"""Build the 11-pressure-level primary dataset from verified Revision_002."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT / "Outputs" / "Datasets"
SOURCE = DATASET_ROOT / "Revision_002"
DESTINATION = DATASET_ROOT / "Revision_003"
LEVELS = (1000, 925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
META = ("hour_bjt", "hour_utc", "split_blocked", "split_strict_24h", "received_power_dbm")
EXPECTED_FEATURE_COUNTS = {
    "S_local": 117,
    "S_plus_B": 188,
    "S_plus_B_plus_L": 268,
    "S_plus_B_plus_pressure_levels": 364,
}
FEATURE_NAMES = {
    "S_local": "S_local",
    "S_plus_B": "S_plus_B",
    "S_plus_B_plus_L": "S_plus_B_plus_L5_original",
    "S_plus_B_plus_pressure_levels": "S_plus_B_plus_P11_primary",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_feature_sets() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = pd.read_csv(SOURCE / "feature_sets.csv")
    features = {
        name: group.sort_values("feature_order")["column"].tolist()
        for name, group in frame.groupby("feature_set", sort=False)
    }
    require(set(features) == set(EXPECTED_FEATURE_COUNTS), "Unexpected source feature sets")
    for name, expected in EXPECTED_FEATURE_COUNTS.items():
        require(len(features[name]) == expected, f"Unexpected {name} feature count")
        require(len(features[name]) == len(set(features[name])), f"Duplicate feature in {name}")
    require(set(features["S_local"]) <= set(features["S_plus_B"]), "S_local not inside S_plus_B")
    require(
        set(features["S_plus_B"]) <= set(features["S_plus_B_plus_pressure_levels"]),
        "S_plus_B not inside 11-level feature set",
    )
    require(
        set(features["S_plus_B"]) <= set(features["S_plus_B_plus_L"]),
        "S_plus_B not inside five-layer feature set",
    )
    return frame, features


def pressure_groups(primary_features: list[str], base_features: list[str]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {level: [] for level in LEVELS}
    for column in primary_features:
        if column in base_features:
            continue
        match = re.fullmatch(r"p(\d+)__.+", column)
        require(match is not None, f"Unclassified pressure feature: {column}")
        level = int(match.group(1))
        require(level in groups, f"Unexpected pressure level: {level}")
        groups[level].append(column)
    for level in LEVELS:
        require(len(groups[level]) == 16, f"Expected 16 variables at {level} hPa")
    return groups


def read_model_tables(features: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    filenames = {
        "p11_all": "model_table_pressure_levels_all_measurements.parquet",
        "p11_high": "model_table_pressure_levels_high_coverage_95.parquet",
        "l5_all": "model_table_five_layers_all_measurements.parquet",
        "l5_high": "model_table_five_layers_high_coverage_95.parquet",
    }
    tables = {key: pd.read_parquet(SOURCE / name) for key, name in filenames.items()}
    for key, frame in tables.items():
        feature_name = "S_plus_B_plus_pressure_levels" if key.startswith("p11") else "S_plus_B_plus_L"
        expected = set(META) | {"received_power_w"} | set(features[feature_name])
        require(set(frame.columns) == expected, f"Column mismatch: {key}")
        require(frame["hour_bjt"].is_unique, f"Duplicate time in {key}")
    require(len(tables["p11_all"]) == 3869, "Unexpected all-measurement hour count")
    require(len(tables["p11_high"]) == 3529, "Unexpected high-coverage hour count")
    for suffix in ("all", "high"):
        pd.testing.assert_frame_equal(
            tables[f"p11_{suffix}"].loc[:, list(META)].reset_index(drop=True),
            tables[f"l5_{suffix}"].loc[:, list(META)].reset_index(drop=True),
            check_exact=True,
        )
    require(
        set(tables["p11_high"]["hour_bjt"]) <= set(tables["p11_all"]["hour_bjt"]),
        "High-coverage hours are not a subset of all hours",
    )
    return tables


def write_model_table(directory: Path, stem: str, frame: pd.DataFrame, columns: list[str]) -> None:
    require(len(columns) == len(set(columns)), f"Duplicate output column in {stem}")
    selected = frame.loc[:, columns]
    selected.to_parquet(directory / f"{stem}.parquet", index=False, compression="zstd")
    selected.to_csv(directory / f"{stem}.csv", index=False, float_format="%.17g")


def write_feature_metadata(
    directory: Path,
    raw_feature_sets: pd.DataFrame,
    features: dict[str, list[str]],
    groups: dict[int, list[str]],
) -> None:
    renamed = raw_feature_sets.copy()
    renamed["feature_set"] = renamed["feature_set"].map(FEATURE_NAMES)
    renamed["dataset_role"] = renamed["feature_set"].map(
        {
            "S_local": "strict local/surface control",
            "S_plus_B": "no-pressure broad-atmosphere control",
            "S_plus_B_plus_L5_original": "original five-layer control",
            "S_plus_B_plus_P11_primary": "formal 11-pressure-level primary",
        }
    )
    renamed.to_csv(directory / "feature_sets.csv", index=False)
    summary = renamed.groupby("feature_set", sort=False).agg(feature_count=("column", "size"))
    summary.reset_index().to_csv(directory / "feature_set_summary.csv", index=False)

    local = set(features["S_local"])
    base = set(features["S_plus_B"])
    level_of = {column: level for level, columns in groups.items() for column in columns}
    feature_rows = []
    for index, column in enumerate(features["S_plus_B_plus_pressure_levels"], start=1):
        if column in local:
            group_id, kind, level = "S_local", "local_surface", ""
        elif column in base:
            group_id, kind, level = "B_additional", "single_level_or_integrated", ""
        else:
            level = level_of[column]
            group_id, kind = f"P{level}", "pressure_level"
        feature_rows.append(
            {
                "feature_order": index,
                "column": column,
                "group_id": group_id,
                "group_kind": kind,
                "pressure_level_hpa": level,
                "included_in_primary_p11": True,
                "included_in_strict_surface": column in local,
            }
        )
    pd.DataFrame(feature_rows).to_csv(directory / "feature_groups.csv", index=False)

    subset_rows = []
    for mask in range(1 << len(LEVELS)):
        included = [level for bit, level in enumerate(LEVELS) if mask & (1 << bit)]
        excluded = [level for level in LEVELS if level not in included]
        subset_rows.append(
            {
                "subset_id": f"P11_{mask:04X}",
                "mask_decimal": mask,
                "mask_binary": format(mask, "011b"),
                "n_pressure_levels": len(included),
                "retained_levels_hpa": ";".join(map(str, included)),
                "removed_levels_hpa": ";".join(map(str, excluded)),
                "n_pressure_predictors": sum(len(groups[level]) for level in included),
                "n_predictors_with_S_plus_B": len(base) + sum(len(groups[level]) for level in included),
                "n_predictors_with_S_local": len(local) + sum(len(groups[level]) for level in included),
                "is_formal_primary": mask == (1 << len(LEVELS)) - 1,
            }
        )
    pd.DataFrame(subset_rows).to_csv(directory / "subset_catalog.csv", index=False)
    pd.DataFrame(
        {
            "n_pressure_levels": list(range(12)),
            "subset_count": [math.comb(11, count) for count in range(12)],
        }
    ).to_csv(directory / "subset_counts.csv", index=False)

    registry = [
        ("primary_11_pressure", "model_table_primary_p11", "S_plus_B_plus_P11_primary", "formal primary"),
        ("original_five_layers", "model_table_baseline_l5", "S_plus_B_plus_L5_original", "original-paper control"),
        ("no_pressure_broad", "model_table_baseline_p0_broad", "S_plus_B", "zero-pressure control; not pure surface"),
        ("strict_local_surface", "model_table_baseline_surface_local", "S_local", "strict near-surface control"),
    ]
    pd.DataFrame(
        [
            {
                "comparison_id": identifier,
                "model_table_prefix": table,
                "feature_set": feature_set,
                "n_predictors": len(features[next(key for key, name in FEATURE_NAMES.items() if name == feature_set)]),
                "role": role,
            }
            for identifier, table, feature_set, role in registry
        ]
    ).to_csv(directory / "comparison_registry.csv", index=False)


def write_missingness(directory: Path, tables: dict[str, pd.DataFrame], groups: dict[int, list[str]]) -> None:
    rows = []
    for scope, frame in (("all_measurements", tables["p11_all"]), ("high_coverage_95", tables["p11_high"])):
        for split, section in frame.groupby("split_strict_24h", sort=False):
            for level in LEVELS:
                complete = section.loc[:, groups[level]].notna().all(axis=1)
                rows.append(
                    {
                        "sample_scope": scope,
                        "split_strict_24h": split,
                        "pressure_level_hpa": level,
                        "n_hours": len(section),
                        "n_all_16_variables_present": int(complete.sum()),
                        "n_at_least_one_variable_missing": int((~complete).sum()),
                    }
                )
    pd.DataFrame(rows).to_csv(directory / "missingness_by_level_split.csv", index=False)


def main() -> None:
    require(SOURCE.is_dir(), "Revision_002 does not exist")
    require(not DESTINATION.exists(), "Revision_003 already exists; refusing to overwrite it")
    require(SOURCE.resolve().parent == DATASET_ROOT.resolve(), "Source outside dataset root")
    require(DESTINATION.resolve().parent == DATASET_ROOT.resolve(), "Destination outside dataset root")
    source_validation = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    require(source_validation.get("status") == "VERIFIED", "Source dataset is not independently verified")

    raw_feature_sets, features = read_feature_sets()
    groups = pressure_groups(features["S_plus_B_plus_pressure_levels"], features["S_plus_B"])
    tables = read_model_tables(features)
    stage = Path(tempfile.mkdtemp(prefix="_Revision_003_stage_", dir=DATASET_ROOT))
    require(stage.resolve().parent == DATASET_ROOT.resolve(), "Staging directory outside dataset root")

    specifications = [
        ("model_table_primary_p11", "p11", features["S_plus_B_plus_pressure_levels"]),
        ("model_table_baseline_l5", "l5", features["S_plus_B_plus_L"]),
        ("model_table_baseline_p0_broad", "p11", features["S_plus_B"]),
        ("model_table_baseline_surface_local", "p11", features["S_local"]),
    ]
    for stem, source_prefix, predictors in specifications:
        for suffix, scope in (("all", "all_measurements"), ("high", "high_coverage_95")):
            write_model_table(stage, f"{stem}_{scope}", tables[f"{source_prefix}_{suffix}"], [*META, *predictors])

    for name in (
        "era5_hourly_pressure_levels.csv",
        "era5_hourly_pressure_levels.parquet",
        "era5_hourly_five_layers.csv",
        "era5_hourly_five_layers.parquet",
        "measurement_hourly.csv",
        "measurement_hourly.parquet",
        "time_splits.csv",
        "split_summary.csv",
        "monthly_coverage.csv",
    ):
        shutil.copy2(SOURCE / name, stage / name)
    shutil.copy2(Path(__file__).with_name("README.md"), stage / "README.md")
    write_feature_metadata(stage, raw_feature_sets, features, groups)
    write_missingness(stage, tables, groups)

    source_files = [
        "validation_report.json",
        "feature_sets.csv",
        "model_table_pressure_levels_all_measurements.parquet",
        "model_table_pressure_levels_high_coverage_95.parquet",
        "model_table_five_layers_all_measurements.parquet",
        "model_table_five_layers_high_coverage_95.parquet",
    ]
    manifest = {
        "version": "Revision_003",
        "status": "BUILT_AWAITING_INDEPENDENT_VALIDATION",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(SOURCE),
        "source_validation_status": source_validation["status"],
        "source_sha256": {name: digest(SOURCE / name) for name in source_files},
        "primary_model_table_prefix": "model_table_primary_p11",
        "primary_target": "received_power_dbm",
        "primary_predictor_set": "S_plus_B_plus_P11_primary",
        "primary_predictor_count": 364,
        "level_order_hpa": list(LEVELS),
        "level_feature_count": {str(level): len(groups[level]) for level in LEVELS},
        "subset_count": 1 << len(LEVELS),
        "all_measurement_hours": len(tables["p11_all"]),
        "high_coverage_95_hours": len(tables["p11_high"]),
        "split_protocol": "split_strict_24h frozen chronological split; embargo excluded from fitting/evaluation",
        "subset_base_feature_set": "S_plus_B (188 predictors, includes integrated variables)",
        "strict_local_surface_baseline": "S_local (117 predictors), separately saved",
        "same_time_signal_features_as_predictors": False,
        "received_power_w_in_model_tables": False,
        "formal_attenuation_target_available": False,
        "models_trained": 0,
        "event_balanced_proposal": str(DATASET_ROOT / "Event_Balanced_Proposal_001"),
    }
    (stage / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    file_lines = [f"{digest(path)}  {path.name}" for path in sorted(stage.iterdir()) if path.is_file()]
    (stage / "checksums.sha256").write_text("\n".join(file_lines) + "\n", encoding="utf-8")
    require(not DESTINATION.exists(), "Destination appeared during construction")
    stage.rename(DESTINATION)
    print(f"BUILT {DESTINATION}")
    print(f"HOURS all={len(tables['p11_all'])} high_coverage_95={len(tables['p11_high'])}")
    print("PRIMARY 11 pressure levels, 364 candidate ERA5 predictors")
    print("SUBSETS 2048; no models were trained")


if __name__ == "__main__":
    main()
