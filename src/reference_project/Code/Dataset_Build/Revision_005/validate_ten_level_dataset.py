"""Independently verify Revision_005 against frozen Revision_004 sources."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "Outputs" / "Datasets" / "Revision_004"
OUTPUT = PROJECT / "Outputs" / "Datasets" / "Revision_005"
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
META = (
    "hour_bjt", "hour_utc", "split_blocked", "split_strict_24h",
    "split_event_chrono_24h", "received_power_dbm",
)
SCOPES = ("all_measurements", "high_coverage_95")
COPIED = (
    "time_splits.csv", "hourly_event_labels.csv", "event_counts.csv",
    "split_summary.csv", "fast_hour_split_lookup.parquet",
    "fast_split_codes.csv", "fast_window_counts.csv", "fast_window_split.parquet",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def check_integrity(manifest: dict) -> dict:
    require(manifest["version"] == "Revision_005", "Wrong manifest version")
    require(manifest["source_validation_status"] == "VERIFIED",
            "Source validation state not recorded as VERIFIED")
    old_report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    require(old_report["status"] == "VERIFIED", "Source no longer VERIFIED")
    for name, expected in manifest["source_sha256"].items():
        require(sha256(SOURCE / name) == expected, f"Source changed: {name}")
    for name in COPIED:
        require(sha256(OUTPUT / name) == sha256(SOURCE / name),
                f"Copied sidecar changed: {name}")
    lines = (OUTPUT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    checks = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        require(name not in checks, f"Duplicate output checksum: {name}")
        checks[name] = digest
    for name, expected in checks.items():
        require(sha256(OUTPUT / name) == expected, f"Output changed: {name}")
    expected_files = {p.name for p in OUTPUT.iterdir() if p.is_file()} - {
        "checksums.sha256", "validation_report.json", "VALIDATION.md",
    }
    require(set(checks) == expected_files, "Checksum inventory differs from output files")
    return {"source_files": len(manifest["source_sha256"]),
            "copied_sidecars": len(COPIED), "built_files": len(checks)}


def check_features() -> tuple[dict, list[str], dict[int, list[str]]]:
    sets = pd.read_csv(OUTPUT / "feature_sets.csv")
    groups = pd.read_csv(OUTPUT / "feature_groups.csv")
    old_groups = pd.read_csv(SOURCE / "feature_groups.csv")
    by_set = {
        name: part.sort_values("feature_order")["column"].tolist()
        for name, part in sets.groupby("feature_set", sort=False)
    }
    require(set(by_set) == {"S_local", "S_plus_B", "S_plus_B_plus_P10_primary"},
            "Unexpected new feature sets")
    require({name: len(columns) for name, columns in by_set.items()}
            == {"S_local": 117, "S_plus_B": 188,
                "S_plus_B_plus_P10_primary": 348}, "Feature counts are wrong")
    for name in ("S_local", "S_plus_B"):
        old = pd.read_csv(SOURCE / "feature_sets.csv")
        old_columns = old.loc[old["feature_set"] == name].sort_values(
            "feature_order")["column"].tolist()
        require(by_set[name] == old_columns, f"Base order changed: {name}")
    require(groups["column"].is_unique and len(groups) == 348,
            "Feature group table is not 348 unique columns")
    require(groups.sort_values("feature_order")["column"].tolist()
            == by_set["S_plus_B_plus_P10_primary"],
            "Feature group order differs from primary feature set")
    require(not any(c.startswith("p1000__") for c in groups["column"]),
            "1000 hPa remains in feature groups")
    pressure = groups.loc[groups["group_kind"] == "pressure_level"].copy()
    pressure["pressure_level_hpa"] = pressure["pressure_level_hpa"].astype(int)
    require(set(pressure["pressure_level_hpa"]) == set(LEVELS),
            "Retained level set differs")
    by_level = {
        level: pressure.loc[pressure["pressure_level_hpa"] == level]
        .sort_values("feature_order")["column"].tolist()
        for level in LEVELS
    }
    require(all(len(columns) == 16 for columns in by_level.values()),
            "A pressure level does not have 16 predictors")
    expected_order = [*by_set["S_plus_B"],
                      *(c for level in LEVELS for c in by_level[level])]
    require(by_set["S_plus_B_plus_P10_primary"] == expected_order,
            "Pressure level order or base columns differ")
    expected_old = old_groups.loc[
        ~old_groups["column"].str.startswith("p1000__"), "column"
    ]
    require(set(expected_order) == set(expected_old),
            "Feature membership is not source minus 1000 hPa")
    counts = pd.read_csv(OUTPUT / "feature_set_summary.csv").set_index(
        "feature_set")["feature_count"].to_dict()
    require(counts == {name: len(columns) for name, columns in by_set.items()},
            "Feature summary differs")
    registry = pd.read_csv(OUTPUT / "comparison_registry.csv")
    require(set(registry["feature_set"]) == set(by_set),
            "Comparison registry feature sets differ")
    require(all(row.n_predictors == len(by_set[row.feature_set])
                for row in registry.itertuples()),
            "Comparison registry counts differ")
    return {"base": 188, "pressure_levels": 10, "per_level": 16,
            "primary_predictors": 348}, expected_order, by_level


def check_tables(primary: list[str], by_level: dict[int, list[str]]) -> dict:
    counts = {}
    missingness_rows = []
    all_hours = None
    for scope in SCOPES:
        old = pd.read_parquet(SOURCE / f"model_table_primary_p11_{scope}.parquet")
        new = pd.read_parquet(OUTPUT / f"model_table_primary_p10_{scope}.parquet")
        expected = old.loc[:, [*META, *primary]]
        pd.testing.assert_frame_equal(new, expected, check_exact=True)
        require(new["hour_bjt"].is_unique, f"Duplicate new hour: {scope}")
        require(new["received_power_dbm"].notna().all(),
                f"Missing target: {scope}")
        require(len(new.columns) == 354 and not any(
            c.startswith("p1000__") for c in new.columns),
            f"New model table dimensions wrong: {scope}")
        csv = pd.read_csv(
            OUTPUT / f"model_table_primary_p10_{scope}.csv",
            float_precision="round_trip",
        )
        require(csv.columns.tolist() == new.columns.tolist() and len(csv) == len(new),
                f"CSV schema or hours wrong: {scope}")
        numeric_columns = ["received_power_dbm", *primary]
        require(np.allclose(
            csv[numeric_columns].to_numpy(dtype=float),
            new[numeric_columns].to_numpy(dtype=float),
            rtol=1e-14, atol=0.0, equal_nan=True,
        ), f"CSV numeric round-trip differs: {scope}")
        for column in META[:-1]:
            require(csv[column].astype(str).tolist() == new[column].astype(str).tolist(),
                    f"CSV metadata differs: {scope}/{column}")
        counts[scope] = len(new)
        if scope == "all_measurements":
            all_hours = set(new["hour_bjt"])
        else:
            require(set(new["hour_bjt"]) <= all_hours,
                    "High-coverage hours are not a subset of all hours")
            splits = new["split_event_chrono_24h"].value_counts().to_dict()
            require(splits == {"train": 2526, "validation": 442, "test": 489,
                               "embargo": 48, "snow_case": 24},
                    f"High-coverage split sizes changed: {splits}")
        for split, section in new.groupby("split_event_chrono_24h", sort=False):
            for level in LEVELS:
                present = section[by_level[level]].notna().all(axis=1)
                missingness_rows.append({
                    "scope": scope, "split": split, "pressure_level_hpa": level,
                    "hours": len(section), "all_16_present_hours": int(present.sum()),
                    "at_least_one_missing_hours": int((~present).sum()),
                })
    require(counts == {"all_measurements": 3869, "high_coverage_95": 3529},
            f"New hour counts differ: {counts}")
    observed = pd.read_csv(OUTPUT / "missingness_by_level_split.csv")
    expected_missing = pd.DataFrame(missingness_rows)
    keys = ["scope", "split", "pressure_level_hpa"]
    pd.testing.assert_frame_equal(
        observed.sort_values(keys).reset_index(drop=True),
        expected_missing.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    old_missing = pd.read_csv(SOURCE / "missingness_by_level_split.csv")
    old_missing = old_missing.loc[old_missing["pressure_level_hpa"] != 1000]
    pd.testing.assert_frame_equal(
        observed.sort_values(keys).reset_index(drop=True),
        old_missing.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    return {"hours": counts, "high_coverage_split_hours": splits,
            "missingness_rows": len(observed)}


def check_catalogs() -> dict:
    catalog = pd.read_csv(OUTPUT / "subset_catalog.csv", dtype={"mask_binary": str})
    require(len(catalog) == 1024 and catalog["subset_id"].is_unique,
            "Subset catalog size or IDs wrong")
    require(catalog["mask_decimal"].tolist() == list(range(1024)),
            "Subset masks do not cover 0..1023 in order")
    for row in catalog.itertuples():
        included = [level for bit, level in enumerate(LEVELS)
                    if row.mask_decimal & (1 << bit)]
        excluded = [level for level in LEVELS if level not in included]
        require(row.subset_id == f"P10_{row.mask_decimal:04X}", "Subset ID wrong")
        require(row.mask_binary == format(row.mask_decimal, "010b"), "Mask binary wrong")
        require(row.n_pressure_levels == len(included), "Level count wrong")
        require((row.retained_levels_hpa if isinstance(row.retained_levels_hpa, str) else "")
                == ";".join(map(str, included)), "Retained levels wrong")
        require((row.removed_levels_hpa if isinstance(row.removed_levels_hpa, str) else "")
                == ";".join(map(str, excluded)), "Removed levels wrong")
        require(row.n_pressure_predictors == 16 * len(included)
                and row.n_predictors_with_S_plus_B == 188 + 16 * len(included),
                "Subset predictor count wrong")
        require(bool(row.is_formal_primary) == (row.mask_decimal == 1023),
                "Formal-primary marker wrong")
    counts = pd.read_csv(OUTPUT / "subset_counts.csv")
    require(counts["n_pressure_levels"].tolist() == list(range(11)), "Count levels wrong")
    require(counts["subset_count"].tolist() == [math.comb(10, n) for n in range(11)],
            "Binomial subset counts wrong")
    pairs = pd.read_csv(OUTPUT / "layer_pair_catalog.csv")
    require(len(pairs) == 5120 and pairs["pressure_level_hpa"].value_counts().to_dict()
            == {level: 512 for level in LEVELS}, "Layer pair counts wrong")
    require(not pairs.duplicated(["pressure_level_hpa", "subset_without_id"]).any(),
            "Duplicate layer pair")
    for row in pairs.itertuples():
        bit = LEVELS.index(row.pressure_level_hpa)
        without = int(row.subset_without_id.removeprefix("P10_"), 16)
        with_level = int(row.subset_with_id.removeprefix("P10_"), 16)
        require(without & (1 << bit) == 0 and with_level == without | (1 << bit),
                "A pair differs by more than its indexed layer")
        require(row.other_level_count == without.bit_count()
                and row.added_predictors == 16, "Pair metadata wrong")
    return {"subsets": 1024, "layer_pairs": 5120, "pairs_per_level": 512}


def main() -> None:
    require(OUTPUT.is_dir(), "Revision_005 output is missing")
    manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
    checks = [
        ("artifact_and_source_integrity", lambda: check_integrity(manifest)),
        ("feature_registry", check_features),
    ]
    results = []
    integrity = checks[0][1]()
    results.append({"name": checks[0][0], **integrity})
    feature_result, primary, by_level = check_features()
    results.append({"name": "feature_registry", **feature_result})
    table_result = check_tables(primary, by_level)
    results.append({"name": "model_tables_and_splits", **table_result})
    catalog_result = check_catalogs()
    results.append({"name": "subset_and_pair_catalogs", **catalog_result})
    report = {
        "status": "VERIFIED",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": len(results),
        "checks_total": len(results),
        "checks": results,
        "validator_sha256": sha256(Path(__file__)),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "scope_note": "Dataset construction only; no model fit or layer-contribution result.",
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "VALIDATION.md").write_text(
        "# Revision_005 independent validation\n\n"
        f"Status: VERIFIED. Date UTC: {report['validated_utc']}.\n\n"
        "4/4 checks passed: source/output integrity; feature registry; "
        "exact hourly table and split preservation; 1,024 masks and 5,120 "
        "layer pairs. This does not verify model accuracy or physical causality.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
