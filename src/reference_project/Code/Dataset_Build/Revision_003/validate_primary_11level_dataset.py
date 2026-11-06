"""Independently validate Revision_003 against verified Revision_002."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "Outputs" / "Datasets"
SOURCE = ROOT / "Revision_002"
DATASET = ROOT / "Revision_003"
LEVELS = (1000, 925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
META = ("hour_bjt", "hour_utc", "split_blocked", "split_strict_24h", "received_power_dbm")
TABLES = {
    "primary_p11": ("model_table_pressure_levels", "S_plus_B_plus_P11_primary", 364),
    "baseline_l5": ("model_table_five_layers", "S_plus_B_plus_L5_original", 268),
    "baseline_p0_broad": ("model_table_pressure_levels", "S_plus_B", 188),
    "baseline_surface_local": ("model_table_pressure_levels", "S_local", 117),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_checksums() -> dict[str, object]:
    lines = (DATASET / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    names = []
    for line in lines:
        expected, name = line.split("  ", 1)
        names.append(name)
        check(digest(DATASET / name) == expected, f"Checksum mismatch: {name}")
    actual = {path.name for path in DATASET.iterdir() if path.is_file()}
    check(set(names) == actual - {"checksums.sha256", "validation_report.json", "VALIDATION.md"}, "Checksum file inventory mismatch")
    return {"name": "output_artifact_checksums", "files_checked": len(names)}


def verify_sources(manifest: dict[str, object]) -> dict[str, object]:
    source_report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    check(source_report["status"] == "VERIFIED", "Source dataset is not verified")
    source_hashes = manifest["source_sha256"]
    for name, expected in source_hashes.items():
        check(digest(SOURCE / name) == expected, f"Source changed: {name}")
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
        check(digest(SOURCE / name) == digest(DATASET / name), f"Source sidecar changed in copy: {name}")
    return {"name": "verified_source_identity", "source_files_checked": len(source_hashes), "copied_sidecars_checked": 9}


def read_features() -> dict[str, list[str]]:
    frame = pd.read_csv(DATASET / "feature_sets.csv")
    return {
        name: rows.sort_values("feature_order")["column"].tolist()
        for name, rows in frame.groupby("feature_set", sort=False)
    }


def verify_models(features: dict[str, list[str]]) -> dict[str, object]:
    check(set(features) == {value[1] for value in TABLES.values()}, "Feature sets differ from table registry")
    check(set(features["S_local"]) < set(features["S_plus_B"]), "Local surface not inside broad baseline")
    check(set(features["S_plus_B"]) < set(features["S_plus_B_plus_P11_primary"]), "Broad baseline not inside primary")
    check(set(features["S_plus_B"]) < set(features["S_plus_B_plus_L5_original"]), "Broad baseline not inside five-layer table")
    mapping = pd.read_csv(DATASET / "feature_groups.csv")
    check(len(mapping) == 364 and mapping["column"].is_unique, "Feature-group map must contain 364 unique predictors")
    check(set(mapping["column"]) == set(features["S_plus_B_plus_P11_primary"]), "Feature-group map differs from primary predictors")
    counts = mapping.loc[mapping["group_kind"].eq("pressure_level"), "pressure_level_hpa"].value_counts().to_dict()
    check(set(counts) == set(LEVELS), "Wrong pressure-level map")
    check(all(counts[level] == 16 for level in LEVELS), "Each pressure level must have 16 variables")
    check(mapping["group_id"].value_counts().get("S_local") == 117, "Wrong strict local count")
    check(mapping["group_id"].value_counts().get("B_additional") == 71, "Wrong additional broad count")

    checked = 0
    for table_key, (source_prefix, feature_name, n_features) in TABLES.items():
        check(len(features[feature_name]) == n_features, f"Wrong feature count: {feature_name}")
        for scope, source_scope, expected_rows in (
            ("all_measurements", "all_measurements", 3869),
            ("high_coverage_95", "high_coverage_95", 3529),
        ):
            output_stem = f"model_table_{table_key}_{scope}"
            actual = pd.read_parquet(DATASET / f"{output_stem}.parquet")
            source = pd.read_parquet(SOURCE / f"{source_prefix}_{source_scope}.parquet")
            expected = source.loc[:, [*META, *features[feature_name]]]
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            check(len(actual) == expected_rows, f"Wrong hour count: {output_stem}")
            check(actual["hour_bjt"].is_unique, f"Duplicate hour: {output_stem}")
            check(actual.columns[:5].tolist() == list(META), f"Wrong metadata order: {output_stem}")
            check("received_power_w" not in actual and "quality_code" not in actual, f"Target transform or QA leakage: {output_stem}")
            with (DATASET / f"{output_stem}.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                check(next(reader) == actual.columns.tolist(), f"CSV header mismatch: {output_stem}")
                check(sum(1 for _ in reader) == expected_rows, f"CSV row count mismatch: {output_stem}")
            checked += 1
    return {"name": "model_tables_match_verified_source", "table_scopes_checked": checked, "primary_predictors": 364}


def verify_subsets(features: dict[str, list[str]]) -> dict[str, object]:
    catalog = pd.read_csv(
        DATASET / "subset_catalog.csv",
        keep_default_na=False,
        dtype={"subset_id": str, "mask_binary": str, "retained_levels_hpa": str},
    )
    check(len(catalog) == 2048, "Subset catalog must contain 2048 rows")
    check(catalog["mask_decimal"].tolist() == list(range(2048)), "Subset masks incomplete or out of order")
    check(catalog["subset_id"].is_unique, "Duplicate subset identifier")
    observed = Counter(catalog["n_pressure_levels"])
    for n in range(12):
        check(observed[n] == math.comb(11, n), f"Wrong subset count at {n} levels")
    for row in catalog.itertuples(index=False):
        included = [level for bit, level in enumerate(LEVELS) if row.mask_decimal & (1 << bit)]
        check(row.subset_id == f"P11_{row.mask_decimal:04X}", "Subset ID does not match mask")
        check(row.mask_binary == format(row.mask_decimal, "011b"), "Binary mask mismatch")
        check(row.n_pressure_levels == len(included), "Subset cardinality mismatch")
        check(row.retained_levels_hpa == ";".join(map(str, included)), "Retained level list mismatch")
        check(row.n_pressure_predictors == 16 * len(included), "Pressure predictor count mismatch")
        check(row.n_predictors_with_S_plus_B == 188 + 16 * len(included), "Broad-base count mismatch")
        check(row.n_predictors_with_S_local == 117 + 16 * len(included), "Local-base count mismatch")
        check(bool(row.is_formal_primary) == (row.mask_decimal == 2047), "Primary model marker mismatch")
    counts = pd.read_csv(DATASET / "subset_counts.csv")
    check(counts["subset_count"].tolist() == [math.comb(11, n) for n in range(12)], "Subset summary mismatch")
    registry = pd.read_csv(DATASET / "comparison_registry.csv")
    check(set(registry["comparison_id"]) == {"primary_11_pressure", "original_five_layers", "no_pressure_broad", "strict_local_surface"}, "Comparison registry incomplete")
    return {"name": "all_pressure_level_subsets", "combinations_checked": len(catalog), "count_by_level": dict(sorted(observed.items()))}


def verify_missingness() -> dict[str, object]:
    report = pd.read_csv(DATASET / "missingness_by_level_split.csv")
    group_map = pd.read_csv(DATASET / "feature_groups.csv")
    expected_rows = []
    for scope in ("all_measurements", "high_coverage_95"):
        frame = pd.read_parquet(DATASET / f"model_table_primary_p11_{scope}.parquet")
        for split, section in frame.groupby("split_strict_24h", sort=False):
            for level in LEVELS:
                columns = group_map.loc[group_map["pressure_level_hpa"].eq(level), "column"].tolist()
                complete = section[columns].notna().all(axis=1)
                expected_rows.append(
                    (scope, split, level, len(section), int(complete.sum()), int((~complete).sum()))
                )
    observed_rows = [tuple(row) for row in report.itertuples(index=False, name=None)]
    check(observed_rows == expected_rows, "Pressure-level missingness report mismatch")
    affected_1000 = int(
        report.loc[
            report["sample_scope"].eq("all_measurements")
            & report["pressure_level_hpa"].eq(1000),
            "n_at_least_one_variable_missing",
        ].sum()
    )
    check(affected_1000 > 0, "Expected partial 1000 hPa LOS missingness")
    return {"name": "pressure_level_missingness", "rows_checked": len(report), "1000_hpa_affected_all_measurement_hours": affected_1000}


def main() -> None:
    check(DATASET.is_dir(), "Revision_003 does not exist")
    manifest = json.loads((DATASET / "run_manifest.json").read_text(encoding="utf-8"))
    check(manifest["version"] == "Revision_003", "Wrong dataset version")
    check(manifest["primary_predictor_set"] == "S_plus_B_plus_P11_primary", "11-level set not designated primary")
    check(manifest["models_trained"] == 0, "Dataset must not claim model results")
    checks = [
        verify_checksums(),
        verify_sources(manifest),
        verify_models(read_features()),
        verify_subsets(read_features()),
        verify_missingness(),
    ]
    result = {
        "status": "VERIFIED",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope_note": "This validates dataset construction, not model accuracy or the 2048 unrun experiments.",
    }
    (DATASET / "validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATASET / "VALIDATION.md").write_text(
        "# Revision_003 独立验证\n\n"
        "状态：VERIFIED；五项检查全部通过。11 个气压层是正式主输入，所有 2048 个逐层组合已完整枚举。"
        "主／对照模型表与已验证的 Revision_002 逐值一致，来源文件和复制的侧表未变化。"
        "模型尚未训练，本文档不证明预测精度、分层增益或事件性能。\n",
        encoding="utf-8",
    )
    print(f"VERIFIED {DATASET} ({len(checks)}/{len(checks)} checks)")


if __name__ == "__main__":
    main()
