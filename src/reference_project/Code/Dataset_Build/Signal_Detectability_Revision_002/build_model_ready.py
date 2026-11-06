"""Package high-detectability local hourly attenuation into model-ready splits.

Only reads validated Signal_Detectability_Revision_001 and the frozen 348-column
feature catalog. Never edits H5, ERA5, Run_011 or older dataset versions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_001"
FEATURE_SOURCE = PROJECT / "Outputs/Datasets/Revision_006/feature_sets.csv"
OUT = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_002"
MODEL_SOURCE = SOURCE / "local_model_table_p10_detected95.parquet"
AUDIT_SOURCE = SOURCE / "local_hourly_detectability_audit.parquet"
PARENT_MANIFEST = SOURCE / "manifest.json"
TARGET = "clear_sky_attenuation_db"
SPLIT = "split_event_chrono_24h"
OFFICIAL = ("train", "validation", "test")
RESERVED = ("embargo", "snow_case")
EVENTS = ("rain", "heavy_rain", "dry_overcast", "clear_dry", "snow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_parquet(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    for path in (MODEL_SOURCE, AUDIT_SOURCE):
        check(sha256(path) == parent["output_sha256"][path.name],
              f"Revision_001 source hash changed: {path}")
    feature_key = str(FEATURE_SOURCE.relative_to(PROJECT)).replace("/", "\\")
    check(sha256(FEATURE_SOURCE) == parent["input_sha256"][feature_key],
          "Feature catalog hash changed")

    catalog = pd.read_csv(FEATURE_SOURCE)
    feature_table = catalog.loc[catalog.feature_set.eq("S_plus_B_plus_P10_primary")].sort_values(
        "feature_order").reset_index(drop=True)
    features = feature_table.column.tolist()
    check(len(features) == len(set(features)) == 348, "Expected 348 distinct features")
    check(feature_table.feature_order.tolist() == list(range(1, 349)), "Feature order changed")
    check(feature_table.allowed_primary_predictor.all(), "Disallowed predictor")

    model = pd.read_parquet(MODEL_SOURCE)
    audit = pd.read_parquet(AUDIT_SOURCE)
    check(len(model) == 3434 and len(audit) == 3869, "Source row counts changed")
    check(model.hour_utc.is_unique and audit.hour_utc.is_unique, "Duplicate hour")
    check(set(model.hour_utc).issubset(set(audit.hour_utc)), "Audit hour mismatch")
    check(set(model[SPLIT]) == set(OFFICIAL + RESERVED), "Unexpected split")
    check(set(features).issubset(model.columns), "Missing predictor")
    check(np.isfinite(model[TARGET]).all(), "Nonfinite target")
    check(not np.isinf(model[features].to_numpy(dtype=float)).any(), "Infinite predictor")

    model = model.sort_values("hour_utc").reset_index(drop=True)
    schema = ["hour_bjt", "hour_utc", *features, TARGET]
    output_paths: list[Path] = []
    split_rows: dict[str, int] = {}
    for split in OFFICIAL + RESERVED:
        frame = model.loc[model[SPLIT].eq(split), schema].reset_index(drop=True)
        filename = f"{split}.parquet" if split in OFFICIAL else f"reserved_{split}.parquet"
        path = OUT / filename
        write_parquet(frame, path)
        output_paths.append(path)
        split_rows[split] = len(frame)
    check(split_rows == {"train": 2520, "validation": 408, "test": 434,
                         "embargo": 48, "snow_case": 24}, "Unexpected split counts")

    feature_path = OUT / "feature_columns.csv"
    feature_table[["feature_order", "column", "source_subgroup", "representation"]].to_csv(
        feature_path, index=False, encoding="utf-8-sig")
    output_paths.append(feature_path)

    sidecar_columns = ["hour_bjt", "hour_utc", SPLIT, "target_status",
                       "measurement_unique_timestamps", "quality_zero_records",
                       "signal_detected_100ms_fraction", "quality_bit32_fraction", *EVENTS]
    status = audit[sidecar_columns].sort_values("hour_utc").reset_index(drop=True).copy()
    status["use_for_regression"] = (status.target_status.eq("detected_hour_quality0")
                                    & status[SPLIT].isin(OFFICIAL))
    reason = status.target_status.astype(str).copy()
    for split in RESERVED:
        reserved = status.target_status.eq("detected_hour_quality0") & status[SPLIT].eq(split)
        reason.loc[reserved] = f"reserved_{split}"
    status["exclusion_reason"] = reason.where(~status.use_for_regression, "")
    check(int(status.use_for_regression.sum()) == 3362, "Regression count mismatch")
    status_path = OUT / "hourly_sample_status.parquet"
    write_parquet(status, status_path)
    output_paths.append(status_path)

    excluded = status.loc[~status.use_for_regression]
    check(len(excluded) == 507, "Excluded/reserved count mismatch")
    excluded_path = OUT / "excluded_or_reserved_hours.csv"
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    output_paths.append(excluded_path)

    summary_rows, missing_rows = [], []
    for split in OFFICIAL + RESERVED:
        frame = model.loc[model[SPLIT].eq(split)]
        match = status.loc[status[SPLIT].eq(split)]
        row = {"split": split, "measured_hours": len(match), "detected_hours": len(frame),
               "regression_hours": len(frame) if split in OFFICIAL else 0,
               "censored_hours": int(match.target_status.ne("detected_hour_quality0").sum())}
        for event in EVENTS:
            row[f"{event}_detected"] = int(match.loc[
                match.target_status.eq("detected_hour_quality0"), event].sum())
            row[f"{event}_regression"] = int(match.loc[match.use_for_regression, event].sum())
        summary_rows.append(row)
        for feature in features:
            n_missing = int(frame[feature].isna().sum())
            missing_rows.append({"split": split, "column": feature, "rows": len(frame),
                                 "missing_count": n_missing,
                                 "missing_fraction": n_missing / len(frame)})
    summary_path = OUT / "split_event_summary.csv"
    missing_path = OUT / "feature_missingness.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False, encoding="utf-8-sig")
    output_paths.extend((summary_path, missing_path))
    check(not model.loc[model[SPLIT].eq("train"), features].isna().all().any(),
          "At least one feature is entirely absent in train")

    manifest = {
        "revision": "Signal_Detectability_Revision_002",
        "source_revision": parent["revision"],
        "source_rule": parent["record_rule"],
        "hour_rule": parent["hour_rule"],
        "decision_timing": parent["decision_timing"],
        "target": TARGET,
        "feature_set": "S_plus_B_plus_P10_primary",
        "feature_count": 348,
        "model_schema": "hour_bjt, hour_utc, 348 meteorological predictors, clear_sky_attenuation_db",
        "model_inputs": "Only feature_columns.csv; time keys and event labels are not predictors",
        "missing_value_rule": "Retain NaN; fit imputation and normalization on train only",
        "split_rows": split_rows,
        "official_regression_rows": sum(split_rows[s] for s in OFFICIAL),
        "censored_rows": int(status.target_status.ne("detected_hour_quality0").sum()),
        "reserved_detected_rows": sum(split_rows[s] for s in RESERVED),
        "source_sha256": {str(path.relative_to(PROJECT)).replace("/", "\\"): sha256(path)
                          for path in (MODEL_SOURCE, AUDIT_SOURCE, PARENT_MANIFEST, FEATURE_SOURCE)},
        "output_sha256": {path.name: sha256(path) for path in output_paths},
        "builder_sha256": sha256(Path(__file__)),
        "python": sys.version.splitlines()[0],
        "pandas": pd.__version__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps({"split_rows": split_rows, "censored": manifest["censored_rows"],
                      "outputs": [path.name for path in output_paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
