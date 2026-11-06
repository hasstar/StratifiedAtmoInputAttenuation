"""Derive a high-detectability local hourly attenuation set without changing sources.

The existing Run_011 aggregation already contains a W-domain hourly mean using
only quality_code == 0 records. This script applies the frozen equipment gain,
re-estimates the train-only clear reference, and censors hours with fewer than
95% of the 36,000 expected 100 ms slots carrying quality-zero records.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow


PROJECT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MEASUREMENT = PROJECT / "Outputs/Datasets/Revision_002/measurement_hourly.parquet"
PRIOR_AUDIT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004/local_hourly_attenuation_audit.parquet"
PRIOR_MODEL = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004/local_model_table_p10_attenuation_all_measurements.parquet"
CALIBRATION = PROJECT / "Outputs/Datasets/Revision_006/equipment_calibration.json"
FEATURES = PROJECT / "Outputs/Datasets/Revision_006/feature_sets.csv"
OUT = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_001"
EXPECTED_SLOTS = 36_000
MIN_DETECTED_SLOTS = math.ceil(0.95 * EXPECTED_SLOTS)
SPLIT = "split_event_chrono_24h"
TARGET = "clear_sky_attenuation_db"
EVENTS = ("rain", "heavy_rain", "dry_overcast", "snow")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")
    readback = pd.read_parquet(path)
    check(readback.shape == frame.shape, f"Readback shape mismatch: {path.name}")
    check(readback.columns.tolist() == frame.columns.tolist(), f"Readback schema mismatch: {path.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(PRIOR_AUDIT)
    measurement = pd.read_parquet(MEASUREMENT, columns=[
        "hour_utc", "measurement_unique_timestamps", "quality_zero_records",
        "quality_bit32_records", "received_power_quality0_w",
    ])
    model = pd.read_parquet(PRIOR_MODEL)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    check(len(old) == len(measurement) == len(model) == 3869, "Unexpected hourly row counts")
    for label, frame in (("audit", old), ("measurement", measurement), ("model", model)):
        check(frame.hour_utc.is_unique, f"Duplicate hour in {label}")
    check(set(old.hour_utc) == set(measurement.hour_utc) == set(model.hour_utc), "Hour keys differ")
    check(calibration["calibration_kind"] == "retrospective_equipment_gain", "Wrong gain source")
    gain = float(calibration["linear_gain"])
    boundary = pd.Timestamp(calibration["boundary_bjt"])
    check(gain > 0 and boundary.tzinfo is not None, "Invalid gain/boundary")

    audit = old.merge(measurement, on="hour_utc", how="left", validate="one_to_one")
    check(audit.quality_zero_records.notna().all(), "Missing quality count")
    check((audit.quality_zero_records <= audit.measurement_unique_timestamps).all(), "Invalid quality counts")
    check(np.allclose(audit.quality_zero_fraction,
                      audit.quality_zero_records / audit.measurement_unique_timestamps,
                      atol=1e-12), "Quality-zero fraction mismatch")
    check(np.allclose(audit.quality_bit32_fraction,
                      audit.quality_bit32_records / audit.measurement_unique_timestamps,
                      atol=1e-12), "Bit32 fraction mismatch")
    audit["previous_received_power_dbm_for_audit_only"] = audit.observed_power_dbm
    audit["previous_clear_sky_attenuation_db_for_audit_only"] = audit[TARGET]
    audit["signal_detected_100ms_fraction"] = audit.quality_zero_records / EXPECTED_SLOTS
    audit["signal_detected_hour"] = audit.quality_zero_records >= MIN_DETECTED_SLOTS
    check((~audit.signal_detected_hour | audit.measurement_high_coverage_95).all(),
          "Detected hour cannot lack raw coverage")
    audit["hourly_power_quality0_corrected_w"] = audit.received_power_quality0_w * np.where(
        audit.hour_bjt >= boundary, gain, 1.0)
    audit["hourly_power_quality0_corrected_dbm"] = 10 * np.log10(
        audit.hourly_power_quality0_corrected_w * 1000)
    check(np.isfinite(audit.loc[audit.signal_detected_hour,
                               "hourly_power_quality0_corrected_dbm"]).all(), "Missing detected power")

    # Keep the prior clear/dry event definition, but use only the fixed training
    # split and newly valid powers to estimate the reference. No test labels enter.
    candidate = (audit.signal_detected_hour & audit[SPLIT].eq("train") & audit.clear_dry)
    reference = float(audit.loc[candidate, "hourly_power_quality0_corrected_dbm"].median())
    check(int(candidate.sum()) == 1430 and np.isfinite(reference), "Reference candidates changed")
    audit["clear_reference_power_dbm"] = reference
    audit["clear_reference_candidate_count"] = int(candidate.sum())
    audit["clear_reference_method"] = "train_clear_dry_detected95_quality0_median_fixed"
    audit["observed_power_dbm"] = audit.hourly_power_quality0_corrected_dbm.where(audit.signal_detected_hour)
    audit[TARGET] = (reference - audit.observed_power_dbm).where(audit.signal_detected_hour)
    audit["target_status"] = np.select(
        [audit.signal_detected_hour, ~audit.measurement_high_coverage_95],
        ["detected_hour_quality0", "insufficient_raw_coverage"],
        default="quality0_coverage_below_95pct",
    )
    audit["clear_context_power_conflict"] = (audit.clear_dry & (audit[TARGET] > 2.0)).fillna(False)
    check(int(audit.signal_detected_hour.sum()) == 3434, "Unexpected detected-hour count")
    check(audit.loc[~audit.signal_detected_hour, TARGET].isna().all(), "Censored label leaked")

    # Preserve all original model columns/order; change only its target/power
    # metadata. This avoids accidentally carrying the prior floor-valued target.
    update_cols = ["hour_utc", "observed_power_dbm", TARGET, "clear_reference_power_dbm",
                   "target_status", "clear_context_power_conflict"]
    update = audit[update_cols].rename(columns={"observed_power_dbm": "received_power_dbm"})
    model = model.drop(columns=[c for c in update.columns if c != "hour_utc"]).merge(
        update, on="hour_utc", how="left", validate="one_to_one")
    old_order = pd.read_parquet(PRIOR_MODEL).columns.tolist()
    model = model[old_order]
    check(model.columns.tolist() == old_order and len(model) == 3869, "Model schema changed")
    check(model[TARGET].notna().sum() == 3434, "Model target count mismatch")
    feature_table = pd.read_csv(FEATURES)
    features = feature_table.loc[feature_table.feature_set.eq("S_plus_B_plus_P10_primary"), "column"].tolist()
    check(len(features) == len(set(features)) == 348 and set(features).issubset(model.columns),
          "348 meteorological features changed")
    high = model.loc[model[TARGET].notna()].sort_values("hour_bjt").reset_index(drop=True)

    # Original target is retained only in this separately labelled audit table.
    write_parquet(audit.sort_values("hour_bjt").reset_index(drop=True),
                  OUT / "local_hourly_detectability_audit.parquet")
    write_parquet(model.sort_values("hour_bjt").reset_index(drop=True),
                  OUT / "local_model_table_p10_all_measurements_censored.parquet")
    write_parquet(high, OUT / "local_model_table_p10_detected95.parquet")
    old_high = audit.measurement_high_coverage_95
    newly_excluded = audit.loc[old_high & ~audit.signal_detected_hour, [
        "hour_bjt", SPLIT, "measurement_coverage_fraction", "quality_zero_fraction",
        "signal_detected_100ms_fraction", "quality_bit32_fraction",
        "previous_received_power_dbm_for_audit_only",
        "previous_clear_sky_attenuation_db_for_audit_only", *EVENTS,
    ]].sort_values("hour_bjt")
    check(len(newly_excluded) == 95, "Unexpected newly censored hours")
    newly_excluded.to_csv(OUT / "newly_censored_from_previous_high_coverage.csv", index=False)

    summary_rows = []
    for split in ("train", "validation", "test", "embargo", "snow_case"):
        group = audit.loc[audit[SPLIT].eq(split)]
        row = {"split": split, "all_measurement_hours": len(group),
               "previous_raw_high_coverage_hours": int(group.measurement_high_coverage_95.sum()),
               "new_detected95_hours": int(group.signal_detected_hour.sum()),
               "newly_censored_from_previous_high": int((group.measurement_high_coverage_95 &
                                                         ~group.signal_detected_hour).sum())}
        for event in EVENTS:
            row[f"{event}_previous_raw_high"] = int((group.measurement_high_coverage_95 & group[event]).sum())
            row[f"{event}_new_detected95"] = int((group.signal_detected_hour & group[event]).sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "split_event_retention.csv", index=False)
    check(summary.new_detected95_hours.sum() == 3434, "Split counts mismatch")
    check(summary.newly_censored_from_previous_high.sum() == 95, "Exclusion counts mismatch")

    manifest = {
        "revision": "Signal_Detectability_Revision_001",
        "decision_timing": "post-hoc after viewing full-348 validation/test errors; not an untouched final test",
        "record_rule": "Use Run_011 records with quality_code == 0 only; bit32 (<10 dB peak/background) is excluded, as are other quality bits",
        "hour_rule": f"At least {MIN_DETECTED_SLOTS} of {EXPECTED_SLOTS} expected 100 ms slots must be quality-zero",
        "censored_rule": "Retain time/features/event flags but set received_power_dbm and clear_sky_attenuation_db to null for unusable hours; do not average only brief recovered intervals into a whole-hour target",
        "reference_rule": "Median of calibrated quality-zero hourly dBm in detected95 train clear_dry hours",
        "reference_candidate_hours": int(candidate.sum()),
        "reference_dbm": reference,
        "equipment_gain_w_multiplier_after_boundary": gain,
        "boundary_bjt": boundary.isoformat(),
        "rows_all": int(len(model)), "rows_detected95": int(len(high)),
        "rows_newly_censored_from_previous_high": int(len(newly_excluded)),
        "input_sha256": {str(p.relative_to(PROJECT)): sha256(p)
                         for p in (MEASUREMENT, PRIOR_AUDIT, PRIOR_MODEL, CALIBRATION, FEATURES)},
        "builder_sha256": sha256(HERE / "build_detected_hour_dataset.py"),
        "output_sha256": {p.name: sha256(p) for p in OUT.iterdir()
                          if p.is_file() and p.name != "manifest.json"},
        "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Reference {reference:.9f} dBm; new detected hours {len(high)}; newly censored {len(newly_excluded)}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
