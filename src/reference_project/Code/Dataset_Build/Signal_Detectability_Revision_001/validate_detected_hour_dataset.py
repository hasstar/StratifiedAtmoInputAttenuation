"""Independent checks for the signal-detectability local hourly revision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004"
MEASUREMENT = PROJECT / "Outputs/Datasets/Revision_002/measurement_hourly.parquet"
FEATURES = PROJECT / "Outputs/Datasets/Revision_006/feature_sets.csv"
OUTPUT = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_001"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def main() -> None:
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["input_sha256"].items():
        assert digest(PROJECT / rel) == expected, f"Changed input: {rel}"
    for name, expected in manifest["output_sha256"].items():
        assert digest(OUTPUT / name) == expected, f"Changed output: {name}"
    old = pd.read_parquet(SOURCE / "local_model_table_p10_attenuation_all_measurements.parquet")
    new = pd.read_parquet(OUTPUT / "local_model_table_p10_all_measurements_censored.parquet")
    high = pd.read_parquet(OUTPUT / "local_model_table_p10_detected95.parquet")
    audit = pd.read_parquet(OUTPUT / "local_hourly_detectability_audit.parquet")
    measurements = pd.read_parquet(MEASUREMENT, columns=[
        "hour_utc", "quality_zero_records", "received_power_quality0_w"])
    assert len(old) == len(new) == len(audit) == len(measurements) == 3869
    assert len(high) == 3434 and new.clear_sky_attenuation_db.notna().sum() == 3434
    assert old.columns.tolist() == new.columns.tolist()
    assert new.hour_utc.is_unique and high.hour_utc.is_unique and audit.hour_utc.is_unique
    old = old.sort_values("hour_utc").reset_index(drop=True)
    new = new.sort_values("hour_utc").reset_index(drop=True)
    high = high.sort_values("hour_utc").reset_index(drop=True)
    audit = audit.sort_values("hour_utc").reset_index(drop=True)
    measurements = measurements.sort_values("hour_utc").reset_index(drop=True)
    pd.testing.assert_series_equal(old.hour_utc, new.hour_utc)
    pd.testing.assert_series_equal(new.hour_utc, audit.hour_utc)
    pd.testing.assert_series_equal(audit.hour_utc, measurements.hour_utc)
    feature_names = pd.read_csv(FEATURES).query(
        "feature_set == 'S_plus_B_plus_P10_primary'")["column"].tolist()
    assert len(feature_names) == len(set(feature_names)) == 348
    pd.testing.assert_frame_equal(old[feature_names], new[feature_names])
    for column in ("hour_bjt", "hour_utc", "split_event_chrono_24h"):
        pd.testing.assert_series_equal(old[column], new[column])

    eligible = measurements.quality_zero_records.to_numpy() >= 34200
    assert np.array_equal(eligible, audit.signal_detected_hour.to_numpy())
    assert audit.loc[~eligible, "clear_sky_attenuation_db"].isna().all()
    assert audit.loc[~eligible, "observed_power_dbm"].isna().all()
    assert new.loc[~eligible, "clear_sky_attenuation_db"].isna().all()
    assert new.loc[~eligible, "received_power_dbm"].isna().all()
    assert set(high.hour_utc) == set(new.loc[eligible, "hour_utc"])
    assert audit.target_status.value_counts().to_dict() == {
        "detected_hour_quality0": 3434,
        "insufficient_raw_coverage": 340,
        "quality0_coverage_below_95pct": 95,
    }

    gain = manifest["equipment_gain_w_multiplier_after_boundary"]
    boundary = pd.Timestamp(manifest["boundary_bjt"])
    expected_w = measurements.received_power_quality0_w.to_numpy() * np.where(
        audit.hour_bjt >= boundary, gain, 1.0)
    expected_dbm = 10 * np.log10(expected_w[eligible] * 1000)
    np.testing.assert_allclose(audit.loc[eligible, "observed_power_dbm"], expected_dbm,
                               rtol=0, atol=1e-10)
    train_clear = eligible & audit.split_event_chrono_24h.eq("train").to_numpy() & audit.clear_dry.to_numpy()
    assert train_clear.sum() == 1430
    reference = float(np.median(expected_dbm[train_clear[eligible]]))
    assert abs(reference - manifest["reference_dbm"]) < 1e-10
    np.testing.assert_allclose(audit.loc[eligible, "clear_sky_attenuation_db"],
                               reference - expected_dbm, rtol=0, atol=1e-10)
    np.testing.assert_allclose(new.loc[eligible, "clear_sky_attenuation_db"],
                               audit.loc[eligible, "clear_sky_attenuation_db"], rtol=0, atol=1e-10)
    assert new.clear_reference_power_dbm.eq(reference).all()
    assert int((audit.measurement_high_coverage_95 & ~audit.signal_detected_hour).sum()) == 95
    split_counts = high.split_event_chrono_24h.value_counts().to_dict()
    assert split_counts == {"train": 2520, "validation": 408, "test": 434,
                            "embargo": 48, "snow_case": 24}
    report = {
        "status": "PASS",
        "all_measurement_hours": len(new),
        "high_detectability_hours": len(high),
        "newly_censored_from_previous_high_coverage": 95,
        "split_counts": split_counts,
        "reference_dbm_recomputed": reference,
        "meteorological_features_unchanged": len(feature_names),
        "max_abs_power_change_retained_db": float(np.max(np.abs(
            audit.loc[eligible, "observed_power_dbm"].to_numpy() -
            audit.loc[eligible, "previous_received_power_dbm_for_audit_only"].to_numpy()))),
        "note": "Numerical reproducibility only; not physical proof that every excluded row is noise or every retained row is signal.",
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
