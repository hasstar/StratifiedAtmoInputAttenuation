"""Independently check packaged local model tables against frozen source rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_002"
SOURCE = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_001"
TARGET = "clear_sky_attenuation_db"
SPLIT = "split_event_chrono_24h"
NAMES = ("train", "validation", "test", "embargo", "snow_case")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def validate() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        assert digest(PROJECT / relative) == expected, f"Source changed: {relative}"
    for name, expected in manifest["output_sha256"].items():
        assert digest(OUT / name) == expected, f"Output changed: {name}"

    original = pd.read_parquet(SOURCE / "local_model_table_p10_detected95.parquet")
    audit = pd.read_parquet(SOURCE / "local_hourly_detectability_audit.parquet")
    feature_catalog = pd.read_csv(OUT / "feature_columns.csv")
    features = feature_catalog.column.tolist()
    assert feature_catalog.feature_order.tolist() == list(range(1, 349))
    assert len(features) == len(set(features)) == 348
    expected_columns = ["hour_bjt", "hour_utc", *features, TARGET]
    counts = {}
    for split in NAMES:
        name = f"{split}.parquet" if split in NAMES[:3] else f"reserved_{split}.parquet"
        actual = pd.read_parquet(OUT / name)
        expected = original.loc[original[SPLIT].eq(split), expected_columns].sort_values(
            "hour_utc").reset_index(drop=True)
        assert actual.columns.tolist() == expected_columns, f"Incorrect schema: {name}"
        pd.testing.assert_frame_equal(actual, expected)
        assert actual.hour_utc.is_unique and np.isfinite(actual[TARGET]).all()
        assert not np.isinf(actual[features].to_numpy(dtype=float)).any()
        counts[split] = len(actual)
    assert counts == manifest["split_rows"]
    assert sum(counts[s] for s in NAMES[:3]) == 3362

    status = pd.read_parquet(OUT / "hourly_sample_status.parquet")
    assert len(status) == len(audit) == 3869
    assert status.hour_utc.is_unique and TARGET not in status.columns
    expected_flags = (audit.target_status.eq("detected_hour_quality0")
                      & audit[SPLIT].isin(NAMES[:3]))
    flags = audit[["hour_utc"]].assign(expected=expected_flags).merge(
        status[["hour_utc", "use_for_regression"]], on="hour_utc", validate="one_to_one")
    assert flags.expected.equals(flags.use_for_regression)
    assert int(status.use_for_regression.sum()) == 3362
    assert int((~status.use_for_regression).sum()) == 507
    excluded = pd.read_csv(OUT / "excluded_or_reserved_hours.csv")
    assert len(excluded) == 507
    assert not excluded.use_for_regression.any()
    assert status.exclusion_reason.eq("").equals(status.use_for_regression)
    assert status.loc[status.use_for_regression, "quality_zero_records"].ge(34200).all()
    assert (status.loc[status.use_for_regression, "target_status"]
            == "detected_hour_quality0").all()
    assert audit.loc[~audit.signal_detected_hour, TARGET].isna().all()
    assert not original.loc[original[SPLIT].eq("train"), features].isna().all().any()

    missing = pd.read_csv(OUT / "feature_missingness.csv")
    assert len(missing) == 348 * 5
    for split in NAMES:
        actual = pd.read_parquet(OUT / (f"{split}.parquet" if split in NAMES[:3]
                                        else f"reserved_{split}.parquet"))
        expected_missing = actual[features].isna().sum().to_numpy()
        observed_missing = missing.loc[missing.split.eq(split), "missing_count"].to_numpy()
        np.testing.assert_array_equal(observed_missing, expected_missing)

    report = {
        "status": "PASS",
        "checked_model_rows": counts,
        "official_regression_rows": 3362,
        "excluded_or_reserved_rows": 507,
        "censored_rows": int(audit.target_status.ne("detected_hour_quality0").sum()),
        "feature_count": len(features),
        "train_all_null_feature_count": 0,
        "source_and_output_hashes_verified": True,
        "every_split_exactly_matches_source_values": True,
        "events_and_quality_are_not_in_predictor_tables": True,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
