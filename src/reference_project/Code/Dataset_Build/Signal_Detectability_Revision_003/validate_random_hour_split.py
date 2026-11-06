"""Independent checks for the 70/20/10 weather-stratified hourly split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_003"
SOURCE = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_001"
SPLITS = ("train", "test", "validation")
TARGET = "clear_sky_attenuation_db"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        assert sha256(PROJECT / relative) == expected, f"Source hash changed: {relative}"
    for name, expected in manifest["output_sha256"].items():
        assert sha256(OUT / name) == expected, f"Output hash changed: {name}"

    audit = pd.read_parquet(SOURCE / "local_hourly_detectability_audit.parquet")
    prior = pd.read_parquet(SOURCE / "local_model_table_p10_detected95.parquet")
    features = pd.read_csv(OUT / "feature_columns.csv")
    assert features.feature_order.tolist() == list(range(1, 349))
    columns = features.column.tolist()
    assert len(columns) == len(set(columns)) == 348
    schema = ["hour_bjt", "hour_utc", *columns, TARGET]
    expected_counts = {"train": 2404, "test": 687, "validation": 343}

    assignment = pd.read_csv(OUT / "hourly_assignments.csv", parse_dates=["hour_utc"])
    assignment.hour_utc = pd.to_datetime(assignment.hour_utc, utc=True)
    assert len(assignment) == 3434 and assignment.hour_utc.is_unique
    assert set(assignment.hour_utc) == set(prior.hour_utc)
    assert assignment.split.value_counts().to_dict() == expected_counts
    assert assignment.weather_stratum.value_counts().to_dict() == manifest["weather_strata"]

    joined = audit.merge(assignment[["hour_utc", "split", "weather_stratum"]],
                         on="hour_utc", how="left", validate="one_to_one")
    assert joined.loc[joined.signal_detected_hour, "split"].notna().all()
    assert joined.loc[~joined.signal_detected_hour, "split"].isna().all()
    eligibility = joined.loc[joined.signal_detected_hour].copy()
    expected_class = np.select(
        [eligibility.snow, eligibility.heavy_rain, eligibility.rain,
         eligibility.dry_overcast, eligibility.clear_dry],
        ["snow", "heavy_rain", "rain_nonheavy", "dry_overcast", "clear_dry"],
        default="other",
    )
    assert (eligibility.weather_stratum.to_numpy() == expected_class).all()
    assert eligibility.loc[eligibility.split.eq("train") & eligibility.clear_dry].shape[0] == 1127
    reference = eligibility.loc[eligibility.split.eq("train") & eligibility.clear_dry,
                                "hourly_power_quality0_corrected_dbm"].median()
    assert np.isclose(reference, manifest["clear_reference_dbm"], atol=1e-12)

    quota = pd.read_csv(OUT / "weather_stratum_quotas.csv")
    for _, row in quota.iterrows():
        group = eligibility.loc[eligibility.weather_stratum.eq(row.weather_stratum)]
        assert len(group) == row.total_hours
        for split in SPLITS:
            assert int(group.split.eq(split).sum()) == int(row[split])

    # Reproduce the documented random seed and within-stratum shuffle exactly.
    replay = audit.loc[audit.signal_detected_hour].sort_values("hour_utc").copy()
    replay["weather_stratum"] = np.select(
        [replay.snow, replay.heavy_rain, replay.rain,
         replay.dry_overcast, replay.clear_dry],
        ["snow", "heavy_rain", "rain_nonheavy", "dry_overcast", "clear_dry"],
        default="other",
    )
    replay["expected_split"] = ""
    rng = np.random.default_rng(int(manifest["random_seed"]))
    for name in ("clear_dry", "dry_overcast", "other", "rain_nonheavy", "heavy_rain", "snow"):
        indices = replay.index[replay.weather_stratum.eq(name)].to_numpy().copy()
        rng.shuffle(indices)
        counts = quota.loc[quota.weather_stratum.eq(name), list(SPLITS)].iloc[0]
        start = 0
        for split in SPLITS:
            count = int(counts[split])
            replay.loc[indices[start:start + count], "expected_split"] = split
            start += count
    replay = replay[["hour_utc", "expected_split"]].merge(
        assignment[["hour_utc", "split"]], on="hour_utc", validate="one_to_one")
    assert replay.expected_split.eq(replay.split).all()

    summary = pd.read_csv(OUT / "split_event_summary.csv")
    missing = pd.read_csv(OUT / "feature_missingness.csv")
    assert len(missing) == 348 * 3
    tested_rows = []
    for split in SPLITS:
        actual = pd.read_parquet(OUT / f"{split}.parquet")
        expected = prior.merge(assignment[["hour_utc", "split"]], on="hour_utc",
                               validate="one_to_one")
        expected = expected.loc[expected.split.eq(split), ["hour_bjt", "hour_utc", *columns,
                                                           "received_power_dbm"]].sort_values(
                                                               "hour_utc").reset_index(drop=True)
        assert actual.columns.tolist() == schema
        assert len(actual) == expected_counts[split]
        assert actual.hour_utc.is_unique and np.isfinite(actual[TARGET]).all()
        pd.testing.assert_frame_equal(actual[["hour_bjt", "hour_utc", *columns]],
                                      expected[["hour_bjt", "hour_utc", *columns]])
        np.testing.assert_allclose(actual[TARGET].to_numpy(),
                                   reference - expected.received_power_dbm.to_numpy(),
                                   rtol=0, atol=2e-12)
        part = eligibility.loc[eligibility.split.eq(split)]
        line = summary.loc[summary.split.eq(split)].iloc[0]
        assert int(line.hours) == len(actual)
        for event in ("rain", "heavy_rain", "dry_overcast", "clear_dry", "snow"):
            assert int(line[event]) == int(part[event].sum())
        per_feature = missing.loc[missing.split.eq(split)]
        assert per_feature.feature.tolist() == columns
        np.testing.assert_array_equal(per_feature.missing_count.to_numpy(),
                                      actual[columns].isna().sum().to_numpy())
        tested_rows.append(len(actual))
    assert sum(tested_rows) == 3434
    assert not pd.read_parquet(OUT / "train.parquet")[columns].isna().all().any()

    status = pd.read_parquet(OUT / "hourly_sample_status.parquet")
    assert len(status) == 3869 and status.hour_utc.is_unique
    assert TARGET not in status.columns
    assert int(status.use_for_regression.sum()) == 3434
    assert status.loc[~status.use_for_regression, "split"].eq("excluded_unreliable").all()
    excluded = pd.read_csv(OUT / "excluded_unreliable_hours.csv")
    assert len(excluded) == 435 and not excluded.use_for_regression.any()
    assert audit.loc[~audit.signal_detected_hour, TARGET].isna().all()

    report = {"status": "PASS", "hours": expected_counts, "total_model_hours": 3434,
              "weather_event_counts": summary.to_dict(orient="records"),
              "excluded_unreliable_hours": 435,
              "train_only_clear_reference_dbm": float(reference),
              "source_and_output_hashes_verified": True,
              "all_348_features_and_targets_checked_against_source": True,
              "weather_strata_quotas_and_seed_reproduced": True,
              "events_and_quality_not_in_model_inputs": True}
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
