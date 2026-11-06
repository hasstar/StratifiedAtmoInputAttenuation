"""Independent structural/arithmetic audit of three-source hourly labels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004"
STAGE = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_003"
LOCAL_MODEL = PROJECT / "Outputs/Datasets/Revision_006"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local() -> dict[str, int]:
    old = pd.read_parquet(STAGE / "local_hourly_power_audit.parquet")
    got = pd.read_parquet(OUT / "local_hourly_attenuation_audit.parquet")
    check(len(got) == len(old) == 3869, "Local count")
    check(got.hour_utc.equals(old.hour_utc), "Local order/time")
    check(np.array_equal(got.observed_power_dbm, old.observed_power_dbm), "Local power changed")
    ref_samples = old.loc[
        (old.split_event_chrono_24h == "train") & old.clear_dry
        & old.measurement_high_coverage_95 & (old.quality_zero_fraction >= .95),
        "observed_power_dbm"]
    reference = ref_samples.median()
    check(len(ref_samples) == 1430, "Local reference sample count")
    check(np.allclose(got.clear_reference_power_dbm, reference, rtol=0, atol=1e-12),
          "Local reference differs from train-only median")
    check(np.allclose(got.clear_sky_attenuation_db,
                      reference - old.observed_power_dbm, rtol=0, atol=1e-12),
          "Local attenuation formula")
    check(np.array_equal(got.clear_context_power_conflict,
                         old.clear_dry & ((reference - old.observed_power_dbm) > 2)),
          "Local clear-context conflict flags")
    original_model = pd.read_parquet(LOCAL_MODEL / "model_table_primary_p10_all_measurements.parquet")
    model = pd.read_parquet(OUT / "local_model_table_p10_attenuation_all_measurements.parquet")
    check(list(model.columns[:len(original_model.columns)]) == list(original_model.columns),
          "Local model original column layout changed")
    pd.testing.assert_frame_equal(model[original_model.columns], original_model,
                                  check_exact=True)
    check(np.allclose(model.clear_sky_attenuation_db,
                      got.set_index("hour_utc").loc[model.hour_utc,
                                                     "clear_sky_attenuation_db"].to_numpy(),
                      rtol=0, atol=1e-12), "Local model target join")
    high = pd.read_parquet(OUT / "local_model_table_p10_attenuation_high_coverage_95.parquet")
    check(len(high) == 3529, "Local high coverage count")
    check(high.hour_utc.isin(got.loc[got.measurement_high_coverage_95, "hour_utc"]).all(),
          "Local high coverage membership")
    return {"hours": len(got), "reference_samples": len(ref_samples),
            "high_coverage_hours": len(high),
            "clear_context_conflicts": int(got.clear_context_power_conflict.sum())}


def validate_opensat() -> dict[str, int]:
    old = pd.read_parquet(STAGE / "opensat_hourly_power.parquet")
    got = pd.read_parquet(OUT / "opensat_hourly_attenuation.parquet")
    check(len(got) == len(old) == 789480, "OpenSat count")
    pd.testing.assert_frame_equal(got[old.columns.drop(["clear_sky_attenuation_db", "target_status"])],
                                  old.drop(columns=["clear_sky_attenuation_db", "target_status"]),
                                  check_exact=True)
    check(np.allclose(got.clear_sky_attenuation_db,
                      got.clear_reference_power_dbm - got.observed_power_dbm,
                      equal_nan=True, rtol=0, atol=1e-12), "OpenSat attenuation formula")
    check(got.clear_reference_candidate.le(got.coverage_fraction >= .95).all(),
          "OpenSat candidate low coverage")
    candidate = got.loc[got.clear_reference_candidate]
    check((candidate.era5_total_cloud_cover <= .2).all(), "Candidate cloudy")
    check((candidate.era5_total_precip_mm_hour < .1).all(), "Candidate precipitation")
    check((candidate.era5_snowfall_cm_hour == 0.0).all(), "Candidate snowfall")
    check((candidate.radar_rain_mm_hour <= 1e-8).all(), "Candidate radar rain")
    labelled = got.clear_sky_attenuation_db.notna()
    check((got.loc[labelled, "clear_reference_candidate_count"] >= 20).all(),
          "Insufficient references labelled")
    check(set(got.loc[labelled, "clear_reference_radius_days"].unique()) <= {15, 30},
          "Unknown baseline radius")
    check((got.loc[~labelled, "target_status"] != "empirical_opensat_causal_radar_openmeteo_era5_reference").all(),
          "Unlabelled row claims estimated attenuation")
    check((got.weather_source == "openmeteo_models_era5_rounded_proxy").all(),
          "Unrecorded OpenSat weather provenance")
    check((got.loc[got.observed_power_dbm.notna(), "target_status"]
           != "no_power_observation").all(), "Power row marked missing")
    check((got.era5_total_cloud_cover.dropna().between(0, 1)).all(), "ERA5 cloud range")
    check((got.era5_total_precip_mm_hour.dropna() >= -0.001).all(), "ERA5 precipitation range")
    # Refit 90 deterministic link/hour samples from their stored candidate pool,
    # independently of the builder's loop and search implementation.
    checked = 0
    for link_id in got.link_id.drop_duplicates().iloc[::23]:
        group = got.loc[got.link_id == link_id].reset_index(drop=True)
        cand = np.flatnonzero(group.clear_reference_candidate.to_numpy())
        powers = group.observed_power_dbm.to_numpy()
        for hour in range(48, len(group), 367):
            row = group.iloc[hour]
            if not np.isfinite(row.clear_sky_attenuation_db):
                continue
            rad = int(row.clear_reference_radius_days) * 24
            use = cand[(cand >= hour - rad) & (cand < hour)]
            d = np.abs((use % 24) - (hour % 24))
            use = use[np.minimum(d, 24 - d) <= 2]
            check(len(use) == int(row.clear_reference_candidate_count), "OpenSat candidate count")
            check(np.isclose(np.quantile(powers[use], .75), row.clear_reference_power_dbm,
                             rtol=0, atol=1e-10), "OpenSat reference quantile")
            check(np.isclose(np.quantile(powers[use], .75) - np.quantile(powers[use], .25),
                             row.clear_reference_candidate_iqr_db, rtol=0, atol=1e-10),
                  "OpenSat reference IQR")
            checked += 1
    check(checked >= 30, "Too few independent OpenSat reference samples")
    station = pd.read_csv(OUT / "opensat_station_summary.csv")
    check(len(station) == 215, "OpenSat station summary")
    check(station.attenuation_hours.sum() == labelled.sum(), "OpenSat summary count")
    return {"station_hours": len(got), "power_hours": int(got.observed_power_dbm.notna().sum()),
            "labelled_hours": int(labelled.sum()), "clear_candidates": len(candidate),
            "reference_spot_checks": checked,
            "stations_without_labels": int((station.attenuation_hours == 0).sum()),
            "clear_context_conflicts": int(got.clear_context_power_conflict.sum())}


def validate_ceda() -> dict[str, int]:
    old = pd.read_parquet(STAGE / "ceda_gbs_hourly_attenuation.parquet")
    got = pd.read_parquet(OUT / "ceda_gbs_hourly_attenuation.parquet")
    check(len(got) == len(old) == 47568, "CEDA count")
    pd.testing.assert_frame_equal(got[old.columns], old, check_exact=True)
    check(got.clear_sky_attenuation_db.notna().sum() == 46905, "CEDA native count")
    check((got.loc[got.clear_sky_attenuation_db.notna(), "target_status"]
           == "native_ceda_gbs").all(), "CEDA native provenance")
    return {"station_hours": len(got),
            "native_labels": int(got.clear_sky_attenuation_db.notna().sum())}


def main() -> None:
    manifest = json.loads((OUT / "build_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["input_sha256"].items():
        check(sha256(PROJECT / filename) == expected, f"Input SHA256 mismatch: {filename}")
    for filename, expected in manifest["output_sha256"].items():
        check(sha256(OUT / filename) == expected, f"Output SHA256 mismatch: {filename}")
    results = {"local": validate_local(), "opensat": validate_opensat(),
               "ceda_gbs": validate_ceda()}
    report = {"status": "VERIFIED_ARITHMETIC_AND_PROVENANCE_ONLY",
              "verified_utc": datetime.now(timezone.utc).isoformat(),
              "checks": results,
              "limitation": "Physical clear-sky validity and hardware/thermal effects of estimated local and OpenSat labels are not proven by arithmetic QA."}
    path = OUT / "validation_report.json"
    check(not path.exists(), "Refusing overwrite of existing validation report")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
