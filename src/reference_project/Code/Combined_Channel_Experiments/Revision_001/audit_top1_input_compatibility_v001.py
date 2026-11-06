"""Read-only feature/target provenance audit; never load or execute a model.

Records a spatial input mismatch instead of silently substituting nearest-cell
geopotential for the saved model's line-of-sight interpolated feature.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
from audit_top1_available_fields_v001 import DEST, PLAN_SHA, ROOT, read, require, sha

ERA5_CSV = Path(r"E:\0_Workflow_System\01_PhD\Data\era5_campaign_20250930_20260820\csv")
LOS_CODE = Path(r"E:\0_Workflow_System\01_PhD\Papers\3_TAP_被拒\scripts\prepare_era5_los_csv.py")
MODEL_DIR = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE"
ALIGNMENT = ROOT / "Outputs/Datasets/ClearSky_Alignment_Revision_004"
FEATURE = "p1__geopotential"


def east_longitude(value, units):
    require(units in {"degrees_east", "degrees_west"}, "Unknown longitude convention")
    signed = -float(value) if units == "degrees_west" else float(value)
    return (signed + 180.0) % 360.0 - 180.0


def build_report():
    sources = {}

    def source(path):
        path = Path(path)
        sources[str(path)] = sha(path)
        return path

    source(Path(__file__))
    source(Path(__file__).with_name("audit_top1_available_fields_v001.py"))
    plan_path = source(DEST / "plan.json")
    require(sha(plan_path) == PLAN_SHA, "Frozen Top1 plan changed")
    plan = read(plan_path)
    fields = read(source(DEST / "field_audit_001/qa.json"))
    require(fields["status"] == "PASS_REUSED_TOP1_FIELDS_ONLY", "Field audit not passed")
    for path, digest in fields["source_sha256"].items():
        require(sha(source(path)) == digest, "Field audit source changed")
    inventory = read(source(DEST / "model_inventory.json"))
    require(inventory["feature_names"] == [FEATURE] and inventory["feature_count"] == 1,
            "Not the existing one-feature model")
    for rel, digest in inventory["source_sha256"].items():
        require(sha(source(ROOT / rel)) == digest, "Existing model/preprocessing source changed")
    metadata = read(source(ERA5_CSV / "feature_metadata.json"))
    require(metadata["pressure_features"]["geopotential"]["units"] == "m**2 s**-2",
            "Local feature unit differs")
    sampling = metadata["pressure_interpolation"]
    require("intersection" in sampling and "bilinear" in sampling,
            "Local feature definition changed")
    source(LOS_CODE)
    source(ROOT / "Code/Dataset_Build/Revision_001/build_matched_dataset.py")
    los = pd.read_csv(source(ERA5_CSV / "era5_pressure_1hPa_los_hourly_bjt.csv"),
                      usecols=["valid_time_utc", "geopotential", "los_latitude_deg_n",
                               "los_longitude_deg_e", "los_horizontal_displacement_m"])
    los["hour_utc"] = pd.to_datetime(los["valid_time_utc"], utc=True)
    require(not los["hour_utc"].duplicated().any(), "Duplicate local source hour")
    manifest = read(source(MODEL_DIR / "run_manifest.json"))
    local_checks = []
    for split, expected_rows in (("train", 2404), ("validation", 343), ("test", 687)):
        rel = f"Outputs\\Datasets\\Signal_Detectability_Revision_003\\{split}.parquet"
        path = source(ROOT / rel)
        require(sha(path) == manifest["source_sha256"][rel], "Actual model dataset SHA changed")
        frame = pd.read_parquet(path, columns=["hour_utc", FEATURE])
        require(len(frame) == expected_rows, "Frozen model dataset rows changed")
        joined = frame.merge(los, on="hour_utc", validate="one_to_one")
        require(len(joined) == len(frame), "Model feature missing original LOS source")
        difference = np.abs(joined[FEATURE].to_numpy() - joined["geopotential"].to_numpy())
        require(np.isfinite(difference).all() and float(difference.max()) == 0.0,
                "Model feature not identical to original LOS geopotential")
        local_checks.append({
            "historical_split": split, "rows": len(frame), "feature_source_max_abs_difference": 0.0,
            "los_latitude_range": [float(joined.los_latitude_deg_n.min()), float(joined.los_latitude_deg_n.max())],
            "los_longitude_range": [float(joined.los_longitude_deg_e.min()), float(joined.los_longitude_deg_e.max())],
            "horizontal_displacement_m_range": [float(joined.los_horizontal_displacement_m.min()),
                                                 float(joined.los_horizontal_displacement_m.max())],
            "predictions_executed": 0})
    external = read(source(DEST.parent / "metadata.json"))
    gbs = pd.read_parquet(source(ALIGNMENT / "ceda_gbs_hourly_attenuation.parquet"))
    require(not gbs.duplicated(["site", "hour_utc"]).any(), "GBS duplicate station-hour")
    source(ALIGNMENT / "README.md")
    source(ALIGNMENT / "build_manifest.json")
    source(ALIGNMENT / "validation_report.json")
    covered = {t for row in fields["checks"] if row["parent_id"].startswith("GBS_England_")
               for t in row["utc"]}
    covered_times = pd.to_datetime(sorted(covered), utc=True)
    expected_grid = fields["checks"][0]
    require(all(r["latitude"] == expected_grid["latitude"] and r["longitude"] == expected_grid["longitude"]
                for r in fields["checks"]), "This partial audit expects one common England grid")
    require(len(expected_grid["longitude"]) == 1, "Reassess compatibility if grid changes")
    site_checks = []
    for site in ("chilbolton", "sparsholt", "dundee"):
        sm = next(s for s in external["GBS"]["sites"] if s["site"] == site)
        c = sm["scalar_coordinates"]
        lon_unit = sm["variables"]["longitude_receiver"]["attributes"]["units"]
        sat_unit = sm["variables"]["satellite_position"]["attributes"]["units"]
        lon = east_longitude(c["longitude_receiver"], lon_unit)
        matched = gbs[gbs["site"].eq(site) & gbs["hour_utc"].isin(covered_times)].copy()
        if site == "dundee":
            matched = matched.iloc[:0]  # England weather must not be assigned to Dundee.
        valid = np.isfinite(matched["clear_sky_attenuation_db"]) & matched["target_status"].eq("native_ceda_gbs")
        quality = valid & matched["coverage_ge_95"]
        site_checks.append({
            "site": site, "receiver_latitude_deg_n": c["latitude_receiver"],
            "receiver_longitude_deg_e": lon, "source_longitude_units": lon_unit,
            "satellite_longitude_deg_e": east_longitude(c["satellite_position"], sat_unit),
            "source_satellite_longitude_units": sat_unit, "frequency_ghz": c["frequency"],
            "available_weather_hours": 0 if site == "dundee" else len(covered),
            "existing_target_overlap_rows": len(matched), "finite_native_target_rows": int(valid.sum()),
            "native_target_coverage_ge_95_rows": int(quality.sum()),
            "first_overlap_utc": None if matched.empty else matched["hour_utc"].min().isoformat(),
            "last_overlap_utc": None if matched.empty else matched["hour_utc"].max().isoformat(),
            "legal_same_definition_feature_rows": 0,
            "status": "WAITING_WEATHER_AND_SPATIAL_COMPATIBILITY" if site == "dundee" else "BLOCKED_SPATIAL_INPUT_DEFINITION",
        })
    open_path = source(ALIGNMENT / "opensat_hourly_attenuation.parquet")
    open_schema = pq.ParquetFile(open_path)
    require({"hour_utc", "clear_sky_attenuation_db", "target_status"}.issubset(open_schema.schema_arrow.names),
            "OpenSat existing label schema changed")
    for path, digest in sources.items():
        require(sha(path) == digest, "Audit source changed during read: " + path)
    return {
        "status": "BLOCKED_SPATIAL_INPUT_DEFINITION", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "feature": FEATURE, "checkpoint_sha256": inventory["checkpoint_sha256"],
        "checkpoint_and_frozen_preprocessing": "PASS_EXISTING_ONLY", "units_and_pressure_level": "PASS",
        "local_feature_definition": sampling, "local_actual_model_source_checks": local_checks,
        "original_local_feature_units": "m**2 s**-2", "model_time_scale": "hourly",
        "external_available_grid": {"latitude": expected_grid["latitude"], "longitude": expected_grid["longitude"],
                                    "minimum_axes_for_original_bilinear_implementation": [2, 2]},
        "spatial_gap": "England payloads have two latitudes and one longitude. Original LOS bilinear interpolation requires nonzero latitude and longitude spans. No same-definition LOS feature can be constructed from this one-column grid without changing the sampling rule or adding spatial support.",
        "GBS": site_checks,
        "GBS_target": "Existing hourly native relative-clear-sky attenuation uses mean linear transmittance; no reference refit. Target unit/time is compatible in definition, but this does not resolve the input spatial mismatch or cross-band transfer.",
        "OpenSat": {"existing_hourly_label_rows": open_schema.metadata.num_rows, "available_top1_weather_hours": 0,
                    "status": "WAITING_WEATHER_AND_FULL_COMPATIBILITY",
                    "minute_averaging_domain": "Unknown W versus dB; no fabricated second-scale truth",
                    "existing_reference": "Read-only empirical causal radar/Open-Meteo ERA5 reference; not new fitting or a new weather source",
                    "spatial_compatibility": "France planned grid has both axes, but ray-intersection coverage still needs per-link verification; not approved by this audit"},
        "weather_grouping": "Not established here; no new cloud/rain/snow downloads authorized",
        "fast_component_compatibility": "NOT_ESTABLISHED; no CDF fit or seconds/minutes conversion",
        "required_user_decision": [
            "Authorize a separately labelled nearest-grid approximation trial using the existing k=1 checkpoint, without treating it as same-definition external validation",
            "Authorize an independently planned spatial extent adjustment for the same single field to support original LOS interpolation; dates/features/models remain unchanged"],
        "decision_applied": False, "safe_download_continues": True, "download_scope_changed": False,
        "new_fit_calls": 0, "model_loads": 0, "model_predictions": 0, "external_metrics_generated": False,
        "old_results_changed": False, "overall_task_complete": False,
        "source_sha256": sources, "source_files_unchanged_after_read": True,
    }


if __name__ == "__main__":
    report = build_report()
    out = DEST / "input_compatibility_001.json"
    require(not out.exists(), "Immutable compatibility audit exists; use a new version")
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "GBS", "model_predictions", "new_fit_calls")}, indent=2))
