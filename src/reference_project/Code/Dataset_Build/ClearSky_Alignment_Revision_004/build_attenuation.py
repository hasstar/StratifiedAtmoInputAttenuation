"""Build source-specific hourly relative-to-clear-sky attenuation labels.

This is a retrospective empirical *target construction*, not a propagation
model. Local and OpenSat references are estimated and explicitly flagged;
CEDA's native reference is preserved. Never edit the underlying observations.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
STAGE = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_003"
LOCAL_MODEL = PROJECT / "Outputs/Datasets/Revision_006"
OPENSAT_RAW = PROJECT / "Outputs/Datasets/External_OpenSat4Weather_001/extracted"
OUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004"
ERA5 = OUT / "era5_france"
PROXY = OUT / "openmeteo_era5_proxy"
MIN_BASELINE_N = 20
NEAR_DAYS = 15
FAR_DAYS = 30
HOUR_TOLERANCE = 2
OPEN_BASELINE_QUANTILE = 0.75


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(frame: pd.DataFrame, path: Path) -> None:
    require(not path.exists(), f"Refusing overwrite: {path}")
    frame.to_parquet(path, index=False)


def read_zipped_era5(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require(path.is_file(), f"Missing ERA5 month: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(all("/" not in x and "\\" not in x for x in names), "Unsafe ERA5 ZIP member")
        matches = [x for x in names if variable in x or x.endswith(".nc")]
        result = None
        for name in matches:
            with h5py.File(io.BytesIO(archive.read(name)), "r") as source:
                if variable not in source:
                    continue
                values = np.asarray(source[variable][:], dtype=np.float64)
                times = np.asarray(source["valid_time"][:], dtype=np.int64)
                lat = np.asarray(source["latitude"][:], dtype=np.float64)
                lon = np.asarray(source["longitude"][:], dtype=np.float64)
                require(values.shape == (len(times), len(lat), len(lon)),
                        f"ERA5 variable shape mismatch: {path.name}/{variable}")
                units = source[variable].attrs.get("units")
                expected = b"(0 - 1)" if variable == "tcc" else (
                    b"m of water equivalent" if variable == "sf" else b"m")
                require(units == expected, f"Unexpected ERA5 units: {variable}/{units!r}")
                result = (times, lat, lon, values)
                break
        require(result is not None, f"No ERA5 {variable} in {path.name}")
        return result


def match_era5(stations: pd.DataFrame, hours: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    canonical_grid = None
    for variable in ("tcc", "tp", "sf"):
        all_times: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        for month in range(8, 13):
            path = ERA5 / f"era5_france_2022{month:02d}.zip"
            times, lat, lon, values = read_zipped_era5(path, variable)
            grid = (lat.tobytes(), lon.tobytes())
            if canonical_grid is None:
                canonical_grid = grid
            require(grid == canonical_grid, "ERA5 monthly grid changed")
            lat_idx = np.abs(lat[:, None] - stations.site_lat_deg.to_numpy()[None, :]).argmin(axis=0)
            lon_idx = np.abs(lon[:, None] - stations.site_lon_deg.to_numpy()[None, :]).argmin(axis=0)
            all_times.append(times)
            all_values.append(values[:, lat_idx, lon_idx])
        merged_times = np.concatenate(all_times)
        merged_values = np.concatenate(all_values, axis=0)
        require(np.all(np.diff(merged_times) == 3600), f"ERA5 {variable} time gap")
        combined[variable] = (merged_times, merged_values)
    hour_seconds = hours.asi8 // 10**9
    result: dict[str, np.ndarray] = {}
    for variable in ("tcc", "tp", "sf"):
        times, values = combined[variable]
        # ERA5 accumulated tp/sf at t+1 describe the measurement hour [t,t+1).
        requested = hour_seconds + (3600 if variable in ("tp", "sf") else 0)
        indexes = np.searchsorted(times, requested)
        found = indexes < len(times)
        found[found] &= times[indexes[found]] == requested[found]
        matched = np.full((len(hours), len(stations)), np.nan)
        matched[found] = values[indexes[found]]
        result[variable] = matched * (1000.0 if variable in ("tp", "sf") else 1.0)
    result["grid_lat"] = lat[np.abs(lat[:, None] - stations.site_lat_deg.to_numpy()[None, :]).argmin(axis=0)]
    result["grid_lon"] = lon[np.abs(lon[:, None] - stations.site_lon_deg.to_numpy()[None, :]).argmin(axis=0)]
    return result


def match_era5_proxy(stations: pd.DataFrame, hours: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Read archived, explicitly ERA5-selected Open-Meteo values at 0.25° cells.

    The provider rounds cloud to integer percent and rain to 0.1 mm. These
    columns must never be represented as byte-identical official CDS data.
    """
    comparison = json.loads((OUT / "weather_proxy_comparison.json").read_text(encoding="utf-8"))
    require(comparison["compared_cell_hours"] == 1320, "Missing one-day CDS/proxy audit")
    records: dict[tuple[float, float], dict[str, object]] = {}
    for batch in range(6):
        payload = json.loads((PROXY / f"batch_{batch:02d}.json").read_text(encoding="utf-8"))
        for coordinates, response in zip(payload["request"]["coordinates"], payload["response"]):
            key = (float(coordinates[0]), float(coordinates[1]))
            require(key not in records, f"Duplicate weather cell: {key}")
            records[key] = response
    require(len(records) == 55, "Incomplete French weather proxy")
    lat = np.rint(stations.site_lat_deg.to_numpy() * 4) / 4
    lon = np.rint(stations.site_lon_deg.to_numpy() * 4) / 4
    result = {name: np.full((len(hours), len(stations)), np.nan)
              for name in ("tcc", "tp", "snow_cm")}
    expected_hours = hours.tz_convert("UTC").tz_localize(None).strftime("%Y-%m-%dT%H:%M").tolist()
    for index, key in enumerate(zip(lat, lon)):
        require(key in records, f"Missing weather cell: {key}")
        hourly = records[key]["hourly"]
        require(hourly["time"] == expected_hours, f"Weather time mismatch: {key}")
        cloud = np.asarray(hourly["cloud_cover"], dtype=np.float64)
        rain = np.asarray(hourly["precipitation"], dtype=np.float64)
        snow = np.asarray(hourly["snowfall"], dtype=np.float64)
        require((len(cloud), len(rain), len(snow)) == (len(hours),) * 3,
                f"Weather length mismatch: {key}")
        result["tcc"][:, index] = cloud / 100.0
        # The accumulated value timestamped t+1 describes [t,t+1).
        result["tp"][:-1, index] = rain[1:]
        result["snow_cm"][:-1, index] = snow[1:]
    result["grid_lat"] = lat
    result["grid_lon"] = lon
    return result


def match_radar(stations: pd.DataFrame, hours: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    with h5py.File(OPENSAT_RAW / "radar_along_sml_data_2022.nc", "r") as source:
        ids = [x.decode("utf-8") for x in source["sml_id"][:]]
        require(ids == stations.link_id.tolist(), "Radar and OpenSat link order mismatch")
        require(np.array_equal(source["time"][:], np.arange(len(hours) * 12) * 5),
                "Radar time grid mismatch")
        raw = source["rainfall_amount"][:]
        fill = int(source["rainfall_amount"].attrs["_FillValue"][0])
        scale = float(source["rainfall_amount"].attrs["scale_factor"][0])
    require(raw.shape == (len(hours) * 12, len(stations)), "Radar shape mismatch")
    valid = raw != fill
    rainfall = np.where(valid, raw * scale, 0.0).reshape(len(hours), 12, len(stations))
    valid_hour = valid.reshape(len(hours), 12, len(stations)).sum(axis=1)
    sum_hour = rainfall.sum(axis=1)
    cumulative = np.vstack([np.zeros((1, len(stations))), np.cumsum(sum_hour, axis=0)])
    valid_cumulative = np.vstack([np.zeros((1, len(stations)), dtype=np.int64),
                                  np.cumsum(valid_hour, axis=0)])
    sum_4h = np.full_like(sum_hour, np.nan)
    valid_4h = np.zeros_like(valid_hour)
    sum_4h[3:] = cumulative[4:] - cumulative[:-4]
    valid_4h[3:] = valid_cumulative[4:] - valid_cumulative[:-4]
    return {"rain_mm": np.where(valid_hour == 12, sum_hour, np.nan),
            "valid_5min_bins": valid_hour,
            "dry_4h": (valid_4h == 48) & (sum_4h <= 1e-8)}


def reference_for_link(power: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_hours = len(power)
    cand_idx = np.flatnonzero(candidate & np.isfinite(power))
    reference = np.full(n_hours, np.nan)
    count = np.zeros(n_hours, dtype=np.int32)
    radius = np.zeros(n_hours, dtype=np.int8)
    iqr = np.full(n_hours, np.nan)
    if len(cand_idx) < MIN_BASELINE_N:
        return reference, count, radius, iqr
    for hour in np.flatnonzero(np.isfinite(power)):
        for days, code in ((NEAR_DAYS, 15), (FAR_DAYS, 30)):
            boundary = days * 24
            start = np.searchsorted(cand_idx, hour - boundary, side="left")
            # Causal calibration: no target or future signal is used.
            stop = np.searchsorted(cand_idx, hour, side="left")
            matched = cand_idx[start:stop]
            distance = np.abs((matched % 24) - (hour % 24))
            matched = matched[np.minimum(distance, 24 - distance) <= HOUR_TOLERANCE]
            if len(matched) >= MIN_BASELINE_N:
                lower, upper = np.quantile(power[matched], [.25, OPEN_BASELINE_QUANTILE])
                reference[hour] = upper
                iqr[hour] = upper - lower
                count[hour] = len(matched)
                radius[hour] = code
                break
    return reference, count, radius, iqr


def make_local() -> dict[str, object]:
    staged = pd.read_parquet(STAGE / "local_hourly_power_audit.parquet")
    require(len(staged) == 3869 and staged.hour_utc.is_unique, "Local staging changed")
    train_clear = staged.loc[
        (staged.split_event_chrono_24h == "train")
        & staged.clear_dry
        & staged.measurement_high_coverage_95
        & (staged.quality_zero_fraction >= 0.95), "observed_power_dbm"
    ]
    require(len(train_clear) >= 100, "Insufficient local train-only clear candidates")
    reference = float(train_clear.median())
    staged["clear_reference_power_dbm"] = reference
    staged["clear_reference_candidate_count"] = int(len(train_clear))
    staged["clear_reference_method"] = "train_clear_dry_quality_median_fixed"
    staged["clear_sky_attenuation_db"] = reference - staged.observed_power_dbm
    staged["target_status"] = "empirical_local_train_reference"
    staged["clear_context_power_conflict"] = (
        staged.clear_dry & (staged.clear_sky_attenuation_db > 2.0)
    )
    staged["coverage_ge_95"] = staged.measurement_high_coverage_95
    write_new(staged, OUT / "local_hourly_attenuation_audit.parquet")

    model = pd.read_parquet(LOCAL_MODEL / "model_table_primary_p10_all_measurements.parquet")
    require(len(model) == len(staged) and model.hour_utc.is_unique, "Local model table changed")
    supplement = staged[["hour_utc", "clear_sky_attenuation_db",
                         "clear_reference_power_dbm", "clear_reference_candidate_count",
                         "clear_reference_method", "target_status",
                         "clear_context_power_conflict"]]
    model = model.merge(supplement, on="hour_utc", how="left", validate="one_to_one")
    require(model.clear_sky_attenuation_db.notna().all(), "Local model attenuation mismatch")
    write_new(model, OUT / "local_model_table_p10_attenuation_all_measurements.parquet")
    high_hours = staged.loc[staged.measurement_high_coverage_95, "hour_utc"]
    high = model.loc[model.hour_utc.isin(high_hours)].copy()
    require(len(high) == 3529, "Local high-coverage count changed")
    write_new(high, OUT / "local_model_table_p10_attenuation_high_coverage_95.parquet")
    return {"hours": len(staged), "high_coverage_hours": len(high),
            "train_clear_reference_hours": len(train_clear),
            "reference_power_dbm": reference,
            "clear_context_power_conflict_hours": int(staged.clear_context_power_conflict.sum())}


def make_opensat() -> dict[str, object]:
    staged = pd.read_parquet(STAGE / "opensat_hourly_power.parquet")
    hours = pd.DatetimeIndex(staged.hour_utc.drop_duplicates().sort_values())
    stations = staged[["link_id", "site_lat_deg", "site_lon_deg"]].drop_duplicates()
    n_hours, n_links = len(hours), len(stations)
    require((n_hours, n_links) == (3672, 215), "OpenSat dimensions changed")
    require(staged.link_id.tolist() == np.repeat(stations.link_id.to_numpy(), n_hours).tolist(),
            "OpenSat order changed")
    require(np.array_equal(staged.hour_utc.to_numpy().reshape(n_links, n_hours)[0],
                           hours.to_numpy()), "OpenSat hour order changed")
    power = staged.observed_power_dbm.to_numpy().reshape(n_links, n_hours).T
    coverage = staged.coverage_fraction.to_numpy().reshape(n_links, n_hours).T
    weather = match_era5_proxy(stations, hours)
    radar = match_radar(stations, hours)
    clear = (coverage >= 0.95) & (weather["tcc"] <= 0.2) & (
        weather["tp"] < 0.1) & (weather["snow_cm"] == 0.0) & radar["dry_4h"]
    reference = np.full_like(power, np.nan)
    count = np.zeros_like(power, dtype=np.int32)
    radius = np.zeros_like(power, dtype=np.int8)
    reference_iqr = np.full_like(power, np.nan)
    candidate_n = np.zeros(n_links, dtype=np.int64)
    for link in range(n_links):
        candidate_n[link] = int(clear[:, link].sum())
        reference[:, link], count[:, link], radius[:, link], reference_iqr[:, link] = reference_for_link(
            power[:, link], clear[:, link]
        )
    staged["era5_total_cloud_cover"] = weather["tcc"].T.reshape(-1)
    staged["era5_total_precip_mm_hour"] = weather["tp"].T.reshape(-1)
    staged["era5_snowfall_cm_hour"] = weather["snow_cm"].T.reshape(-1)
    staged["weather_source"] = "openmeteo_models_era5_rounded_proxy"
    staged["era5_nearest_grid_lat"] = np.repeat(weather["grid_lat"], n_hours)
    staged["era5_nearest_grid_lon"] = np.repeat(weather["grid_lon"], n_hours)
    staged["radar_rain_mm_hour"] = radar["rain_mm"].T.reshape(-1)
    staged["radar_valid_5min_bins"] = radar["valid_5min_bins"].T.reshape(-1)
    staged["clear_reference_candidate"] = clear.T.reshape(-1)
    staged["clear_reference_power_dbm"] = reference.T.reshape(-1)
    staged["clear_reference_candidate_count"] = count.T.reshape(-1)
    staged["clear_reference_radius_days"] = radius.T.reshape(-1)
    staged["clear_reference_candidate_iqr_db"] = reference_iqr.T.reshape(-1)
    staged["clear_reference_iqr_gt_1db"] = staged.clear_reference_candidate_iqr_db > 1.0
    staged["clear_sky_attenuation_db"] = (
        staged.clear_reference_power_dbm - staged.observed_power_dbm
    )
    staged["target_status"] = np.where(
        staged.observed_power_dbm.isna(), "no_power_observation",
        np.where(staged.clear_reference_power_dbm.isna(), "insufficient_clear_reference",
                 "empirical_opensat_causal_radar_openmeteo_era5_reference")
    )
    staged["clear_context_power_conflict"] = (
        staged.clear_reference_candidate & (staged.clear_sky_attenuation_db > 3.0)
    )
    staged["possible_receiver_floor"] = staged.clear_sky_attenuation_db >= 14.0
    write_new(staged, OUT / "opensat_hourly_attenuation.parquet")
    station_summary = staged.groupby("link_id", sort=False).agg(
        satellite=("satellite", "first"), hardware=("hardware", "first"),
        power_hours=("observed_power_dbm", "count"),
        power_std_db=("observed_power_dbm", "std"),
        attenuation_hours=("clear_sky_attenuation_db", "count"),
        clear_candidates=("clear_reference_candidate", "sum"),
        clear_conflicts=("clear_context_power_conflict", "sum"),
        broad_reference_hours=("clear_reference_iqr_gt_1db", "sum"),
        possible_receiver_floor_hours=("possible_receiver_floor", "sum"),
    ).reset_index()
    clear_spread = staged.loc[staged.clear_reference_candidate].groupby("link_id")[
        "observed_power_dbm"].agg(
        lambda values: values.quantile(.95) - values.quantile(.05)
    )
    station_summary["clear_context_p95_minus_p05_db"] = station_summary.link_id.map(clear_spread)
    station_summary["clear_context_spread_gt_2db"] = (
        station_summary.clear_context_p95_minus_p05_db > 2.0
    )
    require(not (OUT / "opensat_station_summary.csv").exists(), "Summary already exists")
    station_summary.to_csv(OUT / "opensat_station_summary.csv", index=False)
    return {"station_count": n_links, "rows": len(staged),
            "power_hours": int(staged.observed_power_dbm.notna().sum()),
            "attenuation_hours": int(staged.clear_sky_attenuation_db.notna().sum()),
            "reference_candidates": int(clear.sum()),
            "clear_context_power_conflict_hours": int(staged.clear_context_power_conflict.sum()),
            "stations_with_no_candidates": int((candidate_n == 0).sum()),
            "stations_with_no_labels": int((station_summary.attenuation_hours == 0).sum()),
            "missing_era5_end_hour": int(np.isnan(weather["tp"][-1]).sum())}


def make_ceda() -> dict[str, object]:
    native = pd.read_parquet(STAGE / "ceda_gbs_hourly_attenuation.parquet")
    require(len(native) == 47568 and native.clear_sky_attenuation_db.notna().sum() == 46905,
            "CEDA native staging changed")
    native["target_status"] = np.where(native.clear_sky_attenuation_db.notna(),
                                        "native_ceda_gbs", "no_native_attenuation")
    native["coverage_ge_95"] = native.coverage_fraction >= 0.95
    native["clear_reference_method"] = "source_native_clear_sky_attenuation"
    write_new(native, OUT / "ceda_gbs_hourly_attenuation.parquet")
    return {"station_hours": len(native),
            "native_attenuation_hours": int(native.clear_sky_attenuation_db.notna().sum()),
            "high_coverage_native_hours": int((native.coverage_ge_95
                                               & native.clear_sky_attenuation_db.notna()).sum())}


def main() -> None:
    require(not (OUT / "build_manifest.json").exists(), "Refusing manifest overwrite")
    require(json.loads((LOCAL_MODEL / "validation_report.json").read_text(encoding="utf-8"))
            .get("status") == "VERIFIED", "Local Revision_006 not verified")
    stage_status = json.loads((STAGE / "validation_report.json").read_text(encoding="utf-8"))
    require(stage_status.get("status") == "VERIFIED_STAGING_ONLY", "Three-source staging changed")
    local_files = [OUT / name for name in (
        "local_hourly_attenuation_audit.parquet",
        "local_model_table_p10_attenuation_all_measurements.parquet",
        "local_model_table_p10_attenuation_high_coverage_95.parquet")]
    if all(path.is_file() for path in local_files):
        prior = pd.read_parquet(local_files[0])
        local_result = {"hours": len(prior),
                        "high_coverage_hours": int(prior.measurement_high_coverage_95.sum()),
                        "train_clear_reference_hours": int(prior.clear_reference_candidate_count.iloc[0]),
                        "reference_power_dbm": float(prior.clear_reference_power_dbm.iloc[0]),
                        "clear_context_power_conflict_hours": int(prior.clear_context_power_conflict.sum())}
    else:
        require(not any(path.exists() for path in local_files),
                "Partial local output; preserve and investigate before retry")
        local_result = make_local()
    open_files = [OUT / "opensat_hourly_attenuation.parquet",
                  OUT / "opensat_station_summary.csv"]
    if all(path.is_file() for path in open_files):
        prior = pd.read_parquet(open_files[0])
        station = pd.read_csv(open_files[1])
        open_result = {"station_count": len(station), "rows": len(prior),
                       "power_hours": int(prior.observed_power_dbm.notna().sum()),
                       "attenuation_hours": int(prior.clear_sky_attenuation_db.notna().sum()),
                       "reference_candidates": int(prior.clear_reference_candidate.sum()),
                       "clear_context_power_conflict_hours": int(prior.clear_context_power_conflict.sum()),
                       "stations_with_no_candidates": int((station.clear_candidates == 0).sum()),
                       "stations_with_no_labels": int((station.attenuation_hours == 0).sum())}
    else:
        require(not any(path.exists() for path in open_files),
                "Partial OpenSat output; preserve and investigate before retry")
        open_result = make_opensat()
    ceda_file = OUT / "ceda_gbs_hourly_attenuation.parquet"
    if ceda_file.is_file():
        prior = pd.read_parquet(ceda_file)
        ceda_result = {"station_hours": len(prior),
                       "native_attenuation_hours": int(prior.clear_sky_attenuation_db.notna().sum()),
                       "high_coverage_native_hours": int((prior.coverage_ge_95
                                                          & prior.clear_sky_attenuation_db.notna()).sum())}
    else:
        ceda_result = make_ceda()
    results = {"local": local_result, "opensat": open_result, "ceda_gbs": ceda_result}
    inputs = [STAGE / "local_hourly_power_audit.parquet",
              STAGE / "opensat_hourly_power.parquet",
              STAGE / "ceda_gbs_hourly_attenuation.parquet",
              OPENSAT_RAW / "radar_along_sml_data_2022.nc",
              LOCAL_MODEL / "model_table_primary_p10_all_measurements.parquet",
              Path(__file__), OUT / "weather_proxy_comparison.json",
              PROXY / "request_manifest.json"] + [
                  PROXY / f"batch_{batch:02d}.json" for batch in range(6)]
    outputs = [OUT / name for name in (
        "local_hourly_attenuation_audit.parquet",
        "local_model_table_p10_attenuation_all_measurements.parquet",
        "local_model_table_p10_attenuation_high_coverage_95.parquet",
        "opensat_hourly_attenuation.parquet", "opensat_station_summary.csv",
        "ceda_gbs_hourly_attenuation.parquet")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "pandas": pd.__version__,
                "parameters": {"local_reference": "train-only screened clear_dry median dBm",
                               "open_min_n": MIN_BASELINE_N, "open_near_days": NEAR_DAYS,
                               "open_far_days": FAR_DAYS, "open_hour_tolerance": HOUR_TOLERANCE,
                               "open_quantile": OPEN_BASELINE_QUANTILE,
                               "open_reference_is_past_only": True,
                               "cloud_max_fraction": 0.2,
                               "era5_precip_max_mm_hour": 0.1,
                               "snowfall_required_cm_hour": 0.0,
                               "weather_source": "Open-Meteo historical API, models=era5; rounded proxy, not official CDS NetCDF",
                               "weather_proxy_validated_cell_hours_on_one_day": 1320,
                               "radar_prior_and_current_hours_dry": 4},
                "counts": results,
                "input_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in inputs},
                "output_sha256": {p.name: sha256(p) for p in outputs}}
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
