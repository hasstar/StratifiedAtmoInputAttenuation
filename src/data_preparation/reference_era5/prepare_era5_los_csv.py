"""Create one surface CSV and eleven line-of-sight pressure-level CSV files.

ERA5 validity times are UTC. Output rows are the exact 7,800 Beijing-time
hours requested for the campaign. Surface variables are bilinearly
interpolated at BJTU. For every pressure level and hour, the program solves
the intersection between the BJTU-to-125 E GEO ray and that hour's isobaric
geopotential surface, then interpolates all pressure variables at the solved
latitude/longitude.

All ERA5 values remain in their original physical units and missing values are
retained. In particular, a pressure surface below the antenna produces NaNs
and los_intersection_valid=False; it is never extrapolated behind the station.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

VENDOR = Path(__file__).resolve().parents[1] / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import xarray as xr


G0_M_S2 = 9.80665
GEOPOTENTIAL_EARTH_RADIUS_M = 6_371_229.0
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

PRESSURE_SHORT_NAMES = {
    "divergence": "d",
    "fraction_of_cloud_cover": "cc",
    "geopotential": "z",
    "ozone_mass_mixing_ratio": "o3",
    "potential_vorticity": "pv",
    "relative_humidity": "r",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_cloud_liquid_water_content": "clwc",
    "specific_humidity": "q",
    "specific_rain_water_content": "crwc",
    "specific_snow_water_content": "cswc",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "vorticity": "vo",
}
SURFACE_SHORT_NAMES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "msl",
    "boundary_layer_height": "blh",
    "zero_degree_level": "deg0l",
    "total_column_water": "tcw",
    "total_column_water_vapour": "tcwv",
    "total_column_cloud_liquid_water": "tclw",
    "total_column_cloud_ice_water": "tciw",
    "total_column_rain_water": "tcrw",
    "total_column_snow_water": "tcsw",
    "total_column_supercooled_liquid_water": "tcslw",
    "total_precipitation": "tp",
    "convective_precipitation": "cp",
    "large_scale_precipitation": "lsp",
    "mean_total_precipitation_rate": "avg_tprate",
    "mean_convective_precipitation_rate": "avg_cpr",
    "mean_large_scale_precipitation_rate": "avg_lsprate",
    "precipitation_type": "ptype",
    "low_cloud_cover": "lcc",
    "medium_cloud_cover": "mcc",
    "high_cloud_cover": "hcc",
    "total_cloud_cover": "tcc",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "vertical_integral_of_divergence_of_moisture_flux": "vimdf",
    "trapping_layer_base_height": "tplb",
    "trapping_layer_top_height": "tplt",
    "mean_vertical_gradient_of_refractivity_inside_trapping_layer": "dndza",
}
REFERENCE_GEOPOTENTIAL_HEIGHT_M = {
    1000: 110.0,
    925: 760.0,
    850: 1_460.0,
    700: 3_010.0,
    650: 3_590.0,
    500: 5_570.0,
    300: 9_160.0,
    200: 11_780.0,
    30: 23_850.0,
    5: 35_750.0,
    1: 47_850.0,
}
TECHNICAL_VARIABLES = {"number", "expver"}


def _campaign_times(config: dict[str, Any]) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    time_config = config["time_range"]
    start = pd.Timestamp(time_config["start_bjt"], tz=time_config["timezone"])
    end = pd.Timestamp(time_config["end_bjt"], tz=time_config["timezone"])
    bjt = pd.date_range(start, end, freq=time_config["frequency"])
    if len(bjt) != int(time_config["expected_hours"]):
        raise ValueError("Campaign time range does not match expected_hours")
    utc_naive = bjt.tz_convert("UTC").tz_localize(None)
    return bjt, utc_naive


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    normal_radius = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (normal_radius + height_m) * math.cos(lat) * math.cos(lon),
            (normal_radius + height_m) * math.cos(lat) * math.sin(lon),
            (normal_radius * (1.0 - WGS84_E2) + height_m) * math.sin(lat),
        ],
        dtype=np.float64,
    )


def _ecef_to_geodetic(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    lon = np.arctan2(y, x)
    horizontal = np.hypot(x, y)
    lat = np.arctan2(z, horizontal * (1.0 - WGS84_E2))
    for _ in range(7):
        normal_radius = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        height = horizontal / np.cos(lat) - normal_radius
        lat = np.arctan2(
            z,
            horizontal * (1.0 - WGS84_E2 * normal_radius / (normal_radius + height)),
        )
    normal_radius = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    height = horizontal / np.cos(lat) - normal_radius
    return np.degrees(lat), np.degrees(lon), height


def _geopotential_to_geometric_height(geopotential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    geopotential_height = geopotential / G0_M_S2
    geometric_height = (
        GEOPOTENTIAL_EARTH_RADIUS_M
        * geopotential_height
        / (GEOPOTENTIAL_EARTH_RADIUS_M - geopotential_height)
    )
    return geopotential_height, geometric_height


def _ray_geometry(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    station = config["station"]
    satellite = config["satellite"]
    origin = _geodetic_to_ecef(
        float(station["latitude_deg_n"]),
        float(station["longitude_deg_e"]),
        float(station["antenna_phase_center_m_amsl"]),
    )
    satellite_position = _geodetic_to_ecef(
        0.0,
        float(satellite["nominal_longitude_deg_e"]),
        float(satellite["nominal_altitude_km"]) * 1000.0,
    )
    direction = satellite_position - origin
    direction /= np.linalg.norm(direction)
    return origin, direction


def _time_name(dataset: xr.Dataset | xr.DataArray) -> str:
    for name in ("valid_time", "time"):
        if name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError("No valid_time or time coordinate in ERA5 file")


def _lat_lon_names(dataset: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    lat = "latitude" if "latitude" in dataset.coords else "lat"
    lon = "longitude" if "longitude" in dataset.coords else "lon"
    if lat not in dataset.coords or lon not in dataset.coords:
        raise KeyError("No latitude/longitude coordinates in ERA5 file")
    return lat, lon


def _pressure_name(dataset: xr.Dataset | xr.DataArray) -> str:
    for name in ("pressure_level", "pressureLevels", "level"):
        if name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError("No pressure-level coordinate in ERA5 file")


def _collapse_extra_dims(
    data: xr.DataArray, required_dims: Iterable[str]
) -> xr.DataArray:
    required = set(required_dims)
    for dim in list(data.dims):
        if dim in required:
            continue
        if data.sizes[dim] == 1:
            data = data.isel({dim: 0}, drop=True)
        else:
            # ERA5T/final ERA5 can coexist along expver with complementary NaNs.
            data = data.max(dim=dim, skipna=True)
    return data


def _bilinear_time_series(
    values: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate one horizontal field for a different point per time."""
    values = np.asarray(values, dtype=np.float64)
    lat_order = np.argsort(np.asarray(grid_lat, dtype=np.float64))
    lon_order = np.argsort(np.asarray(grid_lon, dtype=np.float64))
    lat_values = np.asarray(grid_lat, dtype=np.float64)[lat_order]
    lon_values = np.asarray(grid_lon, dtype=np.float64)[lon_order]
    values = values[:, lat_order, :][:, :, lon_order]

    lat_hi = np.searchsorted(lat_values, target_lat, side="right")
    lon_hi = np.searchsorted(lon_values, target_lon, side="right")
    lat_hi = np.clip(lat_hi, 1, len(lat_values) - 1)
    lon_hi = np.clip(lon_hi, 1, len(lon_values) - 1)
    lat_lo = lat_hi - 1
    lon_lo = lon_hi - 1
    inside = (
        (target_lat >= lat_values[0])
        & (target_lat <= lat_values[-1])
        & (target_lon >= lon_values[0])
        & (target_lon <= lon_values[-1])
    )
    lat_span = lat_values[lat_hi] - lat_values[lat_lo]
    lon_span = lon_values[lon_hi] - lon_values[lon_lo]
    lat_weight = (target_lat - lat_values[lat_lo]) / lat_span
    lon_weight = (target_lon - lon_values[lon_lo]) / lon_span
    row = np.arange(values.shape[0])
    v00 = values[row, lat_lo, lon_lo]
    v10 = values[row, lat_hi, lon_lo]
    v01 = values[row, lat_lo, lon_hi]
    v11 = values[row, lat_hi, lon_hi]
    result = (
        v00 * (1.0 - lat_weight) * (1.0 - lon_weight)
        + v10 * lat_weight * (1.0 - lon_weight)
        + v01 * (1.0 - lat_weight) * lon_weight
        + v11 * lat_weight * lon_weight
    )
    result[~inside] = np.nan
    return result


def _haversine_distance_m(
    start_lat_deg: float,
    start_lon_deg: float,
    end_lat_deg: np.ndarray,
    end_lon_deg: np.ndarray,
) -> np.ndarray:
    start_lat = np.radians(start_lat_deg)
    end_lat = np.radians(end_lat_deg)
    delta_lat = end_lat - start_lat
    delta_lon = np.radians(end_lon_deg - start_lon_deg)
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(start_lat) * np.cos(end_lat) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * GEOPOTENTIAL_EARTH_RADIUS_M * np.arcsin(np.sqrt(value))


def _solve_intersections(
    geopotential_values: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    level_hpa: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    count = geopotential_values.shape[0]
    origin, direction = _ray_geometry(config)
    station = config["station"]
    station_lat = float(station["latitude_deg_n"])
    station_lon = float(station["longitude_deg_e"])
    antenna_height = float(station["antenna_phase_center_m_amsl"])

    reference_gp_height = REFERENCE_GEOPOTENTIAL_HEIGHT_M[level_hpa]
    reference_geometric = (
        GEOPOTENTIAL_EARTH_RADIUS_M
        * reference_gp_height
        / (GEOPOTENTIAL_EARTH_RADIUS_M - reference_gp_height)
    )
    local_up = np.array(
        [
            math.cos(math.radians(station_lat)) * math.cos(math.radians(station_lon)),
            math.cos(math.radians(station_lat)) * math.sin(math.radians(station_lon)),
            math.sin(math.radians(station_lat)),
        ]
    )
    initial_vertical_slope = float(np.dot(direction, local_up))
    slant = np.full(
        count,
        (reference_geometric - antenna_height) / initial_vertical_slope,
        dtype=np.float64,
    )
    slant = np.clip(slant, -2_000.0, 150_000.0)
    last_gp_height = np.full(count, np.nan)
    last_geometric_height = np.full(count, np.nan)
    last_lat = np.full(count, np.nan)
    last_lon = np.full(count, np.nan)
    last_ray_height = np.full(count, np.nan)

    for _ in range(8):
        points = origin[None, :] + slant[:, None] * direction[None, :]
        lat, lon, ray_height = _ecef_to_geodetic(points)
        geopotential = _bilinear_time_series(
            geopotential_values, grid_lat, grid_lon, lat, lon
        )
        gp_height, geometric_height = _geopotential_to_geometric_height(geopotential)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        local_normal = np.column_stack(
            (
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            )
        )
        vertical_slope = local_normal @ direction
        correction = (geometric_height - ray_height) / vertical_slope
        correction = np.clip(correction, -20_000.0, 20_000.0)
        slant += np.where(np.isfinite(correction), correction, 0.0)
        last_gp_height = gp_height
        last_geometric_height = geometric_height
        last_lat = lat
        last_lon = lon
        last_ray_height = ray_height

    # Re-evaluate at the final slant after the last Newton update.
    points = origin[None, :] + slant[:, None] * direction[None, :]
    lat, lon, ray_height = _ecef_to_geodetic(points)
    geopotential = _bilinear_time_series(geopotential_values, grid_lat, grid_lon, lat, lon)
    gp_height, geometric_height = _geopotential_to_geometric_height(geopotential)
    residual = ray_height - geometric_height
    valid = (
        np.isfinite(residual)
        & (np.abs(residual) <= 5.0)
        & (slant >= 0.0)
        & (geometric_height >= antenna_height)
    )
    horizontal = _haversine_distance_m(station_lat, station_lon, lat, lon)
    return pd.DataFrame(
        {
            "los_latitude_deg_n": np.where(valid, lat, np.nan),
            "los_longitude_deg_e": np.where(valid, lon, np.nan),
            "pressure_geopotential_height_m": np.where(valid, gp_height, np.nan),
            "pressure_geometric_height_m_amsl": np.where(valid, geometric_height, np.nan),
            "los_slant_distance_from_antenna_m": np.where(valid, slant, np.nan),
            "los_horizontal_displacement_m": np.where(valid, horizontal, np.nan),
            "los_intersection_residual_m": np.where(valid, residual, np.nan),
            "los_intersection_valid": valid,
        }
    )


def _job_members(raw_dir: Path, job_id: str) -> list[Path]:
    primary = raw_dir / f"{job_id}.nc"
    if primary.exists():
        return [primary]
    members = sorted(raw_dir.glob(f"{job_id}__*.nc"))
    if members:
        return members
    raise FileNotFoundError(f"Raw ERA5 job is missing: {job_id}")


def _manifest_jobs(raw_dir: Path, prefix: str) -> list[dict[str, Any]]:
    manifest = json.loads((raw_dir / "download_manifest.json").read_text(encoding="utf-8"))
    jobs = [job for job in manifest["jobs"] if job["job_id"].startswith(prefix)]
    if not jobs:
        raise ValueError(f"No manifest jobs start with {prefix!r}")
    return jobs


def _select_campaign_times(
    data: xr.DataArray, expected_utc: pd.DatetimeIndex
) -> tuple[xr.DataArray, pd.DatetimeIndex]:
    time_name = _time_name(data)
    source_times = pd.DatetimeIndex(data[time_name].values)
    keep = source_times.isin(expected_utc)
    data = data.isel({time_name: np.flatnonzero(keep)})
    return data, source_times[keep]


def _geometry_frames(
    raw_dir: Path,
    config: dict[str, Any],
    expected_utc: pd.DatetimeIndex,
) -> dict[int, pd.DataFrame]:
    levels = [int(value) for value in config["pressure_levels_hpa"]]
    geometry = {
        level: pd.DataFrame(
            index=expected_utc,
            columns=[
                "los_latitude_deg_n",
                "los_longitude_deg_e",
                "pressure_geopotential_height_m",
                "pressure_geometric_height_m_amsl",
                "los_slant_distance_from_antenna_m",
                "los_horizontal_displacement_m",
                "los_intersection_residual_m",
                "los_intersection_valid",
            ],
        )
        for level in levels
    }
    jobs = _manifest_jobs(raw_dir, "pressure_geopotential_")
    for job in jobs:
        for path in _job_members(raw_dir, job["job_id"]):
            with xr.open_dataset(path, engine="netcdf4") as dataset:
                if "z" not in dataset.data_vars:
                    continue
                time_name = _time_name(dataset)
                lat_name, lon_name = _lat_lon_names(dataset)
                level_name = _pressure_name(dataset)
                data = _collapse_extra_dims(
                    dataset["z"], [time_name, level_name, lat_name, lon_name]
                )
                data, times = _select_campaign_times(data, expected_utc)
                if len(times) == 0:
                    continue
                grid_lat = np.asarray(data[lat_name].values)
                grid_lon = np.asarray(data[lon_name].values)
                for level in levels:
                    level_data = data.sel({level_name: float(level)}).transpose(
                        time_name, lat_name, lon_name
                    )
                    solved = _solve_intersections(
                        np.asarray(level_data.values), grid_lat, grid_lon, level, config
                    )
                    solved.index = times
                    geometry[level].loc[times, solved.columns] = solved.values
    for level, frame in geometry.items():
        frame["los_intersection_valid"] = (
            frame["los_intersection_valid"].fillna(False).astype(bool)
        )
        for column in frame.columns:
            if column != "los_intersection_valid":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame.index.equals(expected_utc):
            raise RuntimeError(f"Geometry time index changed for {level} hPa")
    return geometry


def _initial_output_frame(
    bjt: pd.DatetimeIndex,
    utc: pd.DatetimeIndex,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_time_bjt": [value.isoformat(sep=" ") for value in bjt],
            "valid_time_utc": [
                value.tz_localize("UTC").isoformat(sep=" ") for value in utc
            ],
        },
        index=utc,
    )


def _metadata_entry(data: xr.DataArray, source_name: str) -> dict[str, Any]:
    return {
        "source_short_name": source_name,
        "long_name": data.attrs.get("long_name", source_name),
        "units": data.attrs.get("units", ""),
        "source": "ERA5 hourly reanalysis; values remain unnormalised",
    }


def _pressure_outputs(
    raw_dir: Path,
    config: dict[str, Any],
    bjt: pd.DatetimeIndex,
    utc: pd.DatetimeIndex,
) -> tuple[dict[int, pd.DataFrame], dict[str, dict[str, Any]]]:
    levels = [int(value) for value in config["pressure_levels_hpa"]]
    geometry = _geometry_frames(raw_dir, config, utc)
    frames: dict[int, pd.DataFrame] = {}
    for level in levels:
        base = _initial_output_frame(bjt, utc)
        base.insert(2, "pressure_level_hpa", level)
        frames[level] = pd.concat([base, geometry[level]], axis=1)
        for variable in config["pressure_variables"]:
            frames[level][variable] = np.nan

    inverse_names = {short: long for long, short in PRESSURE_SHORT_NAMES.items()}
    metadata: dict[str, dict[str, Any]] = {}
    jobs = _manifest_jobs(raw_dir, "pressure_")
    for job in jobs:
        for path in _job_members(raw_dir, job["job_id"]):
            with xr.open_dataset(path, engine="netcdf4") as dataset:
                time_name = _time_name(dataset)
                lat_name, lon_name = _lat_lon_names(dataset)
                level_name = _pressure_name(dataset)
                for short_name in dataset.data_vars:
                    if short_name in TECHNICAL_VARIABLES or short_name not in inverse_names:
                        continue
                    long_name = inverse_names[short_name]
                    data = _collapse_extra_dims(
                        dataset[short_name], [time_name, level_name, lat_name, lon_name]
                    )
                    data, times = _select_campaign_times(data, utc)
                    if len(times) == 0:
                        continue
                    metadata[long_name] = _metadata_entry(data, short_name)
                    grid_lat = np.asarray(data[lat_name].values)
                    grid_lon = np.asarray(data[lon_name].values)
                    for level in levels:
                        coords = geometry[level].loc[times]
                        target_lat = coords["los_latitude_deg_n"].to_numpy(dtype=float)
                        target_lon = coords["los_longitude_deg_e"].to_numpy(dtype=float)
                        level_data = data.sel({level_name: float(level)}).transpose(
                            time_name, lat_name, lon_name
                        )
                        interpolated = _bilinear_time_series(
                            np.asarray(level_data.values),
                            grid_lat,
                            grid_lon,
                            target_lat,
                            target_lon,
                        )
                        valid = coords["los_intersection_valid"].to_numpy(dtype=bool)
                        interpolated[~valid] = np.nan
                        frames[level].loc[times, long_name] = interpolated
    return frames, metadata


def _surface_output(
    raw_dir: Path,
    config: dict[str, Any],
    bjt: pd.DatetimeIndex,
    utc: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    station = config["station"]
    station_lat = float(station["latitude_deg_n"])
    station_lon = float(station["longitude_deg_e"])
    frame = _initial_output_frame(bjt, utc)
    frame.insert(2, "surface_latitude_deg_n", station_lat)
    frame.insert(3, "surface_longitude_deg_e", station_lon)
    frame.insert(4, "antenna_phase_center_m_amsl", float(station["antenna_phase_center_m_amsl"]))
    for variable in config["surface_variables"]:
        frame[variable] = np.nan

    inverse_names = {short: long for long, short in SURFACE_SHORT_NAMES.items()}
    param_id_to_name = {
        int(param_id): name
        for name, param_id in config.get("surface_variable_param_ids", {}).items()
    }
    metadata: dict[str, dict[str, Any]] = {}
    jobs = _manifest_jobs(raw_dir, "surface_")
    for job in jobs:
        for path in _job_members(raw_dir, job["job_id"]):
            with xr.open_dataset(path, engine="netcdf4") as dataset:
                time_name = _time_name(dataset)
                lat_name, lon_name = _lat_lon_names(dataset)
                for short_name in dataset.data_vars:
                    if short_name in TECHNICAL_VARIABLES:
                        continue
                    param_id = dataset[short_name].attrs.get("GRIB_paramId")
                    long_name = param_id_to_name.get(int(param_id)) if param_id is not None else None
                    long_name = long_name or inverse_names.get(short_name)
                    if long_name is None or long_name not in frame.columns:
                        continue
                    data = _collapse_extra_dims(
                        dataset[short_name], [time_name, lat_name, lon_name]
                    )
                    data, times = _select_campaign_times(data, utc)
                    if len(times) == 0:
                        continue
                    metadata[long_name] = _metadata_entry(data, short_name)
                    grid_lat = np.asarray(data[lat_name].values)
                    grid_lon = np.asarray(data[lon_name].values)
                    values = data.transpose(time_name, lat_name, lon_name).values
                    interpolated = _bilinear_time_series(
                        np.asarray(values),
                        grid_lat,
                        grid_lon,
                        np.full(len(times), station_lat),
                        np.full(len(times), station_lon),
                    )
                    frame.loc[times, long_name] = interpolated
    return frame, metadata


def _quality_record(
    path: Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    expected_hours: int,
    level_hpa: int | None = None,
    source_schema_complete: bool = True,
) -> dict[str, Any]:
    times = pd.to_datetime(frame["valid_time_bjt"], utc=True)
    record: dict[str, Any] = {
        "file": path.name,
        "pressure_level_hpa": level_hpa,
        "rows": len(frame),
        "expected_rows": expected_hours,
        "first_time_bjt": frame["valid_time_bjt"].iloc[0],
        "last_time_bjt": frame["valid_time_bjt"].iloc[-1],
        "duplicate_time_count": int(times.duplicated().sum()),
        "feature_count": len(feature_columns),
        "schema_complete": source_schema_complete
        and all(column in frame for column in feature_columns),
        "rows_with_all_features": int(frame[feature_columns].notna().all(axis=1).sum()),
        "total_missing_feature_values": int(frame[feature_columns].isna().sum().sum()),
    }
    if "los_intersection_valid" in frame:
        record["valid_los_intersection_hours"] = int(frame["los_intersection_valid"].sum())
        record["invalid_los_intersection_hours"] = int((~frame["los_intersection_valid"]).sum())
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/era5_campaign_20250930_20260820.json"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/era5_campaign_20250930_20260820/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/era5_campaign_20250930_20260820/csv"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    # The raw manifest records the fully resolved 188-field selection used for
    # the downloads. Reading it here guarantees that CSV preparation uses the
    # exact same schema without consulting any legacy data source.
    manifest_path = args.raw_dir / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["campaign"]
    bjt, utc = _campaign_times(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pressure_frames, pressure_metadata = _pressure_outputs(
        args.raw_dir, config, bjt, utc
    )
    surface_frame, surface_metadata = _surface_output(args.raw_dir, config, bjt, utc)
    missing_pressure_variables = sorted(
        set(config["pressure_variables"]) - set(pressure_metadata)
    )
    missing_surface_variables = sorted(
        set(config["surface_variables"]) - set(surface_metadata)
    )
    pressure_schema_complete = not missing_pressure_variables
    surface_schema_complete = not missing_surface_variables

    quality: list[dict[str, Any]] = []
    pressure_features = list(config["pressure_variables"])
    for level in [int(value) for value in config["pressure_levels_hpa"]]:
        frame = pressure_frames[level].reset_index(drop=True)
        path = args.output_dir / f"era5_pressure_{level}hPa_los_hourly_bjt.csv"
        frame.to_csv(path, index=False, na_rep="")
        quality.append(
            _quality_record(
                path,
                frame,
                pressure_features,
                len(bjt),
                level_hpa=level,
                source_schema_complete=pressure_schema_complete,
            )
        )

    surface_path = args.output_dir / "era5_surface_bjtu_hourly_bjt.csv"
    surface_frame = surface_frame.reset_index(drop=True)
    surface_frame.to_csv(surface_path, index=False, na_rep="")
    quality.append(
        _quality_record(
            surface_path,
            surface_frame,
            list(config["surface_variables"]),
            len(bjt),
            source_schema_complete=surface_schema_complete,
        )
    )
    quality_frame = pd.DataFrame(quality)
    quality_frame.to_csv(args.output_dir / "quality_report.csv", index=False)

    missing_by_file: dict[str, dict[str, int]] = {}
    for level, frame in pressure_frames.items():
        missing_by_file[f"pressure_{level}hPa"] = {
            column: int(frame[column].isna().sum()) for column in pressure_features
        }
    missing_by_file["surface"] = {
        column: int(surface_frame[column].isna().sum())
        for column in config["surface_variables"]
    }
    metadata = {
        "campaign": config,
        "source_time_zone": "UTC",
        "output_primary_time_zone": "Asia/Shanghai (UTC+8)",
        "rows_per_csv": len(bjt),
        "normalization": "none",
        "surface_interpolation": config["sampling"]["surface_method"],
        "pressure_interpolation": config["sampling"]["pressure_method"],
        "geopotential_height_conversion": "H=z/9.80665; geometric height h=R*H/(R-H), R=6371229 m",
        "below_antenna_policy": config["sampling"]["below_antenna_policy"],
        "pressure_features": pressure_metadata,
        "surface_features": surface_metadata,
        "missing_value_counts": missing_by_file,
        "missing_pressure_variables": missing_pressure_variables,
        "missing_surface_variables": missing_surface_variables,
    }
    (args.output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(quality_frame.to_string(index=False))
    bad_rows = quality_frame[quality_frame["rows"] != quality_frame["expected_rows"]]
    bad_schema = quality_frame[~quality_frame["schema_complete"]]
    if not bad_rows.empty or not bad_schema.empty:
        raise SystemExit("CSV validation failed; see quality_report.csv")


if __name__ == "__main__":
    main()
