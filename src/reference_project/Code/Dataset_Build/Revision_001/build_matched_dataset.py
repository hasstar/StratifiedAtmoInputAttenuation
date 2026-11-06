from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TZ_BJT = "Asia/Shanghai"
START_BJT = pd.Timestamp("2025-09-30 00:00:00", tz=TZ_BJT)
END_EXCLUSIVE_BJT = pd.Timestamp("2026-08-21 00:00:00", tz=TZ_BJT)

DEFAULT_ERA5_CSV_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Data\era5_campaign_20250930_20260820\csv"
)
DEFAULT_MEASUREMENT_RUN_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Data\Further analysis of the data results"
    r"\思源楼高轨卫星信标信号实测数据_再分析结果\Runs\Run_009"
)
DEFAULT_OUTPUT_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction"
    r"\Outputs\Datasets\Revision_001"
)

PRESSURE_LEVEL_TO_LAYER = {
    1000: "00_08km",
    925: "00_08km",
    850: "00_08km",
    700: "00_08km",
    650: "00_08km",
    500: "00_08km",
    300: "08_17km",
    200: "08_17km",
    30: "17_28km",
    5: "28_37km",
    1: "37_50km",
}

LAYER_BOUNDS_M = {
    "00_08km": (0.0, 8_000.0),
    "08_17km": (8_000.0, 17_000.0),
    "17_28km": (17_000.0, 28_000.0),
    "28_37km": (28_000.0, 37_000.0),
    "37_50km": (37_000.0, 50_000.0),
}

PRESSURE_GEOMETRY_COLUMNS = [
    "los_latitude_deg_n",
    "los_longitude_deg_e",
    "pressure_geopotential_height_m",
    "pressure_geometric_height_m_amsl",
    "los_slant_distance_from_antenna_m",
    "los_horizontal_displacement_m",
    "los_intersection_residual_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an hourly time-matched measurement and ERA5 data set."
    )
    parser.add_argument("--era5-csv-dir", type=Path, default=DEFAULT_ERA5_CSV_DIR)
    parser.add_argument(
        "--measurement-run-dir", type=Path, default=DEFAULT_MEASUREMENT_RUN_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_identity(path: Path, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        result["sha256"] = sha256_file(path)
    return result


def output_identity(path: Path, output_root: Path) -> dict[str, Any]:
    identity = file_identity(path)
    identity["path"] = path.relative_to(output_root).as_posix()
    return identity


def parse_bjt(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(TZ_BJT)
    return parsed.dt.tz_convert(TZ_BJT)


def require_unique_time(frame: pd.DataFrame, column: str, label: str) -> None:
    duplicates = int(frame[column].duplicated().sum())
    if duplicates:
        raise ValueError(f"{label} contains {duplicates} duplicate timestamps")


def aggregate_measurements(
    overview_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(overview_path)
    required = {"time", "value", "count"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Measurement overview is missing columns: {sorted(missing)}")

    raw["time_bjt"] = parse_bjt(raw["time"])
    require_unique_time(raw, "time_bjt", "measurement 10-minute overview")

    raw_count = len(raw)
    valid_all = raw[
        raw["count"].fillna(0).gt(0)
        & np.isfinite(raw["value"])
        & np.isfinite(raw["count"])
    ].copy()
    valid_all["count"] = valid_all["count"].astype("int64")
    valid_all = valid_all[valid_all["count"].gt(0)].copy()

    before_start = valid_all[valid_all["time_bjt"].lt(START_BJT)]
    after_end = valid_all[valid_all["time_bjt"].ge(END_EXCLUSIVE_BJT)]
    valid = valid_all[
        valid_all["time_bjt"].ge(START_BJT)
        & valid_all["time_bjt"].lt(END_EXCLUSIVE_BJT)
    ].copy()
    if valid.empty:
        raise ValueError("No valid measurements fall inside the requested campaign range")

    valid["measurement_power_w"] = np.power(10.0, (valid["value"] - 30.0) / 10.0)
    valid["weighted_power_sum_w"] = valid["measurement_power_w"] * valid["count"]
    valid["time_bjt"] = valid["time_bjt"].dt.floor("10min")
    valid["hour_bjt"] = valid["time_bjt"].dt.floor("h")

    grouped = valid.groupby("hour_bjt", sort=True, observed=True)
    hourly = grouped.agg(
        measurement_rows=("count", "sum"),
        valid_10min_bins=("count", "size"),
        weighted_power_sum_w=("weighted_power_sum_w", "sum"),
        measurement_first_time_bjt=("time_bjt", "min"),
        measurement_last_time_bjt=("time_bjt", "max"),
        ten_min_power_dbm_min=("value", "min"),
        ten_min_power_dbm_median=("value", "median"),
        ten_min_power_dbm_max=("value", "max"),
        ten_min_power_dbm_std=("value", "std"),
    ).reset_index()

    hourly["measurement_rows"] = hourly["measurement_rows"].astype("int64")
    hourly["valid_10min_bins"] = hourly["valid_10min_bins"].astype("int16")
    if hourly["valid_10min_bins"].gt(6).any():
        bad = hourly.loc[hourly["valid_10min_bins"].gt(6), "hour_bjt"].astype(str).tolist()
        raise ValueError(f"More than six 10-minute bins found in an hour: {bad[:5]}")

    hourly["received_power_w"] = (
        hourly["weighted_power_sum_w"] / hourly["measurement_rows"]
    )
    hourly["received_power_dbm"] = 10.0 * np.log10(
        hourly["received_power_w"] * 1_000.0
    )
    hourly["measurement_10min_coverage_fraction"] = (
        hourly["valid_10min_bins"] / 6.0
    )
    hourly["measurement_full_hour"] = hourly["valid_10min_bins"].eq(6)
    hourly["hour_utc"] = hourly["hour_bjt"].dt.tz_convert("UTC")
    hourly = hourly.drop(columns=["weighted_power_sum_w"])

    if not np.isfinite(hourly["received_power_dbm"]).all():
        raise ValueError("Hourly received power contains non-finite values")
    require_unique_time(hourly, "hour_bjt", "hourly measurements")

    monthly = (
        hourly.assign(month=hourly["hour_bjt"].dt.to_period("M").astype(str))
        .groupby("month", sort=True)
        .agg(
            measurement_hours=("hour_bjt", "size"),
            full_measurement_hours=("measurement_full_hour", "sum"),
            measurement_rows=("measurement_rows", "sum"),
        )
        .reset_index()
    )
    summary = {
        "overview_grid_rows": raw_count,
        "valid_10min_bins_all_dates": len(valid_all),
        "excluded_valid_bins_before_start": len(before_start),
        "excluded_valid_bins_at_or_after_end": len(after_end),
        "valid_10min_bins_in_range": len(valid),
        "measurement_rows_in_range": int(valid["count"].sum()),
        "hourly_measurement_rows": len(hourly),
        "full_measurement_hours": int(hourly["measurement_full_hour"].sum()),
        "partial_measurement_hours": int((~hourly["measurement_full_hour"]).sum()),
        "first_measurement_bjt": hourly["measurement_first_time_bjt"].min().isoformat(),
        "last_measurement_bjt": hourly["measurement_last_time_bjt"].max().isoformat(),
        "monthly_coverage": monthly.to_dict("records"),
    }
    return hourly, summary


def load_surface(
    surface_path: Path, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    surface = pd.read_csv(surface_path)
    surface["hour_bjt"] = parse_bjt(surface["valid_time_bjt"])
    require_unique_time(surface, "hour_bjt", "ERA5 surface")
    if len(surface) != 7_800:
        raise ValueError(f"ERA5 surface row count is {len(surface)}, expected 7800")

    utc = pd.to_datetime(surface["valid_time_utc"], utc=True, errors="raise")
    if not utc.equals(surface["hour_bjt"].dt.tz_convert("UTC")):
        raise ValueError("ERA5 surface BJT and UTC timestamps are inconsistent")

    non_time = [
        column
        for column in surface.columns
        if column not in {"valid_time_bjt", "valid_time_utc", "hour_bjt"}
    ]
    renamed = surface[["hour_bjt", *non_time]].rename(
        columns={column: f"surface__{column}" for column in non_time}
    )
    feature_columns = [
        f"surface__{name}"
        for name in metadata["surface_features"]
        if f"surface__{name}" in renamed.columns
    ]
    return renamed, feature_columns


def load_pressure_levels(
    era5_csv_dir: Path, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], list[str]]:
    paths = sorted(era5_csv_dir.glob("era5_pressure_*hPa_los_hourly_bjt.csv"))
    if len(paths) != len(PRESSURE_LEVEL_TO_LAYER):
        raise ValueError(
            f"Found {len(paths)} pressure CSVs, expected {len(PRESSURE_LEVEL_TO_LAYER)}"
        )

    wide_frames: list[pd.DataFrame] = []
    long_frames: list[pd.DataFrame] = []
    expected_times: pd.Series | None = None
    height_report: list[dict[str, Any]] = []
    pressure_feature_names = list(metadata["pressure_features"])
    aggregation_columns = [*PRESSURE_GEOMETRY_COLUMNS, *pressure_feature_names]

    for path in paths:
        match = re.search(r"pressure_(\d+)hPa", path.name)
        if match is None:
            raise ValueError(f"Cannot parse pressure level from {path.name}")
        level = int(match.group(1))
        if level not in PRESSURE_LEVEL_TO_LAYER:
            raise ValueError(f"Unexpected pressure level {level} hPa")

        frame = pd.read_csv(path)
        frame["hour_bjt"] = parse_bjt(frame["valid_time_bjt"])
        require_unique_time(frame, "hour_bjt", path.name)
        if len(frame) != 7_800:
            raise ValueError(f"{path.name} has {len(frame)} rows, expected 7800")
        utc = pd.to_datetime(frame["valid_time_utc"], utc=True, errors="raise")
        if not utc.equals(frame["hour_bjt"].dt.tz_convert("UTC")):
            raise ValueError(f"BJT and UTC timestamps are inconsistent in {path.name}")

        if expected_times is None:
            expected_times = frame["hour_bjt"].copy()
        elif not expected_times.equals(frame["hour_bjt"]):
            raise ValueError(f"Pressure timestamp grid differs in {path.name}")

        missing_columns = set(aggregation_columns).difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} lacks pressure columns: {sorted(missing_columns)}"
            )

        finite_height = frame["pressure_geometric_height_m_amsl"].dropna()
        low, high = LAYER_BOUNDS_M[PRESSURE_LEVEL_TO_LAYER[level]]
        if not finite_height.empty and (
            finite_height.min() < low or finite_height.max() >= high
        ):
            raise ValueError(
                f"{level} hPa height range crosses its configured layer boundary"
            )
        height_report.append(
            {
                "pressure_level_hpa": level,
                "layer": PRESSURE_LEVEL_TO_LAYER[level],
                "finite_height_hours": int(finite_height.size),
                "height_min_m": None if finite_height.empty else float(finite_height.min()),
                "height_median_m": None
                if finite_height.empty
                else float(finite_height.median()),
                "height_max_m": None if finite_height.empty else float(finite_height.max()),
                "valid_los_hours": int(frame["los_intersection_valid"].fillna(False).sum()),
            }
        )

        long = frame[["hour_bjt", "los_intersection_valid", *aggregation_columns]].copy()
        long["pressure_level_hpa"] = level
        long["layer"] = PRESSURE_LEVEL_TO_LAYER[level]
        long_frames.append(long)

        non_time = [
            column
            for column in frame.columns
            if column not in {"valid_time_bjt", "valid_time_utc", "hour_bjt"}
        ]
        wide = frame[["hour_bjt", *non_time]].rename(
            columns={column: f"p{level}__{column}" for column in non_time}
        )
        wide_frames.append(wide)

    pressure_wide = wide_frames[0]
    for frame in wide_frames[1:]:
        pressure_wide = pressure_wide.merge(
            frame, on="hour_bjt", how="inner", validate="one_to_one"
        )
    if len(pressure_wide) != 7_800:
        raise ValueError("Pressure-level wide table lost rows during merging")

    pressure_long = pd.concat(long_frames, ignore_index=True)
    layer_frames: list[pd.DataFrame] = []
    for layer, bounds in LAYER_BOUNDS_M.items():
        subset = pressure_long[pressure_long["layer"].eq(layer)].copy()
        configured_levels = sorted(
            level for level, configured in PRESSURE_LEVEL_TO_LAYER.items() if configured == layer
        )
        observed_levels = sorted(subset["pressure_level_hpa"].unique().tolist())
        if observed_levels != configured_levels:
            raise ValueError(
                f"Layer {layer} levels differ: {observed_levels} != {configured_levels}"
            )

        means = (
            subset.groupby("hour_bjt", sort=True, observed=True)[aggregation_columns]
            .mean()
            .reset_index()
        )
        valid_counts = (
            subset.assign(valid_flag=subset["los_intersection_valid"].fillna(False).astype(int))
            .groupby("hour_bjt", sort=True, observed=True)["valid_flag"]
            .sum()
            .reset_index(name="n_valid_pressure_levels")
        )
        layer_frame = means.merge(
            valid_counts, on="hour_bjt", how="left", validate="one_to_one"
        )
        layer_frame["n_expected_pressure_levels"] = len(configured_levels)
        layer_frame["valid_pressure_level_fraction"] = (
            layer_frame["n_valid_pressure_levels"] / len(configured_levels)
        )
        layer_frame["height_lower_bound_m"] = bounds[0]
        layer_frame["height_upper_bound_m"] = bounds[1]
        layer_frame = layer_frame.rename(
            columns={
                column: f"layer_{layer}__{column}"
                for column in layer_frame.columns
                if column != "hour_bjt"
            }
        )
        layer_frames.append(layer_frame)

    layer_wide = layer_frames[0]
    for frame in layer_frames[1:]:
        layer_wide = layer_wide.merge(
            frame, on="hour_bjt", how="inner", validate="one_to_one"
        )
    if len(layer_wide) != 7_800:
        raise ValueError("Five-layer table lost rows during merging")

    pressure_feature_columns = [
        column
        for column in pressure_wide.columns
        if re.match(r"p\d+__(" + "|".join(map(re.escape, pressure_feature_names)) + r")$", column)
    ]
    layer_feature_columns = [
        column
        for column in layer_wide.columns
        if any(column.endswith(f"__{name}") for name in pressure_feature_names)
    ]
    report = {
        "level_height_ranges": sorted(
            height_report, key=lambda row: -row["pressure_level_hpa"]
        ),
        "layer_mapping": {
            layer: sorted(
                level
                for level, configured in PRESSURE_LEVEL_TO_LAYER.items()
                if configured == layer
            )
            for layer in LAYER_BOUNDS_M
        },
        "layer_aggregation": "unweighted arithmetic mean across available configured pressure levels; missing levels remain absent from the mean and are counted",
    }
    return (
        pressure_wide,
        layer_wide,
        report,
        pressure_feature_columns,
        layer_feature_columns,
    )


def add_missingness_flags(
    frame: pd.DataFrame,
    surface_features: list[str],
    atmosphere_features: list[str],
    atmosphere_label: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["surface__nonmissing_feature_fraction"] = result[surface_features].notna().mean(axis=1)
    result[f"{atmosphere_label}__nonmissing_feature_fraction"] = (
        result[atmosphere_features].notna().mean(axis=1)
    )
    result["all_selected_era5_features_present"] = (
        result[surface_features].notna().all(axis=1)
        & result[atmosphere_features].notna().all(axis=1)
    )
    return result


def description_and_unit(
    column: str, metadata: dict[str, Any]
) -> tuple[str, str, str]:
    measurement = {
        "hour_bjt": ("time", "Matched hourly timestamp in Asia/Shanghai", "ISO 8601"),
        "hour_utc": ("time", "Same matched hourly timestamp in UTC", "ISO 8601"),
        "measurement_rows": ("measurement_qa", "Number of valid source measurement rows", "count"),
        "valid_10min_bins": ("measurement_qa", "Number of non-empty 10-minute bins in the hour", "count"),
        "measurement_first_time_bjt": ("measurement_qa", "First valid 10-minute bin timestamp", "ISO 8601"),
        "measurement_last_time_bjt": ("measurement_qa", "Last valid 10-minute bin timestamp", "ISO 8601"),
        "ten_min_power_dbm_min": ("measurement_qa", "Minimum 10-minute power in the hour", "dBm"),
        "ten_min_power_dbm_median": ("measurement_qa", "Median 10-minute power in the hour", "dBm"),
        "ten_min_power_dbm_max": ("measurement_qa", "Maximum 10-minute power in the hour", "dBm"),
        "ten_min_power_dbm_std": ("measurement_qa", "Sample standard deviation of 10-minute dBm values", "dB"),
        "received_power_w": ("target", "Row-count-weighted hourly mean power, averaged in W", "W"),
        "received_power_dbm": ("target", "Hourly mean received power converted from received_power_w", "dBm"),
        "measurement_10min_coverage_fraction": ("measurement_qa", "valid_10min_bins divided by six", "1"),
        "measurement_full_hour": ("measurement_qa", "True when all six 10-minute bins are present", "boolean"),
        "surface__nonmissing_feature_fraction": ("era5_qa", "Fraction of configured surface ERA5 features present", "1"),
        "pressure__nonmissing_feature_fraction": ("era5_qa", "Fraction of configured pressure-level features present", "1"),
        "layer__nonmissing_feature_fraction": ("era5_qa", "Fraction of configured five-layer features present", "1"),
        "all_selected_era5_features_present": ("era5_qa", "True when all configured surface and atmosphere features are present", "boolean"),
    }
    if column in measurement:
        return measurement[column]

    if column.startswith("surface__"):
        name = column.split("__", 1)[1]
        info = metadata["surface_features"].get(name)
        if info:
            return "era5_surface", info["long_name"], info["units"]
        geometry = {
            "surface_latitude_deg_n": ("Station latitude", "degree_north"),
            "surface_longitude_deg_e": ("Station longitude", "degree_east"),
            "antenna_phase_center_m_amsl": ("Antenna phase-centre height AMSL", "m"),
        }
        if name in geometry:
            return "era5_surface_geometry", *geometry[name]

    pressure_match = re.match(r"p(\d+)__(.+)", column)
    if pressure_match:
        level, name = pressure_match.groups()
        info = metadata["pressure_features"].get(name)
        if info:
            return f"era5_pressure_{level}hpa", info["long_name"], info["units"]
        return f"era5_pressure_{level}hpa_metadata", name, "see source metadata"

    layer_match = re.match(r"layer_([^_]+_[^_]+km)__(.+)", column)
    if layer_match:
        layer, name = layer_match.groups()
        info = metadata["pressure_features"].get(name)
        if info:
            return f"era5_layer_{layer}", f"Layer mean of {info['long_name']}", info["units"]
        units = {
            "n_valid_pressure_levels": "count",
            "n_expected_pressure_levels": "count",
            "valid_pressure_level_fraction": "1",
            "height_lower_bound_m": "m",
            "height_upper_bound_m": "m",
        }
        return f"era5_layer_{layer}_metadata", name, units.get(name, "see source metadata")

    return "other", column, "unknown"


def build_dictionary(
    datasets: dict[str, pd.DataFrame], metadata: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, frame in datasets.items():
        for position, column in enumerate(frame.columns, start=1):
            group, description, units = description_and_unit(column, metadata)
            rows.append(
                {
                    "dataset": dataset_name,
                    "column_position": position,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "group": group,
                    "units": units,
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.DatetimeTZDtype):
            result[column] = result[column].map(
                lambda value: value.isoformat() if pd.notna(value) else ""
            )
    return result


def write_dataset(
    frame: pd.DataFrame, stem: Path, output_root: Path
) -> dict[str, Any]:
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    csv_ready(frame).to_csv(csv_path, index=False, encoding="utf-8-sig")

    parquet_readback = pd.read_parquet(parquet_path)
    csv_readback = pd.read_csv(csv_path)
    if len(parquet_readback) != len(frame) or len(csv_readback) != len(frame):
        raise ValueError(f"Read-back row count failed for {stem.name}")
    if list(parquet_readback.columns) != list(frame.columns):
        raise ValueError(f"Parquet read-back schema failed for {stem.name}")
    if list(csv_readback.columns) != list(frame.columns):
        raise ValueError(f"CSV read-back schema failed for {stem.name}")

    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "first_hour_bjt": frame["hour_bjt"].min().isoformat(),
        "last_hour_bjt": frame["hour_bjt"].max().isoformat(),
        "csv": output_identity(csv_path, output_root),
        "parquet": output_identity(parquet_path, output_root),
    }


def main() -> int:
    args = parse_args()
    era5_csv_dir = args.era5_csv_dir.resolve()
    measurement_run_dir = args.measurement_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    temp_dir = output_dir.with_name(f"{output_dir.name}.tmp")

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_dir}")

    overview_path = measurement_run_dir / "overview_10min.csv"
    measurement_manifest_path = measurement_run_dir / "run_manifest.json"
    metadata_path = era5_csv_dir / "feature_metadata.json"
    era5_quality_path = era5_csv_dir / "quality_report.csv"
    surface_path = era5_csv_dir / "era5_surface_bjtu_hourly_bjt.csv"
    pressure_paths = sorted(era5_csv_dir.glob("era5_pressure_*hPa_los_hourly_bjt.csv"))
    required_paths = [
        overview_path,
        measurement_manifest_path,
        metadata_path,
        era5_quality_path,
        surface_path,
        *pressure_paths,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required inputs: {missing_paths}")

    input_snapshot_before = {
        str(path): file_identity(path) for path in required_paths
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    started_utc = datetime.now(timezone.utc)
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        measurements, measurement_summary = aggregate_measurements(overview_path)
        surface, surface_feature_columns = load_surface(surface_path, metadata)
        (
            pressure_wide,
            layer_wide,
            pressure_report,
            pressure_feature_columns,
            layer_feature_columns,
        ) = load_pressure_levels(era5_csv_dir, metadata)

        era5_pressure = surface.merge(
            pressure_wide, on="hour_bjt", how="inner", validate="one_to_one"
        )
        era5_layers = surface.merge(
            layer_wide, on="hour_bjt", how="inner", validate="one_to_one"
        )
        if len(era5_pressure) != 7_800 or len(era5_layers) != 7_800:
            raise ValueError("ERA5 table row count changed during surface merge")

        measurement_hours = set(measurements["hour_bjt"])
        era5_hours = set(era5_pressure["hour_bjt"])
        unmatched_measurement_hours = sorted(measurement_hours.difference(era5_hours))
        if unmatched_measurement_hours:
            raise ValueError(
                "Measurement hours without exact ERA5 match: "
                + ", ".join(map(str, unmatched_measurement_hours[:10]))
            )

        pressure_dataset = measurements.merge(
            era5_pressure, on="hour_bjt", how="inner", validate="one_to_one"
        )
        layer_dataset = measurements.merge(
            era5_layers, on="hour_bjt", how="inner", validate="one_to_one"
        )
        pressure_dataset = add_missingness_flags(
            pressure_dataset,
            surface_feature_columns,
            pressure_feature_columns,
            "pressure",
        )
        layer_dataset = add_missingness_flags(
            layer_dataset,
            surface_feature_columns,
            layer_feature_columns,
            "layer",
        )

        for label, frame in {
            "pressure": pressure_dataset,
            "layers": layer_dataset,
        }.items():
            if len(frame) != len(measurements):
                raise ValueError(f"{label} matched data set lost measurement hours")
            require_unique_time(frame, "hour_bjt", f"matched {label} data set")
            if not frame["hour_bjt"].is_monotonic_increasing:
                raise ValueError(f"Matched {label} data set is not time sorted")
            if not np.isfinite(frame["received_power_dbm"]).all():
                raise ValueError(f"Matched {label} data set contains invalid target values")

        full_layer_dataset = layer_dataset[layer_dataset["measurement_full_hour"]].copy()
        if full_layer_dataset.empty:
            raise ValueError("No complete six-bin measurement hours were found")

        datasets = {
            "measurement_hourly": measurements,
            "matched_hourly_pressure_levels_all_measurements": pressure_dataset,
            "matched_hourly_five_layers_all_measurements": layer_dataset,
            "matched_hourly_five_layers_full_measurement_hours": full_layer_dataset,
        }
        output_records: dict[str, Any] = {}
        for name, frame in datasets.items():
            output_records[name] = write_dataset(frame, temp_dir / name, temp_dir)

        dictionary = build_dictionary(datasets, metadata)
        dictionary_path = temp_dir / "data_dictionary.csv"
        dictionary.to_csv(dictionary_path, index=False, encoding="utf-8-sig")

        monthly_coverage = (
            layer_dataset.assign(
                month=layer_dataset["hour_bjt"].dt.to_period("M").astype(str)
            )
            .groupby("month", sort=True)
            .agg(
                matched_hours=("hour_bjt", "size"),
                full_measurement_hours=("measurement_full_hour", "sum"),
                measurement_rows=("measurement_rows", "sum"),
                mean_received_power_dbm=("received_power_dbm", "mean"),
            )
            .reset_index()
        )
        monthly_path = temp_dir / "monthly_coverage.csv"
        monthly_coverage.to_csv(monthly_path, index=False, encoding="utf-8-sig")

        missing_surface = {
            column: int(layer_dataset[column].isna().sum())
            for column in surface_feature_columns
            if layer_dataset[column].isna().any()
        }
        missing_layer = {
            column: int(layer_dataset[column].isna().sum())
            for column in layer_feature_columns
            if layer_dataset[column].isna().any()
        }
        missing_pressure = {
            column: int(pressure_dataset[column].isna().sum())
            for column in pressure_feature_columns
            if pressure_dataset[column].isna().any()
        }

        input_snapshot_after = {
            str(path): file_identity(path) for path in required_paths
        }
        inputs_unchanged = input_snapshot_before == input_snapshot_after
        if not inputs_unchanged:
            raise RuntimeError("One or more read-only input files changed during the run")

        completed_utc = datetime.now(timezone.utc)
        manifest: dict[str, Any] = {
            "status": "VERIFIED",
            "started_utc": started_utc.isoformat(),
            "completed_utc": completed_utc.isoformat(),
            "elapsed_seconds": (completed_utc - started_utc).total_seconds(),
            "requested_time_range": {
                "timezone": TZ_BJT,
                "start_inclusive": START_BJT.isoformat(),
                "end_inclusive": (END_EXCLUSIVE_BJT - pd.Timedelta(hours=1)).isoformat(),
            },
            "measurement_contract": {
                "source": str(overview_path),
                "source_derivation": "Run_009 verified 11-bin integrated peak power",
                "hourly_mean": "convert 10-minute dBm means to W, weight by source row count, average in W, convert once to dBm",
                "empty_periods": "not interpolated; hours without measurements are absent",
                "target": "received_power_dbm and received_power_w; no attenuation label is created",
                "time_interpretation": "source measurement timestamps are treated as Asia/Shanghai local time",
            },
            "era5_contract": {
                "source": str(era5_csv_dir),
                "join": "exact one-to-one match on Asia/Shanghai hourly timestamp",
                "missing_values": "retained as NaN; no target-informed or temporal imputation",
                "pressure_interpolation": metadata["pressure_interpolation"],
                "below_antenna_policy": metadata["below_antenna_policy"],
            },
            "five_layer_contract": pressure_report,
            "measurement_summary": measurement_summary,
            "matched_summary": {
                "matched_hours": len(layer_dataset),
                "full_measurement_hours": len(full_layer_dataset),
                "partial_measurement_hours": len(layer_dataset) - len(full_layer_dataset),
                "unmatched_measurement_hours": 0,
                "first_hour_bjt": layer_dataset["hour_bjt"].min().isoformat(),
                "last_hour_bjt": layer_dataset["hour_bjt"].max().isoformat(),
                "months_with_measurements": monthly_coverage["month"].tolist(),
                "surface_feature_count": len(surface_feature_columns),
                "pressure_feature_count": len(pressure_feature_columns),
                "five_layer_feature_count": len(layer_feature_columns),
                "missing_surface_feature_values_by_column": missing_surface,
                "missing_pressure_feature_values_by_column": missing_pressure,
                "missing_layer_feature_values_by_column": missing_layer,
            },
            "input_files": list(input_snapshot_after.values()),
            "outputs": output_records,
            "auxiliary_outputs": {
                "data_dictionary": output_identity(dictionary_path, temp_dir),
                "monthly_coverage": output_identity(monthly_path, temp_dir),
            },
            "validation": {
                "read_only_inputs_unchanged": inputs_unchanged,
                "measurement_10min_timestamps_unique": True,
                "measurement_hours_unique": True,
                "era5_hour_grids_unique": True,
                "exact_measurement_era5_match": True,
                "hourly_power_averaged_in_linear_watts": True,
                "missing_measurement_hours_not_interpolated": True,
                "reference_power_or_attenuation_not_imposed": True,
                "csv_and_parquet_readback_row_counts_passed": True,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        manifest_path = temp_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        qa_lines = [
            "# Revision_001 数据集质量报告",
            "",
            "- 状态：VERIFIED",
            f"- 构建时间：{completed_utc.astimezone().isoformat()}",
            f"- 时间范围：{START_BJT.isoformat()} 至 {(END_EXCLUSIVE_BJT - pd.Timedelta(hours=1)).isoformat()}",
            f"- 有实测且与 ERA5 精确匹配的小时：{len(layer_dataset):,}",
            f"- 六个 10 分钟箱均有效的完整小时：{len(full_layer_dataset):,}",
            f"- 部分覆盖小时：{len(layer_dataset) - len(full_layer_dataset):,}",
            f"- 实测原始行数（指定范围内）：{measurement_summary['measurement_rows_in_range']:,}",
            f"- 首个匹配小时：{layer_dataset['hour_bjt'].min().isoformat()}",
            f"- 最后匹配小时：{layer_dataset['hour_bjt'].max().isoformat()}",
            "",
            "## 已通过的检查",
            "",
            "- ERA5 与实测按北京时间整点一对一精确匹配，未匹配实测小时为 0。",
            "- 实测小时功率在 W 域按源行数加权平均，最后一次性转换为 dBm。",
            "- 无测量小时没有进入数据集，空 10 分钟箱没有插值。",
            "- 1000 hPa 无有效 LOS 交点的值保持 NaN，没有向天线以下外推。",
            "- 输出 CSV 和 Parquet 已回读核对行数与字段顺序。",
            "- 所有直接输入文件的大小、修改时间和 SHA-256 在运行前后相同。",
            "",
            "## 重要限制",
            "",
            "- 实测 TIME 按北京时间解释；当前 H5 未提供独立时区字段用于再次证明该解释。",
            "- dBm 沿用 Run_009 的 RMS 电压、50 Ω 和 11 频段积分口径，不代表接收链路独立绝对标定。",
            "- 本版不使用 −79.38 dBm 或 −78.56 dBm 构造衰减标签；参考功率须另行校准。",
            "- 五层特征是配置压力层的等权算术平均，不是厚度加权、质量加权或传播损耗积分。",
            "- 同站点跨月份样本不能单独证明跨地点或跨气候泛化。",
            "",
            "## 月份覆盖",
            "",
            monthly_coverage.to_markdown(index=False),
            "",
        ]
        qa_path = temp_dir / "QA.md"
        qa_path.write_text("\n".join(qa_lines), encoding="utf-8")

        manifest["auxiliary_outputs"]["qa"] = output_identity(qa_path, temp_dir)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        checksum_lines = []
        for path in sorted(temp_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        checksum_path = temp_dir / "checksums.sha256"
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")

        temp_dir.replace(output_dir)
        print(json.dumps(manifest["matched_summary"], ensure_ascii=False, indent=2))
        print(f"OUTPUT_DIR={output_dir}")
        return 0
    except Exception:
        failure_path = temp_dir / "FAILED.txt"
        if temp_dir.exists():
            failure_path.write_text(
                f"Build failed at {datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
