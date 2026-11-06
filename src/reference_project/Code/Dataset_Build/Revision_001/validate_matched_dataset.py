from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TZ_BJT = "Asia/Shanghai"
START_BJT = pd.Timestamp("2025-09-30 00:00:00", tz=TZ_BJT)
END_EXCLUSIVE_BJT = pd.Timestamp("2026-08-21 00:00:00", tz=TZ_BJT)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Datasets" / "Revision_001"
BUILD_SCRIPT = Path(__file__).with_name("build_matched_dataset.py")
ERA5_CSV_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Data\era5_campaign_20250930_20260820\csv"
)
MEASUREMENT_OVERVIEW = Path(
    r"E:\0_Workflow_System\01_PhD\Data\Further analysis of the data results"
    r"\思源楼高轨卫星信标信号实测数据_再分析结果\Runs\Run_009\overview_10min.csv"
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


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_bjt(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(TZ_BJT)
    return parsed.dt.tz_convert(TZ_BJT)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def numeric_difference(left: pd.Series, right: pd.Series) -> tuple[bool, float, float]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    same_nan = np.array_equal(np.isnan(left_values), np.isnan(right_values))
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if finite.any():
        abs_diff = np.abs(left_values[finite] - right_values[finite])
        scale = np.maximum(np.abs(right_values[finite]), 1e-300)
        max_abs = float(abs_diff.max())
        max_rel = float((abs_diff / scale).max())
    else:
        max_abs = 0.0
        max_rel = 0.0
    passed = same_nan and bool(
        np.allclose(left_values, right_values, rtol=5e-11, atol=1e-12, equal_nan=True)
    )
    return passed, max_abs, max_rel


def compare_columns(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
    label: str,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    for column in columns:
        if column not in actual.columns or column not in expected.columns:
            failures.append({"column": column, "reason": "missing column"})
            continue
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            passed, column_abs, column_rel = numeric_difference(left, right)
            max_abs = max(max_abs, column_abs)
            max_rel = max(max_rel, column_rel)
            if not passed:
                failures.append(
                    {
                        "column": column,
                        "reason": "numeric mismatch",
                        "max_abs_difference": column_abs,
                        "max_relative_difference": column_rel,
                    }
                )
        else:
            left_text = left.astype("string").fillna("<NA>")
            right_text = right.astype("string").fillna("<NA>")
            if not left_text.equals(right_text):
                failures.append({"column": column, "reason": "value mismatch"})
    return {
        "name": label,
        "passed": not failures,
        "columns_checked": len(columns),
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "failures": failures[:25],
    }


def align_on_hour(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hour_bjt"] = parse_bjt(result["hour_bjt"])
    return result.sort_values("hour_bjt").set_index("hour_bjt", drop=False)


def reconstruct_measurement_hourly() -> pd.DataFrame:
    raw = pd.read_csv(MEASUREMENT_OVERVIEW)
    raw["time_bjt"] = parse_bjt(raw["time"])
    valid = raw[
        raw["count"].fillna(0).gt(0)
        & np.isfinite(raw["value"])
        & np.isfinite(raw["count"])
        & raw["time_bjt"].ge(START_BJT)
        & raw["time_bjt"].lt(END_EXCLUSIVE_BJT)
    ].copy()
    valid["count"] = valid["count"].astype("int64")
    valid["time_bjt"] = valid["time_bjt"].dt.floor("10min")
    valid["hour_bjt"] = valid["time_bjt"].dt.floor("h")
    valid["power_w"] = np.power(10.0, (valid["value"] - 30.0) / 10.0)
    valid["weighted_power_w"] = valid["power_w"] * valid["count"]
    grouped = valid.groupby("hour_bjt", sort=True, observed=True)
    hourly = grouped.agg(
        measurement_rows=("count", "sum"),
        valid_10min_bins=("count", "size"),
        weighted_power_w=("weighted_power_w", "sum"),
        measurement_first_time_bjt=("time_bjt", "min"),
        measurement_last_time_bjt=("time_bjt", "max"),
        ten_min_power_dbm_min=("value", "min"),
        ten_min_power_dbm_median=("value", "median"),
        ten_min_power_dbm_max=("value", "max"),
        ten_min_power_dbm_std=("value", "std"),
    ).reset_index()
    hourly["received_power_w"] = hourly["weighted_power_w"] / hourly["measurement_rows"]
    hourly["received_power_dbm"] = 10.0 * np.log10(hourly["received_power_w"] * 1000.0)
    hourly["measurement_10min_coverage_fraction"] = hourly["valid_10min_bins"] / 6.0
    hourly["measurement_full_hour"] = hourly["valid_10min_bins"].eq(6)
    hourly["hour_utc"] = hourly["hour_bjt"].dt.tz_convert("UTC")
    return hourly.drop(columns=["weighted_power_w"])


def verify_checksum_file() -> dict[str, Any]:
    path = OUTPUT_DIR / "checksums.sha256"
    failures: list[dict[str, str]] = []
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        target = OUTPUT_DIR / name.lstrip(" *")
        checked += 1
        if not target.is_file():
            failures.append({"path": target.name, "reason": "missing"})
        elif sha256_file(target) != expected.upper():
            failures.append({"path": target.name, "reason": "SHA-256 mismatch"})
    return {
        "name": "existing_output_checksums",
        "passed": not failures,
        "files_checked": checked,
        "failures": failures,
    }


def verify_manifest_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for identity in manifest["input_files"]:
        path = Path(identity["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        if path.stat().st_size != identity["bytes"]:
            failures.append({"path": str(path), "reason": "size changed"})
            continue
        if sha256_file(path) != identity["sha256"]:
            failures.append({"path": str(path), "reason": "SHA-256 changed"})
    return {
        "name": "direct_input_files_unchanged",
        "passed": not failures,
        "files_checked": len(manifest["input_files"]),
        "failures": failures,
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "started_utc": started.isoformat(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256_before_run": sha256_file(Path(__file__)),
        "build_script": str(BUILD_SCRIPT),
        "build_script_sha256": sha256_file(BUILD_SCRIPT),
        "checks": [],
    }
    checks: list[dict[str, Any]] = report["checks"]

    manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    measurement = align_on_hour(pd.read_parquet(OUTPUT_DIR / "measurement_hourly.parquet"))
    pressure = align_on_hour(
        pd.read_parquet(OUTPUT_DIR / "matched_hourly_pressure_levels_all_measurements.parquet")
    )
    layers = align_on_hour(
        pd.read_parquet(OUTPUT_DIR / "matched_hourly_five_layers_all_measurements.parquet")
    )
    full_layers = align_on_hour(
        pd.read_parquet(OUTPUT_DIR / "matched_hourly_five_layers_full_measurement_hours.parquet")
    )

    expected_shapes = {
        "measurement_hourly": (3821, 14),
        "pressure_levels": (3821, 483),
        "five_layers": (3821, 348),
        "five_layers_full_measurement_hours": (3769, 348),
    }
    actual_frames = {
        "measurement_hourly": measurement,
        "pressure_levels": pressure,
        "five_layers": layers,
        "five_layers_full_measurement_hours": full_layers,
    }
    shape_failures = [
        {"dataset": name, "actual": list(actual_frames[name].shape), "expected": list(shape)}
        for name, shape in expected_shapes.items()
        if actual_frames[name].shape != shape
    ]
    checks.append(
        {
            "name": "expected_shapes",
            "passed": not shape_failures,
            "expected": {key: list(value) for key, value in expected_shapes.items()},
            "failures": shape_failures,
        }
    )

    temporal_failures: list[str] = []
    for name, frame in actual_frames.items():
        if frame.index.has_duplicates:
            temporal_failures.append(f"{name}: duplicate hours")
        if not frame.index.is_monotonic_increasing:
            temporal_failures.append(f"{name}: hours are not sorted")
        if not (frame.index >= START_BJT).all() or not (frame.index < END_EXCLUSIVE_BJT).all():
            temporal_failures.append(f"{name}: hour outside requested interval")
    if not measurement.index.equals(pressure.index) or not measurement.index.equals(layers.index):
        temporal_failures.append("all-measurement products do not share the same hours")
    expected_full_index = measurement.index[measurement["measurement_full_hour"].astype(bool)]
    if not expected_full_index.equals(full_layers.index):
        temporal_failures.append("six-bin subset does not equal measurement_full_hour filter")
    checks.append(
        {
            "name": "temporal_keys_and_exact_matching",
            "passed": not temporal_failures,
            "first_hour_bjt": measurement.index.min(),
            "last_hour_bjt": measurement.index.max(),
            "unmatched_measurement_hours": int(len(measurement.index.difference(pressure.index))),
            "failures": temporal_failures,
        }
    )

    reconstructed = align_on_hour(reconstruct_measurement_hourly())
    measurement_columns = [column for column in reconstructed.columns if column != "hour_bjt"]
    measurement_check = compare_columns(
        measurement,
        reconstructed,
        measurement_columns,
        "measurement_hourly_independent_reconstruction",
    )
    measurement_check["rows_reconstructed"] = len(reconstructed)
    measurement_check["time_keys_equal"] = measurement.index.equals(reconstructed.index)
    measurement_check["passed"] = bool(
        measurement_check["passed"] and measurement_check["time_keys_equal"]
    )
    checks.append(measurement_check)

    metadata = json.loads((ERA5_CSV_DIR / "feature_metadata.json").read_text(encoding="utf-8"))
    raw_surface = pd.read_csv(ERA5_CSV_DIR / "era5_surface_bjtu_hourly_bjt.csv")
    raw_surface["hour_bjt"] = parse_bjt(raw_surface["valid_time_bjt"])
    surface_columns_raw = [
        column
        for column in raw_surface.columns
        if column not in {"valid_time_bjt", "valid_time_utc", "hour_bjt"}
    ]
    expected_surface = align_on_hour(
        raw_surface[["hour_bjt", *surface_columns_raw]].rename(
            columns={column: f"surface__{column}" for column in surface_columns_raw}
        )
    ).loc[measurement.index]
    checks.append(
        compare_columns(
            pressure,
            expected_surface,
            [f"surface__{column}" for column in surface_columns_raw],
            "all_surface_columns_against_era5_source",
        )
    )

    pressure_source_frames: dict[int, pd.DataFrame] = {}
    pressure_checks: list[dict[str, Any]] = []
    pressure_paths = sorted(ERA5_CSV_DIR.glob("era5_pressure_*hPa_los_hourly_bjt.csv"))
    for path in pressure_paths:
        match = re.search(r"pressure_(\d+)hPa", path.name)
        if match is None:
            continue
        level = int(match.group(1))
        frame = pd.read_csv(path)
        frame["hour_bjt"] = parse_bjt(frame["valid_time_bjt"])
        pressure_source_frames[level] = frame
        raw_columns = [
            column
            for column in frame.columns
            if column not in {"valid_time_bjt", "valid_time_utc", "hour_bjt"}
        ]
        expected_level = align_on_hour(
            frame[["hour_bjt", *raw_columns]].rename(
                columns={column: f"p{level}__{column}" for column in raw_columns}
            )
        ).loc[measurement.index]
        pressure_checks.append(
            compare_columns(
                pressure,
                expected_level,
                [f"p{level}__{column}" for column in raw_columns],
                f"pressure_{level}hPa_against_era5_source",
            )
        )
    pressure_failures = [check for check in pressure_checks if not check["passed"]]
    checks.append(
        {
            "name": "all_pressure_level_columns_against_era5_source",
            "passed": not pressure_failures and len(pressure_checks) == 11,
            "levels_checked": len(pressure_checks),
            "columns_checked": int(sum(check["columns_checked"] for check in pressure_checks)),
            "max_abs_difference": max(
                (check["max_abs_difference"] for check in pressure_checks), default=0.0
            ),
            "failures": pressure_failures,
        }
    )

    aggregation_columns = [*PRESSURE_GEOMETRY_COLUMNS, *metadata["pressure_features"]]
    layer_checks: list[dict[str, Any]] = []
    for layer, bounds in LAYER_BOUNDS_M.items():
        configured_levels = sorted(
            level for level, assigned_layer in PRESSURE_LEVEL_TO_LAYER.items() if assigned_layer == layer
        )
        pieces = []
        for level in configured_levels:
            piece = pressure_source_frames[level][
                ["hour_bjt", "los_intersection_valid", *aggregation_columns]
            ].copy()
            piece["pressure_level_hpa"] = level
            pieces.append(piece)
        long = pd.concat(pieces, ignore_index=True)
        means = long.groupby("hour_bjt", sort=True, observed=True)[aggregation_columns].mean()
        valid_counts = (
            long.assign(valid_flag=long["los_intersection_valid"].fillna(False).astype(int))
            .groupby("hour_bjt", sort=True, observed=True)["valid_flag"]
            .sum()
        )
        expected_layer = means.copy()
        expected_layer["n_valid_pressure_levels"] = valid_counts
        expected_layer["n_expected_pressure_levels"] = len(configured_levels)
        expected_layer["valid_pressure_level_fraction"] = valid_counts / len(configured_levels)
        expected_layer["height_lower_bound_m"] = bounds[0]
        expected_layer["height_upper_bound_m"] = bounds[1]
        expected_layer = expected_layer.add_prefix(f"layer_{layer}__")
        expected_layer.index.name = "hour_bjt"
        expected_layer = expected_layer.reset_index()
        expected_layer = align_on_hour(expected_layer).loc[measurement.index]
        expected_columns = [column for column in expected_layer.columns if column != "hour_bjt"]
        layer_checks.append(
            compare_columns(
                layers,
                expected_layer,
                expected_columns,
                f"layer_{layer}_independent_reconstruction",
            )
        )
    layer_failures = [check for check in layer_checks if not check["passed"]]
    checks.append(
        {
            "name": "five_layers_independent_reconstruction",
            "passed": not layer_failures and len(layer_checks) == 5,
            "layers_checked": len(layer_checks),
            "columns_checked": int(sum(check["columns_checked"] for check in layer_checks)),
            "max_abs_difference": max(
                (check["max_abs_difference"] for check in layer_checks), default=0.0
            ),
            "failures": layer_failures,
        }
    )

    csv_parquet_pairs = [
        "measurement_hourly",
        "matched_hourly_pressure_levels_all_measurements",
        "matched_hourly_five_layers_all_measurements",
        "matched_hourly_five_layers_full_measurement_hours",
    ]
    pair_failures = []
    for stem in csv_parquet_pairs:
        csv_frame = pd.read_csv(OUTPUT_DIR / f"{stem}.csv")
        parquet_frame = pd.read_parquet(OUTPUT_DIR / f"{stem}.parquet")
        if csv_frame.shape != parquet_frame.shape or list(csv_frame.columns) != list(parquet_frame.columns):
            pair_failures.append(
                {
                    "dataset": stem,
                    "csv_shape": list(csv_frame.shape),
                    "parquet_shape": list(parquet_frame.shape),
                    "same_column_order": list(csv_frame.columns) == list(parquet_frame.columns),
                }
            )
    checks.append(
        {
            "name": "csv_parquet_shapes_and_column_order",
            "passed": not pair_failures,
            "pairs_checked": len(csv_parquet_pairs),
            "failures": pair_failures,
        }
    )
    checks.append(verify_checksum_file())
    checks.append(verify_manifest_inputs(manifest))

    monthly = (
        measurement.assign(month=measurement["hour_bjt"].dt.strftime("%Y-%m"))
        .groupby("month", sort=True)
        .agg(
            matched_hours=("hour_bjt", "size"),
            six_bin_hours=("measurement_full_hour", "sum"),
            measurement_rows=("measurement_rows", "sum"),
        )
        .reset_index()
    )
    report["sampling_density_by_month"] = monthly.to_dict("records")
    report["sampling_density_caveat"] = (
        "Six nonempty 10-minute bins indicate temporal-bin coverage only. "
        "They do not imply equal raw measurement-row density across hours or months."
    )
    report["missingness_notes"] = {
        "p1000_temperature_missing_hours": int(pressure["p1000__temperature"].isna().sum()),
        "five_layer_feature_missing_values": int(
            layers.filter(regex=r"^layer_.*__(?:" + "|".join(map(re.escape, metadata["pressure_features"])) + r")$")
            .isna()
            .sum()
            .sum()
        ),
    }

    passed = all(bool(check["passed"]) for check in checks)
    completed = datetime.now(timezone.utc)
    report["status"] = "VERIFIED" if passed else "FAILED"
    report["completed_utc"] = completed.isoformat()
    report["elapsed_seconds"] = (completed - started).total_seconds()
    report["check_count"] = len(checks)
    report["passed_check_count"] = sum(bool(check["passed"]) for check in checks)

    report_path = OUTPUT_DIR / "validation_report.json"
    report_path.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# 独立验证报告",
        "",
        f"- 状态：**{report['status']}**",
        f"- 检查：{report['passed_check_count']} / {report['check_count']} 通过",
        f"- 构建脚本 SHA-256：{report['build_script_sha256']}",
        f"- 验证脚本 SHA-256：{report['validator_sha256_before_run']}",
        "",
        "## 核心结论",
        "",
        f"- 匹配小时：{len(measurement):,}；其中六个十分钟箱均非空：{int(measurement['measurement_full_hour'].sum()):,}。",
        f"- 时间范围：{measurement.index.min().isoformat()} 至 {measurement.index.max().isoformat()}。",
        "- 匹配方法：北京时间整点一对一精确连接；无实测小时不进入数据集，不做时间插值。",
        "- 实测小时功率已从源 10 分钟结果独立在 W 域按原始行数加权复算。",
        "- 地面、11 个压力层和 5 个高度层已逐列对源数据或独立聚合结果复算。",
        "- 注意：六个十分钟箱齐全只代表箱覆盖，不代表不同小时或月份的原始采样点数相同。",
        "",
        "## 检查明细",
        "",
    ]
    for check in checks:
        markdown.append(f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}")
    markdown.extend(["", "## 月度采样密度", "", "| 月份 | 匹配小时 | 六箱小时 | 原始测量行 |", "|---|---:|---:|---:|"])
    for row in report["sampling_density_by_month"]:
        markdown.append(
            f"| {row['month']} | {row['matched_hours']:,} | {int(row['six_bin_hours']):,} | {row['measurement_rows']:,} |"
        )
    markdown.extend(
        [
            "",
            "## 已知边界",
            "",
            "- 数据目标是接收功率，不是衰减；链路参考功率尚未统一，因此没有静默生成衰减标签。",
            "- 1000 hPa 在部分小时位于天线以下，LOS 无有效交点；对应缺测保留为 NaN。",
            "- 五层值是可用配置压力层的非加权算术平均，不是按层厚、质量或传播路径积分。",
            "- 实测时间沿用 Run_009 的北京时间解释；采集系统没有提供独立时区字段。",
            "",
        ]
    )
    validation_md = OUTPUT_DIR / "VALIDATION.md"
    validation_md.write_text("\n".join(markdown), encoding="utf-8")

    checksum_lines = [
        f"{sha256_file(report_path)}  {report_path.name}",
        f"{sha256_file(validation_md)}  {validation_md.name}",
        f"{sha256_file(BUILD_SCRIPT)}  {BUILD_SCRIPT.relative_to(PROJECT_ROOT).as_posix()}",
        f"{sha256_file(Path(__file__))}  {Path(__file__).relative_to(PROJECT_ROOT).as_posix()}",
    ]
    (OUTPUT_DIR / "validation_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": len(checks)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
