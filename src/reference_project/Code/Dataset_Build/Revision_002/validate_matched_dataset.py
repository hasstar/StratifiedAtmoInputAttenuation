from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TZ_BJT = "Asia/Shanghai"
START_NAIVE = pd.Timestamp("2025-09-30 00:00:00")
END_EXCLUSIVE_NAIVE = pd.Timestamp("2026-08-21 00:00:00")
START_MS = int(START_NAIVE.value // 1_000_000)
END_EXCLUSIVE_MS = int(END_EXCLUSIVE_NAIVE.value // 1_000_000)
HOUR_MS = 3_600_000
EXPECTED_100MS_PER_HOUR = 36_000

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Datasets" / "Revision_002"
RUN011_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Data\Further analysis of the data results"
    r"\思源楼高轨卫星信标信号实测数据_再分析结果\Runs\Run_011"
)
ERA5_DIR = Path(r"E:\0_Workflow_System\01_PhD\Data\era5_campaign_20250930_20260820\csv")
RAW_H5_ROOT = Path(r"E:\0_Workflow_System\01_PhD\Data\Raw data_cannot be modified")
REVISION_001_HELPER = Path(__file__).resolve().parents[1] / "Revision_001" / "build_matched_dataset.py"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("revision_001_validation_helper", REVISION_001_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Revision_001 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_path(metric: str) -> Path:
    paths = list((RUN011_DIR / "metrics" / metric).glob("*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected one Parquet for {metric}")
    return paths[0]


def parse_bjt(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(TZ_BJT)
    return parsed.dt.tz_convert(TZ_BJT)


def directory_snapshot(root: Path) -> dict[str, Any]:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: str(p).lower())
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ],
    }


def compare_numeric(left: pd.Series, right: pd.Series, rtol: float = 5e-10, atol: float = 1e-11) -> tuple[bool, float]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    max_abs = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return bool(same_nan and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)), max_abs


def verify_build_checksums() -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    checked = 0
    for line in (OUTPUT_DIR / "checksums.sha256").read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        target = OUTPUT_DIR / name.lstrip(" *")
        checked += 1
        if not target.is_file():
            failures.append({"path": target.name, "reason": "missing"})
        elif sha256_file(target) != expected.upper():
            failures.append({"path": target.name, "reason": "SHA-256 mismatch"})
    return {"name": "build_artifact_checksums", "passed": not failures, "files_checked": checked, "failures": failures}


def verify_source_identities(snapshot: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for identity in snapshot["direct_inputs"]:
        path = Path(identity["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        stat = path.stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            failures.append({"path": str(path), "reason": "size or mtime changed"})
            continue
        if sha256_file(path) != identity["sha256"]:
            failures.append({"path": str(path), "reason": "SHA-256 changed"})
    current_dirs = {
        "raw_h5_root": directory_snapshot(RAW_H5_ROOT),
        "run011": directory_snapshot(RUN011_DIR),
        "era5_campaign": directory_snapshot(ERA5_DIR.parent),
    }
    if current_dirs != snapshot["protected_directories_after"]:
        failures.append({"path": "protected directories", "reason": "directory snapshot changed after build"})
    return {
        "name": "source_files_and_protected_directories_unchanged",
        "passed": not failures,
        "direct_files_checked": len(snapshot["direct_inputs"]),
        "failures": failures,
    }


def verify_shapes_and_time(manifest: dict[str, Any], frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    failures: list[str] = []
    for name, frame in frames.items():
        expected = manifest["outputs"][name]
        if list(frame.shape) != [expected["rows"], expected["columns"]]:
            failures.append(f"{name}: shape {frame.shape} != {(expected['rows'], expected['columns'])}")
        if "hour_bjt" in frame:
            frame["hour_bjt"] = parse_bjt(frame["hour_bjt"])
            if frame["hour_bjt"].duplicated().any():
                failures.append(f"{name}: duplicate hour")
            if not frame["hour_bjt"].is_monotonic_increasing:
                failures.append(f"{name}: hour not sorted")
            if not (frame["hour_bjt"] >= pd.Timestamp(START_NAIVE, tz=TZ_BJT)).all():
                failures.append(f"{name}: time before campaign")
            if not (frame["hour_bjt"] < pd.Timestamp(END_EXCLUSIVE_NAIVE, tz=TZ_BJT)).all():
                failures.append(f"{name}: time after campaign")
    measurement = frames["measurement_hourly"]
    measurement_hours = pd.Index(measurement["hour_bjt"])
    for name in [
        "matched_hourly_pressure_levels_all_measurements",
        "matched_hourly_five_layers_all_measurements",
        "model_table_five_layers_all_measurements",
        "model_table_pressure_levels_all_measurements",
        "run011_hourly_diagnostics",
    ]:
        if not measurement_hours.equals(pd.Index(frames[name]["hour_bjt"])):
            failures.append(f"{name}: hours differ from measurement_hourly")
    high_hours = measurement.loc[measurement["measurement_high_coverage_95"].astype(bool), "hour_bjt"]
    for name in [
        "matched_hourly_five_layers_high_coverage_95",
        "model_table_five_layers_high_coverage_95",
        "model_table_pressure_levels_high_coverage_95",
    ]:
        if not pd.Index(high_hours).equals(pd.Index(frames[name]["hour_bjt"])):
            failures.append(f"{name}: high-coverage membership mismatch")
    return {"name": "output_shapes_and_temporal_keys", "passed": not failures, "failures": failures}


def verify_split_contract(measurement: pd.DataFrame) -> dict[str, Any]:
    h = parse_bjt(measurement["hour_bjt"])
    expected_blocked = np.select(
        [h < pd.Timestamp("2026-05-01", tz=TZ_BJT), h < pd.Timestamp("2026-07-01", tz=TZ_BJT)],
        ["train", "validation"],
        default="test",
    )
    expected_strict = np.select(
        [
            h < pd.Timestamp("2026-04-30", tz=TZ_BJT),
            h < pd.Timestamp("2026-05-01", tz=TZ_BJT),
            h < pd.Timestamp("2026-06-30", tz=TZ_BJT),
            h < pd.Timestamp("2026-07-01", tz=TZ_BJT),
        ],
        ["train", "embargo", "validation", "embargo"],
        default="test",
    )
    blocked_ok = np.array_equal(expected_blocked, measurement["split_blocked"].astype(str).to_numpy())
    strict_ok = np.array_equal(expected_strict, measurement["split_strict_24h"].astype(str).to_numpy())
    return {
        "name": "frozen_chronological_splits",
        "passed": bool(blocked_ok and strict_ok),
        "blocked_counts": measurement["split_blocked"].value_counts().to_dict(),
        "strict_counts": measurement["split_strict_24h"].value_counts().to_dict(),
        "failures": [] if blocked_ok and strict_ok else ["split assignments differ from frozen boundaries"],
    }


def verify_no_model_leakage(frames: dict[str, pd.DataFrame], feature_sets: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    meta = {"hour_bjt", "hour_utc", "split_blocked", "split_strict_24h", "received_power_w", "received_power_dbm"}
    five_features = set(feature_sets.loc[feature_sets["feature_set"].eq("S_plus_B_plus_L"), "column"])
    pressure_features = set(feature_sets.loc[feature_sets["feature_set"].eq("S_plus_B_plus_pressure_levels"), "column"])
    for name in ["model_table_five_layers_all_measurements", "model_table_five_layers_high_coverage_95"]:
        actual = set(frames[name].columns)
        expected = meta | five_features
        if actual != expected:
            failures.append(f"{name}: unexpected or missing columns")
    for name in ["model_table_pressure_levels_all_measurements", "model_table_pressure_levels_high_coverage_95"]:
        actual = set(frames[name].columns)
        expected = meta | pressure_features
        if actual != expected:
            failures.append(f"{name}: unexpected or missing columns")
    for name in [key for key in frames if key.startswith("model_table_")]:
        leaked = [
            column
            for column in frames[name].columns
            if column.startswith("run011__")
            or column.startswith("quality_")
            or column.startswith("source_k")
            or column.startswith("measurement_")
        ]
        if leaked:
            failures.append(f"{name}: leaked same-time or QC columns {leaked[:10]}")
    return {"name": "primary_model_tables_exclude_same_time_signal_diagnostics", "passed": not failures, "failures": failures}


def read_hour(metric: str, hour: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    naive = hour.tz_convert(TZ_BJT).tz_localize(None)
    end = naive + pd.Timedelta(hours=1)
    table = pq.read_table(
        metric_path(metric),
        columns=["time", "value"],
        filters=[("time", ">=", naive.to_pydatetime()), ("time", "<", end.to_pydatetime())],
    )
    t = table.column("time").to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
    value = table.column("value").to_numpy(zero_copy_only=False)
    return t, value


def reconstruct_primary_hour(hour: pd.Timestamp) -> dict[str, Any]:
    t, p = read_hour("peak_power_dbm", hour)
    tq, q = read_hour("quality_code", hour)
    tk, k = read_hour("source_bin_count", hour)
    if not np.array_equal(t, tq) or not np.array_equal(t, tk):
        raise ValueError(f"Primary metric time mismatch for {hour}")
    p = p.astype(np.float64)
    q = q.astype(np.uint16)
    k = k.astype(np.int32)
    order = np.argsort(t, kind="stable")
    t, p, q, k = t[order], p[order], q[order], k[order]
    starts = np.r_[0, np.flatnonzero(np.diff(t) != 0) + 1]
    ends = np.r_[starts[1:], len(t)]
    multiplicity = ends - starts
    w = np.power(10.0, (p - 30.0) / 10.0)
    group_w = np.add.reduceat(w, starts) / multiplicity
    group_p = 10.0 * np.log10(group_w * 1000.0)
    group_q = np.bitwise_or.reduceat(q, starts)
    kmin = np.minimum.reduceat(k, starts)
    kmax = np.maximum.reduceat(k, starts)
    group_k = np.where(kmin == kmax, kmin, -1)
    valid = np.isfinite(group_p) & ((group_q & np.uint16(15)) == 0)
    q0 = valid & (group_q == 0)
    power_w = np.power(10.0, (group_p[valid] - 30.0) / 10.0)
    q0_w = np.power(10.0, (group_p[q0] - 30.0) / 10.0)
    return {
        "measurement_records_raw": len(t),
        "measurement_unique_timestamps": len(starts),
        "measurement_duplicate_records": int((multiplicity - 1).sum()),
        "measurement_coverage_fraction": len(starts) / EXPECTED_100MS_PER_HOUR,
        "measurement_high_coverage_95": len(starts) >= math.ceil(0.95 * EXPECTED_100MS_PER_HOUR),
        "physical_valid_records": int(valid.sum()),
        "physical_invalid_records": int((~valid).sum()),
        "quality_zero_records": int(q0.sum()),
        "quality_nonzero_records": int((group_q != 0).sum()),
        "quality_bit16_records": int(((group_q & np.uint16(16)) != 0).sum()),
        "quality_bit32_records": int(((group_q & np.uint16(32)) != 0).sum()),
        "quality_bits16_and32_records": int(((group_q & np.uint16(48)) == 48).sum()),
        "source_k11_records": int((group_k == 11).sum()),
        "source_k101_records": int((group_k == 101).sum()),
        "source_other_or_mixed_records": int(((group_k != 11) & (group_k != 101)).sum()),
        "received_power_w": float(power_w.mean()),
        "received_power_dbm": float(10.0 * np.log10(power_w.mean() * 1000.0)),
        "received_power_quality0_w": float(q0_w.mean()) if q0_w.size else np.nan,
        "received_power_quality0_dbm": float(10.0 * np.log10(q0_w.mean() * 1000.0)) if q0_w.size else np.nan,
        "peak_power_dbm_record_mean": float(group_p[valid].mean()),
        "peak_power_dbm_record_min": float(group_p[valid].min()),
        "peak_power_dbm_record_max": float(group_p[valid].max()),
        "peak_power_dbm_record_std": float(group_p[valid].std(ddof=1)) if valid.sum() > 1 else np.nan,
    }


def verify_primary_samples(measurement: pd.DataFrame) -> dict[str, Any]:
    positions = sorted(set(np.linspace(0, len(measurement) - 1, 12, dtype=int).tolist()))
    numeric_failures: list[dict[str, Any]] = []
    checked_columns = [
        "measurement_records_raw",
        "measurement_unique_timestamps",
        "measurement_duplicate_records",
        "measurement_coverage_fraction",
        "measurement_high_coverage_95",
        "physical_valid_records",
        "physical_invalid_records",
        "quality_zero_records",
        "quality_nonzero_records",
        "quality_bit16_records",
        "quality_bit32_records",
        "quality_bits16_and32_records",
        "source_k11_records",
        "source_k101_records",
        "source_other_or_mixed_records",
        "received_power_w",
        "received_power_dbm",
        "received_power_quality0_w",
        "received_power_quality0_dbm",
        "peak_power_dbm_record_mean",
        "peak_power_dbm_record_min",
        "peak_power_dbm_record_max",
        "peak_power_dbm_record_std",
    ]
    for pos in positions:
        row = measurement.iloc[pos]
        expected = reconstruct_primary_hour(row["hour_bjt"])
        for column in checked_columns:
            actual = row[column]
            wanted = expected[column]
            if isinstance(wanted, (bool, np.bool_)):
                passed = bool(actual) == bool(wanted)
            elif isinstance(wanted, (int, np.integer)):
                passed = int(actual) == int(wanted)
            else:
                passed = bool(np.isclose(float(actual), float(wanted), rtol=1e-8, atol=2e-9, equal_nan=True))
            if not passed:
                numeric_failures.append({"hour": row["hour_bjt"].isoformat(), "column": column, "actual": actual, "expected": wanted})
    return {
        "name": "independent_primary_hour_reconstruction",
        "passed": not numeric_failures,
        "sample_hours": [measurement.iloc[pos]["hour_bjt"].isoformat() for pos in positions],
        "columns_checked_per_hour": len(checked_columns),
        "failures": numeric_failures[:50],
    }


def verify_diagnostic_samples(diagnostics: pd.DataFrame) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    metric_columns = sorted({match.group(1) for column in diagnostics.columns if (match := re.match(r"run011__(.+)__count$", column))})
    hour = diagnostics.iloc[len(diagnostics) // 2]["hour_bjt"]
    row = diagnostics.loc[diagnostics["hour_bjt"].eq(hour)].iloc[0]
    for metric in metric_columns:
        _, values = read_hour(metric, hour)
        values = values.astype(np.float64)
        values = values[np.isfinite(values)]
        expected = {
            "count": len(values),
            "record_mean": float(values.mean()) if len(values) else np.nan,
            "record_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "record_min": float(values.min()) if len(values) else np.nan,
            "record_max": float(values.max()) if len(values) else np.nan,
        }
        for suffix, wanted in expected.items():
            actual = row[f"run011__{metric}__{suffix}"]
            passed = int(actual) == int(wanted) if suffix == "count" else bool(np.isclose(float(actual), float(wanted), rtol=1e-7, atol=2e-9, equal_nan=True))
            if not passed:
                failures.append({"metric": metric, "suffix": suffix, "actual": actual, "expected": wanted})
    return {"name": "independent_run011_diagnostic_sample", "passed": not failures, "hour": hour.isoformat(), "metrics_checked": len(metric_columns), "failures": failures[:50]}


def verify_era5(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    helper = load_helper()
    metadata = json.loads((ERA5_DIR / "feature_metadata.json").read_text(encoding="utf-8"))
    surface, surface_features = helper.load_surface(ERA5_DIR / "era5_surface_bjtu_hourly_bjt.csv", metadata)
    pressure, layers, _, pressure_features, layer_features = helper.load_pressure_levels(ERA5_DIR, metadata)
    expected_pressure = surface.merge(pressure, on="hour_bjt", how="inner", validate="one_to_one")
    expected_layers = surface.merge(layers, on="hour_bjt", how="inner", validate="one_to_one")
    actual_pressure = frames["era5_hourly_pressure_levels"]
    actual_layers = frames["era5_hourly_five_layers"]
    failures: list[dict[str, Any]] = []
    max_abs = 0.0
    for label, actual, expected, columns in [
        ("surface_in_pressure", actual_pressure, expected_pressure, surface_features),
        ("pressure_levels", actual_pressure, expected_pressure, pressure_features),
        ("surface_in_layers", actual_layers, expected_layers, surface_features),
        ("five_layers", actual_layers, expected_layers, layer_features),
    ]:
        for column in columns:
            passed, difference = compare_numeric(actual[column], expected[column])
            max_abs = max(max_abs, difference)
            if not passed:
                failures.append({"group": label, "column": column, "max_abs_difference": difference})
    return {
        "name": "era5_surface_pressure_and_layer_values_against_sources",
        "passed": not failures,
        "surface_features_checked_twice": len(surface_features),
        "pressure_features_checked": len(pressure_features),
        "layer_features_checked": len(layer_features),
        "max_abs_difference": max_abs,
        "failures": failures[:50],
    }


def verify_monthly(measurement: pd.DataFrame) -> dict[str, Any]:
    actual = pd.read_csv(OUTPUT_DIR / "monthly_coverage.csv")
    work = measurement.copy()
    work["month"] = work["hour_bjt"].dt.strftime("%Y-%m")
    expected = (
        work.groupby("month", sort=True)
        .agg(
            measurement_hours=("hour_bjt", "size"),
            high_coverage_95_hours=("measurement_high_coverage_95", "sum"),
            raw_records=("measurement_records_raw", "sum"),
            unique_timestamps=("measurement_unique_timestamps", "sum"),
            quality_nonzero_records=("quality_nonzero_records", "sum"),
            quality_bit16_records=("quality_bit16_records", "sum"),
            quality_bit32_records=("quality_bit32_records", "sum"),
        )
        .reset_index()
    )
    columns = list(expected.columns)
    passed = actual[columns].astype(str).equals(expected[columns].astype(str))
    return {"name": "monthly_coverage_reconstruction", "passed": bool(passed), "months_checked": len(expected), "failures": [] if passed else ["monthly integer summaries differ"]}


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


def main() -> int:
    started = datetime.now(timezone.utc)
    manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    source_snapshot = json.loads((OUTPUT_DIR / "source_snapshot.json").read_text(encoding="utf-8"))
    dataset_names = list(manifest["outputs"])
    frames = {name: pd.read_parquet(OUTPUT_DIR / f"{name}.parquet") for name in dataset_names}
    for frame in frames.values():
        if "hour_bjt" in frame:
            frame["hour_bjt"] = parse_bjt(frame["hour_bjt"])
    feature_sets = pd.read_csv(OUTPUT_DIR / "feature_sets.csv")

    checks: list[dict[str, Any]] = []
    checks.append(verify_build_checksums())
    checks.append(verify_source_identities(source_snapshot))
    checks.append(verify_shapes_and_time(manifest, frames))
    checks.append(verify_split_contract(frames["measurement_hourly"]))
    checks.append(verify_no_model_leakage(frames, feature_sets))
    checks.append(verify_primary_samples(frames["measurement_hourly"]))
    checks.append(verify_diagnostic_samples(frames["run011_hourly_diagnostics"]))
    checks.append(verify_era5(frames))
    checks.append(verify_monthly(frames["measurement_hourly"]))

    passed = all(check["passed"] for check in checks)
    completed = datetime.now(timezone.utc)
    report = {
        "status": "VERIFIED" if passed else "FAILED",
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__)),
        "builder_sha256": sha256_file(Path(__file__).with_name("build_matched_dataset.py")),
        "checks_passed": sum(bool(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope_limitations": [
            "Validation proves reproducible construction and source agreement, not absolute RF calibration.",
            "The five-layer features remain unweighted arithmetic pressure-level summaries, not path-loss integrals.",
            "Chronological splits are frozen in the data set; downstream preprocessing and tuning must still be fitted only on training/validation data.",
        ],
    }
    report_path = OUTPUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")

    measurement = frames["measurement_hourly"]
    markdown = [
        "# Revision_002 独立验证",
        "",
        f"- 状态：{'VERIFIED' if passed else 'FAILED'}",
        f"- 通过：{report['checks_passed']} / {report['checks_total']} 项",
        f"- 验证时间：{completed.astimezone().isoformat()}",
        f"- 有实测小时：{len(measurement):,}",
        f"- 高覆盖率小时：{int(measurement['measurement_high_coverage_95'].sum()):,}",
        f"- 首末小时：{measurement['hour_bjt'].min().isoformat()} 至 {measurement['hour_bjt'].max().isoformat()}",
        "",
        "## 检查结果",
        "",
    ]
    for check in checks:
        markdown.append(f"- {'PASS' if check['passed'] else 'FAIL'}：`{check['name']}`")
    markdown.extend(
        [
            "",
            "## 验证边界",
            "",
            "- 本验证确认数据构建、时间匹配、W域功率平均、ERA5来源一致性、时间划分和主模型无同刻频谱泄漏。",
            "- 本验证不证明接收链路的绝对功率标定；参考功率冲突仍未解决，因此没有正式衰减标签。",
            "- 五层数据仍是配置压力层等权平均，不是厚度或传播路径积分。",
            "- 下游模型仍必须只在训练／验证区间完成缺失值处理、特征筛选和调参。",
            "",
        ]
    )
    validation_md = OUTPUT_DIR / "VALIDATION.md"
    validation_md.write_text("\n".join(markdown), encoding="utf-8")
    validation_checksum = OUTPUT_DIR / "validation_checksums.sha256"
    validation_checksum.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n{sha256_file(validation_md)}  {validation_md.name}\n",
        encoding="ascii",
    )
    print(json.dumps(json_ready({"status": report["status"], "checks_passed": report["checks_passed"], "checks_total": report["checks_total"], "report": str(report_path)}), ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
