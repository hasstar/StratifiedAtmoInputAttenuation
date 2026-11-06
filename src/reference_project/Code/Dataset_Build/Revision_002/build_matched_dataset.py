from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TZ_BJT = "Asia/Shanghai"
START_NAIVE = pd.Timestamp("2025-09-30 00:00:00")
END_EXCLUSIVE_NAIVE = pd.Timestamp("2026-08-21 00:00:00")
START_BJT = START_NAIVE.tz_localize(TZ_BJT)
END_EXCLUSIVE_BJT = END_EXCLUSIVE_NAIVE.tz_localize(TZ_BJT)
START_MS = int(START_NAIVE.value // 1_000_000)
END_EXCLUSIVE_MS = int(END_EXCLUSIVE_NAIVE.value // 1_000_000)
HOUR_MS = 3_600_000
EXPECTED_100MS_PER_HOUR = 36_000
N_CAMPAIGN_HOURS = int((END_EXCLUSIVE_MS - START_MS) // HOUR_MS)
REORDER_BUFFER_MS = 60_000

DEFAULT_ERA5_CSV_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Data\era5_campaign_20250930_20260820\csv"
)
DEFAULT_RAW_H5_ROOT = Path(
    r"E:\0_Workflow_System\01_PhD\Data\Raw data_cannot be modified"
)
DEFAULT_RUN011_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Data\Further analysis of the data results"
    r"\思源楼高轨卫星信标信号实测数据_再分析结果\Runs\Run_011"
)
DEFAULT_OUTPUT_DIR = Path(
    r"E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction"
    r"\Outputs\Datasets\Revision_002"
)
REVISION_001_HELPER = Path(__file__).resolve().parents[1] / "Revision_001" / "build_matched_dataset.py"

PRIMARY_METRICS = {"peak_power_dbm", "quality_code", "source_bin_count"}
POWER_DBM_DIAGNOSTICS = {
    "total_band_power_dbm",
    "center_bin_power_dbm",
    "background_bin_median_power_dbm",
}

MEASUREMENT_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "hour_bjt": ("identifier", "Matched hour in Asia/Shanghai", "ISO 8601"),
    "hour_utc": ("identifier", "Same matched hour in UTC", "ISO 8601"),
    "measurement_records_raw": ("measurement_qc", "Run_011 rows in the hour before exact-time deduplication", "count"),
    "measurement_unique_timestamps": ("measurement_qc", "Unique Run_011 timestamps after exact-time deduplication", "count"),
    "measurement_duplicate_records": ("measurement_qc", "Extra records sharing an exact timestamp", "count"),
    "measurement_coverage_fraction": ("measurement_qc", "Unique timestamps divided by 36,000 expected 100 ms slots", "1"),
    "measurement_high_coverage_95": ("measurement_qc", "True when at least 95% of 100 ms slots are represented", "boolean"),
    "measurement_first_time_bjt": ("measurement_qc", "First unique measurement timestamp in the hour", "ISO 8601"),
    "measurement_last_time_bjt": ("measurement_qc", "Last unique measurement timestamp in the hour", "ISO 8601"),
    "physical_valid_records": ("measurement_qc", "Unique finite peak-power records without quality bits 1, 2, 4, or 8", "count"),
    "physical_invalid_records": ("measurement_qc", "Unique records excluded by non-finite power or quality bits 1, 2, 4, or 8", "count"),
    "quality_zero_records": ("measurement_qc", "Unique records with quality_code equal to zero", "count"),
    "quality_nonzero_records": ("measurement_qc", "Unique records with any quality flag", "count"),
    "quality_bit16_records": ("measurement_qc", "Unique records whose maximum is outside the centre output bin", "count"),
    "quality_bit32_records": ("measurement_qc", "Unique records with peak-to-background ratio below 10 dB", "count"),
    "quality_bits16_and32_records": ("measurement_qc", "Unique records carrying both bit 16 and bit 32", "count"),
    "quality_zero_fraction": ("measurement_qc", "quality_zero_records divided by unique timestamps", "1"),
    "quality_bit16_fraction": ("measurement_qc", "quality_bit16_records divided by unique timestamps", "1"),
    "quality_bit32_fraction": ("measurement_qc", "quality_bit32_records divided by unique timestamps", "1"),
    "source_k11_records": ("measurement_qc", "Unique timestamps derived from native 11-bin spectra", "count"),
    "source_k101_records": ("measurement_qc", "Unique timestamps derived from native 101-bin spectra", "count"),
    "source_other_or_mixed_records": ("measurement_qc", "Unique timestamps with another or mixed source-bin count", "count"),
    "source_k11_fraction": ("measurement_qc", "source_k11_records divided by unique timestamps", "1"),
    "source_k101_fraction": ("measurement_qc", "source_k101_records divided by unique timestamps", "1"),
    "zero_steps_in_source_order": ("measurement_qc", "Zero timestamp steps in the original Run_011 row order", "count"),
    "negative_steps_in_source_order": ("measurement_qc", "Negative timestamp steps in the original Run_011 row order", "count"),
    "received_power_w": ("target", "Hourly mean peak-bin power after averaging unique physical records in W", "W"),
    "received_power_dbm": ("target", "Primary hourly received-power target converted once from received_power_w", "dBm"),
    "received_power_quality0_w": ("sensitivity_target", "Hourly W-domain mean using only quality_code=0 records", "W"),
    "received_power_quality0_dbm": ("sensitivity_target", "Quality-zero sensitivity target converted from W", "dBm"),
    "peak_power_dbm_record_mean": ("measurement_qc", "Arithmetic record mean of deduplicated peak_power_dbm; descriptive only", "dBm"),
    "peak_power_dbm_record_min": ("measurement_qc", "Minimum deduplicated peak_power_dbm in the hour", "dBm"),
    "peak_power_dbm_record_max": ("measurement_qc", "Maximum deduplicated peak_power_dbm in the hour", "dBm"),
    "peak_power_dbm_record_std": ("measurement_qc", "Sample standard deviation of deduplicated peak_power_dbm", "dB"),
    "split_blocked": ("split", "Frozen contiguous split: train through April, validation May-June, test July-August 20", "category"),
    "split_strict_24h": ("split", "Frozen contiguous split with a 24 h preceding embargo at both boundaries", "category"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Run_011 and ERA5 matched hourly data set")
    parser.add_argument("--era5-csv-dir", type=Path, default=DEFAULT_ERA5_CSV_DIR)
    parser.add_argument("--raw-h5-root", type=Path, default=DEFAULT_RAW_H5_ROOT)
    parser.add_argument("--run011-dir", type=Path, default=DEFAULT_RUN011_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    return parser.parse_args()


def load_revision_001_helper() -> Any:
    if not REVISION_001_HELPER.is_file():
        raise FileNotFoundError(f"Missing Revision_001 helper: {REVISION_001_HELPER}")
    spec = importlib.util.spec_from_file_location("revision_001_dataset_helper", REVISION_001_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Revision_001 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_identity(path: Path, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        result["sha256"] = sha256_file(path)
    return result


def output_identity(path: Path, output_root: Path) -> dict[str, Any]:
    ident = file_identity(path)
    ident["path"] = path.relative_to(output_root).as_posix()
    return ident


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


def metric_path(run_dir: Path, metric: str) -> Path:
    paths = sorted((run_dir / "metrics" / metric).glob("*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected one Parquet for {metric}, found {len(paths)}")
    return paths[0]


def parquet_contract(paths: Iterable[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    expected_rows: int | None = None
    for path in paths:
        pf = pq.ParquetFile(path)
        row_group_rows = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
        item = {
            "rows": pf.metadata.num_rows,
            "row_groups": pf.metadata.num_row_groups,
            "row_group_rows_sha256": hashlib.sha256(np.asarray(row_group_rows, dtype=np.int64).tobytes()).hexdigest().upper(),
            "schema": str(pf.schema_arrow),
        }
        if expected_rows is None:
            expected_rows = pf.metadata.num_rows
        elif pf.metadata.num_rows != expected_rows:
            raise ValueError(f"Metric row count mismatch in {path}")
        result[path.parent.name] = item
    return result


def _add_bincount(target: np.ndarray, index: np.ndarray, weights: np.ndarray | None = None) -> None:
    values = np.bincount(index, weights=weights, minlength=N_CAMPAIGN_HOURS)[:N_CAMPAIGN_HOURS]
    if np.issubdtype(target.dtype, np.integer):
        values = values.astype(target.dtype, copy=False)
    target += values


def aggregate_primary_metrics(
    peak_path: Path,
    quality_path: Path,
    source_bins_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    peak_pf = pq.ParquetFile(peak_path)
    quality_pf = pq.ParquetFile(quality_path)
    source_pf = pq.ParquetFile(source_bins_path)
    row_groups = [peak_pf.metadata.row_group(i).num_rows for i in range(peak_pf.metadata.num_row_groups)]
    for label, pf in {"quality": quality_pf, "source_bin_count": source_pf}.items():
        other = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
        if other != row_groups:
            raise ValueError(f"{label} row-group layout differs from peak_power_dbm")

    shape = (N_CAMPAIGN_HOURS,)
    raw_count = np.zeros(shape, dtype=np.int64)
    unique_count = np.zeros(shape, dtype=np.int64)
    duplicate_count = np.zeros(shape, dtype=np.int64)
    physical_count = np.zeros(shape, dtype=np.int64)
    physical_invalid = np.zeros(shape, dtype=np.int64)
    quality0_count = np.zeros(shape, dtype=np.int64)
    quality_nonzero = np.zeros(shape, dtype=np.int64)
    bit16_count = np.zeros(shape, dtype=np.int64)
    bit32_count = np.zeros(shape, dtype=np.int64)
    both_count = np.zeros(shape, dtype=np.int64)
    k11_count = np.zeros(shape, dtype=np.int64)
    k101_count = np.zeros(shape, dtype=np.int64)
    kother_count = np.zeros(shape, dtype=np.int64)
    zero_steps = np.zeros(shape, dtype=np.int64)
    negative_steps = np.zeros(shape, dtype=np.int64)
    power_sum_w = np.zeros(shape, dtype=np.float64)
    quality0_sum_w = np.zeros(shape, dtype=np.float64)
    dbm_sum = np.zeros(shape, dtype=np.float64)
    dbm_sum_sq = np.zeros(shape, dtype=np.float64)
    dbm_min = np.full(shape, np.inf, dtype=np.float64)
    dbm_max = np.full(shape, -np.inf, dtype=np.float64)
    first_ms = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    last_ms = np.full(shape, np.iinfo(np.int64).min, dtype=np.int64)

    before_rows = 0
    after_rows = 0
    total_rows = 0
    sequential_zero = 0
    sequential_negative = 0
    min_negative_step_ms: int | None = None
    prev_time: int | None = None
    max_seen = np.iinfo(np.int64).min
    finalized_through = np.iinfo(np.int64).min

    tail_t = np.empty(0, dtype=np.int64)
    tail_p = np.empty(0, dtype=np.float64)
    tail_q = np.empty(0, dtype=np.uint16)
    tail_k = np.empty(0, dtype=np.int32)

    def consume_sorted(t: np.ndarray, p: np.ndarray, q: np.ndarray, k: np.ndarray) -> None:
        if t.size == 0:
            return
        starts = np.r_[0, np.flatnonzero(np.diff(t) != 0) + 1]
        ends = np.r_[starts[1:], t.size]
        multiplicity = ends - starts
        w = np.power(10.0, (p - 30.0) / 10.0)
        mean_w = np.add.reduceat(w, starts) / multiplicity
        group_p = 10.0 * np.log10(mean_w * 1000.0)
        group_t = t[starts]
        group_q = np.bitwise_or.reduceat(q, starts)
        k_min = np.minimum.reduceat(k, starts)
        k_max = np.maximum.reduceat(k, starts)
        group_k = np.where(k_min == k_max, k_min, -1)
        duplicate_extra = multiplicity - 1

        in_range = (group_t >= START_MS) & (group_t < END_EXCLUSIVE_MS)
        if not in_range.any():
            return
        group_t = group_t[in_range]
        group_p = group_p[in_range]
        group_q = group_q[in_range]
        group_k = group_k[in_range]
        duplicate_extra = duplicate_extra[in_range]
        h = ((group_t - START_MS) // HOUR_MS).astype(np.int64)

        _add_bincount(unique_count, h)
        _add_bincount(duplicate_count, h, duplicate_extra.astype(np.float64))
        np.minimum.at(first_ms, h, group_t)
        np.maximum.at(last_ms, h, group_t)

        finite = np.isfinite(group_p)
        valid = finite & ((group_q & np.uint16(15)) == 0)
        _add_bincount(physical_count, h[valid])
        _add_bincount(physical_invalid, h[~valid])
        if valid.any():
            valid_w = np.power(10.0, (group_p[valid] - 30.0) / 10.0)
            _add_bincount(power_sum_w, h[valid], valid_w)
            _add_bincount(dbm_sum, h[valid], group_p[valid])
            _add_bincount(dbm_sum_sq, h[valid], group_p[valid] ** 2)
            np.minimum.at(dbm_min, h[valid], group_p[valid])
            np.maximum.at(dbm_max, h[valid], group_p[valid])

        q0 = valid & (group_q == 0)
        _add_bincount(quality0_count, h[q0])
        if q0.any():
            _add_bincount(
                quality0_sum_w,
                h[q0],
                np.power(10.0, (group_p[q0] - 30.0) / 10.0),
            )
        _add_bincount(quality_nonzero, h[group_q != 0])
        _add_bincount(bit16_count, h[(group_q & np.uint16(16)) != 0])
        _add_bincount(bit32_count, h[(group_q & np.uint16(32)) != 0])
        _add_bincount(both_count, h[(group_q & np.uint16(48)) == 48])
        _add_bincount(k11_count, h[group_k == 11])
        _add_bincount(k101_count, h[group_k == 101])
        _add_bincount(kother_count, h[(group_k != 11) & (group_k != 101)])

    peak_iter = peak_pf.iter_batches(batch_size=batch_size, columns=["time", "value"])
    quality_iter = quality_pf.iter_batches(batch_size=batch_size, columns=["time", "value"])
    source_iter = source_pf.iter_batches(batch_size=batch_size, columns=["time", "value"])
    batches = 0
    for peak_batch, quality_batch, source_batch in zip(peak_iter, quality_iter, source_iter, strict=True):
        batches += 1
        t = peak_batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
        p = peak_batch.column(1).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        tq = quality_batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
        q = quality_batch.column(1).to_numpy(zero_copy_only=False).astype(np.uint16, copy=False)
        tk = source_batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
        k = source_batch.column(1).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        if not np.array_equal(t, tq) or not np.array_equal(t, tk):
            raise ValueError(f"Primary metric timestamps differ in batch {batches}")
        if not (len(t) == len(p) == len(q) == len(k)):
            raise ValueError(f"Primary metric batch lengths differ in batch {batches}")

        total_rows += len(t)
        before_rows += int(np.count_nonzero(t < START_MS))
        after_rows += int(np.count_nonzero(t >= END_EXCLUSIVE_MS))
        in_range_raw = (t >= START_MS) & (t < END_EXCLUSIVE_MS)
        if in_range_raw.any():
            h_raw = ((t[in_range_raw] - START_MS) // HOUR_MS).astype(np.int64)
            _add_bincount(raw_count, h_raw)

        if prev_time is None:
            deltas = np.diff(t)
            current_times = t[1:]
        else:
            joined = np.concatenate(([prev_time], t))
            deltas = np.diff(joined)
            current_times = t
        if deltas.size:
            sequential_zero += int(np.count_nonzero(deltas == 0))
            sequential_negative += int(np.count_nonzero(deltas < 0))
            if (deltas < 0).any():
                local_min = int(deltas.min())
                min_negative_step_ms = local_min if min_negative_step_ms is None else min(min_negative_step_ms, local_min)
                if local_min < -REORDER_BUFFER_MS:
                    raise ValueError(
                        f"Observed timestamp reversal {local_min} ms exceeds {-REORDER_BUFFER_MS} ms buffer"
                    )
            step_in_range = (current_times >= START_MS) & (current_times < END_EXCLUSIVE_MS)
            if step_in_range.any():
                h_step = ((current_times[step_in_range] - START_MS) // HOUR_MS).astype(np.int64)
                d_step = deltas[step_in_range]
                _add_bincount(zero_steps, h_step[d_step == 0])
                _add_bincount(negative_steps, h_step[d_step < 0])
        prev_time = int(t[-1])

        combined_t = np.concatenate((tail_t, t))
        combined_p = np.concatenate((tail_p, p))
        combined_q = np.concatenate((tail_q, q))
        combined_k = np.concatenate((tail_k, k))
        order = np.argsort(combined_t, kind="stable")
        combined_t = combined_t[order]
        combined_p = combined_p[order]
        combined_q = combined_q[order]
        combined_k = combined_k[order]
        max_seen = max(max_seen, int(combined_t[-1]))
        threshold = max_seen - REORDER_BUFFER_MS
        cut = int(np.searchsorted(combined_t, threshold, side="right"))
        if cut:
            if combined_t[0] <= finalized_through:
                raise ValueError("A timestamp arrived after its reorder buffer had already been finalized")
            consume_sorted(combined_t[:cut], combined_p[:cut], combined_q[:cut], combined_k[:cut])
            finalized_through = int(combined_t[cut - 1])
        tail_t = combined_t[cut:]
        tail_p = combined_p[cut:]
        tail_q = combined_q[cut:]
        tail_k = combined_k[cut:]
        if batches % 25 == 0:
            print(f"primary metrics: {total_rows:,} rows", flush=True)

    if tail_t.size:
        consume_sorted(tail_t, tail_p, tail_q, tail_k)

    present = physical_count > 0
    hour_naive = pd.date_range(START_NAIVE, periods=N_CAMPAIGN_HOURS, freq="h")
    hours = hour_naive[present].tz_localize(TZ_BJT)
    idx = np.flatnonzero(present)
    n = physical_count[idx]
    mean_w = power_sum_w[idx] / n
    q0_w = np.divide(
        quality0_sum_w[idx],
        quality0_count[idx],
        out=np.full(idx.size, np.nan, dtype=np.float64),
        where=quality0_count[idx] > 0,
    )
    variance = np.divide(
        dbm_sum_sq[idx] - (dbm_sum[idx] ** 2) / n,
        n - 1,
        out=np.full(idx.size, np.nan, dtype=np.float64),
        where=n > 1,
    )
    variance = np.maximum(variance, 0.0, where=np.isfinite(variance), out=variance)
    denom = unique_count[idx].astype(np.float64)
    frame = pd.DataFrame(
        {
            "hour_bjt": hours,
            "hour_utc": hours.tz_convert("UTC"),
            "measurement_records_raw": raw_count[idx],
            "measurement_unique_timestamps": unique_count[idx],
            "measurement_duplicate_records": duplicate_count[idx],
            "measurement_coverage_fraction": unique_count[idx] / EXPECTED_100MS_PER_HOUR,
            "measurement_high_coverage_95": unique_count[idx] >= math.ceil(0.95 * EXPECTED_100MS_PER_HOUR),
            "measurement_first_time_bjt": pd.to_datetime(first_ms[idx], unit="ms").tz_localize(TZ_BJT),
            "measurement_last_time_bjt": pd.to_datetime(last_ms[idx], unit="ms").tz_localize(TZ_BJT),
            "physical_valid_records": physical_count[idx],
            "physical_invalid_records": physical_invalid[idx],
            "quality_zero_records": quality0_count[idx],
            "quality_nonzero_records": quality_nonzero[idx],
            "quality_bit16_records": bit16_count[idx],
            "quality_bit32_records": bit32_count[idx],
            "quality_bits16_and32_records": both_count[idx],
            "quality_zero_fraction": quality0_count[idx] / denom,
            "quality_bit16_fraction": bit16_count[idx] / denom,
            "quality_bit32_fraction": bit32_count[idx] / denom,
            "source_k11_records": k11_count[idx],
            "source_k101_records": k101_count[idx],
            "source_other_or_mixed_records": kother_count[idx],
            "source_k11_fraction": k11_count[idx] / denom,
            "source_k101_fraction": k101_count[idx] / denom,
            "zero_steps_in_source_order": zero_steps[idx],
            "negative_steps_in_source_order": negative_steps[idx],
            "received_power_w": mean_w,
            "received_power_dbm": 10.0 * np.log10(mean_w * 1000.0),
            "received_power_quality0_w": q0_w,
            "received_power_quality0_dbm": 10.0 * np.log10(q0_w * 1000.0),
            "peak_power_dbm_record_mean": dbm_sum[idx] / n,
            "peak_power_dbm_record_min": dbm_min[idx],
            "peak_power_dbm_record_max": dbm_max[idx],
            "peak_power_dbm_record_std": np.sqrt(variance),
        }
    )
    frame["split_blocked"] = np.select(
        [frame["hour_bjt"] < pd.Timestamp("2026-05-01", tz=TZ_BJT), frame["hour_bjt"] < pd.Timestamp("2026-07-01", tz=TZ_BJT)],
        ["train", "validation"],
        default="test",
    )
    frame["split_strict_24h"] = np.select(
        [
            frame["hour_bjt"] < pd.Timestamp("2026-04-30", tz=TZ_BJT),
            frame["hour_bjt"] < pd.Timestamp("2026-05-01", tz=TZ_BJT),
            frame["hour_bjt"] < pd.Timestamp("2026-06-30", tz=TZ_BJT),
            frame["hour_bjt"] < pd.Timestamp("2026-07-01", tz=TZ_BJT),
        ],
        ["train", "embargo", "validation", "embargo"],
        default="test",
    )
    summary = {
        "source_rows": int(total_rows),
        "rows_before_campaign": int(before_rows),
        "rows_at_or_after_campaign_end": int(after_rows),
        "rows_in_campaign_before_deduplication": int(raw_count.sum()),
        "unique_timestamps_in_campaign": int(unique_count.sum()),
        "exact_duplicate_records_in_campaign": int(duplicate_count.sum()),
        "physical_valid_records_in_campaign": int(physical_count.sum()),
        "physical_invalid_records_in_campaign": int(physical_invalid.sum()),
        "hours_with_measurements": int(len(frame)),
        "high_coverage_95_hours": int(frame["measurement_high_coverage_95"].sum()),
        "sequential_zero_steps_all_dates": int(sequential_zero),
        "sequential_negative_steps_all_dates": int(sequential_negative),
        "minimum_negative_step_ms": min_negative_step_ms,
        "reorder_buffer_ms": REORDER_BUFFER_MS,
        "first_measurement_bjt": frame["measurement_first_time_bjt"].min().isoformat(),
        "last_measurement_bjt": frame["measurement_last_time_bjt"].max().isoformat(),
    }
    return frame, summary


def aggregate_diagnostic_metric(path: Path, metric: str, batch_size: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = np.zeros(N_CAMPAIGN_HOURS, dtype=np.int64)
    total = np.zeros(N_CAMPAIGN_HOURS, dtype=np.float64)
    total_sq = np.zeros(N_CAMPAIGN_HOURS, dtype=np.float64)
    minimum = np.full(N_CAMPAIGN_HOURS, np.inf, dtype=np.float64)
    maximum = np.full(N_CAMPAIGN_HOURS, -np.inf, dtype=np.float64)
    linear_w = np.zeros(N_CAMPAIGN_HOURS, dtype=np.float64) if metric in POWER_DBM_DIAGNOSTICS else None
    rows_in_range = 0
    finite_in_range = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["time", "value"]):
        t = batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
        value = batch.column(1).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        in_range = (t >= START_MS) & (t < END_EXCLUSIVE_MS)
        rows_in_range += int(np.count_nonzero(in_range))
        valid = in_range & np.isfinite(value)
        finite_in_range += int(np.count_nonzero(valid))
        if not valid.any():
            continue
        h = ((t[valid] - START_MS) // HOUR_MS).astype(np.int64)
        v = value[valid]
        _add_bincount(count, h)
        _add_bincount(total, h, v)
        _add_bincount(total_sq, h, v * v)
        np.minimum.at(minimum, h, v)
        np.maximum.at(maximum, h, v)
        if linear_w is not None:
            _add_bincount(linear_w, h, np.power(10.0, (v - 30.0) / 10.0))
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    variance = np.divide(
        total_sq - np.divide(total * total, count, out=np.zeros_like(total), where=count > 0),
        count - 1,
        out=np.full_like(total, np.nan),
        where=count > 1,
    )
    variance = np.maximum(variance, 0.0, where=np.isfinite(variance), out=variance)
    columns: dict[str, np.ndarray] = {
        f"run011__{metric}__count": count,
        f"run011__{metric}__record_mean": mean,
        f"run011__{metric}__record_std": np.sqrt(variance),
        f"run011__{metric}__record_min": np.where(count > 0, minimum, np.nan),
        f"run011__{metric}__record_max": np.where(count > 0, maximum, np.nan),
    }
    if linear_w is not None:
        mean_w = np.divide(linear_w, count, out=np.full_like(linear_w, np.nan), where=count > 0)
        columns[f"run011__{metric}__linear_mean_dbm"] = 10.0 * np.log10(mean_w * 1000.0)
    return columns, {"rows_in_campaign": rows_in_range, "finite_rows_in_campaign": finite_in_range}


def build_diagnostics(run_dir: Path, measurement_hours: pd.Series, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_names = sorted(path.name for path in (run_dir / "metrics").iterdir() if path.is_dir())
    diagnostic_names = [name for name in metric_names if name not in PRIMARY_METRICS]
    full = pd.DataFrame(
        {"hour_bjt": pd.date_range(START_NAIVE, periods=N_CAMPAIGN_HOURS, freq="h").tz_localize(TZ_BJT)}
    )
    report: dict[str, Any] = {}
    for position, metric in enumerate(diagnostic_names, start=1):
        columns, item = aggregate_diagnostic_metric(metric_path(run_dir, metric), metric, batch_size)
        for name, values in columns.items():
            full[name] = values
        report[metric] = item
        print(f"diagnostic {position}/{len(diagnostic_names)}: {metric}", flush=True)
    wanted = pd.Index(measurement_hours)
    result = full[full["hour_bjt"].isin(wanted)].copy()
    result["hour_utc"] = result["hour_bjt"].dt.tz_convert("UTC")
    result = result[["hour_bjt", "hour_utc", *[c for c in result.columns if c not in {"hour_bjt", "hour_utc"}]]]
    if len(result) != len(wanted):
        raise ValueError("Diagnostic table does not cover every measurement hour")
    return result, report


def surface_subgroup(column: str) -> str:
    name = column.split("__", 1)[1]
    broad_patterns = (
        "total_column",
        "vertical_integral",
        "vertically_integrated",
        "cloud_cover",
        "cloud_base_height",
        "zero_degree_level",
        "trapping_layer",
        "duct_base_height",
        "k_index",
        "total_totals_index",
        "convective_available_potential_energy",
        "convective_inhibition",
        "toa_",
        "top_",
    )
    return "B_column_integrated_or_broad_atmosphere" if any(token in name for token in broad_patterns) else "S_local_or_surface"


def build_feature_sets(
    surface_features: list[str], pressure_features: list[str], layer_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def add(feature_set: str, columns: Iterable[str], representation: str) -> None:
        for order, column in enumerate(columns, start=1):
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_order": order,
                    "column": column,
                    "source_subgroup": surface_subgroup(column) if column.startswith("surface__") else representation,
                    "representation": representation,
                    "allowed_primary_predictor": True,
                }
            )

    local = [column for column in surface_features if surface_subgroup(column) == "S_local_or_surface"]
    add("S_local", local, "ERA5 single-level local/surface subset")
    add("S_plus_B", surface_features, "All selected ERA5 single-level/surface-product variables")
    add("S_plus_B_plus_L", [*surface_features, *layer_features], "All single-level variables plus five-layer means")
    add("S_plus_B_plus_pressure_levels", [*surface_features, *pressure_features], "All single-level variables plus 11 pressure levels")
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("feature_set", sort=False)
        .agg(feature_count=("column", "size"), unique_feature_count=("column", "nunique"))
        .reset_index()
    )
    summary["recommended_role"] = summary["feature_set"].map(
        {
            "S_local": "ablation baseline",
            "S_plus_B": "single-level/broad-atmosphere baseline",
            "S_plus_B_plus_L": "primary five-layer model",
            "S_plus_B_plus_pressure_levels": "pressure-level sensitivity analysis",
        }
    )
    return frame, summary


def add_missingness_flags(
    helper: Any,
    frame: pd.DataFrame,
    surface_features: list[str],
    atmosphere_features: list[str],
    label: str,
) -> pd.DataFrame:
    return helper.add_missingness_flags(frame, surface_features, atmosphere_features, label)


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.DatetimeTZDtype):
            result[column] = result[column].map(lambda value: value.isoformat() if pd.notna(value) else "")
    return result


def write_dataset(frame: pd.DataFrame, stem: Path, output_root: Path) -> dict[str, Any]:
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    csv_ready(frame).to_csv(csv_path, index=False, encoding="utf-8-sig")
    parquet_readback = pd.read_parquet(parquet_path)
    csv_readback = pd.read_csv(csv_path)
    if parquet_readback.shape != frame.shape or csv_readback.shape != frame.shape:
        raise ValueError(f"Read-back shape failed for {stem.name}")
    if list(parquet_readback.columns) != list(frame.columns) or list(csv_readback.columns) != list(frame.columns):
        raise ValueError(f"Read-back schema failed for {stem.name}")
    result = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "csv": output_identity(csv_path, output_root),
        "parquet": output_identity(parquet_path, output_root),
    }
    if "hour_bjt" in frame.columns and len(frame):
        result["first_hour_bjt"] = frame["hour_bjt"].min().isoformat()
        result["last_hour_bjt"] = frame["hour_bjt"].max().isoformat()
    return result


def describe_column(column: str, helper: Any, metadata: dict[str, Any], predictor_columns: set[str]) -> tuple[str, str, str, bool]:
    if column in MEASUREMENT_DESCRIPTIONS:
        group, description, unit = MEASUREMENT_DESCRIPTIONS[column]
        return group, description, unit, False
    if column.startswith("run011__"):
        return "run011_diagnostic", "Hourly descriptive summary; excluded from the primary atmosphere-to-power model", "see Run_011 metric dictionary", False
    group, description, unit = helper.description_and_unit(column, metadata)
    allowed = column in predictor_columns
    if allowed:
        group = "primary_predictor"
    return group, description, unit, allowed


def build_dictionary(
    datasets: dict[str, pd.DataFrame],
    helper: Any,
    metadata: dict[str, Any],
    predictor_columns: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in datasets.items():
        for position, column in enumerate(frame.columns, start=1):
            group, description, units, allowed = describe_column(column, helper, metadata, predictor_columns)
            rows.append(
                {
                    "dataset": dataset,
                    "column_position": position,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "group": group,
                    "units": units,
                    "description": description,
                    "allowed_as_primary_model_input": allowed,
                }
            )
    return pd.DataFrame(rows)


def monthly_coverage(measurement: pd.DataFrame) -> pd.DataFrame:
    work = measurement.copy()
    work["month"] = work["hour_bjt"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for month, frame in work.groupby("month", sort=True):
        mean_w = float(frame["received_power_w"].mean())
        rows.append(
            {
                "month": month,
                "measurement_hours": len(frame),
                "high_coverage_95_hours": int(frame["measurement_high_coverage_95"].sum()),
                "raw_records": int(frame["measurement_records_raw"].sum()),
                "unique_timestamps": int(frame["measurement_unique_timestamps"].sum()),
                "quality_nonzero_records": int(frame["quality_nonzero_records"].sum()),
                "quality_bit16_records": int(frame["quality_bit16_records"].sum()),
                "quality_bit32_records": int(frame["quality_bit32_records"].sum()),
                "monthly_mean_received_power_dbm_from_hourly_w": 10.0 * math.log10(mean_w * 1000.0),
                "train_hours_blocked": int(frame["split_blocked"].eq("train").sum()),
                "validation_hours_blocked": int(frame["split_blocked"].eq("validation").sum()),
                "test_hours_blocked": int(frame["split_blocked"].eq("test").sum()),
            }
        )
    return pd.DataFrame(rows)


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
    args = parse_args()
    era5_dir = args.era5_csv_dir.resolve()
    raw_root = args.raw_h5_root.resolve()
    run_dir = args.run011_dir.resolve()
    output_dir = args.output_dir.resolve()
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    helper = load_revision_001_helper()
    metadata_path = era5_dir / "feature_metadata.json"
    surface_path = era5_dir / "era5_surface_bjtu_hourly_bjt.csv"
    era5_quality_path = era5_dir / "quality_report.csv"
    pressure_paths = sorted(era5_dir.glob("era5_pressure_*hPa_los_hourly_bjt.csv"))
    run_support = [
        run_dir / "run_manifest.json",
        run_dir / "validation_report.json",
        run_dir / "independent_verification.json",
        run_dir / "duplicate_verification.json",
        run_dir / "file_summary.csv",
        run_dir / "input_catalog.csv",
        run_dir / "metric_dictionary.csv",
        run_dir / "metric_summary.csv",
        run_dir / "QA.md",
    ]
    metric_paths = sorted(run_dir.glob("metrics/*/*.parquet"))
    required = [metadata_path, surface_path, era5_quality_path, *pressure_paths, *run_support, *metric_paths, REVISION_001_HELPER]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    if len(metric_paths) != 24:
        raise ValueError(f"Expected 24 metric Parquets, found {len(metric_paths)}")

    print("hashing direct inputs", flush=True)
    input_identities = [file_identity(path) for path in required]
    input_stats_before = {item["path"]: (item["bytes"], item["mtime_ns"]) for item in input_identities}
    directory_before = {
        "raw_h5_root": directory_snapshot(raw_root),
        "run011": directory_snapshot(run_dir),
        "era5_campaign": directory_snapshot(era5_dir.parent),
    }
    contract = parquet_contract(metric_paths)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "VERIFIED":
        raise ValueError("Run_011 manifest is not VERIFIED")

    started = datetime.now(timezone.utc)
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        print("aggregating primary Run_011 metrics", flush=True)
        measurement, measurement_summary = aggregate_primary_metrics(
            metric_path(run_dir, "peak_power_dbm"),
            metric_path(run_dir, "quality_code"),
            metric_path(run_dir, "source_bin_count"),
            args.batch_size,
        )
        if measurement["hour_bjt"].duplicated().any() or not measurement["hour_bjt"].is_monotonic_increasing:
            raise ValueError("Measurement hours are not unique and sorted")
        if not np.isfinite(measurement["received_power_dbm"]).all():
            raise ValueError("Primary received-power target contains non-finite values")

        print("loading ERA5", flush=True)
        surface, surface_features = helper.load_surface(surface_path, metadata)
        pressure_wide, layer_wide, pressure_report, pressure_features, layer_features = helper.load_pressure_levels(era5_dir, metadata)
        era5_pressure = surface.merge(pressure_wide, on="hour_bjt", how="inner", validate="one_to_one")
        era5_layers = surface.merge(layer_wide, on="hour_bjt", how="inner", validate="one_to_one")
        if len(era5_pressure) != N_CAMPAIGN_HOURS or len(era5_layers) != N_CAMPAIGN_HOURS:
            raise ValueError("ERA5 hour grid is not the expected 7,800 rows")

        matched_pressure = measurement.merge(era5_pressure, on="hour_bjt", how="inner", validate="one_to_one")
        matched_layers = measurement.merge(era5_layers, on="hour_bjt", how="inner", validate="one_to_one")
        if len(matched_pressure) != len(measurement) or len(matched_layers) != len(measurement):
            raise ValueError("One or more measurement hours lack an exact ERA5 match")
        matched_pressure = add_missingness_flags(helper, matched_pressure, surface_features, pressure_features, "pressure")
        matched_layers = add_missingness_flags(helper, matched_layers, surface_features, layer_features, "layer")
        high_layers = matched_layers[matched_layers["measurement_high_coverage_95"]].copy()

        print("aggregating Run_011 diagnostic metrics", flush=True)
        diagnostics, diagnostics_report = build_diagnostics(run_dir, measurement["hour_bjt"], args.batch_size)

        feature_sets, feature_set_summary = build_feature_sets(surface_features, pressure_features, layer_features)
        predictor_columns = set(surface_features) | set(pressure_features) | set(layer_features)
        model_meta = ["hour_bjt", "hour_utc", "split_blocked", "split_strict_24h", "received_power_w", "received_power_dbm"]
        model_layers = matched_layers[[*model_meta, *surface_features, *layer_features]].copy()
        model_pressure = matched_pressure[[*model_meta, *surface_features, *pressure_features]].copy()
        model_layers_high = model_layers[model_layers["hour_bjt"].isin(high_layers["hour_bjt"])].copy()
        model_pressure_high = model_pressure[model_pressure["hour_bjt"].isin(high_layers["hour_bjt"])].copy()
        time_splits = measurement[
            [
                "hour_bjt",
                "hour_utc",
                "split_blocked",
                "split_strict_24h",
                "measurement_high_coverage_95",
                "measurement_coverage_fraction",
            ]
        ].copy()
        monthly = monthly_coverage(measurement)
        split_summary = (
            time_splits.groupby(["split_blocked", "split_strict_24h"], dropna=False, sort=False)
            .agg(hours=("hour_bjt", "size"), high_coverage_95_hours=("measurement_high_coverage_95", "sum"), first_hour=("hour_bjt", "min"), last_hour=("hour_bjt", "max"))
            .reset_index()
        )

        datasets = {
            "measurement_hourly": measurement,
            "era5_hourly_pressure_levels": era5_pressure,
            "era5_hourly_five_layers": era5_layers,
            "matched_hourly_pressure_levels_all_measurements": matched_pressure,
            "matched_hourly_five_layers_all_measurements": matched_layers,
            "matched_hourly_five_layers_high_coverage_95": high_layers,
            "model_table_five_layers_all_measurements": model_layers,
            "model_table_five_layers_high_coverage_95": model_layers_high,
            "model_table_pressure_levels_all_measurements": model_pressure,
            "model_table_pressure_levels_high_coverage_95": model_pressure_high,
            "run011_hourly_diagnostics": diagnostics,
        }
        output_records: dict[str, Any] = {}
        for name, frame in datasets.items():
            print(f"writing {name}: {frame.shape}", flush=True)
            output_records[name] = write_dataset(frame, temp_dir / name, temp_dir)

        auxiliary_frames = {
            "time_splits": time_splits,
            "monthly_coverage": monthly,
            "split_summary": split_summary,
            "feature_sets": feature_sets,
            "feature_set_summary": feature_set_summary,
        }
        auxiliary_outputs: dict[str, Any] = {}
        for name, frame in auxiliary_frames.items():
            path = temp_dir / f"{name}.csv"
            csv_ready(frame).to_csv(path, index=False, encoding="utf-8-sig")
            auxiliary_outputs[name] = output_identity(path, temp_dir)

        dictionary = build_dictionary(datasets, helper, metadata, predictor_columns)
        dictionary_path = temp_dir / "data_dictionary.csv"
        dictionary.to_csv(dictionary_path, index=False, encoding="utf-8-sig")
        auxiliary_outputs["data_dictionary"] = output_identity(dictionary_path, temp_dir)

        directory_after = {
            "raw_h5_root": directory_snapshot(raw_root),
            "run011": directory_snapshot(run_dir),
            "era5_campaign": directory_snapshot(era5_dir.parent),
        }
        if directory_before != directory_after:
            raise RuntimeError("A protected source directory changed during the build")
        for path in required:
            stat = path.stat()
            if input_stats_before[str(path.resolve())] != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Direct input changed during the build: {path}")
        source_snapshot = {
            "direct_inputs": input_identities,
            "protected_directories_before": directory_before,
            "protected_directories_after": directory_after,
            "unchanged": True,
        }
        source_snapshot_path = temp_dir / "source_snapshot.json"
        source_snapshot_path.write_text(json.dumps(json_ready(source_snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        auxiliary_outputs["source_snapshot"] = output_identity(source_snapshot_path, temp_dir)

        completed = datetime.now(timezone.utc)
        manifest = {
            "status": "EXECUTED_PENDING_INDEPENDENT_VALIDATION",
            "revision": "Revision_002",
            "started_utc": started.isoformat(),
            "completed_utc": completed.isoformat(),
            "elapsed_seconds": (completed - started).total_seconds(),
            "requested_time_range": {
                "timezone": TZ_BJT,
                "start_inclusive": START_BJT.isoformat(),
                "end_exclusive": END_EXCLUSIVE_BJT.isoformat(),
            },
            "source": {
                "run011": str(run_dir),
                "run011_status": run_manifest.get("status"),
                "run011_rows_per_metric": run_manifest.get("output", {}).get("rows_per_metric"),
                "era5_csv": str(era5_dir),
                "raw_h5_root_protection_only": str(raw_root),
            },
            "code": {
                "builder": str(Path(__file__).resolve()),
                "builder_sha256": sha256_file(Path(__file__)),
                "revision_001_helper": str(REVISION_001_HELPER),
                "revision_001_helper_sha256": sha256_file(REVISION_001_HELPER),
            },
            "measurement_contract": {
                "primary_target": "received_power_dbm",
                "hourly_mean": "deduplicate exact timestamps; convert peak_power_dbm to W; average equal-duration 100 ms records in W; convert once to dBm",
                "physical_validity": "exclude only non-finite peak power or quality bits 1, 2, 4, 8; retain bits 16 and 32",
                "quality_sensitivity": "received_power_quality0_dbm is reported separately; nonzero quality rows are not silently deleted",
                "coverage": "unique 100 ms timestamps divided by 36,000; >=0.95 is the high-coverage sensitivity subset",
                "time_anomalies": f"source order is audited; a {REORDER_BUFFER_MS} ms stable-sort buffer precedes exact-time deduplication",
                "out_of_window": "records at or after 2026-08-21 00:00 BJT are excluded",
                "attenuation": "not generated because the absolute reference-power conflict is unresolved",
            },
            "era5_contract": {
                "join": "exact one-to-one match on timezone-aware Asia/Shanghai hourly key",
                "missing_measurement_hours": "absent from supervised data; never target-interpolated",
                "missing_era5": "retained as NaN; no imputation in data construction",
                "below_antenna_policy": metadata["below_antenna_policy"],
                "five_layers": pressure_report,
            },
            "leakage_contract": {
                "primary_model_tables": "contain only identifiers, frozen split labels, received-power targets, and ERA5 predictors",
                "same_time_run011_spectral_metrics": "kept only in run011_hourly_diagnostics and excluded from model tables",
                "source_bin_count_and_quality_code": "measurement QA/nuisance metadata only; excluded from model tables",
            },
            "time_split_contract": {
                "blocked": {
                    "train": "hour < 2026-05-01 00:00 BJT",
                    "validation": "2026-05-01 00:00 <= hour < 2026-07-01 00:00 BJT",
                    "test": "hour >= 2026-07-01 00:00 BJT",
                },
                "strict_24h": {
                    "train": "hour < 2026-04-30 00:00 BJT",
                    "embargo_1": "2026-04-30 00:00 <= hour < 2026-05-01 00:00 BJT",
                    "validation": "2026-05-01 00:00 <= hour < 2026-06-30 00:00 BJT",
                    "embargo_2": "2026-06-30 00:00 <= hour < 2026-07-01 00:00 BJT",
                    "test": "hour >= 2026-07-01 00:00 BJT",
                },
                "rule": "feature selection, preprocessing, imputation and tuning must not use final test rows",
            },
            "feature_set_contract": {
                "S_local": "explicit local/surface subset of ERA5 single-level variables",
                "S_plus_B": "all selected ERA5 single-level/surface-product variables; not described as purely ground observations",
                "S_plus_B_plus_L": "primary paper logic: all single-level variables plus five unweighted pressure-layer means",
                "S_plus_B_plus_pressure_levels": "sensitivity table retaining all 11 pressure levels",
            },
            "measurement_summary": measurement_summary,
            "diagnostic_summary": diagnostics_report,
            "feature_counts": {
                "surface_selected": len(surface_features),
                "surface_local_subset": int((feature_sets["feature_set"] == "S_local").sum()),
                "pressure_level_selected": len(pressure_features),
                "five_layer_selected": len(layer_features),
            },
            "parquet_contract": contract,
            "outputs": output_records,
            "auxiliary_outputs": auxiliary_outputs,
            "builder_checks": {
                "protected_source_directories_unchanged": True,
                "primary_metric_timestamps_fully_equal": True,
                "measurement_hours_unique_and_sorted": True,
                "exact_era5_match_for_every_measurement_hour": True,
                "power_averaged_in_linear_w": True,
                "no_target_interpolation": True,
                "same_time_spectral_metrics_excluded_from_model_tables": True,
                "csv_parquet_readback_shape_and_schema": True,
                "independent_validation": "pending validate_matched_dataset.py",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "16.1.0",
            },
        }
        manifest_path = temp_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

        qa_lines = [
            "# Revision_002 数据集质量报告",
            "",
            "- 构建状态：EXECUTED；独立验证见后续 `VALIDATION.md`。",
            f"- 规定时间范围：{START_BJT.isoformat()} 至 {END_EXCLUSIVE_BJT.isoformat()}（结束时刻不含）。",
            f"- Run_011 源行：{measurement_summary['source_rows']:,}。",
            f"- 指定窗口内原始行：{measurement_summary['rows_in_campaign_before_deduplication']:,}。",
            f"- 排除 2026-08-21 及之后记录：{measurement_summary['rows_at_or_after_campaign_end']:,}。",
            f"- 精确时间去重后记录：{measurement_summary['unique_timestamps_in_campaign']:,}。",
            f"- 有实测且与 ERA5 精确匹配的小时：{len(measurement):,}。",
            f"- 覆盖率至少 95% 的小时：{int(measurement['measurement_high_coverage_95'].sum()):,}。",
            f"- 首个实测：{measurement['measurement_first_time_bjt'].min().isoformat()}。",
            f"- 最后实测：{measurement['measurement_last_time_bjt'].max().isoformat()}。",
            "",
            "## 大修口径",
            "",
            "- 保留论文原有 ERA5 同期大气参数到接收功率估计主线。",
            "- 小时功率在 W 域平均；不直接平均 dBm。",
            "- `quality_code != 0` 不整体删除；bit 16/32 保留，并提供质量零敏感性目标。",
            "- 主模型表不包含同时刻频谱形态、质量码或源频谱点数，避免目标泄漏和设备配置捷径。",
            "- 五层仍按原论文边界进行配置压力层等权平均；同时保留 11 个压力层供敏感性分析。",
            "- 五层统计不是厚度、质量或传播损耗积分。",
            "- 正式衰减标签未生成，因为参考功率校准口径尚未统一。",
            "- 同期 ERA5 输入对应 estimation/reconstruction；本数据集本身不证明未来提前预测。",
            "",
            "## 月份覆盖",
            "",
            monthly.to_markdown(index=False),
            "",
            "## 时间划分",
            "",
            split_summary.to_markdown(index=False),
            "",
            "## 限制",
            "",
            "- Run_011 的 10 ms 文件在 100 ms 窗中的原始样本数为 1–10，但该逐窗计数未作为独立指标保存；小时覆盖率按输出 100 ms 时间戳计算。",
            "- `run011_hourly_diagnostics` 是描述性记录汇总，不属于主模型输入；dB类指标的 record_mean 不等同于物理线性平均。",
            "- 同站点跨月份验证不等于跨地点或跨气候泛化。",
        ]
        qa_path = temp_dir / "QA.md"
        qa_path.write_text("\n".join(qa_lines) + "\n", encoding="utf-8")
        auxiliary_outputs["qa"] = output_identity(qa_path, temp_dir)

        manifest["auxiliary_outputs"] = auxiliary_outputs
        manifest_path.write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        checksum_lines = []
        for path in sorted(temp_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        (temp_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
        temp_dir.replace(output_dir)
        print(json.dumps(json_ready({"output": str(output_dir), "measurement_summary": measurement_summary, "feature_counts": manifest["feature_counts"]}), ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception:
        if temp_dir.exists():
            (temp_dir / "FAILED.txt").write_text(
                f"Build failed at {datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
