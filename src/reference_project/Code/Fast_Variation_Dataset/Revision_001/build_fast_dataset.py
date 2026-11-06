from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TZ_BJT = "Asia/Shanghai"
START_NAIVE = pd.Timestamp("2025-09-30 00:00:00")
END_NAIVE = pd.Timestamp("2026-08-21 00:00:00")
START_MS = int(np.datetime64("2025-09-30T00:00:00", "ms").astype(np.int64))
END_MS = int(np.datetime64("2026-08-21T00:00:00", "ms").astype(np.int64))
HOUR_MS = 3_600_000
SECOND_MS = 1_000
EXPECTED_100MS_PER_SECOND = 10
REORDER_BUFFER_MS = 60_000
SEGMENT_GAP_MS = 5_000
WINDOW_MINUTES = (5, 10, 15, 30, 60)

DEFAULT_RUN011 = Path(
    r"E:\0_Workflow_System\01_PhD\Data\Further analysis of the data results"
    r"\思源楼高轨卫星信标信号实测数据_再分析结果\Runs\Run_011"
)
DEFAULT_HOURLY = Path(__file__).resolve().parents[3] / "Outputs" / "Datasets" / "Revision_002" / "measurement_hourly.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "Outputs" / "Datasets" / "Fast_Variation_Revision_001"

BLOCKED_CODES = {"train": 0, "validation": 1, "test": 2}
STRICT_CODES = {"train": 0, "validation": 1, "embargo": 2, "test": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Run_011 100 ms and 1 s fast-varying dataset")
    parser.add_argument("--run011", type=Path, default=DEFAULT_RUN011)
    parser.add_argument("--hourly", type=Path, default=DEFAULT_HOURLY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def metric_path(run011: Path, name: str) -> Path:
    paths = sorted((run011 / "metrics" / name).glob("*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one Parquet for {name}; found {len(paths)}")
    return paths[0]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def parquet_layout(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    groups = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "row_group_rows_sha256": hashlib.sha256(np.asarray(groups, dtype=np.int64).tobytes()).hexdigest().upper(),
        "schema": str(pf.schema_arrow),
    }


FAST100_SCHEMA = pa.schema(
    [
        pa.field("time_bjt", pa.timestamp("ms"), nullable=False),
        pa.field("hour_bjt", pa.timestamp("ms"), nullable=False),
        pa.field("peak_power_dbm", pa.float32(), nullable=True),
        pa.field("peak_power_w", pa.float64(), nullable=True),
        pa.field("slow_hourly_power_dbm", pa.float32(), nullable=True),
        pa.field("slow_hourly_power_w", pa.float64(), nullable=True),
        pa.field("fast_residual_db", pa.float32(), nullable=True),
        pa.field("normalized_power_ratio", pa.float32(), nullable=True),
        pa.field("normalized_envelope", pa.float32(), nullable=True),
        pa.field("quality_code", pa.uint16(), nullable=False),
        pa.field("source_bin_count", pa.int32(), nullable=False),
        pa.field("physical_valid", pa.bool_(), nullable=False),
        pa.field("source_record_multiplicity", pa.uint16(), nullable=False),
        pa.field("gap_from_previous_ms", pa.int64(), nullable=False),
        pa.field("continuous_segment_id", pa.int32(), nullable=False),
        pa.field("split_blocked_code", pa.int8(), nullable=False),
        pa.field("split_strict_24h_code", pa.int8(), nullable=False),
    ],
    metadata={
        b"time_zone": b"Asia/Shanghai wall-clock time; stored as timezone-naive timestamp for compact interoperability",
        b"role": b"traceable 100 ms fast-variation base layer",
    },
)

FAST1_SCHEMA = pa.schema(
    [
        pa.field("second_bjt", pa.timestamp("ms"), nullable=False),
        pa.field("hour_bjt", pa.timestamp("ms"), nullable=False),
        pa.field("records_in_second", pa.uint16(), nullable=False),
        pa.field("physical_valid_records", pa.uint16(), nullable=False),
        pa.field("coverage_fraction", pa.float32(), nullable=False),
        pa.field("high_coverage_90", pa.bool_(), nullable=False),
        pa.field("overfilled_nominal_second", pa.bool_(), nullable=False),
        pa.field("received_power_w", pa.float64(), nullable=True),
        pa.field("received_power_dbm", pa.float32(), nullable=True),
        pa.field("slow_hourly_power_w", pa.float64(), nullable=True),
        pa.field("slow_hourly_power_dbm", pa.float32(), nullable=True),
        pa.field("fast_residual_db", pa.float32(), nullable=True),
        pa.field("normalized_power_ratio", pa.float32(), nullable=True),
        pa.field("normalized_envelope", pa.float32(), nullable=True),
        pa.field("quality_zero_records", pa.uint16(), nullable=False),
        pa.field("quality_nonzero_records", pa.uint16(), nullable=False),
        pa.field("quality_bit16_records", pa.uint16(), nullable=False),
        pa.field("quality_bit32_records", pa.uint16(), nullable=False),
        pa.field("quality_zero_fraction", pa.float32(), nullable=False),
        pa.field("source_k11_records", pa.uint16(), nullable=False),
        pa.field("source_k101_records", pa.uint16(), nullable=False),
        pa.field("source_other_or_mixed_records", pa.uint16(), nullable=False),
        pa.field("continuous_segment_id", pa.int32(), nullable=False),
        pa.field("split_blocked_code", pa.int8(), nullable=False),
        pa.field("split_strict_24h_code", pa.int8(), nullable=False),
    ],
    metadata={
        b"time_zone": b"Asia/Shanghai wall-clock time; stored as timezone-naive timestamp for compact interoperability",
        b"role": b"primary one-second fast-variation analysis layer",
        b"power_aggregation": b"arithmetic mean in linear watts over physical-valid deduplicated records",
    },
)


def _arrow_array(values: np.ndarray, field_type: pa.DataType) -> pa.Array:
    if pa.types.is_timestamp(field_type):
        return pa.array(np.asarray(values, dtype=np.int64).astype("datetime64[ms]"), type=field_type)
    return pa.array(values, type=field_type, from_pandas=True)


class MonthlyParquetWriter:
    def __init__(self, root: Path, prefix: str, schema: pa.Schema) -> None:
        self.root = root
        self.prefix = prefix
        self.schema = schema
        self.root.mkdir(parents=True, exist_ok=False)
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.rows: dict[str, int] = {}

    @staticmethod
    def month_label(ms: int) -> str:
        return str(np.datetime64(int(ms), "ms"))[:7]

    def write(self, data: dict[str, np.ndarray]) -> None:
        n = len(next(iter(data.values())))
        if n == 0:
            return
        times = np.asarray(data[self.schema.names[0]], dtype=np.int64)
        month_values = times.astype("datetime64[ms]").astype("datetime64[M]")
        starts = np.r_[0, np.flatnonzero(month_values[1:] != month_values[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        for start, end in zip(starts, ends, strict=True):
            month = str(month_values[start])
            if month not in self.writers:
                path = self.root / f"{self.prefix}_{month}.parquet"
                self.writers[month] = pq.ParquetWriter(
                    path,
                    self.schema,
                    compression="zstd",
                    compression_level=5,
                    use_dictionary=True,
                    write_statistics=True,
                )
                self.rows[month] = 0
            arrays = [_arrow_array(np.asarray(data[field.name])[start:end], field.type) for field in self.schema]
            table = pa.Table.from_arrays(arrays, schema=self.schema)
            self.writers[month].write_table(table, row_group_size=1_000_000)
            self.rows[month] += end - start

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


@dataclass
class HourMaps:
    baseline_w: np.ndarray
    baseline_dbm: np.ndarray
    split_blocked: np.ndarray
    split_strict: np.ndarray
    present: np.ndarray
    source_frame: pd.DataFrame


def load_hour_maps(path: Path) -> HourMaps:
    frame = pd.read_parquet(path)
    required = {"hour_bjt", "received_power_w", "received_power_dbm", "split_blocked", "split_strict_24h"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Hourly baseline missing columns: {sorted(missing)}")
    if frame["hour_bjt"].dt.tz is None:
        naive = frame["hour_bjt"]
    else:
        naive = frame["hour_bjt"].dt.tz_convert(TZ_BJT).dt.tz_localize(None)
    hour_ms = naive.to_numpy(dtype="datetime64[ms]").astype(np.int64)
    n_hours = int((END_MS - START_MS) // HOUR_MS)
    idx = ((hour_ms - START_MS) // HOUR_MS).astype(np.int64)
    if (idx < 0).any() or (idx >= n_hours).any() or len(np.unique(idx)) != len(idx):
        raise ValueError("Hourly baseline keys are out of range or duplicated")
    baseline_w = np.full(n_hours, np.nan, dtype=np.float64)
    baseline_dbm = np.full(n_hours, np.nan, dtype=np.float64)
    split_blocked = np.full(n_hours, -1, dtype=np.int8)
    split_strict = np.full(n_hours, -1, dtype=np.int8)
    present = np.zeros(n_hours, dtype=bool)
    baseline_w[idx] = frame["received_power_w"].to_numpy(dtype=np.float64)
    baseline_dbm[idx] = frame["received_power_dbm"].to_numpy(dtype=np.float64)
    split_blocked[idx] = frame["split_blocked"].map(BLOCKED_CODES).to_numpy(dtype=np.int8)
    split_strict[idx] = frame["split_strict_24h"].map(STRICT_CODES).to_numpy(dtype=np.int8)
    present[idx] = True
    if not (np.isfinite(baseline_w[idx]).all() and np.isfinite(baseline_dbm[idx]).all()):
        raise ValueError("Hourly baseline contains non-finite values")
    return HourMaps(baseline_w, baseline_dbm, split_blocked, split_strict, present, frame)


class SecondAggregator:
    KEYS = ("time", "power_w", "quality", "source_k", "physical", "segment")

    def __init__(self, writer: MonthlyParquetWriter, hours: HourMaps) -> None:
        self.writer = writer
        self.hours = hours
        self.tail: dict[str, np.ndarray] = {
            "time": np.empty(0, np.int64),
            "power_w": np.empty(0, np.float64),
            "quality": np.empty(0, np.uint16),
            "source_k": np.empty(0, np.int32),
            "physical": np.empty(0, bool),
            "segment": np.empty(0, np.int32),
        }
        self.total_rows = 0
        self.overfilled_seconds = 0

    def feed(self, incoming: dict[str, np.ndarray], final: bool = False) -> None:
        data = {key: np.concatenate((self.tail[key], np.asarray(incoming[key]))) for key in self.KEYS}
        n = data["time"].size
        if n == 0:
            return
        seconds = (data["time"] // SECOND_MS) * SECOND_MS
        if final:
            cut = n
        else:
            cut = int(np.searchsorted(seconds, seconds[-1], side="left"))
        if cut:
            self._consume({key: value[:cut] for key, value in data.items()})
        self.tail = {key: value[cut:] for key, value in data.items()}

    def close(self) -> None:
        self.feed({key: np.empty(0, dtype=value.dtype) for key, value in self.tail.items()}, final=True)

    def _consume(self, data: dict[str, np.ndarray]) -> None:
        t = data["time"]
        sec = (t // SECOND_MS) * SECOND_MS
        starts = np.r_[0, np.flatnonzero(sec[1:] != sec[:-1]) + 1]
        ends = np.r_[starts[1:], len(sec)]
        n = (ends - starts).astype(np.int64)
        physical = data["physical"].astype(np.int64)
        valid_n = np.add.reduceat(physical, starts)
        valid_power = np.where(data["physical"], data["power_w"], 0.0)
        power_sum = np.add.reduceat(valid_power, starts)
        mean_w = np.divide(power_sum, valid_n, out=np.full(len(starts), np.nan), where=valid_n > 0)
        q = data["quality"]
        q0 = np.add.reduceat((q == 0).astype(np.int64), starts)
        qnz = n - q0
        q16 = np.add.reduceat(((q & np.uint16(16)) != 0).astype(np.int64), starts)
        q32 = np.add.reduceat(((q & np.uint16(32)) != 0).astype(np.int64), starts)
        k = data["source_k"]
        k11 = np.add.reduceat((k == 11).astype(np.int64), starts)
        k101 = np.add.reduceat((k == 101).astype(np.int64), starts)
        kother = n - k11 - k101
        seg_min = np.minimum.reduceat(data["segment"], starts)
        seg_max = np.maximum.reduceat(data["segment"], starts)
        segment = np.where(seg_min == seg_max, seg_min, -1).astype(np.int32)
        second = sec[starts]
        hour_idx = ((second - START_MS) // HOUR_MS).astype(np.int64)
        baseline_w = self.hours.baseline_w[hour_idx]
        baseline_dbm = self.hours.baseline_dbm[hour_idx]
        dbm = 10.0 * np.log10(mean_w * 1000.0)
        ratio = mean_w / baseline_w
        envelope = np.sqrt(ratio)
        coverage = np.minimum(n / EXPECTED_100MS_PER_SECOND, 1.0)
        self.overfilled_seconds += int(np.count_nonzero(n > EXPECTED_100MS_PER_SECOND))
        out = {
            "second_bjt": second,
            "hour_bjt": (second // HOUR_MS) * HOUR_MS,
            "records_in_second": n.astype(np.uint16),
            "physical_valid_records": valid_n.astype(np.uint16),
            "coverage_fraction": coverage.astype(np.float32),
            "high_coverage_90": (n >= 9),
            "overfilled_nominal_second": (n > EXPECTED_100MS_PER_SECOND),
            "received_power_w": mean_w,
            "received_power_dbm": dbm.astype(np.float32),
            "slow_hourly_power_w": baseline_w,
            "slow_hourly_power_dbm": baseline_dbm.astype(np.float32),
            "fast_residual_db": (dbm - baseline_dbm).astype(np.float32),
            "normalized_power_ratio": ratio.astype(np.float32),
            "normalized_envelope": envelope.astype(np.float32),
            "quality_zero_records": q0.astype(np.uint16),
            "quality_nonzero_records": qnz.astype(np.uint16),
            "quality_bit16_records": q16.astype(np.uint16),
            "quality_bit32_records": q32.astype(np.uint16),
            "quality_zero_fraction": (q0 / n).astype(np.float32),
            "source_k11_records": k11.astype(np.uint16),
            "source_k101_records": k101.astype(np.uint16),
            "source_other_or_mixed_records": kother.astype(np.uint16),
            "continuous_segment_id": segment,
            "split_blocked_code": self.hours.split_blocked[hour_idx],
            "split_strict_24h_code": self.hours.split_strict[hour_idx],
        }
        self.writer.write(out)
        self.total_rows += len(second)


@dataclass
class BuildStats:
    source_rows: int = 0
    rows_before_campaign: int = 0
    rows_at_or_after_campaign: int = 0
    rows_in_campaign_before_dedup: int = 0
    unique_timestamps: int = 0
    duplicate_extra_records: int = 0
    physical_valid: int = 0
    physical_invalid: int = 0
    quality_nonzero: int = 0
    quality_bit16: int = 0
    quality_bit32: int = 0
    source_k11: int = 0
    source_k101: int = 0
    source_other: int = 0
    gaps_le_150ms: int = 0
    gaps_151_to_500ms: int = 0
    gaps_501_to_5000ms: int = 0
    gaps_over_5000ms: int = 0
    source_zero_steps: int = 0
    source_negative_steps: int = 0
    minimum_source_step_ms: int | None = None
    first_time_ms: int | None = None
    last_time_ms: int | None = None


class FastBuilder:
    def __init__(self, peak: Path, quality: Path, source_k: Path, hours: HourMaps, temp: Path, batch_size: int) -> None:
        self.peak_path = peak
        self.quality_path = quality
        self.source_k_path = source_k
        self.hours = hours
        self.batch_size = batch_size
        self.fast100_writer = MonthlyParquetWriter(temp / "fast_100ms", "fast_100ms", FAST100_SCHEMA)
        self.fast1_writer = MonthlyParquetWriter(temp / "fast_1s", "fast_1s", FAST1_SCHEMA)
        self.seconds = SecondAggregator(self.fast1_writer, hours)
        self.stats = BuildStats()
        self.last_unique_time: int | None = None
        self.segment_id = 1

    def consume_sorted(self, t: np.ndarray, p: np.ndarray, q: np.ndarray, k: np.ndarray) -> None:
        if t.size == 0:
            return
        starts = np.r_[0, np.flatnonzero(t[1:] != t[:-1]) + 1]
        ends = np.r_[starts[1:], len(t)]
        multiplicity = ends - starts
        watts = np.power(10.0, (p - 30.0) / 10.0)
        mean_w = np.add.reduceat(watts, starts) / multiplicity
        peak_dbm = 10.0 * np.log10(mean_w * 1000.0)
        group_t = t[starts]
        group_q = np.bitwise_or.reduceat(q, starts)
        k_min = np.minimum.reduceat(k, starts)
        k_max = np.maximum.reduceat(k, starts)
        group_k = np.where(k_min == k_max, k_min, -1).astype(np.int32)
        multiplicity = multiplicity.astype(np.int64)
        in_range = (group_t >= START_MS) & (group_t < END_MS)
        if not in_range.any():
            return
        group_t = group_t[in_range]
        peak_dbm = peak_dbm[in_range]
        mean_w = mean_w[in_range]
        group_q = group_q[in_range]
        group_k = group_k[in_range]
        multiplicity = multiplicity[in_range]

        h = ((group_t - START_MS) // HOUR_MS).astype(np.int64)
        finite = np.isfinite(peak_dbm)
        physical = finite & ((group_q & np.uint16(15)) == 0)
        baseline_w = self.hours.baseline_w[h]
        baseline_dbm = self.hours.baseline_dbm[h]
        mapped = self.hours.present[h]
        if np.any(physical & ~mapped):
            bad = str(np.datetime64(int(group_t[np.flatnonzero(physical & ~mapped)[0]]), "ms"))
            raise ValueError(f"Physical-valid fast record has no verified hourly baseline: {bad}")
        derived = physical & mapped
        ratio = np.full(len(group_t), np.nan, dtype=np.float64)
        ratio[derived] = mean_w[derived] / baseline_w[derived]
        envelope = np.sqrt(ratio)
        residual = np.full(len(group_t), np.nan, dtype=np.float64)
        residual[derived] = peak_dbm[derived] - baseline_dbm[derived]

        gap = np.empty(len(group_t), dtype=np.int64)
        gap[0] = -1 if self.last_unique_time is None else int(group_t[0] - self.last_unique_time)
        if len(group_t) > 1:
            gap[1:] = np.diff(group_t)
        breaks = gap > SEGMENT_GAP_MS
        segment = self.segment_id + np.cumsum(breaks, dtype=np.int64)
        self.segment_id = int(segment[-1])
        self.last_unique_time = int(group_t[-1])
        positive_gap = gap[gap >= 0]
        self.stats.gaps_le_150ms += int(np.count_nonzero(positive_gap <= 150))
        self.stats.gaps_151_to_500ms += int(np.count_nonzero((positive_gap > 150) & (positive_gap <= 500)))
        self.stats.gaps_501_to_5000ms += int(np.count_nonzero((positive_gap > 500) & (positive_gap <= 5000)))
        self.stats.gaps_over_5000ms += int(np.count_nonzero(positive_gap > 5000))

        self.stats.unique_timestamps += len(group_t)
        self.stats.duplicate_extra_records += int(np.sum(multiplicity - 1))
        self.stats.physical_valid += int(np.count_nonzero(physical))
        self.stats.physical_invalid += int(np.count_nonzero(~physical))
        self.stats.quality_nonzero += int(np.count_nonzero(group_q != 0))
        self.stats.quality_bit16 += int(np.count_nonzero((group_q & np.uint16(16)) != 0))
        self.stats.quality_bit32 += int(np.count_nonzero((group_q & np.uint16(32)) != 0))
        self.stats.source_k11 += int(np.count_nonzero(group_k == 11))
        self.stats.source_k101 += int(np.count_nonzero(group_k == 101))
        self.stats.source_other += int(np.count_nonzero((group_k != 11) & (group_k != 101)))
        if self.stats.first_time_ms is None:
            self.stats.first_time_ms = int(group_t[0])
        self.stats.last_time_ms = int(group_t[-1])

        fast100 = {
            "time_bjt": group_t,
            "hour_bjt": (group_t // HOUR_MS) * HOUR_MS,
            "peak_power_dbm": peak_dbm.astype(np.float32),
            "peak_power_w": mean_w,
            "slow_hourly_power_dbm": baseline_dbm.astype(np.float32),
            "slow_hourly_power_w": baseline_w,
            "fast_residual_db": residual.astype(np.float32),
            "normalized_power_ratio": ratio.astype(np.float32),
            "normalized_envelope": envelope.astype(np.float32),
            "quality_code": group_q,
            "source_bin_count": group_k,
            "physical_valid": physical,
            "source_record_multiplicity": np.minimum(multiplicity, np.iinfo(np.uint16).max).astype(np.uint16),
            "gap_from_previous_ms": gap.astype(np.int64),
            "continuous_segment_id": segment.astype(np.int32),
            "split_blocked_code": self.hours.split_blocked[h],
            "split_strict_24h_code": self.hours.split_strict[h],
        }
        self.fast100_writer.write(fast100)
        self.seconds.feed(
            {
                "time": group_t,
                "power_w": mean_w,
                "quality": group_q,
                "source_k": group_k,
                "physical": physical,
                "segment": segment.astype(np.int32),
            }
        )

    def run(self) -> None:
        peak_pf = pq.ParquetFile(self.peak_path)
        quality_pf = pq.ParquetFile(self.quality_path)
        source_pf = pq.ParquetFile(self.source_k_path)
        layouts = []
        for pf in (peak_pf, quality_pf, source_pf):
            layouts.append([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
        if not (layouts[0] == layouts[1] == layouts[2]):
            raise ValueError("The three Run_011 metrics do not share an identical row-group layout")

        peak_iter = peak_pf.iter_batches(batch_size=self.batch_size, columns=["time", "value"])
        quality_iter = quality_pf.iter_batches(batch_size=self.batch_size, columns=["time", "value"])
        source_iter = source_pf.iter_batches(batch_size=self.batch_size, columns=["time", "value"])
        tail_t = np.empty(0, np.int64)
        tail_p = np.empty(0, np.float64)
        tail_q = np.empty(0, np.uint16)
        tail_k = np.empty(0, np.int32)
        max_seen = np.iinfo(np.int64).min
        finalized_through = np.iinfo(np.int64).min
        prev_source_time: int | None = None

        for batch_number, (bp, bq, bk) in enumerate(zip(peak_iter, quality_iter, source_iter, strict=True), start=1):
            t = bp.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
            p = bp.column(1).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            tq = bq.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
            q = bq.column(1).to_numpy(zero_copy_only=False).astype(np.uint16, copy=False)
            tk = bk.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ms]").astype(np.int64, copy=False)
            k = bk.column(1).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            if not np.array_equal(t, tq) or not np.array_equal(t, tk):
                raise ValueError(f"Metric timestamps differ in batch {batch_number}")
            self.stats.source_rows += len(t)
            self.stats.rows_before_campaign += int(np.count_nonzero(t < START_MS))
            self.stats.rows_at_or_after_campaign += int(np.count_nonzero(t >= END_MS))
            self.stats.rows_in_campaign_before_dedup += int(np.count_nonzero((t >= START_MS) & (t < END_MS)))

            joined = t if prev_source_time is None else np.concatenate(([prev_source_time], t))
            deltas = np.diff(joined)
            if deltas.size:
                self.stats.source_zero_steps += int(np.count_nonzero(deltas == 0))
                self.stats.source_negative_steps += int(np.count_nonzero(deltas < 0))
                local_min = int(deltas.min())
                self.stats.minimum_source_step_ms = local_min if self.stats.minimum_source_step_ms is None else min(self.stats.minimum_source_step_ms, local_min)
                if local_min < -REORDER_BUFFER_MS:
                    raise ValueError(f"Timestamp reversal {local_min} ms exceeds reorder buffer")
            prev_source_time = int(t[-1])

            ct = np.concatenate((tail_t, t))
            cp = np.concatenate((tail_p, p))
            cq = np.concatenate((tail_q, q))
            ck = np.concatenate((tail_k, k))
            order = np.argsort(ct, kind="stable")
            ct, cp, cq, ck = ct[order], cp[order], cq[order], ck[order]
            max_seen = max(max_seen, int(ct[-1]))
            cut = int(np.searchsorted(ct, max_seen - REORDER_BUFFER_MS, side="right"))
            if cut:
                if ct[0] <= finalized_through:
                    raise ValueError("Timestamp arrived after reorder-buffer finalization")
                self.consume_sorted(ct[:cut], cp[:cut], cq[:cut], ck[:cut])
                finalized_through = int(ct[cut - 1])
            tail_t, tail_p, tail_q, tail_k = ct[cut:], cp[cut:], cq[cut:], ck[cut:]
            if batch_number % 10 == 0:
                print(f"source rows processed: {self.stats.source_rows:,}", flush=True)

        self.consume_sorted(tail_t, tail_p, tail_q, tail_k)
        self.seconds.close()
        self.fast100_writer.close()
        self.fast1_writer.close()


def _correlation_by_group(times: np.ndarray, values: np.ndarray, groups: np.ndarray, segment: np.ndarray, lag_ms: int, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    target = times - lag_ms
    previous = np.searchsorted(times, target)
    valid = previous < len(times)
    safe = np.minimum(previous, len(times) - 1)
    valid &= times[safe] == target
    valid &= groups[safe] == groups
    valid &= segment[safe] == segment
    valid &= np.isfinite(values) & np.isfinite(values[safe])
    idx = np.flatnonzero(valid)
    if not len(idx):
        return np.full(group_count, np.nan), np.zeros(group_count, np.int64)
    prev_idx = safe[idx]
    g = groups[idx]
    x = values[prev_idx]
    y = values[idx]
    count = np.bincount(g, minlength=group_count).astype(np.int64)
    sx = np.bincount(g, weights=x, minlength=group_count)
    sy = np.bincount(g, weights=y, minlength=group_count)
    sx2 = np.bincount(g, weights=x * x, minlength=group_count)
    sy2 = np.bincount(g, weights=y * y, minlength=group_count)
    sxy = np.bincount(g, weights=x * y, minlength=group_count)
    numerator = count * sxy - sx * sy
    denominator = np.sqrt(np.maximum(count * sx2 - sx * sx, 0.0) * np.maximum(count * sy2 - sy * sy, 0.0))
    corr = np.divide(numerator, denominator, out=np.full(group_count, np.nan), where=(count >= 3) & (denominator > 0))
    return corr, count


def build_window_statistics(fast1_dir: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_frames: list[pd.DataFrame] = []
    monthly_files = sorted(fast1_dir.glob("fast_1s_*.parquet"))
    for file_index, path in enumerate(monthly_files, start=1):
        frame = pd.read_parquet(path)
        times = frame["second_bjt"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
        residual = frame["fast_residual_db"].to_numpy(dtype=np.float64)
        power = frame["received_power_w"].to_numpy(dtype=np.float64)
        envelope = frame["normalized_envelope"].to_numpy(dtype=np.float64)
        ratio = frame["normalized_power_ratio"].to_numpy(dtype=np.float64)
        segment = frame["continuous_segment_id"].to_numpy(dtype=np.int32)
        for minutes in WINDOW_MINUTES:
            duration_ms = minutes * 60_000
            window_ms = (times // duration_ms) * duration_ms
            unique_windows, group = np.unique(window_ms, return_inverse=True)
            group_count = len(unique_windows)
            work = pd.DataFrame(
                {
                    "group": group,
                    "power": power,
                    "residual": residual,
                    "envelope": envelope,
                    "ratio": ratio,
                    "high": frame["high_coverage_90"].to_numpy(dtype=np.int8),
                    "records": frame["records_in_second"].to_numpy(dtype=np.int64),
                    "q0": frame["quality_zero_records"].to_numpy(dtype=np.int64),
                    "k11": frame["source_k11_records"].to_numpy(dtype=np.int64),
                    "k101": frame["source_k101_records"].to_numpy(dtype=np.int64),
                    "segment": segment,
                    "blocked": frame["split_blocked_code"].to_numpy(dtype=np.int8),
                    "strict": frame["split_strict_24h_code"].to_numpy(dtype=np.int8),
                }
            )
            grouped = work.groupby("group", sort=True, observed=True)
            base = grouped.agg(
                observed_seconds=("power", "count"),
                high_coverage_seconds=("high", "sum"),
                record_count=("records", "sum"),
                quality_zero_records=("q0", "sum"),
                source_k11_records=("k11", "sum"),
                source_k101_records=("k101", "sum"),
                segment_count=("segment", "nunique"),
                split_blocked_min=("blocked", "min"),
                split_blocked_max=("blocked", "max"),
                split_strict_min=("strict", "min"),
                split_strict_max=("strict", "max"),
                received_power_w=("power", "mean"),
                residual_db_mean=("residual", "mean"),
                residual_db_std=("residual", "std"),
                residual_db_min=("residual", "min"),
                residual_db_max=("residual", "max"),
                normalized_ratio_mean=("ratio", "mean"),
                normalized_envelope_mean=("envelope", "mean"),
                normalized_envelope_std=("envelope", "std"),
            ).reset_index(drop=True)
            residual_quantiles = grouped["residual"].quantile([0.05, 0.5, 0.95]).unstack()
            residual_quantiles = residual_quantiles.reindex(index=np.arange(group_count))
            envelope_quantiles = grouped["envelope"].quantile([0.05, 0.5, 0.95]).unstack()
            envelope_quantiles = envelope_quantiles.reindex(index=np.arange(group_count))
            mean_by_group = base["received_power_w"].to_numpy(dtype=np.float64)
            local_envelope = np.sqrt(power / mean_by_group[group])
            local_work = pd.DataFrame({"group": group, "value": local_envelope})
            local_grouped = local_work.groupby("group", sort=True, observed=True)["value"]
            local_stats = local_grouped.agg(["mean", "std"])
            local_quantiles = local_grouped.quantile([0.05, 0.5, 0.95]).unstack().reindex(index=np.arange(group_count))
            autocorr1, pairs1 = _correlation_by_group(times, residual, group, segment, 1_000, group_count)
            autocorr10, pairs10 = _correlation_by_group(times, residual, group, segment, 10_000, group_count)
            finite_residual = np.isfinite(residual)
            residual_count = np.bincount(group, weights=finite_residual.astype(np.int64), minlength=group_count)
            fade1 = np.divide(
                np.bincount(group, weights=(finite_residual & (residual <= -1.0)).astype(np.int64), minlength=group_count),
                residual_count,
                out=np.full(group_count, np.nan),
                where=residual_count > 0,
            )
            fade3 = np.divide(
                np.bincount(group, weights=(finite_residual & (residual <= -3.0)).astype(np.int64), minlength=group_count),
                residual_count,
                out=np.full(group_count, np.nan),
                where=residual_count > 0,
            )
            fade5 = np.divide(
                np.bincount(group, weights=(finite_residual & (residual <= -5.0)).astype(np.int64), minlength=group_count),
                residual_count,
                out=np.full(group_count, np.nan),
                where=residual_count > 0,
            )
            expected = minutes * 60
            records = base["record_count"].to_numpy(dtype=np.float64)
            out = pd.DataFrame(
                {
                    "window_start_bjt": pd.to_datetime(unique_windows, unit="ms"),
                    "window_minutes": np.full(group_count, minutes, dtype=np.int16),
                    "expected_seconds": np.full(group_count, expected, dtype=np.int32),
                    "observed_seconds": base["observed_seconds"].to_numpy(dtype=np.int32),
                    "window_coverage_fraction": base["observed_seconds"].to_numpy(dtype=np.float64) / expected,
                    "high_coverage_seconds": base["high_coverage_seconds"].to_numpy(dtype=np.int32),
                    "high_coverage_fraction": base["high_coverage_seconds"].to_numpy(dtype=np.float64) / expected,
                    "record_count": base["record_count"].to_numpy(dtype=np.int64),
                    "quality_zero_fraction": np.divide(base["quality_zero_records"], records, out=np.full(group_count, np.nan), where=records > 0),
                    "source_k11_fraction": np.divide(base["source_k11_records"], records, out=np.full(group_count, np.nan), where=records > 0),
                    "source_k101_fraction": np.divide(base["source_k101_records"], records, out=np.full(group_count, np.nan), where=records > 0),
                    "continuous_segment_count": base["segment_count"].to_numpy(dtype=np.int32),
                    "split_blocked_code": np.where(base["split_blocked_min"] == base["split_blocked_max"], base["split_blocked_min"], -1).astype(np.int8),
                    "split_strict_24h_code": np.where(base["split_strict_min"] == base["split_strict_max"], base["split_strict_min"], -1).astype(np.int8),
                    "received_power_w": mean_by_group,
                    "received_power_dbm": 10.0 * np.log10(mean_by_group * 1000.0),
                    "fast_residual_db_mean": base["residual_db_mean"],
                    "fast_residual_db_std": base["residual_db_std"],
                    "fast_residual_db_min": base["residual_db_min"],
                    "fast_residual_db_p05": residual_quantiles[0.05].to_numpy(),
                    "fast_residual_db_p50": residual_quantiles[0.5].to_numpy(),
                    "fast_residual_db_p95": residual_quantiles[0.95].to_numpy(),
                    "fast_residual_db_max": base["residual_db_max"],
                    "normalized_power_ratio_mean": base["normalized_ratio_mean"],
                    "hour_normalized_envelope_mean": base["normalized_envelope_mean"],
                    "hour_normalized_envelope_std": base["normalized_envelope_std"],
                    "hour_normalized_envelope_p05": envelope_quantiles[0.05].to_numpy(),
                    "hour_normalized_envelope_p50": envelope_quantiles[0.5].to_numpy(),
                    "hour_normalized_envelope_p95": envelope_quantiles[0.95].to_numpy(),
                    "window_normalized_envelope_mean": local_stats["mean"].to_numpy(),
                    "window_normalized_envelope_std": local_stats["std"].to_numpy(),
                    "window_normalized_envelope_p05": local_quantiles[0.05].to_numpy(),
                    "window_normalized_envelope_p50": local_quantiles[0.5].to_numpy(),
                    "window_normalized_envelope_p95": local_quantiles[0.95].to_numpy(),
                    "fade_below_minus1db_fraction": fade1,
                    "fade_below_minus3db_fraction": fade3,
                    "fade_below_minus5db_fraction": fade5,
                    "residual_autocorrelation_lag1s": autocorr1,
                    "residual_autocorrelation_lag1s_pairs": pairs1,
                    "residual_autocorrelation_lag10s": autocorr10,
                    "residual_autocorrelation_lag10s_pairs": pairs10,
                }
            )
            out["primary_usable"] = (
                (out["window_coverage_fraction"] >= 0.95)
                & (out["high_coverage_fraction"] >= 0.90)
                & (out["continuous_segment_count"] == 1)
                & (out["split_strict_24h_code"] >= 0)
            )
            all_frames.append(out)
        print(f"window statistics: {file_index}/{len(monthly_files)} months", flush=True)
    combined = pd.concat(all_frames, ignore_index=True).sort_values(["window_minutes", "window_start_bjt"], kind="stable")
    combined.to_parquet(output_path, index=False, compression="zstd")
    summary = {
        "rows": len(combined),
        "rows_by_window_minutes": {str(int(k)): int(v) for k, v in combined.groupby("window_minutes").size().items()},
        "primary_usable_by_window_minutes": {str(int(k)): int(v) for k, v in combined.groupby("window_minutes")["primary_usable"].sum().items()},
    }
    return combined, summary


def write_support_files(temp: Path, builder: FastBuilder, windows: pd.DataFrame, manifest: dict[str, Any]) -> None:
    split_rows = [
        {"field": "split_blocked_code", "code": code, "label": label} for label, code in BLOCKED_CODES.items()
    ] + [
        {"field": "split_strict_24h_code", "code": code, "label": label} for label, code in STRICT_CODES.items()
    ]
    pd.DataFrame(split_rows).to_csv(temp / "time_split_codes.csv", index=False, encoding="utf-8-sig")

    dictionary_rows: list[dict[str, str]] = []
    descriptions = {
        "peak_power_dbm": ("Run_011 peak received power after exact-time deduplication", "dBm"),
        "peak_power_w": ("Same peak power converted to linear watts", "W"),
        "slow_hourly_power_dbm": ("Verified Revision_002 hourly W-domain mean", "dBm"),
        "fast_residual_db": ("Instantaneous/one-second power minus verified hourly baseline", "dB"),
        "normalized_power_ratio": ("Fast power divided by verified hourly mean power", "1"),
        "normalized_envelope": ("Square root of normalized power ratio; positive distribution-fitting variable", "1"),
        "quality_code": ("Run_011 bitwise quality flag; nonzero rows retained", "bit mask"),
        "source_bin_count": ("Native spectrum-bin count/configuration marker", "count"),
        "continuous_segment_id": (f"Segment changes after a timestamp gap greater than {SEGMENT_GAP_MS} ms", "identifier"),
        "coverage_fraction": ("Observed unique records divided by 10 nominal 100 ms slots, capped at 1", "1"),
        "primary_usable": ("Window has >=95% seconds, >=90% high-coverage seconds, one segment and one strict split", "boolean"),
    }
    for table_name, schema in (("fast_100ms", FAST100_SCHEMA), ("fast_1s", FAST1_SCHEMA)):
        for field in schema:
            description, unit = descriptions.get(field.name, (field.name.replace("_", " "), "see name"))
            dictionary_rows.append({"table": table_name, "column": field.name, "type": str(field.type), "unit": unit, "description": description})
    for column in windows.columns:
        description, unit = descriptions.get(column, (column.replace("_", " "), "see name"))
        dictionary_rows.append({"table": "fast_window_statistics", "column": column, "type": str(windows[column].dtype), "unit": unit, "description": description})
    pd.DataFrame(dictionary_rows).to_csv(temp / "data_dictionary.csv", index=False, encoding="utf-8-sig")

    month_rows = []
    months = sorted(set(builder.fast100_writer.rows) | set(builder.fast1_writer.rows))
    for month in months:
        month_rows.append(
            {
                "month_bjt": month,
                "fast_100ms_rows": builder.fast100_writer.rows.get(month, 0),
                "fast_1s_rows": builder.fast1_writer.rows.get(month, 0),
                "nominal_100ms_to_1s_ratio": builder.fast100_writer.rows.get(month, 0) / max(builder.fast1_writer.rows.get(month, 0), 1),
            }
        )
    pd.DataFrame(month_rows).to_csv(temp / "monthly_summary.csv", index=False, encoding="utf-8-sig")

    readme = f"""# Fast Variation Revision_001

This package is the fast branch paired with the verified hourly `Revision_002` dataset.

- `fast_100ms/`: traceable deduplicated 100 ms base records, one Parquet per Beijing-time month.
- `fast_1s/`: primary fast-variation analysis records, one Parquet per month; power is averaged in watts.
- `fast_window_statistics.parquet`: non-overlapping 5/10/15/30/60 minute summaries. The 15 minute scale is primary; the others are sensitivity scales.
- `hour_bjt` is the foreign key to the existing hourly measurement/ERA5 tables. ERA5 values are not repeated into every fast row.
- `normalized_envelope` is positive and is the correct stored input for subsequent Weibull/Rice/Rayleigh/Gaussian comparison. No distribution claim is made during data construction.
- Split labels are integer-coded to keep the 129 million-row base compact; mappings are in `time_split_codes.csv`.
- Timestamps are Beijing wall-clock values stored without a Parquet timezone annotation; this is explicit in schema metadata.

Important: this dataset contains received power and fast residuals, not a calibrated attenuation target. The unresolved absolute reference-power conflict remains outside this build.
"""
    (temp / "README.md").write_text(readme, encoding="utf-8")

    stats = manifest["build_summary"]
    qa = f"""# Fast Variation Revision_001 QA

## Result

- Build status: BUILT; independent validation is performed by `validate_fast_dataset.py`.
- Source rows: {stats['source_rows']:,}.
- Campaign rows before exact-time deduplication: {stats['rows_in_campaign_before_dedup']:,}.
- Unique 100 ms-layer timestamps: {stats['unique_timestamps']:,}.
- Exact duplicate extra records: {stats['duplicate_extra_records']:,}.
- One-second rows: {stats['one_second_rows']:,}.
- Continuous segments using a >{SEGMENT_GAP_MS} ms gap: {stats['continuous_segments']:,}.
- Quality-code nonzero records retained: {stats['quality_nonzero']:,}.
- Physical-invalid records retained and explicitly flagged: {stats['physical_invalid']:,}.

## Interpretation guardrails

The 100 ms layer is a traceability and sensitivity layer. The paper's primary fast-variation analysis remains one second. Weather remains hourly and is joined only through `hour_bjt`; duplicating an hourly weather value across fast rows does not create additional independent atmospheric observations. The 15 minute window is the primary distribution-analysis window, while 5/10/30/60 minutes are sensitivity checks. Distribution fitting must use `normalized_envelope` or the documented window-normalized envelope, never raw negative dBm, and no arbitrary offset is permitted.
"""
    (temp / "QA.md").write_text(qa, encoding="utf-8")


def main() -> int:
    args = parse_args()
    run011 = args.run011.resolve()
    hourly = args.hourly.resolve()
    output = args.output.resolve()
    temp = output.with_name(f".{output.name}.building")
    if output.exists() or temp.exists():
        raise FileExistsError(f"Output or temporary path already exists: {output} / {temp}")
    peak = metric_path(run011, "peak_power_dbm")
    quality = metric_path(run011, "quality_code")
    source_k = metric_path(run011, "source_bin_count")
    direct_inputs = [peak, quality, source_k, hourly]
    for path in direct_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    layouts = {path.stem: parquet_layout(path) for path in (peak, quality, source_k)}
    if len({item["rows"] for item in layouts.values()}) != 1:
        raise ValueError("Run_011 primary metrics have different row counts")
    if len({item["row_group_rows_sha256"] for item in layouts.values()}) != 1:
        raise ValueError("Run_011 primary metrics have different row-group layouts")

    print("hashing direct inputs before build", flush=True)
    input_before = {str(path): sha256_file(path) for path in direct_inputs}
    temp.mkdir(parents=True, exist_ok=False)
    try:
        hours = load_hour_maps(hourly)
        builder = FastBuilder(peak, quality, source_k, hours, temp, args.batch_size)
        builder.run()
        stats = builder.stats
        if stats.unique_timestamps != sum(builder.fast100_writer.rows.values()):
            raise ValueError("100 ms writer row count differs from deduplicated timestamp count")
        print("building multi-window statistics", flush=True)
        windows, window_summary = build_window_statistics(temp / "fast_1s", temp / "fast_window_statistics.parquet")

        print("hashing direct inputs after build", flush=True)
        input_after = {str(path): sha256_file(path) for path in direct_inputs}
        if input_before != input_after:
            raise RuntimeError("A protected direct input changed during the build")
        summary = json_ready(vars(stats))
        summary.update(
            {
                "one_second_rows": builder.seconds.total_rows,
                "overfilled_nominal_seconds": builder.seconds.overfilled_seconds,
                "continuous_segments": builder.segment_id,
                "fast_100ms_rows_by_month": builder.fast100_writer.rows,
                "fast_1s_rows_by_month": builder.fast1_writer.rows,
                "first_time_bjt": str(np.datetime64(stats.first_time_ms, "ms")),
                "last_time_bjt": str(np.datetime64(stats.last_time_ms, "ms")),
                "window_summary": window_summary,
            }
        )
        manifest = {
            "status": "BUILT_PENDING_INDEPENDENT_VALIDATION",
            "dataset": "Fast_Variation_Revision_001",
            "created_bjt": datetime.now().astimezone().isoformat(),
            "campaign": {"start_bjt_inclusive": START_NAIVE.isoformat(), "end_bjt_exclusive": END_NAIVE.isoformat()},
            "direct_inputs_sha256_before": input_before,
            "direct_inputs_sha256_after": input_after,
            "run011_layouts": layouts,
            "contracts": {
                "time_semantics": "Run_011 timestamps are Beijing wall-clock timestamps; outputs retain wall-clock values without timezone annotation",
                "deduplication": "60 s stable-sort reorder buffer; equal timestamps merged by W-domain power mean, bitwise-OR quality, and common-or--1 source-bin count",
                "fast_100ms": "traceability/base layer; no interpolation; all unique timestamps retained",
                "fast_1s": "primary paper fast layer; deduplicated physical-valid W-domain mean per clock second; missing seconds not inserted",
                "continuous_segment": f"new segment when adjacent unique timestamps differ by more than {SEGMENT_GAP_MS} ms",
                "slow_baseline": "verified Revision_002 hourly W-domain mean, joined by exact Beijing-time hour",
                "residual": "received dBm minus hourly baseline dBm",
                "normalized_envelope": "sqrt(fast power / hourly mean power); positive and dimensionless",
                "weather": "ERA5 remains hourly and is linked by hour_bjt only; it is not expanded into independent fast samples",
                "windows": "clock-aligned non-overlapping 5/10/15/30/60 min summaries; 15 min primary, others sensitivity",
                "distribution_fitting": "not performed during construction; use positive normalized envelope; no arbitrary offset",
                "attenuation": "not generated because the absolute reference-power conflict remains unresolved",
            },
            "split_code_maps": {"split_blocked_code": BLOCKED_CODES, "split_strict_24h_code": STRICT_CODES},
            "build_summary": summary,
            "schemas": {"fast_100ms": str(FAST100_SCHEMA), "fast_1s": str(FAST1_SCHEMA)},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "pid": os.getpid(),
            },
        }
        write_support_files(temp, builder, windows, manifest)
        (temp / "run_manifest.json").write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        checksum_lines = []
        for path in sorted((p for p in temp.rglob("*") if p.is_file() and p.name != "checksums_build.sha256"), key=lambda p: p.as_posix().lower()):
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(temp).as_posix()}")
        (temp / "checksums_build.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
        temp.replace(output)
        print(json.dumps(json_ready({"output": str(output), "summary": summary}), ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception:
        failure = temp.with_name(f"{output.name}_Failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        failure_source = temp if temp.exists() else output if output.exists() else None
        if failure_source is not None:
            (failure_source / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
            failure_source.replace(failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
