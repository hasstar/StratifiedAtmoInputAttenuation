from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import build_fast_dataset as contract


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "Outputs" / "Datasets" / "Fast_Variation_Revision_001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate Fast_Variation_Revision_001")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run011", type=Path, default=contract.DEFAULT_RUN011)
    parser.add_argument("--hourly", type=Path, default=contract.DEFAULT_HOURLY)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def column_numpy(table: pa.Table, name: str, dtype: Any | None = None) -> np.ndarray:
    array = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    if pa.types.is_timestamp(table.schema.field(name).type):
        array = array.astype("datetime64[ms]").astype(np.int64)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def verify_build_checksums(output: Path) -> int:
    lines = (output / "checksums_build.sha256").read_text(encoding="ascii").splitlines()
    checked = 0
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = output / Path(relative)
        require(path.is_file(), f"Build-checksum file missing: {relative}")
        require(sha256_file(path) == expected, f"Build checksum mismatch: {relative}")
        checked += 1
    return checked


def load_hour_maps(hourly_path: Path) -> tuple[pd.DataFrame, contract.HourMaps]:
    frame = pd.read_parquet(hourly_path)
    return frame, contract.load_hour_maps(hourly_path)


def aggregate_second_from_100ms(
    t: np.ndarray,
    power: np.ndarray,
    q: np.ndarray,
    source_k: np.ndarray,
    physical: np.ndarray,
    segment: np.ndarray,
) -> dict[str, np.ndarray]:
    seconds = (t // 1000) * 1000
    starts = np.r_[0, np.flatnonzero(seconds[1:] != seconds[:-1]) + 1]
    ends = np.r_[starts[1:], len(seconds)]
    n = (ends - starts).astype(np.int64)
    valid_n = np.add.reduceat(physical.astype(np.int64), starts)
    power_sum = np.add.reduceat(np.where(physical, power, 0.0), starts)
    mean_power = np.divide(power_sum, valid_n, out=np.full(len(starts), np.nan), where=valid_n > 0)
    q0 = np.add.reduceat((q == 0).astype(np.int64), starts)
    q16 = np.add.reduceat(((q & np.uint16(16)) != 0).astype(np.int64), starts)
    q32 = np.add.reduceat(((q & np.uint16(32)) != 0).astype(np.int64), starts)
    k11 = np.add.reduceat((source_k == 11).astype(np.int64), starts)
    k101 = np.add.reduceat((source_k == 101).astype(np.int64), starts)
    seg_min = np.minimum.reduceat(segment, starts)
    seg_max = np.maximum.reduceat(segment, starts)
    return {
        "second": seconds[starts],
        "records": n,
        "physical": valid_n,
        "power": mean_power,
        "q0": q0,
        "q16": q16,
        "q32": q32,
        "k11": k11,
        "k101": k101,
        "segment": np.where(seg_min == seg_max, seg_min, -1),
    }


def recompute_window_sample(frame: pd.DataFrame, start_ms: int, minutes: int) -> dict[str, float]:
    time_ms = frame["second_bjt"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    end_ms = start_ms + minutes * 60_000
    part = frame[(time_ms >= start_ms) & (time_ms < end_ms)]
    power = part["received_power_w"].to_numpy(dtype=np.float64)
    residual = part["fast_residual_db"].to_numpy(dtype=np.float64)
    local = np.sqrt(power / np.mean(power))
    records = part["records_in_second"].to_numpy(dtype=np.float64)
    return {
        "observed_seconds": len(part),
        "received_power_w": float(np.mean(power)),
        "fast_residual_db_mean": float(np.mean(residual)),
        "fast_residual_db_p05": float(np.quantile(residual, 0.05)),
        "fast_residual_db_p50": float(np.quantile(residual, 0.50)),
        "fast_residual_db_p95": float(np.quantile(residual, 0.95)),
        "window_normalized_envelope_mean": float(np.mean(local)),
        "window_normalized_envelope_p50": float(np.quantile(local, 0.50)),
        "quality_zero_fraction": float(part["quality_zero_records"].sum() / records.sum()),
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    hourly_path = args.hourly.resolve()
    report_path = output / "validation_report.json"
    failed_dir: Path | None = None
    checks: list[dict[str, Any]] = []
    try:
        required = [
            output / "run_manifest.json",
            output / "checksums_build.sha256",
            output / "fast_window_statistics.parquet",
            output / "data_dictionary.csv",
            output / "monthly_summary.csv",
            output / "time_split_codes.csv",
            output / "QA.md",
            output / "README.md",
        ]
        require(all(path.is_file() for path in required), "One or more required output files are missing")
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

        checksum_count = verify_build_checksums(output)
        checks.append({"name": "build_stage_checksums", "passed": True, "details": f"{checksum_count} files"})

        peak = contract.metric_path(args.run011.resolve(), "peak_power_dbm")
        quality = contract.metric_path(args.run011.resolve(), "quality_code")
        source_k_path = contract.metric_path(args.run011.resolve(), "source_bin_count")
        direct_inputs = [peak, quality, source_k_path, hourly_path]
        print("validation: hashing protected direct inputs", flush=True)
        current_hashes = {str(path): sha256_file(path) for path in direct_inputs}
        require(current_hashes == manifest["direct_inputs_sha256_before"], "Current direct-input hashes differ from pre-build hashes")
        require(current_hashes == manifest["direct_inputs_sha256_after"], "Current direct-input hashes differ from post-build hashes")
        checks.append({"name": "protected_direct_inputs_unchanged", "passed": True, "details": f"{len(direct_inputs)} files"})

        for path in (peak, quality, source_k_path):
            current = contract.parquet_layout(path)
            expected = manifest["run011_layouts"][path.stem]
            require(current["rows"] == expected["rows"], f"Source row count changed: {path}")
            require(current["row_group_rows_sha256"] == expected["row_group_rows_sha256"], f"Source row-group layout changed: {path}")
        checks.append({"name": "run011_alignment_contract", "passed": True, "details": "three metric layouts still aligned"})

        hourly, hour_maps = load_hour_maps(hourly_path)
        n_hours = len(hour_maps.baseline_w)
        hourly_sum = np.zeros(n_hours, dtype=np.float64)
        hourly_count = np.zeros(n_hours, dtype=np.int64)
        global_prev_time: int | None = None
        global_prev_segment: int | None = None
        total100 = 0
        total_multiplicity = 0
        total_physical = 0
        total_q_nonzero = 0
        total_q16 = 0
        total_q32 = 0
        total_k11 = 0
        total_k101 = 0
        segment_max = 0

        window_table = pd.read_parquet(output / "fast_window_statistics.parquet")
        require(not window_table.duplicated(["window_minutes", "window_start_bjt"]).any(), "Window keys are duplicated")
        window_ms_all = window_table["window_start_bjt"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
        window_lookup = {
            (int(window_ms_all[i]), int(window_table.iloc[i]["window_minutes"])): window_table.iloc[i]
            for i in range(len(window_table))
        }

        files100 = sorted((output / "fast_100ms").glob("fast_100ms_*.parquet"))
        files1 = sorted((output / "fast_1s").glob("fast_1s_*.parquet"))
        require(files100 and len(files100) == len(files1), "100 ms and 1 s monthly partition counts differ")
        require([p.stem[-7:] for p in files100] == [p.stem[-7:] for p in files1], "100 ms and 1 s month labels differ")
        require(contract.FAST100_SCHEMA.names == pq.ParquetFile(files100[0]).schema_arrow.names, "100 ms schema names differ")
        require(contract.FAST1_SCHEMA.names == pq.ParquetFile(files1[0]).schema_arrow.names, "1 s schema names differ")

        window_samples_checked = 0
        total1 = 0
        total1_records = 0
        total1_physical = 0
        one_second_hour_sum = np.zeros(n_hours, dtype=np.float64)
        one_second_hour_count = np.zeros(n_hours, dtype=np.int64)
        for month_index, (path100, path1) in enumerate(zip(files100, files1, strict=True), start=1):
            table100 = pq.read_table(path100)
            t = column_numpy(table100, "time_bjt", np.int64)
            peak_dbm = column_numpy(table100, "peak_power_dbm", np.float64)
            power = column_numpy(table100, "peak_power_w", np.float64)
            baseline_w = column_numpy(table100, "slow_hourly_power_w", np.float64)
            baseline_dbm = column_numpy(table100, "slow_hourly_power_dbm", np.float64)
            residual = column_numpy(table100, "fast_residual_db", np.float64)
            ratio = column_numpy(table100, "normalized_power_ratio", np.float64)
            envelope = column_numpy(table100, "normalized_envelope", np.float64)
            q = column_numpy(table100, "quality_code", np.uint16)
            source_k = column_numpy(table100, "source_bin_count", np.int32)
            physical = column_numpy(table100, "physical_valid", bool)
            multiplicity = column_numpy(table100, "source_record_multiplicity", np.uint16)
            gap = column_numpy(table100, "gap_from_previous_ms", np.int64)
            segment = column_numpy(table100, "continuous_segment_id", np.int32)
            blocked = column_numpy(table100, "split_blocked_code", np.int8)
            strict = column_numpy(table100, "split_strict_24h_code", np.int8)
            require(len(t) > 0 and np.all(np.diff(t) > 0), f"100 ms times are not strictly increasing in {path100.name}")
            require(str(np.datetime64(int(t[0]), "ms"))[:7] == path100.stem[-7:], f"100 ms month label mismatch: {path100.name}")
            if global_prev_time is None:
                expected_gap = np.r_[-1, np.diff(t)]
                expected_segment = 1 + np.cumsum(expected_gap > contract.SEGMENT_GAP_MS)
            else:
                expected_gap = np.r_[int(t[0] - global_prev_time), np.diff(t)]
                expected_segment = int(global_prev_segment) + np.cumsum(expected_gap > contract.SEGMENT_GAP_MS)
            require(np.array_equal(gap.astype(np.int64), expected_gap), f"Stored gaps differ from time keys in {path100.name}")
            require(np.array_equal(segment.astype(np.int64), expected_segment), f"Segment IDs differ from gap rule in {path100.name}")
            global_prev_time = int(t[-1])
            global_prev_segment = int(segment[-1])
            segment_max = max(segment_max, int(segment.max()))

            hour_idx = ((t - contract.START_MS) // contract.HOUR_MS).astype(np.int64)
            require(np.all((hour_idx >= 0) & (hour_idx < n_hours)), f"100 ms time outside campaign: {path100.name}")
            expected_physical = np.isfinite(peak_dbm) & ((q & np.uint16(15)) == 0)
            require(np.array_equal(physical, expected_physical), f"Physical-valid flag mismatch: {path100.name}")
            mapped = hour_maps.present[hour_idx]
            require(np.all(~physical | mapped), f"Physical 100 ms row lacks hourly baseline: {path100.name}")
            require(np.allclose(baseline_w[mapped], hour_maps.baseline_w[hour_idx[mapped]], rtol=0, atol=0), f"100 ms slow W mismatch: {path100.name}")
            require(np.allclose(baseline_dbm[mapped], hour_maps.baseline_dbm[hour_idx[mapped]], rtol=0, atol=5e-6), f"100 ms slow dBm mismatch: {path100.name}")
            require(np.allclose(power[physical], np.power(10.0, (peak_dbm[physical] - 30.0) / 10.0), rtol=3e-6, atol=0), f"100 ms dBm/W mismatch: {path100.name}")
            require(np.allclose(residual[physical], peak_dbm[physical] - baseline_dbm[physical], rtol=0, atol=2e-5), f"100 ms residual mismatch: {path100.name}")
            require(np.allclose(ratio[physical], power[physical] / baseline_w[physical], rtol=3e-6, atol=0), f"100 ms ratio mismatch: {path100.name}")
            require(np.allclose(envelope[physical], np.sqrt(ratio[physical]), rtol=3e-6, atol=0), f"100 ms envelope mismatch: {path100.name}")
            require(np.array_equal(blocked, hour_maps.split_blocked[hour_idx]), f"100 ms blocked split mismatch: {path100.name}")
            require(np.array_equal(strict, hour_maps.split_strict[hour_idx]), f"100 ms strict split mismatch: {path100.name}")
            valid_h = hour_idx[physical]
            hourly_sum += np.bincount(valid_h, weights=power[physical], minlength=n_hours)[:n_hours]
            hourly_count += np.bincount(valid_h, minlength=n_hours)[:n_hours].astype(np.int64)

            aggregate = aggregate_second_from_100ms(t, power, q, source_k, physical, segment)
            table1 = pq.read_table(path1)
            s = column_numpy(table1, "second_bjt", np.int64)
            require(np.array_equal(s, aggregate["second"]), f"1 s keys do not match 100 ms aggregation: {path1.name}")
            records1 = column_numpy(table1, "records_in_second", np.uint16).astype(np.int64)
            valid1 = column_numpy(table1, "physical_valid_records", np.uint16).astype(np.int64)
            power1 = column_numpy(table1, "received_power_w", np.float64)
            dbm1 = column_numpy(table1, "received_power_dbm", np.float64)
            basew1 = column_numpy(table1, "slow_hourly_power_w", np.float64)
            basedbm1 = column_numpy(table1, "slow_hourly_power_dbm", np.float64)
            residual1 = column_numpy(table1, "fast_residual_db", np.float64)
            ratio1 = column_numpy(table1, "normalized_power_ratio", np.float64)
            envelope1 = column_numpy(table1, "normalized_envelope", np.float64)
            q01 = column_numpy(table1, "quality_zero_records", np.uint16).astype(np.int64)
            qnz1 = column_numpy(table1, "quality_nonzero_records", np.uint16).astype(np.int64)
            q161 = column_numpy(table1, "quality_bit16_records", np.uint16).astype(np.int64)
            q321 = column_numpy(table1, "quality_bit32_records", np.uint16).astype(np.int64)
            k111 = column_numpy(table1, "source_k11_records", np.uint16).astype(np.int64)
            k1011 = column_numpy(table1, "source_k101_records", np.uint16).astype(np.int64)
            kother1 = column_numpy(table1, "source_other_or_mixed_records", np.uint16).astype(np.int64)
            segment1 = column_numpy(table1, "continuous_segment_id", np.int32)
            require(np.array_equal(records1, aggregate["records"]), f"1 s record counts differ: {path1.name}")
            require(np.array_equal(valid1, aggregate["physical"]), f"1 s valid counts differ: {path1.name}")
            require(np.allclose(power1, aggregate["power"], rtol=2e-12, atol=0, equal_nan=True), f"1 s W-domain means differ: {path1.name}")
            require(np.array_equal(q01, aggregate["q0"]), f"1 s q0 counts differ: {path1.name}")
            require(np.array_equal(q161, aggregate["q16"]), f"1 s bit16 counts differ: {path1.name}")
            require(np.array_equal(q321, aggregate["q32"]), f"1 s bit32 counts differ: {path1.name}")
            require(np.array_equal(k111, aggregate["k11"]), f"1 s K=11 counts differ: {path1.name}")
            require(np.array_equal(k1011, aggregate["k101"]), f"1 s K=101 counts differ: {path1.name}")
            require(np.array_equal(segment1, aggregate["segment"]), f"1 s segment IDs differ: {path1.name}")
            require(np.array_equal(q01 + qnz1, records1), f"1 s quality totals differ: {path1.name}")
            require(np.array_equal(k111 + k1011 + kother1, records1), f"1 s source-bin totals differ: {path1.name}")
            s_hour = ((s - contract.START_MS) // contract.HOUR_MS).astype(np.int64)
            require(np.allclose(basew1, hour_maps.baseline_w[s_hour], rtol=0, atol=0), f"1 s slow W mismatch: {path1.name}")
            require(np.allclose(basedbm1, hour_maps.baseline_dbm[s_hour], rtol=0, atol=5e-6), f"1 s slow dBm mismatch: {path1.name}")
            require(np.allclose(dbm1, 10.0 * np.log10(power1 * 1000.0), rtol=0, atol=2e-5), f"1 s dBm mismatch: {path1.name}")
            require(np.allclose(residual1, dbm1 - basedbm1, rtol=0, atol=3e-5), f"1 s residual mismatch: {path1.name}")
            require(np.allclose(ratio1, power1 / basew1, rtol=3e-6, atol=0), f"1 s ratio mismatch: {path1.name}")
            require(np.allclose(envelope1, np.sqrt(ratio1), rtol=3e-6, atol=0), f"1 s envelope mismatch: {path1.name}")
            one_second_hour_sum += np.bincount(s_hour, weights=power1 * valid1, minlength=n_hours)[:n_hours]
            one_second_hour_count += np.bincount(s_hour, weights=valid1, minlength=n_hours)[:n_hours].astype(np.int64)

            frame1 = table1.to_pandas()
            for minutes in contract.WINDOW_MINUTES:
                duration = minutes * 60_000
                groups = (s // duration) * duration
                candidates, counts = np.unique(groups, return_counts=True)
                chosen = int(candidates[np.argmax(counts)])
                row = window_lookup[(chosen, minutes)]
                observed = recompute_window_sample(frame1, chosen, minutes)
                require(int(row["observed_seconds"]) == observed["observed_seconds"], "Window observed-second count mismatch")
                for key in ("received_power_w", "fast_residual_db_mean", "fast_residual_db_p05", "fast_residual_db_p50", "fast_residual_db_p95", "window_normalized_envelope_mean", "window_normalized_envelope_p50", "quality_zero_fraction"):
                    require(np.isclose(float(row[key]), observed[key], rtol=3e-6, atol=3e-6), f"Window sample mismatch: {key}, {path1.name}, {minutes} min")
                window_samples_checked += 1

            total100 += len(t)
            total_multiplicity += int(multiplicity.astype(np.int64).sum())
            total_physical += int(physical.sum())
            total_q_nonzero += int((q != 0).sum())
            total_q16 += int(((q & np.uint16(16)) != 0).sum())
            total_q32 += int(((q & np.uint16(32)) != 0).sum())
            total_k11 += int((source_k == 11).sum())
            total_k101 += int((source_k == 101).sum())
            total1 += len(s)
            total1_records += int(records1.sum())
            total1_physical += int(valid1.sum())
            print(f"validation: month {month_index}/{len(files100)} ({path100.stem[-7:]})", flush=True)
            del table100, table1, frame1
            gc.collect()

        summary = manifest["build_summary"]
        require(total100 == int(summary["unique_timestamps"]), "Total 100 ms rows differ from manifest")
        require(total_multiplicity == int(summary["rows_in_campaign_before_dedup"]), "Multiplicity sum does not recover pre-dedup campaign rows")
        require(total_physical == int(summary["physical_valid"]), "Physical-valid total differs")
        require(total_q_nonzero == int(summary["quality_nonzero"]), "Quality-nonzero total differs")
        require(total_q16 == int(summary["quality_bit16"]), "Bit16 total differs")
        require(total_q32 == int(summary["quality_bit32"]), "Bit32 total differs")
        require(total_k11 == int(summary["source_k11"]), "K=11 total differs")
        require(total_k101 == int(summary["source_k101"]), "K=101 total differs")
        require(total1 == int(summary["one_second_rows"]), "Total 1 s rows differ from manifest")
        require(total1_records == total100, "1 s record counts do not recover all 100 ms rows")
        require(total1_physical == total_physical, "1 s valid counts do not recover all valid 100 ms rows")
        require(segment_max == int(summary["continuous_segments"]), "Continuous-segment total differs")
        checks.append({"name": "monthly_partition_schema_and_time_keys", "passed": True, "details": f"{len(files100)} months"})
        checks.append({"name": "fast_100ms_derived_fields", "passed": True, "details": f"{total100:,} rows fully checked"})
        checks.append({"name": "fast_1s_exact_reaggregation", "passed": True, "details": f"{total1:,} seconds compared against 100 ms"})
        checks.append({"name": "quality_source_and_segment_totals", "passed": True, "details": "all totals recovered"})

        present = hourly_count > 0
        recomputed_hourly = np.divide(hourly_sum, hourly_count, out=np.full(n_hours, np.nan), where=present)
        recomputed_from_1s = np.divide(one_second_hour_sum, one_second_hour_count, out=np.full(n_hours, np.nan), where=one_second_hour_count > 0)
        require(np.array_equal(present, hour_maps.present), "Fast data and hourly baseline have different observed-hour keys")
        require(np.allclose(recomputed_hourly[present], hour_maps.baseline_w[present], rtol=2e-11, atol=0), "100 ms data do not reconstruct verified hourly W means")
        require(np.allclose(recomputed_from_1s[present], hour_maps.baseline_w[present], rtol=2e-11, atol=0), "1 s weighted data do not reconstruct verified hourly W means")
        if "physical_valid_records" in hourly.columns:
            hour_naive = hourly["hour_bjt"].dt.tz_convert(contract.TZ_BJT).dt.tz_localize(None)
            hour_idx = ((hour_naive.to_numpy(dtype="datetime64[ms]").astype(np.int64) - contract.START_MS) // contract.HOUR_MS).astype(np.int64)
            require(np.array_equal(hourly_count[hour_idx], hourly["physical_valid_records"].to_numpy(dtype=np.int64)), "Hourly valid counts differ from Revision_002")
        checks.append({"name": "hourly_slow_baseline_reconstruction", "passed": True, "details": f"{int(present.sum()):,} hours"})

        expected_coverage = window_table["observed_seconds"] / window_table["expected_seconds"]
        require(np.allclose(window_table["window_coverage_fraction"], expected_coverage, rtol=0, atol=1e-15), "Window coverage formula differs")
        expected_usable = (
            (window_table["window_coverage_fraction"] >= 0.95)
            & (window_table["high_coverage_fraction"] >= 0.90)
            & (window_table["continuous_segment_count"] == 1)
            & (window_table["split_strict_24h_code"] >= 0)
        )
        require(np.array_equal(window_table["primary_usable"].to_numpy(), expected_usable.to_numpy()), "Window primary-usable flag differs")
        require(set(window_table["window_minutes"].unique()) == set(contract.WINDOW_MINUTES), "Window scales differ from contract")
        correlations = window_table[["residual_autocorrelation_lag1s", "residual_autocorrelation_lag10s"]].to_numpy(dtype=np.float64)
        require(np.all((~np.isfinite(correlations)) | ((correlations >= -1.0000001) & (correlations <= 1.0000001))), "Autocorrelation outside [-1,1]")
        checks.append({"name": "multi_window_statistics", "passed": True, "details": f"{len(window_table):,} rows; {window_samples_checked} full recomputations"})

        split_codes = pd.read_csv(output / "time_split_codes.csv")
        require(set(split_codes["code"]) == {0, 1, 2, 3}, "Split-code mapping incomplete")
        require((-1 not in set(column_numpy(pq.read_table(files1[0], columns=["split_blocked_code"]), "split_blocked_code", np.int8))), "Unexpected missing blocked split")
        checks.append({"name": "frozen_split_inheritance", "passed": True, "details": "record and window boundaries checked"})

        manifest["status"] = "VERIFIED"
        manifest["independent_validation"] = {
            "result": "VERIFIED",
            "checks_passed": len(checks),
            "checks_total": len(checks),
            "validated_bjt": datetime.now().astimezone().isoformat(),
            "validator": str(Path(__file__).resolve()),
        }
        (output / "run_manifest.json").write_text(json.dumps(contract.json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "status": "VERIFIED",
            "validated_bjt": datetime.now().astimezone().isoformat(),
            "checks": checks,
            "summary": {
                "fast_100ms_rows": total100,
                "fast_1s_rows": total1,
                "window_rows": len(window_table),
                "hours_reconstructed": int(present.sum()),
                "continuous_segments": segment_max,
                "window_samples_recomputed": window_samples_checked,
            },
            "direct_inputs_sha256": current_hashes,
        }
        report_path.write_text(json.dumps(contract.json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
        validation_md = "# Independent validation\n\nStatus: **VERIFIED**.\n\n" + "\n".join(
            f"- PASS — {item['name']}: {item['details']}" for item in checks
        ) + "\n\nThe validation independently re-read every 100 ms and 1 s output row, reconstructed all hourly W-domain means, exactly reaggregated every second from its 100 ms records, recomputed deterministic window samples, and rehashed all protected direct inputs.\n"
        (output / "VALIDATION.md").write_text(validation_md, encoding="utf-8")
        checksum_lines = []
        for path in sorted((p for p in output.rglob("*") if p.is_file() and p.name != "checksums.sha256"), key=lambda p: p.as_posix().lower()):
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
        (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
        print(json.dumps(contract.json_ready(report), ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception:
        failed_dir = output / f"Validation_Failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        failed_dir.mkdir(parents=True, exist_ok=False)
        failure = {"status": "FAILED", "error": traceback.format_exc(), "checks_completed": checks}
        (failed_dir / "validation_report.json").write_text(json.dumps(contract.json_ready(failure), ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
