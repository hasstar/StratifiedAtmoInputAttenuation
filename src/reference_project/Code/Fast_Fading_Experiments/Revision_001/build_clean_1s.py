"""Reaggregate quality-zero 100 ms powers into a clean 1 s rapid series.

The old Fast_Variation_Revision_002 1 s table includes low-quality records and
is intentionally not used. This script only creates derived observations; it
does not select a stationary window or fit a distribution.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
FAST = ROOT / "Outputs/Datasets/Fast_Variation_Revision_002/fast_100ms"
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001"
ALIGN = OUT / "alignment_by_hour.csv"
ALIGN_QA = OUT / "alignment_qa.json"
MIN_RECORDS_PER_PRIMARY_SECOND = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_hours(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for hour, group in frame.groupby("hour_bjt", sort=True):
        good = group.loc[group.primary_usable_1s].sort_values("second_bjt")
        residual = good.fast_residual_db.to_numpy(dtype=float)
        time_s = good.second_bjt.astype("int64").to_numpy() // 1_000_000_000
        consecutive = np.diff(time_s) == 1
        if consecutive.sum() >= 3 and np.std(residual[:-1][consecutive]) > 0 and np.std(
            residual[1:][consecutive]
        ) > 0:
            lag1 = float(np.corrcoef(residual[:-1][consecutive],
                                     residual[1:][consecutive])[0, 1])
        else:
            lag1 = None
        rows.append({
            "hour_bjt": hour,
            "all_observed_seconds": len(group),
            "primary_seconds": len(good),
            "quality_zero_records": int(group.quality_zero_records.sum()),
            "primary_1s_coverage_fraction": len(good) / 3600,
            "fast_residual_std_db": float(np.std(residual, ddof=1)) if len(residual) > 1 else None,
            "fast_residual_p05_db": float(np.quantile(residual, 0.05)) if len(residual) else None,
            "fast_residual_p50_db": float(np.quantile(residual, 0.50)) if len(residual) else None,
            "fast_residual_p95_db": float(np.quantile(residual, 0.95)) if len(residual) else None,
            "fast_residual_lag1s_corr": lag1,
            "fade_plus_1db_fraction": float(np.mean(residual >= 1)) if len(residual) else None,
            "fade_plus_3db_fraction": float(np.mean(residual >= 3)) if len(residual) else None,
            "weather_stratum": group.weather_stratum.iloc[0],
            "split": group.split.iloc[0],
            "rain": bool(group.rain.iloc[0]),
            "dry_overcast": bool(group.dry_overcast.iloc[0]),
            "clear_dry": bool(group.clear_dry.iloc[0]),
        })
    return rows


def main() -> None:
    audit = json.loads(ALIGN_QA.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("Stage 0 alignment QA is not PASS")
    aligned = pd.read_csv(ALIGN, parse_dates=["hour_bjt"])
    eligible = aligned.loc[aligned.use_for_regression].copy()
    if len(eligible) != 3434:
        raise RuntimeError("Detected-hour pool changed")
    expected_month_names = {
        f"clean_1s_{month}.parquet"
        for month in eligible.hour_bjt.dt.strftime("%Y-%m").unique()
    }
    hour_ns = eligible.hour_bjt.dt.tz_localize(None).to_numpy(
        dtype="datetime64[ns]"
    ).view("int64")
    if not np.all(np.diff(hour_ns) > 0):
        raise RuntimeError("Hour mapping is not strictly increasing")
    expected_count = int(eligible.recomputed_quality_zero_records.sum())
    baseline_w = eligible.recomputed_mean_power_w_diagnostic.to_numpy(dtype=float)
    if not np.isfinite(baseline_w).all() or (baseline_w <= 0).any():
        raise RuntimeError("Nonpositive detected-hour baseline")

    destination = OUT / "clean_1s"
    destination.mkdir(parents=True, exist_ok=True)
    partitions = sorted(FAST.glob("fast_100ms_*.parquet"))
    summaries: list[dict] = []
    output_files = {}
    total_records = 0
    total_seconds = 0
    total_primary_seconds = 0
    for path in partitions:
        chunks = []
        source = pq.ParquetFile(path)
        for batch in source.iter_batches(
            batch_size=250_000,
            columns=["time_bjt", "hour_bjt", "peak_power_w", "quality_code",
                     "physical_valid", "source_bin_count"],
        ):
            time_ms = batch.column(0).to_numpy(zero_copy_only=False).astype(
                "datetime64[ms]"
            ).view("int64")
            observed_hour_ns = batch.column(1).to_numpy(
                zero_copy_only=False
            ).astype("datetime64[ns]").view("int64")
            power_w = batch.column(2).to_numpy(zero_copy_only=False)
            quality = batch.column(3).to_numpy(zero_copy_only=False)
            physical = batch.column(4).to_numpy(zero_copy_only=False)
            bins = batch.column(5).to_numpy(zero_copy_only=False)
            idx = np.searchsorted(hour_ns, observed_hour_ns)
            in_pool = idx < len(hour_ns)
            in_pool[in_pool] &= hour_ns[idx[in_pool]] == observed_hour_ns[in_pool]
            take = in_pool & (quality == 0) & physical & np.isfinite(power_w) & (power_w > 0)
            if not take.any():
                continue
            second_ns = (time_ms[take] // 1000) * 1_000_000_000
            hour_from_second = (second_ns // 3_600_000_000_000) * 3_600_000_000_000
            if not np.array_equal(hour_from_second, observed_hour_ns[take]):
                raise RuntimeError("A source 100 ms timestamp does not belong to its hour")
            frame = pd.DataFrame({
                "second_ns": second_ns,
                "power_w": power_w[take],
                "k11": (bins[take] == 11).astype(np.int8),
                "k101": (bins[take] == 101).astype(np.int8),
            })
            chunk = frame.groupby("second_ns", sort=True).agg(
                power_sum_w=("power_w", "sum"),
                quality_zero_records=("power_w", "size"),
                k11_records=("k11", "sum"),
                k101_records=("k101", "sum"),
            ).reset_index()
            chunks.append(chunk)
        if not chunks:
            continue
        month = pd.concat(chunks, ignore_index=True).groupby(
            "second_ns", sort=True, as_index=False
        )[["power_sum_w", "quality_zero_records", "k11_records", "k101_records"]].sum()
        sec_ns = month.second_ns.to_numpy(dtype=np.int64)
        corresponding_hour_ns = (sec_ns // 3_600_000_000_000) * 3_600_000_000_000
        idx = np.searchsorted(hour_ns, corresponding_hour_ns)
        if (idx >= len(hour_ns)).any() or not np.array_equal(hour_ns[idx], corresponding_hour_ns):
            raise RuntimeError("An aggregated second is outside the detected-hour pool")
        mean_w = month.power_sum_w.to_numpy(dtype=float) / month.quality_zero_records.to_numpy()
        base_w = baseline_w[idx]
        result = pd.DataFrame({
            "second_bjt": pd.to_datetime(sec_ns).tz_localize("Asia/Shanghai"),
            "hour_bjt": pd.to_datetime(corresponding_hour_ns).tz_localize("Asia/Shanghai"),
            "quality_zero_records": month.quality_zero_records.to_numpy(dtype=np.int32),
            "k11_records": month.k11_records.to_numpy(dtype=np.int32),
            "k101_records": month.k101_records.to_numpy(dtype=np.int32),
            "received_power_w": mean_w,
            "hour_mean_power_w": base_w,
            "fast_residual_db": 10.0 * np.log10(base_w / mean_w),
            "normalized_power_ratio": mean_w / base_w,
            "normalized_envelope": np.sqrt(mean_w / base_w),
            "primary_usable_1s": month.quality_zero_records.to_numpy() >= MIN_RECORDS_PER_PRIMARY_SECOND,
            "weather_stratum": eligible.weather_stratum.iloc[idx].to_numpy(),
            "split": eligible.split.iloc[idx].to_numpy(),
            "rain": eligible.rain.iloc[idx].to_numpy(dtype=bool),
            "dry_overcast": eligible.dry_overcast.iloc[idx].to_numpy(dtype=bool),
            "clear_dry": eligible.clear_dry.iloc[idx].to_numpy(dtype=bool),
        })
        name = path.name.replace("fast_100ms_", "clean_1s_")
        output = destination / name
        result.to_parquet(output, index=False, compression="zstd")
        output_files[name] = sha256(output)
        total_records += int(result.quality_zero_records.sum())
        total_seconds += len(result)
        total_primary_seconds += int(result.primary_usable_1s.sum())
        summaries.extend(summarize_hours(result))
        print(f"Built {name}: {len(result):,} seconds, {int(result.quality_zero_records.sum()):,} records", flush=True)

    by_hour = pd.DataFrame(summaries).sort_values("hour_bjt")
    hours_complete = bool(len(by_hour) == len(eligible) and by_hour.hour_bjt.is_unique)
    count_matches = bool(
        hours_complete
        and np.array_equal(
            by_hour.hour_bjt.dt.tz_localize(None).astype("int64").to_numpy(),
            eligible.hour_bjt.dt.tz_localize(None).astype("int64").to_numpy(),
        )
        and np.array_equal(
            by_hour.quality_zero_records.to_numpy(dtype=np.int64),
            eligible.recomputed_quality_zero_records.to_numpy(dtype=np.int64),
        )
    )
    finite = bool(total_seconds > 0 and all(np.isfinite(
        pd.read_parquet(destination / name,
                        columns=["fast_residual_db"]).fast_residual_db.to_numpy()
    ).all() for name in output_files))
    # 2025-09 contains raw samples but no hour passing the 95% quality-zero
    # rule; an empty derived partition must not be fabricated for that month.
    qa_pass = (hours_complete and count_matches and total_records == expected_count
               and finite and set(output_files) == expected_month_names)
    by_hour.to_csv(OUT / "clean_1s_hourly_summary.csv", index=False)
    report = {
        "status": "PASS" if qa_pass else "FAIL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "detected_hours": len(eligible),
        "clean_1s_hours": len(by_hour),
        "expected_quality_zero_100ms_records": expected_count,
        "reaggregated_quality_zero_100ms_records": total_records,
        "clean_1s_observed_seconds": total_seconds,
        "primary_seconds_at_least_9_records": total_primary_seconds,
        "source_100ms_partitions": len(partitions),
        "derived_nonempty_months": len(output_files),
        "hourly_counts_match": count_matches,
        "all_residuals_finite": finite,
        "alignment_qa_sha256": sha256(ALIGN_QA),
        "script_sha256": sha256(Path(__file__)),
        "monthly_parquet_sha256": output_files,
        "note": "No stationarity interval or conditional fading distribution selected yet.",
    }
    (OUT / "clean_1s_qa.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not qa_pass:
        raise SystemExit("Clean 1 s QA failed; do not fit distributions")


if __name__ == "__main__":
    main()
