"""Recompute latest hourly targets from calibrated, quality-zero 100 ms data.

This is an alignment/quality gate only. It does not fit a rapid-fading model.
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
FAST = ROOT / "Outputs/Datasets/Fast_Variation_Revision_002"
HOURLY = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001"
EXPECTED_SOURCE_ROWS = 129_822_732
EXPECTED_HOURS = 3_869
EXPECTED_DETECTED_HOURS = 3_434
MAX_ABS_ATTENUATION_DIFF_DB = 0.001


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((HOURLY / "manifest.json").read_text(encoding="utf-8"))
    reference_dbm = float(manifest["clear_reference_dbm"])
    status = pd.read_parquet(HOURLY / "hourly_sample_status.parquet")
    status = status.sort_values("hour_bjt").reset_index(drop=True)
    if len(status) != EXPECTED_HOURS or not status.hour_bjt.is_unique:
        raise RuntimeError("The official hourly status table changed")

    labels = pd.concat(
        [pd.read_parquet(HOURLY / f"{split}.parquet",
                         columns=["hour_bjt", "clear_sky_attenuation_db"])
         for split in ("train", "validation", "test")],
        ignore_index=True,
    )
    if len(labels) != EXPECTED_DETECTED_HOURS or not labels.hour_bjt.is_unique:
        raise RuntimeError("The official numerical target table changed")
    status = status.merge(labels, on="hour_bjt", how="left", validate="one_to_one")
    eligible = status.use_for_regression.to_numpy(dtype=bool)
    if eligible.sum() != EXPECTED_DETECTED_HOURS:
        raise RuntimeError("The official detected-hour count changed")

    # The official table stores timezone-aware Beijing hours, whereas the older
    # fast Parquet stores local Beijing wall-clock timestamps without tz metadata.
    # Remove tz metadata without shifting the clock: converting to UTC would
    # silently pair every 100 ms observation with an hour eight hours earlier.
    hours = status.hour_bjt.dt.tz_localize(None).to_numpy(
        dtype="datetime64[ns]"
    ).view("int64")
    counts = np.zeros(len(status), dtype=np.int64)
    sums_w = np.zeros(len(status), dtype=np.float64)
    k11 = np.zeros(len(status), dtype=np.int64)
    k101 = np.zeros(len(status), dtype=np.int64)
    source_rows = 0
    quality_zero_rows = 0
    bad_quality_zero_power = 0
    unmatched_quality_zero_rows = 0
    partitions = sorted((FAST / "fast_100ms").glob("fast_100ms_*.parquet"))
    if not partitions:
        raise RuntimeError("No calibrated 100 ms partitions found")

    for path in partitions:
        parquet = pq.ParquetFile(path)
        part_rows = 0
        for batch in parquet.iter_batches(
            batch_size=250_000,
            columns=["hour_bjt", "peak_power_w", "quality_code",
                     "physical_valid", "source_bin_count"],
        ):
            n = batch.num_rows
            part_rows += n
            source_rows += n
            hour_ns = batch.column(0).to_numpy(zero_copy_only=False).astype(
                "datetime64[ns]"
            ).view("int64")
            power_w = batch.column(1).to_numpy(zero_copy_only=False)
            quality = batch.column(2).to_numpy(zero_copy_only=False)
            physical = batch.column(3).to_numpy(zero_copy_only=False)
            source_bins = batch.column(4).to_numpy(zero_copy_only=False)
            quality_zero = quality == 0
            quality_zero_rows += int(quality_zero.sum())
            idx = np.searchsorted(hours, hour_ns)
            matched = idx < len(hours)
            matched[matched] &= hours[idx[matched]] == hour_ns[matched]
            unmatched_quality_zero_rows += int((quality_zero & ~matched).sum())
            usable = quality_zero & matched
            bad_power = usable & (~physical | ~np.isfinite(power_w) | (power_w <= 0))
            bad_quality_zero_power += int(bad_power.sum())
            usable &= ~bad_power
            selected = idx[usable]
            counts += np.bincount(selected, minlength=len(hours))
            sums_w += np.bincount(selected, weights=power_w[usable],
                                  minlength=len(hours))
            k11 += np.bincount(selected[source_bins[usable] == 11],
                               minlength=len(hours))
            k101 += np.bincount(selected[source_bins[usable] == 101],
                                minlength=len(hours))
        print(f"Scanned {path.name}: {part_rows:,} rows", flush=True)

    mean_w = np.divide(sums_w, counts, out=np.full(len(status), np.nan),
                       where=counts > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recomputed_power_dbm = 10.0 * np.log10(mean_w * 1000.0)
    recomputed_attenuation_db = reference_dbm - recomputed_power_dbm
    official_attenuation = status.clear_sky_attenuation_db.to_numpy(dtype=float)
    differences = recomputed_attenuation_db[eligible] - official_attenuation[eligible]
    count_errors = counts - status.quality_zero_records.to_numpy(dtype=np.int64)
    finite_differences = differences[np.isfinite(differences)]
    max_abs_diff = (float(np.max(np.abs(finite_differences)))
                    if len(finite_differences) else None)
    max_abs_count_error = int(np.max(np.abs(count_errors)))
    detected_all_finite = bool(np.isfinite(differences).all())
    censored_without_label = bool(np.isnan(official_attenuation[~eligible]).all())
    qa_pass = (
        source_rows == EXPECTED_SOURCE_ROWS
        and len(status) == EXPECTED_HOURS
        and int(eligible.sum()) == EXPECTED_DETECTED_HOURS
        and max_abs_count_error == 0
        and bad_quality_zero_power == 0
        and unmatched_quality_zero_rows == 0
        and detected_all_finite
        and max_abs_diff is not None
        and max_abs_diff <= MAX_ABS_ATTENUATION_DIFF_DB
        and censored_without_label
    )

    by_hour = status[["hour_bjt", "split", "weather_stratum", "rain",
                      "heavy_rain", "dry_overcast", "clear_dry", "snow",
                      "use_for_regression", "quality_zero_records",
                      "clear_sky_attenuation_db"]].copy()
    by_hour["recomputed_quality_zero_records"] = counts
    by_hour["recomputed_mean_power_w_diagnostic"] = mean_w
    by_hour["recomputed_attenuation_db_detected_only"] = np.where(
        eligible, recomputed_attenuation_db, np.nan
    )
    by_hour["attenuation_difference_db_detected_only"] = np.where(
        eligible, recomputed_attenuation_db - official_attenuation, np.nan
    )
    by_hour["quality_zero_k11_records"] = k11
    by_hour["quality_zero_k101_records"] = k101
    result = {
        "status": "PASS" if qa_pass else "FAIL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_100ms_partitions": len(partitions),
        "source_100ms_rows": source_rows,
        "quality_zero_source_rows": quality_zero_rows,
        "official_hours": len(status),
        "detected_hours": int(eligible.sum()),
        "censored_hours": int((~eligible).sum()),
        "bad_quality_zero_power_rows": bad_quality_zero_power,
        "unmatched_quality_zero_rows": unmatched_quality_zero_rows,
        "max_abs_quality_zero_count_error": max_abs_count_error,
        "max_abs_attenuation_difference_db": max_abs_diff,
        "nonfinite_detected_hour_differences": int((~np.isfinite(differences)).sum()),
        "attenuation_tolerance_db": MAX_ABS_ATTENUATION_DIFF_DB,
        "clear_reference_dbm": reference_dbm,
        "censored_hours_remain_unlabeled": censored_without_label,
        "source_manifest_sha256": file_sha256(HOURLY / "manifest.json"),
        "fast_checksums_sha256": file_sha256(FAST / "checksums.sha256"),
        "script_sha256": file_sha256(Path(__file__)),
        "note": "This is a source alignment gate, not a fitted fading distribution.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    by_hour.to_csv(OUT / "alignment_by_hour.csv", index=False)
    (OUT / "alignment_qa.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not qa_pass:
        raise SystemExit("Alignment QA failed; do not fit fast-variation distributions")


if __name__ == "__main__":
    main()
