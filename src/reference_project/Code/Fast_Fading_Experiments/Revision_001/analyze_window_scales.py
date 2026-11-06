"""Train-only descriptive check of within-window power-mean drift.

This intentionally does not select a stationary window or fit a distribution.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001"
WINDOW_MINUTES = (1, 2, 5, 10, 15, 30, 60)
MIN_HALF_COVERAGE = 0.90
MAX_GAP_SECONDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_month(path: Path, minutes: int) -> pd.DataFrame:
    data = pd.read_parquet(
        path,
        columns=["second_bjt", "hour_bjt", "quality_zero_records",
                 "received_power_w", "primary_usable_1s", "split",
                 "rain", "dry_overcast", "clear_dry"],
    )
    data = data.loc[data.split.eq("train") & data.primary_usable_1s].copy()
    if data.empty:
        return pd.DataFrame()
    seconds_in_hour = data.second_bjt.dt.minute.to_numpy() * 60 + data.second_bjt.dt.second.to_numpy()
    window_seconds = minutes * 60
    data["slot"] = seconds_in_hour // window_seconds
    data["half"] = (seconds_in_hour % window_seconds) >= (window_seconds // 2)
    data["power_weighted_w"] = data.received_power_w * data.quality_zero_records
    time_s = data.second_bjt.astype("int64") // 1_000_000_000
    data["time_s"] = time_s
    data["gap_s"] = data.groupby(["hour_bjt", "slot"], sort=False).time_s.diff().fillna(1)
    halves = data.groupby(["hour_bjt", "slot", "half"], sort=True).agg(
        observed_seconds=("time_s", "size"),
        power_weighted_sum_w=("power_weighted_w", "sum"),
        quality_zero_records=("quality_zero_records", "sum"),
        max_gap_s=("gap_s", "max"),
        rain=("rain", "first"),
        dry_overcast=("dry_overcast", "first"),
        clear_dry=("clear_dry", "first"),
    ).reset_index()
    first = halves.loc[~halves.half].drop(columns="half").set_index(["hour_bjt", "slot"])
    second = halves.loc[halves.half].drop(columns="half").set_index(["hour_bjt", "slot"])
    paired = first.join(second, how="inner", lsuffix="_first", rsuffix="_second").reset_index()
    half_seconds = window_seconds // 2
    good = (
        (paired.observed_seconds_first >= np.ceil(MIN_HALF_COVERAGE * half_seconds))
        & (paired.observed_seconds_second >= np.ceil(MIN_HALF_COVERAGE * half_seconds))
        & (paired.max_gap_s_first <= MAX_GAP_SECONDS)
        & (paired.max_gap_s_second <= MAX_GAP_SECONDS)
    )
    paired = paired.loc[good].copy()
    if paired.empty:
        return pd.DataFrame()
    mean_first = paired.power_weighted_sum_w_first / paired.quality_zero_records_first
    mean_second = paired.power_weighted_sum_w_second / paired.quality_zero_records_second
    paired["half_power_mean_difference_abs_db"] = np.abs(10.0 * np.log10(mean_first / mean_second))
    paired["window_minutes"] = minutes
    paired["window_start_bjt"] = paired.hour_bjt + pd.to_timedelta(
        paired.slot * window_seconds, unit="s"
    )
    return paired[["window_start_bjt", "hour_bjt", "window_minutes",
                   "observed_seconds_first", "observed_seconds_second",
                   "half_power_mean_difference_abs_db", "rain_first",
                   "dry_overcast_first", "clear_dry_first"]].rename(columns={
        "rain_first": "rain",
        "dry_overcast_first": "dry_overcast",
        "clear_dry_first": "clear_dry",
    })


def main() -> None:
    source_qa = json.loads((OUT / "clean_1s_qa.json").read_text(encoding="utf-8"))
    if source_qa["status"] != "PASS":
        raise RuntimeError("The clean 1 s source has not passed QA")
    files = sorted((OUT / "clean_1s").glob("clean_1s_*.parquet"))
    if set(path.name for path in files) != set(source_qa["monthly_parquet_sha256"]):
        raise RuntimeError("Clean 1 s partition set changed")
    if any(sha256(path) != source_qa["monthly_parquet_sha256"][path.name]
           for path in files):
        raise RuntimeError("Clean 1 s partition fingerprint changed")

    pieces = []
    for path in files:
        for minutes in WINDOW_MINUTES:
            piece = summarize_month(path, minutes)
            if not piece.empty:
                pieces.append(piece)
        print(f"Analyzed {path.name}", flush=True)
    if not pieces:
        raise RuntimeError("No eligible training window")
    windows = pd.concat(pieces, ignore_index=True).sort_values(
        ["window_minutes", "window_start_bjt"]
    )
    if not np.isfinite(windows.half_power_mean_difference_abs_db).all():
        raise RuntimeError("Nonfinite half-window difference")
    windows.to_parquet(OUT / "train_window_half_power_drift.parquet",
                       index=False, compression="zstd")

    summary = []
    groups = {
        "clear_dry": windows.clear_dry,
        "dry_overcast": windows.dry_overcast,
        "rain_including_heavy": windows.rain,
        "all_train": np.ones(len(windows), dtype=bool),
    }
    for minutes in WINDOW_MINUTES:
        current = windows.loc[windows.window_minutes.eq(minutes)]
        for event in groups:
            if event == "all_train":
                subset = current
            else:
                subset = current.loc[current[event.split("_including_")[0]]]
            by_hour = subset.groupby("hour_bjt").half_power_mean_difference_abs_db.median()
            summary.append({
                "window_minutes": minutes,
                "event": event,
                "n_windows": len(subset),
                "n_hours": len(by_hour),
                "median_hourly_median_half_drift_db": float(by_hour.median()) if len(by_hour) else None,
                "p90_hourly_median_half_drift_db": float(by_hour.quantile(0.90)) if len(by_hour) else None,
            })
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUT / "train_window_scale_summary.csv", index=False)
    # The coverage/gap gate admits different hour sets at different scales.
    # A paired comparison must use only hours represented at every scale.
    hour_medians = windows.groupby(
        ["hour_bjt", "window_minutes"]
    ).half_power_mean_difference_abs_db.median().unstack()
    common_hours = hour_medians.dropna().copy()
    hour_events = windows.groupby("hour_bjt")[["rain", "dry_overcast", "clear_dry"]].first()
    paired = common_hours.join(hour_events, how="left")
    paired.to_csv(OUT / "train_window_paired_hours.csv")
    paired_rows = []
    for minutes in WINDOW_MINUTES:
        for event, mask in {
            "clear_dry": paired.clear_dry,
            "dry_overcast": paired.dry_overcast,
            "rain_including_heavy": paired.rain,
            "all_train": np.ones(len(paired), dtype=bool),
        }.items():
            values = paired.loc[mask, minutes]
            paired_rows.append({
                "window_minutes": minutes,
                "event": event,
                "same_hours_at_all_scales": len(values),
                "median_hourly_half_drift_db": float(values.median()) if len(values) else None,
                "p90_hourly_half_drift_db": float(values.quantile(0.90)) if len(values) else None,
            })
    pd.DataFrame(paired_rows).to_csv(OUT / "train_window_paired_summary.csv", index=False)
    report = {
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "train-hours-only descriptive diagnostics",
        "window_minutes": list(WINDOW_MINUTES),
        "min_half_coverage": MIN_HALF_COVERAGE,
        "max_gap_seconds": MAX_GAP_SECONDS,
        "eligible_window_records": len(windows),
        "hours_with_all_window_scales": len(common_hours),
        "clean_1s_qa_sha256": sha256(OUT / "clean_1s_qa.json"),
        "script_sha256": sha256(Path(__file__)),
        "note": "No stationarity interval or distribution was selected from these descriptive scores.",
    }
    (OUT / "window_scale_qa.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
