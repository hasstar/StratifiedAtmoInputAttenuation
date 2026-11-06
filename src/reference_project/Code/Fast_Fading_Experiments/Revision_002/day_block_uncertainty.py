"""Paired calendar-day bootstrap of verified train-only window diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001"
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_002"
WINDOWS = (1, 2, 5, 10, 15, 30, 60)
EVENTS = ("all_train", "clear_dry", "dry_overcast", "rain")
CONTRASTS = ((15, 10), (30, 15), (60, 15))
REPLICATES = 2000
SEED = 20261006


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inputs() -> tuple[pd.DataFrame, dict]:
    qa = json.loads((SOURCE / "window_scale_qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS":
        raise RuntimeError("Revision 001 window QA is not PASS")
    frame = pd.read_csv(SOURCE / "train_window_paired_hours.csv")
    frame["hour_bjt"] = pd.to_datetime(frame.hour_bjt)
    frame["date_bjt"] = frame.hour_bjt.dt.strftime("%Y-%m-%d")
    if not frame.hour_bjt.is_unique or len(frame) != qa["hours_with_all_window_scales"]:
        raise RuntimeError("Paired-hour mapping changed")
    values = frame[[str(w) for w in WINDOWS]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("Invalid window diagnostic")
    # Independently establish that the paired table contains official train hours.
    official = pd.read_csv(SOURCE / "clean_1s_hourly_summary.csv")
    train = set(pd.to_datetime(official.loc[official.split.eq("train"), "hour_bjt"]))
    if not set(frame.hour_bjt).issubset(train):
        raise RuntimeError("A paired hour is outside the training split")
    return frame, {
        "paired_hours_sha256": sha256(SOURCE / "train_window_paired_hours.csv"),
        "window_scale_qa_sha256": sha256(SOURCE / "window_scale_qa.json"),
        "hour_summary_sha256": sha256(SOURCE / "clean_1s_hourly_summary.csv"),
        "script_sha256": sha256(Path(__file__)),
    }


def run() -> None:
    frame, fingerprints = inputs()
    prior = OUT / "day_block_qa.json"
    if prior.exists():
        old = json.loads(prior.read_text(encoding="utf-8"))
        if old.get("status") == "PASS" and old["fingerprints"] == fingerprints:
            verify()
            print("Stage A already complete; outputs verified", flush=True)
            return
        raise RuntimeError("Existing stage A has a different fingerprint or failed QA; diagnose before rerun")
    summary, contrast, draws = [], [], []
    for event_i, event in enumerate(EVENTS):
        group = frame if event == "all_train" else frame.loc[frame[event]].copy()
        values = group[[str(w) for w in WINDOWS]].to_numpy(dtype=float)
        days = sorted(group.date_bjt.unique())
        day_rows = [np.flatnonzero(group.date_bjt.to_numpy() == day) for day in days]
        rng = np.random.default_rng(SEED + event_i)
        medians = np.empty((REPLICATES, len(WINDOWS)))
        p90s = np.empty_like(medians)
        for b in range(REPLICATES):
            chosen = rng.integers(0, len(days), size=len(days))
            idx = np.concatenate([day_rows[i] for i in chosen])
            current = values[idx]
            medians[b] = np.median(current, axis=0)
            p90s[b] = np.quantile(current, 0.90, axis=0)
        for j, minutes in enumerate(WINDOWS):
            summary.append({
                "event": event, "window_minutes": minutes,
                "n_hours": len(group), "n_days": len(days),
                "median_db": float(np.median(values[:, j])),
                "median_ci95_low_db": float(np.quantile(medians[:, j], 0.025)),
                "median_ci95_high_db": float(np.quantile(medians[:, j], 0.975)),
                "p90_db": float(np.quantile(values[:, j], 0.90)),
                "p90_ci95_low_db": float(np.quantile(p90s[:, j], 0.025)),
                "p90_ci95_high_db": float(np.quantile(p90s[:, j], 0.975)),
            })
            draws.append(pd.DataFrame({
                "event": event, "window_minutes": minutes,
                "replicate": np.arange(REPLICATES),
                "median_db": medians[:, j], "p90_db": p90s[:, j],
            }))
        for bigger, smaller in CONTRASTS:
            j, k = WINDOWS.index(bigger), WINDOWS.index(smaller)
            difference = p90s[:, j] - p90s[:, k]
            contrast.append({
                "event": event, "contrast_minutes": f"{bigger}-{smaller}",
                "n_hours": len(group), "n_days": len(days),
                "p90_difference_db": float(np.quantile(values[:, j], 0.90) - np.quantile(values[:, k], 0.90)),
                "ci95_low_db": float(np.quantile(difference, 0.025)),
                "ci95_high_db": float(np.quantile(difference, 0.975)),
            })
        print(f"{event}: {len(group)} paired hours, {len(days)} days, {REPLICATES} date-block draws", flush=True)
    pd.DataFrame(summary).to_csv(OUT / "window_uncertainty_summary.csv", index=False)
    pd.DataFrame(contrast).to_csv(OUT / "window_contrast_summary.csv", index=False)
    pd.concat(draws, ignore_index=True).to_parquet(OUT / "day_bootstrap_draws.parquet", index=False, compression="zstd")
    result = {
        "status": "PASS", "stage": "A_day_block_uncertainty",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "n_paired_hours": len(frame), "n_days": int(frame.date_bjt.nunique()),
        "replicates_per_event": REPLICATES, "seed": SEED,
        "summary_rows": len(summary), "contrast_rows": len(contrast),
        "bootstrap_rows": REPLICATES * len(EVENTS) * len(WINDOWS),
        "fingerprints": fingerprints,
        "output_sha256": {p.name: sha256(p) for p in [OUT / "window_uncertainty_summary.csv", OUT / "window_contrast_summary.csv", OUT / "day_bootstrap_draws.parquet"]},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "interpretation": "Descriptive day-block uncertainty; no stationary window or generator selected.",
    }
    write_json(prior, result)
    verify()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def verify() -> None:
    frame, fingerprints = inputs()
    qa = json.loads((OUT / "day_block_qa.json").read_text(encoding="utf-8"))
    if qa["fingerprints"] != fingerprints or any(sha256(OUT / name) != digest for name, digest in qa["output_sha256"].items()):
        raise RuntimeError("Stage A fingerprint mismatch")
    summary = pd.read_csv(OUT / "window_uncertainty_summary.csv")
    contrast = pd.read_csv(OUT / "window_contrast_summary.csv")
    draws = pd.read_parquet(OUT / "day_bootstrap_draws.parquet")
    if (len(summary), len(contrast), len(draws)) != (28, 12, 56000):
        raise RuntimeError("Stage A record count mismatch")
    errors = []
    for row in summary.itertuples(index=False):
        group = frame if row.event == "all_train" else frame.loc[frame[row.event]]
        sample = group[str(row.window_minutes)].to_numpy(dtype=float)
        boot = draws.loc[draws.event.eq(row.event) & draws.window_minutes.eq(row.window_minutes)].sort_values("replicate")
        if len(boot) != REPLICATES or not np.array_equal(boot.replicate, np.arange(REPLICATES)):
            raise RuntimeError("Missing bootstrap replicate")
        pairs = ((row.median_db, np.median(sample)), (row.p90_db, np.quantile(sample, .90)),
                 (row.median_ci95_low_db, boot.median_db.quantile(.025)),
                 (row.median_ci95_high_db, boot.median_db.quantile(.975)),
                 (row.p90_ci95_low_db, boot.p90_db.quantile(.025)),
                 (row.p90_ci95_high_db, boot.p90_db.quantile(.975)))
        errors.extend(abs(float(a) - float(b)) for a, b in pairs)
        if row.n_hours != len(group) or row.n_days != group.date_bjt.nunique():
            raise RuntimeError("Stage A sample unit mismatch")
    for row in contrast.itertuples(index=False):
        big, small = map(int, row.contrast_minutes.split("-"))
        boot = draws.loc[draws.event.eq(row.event)].pivot(index="replicate", columns="window_minutes", values="p90_db")
        diff = boot[big] - boot[small]
        group = frame if row.event == "all_train" else frame.loc[frame[row.event]]
        errors.extend((abs(row.ci95_low_db - diff.quantile(.025)), abs(row.ci95_high_db - diff.quantile(.975)),
                       abs(row.p90_difference_db - (group[str(big)].quantile(.90) - group[str(small)].quantile(.90)))))
    if not np.isfinite(errors).all() or max(errors) > 1e-12:
        raise RuntimeError("Stage A metrics do not recompute")
    write_json(OUT / "day_block_verification.json", {
        "status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "recomputed_numeric_entries": len(errors), "max_absolute_difference": float(max(errors)),
        "source_hours": len(frame), "source_days": int(frame.date_bjt.nunique()),
        "day_block_qa_sha256": sha256(OUT / "day_block_qa.json"),
    })
    print("Stage A independent recomputation PASS", flush=True)


def status() -> None:
    for name in ("day_block_qa.json", "day_block_verification.json"):
        p = OUT / name
        print(p.read_text(encoding="utf-8") if p.exists() else f"{name}: pending", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "status"))
    args = parser.parse_args()
    {"run": run, "verify": verify, "status": status}[args.command]()
