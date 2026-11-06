"""Train-only moment, distribution-shape and quality sensitivity diagnostics.

The data-derived outputs describe windows; they do not select a stationary
interval or fit a weather-conditional generator. Monthly checkpoints allow
identical-command recovery without duplicating completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001"
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_002"
MONTHS = OUT / "window_months"
WINDOWS = (1, 2, 5, 10, 15, 30, 60)
GATES = {
    "primary90_gap5": {"min_records": 9, "half_coverage": .90, "max_missing_run": 5},
    "primary95_gap5": {"min_records": 9, "half_coverage": .95, "max_missing_run": 5},
    "complete_primary": {"min_records": 9, "half_coverage": 1., "max_missing_run": 0},
    "complete_10of10": {"min_records": 10, "half_coverage": 1., "max_missing_run": 0},
}
METRICS = ("half_power_drift_db", "half_sd_difference_db", "half_p05_difference_db",
           "half_p50_difference_db", "half_p95_difference_db", "half_lag1_difference",
           "local_shape_wasserstein_db")
PRIMARY_BOOTSTRAP_METRICS = ("half_power_drift_db", "half_sd_difference_db")
EVENTS = ("all_train", "clear_dry", "dry_overcast", "rain")
REPLICATES = 2000
SEED = 20261006


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def expected_manifest() -> dict:
    source_qa = json.loads((SOURCE / "clean_1s_qa.json").read_text(encoding="utf-8"))
    power_qa = json.loads((SOURCE / "power_normalization_qa.json").read_text(encoding="utf-8"))
    day_qa = json.loads((OUT / "day_block_verification.json").read_text(encoding="utf-8"))
    if any(x["status"] != "PASS" for x in (source_qa, power_qa, day_qa)):
        raise RuntimeError("An upstream QA is not PASS")
    return {
        "revision": "Fast_Fading_Revision_002", "stage": "B_train_window_diagnostics",
        "scope": "official training hours only", "windows_minutes": list(WINDOWS),
        "quality_gates": GATES, "metrics": list(METRICS),
        "source_partitions": source_qa["monthly_parquet_sha256"],
        "source_clean_qa_sha256": sha256(SOURCE / "clean_1s_qa.json"),
        "source_power_qa_sha256": sha256(SOURCE / "power_normalization_qa.json"),
        "hour_summary_sha256": sha256(SOURCE / "clean_1s_hourly_summary.csv"),
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "day_bootstrap_replicates": REPLICATES,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
    }


def manifest() -> dict:
    current = expected_manifest()
    p = OUT / "run_manifest.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old["frozen_design"] != current:
            raise RuntimeError("Frozen stage B fingerprint/design changed")
        return old
    created = {"created_utc": now(), "frozen_design": current}
    write_json(p, created)
    return created


@contextmanager
def worker_lock(command: str):
    # Windows file lock is released by the OS after a crash. Metadata remains
    # for diagnosis, but its mere existence is not treated as a live worker.
    p = OUT / "worker.lock"
    with p.open("a+b") as handle:
        if p.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {"pid": os.getpid(), "command": command, "started_utc": now(), "status": "running"}
        write_json(OUT / "active_worker.json", metadata)
        try:
            yield
        except BaseException as error:
            metadata.update(status="failed", ended_utc=now(), error=str(error))
            write_json(OUT / "active_worker.json", metadata)
            raise
        else:
            metadata.update(status="complete", ended_utc=now())
            write_json(OUT / "active_worker.json", metadata)
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def longest_missing(valid: np.ndarray) -> int:
    idx = np.flatnonzero(valid)
    return int(np.max(np.diff(np.r_[-1, idx, len(valid)]) - 1))


def lag1(value: np.ndarray, valid: np.ndarray) -> float:
    good = valid[:-1] & valid[1:]
    a, b = value[:-1][good], value[1:][good]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def hour_arrays(group: pd.DataFrame) -> dict:
    slot = (group.second_bjt - group.hour_bjt).dt.total_seconds().to_numpy(dtype=int)
    if (slot < 0).any() or (slot >= 3600).any() or len(np.unique(slot)) != len(slot):
        raise RuntimeError("A second is duplicated or outside its hour")
    p = np.full(3600, np.nan)
    count = np.zeros(3600, dtype=np.int16)
    k11 = np.zeros_like(count)
    k101 = np.zeros_like(count)
    p[slot] = group.received_power_w
    count[slot] = group.quality_zero_records
    k11[slot] = group.k11_records
    k101[slot] = group.k101_records
    if not group.hour_mean_power_w.nunique() == 1:
        raise RuntimeError("Hour baseline changed inside an hour")
    base = float(group.hour_mean_power_w.iloc[0])
    f = 10 * np.log10(base / p)
    if not np.allclose(f[slot], group.fast_residual_db, rtol=0, atol=1e-10):
        raise RuntimeError("Residual and power identity mismatch")
    for column in ("rain", "dry_overcast", "clear_dry", "weather_stratum"):
        if group[column].nunique() != 1:
            raise RuntimeError("Weather changed inside an hourly-labeled group")
    return {"power": p, "count": count, "f": f, "base": base, "k11": k11, "k101": k101}


def calculate_hour(group: pd.DataFrame) -> list[dict]:
    arrays = hour_arrays(group)
    hour = group.hour_bjt.iloc[0]
    date = hour.strftime("%Y-%m-%d")
    epoch = "before_device_change" if date < "2026-05-14" else ("transition_day" if date == "2026-05-14" else "after_device_change")
    rows = []
    for min_records in (9, 10):
        eligible_gates = {key: rule for key, rule in GATES.items() if rule["min_records"] == min_records}
        valid_hour = (arrays["count"] >= min_records) & np.isfinite(arrays["power"]) & (arrays["power"] > 0)
        for minutes in WINDOWS:
            width = minutes * 60
            half = width // 2
            for slot in range(3600 // width):
                start = slot * width
                p = arrays["power"][start:start + width]
                f = arrays["f"][start:start + width]
                counts = arrays["count"][start:start + width]
                valid = valid_hour[start:start + width]
                first, second = valid[:half], valid[half:]
                n1, n2 = int(first.sum()), int(second.sum())
                missing1, missing2 = longest_missing(first), longest_missing(second)
                chosen = [key for key, rule in eligible_gates.items()
                          if n1 >= int(np.ceil(rule["half_coverage"] * half))
                          and n2 >= int(np.ceil(rule["half_coverage"] * half))
                          and max(missing1, missing2) <= rule["max_missing_run"]]
                if not chosen:
                    continue
                v1, v2 = f[:half][first], f[half:][second]
                power1, power2 = p[:half][first], p[half:][second]
                weight1, weight2 = counts[:half][first], counts[half:][second]
                mean1 = float(np.average(power1, weights=weight1))
                mean2 = float(np.average(power2, weights=weight2))
                q1, q2 = np.quantile(v1, [.05, .5, .95]), np.quantile(v2, [.05, .5, .95])
                local1, local2 = 10 * np.log10(mean1 / power1), 10 * np.log10(mean2 / power2)
                c1, c2 = lag1(f[:half], first), lag1(f[half:], second)
                sd1, sd2 = float(np.std(v1, ddof=1)), float(np.std(v2, ddof=1))
                common = {
                    "hour_bjt": hour, "date_bjt": date, "equipment_epoch": epoch,
                    "window_minutes": minutes, "slot": slot,
                    "window_start_bjt": hour + pd.Timedelta(seconds=start),
                    "n_seconds_first": n1, "n_seconds_second": n2,
                    "longest_missing_first_s": missing1, "longest_missing_second_s": missing2,
                    "mean_power_first_w": mean1, "mean_power_second_w": mean2,
                    "half_power_drift_db": abs(float(10 * np.log10(mean1 / mean2))),
                    "half_sd_difference_db": abs(sd1 - sd2),
                    "half_p05_difference_db": abs(float(q1[0] - q2[0])),
                    "half_p50_difference_db": abs(float(q1[1] - q2[1])),
                    "half_p95_difference_db": abs(float(q1[2] - q2[2])),
                    "half_lag1_difference": abs(c1 - c2) if np.isfinite(c1) and np.isfinite(c2) else np.nan,
                    "local_shape_wasserstein_db": float(wasserstein_distance(local1, local2)),
                    "fraction_k11_records": float(arrays["k11"][start:start + width][valid].sum() / counts[valid].sum()),
                    "fraction_k101_records": float(arrays["k101"][start:start + width][valid].sum() / counts[valid].sum()),
                    "rain": bool(group.rain.iloc[0]), "dry_overcast": bool(group.dry_overcast.iloc[0]),
                    "clear_dry": bool(group.clear_dry.iloc[0]), "weather_stratum": group.weather_stratum.iloc[0],
                }
                for gate in chosen:
                    rows.append(dict(common, quality_gate=gate))
    return rows


def read_train(path: Path) -> pd.DataFrame:
    columns = ["second_bjt", "hour_bjt", "received_power_w", "hour_mean_power_w", "fast_residual_db",
               "quality_zero_records", "k11_records", "k101_records", "split", "weather_stratum",
               "rain", "dry_overcast", "clear_dry"]
    frame = pd.read_parquet(path, columns=columns, filters=[("split", "==", "train")])
    if not frame.split.eq("train").all():
        raise RuntimeError("A non-training row reached diagnostics")
    return frame.sort_values(["hour_bjt", "second_bjt"])


def run() -> None:
    frozen = manifest()
    MONTHS.mkdir(exist_ok=True)
    with worker_lock("window_diagnostics.py run"):
        for name, digest in frozen["frozen_design"]["source_partitions"].items():
            source = SOURCE / "clean_1s" / name
            if sha256(source) != digest:
                raise RuntimeError(f"Source fingerprint changed: {name}")
            stem = name.replace("clean_1s_", "windows_").replace(".parquet", "")
            target, meta_path = MONTHS / f"{stem}.parquet", MONTHS / f"{stem}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (meta["source_sha256"] != digest or meta["script_sha256"] != frozen["frozen_design"]["script_sha256"]
                        or not target.exists() or sha256(target) != meta["output_sha256"]):
                    raise RuntimeError(f"Checkpoint fingerprint mismatch: {stem}")
                print(f"Skip verified checkpoint {stem}: {meta['rows']:,} windows", flush=True)
                continue
            frame = read_train(source)
            rows = []
            n_hours = frame.hour_bjt.nunique()
            for i, (_, group) in enumerate(frame.groupby("hour_bjt", sort=True)):
                rows.extend(calculate_hour(group))
                if (i + 1) % 50 == 0:
                    print(f"{stem}: {i + 1}/{n_hours} train hours, {len(rows):,} window states", flush=True)
            if not rows:
                raise RuntimeError(f"No eligible window at all in {stem}")
            result = pd.DataFrame(rows).sort_values(["quality_gate", "window_minutes", "hour_bjt", "slot"])
            temp = target.with_suffix(".tmp.parquet")
            result.to_parquet(temp, index=False, compression="zstd")
            temp.replace(target)
            counts = {f"{gate}|{minutes}": int((result.quality_gate.eq(gate) & result.window_minutes.eq(minutes)).sum())
                      for gate in GATES for minutes in WINDOWS}
            write_json(meta_path, {
                "status": "complete", "completed_utc": now(), "source_sha256": digest,
                "script_sha256": frozen["frozen_design"]["script_sha256"], "output_sha256": sha256(target),
                "processed_train_hours": int(n_hours), "rows": len(result), "state_counts": counts,
            })
            print(f"Saved {stem}: {n_hours} train hours, {len(result):,} windows", flush=True)
    verify()


def verify() -> None:
    frozen = manifest()["frozen_design"]
    counts, hours, checked = {}, set(), 0
    max_error = 0.
    for name, digest in frozen["source_partitions"].items():
        stem = name.replace("clean_1s_", "windows_").replace(".parquet", "")
        p, mp = MONTHS / f"{stem}.parquet", MONTHS / f"{stem}.json"
        if not mp.exists() or not p.exists():
            raise RuntimeError(f"Incomplete month: {stem}")
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta["source_sha256"] != digest or sha256(p) != meta["output_sha256"]:
            raise RuntimeError(f"Wrong fingerprint in {stem}")
        data = pd.read_parquet(p)
        if len(data) != meta["rows"] or data.duplicated(["quality_gate", "hour_bjt", "window_minutes", "slot"]).any():
            raise RuntimeError("Count or unique key mismatch")
        required = data[[c for c in METRICS if c != "half_lag1_difference"]].to_numpy(dtype=float)
        if not np.isfinite(required).all() or (required < 0).any():
            raise RuntimeError("Invalid required endpoint")
        if not np.allclose(data.half_power_drift_db, np.abs(10 * np.log10(data.mean_power_first_w / data.mean_power_second_w)), atol=1e-12, rtol=0):
            raise RuntimeError("Power drift identity mismatch")
        for gate, rule in GATES.items():
            part = data.loc[data.quality_gate.eq(gate)]
            half = part.window_minutes.to_numpy() * 30
            if ((part.n_seconds_first < np.ceil(rule["half_coverage"] * half)).any()
                    or (part.n_seconds_second < np.ceil(rule["half_coverage"] * half)).any()
                    or (part.longest_missing_first_s > rule["max_missing_run"]).any()
                    or (part.longest_missing_second_s > rule["max_missing_run"]).any()):
                raise RuntimeError("Quality gate violation")
        computed_start = data.hour_bjt + pd.to_timedelta(data.slot * data.window_minutes * 60, unit="s")
        if not computed_start.equals(data.window_start_bjt) or (data.slot < 0).any() or (data.slot * data.window_minutes >= 60).any():
            raise RuntimeError("Window boundary mismatch")
        source = read_train(SOURCE / "clean_1s" / name)
        source_hours = set(source.hour_bjt)
        if not set(data.hour_bjt).issubset(source_hours) or meta["processed_train_hours"] != len(source_hours):
            raise RuntimeError("Training scope mismatch")
        # Recompute one deterministic actual window for every gate/length from
        # source observations, independently of the saved window endpoints.
        hour_groups = {h: g for h, g in source.groupby("hour_bjt", sort=False)}
        for (_, minutes), part in data.groupby(["quality_gate", "window_minutes"], sort=True):
            row = part.iloc[0]
            rule = GATES[row.quality_gate]
            group = hour_groups[row.hour_bjt]
            times = (group.second_bjt - row.window_start_bjt).dt.total_seconds()
            group = group.loc[(times >= 0) & (times < minutes * 60) & (group.quality_zero_records >= rule["min_records"])].copy()
            group["relative_s"] = (group.second_bjt - row.window_start_bjt).dt.total_seconds()
            halves = (group.loc[group.relative_s < minutes * 30], group.loc[group.relative_s >= minutes * 30])
            means = [np.sum(g.received_power_w * g.quality_zero_records) / np.sum(g.quality_zero_records) for g in halves]
            sd = [np.std(g.fast_residual_db, ddof=1) for g in halves]
            errors = (abs(row.half_power_drift_db - abs(10 * np.log10(means[0] / means[1]))),
                      abs(row.half_sd_difference_db - abs(sd[0] - sd[1])))
            max_error = max(max_error, *errors)
            if [len(g) for g in halves] != [row.n_seconds_first, row.n_seconds_second]:
                raise RuntimeError("Source window coverage does not recompute")
            checked += 1
        counts[p.name] = len(data)
        hours.update(source_hours)
    if max_error > 1e-10 or len(hours) != 2404:
        raise RuntimeError("Source recomputation or official train-hour count failed")
    write_json(OUT / "window_diagnostics_qa.json", {
        "status": "PASS", "completed_utc": now(), "monthly_results": len(counts), "rows_by_month": counts,
        "window_states": sum(counts.values()), "official_training_hours": len(hours),
        "source_windows_independently_recomputed": checked, "max_mean_or_sd_recompute_error": max_error,
        "run_manifest_sha256": sha256(OUT / "run_manifest.json"),
        "note": "Diagnostics only; no operational stationarity length selected.",
    })
    print(f"Stage B window QA PASS: {sum(counts.values()):,} states in {len(counts)} months", flush=True)


def report() -> None:
    qa = json.loads((OUT / "window_diagnostics_qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS" or qa["run_manifest_sha256"] != sha256(OUT / "run_manifest.json"):
        raise RuntimeError("Stage B QA not valid")
    data = pd.concat([pd.read_parquet(MONTHS / name) for name in qa["rows_by_month"]], ignore_index=True)
    keys = ["quality_gate", "hour_bjt", "date_bjt", "equipment_epoch", "rain", "dry_overcast", "clear_dry", "window_minutes"]
    hourly = data.groupby(keys, sort=True, dropna=False)[list(METRICS)].median().reset_index()
    hourly.to_parquet(OUT / "hour_window_diagnostics.parquet", index=False, compression="zstd")
    summary, uncertainty = [], []
    for gate in GATES:
        part = hourly.loc[hourly.quality_gate.eq(gate)]
        wide = part.pivot(index="hour_bjt", columns="window_minutes", values="half_power_drift_db").reindex(columns=WINDOWS)
        common = wide.dropna().index
        paired = part.loc[part.hour_bjt.isin(common)]
        for event_i, event in enumerate(EVENTS):
            group = paired if event == "all_train" else paired.loc[paired[event]]
            for metric in METRICS:
                for minutes in WINDOWS:
                    selected = group.loc[group.window_minutes.eq(minutes), metric].dropna()
                    summary.append({"quality_gate": gate, "event": event, "metric": metric, "window_minutes": minutes,
                                    "n_hours": len(selected), "n_days": int(group.loc[group.window_minutes.eq(minutes) & group[metric].notna(), "date_bjt"].nunique()),
                                    "median": float(selected.median()) if len(selected) else None,
                                    "p90": float(selected.quantile(.90)) if len(selected) else None})
            if group.empty:
                continue
            matrix = group.pivot(index="hour_bjt", columns="window_minutes", values=list(PRIMARY_BOOTSTRAP_METRICS))
            matrix = matrix.reindex(columns=pd.MultiIndex.from_product([PRIMARY_BOOTSTRAP_METRICS, WINDOWS]))
            calendar = pd.Series(matrix.index.strftime("%Y-%m-%d"), index=matrix.index)
            days = sorted(calendar.unique())
            idx_by_day = [np.flatnonzero(calendar.to_numpy() == day) for day in days]
            values = matrix.to_numpy(dtype=float)
            rng = np.random.default_rng(SEED + list(GATES).index(gate) * len(EVENTS) + event_i)
            draws = np.empty((REPLICATES, values.shape[1]))
            for b in range(REPLICATES):
                idx = np.concatenate([idx_by_day[i] for i in rng.integers(0, len(days), size=len(days))])
                draws[b] = np.quantile(values[idx], .90, axis=0)
            for j, (metric, minutes) in enumerate(matrix.columns):
                uncertainty.append({"quality_gate": gate, "event": event, "metric": metric, "window_minutes": minutes,
                                    "n_hours": len(matrix), "n_days": len(days),
                                    "p90": float(np.quantile(values[:, j], .90)),
                                    "ci95_low": float(np.quantile(draws[:, j], .025)),
                                    "ci95_high": float(np.quantile(draws[:, j], .975))})
    pd.DataFrame(summary).to_csv(OUT / "quality_window_paired_summary.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(OUT / "quality_window_day_uncertainty.csv", index=False)
    write_json(OUT / "window_report_qa.json", {
        "status": "PASS", "completed_utc": now(), "hour_window_states": len(hourly),
        "paired_summary_rows": len(summary), "primary_uncertainty_rows": len(uncertainty),
        "window_diagnostics_qa_sha256": sha256(OUT / "window_diagnostics_qa.json"),
        "output_sha256": {p.name: sha256(p) for p in [OUT / "hour_window_diagnostics.parquet", OUT / "quality_window_paired_summary.csv", OUT / "quality_window_day_uncertainty.csv"]},
        "interpretation": "Quality-sensitive effect estimates; no final window selected.",
    })
    print("Stage B paired/hourly report saved", flush=True)


def status() -> None:
    for name in ("active_worker.json", "window_diagnostics_qa.json", "window_report_qa.json"):
        p = OUT / name
        print(p.read_text(encoding="utf-8") if p.exists() else f"{name}: pending", flush=True)
    metadata = sorted(MONTHS.glob("windows_*.json")) if MONTHS.exists() else []
    print(json.dumps({"completed_month_checkpoints": len(metadata), "expected_months": 9,
                      "monthly_window_states": {p.stem: json.loads(p.read_text(encoding="utf-8"))["rows"] for p in metadata}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "report", "status"))
    args = parser.parse_args()
    {"run": run, "verify": verify, "report": report, "status": status}[args.command]()
