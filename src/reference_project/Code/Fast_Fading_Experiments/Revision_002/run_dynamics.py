"""C2: date-isolated, common-marginal temporal-generator comparison.

All input reads are official TRAIN hours. The final C1 all-training CDF is
fingerprinted as provenance, NEVER loaded as a fold's distribution. Observed
gaps remain gaps; synthetic joins are included in scores, not censored away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import lfilter
from scipy.special import ndtr, ndtri

import fit_marginals as fm
from window_diagnostics import OUT, SOURCE, now, read_train, sha256, worker_lock, write_json

HERE = Path(__file__).resolve().parent
DYN = OUT / "dynamics"
LAGS = (1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1800)
LENGTHS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3600)
CANDIDATES = ("iid", "gaussian_ar1") + tuple(f"block_{n:04d}" for n in LENGTHS)
SEED = 20261007
SIZE = 3600
CHUNK = 32  # Resume granularity only; it never changes scientific weighting.
THRESHOLD_P = (.95, .99)


@dataclass
class Series:
    hour: str
    day: str
    event: str
    fold: int
    q: np.ndarray
    records: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.records >= 9

    @property
    def f(self) -> np.ndarray:
        return -10 * np.log10(self.q)

    def as_marginal_hour(self) -> fm.Hour:
        good = self.valid
        q = self.q[good]
        w = self.records[good].astype(float)
        w /= w.sum()
        f = -10 * np.log10(q)
        order = np.argsort(f, kind="stable")
        fs, ws = f[order], w[order]
        cw, cwf = np.r_[0., np.cumsum(ws)], np.r_[0., np.cumsum(ws * fs)]
        cw[-1] = 1.
        return fm.Hour(self.hour, self.day, self.event, self.fold, q, w, fs, cw, cwf,
                       int(self.records[good].sum()))


def stable_seed(*parts: object) -> int:
    value = "|".join(map(str, (SEED,) + parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2 ** 63)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def core_design() -> dict:
    for name in ("marginal_model_qa.json", "marginal_document_qa.json", "window_report_qa.json"):
        if read_json(OUT / name)["status"] != "PASS":
            raise RuntimeError(f"Upstream {name} is not PASS")
    marginal, assignment = fm.frozen_manifest()
    if len(assignment) != 2404:
        raise RuntimeError("Training pool changed")
    return {
        "scope": "2404 official training hours only; C1 full-date-isolated 5-fold CV",
        "classes": list(fm.CLASSES), "candidates": list(CANDIDATES), "lags_s": list(LAGS),
        "block_lengths_s": list(LENGTHS), "seed": SEED, "generated_seconds": SIZE,
        "min_quality_zero_records": 9, "pit_clip": 1e-10, "phi_abs_limit": 1 - 1e-6,
        "score": "VS-p=1; per-lag, then per-hour, then per-date equal weighting",
        "mc_square": "cross products of independent realization increment means",
        "tail_probabilities": list(THRESHOLD_P), "resume_chunk_hours": CHUNK,
        "source_partitions": marginal["frozen_design"]["source_partitions"],
        "marginal_manifest_sha256": sha256(OUT / "marginal_manifest.json"),
        "marginal_model_qa_sha256": sha256(OUT / "marginal_model_qa.json"),
        "fold_days_sha256": sha256(OUT / "marginal_fold_days.csv"),
        "fold_hours_sha256": sha256(OUT / "marginal_fold_hours.csv"),
        "plan_sha256": sha256(HERE / "DYNAMICS_PLAN.md"),
        "implementation_sha256": sha256(HERE / "DYNAMICS_IMPLEMENTATION.md"),
        "script_sha256": sha256(Path(__file__)),
        "numerical_tests_sha256": sha256(HERE / "test_dynamics_numerics.py"),
        "reused_scripts_sha256": {n: sha256(HERE / n) for n in ("window_diagnostics.py", "fit_marginals.py")},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "final_validation_test_reads": 0,
    }


def freeze_manifest(replicates: int) -> dict:
    core = core_design()
    qa = read_json(DYN / "numerical_tests_qa.json")
    if qa["status"] != "PASS" or qa["script_sha256"] != core["script_sha256"] or \
            qa["test_script_sha256"] != core["numerical_tests_sha256"]:
        raise RuntimeError("Current code lacks passing numerical tests")
    DYN.mkdir(parents=True, exist_ok=True)
    base_path = DYN / "design_manifest.json"
    if base_path.exists():
        if read_json(base_path)["frozen_design"] != core:
            raise RuntimeError("Frozen dynamics code/design/source changed")
    else:
        write_json(base_path, {"created_utc": now(), "frozen_design": core})
    trial_dir = DYN / f"r{replicates:02d}"
    trial_dir.mkdir(exist_ok=True)
    value = {"core_manifest_sha256": sha256(base_path), "replicates": replicates,
             "half_replicates": replicates // 2, "expected_states": 70}
    path = trial_dir / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["frozen_design"] != value:
            raise RuntimeError("MC precision manifest changed")
    else:
        manifest = {"created_utc": now(), "frozen_design": value}
        write_json(path, manifest)
    return manifest


def load_series(core: dict) -> list[Series]:
    assignment = pd.read_csv(OUT / "marginal_fold_hours.csv")
    mapping = {str(row.hour_bjt): row for row in assignment.itertuples(index=False)}
    series = []
    for name, expected in core["source_partitions"].items():
        path = SOURCE / "clean_1s" / name
        if sha256(path) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
        frame = read_train(path)
        if not frame.split.eq("train").all():
            raise RuntimeError("Nontraining input entered dynamics")
        for hour, group in frame.groupby("hour_bjt", sort=True):
            key = str(hour)
            row = mapping[key]
            slot = ((group.second_bjt - hour).dt.total_seconds()).to_numpy(dtype=int)
            if (slot < 0).any() or (slot >= SIZE).any() or len(np.unique(slot)) != len(slot):
                raise RuntimeError("Duplicate/out-of-hour observed seconds")
            q = np.full(SIZE, np.nan)
            records = np.zeros(SIZE, dtype=np.uint8)
            q[slot] = (group.received_power_w / group.hour_mean_power_w).to_numpy(dtype=float)
            records[slot] = group.quality_zero_records.to_numpy(dtype=np.uint8)
            good = records >= 9
            if not good.any() or not np.isfinite(q[good]).all() or (q[good] <= 0).any():
                raise RuntimeError("Invalid training power ratio")
            actual_f = group.fast_residual_db.to_numpy(dtype=float)
            if np.max(np.abs(actual_f - (-10 * np.log10(q[slot])))) > 1e-9:
                raise RuntimeError("Hourly F definition changed")
            series.append(Series(key, str(row.date_bjt), str(row.event), int(row.fold), q, records))
        print(f"Source checked: {name}, {len(series)} training hours loaded", flush=True)
    series.sort(key=lambda h: h.hour)
    if len(series) != 2404 or len({h.hour for h in series}) != 2404:
        raise RuntimeError("Missing/duplicate training-hour scope")
    if pd.DataFrame({"day": [h.day for h in series], "fold": [h.fold for h in series]}).groupby("day").fold.nunique().max() != 1:
        raise RuntimeError("A date is split across folds")
    return series


def valid_runs(valid: np.ndarray) -> np.ndarray:
    change = np.diff(np.r_[False, valid, False].astype(np.int8))
    return np.column_stack((np.flatnonzero(change == 1), np.flatnonzero(change == -1))).astype(np.int32)


def pair_masks(valid: np.ndarray) -> list[np.ndarray]:
    bad = np.r_[0, np.cumsum(~valid)]
    # This checks the entire inclusive interval, not just its endpoints.
    return [(bad[lag + 1:] - bad[:len(valid) - lag]) == 0 for lag in LAGS]


def increments(f: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, seconds, correlations = [], [], []
    for lag, good in zip(LAGS, masks):
        a, b = f[:-lag][good], f[lag:][good]
        if len(a) == 0:
            means.append(np.nan); seconds.append(np.nan); correlations.append(np.nan)
            continue
        delta = np.abs(b - a)
        means.append(float(delta.mean()))
        seconds.append(float(np.mean(delta ** 2)))
        ac, bc = a - a.mean(), b - b.mean()
        denominator = math.sqrt(float(np.dot(ac, ac) * np.dot(bc, bc)))
        correlations.append(float(np.dot(ac, bc) / denominator) if denominator > 0 and len(a) >= 3 else np.nan)
    return np.array(means), np.array(seconds), np.array(correlations)


def score_from_moments(m1: np.ndarray, m2: np.ndarray, generated: np.ndarray) -> dict:
    r = generated.shape[0]
    if r < 2:
        raise ValueError("Independent-realization correction needs R>=2")
    good = np.isfinite(m1) & np.isfinite(m2)
    if not good.any() or not np.isfinite(generated[:, good]).all():
        raise RuntimeError("No available lag or invalid generated increments")
    g = generated[:, good]
    average = g.mean(axis=0)
    cross = (g.sum(axis=0) ** 2 - (g ** 2).sum(axis=0)) / (r * (r - 1))
    corrected = m2[good] - 2 * m1[good] * average + cross
    plugin = m2[good] - 2 * m1[good] * average + average ** 2
    return {"vs": float(corrected.mean()), "vs_plugin": float(plugin.mean()),
            "lags_used": int(good.sum()), "lag_vs": corrected.tolist(),
            "mean_increment": average.tolist()}


def inverse_q(model: dict, uniforms: np.ndarray) -> np.ndarray:
    if not np.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms > 1).any():
        raise RuntimeError("Invalid CDF probability")
    indices = np.minimum(np.searchsorted(model["support_cdf"], uniforms, side="left"),
                         len(model["support_q"]) - 1)
    return model["support_q"][indices]


def midpoint_z(model: dict, raw_q: np.ndarray) -> tuple[np.ndarray, int]:
    q = raw_q / model["power_normalizer"]
    support, cdf = model["support_q"], model["support_cdf"]
    left, right = np.searchsorted(support, q, "left"), np.searchsorted(support, q, "right")
    lo = np.where(left > 0, cdf[np.maximum(left - 1, 0)], 0.)
    hi = np.where(right > 0, cdf[np.maximum(right - 1, 0)], 0.)
    pit = (lo + hi) / 2
    clipped = int(((pit < 1e-10) | (pit > 1 - 1e-10)).sum())
    return ndtri(np.clip(pit, 1e-10, 1 - 1e-10)), clipped


def estimate_phi(model: dict, hours: list[Series]) -> dict:
    numerator = denominator = 0.
    included = pairs = clipped = 0
    for hour in hours:
        good = hour.valid
        pair = good[:-1] & good[1:]
        if not pair.any():
            continue
        z = np.full(SIZE, np.nan)
        z[good], c = midpoint_z(model, hour.q[good])
        a, b = z[:-1][pair], z[1:][pair]
        numerator += float(np.mean(a * b))
        denominator += float(np.mean((a ** 2 + b ** 2) / 2))
        included += 1; pairs += len(a); clipped += c
    if denominator <= 0 or included == 0:
        raise RuntimeError("No nondegenerate adjacent pairs for AR coefficient")
    raw = numerator / denominator
    phi = float(np.clip(raw, -1 + 1e-6, 1 - 1e-6))
    return {"phi_raw": raw, "phi": phi, "boundary_clipped": bool(phi != raw),
            "pit_clipped_seconds": clipped, "hours_with_pairs": included, "adjacent_pairs": pairs,
            "equal_hour_numerator_sum": numerator, "equal_hour_denominator_sum": denominator}


def ar_normal(phi: float, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    if abs(phi) >= 1:
        raise ValueError("AR process must be stationary")
    innovations = rng.standard_normal(size)
    rest, _ = lfilter([math.sqrt(1 - phi ** 2)], [1., -phi], innovations[1:],
                      zi=[phi * innovations[0]])
    return np.r_[innovations[0], rest]


def block_library(hours: list[Series], length: int) -> list[dict]:
    library = []
    for h in hours:
        runs = valid_runs(h.valid)
        runs = runs[(runs[:, 1] - runs[:, 0]) >= length]
        if len(runs):
            counts = runs[:, 1] - runs[:, 0] - length + 1
            library.append({"hour": h, "runs": runs, "cumulative": np.cumsum(counts),
                            "n_starts": int(counts.sum())})
    return library


def block_template(library: list[dict], length: int, rng: np.random.Generator,
                   size: int = SIZE) -> tuple[np.ndarray, list[dict]]:
    if not library:
        raise ValueError("No eligible contiguous blocks")
    template = np.empty(size)
    used = []
    for pos in range(0, size, length):
        entry = library[int(rng.integers(len(library)))]  # Hour-balanced, not length-biased.
        k = int(rng.integers(entry["n_starts"]))
        j = int(np.searchsorted(entry["cumulative"], k, side="right"))
        preceding = int(entry["cumulative"][j - 1]) if j else 0
        start = int(entry["runs"][j, 0]) + k - preceding
        n = min(length, size - pos)
        part = entry["hour"].q[start:start + n]
        if len(part) != n or not entry["hour"].valid[start:start + length].all():
            raise RuntimeError("Block crossed a real gap/hour")
        template[pos:pos + n] = part
        used.append({"hour": entry["hour"].hour, "start_s": start, "length_used_s": n})
    return template, used


def rank_reorder(q: np.ndarray, template: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if len(q) != len(template) or not np.isfinite(template).all():
        raise ValueError("Invalid rank template")
    order = np.lexsort((rng.random(len(q)), template))
    result = np.empty(len(q))
    result[order] = np.sort(q)
    phase = int(rng.integers(len(q)))
    return np.roll(result, phase), phase


def generate(candidate: str, model: dict, library: list[dict] | None,
             fold: int, hour: str, repetition: int) -> tuple[np.ndarray, dict]:
    marginal_seed = stable_seed("marginal", fold, hour, repetition)
    dynamics_seed = stable_seed("dependence", candidate, fold, hour, repetition)
    rng = np.random.default_rng(dynamics_seed)
    meta = {"marginal_seed": marginal_seed, "dynamics_seed": dynamics_seed,
            "phase_s": 0, "synthetic_joins_before_phase": 0}
    if candidate == "gaussian_ar1":
        q = inverse_q(model, ndtr(ar_normal(model["ar"]["phi"], rng)))
    else:
        q = inverse_q(model, np.random.default_rng(marginal_seed).random(SIZE))
        if candidate.startswith("block_"):
            length = int(candidate.split("_")[1])
            template, pieces = block_template(library, length, rng)
            q, phase = rank_reorder(q, template, rng)
            meta.update(phase_s=phase, synthetic_joins_before_phase=len(pieces) - 1,
                        block_count=len(pieces), circular_phase_seam_in_hour=bool(phase),
                        block_provenance_sha256=hashlib.sha256(json.dumps(pieces, sort_keys=True).encode()).hexdigest())
    if not np.isfinite(q).all() or (q <= 0).any():
        raise RuntimeError("Generated nonpositive/nonfinite power")
    return q, meta


def episode_arrays(f: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    fades, intervals = [], []
    fade_censored = interval_censored = 0
    for start, stop in valid_runs(valid):
        state = f[start:stop] > threshold
        changes = np.r_[0, np.flatnonzero(np.diff(state.astype(np.int8))) + 1, len(state)]
        for j, (a, b) in enumerate(zip(changes[:-1], changes[1:])):
            censored = a == 0 or b == len(state)
            if state[a]:
                if censored: fade_censored += 1
                else: fades.append(int(b - a))
            else:
                if censored: interval_censored += 1
                else: intervals.append(int(b - a))
    return {"complete_fades_s": fades, "complete_interfade_s": intervals,
            "censored_fades": fade_censored, "censored_interfade": interval_censored,
            "exceedance_seconds": int(((f > threshold) & valid).sum()),
            "observed_seconds": int(valid.sum()), "threshold_db": float(threshold)}


def describe(f: np.ndarray, q: np.ndarray, valid: np.ndarray, thresholds: np.ndarray) -> dict:
    actual = f[valid]
    slope = np.diff(f)[valid[:-1] & valid[1:]]
    positive, negative = slope[slope > 0], -slope[slope < 0]
    result = {"f_mean_db": float(actual.mean()), "f_sd_db": float(actual.std()),
              "f_quantiles_db": np.quantile(actual, [.01, .05, .5, .95, .99]).tolist(),
              "mean_q_observed_positions": float(q[valid].mean()),
              "slope_absolute_p50_dbps": float(np.median(np.abs(slope))) if len(slope) else None,
              "slope_absolute_p90_dbps": float(np.quantile(np.abs(slope), .9)) if len(slope) else None,
              "slope_positive_count": len(positive), "slope_negative_count": len(negative),
              "slope_positive_p90_dbps": float(np.quantile(positive, .9)) if len(positive) else None,
              "slope_negative_magnitude_p90_dbps": float(np.quantile(negative, .9)) if len(negative) else None}
    for label, threshold in zip(("95", "99"), thresholds):
        for name, value in episode_arrays(f, valid, threshold).items():
            result[f"tail{label}_{name}"] = value
    return result


def fit_context(fold: int, series: list[Series], core_sha: str) -> tuple[dict, dict[str, list[Series]]]:
    directory = DYN / "folds" / f"fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    train = [h for h in series if h.fold != fold]
    held = [h for h in series if h.fold == fold]
    if set(h.day for h in train) & set(h.day for h in held):
        raise RuntimeError("Date leakage")
    training_by_class = {event: [h for h in train if h.event == event] for event in fm.CLASSES}
    path = directory / "context.json"
    if path.exists():
        meta = read_json(path)
        if meta["core_manifest_sha256"] != core_sha or meta["fit_hours"] != [h.hour for h in train] or meta["held_hours"] != [h.hour for h in held]:
            raise RuntimeError("Fold context provenance changed")
        models = {}
        for event, component in meta["components"].items():
            support_path = directory / component["support_file"]
            if sha256(support_path) != component["support_sha256"]:
                raise RuntimeError("Fold CDF cache changed")
            with np.load(support_path) as arrays:
                models[event] = dict(component["model"], support_q=arrays["q"], support_cdf=arrays["cdf"])
        return models, training_by_class
    models, components = {}, {}
    for event, hours in training_by_class.items():
        if not hours:
            raise RuntimeError("Fold lacks an entire weather class")
        model = fm.fit("empirical", [h.as_marginal_hour() for h in hours])
        model["ar"] = estimate_phi(model, hours)
        model["tail_thresholds_db"] = fm.quantile_f(model, np.array(THRESHOLD_P)).tolist()
        model["discrete_unit_power_moment"] = float(np.dot(np.diff(np.r_[0., model["support_cdf"]]), model["support_q"]))
        if abs(model["discrete_unit_power_moment"] - 1) > 1e-10:
            raise RuntimeError("Fold empirical CDF unit power failed")
        support_path = directory / f"{event}_support.npz"
        atomic_npz(support_path, q=model["support_q"], cdf=model["support_cdf"])
        runs = [(i, a, b) for i, h in enumerate(hours) for a, b in valid_runs(h.valid)]
        library_path = directory / f"{event}_contiguous_run_library.npz"
        atomic_npz(library_path, hours=np.array([h.hour for h in hours]), runs=np.array(runs, dtype=np.int32))
        scalar_model = {k: v for k, v in model.items() if not isinstance(v, np.ndarray)}
        components[event] = {"model": scalar_model, "support_file": support_path.name,
                             "support_sha256": sha256(support_path), "run_library_file": library_path.name,
                             "run_library_sha256": sha256(library_path), "fit_hours": [h.hour for h in hours]}
        models[event] = model
        print(f"Fold {fold}, {event}: CDF fit on {len(hours)} hours; phi={model['ar']['phi']:.8f}", flush=True)
    meta = {"created_utc": now(), "core_manifest_sha256": core_sha, "fold": fold,
            "fit_hours": [h.hour for h in train], "held_hours": [h.hour for h in held],
            "fit_days": sorted({h.day for h in train}), "held_days": sorted({h.day for h in held}),
            "components": components}
    write_json(path, meta)
    return models, training_by_class


def candidate_fold(candidate: str, fold: int, reps: int, models: dict, pools: dict,
                   held: list[Series], manifest_sha: str) -> None:
    trial_dir = DYN / f"r{reps:02d}" / "trials"
    trial_dir.mkdir(exist_ok=True)
    trial_path = trial_dir / f"{candidate}__fold_{fold}.json"
    if trial_path.exists():
        old = read_json(trial_path)
        if old["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Completed trial has another manifest")
        for name, digest in old["output_sha256"].items():
            if sha256(DYN / f"r{reps:02d}" / name) != digest:
                raise RuntimeError("Completed trial output changed")
        print(f"Resume skip verified: {candidate} fold={fold}", flush=True)
        return
    libraries, coverage = {}, {}
    if candidate.startswith("block_"):
        length = int(candidate.split("_")[1])
        for event in fm.CLASSES:
            library = block_library(pools[event], length)
            libraries[event] = library
            coverage[event] = {"eligible_hours": len(library), "fit_hours": len(pools[event]),
                               "eligible_days": len({e['hour'].day for e in library}),
                               "legal_starts": sum(e["n_starts"] for e in library),
                               "excluded_hours": [h.hour for h in pools[event] if not any(e['hour'].hour == h.hour for e in library)]}
        absent = [event for event in fm.CLASSES if not libraries[event]]
        if absent:
            write_json(trial_path, {"status": "ineligible_no_contiguous_blocks", "candidate": candidate,
                                    "fold": fold, "replicates": reps, "manifest_sha256": manifest_sha,
                                    "coverage": coverage, "missing_classes": absent,
                                    "held_hours_expected": [h.hour for h in held], "output_sha256": {},
                                    "completed_utc": now(), "numerical_failure": False})
            print(f"Ineligible: {candidate} fold={fold}, no blocks in {absent}", flush=True)
            return
    directory = DYN / f"r{reps:02d}" / "chunks" / f"{candidate}__fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for begin in range(0, len(held), CHUNK):
        part = held[begin:begin + CHUNK]
        stem = f"hours_{begin:04d}"
        scores_path, reps_path = directory / f"{stem}_scores.parquet", directory / f"{stem}_replicates.parquet"
        observed_path, marker_path = directory / f"{stem}_observed.parquet", directory / f"{stem}.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            if marker["manifest_sha256"] != manifest_sha or marker["hours"] != [h.hour for h in part]:
                raise RuntimeError("Checkpoint scope/fingerprint changed")
            for name, digest in marker["output_sha256"].items():
                if sha256(directory / name) != digest:
                    raise RuntimeError("Checkpoint output changed")
        else:
            score_rows, rep_rows, observed_rows = [], [], []
            for hour in part:
                model = models[hour.event]
                thresholds = np.array(model["tail_thresholds_db"])
                masks = pair_masks(hour.valid)
                f = hour.f
                m1, m2, acf = increments(f, masks)
                info = {"hour": hour.hour, "day": hour.day, "event": hour.event, "fold": fold, "candidate": candidate}
                observed_rows.append(dict(info, n_primary_seconds=int(hour.valid.sum()), n_records=int(hour.records[hour.valid].sum()),
                                          pair_counts=[int(g.sum()) for g in masks], increment_mean=m1.tolist(),
                                          increment_second_moment=m2.tolist(), acf=acf.tolist(),
                                          **describe(f, hour.q, hour.valid, thresholds)))
                generated = []
                for repetition in range(reps):
                    q, meta = generate(candidate, model, libraries.get(hour.event), fold, hour.hour, repetition)
                    g = -10 * np.log10(q)
                    gm1, gm2, gacf = increments(g, masks)
                    generated.append(gm1)
                    mean_q = float(q.mean())
                    rep_rows.append(dict(info, repetition=repetition, **meta, mean_q_full_hour=mean_q,
                                         realized_hour_mean_attenuation_drift_db=-10 * math.log10(mean_q),
                                         increment_mean=gm1.tolist(), increment_second_moment=gm2.tolist(), acf=gacf.tolist(),
                                         **describe(g, q, hour.valid, thresholds)))
                    # One deterministic reproduction artifact per weather/fold/candidate.
                    if hour.hour == next(h.hour for h in held if h.event == hour.event) and repetition == 0:
                        example_path = directory / f"example_{hour.event}.npz"
                        atomic_npz(example_path, generated_q=q, actual_q=hour.q, valid=hour.valid)
                generated = np.array(generated)
                full, half = score_from_moments(m1, m2, generated), score_from_moments(m1, m2, generated[:reps // 2])
                score_rows.append(dict(info, vs=full["vs"], vs_plugin=full["vs_plugin"],
                                       vs_half=half["vs"], vs_half_plugin=half["vs_plugin"],
                                       lags_used=full["lags_used"], lag_vs=full["lag_vs"],
                                       mean_increment=full["mean_increment"]))
            atomic_parquet(scores_path, pd.DataFrame(score_rows))
            atomic_parquet(reps_path, pd.DataFrame(rep_rows))
            atomic_parquet(observed_path, pd.DataFrame(observed_rows))
            marker = {"completed_utc": now(), "manifest_sha256": manifest_sha, "hours": [h.hour for h in part],
                      "output_sha256": {p.name: sha256(p) for p in (scores_path, reps_path, observed_path)}}
            write_json(marker_path, marker)
            print(f"{candidate} fold={fold} hours={begin + len(part)}/{len(held)} R={reps}", flush=True)
        for path in (scores_path, reps_path, observed_path, marker_path):
            output_hashes[str(path.relative_to(DYN / f"r{reps:02d}"))] = sha256(path)
    for path in directory.glob("example_*.npz"):
        output_hashes[str(path.relative_to(DYN / f"r{reps:02d}"))] = sha256(path)
    write_json(trial_path, {"status": "complete", "candidate": candidate, "fold": fold, "replicates": reps,
                            "manifest_sha256": manifest_sha, "held_hours": [h.hour for h in held],
                            "hour_scores": len(held), "realizations": len(held) * reps, "coverage": coverage,
                            "output_sha256": output_hashes, "completed_utc": now(), "numerical_failure": False})


def run(reps: int) -> None:
    manifest = freeze_manifest(reps)
    core = read_json(DYN / "design_manifest.json")["frozen_design"]
    series = load_series(core)
    core_sha, manifest_sha = sha256(DYN / "design_manifest.json"), sha256(DYN / f"r{reps:02d}" / "manifest.json")
    for fold in range(5):
        models, pools = fit_context(fold, series, core_sha)
        held = [h for h in series if h.fold == fold]
        for candidate in CANDIDATES:
            try:
                candidate_fold(candidate, fold, reps, models, pools, held, manifest_sha)
            except Exception as error:
                failure_dir = DYN / f"r{reps:02d}" / "failures"
                failure_dir.mkdir(exist_ok=True)
                failure_path = failure_dir / f"{candidate}__fold_{fold}__{stable_seed(now())}.json"
                write_json(failure_path, {"status": "failed", "candidate": candidate, "fold": fold,
                           "replicates": reps, "manifest_sha256": manifest_sha, "failed_utc": now(),
                           "error": str(error), "traceback": traceback.format_exc(),
                           "note": "Not ineligible; original trace retained across identical-command recovery."})
                raise
    verify(reps)


def frames(reps: int, kind: str) -> pd.DataFrame:
    paths = sorted((DYN / f"r{reps:02d}" / "chunks").glob(f"*/*_{kind}.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def verify(reps: int) -> None:
    freeze_manifest(reps)
    core = read_json(DYN / "design_manifest.json")["frozen_design"]
    directory = DYN / f"r{reps:02d}"
    manifest_sha = sha256(directory / "manifest.json")
    trials = sorted((directory / "trials").glob("*.json"))
    expected = {(candidate, fold) for candidate in CANDIDATES for fold in range(5)}
    actual, numerical_failures = set(), []
    series = load_series(core)
    mapping = {h.hour: h for h in series}
    independent_max = gap_max = power_error = 0.
    complete_hours = realizations = ineligible = 0
    audit_rows = []
    if len(trials) != 70:
        raise RuntimeError(f"Expected 70 states, found {len(trials)}")
    contexts = {}
    for path in trials:
        trial = read_json(path)
        key = (trial["candidate"], trial["fold"])
        if key in actual or trial["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Duplicate trial or fingerprint mismatch")
        actual.add(key)
        if trial["numerical_failure"]:
            numerical_failures.append(str(path))
        for name, digest in trial["output_sha256"].items():
            if sha256(directory / name) != digest:
                raise RuntimeError("Trial output hash mismatch")
        fold, candidate = trial["fold"], trial["candidate"]
        if fold not in contexts:
            if not (DYN / "folds" / f"fold_{fold}" / "context.json").exists():
                raise RuntimeError("QA must not create a missing fold fit")
            contexts[fold] = fit_context(fold, series, sha256(DYN / "design_manifest.json"))
        models, pools = contexts[fold]
        context = read_json(DYN / "folds" / f"fold_{fold}" / "context.json")
        if set(context["fit_days"]) & set(context["held_days"]):
            raise RuntimeError("Date leakage in cache")
        for event, component in context["components"].items():
            if sha256(DYN / "folds" / f"fold_{fold}" / component["run_library_file"]) != component["run_library_sha256"]:
                raise RuntimeError("Run library altered")
            model = models[event]
            moment = float(np.dot(np.diff(np.r_[0., model["support_cdf"]]), model["support_q"]))
            power_error = max(power_error, abs(moment - 1))
        libraries = {event: block_library(pools[event], int(candidate.split("_")[1])) for event in fm.CLASSES} if candidate.startswith("block_") else {}
        if trial["status"] == "ineligible_no_contiguous_blocks":
            if not any(not libraries[event] for event in fm.CLASSES):
                raise RuntimeError("False no-block eligibility declaration")
            ineligible += 1
            continue
        if trial["status"] != "complete":
            raise RuntimeError("Nonterminal numerical/partial trial")
        chunk_dir = directory / "chunks" / f"{candidate}__fold_{fold}"
        scores = pd.concat([pd.read_parquet(p) for p in sorted(chunk_dir.glob("*_scores.parquet"))], ignore_index=True)
        generated_records = pd.concat([pd.read_parquet(p) for p in sorted(chunk_dir.glob("*_replicates.parquet"))], ignore_index=True)
        observed = pd.concat([pd.read_parquet(p) for p in sorted(chunk_dir.glob("*_observed.parquet"))], ignore_index=True)
        hours = [h.hour for h in series if h.fold == fold]
        if sorted(scores.hour) != hours or sorted(observed.hour) != hours or scores.hour.duplicated().any():
            raise RuntimeError("Fold held-hour membership differs")
        if len(generated_records) != len(hours) * reps or generated_records.duplicated(["hour", "repetition"]).any():
            raise RuntimeError("Realization count/duplicates")
        if set(generated_records.repetition) != set(range(reps)):
            raise RuntimeError("Invalid realization identities")
        complete_hours += len(hours); realizations += len(generated_records)
        obs_by_hour = {row.hour: row for row in observed.itertuples(index=False)}
        rep_by_hour = {hour: part.sort_values("repetition") for hour, part in generated_records.groupby("hour", sort=False)}
        for row in scores.itertuples(index=False):
            obs = obs_by_hour[row.hour]
            generated = np.stack(rep_by_hour[row.hour].increment_mean)
            full = score_from_moments(np.array(obs.increment_mean), np.array(obs.increment_second_moment), generated)
            half = score_from_moments(np.array(obs.increment_mean), np.array(obs.increment_second_moment), generated[:reps // 2])
            if max(abs(full["vs"] - row.vs), abs(half["vs"] - row.vs_half), abs(full["vs_plugin"] - row.vs_plugin)) > 1e-10:
                raise RuntimeError("Saved-moment score recomputation failed")
        # Independent literal point-pair formula for one held hour in each trial.
        hour = mapping[hours[0]]
        model = models[hour.event]
        masks = pair_masks(hour.valid)
        generated_f = np.stack([-10 * np.log10(generate(candidate, model, libraries.get(hour.event), fold, hour.hour, r)[0]) for r in range(reps)])
        literal = []
        f = hour.f
        for lag, good in zip(LAGS, masks):
            if not good.any(): continue
            obs_d = np.abs(f[lag:][good] - f[:-lag][good])
            g = np.array([np.abs(x[lag:][good] - x[:-lag][good]).mean() for x in generated_f])
            crossed = sum(g[r] * g[s] for r in range(reps) for s in range(reps) if r != s) / (reps * (reps - 1))
            literal.append(float(np.mean(obs_d ** 2 - 2 * obs_d * g.mean() + crossed)))
            # Independently inspect every inclusive observed interval used.
            for start in np.flatnonzero(good):
                gap_max = max(gap_max, int((~hour.valid[start:start + lag + 1]).sum()))
        saved = float(scores.loc[scores.hour.eq(hour.hour), "vs"].iloc[0])
        error = abs(float(np.mean(literal)) - saved)
        independent_max = max(independent_max, error)
        audit_rows.append({"candidate": candidate, "fold": fold, "hour": hour.hour,
                           "independent_literal_vs": float(np.mean(literal)), "saved_vs": saved, "abs_error": error})
    if actual != expected or numerical_failures or gap_max != 0 or independent_max > 1e-9 or power_error > 1e-10:
        raise RuntimeError("Independent dynamics audit failed")
    audit_path = directory / "independent_score_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    write_json(directory / "qa.json", {"status": "PASS", "completed_utc": now(), "trial_states": len(trials),
               "complete_trial_states": len(trials) - ineligible, "ineligible_trial_states": ineligible,
               "fold_out_hour_scores": complete_hours, "realization_records": realizations,
               "independent_recomputed_trials": len(audit_rows), "max_literal_score_abs_error": independent_max,
               "max_unit_power_moment_error": power_error, "observed_gap_crossings": int(gap_max),
               "date_leakage": 0, "final_validation_test_reads": 0, "numerical_failures": numerical_failures,
               "manifest_sha256": manifest_sha, "trial_hashes": {p.name: sha256(p) for p in trials},
               "audit_sha256": sha256(audit_path),
               "interpretation": "Development-fold temporal comparison only; C3 and final evaluation pending."})
    print(f"QA PASS: {len(trials)} states, {complete_hours} hour scores, {realizations} realizations", flush=True)


def block_bootstrap_effect(day_frame: pd.DataFrame, candidate: str, length: int, reps: int = 2000) -> dict:
    pivot = day_frame.pivot(index="day", columns="candidate", values="vs").sort_index()
    delta = (pivot["iid"] - pivot[candidate]).to_numpy()
    dates = pd.to_datetime(pivot.index)
    if length == 1:
        blocks = [np.array([i]) for i in range(len(delta))]
    else:
        blocks = [np.arange(i, i + length) for i in range(len(delta) - length + 1)
                  if (dates[i + length - 1] - dates[i]).days == length - 1]
    if not blocks:
        return {"candidate": candidate, "block_days": length, "status": "no_complete_contiguous_date_blocks"}
    rng = np.random.default_rng(stable_seed("date-bootstrap", candidate, length))
    draws = []
    for _ in range(reps):
        indices = np.concatenate([blocks[int(i)] for i in rng.integers(len(blocks), size=math.ceil(len(delta) / length))])[:len(delta)]
        draws.append(float(delta[indices].mean()))
    return {"candidate": candidate, "block_days": length, "status": "complete", "n_days": len(delta),
            "eligible_blocks": len(blocks), "iid_minus_candidate_vs": float(delta.mean()),
            "ci025": float(np.quantile(draws, .025)), "ci975": float(np.quantile(draws, .975)),
            "bootstrap_replicates": reps, "eligible_date_union": len(set(np.concatenate(blocks).tolist())),
            "note": "3/7-day eligible-block sensitivity is conditional on available complete date blocks"}


def report(reps: int) -> None:
    directory = DYN / f"r{reps:02d}"
    qa = read_json(directory / "qa.json")
    if qa["status"] != "PASS":
        raise RuntimeError("Report requires QA PASS")
    freeze_manifest(reps)
    for name, digest in qa["trial_hashes"].items():
        if sha256(directory / "trials" / name) != digest:
            raise RuntimeError("Trial changed since QA")
    scores = frames(reps, "scores")
    trials = [read_json(p) for p in sorted((directory / "trials").glob("*.json"))]
    eligible = [c for c in CANDIDATES if all(t["status"] == "complete" for t in trials if t["candidate"] == c)]
    scores = scores.loc[scores.candidate.isin(eligible)].copy()
    if any(len(scores.loc[scores.candidate.eq(c)]) != 2404 for c in eligible):
        raise RuntimeError("Eligible candidate lacks full training-hour coverage")
    daily = scores.groupby(["candidate", "day"], as_index=False)[["vs", "vs_half", "vs_plugin"]].mean()
    ranking = daily.groupby("candidate", as_index=False).agg(vs=("vs", "mean"), vs_half=("vs_half", "mean"),
                                                              vs_plugin=("vs_plugin", "mean"), n_days=("day", "size"))
    ranking = ranking.sort_values(["vs", "candidate"]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    half_winner = ranking.sort_values(["vs_half", "candidate"]).candidate.iloc[0]
    winner = ranking.candidate.iloc[0]
    stable = bool(winner == half_winner)
    status = "dependency_family_frozen" if stable else ("selection_unresolved_at_max_precision" if reps == 32 else "requires_monte_carlo_refinement")
    weather_daily = scores.groupby(["candidate", "event", "day"], as_index=False).vs.mean()
    weather = weather_daily.groupby(["candidate", "event"], as_index=False).agg(vs=("vs", "mean"), n_days=("day", "size"))
    weather = weather.merge(scores.groupby(["candidate", "event"]).size().rename("n_hours").reset_index(), on=["candidate", "event"])
    effects = pd.DataFrame([block_bootstrap_effect(daily, c, length) for c in eligible if c != "iid" for length in (1, 3, 7)])
    paths = {"candidate_ranking.csv": ranking, "hour_scores.csv": scores, "date_scores.csv": daily,
             "weather_scores.csv": weather, "paired_date_effects.csv": effects}
    for name, frame in paths.items():
        frame.to_csv(directory / name, index=False)
    selection = {"status": status, "created_utc": now(), "replicates": reps, "half_replicates": reps // 2,
                 "winner": winner, "half_budget_winner": half_winner, "eligible_candidates": eligible,
                 "ineligible_candidates": [c for c in CANDIDATES if c not in eligible],
                 "next_replicates": None if stable or reps == 32 else reps * 2,
                 "qa_sha256": sha256(directory / "qa.json"), "design_manifest_sha256": sha256(DYN / "design_manifest.json"),
                 "frozen_dynamic_parameters": {"family": winner, "block_length_s": int(winner.split("_")[1]) if winner.startswith("block_") else None},
                 "final_validation_test_reads": 0, "whole_fast_fading_stage_complete": False,
                 "output_sha256": {name: sha256(directory / name) for name in paths},
                 "decision_scope": "Common marginal dependence comparison on training-date folds; not a stationary-window selection."}
    write_json(directory / "selection.json", selection)
    lines = ["# C2 动态生成对照（训练日期内）", "", f"精度：{reps}个实现；状态：`{status}`。",
             f"主评分第一名：{winner}；前{reps // 2}实现第一名：{half_winner}。", "",
             "VS-1单位dB²，越低越好；不是逐点预测RMSE。天气表只诊断，不回调总体评分。", "",
             "| 方法 | 日期平均VS-1 | 半预算VS-1 | 名次 |", "|---|---:|---:|---:|"]
    for row in ranking.itertuples(index=False):
        lines.append(f"| {row.candidate} | {row.vs:.8g} | {row.vs_half:.8g} | {row.rank} |")
    lines.extend(["", "每个有资格候选使用相同2404小时，完整日期隔离；CDF与动态参数只在相应折训练日期拟合。",
                  "缺少合法连续片段者为无资格，不是数值失败；其状态和覆盖量仍保留。",
                  "此阶段尚未宣布平稳区间，尚未评价正式验证/测试，也未完成整个快变模型。",
                  "有限实现Q均值不强制为1；总体功率矩已核验。全部接缝包含于生成评分。",
                  "日期/3日/7日bootstrap为开发期配对不确定性；连续块仅覆盖存在的完整日期段，覆盖量见CSV。", ""])
    report_path = directory / "DYNAMICS_RESULTS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(directory / "report_qa.json", {"status": "PASS", "created_utc": now(),
               "selection_sha256": sha256(directory / "selection.json"), "report_sha256": sha256(report_path),
               "n_eligible_candidates": len(eligible), "hour_scores": len(scores),
               "main_half_winner_consistent": stable, "final_validation_test_reads": 0})
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def status(reps: int) -> None:
    directory = DYN / f"r{reps:02d}"
    trials = [read_json(p) for p in sorted((directory / "trials").glob("*.json"))]
    design_path = DYN / "design_manifest.json"
    fingerprints = None
    if design_path.exists():
        design = read_json(design_path)["frozen_design"]
        fingerprints = {"script": sha256(Path(__file__)) == design["script_sha256"],
                        "plan": sha256(HERE / "DYNAMICS_PLAN.md") == design["plan_sha256"],
                        "tests": sha256(HERE / "test_dynamics_numerics.py") == design["numerical_tests_sha256"]}
    print(json.dumps({"replicates": reps, "expected_states": 70, "terminal_states": len(trials),
          "complete_states": sum(t["status"] == "complete" for t in trials),
          "ineligible_states": sum(t["status"] == "ineligible_no_contiguous_blocks" for t in trials),
          "completed_hour_scores": sum(t.get("hour_scores", 0) for t in trials),
          "chunk_checkpoints": len(list((directory / "chunks").glob("*/*.json"))),
          "retained_failure_records": len(list((directory / "failures").glob("*.json"))),
          "fingerprints": fingerprints,
          "active_worker_metadata": read_json(OUT / "active_worker.json") if (OUT / "active_worker.json").exists() else None,
          "qa": read_json(directory / "qa.json") if (directory / "qa.json").exists() else None,
          "selection": read_json(directory / "selection.json") if (directory / "selection.json").exists() else None},
          ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "report", "status"))
    parser.add_argument("--replicates", type=int, default=8, choices=(8, 16, 32))
    args = parser.parse_args()
    if args.command == "status":
        status(args.replicates)
    else:
        with worker_lock(f"run_dynamics.py {args.command} --replicates {args.replicates}"):
            {"run": run, "verify": verify, "report": report}[args.command](args.replicates)


if __name__ == "__main__":
    main()
