"""C3 train-date-only model-compatible windows, not proof of stationarity.

All reference banks use C2 fold-training CDFs/libraries. Final all-train
generator parameters are NEVER used for this development-fold diagnosis.
"""
from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import run_dynamics as dyn
from window_diagnostics import OUT, MONTHS, METRICS, GATES, now, sha256, worker_lock, write_json

HERE = Path(__file__).resolve().parent
DEST = OUT / "stationarity"
ENDPOINTS = METRICS[:6]
BANK = 64
REPS = 2 * BANK
CHUNK = 8
ALPHA = .05


def endpoint_bank(q: np.ndarray, records: np.ndarray, start: int, width: int, min_records: int) -> np.ndarray:
    """Vectorized (realization, six endpoints), matched observation masks."""
    q = np.atleast_2d(q)[:, start:start + width]
    n = records[start:start + width]
    valid = n >= min_records
    half = width // 2
    values, summaries, correlations = [], [], []
    for a, b in ((0, half), (half, width)):
        mask = valid[a:b]
        if mask.sum() < 3:
            raise ValueError("Insufficient half-window observations")
        raw = q[:, a:b]
        f = -10 * np.log10(raw[:, mask])
        if not np.isfinite(f).all():
            raise RuntimeError("Nonfinite eligible powers")
        weight = n[a:b][mask].astype(float)
        mean_q = (raw[:, mask] * weight).sum(axis=1) / weight.sum()
        values.append(mean_q)
        summaries.append(np.column_stack((f.std(axis=1, ddof=1), np.quantile(f, [.05, .5, .95], axis=1).T)))
        pair = mask[:-1] & mask[1:]
        left = -10 * np.log10(raw[:, :-1][:, pair])
        right = -10 * np.log10(raw[:, 1:][:, pair])
        if pair.sum() < 3:
            correlations.append(np.full(len(q), np.nan))
        else:
            left -= left.mean(axis=1, keepdims=True)
            right -= right.mean(axis=1, keepdims=True)
            den = np.sqrt((left ** 2).sum(axis=1) * (right ** 2).sum(axis=1))
            c = np.full(len(q), np.nan)
            np.divide((left * right).sum(axis=1), den, out=c, where=den > 0)
            correlations.append(c)
    return np.column_stack((np.abs(10 * np.log10(values[0] / values[1])),
                            np.abs(summaries[0] - summaries[1]),
                            np.abs(correlations[0] - correlations[1])))


def standardized_max(value: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    value = np.atleast_2d(value)
    z = np.zeros_like(value)
    nonzero = scale > 0
    z[:, nonzero] = (value[:, nonzero] - center[nonzero]) / scale[nonzero]
    z[:, ~nonzero] = np.where(value[:, ~nonzero] > center[~nonzero], np.inf, 0.)
    return z.max(axis=1)


def compatibility(observed: np.ndarray, generated: np.ndarray, half_precision: bool = False) -> dict:
    if generated.shape != (REPS, 6):
        raise ValueError("Reference bank size mismatch")
    if not np.isfinite(observed).all() or not np.isfinite(generated).all():
        return {"status": "incomplete_endpoints", "p": None, "compatible": None}
    n = BANK // 2 if half_precision else BANK
    fit, calibrate = generated[:n], generated[BANK:BANK + n]
    center, scale = fit.mean(axis=0), fit.std(axis=0, ddof=1)
    obs_t = standardized_max(observed, center, scale)[0]
    reference_t = standardized_max(calibrate, center, scale)
    p = float((1 + np.sum(reference_t >= obs_t)) / (n + 1))
    # T can be infinite in the explicitly documented zero-scale case. JSON
    # stores flags; the binary reference bank retains the exact endpoints.
    return {"status": "complete", "p": p, "compatible": p > ALPHA,
            "observed_t": float(obs_t) if np.isfinite(obs_t) else None,
            "observed_t_positive_infinity": bool(np.isposinf(obs_t)),
            "reference_t_infinities": int(np.isposinf(reference_t).sum()),
            "center": center.tolist(), "scale": scale.tolist()}


def manifest(create: bool = True) -> dict:
    selection = dyn.read_json(OUT / "dynamics/r16/selection.json")
    numerical = dyn.read_json(DEST / "numerical_tests_qa.json")
    final_qa = dyn.read_json(OUT / "final_generator/qa.json")
    window_qa = dyn.read_json(OUT / "window_diagnostics_qa.json")
    if (selection["status"] != "dependency_family_frozen" or selection["winner"] != "block_0256"
            or numerical["status"] != "PASS" or final_qa["status"] != "PASS"
            or window_qa["status"] != "PASS" or numerical["script_sha256"] != sha256(Path(__file__))):
        raise RuntimeError("C3 upstream gate failed")
    dyn.freeze_manifest(16)
    core = dyn.read_json(OUT / "dynamics/design_manifest.json")["frozen_design"]
    windows = {}
    for filename in window_qa["rows_by_month"]:
        path = MONTHS / filename
        metadata = dyn.read_json(path.with_suffix(".json"))
        if sha256(path) != metadata["output_sha256"]:
            raise RuntimeError("B window changed")
        windows[filename] = metadata["output_sha256"]
    fold_files = {}
    for fold in range(5):
        directory = OUT / "dynamics/folds" / f"fold_{fold}"
        context = dyn.read_json(directory / "context.json")
        if set(context["fit_days"]) & set(context["held_days"]):
            raise RuntimeError("C3 date isolation failed")
        for component in context["components"].values():
            if not set(component["fit_hours"]).issubset(context["fit_hours"]):
                raise RuntimeError("C3 component has non-fit hours")
            for field in ("support", "run_library"):
                path = directory / component[field + "_file"]
                if sha256(path) != component[field + "_sha256"]:
                    raise RuntimeError("C2 fold cache changed")
                fold_files[str(path.relative_to(OUT))] = sha256(path)
        fold_files[str((directory / "context.json").relative_to(OUT))] = sha256(directory / "context.json")
    design = {
        "stage": "C3 model-compatible training-date windows", "script_sha256": sha256(Path(__file__)),
        "plan_sha256": sha256(HERE / "STATIONARITY_PLAN.md"),
        "test_script_sha256": sha256(HERE / "test_stationarity_numerics.py"),
        "numerical_tests_qa_sha256": sha256(DEST / "numerical_tests_qa.json"),
        "core_dynamics_sha256": sha256(OUT / "dynamics/design_manifest.json"),
        "selection_sha256": sha256(OUT / "dynamics/r16/selection.json"),
        "final_generator_qa_sha256": sha256(OUT / "final_generator/qa.json"),
        "window_sources": windows, "source_partitions": core["source_partitions"],
        "fold_cache_files": fold_files,
        "fold_hours_sha256": sha256(OUT / "marginal_fold_hours.csv"),
        "endpoint_names": list(ENDPOINTS), "bank_sizes": [BANK, BANK], "alpha": ALPHA,
        "chunk_hours": CHUNK, "expected_training_hours": 2404, "expected_window_states": 653430,
        "final_validation_test_reads": 0,
    }
    path = DEST / "manifest.json"
    if path.exists():
        result = dyn.read_json(path)
        if result["frozen_design"] != design:
            raise RuntimeError("C3 frozen design/source changed")
        return result
    if not create:
        raise RuntimeError("C3 manifest not yet frozen")
    write_json(path, {"created_utc": now(), "frozen_design": design})
    return dyn.read_json(path)


def load_windows(frozen: dict) -> pd.DataFrame:
    data = pd.concat([pd.read_parquet(MONTHS / p) for p in frozen["frozen_design"]["window_sources"]], ignore_index=True)
    data["hour_key"] = data.hour_bjt.astype(str)
    data["minimum_records"] = data.quality_gate.map({key: rule["min_records"] for key, rule in GATES.items()})
    return data.sort_values(["hour_key", "minimum_records", "window_minutes", "slot", "quality_gate"])


def one_hour(h: dyn.Series, windows: pd.DataFrame, model: dict, library: list) -> tuple[pd.DataFrame, np.ndarray, dict]:
    bank, seeds = [], []
    for r in range(REPS):
        q, seed = dyn.generate("block_0256", model, library, h.fold, "C3-reference-" + h.hour, r)
        bank.append(q)
        seeds.append(seed)
    power = np.stack(bank)
    indices, reference, rows, maximum_error = {}, [], [], 0.
    for key, group in windows.groupby(["minimum_records", "window_minutes", "slot"], sort=True):
        minimum, minutes, slot = map(int, key)
        width, start = minutes * 60, slot * minutes * 60
        obs = endpoint_bank(h.q, h.records, start, width, minimum)[0]
        generated = endpoint_bank(power, h.records, start, width, minimum)
        saved_obs = group.iloc[0][list(ENDPOINTS)].to_numpy(dtype=float)
        finite = np.isfinite(obs) & np.isfinite(saved_obs)
        if not np.array_equal(np.isnan(obs), np.isnan(saved_obs)):
            raise RuntimeError("B endpoint missingness differs")
        error = float(np.max(np.abs(obs[finite] - saved_obs[finite])))
        maximum_error = max(maximum_error, error)
        if error > 1e-9:
            raise RuntimeError("Source B endpoints do not reproduce")
        # Use the immutable B endpoints after independent source reproduction;
        # avoid microscopic reaggregation rounding changing a rank tie.
        main, half = compatibility(saved_obs, generated), compatibility(saved_obs, generated, True)
        index = len(reference)
        indices[key] = index
        reference.append(generated)
        for record in group.to_dict("records"):
            record.update(event=h.event, fold=h.fold, reference_index_within_hour=index,
                          compatibility_status=main["status"], reference_tail_probability=main["p"],
                          compatible=main["compatible"], half_reference_tail_probability=half["p"],
                          half_compatible=half["compatible"], observed_joint_t=main.get("observed_t"),
                          joint_t_positive_infinity=main.get("observed_t_positive_infinity", False),
                          reference_t_infinities=main.get("reference_t_infinities", 0),
                          reference_endpoint_center=main.get("center"), reference_endpoint_scale=main.get("scale"))
            rows.append(record)
    return pd.DataFrame(rows), np.stack(reference) if reference else np.empty((0, REPS, 6)), {
        "hour": h.hour, "day": h.day, "fold": h.fold, "event": h.event,
        "window_states": len(rows), "unique_reference_windows": len(reference),
        "source_endpoint_max_error": maximum_error, "seeds": seeds,
    }


def run() -> None:
    frozen = manifest()
    chunks = DEST / "chunks"
    chunks.mkdir(exist_ok=True)
    with worker_lock("run_stationarity.py run"):
        try:
            core = dyn.read_json(OUT / "dynamics/design_manifest.json")["frozen_design"]
            series = dyn.load_series(core)
            windows = load_windows(frozen)
            by_hour = {h: part for h, part in windows.groupby("hour_key", sort=False)}
            if set(by_hour) - {h.hour for h in series}:
                raise RuntimeError("Nontraining B windows")
            for fold in range(5):
                models, pools = dyn.fit_context(fold, series, sha256(OUT / "dynamics/design_manifest.json"))
                libraries = {event: dyn.block_library(pool, 256) for event, pool in pools.items()}
                held = [h for h in series if h.fold == fold]
                if any(not x for x in libraries.values()):
                    raise RuntimeError("Frozen winner lacks a fold weather library")
                for offset in range(0, len(held), CHUNK):
                    stem = f"fold_{fold}_hours_{offset:04d}"
                    marker = chunks / (stem + ".json")
                    part_hours = held[offset:offset + CHUNK]
                    if marker.exists():
                        saved = dyn.read_json(marker)
                        if saved["manifest_sha256"] != sha256(DEST / "manifest.json"):
                            raise RuntimeError("Checkpoint manifest changed")
                        for name, digest in saved["output_sha256"].items():
                            if sha256(chunks / name) != digest:
                                raise RuntimeError("C3 checkpoint changed")
                        if saved["hours"] != [h.hour for h in part_hours]:
                            raise RuntimeError("Checkpoint hour order changed")
                        print("Skip verified " + stem, flush=True)
                        continue
                    frames, refs, metadata = [], {}, []
                    for i, h in enumerate(part_hours):
                        part = by_hour.get(h.hour, windows.iloc[:0])
                        result, reference, meta = one_hour(h, part, models[h.event], libraries[h.event])
                        if not result.empty:
                            frames.append(result)
                        refs[f"hour_{i}"] = reference
                        metadata.append(meta)
                    target, bank_path = chunks / (stem + ".parquet"), chunks / (stem + "_reference.npz")
                    empty = windows.iloc[:0].copy()
                    dyn.atomic_parquet(target, pd.concat(frames, ignore_index=True) if frames else empty)
                    dyn.atomic_npz(bank_path, **refs)
                    write_json(marker, {
                        "status": "complete", "completed_utc": now(), "manifest_sha256": sha256(DEST / "manifest.json"),
                        "hours": [h.hour for h in part_hours], "metadata": metadata,
                        "rows": sum(len(x) for x in frames),
                        "output_sha256": {target.name: sha256(target), bank_path.name: sha256(bank_path)},
                    })
                    print(f"Saved {stem}: {offset + len(part_hours)}/{len(held)} fold hours", flush=True)
            verify()
        except BaseException:
            failure = DEST / ("failure_" + now().replace(":", "-") + ".json")
            write_json(failure, {"failed_utc": now(), "traceback": traceback.format_exc()})
            raise


def verify() -> dict:
    frozen = manifest(False)
    windows = load_windows(frozen)
    results, hours, independent_windows, errors, hashes = [], [], 0, [], {}
    for marker in sorted((DEST / "chunks").glob("fold_*.json")):
        meta = dyn.read_json(marker)
        if meta["manifest_sha256"] != sha256(DEST / "manifest.json"):
            raise RuntimeError("C3 checkpoint manifest mismatch")
        hashes[marker.name] = sha256(marker)
        for name, digest in meta["output_sha256"].items():
            if sha256(marker.parent / name) != digest:
                raise RuntimeError("C3 output hash changed")
        frame = pd.read_parquet(marker.with_suffix(".parquet"))
        if len(frame) != meta["rows"]:
            raise RuntimeError("Window count mismatch")
        with np.load(marker.parent / (marker.stem + "_reference.npz")) as references:
            for i, hour in enumerate(meta["hours"]):
                rows = frame.loc[frame.hour_key.eq(hour)]
                ref = references[f"hour_{i}"]
                if rows.empty:
                    if len(ref):
                        raise RuntimeError("Zero-window hour has reference windows")
                    continue
                # Every endpoint bank/window is independently recalibrated.
                for _, repeated in rows.groupby("reference_index_within_hour", sort=True):
                    row = repeated.iloc[0]
                    bank = ref[int(row.reference_index_within_hour)]
                    obs = row[list(ENDPOINTS)].to_numpy(dtype=float)
                    for half, col in ((False, "reference_tail_probability"), (True, "half_reference_tail_probability")):
                        if np.isfinite(obs).all() and np.isfinite(bank).all():
                            n = BANK // 2 if half else BANK
                            fit, calibration = bank[:n], bank[BANK:BANK + n]
                            average = np.sum(fit, axis=0) / n
                            deviation = np.sqrt(np.sum((fit - average) ** 2, axis=0) / (n - 1))
                            observed_terms, calibration_terms = [], []
                            for j in range(6):
                                if deviation[j] > 0:
                                    observed_terms.append((obs[j] - average[j]) / deviation[j])
                                    calibration_terms.append((calibration[:, j] - average[j]) / deviation[j])
                                else:
                                    observed_terms.append(np.inf if obs[j] > average[j] else 0.)
                                    calibration_terms.append(np.where(calibration[:, j] > average[j], np.inf, 0.))
                            threshold = max(observed_terms)
                            extremes = np.array(calibration_terms).max(axis=0)
                            literal_p = (1 + sum(x >= threshold for x in extremes)) / (n + 1)
                            errors.append(abs(float(row[col]) - literal_p))
                            if not repeated[col].eq(row[col]).all():
                                raise RuntimeError("Duplicate quality states have different references")
                            verdict = "half_compatible" if half else "compatible"
                            if not repeated[verdict].eq(literal_p > ALPHA).all():
                                raise RuntimeError("Reference classification mismatch")
                        elif pd.notna(row[col]):
                            raise RuntimeError("Incomplete reference was silently scored")
                    independent_windows += 1
        results.append(frame)
        hours.extend(meta["hours"])
        errors.extend(m["source_endpoint_max_error"] for m in meta["metadata"])
    all_rows = pd.concat(results, ignore_index=True)
    key = ["quality_gate", "hour_key", "window_minutes", "slot"]
    if (len(hours) != 2404 or len(set(hours)) != 2404 or len(all_rows) != 653430
            or all_rows.duplicated(key).any()
            or set(map(tuple, all_rows[key].to_numpy())) != set(map(tuple, windows[key].to_numpy()))):
        raise RuntimeError("C3 official scope/completeness mismatch")
    if max(errors) > 1e-9:
        raise RuntimeError("C3 independent recalibration failed")
    complete = all_rows.compatibility_status.eq("complete")
    if ((all_rows.loc[complete, "reference_tail_probability"] <= 0).any()
            or (all_rows.loc[complete, "reference_tail_probability"] > 1).any()):
        raise RuntimeError("Invalid positive reference tail probabilities")
    assignment = pd.read_csv(OUT / "marginal_fold_hours.csv")
    if set(hours) != set(assignment.hour_bjt.astype(str)):
        raise RuntimeError("Training hour list changed")
    qa = {"status": "PASS", "completed_utc": now(), "training_hours": len(hours),
          "window_states": len(all_rows), "complete_endpoint_states": int(complete.sum()),
          "incomplete_endpoint_states": int((~complete).sum()), "reference_windows_recalibrated": independent_windows,
          "max_source_or_recalibration_error": max(errors), "checkpoint_hashes": hashes,
          "manifest_sha256": sha256(DEST / "manifest.json"), "final_validation_test_reads": 0,
          "interpretation": "Conditional model compatibility, not proof of physical stationarity"}
    path = DEST / "qa.json"
    if path.exists():
        old = dyn.read_json(path)
        if any(old[k] != v for k, v in qa.items() if k != "completed_utc"):
            raise RuntimeError("Existing C3 QA differs")
        return old
    write_json(path, qa)
    print("C3 window QA PASS", len(all_rows), flush=True)
    return qa


def summary_interval(values: pd.Series, block_days: int, tag: str) -> dict:
    dates = pd.to_datetime(values.index)
    blocks = [np.arange(i, i + block_days) for i in range(len(values) - block_days + 1)
              if (dates[i + block_days - 1] - dates[i]).days == block_days - 1]
    if not blocks:
        return {"status": "no_contiguous_date_blocks", "ci025": None, "ci975": None, "date_union": 0}
    rng = np.random.default_rng(dyn.stable_seed("C3-bootstrap", tag, block_days))
    a = values.to_numpy(dtype=float)
    samples = np.empty(2000)
    for j in range(2000):
        idx = np.concatenate([blocks[k] for k in rng.integers(len(blocks), size=math.ceil(len(a) / block_days))])[:len(a)]
        samples[j] = a[idx].mean()
    return {"status": "complete", "ci025": float(np.quantile(samples, .025)),
            "ci975": float(np.quantile(samples, .975)), "eligible_blocks": len(blocks),
            "date_union": len(set(np.concatenate(blocks).tolist()))}


def report() -> dict:
    frozen = manifest(False)
    qa = dyn.read_json(DEST / "qa.json")
    if qa["status"] != "PASS" or qa["manifest_sha256"] != sha256(DEST / "manifest.json"):
        raise RuntimeError("C3 report requires QA PASS")
    for name, digest in qa["checkpoint_hashes"].items():
        if sha256(DEST / "chunks" / name) != digest:
            raise RuntimeError("C3 checkpoint changed since QA")
    data = pd.concat([pd.read_parquet(p) for p in sorted((DEST / "chunks").glob("fold_*.parquet"))], ignore_index=True)
    keys = ["quality_gate", "window_minutes", "event", "date_bjt", "hour_key"]
    good = data.loc[data.compatibility_status.eq("complete")].copy()
    good["classification_changed"] = good.compatible != good.half_compatible
    hourly = good.groupby(keys, as_index=False)[["compatible", "half_compatible", "classification_changed"]].mean()
    overall = hourly.copy()
    overall["event"] = "all_train"
    hourly = pd.concat([hourly, overall], ignore_index=True)
    daily = hourly.groupby(keys[:-1], as_index=False)[["compatible", "half_compatible", "classification_changed"]].mean()
    summaries = []
    for key, part in daily.groupby(["quality_gate", "window_minutes", "event"], sort=True):
        gate, minutes, event = key
        population = data.loc[data.quality_gate.eq(gate) & data.window_minutes.eq(minutes)]
        if event != "all_train":
            population = population.loc[population.event.eq(event)]
        for col in ("compatible", "half_compatible", "classification_changed"):
            values = part.set_index("date_bjt")[col].sort_index()
            for days in (1, 3, 7):
                summaries.append({"quality_gate": gate, "window_minutes": int(minutes), "event": event,
                                  "quantity": col, "mean_date_proportion": float(values.mean()),
                                  "n_days": len(values), "n_hours": int(population.hour_key.nunique()),
                                  "window_states": len(population),
                                  "incomplete_window_states": int(population.compatibility_status.ne("complete").sum()),
                                  "block_days": days, **summary_interval(values, days, f"{key}|{col}")})
    summary = pd.DataFrame(summaries)
    suggestions = []
    for event in sorted(daily.event.unique()):
        for coverage in (.80, .90, .95):
            part = summary.loc[summary.quality_gate.eq("primary90_gap5") & summary.event.eq(event)
                               & summary.block_days.eq(1) & summary.quantity.isin(("compatible", "half_compatible"))]
            acceptable = []
            for minutes in (1, 2, 5, 10, 15, 30, 60):
                prefix = part.loc[part.window_minutes.le(minutes)]
                expected = 2 * sum(x <= minutes for x in (1, 2, 5, 10, 15, 30, 60))
                if len(prefix) == expected and prefix.ci025.notna().all() and prefix.ci025.ge(coverage).all():
                    acceptable.append(minutes)
            suggestions.append({"event": event, "engineering_coverage": coverage,
                                "supported_fixed_window_minutes": max(acceptable) if acceptable else None,
                                "status": "supported_under_model_reference" if acceptable else "no_common_fixed_window_supported"})
    for name, frame in (("hour_summary.parquet", hourly), ("date_summary.parquet", daily),
                        ("window_compatibility.csv", summary), ("engineering_window_suggestions.csv", pd.DataFrame(suggestions))):
        if name.endswith("parquet"):
            dyn.atomic_parquet(DEST / name, frame)
        else:
            frame.to_csv(DEST / name, index=False)
    decision = {"created_utc": now(), "status": "frozen_model_compatibility_diagnosis",
                "qa_sha256": sha256(DEST / "qa.json"), "suggestions": suggestions,
                "final_validation_test_reads": 0, "whole_fast_fading_stage_complete": False,
                "output_sha256": {name: sha256(DEST / name) for name in ("hour_summary.parquet", "date_summary.parquet",
                                    "window_compatibility.csv", "engineering_window_suggestions.csv")}}
    write_json(DEST / "selection.json", decision)
    return decision


def status() -> dict:
    markers = list((DEST / "chunks").glob("fold_*.json")) if (DEST / "chunks").exists() else []
    meta = [dyn.read_json(p) for p in markers]
    return {"completed_checkpoints": len(meta), "processed_training_hours": sum(len(m["hours"]) for m in meta),
            "window_states": sum(m["rows"] for m in meta),
            "manifest": dyn.read_json(DEST / "manifest.json") if (DEST / "manifest.json").exists() else None,
            "qa": dyn.read_json(DEST / "qa.json") if (DEST / "qa.json").exists() else None,
            "selection": dyn.read_json(DEST / "selection.json") if (DEST / "selection.json").exists() else None,
            "active_worker": dyn.read_json(OUT / "active_worker.json"),
            "failure_records": len(list(DEST.glob("failure_*.json")))}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "report", "status"))
    args = parser.parse_args()
    if args.command == "run":
        run()
    else:
        function = {"verify": verify, "report": report, "status": status}[args.command]
        print(json.dumps(function(), ensure_ascii=False, indent=2), flush=True)
