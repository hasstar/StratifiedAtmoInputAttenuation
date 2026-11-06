"""Fresh ten-model search with formal-validation per-round traces; no test access."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import common as c

VERSION = "validation-stopped-ten-model-v14-self-contained-20260930"
OUT = c.OUT


def manifest_candidate(frame: pd.DataFrame, names: list[str], _folds: list[dict]) -> dict:
    source = [c.DATA / x for x in ("train.parquet", "validation.parquet",
                                   "feature_columns.csv", "hourly_sample_status.parquet", "manifest.json")]
    source += [c.HERE / "models.py", c.HERE / "support.py",
               c.HERE / "_vendor/catboost/__init__.py",
               c.HERE / "_vendor/catboost/_catboost.pyd",
               c.HERE / "_vendor/lightgbm/__init__.py",
               c.HERE / "_vendor/lightgbm/bin/lib_lightgbm.dll",
               c.HERE / "PROTOCOL.md",
               c.HERE / "common.py", c.HERE / "run_selection.py"]
    candidate = c.hash_sources(source, VERSION)
    candidate.update({
        "target": c.TARGET, "n_train": 2404, "n_validation": 343,
        "feature_names": names, "models": list(c.MODEL_NAMES),
        "configs": {name: c.candidate_configs(name) for name in c.MODEL_NAMES},
        "seeds": list(c.SEEDS), "test_not_read": True,
        "round_policy": "seed17 formal-validation minimum RMSE, earliest tie; fixed-round 3-seed refits",
        "selection_rule": "50% normalized validation RMSE + 50% normalized log single-row latency",
    })
    return candidate


def one_trial(frame: pd.DataFrame, names: list[str],
              name: str, config_id: int, config: dict, fingerprint: str) -> dict:
    fit = np.flatnonzero(frame.split.to_numpy() == "train")
    hold = np.flatnonzero(frame.split.to_numpy() == "validation")
    raw_fit = c.raw(frame, fit, names)
    raw_hold = c.raw(frame, hold, names)
    xf, xh, _, _ = c.preprocess(raw_fit, raw_hold, name)
    yf = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    yh = frame.loc[hold, c.TARGET].to_numpy(dtype=float)
    model, history, pred, meta = c.fit_logged(
        name, config, c.SEEDS[0], tuple(range(10)), xf, yf, xh, yh)
    c.require(abs(c.rmse(yh, pred) - min(row["holdout_rmse_db"] for row in history)) < 1e-5,
              "Selected round differs from minimum validation curve")
    final_rounds = meta["selected_round"]
    del model
    gc.collect()
    return {
        "fingerprint": fingerprint, "model": name, "config_id": config_id,
        "config": config, "seed": 17, "n_train": len(fit), "n_validation": len(hold),
        "round_history": history, **meta,
        "validation_truth_db": yh.tolist(),
        "validation_prediction_db": pred.tolist(),
        "validation_metrics": c.metrics(frame, hold, pred),
        "formal_rounds": final_rounds,
        "formal_rounds_reason": "lowest formal-validation RMSE, earliest tie"
                                 if final_rounds is not None else "solver fit; no epoch/tree round",
        "completed_utc": c.prior.utc_now(),
    }


def one_validation(frame: pd.DataFrame, names: list[str], name: str,
                   trial: dict, seed: int, fingerprint: str) -> dict:
    fit = np.flatnonzero(frame.split.to_numpy() == "train")
    hold = np.flatnonzero(frame.split.to_numpy() == "validation")
    raw_fit, raw_hold = c.raw(frame, fit, names), c.raw(frame, hold, names)
    xf, xh, imputer, scaler = c.preprocess(raw_fit, raw_hold, name)
    yf = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    model, history, _, meta = c.fit_logged(
        name, trial["config"], seed, tuple(range(10)), xf, yf,
        fixed_rounds=trial["formal_rounds"])
    pred = np.asarray(model.predict(xh), dtype=float).reshape(-1)
    c.require(len(pred) == 343 and np.isfinite(pred).all(), "Invalid validation predictions")
    timing = c.measure_latency(model, imputer, scaler, raw_hold)
    return {
        "fingerprint": fingerprint, "model": name, "config_id": trial["config_id"],
        "config": trial["config"], "seed": seed,
        "n_train": len(fit), "n_validation": len(hold),
        "formal_rounds": trial["formal_rounds"],
        "formal_rounds_reason": trial["formal_rounds_reason"],
        "round_history": history, **meta,
        "solver_termination": ("preset max_iter/tol or libsvm convergence"
                               if trial["formal_rounds"] is None else None),
        "validation_hour_bjt": frame.loc[hold, "hour_bjt"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist(),
        "validation_truth_db": frame.loc[hold, c.TARGET].to_numpy(dtype=float).tolist(),
        "validation_prediction_db": pred.tolist(),
        "metrics": c.metrics(frame, hold, pred), "latency": timing,
        "completed_utc": c.prior.utc_now(),
    }


def run(frame: pd.DataFrame, names: list[str], manifest: dict,
        requested: set[str] | None) -> None:
    fingerprint = manifest["fingerprint"]
    for name in c.MODEL_NAMES:
        if requested is not None and name not in requested:
            continue
        trials = []
        for config_id, config in enumerate(c.candidate_configs(name)):
            path = OUT / "trials" / f"{name}_{config_id:02d}.json"
            record = c.read_record(path, fingerprint)
            if record is None:
                record = one_trial(frame, names, name, config_id, config, fingerprint)
                c.prior.atomic_json(path, record)
                print(f"TUNING {name} config={config_id} validation_RMSE="
                      f"{record['validation_metrics']['overall']['rmse']:.6f} "
                      f"rounds={record['formal_rounds']}", flush=True)
            c.require(record["model"] == name and record["config_id"] == config_id and
                      len(record["validation_prediction_db"]) == 343, "Incomplete tuning trial")
            trials.append(record)
        best = min(trials, key=lambda row: (row["validation_metrics"]["overall"]["rmse"],
                                            row["validation_metrics"]["overall"]["mae"],
                                            row["config_id"]))
        for seed in c.SEEDS:
            path = OUT / "validation" / f"{name}_seed{seed}.json"
            record = c.read_record(path, fingerprint)
            if record is None:
                record = one_validation(frame, names, name, best, seed, fingerprint)
                c.prior.atomic_json(path, record)
                print(f"VALIDATION {name} seed={seed} "
                      f"RMSE={record['metrics']['overall']['rmse']:.6f} "
                      f"latency_ms={record['latency']['median_ms']:.5f}", flush=True)
            c.require(record["model"] == name and record["seed"] == seed and
                      record["config_id"] == best["config_id"], "Validation config mismatch")
    if requested is None:
        make_selection(fingerprint)


def make_selection(fingerprint: str) -> dict:
    rows = []
    for name in c.MODEL_NAMES:
        trials = [c.read_record(OUT / "trials" / f"{name}_{i:02d}.json", fingerprint)
                  for i in range(4)]
        c.require(all(item is not None for item in trials), f"Missing trial for {name}")
        best = min(trials, key=lambda row: (row["validation_metrics"]["overall"]["rmse"],
                                            row["validation_metrics"]["overall"]["mae"],
                                            row["config_id"]))
        seeds = [c.read_record(OUT / "validation" / f"{name}_seed{seed}.json", fingerprint)
                 for seed in c.SEEDS]
        c.require(all(item is not None for item in seeds), f"Missing seed for {name}")
        row = {"name": name, "best_config_id": best["config_id"],
               "best_config": best["config"], "formal_rounds": best["formal_rounds"],
               "formal_rounds_reason": best["formal_rounds_reason"],
               "candidate_validation_rmse_db": best["validation_metrics"]["overall"]["rmse"]}
        for group in c.GROUPS:
            for metric in ("rmse", "mae", "bias"):
                values = [item["metrics"][group][metric] for item in seeds]
                row[f"{group}_{metric}_mean"] = None if values[0] is None else float(np.mean(values))
                row[f"{group}_{metric}_sd_seed"] = None if values[0] is None else float(np.std(values, ddof=1))
            row[f"{group}_n"] = seeds[0]["metrics"][group]["n"]
        row["event_rmse_mean"] = float(np.mean([row[f"{event}_rmse_mean"] for event in
                                                   ("rain", "heavy_rain", "dry_overcast")]))
        row["latency_median_ms"] = float(np.median([item["latency"]["median_ms"] for item in seeds]))
        row["fit_seconds_median"] = float(np.median([item["fit_seconds"] for item in seeds]))
        rows.append(row)
    ranked, summary = c.cleaned.rank_models(rows)
    result = {"fingerprint": fingerprint, "status": "TEN_MODELS_VALIDATION_SELECTED",
              "selection_rule": "50% normalized validation RMSE + 50% normalized log single-row latency",
              "model_rows": ranked, **summary, "test_not_read": True,
              "completed_utc": c.prior.utc_now()}
    c.prior.atomic_json(OUT / "model_selection.json", result)
    pd.DataFrame(ranked).drop(columns=["best_config"]).to_csv(OUT / "model_ranking.csv", index=False)
    print(f"MODEL_SELECTED {result['primary_winner']} RMSE_ONLY={result['rmse_only_winner']}",
          flush=True)
    return result


def status() -> None:
    path = OUT / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    selection = OUT / "model_selection.json"
    print(json.dumps({"status": "SELECTED" if selection.exists() else "IN_PROGRESS" if manifest else "NOT_STARTED",
                      "fingerprint": None if manifest is None else manifest["fingerprint"],
                      "cv_trials": len(list((OUT / "trials").glob("*.json"))),
                      "validation_fits": len(list((OUT / "validation").glob("*.json"))),
                      "winner": None if not selection.exists() else
                                json.loads(selection.read_text(encoding="utf-8"))["primary_winner"]},
                     ensure_ascii=False, indent=2))


def smoke(frame: pd.DataFrame, names: list[str]) -> None:
    fit = np.flatnonzero(frame.split.to_numpy() == "train")[:120]
    hold = np.flatnonzero(frame.split.to_numpy() == "train")[120:160]
    for name in c.MODEL_NAMES:
        config = dict(c.candidate_configs(name)[0])
        for key in ("n_estimators", "iterations", "epochs"):
            if key in config:
                config[key] = 4
        xf, xh, _, _ = c.preprocess(c.raw(frame, fit, names), c.raw(frame, hold, names), name)
        yf = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
        yh = frame.loc[hold, c.TARGET].to_numpy(dtype=float)
        _, history, pred, meta = c.fit_logged(name, config, 17, tuple(range(10)),
                                               xf, yf, xh, yh)
        c.require(len(history) >= 1 and len(pred) == len(hold), f"Smoke failed: {name}")
        print(f"SMOKE {name} rounds={len(history)} selected={meta['selected_round']} "
              f"rmse={c.rmse(yh, pred):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "smoke", "run", "status"))
    parser.add_argument("--models", nargs="+", choices=c.MODEL_NAMES)
    args = parser.parse_args()
    if args.command == "status":
        status()
        return
    frame, names, folds = c.load_training()
    manifest = manifest_candidate(frame, names, folds)
    if args.command == "preflight":
        print(json.dumps({"status": "PREFLIGHT_OK", "fingerprint": manifest["fingerprint"],
                          "n_train": 2404, "n_validation": 343, "n_features": len(names),
                          "models": manifest["models"], "round_policy": manifest["round_policy"]},
                         ensure_ascii=False, indent=2))
        return
    if args.command == "smoke":
        smoke(frame, names)
        return
    with c.writer_lock():
        frozen = c.check_manifest(OUT / "run_manifest.json", manifest)
        run(frame, names, frozen, None if args.models is None else set(args.models))


if __name__ == "__main__":
    main()
