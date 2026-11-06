"""Run the frozen ten-layer residual experiment on train-selected ERA5 columns.

Revision 014 supplies the estimators and residual experiment machinery. This
adapter redirects all writes to Revision 017 and removes columns rejected by
the training-only screen. It does not modify any previous revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiment as stage
import run_layers as legacy
import validate_layers as validator

c = stage.c
OUT = stage.OUT
LAYER_DIR = OUT / "layers"
LAYER_MANIFEST = LAYER_DIR / "run_manifest.json"
LAYER_VERSION = "correlation-pruned-1024-residual-v17-20261001"

c.OUT = OUT
legacy.OUT = OUT
legacy.SELECTION = OUT / "model_selection.json"
legacy.MODEL_QA = stage.MODEL_QA


def configure(saved: dict, frame: pd.DataFrame, names: list[str]) -> None:
    """Bind the old engine to this run's columns, records, and held-out data."""
    selected = set(saved["feature_names_retained"])
    base_positions = [i for i, name in enumerate(names[:188]) if name in selected]
    full_positions = [i for i, name in enumerate(names) if name in selected]
    c.require(len(base_positions) == saved["retained_base_count"] and
              len(full_positions) == saved["n_retained"], "Selected feature positions changed")

    def preprocess(xfit: np.ndarray, xhold: np.ndarray | None, model_name: str):
        width = xfit.shape[1]
        c.require(width in (188, 348) and (xhold is None or xhold.shape[1] == width),
                  "Unexpected baseline or full-feature input width")
        if model_name in stage.STRUCTURED:
            return stage.ORIGINAL_PREPROCESS(xfit, xhold, model_name)
        positions = base_positions if width == 188 else full_positions
        imputer = stage.SubsetImputer(positions)
        xf = imputer.fit_transform(xfit)
        xh = None if xhold is None else imputer.transform(xhold)
        scaler = stage.StandardScaler() if model_name in c.SCALED_MODELS else None
        if scaler is not None:
            xf = scaler.fit_transform(xf)
            if xh is not None:
                xh = scaler.transform(xh)
        return xf, xh, imputer, scaler

    c.preprocess = preprocess

    def fit_base(local_frame: pd.DataFrame, fit: np.ndarray, hold: np.ndarray,
                 local_names: list[str], chosen: dict,
                 with_history: bool = False) -> tuple[np.ndarray, dict | None]:
        c.require(len(local_names) == 188, "Baseline must use original 188-column layout")
        name = chosen["name"]
        raw_fit = c.raw(local_frame, fit, local_names)
        raw_hold = c.raw(local_frame, hold, local_names)
        xf, xh, _, _ = c.preprocess(raw_fit, raw_hold, name)
        yf = local_frame.loc[fit, c.TARGET].to_numpy(dtype=float)
        if with_history:
            model, history, _, meta = c.fit_logged(
                name, chosen["best_config"], 17, tuple(), xf, yf,
                fixed_rounds=chosen["formal_rounds"])
        else:
            config = c.model_config(name, chosen["best_config"], chosen["formal_rounds"])
            model = c.make_estimator(name, config, 17, xf.shape[1], tuple())
            model.fit(xf, yf)
            history, meta = None, None
        pred = np.asarray(model.predict(xh), dtype=float).reshape(-1)
        c.require(len(pred) == len(hold) and np.isfinite(pred).all(),
                  "Invalid pruned baseline prediction")
        return pred, None if not with_history else {"round_history": history, **meta}

    legacy.fit_base = fit_base

    def columns_for_mask(local_names: list[str], mask: int) -> tuple[list[int], list[str]]:
        c.require(len(local_names) == 348 and 0 <= mask < 1024,
                  "Unexpected layer feature layout or mask")
        active = [i for i in range(10) if mask & (1 << i)]
        features = [name for i in active
                    for name in local_names[188 + 16 * i:188 + 16 * (i + 1)]
                    if name in selected]
        c.require(len(features) == sum(saved["retained_per_pressure_level"][str(c.LEVELS[i])]
                                       for i in active),
                  "Selected layer feature count changed")
        return active, features

    legacy.columns_for_mask = columns_for_mask

    original_test_loader = stage.source.load_test
    dropped = [name for name in names if name not in selected]

    def load_masked_test(local_names: list[str]) -> pd.DataFrame:
        c.require(local_names == names, "Test feature order changed")
        test = original_test_loader(local_names)
        test.loc[:, dropped] = np.nan
        return test

    c.residual.load_test = load_masked_test
    c.load_training = lambda: (frame, names, c.layer_folds(frame))


def layer_manifest(frame: pd.DataFrame, names: list[str],
                   folds: list[dict], chosen: dict) -> dict:
    paths = [stage.DATA / name for name in
             ("train.parquet", "validation.parquet", "feature_columns.csv",
              "hourly_sample_status.parquet")]
    paths += [stage.SCREEN, stage.MODEL_QA, OUT / "model_selection.json",
              stage.HERE / "PROTOCOL.md", stage.HERE / "run_experiment.py",
              Path(__file__).resolve(), stage.PRIOR / "models.py",
              stage.PRIOR / "common.py", stage.PRIOR / "support.py",
              stage.PRIOR / "run_layers.py", stage.PRIOR / "validate_layers.py"]
    payload = {
        "version": LAYER_VERSION,
        "source_hashes": stage.hashes(paths),
        "feature_fingerprint": json.loads(stage.SCREEN.read_text(encoding="utf-8"))["fingerprint"],
        "winner": chosen["name"], "winner_config": chosen["best_config"],
        "winner_formal_rounds": chosen["formal_rounds"],
        "n_train": 2404, "n_validation": 343,
        "feature_names_original": names,
        "feature_names_retained": json.loads(stage.SCREEN.read_text(encoding="utf-8"))["feature_names_retained"],
        "levels_hpa": list(c.LEVELS), "residual_config": c.RESIDUAL_CONFIG,
        "alpha_candidates": list(c.ALPHAS), "seed": 17,
        "layer_fold_policy": "3 GroupShuffleSplit folds by Beijing calendar day, 20% held-out dates",
        "folds": [{key: value for key, value in fold.items() if not key.endswith("_index")}
                  for fold in folds],
        "test_not_read": True,
    }
    payload["fingerprint"] = stage.fingerprint(payload)
    payload["created_utc"] = stage.source.utc_now()
    return payload


def status() -> dict:
    selection = LAYER_DIR / "layer_selection.json"
    validation_qa = LAYER_DIR / "validation_qa.json"
    test = LAYER_DIR / "final_test.json"
    final_qa = LAYER_DIR / "final_qa.json"
    return {
        "phase": "FINAL_QA_PASS" if final_qa.exists() else
                 "TEST_WRITTEN" if test.exists() else
                 "VALIDATION_QA_PASS" if validation_qa.exists() else
                 "REPORT_WRITTEN" if selection.exists() else
                 "SUBSETS_RUNNING" if LAYER_MANIFEST.exists() else "NOT_STARTED",
        "manifest": LAYER_MANIFEST.exists(),
        "baseline_cache": (LAYER_DIR / "baseline_cache.json").exists(),
        "completed_masks": len(list((LAYER_DIR / "subsets").glob("mask_*.json"))),
        "report": selection.exists(), "validation_qa": validation_qa.exists(),
        "test": test.exists(), "final_qa": final_qa.exists(),
    }


def run_validation(stage_name: str) -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["validate_layers.py", stage_name]
        validator.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "preflight", "smoke", "baseline",
                                           "subsets", "report", "verify", "test", "verify-all"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return
    saved = stage.load_screen()
    c.require(stage.MODEL_QA.exists(), "Ten-model QA must pass before the layer stage")
    qa = json.loads(stage.MODEL_QA.read_text(encoding="utf-8"))
    c.require(qa["status"] == "PASS" and
              qa["feature_fingerprint"] == saved["fingerprint"],
              "Ten-model QA fingerprint mismatch")
    frame, names = stage.masked_training(saved)
    configure(saved, frame, names)
    legacy.manifest_candidate = layer_manifest
    chosen = legacy.winner()
    folds = c.layer_folds(frame)
    candidate = layer_manifest(frame, names, folds, chosen)
    if args.command == "preflight":
        print(json.dumps({"status": "PREFLIGHT_OK", "fingerprint": candidate["fingerprint"],
                          "winner": chosen["name"], "retained_features": saved["n_retained"],
                          "retained_base_features": saved["retained_base_count"],
                          "retained_per_level": saved["retained_per_pressure_level"]},
                         ensure_ascii=False, indent=2))
        return
    if args.command == "smoke":
        first = next(i for i in range(10)
                     if saved["retained_per_pressure_level"][str(c.LEVELS[i])] > 0)
        active, features = legacy.columns_for_mask(names, 1 << first)
        fit = np.flatnonzero(frame.split.to_numpy() == "train")[:120]
        hold = np.flatnonzero(frame.split.to_numpy() == "train")[120:160]
        pred, _ = legacy.fit_base(frame, fit, hold, names[:188], chosen)
        c.require(active == [first] and len(features) > 0 and len(pred) == 40,
                  "Layer adapter smoke check failed")
        print(json.dumps({"status": "SMOKE_OK", "baseline_prediction_count": len(pred),
                          "first_level_hpa": c.LEVELS[first],
                          "first_level_retained_features": len(features)}))
        return
    with c.writer_lock():
        manifest = c.check_manifest(LAYER_MANIFEST, candidate)
        fingerprint = manifest["fingerprint"]
        if args.command in ("verify", "verify-all"):
            run_validation("validation" if args.command == "verify" else "all")
            return
        cache = legacy.baseline_cache(frame, names, folds, chosen, fingerprint)
        if args.command == "baseline":
            return
        if args.command == "subsets":
            c.require(args.limit is None or args.limit > 0, "--limit must be positive")
            legacy.run_subsets(frame, names, cache, fingerprint, args.limit)
        elif args.command == "report":
            legacy.run_report(fingerprint)
        elif args.command == "test":
            legacy.run_test(frame, names, chosen, cache, fingerprint)


if __name__ == "__main__":
    main()
