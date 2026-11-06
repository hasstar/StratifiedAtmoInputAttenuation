"""Retrain one residual head per pressure-layer mask, then evaluate one locked test."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

import common as c

VERSION = "validation-stopped-1024-residual-v14-self-contained-20260930"
OUT = c.OUT
SELECTION = OUT / "model_selection.json"
MODEL_QA = OUT / "model_selection_qa.json"


def winner() -> dict:
    c.require(SELECTION.exists() and MODEL_QA.exists(), "Ten-model selection and QA are required")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    qa = json.loads(MODEL_QA.read_text(encoding="utf-8"))
    c.require(selection["status"] == "TEN_MODELS_VALIDATION_SELECTED" and
              qa["status"] == "PASS" and qa["fingerprint"] == selection["fingerprint"],
              "Ten-model validation gate failed")
    return next(row for row in selection["model_rows"]
                if row["name"] == selection["primary_winner"])


def manifest_candidate(frame: pd.DataFrame, names: list[str], folds: list[dict], chosen: dict) -> dict:
    source = [c.DATA / x for x in ("train.parquet", "validation.parquet",
                                   "feature_columns.csv", "hourly_sample_status.parquet", "manifest.json")]
    source += [c.HERE / "models.py", c.HERE / "support.py",
               c.HERE / "_vendor/catboost/__init__.py",
               c.HERE / "_vendor/catboost/_catboost.pyd",
               c.HERE / "_vendor/lightgbm/__init__.py",
               c.HERE / "_vendor/lightgbm/bin/lib_lightgbm.dll",
               c.HERE / "PROTOCOL.md",
               c.HERE / "common.py", c.HERE / "run_layers.py", SELECTION, MODEL_QA]
    candidate = c.hash_sources(source, VERSION)
    candidate.update({"winner": chosen["name"], "winner_config": chosen["best_config"],
                      "winner_formal_rounds": chosen["formal_rounds"],
                      "n_train": 2404, "n_validation": 343, "feature_names": names,
                      "levels_hpa": list(c.LEVELS), "residual_config": c.RESIDUAL_CONFIG,
                      "alpha_candidates": list(c.ALPHAS), "seed": 17,
                      "layer_fold_policy": "3 GroupShuffleSplit folds by Beijing calendar day, 20% held-out dates",
                      "folds": [{k: v for k, v in fold.items() if not k.endswith("_index")}
                                for fold in folds], "test_not_read": True})
    return candidate


def fit_base(frame: pd.DataFrame, fit: np.ndarray, hold: np.ndarray, names: list[str],
             chosen: dict, with_history: bool = False) -> tuple[np.ndarray, dict | None]:
    name = chosen["name"]
    raw_fit, raw_hold = c.raw(frame, fit, names), c.raw(frame, hold, names)
    xf, xh, _, _ = c.preprocess(raw_fit, raw_hold, name)
    yf = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    if with_history:
        model, history, _, meta = c.fit_logged(
            name, chosen["best_config"], 17, tuple(), xf, yf,
            fixed_rounds=chosen["formal_rounds"])
    else:
        config = c.model_config(name, chosen["best_config"], chosen["formal_rounds"])
        model = c.make_estimator(name, config, 17, len(names), tuple())
        model.fit(xf, yf)
        history, meta = None, None
    pred = np.asarray(model.predict(xh), dtype=float).reshape(-1)
    c.require(len(pred) == len(hold) and np.isfinite(pred).all(), "Invalid baseline prediction")
    return pred, None if not with_history else {"round_history": history, **meta}


def crossfit_base(frame: pd.DataFrame, positions: np.ndarray, names: list[str],
                  chosen: dict) -> tuple[np.ndarray, list[dict]]:
    positions = np.asarray(positions, dtype=int)
    groups = frame.loc[positions, "hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    y = frame.loc[positions, c.TARGET].to_numpy(dtype=float)
    pred = np.full(len(positions), np.nan, dtype=float)
    covered = np.zeros(len(positions), dtype=int)
    audit = []
    for inner_id, (fit_local, hold_local) in enumerate(
            GroupKFold(n_splits=5).split(positions, y, groups), start=1):
        c.require(set(groups[fit_local]).isdisjoint(set(groups[hold_local])),
                  "Same day crossed OOF fit/hold")
        prediction, _ = fit_base(frame, positions[fit_local], positions[hold_local],
                                 names, chosen)
        pred[hold_local] = prediction
        covered[hold_local] += 1
        audit.append({"inner_fold": inner_id, "n_fit": len(fit_local),
                      "n_holdout": len(hold_local),
                      "holdout_global_indices": positions[hold_local].tolist()})
    c.require(np.all(covered == 1) and np.isfinite(pred).all(),
              "OOF baseline not complete")
    return pred, audit


def baseline_cache(frame: pd.DataFrame, names: list[str], folds: list[dict],
                   chosen: dict, fingerprint: str) -> dict:
    path = OUT / "layers" / "baseline_cache.json"
    cached = c.read_record(path, fingerprint)
    if cached is not None:
        return cached
    records = []
    for fold in folds:
        fit = np.asarray(fold["train_index"], dtype=int)
        hold = np.asarray(fold["validation_index"], dtype=int)
        oof, audit = crossfit_base(frame, fit, names[:188], chosen)
        base_hold, base_history = fit_base(frame, fit, hold, names[:188], chosen,
                                           with_history=True)
        truth_fit = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
        truth_hold = frame.loc[hold, c.TARGET].to_numpy(dtype=float)
        records.append({"fold_id": fold["id"], "fit_indices": fit.tolist(),
                        "holdout_indices": hold.tolist(), "inner_folds": audit,
                        "oof_base_fit_db": oof.tolist(),
                        "residual_fit_db": (truth_fit - oof).tolist(),
                        "base_holdout_db": base_hold.tolist(),
                        "residual_holdout_db": (truth_hold - base_hold).tolist(),
                        "truth_holdout_db": truth_hold.tolist(),
                        "base_model_train_trace": base_history})
        print(f"BASELINE CV FOLD {fold['id']}/3", flush=True)
    fit = np.flatnonzero(frame.split.to_numpy() == "train")
    hold = np.flatnonzero(frame.split.to_numpy() == "validation")
    oof, audit = crossfit_base(frame, fit, names[:188], chosen)
    base_hold, base_history = fit_base(frame, fit, hold, names[:188], chosen,
                                       with_history=True)
    truth_fit = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    cache = {"fingerprint": fingerprint, "winner": chosen["name"],
             "winner_config": chosen["best_config"],
             "winner_formal_rounds": chosen["formal_rounds"],
             "cv_folds": records,
             "full_train": {"fit_indices": fit.tolist(), "validation_indices": hold.tolist(),
                            "inner_folds": audit, "oof_base_train_db": oof.tolist(),
                            "residual_train_db": (truth_fit - oof).tolist(),
                            "base_validation_db": base_hold.tolist(),
                            "base_model_train_trace": base_history,
                            "base_validation_metrics": c.metrics(frame, hold, base_hold)},
             "completed_utc": c.prior.utc_now()}
    c.prior.atomic_json(path, cache)
    print("BASELINE CACHE COMPLETE", flush=True)
    return cache


def columns_for_mask(names: list[str], mask: int) -> tuple[list[int], list[str]]:
    active = [i for i in range(10) if mask & (1 << i)]
    features = [name for i in active for name in names[188 + 16 * i:188 + 16 * (i + 1)]]
    c.require(len(features) == 16 * len(active), "Layer feature count changed")
    return active, features


def g_model(rounds: int = 120) -> XGBRegressor:
    return XGBRegressor(**{**c.RESIDUAL_CONFIG, "n_estimators": rounds},
                        objective="reg:squarederror", tree_method="hist",
                        eval_metric="rmse", random_state=17, n_jobs=2, verbosity=0)


def fit_g(xfit_raw: np.ndarray, yfit: np.ndarray,
          xhold_raw: np.ndarray | None, yhold: np.ndarray | None,
          rounds: int = 120):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xf = imputer.fit_transform(xfit_raw)
    xh = None if xhold_raw is None else imputer.transform(xhold_raw)
    model = g_model(rounds)
    evaluation = [(xf, yfit)]
    if xh is not None:
        evaluation.append((xh, yhold))
    start = time.perf_counter()
    model.fit(xf, yfit, eval_set=evaluation, verbose=False)
    seconds = time.perf_counter() - start
    log = model.evals_result()
    train_curve = log["validation_0"]["rmse"]
    hold_curve = log["validation_1"]["rmse"] if xh is not None else None
    trace = [{"round": i + 1, "train_residual_rmse_db": float(v),
              "holdout_residual_rmse_db": None if hold_curve is None else float(hold_curve[i])}
             for i, v in enumerate(train_curve)]
    best = rounds if hold_curve is None else min(range(1, rounds + 1),
                                                 key=lambda i: (hold_curve[i - 1], i))
    pred = None if xh is None else np.asarray(model.predict(xh, iteration_range=(0, best)), dtype=float)
    return model, imputer, trace, best, pred, seconds


def one_mask(frame: pd.DataFrame, names: list[str], mask: int, cache: dict,
             fingerprint: str) -> dict:
    active, features = columns_for_mask(names, mask)
    full = cache["full_train"]
    fit_all = np.asarray(full["fit_indices"], dtype=int)
    valid = np.asarray(full["validation_indices"], dtype=int)
    base_valid = np.asarray(full["base_validation_db"], dtype=float)
    y_valid = frame.loc[valid, c.TARGET].to_numpy(dtype=float)
    if mask == 0:
        cv_y = np.concatenate([np.asarray(f["truth_holdout_db"], dtype=float)
                               for f in cache["cv_folds"]])
        cv_base = np.concatenate([np.asarray(f["base_holdout_db"], dtype=float)
                                  for f in cache["cv_folds"]])
        return {"fingerprint": fingerprint, "mask": mask, "levels_hpa": [],
                "features": [], "n_layer_features": 0, "config": None,
                "folds": [], "alpha_scores": {str(a): c.rmse(cv_y, cv_base) for a in c.ALPHAS},
                "alpha": 0.0, "formal_rounds": 0,
                "stopping_reason": "No layer features; Base188 control",
                "cv_pooled_rmse_db": c.rmse(cv_y, cv_base),
                "final_train_round_history": [], "g_validation_db": [0.0] * len(valid),
                "validation_prediction_db": base_valid.tolist(),
                "validation_metrics": c.metrics(frame, valid, base_valid),
                "completed_utc": c.prior.utc_now()}
    fold_rows = []
    truth_cv, base_cv, correction_cv = [], [], []
    for fold in cache["cv_folds"]:
        fit = np.asarray(fold["fit_indices"], dtype=int)
        hold = np.asarray(fold["holdout_indices"], dtype=int)
        yfit = np.asarray(fold["residual_fit_db"], dtype=float)
        yhold = np.asarray(fold["residual_holdout_db"], dtype=float)
        _, _, history, best, pred, seconds = fit_g(c.raw(frame, fit, features), yfit,
                                                    c.raw(frame, hold, features), yhold)
        fold_rows.append({"fold_id": fold["fold_id"], "n_fit": len(fit),
                          "n_holdout": len(hold), "round_history": history,
                          "trained_rounds": 120, "selected_round": best,
                          "stopping_reason": "minimum training-internal day-grouped holdout residual RMSE",
                          "fit_seconds": seconds,
                          "g_holdout_db": pred.tolist()})
        truth_cv.append(np.asarray(fold["truth_holdout_db"], dtype=float))
        base_cv.append(np.asarray(fold["base_holdout_db"], dtype=float))
        correction_cv.append(pred)
    truth, base, correction = (np.concatenate(part) for part in
                               (truth_cv, base_cv, correction_cv))
    scores = {str(alpha): c.rmse(truth, base + alpha * correction)
              for alpha in c.ALPHAS}
    alpha = min(c.ALPHAS, key=lambda a: (scores[str(a)], a))
    formal_rounds = int(np.median([f["selected_round"] for f in fold_rows]))
    model, imputer, trace, _, _, seconds = fit_g(
        c.raw(frame, fit_all, features),
        np.asarray(full["residual_train_db"], dtype=float), None, None,
        rounds=formal_rounds)
    g_valid = np.asarray(model.predict(imputer.transform(c.raw(frame, valid, features))), dtype=float)
    final_pred = base_valid + alpha * g_valid
    c.require(np.isfinite(final_pred).all(), "Invalid subset prediction")
    return {"fingerprint": fingerprint, "mask": mask,
            "levels_hpa": [c.LEVELS[i] for i in active],
            "features": features, "n_layer_features": len(features),
            "config": c.RESIDUAL_CONFIG, "folds": fold_rows,
            "alpha_scores": scores, "alpha": alpha,
            "formal_rounds": formal_rounds,
            "stopping_reason": "median of three training-internal best residual-tree counts",
            "cv_pooled_rmse_db": scores[str(alpha)],
            "final_train_round_history": trace, "final_fit_seconds": seconds,
            "g_validation_db": g_valid.tolist(),
            "validation_prediction_db": final_pred.tolist(),
            "validation_metrics": c.metrics(frame, valid, final_pred),
            "completed_utc": c.prior.utc_now()}


def run_subsets(frame: pd.DataFrame, names: list[str], cache: dict,
                fingerprint: str, limit: int | None) -> None:
    done = 0
    for mask in range(1024):
        path = OUT / "layers" / "subsets" / f"mask_{mask:04d}.json"
        if c.read_record(path, fingerprint) is not None:
            continue
        if limit is not None and done >= limit:
            break
        result = one_mask(frame, names, mask, cache, fingerprint)
        c.prior.atomic_json(path, result)
        done += 1
        print(f"SUBSET {mask + 1}/1024 levels={len(result['levels_hpa'])} "
              f"alpha={result['alpha']:.2f} CV={result['cv_pooled_rmse_db']:.5f} "
              f"VAL={result['validation_metrics']['overall']['rmse']:.5f}", flush=True)
    print(f"NEW_SUBSETS {done}", flush=True)


def run_report(fingerprint: str) -> dict:
    records = {}
    for mask in range(1024):
        item = c.read_record(OUT / "layers" / "subsets" / f"mask_{mask:04d}.json", fingerprint)
        c.require(item is not None and item["mask"] == mask, f"Missing subset {mask}")
        records[mask] = item
    rows = []
    for mask, item in records.items():
        row = {"mask": mask, "n_levels": len(item["levels_hpa"]),
               "levels_hpa": ";".join(map(str, item["levels_hpa"])),
               "n_layer_features": item["n_layer_features"],
               "alpha": item["alpha"], "formal_rounds": item["formal_rounds"],
               "cv_pooled_rmse_db": item["cv_pooled_rmse_db"]}
        for group in c.GROUPS:
            for metric in ("rmse", "mae", "bias"):
                row[f"validation_{group}_{metric}_db"] = item["validation_metrics"][group][metric]
            row[f"validation_{group}_n"] = item["validation_metrics"][group]["n"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "layers" / "subset_summary.csv", index=False)
    pairs = []
    for level_index, level_hpa in enumerate(c.LEVELS):
        bit = 1 << level_index
        for without in range(1024):
            if without & bit:
                continue
            with_layer = without | bit
            row = {"level_hpa": level_hpa, "without_mask": without,
                   "with_mask": with_layer, "other_level_count": without.bit_count()}
            for group in c.GROUPS:
                a = records[without]["validation_metrics"][group]["rmse"]
                b = records[with_layer]["validation_metrics"][group]["rmse"]
                row[f"validation_{group}_rmse_improvement_db"] = None if a is None or b is None else a - b
            pairs.append(row)
    c.require(len(pairs) == 5120, "Layer pair count changed")
    paired = pd.DataFrame(pairs)
    paired.to_csv(OUT / "layers" / "layer_pairwise.csv", index=False)
    layer_rows = []
    for level in c.LEVELS:
        subset = paired.loc[paired.level_hpa.eq(level)]
        row = {"level_hpa": level, "paired_contexts": len(subset)}
        for group in ("overall", "rain", "heavy_rain", "dry_overcast"):
            values = subset[f"validation_{group}_rmse_improvement_db"].dropna()
            row[f"{group}_mean_improvement_db"] = float(values.mean())
            row[f"{group}_median_improvement_db"] = float(values.median())
            row[f"{group}_fraction_improved"] = float((values > 0).mean())
        layer_rows.append(row)
    pd.DataFrame(layer_rows).to_csv(OUT / "layers" / "layer_summary.csv", index=False)
    chosen = min(records.values(), key=lambda item: (item["cv_pooled_rmse_db"],
                                                     len(item["levels_hpa"]), item["mask"]))
    result = {"fingerprint": fingerprint, "status": "ALL_1024_VALIDATION_WRITTEN_PENDING_QA",
              "n_subsets": 1024, "n_layer_pairs": 5120,
              "selected_mask_train_only": chosen["mask"],
              "selected_levels_hpa": chosen["levels_hpa"],
              "selected_alpha": chosen["alpha"],
              "selected_formal_rounds": chosen["formal_rounds"],
              "selection_basis": "minimum pooled training-internal RMSE, then fewer levels/mask",
              "test_not_read": True, "completed_utc": c.prior.utc_now()}
    c.prior.atomic_json(OUT / "layers" / "layer_selection.json", result)
    print(json.dumps({"status": result["status"], "selected_mask": chosen["mask"],
                      "levels": chosen["levels_hpa"]}, ensure_ascii=False), flush=True)
    return result


def trained_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame,
                    names: list[str], chosen: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    name = chosen["name"]
    fit = np.flatnonzero(train_frame.split.to_numpy() == "train")
    valid = np.flatnonzero(train_frame.split.to_numpy() == "validation")
    fit_raw = c.raw(train_frame, fit, names)
    valid_raw = c.raw(train_frame, valid, names)
    test_raw = c.raw(test_frame, np.arange(len(test_frame)), names)
    xf, xv, imputer, scaler = c.preprocess(fit_raw, valid_raw, name)
    xt = imputer.transform(test_raw)
    if scaler is not None:
        xt = scaler.transform(xt)
    yfit = train_frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    model, history, _, meta = c.fit_logged(name, chosen["best_config"], 17,
                                            tuple(range(10)) if len(names) == 348 else tuple(),
                                            xf, yfit, fixed_rounds=chosen["formal_rounds"])
    pred_valid = np.asarray(model.predict(xv), dtype=float).reshape(-1)
    pred_test = np.asarray(model.predict(xt), dtype=float).reshape(-1)
    return pred_valid, pred_test, {"config": chosen["best_config"], "seed": 17,
                                   "formal_rounds": chosen["formal_rounds"],
                                   "round_history": history, **meta}


def run_test(frame: pd.DataFrame, names: list[str], chosen: dict,
             cache: dict, fingerprint: str) -> dict:
    path = OUT / "layers" / "final_test.json"
    existing = c.read_record(path, fingerprint)
    if existing is not None:
        return existing
    choice = c.read_record(OUT / "layers" / "layer_selection.json", fingerprint)
    qa_path = OUT / "layers" / "validation_qa.json"
    c.require(choice is not None and choice["n_subsets"] == 1024 and qa_path.exists(),
              "All 1024 subsets and their validation QA must complete before test")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    c.require(qa["status"] == "PASS" and qa["fingerprint"] == fingerprint,
              "Layer validation QA gate failed")
    mask = choice["selected_mask_train_only"]
    selected = c.read_record(OUT / "layers" / "subsets" / f"mask_{mask:04d}.json", fingerprint)
    all_layers = c.read_record(OUT / "layers" / "subsets" / "mask_1023.json", fingerprint)
    c.require(selected is not None and all_layers is not None, "Final masks missing")
    test_frame = c.residual.load_test(names)
    valid = np.flatnonzero(frame.split.to_numpy() == "validation")
    base_valid, base_test, base_info = trained_predict(frame, test_frame, names[:188], chosen)
    reference_base = np.asarray(cache["full_train"]["base_validation_db"], dtype=float)
    c.require(np.max(np.abs(base_valid - reference_base)) < 1e-6,
              "Refit Base188 differs from locked validation predictor")
    direct_valid, direct_test, direct_info = trained_predict(frame, test_frame, names, chosen)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    reference_direct = json.loads((OUT / "validation" /
                                   f"{chosen['name']}_seed17.json").read_text(encoding="utf-8"))
    c.require(np.max(np.abs(direct_valid - np.asarray(reference_direct["validation_prediction_db"]))) < 1e-6,
              "Refit full348 differs from locked model-selection predictor")
    fit = np.flatnonzero(frame.split.to_numpy() == "train")
    residual_target = np.asarray(cache["full_train"]["residual_train_db"], dtype=float)
    predictions = {"Base188": base_test, "Direct348": direct_test}
    residual_info = {}
    for label, record in (("SelectedResidual", selected), ("Full10Residual", all_layers)):
        if record["mask"] == 0:
            predictions[label] = base_test.copy()
            residual_info[label] = {"mask": 0, "alpha": 0.0, "formal_rounds": 0,
                                    "seed": 17}
            continue
        features = record["features"]
        model, imputer, history, _, _, seconds = fit_g(
            c.raw(frame, fit, features), residual_target, None, None,
            rounds=record["formal_rounds"])
        g_valid = np.asarray(model.predict(imputer.transform(c.raw(frame, valid, features))), dtype=float)
        c.require(np.max(np.abs(g_valid - np.asarray(record["g_validation_db"]))) < 1e-6,
                  f"Refit {label} differs from locked validation predictor")
        g_test = np.asarray(model.predict(imputer.transform(
            c.raw(test_frame, np.arange(len(test_frame)), features))), dtype=float)
        predictions[label] = base_test + float(record["alpha"]) * g_test
        residual_info[label] = {"mask": record["mask"], "levels_hpa": record["levels_hpa"],
                                "seed": 17,
                                "alpha": record["alpha"], "formal_rounds": record["formal_rounds"],
                                "config": c.RESIDUAL_CONFIG,
                                "fit_seconds": seconds, "round_history": history}
    result = {"fingerprint": fingerprint, "status": "TEST_WRITTEN_PENDING_QA",
              "test_parquet_sha256": c.prior.sha256(c.DATA / "test.parquet"),
              "n_train": 2404, "n_test": len(test_frame),
              "prediction_seed": 17,
              "selected_mask_train_only": mask,
              "hour_bjt": test_frame.hour_bjt.dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist(),
              "truth_db": test_frame[c.TARGET].to_numpy(dtype=float).tolist(),
              "events": {event: test_frame[event].astype(bool).tolist()
                         for event in ("clear_dry", "rain", "heavy_rain", "dry_overcast", "snow")},
              "predictions_db": {key: val.tolist() for key, val in predictions.items()},
              "metrics": {key: c.metrics(test_frame, np.arange(len(test_frame)), val)
                          for key, val in predictions.items()},
              "base_fit": base_info, "direct_fit": direct_info,
              "residual_fits": residual_info,
              "no_test_based_model_change": True, "completed_utc": c.prior.utc_now()}
    c.prior.atomic_json(path, result)
    print(json.dumps({"status": result["status"], "selected_mask": mask,
                      "overall_test_rmse_db": {k: v["overall"]["rmse"]
                                               for k, v in result["metrics"].items()}},
                     ensure_ascii=False), flush=True)
    return result


def status() -> None:
    manifest = OUT / "layers" / "run_manifest.json"
    selection = OUT / "layers" / "layer_selection.json"
    qa = OUT / "layers" / "validation_qa.json"
    test = OUT / "layers" / "final_test.json"
    print(json.dumps({"status": "TEST_WRITTEN" if test.exists() else
                      "VALIDATED" if qa.exists() else "REPORTED" if selection.exists() else
                      "IN_PROGRESS" if manifest.exists() else "NOT_STARTED",
                      "baseline_cache": (OUT / "layers" / "baseline_cache.json").exists(),
                      "completed_subsets": len(list((OUT / "layers" / "subsets").glob("mask_*.json"))),
                      "selection": selection.exists(), "qa": qa.exists(), "test": test.exists()},
                     ensure_ascii=False, indent=2))


def smoke() -> None:
    frame, names, _ = c.load_training()
    active, features = columns_for_mask(names, 1)
    fit = np.flatnonzero(frame.split.to_numpy() == "train")[:120]
    hold = np.flatnonzero(frame.split.to_numpy() == "train")[120:160]
    residual_fit = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    residual_hold = frame.loc[hold, c.TARGET].to_numpy(dtype=float)
    _, _, history, best, pred, _ = fit_g(c.raw(frame, fit, features), residual_fit,
                                          c.raw(frame, hold, features), residual_hold,
                                          rounds=4)
    c.require(active == [0] and len(features) == 16 and len(history) == 4 and
              len(pred) == len(hold), "Layer smoke test failed")
    print(json.dumps({"status": "SMOKE_OK", "mask": 1, "best_round": best,
                      "holdout_rmse_db": c.rmse(residual_hold, pred)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "preflight", "baseline", "subsets", "report", "test", "status"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "status":
        status()
        return
    if args.command == "smoke":
        smoke()
        return
    frame, names, _ = c.load_training()
    folds = c.layer_folds(frame)
    chosen = winner()
    candidate = manifest_candidate(frame, names, folds, chosen)
    if args.command == "preflight":
        print(json.dumps({"status": "PREFLIGHT_OK", "fingerprint": candidate["fingerprint"],
                          "winner": chosen["name"], "config": chosen["best_config"],
                          "formal_rounds": chosen["formal_rounds"]}, ensure_ascii=False, indent=2))
        return
    with c.writer_lock():
        manifest = c.check_manifest(OUT / "layers" / "run_manifest.json", candidate)
        fingerprint = manifest["fingerprint"]
        cache = baseline_cache(frame, names, folds, chosen, fingerprint)
        if args.command == "baseline":
            return
        if args.command == "subsets":
            c.require(args.limit is None or args.limit > 0, "--limit must be positive")
            run_subsets(frame, names, cache, fingerprint, args.limit)
            return
        if args.command == "report":
            run_report(fingerprint)
            return
        run_test(frame, names, chosen, cache, fingerprint)


if __name__ == "__main__":
    main()
