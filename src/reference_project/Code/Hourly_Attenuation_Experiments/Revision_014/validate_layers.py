"""Recompute all 1024 saved subset scores, 5120 pairs, and final test metrics."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common as c
import run_layers as layers


def close(a, b, tol=1e-9) -> None:
    c.require(abs(float(a) - float(b)) <= tol, f"Metric mismatch {a} vs {b}")


def validate_metrics(frame: pd.DataFrame, positions: np.ndarray,
                     prediction: np.ndarray, saved: dict) -> None:
    replay = c.metrics(frame, positions, prediction)
    for group in c.GROUPS:
        c.require(replay[group]["n"] == saved[group]["n"] and
                  replay[group]["days"] == saved[group]["days"], "Event count/day mismatch")
        for metric in ("rmse", "mae", "bias"):
            if replay[group][metric] is None:
                c.require(saved[group][metric] is None, "Empty group metric mismatch")
            else:
                close(replay[group][metric], saved[group][metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "all"))
    args = parser.parse_args()
    frame, names, _ = c.load_training()
    folds = c.layer_folds(frame)
    chosen = layers.winner()
    candidate = layers.manifest_candidate(frame, names, folds, chosen)
    manifest = json.loads((c.OUT / "layers" / "run_manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest["fingerprint"]
    c.require(fingerprint == candidate["fingerprint"], "Source/data/choice fingerprint mismatch")
    cache = c.read_record(c.OUT / "layers" / "baseline_cache.json", fingerprint)
    c.require(cache is not None and len(cache["cv_folds"]) == 3,
              "Baseline cache missing/incomplete")
    for saved_fold, source_fold in zip(cache["cv_folds"], folds):
        fit = np.asarray(saved_fold["fit_indices"], dtype=int)
        hold = np.asarray(saved_fold["holdout_indices"], dtype=int)
        c.require(np.array_equal(fit, source_fold["train_index"]) and
                  np.array_equal(hold, source_fold["validation_index"]) and
                  set(fit).isdisjoint(set(hold)), "Layer fold indices changed")
        fit_days = set(frame.loc[fit, "hour_bjt"].dt.strftime("%Y-%m-%d"))
        hold_days = set(frame.loc[hold, "hour_bjt"].dt.strftime("%Y-%m-%d"))
        c.require(fit_days.isdisjoint(hold_days), "Same day crossed layer fit/hold")
        inner_hold = [np.asarray(row["holdout_global_indices"], dtype=int)
                      for row in saved_fold["inner_folds"]]
        c.require(len(inner_hold) == 5 and
                  sorted(np.concatenate(inner_hold).tolist()) == sorted(fit.tolist()),
                  "Inner OOF fold coverage incomplete/duplicated")
        for part in inner_hold:
            part_days = set(frame.loc[part, "hour_bjt"].dt.strftime("%Y-%m-%d"))
            other_days = set(frame.loc[np.setdiff1d(fit, part), "hour_bjt"].dt.strftime("%Y-%m-%d"))
            c.require(part_days.isdisjoint(other_days), "Same day crossed inner OOF fold")
        yfit = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
        yhold = frame.loc[hold, c.TARGET].to_numpy(dtype=float)
        c.require(np.allclose(yfit - np.asarray(saved_fold["oof_base_fit_db"]),
                              np.asarray(saved_fold["residual_fit_db"]), atol=1e-12) and
                  np.allclose(yhold - np.asarray(saved_fold["base_holdout_db"]),
                              np.asarray(saved_fold["residual_holdout_db"]), atol=1e-12),
                  "Saved residual targets do not equal truth minus out-of-fold baseline")
    final = cache["full_train"]
    full_fit = np.asarray(final["fit_indices"], dtype=int)
    full_inner = [np.asarray(row["holdout_global_indices"], dtype=int)
                  for row in final["inner_folds"]]
    c.require(len(full_inner) == 5 and
              sorted(np.concatenate(full_inner).tolist()) == sorted(full_fit.tolist()),
              "Full-train inner OOF coverage incomplete/duplicated")
    c.require(np.allclose(frame.loc[full_fit, c.TARGET].to_numpy(dtype=float) -
                          np.asarray(final["oof_base_train_db"]),
                          np.asarray(final["residual_train_db"]), atol=1e-12),
              "Full-train residual targets changed")
    valid = np.asarray(final["validation_indices"], dtype=int)
    base_valid = np.asarray(final["base_validation_db"], dtype=float)
    c.require(len(valid) == 343 and len(base_valid) == 343, "Validation baseline count")
    validate_metrics(frame, valid, base_valid, final["base_validation_metrics"])
    records = {}
    for mask in range(1024):
        path = c.OUT / "layers" / "subsets" / f"mask_{mask:04d}.json"
        record = c.read_record(path, fingerprint)
        c.require(record is not None and record["mask"] == mask, f"Subset missing: {mask}")
        active, expected_features = layers.columns_for_mask(names, mask)
        c.require(record["levels_hpa"] == [c.LEVELS[i] for i in active] and
                  record["features"] == expected_features and
                  record["n_layer_features"] == len(expected_features),
                  f"Subset feature list mismatch: {mask}")
        pred = np.asarray(record["validation_prediction_db"], dtype=float)
        g = np.asarray(record["g_validation_db"], dtype=float)
        c.require(len(pred) == len(g) == len(valid) and np.isfinite(pred).all(),
                  "Subset prediction length/finite mismatch")
        c.require(np.max(np.abs(pred - (base_valid + record["alpha"] * g))) < 1e-9,
                  "Saved subset prediction does not equal base+alpha*g")
        validate_metrics(frame, valid, pred, record["validation_metrics"])
        cv_y, cv_base, cv_g = [], [], []
        if mask == 0:
            c.require(record["alpha"] == 0 and record["formal_rounds"] == 0 and
                      record["folds"] == [] and np.all(g == 0), "Mask zero is not Base188")
        else:
            c.require(len(record["folds"]) == 3 and record["config"] == c.RESIDUAL_CONFIG,
                      "Residual model count/config mismatch")
            selected_rounds = []
            for saved_fold, base_fold in zip(record["folds"], cache["cv_folds"]):
                c.require(saved_fold["fold_id"] == base_fold["fold_id"] and
                          saved_fold["n_fit"] == len(base_fold["fit_indices"]) and
                          saved_fold["n_holdout"] == len(base_fold["holdout_indices"]),
                          "Subset training fold identity mismatch")
                curve = saved_fold["round_history"]
                c.require(len(curve) == 120 and all(
                    np.isfinite(row["train_residual_rmse_db"]) and
                    np.isfinite(row["holdout_residual_rmse_db"]) for row in curve),
                    "Subset per-tree history missing/nonfinite")
                best = min(curve, key=lambda row: (row["holdout_residual_rmse_db"], row["round"]))
                c.require(saved_fold["selected_round"] == best["round"],
                          "Selected residual round mismatch")
                selected_rounds.append(best["round"])
                cv_g.append(np.asarray(saved_fold["g_holdout_db"], dtype=float))
            c.require(record["formal_rounds"] == int(np.median(selected_rounds)) and
                      len(record["final_train_round_history"]) == record["formal_rounds"],
                      "Final residual round/history mismatch")
        for base_fold in cache["cv_folds"]:
            cv_y.append(np.asarray(base_fold["truth_holdout_db"], dtype=float))
            cv_base.append(np.asarray(base_fold["base_holdout_db"], dtype=float))
        truth = np.concatenate(cv_y)
        base = np.concatenate(cv_base)
        correction = np.zeros_like(base) if mask == 0 else np.concatenate(cv_g)
        scores = {str(alpha): c.rmse(truth, base + alpha * correction) for alpha in c.ALPHAS}
        for key, value in scores.items():
            close(value, record["alpha_scores"][key])
        selected_alpha = min(c.ALPHAS, key=lambda a: (scores[str(a)], a))
        c.require(record["alpha"] == selected_alpha, "Residual alpha selection mismatch")
        close(scores[str(selected_alpha)], record["cv_pooled_rmse_db"])
        records[mask] = record
    selection = c.read_record(c.OUT / "layers" / "layer_selection.json", fingerprint)
    c.require(selection is not None and selection["n_subsets"] == 1024 and
              selection["n_layer_pairs"] == 5120, "Layer analysis incomplete")
    selected = min(records.values(), key=lambda item: (item["cv_pooled_rmse_db"],
                                                    len(item["levels_hpa"]), item["mask"]))
    c.require(selection["selected_mask_train_only"] == selected["mask"] and
              selection["selected_alpha"] == selected["alpha"],
              "Final mask not selected from training-internal CV")
    pairwise = pd.read_csv(c.OUT / "layers" / "layer_pairwise.csv")
    c.require(len(pairwise) == 5120, "Pair table count mismatch")
    for row in pairwise.itertuples(index=False):
        c.require(int(row.with_mask) == (int(row.without_mask) | (1 << c.LEVELS.index(int(row.level_hpa)))),
                  "Layer pair mask mismatch")
        for group in c.GROUPS:
            value = getattr(row, f"validation_{group}_rmse_improvement_db")
            a = records[int(row.without_mask)]["validation_metrics"][group]["rmse"]
            b = records[int(row.with_mask)]["validation_metrics"][group]["rmse"]
            if a is None or b is None:
                c.require(pd.isna(value), "Empty event pair should be NaN")
            else:
                close(value, a - b)
    result = {"status": "PASS", "stage": args.stage, "fingerprint": fingerprint,
              "validated_subsets": 1024, "validated_layer_pairs": 5120,
              "selected_mask_train_only": selected["mask"],
              "all_validation_metrics_recomputed": True,
              "validated_utc": c.prior.utc_now()}
    if args.stage == "all":
        test = c.read_record(c.OUT / "layers" / "final_test.json", fingerprint)
        c.require(test is not None and test["status"] == "TEST_WRITTEN_PENDING_QA" and
                  test["selected_mask_train_only"] == selected["mask"] and
                  test["no_test_based_model_change"], "Final test record missing/changed")
        test_frame = c.residual.load_test(names)
        truth = np.asarray(test["truth_db"], dtype=float)
        c.require(len(truth) == len(test_frame) == 687 and
                  np.array_equal(truth, test_frame[c.TARGET].to_numpy(dtype=float)),
                  "Test target/order mismatch")
        positions = np.arange(687)
        c.require(set(test["predictions_db"]) ==
                  {"Base188", "Direct348", "SelectedResidual", "Full10Residual"},
                  "Unexpected final test arms")
        for arm, values in test["predictions_db"].items():
            prediction = np.asarray(values, dtype=float)
            c.require(len(prediction) == 687 and np.isfinite(prediction).all(),
                      "Invalid final test predictions")
            validate_metrics(test_frame, positions, prediction, test["metrics"][arm])
        c.require(test["test_parquet_sha256"] == c.prior.sha256(c.DATA / "test.parquet"),
                  "Test parquet hash changed")
        result["n_test_hours"] = 687
        result["all_test_metrics_recomputed"] = True
    path = c.OUT / "layers" / ("final_qa.json" if args.stage == "all" else "validation_qa.json")
    c.prior.atomic_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
