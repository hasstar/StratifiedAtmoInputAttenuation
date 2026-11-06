"""Resumable BiLSTM 1024-subset refits and independent 156-to-1 refitted RFE."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import joblib
import numpy as np

import engine as e


def state_path(kind: str, key: int) -> Path:
    stem = f"mask_{key:04d}" if kind == "layer" else f"k_{key:03d}"
    return e.OUT / kind / "states" / f"{stem}.json"


def model_path(kind: str, key: int) -> Path:
    stem = f"mask_{key:04d}" if kind == "layer" else f"k_{key:03d}"
    return e.OUT / kind / "models" / f"{stem}.joblib"


def evaluation_path(kind: str, key: int) -> Path:
    stem = f"mask_{key:04d}" if kind == "layer" else f"k_{key:03d}"
    return e.OUT / kind / "evaluation" / f"{stem}.json"


def qa_path(kind: str, stage: str) -> Path:
    return e.OUT / kind / f"{stage}_qa.json"


def selection_path(kind: str) -> Path:
    return e.OUT / kind / "train_selection.json"


def names_for_mask(mask: int, retained: list[str]) -> list[str]:
    active = {level for index, level in enumerate(e.LEVELS) if mask & (1 << index)}
    return [name for name in retained if name.startswith("surface__") or
            any(name.startswith(f"p{level}__") for level in active)]


def make_state(kind: str, key: int, names: list[str], frame, folds: list[dict],
               manifest: dict) -> dict:
    y = frame[e.TARGET].to_numpy(dtype=float)
    fold_records = []
    importance_rows = []
    if kind == "rfe" and key > 1:
        importance_rows = [{"name": name, "fold_delta_rmse_db": []} for name in names]
    for fold in folds:
        fit, hold = fold["fit"], fold["hold"]
        bundle, xf = e.make_bundle(frame, fit, names, manifest["original_names"])
        xh = e.transform(bundle, e.raw_x(frame.iloc[hold], names))
        bundle, history, seconds = e.train_model(bundle, xf, y[fit], xh, y[hold])
        fit_pred = bundle["model"].predict(xf)
        hold_pred = bundle["model"].predict(xh)
        hold_frame = frame.iloc[hold]
        fit_scores = e.scores(y[fit], fit_pred, frame.iloc[fit])
        hold_scores = e.scores(y[hold], hold_pred, hold_frame)
        baseline = hold_scores["overall"]["rmse"]
        if importance_rows:
            for column, item in enumerate(importance_rows):
                changed = xh.copy()
                rng = np.random.default_rng(e.perm_seed(manifest, key, fold["fold"],
                                                        item["name"]))
                slot = bundle["slot_indices"][column]
                changed[:, slot] = xh[rng.permutation(len(hold)), slot]
                permuted = bundle["model"].predict(changed)
                item["fold_delta_rmse_db"].append(e.rmse(y[hold], permuted) - baseline)
        fold_records.append({"fold": fold["fold"], "n_fit": len(fit),
                             "n_hold": len(hold), "fit_days": fold["fit_days"],
                             "hold_days": fold["hold_days"],
                             "fit_index_sha256": fold["fit_index_sha256"],
                             "hold_index_sha256": fold["hold_index_sha256"],
                             "fit_seconds": seconds, "trained_epochs": len(history),
                             "stop_reason": manifest["stopping_reason"],
                             "epoch_history": history, "fit_metrics": fit_scores,
                             "hold_metrics": hold_scores,
                             "hold_predictions": e.hourly(hold_frame, y[hold], hold_pred)})
    for item in importance_rows:
        item["mean_delta_rmse_db"] = float(np.mean(item["fold_delta_rmse_db"]))
    removed = (min(importance_rows,
                   key=lambda item: (item["mean_delta_rmse_db"], item["name"]))["name"]
               if importance_rows else None)
    full, xf = e.make_bundle(frame, np.arange(len(frame)), names,
                             manifest["original_names"])
    full, full_history, full_seconds = e.train_model(full, xf, y)
    full_pred = full["model"].predict(xf)
    target = model_path(kind, key)
    model_hash = e.save_bundle(target, {"fingerprint": manifest["fingerprint"],
                                        "kind": kind, "key": key, "bundle": full})
    rmse_by_fold = np.array([row["hold_metrics"]["overall"]["rmse"]
                             for row in fold_records], dtype=float)
    state = {"fingerprint": manifest["fingerprint"], "kind": kind, "key": key,
             "feature_names": names, "n_features": len(names),
             "n_base": sum(name.startswith("surface__") for name in names),
             "n_pressure": sum(not name.startswith("surface__") for name in names),
             "active_positions": list(e.pressure_positions(names)),
             "levels_hpa": [e.LEVELS[index] for index in e.pressure_positions(names)],
             "cv_mean_rmse_db": float(rmse_by_fold.mean()),
             "cv_se_rmse_db": float(rmse_by_fold.std(ddof=1) / np.sqrt(3)),
             "folds": fold_records, "permutation_importance": importance_rows,
             "removed_feature_next": removed,
             "full_train_fit": {"n": len(frame), "fit_seconds": full_seconds,
                                "trained_epochs": len(full_history),
                                "stop_reason": manifest["stopping_reason"],
                                "epoch_history": full_history,
                                "metrics": e.scores(y, full_pred, frame)},
             "model_sha256": model_hash, "completed_utc": e.now()}
    e.atomic_json(state_path(kind, key), state)
    return state


def check_existing(kind: str, key: int, expected_names: list[str], fp: str) -> dict:
    state = e.read(state_path(kind, key))
    target = model_path(kind, key)
    e.require(state["fingerprint"] == fp and state["kind"] == kind and
              state["key"] == key and state["feature_names"] == expected_names and
              target.exists() and state["model_sha256"] == e.sha(target),
              f"Existing {kind} checkpoint {key} is inconsistent")
    return state


def prepare() -> dict:
    with e.writer_lock():
        manifest = e.get_manifest()
    return {"status": "prepared", "fingerprint": manifest["fingerprint"],
            "n_features": len(manifest["retained_names"]), "n_masks": 1024}


def smoke() -> dict:
    manifest = e.get_manifest(False)
    frame = e.frame_for("train", manifest["retained_names"]).iloc[:150]
    names = names_for_mask(1023, manifest["retained_names"])
    bundle, xf = e.make_bundle(frame, np.arange(120), names, manifest["original_names"])
    import torch
    from models import VerticalProfileNet
    net = VerticalProfileNet("bilstm", bundle["active_positions"], e.CONFIG["dropout"])
    x = torch.as_tensor(xf[:8], dtype=torch.float32)
    target = torch.as_tensor(frame[e.TARGET].iloc[:8].to_numpy(), dtype=torch.float32)
    loss = torch.mean((net(x) - target) ** 2)
    loss.backward()
    e.require(torch.isfinite(loss).item(), "Nonfinite BiLSTM smoke loss")
    return {"status": "PASS", "train_only_smoke_loss": float(loss.item()),
            "full_layout_width": xf.shape[1]}


def layer_run() -> dict:
    with e.writer_lock():
        manifest = e.get_manifest(False)
        frame = e.frame_for("train", manifest["retained_names"])
        fold_list = e.folds(frame)
        for mask in range(1024):
            names = names_for_mask(mask, manifest["retained_names"])
            path = state_path("layer", mask)
            if path.exists():
                check_existing("layer", mask, names, manifest["fingerprint"])
            else:
                state = make_state("layer", mask, names, frame, fold_list, manifest)
                print(json.dumps({"completed_mask": mask,
                                  "cv_rmse_db": state["cv_mean_rmse_db"],
                                  "n_features": state["n_features"]}), flush=True)
    return {"status": "layer_run_complete", "states": 1024, "refits": 4096}


def choose_layer(rows: list[dict]) -> dict:
    best = min(rows, key=lambda row: (row["cv_mean_rmse_db"], row["key"]))
    threshold = best["cv_mean_rmse_db"] + best["cv_se_rmse_db"]
    eligible = [row for row in rows if row["cv_mean_rmse_db"] <= threshold]
    chosen = min(eligible, key=lambda row: (len(row["levels_hpa"]),
                                            row["cv_mean_rmse_db"], row["key"]))
    indexed = {row["key"]: row for row in rows}
    effects = []
    for index, level in enumerate(e.LEVELS):
        deltas = np.array([indexed[mask]["cv_mean_rmse_db"] -
                           indexed[mask | (1 << index)]["cv_mean_rmse_db"]
                           for mask in range(1024) if not mask & (1 << index)])
        effects.append({"level_hpa": level, "paired_contexts": len(deltas),
                        "mean_improvement_db": float(deltas.mean()),
                        "median_improvement_db": float(np.median(deltas)),
                        "fraction_helpful": float(np.mean(deltas > 0))})
    return {"rule": "training-only one-SE; fewest levels, then lowest CV RMSE",
            "minimum_cv_mask": best["key"], "minimum_cv_rmse_db": best["cv_mean_rmse_db"],
            "minimum_cv_se_db": best["cv_se_rmse_db"], "one_se_threshold_db": threshold,
            "n_eligible_masks": len(eligible), "selected_mask": chosen["key"],
            "selected_levels_hpa": chosen["levels_hpa"],
            "selected_cv_rmse_db": chosen["cv_mean_rmse_db"],
            "train_fold_level_effects": effects,
            "selection_uses_validation_or_test": False}


def verify_run(kind: str) -> dict:
    with e.writer_lock():
        manifest = e.get_manifest(False)
        e.require(not (e.OUT / kind / "evaluation").exists(),
                  f"{kind} holdout evaluation exists before train QA")
        frame = e.frame_for("train", manifest["retained_names"])
        fold_list = e.folds(frame)
        expected = 1024 if kind == "layer" else 156
        keys = range(1024) if kind == "layer" else range(156, 0, -1)
        names = manifest["retained_names"].copy()
        rows = []
        for key in keys:
            current_names = (names_for_mask(key, manifest["retained_names"])
                             if kind == "layer" else names)
            state = check_existing(kind, key, current_names, manifest["fingerprint"])
            e.require(state["n_features"] == len(current_names) and
                      len(state["folds"]) == 3 and
                      state["full_train_fit"]["trained_epochs"] == e.EPOCHS and
                      len(state["full_train_fit"]["epoch_history"]) == e.EPOCHS,
                      f"{kind} state structure/epoch count differs at {key}")
            fold_rmse = []
            for recorded, fold in zip(state["folds"], fold_list):
                e.require(recorded["fold"] == fold["fold"] and
                          recorded["fit_index_sha256"] == fold["fit_index_sha256"] and
                          recorded["hold_index_sha256"] == fold["hold_index_sha256"] and
                          recorded["trained_epochs"] == e.EPOCHS and
                          len(recorded["epoch_history"]) == e.EPOCHS,
                          f"{kind} fold/epoch mismatch at {key}")
                e.check_scores(recorded["hold_metrics"], recorded["hold_predictions"])
                fold_rmse.append(recorded["hold_metrics"]["overall"]["rmse"])
            fold_rmse = np.asarray(fold_rmse)
            e.require(np.isclose(state["cv_mean_rmse_db"], fold_rmse.mean()) and
                      np.isclose(state["cv_se_rmse_db"],
                                 fold_rmse.std(ddof=1) / np.sqrt(3)),
                      f"{kind} CV score mismatch at {key}")
            if kind == "rfe":
                importance = state["permutation_importance"]
                e.require(len(importance) == (key if key > 1 else 0),
                          f"RFE permutation count differs at {key}")
                if key > 1:
                    e.require([row["name"] for row in importance] == names and
                              all(len(row["fold_delta_rmse_db"]) == 3 and
                                  np.isclose(row["mean_delta_rmse_db"],
                                             np.mean(row["fold_delta_rmse_db"]))
                                  for row in importance),
                              f"RFE importance mismatch at {key}")
                    weakest = min(importance, key=lambda item:
                                  (item["mean_delta_rmse_db"], item["name"]))["name"]
                    e.require(state["removed_feature_next"] == weakest,
                              f"RFE deletion differs at {key}")
                    names = [name for name in names if name != weakest]
                else:
                    e.require(state["removed_feature_next"] is None,
                              "RFE k=1 has an extra deletion")
            model_record = joblib.load(model_path(kind, key))
            bundle = model_record["bundle"]
            e.require(model_record["fingerprint"] == manifest["fingerprint"] and
                      model_record["kind"] == kind and model_record["key"] == key and
                      bundle["names"] == current_names and
                      bundle["model"].net is not None and
                      len(bundle["active_positions"]) == len(state["levels_hpa"]),
                      f"{kind} model identity differs at {key}")
            rows.append(state)
        e.require(len(list((e.OUT / kind / "states").glob("*.json"))) == expected and
                  len(list((e.OUT / kind / "models").glob("*.joblib"))) == expected,
                  f"{kind} state/model inventory differs")
        if kind == "layer":
            choice = choose_layer(rows)
        else:
            best = min(rows, key=lambda row: (row["cv_mean_rmse_db"], row["key"]))
            threshold = best["cv_mean_rmse_db"] + best["cv_se_rmse_db"]
            chosen = min((row for row in rows if row["cv_mean_rmse_db"] <= threshold),
                         key=lambda row: (row["key"], row["cv_mean_rmse_db"]))
            choice = {"rule": "training-only one-SE; smallest k",
                      "minimum_cv_k": best["key"],
                      "minimum_cv_rmse_db": best["cv_mean_rmse_db"],
                      "minimum_cv_se_db": best["cv_se_rmse_db"],
                      "one_se_threshold_db": threshold, "selected_k": chosen["key"],
                      "selected_feature_names": chosen["feature_names"],
                      "selection_uses_validation_or_test": False}
        choice.update({"fingerprint": manifest["fingerprint"], "frozen_utc": e.now()})
        if selection_path(kind).exists():
            frozen = e.read(selection_path(kind))
            e.require(frozen["fingerprint"] == choice["fingerprint"] and
                      frozen.get("selected_mask", frozen.get("selected_k")) ==
                      choice.get("selected_mask", choice.get("selected_k")),
                      f"{kind} train-only choice changed")
        else:
            e.atomic_json(selection_path(kind), choice)
        qa = {"status": "PASS", "fingerprint": manifest["fingerprint"],
              "n_states": expected, "n_models": expected,
              "n_refits": expected * 4, "n_deletions": 155 if kind == "rfe" else None,
              "holdouts_not_read": True, "verified_utc": e.now()}
        e.atomic_json(qa_path(kind, "train"), qa)
        return qa


def run_rfe() -> dict:
    with e.writer_lock():
        manifest = e.get_manifest(False)
        frame = e.frame_for("train", manifest["retained_names"])
        fold_list = e.folds(frame)
        names = manifest["retained_names"].copy()
        for k in range(156, 0, -1):
            if state_path("rfe", k).exists():
                state = check_existing("rfe", k, names, manifest["fingerprint"])
            else:
                state = make_state("rfe", k, names, frame, fold_list, manifest)
                print(json.dumps({"completed_k": k,
                                  "cv_rmse_db": state["cv_mean_rmse_db"],
                                  "removed_next": state["removed_feature_next"]}), flush=True)
            if k > 1:
                e.require(state["removed_feature_next"] in names,
                          f"RFE deletion missing from k={k}")
                names = [name for name in names if name != state["removed_feature_next"]]
    return {"status": "rfe_run_complete", "states": 156, "refits": 624,
            "deletions": 155}


def evaluate(kind: str) -> dict:
    with e.writer_lock():
        manifest = e.get_manifest(False)
        qa = e.read(qa_path(kind, "train"))
        selected = e.read(selection_path(kind))
        e.require(qa["status"] == "PASS" and
                  qa["fingerprint"] == selected["fingerprint"] == manifest["fingerprint"],
                  f"{kind} train-only QA/selection gate failed")
        frames = {split: e.frame_for(split, manifest["retained_names"])
                  for split in ("validation", "test")}
        keys = range(1024) if kind == "layer" else range(156, 0, -1)
        for key in keys:
            path = evaluation_path(kind, key)
            state = e.read(state_path(kind, key))
            if path.exists():
                saved = e.read(path)
                e.require(saved["fingerprint"] == manifest["fingerprint"] and
                          saved["model_sha256"] == state["model_sha256"] and
                          saved["key"] == key, f"{kind} evaluation checkpoint changed")
                continue
            model_record = joblib.load(model_path(kind, key))
            bundle = model_record["bundle"]
            e.require(bundle["names"] == state["feature_names"],
                      f"{kind} saved model names differ at {key}")
            results = {}
            for split, frame in frames.items():
                truth = frame[e.TARGET].to_numpy(dtype=float)
                pred = e.predict(bundle, e.raw_x(frame, bundle["names"]))
                results[split] = {"metrics": e.scores(truth, pred, frame),
                                  "hourly_predictions": e.hourly(frame, truth, pred)}
            e.atomic_json(path, {"fingerprint": manifest["fingerprint"],
                                 "kind": kind, "key": key,
                                 "feature_names": bundle["names"],
                                 "model_sha256": state["model_sha256"],
                                 "results": results, "scored_utc": e.now(),
                                 "model_refitted_during_evaluation": False})
            print(json.dumps({"evaluated": kind, "key": key,
                              "test_rmse_db": results["test"]["metrics"]["overall"]["rmse"]}),
                  flush=True)
    return {"status": f"{kind}_evaluation_complete",
            "n_models_scored": 1024 if kind == "layer" else 156}


def verify_evaluation(kind: str) -> dict:
    with e.writer_lock():
        manifest = e.get_manifest(False)
        train_qa = e.read(qa_path(kind, "train"))
        chosen = e.read(selection_path(kind))
        e.require(train_qa["status"] == "PASS" and
                  train_qa["fingerprint"] == chosen["fingerprint"] == manifest["fingerprint"],
                  f"{kind} training gate invalid")
        expected = 1024 if kind == "layer" else 156
        e.require(len(list((e.OUT / kind / "evaluation").glob("*.json"))) == expected,
                  f"{kind} evaluation inventory differs")
        keys = range(1024) if kind == "layer" else range(156, 0, -1)
        for key in keys:
            state = e.read(state_path(kind, key))
            result = e.read(evaluation_path(kind, key))
            e.require(result["fingerprint"] == manifest["fingerprint"] and
                      result["key"] == key and
                      result["feature_names"] == state["feature_names"] and
                      result["model_sha256"] == state["model_sha256"] and
                      result["model_refitted_during_evaluation"] is False and
                      result["scored_utc"] >= chosen["frozen_utc"],
                      f"{kind} evaluation identity/order differs at {key}")
            for split, count in (("validation", 343), ("test", 687)):
                record = result["results"][split]
                e.require(len(record["hourly_predictions"]["truth_db"]) == count,
                          f"{kind} {split} count differs at {key}")
                e.check_scores(record["metrics"], record["hourly_predictions"])
        qa = {"status": "PASS", "fingerprint": manifest["fingerprint"],
              "n_models_scored": expected, "validation_prediction_count": expected * 343,
              "test_prediction_count": expected * 687,
              "choice_frozen_before_holdout": True,
              "no_refit_in_evaluation": True,
              "test_historically_accessed": True,
              "verified_utc": e.now()}
        e.atomic_json(qa_path(kind, "evaluation"), qa)
        return qa


def report(kind: str) -> dict:
    with e.writer_lock():
        manifest = e.get_manifest(False)
        train_qa = e.read(qa_path(kind, "train"))
        eval_qa = e.read(qa_path(kind, "evaluation"))
        chosen = e.read(selection_path(kind))
        e.require(train_qa["status"] == eval_qa["status"] == "PASS" and
                  train_qa["fingerprint"] == eval_qa["fingerprint"] ==
                  chosen["fingerprint"] == manifest["fingerprint"],
                  f"{kind} final QA gate failed")
        keys = range(1024) if kind == "layer" else range(156, 0, -1)
        rows, name_rows = [], []
        for key in keys:
            state = e.read(state_path(kind, key))
            evaluation = e.read(evaluation_path(kind, key))
            row = {"mask" if kind == "layer" else "k": key,
                   "n_features": state["n_features"], "n_base": state["n_base"],
                   "n_pressure": state["n_pressure"],
                   "levels_hpa": ";".join(map(str, state["levels_hpa"])),
                   "cv_mean_rmse_db": state["cv_mean_rmse_db"],
                   "cv_se_rmse_db": state["cv_se_rmse_db"],
                   "train_rmse_db": state["full_train_fit"]["metrics"]["overall"]["rmse"],
                   "epochs": state["full_train_fit"]["trained_epochs"]}
            if kind == "rfe":
                row["removed_feature_next"] = state["removed_feature_next"]
            for split in ("validation", "test"):
                for event in ("overall", *e.EVENTS):
                    metric = evaluation["results"][split]["metrics"][event]
                    for field in ("n", "rmse", "mae", "bias"):
                        row[f"{split}_{event}_{field}"] = metric[field]
            rows.append(row)
            name_rows.append({"mask" if kind == "layer" else "k": key,
                              "feature_names_json": json.dumps(state["feature_names"],
                                                               ensure_ascii=False)})
        for filename, table in (("metrics_all_states.csv", rows),
                                ("feature_names_all_states.csv", name_rows)):
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
            e.atomic_text(e.OUT / kind / filename, buffer.getvalue())
        if kind == "layer":
            selected_key = chosen["selected_mask"]
            highlights = {label: rows[key] for label, key in
                          (("base", 0), ("full", 1023), ("train_selected", selected_key))}
            effects = []
            by_key = {row["mask"]: row for row in rows}
            for index, level in enumerate(e.LEVELS):
                deltas = np.array([by_key[m]["test_overall_rmse"] -
                                   by_key[m | (1 << index)]["test_overall_rmse"]
                                   for m in range(1024) if not m & (1 << index)])
                effects.append({"level_hpa": level,
                                "mean_test_improvement_db": float(deltas.mean()),
                                "median_test_improvement_db": float(np.median(deltas)),
                                "fraction_test_contexts_helpful": float(np.mean(deltas > 0))})
            final = {"selected_mask_train_one_se": selected_key,
                     "train_level_effects": chosen["train_fold_level_effects"],
                     "descriptive_test_level_effects": effects,
                     "highlights": highlights}
        else:
            selected_key = chosen["selected_k"]
            by_key = {row["k"]: row for row in rows}
            final = {"selected_k_train_one_se": selected_key,
                     "selected_feature_names": chosen["selected_feature_names"],
                     "k10_feature_names": e.read(state_path("rfe", 10))["feature_names"],
                     "highlights": {label: by_key[key] for label, key in
                                    (("full", 156), ("train_selected", selected_key),
                                     ("k10", 10), ("k1", 1))}}
        final.update({"fingerprint": manifest["fingerprint"], "kind": kind,
                      "n_states": len(rows), "n_refits": len(rows) * 4,
                      "test_historically_accessed": True,
                      "test_curve_descriptive_not_selection": True,
                      "same_day_train_test_overlap": True, "created_utc": e.now()})
        e.atomic_json(e.OUT / kind / "final_report.json", final)
        title = ("1024 pressure-subset direct refits" if kind == "layer"
                 else "156-to-1 column permutation RFE")
        prose = [f"# Revision 025 BiLSTM — {title}", "",
                 f"QA PASS: {len(rows)} states, {len(rows)*4} new BiLSTM fits at 62 epochs each. "
                 "The 156-column screen and BiLSTM hyperparameters came from Revision 023. "
                 "No removed pressure subset or RFE column was evaluated by merely masking a fixed model.", "",
                 "Every retained feature list, epoch curve, train-fold hourly prediction, "
                 "fitted model digest, validation/test hourly prediction and weather metric "
                 "is saved in the states, models, evaluation and CSV files.", "",
                 "| State | CV RMSE (dB) | Test overall RMSE (dB) | Test rain RMSE (dB) | Test dry-overcast RMSE (dB) |",
                 "|---|---:|---:|---:|---:|"]
        for label, row in final["highlights"].items():
            prose.append(f"| {label} | {row['cv_mean_rmse_db']:.4f} | "
                         f"{row['test_overall_rmse']:.4f} | {row['test_rain_rmse']:.4f} | "
                         f"{row['test_dry_overcast_rmse']:.4f} |")
        prose += ["", "The train-only one-SE rule froze the illustrative state before validation/test scoring. "
                  "Full test curves and layer effects are descriptive, not a second model-selection rule. "
                  "Rain includes all strong-rain hours. The existing test data were previously viewed "
                  "and share calendar days with training; they are not fresh blind or unseen-day evidence.", ""]
        if kind == "rfe":
            prose += ["## Ten retained features", ""]
            prose += [f"{i}. `{name}`" for i, name in
                      enumerate(final["k10_feature_names"], 1)]
            prose.append("")
        e.atomic_text(e.OUT / kind / "RESULTS.md", "\n".join(prose))
        return {"status": "reported", "kind": kind, "selected": selected_key,
                "states": len(rows)}


def status() -> dict:
    result = {"manifest_exists": e.MANIFEST.exists(),
              "fingerprint": e.read(e.MANIFEST)["fingerprint"]
              if e.MANIFEST.exists() else None}
    for kind in ("layer", "rfe"):
        result[kind] = {"states": len(list((e.OUT / kind / "states").glob("*.json"))),
                        "models": len(list((e.OUT / kind / "models").glob("*.joblib"))),
                        "evaluations": len(list((e.OUT / kind / "evaluation").glob("*.json"))),
                        "train_qa": e.read(qa_path(kind, "train"))["status"]
                        if qa_path(kind, "train").exists() else None,
                        "evaluation_qa": e.read(qa_path(kind, "evaluation"))["status"]
                        if qa_path(kind, "evaluation").exists() else None,
                        "reported": (e.OUT / kind / "RESULTS.md").exists()}
    return result


def main() -> None:
    actions = {"prepare": prepare, "smoke": smoke, "layer-run": layer_run,
               "layer-verify-run": lambda: verify_run("layer"),
               "layer-evaluate": lambda: evaluate("layer"),
               "layer-verify-evaluation": lambda: verify_evaluation("layer"),
               "layer-report": lambda: report("layer"),
               "rfe-run": run_rfe,
               "rfe-verify-run": lambda: verify_run("rfe"),
               "rfe-evaluate": lambda: evaluate("rfe"),
               "rfe-verify-evaluation": lambda: verify_evaluation("rfe"),
               "rfe-report": lambda: report("rfe"), "status": status}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=actions)
    args = parser.parse_args()
    print(json.dumps(actions[args.command](), ensure_ascii=False, indent=2,
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
