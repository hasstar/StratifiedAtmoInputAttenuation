"""Train-only permutation RFE with a refitted RBF-SVR at every feature count.

Only the explicit ``evaluate`` command reads validation or test data. Older
experiments and all source data are read-only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_024_SVR_RFE"
PARENT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation"
DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
TARGET = "clear_sky_attenuation_db"
EVENTS = ("rain", "dry_overcast", "heavy_rain")
CONFIG = {"C": 10.0, "epsilon": 0.1, "gamma": "scale"}
VERSION = "revision024-svr-permutation-rfe-v1"
MANIFEST = OUT / "run_manifest.json"
SELECTION = OUT / "train_selection.json"
RUN_QA = OUT / "run_qa.json"
EVALUATION_QA = OUT / "evaluation_qa.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def writer_lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rfe.writer.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another Revision 024 writer is active") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def parent_plan() -> list[str]:
    feature_qa = read(PARENT / "feature_selection_qa.json")
    model_qa = read(PARENT / "model_selection_qa.json")
    layer_qa = read(PARENT / "layer_qa.json")
    require(all(item.get("status") == "PASS" for item in
                (feature_qa, model_qa, layer_qa)), "Revision 023 QA gate failed")
    names = read(PARENT / "feature_selection.json")["feature_names_retained"]
    parent_state = read(PARENT / "layer_subsets/mask_1023.json")
    parent_manifest = read(PARENT / "layer_manifest.json")
    require(len(names) == 156 and len(set(names)) == 156 and
            parent_state["feature_names"] == names,
            "The frozen full-layer 156-column input changed")
    require(parent_state["selected_config_id"] == 1 and
            parent_manifest["configs"][1] == CONFIG and
            parent_state["fingerprint"] == parent_manifest["fingerprint"],
            "The frozen full-layer SVR configuration changed")
    return names


def frame_for(split: str, names: list[str]) -> pd.DataFrame:
    require(split in ("train", "validation", "test"), "Unknown partition")
    frame = pd.read_parquet(DATA / f"{split}.parquet",
                            columns=["hour_bjt", *names, TARGET])
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet",
                             columns=["hour_bjt", "split", *EVENTS])
    frame = frame.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    expected = {"train": 2404, "validation": 343, "test": 687}[split]
    require(len(frame) == expected and frame["split"].eq(split).all() and
            frame[TARGET].notna().all() and
            frame["hour_bjt"].is_unique, f"{split} partition/schema changed")
    return frame.sort_values("hour_bjt").reset_index(drop=True)


def x_y(frame: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[names].to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    y = frame[TARGET].to_numpy(dtype=float)
    require(np.isfinite(y).all(), "Nonfinite target")
    return x, y


def grouped_folds(frame: pd.DataFrame) -> list[dict]:
    days = frame["hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    splitter = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=20261002)
    result = []
    for fold_id, (fit, hold) in enumerate(splitter.split(frame, groups=days), 1):
        require(set(days[fit]).isdisjoint(days[hold]), "Fold day overlap")
        result.append({"fold": fold_id, "fit": fit, "hold": hold,
                       "fit_days": len(set(days[fit])),
                       "hold_days": len(set(days[hold])),
                       "fit_index_sha256": fingerprint(fit.tolist()),
                       "hold_index_sha256": fingerprint(hold.tolist())})
    return result


def estimator() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", **CONFIG)),
    ])


def metrics(truth: np.ndarray, pred: np.ndarray, labels: dict) -> dict:
    require(len(truth) == len(pred) and np.isfinite(pred).all(),
            "Invalid prediction values")
    masks = {"overall": np.ones(len(truth), dtype=bool)}
    masks.update({event: np.asarray(labels[event], dtype=bool) for event in EVENTS})
    result = {}
    for key, mask in masks.items():
        err = pred[mask] - truth[mask]
        result[key] = {"n": int(mask.sum()),
                       "rmse": float(np.sqrt(np.mean(err * err))) if len(err) else None,
                       "mae": float(np.mean(np.abs(err))) if len(err) else None,
                       "bias": float(np.mean(err)) if len(err) else None}
    return result


def labels_for(frame: pd.DataFrame) -> dict:
    return {event: frame[event].to_numpy(dtype=bool) for event in EVENTS}


def source_paths() -> list[Path]:
    return [DATA / "train.parquet", DATA / "hourly_sample_status.parquet",
            PARENT / "feature_selection.json",
            PARENT / "feature_selection_qa.json",
            PARENT / "model_selection_qa.json",
            PARENT / "layer_manifest.json", PARENT / "layer_qa.json",
            PARENT / "layer_subsets/mask_1023.json",
            PARENT / "layer_models/mask_1023.joblib",
            HERE / "PROTOCOL.md", HERE / "run_rfe.py"]


def candidate_manifest() -> dict:
    names = parent_plan()
    frame = frame_for("train", names)
    folds = grouped_folds(frame)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}
    payload = {"version": VERSION, "source_sha256": sources,
               "target": TARGET, "n_train": len(frame),
               "feature_names_initial": names, "n_features_initial": len(names),
               "estimator": {"pipeline": ["median imputer", "StandardScaler", "SVR"],
                             "kernel": "rbf", **CONFIG},
               "folds": [{key: value for key, value in fold.items()
                          if key not in ("fit", "hold")} for fold in folds],
               "permutation": "one deterministic shuffle per column and fold; mean RMSE increase",
               "deletion": "smallest mean increase, tie by feature name",
               "operating_k_rule": "smallest k within one SE of minimum train-fold mean RMSE",
               "holdouts_not_read": True}
    payload["fingerprint"] = fingerprint(payload)
    payload["created_utc"] = now()
    return payload


def get_manifest() -> dict:
    candidate = candidate_manifest()
    if MANIFEST.exists():
        frozen = read(MANIFEST)
        require(candidate["fingerprint"] == frozen["fingerprint"],
                "Frozen source/protocol/config/fold fingerprint changed")
        return frozen
    atomic_json(MANIFEST, candidate)
    return candidate


def state_path(k: int) -> Path:
    return OUT / "states" / f"k_{k:03d}.json"


def model_path(k: int) -> Path:
    return OUT / "models" / f"k_{k:03d}.joblib"


def evaluation_path(k: int) -> Path:
    return OUT / "evaluation" / f"k_{k:03d}.json"


def perm_seed(manifest: dict, k: int, fold: int, name: str) -> int:
    content = f"{manifest['fingerprint']}|{k}|{fold}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "big")


def score_record(frame: pd.DataFrame, truth: np.ndarray, pred: np.ndarray) -> dict:
    return {"hour_bjt": frame["hour_bjt"].astype(str).tolist(),
            "truth_db": truth.tolist(), "prediction_db": pred.tolist(),
            **{event: frame[event].astype(bool).tolist() for event in EVENTS}}


def fit_state(manifest: dict, frame: pd.DataFrame, folds: list[dict],
              names: list[str]) -> dict:
    k = len(names)
    x, y = x_y(frame, names)
    fold_rows = []
    importance_by_fold = np.zeros((3, k), dtype=float)
    for index, fold in enumerate(folds):
        fit, hold = fold["fit"], fold["hold"]
        started = time.perf_counter()
        model = estimator()
        model.fit(x[fit], y[fit])
        fit_seconds = time.perf_counter() - started
        train_pred = model.predict(x[fit])
        hold_pred = model.predict(x[hold])
        fit_frame = frame.iloc[fit]
        hold_frame = frame.iloc[hold]
        fit_metrics = metrics(y[fit], train_pred, labels_for(fit_frame))
        hold_metrics = metrics(y[hold], hold_pred, labels_for(hold_frame))
        baseline = hold_metrics["overall"]["rmse"]
        x_hold = x[hold]
        if k > 1:
            for col, name in enumerate(names):
                rng = np.random.default_rng(perm_seed(manifest, k, fold["fold"], name))
                altered = x_hold.copy()
                altered[:, col] = x_hold[rng.permutation(len(hold)), col]
                predicted = model.predict(altered)
                permuted = float(np.sqrt(np.mean((predicted - y[hold]) ** 2)))
                importance_by_fold[index, col] = permuted - baseline
        fold_rows.append({"fold": fold["fold"], "n_fit": len(fit),
                          "n_hold": len(hold), "fit_days": fold["fit_days"],
                          "hold_days": fold["hold_days"],
                          "fit_index_sha256": fold["fit_index_sha256"],
                          "hold_index_sha256": fold["hold_index_sha256"],
                          "fit_metrics": fit_metrics, "hold_metrics": hold_metrics,
                          "solver_iterations": int(model.named_steps["svr"].n_iter_),
                          "fit_seconds": fit_seconds,
                          "hold_predictions": score_record(hold_frame, y[hold], hold_pred)})
    mean_importance = importance_by_fold.mean(axis=0)
    importances = [
        {"name": name, "fold_delta_rmse_db": importance_by_fold[:, col].tolist(),
         "mean_delta_rmse_db": float(mean_importance[col])}
        for col, name in enumerate(names)
    ] if k > 1 else []
    removed = (min(importances, key=lambda row: (row["mean_delta_rmse_db"],
                                              row["name"]))["name"]
               if k > 1 else None)
    started = time.perf_counter()
    full_model = estimator()
    full_model.fit(x, y)
    full_fit_seconds = time.perf_counter() - started
    full_pred = full_model.predict(x)
    parent_max_abs_diff = None
    if k == 156:
        parent_bundle = joblib.load(PARENT / "layer_models/mask_1023.joblib")
        require(parent_bundle["names"] == names and parent_bundle["config"] == CONFIG,
                "Revision 023 parent model differs from frozen design")
        parent_pred = parent_bundle["model"].predict(x)
        parent_max_abs_diff = float(np.max(np.abs(parent_pred - full_pred)))
        require(parent_max_abs_diff < 1e-9,
                "Full-156 refit fails parent-model equivalence")
    target = model_path(k)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    joblib.dump({"fingerprint": manifest["fingerprint"], "k": k,
                 "names": names, "config": CONFIG, "model": full_model},
                temp, compress=3)
    os.replace(temp, target)
    fold_rmse = np.array([row["hold_metrics"]["overall"]["rmse"]
                          for row in fold_rows], dtype=float)
    state = {"fingerprint": manifest["fingerprint"], "k": k,
             "feature_names": names, "n_base": sum(n.startswith("surface__") for n in names),
             "n_pressure": sum(not n.startswith("surface__") for n in names),
             "folds": fold_rows, "cv_mean_rmse_db": float(fold_rmse.mean()),
             "cv_se_rmse_db": float(fold_rmse.std(ddof=1) / np.sqrt(3)),
             "permutation_importance": importances,
             "removed_feature_next": removed,
             "full_train_fit": {"n": len(frame),
                                "metrics": metrics(y, full_pred, labels_for(frame)),
                                "solver_iterations": int(full_model.named_steps["svr"].n_iter_),
                                "fit_seconds": full_fit_seconds,
                                "training_rounds": None,
                                "stop_reason": "libsvm optimization convergence; no epoch rounds",
                                "parent_k156_max_abs_prediction_diff_db": parent_max_abs_diff},
             "model_sha256": digest(target), "completed_utc": now()}
    atomic_json(state_path(k), state)
    return state


def prepare() -> dict:
    with writer_lock():
        manifest = get_manifest()
    return {"fingerprint": manifest["fingerprint"],
            "n_features": manifest["n_features_initial"], "status": "prepared"}


def run() -> dict:
    with writer_lock():
        manifest = get_manifest()
        frame = frame_for("train", manifest["feature_names_initial"])
        folds = grouped_folds(frame)
        names = manifest["feature_names_initial"].copy()
        for k in range(len(names), 0, -1):
            path = state_path(k)
            model_file = model_path(k)
            if path.exists():
                state = read(path)
                require(state["fingerprint"] == manifest["fingerprint"] and
                        state["k"] == k and state["feature_names"] == names and
                        model_file.exists() and
                        state["model_sha256"] == digest(model_file),
                        f"Existing k={k} checkpoint is inconsistent")
            else:
                state = fit_state(manifest, frame, folds, names)
                print(json.dumps({"completed_k": k,
                                  "cv_rmse_db": state["cv_mean_rmse_db"],
                                  "removed_next": state["removed_feature_next"]}),
                      flush=True)
            if k > 1:
                require(state["removed_feature_next"] in names,
                        f"Invalid removed feature at k={k}")
                names = [name for name in names if name != state["removed_feature_next"]]
        return {"status": "run_complete", "n_states": 156,
                "n_refits": 624, "n_deletions": 155}


def close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(a, b, rtol=tolerance, atol=tolerance))


def check_metrics(saved: dict, prediction: dict) -> None:
    truth = np.asarray(prediction["truth_db"], dtype=float)
    pred = np.asarray(prediction["prediction_db"], dtype=float)
    require(len(prediction["hour_bjt"]) == len(truth) == len(pred),
            "Hourly prediction length mismatch")
    expected = metrics(truth, pred, prediction)
    for group, expected_row in expected.items():
        row = saved[group]
        require(row["n"] == expected_row["n"], f"{group} sample count mismatch")
        for metric in ("rmse", "mae", "bias"):
            if expected_row[metric] is None:
                require(row[metric] is None, f"{group} null score mismatch")
            else:
                require(close(row[metric], expected_row[metric]),
                        f"{group} {metric} recomputation mismatch")


def verify_run() -> dict:
    with writer_lock():
        manifest = get_manifest()
        require(not (OUT / "evaluation").exists(),
                "Holdout evaluation exists before train-only QA")
        frame = frame_for("train", manifest["feature_names_initial"])
        folds = grouped_folds(frame)
        names = manifest["feature_names_initial"].copy()
        rows = []
        for k in range(156, 0, -1):
            state = read(state_path(k))
            model_file = model_path(k)
            require(state["fingerprint"] == manifest["fingerprint"] and
                    state["k"] == k and state["feature_names"] == names and
                    len(names) == k and state["model_sha256"] == digest(model_file),
                    f"State/model/path mismatch at k={k}")
            require(len(state["folds"]) == 3 and
                    len(state["permutation_importance"]) == (k if k > 1 else 0),
                    f"Fold/permutation count mismatch at k={k}")
            for recorded, fold in zip(state["folds"], folds):
                require(recorded["fold"] == fold["fold"] and
                        recorded["fit_index_sha256"] == fold["fit_index_sha256"] and
                        recorded["hold_index_sha256"] == fold["hold_index_sha256"] and
                        recorded["n_fit"] == len(fold["fit"]) and
                        recorded["n_hold"] == len(fold["hold"]),
                        f"Fold identity mismatch at k={k}")
                check_metrics(recorded["hold_metrics"], recorded["hold_predictions"])
                require(recorded["solver_iterations"] > 0 and
                        recorded["fit_seconds"] >= 0,
                        f"Fold training record invalid at k={k}")
            rmse = np.array([fold["hold_metrics"]["overall"]["rmse"]
                             for fold in state["folds"]])
            require(close(state["cv_mean_rmse_db"], rmse.mean()) and
                    close(state["cv_se_rmse_db"], rmse.std(ddof=1) / np.sqrt(3)),
                    f"Cross-fold score mismatch at k={k}")
            if k > 1:
                require([row["name"] for row in state["permutation_importance"]] == names,
                        f"Importance column order mismatch at k={k}")
                for row in state["permutation_importance"]:
                    require(len(row["fold_delta_rmse_db"]) == 3 and
                            close(row["mean_delta_rmse_db"],
                                  np.mean(row["fold_delta_rmse_db"])),
                            f"Importance average mismatch at k={k}")
                weakest = min(state["permutation_importance"],
                              key=lambda row: (row["mean_delta_rmse_db"], row["name"]))
                require(state["removed_feature_next"] == weakest["name"],
                        f"Deletion ranking mismatch at k={k}")
                names = [name for name in names if name != weakest["name"]]
            else:
                require(state["removed_feature_next"] is None,
                        "k=1 must not delete another feature")
            bundle = joblib.load(model_file)
            require(bundle["fingerprint"] == manifest["fingerprint"] and
                    bundle["k"] == k and bundle["names"] == state["feature_names"] and
                    bundle["config"] == CONFIG and
                    bundle["model"].named_steps["svr"].n_features_in_ == k and
                    state["full_train_fit"]["solver_iterations"] > 0,
                    f"Saved refit model mismatch at k={k}")
            rows.append(state)
        require(len(list((OUT / "states").glob("k_*.json"))) == 156 and
                len(list((OUT / "models").glob("k_*.joblib"))) == 156,
                "State/model file inventory mismatch")
        require(rows[0]["full_train_fit"]["parent_k156_max_abs_prediction_diff_db"] < 1e-9,
                "Parent 156-model equivalence missing")
        best = min(rows, key=lambda row: (row["cv_mean_rmse_db"], row["k"]))
        threshold = best["cv_mean_rmse_db"] + best["cv_se_rmse_db"]
        eligible = [row for row in rows if row["cv_mean_rmse_db"] <= threshold]
        chosen = min(eligible, key=lambda row: (row["k"], row["cv_mean_rmse_db"]))
        selected = {"fingerprint": manifest["fingerprint"],
                    "rule": manifest["operating_k_rule"],
                    "minimum_cv_k": best["k"],
                    "minimum_cv_rmse_db": best["cv_mean_rmse_db"],
                    "minimum_cv_se_db": best["cv_se_rmse_db"],
                    "one_se_threshold_db": threshold,
                    "selected_k": chosen["k"],
                    "selected_feature_names": chosen["feature_names"],
                    "selected_cv_rmse_db": chosen["cv_mean_rmse_db"],
                    "frozen_utc": now(), "validation_test_not_read": True}
        atomic_json(SELECTION, selected)
        qa = {"status": "PASS", "fingerprint": manifest["fingerprint"],
              "n_states": 156, "n_saved_models": 156,
              "n_deletions": 155, "n_refits": 624,
              "parent_model_equivalent": True,
              "selected_k": chosen["k"], "holdouts_not_read": True,
              "verified_utc": now()}
        atomic_json(RUN_QA, qa)
        return qa


def evaluate() -> dict:
    with writer_lock():
        manifest = get_manifest()
        qa = read(RUN_QA)
        selected = read(SELECTION)
        require(qa["status"] == "PASS" and
                qa["fingerprint"] == selected["fingerprint"] == manifest["fingerprint"],
                "Train-only QA/selection gate failed")
        names = manifest["feature_names_initial"]
        frames = {split: frame_for(split, names)
                  for split in ("validation", "test")}
        for k in range(156, 0, -1):
            path = evaluation_path(k)
            if path.exists():
                existing = read(path)
                require(existing["fingerprint"] == manifest["fingerprint"] and
                        existing["k"] == k and
                        existing["model_sha256"] == read(state_path(k))["model_sha256"],
                        f"Existing evaluation mismatch at k={k}")
                continue
            state = read(state_path(k))
            bundle = joblib.load(model_path(k))
            require(bundle["names"] == state["feature_names"],
                    f"Saved model input list changed at k={k}")
            results = {}
            for split, frame in frames.items():
                x, y = x_y(frame, bundle["names"])
                pred = bundle["model"].predict(x)
                results[split] = {"metrics": metrics(y, pred, labels_for(frame)),
                                  "hourly_predictions": score_record(frame, y, pred)}
            atomic_json(path, {"fingerprint": manifest["fingerprint"],
                               "k": k, "feature_names": bundle["names"],
                               "model_sha256": state["model_sha256"],
                               "results": results, "scored_utc": now(),
                               "model_refitted_during_evaluation": False})
            print(json.dumps({"evaluated_k": k,
                              "test_rmse_db": results["test"]["metrics"]["overall"]["rmse"]}),
                  flush=True)
        return {"status": "evaluation_complete", "n_models_scored": 156,
                "validation_hours_each": 343, "test_hours_each": 687}


def verify_evaluation() -> dict:
    with writer_lock():
        manifest = get_manifest()
        run_qa = read(RUN_QA)
        selected = read(SELECTION)
        require(run_qa["status"] == "PASS" and
                run_qa["fingerprint"] == selected["fingerprint"] == manifest["fingerprint"],
                "Training gate no longer valid")
        require(len(list((OUT / "evaluation").glob("k_*.json"))) == 156,
                "Evaluation record inventory mismatch")
        for k in range(156, 0, -1):
            state = read(state_path(k))
            record = read(evaluation_path(k))
            require(record["fingerprint"] == manifest["fingerprint"] and
                    record["k"] == k and
                    record["feature_names"] == state["feature_names"] and
                    record["model_sha256"] == state["model_sha256"] and
                    record["model_refitted_during_evaluation"] is False,
                    f"Evaluation identity mismatch at k={k}")
            require(record["scored_utc"] >= selected["frozen_utc"],
                    f"Holdout scored before k was frozen at k={k}")
            for split, expected_n in (("validation", 343), ("test", 687)):
                item = record["results"][split]
                require(len(item["hourly_predictions"]["truth_db"]) == expected_n,
                        f"Holdout count mismatch at k={k} {split}")
                check_metrics(item["metrics"], item["hourly_predictions"])
        qa = {"status": "PASS", "fingerprint": manifest["fingerprint"],
              "n_models_scored": 156, "n_validation_predictions": 156 * 343,
              "n_test_predictions": 156 * 687,
              "selection_frozen_before_holdout": True,
              "no_refit_in_evaluation": True,
              "test_historically_accessed": True, "verified_utc": now()}
        atomic_json(EVALUATION_QA, qa)
        return qa


def report() -> dict:
    with writer_lock():
        manifest = get_manifest()
        run_qa = read(RUN_QA)
        eval_qa = read(EVALUATION_QA)
        selected = read(SELECTION)
        require(run_qa["status"] == eval_qa["status"] == "PASS" and
                run_qa["fingerprint"] == eval_qa["fingerprint"] ==
                selected["fingerprint"] == manifest["fingerprint"],
                "Final QA gate failed")
        rows = []
        names_rows = []
        for k in range(156, 0, -1):
            state = read(state_path(k))
            record = read(evaluation_path(k))
            row = {"k": k, "n_base": state["n_base"],
                   "n_pressure": state["n_pressure"],
                   "train_cv_mean_rmse_db": state["cv_mean_rmse_db"],
                   "train_cv_se_rmse_db": state["cv_se_rmse_db"],
                   "train_fit_rmse_db": state["full_train_fit"]["metrics"]["overall"]["rmse"],
                   "solver_iterations": state["full_train_fit"]["solver_iterations"],
                   "removed_feature_next": state["removed_feature_next"],
                   "one_se_selected": k == selected["selected_k"]}
            for split in ("validation", "test"):
                for event in ("overall", *EVENTS):
                    event_row = record["results"][split]["metrics"][event]
                    for key in ("n", "rmse", "mae", "bias"):
                        row[f"{split}_{event}_{key}"] = event_row[key]
            rows.append(row)
            names_rows.append({"k": k,
                               "feature_names_json": json.dumps(state["feature_names"],
                                                                ensure_ascii=False)})
        for path, table in ((OUT / "rfe_metrics_all_k.csv", rows),
                            (OUT / "retained_features_all_k.csv", names_rows)):
            import io
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
            atomic_text(path, buffer.getvalue())
        by_k = {row["k"]: row for row in rows}
        selected_k = selected["selected_k"]
        k10 = read(state_path(10))["feature_names"]
        result = {"fingerprint": manifest["fingerprint"],
                  "selected_k_train_one_se": selected_k,
                  "selected_feature_names": selected["selected_feature_names"],
                  "k10_feature_names": k10,
                  "k156": by_k[156], "k10": by_k[10], "k1": by_k[1],
                  "selected_k_row": by_k[selected_k],
                  "n_states": 156, "n_refits": 624,
                  "test_historically_accessed": True,
                  "same_day_train_test_overlap": True,
                  "test_curve_descriptive_not_a_selection_rule": True,
                  "created_utc": now()}
        atomic_json(OUT / "final_report.json", result)
        def fmt(k: int) -> str:
            row = by_k[k]
            return (f"| {k} | {row['n_base']} | {row['n_pressure']} | "
                    f"{row['test_overall_rmse']:.4f} | "
                    f"{row['test_rain_rmse']:.4f} | "
                    f"{row['test_dry_overcast_rmse']:.4f} |")
        prose = ["# Revision 024 SVR-RBF permutation RFE results", "",
                 "The path was selected using only three day-disjoint folds of the 2404 training hours. The 343-hour validation and 687-hour test splits did not choose a deletion. The all-156-column direct-refit SVR from Revision 023 fixed C=10, epsilon=0.1, gamma=scale. Each k from 156 to 1 was newly fit; no zero-mask result appears here.", "",
                 f"Independent QA PASS: 156 states, 156 saved models, 624 fits and 155 one-column deletions. The train-fold one-SE rule froze k={selected_k} before holdout evaluation. SVR uses a convergent optimization solver, not epochs; solver iterations are recorded per fit.", "",
                 "| k | Base columns | Pressure columns | Test overall RMSE (dB) | Test rain RMSE (dB) | Test dry-overcast RMSE (dB) |",
                 "|---:|---:|---:|---:|---:|---:|",
                 fmt(156), fmt(selected_k), fmt(10), fmt(1), "",
                 "## Ten-column state", ""]
        prose += [f"{index}. `{name}`" for index, name in enumerate(k10, 1)]
        prose += ["", "The complete 156-to-1 metric trajectory is `rfe_metrics_all_k.csv`; exact ordered names at every k are in `retained_features_all_k.csv` and the state JSON files. Per-hour validation/test predictions and weather-event scores are in `evaluation/k_*.json`. The above test curve is descriptive; choosing its minimum k after seeing it would require a new independent assessment.", "",
                  "The test split was inspected in previous revisions and its calendar days overlap training. Therefore these metrics are not a fresh blind or unseen-day generalization test. Rain includes strong rain.", ""]
        atomic_text(OUT / "RESULTS.md", "\n".join(prose))
        return {"status": "reported", "selected_k": selected_k,
                "k10_names": k10, "files": ["rfe_metrics_all_k.csv",
                                               "retained_features_all_k.csv",
                                               "final_report.json", "RESULTS.md"]}


def status() -> dict:
    state_files = list((OUT / "states").glob("k_*.json"))
    model_files = list((OUT / "models").glob("k_*.joblib"))
    eval_files = list((OUT / "evaluation").glob("k_*.json"))
    return {"manifest_exists": MANIFEST.exists(),
            "fingerprint": read(MANIFEST)["fingerprint"] if MANIFEST.exists() else None,
            "n_states": len(state_files), "n_models": len(model_files),
            "n_evaluated": len(eval_files),
            "lowest_completed_k": min((int(path.stem.split("_")[1]) for path in state_files),
                                      default=None),
            "train_qa": read(RUN_QA)["status"] if RUN_QA.exists() else None,
            "evaluation_qa": read(EVALUATION_QA)["status"] if EVALUATION_QA.exists() else None,
            "reported": (OUT / "RESULTS.md").exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "status", "verify-run",
                                            "evaluate", "verify-evaluation", "report"))
    args = parser.parse_args()
    action = {"prepare": prepare, "run": run, "status": status,
              "verify-run": verify_run, "evaluate": evaluate,
              "verify-evaluation": verify_evaluation, "report": report}[args.command]
    print(json.dumps(action(), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
