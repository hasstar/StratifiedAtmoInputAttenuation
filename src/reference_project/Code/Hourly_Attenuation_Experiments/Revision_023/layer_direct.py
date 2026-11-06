"""Revision 023: exhaustive, directly refitted RBF-SVR pressure-layer comparison.

Selection is train-only and calendar-day grouped. The test partition is opened
only by the explicit ``test``/``verify-test``/``report`` commands.
"""

from __future__ import annotations

import argparse
import contextlib
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
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation"
DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
TARGET = "clear_sky_attenuation_db"
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
EVENTS = ("rain", "dry_overcast", "heavy_rain")
VERSION = "revision023-direct-refit-all-1024-v1"
MANIFEST = OUT / "layer_manifest.json"
SELECTION = OUT / "layer_selection.json"
QA = OUT / "layer_qa.json"
TEST_MANIFEST = OUT / "layer_test_manifest.json"
TEST_QA = OUT / "layer_test_qa.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def writer_lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "layer.writer.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another Revision 023 layer writer is active") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def feature_plan() -> tuple[list[str], list[str], list[dict]]:
    selected = read(OUT / "feature_selection.json")
    model_qa = read(OUT / "model_selection_qa.json")
    model_choice = read(OUT / "model_selection.json")
    require(model_qa.get("status") == "PASS" and
            model_qa["fingerprint"] == model_choice["fingerprint"] and
            model_choice["primary_winner"] == "svr_rbf", "Model QA/learner gate failed")
    names = selected["feature_names_retained"]
    base = [name for name in names if name.startswith("surface__")]
    groups = [[name for name in names if name.startswith(f"p{level}__")]
              for level in LEVELS]
    require(len(base) == 50 and len(names) == 156 and all(groups) and
            len(base) + sum(map(len, groups)) == len(names), "Screened group schema changed")
    winner = next(row for row in model_choice["model_rows"] if row["name"] == "svr_rbf")
    configs = read(OUT / "run_manifest.json")["configs"]["svr_rbf"]
    require(len(configs) == 4 and winner["best_config"] == configs[winner["best_config_id"]],
            "Frozen SVR configuration list changed")
    return base, names, configs


def train_frame() -> pd.DataFrame:
    base, names, _ = feature_plan()
    frame = pd.read_parquet(DATA / "train.parquet", columns=["hour_bjt", *names, TARGET])
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet",
                             columns=["hour_bjt", "split", *EVENTS])
    frame = frame.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    require(len(frame) == 2404 and frame["split"].eq("train").all() and
            frame[TARGET].notna().all() and len(base) == 50,
            "Training partition/schema changed")
    return frame.sort_values("hour_bjt").reset_index(drop=True)


def holdout_frame(split: str) -> pd.DataFrame:
    require(split in ("validation", "test"), "Unknown holdout split")
    _, names, _ = feature_plan()
    frame = pd.read_parquet(DATA / f"{split}.parquet",
                            columns=["hour_bjt", *names, TARGET])
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet",
                             columns=["hour_bjt", "split", *EVENTS])
    frame = frame.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    expected = 343 if split == "validation" else 687
    require(len(frame) == expected and frame["split"].eq(split).all() and
            frame[TARGET].notna().all(), f"{split} partition/schema changed")
    return frame.sort_values("hour_bjt").reset_index(drop=True)


def folds(frame: pd.DataFrame) -> list[dict]:
    groups = frame["hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    splitter = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=20261002)
    result = []
    for fold_id, (fit, hold) in enumerate(splitter.split(frame, groups=groups), 1):
        require(set(groups[fit]).isdisjoint(groups[hold]), "Calendar-day fold overlap")
        result.append({"fold": fold_id, "fit": fit, "hold": hold,
                       "fit_days": int(len(set(groups[fit]))),
                       "hold_days": int(len(set(groups[hold]))),
                       "fit_index_sha256": fingerprint(fit.tolist()),
                       "hold_index_sha256": fingerprint(hold.tolist())})
    return result


def names_for(mask: int, base: list[str], all_names: list[str]) -> list[str]:
    levels = {level for i, level in enumerate(LEVELS) if mask & (1 << i)}
    selected = [name for name in all_names if name in set(base) or
                any(name.startswith(f"p{level}__") for level in levels)]
    require(len(selected) == len(base) + sum(name.startswith(f"p{level}__")
                                             for name in all_names for level in levels),
            "Mask feature accounting failed")
    return selected


def estimator(config: dict) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", **config)),
    ])


def scores(truth: np.ndarray, pred: np.ndarray, frame: pd.DataFrame) -> dict:
    require(len(truth) == len(pred) == len(frame) and np.isfinite(pred).all(),
            "Invalid prediction shape or value")
    masks = {"overall": np.ones(len(frame), dtype=bool)}
    masks.update({label: frame[label].to_numpy(dtype=bool) for label in EVENTS})
    answer = {}
    for label, mask in masks.items():
        error = pred[mask] - truth[mask]
        answer[label] = {"n": int(mask.sum()),
                         "rmse": None if not len(error) else float(np.sqrt(np.mean(error ** 2))),
                         "mae": None if not len(error) else float(np.mean(np.abs(error))),
                         "bias": None if not len(error) else float(np.mean(error))}
    return answer


def candidate_manifest() -> dict:
    base, names, configs = feature_plan()
    paths = [DATA / "train.parquet", DATA / "feature_columns.csv",
             DATA / "hourly_sample_status.parquet", HERE / "PROTOCOL.md",
             HERE / "LAYER_PLAN.md", HERE / "layer_direct.py",
             OUT / "feature_selection.json", OUT / "feature_selection_qa.json",
             OUT / "run_manifest.json", OUT / "model_selection.json",
             OUT / "model_selection_qa.json"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    frame = train_frame()
    fold_list = folds(frame)
    payload = {"version": VERSION, "source_hashes": sources,
               "learner": "svr_rbf", "configs": configs, "levels_hpa": list(LEVELS),
               "base_names": base, "all_names": names, "n_train": len(frame),
               "folds": [{k: v for k, v in item.items() if k not in ("fit", "hold")}
                         for item in fold_list],
               "selection": "three day-disjoint training folds; one-SE fewest-level mask",
               "test_not_read": True}
    payload["fingerprint"] = fingerprint(payload)
    payload["created_utc"] = utc_now()
    return payload


def get_manifest() -> dict:
    candidate = candidate_manifest()
    if MANIFEST.exists():
        frozen = read(MANIFEST)
        require(candidate["fingerprint"] == frozen["fingerprint"],
                "Frozen layer fingerprint changed; do not resume under altered sources")
        return frozen
    atomic_json(MANIFEST, candidate)
    return candidate


def subset_path(mask: int) -> Path:
    return OUT / "layer_subsets" / f"mask_{mask:04d}.json"


def model_path(mask: int) -> Path:
    return OUT / "layer_models" / f"mask_{mask:04d}.joblib"


def run_one(mask: int, frame: pd.DataFrame, fold_list: list[dict], manifest: dict) -> None:
    path = subset_path(mask)
    model_file = model_path(mask)
    if path.exists():
        record = read(path)
        require(record["fingerprint"] == manifest["fingerprint"] and
                model_file.exists() and record["model_sha256"] == sha(model_file),
                f"Existing subset/model does not match frozen run: {mask}")
        return
    names = names_for(mask, manifest["base_names"], manifest["all_names"])
    x = frame[names].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    trials = []
    predictions = {}
    for config_id, config in enumerate(manifest["configs"]):
        fold_rows = []
        prediction_rows = []
        for fold in fold_list:
            fit, hold = fold["fit"], fold["hold"]
            started = time.perf_counter()
            model = estimator(config)
            model.fit(x[fit], y[fit])
            fit_pred = model.predict(x[fit])
            hold_pred = model.predict(x[hold])
            fit_seconds = time.perf_counter() - started
            fit_score = scores(y[fit], fit_pred, frame.iloc[fit])
            hold_score = scores(y[hold], hold_pred, frame.iloc[hold])
            fold_rows.append({"fold": fold["fold"], "n_fit": len(fit),
                              "n_hold": len(hold), "fit_days": fold["fit_days"],
                              "hold_days": fold["hold_days"],
                              "fit_index_sha256": fold["fit_index_sha256"],
                              "hold_index_sha256": fold["hold_index_sha256"],
                              "fit_metrics": fit_score, "hold_metrics": hold_score,
                              "solver_iterations": int(model.named_steps["svr"].n_iter_),
                              "fit_seconds": fit_seconds})
            prediction_rows.append({
                "fold": fold["fold"],
                "hour_bjt": frame.iloc[hold]["hour_bjt"].astype(str).tolist(),
                "truth_db": y[hold].tolist(), "prediction_db": hold_pred.tolist(),
                "rain": frame.iloc[hold]["rain"].astype(bool).tolist(),
                "dry_overcast": frame.iloc[hold]["dry_overcast"].astype(bool).tolist(),
                "heavy_rain": frame.iloc[hold]["heavy_rain"].astype(bool).tolist(),
            })
        rmse = np.array([row["hold_metrics"]["overall"]["rmse"] for row in fold_rows])
        mae = np.array([row["hold_metrics"]["overall"]["mae"] for row in fold_rows])
        trials.append({"config_id": config_id, "config": config,
                       "mean_hold_rmse_db": float(rmse.mean()),
                       "se_hold_rmse_db": float(rmse.std(ddof=1) / np.sqrt(len(rmse))),
                       "mean_hold_mae_db": float(mae.mean()), "folds": fold_rows})
        predictions[config_id] = prediction_rows
    best = min(trials, key=lambda row: (row["mean_hold_rmse_db"],
                                        row["mean_hold_mae_db"], row["config_id"]))
    started = time.perf_counter()
    final = estimator(best["config"])
    final.fit(x, y)
    train_pred = final.predict(x)
    final_seconds = time.perf_counter() - started
    model_file.parent.mkdir(parents=True, exist_ok=True)
    temp = model_file.with_name(f".{model_file.name}.{os.getpid()}.tmp")
    joblib.dump({"fingerprint": manifest["fingerprint"], "mask": mask,
                 "names": names, "config": best["config"], "model": final},
                temp, compress=3)
    os.replace(temp, model_file)
    record = {"fingerprint": manifest["fingerprint"], "mask": mask,
              "levels_hpa": [level for i, level in enumerate(LEVELS) if mask & (1 << i)],
              "feature_names": names, "n_features": len(names),
              "config_trials": trials, "selected_config_id": best["config_id"],
              "selected_cv_mean_rmse_db": best["mean_hold_rmse_db"],
              "selected_cv_se_rmse_db": best["se_hold_rmse_db"],
              "selected_cv_mean_mae_db": best["mean_hold_mae_db"],
              "selected_fold_predictions": predictions[best["config_id"]],
              "full_train_fit": {"n": len(frame), "metrics": scores(y, train_pred, frame),
                                 "solver_iterations": int(final.named_steps["svr"].n_iter_),
                                 "fit_seconds": final_seconds,
                                 "training_rounds": None,
                                 "stop_reason": "libsvm solver convergence; no epoch round"},
              "model_sha256": sha(model_file), "completed_utc": utc_now()}
    atomic_json(path, record)


def choose_mask(rows: list[dict]) -> dict:
    lowest = min(rows, key=lambda row: (row["selected_cv_mean_rmse_db"], row["mask"]))
    threshold = lowest["selected_cv_mean_rmse_db"] + lowest["selected_cv_se_rmse_db"]
    eligible = [row for row in rows if row["selected_cv_mean_rmse_db"] <= threshold]
    chosen = min(eligible, key=lambda row: (len(row["levels_hpa"]),
                                            row["selected_cv_mean_rmse_db"], row["mask"]))
    level_effects = []
    by_mask = {row["mask"]: row for row in rows}
    for bit, level in enumerate(LEVELS):
        delta = np.array([
            by_mask[mask]["selected_cv_mean_rmse_db"] -
            by_mask[mask | (1 << bit)]["selected_cv_mean_rmse_db"]
            for mask in range(1024) if not mask & (1 << bit)
        ])
        level_effects.append({"level_hpa": level, "paired_contexts": len(delta),
                              "mean_improvement_db": float(delta.mean()),
                              "median_improvement_db": float(np.median(delta)),
                              "fraction_helpful": float(np.mean(delta > 0)),
                              "min_improvement_db": float(delta.min()),
                              "max_improvement_db": float(delta.max())})
    return {"selection_scope": "training-only day-disjoint folds",
            "minimum_cv_mask": lowest["mask"],
            "minimum_cv_rmse_db": lowest["selected_cv_mean_rmse_db"],
            "one_se_threshold_db": float(threshold),
            "masks_within_one_se": len(eligible),
            "selected_mask": chosen["mask"],
            "selected_levels_hpa": chosen["levels_hpa"],
            "selected_cv_rmse_db": chosen["selected_cv_mean_rmse_db"],
            "base_mask": 0, "full_mask": 1023,
            "level_effects": level_effects}


def run() -> dict:
    require(read(OUT / "model_selection_qa.json")["status"] == "PASS",
            "Ten-model QA gate has not passed")
    with writer_lock():
        manifest = get_manifest()
        frame = train_frame()
        fold_list = folds(frame)
        for mask in range(1024):
            run_one(mask, frame, fold_list, manifest)
            if mask % 16 == 0 or mask == 1023:
                print(f"REFIT mask={mask}/1023", flush=True)
        rows = [read(subset_path(mask)) for mask in range(1024)]
        choice = choose_mask(rows)
        choice.update({"fingerprint": manifest["fingerprint"],
                       "n_subsets": 1024, "n_cv_fits": 1024 * 4 * 3,
                       "n_full_train_fits": 1024, "test_not_read": True,
                       "completed_utc": utc_now()})
        if SELECTION.exists():
            frozen = read(SELECTION)
            require(frozen["fingerprint"] == choice["fingerprint"] and
                    frozen["selected_mask"] == choice["selected_mask"],
                    "Existing subset choice changed")
        else:
            atomic_json(SELECTION, choice)
        # Validation is purely diagnostic, after train-only selection freezes.
        validation = holdout_frame("validation")
        diagnostics = {"fingerprint": manifest["fingerprint"],
                       "selected_mask": choice["selected_mask"],
                       "selection_uses_validation": False, "models": []}
        for mask in sorted({0, 1023, choice["selected_mask"]}):
            bundle = joblib.load(model_path(mask))
            x = validation[bundle["names"]].to_numpy(dtype=float)
            x[~np.isfinite(x)] = np.nan
            pred = bundle["model"].predict(x)
            diagnostics["models"].append({
                "mask": mask, "config": bundle["config"], "feature_names": bundle["names"],
                "hour_bjt": validation["hour_bjt"].astype(str).tolist(),
                "truth_db": validation[TARGET].to_numpy(dtype=float).tolist(),
                "prediction_db": pred.tolist(),
                "metrics": scores(validation[TARGET].to_numpy(dtype=float), pred, validation)})
        atomic_json(OUT / "layer_validation_diagnostics.json", diagnostics)
        return {"n_subsets": 1024, "selected_mask": choice["selected_mask"],
                "selected_levels_hpa": choice["selected_levels_hpa"],
                "test_not_read": True}


def verify_run() -> dict:
    manifest = get_manifest()
    require(SELECTION.exists(), "Layer selection not complete")
    choice = read(SELECTION)
    require(choice["fingerprint"] == manifest["fingerprint"] and
            choice["n_subsets"] == 1024 and choice["test_not_read"],
            "Layer selection fingerprint/count invalid")
    rows = []
    expected_configs = manifest["configs"]
    for mask in range(1024):
        path, model_file = subset_path(mask), model_path(mask)
        require(path.exists() and model_file.exists(), f"Missing subset/model {mask}")
        row = read(path)
        expected_names = names_for(mask, manifest["base_names"], manifest["all_names"])
        require(row["fingerprint"] == manifest["fingerprint"] and
                row["mask"] == mask and row["feature_names"] == expected_names and
                row["n_features"] == len(expected_names) and
                len(row["config_trials"]) == 4 and
                row["model_sha256"] == sha(model_file),
                f"Subset/model fingerprint or features wrong: {mask}")
        for config_id, trial in enumerate(row["config_trials"]):
            require(trial["config_id"] == config_id and
                    trial["config"] == expected_configs[config_id] and
                    len(trial["folds"]) == 3,
                    f"Candidate/folds missing for mask {mask}")
            mean = float(np.mean([fold["hold_metrics"]["overall"]["rmse"]
                                  for fold in trial["folds"]]))
            require(abs(mean - trial["mean_hold_rmse_db"]) < 1e-11,
                    f"CV mean mismatch for mask {mask}")
        best = min(row["config_trials"], key=lambda item: (
            item["mean_hold_rmse_db"], item["mean_hold_mae_db"], item["config_id"]))
        require(row["selected_config_id"] == best["config_id"] and
                len(row["selected_fold_predictions"]) == 3,
                f"Mask config/predictions wrong: {mask}")
        for prediction, fold in zip(row["selected_fold_predictions"], best["folds"]):
            truth = np.asarray(prediction["truth_db"], dtype=float)
            pred = np.asarray(prediction["prediction_db"], dtype=float)
            require(len(truth) == len(pred) == fold["n_hold"] and
                    abs(float(np.sqrt(np.mean((pred - truth) ** 2))) -
                        fold["hold_metrics"]["overall"]["rmse"]) < 1e-11,
                    f"CV prediction mismatch: {mask}")
        rows.append(row)
    expected = choose_mask(rows)
    require(choice["selected_mask"] == expected["selected_mask"] and
            choice["minimum_cv_mask"] == expected["minimum_cv_mask"] and
            len(choice["level_effects"]) == 10 and
            all(item["paired_contexts"] == 512 for item in choice["level_effects"]),
            "Train-only one-SE choice or level effects changed")
    diagnostic = read(OUT / "layer_validation_diagnostics.json")
    require(diagnostic["fingerprint"] == manifest["fingerprint"] and
            diagnostic["selection_uses_validation"] is False and
            {r["mask"] for r in diagnostic["models"]} ==
            {0, 1023, choice["selected_mask"]}, "Validation diagnostics incomplete")
    qa = {"status": "PASS", "stage": "train_only_direct_refit",
          "fingerprint": manifest["fingerprint"], "n_subsets": 1024,
          "n_cv_fits": 1024 * 4 * 3, "n_full_train_fits": 1024,
          "selected_mask": choice["selected_mask"], "test_not_read": True,
          "verified_utc": utc_now()}
    atomic_json(QA, qa)
    return qa


def test_path(mask: int) -> Path:
    return OUT / "layer_test" / f"mask_{mask:04d}.json"


def test() -> dict:
    require(QA.exists() and read(QA)["status"] == "PASS", "Layer QA gate not passed")
    with writer_lock():
        manifest = get_manifest()
        frame = holdout_frame("test")
        test_hash = sha(DATA / "test.parquet")
        frozen = {"fingerprint": manifest["fingerprint"],
                  "test_parquet_sha256": test_hash, "n_test": len(frame),
                  "scoring_plan": "score all 1024 frozen models once; selection cannot change",
                  "created_utc": utc_now()}
        if TEST_MANIFEST.exists():
            require(read(TEST_MANIFEST)["test_parquet_sha256"] == test_hash,
                    "Test source changed during batch score")
        else:
            atomic_json(TEST_MANIFEST, frozen)
        y = frame[TARGET].to_numpy(dtype=float)
        for mask in range(1024):
            path = test_path(mask)
            if path.exists():
                require(read(path)["fingerprint"] == manifest["fingerprint"],
                        f"Existing test record fingerprint mismatch: {mask}")
                continue
            bundle = joblib.load(model_path(mask))
            require(bundle["fingerprint"] == manifest["fingerprint"] and
                    bundle["mask"] == mask, f"Saved model identity wrong: {mask}")
            x = frame[bundle["names"]].to_numpy(dtype=float)
            x[~np.isfinite(x)] = np.nan
            pred = bundle["model"].predict(x)
            atomic_json(path, {"fingerprint": manifest["fingerprint"],
                               "mask": mask, "feature_names": bundle["names"],
                               "model_sha256": sha(model_path(mask)),
                               "hour_bjt": frame["hour_bjt"].astype(str).tolist(),
                               "truth_db": y.tolist(), "prediction_db": pred.tolist(),
                               "rain": frame["rain"].astype(bool).tolist(),
                               "dry_overcast": frame["dry_overcast"].astype(bool).tolist(),
                               "heavy_rain": frame["heavy_rain"].astype(bool).tolist(),
                               "metrics": scores(y, pred, frame),
                               "scored_utc": utc_now()})
            if mask % 64 == 0 or mask == 1023:
                print(f"TEST_SCORE mask={mask}/1023", flush=True)
    return {"test_records": len(list((OUT / "layer_test").glob("mask_*.json"))),
            "n_test": 687}


def verify_test() -> dict:
    require(QA.exists() and read(QA)["status"] == "PASS", "Layer QA absent")
    manifest = get_manifest()
    frozen = read(TEST_MANIFEST)
    require(frozen["fingerprint"] == manifest["fingerprint"] and
            frozen["test_parquet_sha256"] == sha(DATA / "test.parquet"),
            "Test manifest mismatch")
    reference_hours = None
    reference_truth = None
    for mask in range(1024):
        row = read(test_path(mask))
        pred = np.asarray(row["prediction_db"], dtype=float)
        truth = np.asarray(row["truth_db"], dtype=float)
        require(row["fingerprint"] == manifest["fingerprint"] and row["mask"] == mask and
                row["feature_names"] == read(subset_path(mask))["feature_names"] and
                row["model_sha256"] == sha(model_path(mask)) and
                len(pred) == len(truth) == len(row["hour_bjt"]) == 687 and
                abs(float(np.sqrt(np.mean((pred - truth) ** 2))) -
                    row["metrics"]["overall"]["rmse"]) < 1e-11,
                f"Test metrics/source mismatch: {mask}")
        if reference_hours is None:
            reference_hours, reference_truth = row["hour_bjt"], truth
        else:
            require(row["hour_bjt"] == reference_hours and
                    np.array_equal(truth, reference_truth),
                    f"Test hours/truth changed by mask: {mask}")
    qa = {"status": "PASS", "fingerprint": manifest["fingerprint"],
          "n_test_models": 1024, "n_test_hours_each": 687,
          "selection_frozen_before_test": True,
          "test_historically_accessed": True, "verified_utc": utc_now()}
    atomic_json(TEST_QA, qa)
    return qa


def paired_day_bootstrap(base: dict, candidate: dict, event: str,
                         seed: int = 20261002, n_rep: int = 2000) -> dict:
    y = np.asarray(base["truth_db"], dtype=float)
    first = np.asarray(base["prediction_db"], dtype=float)
    second = np.asarray(candidate["prediction_db"], dtype=float)
    hours = pd.to_datetime(base["hour_bjt"])
    dates = np.asarray(hours.strftime("%Y-%m-%d"))
    event_mask = np.ones(len(y), dtype=bool) if event == "overall" else np.asarray(
        base[event], dtype=bool)
    unique_days = np.unique(dates)
    indices = {day: np.flatnonzero((dates == day) & event_mask) for day in unique_days}
    event_days = [day for day in unique_days if len(indices[day])]
    require(event_days, f"No test days for {event}")
    observed = float(np.sqrt(np.mean((first[event_mask] - y[event_mask]) ** 2)) -
                     np.sqrt(np.mean((second[event_mask] - y[event_mask]) ** 2)))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_rep):
        sampled = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx = np.concatenate([indices[day] for day in sampled if len(indices[day])])
        if len(idx) == 0:
            continue
        delta = float(np.sqrt(np.mean((first[idx] - y[idx]) ** 2)) -
                      np.sqrt(np.mean((second[idx] - y[idx]) ** 2)))
        deltas.append(delta)
    require(len(deltas) >= 1000, "Insufficient nonempty bootstrap replicates")
    return {"event": event, "n_hours": int(event_mask.sum()),
            "n_days_with_event": len(event_days), "observed_improvement_db": observed,
            "ci95_percentile_db": np.percentile(deltas, [2.5, 97.5]).tolist(),
            "bootstrap_replicates": len(deltas),
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0))}


def report() -> dict:
    require(TEST_QA.exists() and read(TEST_QA)["status"] == "PASS",
            "Final test QA gate not passed")
    manifest, choice = get_manifest(), read(SELECTION)
    rows = [read(subset_path(mask)) for mask in range(1024)]
    results = []
    for row in rows:
        test_row = read(test_path(row["mask"]))
        results.append({"mask": row["mask"], "levels_hpa": row["levels_hpa"],
                        "n_features": row["n_features"],
                        "config": manifest["configs"][row["selected_config_id"]],
                        "cv_mean_rmse_db": row["selected_cv_mean_rmse_db"],
                        "cv_se_rmse_db": row["selected_cv_se_rmse_db"],
                        "test_metrics": test_row["metrics"]})
    base = read(test_path(0))
    comparisons = []
    for mask in sorted({1023, choice["selected_mask"]}):
        candidate = read(test_path(mask))
        comparisons.append({"candidate_mask": mask,
                            "candidate_levels_hpa": rows[mask]["levels_hpa"],
                            "paired_day_bootstrap": [paired_day_bootstrap(base, candidate, event)
                                                     for event in ("overall", "rain", "dry_overcast")]})
    train_hours = pd.read_parquet(DATA / "train.parquet", columns=["hour_bjt"])
    train_days = set(train_hours["hour_bjt"].dt.strftime("%Y-%m-%d"))
    test_hours = pd.to_datetime(base["hour_bjt"])
    test_days = set(test_hours.strftime("%Y-%m-%d"))
    report_data = {"fingerprint": manifest["fingerprint"],
                   "selection": choice, "all_1024_exploratory_results": results,
                   "frozen_comparisons": comparisons,
                   "same_day_train_test_overlap": {
                       "train_days": len(train_days), "test_days": len(test_days),
                       "overlap_days": len(train_days & test_days),
                       "fraction_test_days_overlapping_train":
                           len(train_days & test_days) / len(test_days)},
                   "interpretation_boundary": (
                       "Historic test access and same-day train/test overlap; "
                       "test sweep descriptive only; no test-driven selection; "
                       "layer differences predictive, not causal."),
                   "completed_utc": utc_now()}
    atomic_json(OUT / "layer_final_report.json", report_data)
    pd.DataFrame([{**{k: v for k, v in row.items() if k != "test_metrics"},
                   "test_overall_rmse_db": row["test_metrics"]["overall"]["rmse"],
                   "test_rain_rmse_db": row["test_metrics"]["rain"]["rmse"],
                   "test_dry_overcast_rmse_db": row["test_metrics"]["dry_overcast"]["rmse"]}
                  for row in results]).to_csv(OUT / "layer_all_1024_summary.csv", index=False)
    return {"status": "COMPLETE", "n_subsets": 1024,
            "selected_mask": choice["selected_mask"],
            "selected_levels_hpa": choice["selected_levels_hpa"],
            "test_base_rmse_db": base["metrics"]["overall"]["rmse"],
            "test_full_rmse_db": read(test_path(1023))["metrics"]["overall"]["rmse"]}


def status() -> dict:
    def count(directory: str, pattern: str) -> int:
        place = OUT / directory
        return len(list(place.glob(pattern))) if place.exists() else 0
    return {"model_qa_pass": (OUT / "model_selection_qa.json").exists() and
            read(OUT / "model_selection_qa.json").get("status") == "PASS",
            "manifest_fingerprint": read(MANIFEST).get("fingerprint") if MANIFEST.exists() else None,
            "subset_records": count("layer_subsets", "mask_*.json"),
            "saved_models": count("layer_models", "mask_*.joblib"),
            "selection_frozen": SELECTION.exists(),
            "layer_qa_pass": QA.exists() and read(QA).get("status") == "PASS",
            "test_records": count("layer_test", "mask_*.json"),
            "test_qa_pass": TEST_QA.exists() and read(TEST_QA).get("status") == "PASS",
            "report_exists": (OUT / "layer_final_report.json").exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify-run", "test", "verify-test",
                                            "report", "status"))
    args = parser.parse_args()
    result = {"run": run, "verify-run": verify_run, "test": test,
              "verify-test": verify_test, "report": report, "status": status}[args.command]()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
