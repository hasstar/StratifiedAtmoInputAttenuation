"""Shared, self-contained training helpers for Revision 014 (no test reads here)."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
import support as prior  # noqa: E402
import support as cleaned  # noqa: E402
import support as residual  # noqa: E402
from models import (  # noqa: E402
    MODEL_NAMES, NEURAL_MODELS, SCALED_MODELS, FlatMLP, VerticalProfileNet,
    TorchRegressor, candidate_configs, make_estimator,
)

OUT = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_014_SelfContained"
DATA = cleaned.DATA
LEVELS = cleaned.LEVELS
TARGET = cleaned.TARGET
SEEDS = (17, 29, 43)
GROUPS = ("overall", "clear_dry", "rain", "heavy_rain", "dry_overcast", "snow", "deep_fade_gt5")
RESIDUAL_CONFIG = {
    "n_estimators": 120, "max_depth": 2, "learning_rate": 0.03,
    "min_child_weight": 10, "reg_lambda": 5.0,
    "subsample": 0.8, "colsample_bytree": 0.8,
}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_training() -> tuple[pd.DataFrame, list[str], list[dict]]:
    frame, names, folds = residual.load_training()
    require(len(names) == 348 and len(frame) == 2747, "Training/validation schema changed")
    require(frame.split.value_counts().to_dict() == {"train": 2404, "validation": 343},
            "Training/validation counts changed")
    require(len(folds) == 3 and all(fold["n_validation"] == 300 for fold in folds),
            "Internal time folds changed")
    return frame, names, folds


def layer_folds(frame: pd.DataFrame) -> list[dict]:
    """Three fixed day-grouped training-only checks matching the random-hour task."""
    train = np.flatnonzero(frame.split.to_numpy() == "train")
    groups = frame.loc[train, "hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    splitter = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=20260930)
    folds = []
    for fold_id, (fit_local, hold_local) in enumerate(splitter.split(train, groups=groups), 1):
        fit = train[fit_local]
        hold = train[hold_local]
        require(set(groups[fit_local]).isdisjoint(set(groups[hold_local])),
                "Same day crossed layer fit/holdout")
        folds.append({"id": fold_id, "train_index": fit,
                      "validation_index": hold,
                      "n_train": len(fit), "n_validation": len(hold),
                      "fit_days": len(np.unique(groups[fit_local])),
                      "holdout_days": len(np.unique(groups[hold_local])),
                      "split_seed": 20260930})
    return folds


def hash_sources(paths: list[Path], version: str) -> dict:
    hashes = {str(p.relative_to(PROJECT)): prior.sha256(p) for p in paths}
    fingerprint = hashlib.sha256(json.dumps(
        {"version": version, "source_hashes": hashes}, sort_keys=True
    ).encode()).hexdigest()
    return {"version": version, "fingerprint": fingerprint, "source_hashes": hashes,
            "created_utc": prior.utc_now(), "python": sys.version,
            "platform": platform.platform(),
            "package_versions": {p: __import__(p).__version__ for p in
                                 ("numpy", "pandas", "sklearn", "torch", "xgboost", "lightgbm", "catboost")}}


def check_manifest(path: Path, candidate: dict) -> dict:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        require(old["fingerprint"] == candidate["fingerprint"],
                f"Inputs/code changed; create a new revision: {path}")
        return old
    prior.atomic_json(path, candidate)
    return candidate


def read_record(path: Path, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    require(row.get("fingerprint") == fingerprint, f"Record fingerprint mismatch: {path}")
    return row


@contextlib.contextmanager
def writer_lock() -> None:
    import msvcrt

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "writer.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another Revision 014 writer is active") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def raw(frame: pd.DataFrame, positions: np.ndarray, names: list[str]) -> np.ndarray:
    x = frame.loc[positions, names].to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    return x


def preprocess(xfit: np.ndarray, xhold: np.ndarray | None, model_name: str):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(xfit)
    hold = None if xhold is None else imputer.transform(xhold)
    scaler = StandardScaler() if model_name in SCALED_MODELS else None
    if scaler is not None:
        train = scaler.fit_transform(train)
        if hold is not None:
            hold = scaler.transform(hold)
    return train, hold, imputer, scaler


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(pred) - np.asarray(truth)))))


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def metrics(frame: pd.DataFrame, positions: np.ndarray, pred: np.ndarray) -> dict:
    y = frame.loc[positions, TARGET].to_numpy(dtype=float)
    require(len(y) == len(pred) and np.isfinite(pred).all(), "Invalid predictions")
    day = frame.loc[positions, "hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    masks = {"overall": np.ones(len(y), dtype=bool),
             "deep_fade_gt5": y > 5.0}
    for label in ("clear_dry", "rain", "heavy_rain", "dry_overcast", "snow"):
        masks[label] = frame.loc[positions, label].to_numpy(dtype=bool)
    result = {}
    for label in GROUPS:
        mask = masks[label]
        err = np.asarray(pred)[mask] - y[mask]
        result[label] = {
            "n": int(mask.sum()), "days": int(len(np.unique(day[mask]))),
            "rmse": None if len(err) == 0 else float(np.sqrt(np.mean(err ** 2))),
            "mae": None if len(err) == 0 else float(np.mean(np.abs(err))),
            "bias": None if len(err) == 0 else float(np.mean(err)),
        }
    return result


def model_config(name: str, base: dict, rounds: int | None) -> dict:
    config = dict(base)
    key = {"xgboost": "n_estimators", "lightgbm": "n_estimators",
           "catboost": "iterations", "random_forest": "n_estimators"}.get(name)
    if name in NEURAL_MODELS:
        key = "epochs"
    if key is not None and rounds is not None:
        config[key] = int(rounds)
    return config


def _predict_at(model, name: str, x: np.ndarray, round_number: int) -> np.ndarray:
    if name == "xgboost":
        return np.asarray(model.predict(x, iteration_range=(0, round_number)), dtype=float)
    if name == "lightgbm":
        return np.asarray(model.predict(x, num_iteration=round_number), dtype=float)
    if name == "catboost":
        return np.asarray(model.predict(x, ntree_end=round_number), dtype=float)
    raise ValueError(name)


def _boost_hist(model, name: str, xfit: np.ndarray, yfit: np.ndarray,
                xhold: np.ndarray | None, yhold: np.ndarray | None) -> list[dict]:
    if name == "xgboost":
        model.set_params(eval_metric="rmse")
        evaluation = [(xfit, yfit)]
        if xhold is not None:
            evaluation.append((xhold, yhold))
        model.fit(xfit, yfit, eval_set=evaluation, verbose=False)
        result = model.evals_result()
        train_curve = result["validation_0"]["rmse"]
        hold_curve = result["validation_1"]["rmse"] if xhold is not None else None
    elif name == "lightgbm":
        import lightgbm as lgb
        evaluation = [(xfit, yfit)]
        if xhold is not None:
            evaluation.append((xhold, yhold))
        logs: dict = {}
        model.fit(xfit, yfit, eval_set=evaluation, eval_metric="rmse",
                  callbacks=[lgb.record_evaluation(logs)])
        keys = list(logs)
        train_curve = logs[keys[0]]["rmse"]
        hold_curve = logs[keys[1]]["rmse"] if xhold is not None else None
    else:
        model.fit(xfit, yfit, eval_set=None if xhold is None else (xhold, yhold),
                  verbose=False, use_best_model=False)
        logs = model.get_evals_result()
        train_curve = logs["learn"]["RMSE"]
        hold_curve = logs["validation"]["RMSE"] if xhold is not None else None
    return [{"round": i + 1, "train_rmse_db": float(value),
             "holdout_rmse_db": None if hold_curve is None else float(hold_curve[i])}
            for i, value in enumerate(train_curve)]


def _forest_hist(model, xfit: np.ndarray, yfit: np.ndarray,
                 xhold: np.ndarray | None, yhold: np.ndarray | None) -> tuple[list[dict], list[np.ndarray]]:
    model.fit(xfit, yfit)
    fit_total = np.zeros(len(yfit), dtype=float)
    hold_total = None if xhold is None else np.zeros(len(yhold), dtype=float)
    hold_preds = []
    history = []
    for i, tree in enumerate(model.estimators_, 1):
        fit_total += tree.predict(xfit)
        if hold_total is not None:
            hold_total += tree.predict(xhold)
            hold_preds.append((hold_total / i).copy())
        history.append({"round": i, "train_rmse_db": rmse(yfit, fit_total / i),
                        "holdout_rmse_db": None if hold_total is None else rmse(yhold, hold_total / i)})
    return history, hold_preds


def _neural_hist(name: str, config: dict, seed: int, positions: tuple[int, ...],
                 xfit: np.ndarray, yfit: np.ndarray,
                 xhold: np.ndarray | None, yhold: np.ndarray | None):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = TorchRegressor(name, config, seed, xfit.shape[1], positions)
    model.net = FlatMLP(xfit.shape[1], config["dropout"]) if name == "mlp" else \
        VerticalProfileNet(name, positions, config["dropout"])
    model.target_mean = float(np.mean(yfit))
    model.target_scale = max(float(np.std(yfit)), 1e-6)
    xx = torch.as_tensor(np.asarray(xfit, dtype=np.float32))
    yy = torch.as_tensor(((yfit - model.target_mean) / model.target_scale).astype(np.float32))
    xh = None if xhold is None else torch.as_tensor(np.asarray(xhold, dtype=np.float32))
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_state = None
    best_pred = None
    best_rmse = math.inf
    for epoch in range(1, config["epochs"] + 1):
        model.net.train()
        order = torch.randperm(len(xx), generator=generator)
        for batch in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model.net(xx[batch]) - yy[batch]) ** 2)
            loss.backward()
            optimizer.step()
        model.net.eval()
        with torch.no_grad():
            train_pred = model.net(xx).numpy().astype(float) * model.target_scale + model.target_mean
            hold_pred = None if xh is None else \
                model.net(xh).numpy().astype(float) * model.target_scale + model.target_mean
        train_rmse = rmse(yfit, train_pred)
        hold_rmse = None if hold_pred is None else rmse(yhold, hold_pred)
        history.append({"round": epoch, "train_rmse_db": train_rmse,
                        "holdout_rmse_db": hold_rmse})
        if hold_rmse is not None and hold_rmse < best_rmse:
            best_rmse = hold_rmse
            best_state = copy.deepcopy(model.net.state_dict())
            best_pred = hold_pred.copy()
    if best_state is not None:
        model.net.load_state_dict(best_state)
    model.net.eval()
    return model, history, best_pred


def fit_logged(name: str, config: dict, seed: int, active_positions: tuple[int, ...],
               xfit: np.ndarray, yfit: np.ndarray,
               xhold: np.ndarray | None = None, yhold: np.ndarray | None = None,
               fixed_rounds: int | None = None):
    """Return fitted estimator, per-round history, selected holdout prediction, metadata."""
    require((xhold is None) == (yhold is None), "Holdout X/y mismatch")
    cfg = model_config(name, config, fixed_rounds)
    started = time.perf_counter()
    if name in NEURAL_MODELS:
        model, history, best_neural = _neural_hist(name, cfg, seed, active_positions,
                                                   xfit, yfit, xhold, yhold)
        best_round = len(history) if xhold is None else min(
            history, key=lambda row: (row["holdout_rmse_db"], row["round"]))["round"]
        prediction = None if xhold is None else best_neural
    else:
        model = make_estimator(name, cfg, seed, xfit.shape[1], active_positions)
        if name in ("xgboost", "lightgbm", "catboost"):
            history = _boost_hist(model, name, xfit, yfit, xhold, yhold)
            best_round = len(history) if xhold is None else min(
                history, key=lambda row: (row["holdout_rmse_db"], row["round"]))["round"]
            prediction = None if xhold is None else _predict_at(model, name, xhold, best_round)
        elif name == "random_forest":
            history, hold_preds = _forest_hist(model, xfit, yfit, xhold, yhold)
            best_round = len(history) if xhold is None else min(
                history, key=lambda row: (row["holdout_rmse_db"], row["round"]))["round"]
            prediction = None if xhold is None else hold_preds[best_round - 1]
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(xfit, yfit)
            prediction = None if xhold is None else np.asarray(model.predict(xhold), dtype=float)
            best_round = None
            history = [{"round": None, "train_rmse_db": rmse(yfit, model.predict(xfit)),
                        "holdout_rmse_db": None if prediction is None else rmse(yhold, prediction)}]
    seconds = time.perf_counter() - started
    solver_iter = getattr(model, "n_iter_", None)
    if isinstance(solver_iter, np.ndarray):
        solver_iter = solver_iter.tolist()
    if isinstance(solver_iter, np.integer):
        solver_iter = int(solver_iter)
    meta = {"candidate_max_rounds": (config.get("n_estimators") or config.get("iterations") or
                                      config.get("epochs")),
            "trained_rounds": None if best_round is None else len(history),
            "selected_round": best_round, "solver_iterations": solver_iter,
            "fit_seconds": seconds,
            "selection_reason": ("minimum specified holdout RMSE, earliest tie"
                                 if xhold is not None and best_round is not None else
                                 "fixed preselected number of rounds"
                                 if fixed_rounds is not None else
                                 "solver convergence or preset iteration cap")}
    if prediction is not None:
        require(len(prediction) == len(yhold) and np.isfinite(prediction).all(),
                "Non-finite holdout prediction")
    return model, history, prediction, meta


def measure_latency(model, imputer, scaler, raw_validation: np.ndarray) -> dict:
    positions = np.linspace(0, len(raw_validation) - 1, 100, dtype=int)
    def predict_one(raw_one):
        start = time.perf_counter_ns()
        x = imputer.transform(raw_one)
        if scaler is not None:
            x = scaler.transform(x)
        result = np.asarray(model.predict(x), dtype=float).reshape(-1)
        end = time.perf_counter_ns()
        require(len(result) == 1 and np.isfinite(result).all(), "Invalid single-row prediction")
        return int(end - start)
    for i in positions[:20]:
        predict_one(raw_validation[i:i + 1])
    samples = [predict_one(raw_validation[i:i + 1]) for _ in range(3) for i in positions]
    return {"end_to_end_ns": samples, "median_ms": float(np.median(samples) / 1e6),
            "p95_ms": float(np.percentile(samples, 95) / 1e6),
            "warmup_calls": 20, "timed_calls": len(samples)}
