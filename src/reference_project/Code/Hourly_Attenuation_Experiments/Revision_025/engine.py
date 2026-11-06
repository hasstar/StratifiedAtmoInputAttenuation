"""Revision 025 shared, train-fitted BiLSTM utilities; no implicit holdout reads."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD_MODELS = HERE.parent / "Revision_014"
sys.path.insert(0, str(OLD_MODELS))
from models import TorchRegressor, VerticalProfileNet  # noqa: E402

OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_025_BiLSTM"
PARENT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation"
DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
TARGET = "clear_sky_attenuation_db"
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
EVENTS = ("rain", "dry_overcast", "heavy_rain")
SEED = 17
EPOCHS = 62
CONFIG = {"epochs": EPOCHS, "lr": 0.001, "dropout": 0.2, "weight_decay": 0.001,
          "batch_size": 128}
MANIFEST = OUT / "run_manifest.json"
VERSION = "revision025-bilstm-subsets-and-permutation-rfe-v1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_bundle(path: Path, bundle: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, path)
    return sha(path)


@contextlib.contextmanager
def writer_lock():
    import msvcrt

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "revision025.writer.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another Revision 025 writer is active") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def parent_plan() -> tuple[list[str], list[str]]:
    for name in ("feature_selection_qa.json", "model_selection_qa.json"):
        require(read(PARENT / name)["status"] == "PASS", f"Parent {name} QA failed")
    retained = read(PARENT / "feature_selection.json")["feature_names_retained"]
    original = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    require(len(retained) == 156 and len(set(retained)) == 156 and len(original) == 348 and
            len(set(original)) == 348 and set(retained) <= set(original),
            "Frozen 156/348 input schema changed")
    require(all(name.startswith("surface__") for name in original[:188]) and
            all(len([name for name in original if name.startswith(f"p{level}__")]) == 16
                for level in LEVELS) and
            sum(name.startswith("surface__") for name in retained) == 50 and
            all(any(name.startswith(f"p{level}__") for name in retained) for level in LEVELS),
            "Vertical layout or screen changed")
    rows = read(PARENT / "model_selection.json")["model_rows"]
    bilstm = next(row for row in rows if row["name"] == "bilstm")
    require(bilstm["best_config_id"] == 2 and bilstm["formal_rounds"] == EPOCHS and
            bilstm["best_config"] == {key: CONFIG[key] for key in
                                      ("epochs", "lr", "dropout", "weight_decay")}
            | {"epochs": 80},
            "Frozen parent BiLSTM configuration changed")
    return retained, original


def frame_for(split: str, retained: list[str]) -> pd.DataFrame:
    require(split in ("train", "validation", "test"), "Unknown partition")
    frame = pd.read_parquet(DATA / f"{split}.parquet",
                            columns=["hour_bjt", *retained, TARGET])
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet",
                             columns=["hour_bjt", "split", *EVENTS])
    frame = frame.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    expected = {"train": 2404, "validation": 343, "test": 687}[split]
    require(len(frame) == expected and frame["split"].eq(split).all() and
            frame["hour_bjt"].is_unique and frame[TARGET].notna().all(),
            f"{split} sample schema changed")
    return frame.sort_values("hour_bjt").reset_index(drop=True)


def folds(frame: pd.DataFrame) -> list[dict]:
    days = frame["hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    splitter = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=20261002)
    result = []
    for fold_id, (fit, hold) in enumerate(splitter.split(frame, groups=days), 1):
        require(set(days[fit]).isdisjoint(days[hold]), "Day leakage across train fold")
        result.append({"fold": fold_id, "fit": fit, "hold": hold,
                       "fit_days": len(set(days[fit])), "hold_days": len(set(days[hold])),
                       "fit_index_sha256": fingerprint(fit.tolist()),
                       "hold_index_sha256": fingerprint(hold.tolist())})
    return result


def candidate_manifest() -> dict:
    retained, original = parent_plan()
    frame = frame_for("train", retained)
    fold_list = folds(frame)
    files = [DATA / "train.parquet", DATA / "feature_columns.csv",
             DATA / "hourly_sample_status.parquet", PARENT / "feature_selection.json",
             PARENT / "feature_selection_qa.json", PARENT / "model_selection.json",
             PARENT / "model_selection_qa.json", OLD_MODELS / "models.py",
             HERE / "PROTOCOL.md", HERE / "engine.py", HERE / "run_experiment.py"]
    payload = {"version": VERSION,
               "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
               "target": TARGET, "n_train": len(frame), "retained_names": retained,
               "original_names": original, "levels_hpa": list(LEVELS),
               "model": "Revision 014 vertical BiLSTM", "config": CONFIG, "seed": SEED,
               "stopping_reason": "62 epochs selected by parent full-input validation; fixed for every refit",
               "folds": [{key: value for key, value in fold.items()
                          if key not in ("fit", "hold")} for fold in fold_list],
               "selection_scope": "train-only day-grouped folds, overall RMSE",
               "no_test_access_during_training": True}
    payload["fingerprint"] = fingerprint(payload)
    payload["created_utc"] = now()
    return payload


def get_manifest(create: bool = True) -> dict:
    candidate = candidate_manifest()
    if MANIFEST.exists():
        frozen = read(MANIFEST)
        require(frozen["fingerprint"] == candidate["fingerprint"],
                "Source/code/protocol fingerprint differs from frozen manifest")
        return frozen
    require(create, "Run prepare first")
    atomic_json(MANIFEST, candidate)
    return candidate


def pressure_positions(names: list[str]) -> tuple[int, ...]:
    return tuple(i for i, level in enumerate(LEVELS)
                 if any(name.startswith(f"p{level}__") for name in names))


def layout(original: list[str], names: list[str]) -> tuple[list[str], tuple[int, ...], list[int]]:
    positions = pressure_positions(names)
    slots = original[:188] + [name for index in positions for name in original
                              if name.startswith(f"p{LEVELS[index]}__")]
    indices = [slots.index(name) for name in names]
    require(len(slots) == 188 + 16 * len(positions) and len(indices) == len(names),
            "BiLSTM layout differs from original 188+16-per-level format")
    return slots, positions, indices


def raw_x(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = frame[names].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def transform(bundle: dict, raw: np.ndarray) -> np.ndarray:
    selected = bundle["scaler"].transform(bundle["imputer"].transform(raw))
    dense = np.zeros((len(raw), bundle["layout_width"]), dtype=np.float64)
    dense[:, bundle["slot_indices"]] = selected
    require(np.isfinite(dense).all(), "Nonfinite preprocessed BiLSTM input")
    return dense


def make_bundle(frame: pd.DataFrame, fit_index: np.ndarray,
                names: list[str], original: list[str]) -> tuple[dict, np.ndarray]:
    slots, positions, indices = layout(original, names)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    xf = raw_x(frame.iloc[fit_index], names)
    scaler.fit(imputer.fit_transform(xf))
    bundle = {"names": names, "active_positions": positions,
              "slot_indices": indices, "layout_width": len(slots),
              "imputer": imputer, "scaler": scaler, "model": None,
              "config": CONFIG, "seed": SEED}
    return bundle, transform(bundle, xf)


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def scores(truth: np.ndarray, pred: np.ndarray, frame: pd.DataFrame) -> dict:
    require(len(truth) == len(pred) == len(frame) and np.isfinite(pred).all(),
            "Invalid prediction array")
    masks = {"overall": np.ones(len(frame), dtype=bool)}
    masks.update({event: frame[event].to_numpy(dtype=bool) for event in EVENTS})
    result = {}
    for event, mask in masks.items():
        errors = pred[mask] - truth[mask]
        result[event] = {"n": int(mask.sum()),
                         "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                         "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
                         "bias": float(np.mean(errors)) if len(errors) else None}
    return result


def hourly(frame: pd.DataFrame, truth: np.ndarray, pred: np.ndarray) -> dict:
    return {"hour_bjt": frame["hour_bjt"].astype(str).tolist(),
            "truth_db": truth.tolist(), "prediction_db": pred.tolist(),
            **{event: frame[event].astype(bool).tolist() for event in EVENTS}}


def check_scores(saved: dict, prediction: dict) -> None:
    truth = np.asarray(prediction["truth_db"], dtype=float)
    pred = np.asarray(prediction["prediction_db"], dtype=float)
    expected = scores(truth, pred, pd.DataFrame(prediction))
    require(len(prediction["hour_bjt"]) == len(truth), "Hour/prediction length differs")
    for event, row in expected.items():
        require(saved[event]["n"] == row["n"], f"{event} sample count differs")
        for key in ("rmse", "mae", "bias"):
            left, right = saved[event][key], row[key]
            require((left is None and right is None) or
                    (left is not None and right is not None and
                     np.isclose(left, right, atol=1e-9, rtol=1e-9)),
                    f"{event} {key} does not recompute")


def train_model(bundle: dict, xf: np.ndarray, yf: np.ndarray,
                xh: np.ndarray | None = None, yh: np.ndarray | None = None
                ) -> tuple[dict, list[dict], float]:
    require((xh is None) == (yh is None), "Holdout X/y mismatch")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    model = TorchRegressor("bilstm", CONFIG, SEED, xf.shape[1], bundle["active_positions"])
    model.net = VerticalProfileNet("bilstm", bundle["active_positions"], CONFIG["dropout"])
    model.target_mean = float(np.mean(yf))
    model.target_scale = max(float(np.std(yf)), 1e-6)
    xx = torch.as_tensor(np.asarray(xf, dtype=np.float32))
    yy = torch.as_tensor(((yf - model.target_mean) / model.target_scale).astype(np.float32))
    hh = None if xh is None else torch.as_tensor(np.asarray(xh, dtype=np.float32))
    optim = torch.optim.AdamW(model.net.parameters(), lr=CONFIG["lr"],
                              weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator().manual_seed(SEED)
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.net.train()
        for batch in torch.randperm(len(xx), generator=generator).split(CONFIG["batch_size"]):
            optim.zero_grad(set_to_none=True)
            loss = torch.mean((model.net(xx[batch]) - yy[batch]) ** 2)
            loss.backward()
            optim.step()
        model.net.eval()
        with torch.no_grad():
            tp = model.net(xx).numpy().astype(float) * model.target_scale + model.target_mean
            hp = None if hh is None else (model.net(hh).numpy().astype(float) *
                                          model.target_scale + model.target_mean)
        history.append({"epoch": epoch, "train_rmse_db": rmse(yf, tp),
                        "holdout_rmse_db": None if hp is None else rmse(yh, hp)})
    bundle["model"] = model
    return bundle, history, time.perf_counter() - started


def predict(bundle: dict, raw: np.ndarray) -> np.ndarray:
    return np.asarray(bundle["model"].predict(transform(bundle, raw)), dtype=float)


def perm_seed(manifest: dict, k: int, fold: int, name: str) -> int:
    value = f"{manifest['fingerprint']}|{k}|{fold}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")

