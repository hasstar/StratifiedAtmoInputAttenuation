"""Revision 015: training-only feature views and event-focused XGBoost comparison.

Commands: selftest, prepare, run [--limit N], status, report, verify, test, document.
The test parquet is opened only by test/verify-after-test/document-after-test.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DATA = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_003"
FEATURE_SETS = PROJECT / "Outputs/Datasets/Revision_002/feature_sets.csv"
OUT = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_015_Event_Features"
TRIALS = OUT / "trials"
MODELS = OUT / "models"
TARGET = "clear_sky_attenuation_db"
SEED = 20260930
MODEL_SEED = 17
VIEWS = ("full348", "local117", "base188", "clean348", "prune99", "prune95")
CONFIGS = {
    "reference": dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                      min_child_weight=1, gamma=0.0, reg_lambda=1.0,
                      colsample_bytree=0.8, subsample=0.8),
    "conservative": dict(n_estimators=600, max_depth=3, learning_rate=0.04,
                         min_child_weight=5, gamma=0.05, reg_lambda=5.0,
                         colsample_bytree=0.65, subsample=0.8),
    "profile": dict(n_estimators=800, max_depth=4, learning_rate=0.03,
                    min_child_weight=3, gamma=0.01, reg_lambda=3.0,
                    colsample_bytree=0.75, subsample=0.8),
    "shallow": dict(n_estimators=900, max_depth=2, learning_rate=0.03,
                    min_child_weight=7, gamma=0.02, reg_lambda=8.0,
                    colsample_bytree=0.6, subsample=0.9),
}
WEIGHTS = {
    "uniform": (1.0, 1.0),
    "event_moderate": (2.0, 1.5),
    "event_strong": (4.0, 2.0),
}
EVENTS = ("overall", "rain", "dry_overcast", "clear_dry")
ROLES = ("winner", "reference", "base188", "local117")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def writer_lock():
    import msvcrt

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "writer.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another Revision 015 writer is active") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def source_paths() -> list[Path]:
    return [
        DATA / "train.parquet",
        DATA / "validation.parquet",
        DATA / "feature_columns.csv",
        DATA / "hourly_sample_status.parquet",
        DATA / "excluded_unreliable_hours.csv",
        FEATURE_SETS,
        HERE / "PROTOCOL.md",
        HERE / "run_feature_experiment.py",
    ]


def fingerprint_record() -> dict:
    paths = source_paths()
    hashes = {str(p.relative_to(PROJECT)): sha256(p) for p in paths}
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "fingerprint": fingerprint,
        "source_sha256": hashes,
        "created_utc": utc_now(),
        "python": sys.version,
        "pandas": pd.__version__,
        "xgboost": __import__("xgboost").__version__,
        "sklearn": __import__("sklearn").__version__,
        "seed": SEED,
        "model_seed": MODEL_SEED,
        "views": VIEWS,
        "configs": CONFIGS,
        "weights": WEIGHTS,
        "expected_trials": len(VIEWS) * len(CONFIGS) * len(WEIGHTS),
        "expected_fits": len(VIEWS) * len(CONFIGS) * len(WEIGHTS) * 3,
        "test_not_in_selection_fingerprint": True,
    }


def checked_manifest() -> dict:
    path = OUT / "run_manifest.json"
    require(path.exists(), "Run prepare first")
    saved = read_json(path)
    current = fingerprint_record()
    require(saved["fingerprint"] == current["fingerprint"],
            "Input, protocol, or code hash changed; stop and create a new revision")
    return saved


def names_and_local() -> tuple[list[str], list[str]]:
    names = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    require(len(names) == len(set(names)) == 348, "Feature schema mismatch")
    source = pd.read_csv(FEATURE_SETS)
    local = source.loc[source.feature_set.eq("S_local")].sort_values(
        "feature_order"
    )["column"].tolist()
    require(len(local) == len(set(local)) == 117, "Local feature schema mismatch")
    require(set(local).issubset(set(names[:188])), "Local set not inside base188")
    return names, local


def load_partition(split: str, names: list[str]) -> pd.DataFrame:
    require(split in ("train", "validation", "test"), "Invalid split")
    table = pd.read_parquet(DATA / f"{split}.parquet")
    require(list(table.columns) == ["hour_bjt", "hour_utc", *names, TARGET],
            f"{split} column schema mismatch")
    require(len(table) == {"train": 2404, "validation": 343, "test": 687}[split],
            f"{split} row count mismatch")
    require(table.hour_bjt.is_unique and np.isfinite(table[TARGET]).all(),
            f"{split} time/target invalid")
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet", columns=[
        "hour_bjt", "split", "use_for_regression", "rain", "dry_overcast", "clear_dry"
    ])
    require(status.hour_bjt.is_unique, "Duplicate status time")
    table = table.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    require(table["split"].eq(split).all(), f"{split} status mismatch")
    require(table["use_for_regression"].eq(True).all(), f"{split} quality mismatch")
    for event in ("rain", "dry_overcast", "clear_dry"):
        require(table[event].notna().all(), f"{split} missing {event}")
        table[event] = table[event].astype(bool)
    require(not (table.rain & table.dry_overcast).any(),
            "Rain and dry overcast must be disjoint")
    return table.reset_index(drop=True)


def grouped_folds(train: pd.DataFrame) -> list[dict]:
    days = train.hour_bjt.dt.strftime("%Y-%m-%d").to_numpy()
    splitter = GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=SEED)
    result = []
    for number, (fit, hold) in enumerate(splitter.split(train, groups=days), 1):
        require(set(days[fit]).isdisjoint(set(days[hold])), "Day leakage in fold")
        require(train.iloc[hold].rain.sum() >= 20, "Too few holdout rainy hours")
        require(train.iloc[hold].dry_overcast.sum() >= 70,
                "Too few holdout overcast hours")
        result.append({
            "fold": number,
            "fit_indices": fit.tolist(),
            "hold_indices": hold.tolist(),
            "fit_days": int(len(set(days[fit]))),
            "hold_days": int(len(set(days[hold]))),
            "hold_rain": int(train.iloc[hold].rain.sum()),
            "hold_dry_overcast": int(train.iloc[hold].dry_overcast.sum()),
        })
    return result


def feature_view(xfit: pd.DataFrame, names: list[str], local: list[str],
                 view: str) -> list[str]:
    require(view in VIEWS, f"Invalid view {view}")
    if view == "local117":
        return list(local)
    if view == "base188":
        return list(names[:188])
    if view == "full348":
        return list(names)

    cleaned = [
        name for name in names
        if xfit[name].nunique(dropna=True) > 1
        and float(xfit[name].isna().mean()) <= 0.5
    ]
    if view == "clean348":
        return cleaned
    base = [name for name in cleaned if name in set(names[:188])]
    cutoff = 0.99 if view == "prune99" else 0.95
    matrix = xfit[base].corr(
        method="spearman", min_periods=max(50, int(len(xfit) * 0.5))
    ).to_numpy()
    parent = list(range(len(base)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= cutoff:
                parent[find(i)] = find(j)
    clusters: dict[int, list[str]] = {}
    for i, name in enumerate(base):
        clusters.setdefault(find(i), []).append(name)
    order = {name: i for i, name in enumerate(names)}
    keep_base = {
        min(group, key=lambda name: (float(xfit[name].isna().mean()), order[name]))
        for group in clusters.values()
    }
    result = [name for name in cleaned if name not in base or name in keep_base]
    require(result and len(result) == len(set(result)), "Invalid selected columns")
    return result


def sample_weights(frame: pd.DataFrame, weight_name: str) -> np.ndarray:
    rain_weight, cloud_weight = WEIGHTS[weight_name]
    weights = np.ones(len(frame), dtype=float)
    weights[frame.rain.to_numpy()] = rain_weight
    weights[frame.dry_overcast.to_numpy()] = cloud_weight
    return weights


def predictor(config_name: str) -> XGBRegressor:
    return XGBRegressor(
        **CONFIGS[config_name], objective="reg:squarederror",
        tree_method="hist", eval_metric="rmse",
        random_state=MODEL_SEED, n_jobs=2, verbosity=0,
    )


def scores(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    truth = frame[TARGET].to_numpy(dtype=float)
    require(len(truth) == len(pred) and np.isfinite(pred).all(),
            "Prediction invalid")
    result = {}
    masks = {
        "overall": np.ones(len(frame), dtype=bool),
        "rain": frame.rain.to_numpy(dtype=bool),
        "dry_overcast": frame.dry_overcast.to_numpy(dtype=bool),
        "clear_dry": frame.clear_dry.to_numpy(dtype=bool),
    }
    for name in EVENTS:
        mask = masks[name]
        require(mask.any(), f"Empty event group {name}")
        error = np.asarray(pred)[mask] - truth[mask]
        result[name] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    result["J"] = float(
        0.4 * result["rain"]["rmse"]
        + 0.4 * result["dry_overcast"]["rmse"]
        + 0.2 * result["overall"]["rmse"]
    )
    return result


def spec_id(view: str, config: str, weight: str) -> str:
    return f"{view}__{config}__{weight}"


def all_specs() -> list[dict]:
    return [
        {"view": view, "config": config, "weight": weight,
         "id": spec_id(view, config, weight)}
        for view in VIEWS for config in CONFIGS for weight in WEIGHTS
    ]


def audit_features(train: pd.DataFrame, names: list[str],
                   fingerprint: str) -> dict:
    x = train[names].replace([np.inf, -np.inf], np.nan)
    missing = x.isna().mean()
    constant = x.nunique(dropna=True)
    correlation = x.corr(method="spearman", min_periods=1000).to_numpy()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = correlation[i, j]
            if np.isfinite(value) and abs(value) >= 0.95:
                pairs.append({
                    "left": names[i], "right": names[j],
                    "spearman_rho": float(value),
                    "category": "base_base" if j < 188 else
                                "base_pressure" if i < 188 else "pressure_pressure",
                })
    excluded = pd.read_csv(DATA / "excluded_unreliable_hours.csv")
    rain_excluded = excluded.loc[excluded.rain.astype(bool)]
    return {
        "fingerprint": fingerprint,
        "train_hours": len(train),
        "feature_count": len(names),
        "constant_features": constant.index[constant.le(1)].tolist(),
        "missing_gt50_features": missing.index[missing.gt(0.5)].tolist(),
        "high_correlation_threshold": 0.95,
        "high_correlation_min_pairwise_rows": 1000,
        "high_correlation_pairs": pairs,
        "high_correlation_pair_counts": {
            category: sum(pair["category"] == category for pair in pairs)
            for category in ("base_base", "base_pressure", "pressure_pressure")
        },
        "valid_rain_hours": int(train.rain.sum()),
        "excluded_rain_hours_all_splits": len(rain_excluded),
        "excluded_rain_reasons": {
            str(key): int(value)
            for key, value in rain_excluded.exclusion_reason.value_counts().items()
        },
    }


def fit_fold(train: pd.DataFrame, names: list[str], local: list[str],
             fold: dict, spec: dict) -> dict:
    fit = train.iloc[fold["fit_indices"]].reset_index(drop=True)
    hold = train.iloc[fold["hold_indices"]].reset_index(drop=True)
    columns = feature_view(fit[names], names, local, spec["view"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xfit = imputer.fit_transform(fit[columns].replace([np.inf, -np.inf], np.nan))
    xhold = imputer.transform(hold[columns].replace([np.inf, -np.inf], np.nan))
    require(xfit.shape[1] == xhold.shape[1] == len(columns), "Imputation mismatch")
    yfit = fit[TARGET].to_numpy(dtype=float)
    yhold = hold[TARGET].to_numpy(dtype=float)
    model = predictor(spec["config"])
    started = time.perf_counter()
    model.fit(
        xfit, yfit, sample_weight=sample_weights(fit, spec["weight"]),
        eval_set=[(xfit, yfit), (xhold, yhold)], verbose=False,
    )
    elapsed = time.perf_counter() - started
    prediction = np.asarray(model.predict(xhold), dtype=float)
    history = model.evals_result()
    train_curve = history["validation_0"]["rmse"]
    hold_curve = history["validation_1"]["rmse"]
    require(len(train_curve) == len(hold_curve) == CONFIGS[spec["config"]]["n_estimators"],
            "Incomplete training curve")
    return {
        "fold": fold["fold"],
        "n_fit": len(fit),
        "n_hold": len(hold),
        "fit_days": fold["fit_days"],
        "hold_days": fold["hold_days"],
        "feature_names": columns,
        "feature_count": len(columns),
        "train_metrics": scores(fit, np.asarray(model.predict(xfit), dtype=float)),
        "hold_metrics": scores(hold, prediction),
        "hold_indices": fold["hold_indices"],
        "hold_hour_bjt": hold.hour_bjt.astype(str).tolist(),
        "hold_truth_db": yhold.tolist(),
        "hold_prediction_db": prediction.tolist(),
        "train_curve_rmse_db": [float(v) for v in train_curve],
        "hold_curve_rmse_db": [float(v) for v in hold_curve],
        "trained_trees": len(train_curve),
        "stopping_reason": "pre-registered fixed tree count; no holdout early stopping",
        "fit_seconds": float(elapsed),
    }


def prepare() -> None:
    with writer_lock():
        manifest = fingerprint_record()
        path = OUT / "run_manifest.json"
        if path.exists():
            require(read_json(path)["fingerprint"] == manifest["fingerprint"],
                    "Existing manifest differs; stop")
        else:
            names, local = names_and_local()
            train = load_partition("train", names)
            folds = grouped_folds(train)
            manifest["folds"] = folds
            manifest["train_event_counts"] = {
                key: int(train[key].sum()) for key in ("rain", "dry_overcast", "clear_dry")
            }
            manifest["local117_names"] = local
            atomic_json(path, manifest)
            atomic_json(OUT / "feature_audit.json",
                        audit_features(train, names, manifest["fingerprint"]))
        print(json.dumps({
            "status": "prepared", "fingerprint": manifest["fingerprint"],
            "expected_trials": manifest["expected_trials"],
            "expected_fits": manifest["expected_fits"],
        }))


def selftest() -> None:
    names, local = names_and_local()
    train = load_partition("train", names)
    folds = grouped_folds(train)
    for fold in folds:
        xfit = train.iloc[fold["fit_indices"]][names]
        counts = {view: len(feature_view(xfit, names, local, view)) for view in VIEWS}
        require(counts["local117"] == 117 and counts["base188"] == 188
                and counts["full348"] == 348, "Fixed views mismatch")
        require(counts["prune95"] <= counts["prune99"] <= counts["clean348"] <= 348,
                "Pruned feature counts mismatch")
        print(json.dumps({"fold": fold["fold"], "hold_rain": fold["hold_rain"],
                          "hold_overcast": fold["hold_dry_overcast"],
                          "view_feature_counts": counts}))


def run(limit: int | None) -> None:
    with writer_lock():
        manifest = checked_manifest()
        names, local = names_and_local()
        train = load_partition("train", names)
        count = 0
        for spec in all_specs():
            path = TRIALS / f"trial_{spec['id']}.json"
            if path.exists():
                row = read_json(path)
                require(row.get("fingerprint") == manifest["fingerprint"]
                        and row.get("spec") == spec and len(row.get("folds", [])) == 3,
                        f"Corrupt or mismatched trial: {path}")
                continue
            folds = [fit_fold(train, names, local, fold, spec)
                     for fold in manifest["folds"]]
            row = {
                "fingerprint": manifest["fingerprint"],
                "spec": spec,
                "config_parameters": CONFIGS[spec["config"]],
                "event_weights": WEIGHTS[spec["weight"]],
                "folds": folds,
                "completed_utc": utc_now(),
            }
            atomic_json(path, row)
            count += 1
            print(json.dumps({
                "trial": spec["id"], "completed_this_run": count,
                "fold_scores": [round(f["hold_metrics"]["J"], 6) for f in folds],
                "features": [f["feature_count"] for f in folds],
            }), flush=True)
            if limit is not None and count >= limit:
                break


def status() -> None:
    path = OUT / "run_manifest.json"
    if not path.exists():
        print(json.dumps({"stage": "not_prepared"}))
        return
    manifest = checked_manifest()
    done = 0
    for spec in all_specs():
        path = TRIALS / f"trial_{spec['id']}.json"
        if path.exists():
            row = read_json(path)
            require(row.get("fingerprint") == manifest["fingerprint"]
                    and len(row.get("folds", [])) == 3,
                    f"Trial fingerprint/fold mismatch: {path}")
            done += 1
    print(json.dumps({
        "stage": "test_complete" if (OUT / "final_test.json").exists()
        else "reported" if (OUT / "selection.json").exists()
        else "running" if done else "prepared",
        "fingerprint": manifest["fingerprint"],
        "trials_complete": done,
        "trials_expected": manifest["expected_trials"],
        "fits_complete": done * 3,
        "fits_expected": manifest["expected_fits"],
        "selection_exists": (OUT / "selection.json").exists(),
        "validation_exists": (OUT / "validation.json").exists(),
        "qa_exists": (OUT / "qa.json").exists(),
        "test_exists": (OUT / "final_test.json").exists(),
    }))


def validate_trial(row: dict, manifest: dict, train: pd.DataFrame) -> dict:
    require(row["fingerprint"] == manifest["fingerprint"], "Trial fingerprint mismatch")
    require(len(row["folds"]) == 3, "Trial fold count mismatch")
    totals = {name: [] for name in ("J", "overall", "rain", "dry_overcast")}
    feature_counts = []
    for saved, source in zip(row["folds"], manifest["folds"]):
        require(saved["fold"] == source["fold"], "Fold identity mismatch")
        require(saved["hold_indices"] == source["hold_indices"], "Fold hour mismatch")
        hold = train.iloc[source["hold_indices"]].reset_index(drop=True)
        require(len(saved["hold_prediction_db"]) == len(hold), "Prediction count mismatch")
        require(np.allclose(saved["hold_truth_db"], hold[TARGET], atol=1e-12),
                "Stored truth mismatch")
        recalc = scores(hold, np.asarray(saved["hold_prediction_db"], dtype=float))
        for name in EVENTS:
            for metric in ("n", "rmse", "mae", "bias"):
                require(np.isclose(recalc[name][metric], saved["hold_metrics"][name][metric],
                                   atol=1e-10), f"Recomputed {name}/{metric} mismatch")
        require(np.isclose(recalc["J"], saved["hold_metrics"]["J"], atol=1e-10),
                "Recomputed J mismatch")
        require(len(saved["feature_names"]) == saved["feature_count"],
                "Feature list count mismatch")
        require(len(saved["train_curve_rmse_db"]) == len(saved["hold_curve_rmse_db"])
                == CONFIGS[row["spec"]["config"]]["n_estimators"],
                "Round history mismatch")
        feature_counts.append(saved["feature_count"])
        totals["J"].append(recalc["J"])
        for name in ("overall", "rain", "dry_overcast"):
            totals[name].append(recalc[name]["rmse"])
    return {
        "id": row["spec"]["id"], **row["spec"],
        "mean_J": float(np.mean(totals["J"])),
        "mean_overall_rmse": float(np.mean(totals["overall"])),
        "mean_rain_rmse": float(np.mean(totals["rain"])),
        "mean_overcast_rmse": float(np.mean(totals["dry_overcast"])),
        "std_J": float(np.std(totals["J"], ddof=1)),
        "mean_features": float(np.mean(feature_counts)),
        "fold_J": totals["J"],
        "fold_feature_counts": feature_counts,
    }


def load_cv_results(manifest: dict, train: pd.DataFrame) -> list[dict]:
    rows = []
    for spec in all_specs():
        path = TRIALS / f"trial_{spec['id']}.json"
        require(path.exists(), f"Missing trial: {path.name}")
        record = read_json(path)
        require(record["spec"] == spec, f"Trial spec mismatch: {path.name}")
        rows.append(validate_trial(record, manifest, train))
    require(len(rows) == manifest["expected_trials"] == 72, "CV grid incomplete")
    return rows


def choose(rows: list[dict]) -> dict:
    def ranking(row):
        return (row["mean_J"], row["mean_features"], row["id"])

    reference = min(
        (r for r in rows if r["view"] == "full348" and r["weight"] == "uniform"),
        key=ranking,
    )
    feasible_overall = [
        r for r in rows
        if r["mean_overall_rmse"] <= 1.05 * reference["mean_overall_rmse"]
    ]
    require(feasible_overall, "Reference must satisfy overall constraint")
    feasible_both = [
        r for r in feasible_overall
        if r["mean_rain_rmse"] < reference["mean_rain_rmse"]
        and r["mean_overcast_rmse"] < reference["mean_overcast_rmse"]
    ]
    winner = min(feasible_both or feasible_overall, key=ranking)
    baseline_188 = min(
        (r for r in rows if r["view"] == "base188" and r["weight"] == "uniform"),
        key=ranking,
    )
    baseline_117 = min(
        (r for r in rows if r["view"] == "local117" and r["weight"] == "uniform"),
        key=ranking,
    )
    role_rows = {
        "winner": winner,
        "reference": reference,
        "base188": baseline_188,
        "local117": baseline_117,
    }
    return {
        "role_specs": {role: row["id"] for role, row in role_rows.items()},
        "both_event_improvement_in_training_cv": bool(feasible_both),
        "n_jointly_feasible": len(feasible_both),
        "n_overall_feasible": len(feasible_overall),
        "selection_rule": "pre-registered dual-event/overall constraint then minimum J",
        "reference_cv": reference,
        "winner_cv": winner,
    }


def fit_formal(role: str, spec: dict, train: pd.DataFrame, validation: pd.DataFrame,
               names: list[str], local: list[str], fingerprint: str) -> dict:
    columns = feature_view(train[names], names, local, spec["view"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xfit = imputer.fit_transform(train[columns].replace([np.inf, -np.inf], np.nan))
    xval = imputer.transform(validation[columns].replace([np.inf, -np.inf], np.nan))
    yfit = train[TARGET].to_numpy(dtype=float)
    yval = validation[TARGET].to_numpy(dtype=float)
    model = predictor(spec["config"])
    started = time.perf_counter()
    model.fit(
        xfit, yfit, sample_weight=sample_weights(train, spec["weight"]),
        eval_set=[(xfit, yfit), (xval, yval)], verbose=False,
    )
    elapsed = time.perf_counter() - started
    train_pred = np.asarray(model.predict(xfit), dtype=float)
    val_pred = np.asarray(model.predict(xval), dtype=float)
    curve = model.evals_result()
    model_path = MODELS / f"{role}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODELS / f".{role}.{os.getpid()}.tmp.json"
    model.save_model(temporary)
    os.replace(temporary, model_path)
    require(len(imputer.statistics_) == len(columns)
            and np.isfinite(imputer.statistics_).all(), "Invalid formal imputer")
    return {
        "role": role,
        "spec": spec,
        "fingerprint": fingerprint,
        "model_file": str(model_path.relative_to(OUT)),
        "model_sha256": sha256(model_path),
        "feature_names": columns,
        "feature_count": len(columns),
        "imputer_medians": imputer.statistics_.astype(float).tolist(),
        "config_parameters": CONFIGS[spec["config"]],
        "event_weights": WEIGHTS[spec["weight"]],
        "trained_trees": CONFIGS[spec["config"]]["n_estimators"],
        "stopping_reason": "fixed trees chosen by train-only grouped CV",
        "fit_seconds": float(elapsed),
        "train_metrics": scores(train, train_pred),
        "validation_metrics": scores(validation, val_pred),
        "validation_truth_db": yval.tolist(),
        "validation_prediction_db": val_pred.tolist(),
        "rounds_train_rmse_db": [float(v) for v in curve["validation_0"]["rmse"]],
        "rounds_validation_rmse_db": [float(v) for v in curve["validation_1"]["rmse"]],
    }


def report() -> None:
    with writer_lock():
        manifest = checked_manifest()
        names, local = names_and_local()
        train = load_partition("train", names)
        validation = load_partition("validation", names)
        rows = load_cv_results(manifest, train)
        selection = choose(rows)
        selection["fingerprint"] = manifest["fingerprint"]
        selection["selected_utc"] = utc_now()
        path = OUT / "selection.json"
        if path.exists():
            require(read_json(path)["role_specs"] == selection["role_specs"],
                    "Frozen selection differs; stop")
        else:
            atomic_json(path, selection)
        flat = [{k: v for k, v in row.items() if k not in ("fold_J", "fold_feature_counts")}
                for row in rows]
        pd.DataFrame(flat).to_csv(OUT / "cv_results.csv", index=False)
        by_id = {row["id"]: row for row in all_specs()}
        formal = {
            role: fit_formal(role, by_id[selection["role_specs"][role]],
                             train, validation, names, local, manifest["fingerprint"])
            for role in ROLES
        }
        atomic_json(OUT / "validation.json", {
            "fingerprint": manifest["fingerprint"],
            "selected_roles": selection["role_specs"],
            "n_train": len(train),
            "n_validation": len(validation),
            "validation_hour_bjt": validation.hour_bjt.astype(str).tolist(),
            "models": formal,
            "validation_not_used_to_select": True,
            "completed_utc": utc_now(),
        })
        print(json.dumps({
            "status": "reported",
            "winner": selection["role_specs"]["winner"],
            "reference": selection["role_specs"]["reference"],
            "both_event_improvement_cv": selection["both_event_improvement_in_training_cv"],
            "validation_winner": formal["winner"]["validation_metrics"],
        }, ensure_ascii=False))


def verify() -> None:
    manifest = checked_manifest()
    names, _ = names_and_local()
    train = load_partition("train", names)
    audit = read_json(OUT / "feature_audit.json")
    require(audit["fingerprint"] == manifest["fingerprint"]
            and audit["feature_count"] == 348
            and len(audit["high_correlation_pairs"]) ==
                sum(audit["high_correlation_pair_counts"].values()),
            "Feature audit mismatch")
    rows = load_cv_results(manifest, train)
    selection = read_json(OUT / "selection.json")
    require(selection["fingerprint"] == manifest["fingerprint"],
            "Selection fingerprint mismatch")
    require(selection["role_specs"] == choose(rows)["role_specs"],
            "Selection rule recomputation mismatch")
    validation = read_json(OUT / "validation.json")
    require(validation["fingerprint"] == manifest["fingerprint"]
            and validation["selected_roles"] == selection["role_specs"],
            "Validation selection mismatch")
    val = load_partition("validation", names)
    require(validation["validation_hour_bjt"] == val.hour_bjt.astype(str).tolist(),
            "Validation hour mismatch")
    for role in ROLES:
        row = validation["models"][role]
        require(row["fingerprint"] == manifest["fingerprint"], "Model fingerprint mismatch")
        model_file = OUT / row["model_file"]
        require(model_file.exists() and sha256(model_file) == row["model_sha256"],
                f"Missing or changed frozen model {role}")
        require(len(row["feature_names"]) == row["feature_count"]
                == len(row["imputer_medians"]), "Formal feature/imputer mismatch")
        pred = np.asarray(row["validation_prediction_db"], dtype=float)
        frozen = predict_frozen(row, val)
        require(np.allclose(pred, frozen, atol=1e-6),
                f"Saved validation prediction/model mismatch {role}")
        recalc = scores(val, pred)
        for name in EVENTS:
            require(np.isclose(recalc[name]["rmse"],
                               row["validation_metrics"][name]["rmse"], atol=1e-10),
                    f"Validation {role}/{name} RMSE mismatch")
        require(len(row["rounds_train_rmse_db"])
                == len(row["rounds_validation_rmse_db"])
                == row["trained_trees"], f"Formal rounds mismatch {role}")
    if (OUT / "final_test.json").exists():
        test = read_json(OUT / "final_test.json")
        require(test["fingerprint"] == manifest["fingerprint"], "Test fingerprint mismatch")
        require(test["test_parquet_sha256"] == sha256(DATA / "test.parquet"),
                "Test parquet changed")
        test_frame = load_partition("test", names)
        for role in ROLES:
            pred = np.asarray(test["models"][role]["prediction_db"], dtype=float)
            recalc = scores(test_frame, pred)
            for name in EVENTS:
                require(np.isclose(recalc[name]["rmse"],
                                   test["models"][role]["metrics"][name]["rmse"],
                                   atol=1e-10), "Test metric recomputation mismatch")
    qa = {
        "status": "PASS",
        "fingerprint": manifest["fingerprint"],
        "trials": len(rows),
        "fits": len(rows) * 3,
        "formal_models": len(ROLES),
        "test_checked": (OUT / "final_test.json").exists(),
        "verified_utc": utc_now(),
    }
    atomic_json(OUT / "qa.json", qa)
    print(json.dumps(qa))


def predict_frozen(row: dict, frame: pd.DataFrame) -> np.ndarray:
    path = OUT / row["model_file"]
    require(sha256(path) == row["model_sha256"], "Frozen model hash mismatch")
    model = XGBRegressor()
    model.load_model(path)
    x = frame[row["feature_names"]].to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    medians = np.asarray(row["imputer_medians"], dtype=float)
    x = np.where(np.isnan(x), medians[None, :], x)
    result = np.asarray(model.predict(x), dtype=float)
    require(np.isfinite(result).all(), "Nonfinite frozen predictions")
    return result


def bootstrap_difference(frame: pd.DataFrame, reference: np.ndarray,
                         candidate: np.ndarray, event: str) -> dict:
    mask = (np.ones(len(frame), dtype=bool) if event == "overall"
            else frame[event].to_numpy(dtype=bool))
    indices = np.flatnonzero(mask)
    days = frame.hour_bjt.dt.strftime("%Y-%m-%d").to_numpy()
    grouped = {}
    for index in indices:
        grouped.setdefault(days[index], []).append(index)
    chunks = [np.asarray(value, dtype=int) for value in grouped.values()]
    truth = frame[TARGET].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED + EVENTS.index(event))

    def rmse(pred: np.ndarray, select: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred[select] - truth[select]) ** 2)))

    observed = rmse(candidate, indices) - rmse(reference, indices)
    draws = np.empty(4000, dtype=float)
    for i in range(len(draws)):
        choice = rng.integers(0, len(chunks), size=len(chunks))
        draw = np.concatenate([chunks[j] for j in choice])
        draws[i] = rmse(candidate, draw) - rmse(reference, draw)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n_hours": int(len(indices)),
        "n_days": len(chunks),
        "candidate_minus_reference_rmse_db": observed,
        "day_block_bootstrap_95pct_db": [float(low), float(high)],
        "draws": len(draws),
        "seed": SEED + EVENTS.index(event),
    }


def test() -> None:
    with writer_lock():
        manifest = checked_manifest()
        qa = read_json(OUT / "qa.json")
        require(qa["status"] == "PASS" and qa["fingerprint"] == manifest["fingerprint"],
                "Pretest verification missing")
        path = OUT / "final_test.json"
        require(not path.exists(), "Fixed test already completed; do not repeat")
        validation = read_json(OUT / "validation.json")
        names, _ = names_and_local()
        frame = load_partition("test", names)
        models = {}
        predictions = {}
        for role in ROLES:
            prediction = predict_frozen(validation["models"][role], frame)
            predictions[role] = prediction
            models[role] = {
                "spec": validation["models"][role]["spec"],
                "model_sha256": validation["models"][role]["model_sha256"],
                "metrics": scores(frame, prediction),
                "prediction_db": prediction.tolist(),
            }
        differences = {
            role: {event: bootstrap_difference(frame, predictions["reference"],
                                               predictions[role], event)
                   for event in ("overall", "rain", "dry_overcast")}
            for role in ("winner", "base188", "local117")
        }
        atomic_json(path, {
            "fingerprint": manifest["fingerprint"],
            "test_parquet_sha256": sha256(DATA / "test.parquet"),
            "n_test": len(frame),
            "hour_bjt": frame.hour_bjt.astype(str).tolist(),
            "truth_db": frame[TARGET].astype(float).tolist(),
            "events": {event: frame[event].astype(bool).tolist()
                       for event in ("rain", "dry_overcast", "clear_dry")},
            "models": models,
            "paired_day_bootstrap": differences,
            "no_test_based_selection": True,
            "completed_utc": utc_now(),
        })
        print(json.dumps({
            "status": "test_complete",
            "winner": models["winner"]["spec"]["id"],
            "winner_metrics": models["winner"]["metrics"],
        }, ensure_ascii=False))


def document() -> None:
    manifest = checked_manifest()
    qa = read_json(OUT / "qa.json")
    require(qa["status"] == "PASS" and qa["test_checked"],
            "Final verification required before document")
    selection = read_json(OUT / "selection.json")
    validation = read_json(OUT / "validation.json")
    test_row = read_json(OUT / "final_test.json")
    lines = [
        "# Revision 015：雨天/阴天导向特征实验结果",
        "",
        f"输入与代码指纹：`{manifest['fingerprint']}`。训练 2404、验证 343、测试 687 小时。",
        "雨天包含所有原雨天记录，不另分强雨。目标是同小时相对晴空衰减（dB）。",
        "现有测试小时曾在历史实验中查看；本次锁定测试不称全新盲测。",
        "",
        "## 训练内选择",
        "",
        f"- 完整记录：{qa['trials']} 个候选、{qa['fits']} 次日分组拟合；逐树曲线、逐小时预测及特征名在 `trials/`。",
        f"- 胜出：`{selection['role_specs']['winner']}`；全量均匀权重参照：`{selection['role_specs']['reference']}`。",
        f"- 同时改善两种天气的训练内候选是否存在：`{selection['both_event_improvement_in_training_cv']}`；数量：{selection['n_jointly_feasible']}。",
        "- 训练内目标 J = 0.4×雨 RMSE + 0.4×干燥阴天 RMSE + 0.2×总体 RMSE；总体不得超过参照的 105%。",
        "",
        "## 固定方案的验证与测试",
        "",
        "| 方案 | 列数 | 验证总体 | 验证雨 | 验证阴天 | 测试总体 | 测试雨 | 测试阴天 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        v = validation["models"][role]
        t = test_row["models"][role]
        vm, tm = v["validation_metrics"], t["metrics"]
        lines.append(
            f"| {role} (`{v['spec']['id']}`) | {v['feature_count']} | "
            f"{vm['overall']['rmse']:.4f} | {vm['rain']['rmse']:.4f} | "
            f"{vm['dry_overcast']['rmse']:.4f} | {tm['overall']['rmse']:.4f} | "
            f"{tm['rain']['rmse']:.4f} | {tm['dry_overcast']['rmse']:.4f} |"
        )
    lines += [
        "",
        "完整超参数、固定树数、样本权重、每轮训练/验证 RMSE、输入列名和逐小时预测见 `validation.json`；",
        "最终测试各天气 n、RMSE、MAE、偏差与按日配对 bootstrap 区间见 `final_test.json`。",
        "所有方案均保留，即使某项未改善亦不删除。雨天结果只适用于质量合格、可检出的雨小时。",
        "",
    ]
    path = OUT / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "documented", "path": str(path)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "selftest", "prepare", "run", "status", "report", "verify", "test", "document"
    ])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "prepare":
        prepare()
    elif args.command == "run":
        run(args.limit)
    elif args.command == "status":
        status()
    elif args.command == "report":
        report()
    elif args.command == "verify":
        verify()
    elif args.command == "test":
        test()
    else:
        document()


if __name__ == "__main__":
    main()
