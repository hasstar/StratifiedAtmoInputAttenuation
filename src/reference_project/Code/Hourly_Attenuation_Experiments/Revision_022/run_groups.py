"""Training-only, unconstrained group selection for Revision 022."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_022_Natural_Selection"
MANIFEST = OUT / "group_manifest.json"
PLAN = OUT / "group_plan.json"
PLAN_QA = OUT / "group_plan_qa.json"
GROUP_QA = OUT / "group_search_qa.json"
SELECTION = OUT / "group_selection.json"
TARGET = "clear_sky_attenuation_db"
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
GROUPS = ("base", *(f"p{level}" for level in LEVELS))
EVENTS = ("overall", "rain", "dry_overcast", "clear_dry", "heavy_rain")
CONFIG = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03,
          "objective": "reg:squarederror", "tree_method": "hist", "subsample": 0.8,
          "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_jobs": 2,
          "eval_metric": "rmse", "early_stopping_rounds": 30, "verbosity": 0}
FOLD_SEEDS = (17, 29, 43)
VERSION = "revision022-natural-group-v1-20261002"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(data: dict) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@contextmanager
def writer_lock():
    lock = OUT / "group_worker.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_utc": utc_now()}))
    except FileExistsError as exc:
        raise RuntimeError(f"Another/stale worker lock exists: {lock}; inspect process first") from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def source_paths() -> list[Path]:
    return [HERE / "PROTOCOL.md", HERE / "run_groups.py",
            DATA / "feature_columns.csv", DATA / "train.parquet",
            DATA / "hourly_sample_status.parquet"]


def load_training() -> tuple[pd.DataFrame, list[str]]:
    names = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    require(len(names) == len(set(names)) == 348 and
            all(name.startswith("surface__") for name in names[:188]) and
            all(name.startswith("p") for name in names[188:]),
            "Original 348-column schema changed")
    frame = pd.read_parquet(DATA / "train.parquet")
    require(len(frame) == 2404 and list(frame.columns) ==
            ["hour_bjt", "hour_utc", *names, TARGET], "Training split changed")
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet",
                             columns=["hour_bjt", "split", "use_for_regression",
                                      "rain", "heavy_rain", "dry_overcast", "clear_dry"],
                             filters=[("split", "==", "train")])
    require(len(status) == 2404 and status["hour_bjt"].is_unique,
            "Training event metadata changed")
    frame = frame.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    frame = frame.sort_values("hour_bjt").reset_index(drop=True)
    require(frame["split"].eq("train").all() and
            frame["use_for_regression"].eq(True).all() and
            np.isfinite(frame[TARGET].to_numpy(dtype=float)).all() and
            frame[["rain", "heavy_rain", "dry_overcast", "clear_dry"]].notna().all().all(),
            "Training targets or event flags invalid")
    require(int(frame["rain"].sum()) == 179, "Training rain population changed")
    return frame, names


def make_folds(frame: pd.DataFrame) -> tuple[list[dict], dict]:
    category = np.where(frame["rain"].to_numpy(dtype=bool), "rain",
                 np.where(frame["dry_overcast"].to_numpy(dtype=bool),
                          "dry_overcast", "other"))
    days = frame["hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    best: tuple[float, int, list[dict]] | None = None
    for seed in range(64):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        folds = []
        for i, (fit, hold) in enumerate(splitter.split(
                np.zeros(len(frame)), category, groups=days)):
            fit_days, hold_days = set(days[fit]), set(days[hold])
            require(fit_days.isdisjoint(hold_days), "Same day crossed inner folds")
            counts = {label: int(np.sum(category[hold] == label))
                      for label in ("rain", "dry_overcast", "other")}
            folds.append({"id": i + 1, "fit_index": fit.tolist(),
                          "hold_index": hold.tolist(), "hold_counts": counts,
                          "fit_days": len(fit_days), "hold_days": len(hold_days)})
        if any(fold["hold_counts"]["rain"] == 0 or
               fold["hold_counts"]["dry_overcast"] == 0 for fold in folds):
            continue
        totals = {label: int(np.sum(category == label))
                  for label in ("rain", "dry_overcast", "other")}
        imbalance = sum((fold["hold_counts"][label] - totals[label] / 3) ** 2 /
                        max(totals[label], 1) for fold in folds
                        for label in ("rain", "dry_overcast", "other"))
        item = (float(imbalance), seed, folds)
        if best is None or item[:2] < best[:2]:
            best = item
    require(best is not None, "Could not form three day-disjoint event-complete folds")
    return best[2], {"strategy": "StratifiedGroupKFold_day", "n_splits": 3,
                     "candidate_seeds_checked": 64, "chosen_seed": best[1],
                     "event_balance_objective": best[0]}


def plan_candidate() -> dict:
    frame, names = load_training()
    excluded = {}
    eligible = []
    for name in names:
        values = frame[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            excluded[name] = "all_missing_in_training"
        elif float(finite.min()) == float(finite.max()):
            excluded[name] = "constant_in_training"
        else:
            eligible.append(name)
    group_map = {group: [name for name in eligible if
                 (name.startswith("surface__") if group == "base"
                  else name.startswith(f"{group}__"))] for group in GROUPS}
    require(all(group_map.values()) and sum(map(len, group_map.values())) ==
            len(eligible), "Candidate group assignment invalid")
    folds, strategy = make_folds(frame)
    return {"original_feature_names": names, "eligible_feature_names": eligible,
            "excluded_training_quality": excluded, "group_feature_names": group_map,
            "group_order": list(GROUPS), "fold_strategy": strategy, "folds": folds,
            "n_train": 2404, "n_rain_train": int(frame["rain"].sum()),
            "n_dry_overcast_train": int(frame["dry_overcast"].sum())}


def manifest_candidate(plan: dict) -> dict:
    payload = {"version": VERSION,
               "source_hashes": {str(path.relative_to(ROOT)): sha256(path)
                                 for path in source_paths()},
               "plan_content_sha256": content_hash(plan),
               "feature_selection_uses_test": False,
               "model_family": "xgboost", "model_config": CONFIG,
               "score": "equal_mean_RMSE_overall_rain_dry_overcast",
               "group_rule": "greedy_one_group_drop_refit_all_candidates",
               "selection_rule": "smallest_feature_count_within_one_SE_of_path_minimum",
               "no_layer_quota": True, "no_feature_count_bounds": True,
               "historically_accessed_test": True}
    payload["fingerprint"] = content_hash(payload)
    return payload


def prepare() -> dict:
    plan = plan_candidate()
    manifest = manifest_candidate(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() or PLAN.exists():
        require(MANIFEST.exists() and PLAN.exists() and
                read_json(MANIFEST)["fingerprint"] == manifest["fingerprint"] and
                content_hash(read_json(PLAN)) == content_hash(plan),
                "Existing frozen group plan differs; create a new revision")
        return {"fingerprint": manifest["fingerprint"],
                "n_eligible": len(plan["eligible_feature_names"]),
                "n_excluded": len(plan["excluded_training_quality"])}
    atomic_json(PLAN, plan)
    manifest["created_utc"] = utc_now()
    atomic_json(MANIFEST, manifest)
    return {"fingerprint": manifest["fingerprint"],
            "n_eligible": len(plan["eligible_feature_names"]),
            "n_excluded": len(plan["excluded_training_quality"]),
            "group_sizes": {k: len(v) for k, v in plan["group_feature_names"].items()},
            "fold_seed": plan["fold_strategy"]["chosen_seed"]}


def checked() -> tuple[dict, dict]:
    require(MANIFEST.exists() and PLAN.exists(), "Run prepare first")
    plan = read_json(PLAN)
    frozen = read_json(MANIFEST)
    require(content_hash(plan) == frozen["plan_content_sha256"] and
            frozen["fingerprint"] == manifest_candidate(plan_candidate())["fingerprint"],
            "Frozen group plan/source fingerprint changed")
    return plan, frozen


def verify_prepare() -> dict:
    plan, manifest = checked()
    frame, names = load_training()
    require(len(names) == 348 and len(plan["eligible_feature_names"]) +
            len(plan["excluded_training_quality"]) == 348 and
            all(len(fold["fit_index"]) > 0 and len(fold["hold_index"]) > 0
                for fold in plan["folds"]), "Feature eligibility/folds invalid")
    for fold in plan["folds"]:
        fit, hold = fold["fit_index"], fold["hold_index"]
        require(set(fit).isdisjoint(hold) and
                set(frame.loc[fit, "hour_bjt"].dt.strftime("%Y-%m-%d")).isdisjoint(
                    frame.loc[hold, "hour_bjt"].dt.strftime("%Y-%m-%d")),
                "Fold leakage by calendar day")
        require(int(frame.loc[hold, "rain"].sum()) ==
                fold["hold_counts"]["rain"] and
                int(frame.loc[hold, "dry_overcast"].sum()) ==
                fold["hold_counts"]["dry_overcast"], "Fold event counts changed")
    qa = {"status": "PASS", "stage": "group_plan",
          "fingerprint": manifest["fingerprint"],
          "eligible_columns": len(plan["eligible_feature_names"]),
          "excluded_columns": len(plan["excluded_training_quality"]),
          "folds": len(plan["folds"]), "test_target_read": False,
          "verified_utc": utc_now()}
    atomic_json(PLAN_QA, qa)
    return qa


def names_for_mask(mask: int, plan: dict) -> list[str]:
    require(0 < mask < (1 << len(GROUPS)), "Group mask out of bounds")
    selected = {group for i, group in enumerate(GROUPS) if mask & (1 << i)}
    names = [name for name in plan["eligible_feature_names"] if
             ("base" if name.startswith("surface__") else name.split("__", 1)[0])
             in selected]
    require(len(names) == sum(len(plan["group_feature_names"][group])
                              for group in selected), "Mask feature list mismatch")
    return names


def metric_record(frame: pd.DataFrame, hold: np.ndarray,
                  pred: np.ndarray) -> dict:
    truth = frame.loc[hold, TARGET].to_numpy(dtype=float)
    require(len(truth) == len(pred) and np.isfinite(pred).all(),
            "Invalid validation prediction")
    result = {}
    for event in EVENTS:
        mask = (np.ones(len(hold), dtype=bool) if event == "overall" else
                frame.loc[hold, event].to_numpy(dtype=bool))
        err = pred[mask] - truth[mask]
        result[event] = {"n": int(mask.sum()),
                         "rmse": None if len(err) == 0 else
                         float(np.sqrt(np.mean(err ** 2))),
                         "mae": None if len(err) == 0 else
                         float(np.mean(np.abs(err)))}
    require(all(result[event]["n"] > 0 for event in
                ("overall", "rain", "dry_overcast")),
            "Event-complete fold assumption failed")
    result["macro_rmse"] = float(np.mean([
        result[event]["rmse"] for event in
        ("overall", "rain", "dry_overcast")]))
    return result


def candidate_path(mask: int) -> Path:
    return OUT / "group_candidates" / f"mask_{mask:04d}.json"


def state_path(count: int) -> Path:
    return OUT / "group_states" / f"groups_{count:02d}.json"


def fit_candidate(mask: int, plan: dict, manifest: dict,
                  frame: pd.DataFrame) -> dict:
    names = names_for_mask(mask, plan)
    folds = []
    for fold, seed in zip(plan["folds"], FOLD_SEEDS):
        fit = np.asarray(fold["fit_index"], dtype=int)
        hold = np.asarray(fold["hold_index"], dtype=int)
        xfit = frame.loc[fit, names].to_numpy(dtype=float, copy=True)
        xhold = frame.loc[hold, names].to_numpy(dtype=float, copy=True)
        xfit[~np.isfinite(xfit)] = np.nan
        xhold[~np.isfinite(xhold)] = np.nan
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        xfit = imputer.fit_transform(xfit)
        xhold = imputer.transform(xhold)
        require(xfit.shape == (len(fit), len(names)) and
                xhold.shape == (len(hold), len(names)) and
                np.isfinite(xfit).all() and np.isfinite(xhold).all(),
                f"Fold preprocessing invalid mask={mask}")
        yfit = frame.loc[fit, TARGET].to_numpy(dtype=float)
        yhold = frame.loc[hold, TARGET].to_numpy(dtype=float)
        model = XGBRegressor(**CONFIG, random_state=seed)
        start = time.perf_counter()
        model.fit(xfit, yfit, eval_set=[(xfit, yfit), (xhold, yhold)], verbose=False)
        fit_seconds = time.perf_counter() - start
        curve = model.evals_result()
        best_round = int(model.best_iteration) + 1
        pred = np.asarray(model.predict(xhold, iteration_range=(0, best_round)),
                          dtype=float)
        scores = metric_record(frame, hold, pred)
        require(abs(scores["overall"]["rmse"] -
                    float(curve["validation_1"]["rmse"][best_round - 1])) < 1e-5,
                f"Selected validation round mismatch mask={mask}")
        folds.append({"id": fold["id"], "seed": seed,
                      "fit_n": len(fit), "hold_n": len(hold),
                      "best_round": best_round,
                      "trained_rounds": len(curve["validation_1"]["rmse"]),
                      "stop_reason": "validation_patience_30" if
                      len(curve["validation_1"]["rmse"]) < 300 else "round_cap_300",
                      "fit_seconds": fit_seconds, "metrics": scores,
                      "training_rmse_by_round": curve["validation_0"]["rmse"],
                      "holdout_rmse_by_round": curve["validation_1"]["rmse"],
                      "holdout_prediction_db": pred.tolist()})
    values = np.asarray([fold["metrics"]["macro_rmse"] for fold in folds])
    return {"fingerprint": manifest["fingerprint"], "mask": mask,
            "groups": [group for i, group in enumerate(GROUPS) if mask & (1 << i)],
            "feature_names": names, "n_features": len(names),
            "config": CONFIG, "folds": folds,
            "macro_mean_db": float(values.mean()),
            "macro_se_db": float(values.std(ddof=1) / np.sqrt(len(values))),
            "completed_utc": utc_now()}


def candidate(mask: int, plan: dict, manifest: dict,
              frame: pd.DataFrame) -> dict:
    path = candidate_path(mask)
    if path.exists():
        row = read_json(path)
        require(row["fingerprint"] == manifest["fingerprint"] and
                row["mask"] == mask and
                row["feature_names"] == names_for_mask(mask, plan),
                f"Checkpoint mismatch: {path}")
        return row
    row = fit_candidate(mask, plan, manifest, frame)
    atomic_json(path, row)
    print(f"CANDIDATE mask={mask} groups={len(row['groups'])} "
          f"features={row['n_features']} macro={row['macro_mean_db']:.6f}", flush=True)
    return row


def choose_path(rows: list[dict]) -> dict:
    best = min(rows, key=lambda row: (row["macro_mean_db"], row["mask"]))
    limit = best["macro_mean_db"] + best["macro_se_db"]
    selected = min((row for row in rows if row["macro_mean_db"] <= limit + 1e-12),
                   key=lambda row: (row["n_features"], row["macro_mean_db"], row["mask"]))
    return {"minimum_score_mask": best["mask"], "minimum_macro_db":
            best["macro_mean_db"], "one_se_db": best["macro_se_db"],
            "eligibility_threshold_db": limit, "selected_mask": selected["mask"],
            "selected_groups": selected["groups"],
            "selected_feature_names": selected["feature_names"],
            "selected_macro_db": selected["macro_mean_db"],
            "selected_n_features": selected["n_features"]}


def groups() -> None:
    plan, manifest = checked()
    require(PLAN_QA.exists() and read_json(PLAN_QA)["status"] == "PASS" and
            read_json(PLAN_QA)["fingerprint"] == manifest["fingerprint"],
            "Group plan QA must pass")
    frame, _ = load_training()
    with writer_lock():
        mask = (1 << len(GROUPS)) - 1
        path_rows = []
        for count in range(len(GROUPS), 0, -1):
            current = candidate(mask, plan, manifest, frame)
            path_rows.append(current)
            options = []
            if count > 1:
                for i, group in enumerate(GROUPS):
                    if mask & (1 << i):
                        row = candidate(mask & ~(1 << i), plan, manifest, frame)
                        options.append((row["macro_mean_db"], i, group, row))
                _, _, removed, chosen = min(options, key=lambda item: item[:2])
                next_mask = chosen["mask"]
            else:
                removed, next_mask = None, None
            state = {"fingerprint": manifest["fingerprint"],
                     "group_count": count, "current_mask": mask,
                     "current_feature_names": current["feature_names"],
                     "current_macro_db": current["macro_mean_db"],
                     "candidate_masks": [row[3]["mask"] for row in options],
                     "removed_group_for_next": removed, "next_mask": next_mask,
                     "completed_utc": utc_now()}
            path = state_path(count)
            if path.exists():
                old = read_json(path)
                require(old["fingerprint"] == state["fingerprint"] and
                        old["current_mask"] == mask and
                        old["removed_group_for_next"] == removed and
                        old["next_mask"] == next_mask,
                        f"Group state checkpoint mismatch: {path}")
            else:
                atomic_json(path, state)
            print(f"STATE groups={count} mask={mask} removed={removed} "
                  f"macro={current['macro_mean_db']:.6f}", flush=True)
            if next_mask is not None:
                mask = next_mask
        decision = choose_path(path_rows)
        decision.update({"fingerprint": manifest["fingerprint"],
                         "path_masks": [row["mask"] for row in path_rows],
                         "completed_utc": utc_now()})
        if SELECTION.exists():
            old = read_json(SELECTION)
            require(old["fingerprint"] == decision["fingerprint"] and
                    old["selected_mask"] == decision["selected_mask"],
                    "Group selection checkpoint mismatch")
        else:
            atomic_json(SELECTION, decision)
        print(f"GROUP_PATH_FROZEN selected_mask={decision['selected_mask']} "
              f"selected_groups={decision['selected_groups']}", flush=True)


def verify_groups() -> dict:
    plan, manifest = checked()
    require(PLAN_QA.exists() and read_json(PLAN_QA)["status"] == "PASS",
            "Group plan QA missing")
    require(len(list((OUT / "group_states").glob("groups_*.json"))) == 11 and
            len(list((OUT / "group_candidates").glob("mask_*.json"))) == 66 and
            SELECTION.exists(), "Group path artifact count mismatch")
    frame, _ = load_training()
    mask = (1 << len(GROUPS)) - 1
    path_rows = []
    for count in range(len(GROUPS), 0, -1):
        state = read_json(state_path(count))
        row = read_json(candidate_path(mask))
        require(state["fingerprint"] == row["fingerprint"] ==
                manifest["fingerprint"] and state["current_mask"] == mask and
                row["feature_names"] == names_for_mask(mask, plan) and
                state["current_feature_names"] == row["feature_names"] and
                abs(state["current_macro_db"] - row["macro_mean_db"]) < 1e-12,
                f"Group path mismatch count={count}")
        path_rows.append(row)
        options = []
        if count > 1:
            for i, group in enumerate(GROUPS):
                if mask & (1 << i):
                    option = read_json(candidate_path(mask & ~(1 << i)))
                    require(option["fingerprint"] == manifest["fingerprint"] and
                            option["feature_names"] ==
                            names_for_mask(option["mask"], plan),
                            f"Candidate identity mismatch mask={option['mask']}")
                    options.append((option["macro_mean_db"], i, group, option))
            _, _, removed, chosen = min(options, key=lambda item: item[:2])
            require(state["candidate_masks"] == [x[3]["mask"] for x in options] and
                    state["removed_group_for_next"] == removed and
                    state["next_mask"] == chosen["mask"],
                    f"Greedy deletion mismatch count={count}")
            mask = chosen["mask"]
        else:
            require(state["removed_group_for_next"] is None and
                    state["next_mask"] is None, "Final group state invalid")
    seen = set()
    for path in (OUT / "group_candidates").glob("mask_*.json"):
        row = read_json(path)
        require(row["mask"] not in seen and row["fingerprint"] ==
                manifest["fingerprint"] and len(row["folds"]) == 3 and
                row["feature_names"] == names_for_mask(row["mask"], plan),
                f"Candidate invalid: {path}")
        seen.add(row["mask"])
        scores = []
        for fold_row, fold in zip(row["folds"], plan["folds"]):
            hold = np.asarray(fold["hold_index"], dtype=int)
            pred = np.asarray(fold_row["holdout_prediction_db"], dtype=float)
            replay = metric_record(frame, hold, pred)
            require(fold_row["id"] == fold["id"] and len(pred) == len(hold) and
                    fold_row["fit_n"] == len(fold["fit_index"]) and
                    fold_row["hold_n"] == len(hold) and
                    1 <= fold_row["best_round"] <= fold_row["trained_rounds"] <= 300 and
                    len(fold_row["training_rmse_by_round"]) ==
                    len(fold_row["holdout_rmse_by_round"]) ==
                    fold_row["trained_rounds"] and
                    all(abs(replay[event][metric] -
                            fold_row["metrics"][event][metric]) < 1e-10
                        for event in ("overall", "rain", "dry_overcast")
                        for metric in ("rmse", "mae")) and
                    abs(replay["macro_rmse"] -
                        fold_row["metrics"]["macro_rmse"]) < 1e-10,
                    f"Candidate metric/curve mismatch mask={row['mask']}")
            scores.append(replay["macro_rmse"])
        values = np.asarray(scores)
        require(abs(float(values.mean()) - row["macro_mean_db"]) < 1e-10 and
                abs(float(values.std(ddof=1) / np.sqrt(3)) -
                    row["macro_se_db"]) < 1e-10,
                f"Candidate fold score mismatch mask={row['mask']}")
    selected = read_json(SELECTION)
    expected = choose_path(path_rows)
    require(selected["fingerprint"] == manifest["fingerprint"] and
            selected["path_masks"] == [row["mask"] for row in path_rows] and
            all(selected[key] == value for key, value in expected.items()),
            "Group-stage one-SE decision mismatch")
    qa = {"status": "PASS", "stage": "group_search",
          "fingerprint": manifest["fingerprint"], "group_states": 11,
          "candidate_subsets": 66, "candidate_fits": 198,
          "selected_mask": selected["selected_mask"],
          "selected_groups": selected["selected_groups"],
          "selected_n_features": selected["selected_n_features"],
          "test_target_read": False, "verified_utc": utc_now()}
    atomic_json(GROUP_QA, qa)
    return qa


def status() -> dict:
    def count(folder: str, pattern: str) -> int:
        return len(list((OUT / folder).glob(pattern)))
    return {"manifest": MANIFEST.exists(),
            "plan_qa": read_json(PLAN_QA).get("status") if PLAN_QA.exists() else None,
            "group_states": count("group_states", "groups_*.json"),
            "group_candidates": count("group_candidates", "mask_*.json"),
            "selected": SELECTION.exists(),
            "group_qa": read_json(GROUP_QA).get("status") if GROUP_QA.exists() else None,
            "worker_lock": (OUT / "group_worker.lock").exists()}


def smoke() -> dict:
    frame, names = load_training()
    x = frame.loc[:159, names[:5]].to_numpy(dtype=float, copy=True)
    hold = frame.loc[160:239, names[:5]].to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    hold[~np.isfinite(hold)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x, hold = imputer.fit_transform(x), imputer.transform(hold)
    model = XGBRegressor(**{**CONFIG, "n_estimators": 20,
                            "early_stopping_rounds": 3}, random_state=17)
    model.fit(x, frame.loc[:159, TARGET].to_numpy(dtype=float),
              eval_set=[(hold, frame.loc[160:239, TARGET].to_numpy(dtype=float))],
              verbose=False)
    pred = model.predict(hold)
    require(len(pred) == 80 and np.isfinite(pred).all(), "Smoke fit failed")
    return {"status": "PASS", "best_round": int(model.best_iteration) + 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "prepare", "verify-prepare",
                                            "groups", "verify-groups", "status"))
    args = parser.parse_args()
    action = {"smoke": smoke, "prepare": prepare,
              "verify-prepare": verify_prepare, "groups": groups,
              "verify-groups": verify_groups, "status": status}[args.command]
    result = action()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
