"""Training-only, held-out-permutation recursive column selection for Revision 022.

This file never imports the formal validation or test split. The base-only and
group-selected branches are both recorded; identical pools share computation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

import run_groups as g

HERE = Path(__file__).resolve().parent
OUT = g.OUT
MANIFEST = OUT / "column_manifest.json"
PLAN = OUT / "column_plan.json"
PLAN_QA = OUT / "column_plan_qa.json"
SELECT_QA = OUT / "column_selection_qa.json"
VERSION = "revision022-permutation-rfe-v1-20261002"


@contextmanager
def writer_lock():
    path = OUT / "column_worker.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_utc": g.utc_now()}))
    except FileExistsError as exc:
        raise RuntimeError(f"Column worker lock exists: {path}; inspect process first") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def pool_plan() -> dict:
    group_plan, group_manifest = g.checked()
    g.require(g.GROUP_QA.exists() and g.read_json(g.GROUP_QA)["status"] == "PASS",
              "Group search QA must PASS before column selection")
    group_choice = g.read_json(g.SELECTION)
    g.require(group_choice["fingerprint"] == group_manifest["fingerprint"],
              "Frozen group choice fingerprint mismatch")
    base = group_plan["group_feature_names"]["base"]
    selected = group_choice["selected_feature_names"]
    g.require(bool(base) and bool(selected) and
              set(base).issubset(group_plan["eligible_feature_names"]) and
              set(selected).issubset(group_plan["eligible_feature_names"]),
              "Column pools invalid")
    alias = "base" if base == selected else None
    return {"pools": {"base": base, "group_selected": selected},
            "computed_branches": ["base"] if alias else ["base", "group_selected"],
            "group_selected_alias_of": alias,
            "group_search_fingerprint": group_manifest["fingerprint"],
            "group_selection_sha256": g.sha256(g.SELECTION),
            "group_qa_sha256": g.sha256(g.GROUP_QA),
            "folds_sha256": g.content_hash({"folds": group_plan["folds"]}),
            "method": "heldout_one_permutation_per_column_per_fold; drop_min_mean_delta",
            "score": "equal_mean_RMSE_overall_rain_dry_overcast",
            "one_se_rule": "smallest_k_within_SE_of_path_minimum",
            "event_columns_as_predictors": False,
            "validation_or_test_for_selection": False,
            "no_forced_vertical_feature_or_feature_count": True}


def manifest_candidate(plan: dict) -> dict:
    sources = [HERE / "PROTOCOL.md", HERE / "run_columns.py",
               g.MANIFEST, g.PLAN, g.GROUP_QA, g.SELECTION]
    payload = {"version": VERSION,
               "source_hashes": {str(p.relative_to(g.ROOT)): g.sha256(p) for p in sources},
               "plan_content_sha256": g.content_hash(plan),
               "group_fingerprint": plan["group_search_fingerprint"],
               "model_config": g.CONFIG, "inner_fold_seeds": list(g.FOLD_SEEDS),
               "permutation_repeats": 1,
               "test_target_read": False}
    payload["fingerprint"] = g.content_hash(payload)
    return payload


def prepare() -> dict:
    plan = pool_plan()
    manifest = manifest_candidate(plan)
    if MANIFEST.exists() or PLAN.exists():
        g.require(MANIFEST.exists() and PLAN.exists() and
                  g.read_json(MANIFEST)["fingerprint"] == manifest["fingerprint"] and
                  g.content_hash(g.read_json(PLAN)) == g.content_hash(plan),
                  "Frozen column plan differs; create new stage/revision")
    else:
        g.atomic_json(PLAN, plan)
        manifest["created_utc"] = g.utc_now()
        g.atomic_json(MANIFEST, manifest)
    return {"fingerprint": manifest["fingerprint"],
            "branch_sizes": {k: len(v) for k, v in plan["pools"].items()},
            "group_selected_alias_of": plan["group_selected_alias_of"]}


def checked() -> tuple[dict, dict, dict]:
    g.require(PLAN.exists() and MANIFEST.exists(), "Run column prepare first")
    plan, frozen = g.read_json(PLAN), g.read_json(MANIFEST)
    g.require(g.content_hash(plan) == frozen["plan_content_sha256"] and
              frozen["fingerprint"] == manifest_candidate(pool_plan())["fingerprint"],
              "Frozen column source/plan fingerprint changed")
    group_plan, _ = g.checked()
    return plan, frozen, group_plan


def verify_prepare() -> dict:
    plan, manifest, group_plan = checked()
    frame, _ = g.load_training()
    g.require(len(frame) == 2404 and len(group_plan["folds"]) == 3 and
              all(len(names) > 0 and len(set(names)) == len(names)
                  for names in plan["pools"].values()),
              "Column pool or training folds invalid")
    for branch, names in plan["pools"].items():
        g.require(all(name in group_plan["eligible_feature_names"] for name in names),
                  f"Ineligible column in {branch}")
    if plan["group_selected_alias_of"]:
        g.require(plan["pools"]["base"] == plan["pools"]["group_selected"],
                  "Alias pools differ")
    qa = {"status": "PASS", "stage": "column_plan",
          "fingerprint": manifest["fingerprint"],
          "computed_branches": plan["computed_branches"],
          "group_selected_alias_of": plan["group_selected_alias_of"],
          "formal_validation_or_test_read": False, "verified_utc": g.utc_now()}
    g.atomic_json(PLAN_QA, qa)
    return qa


def state_path(branch: str, k: int) -> Path:
    return OUT / "column_states" / branch / f"k_{k:03d}.json"


def decision_path(branch: str) -> Path:
    return OUT / f"column_selection_{branch}.json"


def perm_seed(branch: str, k: int, fold_id: int, original_index: int) -> int:
    branch_id = 1 if branch == "base" else 2
    return (branch_id * 1_000_000_007 + k * 1_000_003 +
            fold_id * 10_007 + original_index * 101) % (2 ** 32)


def fit_state(branch: str, names: list[str], pool: list[str],
              frame, group_plan: dict, fingerprint: str) -> dict:
    k = len(names)
    folds_out = []
    deltas = np.zeros((3, k), dtype=float)
    original_indices = [pool.index(name) for name in names]
    for fold_pos, (fold, seed) in enumerate(zip(group_plan["folds"], g.FOLD_SEEDS)):
        fit = np.asarray(fold["fit_index"], dtype=int)
        hold = np.asarray(fold["hold_index"], dtype=int)
        xfit = frame.loc[fit, names].to_numpy(dtype=float, copy=True)
        xhold = frame.loc[hold, names].to_numpy(dtype=float, copy=True)
        xfit[~np.isfinite(xfit)] = np.nan
        xhold[~np.isfinite(xhold)] = np.nan
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        xfit = imputer.fit_transform(xfit)
        xhold = imputer.transform(xhold)
        g.require(xfit.shape == (len(fit), k) and xhold.shape == (len(hold), k) and
                  np.isfinite(xfit).all() and np.isfinite(xhold).all(),
                  f"Fold imputation invalid {branch} k={k}")
        yfit = frame.loc[fit, g.TARGET].to_numpy(dtype=float)
        yhold = frame.loc[hold, g.TARGET].to_numpy(dtype=float)
        model = XGBRegressor(**g.CONFIG, random_state=seed)
        start = time.perf_counter()
        model.fit(xfit, yfit, eval_set=[(xfit, yfit), (xhold, yhold)], verbose=False)
        seconds = time.perf_counter() - start
        curve = model.evals_result()
        best_round = int(model.best_iteration) + 1
        pred = np.asarray(model.predict(xhold, iteration_range=(0, best_round)),
                          dtype=float)
        metrics = g.metric_record(frame, hold, pred)
        g.require(abs(metrics["overall"]["rmse"] -
                      float(curve["validation_1"]["rmse"][best_round - 1])) < 1e-5,
                  f"Best round replay mismatch {branch} k={k}")
        if k > 1:
            scratch = xhold.copy()
            for j, original_index in enumerate(original_indices):
                perm = np.random.default_rng(perm_seed(
                    branch, k, int(fold["id"]), original_index)).permutation(len(hold))
                scratch[:, j] = xhold[perm, j]
                changed = np.asarray(model.predict(
                    scratch, iteration_range=(0, best_round)), dtype=float)
                changed_score = g.metric_record(frame, hold, changed)["macro_rmse"]
                deltas[fold_pos, j] = changed_score - metrics["macro_rmse"]
                scratch[:, j] = xhold[:, j]
        folds_out.append({"id": fold["id"], "seed": seed, "fit_n": len(fit),
                          "hold_n": len(hold), "best_round": best_round,
                          "trained_rounds": len(curve["validation_1"]["rmse"]),
                          "stop_reason": "validation_patience_30" if
                          len(curve["validation_1"]["rmse"]) < 300 else
                          "round_cap_300", "fit_seconds": seconds,
                          "metrics": metrics,
                          "training_rmse_by_round": curve["validation_0"]["rmse"],
                          "holdout_rmse_by_round": curve["validation_1"]["rmse"],
                          "holdout_prediction_db": pred.tolist(),
                          "permutation_macro_delta_db": deltas[fold_pos].tolist()})
    scores = np.asarray([fold["metrics"]["macro_rmse"] for fold in folds_out])
    mean_delta = deltas.mean(axis=0)
    removed = names[int(np.argmin(mean_delta))] if k > 1 else None
    return {"fingerprint": fingerprint, "branch": branch, "k": k,
            "feature_names": names, "removed_for_next": removed,
            "permutation_mean_delta_db": mean_delta.tolist(),
            "macro_mean_db": float(scores.mean()),
            "macro_se_db": float(scores.std(ddof=1) / np.sqrt(3)),
            "config": g.CONFIG, "folds": folds_out,
            "completed_utc": g.utc_now()}


def choose_path(rows: list[dict]) -> dict:
    best = min(rows, key=lambda row: (row["macro_mean_db"], row["k"]))
    limit = best["macro_mean_db"] + best["macro_se_db"]
    selected = min((row for row in rows if row["macro_mean_db"] <= limit + 1e-12),
                   key=lambda row: (row["k"], row["macro_mean_db"]))
    return {"minimum_k": best["k"], "minimum_macro_db": best["macro_mean_db"],
            "one_se_db": best["macro_se_db"],
            "eligibility_threshold_db": limit,
            "selected_k": selected["k"],
            "selected_feature_names": selected["feature_names"],
            "selected_macro_db": selected["macro_mean_db"]}


def select() -> None:
    plan, manifest, group_plan = checked()
    g.require(PLAN_QA.exists() and g.read_json(PLAN_QA)["status"] == "PASS",
              "Column plan QA must PASS")
    frame, _ = g.load_training()
    with writer_lock():
        for branch in plan["computed_branches"]:
            pool = plan["pools"][branch]
            names = pool.copy()
            rows = []
            while names:
                k = len(names)
                path = state_path(branch, k)
                if path.exists():
                    row = g.read_json(path)
                    g.require(row["fingerprint"] == manifest["fingerprint"] and
                              row["branch"] == branch and row["k"] == k and
                              row["feature_names"] == names,
                              f"Column checkpoint mismatch {path}")
                else:
                    row = fit_state(branch, names, pool, frame, group_plan,
                                    manifest["fingerprint"])
                    g.atomic_json(path, row)
                rows.append(row)
                print(f"COLUMN branch={branch} k={k} "
                      f"macro={row['macro_mean_db']:.6f} "
                      f"remove={row['removed_for_next']}", flush=True)
                if k == 1:
                    break
                g.require(row["removed_for_next"] in names,
                          f"Invalid removal at {branch} k={k}")
                names = [name for name in names if name != row["removed_for_next"]]
            decision = choose_path(rows)
            decision.update({"fingerprint": manifest["fingerprint"],
                             "branch": branch, "alias_of": None,
                             "path_k": [row["k"] for row in rows],
                             "completed_utc": g.utc_now()})
            path = decision_path(branch)
            if path.exists():
                old = g.read_json(path)
                g.require(old["fingerprint"] == decision["fingerprint"] and
                          old["selected_k"] == decision["selected_k"] and
                          old["selected_feature_names"] ==
                          decision["selected_feature_names"],
                          f"Decision checkpoint mismatch {branch}")
            else:
                g.atomic_json(path, decision)
        alias = plan["group_selected_alias_of"]
        if alias:
            canonical = g.read_json(decision_path(alias))
            equivalent = {**canonical, "branch": "group_selected",
                          "alias_of": alias, "identical_pool_and_path": True}
            path = decision_path("group_selected")
            if path.exists():
                old = g.read_json(path)
                g.require(old["fingerprint"] == equivalent["fingerprint"] and
                          old["selected_feature_names"] ==
                          equivalent["selected_feature_names"],
                          "Alias decision checkpoint mismatch")
            else:
                g.atomic_json(path, equivalent)
        print("COLUMN_PATHS_FROZEN", flush=True)


def verify_selection() -> dict:
    plan, manifest, group_plan = checked()
    g.require(PLAN_QA.exists() and g.read_json(PLAN_QA)["status"] == "PASS",
              "Column plan QA missing")
    frame, _ = g.load_training()
    total_states = 0
    for branch in plan["computed_branches"]:
        pool = plan["pools"][branch]
        names = pool.copy()
        rows = []
        files = list((OUT / "column_states" / branch).glob("k_*.json"))
        g.require(len(files) == len(pool), f"State count mismatch {branch}")
        while names:
            k = len(names)
            row = g.read_json(state_path(branch, k))
            g.require(row["fingerprint"] == manifest["fingerprint"] and
                      row["branch"] == branch and row["k"] == k and
                      row["feature_names"] == names and len(row["folds"]) == 3 and
                      len(row["permutation_mean_delta_db"]) == k,
                      f"State identity mismatch {branch} k={k}")
            fold_scores = []
            fold_deltas = []
            for fold_row, fold in zip(row["folds"], group_plan["folds"]):
                hold = np.asarray(fold["hold_index"], dtype=int)
                pred = np.asarray(fold_row["holdout_prediction_db"], dtype=float)
                replay = g.metric_record(frame, hold, pred)
                g.require(fold_row["id"] == fold["id"] and
                          fold_row["seed"] == g.FOLD_SEEDS[fold["id"] - 1] and
                          fold_row["fit_n"] == len(fold["fit_index"]) and
                          fold_row["hold_n"] == len(hold) and
                          1 <= fold_row["best_round"] <=
                          fold_row["trained_rounds"] <= 300 and
                          len(fold_row["training_rmse_by_round"]) ==
                          len(fold_row["holdout_rmse_by_round"]) ==
                          fold_row["trained_rounds"] and
                          len(fold_row["permutation_macro_delta_db"]) == k and
                          np.isfinite(fold_row["permutation_macro_delta_db"]).all() and
                          all(abs(replay[event][metric] -
                                  fold_row["metrics"][event][metric]) < 1e-10
                              for event in ("overall", "rain", "dry_overcast")
                              for metric in ("rmse", "mae")) and
                          abs(replay["macro_rmse"] -
                              fold_row["metrics"]["macro_rmse"]) < 1e-10,
                          f"Fold metric/curve mismatch {branch} k={k}")
                fold_scores.append(replay["macro_rmse"])
                fold_deltas.append(fold_row["permutation_macro_delta_db"])
            scores = np.asarray(fold_scores)
            deltas = np.asarray(fold_deltas)
            g.require(abs(float(scores.mean()) - row["macro_mean_db"]) < 1e-10 and
                      abs(float(scores.std(ddof=1) / np.sqrt(3)) -
                          row["macro_se_db"]) < 1e-10 and
                      np.allclose(deltas.mean(axis=0),
                                  row["permutation_mean_delta_db"], atol=1e-12),
                      f"State aggregate mismatch {branch} k={k}")
            expected_remove = names[int(np.argmin(deltas.mean(axis=0)))] if k > 1 else None
            g.require(row["removed_for_next"] == expected_remove,
                      f"Permutation path mismatch {branch} k={k}")
            rows.append(row)
            total_states += 1
            if k == 1:
                break
            names = [name for name in names if name != expected_remove]
        decision = g.read_json(decision_path(branch))
        expected = choose_path(rows)
        g.require(decision["fingerprint"] == manifest["fingerprint"] and
                  decision["branch"] == branch and decision["alias_of"] is None and
                  decision["path_k"] == list(range(len(pool), 0, -1)) and
                  all(decision[key] == value for key, value in expected.items()),
                  f"One-SE choice mismatch {branch}")
    alias = plan["group_selected_alias_of"]
    if alias:
        canonical = g.read_json(decision_path(alias))
        equivalent = g.read_json(decision_path("group_selected"))
        g.require(equivalent["alias_of"] == alias and
                  equivalent["fingerprint"] == canonical["fingerprint"] and
                  equivalent["selected_feature_names"] ==
                  canonical["selected_feature_names"] and
                  equivalent["selected_k"] == canonical["selected_k"],
                  "Identical-pool branch alias mismatch")
    qa = {"status": "PASS", "stage": "column_selection",
          "fingerprint": manifest["fingerprint"],
          "computed_branches": plan["computed_branches"],
          "group_selected_alias_of": alias,
          "states": total_states, "fold_fits": total_states * 3,
          "branch_selected_k": {branch: g.read_json(decision_path(branch))["selected_k"]
                                for branch in plan["pools"]},
          "formal_validation_or_test_read": False,
          "verified_utc": g.utc_now()}
    g.atomic_json(SELECT_QA, qa)
    return qa


def status() -> dict:
    counts = {}
    for branch in ("base", "group_selected"):
        counts[branch] = len(list((OUT / "column_states" / branch).glob("k_*.json")))
    return {"manifest": MANIFEST.exists(),
            "plan_qa": g.read_json(PLAN_QA).get("status") if PLAN_QA.exists() else None,
            "state_counts": counts,
            "decisions": {branch: decision_path(branch).exists()
                          for branch in counts},
            "selection_qa": g.read_json(SELECT_QA).get("status") if
            SELECT_QA.exists() else None,
            "worker_lock": (OUT / "column_worker.lock").exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify-prepare", "select",
                                            "verify-selection", "status"))
    args = parser.parse_args()
    result = {"prepare": prepare, "verify-prepare": verify_prepare,
              "select": select, "verify-selection": verify_selection,
              "status": status}[args.command]()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
