"""Greedy 200-to-1 train-selected masking of frozen Revision 019 XGBoost models.

The trained models are loaded, never fitted here. All writes stay in Revision 020.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "Revision_019"
sys.path.insert(0, str(PRIOR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_experiment as base  # noqa: E402

OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_020_Sequential_Zero_Mask"
MANIFEST = OUT / "run_manifest.json"
PLAN_QA = OUT / "plan_qa.json"
SELECTION_QA = OUT / "selection_qa.json"
EVALUATION_QA = OUT / "evaluation_qa.json"
SUMMARY = OUT / "results_path.csv"
FEATURE_PATH = OUT / "feature_path.csv"
SEEDS = (17, 29, 43)
N_FEATURES = 200
VERSION = "frozen-xgboost-greedy-train-mask-v20-20261002"

# The shared lock helper otherwise points at the read-only Revision 019 output.
base.c.OUT = OUT


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    base.save_json(path, payload)


def sha(path: Path) -> str:
    return base.sha(path)


def digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def prior_identity() -> tuple[dict, dict, dict]:
    plan = base.checked_plan()
    prior_manifest = base.checked_manifest()
    qa = read_json(base.MASK_QA)
    ranking = read_json(base.RANKING)
    require(qa["status"] == "PASS" and qa["fingerprint"] == prior_manifest["fingerprint"],
            "Revision 019 final QA is not valid")
    require(ranking["primary_winner"] == "xgboost" and
            ranking["fingerprint"] == prior_manifest["fingerprint"],
            "The frozen winning model changed")
    require(len(plan["retained_feature_names"]) == N_FEATURES and
            len(set(plan["retained_feature_names"])) == N_FEATURES,
            "Expected 200 distinct model inputs")
    return plan, prior_manifest, ranking


def manifest_candidate() -> dict:
    plan, prior_manifest, ranking = prior_identity()
    files = [HERE / "run_elimination.py", HERE / "PROTOCOL.md",
             base.PLAN, base.MANIFEST, base.MASK_QA, base.RANKING,
             base.DATA / "train.parquet"]
    for seed in SEEDS:
        fitted = base.model_path("xgboost", seed)
        record = read_json(base.test_record_path("xgboost", seed))
        require(sha(fitted) == record["fitted_model_sha256"] and
                record["fingerprint"] == prior_manifest["fingerprint"],
                f"Frozen XGBoost model changed: seed {seed}")
        files += [fitted, base.test_record_path("xgboost", seed)]
    payload = {
        "version": VERSION,
        "source_hashes": source_hashes(files),
        "prior_fingerprint": prior_manifest["fingerprint"],
        "feature_plan_fingerprint": plan["fingerprint"],
        "frozen_model_family": ranking["primary_winner"],
        "seeds": list(SEEDS),
        "ordered_feature_names": plan["retained_feature_names"],
        "n_start": N_FEATURES, "n_end": 1,
        "selection_data": "2404 training hours only; no validation/test/events",
        "selection_rule": "At each k, zero each remaining coordinate in turn; choose lowest mean training RMSE across the three frozen models; exact ties break by original feature order",
        "model_rule": "fixed Revision 019 fitted XGBoost; no refitting or retuning",
        "zero_reference": "final processed input zero; raw input centered by training median",
        "test_role": "describe the pre-frozen path, never choose a count or change the path",
        "test_historically_accessed": True,
    }
    payload["fingerprint"] = digest(payload)
    return payload


def checked_manifest() -> dict:
    candidate = manifest_candidate()
    require(MANIFEST.exists(), "Run prepare first")
    frozen = read_json(MANIFEST)
    require(frozen["fingerprint"] == candidate["fingerprint"],
            "Revision 020 manifest or an input source changed")
    return frozen


def prepare() -> dict:
    payload = manifest_candidate()
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        require(read_json(MANIFEST)["fingerprint"] == payload["fingerprint"],
                "Existing manifest differs")
        return read_json(MANIFEST)
    payload["created_utc"] = base.c.prior.utc_now()
    save_json(MANIFEST, payload)
    return payload


def verify_plan() -> dict:
    manifest = checked_manifest()
    plan, _, _ = prior_identity()
    require(plan["retained_feature_names"] == manifest["ordered_feature_names"],
            "Retained feature order changed")
    require(len(manifest["source_hashes"]) == 13,
            "Expected code, training, and three model+record digests")
    qa = {"status": "PASS", "stage": "source-and-model-plan",
          "fingerprint": manifest["fingerprint"], "n_features": N_FEATURES,
          "model": "xgboost", "seeds": list(SEEDS),
          "training_only_path_selection": True,
          "verified_utc": base.c.prior.utc_now()}
    save_json(PLAN_QA, qa)
    return qa


def model_bundle(manifest: dict) -> dict[int, dict]:
    loaded = {}
    for seed in SEEDS:
        file = base.model_path("xgboost", seed)
        with file.open("rb") as handle:
            item = pickle.load(handle)
        require(item["fingerprint"] == manifest["prior_fingerprint"] and
                item["name"] == "xgboost" and item["seed"] == seed,
                f"Fitted model identity mismatch: seed {seed}")
        require(item["scaler"] is None, "Unexpected XGBoost scaler")
        loaded[seed] = item
    return loaded


def transform(frame: pd.DataFrame, plan: dict, bundle: dict[int, dict]) -> dict[int, np.ndarray]:
    pos = np.arange(len(frame))
    raw = base.raw_input(frame, pos, plan, "xgboost")
    x = {seed: item["imputer"].transform(raw) for seed, item in bundle.items()}
    require(all(value.shape == (len(frame), N_FEATURES) and np.isfinite(value).all()
                for value in x.values()), "Transformed input changed")
    return x


def train_inputs(manifest: dict) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, dict]]:
    plan = base.checked_plan()
    names = plan["all_feature_names"]
    frame = pd.read_parquet(base.DATA / "train.parquet",
                            columns=[*names, base.c.TARGET]).reset_index(drop=True)
    require(len(frame) == 2404, "Training split size changed")
    y = frame[base.c.TARGET].to_numpy(dtype=float)
    require(np.isfinite(y).all(), "Training labels are invalid")
    bundle = model_bundle(manifest)
    x = transform(frame, plan, bundle)
    for seed in SEEDS:
        pred = np.asarray(bundle[seed]["model"].predict(x[seed]), dtype=float)
        reference = read_json(base.test_record_path("xgboost", seed))
        train_rmse = base.c.rmse(y, pred)
        require(abs(train_rmse - reference["round_history"][-1]["train_rmse_db"]) < 1e-7,
                f"Frozen train prediction mismatch: seed {seed}")
    return y, x, bundle


def state_path(k: int) -> Path:
    return OUT / "states" / f"k_{k:03d}.json"


def candidate_path(from_k: int) -> Path:
    return OUT / "candidates" / f"from_{from_k:03d}.json"


def evaluation_path(k: int, seed: int) -> Path:
    return OUT / "evaluation" / f"k_{k:03d}_seed{seed}.json"


def state_at(k: int, fp: str) -> dict | None:
    return base.c.read_record(state_path(k), fp)


def candidate_at(from_k: int, fp: str) -> dict | None:
    return base.c.read_record(candidate_path(from_k), fp)


def masked_inputs(x: dict[int, np.ndarray], eliminated: list[str],
                  indices: dict[str, int]) -> dict[int, np.ndarray]:
    cols = [indices[name] for name in eliminated]
    result = {seed: value.copy() for seed, value in x.items()}
    for value in result.values():
        value[:, cols] = 0.0
    return result


def score_candidate(feature: str, current: dict[int, np.ndarray],
                    indices: dict[str, int], bundle: dict[int, dict],
                    y: np.ndarray) -> dict:
    per_seed = {}
    index = indices[feature]
    for seed in SEEDS:
        trial = current[seed].copy()
        trial[:, index] = 0.0
        pred = np.asarray(bundle[seed]["model"].predict(trial), dtype=float).reshape(-1)
        per_seed[str(seed)] = base.c.rmse(y, pred)
    return {"feature": feature, "original_index": index,
            "train_rmse_by_seed": per_seed,
            "train_rmse_mean": float(np.mean(list(per_seed.values())))}


def select() -> None:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(PLAN_QA.exists() and read_json(PLAN_QA)["status"] == "PASS",
            "Plan QA must pass before selection")
    names = manifest["ordered_feature_names"]
    indices = {name: i for i, name in enumerate(names)}
    y, x, bundle = train_inputs(manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    with base.c.writer_lock():
        full = state_at(N_FEATURES, fp)
        if full is None:
            baseline = {str(seed): base.c.rmse(y, bundle[seed]["model"].predict(x[seed]))
                        for seed in SEEDS}
            full = {"fingerprint": fp, "k": N_FEATURES,
                    "retained_feature_names": names, "eliminated_feature_names": [],
                    "retained_original_indices": list(range(N_FEATURES)),
                    "removed_feature": None, "from_k": None,
                    "train_rmse_by_seed": baseline,
                    "train_rmse_mean": float(np.mean(list(baseline.values()))),
                    "selection_candidate_sha256": None,
                    "completed_utc": base.c.prior.utc_now()}
            save_json(state_path(N_FEATURES), full)
            print(f"STATE k={N_FEATURES} full-input training RMSE={full['train_rmse_mean']:.6f}",
                  flush=True)
        require(full["retained_feature_names"] == names and
                full["eliminated_feature_names"] == [], "Full state changed")
        for from_k in range(N_FEATURES, 1, -1):
            current = state_at(from_k, fp)
            require(current is not None and current["k"] == from_k and
                    len(current["retained_feature_names"]) == from_k,
                    f"Missing or invalid state at k={from_k}")
            already = state_at(from_k - 1, fp)
            if already is not None:
                require(already["k"] == from_k - 1 and
                        already["from_k"] == from_k and
                        already["removed_feature"] in current["retained_feature_names"],
                        f"Invalid resumed state at k={from_k - 1}")
                continue
            masked = masked_inputs(x, current["eliminated_feature_names"], indices)
            candidates = candidate_at(from_k, fp)
            if candidates is None:
                rows = [score_candidate(feature, masked, indices, bundle, y)
                        for feature in current["retained_feature_names"]]
                selected = min(rows, key=lambda row: (row["train_rmse_mean"],
                                                      row["original_index"]))
                candidates = {"fingerprint": fp, "from_k": from_k,
                              "to_k": from_k - 1,
                              "current_retained_feature_names": current["retained_feature_names"],
                              "candidate_scores": rows,
                              "selected_feature": selected["feature"],
                              "selection_objective": "minimum mean training RMSE over three frozen models",
                              "completed_utc": base.c.prior.utc_now()}
                save_json(candidate_path(from_k), candidates)
            else:
                require(candidates["current_retained_feature_names"] ==
                        current["retained_feature_names"] and
                        len(candidates["candidate_scores"]) == from_k,
                        f"Invalid candidate checkpoint at k={from_k}")
                selected = min(candidates["candidate_scores"],
                               key=lambda row: (row["train_rmse_mean"], row["original_index"]))
                require(selected["feature"] == candidates["selected_feature"],
                        "Saved candidate winner changed")
            removed = selected["feature"]
            retained = [name for name in current["retained_feature_names"] if name != removed]
            eliminated = [*current["eliminated_feature_names"], removed]
            require(len(retained) == from_k - 1 and len(eliminated) == N_FEATURES - from_k + 1,
                    "Removal did not eliminate exactly one feature")
            next_state = {"fingerprint": fp, "k": from_k - 1,
                          "retained_feature_names": retained,
                          "eliminated_feature_names": eliminated,
                          "retained_original_indices": [indices[name] for name in retained],
                          "removed_feature": removed, "from_k": from_k,
                          "train_rmse_by_seed": selected["train_rmse_by_seed"],
                          "train_rmse_mean": selected["train_rmse_mean"],
                          "selection_candidate_sha256": sha(candidate_path(from_k)),
                          "completed_utc": base.c.prior.utc_now()}
            save_json(state_path(from_k - 1), next_state)
            print(f"STATE k={from_k - 1} removed={removed} "
                  f"train_RMSE={selected['train_rmse_mean']:.6f}", flush=True)
        print("SELECTION_PATH_FROZEN 200 STATES; 199 REMOVALS", flush=True)


def verify_selection() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    names = manifest["ordered_feature_names"]
    indices = {name: i for i, name in enumerate(names)}
    require(len(list((OUT / "states").glob("k_*.json"))) == N_FEATURES,
            "Expected exactly 200 state files")
    require(len(list((OUT / "candidates").glob("from_*.json"))) == N_FEATURES - 1,
            "Expected exactly 199 candidate files")
    y, x, bundle = train_inputs(manifest)
    assessed = 0
    previous = None
    for k in range(N_FEATURES, 0, -1):
        state = state_at(k, fp)
        require(state is not None and state["k"] == k and
                len(state["retained_feature_names"]) == k and
                len(state["eliminated_feature_names"]) == N_FEATURES - k and
                set(state["retained_feature_names"]) |
                set(state["eliminated_feature_names"]) == set(names) and
                not (set(state["retained_feature_names"]) &
                     set(state["eliminated_feature_names"])),
                f"Invalid feature partition at k={k}")
        require(state["retained_original_indices"] ==
                [indices[name] for name in state["retained_feature_names"]],
                f"Original index list changed at k={k}")
        if previous is not None:
            require(state["from_k"] == k + 1 and
                    state["retained_feature_names"] ==
                    [name for name in previous["retained_feature_names"]
                     if name != state["removed_feature"]] and
                    state["eliminated_feature_names"] ==
                    [*previous["eliminated_feature_names"], state["removed_feature"]],
                    f"State chain is broken at k={k}")
            candidate = candidate_at(k + 1, fp)
            require(candidate is not None and candidate["from_k"] == k + 1 and
                    candidate["to_k"] == k and
                    candidate["current_retained_feature_names"] ==
                    previous["retained_feature_names"] and
                    len(candidate["candidate_scores"]) == k + 1 and
                    [row["feature"] for row in candidate["candidate_scores"]] ==
                    previous["retained_feature_names"],
                    f"Candidate inventory mismatch at k={k}")
            assessed += len(candidate["candidate_scores"])
            best = min(candidate["candidate_scores"],
                       key=lambda row: (row["train_rmse_mean"], row["original_index"]))
            require(best["feature"] == state["removed_feature"] ==
                    candidate["selected_feature"] and
                    best["train_rmse_by_seed"] == state["train_rmse_by_seed"] and
                    abs(best["train_rmse_mean"] - state["train_rmse_mean"]) < 1e-12 and
                    sha(candidate_path(k + 1)) == state["selection_candidate_sha256"],
                    f"Greedy decision or candidate digest mismatch at k={k}")
        xm = masked_inputs(x, state["eliminated_feature_names"], indices)
        for seed in SEEDS:
            pred = bundle[seed]["model"].predict(xm[seed])
            require(abs(base.c.rmse(y, pred) -
                        state["train_rmse_by_seed"][str(seed)]) < 1e-9,
                    f"Training score replay mismatch at k={k}/seed={seed}")
        require(abs(float(np.mean(list(state["train_rmse_by_seed"].values()))) -
                    state["train_rmse_mean"]) < 1e-12,
                f"Training score mean mismatch at k={k}")
        previous = state
    require(assessed == 20099 and len(previous["retained_feature_names"]) == 1,
            "Selection path did not reach one feature")
    qa = {"status": "PASS", "stage": "training-only-selection",
          "fingerprint": fp, "states": N_FEATURES,
          "removals": N_FEATURES - 1, "candidate_assessments": assessed,
          "model_fits_in_this_revision": 0,
          "last_remaining_feature": previous["retained_feature_names"][0],
          "test_not_used_to_select_path": True,
          "verified_utc": base.c.prior.utc_now()}
    save_json(SELECTION_QA, qa)
    return qa


def evaluate() -> None:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(SELECTION_QA.exists() and read_json(SELECTION_QA)["status"] == "PASS",
            "Selection QA must pass before test evaluation")
    plan = base.checked_plan()
    names = manifest["ordered_feature_names"]
    indices = {name: i for i, name in enumerate(names)}
    test = base.test_frame(plan["all_feature_names"])
    require(len(test) == 687, "Test set changed")
    bundle = model_bundle(manifest)
    x = transform(test, plan, bundle)
    pos = np.arange(len(test))
    baseline = {}
    for seed in SEEDS:
        pred = np.asarray(bundle[seed]["model"].predict(x[seed]), dtype=float)
        prior_record = read_json(base.test_record_path("xgboost", seed))
        require(np.max(np.abs(pred - np.asarray(prior_record["test_prediction_db"]))) < 1e-8,
                f"Full-input test prediction differs from Revision 019: seed {seed}")
        baseline[seed] = pred
    OUT.mkdir(parents=True, exist_ok=True)
    with base.c.writer_lock():
        for k in range(N_FEATURES, 0, -1):
            state = state_at(k, fp)
            require(state is not None, f"Selection state missing: k={k}")
            xm = masked_inputs(x, state["eliminated_feature_names"], indices)
            for seed in SEEDS:
                path = evaluation_path(k, seed)
                old = base.c.read_record(path, fp)
                if old is not None:
                    require(old["state_sha256"] == sha(state_path(k)) and
                            old["fitted_model_sha256"] ==
                            read_json(base.test_record_path("xgboost", seed))["fitted_model_sha256"],
                            f"Resume identity mismatch at k={k}/seed={seed}")
                    continue
                pred = np.asarray(bundle[seed]["model"].predict(xm[seed]), dtype=float)
                metrics = base.score(test, pos, pred)
                record = {"fingerprint": fp, "k": k, "seed": seed,
                          "state_sha256": sha(state_path(k)),
                          "fitted_model_sha256": sha(base.model_path("xgboost", seed)),
                          "removed_feature_to_reach_k": state["removed_feature"],
                          "n_retained_features": k,
                          "n_eliminated_features": N_FEATURES - k,
                          "test_prediction_db": pred.tolist(), "test_metrics": metrics,
                          "mean_absolute_prediction_shift_from_full_db": float(
                              np.mean(np.abs(pred - baseline[seed]))),
                          "completed_utc": base.c.prior.utc_now()}
                save_json(path, record)
            if k % 10 == 0 or k <= 10:
                print(f"EVALUATED k={k}", flush=True)
        feature_rows = []
        metric_rows = []
        for k in range(N_FEATURES, 0, -1):
            state = state_at(k, fp)
            records = [base.c.read_record(evaluation_path(k, seed), fp) for seed in SEEDS]
            require(all(record is not None for record in records),
                    f"Evaluation incomplete at k={k}")
            feature_rows.append({
                "k": k, "removed_feature_to_reach_k": state["removed_feature"] or "",
                "retained_feature_names_json": json.dumps(
                    state["retained_feature_names"], ensure_ascii=False),
                "eliminated_feature_names_json": json.dumps(
                    state["eliminated_feature_names"], ensure_ascii=False),
                "train_rmse_mean_db": state["train_rmse_mean"]})
            metric_rows.append({
                "k": k, "removed_feature_to_reach_k": state["removed_feature"] or "",
                "train_rmse_mean_db": state["train_rmse_mean"],
                "test_overall_rmse_mean_db": float(np.mean([
                    record["test_metrics"]["overall"]["rmse"] for record in records])),
                "test_overall_rmse_sd_seed_db": float(np.std([
                    record["test_metrics"]["overall"]["rmse"] for record in records], ddof=1)),
                "test_overall_mae_mean_db": float(np.mean([
                    record["test_metrics"]["overall"]["mae"] for record in records])),
                "test_rain_rmse_mean_db": float(np.mean([
                    record["test_metrics"]["rain"]["rmse"] for record in records])),
                "test_dry_overcast_rmse_mean_db": float(np.mean([
                    record["test_metrics"]["dry_overcast"]["rmse"] for record in records])),
                "mean_absolute_prediction_shift_from_full_db": float(np.mean([
                    record["mean_absolute_prediction_shift_from_full_db"]
                    for record in records]))})
        pd.DataFrame(feature_rows).to_csv(FEATURE_PATH, index=False)
        pd.DataFrame(metric_rows).to_csv(SUMMARY, index=False)
        print("EVALUATION_WRITTEN 600 RECORDS; NO RETRAINING", flush=True)


def verify_evaluation() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(SELECTION_QA.exists() and read_json(SELECTION_QA)["status"] == "PASS",
            "Selection QA missing")
    require(len(list((OUT / "evaluation").glob("k_*.json"))) == 600 and
            SUMMARY.exists() and FEATURE_PATH.exists(),
            "Expected 600 evaluations and two CSV summaries")
    plan = base.checked_plan()
    test = base.test_frame(plan["all_feature_names"])
    truth = test[base.c.TARGET].to_numpy(dtype=float)
    pos = np.arange(len(test))
    metric_csv = pd.read_csv(SUMMARY)
    feature_csv = pd.read_csv(FEATURE_PATH)
    require(len(metric_csv) == len(feature_csv) == N_FEATURES and
            metric_csv.k.tolist() == feature_csv.k.tolist() ==
            list(range(N_FEATURES, 0, -1)), "Summary state order is incomplete")
    for k in range(N_FEATURES, 0, -1):
        state = state_at(k, fp)
        require(json.loads(feature_csv.loc[N_FEATURES - k, "retained_feature_names_json"]) ==
                state["retained_feature_names"], f"Feature CSV mismatch at k={k}")
        per_seed = []
        for seed in SEEDS:
            row = base.c.read_record(evaluation_path(k, seed), fp)
            require(row is not None and row["k"] == k and row["seed"] == seed and
                    row["state_sha256"] == sha(state_path(k)) and
                    row["fitted_model_sha256"] == sha(base.model_path("xgboost", seed)),
                    f"Invalid evaluation identity at k={k}/seed={seed}")
            pred = np.asarray(row["test_prediction_db"], dtype=float)
            require(len(pred) == 687 and np.isfinite(pred).all() and
                    abs(base.c.rmse(truth, pred) -
                        row["test_metrics"]["overall"]["rmse"]) < 1e-10,
                    f"Test metric mismatch at k={k}/seed={seed}")
            replay = base.score(test, pos, pred)
            for group in ("overall", "rain", "dry_overcast"):
                require(replay[group]["n"] == row["test_metrics"][group]["n"] and
                        abs(replay[group]["rmse"] -
                            row["test_metrics"][group]["rmse"]) < 1e-10,
                        f"Weather metric mismatch at k={k}/seed={seed}/{group}")
            if k == N_FEATURES:
                prior_record = read_json(base.test_record_path("xgboost", seed))
                require(np.max(np.abs(pred - np.asarray(
                    prior_record["test_prediction_db"]))) < 1e-8,
                    f"Full baseline mismatch at seed={seed}")
            per_seed.append(row)
        row_csv = metric_csv.loc[N_FEATURES - k]
        for group, column in (("overall", "test_overall_rmse_mean_db"),
                              ("rain", "test_rain_rmse_mean_db"),
                              ("dry_overcast", "test_dry_overcast_rmse_mean_db")):
            actual = float(np.mean([record["test_metrics"][group]["rmse"]
                                    for record in per_seed]))
            require(abs(actual - float(row_csv[column])) < 1e-10,
                    f"Summary metric mismatch at k={k}/{group}")
    qa = {"status": "PASS", "stage": "all",
          "fingerprint": fp, "states": N_FEATURES,
          "test_prediction_records": 600, "test_hours": 687,
          "no_refitting_or_retuning": True,
          "full_baseline_matches_revision_019": True,
          "test_historically_accessed": True,
          "test_not_used_to_select_path": True,
          "verified_utc": base.c.prior.utc_now()}
    save_json(EVALUATION_QA, qa)
    return qa


def report() -> Path:
    require(EVALUATION_QA.exists() and read_json(EVALUATION_QA)["status"] == "PASS",
            "Final evaluation QA must pass before reporting")
    manifest = checked_manifest()
    table = pd.read_csv(SUMMARY).set_index("k")
    last = state_at(1, manifest["fingerprint"])["retained_feature_names"][0]
    selected_k = [200, 100, 50, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    lines = ["# Revision 020: fixed-XGBoost sequential input masking", "",
             "This is a 200-to-1 greedy path selected by overall TRAINING RMSE only. "
             "The XGBoost models were loaded from Revision 019 and never refitted.", "",
             "Each eliminated meteorological input was set to its training-reference zero at "
             "the final model input. This is not a physically absent variable, classical "
             "retrained RFE, or a causal feature contribution.", "",
             f"Final remaining feature at k=1: `{last}`.", "",
             "| Retained k | Train RMSE (dB) | Test overall RMSE (dB) | "
             "Test rain RMSE (dB) | Test dry-overcast RMSE (dB) |",
             "|---:|---:|---:|---:|---:|"]
    for k in selected_k:
        row = table.loc[k]
        lines.append(f"| {k} | {row.train_rmse_mean_db:.4f} | "
                     f"{row.test_overall_rmse_mean_db:.4f} | "
                     f"{row.test_rain_rmse_mean_db:.4f} | "
                     f"{row.test_dry_overcast_rmse_mean_db:.4f} |")
    lines += ["", "Complete feature names at every k are in `feature_path.csv` and "
              "`states/k_*.json`. All 200 test metric rows are in `results_path.csv`; "
              "three per-seed, per-hour prediction records at each k are in `evaluation/`.", "",
              "The test set was historically accessed and chosen XGBoost had already been "
              "ranked on this test set. The test curve describes the fixed, training-selected "
              "path; it must not be mined for a winning k or presented as independent blind "
              "generalization. Rain includes heavy rain. A deployable sparse model would "
              "require a separately specified retraining and independent validation study.", ""]
    path = OUT / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def status() -> dict:
    phase = ("FINAL_QA_PASS" if EVALUATION_QA.exists() and
             read_json(EVALUATION_QA).get("status") == "PASS" else
             "SELECTION_QA_PASS" if SELECTION_QA.exists() and
             read_json(SELECTION_QA).get("status") == "PASS" else
             "SELECTING" if (OUT / "states").exists() else
             "PREPARED" if MANIFEST.exists() else "NOT_PREPARED")
    return {"phase": phase, "manifest": MANIFEST.exists(),
            "fingerprint": None if not MANIFEST.exists() else
            read_json(MANIFEST)["fingerprint"],
            "states": len(list((OUT / "states").glob("k_*.json"))),
            "candidate_files": len(list((OUT / "candidates").glob("from_*.json"))),
            "test_prediction_records": len(list((OUT / "evaluation").glob("k_*.json"))),
            "report": (OUT / "RESULTS.md").exists()}


def smoke() -> dict:
    manifest = checked_manifest()
    y, x, bundle = train_inputs(manifest)
    names = manifest["ordered_feature_names"]
    indices = {name: i for i, name in enumerate(names)}
    subset = {seed: value[:24] for seed, value in x.items()}
    first = names[0]
    masked = masked_inputs(subset, [first], indices)
    require(all(np.all(value[:, indices[first]] == 0.0) for value in masked.values()),
            "Mask did not set exact final-input zero")
    sample = {str(seed): float(base.c.rmse(y[:24],
              bundle[seed]["model"].predict(masked[seed]))) for seed in SEEDS}
    return {"status": "PASS", "sample_hours": 24,
            "masked_feature": first, "train_sample_rmse_by_seed": sample,
            "model_fits": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-plan", "select",
                                            "verify-selection", "evaluate",
                                            "verify-evaluation", "report", "status", "smoke"))
    args = parser.parse_args()
    actions = {"prepare": prepare, "verify-plan": verify_plan,
               "select": select, "verify-selection": verify_selection,
               "evaluate": evaluate, "verify-evaluation": verify_evaluation,
               "report": report, "status": status, "smoke": smoke}
    result = actions[args.command]()
    if result is not None:
        print(json.dumps(str(result) if isinstance(result, Path) else result,
                         ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
