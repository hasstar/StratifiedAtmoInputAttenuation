"""Checkpointed, genuinely refitted XGBoost RFE on the local hourly dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "Revision_019"
sys.path.insert(0, str(PRIOR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
import run_experiment as base  # noqa: E402

OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_021_RFE_Refit"
MANIFEST = OUT / "run_manifest.json"
PLAN_QA = OUT / "plan_qa.json"
RFE_QA = OUT / "rfe_qa.json"
TEST_QA = OUT / "test_qa.json"
SUMMARY = OUT / "results_path.csv"
FEATURE_PATH = OUT / "feature_path.csv"
SEEDS = (17, 29, 43)
N_FEATURES = 200
MAX_ROUNDS = 300
PATIENCE = 30
VERSION = "xgboost-three-seed-retrained-rfe-v21-20261002"

# The inherited lock helper must write only into this revision.
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


def fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prior_identity() -> tuple[dict, dict, dict, dict]:
    plan = base.checked_plan()
    prior_manifest = base.checked_manifest()
    freeze = read_json(base.FREEZE)
    ranking = read_json(base.RANKING)
    qa = read_json(base.MASK_QA)
    require(qa["status"] == "PASS" and qa["fingerprint"] == prior_manifest["fingerprint"],
            "Revision 019 final QA did not pass")
    require(ranking["primary_winner"] == "xgboost" and
            ranking["fingerprint"] == prior_manifest["fingerprint"],
            "The previously selected model family changed")
    require(freeze["fingerprint"] == prior_manifest["fingerprint"],
            "Revision 019 hyperparameter freeze changed")
    chosen = next(row for row in freeze["models"] if row["name"] == "xgboost")
    require(chosen["config"] == {"n_estimators": 300, "max_depth": 5,
                                  "learning_rate": 0.03} and
            len(plan["retained_feature_names"]) == N_FEATURES,
            "Frozen configuration or 200-column feature list changed")
    return plan, prior_manifest, ranking, chosen


def manifest_candidate() -> dict:
    plan, prior_manifest, ranking, chosen = prior_identity()
    sources = [HERE / "PROTOCOL.md", HERE / "run_rfe.py",
               base.DATA / "train.parquet", base.DATA / "validation.parquet",
               base.PLAN, base.MANIFEST, base.FREEZE, base.RANKING, base.MASK_QA,
               PRIOR / "run_experiment.py", PRIOR.parent / "Revision_014" / "common.py",
               PRIOR.parent / "Revision_014" / "models.py",
               PRIOR.parent / "Revision_014" / "support.py"]
    payload = {
        "version": VERSION,
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "prior_manifest_fingerprint": prior_manifest["fingerprint"],
        "prior_feature_plan_fingerprint": plan["fingerprint"],
        "model_family": ranking["primary_winner"],
        "start_feature_names": plan["retained_feature_names"],
        "n_start": N_FEATURES, "n_end": 1,
        "n_train": 2404, "n_validation": 343,
        "fixed_config": chosen["config"], "seeds": list(SEEDS),
        "max_rounds": MAX_ROUNDS, "early_stopping_patience": PATIENCE,
        "training_rule": "fresh XGBoost fit for every k and seed; validation only chooses effective best round",
        "removal_rule": "minimum three-seed mean normalized total_gain through selected round; original-order tie break",
        "test_rule": "evaluate only after 200-state RFE path and QA are frozen; never choose k from test",
        "test_historically_accessed": True,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def prepare() -> dict:
    payload = manifest_candidate()
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        require(read_json(MANIFEST)["fingerprint"] == payload["fingerprint"],
                "Existing RFE manifest differs from source/code")
        return read_json(MANIFEST)
    payload["created_utc"] = base.c.prior.utc_now()
    save_json(MANIFEST, payload)
    return payload


def checked_manifest() -> dict:
    require(MANIFEST.exists(), "Run prepare first")
    frozen = read_json(MANIFEST)
    replay = manifest_candidate()
    require(frozen["fingerprint"] == replay["fingerprint"],
            "RFE source or code fingerprint changed")
    return frozen


def verify_plan() -> dict:
    frozen = checked_manifest()
    names = frozen["start_feature_names"]
    require(len(names) == len(set(names)) == N_FEATURES and
            sum(name.startswith("surface__") for name in names) == 88 and
            sum(name.startswith("p") for name in names) == 112,
            "Starting features are not the frozen 88+112 list")
    require(frozen["fixed_config"]["n_estimators"] == MAX_ROUNDS and
            frozen["early_stopping_patience"] == PATIENCE,
            "Training budget changed")
    qa = {"status": "PASS", "stage": "plan", "fingerprint": frozen["fingerprint"],
          "n_features": N_FEATURES, "n_train": 2404, "n_validation": 343,
          "test_not_read_for_rfe": True, "verified_utc": base.c.prior.utc_now()}
    save_json(PLAN_QA, qa)
    return qa


def train_path(k: int, seed: int) -> Path:
    return OUT / "training" / f"k_{k:03d}_seed{seed}.json"


def model_path(k: int, seed: int) -> Path:
    return OUT / "models" / f"k_{k:03d}_seed{seed}.pkl"


def state_path(k: int) -> Path:
    return OUT / "states" / f"k_{k:03d}.json"


def test_path(k: int, seed: int) -> Path:
    return OUT / "test" / f"k_{k:03d}_seed{seed}.json"


def record_at(path: Path, fp: str) -> dict | None:
    return base.c.read_record(path, fp)


def save_pickle(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha(path)


def raw_frame(frame: pd.DataFrame, names: list[str], medians: dict[str, float]) -> np.ndarray:
    raw = frame.loc[:, names].to_numpy(dtype=float, copy=True)
    raw[~np.isfinite(raw)] = np.nan
    for i, name in enumerate(names):
        raw[:, i] -= medians[name]
    return raw


def training_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    plan = base.checked_plan()
    names = plan["retained_feature_names"]
    columns = [*names, base.c.TARGET]
    train = pd.read_parquet(base.DATA / "train.parquet", columns=columns).reset_index(drop=True)
    valid = pd.read_parquet(base.DATA / "validation.parquet", columns=columns).reset_index(drop=True)
    require(len(train) == 2404 and len(valid) == 343 and
            np.isfinite(train[base.c.TARGET]).all() and
            np.isfinite(valid[base.c.TARGET]).all(),
            "Train/validation splits or targets changed")
    return plan, train, valid


def importance_from_booster(model, names: list[str], best_round: int) -> tuple[list[float], list[float]]:
    booster = model.get_booster()[:best_round]
    gains = booster.get_score(importance_type="total_gain")
    raw = [float(gains.get(f"f{i}", 0.0)) for i in range(len(names))]
    require(np.isfinite(raw).all() and min(raw) >= 0,
            "Invalid XGBoost total_gain importance")
    total = sum(raw)
    normalized = [float(value / total) if total > 0 else 0.0 for value in raw]
    return raw, normalized


def fit_one(k: int, seed: int, names: list[str], manifest: dict,
            plan: dict, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    fp = manifest["fingerprint"]
    ytrain = train[base.c.TARGET].to_numpy(dtype=float)
    yvalid = valid[base.c.TARGET].to_numpy(dtype=float)
    xtrain_raw = raw_frame(train, names, plan["reference_median_by_feature"])
    xvalid_raw = raw_frame(valid, names, plan["reference_median_by_feature"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xtrain = imputer.fit_transform(xtrain_raw)
    xvalid = imputer.transform(xvalid_raw)
    require(xtrain.shape == (2404, k) and xvalid.shape == (343, k) and
            np.isfinite(xtrain).all() and np.isfinite(xvalid).all(),
            f"Preprocessing changed at k={k}/seed={seed}")

    model = base.c.make_estimator("xgboost", manifest["fixed_config"], seed,
                                  k, tuple(range(10)))
    model.set_params(eval_metric="rmse", early_stopping_rounds=PATIENCE)
    started = time.perf_counter()
    model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xvalid, yvalid)],
              verbose=False)
    fit_seconds = time.perf_counter() - started
    logged = model.evals_result()
    train_curve = logged["validation_0"]["rmse"]
    valid_curve = logged["validation_1"]["rmse"]
    trained_rounds = len(train_curve)
    best_round = int(model.best_iteration) + 1
    require(len(valid_curve) == trained_rounds and
            1 <= best_round <= trained_rounds <= MAX_ROUNDS and
            best_round == min(range(1, trained_rounds + 1),
                              key=lambda index: (float(valid_curve[index - 1]), index)),
            f"XGBoost best round did not match the validation curve at k={k}/seed={seed}")
    history = [{"round": i + 1, "train_rmse_db": float(train_curve[i]),
                "validation_rmse_db": float(valid_curve[i])}
               for i in range(trained_rounds)]
    train_pred = np.asarray(model.predict(xtrain, iteration_range=(0, best_round)), dtype=float)
    valid_pred = np.asarray(model.predict(xvalid, iteration_range=(0, best_round)), dtype=float)
    train_rmse = base.c.rmse(ytrain, train_pred)
    valid_rmse = base.c.rmse(yvalid, valid_pred)
    require(abs(valid_rmse - float(valid_curve[best_round - 1])) < 1e-5,
            f"Selected-round validation prediction mismatch at k={k}/seed={seed}")
    raw_gain, norm_gain = importance_from_booster(model, names, best_round)
    stored_model_sha = save_pickle(model_path(k, seed), {
        "fingerprint": fp, "k": k, "seed": seed, "feature_names": names,
        "best_round": best_round, "model": model, "imputer": imputer})
    row = {"fingerprint": fp, "k": k, "seed": seed,
           "feature_names": names, "config": manifest["fixed_config"],
           "maximum_rounds": MAX_ROUNDS, "early_stopping_patience": PATIENCE,
           "trained_rounds": trained_rounds, "effective_best_round": best_round,
           "stop_reason": ("validation_no_improvement_for_30_rounds"
                           if trained_rounds < MAX_ROUNDS else "300_round_cap"),
           "selection_reason": "minimum validation RMSE among trained rounds",
           "fit_seconds": fit_seconds, "n_train": 2404, "n_validation": 343,
           "round_history": history, "training_prediction_db": train_pred.tolist(),
           "validation_prediction_db": valid_pred.tolist(),
           "train_rmse_db": train_rmse, "validation_rmse_db": valid_rmse,
           "raw_total_gain_by_feature": raw_gain,
           "normalized_total_gain_by_feature": norm_gain,
           "fitted_model_sha256": stored_model_sha,
           "completed_utc": base.c.prior.utc_now()}
    save_json(train_path(k, seed), row)
    return row


def group_counts(names: list[str]) -> dict:
    levels = {str(level): sum(name.startswith(f"p{level}__") for name in names)
              for level in base.c.LEVELS}
    return {"base": sum(name.startswith("surface__") for name in names),
            "pressure_levels": levels}


def rfe() -> None:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(PLAN_QA.exists() and read_json(PLAN_QA)["status"] == "PASS",
            "Plan QA must pass before RFE")
    plan, train, valid = training_data()
    original = manifest["start_feature_names"]
    original_index = {name: i for i, name in enumerate(original)}
    names = list(original)
    OUT.mkdir(parents=True, exist_ok=True)
    with base.c.writer_lock():
        for k in range(N_FEATURES, 0, -1):
            require(len(names) == k, f"RFE path length changed before k={k}")
            records = []
            for seed in SEEDS:
                path = train_path(k, seed)
                old = record_at(path, fp)
                if old is None:
                    old = fit_one(k, seed, names, manifest, plan, train, valid)
                    print(f"FIT k={k} seed={seed} best={old['effective_best_round']} "
                          f"trained={old['trained_rounds']} valid_RMSE="
                          f"{old['validation_rmse_db']:.6f}", flush=True)
                else:
                    require(old["feature_names"] == names and
                            old["fitted_model_sha256"] == sha(model_path(k, seed)),
                            f"Resumed fit mismatch at k={k}/seed={seed}")
                records.append(old)
            gains = np.asarray([row["normalized_total_gain_by_feature"]
                                for row in records], dtype=float)
            require(gains.shape == (len(SEEDS), k) and np.isfinite(gains).all(),
                    f"Feature gain shape changed at k={k}")
            mean_gain = gains.mean(axis=0)
            removed = (None if k == 1 else names[min(range(k),
                       key=lambda i: (float(mean_gain[i]), original_index[names[i]]))])
            state = {"fingerprint": fp, "k": k, "retained_feature_names": names,
                     "removed_feature_for_next_k": removed,
                     "mean_normalized_total_gain_by_feature": mean_gain.tolist(),
                     "group_counts": group_counts(names),
                     "training_rmse_mean_db": float(np.mean([
                         row["train_rmse_db"] for row in records])),
                     "validation_rmse_mean_db": float(np.mean([
                         row["validation_rmse_db"] for row in records])),
                     "best_round_by_seed": {str(row["seed"]): row["effective_best_round"]
                                            for row in records},
                     "model_sha256_by_seed": {str(row["seed"]): row["fitted_model_sha256"]
                                              for row in records},
                     "completed_utc": base.c.prior.utc_now()}
            old_state = record_at(state_path(k), fp)
            if old_state is None:
                save_json(state_path(k), state)
            else:
                require(old_state["retained_feature_names"] == names and
                        old_state["removed_feature_for_next_k"] == removed and
                        old_state["model_sha256_by_seed"] == state["model_sha256_by_seed"],
                        f"Resumed RFE state mismatch at k={k}")
            print(f"STATE k={k} next_remove={removed} "
                  f"valid_RMSE={state['validation_rmse_mean_db']:.6f}", flush=True)
            if removed is not None:
                names = [name for name in names if name != removed]
        print("RFE_PATH_FROZEN 200 STATES; 600 REFITS; 199 DELETIONS", flush=True)


def verify_rfe() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(read_json(PLAN_QA)["status"] == "PASS", "Plan QA missing")
    require(not any((OUT / "test").glob("k_*.json")),
            "Test records exist before the RFE path QA")
    plan, train, valid = training_data()
    require(len(list((OUT / "states").glob("k_*.json"))) == N_FEATURES and
            len(list((OUT / "training").glob("k_*_seed*.json"))) == N_FEATURES * len(SEEDS) and
            len(list((OUT / "models").glob("k_*_seed*.pkl"))) == N_FEATURES * len(SEEDS),
            "RFE artifact counts incomplete or unexpected")
    names = list(manifest["start_feature_names"])
    original_index = {name: i for i, name in enumerate(names)}
    for k in range(N_FEATURES, 0, -1):
        state = record_at(state_path(k), fp)
        require(state is not None and state["k"] == k and
                state["retained_feature_names"] == names and
                state["group_counts"] == group_counts(names), f"Invalid state k={k}")
        records = []
        for seed in SEEDS:
            row = record_at(train_path(k, seed), fp)
            require(row is not None and row["feature_names"] == names and
                    row["k"] == k and row["seed"] == seed and
                    row["config"] == manifest["fixed_config"] and
                    row["fitted_model_sha256"] == sha(model_path(k, seed)),
                    f"Training artifact mismatch k={k}/seed={seed}")
            rounds = row["round_history"]
            require(1 <= row["effective_best_round"] <= len(rounds) ==
                    row["trained_rounds"] <= MAX_ROUNDS and
                    len(row["training_prediction_db"]) == 2404 and
                    len(row["validation_prediction_db"]) == 343 and
                    len(row["raw_total_gain_by_feature"]) == k and
                    len(row["normalized_total_gain_by_feature"]) == k,
                    f"Incomplete fitted path k={k}/seed={seed}")
            best = min(rounds, key=lambda item: (item["validation_rmse_db"],
                                                       item["round"]))["round"]
            require(row["effective_best_round"] == best and
                    all(item["round"] == i + 1 and
                        np.isfinite(item["train_rmse_db"]) and
                        np.isfinite(item["validation_rmse_db"])
                        for i, item in enumerate(rounds)),
                    f"Round history mismatch k={k}/seed={seed}")
            tr = base.c.rmse(train[base.c.TARGET].to_numpy(dtype=float),
                             np.asarray(row["training_prediction_db"], dtype=float))
            va = base.c.rmse(valid[base.c.TARGET].to_numpy(dtype=float),
                             np.asarray(row["validation_prediction_db"], dtype=float))
            require(abs(tr - row["train_rmse_db"]) < 1e-9 and
                    abs(va - row["validation_rmse_db"]) < 1e-9 and
                    abs(va - rounds[best - 1]["validation_rmse_db"]) < 1e-5,
                    f"Prediction/metric mismatch k={k}/seed={seed}")
            gains = np.asarray(row["raw_total_gain_by_feature"], dtype=float)
            total = float(gains.sum())
            expected_norm = gains / total if total > 0 else np.zeros(k)
            require(np.isfinite(gains).all() and (gains >= 0).all() and
                    np.allclose(expected_norm, row["normalized_total_gain_by_feature"],
                                rtol=0, atol=1e-12),
                    f"Feature importance mismatch k={k}/seed={seed}")
            with model_path(k, seed).open("rb") as handle:
                saved = pickle.load(handle)
            require(saved["fingerprint"] == fp and saved["feature_names"] == names and
                    saved["best_round"] == best and saved["seed"] == seed and
                    saved["k"] == k and
                    saved["imputer"].statistics_.shape == (k,),
                    f"Serialized model metadata mismatch k={k}/seed={seed}")
            raw_gain, normalized = importance_from_booster(saved["model"], names, best)
            require(np.allclose(raw_gain, gains, rtol=1e-6, atol=1e-6) and
                    np.allclose(normalized, expected_norm, rtol=0, atol=1e-12),
                    f"Serialized model gain mismatch k={k}/seed={seed}")
            records.append(row)
        mean_gain = np.mean([row["normalized_total_gain_by_feature"]
                             for row in records], axis=0)
        removed = None if k == 1 else names[min(range(k), key=lambda i:
                    (float(mean_gain[i]), original_index[names[i]]))]
        require(np.allclose(mean_gain, state["mean_normalized_total_gain_by_feature"],
                            rtol=0, atol=1e-12) and
                state["removed_feature_for_next_k"] == removed and
                state["model_sha256_by_seed"] ==
                {str(row["seed"]): row["fitted_model_sha256"] for row in records},
                f"Recursive deletion mismatch k={k}")
        if removed is not None:
            names = [name for name in names if name != removed]
    qa = {"status": "PASS", "stage": "rfe", "fingerprint": fp,
          "states": 200, "fitted_models": 600, "deletions": 199,
          "test_not_read_by_rfe": True, "final_feature": names[0],
          "verified_utc": base.c.prior.utc_now()}
    save_json(RFE_QA, qa)
    return qa


def test() -> None:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(RFE_QA.exists() and read_json(RFE_QA)["status"] == "PASS" and
            read_json(RFE_QA)["fingerprint"] == fp,
            "RFE QA must pass before the test split is opened")
    plan = base.checked_plan()
    frame = base.test_frame(plan["all_feature_names"])
    truth = frame[base.c.TARGET].to_numpy(dtype=float)
    require(len(frame) == 687 and np.isfinite(truth).all(), "Test split changed")
    with base.c.writer_lock():
        for k in range(N_FEATURES, 0, -1):
            state = record_at(state_path(k), fp)
            require(state is not None, f"RFE state missing k={k}")
            names = state["retained_feature_names"]
            for seed in SEEDS:
                existing = record_at(test_path(k, seed), fp)
                if existing is not None:
                    require(existing["feature_names"] == names and
                            existing["fitted_model_sha256"] == sha(model_path(k, seed)) and
                            existing["state_sha256"] == sha(state_path(k)),
                            f"Resumed test record mismatch k={k}/seed={seed}")
                    continue
                train_row = record_at(train_path(k, seed), fp)
                require(train_row is not None and
                        train_row["fitted_model_sha256"] == sha(model_path(k, seed)),
                        f"Fitted model changed k={k}/seed={seed}")
                with model_path(k, seed).open("rb") as handle:
                    saved = pickle.load(handle)
                require(saved["fingerprint"] == fp and saved["feature_names"] == names,
                        f"Model identity changed k={k}/seed={seed}")
                raw = raw_frame(frame, names, plan["reference_median_by_feature"])
                x = saved["imputer"].transform(raw)
                require(x.shape == (687, k) and np.isfinite(x).all(),
                        f"Test preprocessing invalid k={k}/seed={seed}")
                pred = np.asarray(saved["model"].predict(
                    x, iteration_range=(0, saved["best_round"])), dtype=float)
                metrics = base.score(frame, np.arange(687), pred)
                row = {"fingerprint": fp, "k": k, "seed": seed,
                       "feature_names": names, "n_test": 687,
                       "effective_best_round": saved["best_round"],
                       "fitted_model_sha256": sha(model_path(k, seed)),
                       "state_sha256": sha(state_path(k)),
                       "test_prediction_db": pred.tolist(), "test_metrics": metrics,
                       "completed_utc": base.c.prior.utc_now()}
                save_json(test_path(k, seed), row)
                print(f"TEST k={k} seed={seed} RMSE="
                      f"{metrics['overall']['rmse']:.6f}", flush=True)
        print("TEST_SCORED 600 FITTED MODELS; NO RETRAINING", flush=True)


def verify_test() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(RFE_QA.exists() and read_json(RFE_QA)["status"] == "PASS",
            "RFE QA must pass")
    require(len(list((OUT / "test").glob("k_*_seed*.json"))) == 600,
            "Expected exactly 600 test records")
    plan = base.checked_plan()
    frame = base.test_frame(plan["all_feature_names"])
    rows = []
    feature_rows = []
    for k in range(N_FEATURES, 0, -1):
        state = record_at(state_path(k), fp)
        names = state["retained_feature_names"]
        feature_rows.append({"k": k, "n_base": state["group_counts"]["base"],
                             "feature_names_json": json.dumps(names, ensure_ascii=False),
                             "removed_for_next_k":
                             state["removed_feature_for_next_k"] or ""})
        seed_records = []
        for seed in SEEDS:
            row = record_at(test_path(k, seed), fp)
            require(row is not None and row["k"] == k and row["seed"] == seed and
                    row["feature_names"] == names and row["n_test"] == 687 and
                    row["state_sha256"] == sha(state_path(k)) and
                    row["fitted_model_sha256"] == sha(model_path(k, seed)) and
                    row["effective_best_round"] ==
                    read_json(train_path(k, seed))["effective_best_round"],
                    f"Test artifact identity mismatch k={k}/seed={seed}")
            pred = np.asarray(row["test_prediction_db"], dtype=float)
            require(len(pred) == 687 and np.isfinite(pred).all(),
                    f"Test predictions invalid k={k}/seed={seed}")
            recomputed = base.score(frame, np.arange(687), pred)
            for group in base.c.GROUPS:
                expected = recomputed[group]
                recorded = row["test_metrics"][group]
                require(expected["n"] == recorded["n"] and
                        expected["days"] == recorded["days"] and
                        all((expected[field] is None and recorded[field] is None) or
                            (expected[field] is not None and recorded[field] is not None
                             and abs(expected[field] - recorded[field]) < 1e-9)
                            for field in ("rmse", "mae", "bias")),
                        f"Test metric mismatch k={k}/seed={seed}/{group}")
            seed_records.append(row)
        summary = {"k": k, "n_features": k,
                   "n_base": state["group_counts"]["base"],
                   "removed_for_next_k": state["removed_feature_for_next_k"] or "",
                   "train_rmse_mean_db": state["training_rmse_mean_db"],
                   "validation_rmse_mean_db": state["validation_rmse_mean_db"],
                   "best_rounds_json": json.dumps(state["best_round_by_seed"]),
                   "feature_names_json": json.dumps(names, ensure_ascii=False)}
        for group in base.c.GROUPS:
            for metric in ("rmse", "mae", "bias"):
                values = [row["test_metrics"][group][metric] for row in seed_records]
                summary[f"test_{group}_{metric}_mean"] = (
                    float(np.mean(values)) if all(x is not None for x in values) else np.nan)
                summary[f"test_{group}_{metric}_sd"] = (
                    float(np.std(values, ddof=1)) if all(x is not None for x in values)
                    else np.nan)
            summary[f"test_{group}_n"] = seed_records[0]["test_metrics"][group]["n"]
        rows.append(summary)
    pd.DataFrame(rows).to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).to_csv(FEATURE_PATH, index=False, encoding="utf-8-sig")
    qa = {"status": "PASS", "stage": "test", "fingerprint": fp,
          "test_records": 600, "test_hours": 687, "path_rows": 200,
          "test_used_for_rfe_selection": False,
          "test_historically_accessed": True,
          "summary_sha256": sha(SUMMARY), "feature_path_sha256": sha(FEATURE_PATH),
          "verified_utc": base.c.prior.utc_now()}
    save_json(TEST_QA, qa)
    return qa


def report() -> dict:
    qa = read_json(TEST_QA)
    require(qa["status"] == "PASS" and qa["fingerprint"] ==
            checked_manifest()["fingerprint"] and
            qa["summary_sha256"] == sha(SUMMARY) and
            qa["feature_path_sha256"] == sha(FEATURE_PATH),
            "Test QA must pass before reporting")
    table = pd.read_csv(SUMMARY)
    chosen = table[table.k.isin([200, 100, 50, 20, 10, 5, 4, 3, 2, 1])]
    lines = ["# Revision 021: retrained RFE results", "",
             "The 200-to-1 elimination path was chosen solely on XGBoost training-set "
             "tree gain, with validation used for early stopping. Each k and each of three "
             "seeds was trained afresh (600 fits). No zero masking was used.", "",
             "The test split was opened only after the full RFE path passed QA. It was "
             "previously accessed and the XGBoost family was previously test-ranked; "
             "therefore these are exploratory holdout comparisons, not a new blind test. "
             "Do not select a final k from the test curve.", "",
             "| k | Train RMSE | Validation RMSE | Test RMSE | Rain RMSE | Dry-overcast RMSE |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in chosen.itertuples():
        lines.append(f"| {row.k} | {row.train_rmse_mean_db:.3f} | "
                     f"{row.validation_rmse_mean_db:.3f} | "
                     f"{row.test_overall_rmse_mean:.3f} | "
                     f"{row.test_rain_rmse_mean:.3f} | "
                     f"{row.test_dry_overcast_rmse_mean:.3f} |")
    last = read_json(state_path(1))
    lines += ["", f"The final one-feature state retains `{last['retained_feature_names'][0]}`.",
              "", f"Full 200-row metrics: `{SUMMARY.name}`. Complete feature lists: "
              f"`{FEATURE_PATH.name}`. Every state, training curve, fitted model, and "
              "per-hour test prediction is saved under this output directory.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "REPORTED", "result": str(OUT / "RESULTS.md")}


def status() -> dict:
    def count(folder: str, pattern: str) -> int:
        return len(list((OUT / folder).glob(pattern)))
    return {"manifest": MANIFEST.exists(),
            "plan_qa": read_json(PLAN_QA).get("status") if PLAN_QA.exists() else None,
            "states": count("states", "k_*.json"),
            "training_records": count("training", "k_*_seed*.json"),
            "fitted_models": count("models", "k_*_seed*.pkl"),
            "rfe_qa": read_json(RFE_QA).get("status") if RFE_QA.exists() else None,
            "test_records": count("test", "k_*_seed*.json"),
            "test_qa": read_json(TEST_QA).get("status") if TEST_QA.exists() else None,
            "report": (OUT / "RESULTS.md").exists()}


def smoke() -> dict:
    plan, train, valid = training_data()
    names = plan["retained_feature_names"][:5]
    xtrain = raw_frame(train.iloc[:160], names, plan["reference_median_by_feature"])
    xvalid = raw_frame(valid.iloc[:80], names, plan["reference_median_by_feature"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xtrain = imputer.fit_transform(xtrain)
    xvalid = imputer.transform(xvalid)
    model = base.c.make_estimator("xgboost", {"n_estimators": 20, "max_depth": 3,
                    "learning_rate": 0.03}, SEEDS[0], 5, tuple(range(10)))
    model.set_params(eval_metric="rmse", early_stopping_rounds=3)
    model.fit(xtrain, train[base.c.TARGET].to_numpy(dtype=float)[:160],
              eval_set=[(xtrain, train[base.c.TARGET].to_numpy(dtype=float)[:160]),
                        (xvalid, valid[base.c.TARGET].to_numpy(dtype=float)[:80])],
              verbose=False)
    best = int(model.best_iteration) + 1
    raw_gain, normalized = importance_from_booster(model, names, best)
    pred = model.predict(xvalid, iteration_range=(0, best))
    require(np.isfinite(pred).all() and len(raw_gain) == len(normalized) == 5,
            "RFE smoke test failed")
    return {"status": "PASS", "best_round": best,
            "trained_rounds": len(model.evals_result()["validation_1"]["rmse"]),
            "gain_total": float(sum(raw_gain))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "prepare", "verify-plan",
                    "rfe", "verify-rfe", "test", "verify-test", "report", "status"))
    args = parser.parse_args()
    action = {"smoke": smoke, "prepare": prepare, "verify-plan": verify_plan,
              "rfe": rfe, "verify-rfe": verify_rfe, "test": test,
              "verify-test": verify_test, "report": report, "status": status}[args.command]
    result = action()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
