"""Train ten frozen model families, rank on test, then zero-mask the winner.

All generated files are confined to the Revision 019 output directory. Old
source datasets and experiment revisions are read-only runtime dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "Revision_014"
sys.path.insert(0, str(PRIOR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import common as c  # noqa: E402

DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_019_Test_Ranked_Zero_Mask"
PLAN = OUT / "feature_plan.json"
MANIFEST = OUT / "run_manifest.json"
FREEZE = OUT / "validation_freeze.json"
TRAIN_QA = OUT / "training_qa.json"
RANKING = OUT / "test_ranking.json"
TEST_QA = OUT / "test_ranking_qa.json"
MASK_QA = OUT / "zero_mask_qa.json"
SEEDS = (17, 29, 43)
THRESHOLD = 0.90
MIN_PAIR_OBS = 200
MAX_MISSING = 0.50
STRUCTURED = frozenset(("transformer", "gcn", "bilstm"))
VERSION = "test-ranked-ten-model-zero-mask-v19-20261001"

c.OUT = OUT  # The shared lock writes only inside this new revision.


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return c.prior.sha256(path)


def fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    c.prior.atomic_json(path, payload)


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def feature_screen() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Use only training-split meteorological columns, not labels/events."""
    names = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    require(len(names) == len(set(names)) == 348, "348 ordered features required")
    train = pd.read_parquet(DATA / "train.parquet", columns=names)
    require(len(train) == 2404, "Training split changed")
    train = train.replace([np.inf, -np.inf], np.nan)
    missing = train.isna().mean()
    unique = train.nunique(dropna=True)
    eligible = [name for name in names if missing[name] <= MAX_MISSING and unique[name] > 1]
    corr = train[eligible].corr(method="spearman", min_periods=MIN_PAIR_OBS).abs()
    order = sorted(eligible, key=lambda name: (float(missing[name]), names.index(name)))
    selected: list[str] = []
    redundant: dict[str, tuple[str, float]] = {}
    for name in order:
        match = next((kept for kept in selected if corr.loc[name, kept] >= THRESHOLD), None)
        if match is None:
            selected.append(name)
        else:
            redundant[name] = (match, float(corr.loc[name, match]))
    selected_set = set(selected)
    selected = [name for name in names if name in selected_set]
    remaining = corr.loc[selected, selected].to_numpy()
    max_remaining = float(np.nanmax(remaining[np.triu_indices(len(selected), 1)]))
    medians = {name: float(train[name].median()) for name in selected}
    levels = {str(level): sum(name.startswith(f"p{level}__") for name in selected)
              for level in c.LEVELS}
    require(len(selected) == 200 and max_remaining < THRESHOLD,
            "Expected 200 independent retained features")
    require(all(levels.values()) and len(c.LEVELS) == 10,
            "At least one pressure level lost all retained features")
    require(all(np.isfinite(value) for value in medians.values()), "Nonfinite train median")

    audit = []
    for index, name in enumerate(names, 1):
        why = ("missing_gt_50_percent" if missing[name] > MAX_MISSING else
               "constant_or_empty" if unique[name] <= 1 else
               "correlated" if name in redundant else "retained")
        match = redundant.get(name)
        audit.append({"feature_order": index, "column": name, "decision": why,
                      "train_missing_fraction": float(missing[name]),
                      "train_nonmissing_unique_values": int(unique[name]),
                      "representative": "" if match is None else match[0],
                      "absolute_spearman_with_representative": None if match is None else match[1]})
    pairs = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= THRESHOLD:
                pairs.append({"left": left, "right": right, "absolute_spearman": float(value)})
    require(len(pairs) == 542, "Training correlation inventory changed")

    payload = {
        "version": VERSION + "-feature-plan",
        "source_hashes": source_hashes([DATA / "train.parquet", DATA / "feature_columns.csv",
                                        HERE / "PROTOCOL.md", HERE / "run_experiment.py"]),
        "selection_scope": "train meteorological columns only; no target/events/validation/test",
        "threshold_absolute_spearman": THRESHOLD,
        "minimum_pairwise_complete_hours": MIN_PAIR_OBS,
        "maximum_missing_fraction": MAX_MISSING,
        "all_feature_names": names,
        "retained_feature_names": selected,
        "retained_base_count": sum(name.startswith("surface__") for name in selected),
        "retained_per_pressure_level": levels,
        "reference_median_by_feature": medians,
        "n_original": len(names), "n_retained": len(selected),
        "n_high_correlation_pairs_original": len(pairs),
        "max_remaining_absolute_spearman": max_remaining,
    }
    payload["fingerprint"] = fingerprint(payload)
    payload["prepared_utc"] = c.prior.utc_now()
    return payload, pd.DataFrame(audit), pd.DataFrame(pairs)


def prepare() -> dict:
    payload, audit, pairs = feature_screen()
    if PLAN.exists():
        require(read_json(PLAN)["fingerprint"] == payload["fingerprint"],
                "Feature sources or code changed after freezing")
        return read_json(PLAN)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "feature_audit.csv", index=False)
    pairs.to_csv(OUT / "high_correlation_pairs.csv", index=False)
    save_json(PLAN, payload)
    return payload


def checked_plan() -> dict:
    require(PLAN.exists(), "Run prepare first")
    frozen = read_json(PLAN)
    replay, _, _ = feature_screen()
    require(frozen["fingerprint"] == replay["fingerprint"], "Feature plan replay mismatch")
    return frozen


def verify_plan() -> dict:
    frozen = checked_plan()
    replay, audit, pairs = feature_screen()
    old_audit = pd.read_csv(OUT / "feature_audit.csv")
    old_pairs = pd.read_csv(OUT / "high_correlation_pairs.csv")
    require(len(audit) == len(old_audit) == 348 and
            audit.column.tolist() == old_audit.column.tolist() and
            audit.decision.tolist() == old_audit.decision.tolist(), "Feature audit mismatch")
    require(len(pairs) == len(old_pairs) == 542 and
            pairs[["left", "right"]].equals(old_pairs[["left", "right"]]),
            "High-correlation pair audit mismatch")
    require(replay["fingerprint"] == frozen["fingerprint"] and
            frozen["retained_base_count"] == 88 and
            sum(frozen["retained_per_pressure_level"].values()) == 112,
            "Feature group count mismatch")
    qa = {"status": "PASS", "stage": "feature-plan", "fingerprint": frozen["fingerprint"],
          "n_original": 348, "n_retained": 200, "n_high_correlation_pairs": 542,
          "all_ten_levels_nonempty": True, "verified_utc": c.prior.utc_now()}
    save_json(OUT / "feature_plan_qa.json", qa)
    return qa


def manifest_candidate(plan: dict) -> dict:
    paths = [DATA / "train.parquet", DATA / "validation.parquet",
             DATA / "feature_columns.csv", DATA / "hourly_sample_status.parquet",
             PLAN, HERE / "PROTOCOL.md", HERE / "run_experiment.py",
             PRIOR / "common.py", PRIOR / "models.py", PRIOR / "support.py"]
    payload = {"version": VERSION + "-train", "source_hashes": source_hashes(paths),
               "feature_fingerprint": plan["fingerprint"],
               "retained_features": plan["retained_feature_names"],
               "n_train": 2404, "n_validation": 343, "models": list(c.MODEL_NAMES),
               "candidate_configs": {name: c.candidate_configs(name) for name in c.MODEL_NAMES},
               "seeds": list(SEEDS),
               "validation_role": "select configuration and stopping round within each family only",
               "test_ranking_rule": "50% normalized overall test RMSE + 50% normalized log single-row latency",
               "zero_reference": "final processed model input zero; train-median raw centering",
               "test_not_read": True}
    payload["fingerprint"] = fingerprint(payload)
    payload["created_utc"] = c.prior.utc_now()
    return payload


def checked_manifest() -> dict:
    plan = checked_plan()
    require((OUT / "feature_plan_qa.json").exists() and
            read_json(OUT / "feature_plan_qa.json")["status"] == "PASS",
            "Feature-plan QA must pass")
    candidate = manifest_candidate(plan)
    if MANIFEST.exists():
        frozen = read_json(MANIFEST)
        require(frozen["fingerprint"] == candidate["fingerprint"],
                "Training manifest fingerprint changed")
        return frozen
    save_json(MANIFEST, candidate)
    return candidate


def training_frame(plan: dict) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    frame, names, _ = c.load_training()
    require(names == plan["all_feature_names"], "Original feature order changed")
    fit = np.flatnonzero(frame.split.to_numpy() == "train")
    valid = np.flatnonzero(frame.split.to_numpy() == "validation")
    require(len(fit) == 2404 and len(valid) == 343, "Split counts changed")
    return frame, names, fit, valid


def raw_input(frame: pd.DataFrame, positions: np.ndarray, plan: dict,
              model_name: str) -> np.ndarray:
    names = plan["all_feature_names"]
    keep = set(plan["retained_feature_names"])
    raw = frame.loc[positions, names].to_numpy(dtype=float, copy=True)
    raw[~np.isfinite(raw)] = np.nan
    for i, name in enumerate(names):
        if name in keep:
            raw[:, i] -= plan["reference_median_by_feature"][name]
        else:
            raw[:, i] = np.nan
    if model_name not in STRUCTURED:
        indices = [i for i, name in enumerate(names) if name in keep]
        raw = raw[:, indices]
    return raw


def processed(frame: pd.DataFrame, fit: np.ndarray, hold: np.ndarray,
              plan: dict, name: str):
    raw_fit = raw_input(frame, fit, plan, name)
    raw_hold = raw_input(frame, hold, plan, name)
    xf, xh, imputer, scaler = c.preprocess(raw_fit, raw_hold, name)
    require(xf.shape[1] == (348 if name in STRUCTURED else 200),
            f"Unexpected model input width for {name}")
    return xf, xh, imputer, scaler, raw_hold


def score(frame: pd.DataFrame, positions: np.ndarray, pred: np.ndarray) -> dict:
    return c.metrics(frame, positions, np.asarray(pred, dtype=float).reshape(-1))


def training_paths(name: str, config_id: int | None = None, seed: int | None = None) -> Path:
    if config_id is not None:
        return OUT / "trials" / f"{name}_{config_id:02d}.json"
    require(seed is not None, "Training seed missing")
    return OUT / "validation" / f"{name}_seed{seed}.json"


def train() -> None:
    plan = checked_plan()
    manifest = checked_manifest()
    frame, _, fit, valid = training_frame(plan)
    fp = manifest["fingerprint"]
    yfit = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    yval = frame.loc[valid, c.TARGET].to_numpy(dtype=float)
    with c.writer_lock():
        for name in c.MODEL_NAMES:
            xf, xv, _, _, _ = processed(frame, fit, valid, plan, name)
            trials = []
            for config_id, config in enumerate(c.candidate_configs(name)):
                path = training_paths(name, config_id=config_id)
                old = c.read_record(path, fp)
                if old is None:
                    model, curve, pred, meta = c.fit_logged(
                        name, config, SEEDS[0], tuple(range(10)), xf, yfit, xv, yval)
                    del model
                    values = score(frame, valid, pred)
                    old = {"fingerprint": fp, "model": name, "config_id": config_id,
                           "config": config, "seed": SEEDS[0],
                           "n_train": len(fit), "n_validation": len(valid),
                           "round_history": curve, **meta,
                           "formal_rounds": meta["selected_round"],
                           "validation_prediction_db": np.asarray(pred).tolist(),
                           "validation_metrics": values, "completed_utc": c.prior.utc_now()}
                    save_json(path, old)
                    print(f"TRIAL {name} {config_id + 1}/4 RMSE={values['overall']['rmse']:.6f} "
                          f"round={old['formal_rounds']}", flush=True)
                trials.append(old)
            best = min(trials, key=lambda row: (
                row["validation_metrics"]["overall"]["rmse"],
                row["validation_metrics"]["overall"]["mae"], row["config_id"]))
            for seed in SEEDS:
                path = training_paths(name, seed=seed)
                old = c.read_record(path, fp)
                if old is not None:
                    require(old["config_id"] == best["config_id"], "Validation config drift")
                    continue
                model, curve, _, meta = c.fit_logged(
                    name, best["config"], seed, tuple(range(10)), xf, yfit,
                    fixed_rounds=best["formal_rounds"])
                pred = np.asarray(model.predict(xv), dtype=float).reshape(-1)
                del model
                values = score(frame, valid, pred)
                old = {"fingerprint": fp, "model": name, "config_id": best["config_id"],
                       "config": best["config"], "seed": seed,
                       "formal_rounds": best["formal_rounds"],
                       "n_train": len(fit), "n_validation": len(valid),
                       "round_history": curve, **meta,
                       "validation_prediction_db": pred.tolist(),
                       "validation_metrics": values, "completed_utc": c.prior.utc_now()}
                save_json(path, old)
                print(f"VALIDATION {name} seed={seed} RMSE={values['overall']['rmse']:.6f}",
                      flush=True)
        rows = []
        for name in c.MODEL_NAMES:
            trials = [c.read_record(training_paths(name, config_id=i), fp) for i in range(4)]
            best = min(trials, key=lambda row: (
                row["validation_metrics"]["overall"]["rmse"],
                row["validation_metrics"]["overall"]["mae"], row["config_id"]))
            seeds = [c.read_record(training_paths(name, seed=s), fp) for s in SEEDS]
            rows.append({"name": name, "config_id": best["config_id"],
                         "config": best["config"], "formal_rounds": best["formal_rounds"],
                         "validation_overall_rmse_mean": float(np.mean([
                             item["validation_metrics"]["overall"]["rmse"] for item in seeds])),
                         "validation_overall_rmse_by_seed": {str(item["seed"]):
                             item["validation_metrics"]["overall"]["rmse"] for item in seeds}})
        freeze = {"fingerprint": fp, "status": "TEN_FAMILIES_CONFIGURED_NOT_RANKED",
                  "n_candidate_trials": 40, "n_validation_refits": 30,
                  "models": rows, "winner": None, "test_not_read": True,
                  "completed_utc": c.prior.utc_now()}
        save_json(FREEZE, freeze)
        print("TEN_CONFIGS_FROZEN; NO FAMILY SELECTED", flush=True)


def verify_train() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(FREEZE.exists(), "Validation freeze missing")
    freeze = read_json(FREEZE)
    require(freeze["fingerprint"] == fp and freeze["winner"] is None and
            freeze["test_not_read"] is True and len(freeze["models"]) == 10,
            "Validation freeze incorrectly selected a model")
    frame, _, _, valid = training_frame(checked_plan())
    y = frame.loc[valid, c.TARGET].to_numpy(dtype=float)
    for row in freeze["models"]:
        name = row["name"]
        trials = [c.read_record(training_paths(name, config_id=i), fp) for i in range(4)]
        require(all(item is not None for item in trials), f"Missing trial: {name}")
        best = min(trials, key=lambda item: (
            item["validation_metrics"]["overall"]["rmse"],
            item["validation_metrics"]["overall"]["mae"], item["config_id"]))
        require(best["config_id"] == row["config_id"] and
                best["formal_rounds"] == row["formal_rounds"],
                f"Validation-selected config/round mismatch: {name}")
        for item in trials + [c.read_record(training_paths(name, seed=s), fp) for s in SEEDS]:
            require(item is not None and len(item["validation_prediction_db"]) == 343 and
                    len(item["round_history"]) >= 1, f"Incomplete model record: {name}")
            pred = np.asarray(item["validation_prediction_db"], dtype=float)
            require(abs(c.rmse(y, pred) - item["validation_metrics"]["overall"]["rmse"]) < 1e-10,
                    f"Validation RMSE mismatch: {name}")
        for seed in SEEDS:
            item = c.read_record(training_paths(name, seed=seed), fp)
            require(item["config_id"] == best["config_id"] and
                    item["formal_rounds"] == best["formal_rounds"],
                    f"Seed refit mismatch: {name}/{seed}")
    qa = {"status": "PASS", "stage": "training-validation", "fingerprint": fp,
          "candidate_trials": 40, "validation_refits": 30,
          "no_model_winner_from_validation": True, "test_not_read": True,
          "verified_utc": c.prior.utc_now()}
    save_json(TRAIN_QA, qa)
    return qa


def save_pickle(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha(path)


def test_frame(names: list[str]) -> pd.DataFrame:
    table = c.residual.load_test(names)
    require(len(table) == 687, "Test split changed")
    return table


def test_record_path(name: str, seed: int) -> Path:
    return OUT / "test_models" / f"{name}_seed{seed}.json"


def model_path(name: str, seed: int) -> Path:
    return OUT / "fitted_models" / f"{name}_seed{seed}.pkl"


def rank_test_records(fp: str) -> dict:
    rows = []
    for name in c.MODEL_NAMES:
        records = [c.read_record(test_record_path(name, seed), fp) for seed in SEEDS]
        require(all(item is not None for item in records), f"Test results incomplete: {name}")
        rmse = [item["test_metrics"]["overall"]["rmse"] for item in records]
        rain = [item["test_metrics"]["rain"]["rmse"] for item in records]
        cloudy = [item["test_metrics"]["dry_overcast"]["rmse"] for item in records]
        latency = [item["latency"]["median_ms"] for item in records]
        rows.append({"name": name, "overall_rmse_mean": float(np.mean(rmse)),
                     "overall_rmse_sd_seed": float(np.std(rmse, ddof=1)),
                     "rain_rmse_mean": float(np.mean(rain)),
                     "dry_overcast_rmse_mean": float(np.mean(cloudy)),
                     "event_rmse_mean": float(np.mean([*rain, *cloudy])),
                     "latency_median_ms": float(np.median(latency)),
                     "test_hours": 687})
    ranked, summary = c.cleaned.rank_models(rows)
    return {"fingerprint": fp, "status": "TEST_RANKED",
            "ranking_rule": "50% normalized test overall RMSE + 50% normalized log single-row latency",
            "model_rows": ranked, **summary, "n_test": 687,
            "historically_accessed_test": True,
            "completed_utc": c.prior.utc_now()}


def test_models() -> None:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(TRAIN_QA.exists() and read_json(TRAIN_QA)["status"] == "PASS",
            "Training-validation QA must pass before test")
    freeze = read_json(FREEZE)
    plan = checked_plan()
    frame, names, fit, _ = training_frame(plan)
    test = test_frame(names)
    yfit = frame.loc[fit, c.TARGET].to_numpy(dtype=float)
    tpos = np.arange(len(test))
    with c.writer_lock():
        for row in freeze["models"]:
            name = row["name"]
            raw_train = raw_input(frame, fit, plan, name)
            raw_test = raw_input(test, tpos, plan, name)
            xf, xt, imputer, scaler = c.preprocess(raw_train, raw_test, name)
            for seed in SEEDS:
                path = test_record_path(name, seed)
                stored = c.read_record(path, fp)
                if stored is not None:
                    require(model_path(name, seed).exists() and
                            sha(model_path(name, seed)) == stored["fitted_model_sha256"],
                            f"Saved fitted model missing or changed: {name}/{seed}")
                    continue
                model, history, _, meta = c.fit_logged(
                    name, row["config"], seed, tuple(range(10)), xf, yfit,
                    fixed_rounds=row["formal_rounds"])
                pred = np.asarray(model.predict(xt), dtype=float).reshape(-1)
                timing = c.measure_latency(model, imputer, scaler, raw_test)
                model_hash = save_pickle(model_path(name, seed),
                                         {"fingerprint": fp, "name": name, "seed": seed,
                                          "model": model, "imputer": imputer,
                                          "scaler": scaler})
                metrics = score(test, tpos, pred)
                record = {"fingerprint": fp, "name": name, "seed": seed,
                          "config": row["config"], "formal_rounds": row["formal_rounds"],
                          "trained_rounds": meta["trained_rounds"],
                          "round_history": history, "n_train": len(fit), "n_test": len(test),
                          "test_prediction_db": pred.tolist(), "test_metrics": metrics,
                          "latency": timing, "fitted_model_sha256": model_hash,
                          "completed_utc": c.prior.utc_now()}
                save_json(path, record)
                print(f"TEST {name} seed={seed} RMSE={metrics['overall']['rmse']:.6f} "
                      f"latency_ms={timing['median_ms']:.5f}", flush=True)
        ranking = rank_test_records(fp)
        save_json(RANKING, ranking)
        pd.DataFrame(ranking["model_rows"]).to_csv(OUT / "test_ranking.csv", index=False)
        print(f"TEST_WINNER {ranking['primary_winner']} "
              f"RMSE_ONLY={ranking['rmse_only_winner']}", flush=True)


def verify_test() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(TRAIN_QA.exists() and read_json(TRAIN_QA)["status"] == "PASS",
            "Training QA missing")
    require(RANKING.exists(), "Test ranking missing")
    saved = read_json(RANKING)
    plan = checked_plan()
    test = test_frame(plan["all_feature_names"])
    truth = test[c.TARGET].to_numpy(dtype=float)
    for name in c.MODEL_NAMES:
        for seed in SEEDS:
            row = c.read_record(test_record_path(name, seed), fp)
            require(row is not None and row["n_train"] == 2404 and row["n_test"] == 687,
                    f"Missing test row: {name}/{seed}")
            model_file = model_path(name, seed)
            require(model_file.exists() and sha(model_file) == row["fitted_model_sha256"],
                    f"Fitted model digest mismatch: {name}/{seed}")
            pred = np.asarray(row["test_prediction_db"], dtype=float)
            require(len(pred) == len(truth) and np.isfinite(pred).all() and
                    abs(c.rmse(truth, pred) - row["test_metrics"]["overall"]["rmse"]) < 1e-10,
                    f"Test metric mismatch: {name}/{seed}")
            recheck = score(test, np.arange(len(test)), pred)
            for group in c.GROUPS:
                old, now = row["test_metrics"][group], recheck[group]
                require(old["n"] == now["n"] and
                        (old["rmse"] is None and now["rmse"] is None or
                         old["rmse"] is not None and now["rmse"] is not None and
                         abs(old["rmse"] - now["rmse"]) < 1e-10),
                        f"Event metric mismatch: {name}/{seed}/{group}")
    replay = rank_test_records(fp)
    require(saved["primary_winner"] == replay["primary_winner"] and
            saved["rmse_only_winner"] == replay["rmse_only_winner"] and
            [r["name"] for r in saved["model_rows"]] ==
            [r["name"] for r in replay["model_rows"]], "Test ranking changed")
    qa = {"status": "PASS", "stage": "ten-model-test-ranking", "fingerprint": fp,
          "candidate_trials": 40, "validation_refits": 30, "test_model_fits": 30,
          "n_test": 687, "winner": saved["primary_winner"],
          "historically_accessed_test": True, "verified_utc": c.prior.utc_now()}
    save_json(TEST_QA, qa)
    return qa


def mask_specs() -> list[tuple[str, set[int]]]:
    all_levels = set(c.LEVELS)
    specs = [("full", set()), ("base_only", all_levels)]
    specs += [(f"drop_p{level}", {level}) for level in c.LEVELS]
    specs += [(f"only_p{level}", all_levels - {level}) for level in c.LEVELS]
    require(len(specs) == 22 and len(set(name for name, _ in specs)) == 22,
            "Zero-mask plan must have 22 unique configurations")
    return specs


def mask_path(label: str, seed: int) -> Path:
    return OUT / "zero_masks" / f"{label}_seed{seed}.json"


def zero_masks() -> None:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(TEST_QA.exists() and read_json(TEST_QA)["status"] == "PASS",
            "Ten-model test-ranking QA must pass before zero masks")
    ranking = read_json(RANKING)
    name = ranking["primary_winner"]
    plan = checked_plan()
    test = test_frame(plan["all_feature_names"])
    tpos = np.arange(len(test))
    retained = plan["all_feature_names"] if name in STRUCTURED else plan["retained_feature_names"]
    with c.writer_lock():
        for seed in SEEDS:
            result = c.read_record(test_record_path(name, seed), fp)
            path = model_path(name, seed)
            require(sha(path) == result["fitted_model_sha256"], "Frozen model changed")
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            require(saved["fingerprint"] == fp and saved["name"] == name and
                    saved["seed"] == seed, "Fitted model identity mismatch")
            raw = raw_input(test, tpos, plan, name)
            imputer, scaler, model = saved["imputer"], saved["scaler"], saved["model"]
            xt = imputer.transform(raw)
            if scaler is not None:
                xt = scaler.transform(xt)
            full = np.asarray(model.predict(xt), dtype=float).reshape(-1)
            require(np.max(np.abs(full - np.asarray(result["test_prediction_db"]))) < 1e-8,
                    "Frozen full-input prediction differs from ten-model comparison")
            for label, hidden in mask_specs():
                path_out = mask_path(label, seed)
                old = c.read_record(path_out, fp)
                if old is not None:
                    require(old["fitted_model_sha256"] == result["fitted_model_sha256"],
                            "Zero-mask model changed")
                    continue
                xm = xt.copy()
                indices = [i for i, feature in enumerate(retained)
                           if feature.startswith("p") and
                           int(feature.split("__", 1)[0][1:]) in hidden]
                xm[:, indices] = 0.0
                pred = np.asarray(model.predict(xm), dtype=float).reshape(-1)
                met = score(test, tpos, pred)
                row = {"fingerprint": fp, "winner": name, "seed": seed,
                       "label": label, "zeroed_levels_hpa": sorted(hidden),
                       "zeroed_retained_feature_names": [retained[i] for i in indices],
                       "zeroed_model_input_coordinates": len(indices),
                       "fitted_model_sha256": result["fitted_model_sha256"],
                       "test_prediction_db": pred.tolist(), "metrics": met,
                       "mean_prediction_shift_db": float(np.mean(pred - full)),
                       "mean_absolute_prediction_shift_db": float(np.mean(np.abs(pred - full))),
                       "completed_utc": c.prior.utc_now()}
                save_json(path_out, row)
                print(f"MASK {label} seed={seed} RMSE={met['overall']['rmse']:.6f}", flush=True)
        rows = []
        for label, hidden in mask_specs():
            records = [c.read_record(mask_path(label, seed), fp) for seed in SEEDS]
            require(all(item is not None for item in records), "Zero-mask records incomplete")
            rows.append({"label": label, "zeroed_levels_hpa": "+".join(map(str, sorted(hidden))),
                         "zeroed_coordinates": records[0]["zeroed_model_input_coordinates"],
                         "overall_rmse_mean": float(np.mean([r["metrics"]["overall"]["rmse"] for r in records])),
                         "rain_rmse_mean": float(np.mean([r["metrics"]["rain"]["rmse"] for r in records])),
                         "dry_overcast_rmse_mean": float(np.mean([
                             r["metrics"]["dry_overcast"]["rmse"] for r in records])),
                         "mean_absolute_prediction_shift_db": float(np.mean([
                             r["mean_absolute_prediction_shift_db"] for r in records]))})
        pd.DataFrame(rows).to_csv(OUT / "zero_mask_summary.csv", index=False)
        print("ZERO_MASKS_WRITTEN 66; NO MASK-SPECIFIC TRAINING", flush=True)


def verify_all() -> dict:
    manifest = checked_manifest()
    fp = manifest["fingerprint"]
    require(TEST_QA.exists() and read_json(TEST_QA)["status"] == "PASS",
            "Ten-model test QA missing")
    require((OUT / "zero_mask_summary.csv").exists(), "Zero-mask summary missing")
    ranking = read_json(RANKING)
    name = ranking["primary_winner"]
    plan = checked_plan()
    test = test_frame(plan["all_feature_names"])
    truth = test[c.TARGET].to_numpy(dtype=float)
    expected = {label: sorted(hidden) for label, hidden in mask_specs()}
    for seed in SEEDS:
        model_test = c.read_record(test_record_path(name, seed), fp)
        full = np.asarray(model_test["test_prediction_db"], dtype=float)
        for label, levels in expected.items():
            record = c.read_record(mask_path(label, seed), fp)
            require(record is not None and record["winner"] == name and
                    record["zeroed_levels_hpa"] == levels and
                    record["fitted_model_sha256"] == model_test["fitted_model_sha256"],
                    f"Mask record identity mismatch: {label}/{seed}")
            pred = np.asarray(record["test_prediction_db"], dtype=float)
            require(len(pred) == 687 and np.isfinite(pred).all() and
                    abs(c.rmse(truth, pred) - record["metrics"]["overall"]["rmse"]) < 1e-10,
                    f"Mask metric mismatch: {label}/{seed}")
            if label == "full":
                require(np.max(np.abs(pred - full)) < 1e-8,
                        f"Full mask baseline changed: {seed}")
            require(abs(float(np.mean(np.abs(pred - full))) -
                        record["mean_absolute_prediction_shift_db"]) < 1e-10,
                    f"Prediction-shift mismatch: {label}/{seed}")
    summary = pd.read_csv(OUT / "zero_mask_summary.csv")
    require(len(summary) == 22 and set(summary.label) == set(expected),
            "Zero-mask summary incomplete")
    qa = {"status": "PASS", "stage": "all", "fingerprint": fp,
          "winner_test_ranked": name, "frozen_test_models": 30,
          "zero_mask_configs": 22, "zero_mask_prediction_records": 66,
          "no_mask_specific_training": True, "historically_accessed_test": True,
          "verified_utc": c.prior.utc_now()}
    save_json(MASK_QA, qa)
    return qa


def report() -> Path:
    require(MASK_QA.exists() and read_json(MASK_QA)["status"] == "PASS",
            "Final zero-mask QA must pass before report")
    plan, ranking = read_json(PLAN), read_json(RANKING)
    ranked = ranking["model_rows"]
    masks = pd.read_csv(OUT / "zero_mask_summary.csv")
    lines = ["# Revision 019 results", "",
             "This is a test-ranked model benchmark and a fixed-weight zero-reference sensitivity study, not a fresh blind test or a causal experiment. No residual model or 1024-subset refitting was used.", "",
             f"- Data: 2404 training, 343 validation, 687 test hours; target `{c.TARGET}`.",
             f"- Train-only correlation screen: 348 to {plan['n_retained']} columns; "
             f"{plan['retained_base_count']} base and 112 across all ten pressure levels.",
             "- Validation selected four-configuration family hyperparameters and stopping rounds; it did not rank model families.",
             "- All ten families were then evaluated on the same test hours at seeds 17, 29, 43. Test overall RMSE and single-row latency were weighted 50/50 to rank families.",
             f"- Test-ranked winner: **{ranking['primary_winner']}**; pure-RMSE winner: "
             f"**{ranking['rmse_only_winner']}**.", "",
             "| Test rank | Model | Overall RMSE (dB) | Rain RMSE (dB) | Dry-overcast RMSE (dB) | Median single-row latency (ms) |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in ranked:
        lines.append(f"| {row['rank_50_50']} | {row['name']} | {row['overall_rmse_mean']:.4f} | "
                     f"{row['rain_rmse_mean']:.4f} | {row['dry_overcast_rmse_mean']:.4f} | "
                     f"{row['latency_median_ms']:.4f} |")
    lines += ["", "## Fixed-weight zero-mask sensitivity", "",
              "The selected fitted models were loaded from the benchmark stage. Masking changed only final numerical input coordinates to zero; no model was refitted for any mask. Zero means a train-reference input, not physically absent atmosphere.", "",
              "| Mask | Zeroed levels (hPa) | Overall RMSE | Rain RMSE | Dry-overcast RMSE | Mean absolute prediction shift |",
              "|---|---|---:|---:|---:|---:|"]
    for row in masks.itertuples(index=False):
        levels = "none" if pd.isna(row.zeroed_levels_hpa) or not row.zeroed_levels_hpa else row.zeroed_levels_hpa
        lines.append(f"| {row.label} | {levels} | {row.overall_rmse_mean:.4f} | "
                     f"{row.rain_rmse_mean:.4f} | {row.dry_overcast_rmse_mean:.4f} | "
                     f"{row.mean_absolute_prediction_shift_db:.4f} |")
    lines += ["", "## Evidence and limits", "",
              "All per-configuration training histories, parameters, validation predictions, test predictions, metrics, model digests, frozen feature names, and QA reports are retained in this revision output directory. Weather event labels were used for evaluation only. Rain includes heavy rain.",
              "The local test set was accessed in historical experiments, and this revision explicitly uses it to select the winning model family. The numbers characterize this test benchmark; they must not be described as newly blind independent validation. Random-hour splitting also permits neighboring hours of one weather event across partitions. External sites or new time blocks are needed for stronger generalization claims.", ""]
    path = OUT / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def status() -> dict:
    return {"phase": "FINAL_QA_PASS" if MASK_QA.exists() else
            "ZERO_MASKS" if (OUT / "zero_mask_summary.csv").exists() else
            "TEST_QA_PASS" if TEST_QA.exists() else
            "TEST_RANKED" if RANKING.exists() else
            "TRAIN_QA_PASS" if TRAIN_QA.exists() else
            "TRAIN_FROZEN" if FREEZE.exists() else
            "PREPARED" if PLAN.exists() else "NOT_STARTED",
            "feature_plan": PLAN.exists(), "manifest": MANIFEST.exists(),
            "candidate_trials": len(list((OUT / "trials").glob("*.json"))),
            "validation_refits": len(list((OUT / "validation").glob("*.json"))),
            "test_model_records": len(list((OUT / "test_models").glob("*.json"))),
            "zero_mask_records": len(list((OUT / "zero_masks").glob("*.json"))),
            "fingerprint": read_json(MANIFEST)["fingerprint"] if MANIFEST.exists() else None}


def smoke() -> dict:
    """Fit tiny training-only examples, including one vertical neural model."""
    plan, _, _ = feature_screen()
    frame, _, fit, _ = training_frame(plan)
    tiny_fit, tiny_hold = fit[:120], fit[120:160]
    result = {}
    for name in ("xgboost", "transformer"):
        xf, xh, imputer, scaler, raw_hold = processed(
            frame, tiny_fit, tiny_hold, plan, name)
        config = dict(c.candidate_configs(name)[0])
        config["n_estimators" if name == "xgboost" else "epochs"] = 4
        yfit = frame.loc[tiny_fit, c.TARGET].to_numpy(dtype=float)
        yhold = frame.loc[tiny_hold, c.TARGET].to_numpy(dtype=float)
        model, curve, prediction, _ = c.fit_logged(
            name, config, 17, tuple(range(10)), xf, yfit, xh, yhold)
        require(len(prediction) == 40 and len(curve) == 4,
                f"Tiny model fitting failed: {name}")
        packed = pickle.loads(pickle.dumps({"model": model, "imputer": imputer,
                                            "scaler": scaler}))
        transformed = packed["imputer"].transform(raw_hold)
        if packed["scaler"] is not None:
            transformed = packed["scaler"].transform(transformed)
        require(len(packed["model"].predict(transformed)) == 40,
                f"Fitted-model serialization failed: {name}")
        result[name] = {"input_columns": int(xf.shape[1]), "rounds": len(curve),
                        "holdout_rmse_db": c.rmse(yhold, prediction)}
    return {"status": "SMOKE_OK", "training_only": True, "models": result}


def main() -> None:
    commands = ("prepare", "verify-plan", "train", "verify-train", "test-models",
                "verify-test", "zero-masks", "verify-all", "report", "status", "smoke")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    args = parser.parse_args()
    action = args.command
    result = {"prepare": prepare, "verify-plan": verify_plan,
              "train": train, "verify-train": verify_train,
              "test-models": test_models, "verify-test": verify_test,
              "zero-masks": zero_masks, "verify-all": verify_all,
              "report": report, "status": status, "smoke": smoke}[action]()
    if result is not None:
        print(json.dumps(str(result) if isinstance(result, Path) else result,
                         ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
