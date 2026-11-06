"""Frozen validation fit and final test-only assessment for Revision 022.

Feature choices are loaded from PASSed training-only selection checkpoints. The
test split is not loaded by prepare, verify-prepare, fit or verify-fit.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

import run_groups as g
import run_columns as c

HERE = Path(__file__).resolve().parent
OUT = g.OUT
PLAN = OUT / "final_plan.json"
MANIFEST = OUT / "final_manifest.json"
PLAN_QA = OUT / "final_plan_qa.json"
FIT_QA = OUT / "final_fit_qa.json"
TEST_QA = OUT / "final_test_qa.json"
SUMMARY = OUT / "final_summary.json"
REPORT = OUT / "FINAL_REPORT.md"
SEEDS = g.FOLD_SEEDS
VERSION = "revision022-final-validation-test-v1-20261002"


@contextmanager
def writer_lock(stage: str):
    path = OUT / f"final_{stage}_worker.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "utc": g.utc_now()}))
    except FileExistsError as exc:
        raise RuntimeError(f"Another/stale worker lock exists: {path}") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def plans() -> dict:
    group_plan, group_manifest = g.checked()
    column_plan, column_manifest, _ = c.checked()
    g.require(g.GROUP_QA.exists() and g.read_json(g.GROUP_QA)["status"] == "PASS" and
              c.SELECT_QA.exists() and g.read_json(c.SELECT_QA)["status"] == "PASS",
              "All feature-selection QA gates must PASS")
    base_choice = g.read_json(c.decision_path("base"))
    group_choice = g.read_json(c.decision_path("group_selected"))
    base = column_plan["pools"]["base"]
    selected_group = column_plan["pools"]["group_selected"]
    all_eligible = group_plan["eligible_feature_names"]
    smaller_k = min(base_choice["selected_k"], group_choice["selected_k"])
    # The group-stage choice is base-only in this run. Record semantic aliases
    # rather than refitting exact duplicates and pretending they are independent.
    sets = {"all_eligible_full": all_eligible,
            "base_full": base,
            "group_full": selected_group,
            "base_selected": base_choice["selected_feature_names"],
            "group_selected": group_choice["selected_feature_names"],
            "matched_base": g.read_json(c.state_path("base", smaller_k))["feature_names"],
            "matched_group": g.read_json(c.state_path(
                column_plan["group_selected_alias_of"] or "group_selected",
                smaller_k))["feature_names"]}
    unique: dict[str, list[str]] = {}
    aliases = {}
    for name, features in sets.items():
        source = next((key for key, old in unique.items() if old == features), None)
        if source is None:
            unique[name] = features
        else:
            aliases[name] = source
    g.require(all(0 < len(features) <= len(all_eligible) and
                  len(features) == len(set(features)) and
                  set(features).issubset(all_eligible)
                  for features in sets.values()), "Final feature sets invalid")
    return {"sets": sets, "unique_sets": unique, "aliases": aliases,
            "smaller_selected_k": smaller_k,
            "group_selected": g.read_json(g.SELECTION)["selected_groups"],
            "group_fingerprint": group_manifest["fingerprint"],
            "column_fingerprint": column_manifest["fingerprint"],
            "config": g.CONFIG, "seeds": list(SEEDS),
            "validation_role": "best_round_and_early_stopping_only",
            "test_role": "final_metrics_only_no_selection",
            "target": g.TARGET,
            "historically_accessed_test": True}


def manifest_candidate(plan: dict) -> dict:
    sources = [HERE / "PROTOCOL.md", HERE / "run_final.py",
               g.MANIFEST, g.GROUP_QA, g.SELECTION,
               c.MANIFEST, c.SELECT_QA,
               c.decision_path("base"), c.decision_path("group_selected"),
               g.DATA / "validation.parquet", g.DATA / "train.parquet",
               g.DATA / "hourly_sample_status.parquet"]
    payload = {"version": VERSION, "plan_sha256": g.content_hash(plan),
               "source_hashes": {str(p.relative_to(g.ROOT)): g.sha256(p)
                                 for p in sources},
               "test_values_read_for_freeze": False,
               "no_test_driven_selection": True}
    payload["fingerprint"] = g.content_hash(payload)
    return payload


def prepare() -> dict:
    plan = plans()
    frozen = manifest_candidate(plan)
    if PLAN.exists() or MANIFEST.exists():
        g.require(PLAN.exists() and MANIFEST.exists() and
                  g.content_hash(g.read_json(PLAN)) == g.content_hash(plan) and
                  g.read_json(MANIFEST)["fingerprint"] == frozen["fingerprint"],
                  "Frozen final plan differs; new revision needed")
    else:
        g.atomic_json(PLAN, plan)
        frozen["created_utc"] = g.utc_now()
        g.atomic_json(MANIFEST, frozen)
    return {"fingerprint": frozen["fingerprint"],
            "unique_set_sizes": {k: len(v) for k, v in plan["unique_sets"].items()},
            "aliases": plan["aliases"]}


def checked() -> tuple[dict, dict]:
    g.require(PLAN.exists() and MANIFEST.exists(), "Run final prepare first")
    plan, saved = g.read_json(PLAN), g.read_json(MANIFEST)
    g.require(g.content_hash(plan) == saved["plan_sha256"] and
              saved["fingerprint"] == manifest_candidate(plans())["fingerprint"],
              "Final plan/source fingerprint changed")
    return plan, saved


def load_split(split: str, names: list[str]) -> pd.DataFrame:
    g.require(split in ("validation", "test"), "Invalid formal split")
    frame = pd.read_parquet(g.DATA / f"{split}.parquet")
    expected = 343 if split == "validation" else 687
    g.require(len(frame) == expected and
              list(frame.columns) == ["hour_bjt", "hour_utc", *names, g.TARGET] and
              frame["hour_bjt"].is_unique and
              np.isfinite(frame[g.TARGET].to_numpy(dtype=float)).all(),
              f"Formal {split} split/schema changed")
    metadata = pd.read_parquet(g.DATA / "hourly_sample_status.parquet",
                               columns=["hour_bjt", "split", "use_for_regression",
                                        "rain", "heavy_rain", "dry_overcast", "clear_dry"],
                               filters=[("split", "==", split)])
    g.require(len(metadata) == expected and metadata["hour_bjt"].is_unique,
              f"Formal {split} event metadata changed")
    frame = frame.merge(metadata, on="hour_bjt", how="left", validate="one_to_one")
    frame = frame.sort_values("hour_bjt").reset_index(drop=True)
    g.require(frame["split"].eq(split).all() and
              frame["use_for_regression"].eq(True).all() and
              frame[["rain", "heavy_rain", "dry_overcast", "clear_dry"]].notna().all().all(),
              f"Formal {split} event join invalid")
    return frame


def verify_prepare() -> dict:
    plan, manifest = checked()
    train, original = g.load_training()
    validation = load_split("validation", original)
    g.require(len(train) == 2404 and len(validation) == 343 and
              set(train["hour_bjt"]).isdisjoint(set(validation["hour_bjt"])) and
              int(validation["rain"].sum()) == 25 and
              int(validation["dry_overcast"].sum()) == 79 and
              plan["seeds"] == list(SEEDS) and plan["config"] == g.CONFIG and
              all(features == plan["sets"][name]
                  for name, features in plan["unique_sets"].items()) and
              all(plan["sets"][name] == plan["sets"][source]
                  for name, source in plan["aliases"].items()),
              "Final plan, training, validation or alias QA failed")
    qa = {"status": "PASS", "stage": "final_plan",
          "fingerprint": manifest["fingerprint"],
          "unique_set_sizes": {k: len(v) for k, v in plan["unique_sets"].items()},
          "aliases": plan["aliases"],
          "n_train": len(train), "n_validation": len(validation),
          "test_target_read": False, "verified_utc": g.utc_now()}
    g.atomic_json(PLAN_QA, qa)
    return qa


def model_path(name: str, seed: int) -> Path:
    return OUT / "final_models" / f"{name}_seed{seed}.json"


def fit_path(name: str, seed: int) -> Path:
    return OUT / "final_fit" / f"{name}_seed{seed}.json"


def test_path(name: str, seed: int) -> Path:
    return OUT / "final_test" / f"{name}_seed{seed}.json"


def raw(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = frame.loc[:, names].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def transform(raw_values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    values = raw_values.copy()
    mask = np.isnan(values)
    if mask.any():
        values[mask] = np.broadcast_to(medians, values.shape)[mask]
    g.require(np.isfinite(values).all(), "Nonfinite model input after training imputation")
    return values


def score(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return g.metric_record(frame, np.arange(len(frame), dtype=int), prediction)


def save_model(model: XGBRegressor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    model.save_model(str(tmp))
    os.replace(tmp, path)
    return g.sha256(path)


def fit() -> None:
    plan, manifest = checked()
    g.require(PLAN_QA.exists() and g.read_json(PLAN_QA)["status"] == "PASS",
              "Final plan QA must PASS")
    train, original = g.load_training()
    validation = load_split("validation", original)
    ytrain = train[g.TARGET].to_numpy(dtype=float)
    yval = validation[g.TARGET].to_numpy(dtype=float)
    with writer_lock("fit"):
        for name, features in plan["unique_sets"].items():
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            xtrain = imputer.fit_transform(raw(train, features))
            xval = imputer.transform(raw(validation, features))
            g.require(xtrain.shape == (2404, len(features)) and
                      xval.shape == (343, len(features)) and
                      np.isfinite(xtrain).all() and np.isfinite(xval).all(),
                      f"Train-only imputation invalid for {name}")
            medians = np.asarray(imputer.statistics_, dtype=float)
            for seed in SEEDS:
                record_file, model_file = fit_path(name, seed), model_path(name, seed)
                if record_file.exists():
                    old = g.read_json(record_file)
                    g.require(old["fingerprint"] == manifest["fingerprint"] and
                              old["features"] == features and old["seed"] == seed and
                              model_file.exists() and
                              old["model_sha256"] == g.sha256(model_file),
                              f"Existing final-fit checkpoint changed: {name}/{seed}")
                    continue
                model = XGBRegressor(**g.CONFIG, random_state=seed)
                model.fit(xtrain, ytrain,
                          eval_set=[(xtrain, ytrain), (xval, yval)], verbose=False)
                curve = model.evals_result()
                best_round = int(model.best_iteration) + 1
                pred_train = np.asarray(model.predict(
                    xtrain, iteration_range=(0, best_round)), dtype=float)
                pred_val = np.asarray(model.predict(
                    xval, iteration_range=(0, best_round)), dtype=float)
                val_metrics = score(validation, pred_val)
                g.require(abs(val_metrics["overall"]["rmse"] -
                              float(curve["validation_1"]["rmse"][best_round - 1])) < 1e-5,
                          f"Final validation round mismatch: {name}/{seed}")
                model_digest = save_model(model, model_file)
                row = {"fingerprint": manifest["fingerprint"], "name": name,
                       "features": features, "n_features": len(features),
                       "seed": seed, "config": g.CONFIG,
                       "n_train": len(train), "n_validation": len(validation),
                       "imputer_medians_training_only": medians.tolist(),
                       "best_round": best_round,
                       "trained_rounds": len(curve["validation_1"]["rmse"]),
                       "stop_reason": "validation_patience_30" if
                       len(curve["validation_1"]["rmse"]) < 300 else "round_cap_300",
                       "train_rmse_by_round": curve["validation_0"]["rmse"],
                       "validation_rmse_by_round": curve["validation_1"]["rmse"],
                       "train_prediction_db": pred_train.tolist(),
                       "validation_prediction_db": pred_val.tolist(),
                       "train_metrics": score(train, pred_train),
                       "validation_metrics": val_metrics,
                       "model_sha256": model_digest, "completed_utc": g.utc_now()}
                g.atomic_json(record_file, row)
                print(f"FIT {name} seed={seed} n={len(features)} "
                      f"val_rmse={val_metrics['overall']['rmse']:.6f} "
                      f"best_round={best_round}", flush=True)
        print("FINAL_FITS_COMPLETE", flush=True)


def verify_fit() -> dict:
    plan, manifest = checked()
    g.require(PLAN_QA.exists() and g.read_json(PLAN_QA)["status"] == "PASS",
              "Final plan QA missing")
    train, original = g.load_training()
    validation = load_split("validation", original)
    expected_records = len(plan["unique_sets"]) * len(SEEDS)
    g.require(len(list((OUT / "final_fit").glob("*.json"))) == expected_records and
              len(list((OUT / "final_models").glob("*.json"))) == expected_records,
              "Final fit/model artifact count mismatch")
    for name, features in plan["unique_sets"].items():
        raw_train, raw_val = raw(train, features), raw(validation, features)
        expected_medians = SimpleImputer(strategy="median",
                                         keep_empty_features=True).fit(raw_train).statistics_
        for seed in SEEDS:
            row = g.read_json(fit_path(name, seed))
            model_file = model_path(name, seed)
            g.require(row["fingerprint"] == manifest["fingerprint"] and
                      row["name"] == name and row["features"] == features and
                      row["seed"] == seed and row["n_train"] == 2404 and
                      row["n_validation"] == 343 and row["config"] == g.CONFIG and
                      model_file.exists() and g.sha256(model_file) == row["model_sha256"] and
                      1 <= row["best_round"] <= row["trained_rounds"] <= 300 and
                      len(row["train_rmse_by_round"]) ==
                      len(row["validation_rmse_by_round"]) == row["trained_rounds"],
                      f"Final fit identity/curve invalid {name}/{seed}")
            medians = np.asarray(row["imputer_medians_training_only"], dtype=float)
            g.require(np.allclose(medians, expected_medians, atol=0, rtol=0),
                      f"Imputation leaked non-training values {name}/{seed}")
            model = XGBRegressor()
            model.load_model(str(model_file))
            for frame, values, saved_key, metrics_key in (
                    (train, raw_train, "train_prediction_db", "train_metrics"),
                    (validation, raw_val, "validation_prediction_db", "validation_metrics")):
                pred = np.asarray(model.predict(
                    transform(values, medians),
                    iteration_range=(0, row["best_round"])), dtype=float)
                saved = np.asarray(row[saved_key], dtype=float)
                replay = score(frame, pred)
                g.require(len(saved) == len(pred) and
                          np.allclose(pred, saved, atol=1e-9, rtol=0) and
                          all(abs(replay[event][metric] -
                                  row[metrics_key][event][metric]) < 1e-9
                              for event in ("overall", "rain", "dry_overcast")
                              for metric in ("rmse", "mae")),
                          f"Final fit prediction/metric replay failed {name}/{seed}")
    qa = {"status": "PASS", "stage": "final_validation_fit",
          "fingerprint": manifest["fingerprint"],
          "unique_feature_sets": len(plan["unique_sets"]),
          "model_fits": expected_records,
          "n_train": len(train), "n_validation": len(validation),
          "test_target_read": False, "verified_utc": g.utc_now()}
    g.atomic_json(FIT_QA, qa)
    return qa


def test() -> None:
    plan, manifest = checked()
    g.require(PLAN_QA.exists() and g.read_json(PLAN_QA)["status"] == "PASS" and
              FIT_QA.exists() and g.read_json(FIT_QA)["status"] == "PASS" and
              g.read_json(FIT_QA)["fingerprint"] == manifest["fingerprint"],
              "Frozen plan and final-fit QA must PASS before opening test")
    original = g.read_json(g.PLAN)["original_feature_names"]
    frame = load_split("test", original)
    g.require(int(frame["rain"].sum()) == 51 and
              int(frame["dry_overcast"].sum()) == 158,
              "Formal test event counts changed")
    with writer_lock("test"):
        for name, features in plan["unique_sets"].items():
            raw_test = raw(frame, features)
            for seed in SEEDS:
                path = test_path(name, seed)
                if path.exists():
                    old = g.read_json(path)
                    g.require(old["fingerprint"] == manifest["fingerprint"] and
                              old["name"] == name and old["seed"] == seed and
                              old["features"] == features,
                              f"Test checkpoint mismatch {name}/{seed}")
                    continue
                fitted = g.read_json(fit_path(name, seed))
                model_file = model_path(name, seed)
                g.require(fitted["model_sha256"] == g.sha256(model_file),
                          f"Frozen model file changed {name}/{seed}")
                model = XGBRegressor()
                model.load_model(str(model_file))
                medians = np.asarray(fitted["imputer_medians_training_only"], dtype=float)
                pred = np.asarray(model.predict(
                    transform(raw_test, medians),
                    iteration_range=(0, fitted["best_round"])), dtype=float)
                row = {"fingerprint": manifest["fingerprint"], "name": name,
                       "features": features, "seed": seed,
                       "model_sha256": fitted["model_sha256"],
                       "best_round": fitted["best_round"],
                       "n_test": len(frame),
                       "test_hour_bjt": frame["hour_bjt"].astype(str).tolist(),
                       "test_prediction_db": pred.tolist(),
                       "test_metrics": score(frame, pred),
                       "completed_utc": g.utc_now()}
                g.atomic_json(path, row)
                print(f"TEST {name} seed={seed} "
                      f"rmse={row['test_metrics']['overall']['rmse']:.6f}", flush=True)
        print("FINAL_TEST_SCORING_COMPLETE", flush=True)


def event_mask(frame: pd.DataFrame, event: str) -> np.ndarray:
    return (np.ones(len(frame), dtype=bool) if event == "overall" else
            frame[event].to_numpy(dtype=bool))


def bootstrap_improvement(frame: pd.DataFrame,
                          baseline: list[np.ndarray], candidate: list[np.ndarray],
                          event: str, seed: int) -> dict:
    mask = event_mask(frame, event)
    truth = frame.loc[mask, g.TARGET].to_numpy(dtype=float)
    dates = frame.loc[mask, "hour_bjt"].dt.strftime("%Y-%m-%d").to_numpy()
    days = np.unique(dates)
    g.require(len(days) > 0, f"Empty event for bootstrap: {event}")
    positions = [np.flatnonzero(dates == day) for day in days]
    base = np.stack([arr[mask] for arr in baseline])
    cand = np.stack([arr[mask] for arr in candidate])

    def improvement(pos: np.ndarray) -> float:
        base_rmse = np.sqrt(np.mean((base[:, pos] - truth[pos]) ** 2, axis=1)).mean()
        cand_rmse = np.sqrt(np.mean((cand[:, pos] - truth[pos]) ** 2, axis=1)).mean()
        return float(base_rmse - cand_rmse)

    observed = improvement(np.arange(len(truth)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        sampled = rng.integers(0, len(days), size=len(days))
        draws.append(improvement(np.concatenate([positions[i] for i in sampled])))
    low, high = np.quantile(draws, (0.025, 0.975))
    return {"event": event, "n_hours": len(truth), "n_days": len(days),
            "positive_means_candidate_better": True,
            "baseline_minus_candidate_rmse_mean_seed_db": observed,
            "paired_day_bootstrap_95pct_db": [float(low), float(high)],
            "bootstrap_replicates": len(draws), "seed": seed}


def build_summary(frame: pd.DataFrame, plan: dict, fp: str) -> dict:
    predictions: dict[str, list[np.ndarray]] = {}
    rows = []
    for name, features in plan["unique_sets"].items():
        records = [g.read_json(test_path(name, seed)) for seed in SEEDS]
        g.require(all(row["fingerprint"] == fp and row["features"] == features
                      for row in records), f"Missing/changed test records for {name}")
        predictions[name] = [np.asarray(row["test_prediction_db"], dtype=float)
                             for row in records]
        by_event = {}
        for event in ("overall", "rain", "dry_overcast", "clear_dry", "heavy_rain"):
            by_event[event] = {"n": records[0]["test_metrics"][event]["n"]}
            for metric in ("rmse", "mae"):
                values = [row["test_metrics"][event][metric] for row in records]
                by_event[event][metric + "_mean_seed"] = (
                    None if any(value is None for value in values) else
                    float(np.mean(values)))
                by_event[event][metric + "_sd_seed"] = (
                    None if any(value is None for value in values) else
                    float(np.std(values, ddof=1)))
        rows.append({"name": name, "n_features": len(features),
                     "feature_names": features, "metrics": by_event,
                     "validation_best_rounds":
                     [g.read_json(fit_path(name, seed))["best_round"] for seed in SEEDS]})
    comparisons = []
    for baseline_name, candidate_name in (("base_full", "all_eligible_full"),
                                          ("base_full", "base_selected")):
        for event_id, event in enumerate(("overall", "rain", "dry_overcast")):
            comparisons.append({"baseline": baseline_name,
                                "candidate": candidate_name,
                                **bootstrap_improvement(
                                    frame, predictions[baseline_name],
                                    predictions[candidate_name], event,
                                    22000 + 100 * len(comparisons) + event_id)})
    return {"fingerprint": fp, "status": "FINAL_TEST_EVALUATED",
            "historically_accessed_test": True,
            "test_not_used_to_choose_features_or_rounds": True,
            "n_train": 2404, "n_validation": 343, "n_test": len(frame),
            "group_selected": plan["group_selected"],
            "set_aliases": plan["aliases"],
            "model_rows": rows,
            "comparisons": comparisons,
            "completed_utc": g.utc_now()}


def verify_test() -> dict:
    plan, manifest = checked()
    g.require(FIT_QA.exists() and g.read_json(FIT_QA)["status"] == "PASS",
              "Final-fit QA missing")
    original = g.read_json(g.PLAN)["original_feature_names"]
    frame = load_split("test", original)
    expected = len(plan["unique_sets"]) * len(SEEDS)
    g.require(len(list((OUT / "final_test").glob("*.json"))) == expected,
              "Test result count mismatch")
    for name, features in plan["unique_sets"].items():
        for seed in SEEDS:
            row = g.read_json(test_path(name, seed))
            fit = g.read_json(fit_path(name, seed))
            pred = np.asarray(row["test_prediction_db"], dtype=float)
            replay = score(frame, pred)
            g.require(row["fingerprint"] == manifest["fingerprint"] and
                      row["name"] == name and row["features"] == features and
                      row["seed"] == seed and row["n_test"] == len(frame) and
                      row["model_sha256"] == fit["model_sha256"] and
                      row["best_round"] == fit["best_round"] and
                      row["test_hour_bjt"] == frame["hour_bjt"].astype(str).tolist() and
                      len(pred) == len(frame) and np.isfinite(pred).all() and
                      all(abs(replay[event][metric] -
                              row["test_metrics"][event][metric]) < 1e-10
                          for event in ("overall", "rain", "dry_overcast")
                          for metric in ("rmse", "mae")),
                      f"Test record replay failed {name}/{seed}")
            model = XGBRegressor()
            model.load_model(str(model_path(name, seed)))
            medians = np.asarray(fit["imputer_medians_training_only"], dtype=float)
            model_pred = np.asarray(model.predict(
                transform(raw(frame, features), medians),
                iteration_range=(0, fit["best_round"])), dtype=float)
            g.require(np.allclose(model_pred, pred, atol=1e-9, rtol=0),
                      f"Frozen-model test prediction mismatch {name}/{seed}")
    summary = build_summary(frame, plan, manifest["fingerprint"])
    if SUMMARY.exists():
        old = g.read_json(SUMMARY)
        g.require(old["fingerprint"] == summary["fingerprint"] and
                  old["model_rows"] == summary["model_rows"] and
                  old["comparisons"] == summary["comparisons"],
                  "Final summary changed unexpectedly")
    else:
        g.atomic_json(SUMMARY, summary)
    qa = {"status": "PASS", "stage": "final_test",
          "fingerprint": manifest["fingerprint"],
          "test_records": expected, "n_test": len(frame),
          "no_retraining_on_test": True,
          "selected_features_not_changed_after_test": True,
          "verified_utc": g.utc_now()}
    g.atomic_json(TEST_QA, qa)
    return qa


def report() -> dict:
    plan, manifest = checked()
    g.require(TEST_QA.exists() and g.read_json(TEST_QA)["status"] == "PASS" and
              g.read_json(TEST_QA)["fingerprint"] == manifest["fingerprint"],
              "Final test QA must PASS before reporting")
    summary = g.read_json(SUMMARY)
    decision = g.read_json(c.decision_path("base"))
    lines = ["# Revision 022: unconstrained meteorological selection",
             "", "## Scope and result status", "",
             "Training-only feature selection and the frozen final assessment are complete.",
             "The same local test set was accessed in earlier experiments; these are not new blind-test results.",
             "No pressure-level quota or desired final feature count was imposed.", "",
             f"Group-stage one-SE selection: `{', '.join(plan['group_selected'])}`.",
             f"The minimum-score group-path state was mask {g.read_json(g.SELECTION)['minimum_score_mask']}, "
             "which is not the one-SE selected state.",
             f"Column one-SE selection: {decision['selected_k']} features; "
             f"minimum-score path state: {decision['minimum_k']} features.",
             "The group-selected and base paths were identical, so duplicate model fits were not fabricated.",
             "", "## Selected features", ""]
    lines.extend(f"- `{name}`" for name in decision["selected_feature_names"])
    lines.extend(["", "## Formal test metrics (three-seed means)", "",
                  "| Frozen input set | Features | Overall RMSE | Rain RMSE | Dry-overcast RMSE | Best rounds |",
                  "|---|---:|---:|---:|---:|---|"])
    for row in summary["model_rows"]:
        metric = row["metrics"]
        lines.append(f"| {row['name']} | {row['n_features']} | "
                     f"{metric['overall']['rmse_mean_seed']:.4f} | "
                     f"{metric['rain']['rmse_mean_seed']:.4f} | "
                     f"{metric['dry_overcast']['rmse_mean_seed']:.4f} | "
                     f"{row['validation_best_rounds']} |")
    lines.extend(["", "All RMSE values are in dB; rain includes heavy rain. "
                  "Full-pool (all eligible vs base) comparisons have unequal feature budgets; "
                  "the naturally selected group and base subsets are identical, so a "
                  "nontrivial matched-budget vertical comparison is unavailable in this run.",
                  "", "## Paired day-block uncertainty", "",
                  "Positive improvement means the candidate has lower RMSE than the baseline.", "",
                  "| Baseline → Candidate | Event | Improvement (dB) | 95% day-bootstrap interval (dB) | Days |",
                  "|---|---|---:|---|---:|"])
    for item in summary["comparisons"]:
        ci = item["paired_day_bootstrap_95pct_db"]
        lines.append(f"| {item['baseline']} → {item['candidate']} | {item['event']} | "
                     f"{item['baseline_minus_candidate_rmse_mean_seed_db']:+.4f} | "
                     f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {item['n_days']} |")
    lines.extend(["", "## Audit boundary", "",
                  "Feature eligibility, group and column paths used the 2404 training hours only. "
                  "The 343 validation hours set the final boosting rounds. "
                  "The 687 test hours only produced frozen-model metrics and uncertainty intervals. "
                  "Event flags were never model inputs. The external stations and 100 ms fast component "
                  "were not included.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "PASS", "report": str(REPORT),
            "model_rows": len(summary["model_rows"]),
            "comparisons": len(summary["comparisons"])}


def status() -> dict:
    plan = g.read_json(PLAN) if PLAN.exists() else None
    return {"manifest": MANIFEST.exists(),
            "plan_qa": g.read_json(PLAN_QA).get("status") if PLAN_QA.exists() else None,
            "expected_unique_sets": len(plan["unique_sets"]) if plan else None,
            "fit_records": len(list((OUT / "final_fit").glob("*.json"))),
            "fit_qa": g.read_json(FIT_QA).get("status") if FIT_QA.exists() else None,
            "test_records": len(list((OUT / "final_test").glob("*.json"))),
            "test_qa": g.read_json(TEST_QA).get("status") if TEST_QA.exists() else None,
            "report": REPORT.exists(),
            "fit_worker_lock": (OUT / "final_fit_worker.lock").exists(),
            "test_worker_lock": (OUT / "final_test_worker.lock").exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify-prepare", "fit",
                                            "verify-fit", "test", "verify-test",
                                            "report", "status"))
    args = parser.parse_args()
    action = {"prepare": prepare, "verify-prepare": verify_prepare,
              "fit": fit, "verify-fit": verify_fit, "test": test,
              "verify-test": verify_test, "report": report,
              "status": status}[args.command]
    result = action()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
