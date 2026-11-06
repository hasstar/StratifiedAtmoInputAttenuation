"""Independently audit all forty validation curves, refits, and ten-model ranking."""

from __future__ import annotations

import json
import math

import numpy as np

import common as c
import run_selection as code


def close(a, b, tol: float = 1e-8) -> None:
    c.require(abs(float(a) - float(b)) <= tol, f"Mismatch: {a} vs {b}")


def check_metrics(frame, rows, pred, saved):
    computed = c.metrics(frame, rows, pred)
    for group in c.GROUPS:
        c.require(computed[group]["n"] == saved[group]["n"] and
                  computed[group]["days"] == saved[group]["days"], "Event counts changed")
        for metric in ("rmse", "mae", "bias"):
            if computed[group][metric] is None:
                c.require(saved[group][metric] is None, "Empty metric mismatch")
            else:
                close(computed[group][metric], saved[group][metric])


def main() -> None:
    frame, names, folds = c.load_training()
    candidate = code.manifest_candidate(frame, names, folds)
    manifest = json.loads((c.OUT / "run_manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest["fingerprint"]
    c.require(fingerprint == candidate["fingerprint"], "Source/data/protocol fingerprint changed")
    c.require(not (c.OUT / "layers" / "final_test.json").exists(),
              "Selection QA ran after final test")
    valid = np.flatnonzero(frame.split.to_numpy() == "validation")
    truth = frame.loc[valid, c.TARGET].to_numpy(dtype=float)
    trials_n, seeds_n = 0, 0
    for name in c.MODEL_NAMES:
        trials = []
        for config_id, config in enumerate(c.candidate_configs(name)):
            item = c.read_record(c.OUT / "trials" / f"{name}_{config_id:02d}.json", fingerprint)
            c.require(item is not None and item["model"] == name and
                      item["config_id"] == config_id and item["config"] == config and
                      item["seed"] == 17 and item["n_train"] == 2404 and
                      item["n_validation"] == 343, "Missing/corrupt candidate fit")
            y = np.asarray(item["validation_truth_db"], dtype=float)
            p = np.asarray(item["validation_prediction_db"], dtype=float)
            c.require(np.array_equal(y, truth) and len(p) == 343 and np.isfinite(p).all(),
                      "Candidate prediction truth/order/finite mismatch")
            history = item["round_history"]
            c.require(len(history) >= 1 and all(math.isfinite(row["train_rmse_db"])
                                                for row in history), "Candidate curve invalid")
            if name in ("elastic_net", "svr_rbf"):
                c.require(len(history) == 1 and item["formal_rounds"] is None,
                          "Non-epoch model has fake rounds")
            else:
                best = min(history, key=lambda row: (row["holdout_rmse_db"], row["round"]))
                c.require(item["formal_rounds"] == item["selected_round"] == best["round"],
                          "Validation-stopped round mismatch")
                c.require(item["formal_rounds_reason"] ==
                          "lowest formal-validation RMSE, earliest tie",
                          "Authoritative stopping reason changed")
                close(best["holdout_rmse_db"], c.rmse(y, p), 1e-5)
            check_metrics(frame, valid, p, item["validation_metrics"])
            trials.append(item)
            trials_n += 1
        best = min(trials, key=lambda row: (row["validation_metrics"]["overall"]["rmse"],
                                            row["validation_metrics"]["overall"]["mae"],
                                            row["config_id"]))
        for seed in c.SEEDS:
            item = c.read_record(c.OUT / "validation" / f"{name}_seed{seed}.json", fingerprint)
            c.require(item is not None and item["model"] == name and
                      item["seed"] == seed and item["config_id"] == best["config_id"] and
                      item["config"] == best["config"] and
                      item["formal_rounds"] == best["formal_rounds"] and
                      item["n_train"] == 2404 and item["n_validation"] == 343,
                      "Wrong fixed-round formal refit")
            y = np.asarray(item["validation_truth_db"], dtype=float)
            p = np.asarray(item["validation_prediction_db"], dtype=float)
            c.require(np.array_equal(y, truth) and len(p) == 343 and np.isfinite(p).all(),
                      "Formal prediction invalid")
            if best["formal_rounds"] is not None:
                c.require(len(item["round_history"]) == best["formal_rounds"],
                          "Formal fit ended at wrong round")
            check_metrics(frame, valid, p, item["metrics"])
            samples = np.asarray(item["latency"]["end_to_end_ns"], dtype=float)
            c.require(len(samples) == 300 and np.isfinite(samples).all() and
                      (samples > 0).all(), "Single-row latency samples invalid")
            close(np.median(samples) / 1e6, item["latency"]["median_ms"])
            close(np.percentile(samples, 95) / 1e6, item["latency"]["p95_ms"])
            seeds_n += 1
    choice = c.read_record(c.OUT / "model_selection.json", fingerprint)
    c.require(choice is not None and choice["status"] == "TEN_MODELS_VALIDATION_SELECTED" and
              len(choice["model_rows"]) == 10, "Model ranking missing")
    rows = {row["name"]: row for row in choice["model_rows"]}
    c.require(len(rows) == 10, "Duplicate ranked model")
    for name in c.MODEL_NAMES:
        parts = [c.read_record(c.OUT / "validation" / f"{name}_seed{seed}.json", fingerprint)
                 for seed in c.SEEDS]
        close(rows[name]["overall_rmse_mean"],
              np.mean([part["metrics"]["overall"]["rmse"] for part in parts]))
        close(rows[name]["latency_median_ms"],
              np.median([part["latency"]["median_ms"] for part in parts]))
    ranked, summary = c.cleaned.rank_models([dict(rows[name]) for name in c.MODEL_NAMES])
    c.require(summary["primary_winner"] == choice["primary_winner"] and
              summary["rmse_only_winner"] == choice["rmse_only_winner"] and
              [row["name"] for row in ranked] == [row["name"] for row in choice["model_rows"]],
              "Composite ranking mismatch")
    c.require((trials_n, seeds_n) == (40, 30), "Ten-model count mismatch")
    report = {"status": "PASS", "fingerprint": fingerprint,
              "candidate_fits": trials_n, "fixed_round_seed_refits": seeds_n,
              "winner": choice["primary_winner"], "metrics_recomputed": True,
              "test_not_read": True, "validated_utc": c.prior.utc_now()}
    c.prior.atomic_json(c.OUT / "model_selection_qa.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
