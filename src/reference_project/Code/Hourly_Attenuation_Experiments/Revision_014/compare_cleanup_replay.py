"""One-time numerical equivalence check before removing obsolete experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import support


OLD = support.PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_013_ValidationStopped"
NEW = support.PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_014_SelfContained"
TOLERANCE_DB = 1e-5
MODELS = ("elastic_net", "svr_rbf", "random_forest", "xgboost", "lightgbm",
          "catboost", "mlp", "transformer", "gcn", "bilstm")


def read(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def max_difference(a: object, b: object) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("Prediction shapes or finite status differ")
    return float(np.max(np.abs(left - right)))


def require_close(value: float, label: str) -> None:
    if value > TOLERANCE_DB:
        raise RuntimeError(f"{label} differs by {value:.9g} dB")


def main() -> None:
    for root in (OLD, NEW):
        for relative in ("model_selection_qa.json", "layers/validation_qa.json",
                         "layers/final_qa.json"):
            qa = read(root, relative)
            if qa["status"] != "PASS":
                raise RuntimeError(f"QA not PASS: {root / relative}")
    old_selection = read(OLD, "model_selection.json")
    new_selection = read(NEW, "model_selection.json")
    if old_selection["primary_winner"] != new_selection["primary_winner"]:
        raise RuntimeError("Winner changed in self-contained replay")
    old_rows = {row["name"]: row for row in old_selection["model_rows"]}
    new_rows = {row["name"]: row for row in new_selection["model_rows"]}
    trial_max = 0.0
    formal_max = 0.0
    for model in MODELS:
        for key in ("best_config_id", "best_config", "formal_rounds"):
            if old_rows[model][key] != new_rows[model][key]:
                raise RuntimeError(f"Selected {key} changed for {model}")
        for config_id in range(4):
            relative = f"trials/{model}_{config_id:02d}.json"
            old = read(OLD, relative)
            new = read(NEW, relative)
            if old["formal_rounds"] != new["formal_rounds"]:
                raise RuntimeError(f"Candidate round changed: {relative}")
            difference = max_difference(old["validation_prediction_db"],
                                        new["validation_prediction_db"])
            require_close(difference, relative)
            trial_max = max(trial_max, difference)
        for seed in (17, 29, 43):
            relative = f"validation/{model}_seed{seed}.json"
            old = read(OLD, relative)
            new = read(NEW, relative)
            difference = max_difference(old["validation_prediction_db"],
                                        new["validation_prediction_db"])
            require_close(difference, relative)
            formal_max = max(formal_max, difference)

    old_layers = read(OLD, "layers/layer_selection.json")
    new_layers = read(NEW, "layers/layer_selection.json")
    if old_layers["selected_mask_train_only"] != new_layers["selected_mask_train_only"]:
        raise RuntimeError("Selected pressure-layer mask changed")
    mask_max = 0.0
    for mask in range(1024):
        relative = f"layers/subsets/mask_{mask:04d}.json"
        old = read(OLD, relative)
        new = read(NEW, relative)
        for key in ("features", "alpha", "formal_rounds", "levels_hpa"):
            if old[key] != new[key]:
                raise RuntimeError(f"{relative} differs in {key}")
        difference = max_difference(old["validation_prediction_db"],
                                    new["validation_prediction_db"])
        require_close(difference, relative)
        mask_max = max(mask_max, difference)

    old_test = read(OLD, "layers/final_test.json")
    new_test = read(NEW, "layers/final_test.json")
    arm_max = {}
    for arm in ("Base188", "SelectedResidual", "Full10Residual", "Direct348"):
        difference = max_difference(old_test["predictions_db"][arm],
                                    new_test["predictions_db"][arm])
        require_close(difference, f"final test {arm}")
        arm_max[arm] = difference
    report = {
        "status": "PASS", "tolerance_db": TOLERANCE_DB,
        "old_model_fingerprint": old_selection["fingerprint"],
        "new_model_fingerprint": new_selection["fingerprint"],
        "old_layer_fingerprint": old_layers["fingerprint"],
        "new_layer_fingerprint": new_layers["fingerprint"],
        "winner": new_selection["primary_winner"],
        "selected_mask": new_layers["selected_mask_train_only"],
        "compared_trials": 40, "compared_formal_refits": 30,
        "compared_masks": 1024, "compared_test_arms": 4,
        "max_trial_prediction_difference_db": trial_max,
        "max_formal_prediction_difference_db": formal_max,
        "max_mask_prediction_difference_db": mask_max,
        "max_test_arm_prediction_difference_db": arm_max,
        "latency_note": "Timing noise is not required to match exactly; winner must match",
        "completed_utc": support.utc_now(),
    }
    support.atomic_json(NEW / "REPRODUCTION_CHECK.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
