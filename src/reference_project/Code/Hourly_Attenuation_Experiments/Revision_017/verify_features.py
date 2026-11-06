"""Independent read-only replay of the train-only feature screen and its audit."""

from __future__ import annotations

import json

import pandas as pd

import run_experiment as experiment


def main() -> None:
    saved = json.loads(experiment.SCREEN.read_text(encoding="utf-8"))
    replay, audit, pairs = experiment.train_only_screen()
    recorded_audit = pd.read_csv(experiment.OUT / "feature_audit.csv")
    recorded_pairs = pd.read_csv(experiment.OUT / "high_correlation_pairs.csv")
    experiment.require(saved["fingerprint"] == replay["fingerprint"] and
                       saved["feature_names_retained"] == replay["feature_names_retained"],
                       "Feature-screen source, order, or result changed")
    experiment.require(len(audit) == len(recorded_audit) == 348 and
                       audit.column.tolist() == recorded_audit.column.tolist() and
                       audit.decision.tolist() == recorded_audit.decision.tolist(),
                       "Feature audit rows differ from replay")
    experiment.require(len(pairs) == len(recorded_pairs) == 354 and
                       pairs[["left", "right"]].equals(recorded_pairs[["left", "right"]]),
                       "High-correlation pair list differs from replay")
    experiment.require(saved["n_retained"] == 232 and
                       saved["max_remaining_absolute_spearman"] < 0.95 and
                       all(count > 0 for count in
                           saved["retained_per_pressure_level"].values()),
                       "Feature count, threshold, or pressure-level coverage failed")
    result = {
        "status": "PASS", "fingerprint": saved["fingerprint"],
        "n_original": saved["n_original"], "n_retained": saved["n_retained"],
        "n_high_correlation_pairs_initial": len(pairs),
        "max_remaining_absolute_spearman": saved["max_remaining_absolute_spearman"],
        "all_ten_levels_nonempty": True,
        "no_target_event_validation_or_test_read": True,
        "verified_utc": experiment.source.utc_now(),
    }
    experiment.source.atomic_json(experiment.OUT / "feature_selection_qa.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
