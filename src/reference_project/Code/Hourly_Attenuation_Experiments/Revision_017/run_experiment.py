"""Revision 017: train-only correlation pruning and ten-model evaluation.

Run this module from any directory using the project's Python environment.
The older Revision 014 model implementations are imported read-only; all
records, locks and QA files are redirected to this revision's output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / "Code/Hourly_Attenuation_Experiments/Revision_014"
sys.path.insert(0, str(PRIOR))
import common as c  # noqa: E402
import run_selection as prior_selection  # noqa: E402
import support as source  # noqa: E402

OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_017_Correlation_Pruned"
DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
SCREEN = OUT / "feature_selection.json"
MANIFEST = OUT / "run_manifest.json"
MODEL_QA = OUT / "model_selection_qa.json"
THRESHOLD = 0.95
MAX_MISSING = 0.50
MIN_PAIR_OBS = 200
VERSION = "correlation-pruned-ten-models-v17-20261001"
STRUCTURED = frozenset(("transformer", "gcn", "bilstm"))

# Redirect imported legacy helpers before any command can write a record.
c.OUT = OUT
prior_selection.OUT = OUT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def hashes(paths: list[Path]) -> dict:
    return {str(path.relative_to(ROOT)): source.sha256(path) for path in paths}


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def train_only_screen() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    columns = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    require(len(columns) == len(set(columns)) == 348, "Expected 348 unique feature columns")
    # Column-selecting Parquet read guarantees that no label, event, validation or
    # test values can participate in the feature-screening calculation.
    train = pd.read_parquet(DATA / "train.parquet", columns=columns)
    require(len(train) == 2404, "Training-hour count changed")
    train = train.replace([np.inf, -np.inf], np.nan)
    miss = train.isna().mean()
    unique = train.nunique(dropna=True)
    eligible = [name for name in columns if miss[name] <= MAX_MISSING and unique[name] > 1]
    corr = train[eligible].corr(method="spearman", min_periods=MIN_PAIR_OBS).abs()
    ordered = sorted(eligible, key=lambda name: (float(miss[name]), columns.index(name)))
    selected, redundant = [], {}
    for name in ordered:
        match = next((kept for kept in selected if corr.loc[name, kept] >= THRESHOLD), None)
        if match is None:
            selected.append(name)
        else:
            redundant[name] = {"representative": match, "absolute_spearman": float(corr.loc[name, match])}
    selected = [name for name in columns if name in set(selected)]
    index = np.triu_indices(len(selected), 1)
    retained_corr = corr.loc[selected, selected].to_numpy()[index]
    max_remaining = float(np.nanmax(retained_corr))
    require(max_remaining < THRESHOLD, "A high-correlation pair survived pruning")
    per_layer = {str(level): sum(name.startswith(f"p{level}__") for name in selected)
                 for level in source.LEVELS}
    require(all(count > 0 for count in per_layer.values()),
            f"At least one pressure level has no retained feature: {per_layer}")
    require(sum(name.startswith("surface__") for name in selected) > 0,
            "No base feature remained")

    audit = []
    for order, name in enumerate(columns, 1):
        reason = ("missing_gt_50_percent" if miss[name] > MAX_MISSING else
                  "constant_or_empty" if unique[name] <= 1 else
                  "correlated" if name in redundant else "retained")
        item = redundant.get(name, {})
        audit.append({"feature_order": order, "column": name,
                      "train_missing_fraction": float(miss[name]),
                      "train_nonmissing_unique_values": int(unique[name]),
                      "decision": reason,
                      "representative": item.get("representative", ""),
                      "absolute_spearman_with_representative": item.get("absolute_spearman")})
    pairs = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= THRESHOLD:
                pairs.append({"left": left, "right": right, "absolute_spearman": float(value)})
    source_hashes = hashes([DATA / "train.parquet", DATA / "feature_columns.csv",
                            HERE / "PROTOCOL.md", HERE / "run_experiment.py"])
    header = {
        "version": VERSION, "source_hashes": source_hashes,
        "selection_scope": "train meteorological columns only; no target/events/validation/test",
        "method": "nonconstant; <=50% missing; Spearman absolute rho <0.95 greedy independent set",
        "correlation_threshold": THRESHOLD,
        "maximum_train_missing_fraction": MAX_MISSING,
        "minimum_pairwise_complete_hours": MIN_PAIR_OBS,
        "feature_names_original": columns,
        "feature_names_retained": selected,
        "n_original": len(columns), "n_retained": len(selected),
        "n_dropped_missing": int(sum(miss[name] > MAX_MISSING for name in columns)),
        "n_dropped_constant": int(sum(miss[name] <= MAX_MISSING and unique[name] <= 1 for name in columns)),
        "n_dropped_correlated": len(redundant),
        "n_high_correlation_pairs_initial": len(pairs),
        "max_remaining_absolute_spearman": max_remaining,
        "retained_base_count": sum(name.startswith("surface__") for name in selected),
        "retained_per_pressure_level": per_layer,
        "prepared_utc": source.utc_now(),
    }
    header["fingerprint"] = fingerprint({key: value for key, value in header.items()
                                         if key != "prepared_utc"})
    require(len(selected) + header["n_dropped_missing"] +
            header["n_dropped_constant"] + header["n_dropped_correlated"] == 348,
            "Pruning counts do not sum to 348")
    return header, pd.DataFrame(audit), pd.DataFrame(pairs)


def prepare() -> dict:
    selection, audit, pairs = train_only_screen()
    if SCREEN.exists():
        old = json.loads(SCREEN.read_text(encoding="utf-8"))
        require(old["fingerprint"] == selection["fingerprint"],
                "Screen source or code changed; create a new revision")
        require((OUT / "feature_audit.csv").exists() and (OUT / "high_correlation_pairs.csv").exists(),
                "Screen audit files missing")
        return old
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "feature_audit.csv", index=False)
    pairs.to_csv(OUT / "high_correlation_pairs.csv", index=False)
    source.atomic_json(SCREEN, selection)
    return selection


def load_screen() -> dict:
    require(SCREEN.exists(), "Run prepare first")
    saved = json.loads(SCREEN.read_text(encoding="utf-8"))
    current, _, _ = train_only_screen()
    require(current["fingerprint"] == saved["fingerprint"],
            "Feature screen no longer matches its sources")
    return saved


def masked_training(saved: dict) -> tuple[pd.DataFrame, list[str]]:
    frame, names, _ = source.load_training()
    require(names == saved["feature_names_original"], "Feature order changed")
    dropped = [name for name in names if name not in set(saved["feature_names_retained"])]
    frame.loc[:, dropped] = np.nan
    return frame, names


class SubsetImputer:
    """Median imputer that also slices the frozen training-selected columns."""

    def __init__(self, positions: list[int]):
        self.positions = np.asarray(positions, dtype=int)
        self.inner = SimpleImputer(strategy="median", keep_empty_features=True)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.inner.fit_transform(values[:, self.positions])

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.inner.transform(values[:, self.positions])


ORIGINAL_PREPROCESS = c.preprocess


def configure_pruned_preprocess(saved: dict) -> None:
    keep = set(saved["feature_names_retained"])
    indices = [i for i, name in enumerate(saved["feature_names_original"]) if name in keep]

    def pruned(xfit: np.ndarray, xhold: np.ndarray | None, model_name: str):
        require(xfit.shape[1] == 348, "Ten-model input layout must have 348 source slots")
        if model_name in STRUCTURED:
            return ORIGINAL_PREPROCESS(xfit, xhold, model_name)
        imputer = SubsetImputer(indices)
        xf = imputer.fit_transform(xfit)
        xh = None if xhold is None else imputer.transform(xhold)
        scaler = StandardScaler() if model_name in c.SCALED_MODELS else None
        if scaler is not None:
            xf = scaler.fit_transform(xf)
            if xh is not None:
                xh = scaler.transform(xh)
        return xf, xh, imputer, scaler

    c.preprocess = pruned


def model_manifest(saved: dict) -> dict:
    source_hashes = hashes([DATA / "train.parquet", DATA / "validation.parquet",
                            DATA / "feature_columns.csv", DATA / "hourly_sample_status.parquet",
                            SCREEN, HERE / "PROTOCOL.md", HERE / "run_experiment.py",
                            PRIOR / "models.py", PRIOR / "common.py", PRIOR / "support.py",
                            PRIOR / "run_selection.py"])
    payload = {"version": VERSION + "-model-fit", "source_hashes": source_hashes,
               "feature_fingerprint": saved["fingerprint"],
               "retained_features": saved["feature_names_retained"],
               "models": list(c.MODEL_NAMES),
               "configs": {name: c.candidate_configs(name) for name in c.MODEL_NAMES},
               "seeds": list(c.SEEDS), "n_train": 2404, "n_validation": 343,
               "selection_rule": "50% normalized validation overall RMSE + 50% normalized log single-row latency",
               "test_not_read": True}
    payload["fingerprint"] = fingerprint(payload)
    payload["created_utc"] = source.utc_now()
    return payload


def run_models(requested: set[str] | None) -> None:
    saved = load_screen()
    frame, names = masked_training(saved)
    configure_pruned_preprocess(saved)
    candidate = model_manifest(saved)
    with c.writer_lock():
        frozen = c.check_manifest(MANIFEST, candidate)
        prior_selection.run(frame, names, frozen, requested)


def verify_models() -> dict:
    saved = load_screen()
    require(MANIFEST.exists(), "Model manifest missing")
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(frozen["fingerprint"] == model_manifest(saved)["fingerprint"],
            "Model manifest fingerprint changed")
    fp = frozen["fingerprint"]
    counts = {"candidate_trials": 0, "seed_validation_fits": 0}
    for name in c.MODEL_NAMES:
        trials = []
        for config_id in range(4):
            record = c.read_record(OUT / "trials" / f"{name}_{config_id:02d}.json", fp)
            require(record is not None and record["model"] == name and
                    record["config_id"] == config_id and
                    len(record["validation_truth_db"]) == len(record["validation_prediction_db"]) == 343,
                    f"Invalid candidate trial {name}/{config_id}")
            require(abs(c.rmse(record["validation_truth_db"], record["validation_prediction_db"]) -
                        record["validation_metrics"]["overall"]["rmse"]) < 1e-10,
                    f"Candidate RMSE mismatch {name}/{config_id}")
            require(record["round_history"] and record["config"] == c.candidate_configs(name)[config_id],
                    f"Missing rounds or altered parameters {name}/{config_id}")
            trials.append(record)
            counts["candidate_trials"] += 1
        best = min(trials, key=lambda row: (row["validation_metrics"]["overall"]["rmse"],
                                            row["validation_metrics"]["overall"]["mae"],
                                            row["config_id"]))
        for seed in c.SEEDS:
            record = c.read_record(OUT / "validation" / f"{name}_seed{seed}.json", fp)
            require(record is not None and record["model"] == name and
                    record["config_id"] == best["config_id"] and record["seed"] == seed and
                    len(record["validation_prediction_db"]) == 343,
                    f"Invalid seed validation {name}/{seed}")
            require(abs(c.rmse(record["validation_truth_db"], record["validation_prediction_db"]) -
                        record["metrics"]["overall"]["rmse"]) < 1e-10,
                    f"Seed RMSE mismatch {name}/{seed}")
            require(record["round_history"] and record["latency"]["median_ms"] > 0,
                    f"Missing round history or latency {name}/{seed}")
            counts["seed_validation_fits"] += 1
    selection_path = OUT / "model_selection.json"
    require(selection_path.exists(), "Ten-model selection missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(selection["fingerprint"] == fp and len(selection["model_rows"]) == 10 and
            selection["status"] == "TEN_MODELS_VALIDATION_SELECTED" and
            selection["test_not_read"] is True,
            "Selection status/fingerprint invalid")
    ranked, summary = source.rank_models([dict(row) for row in selection["model_rows"]])
    require(summary["primary_winner"] == selection["primary_winner"] and
            [r["name"] for r in ranked] == [r["name"] for r in selection["model_rows"]],
            "Model ranking not reproducible")
    qa = {"status": "PASS", "fingerprint": fp, "feature_fingerprint": saved["fingerprint"],
          **counts, "retained_features": len(saved["feature_names_retained"]),
          "winner": selection["primary_winner"], "test_not_read": True,
          "verified_utc": source.utc_now()}
    source.atomic_json(MODEL_QA, qa)
    return qa


def status() -> dict:
    def read(path: Path) -> dict | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    screen = read(SCREEN)
    selection = read(OUT / "model_selection.json")
    return {"phase": "MODELS_QA_PASS" if MODEL_QA.exists() else
            "MODELS_SELECTED" if selection else "MODELS_RUNNING" if MANIFEST.exists() else
            "FEATURES_PREPARED" if screen else "NOT_STARTED",
            "feature_fingerprint": None if screen is None else screen["fingerprint"],
            "retained_features": None if screen is None else screen["n_retained"],
            "model_fingerprint": None if not MANIFEST.exists() else read(MANIFEST)["fingerprint"],
            "candidate_trials": len(list((OUT / "trials").glob("*.json"))),
            "seed_validation_fits": len(list((OUT / "validation").glob("*.json"))),
            "winner": None if selection is None else selection["primary_winner"],
            "model_qa_pass": MODEL_QA.exists() and read(MODEL_QA).get("status") == "PASS",
            "layer_subsets": len(list((OUT / "layers" / "subsets").glob("mask_*.json"))),
            "layer_report": (OUT / "layers" / "layer_selection.json").exists(),
            "layer_qa_pass": (OUT / "layers" / "validation_qa.json").exists(),
            "test_written": (OUT / "layers" / "final_test.json").exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "models", "verify-models", "status"))
    parser.add_argument("--models", nargs="+", choices=c.MODEL_NAMES)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif args.command == "models":
        run_models(None if args.models is None else set(args.models))
    elif args.command == "verify-models":
        print(json.dumps(verify_models(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
