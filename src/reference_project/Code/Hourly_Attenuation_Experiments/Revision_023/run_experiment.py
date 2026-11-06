"""Revision 023: physical-domain and train-only correlation screen; ten-model run.

The frozen Revision 014 estimators and Revision 017 orchestration are imported
read-only. All records are redirected to a new Revision 023 output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "Revision_017" / "run_experiment.py"
_spec = importlib.util.spec_from_file_location("revision017_core_for_revision023", PREVIOUS)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Revision 017 ten-model core could not be loaded")
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)

OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation"
DATA = _core.DATA
SCREEN = OUT / "feature_selection.json"
FEATURE_QA = OUT / "feature_selection_qa.json"
MANIFEST = OUT / "run_manifest.json"
MODEL_QA = OUT / "model_selection_qa.json"
THRESHOLD = 0.90
MAX_MISSING = 0.50
MIN_PAIR_OBS = 200
VERSION = "physical-atmosphere-two-correlation-ten-model-v23-20261002"

_core.HERE = HERE
_core.OUT = OUT
_core.SCREEN = SCREEN
_core.MANIFEST = MANIFEST
_core.MODEL_QA = MODEL_QA
_core.THRESHOLD = THRESHOLD
_core.VERSION = VERSION
_core.c.OUT = OUT
_core.prior_selection.OUT = OUT

# These patterns are fixed before reading outcomes. They apply only to the
# 188 surface__/single-level columns; pressure-level atmospheric fields remain
# eligible and may subsequently be removed by objective train-only screening.
PHYSICAL_EXCLUSIONS = {
    "terrestrial_state": (
        r"(^|_)soil_", r"volumetric_soil_water", r"leaf_area_index",
        r"albedo", r"roughness", r"^skin_", r"friction_velocity",
        r"^snow_density$", r"^snow_depth$", r"^temperature_of_snow_layer$",
        r"snowmelt", r"snow_evaporation", r"runoff",
    ),
    "surface_exchange": (
        r"sensible_heat_flux", r"latent_heat_flux",
        r"instantaneous_moisture_flux", r"surface_stress",
        r"^evaporation$", r"^potential_evaporation$",
        r"evaporation_rate",
    ),
    "radiative_energy_proxy": (
        r"radiation", r"short_wave", r"long_wave", r"solar",
        r"^toa_incident", r"^top_net_thermal", r"^total_sky_direct",
    ),
}


def physical_reason(name: str) -> str | None:
    if not name.startswith("surface__"):
        return None
    suffix = name.split("__", 1)[1]
    for category, patterns in PHYSICAL_EXCLUSIONS.items():
        if any(re.search(pattern, suffix) for pattern in patterns):
            return category
    return None


def correlated(value: float) -> bool:
    return bool(pd.notna(value) and value >= THRESHOLD)


def train_only_screen() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    columns = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    _core.require(len(columns) == len(set(columns)) == 348,
                  "Expected exactly 348 original meteorological columns")
    # Read ONLY the training weather columns. Neither target/event nor formal
    # validation/test fields can influence this physical/redundancy screen.
    train = pd.read_parquet(DATA / "train.parquet", columns=columns)
    _core.require(len(train) == 2404, "Training-hour count changed")
    train = train.replace([np.inf, -np.inf], np.nan)
    missing = train.isna().mean()
    unique = train.nunique(dropna=True)
    physical = {name: physical_reason(name) for name in columns}
    eligible = [name for name in columns if physical[name] is None and
                missing[name] <= MAX_MISSING and unique[name] > 1]
    _core.require(len(eligible) > 1, "No meteorological candidates survived")
    spearman = train[eligible].corr(method="spearman", min_periods=MIN_PAIR_OBS).abs()
    pearson = train[eligible].corr(method="pearson", min_periods=MIN_PAIR_OBS).abs()
    order = sorted(eligible, key=lambda name: (
        float(missing[name]), -int(unique[name]),
        hashlib.sha256(name.encode("utf-8")).hexdigest()))
    retained: list[str] = []
    redundant: dict[str, dict] = {}
    for name in order:
        blockers = []
        for kept in retained:
            sr = spearman.loc[name, kept]
            pr = pearson.loc[name, kept]
            if correlated(sr) or correlated(pr):
                overlap = int((train[name].notna() & train[kept].notna()).sum())
                blockers.append((max(sr if pd.notna(sr) else -1,
                                     pr if pd.notna(pr) else -1), kept,
                                 None if pd.isna(sr) else float(sr),
                                 None if pd.isna(pr) else float(pr), overlap))
        if not blockers:
            retained.append(name)
        else:
            _, kept, sr, pr, overlap = max(blockers, key=lambda item: item[0])
            redundant[name] = {"representative": kept,
                               "absolute_spearman": sr,
                               "absolute_pearson": pr,
                               "pairwise_complete_hours": overlap}
    retained_set = set(retained)
    retained = [name for name in columns if name in retained_set]
    for i, left in enumerate(retained):
        for right in retained[i + 1:]:
            _core.require(not correlated(spearman.loc[left, right]) and
                          not correlated(pearson.loc[left, right]),
                          f"High-correlation edge survived: {left}, {right}")
    audit = []
    for index, name in enumerate(columns, 1):
        reason = (f"physical_{physical[name]}" if physical[name] is not None else
                  "missing_gt_50_percent" if missing[name] > MAX_MISSING else
                  "constant_or_empty" if unique[name] <= 1 else
                  "correlated" if name in redundant else "retained")
        match = redundant.get(name, {})
        audit.append({"feature_order": index, "column": name,
                      "train_missing_fraction": float(missing[name]),
                      "train_nonmissing_unique_values": int(unique[name]),
                      "decision": reason,
                      "representative": match.get("representative", ""),
                      "absolute_spearman_with_representative":
                      match.get("absolute_spearman"),
                      "absolute_pearson_with_representative":
                      match.get("absolute_pearson"),
                      "pairwise_complete_hours":
                      match.get("pairwise_complete_hours")})
    pairs = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            sr, pr = spearman.loc[left, right], pearson.loc[left, right]
            if correlated(sr) or correlated(pr):
                pairs.append({"left": left, "right": right,
                              "absolute_spearman": None if pd.isna(sr) else float(sr),
                              "absolute_pearson": None if pd.isna(pr) else float(pr),
                              "pairwise_complete_hours":
                              int((train[left].notna() & train[right].notna()).sum())})
    physical_counts = {category: sum(physical[name] == category for name in columns)
                       for category in PHYSICAL_EXCLUSIONS}
    source_hashes = _core.hashes([DATA / "train.parquet", DATA / "feature_columns.csv",
                                  HERE / "PROTOCOL.md", HERE / "run_experiment.py",
                                  PREVIOUS])
    header = {
        "version": VERSION, "source_hashes": source_hashes,
        "selection_scope": "train weather columns only; no target/events/validation/test",
        "method": "physical exclusion; <=50% missing; nonconstant; train-only Pearson OR Spearman absolute rho <0.90 greedy independent set",
        "correlation_threshold": THRESHOLD,
        "correlation_methods": ["pearson", "spearman"],
        "maximum_train_missing_fraction": MAX_MISSING,
        "minimum_pairwise_complete_hours": MIN_PAIR_OBS,
        "physical_exclusion_patterns": {k: list(v) for k, v in PHYSICAL_EXCLUSIONS.items()},
        "tie_break": "training missing ascending; observed unique descending; sha256(column) ascending",
        "feature_names_original": columns,
        "feature_names_physically_eligible": [name for name in columns if physical[name] is None],
        "feature_names_quality_eligible": eligible,
        "feature_names_retained": retained,
        "n_original": len(columns), "n_retained": len(retained),
        "n_dropped_physical": sum(physical_counts.values()),
        "physical_exclusion_counts": physical_counts,
        "n_dropped_missing": int(sum(physical[name] is None and
                                     bool(missing[name] > MAX_MISSING)
                                     for name in columns)),
        "n_dropped_constant": int(sum(physical[name] is None and
                                      bool(missing[name] <= MAX_MISSING) and
                                      bool(unique[name] <= 1)
                                      for name in columns)),
        "n_dropped_correlated": len(redundant),
        "n_high_correlation_pairs_initial": len(pairs),
        "retained_base_count": sum(name.startswith("surface__") for name in retained),
        "retained_per_pressure_level": {
            str(level): sum(name.startswith(f"p{level}__") for name in retained)
            for level in _core.source.LEVELS},
        "prepared_utc": _core.source.utc_now(),
    }
    header["fingerprint"] = _core.fingerprint({
        key: value for key, value in header.items() if key != "prepared_utc"})
    _core.require(sum((header["n_retained"], header["n_dropped_physical"],
                       header["n_dropped_missing"], header["n_dropped_constant"],
                       header["n_dropped_correlated"])) == 348,
                  "Original column accounting failed")
    return header, pd.DataFrame(audit), pd.DataFrame(pairs)


_core.train_only_screen = train_only_screen


def verify_features() -> dict:
    saved = _core.load_screen()
    audit_path, pairs_path = OUT / "feature_audit.csv", OUT / "high_correlation_pairs.csv"
    _core.require(audit_path.exists() and pairs_path.exists(), "Feature audit files absent")
    current, audit, pairs = train_only_screen()
    old_audit = pd.read_csv(audit_path).fillna("")
    _core.require(saved["fingerprint"] == current["fingerprint"] and
                  len(old_audit) == len(audit) == 348 and
                  old_audit["column"].tolist() == audit["column"].tolist() and
                  old_audit["decision"].tolist() == audit["decision"].tolist() and
                  len(pd.read_csv(pairs_path)) == len(pairs) and
                  sum(saved["retained_per_pressure_level"].values()) +
                  saved["retained_base_count"] == saved["n_retained"] and
                  all(physical_reason(name) is None for name in
                      saved["feature_names_retained"]),
                  "Feature audit/physical/level/fingerprint QA failed")
    qa = {"status": "PASS", "stage": "train_only_feature_screen",
          "fingerprint": saved["fingerprint"],
          "n_original": 348,
          "n_physically_excluded": saved["n_dropped_physical"],
          "n_missing_excluded": saved["n_dropped_missing"],
          "n_constant_excluded": saved["n_dropped_constant"],
          "n_correlated_excluded": saved["n_dropped_correlated"],
          "n_retained": saved["n_retained"],
          "retained_per_pressure_level": saved["retained_per_pressure_level"],
          "n_high_correlation_pairs_initial": len(pairs),
          "target_event_validation_test_read": False,
          "verified_utc": _core.source.utc_now()}
    _core.source.atomic_json(FEATURE_QA, qa)
    return qa


_original_model_manifest = _core.model_manifest


def model_manifest(saved: dict) -> dict:
    payload = _original_model_manifest(saved)
    payload["source_hashes"][str(PREVIOUS.relative_to(ROOT))] = _core.source.sha256(PREVIOUS)
    payload["source_hashes"][str(FEATURE_QA.relative_to(ROOT))] = _core.source.sha256(FEATURE_QA)
    payload["fingerprint"] = _core.fingerprint({
        key: value for key, value in payload.items()
        if key not in ("fingerprint", "created_utc")})
    return payload


_core.model_manifest = model_manifest


def run_models(requested: set[str] | None) -> None:
    _core.require(FEATURE_QA.exists() and
                  json.loads(FEATURE_QA.read_text(encoding="utf-8"))["status"] == "PASS",
                  "Feature screen QA must PASS before model training")
    _core.run_models(requested)


def verify_models() -> dict:
    _core.require(FEATURE_QA.exists() and
                  json.loads(FEATURE_QA.read_text(encoding="utf-8"))["status"] == "PASS",
                  "Feature screen QA missing")
    return _core.verify_models()


def status() -> dict:
    result = _core.status()
    result["feature_qa_pass"] = FEATURE_QA.exists() and json.loads(
        FEATURE_QA.read_text(encoding="utf-8")).get("status") == "PASS"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-features", "models",
                                            "verify-models", "status"))
    parser.add_argument("--models", nargs="+", choices=_core.c.MODEL_NAMES)
    args = parser.parse_args()
    result = {"prepare": _core.prepare,
              "verify-features": verify_features,
              "models": lambda: run_models(None if args.models is None else set(args.models)),
              "verify-models": verify_models,
              "status": status}[args.command]()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
