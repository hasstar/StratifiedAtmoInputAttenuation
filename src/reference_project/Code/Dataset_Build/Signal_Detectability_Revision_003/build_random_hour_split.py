"""Build a reproducible 70/20/10 weather-stratified random hourly dataset.

Weather flags and the quality eligibility are used for allocation. Neither
received power nor attenuation values are consulted when assigning hours.
The fixed equipment calibration is retained; the clear reference is recomputed
from the NEW training clear/dry hours after allocation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_001"
FEATURE_SOURCE = PROJECT / "Outputs/Datasets/Revision_006/feature_sets.csv"
OUT = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_003"
AUDIT_FILE = SOURCE / "local_hourly_detectability_audit.parquet"
MODEL_FILE = SOURCE / "local_model_table_p10_detected95.parquet"
PARENT_MANIFEST = SOURCE / "manifest.json"
SPLITS = ("train", "test", "validation")
WEIGHTS = np.array((7, 2, 1), dtype=int)
SEED = 20260928
CLASSES = ("clear_dry", "dry_overcast", "other", "rain_nonheavy", "heavy_rain", "snow")
EVENTS = ("rain", "heavy_rain", "dry_overcast", "clear_dry", "snow")
TARGET = "clear_sky_attenuation_db"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def quotas(size: int) -> list[tuple[int, int, int]]:
    """All class allocations within floor/ceil of exact 70/20/10 quotas."""
    numerator = size * WEIGHTS
    floors = numerator // 10
    leftover = size - int(floors.sum())
    fractional = np.flatnonzero(numerator % 10)
    choices = []
    for selected in itertools.combinations(fractional.tolist(), leftover):
        candidate = floors.copy()
        candidate[list(selected)] += 1
        choices.append(tuple(int(v) for v in candidate))
    return choices


def choose_quotas(sizes: dict[str, int]) -> tuple[dict[str, tuple[int, int, int]], tuple[int, int, int]]:
    total = sum(sizes.values())
    numerators = total * WEIGHTS
    exact = numerators // 10
    missing = total - int(exact.sum())
    remainder_order = sorted(range(3), key=lambda i: (-int(numerators[i] % 10), i))
    exact[remainder_order[:missing]] += 1
    targets = tuple(int(v) for v in exact)

    alternatives = [quotas(sizes[name]) for name in CLASSES]
    feasible = []
    for option in itertools.product(*alternatives):
        column_totals = tuple(sum(row[j] for row in option) for j in range(3))
        if column_totals != targets:
            continue
        squared_integer_error = sum(
            sum((10 * row[j] - sizes[name] * int(WEIGHTS[j])) ** 2 for j in range(3))
            for name, row in zip(CLASSES, option)
        )
        feasible.append((squared_integer_error, option))
    check(bool(feasible), "No exact global 70/20/10 stratified allocation")
    _, best = min(feasible)
    return dict(zip(CLASSES, best)), targets


def classify(frame: pd.DataFrame) -> pd.Series:
    check((~frame.heavy_rain | frame.rain).all(), "Heavy rain is not a rain subset")
    check((~frame.snow | (~frame.rain & ~frame.clear_dry & ~frame.dry_overcast)).all(),
          "Snow overlaps another weather class")
    classes = np.select(
        [frame.snow, frame.heavy_rain, frame.rain, frame.dry_overcast, frame.clear_dry],
        ["snow", "heavy_rain", "rain_nonheavy", "dry_overcast", "clear_dry"],
        default="other",
    )
    return pd.Series(classes, index=frame.index, name="weather_stratum")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    for path in (AUDIT_FILE, MODEL_FILE):
        check(sha256(path) == parent["output_sha256"][path.name],
              f"Upstream dataset changed: {path}")
    feature_key = str(FEATURE_SOURCE.relative_to(PROJECT)).replace("/", "\\")
    check(sha256(FEATURE_SOURCE) == parent["input_sha256"][feature_key],
          "Frozen 348-feature catalog changed")

    audit = pd.read_parquet(AUDIT_FILE)
    model = pd.read_parquet(MODEL_FILE)
    check(len(audit) == 3869 and len(model) == 3434, "Source row counts changed")
    check(audit.hour_utc.is_unique and model.hour_utc.is_unique, "Duplicate hourly timestamp")
    eligible = audit.loc[audit.signal_detected_hour].sort_values("hour_utc").copy()
    check(set(eligible.hour_utc) == set(model.hour_utc), "Source eligible hours mismatch")
    eligible["weather_stratum"] = classify(eligible)
    sizes = {name: int(eligible.weather_stratum.eq(name).sum()) for name in CLASSES}
    check(sum(sizes.values()) == 3434 and all(sizes.values()), "Unexpected weather classes")
    allocation, split_targets = choose_quotas(sizes)

    rng = np.random.default_rng(SEED)
    eligible["split"] = ""
    for name in CLASSES:
        indices = eligible.index[eligible.weather_stratum.eq(name)].to_numpy().copy()
        rng.shuffle(indices)
        start = 0
        for split, count in zip(SPLITS, allocation[name]):
            eligible.loc[indices[start:start + count], "split"] = split
            start += count
        check(start == len(indices), f"Stratum allocation error: {name}")
    check(not eligible.split.eq("").any(), "Unassigned eligible hour")
    check(tuple(int(eligible.split.eq(s).sum()) for s in SPLITS) == split_targets,
          "Final split sizes differ from quotas")

    # Under the new split, the reference must not come from prior validation/test.
    candidates = eligible.loc[eligible.split.eq("train") & eligible.clear_dry]
    reference_dbm = float(candidates.hourly_power_quality0_corrected_dbm.median())
    check(len(candidates) > 1000 and np.isfinite(reference_dbm), "Invalid train-only clear reference")

    feature_catalog = pd.read_csv(FEATURE_SOURCE)
    feature_rows = feature_catalog.loc[
        feature_catalog.feature_set.eq("S_plus_B_plus_P10_primary")].sort_values("feature_order")
    features = feature_rows.column.tolist()
    check(len(features) == len(set(features)) == 348, "Feature schema changed")
    check(feature_rows.feature_order.tolist() == list(range(1, 349)), "Feature order changed")
    check(feature_rows.allowed_primary_predictor.all(), "Disallowed predictor")
    check(set(features).issubset(model.columns), "Predictor absent from source model table")
    check(not model[features].isna().all().any(), "A predictor is all missing")

    assignments = eligible[["hour_utc", "split", "weather_stratum",
                            "hourly_power_quality0_corrected_dbm"]]
    model = model.merge(assignments, on="hour_utc", how="left", validate="one_to_one")
    check(model.split.notna().all(), "Model hour lacks split")
    check(np.isfinite(model.hourly_power_quality0_corrected_dbm).all(), "Missing accepted power")
    model[TARGET] = reference_dbm - model.hourly_power_quality0_corrected_dbm
    check(np.isfinite(model[TARGET]).all(), "Nonfinite relabeled target")
    check(not model.loc[model.split.eq("train"), features].isna().all().any(),
          "Train-only all-missing predictor")

    schema = ["hour_bjt", "hour_utc", *features, TARGET]
    produced: list[Path] = []
    for split in SPLITS:
        part = model.loc[model.split.eq(split), schema].sort_values("hour_utc").reset_index(drop=True)
        path = OUT / f"{split}.parquet"
        part.to_parquet(path, index=False, compression="zstd")
        pd.testing.assert_frame_equal(part, pd.read_parquet(path))
        produced.append(path)

    status_cols = ["hour_bjt", "hour_utc", "target_status",
                   "measurement_unique_timestamps", "quality_zero_records",
                   "signal_detected_100ms_fraction", "quality_bit32_fraction", *EVENTS]
    status = audit[status_cols].merge(
        eligible[["hour_utc", "split", "weather_stratum"]],
        on="hour_utc", how="left", validate="one_to_one")
    status["split"] = status.split.fillna("excluded_unreliable")
    status["weather_stratum"] = status.weather_stratum.fillna("unassigned_unreliable")
    status["use_for_regression"] = status.split.isin(SPLITS)
    status["exclusion_reason"] = np.where(status.use_for_regression, "", status.target_status)
    status = status.sort_values("hour_utc").reset_index(drop=True)
    check(int(status.use_for_regression.sum()) == 3434 and len(status) == 3869,
          "Audit status count mismatch")
    status_path = OUT / "hourly_sample_status.parquet"
    status.to_parquet(status_path, index=False, compression="zstd")
    produced.append(status_path)
    excluded_path = OUT / "excluded_unreliable_hours.csv"
    status.loc[~status.use_for_regression].to_csv(excluded_path, index=False, encoding="utf-8-sig")
    produced.append(excluded_path)

    assignment_path = OUT / "hourly_assignments.csv"
    eligible[["hour_bjt", "hour_utc", "weather_stratum", "split"]].sort_values(
        "hour_utc").to_csv(assignment_path, index=False, encoding="utf-8-sig")
    produced.append(assignment_path)
    quota_path = OUT / "weather_stratum_quotas.csv"
    quota_rows = [{"weather_stratum": name, "total_hours": sizes[name],
                   **{split: allocation[name][j] for j, split in enumerate(SPLITS)}}
                  for name in CLASSES]
    pd.DataFrame(quota_rows).to_csv(quota_path, index=False, encoding="utf-8-sig")
    produced.append(quota_path)
    feature_path = OUT / "feature_columns.csv"
    feature_rows[["feature_order", "column", "source_subgroup", "representation"]].to_csv(
        feature_path, index=False, encoding="utf-8-sig")
    produced.append(feature_path)

    summary_rows, missing_rows = [], []
    for split in SPLITS:
        frame = model.loc[model.split.eq(split)]
        events = eligible.loc[eligible.split.eq(split)]
        summary_rows.append({"split": split, "hours": len(frame),
                             "fraction": len(frame) / len(model),
                             **{event: int(events[event].sum()) for event in EVENTS}})
        for feature in features:
            missing_count = int(frame[feature].isna().sum())
            missing_rows.append({"split": split, "feature": feature, "rows": len(frame),
                                 "missing_count": missing_count,
                                 "missing_fraction": missing_count / len(frame)})
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT / "split_event_summary.csv"
    missing_path = OUT / "feature_missingness.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False, encoding="utf-8-sig")
    produced.extend((summary_path, missing_path))

    manifest = {
        "revision": "Signal_Detectability_Revision_003",
        "source_revision": parent["revision"],
        "split_method": "Fixed-seed, per-hour random sampling within mutually exclusive weather strata",
        "random_seed": SEED,
        "split_order_and_weights": dict(zip(SPLITS, WEIGHTS.tolist())),
        "split_targets": dict(zip(SPLITS, split_targets)),
        "weather_strata": sizes,
        "clear_reference_dbm": reference_dbm,
        "clear_reference_train_candidate_hours": len(candidates),
        "target": TARGET,
        "feature_count": len(features),
        "censored_hours_not_in_regression": 435,
        "caution": "Random hours may share weather events across sets; results do not demonstrate independent temporal generalization. Older test outcomes were viewed before this split.",
        "source_sha256": {str(p.relative_to(PROJECT)).replace("/", "\\"): sha256(p)
                          for p in (AUDIT_FILE, MODEL_FILE, FEATURE_SOURCE, PARENT_MANIFEST)},
        "output_sha256": {p.name: sha256(p) for p in produced},
        "builder_sha256": sha256(Path(__file__)),
        "python": sys.version.splitlines()[0],
        "pandas": pd.__version__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Reference: {reference_dbm:.12f} dBm from {len(candidates)} train clear/dry hours")


if __name__ == "__main__":
    build()
