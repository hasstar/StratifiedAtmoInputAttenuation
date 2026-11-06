"""Independently validate the event-aware Revision_004 dataset package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
DATASETS = PROJECT / "Outputs" / "Datasets"
SOURCE = DATASETS / "Revision_003"
DEST = DATASETS / "Revision_004"
FAST = DATASETS / "Fast_Variation_Revision_001"
PRIOR_EVENTS = DATASETS / "Event_Balanced_Proposal_001" / "hourly_event_split.csv"
SPLIT = "split_event_chrono_24h"
PREFIXES = (
    "model_table_primary_p11",
    "model_table_baseline_l5",
    "model_table_baseline_p0_broad",
    "model_table_baseline_surface_local",
)
EXPECTED_HIGH = {"train": 2526, "validation": 442, "test": 489,
                 "embargo": 48, "snow_case": 24}
EXPECTED_HEAVY_HOURS = {"train": 17, "validation": 26, "test": 25}
EXPECTED_HEAVY_RUNS = {"train": 8, "validation": 6, "test": 8}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_checksums() -> dict[str, object]:
    lines = (DEST / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    listed = {}
    for line in lines:
        digest, filename = line.split("  ", 1)
        require(filename not in listed, f"Duplicate checksum entry: {filename}")
        listed[filename] = digest
    actual = {
        path.name for path in DEST.iterdir()
        if path.is_file() and path.name not in {"checksums.sha256", "validation_report.json", "VALIDATION.md"}
    }
    require(set(listed) == actual, "Checksum list does not match built artifacts")
    for filename, expected in listed.items():
        require(sha256(DEST / filename) == expected, f"Checksum mismatch: {filename}")
    return {"name": "built_artifact_integrity", "files_checked": len(listed)}


def verify_sources() -> dict[str, object]:
    manifest = json.loads((DEST / "run_manifest.json").read_text(encoding="utf-8"))
    source_report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    require(source_report["status"] == "VERIFIED", "Source validation status changed")
    require(manifest["split_design_uses_target"] is False, "Split provenance is mislabeled")
    for filename, expected in manifest["source_sha256"].items():
        require(sha256(SOURCE / filename) == expected, f"Source file changed: {filename}")
    require(
        sha256(FAST / "fast_window_statistics.parquet") == manifest["fast_source_sha256"],
        "Fast-window source changed",
    )
    for filename in (
        "feature_sets.csv", "feature_set_summary.csv", "feature_groups.csv",
        "subset_catalog.csv", "subset_counts.csv", "comparison_registry.csv",
    ):
        require(sha256(SOURCE / filename) == sha256(DEST / filename),
                f"Feature catalog changed: {filename}")
    subsets = pd.read_csv(DEST / "subset_catalog.csv")
    require(len(subsets) == 2048 and subsets["subset_id"].is_unique,
            "The 11-level subset catalog changed")
    return {"name": "verified_source_and_feature_catalog",
            "source_files_checked": len(manifest["source_sha256"]) + 1,
            "copied_feature_sidecars_checked": 6}


def read_splits() -> pd.DataFrame:
    old = pd.read_csv(SOURCE / "time_splits.csv")
    new = pd.read_csv(DEST / "time_splits.csv")
    require(list(new.columns) == list(old.columns) + [SPLIT], "Split sidecar schema changed")
    for frame in (old, new):
        frame["hour_bjt"] = pd.to_datetime(frame["hour_bjt"]).dt.tz_convert("Asia/Shanghai")
        frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True)
    pd.testing.assert_frame_equal(new[old.columns], old,
                                  check_exact=False, rtol=0, atol=1e-15)
    require(len(new) == 3869 and new["hour_bjt"].is_unique, "Invalid measured-hour key")
    require(new[SPLIT].notna().all(), "Unassigned measured hour")
    return new


def verify_tables(splits: pd.DataFrame) -> dict[str, object]:
    lookup = splits.set_index("hour_bjt")[SPLIT]
    checked = 0
    for prefix in PREFIXES:
        for scope in ("all_measurements", "high_coverage_95"):
            stem = f"{prefix}_{scope}"
            old = pd.read_parquet(SOURCE / f"{stem}.parquet")
            new = pd.read_parquet(DEST / f"{stem}.parquet")
            require(SPLIT in new.columns and SPLIT not in old.columns,
                    f"Missing new split column: {stem}")
            pd.testing.assert_frame_equal(new.drop(columns=SPLIT)[old.columns], old,
                                          check_exact=True)
            require(new[SPLIT].equals(new["hour_bjt"].map(lookup).rename(SPLIT)),
                    f"Split map mismatch: {stem}")
            with (DEST / f"{stem}.csv").open("r", encoding="utf-8") as stream:
                require(stream.readline().strip().split(",") == new.columns.tolist(),
                        f"CSV header mismatch: {stem}")
            checked += 1
    return {"name": "model_tables_preserve_targets_and_predictors",
            "tables_checked": checked}


def verify_events(splits: pd.DataFrame) -> dict[str, object]:
    events = pd.read_csv(DEST / "hourly_event_labels.csv")
    events["hour_bjt"] = pd.to_datetime(events["hour_bjt"])
    require(len(events) == len(splits) and events["hour_bjt"].is_unique,
            "Event sidecar hour keys differ")
    joined = events.merge(
        splits[["hour_bjt", SPLIT, "measurement_high_coverage_95"]],
        on="hour_bjt", suffixes=("_event", "_split"), validate="one_to_one",
    )
    require((joined[f"{SPLIT}_event"] == joined[f"{SPLIT}_split"]).all(),
            "Event split does not match split sidecar")
    high = events.loc[events["measurement_high_coverage_95"]].copy()
    require(high[SPLIT].value_counts().to_dict() == EXPECTED_HIGH,
            "High-coverage split size mismatch")
    # Existing independently generated ERA5 event labels are a cross-check.
    prior = pd.read_csv(PRIOR_EVENTS)
    prior["hour_bjt"] = pd.to_datetime(prior["hour_bjt"])
    compare = high.merge(
        prior[["hour_bjt", "rain", "heavy_rain", "dry_overcast", "clear_dry", "snow"]],
        on="hour_bjt", suffixes=("_new", "_prior"), validate="one_to_one",
    )
    require(len(compare) == 3529, "Prior event cross-check is incomplete")
    for event in ("rain", "heavy_rain", "dry_overcast", "clear_dry", "snow"):
        require((compare[f"{event}_new"] == compare[f"{event}_prior"]).all(),
                f"Weather event labels changed: {event}")
    heavy = high.loc[high["heavy_rain"]]
    hours = heavy[SPLIT].value_counts().to_dict()
    runs = heavy.groupby(SPLIT)["heavy_rain_run_id"].nunique().to_dict()
    require(hours == EXPECTED_HEAVY_HOURS, f"Strong-rain hours differ: {hours}")
    require(runs == EXPECTED_HEAVY_RUNS, f"Strong-rain runs differ: {runs}")
    require(set(events.loc[events["snow"], SPLIT]) == {"snow_case"},
            "Observed snow entered a formal split")
    require(events.loc[events[SPLIT] == "snow_case", "hour_bjt"].dt.strftime("%Y-%m-%d").eq("2025-12-12").all(),
            "Snow case includes another date")
    for event in ("rain", "heavy_rain", "dry_overcast", "snow"):
        active = events.loc[events[event] & events[SPLIT].isin(("train", "validation", "test"))]
        if active.empty:
            continue
        require(active.groupby(f"{event}_run_id")[SPLIT].nunique().max() <= 1,
                f"A {event} episode crosses formal splits")
    group = events.loc[events[SPLIT].isin(("train", "validation", "test"))]
    tmax = group.loc[group[SPLIT] == "train", "hour_bjt"].max()
    vmin = group.loc[group[SPLIT] == "validation", "hour_bjt"].min()
    vmax = group.loc[group[SPLIT] == "validation", "hour_bjt"].max()
    emin = group.loc[group[SPLIT] == "test", "hour_bjt"].min()
    require(tmax + pd.Timedelta(hours=24) < vmin, "Train/validation embargo too short")
    require(vmax + pd.Timedelta(hours=24) < emin, "Validation/test embargo too short")
    counts = pd.read_csv(DEST / "event_counts.csv")
    reported = counts.loc[
        (counts["scope"] == "high_coverage_95")
        & (counts["scheme"] == "event_aware_chronological")
        & (counts["event"] == "heavy_rain")
        & (counts["split"].isin(EXPECTED_HEAVY_HOURS))
    ]
    require(reported.set_index("split")["hours"].to_dict() == EXPECTED_HEAVY_HOURS,
            "Reported strong-rain hours differ")
    require(reported.set_index("split")["contiguous_event_runs"].to_dict() == EXPECTED_HEAVY_RUNS,
            "Reported strong-rain runs differ")
    return {"name": "weather_events_and_chronological_isolation",
            "all_measurement_hours_checked": len(events),
            "high_coverage_hours_checked": len(high),
            "heavy_rain_hours": hours, "heavy_rain_runs": runs}


def verify_missingness() -> dict[str, object]:
    table = pd.read_parquet(DEST / "model_table_primary_p11_high_coverage_95.parquet")
    group = pd.read_csv(DEST / "feature_groups.csv")
    p1000 = group.loc[group["group_id"] == "P1000", "column"].tolist()
    require(len(p1000) == 16, "1000 hPa feature group changed")
    missing = pd.read_csv(DEST / "missingness_by_level_split.csv")
    reported = missing.loc[
        (missing["scope"] == "high_coverage_95")
        & (missing["pressure_level_hpa"] == 1000)
    ].set_index("split")
    for split, frame in table.groupby(SPLIT):
        complete = int(frame[p1000].notna().all(axis=1).sum())
        require(int(reported.loc[split, "hours"]) == len(frame),
                f"1000 hPa denominator differs in {split}")
        require(int(reported.loc[split, "all_16_present_hours"]) == complete,
                f"1000 hPa completeness differs in {split}")
    return {"name": "pressure_level_missingness", "1000_hpa_splits_checked": len(reported)}


def verify_fast(splits: pd.DataFrame) -> dict[str, object]:
    lookup = pd.read_parquet(DEST / "fast_hour_split_lookup.parquet")
    require(len(lookup) == len(splits), "Fast hourly lookup missing hours")
    require(lookup["hour_bjt"].is_unique and lookup[SPLIT].notna().all(),
            "Fast hourly lookup key is invalid")
    mapping = splits[["hour_bjt", SPLIT]].copy()
    mapping["hour_bjt"] = mapping["hour_bjt"].dt.tz_localize(None)
    mapping = mapping.merge(lookup, on="hour_bjt", suffixes=("_hour", "_fast"),
                            validate="one_to_one")
    require((mapping[f"{SPLIT}_hour"] == mapping[f"{SPLIT}_fast"]).all(),
            "Fast lookup split mismatch")
    fast = pd.read_parquet(DEST / "fast_window_split.parquet")
    source_rows = len(pd.read_parquet(FAST / "fast_window_statistics.parquet",
                                      columns=["window_start_bjt"]))
    require(len(fast) == source_rows == 95968, "Fast-window row count changed")
    require(fast[SPLIT].notna().all(), "Unassigned fast window")
    require(fast["window_start_bjt"].add(pd.to_timedelta(fast["window_minutes"], unit="m")).le(
        fast["hour_bjt"] + pd.Timedelta(hours=1)
    ).all(), "A fast window crosses an hourly boundary")
    reported = pd.read_csv(DEST / "fast_window_counts.csv")
    primary = fast.loc[fast["window_minutes"] == 15]
    for row in reported.itertuples(index=False):
        sample = primary.loc[
            (primary[SPLIT] == row.split) & primary[row.window_rule]
        ]
        require(len(sample) == row.windows, f"Fast-window count differs: {row.split}")
        require(int(sample["heavy_rain"].sum()) == row.heavy_rain,
                f"Fast strong-rain count differs: {row.split}")
    return {"name": "fast_branch_split_alignment",
            "windows_checked": len(fast), "hour_lookup_rows": len(lookup)}


def main() -> None:
    checks = [
        verify_checksums(),
        verify_sources(),
    ]
    splits = read_splits()
    checks.extend([
        verify_tables(splits),
        verify_events(splits),
        verify_missingness(),
        verify_fast(splits),
    ])
    report = {
        "status": "VERIFIED",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope_note": "Verifies split/data construction, not feature-selection or model performance",
    }
    (DEST / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DEST / "VALIDATION.md").write_text(
        "# Revision_004 validation\n\n"
        f"Status: **{report['status']}**; {len(checks)}/{len(checks)} checks passed.\n\n"
        "Targets and all 11-level/5-layer predictors exactly match verified Revision_003. "
        "Strong-rain hours are 17/26/25 and continuous runs are 8/6/8 "
        "in train/validation/test. All formal splits are chronological and separated "
        "by at least 24 hours; the snow day is reserved separately. "
        "Fast-window split mappings are aligned. No model was trained.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
