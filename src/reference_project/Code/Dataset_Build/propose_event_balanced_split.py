"""Build a separate, weather-balanced split proposal without changing Revision_002.

The existing chronological split remains the primary cross-period evaluation.
This proposal groups whole calendar weeks (and merges weeks sharing a weather
episode), balances ERA5 weather categories and a received-power tail proxy,
and holds the only snow week out for a descriptive test case. This supplement
is not a natural-prevalence or future-time test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Outputs" / "Datasets" / "Revision_002"
DEFAULT_OUTPUT = ROOT / "Outputs" / "Datasets" / "Event_Balanced_Proposal_001"
SPLITS = ("train", "validation", "test")
TARGET_SHARES = np.array([0.60, 0.20, 0.20])
WEATHER_COLUMNS = ("rain", "heavy_rain", "dry_overcast", "clear_dry")
BALANCE_COLUMNS = ("hours",) + WEATHER_COLUMNS + (
    "low_power_le_minus85_dbm", "low_power_days", "fast_minus5db_1pct_windows",
)
BALANCE_WEIGHTS = np.array([2.0, 3.0, 3.5, 2.0, 0.7, 5.0, 1.0, 3.0])
EMBARGO_HOURS = 24


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(
        SOURCE / "model_table_five_layers_high_coverage_95.csv",
        usecols=["hour_bjt", "split_strict_24h", "received_power_dbm"],
    )
    model["t"] = pd.to_datetime(model["hour_bjt"]).dt.tz_localize(None)
    surface = pd.read_csv(
        SOURCE / "era5_hourly_five_layers.csv",
        usecols=[
            "hour_bjt",
            "surface__total_precipitation",
            "surface__snowfall",
            "surface__total_cloud_cover",
        ],
    )
    surface["t"] = pd.to_datetime(surface["hour_bjt"]).dt.tz_localize(None)
    clouds = surface[["t", "surface__total_cloud_cover"]]
    accumulated = surface[
        ["t", "surface__total_precipitation", "surface__snowfall"]
    ].copy()
    # ERA5 accumulated values stamped t+1 describe the measurement hour [t,t+1).
    accumulated["t"] -= pd.Timedelta(hours=1)
    weather = clouds.merge(accumulated, on="t", how="left", validate="one_to_one")
    weather = weather.sort_values("t").reset_index(drop=True)
    p = weather["surface__total_precipitation"] * 1000.0
    snow = weather["surface__snowfall"] * 1000.0
    cloud = weather["surface__total_cloud_cover"]
    known = p.notna() & snow.notna() & cloud.notna()
    weather["weather_known"] = known
    weather["rain"] = known & (p >= 0.1) & (snow < 0.1)
    weather["heavy_rain"] = known & (p >= 1.0) & (snow < 0.1)
    weather["dry_overcast"] = known & (p < 0.1) & (snow < 0.1) & (cloud >= 0.8)
    weather["clear_dry"] = known & (p < 0.1) & (snow < 0.1) & (cloud <= 0.2)
    weather["snow"] = known & (snow >= 0.1)
    gap = weather["t"].diff().ne(pd.Timedelta(hours=1))
    for name in ("rain", "heavy_rain", "dry_overcast", "snow"):
        active = weather[name]
        starts = active & (~active.shift(fill_value=False) | gap)
        weather[f"{name}_run_id"] = starts.cumsum().where(active).astype("Int64")
    selected = [
        "t", "weather_known", *WEATHER_COLUMNS, "snow",
        "rain_run_id", "heavy_rain_run_id", "dry_overcast_run_id", "snow_run_id",
    ]
    model = model.merge(weather[selected], on="t", how="left", validate="one_to_one")
    assert len(model) == 3529 and model["t"].is_unique
    for name in (*WEATHER_COLUMNS, "snow", "weather_known"):
        model[name] = model[name].fillna(False).astype(bool)
    model["low_power_le_minus85_dbm"] = model["received_power_dbm"] <= -85.0
    model = model.sort_values("t").reset_index(drop=True)
    low = model["low_power_le_minus85_dbm"]
    low_starts = low & (
        ~low.shift(fill_value=False)
        | model["t"].diff().ne(pd.Timedelta(hours=1))
    )
    model["low_power_le_minus85_dbm_run_id"] = low_starts.cumsum().where(low).astype("Int64")
    model["week_start"] = model["t"].dt.normalize() - pd.to_timedelta(
        model["t"].dt.dayofweek, unit="D"
    )
    return model.sort_values("t").reset_index(drop=True), weather


def episode_merged_blocks(model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = sorted(model["week_start"].unique())
    week_index = {week: i for i, week in enumerate(weeks)}
    parent = list(range(len(weeks)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for event in ("rain", "heavy_rain", "dry_overcast", "snow", "low_power_le_minus85_dbm"):
        for _, frame in model.loc[model[event]].groupby(f"{event}_run_id"):
            occupied = [week_index[week] for week in frame["week_start"].unique()]
            for idx in occupied[1:]:
                union(occupied[0], idx)

    roots = {week: root(i) for i, week in enumerate(weeks)}
    ordered_roots = sorted(set(roots.values()))
    root_to_group = {value: i for i, value in enumerate(ordered_roots)}
    model = model.copy()
    model["block_id"] = model["week_start"].map(roots).map(root_to_group).astype(int)
    rows = []
    for block_id, frame in model.groupby("block_id", sort=True):
        row = {
            "block_id": int(block_id),
            "first_week": frame["week_start"].min(),
            "last_week": frame["week_start"].max(),
            "first_hour": frame["t"].min(),
            "last_hour": frame["t"].max(),
            "hours": len(frame),
            **{event: int(frame[event].sum()) for event in (*WEATHER_COLUMNS, "snow", "low_power_le_minus85_dbm")},
            "low_power_days": frame.loc[frame["low_power_le_minus85_dbm"], "t"].dt.normalize().nunique(),
        }
        rows.append(row)
    return model, pd.DataFrame(rows).sort_values("first_week").reset_index(drop=True)


def add_fast_planning_counts(model: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "Outputs" / "Datasets" / "Fast_Variation_Revision_001" / "fast_window_statistics.parquet"
    fast = pd.read_parquet(
        path,
        columns=[
            "window_start_bjt", "window_minutes", "window_coverage_fraction",
            "high_coverage_fraction", "fade_below_minus5db_fraction",
        ],
    )
    selected = fast.loc[
        (fast["window_minutes"] == 15)
        & (fast["window_coverage_fraction"] >= 0.95)
        & (fast["high_coverage_fraction"] >= 0.90)
        & (fast["fade_below_minus5db_fraction"] >= 0.01)
    ].copy()
    selected["t"] = selected["window_start_bjt"].dt.floor("h")
    selected = selected.merge(model[["t", "block_id"]], on="t", how="inner", validate="many_to_one")
    counts = selected.groupby("block_id").size()
    blocks = blocks.copy()
    blocks["fast_minus5db_1pct_windows"] = blocks["block_id"].map(counts).fillna(0).astype(int)
    return blocks


def optimize(blocks: pd.DataFrame) -> pd.DataFrame:
    n, n_splits, n_features = len(blocks), len(SPLITS), len(BALANCE_COLUMNS)
    block_count = blocks.set_index("block_id")["hours"]
    edges = []
    for i in range(n - 1):
        if blocks.loc[i, "last_week"] + pd.Timedelta(days=7) != blocks.loc[i + 1, "first_week"]:
            continue
        if blocks.loc[i + 1, "first_hour"] - blocks.loc[i, "last_hour"] > pd.Timedelta(hours=24):
            continue
        edges.append((i, i + 1))
    x_end = n * n_splits
    plus_start = x_end
    minus_start = plus_start + n_features * n_splits
    edge_start = minus_start + n_features * n_splits
    n_variables = edge_start + len(edges)
    objective = np.zeros(n_variables)
    integrality = np.zeros(n_variables, dtype=np.int8)
    integrality[:x_end] = 1
    integrality[edge_start:] = 1
    lower = np.zeros(n_variables)
    upper = np.full(n_variables, np.inf)
    upper[:x_end] = 1
    upper[edge_start:] = 1
    for feature_index, feature in enumerate(BALANCE_COLUMNS):
        total = float(blocks[feature].sum())
        weight = BALANCE_WEIGHTS[feature_index] / total
        for split_index in range(n_splits):
            j = feature_index * n_splits + split_index
            objective[plus_start + j] = weight
            objective[minus_start + j] = weight
    objective[edge_start:] = 0.04
    rows, lows, highs = [], [], []

    def constrain(coefficients: np.ndarray, low: float, high: float) -> None:
        rows.append(coefficients)
        lows.append(low)
        highs.append(high)

    for group in range(n):
        row = np.zeros(n_variables)
        row[group * n_splits : (group + 1) * n_splits] = 1
        constrain(row, 1, 1)
    for feature_index, feature in enumerate(BALANCE_COLUMNS):
        counts = blocks[feature].to_numpy(dtype=float)
        total = float(counts.sum())
        for split_index in range(n_splits):
            j = feature_index * n_splits + split_index
            row = np.zeros(n_variables)
            row[split_index:x_end:n_splits] = counts
            row[plus_start + j] = -1
            row[minus_start + j] = 1
            target = TARGET_SHARES[split_index] * total
            constrain(row, target, target)
    for group in blocks.loc[blocks["snow"] > 0, "block_id"]:
        row = np.zeros(n_variables)
        row[int(group) * n_splits + 2] = 1
        constrain(row, 1, 1)
    minimum = {
        "hours": (1500, 400, 400),
        "rain": (60, 20, 20),
        "heavy_rain": (20, 12, 12),
        "dry_overcast": (200, 50, 50),
        "low_power_le_minus85_dbm": (20, 10, 10),
        "low_power_days": (5, 3, 3),
        "fast_minus5db_1pct_windows": (40, 20, 20),
    }
    for feature, limits in minimum.items():
        counts = blocks[feature].to_numpy(dtype=float)
        for split_index, limit in enumerate(limits):
            row = np.zeros(n_variables)
            row[split_index:x_end:n_splits] = counts
            constrain(row, limit, np.inf)
    for edge_index, (a, b) in enumerate(edges):
        for split_index in range(n_splits):
            row = np.zeros(n_variables)
            row[a * n_splits + split_index] = 1
            row[b * n_splits + split_index] = -1
            row[edge_start + edge_index] = -1
            constrain(row, -np.inf, 0)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.vstack(rows), lows, highs),
        options={"time_limit": 90, "mip_rel_gap": 0.001},
    )
    if result.x is None:
        raise RuntimeError(f"Weather balance optimization failed: {result.message}")
    assignment = np.argmax(result.x[:x_end].reshape(n, n_splits), axis=1)
    blocks = blocks.copy()
    blocks["proposed_split_before_embargo"] = [SPLITS[i] for i in assignment]
    blocks["optimization_status"] = result.message
    blocks["optimization_objective"] = result.fun
    assert block_count.sum() == 3529
    return blocks


def apply_embargo(model: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    model = model.copy().sort_values("t").reset_index(drop=True)
    split_map = blocks.set_index("block_id")["proposed_split_before_embargo"]
    model["proposed_split_before_embargo"] = model["block_id"].map(split_map)
    model["event_balanced_split"] = model["proposed_split_before_embargo"]
    model["embargo_reason"] = ""
    last_seen = {split: None for split in SPLITS}
    for idx, row in model.iterrows():
        t, split = row["t"], row["proposed_split_before_embargo"]
        if any(
            other != split and last_seen[other] is not None
            and t - last_seen[other] <= pd.Timedelta(hours=EMBARGO_HOURS)
            for other in SPLITS
        ):
            model.at[idx, "event_balanced_split"] = "embargo"
            model.at[idx, "embargo_reason"] = "24h_cross_split"
        last_seen[split] = t
    for event in ("rain", "heavy_rain", "dry_overcast", "snow", "low_power_le_minus85_dbm"):
        key = f"{event}_run_id"
        active = model.loc[model[event] & model["event_balanced_split"].isin(SPLITS)]
        crossing = active.groupby(key)["event_balanced_split"].nunique()
        bad = set(crossing[crossing > 1].index)
        if bad:
            mask = model[key].isin(bad)
            model.loc[mask, "event_balanced_split"] = "embargo"
            model.loc[mask, "embargo_reason"] = "cross_split_weather_episode"
    return model


def counts_table(model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, split_column in (
        ("original_chronological", "split_strict_24h"),
        ("supplementary_event_balanced", "event_balanced_split"),
    ):
        for split in (*SPLITS, "embargo"):
            frame = model.loc[model[split_column] == split]
            for event in (*WEATHER_COLUMNS, "snow", "low_power_le_minus85_dbm"):
                selected = frame.loc[frame[event]]
                run_key = f"{event}_run_id"
                rows.append(
                    {
                        "scheme": label,
                        "split": split,
                        "event": event,
                        "hours": len(selected),
                        "calendar_days": selected["t"].dt.normalize().nunique(),
                        "contiguous_event_runs": selected[run_key].nunique() if run_key in selected else np.nan,
                        "total_split_hours": len(frame),
                    }
                )
    return pd.DataFrame(rows)


def fast_window_counts(model: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "Outputs" / "Datasets" / "Fast_Variation_Revision_001" / "fast_window_statistics.parquet"
    columns = [
        "window_start_bjt", "window_minutes", "window_coverage_fraction",
        "high_coverage_fraction", "continuous_segment_count",
        "fade_below_minus5db_fraction",
    ]
    fast = pd.read_parquet(path, columns=columns)
    fast = fast.loc[fast["window_minutes"] == 15].copy()
    fast["t"] = fast["window_start_bjt"].dt.floor("h")
    selected = model[["t", "event_balanced_split", "rain", "heavy_rain", "dry_overcast", "snow"]]
    fast = fast.merge(selected, on="t", how="inner", validate="many_to_one")
    fast["distribution_candidate"] = (
        (fast["window_coverage_fraction"] >= 0.95)
        & (fast["high_coverage_fraction"] >= 0.90)
    )
    fast["continuity_eligible"] = fast["distribution_candidate"] & (
        fast["continuous_segment_count"] == 1
    )
    fast["any_minus5db"] = fast["fade_below_minus5db_fraction"] > 0
    fast["sustained_minus5db"] = fast["fade_below_minus5db_fraction"] >= 0.01
    fast["day"] = fast["window_start_bjt"].dt.normalize()
    rows = []
    for split in (*SPLITS, "embargo"):
        frame = fast.loc[fast["event_balanced_split"] == split]
        for flag in ("distribution_candidate", "continuity_eligible"):
            subset = frame.loc[frame[flag]]
            rows.append(
                {
                    "split": split,
                    "window_rule": flag,
                    "windows": len(subset),
                    **{event: int(subset[event].sum()) for event in ("rain", "heavy_rain", "dry_overcast", "snow", "any_minus5db", "sustained_minus5db")},
                    "sustained_minus5db_days": subset.loc[subset["sustained_minus5db"], "day"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def verify(model: pd.DataFrame) -> None:
    assert len(model) == 3529 and model["t"].is_unique
    assert model["event_balanced_split"].notna().all()
    assert set(model.loc[model["snow"], "event_balanced_split"]) == {"test"}
    for event in ("rain", "heavy_rain", "dry_overcast", "snow", "low_power_le_minus85_dbm"):
        active = model.loc[model[event] & model["event_balanced_split"].isin(SPLITS)]
        assert active.groupby(f"{event}_run_id")["event_balanced_split"].nunique().max() <= 1
    retained = model.loc[model["event_balanced_split"].isin(SPLITS)]
    for i, row in retained.iterrows():
        later = retained.loc[(retained["t"] > row["t"]) & (retained["t"] <= row["t"] + pd.Timedelta(hours=24))]
        assert (later["event_balanced_split"] == row["event_balanced_split"]).all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write a new sidecar proposal and audit files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model, _ = load_events()
    model, blocks = episode_merged_blocks(model)
    blocks = add_fast_planning_counts(model, blocks)
    blocks = optimize(blocks)
    model = apply_embargo(model, blocks)
    verify(model)
    counts = counts_table(model)
    fast_counts = fast_window_counts(model)
    print(counts.to_string(index=False))
    print("\n15-minute fast windows:\n", fast_counts.to_string(index=False))
    if args.write:
        args.output.mkdir(parents=True, exist_ok=False)
        hour_columns = [
            "hour_bjt", "split_strict_24h", "event_balanced_split",
            "proposed_split_before_embargo", "embargo_reason", "block_id",
            "weather_known", *WEATHER_COLUMNS, "snow",
            "rain_run_id", "heavy_rain_run_id", "dry_overcast_run_id", "snow_run_id",
            "low_power_le_minus85_dbm", "low_power_le_minus85_dbm_run_id",
        ]
        model[hour_columns].to_csv(args.output / "hourly_event_split.csv", index=False)
        blocks.to_csv(args.output / "weather_blocks.csv", index=False)
        counts.to_csv(args.output / "event_counts.csv", index=False)
        fast_counts.to_csv(args.output / "fast_window_counts.csv", index=False)
        manifest = {
            "status": "PROPOSAL_ONLY_NOT_PRIMARY_SPLIT",
            "source": str(SOURCE),
            "source_rows": len(model),
            "weather_label_timing": "ERA5 precipitation and snowfall stamped t+1 label measurement hour [t,t+1); cloud at t",
            "weather_thresholds": {
                "rain_mm_per_hour": 0.1,
                "heavy_rain_mm_per_hour": 1.0,
                "snow_mm_water_equivalent_per_hour": 0.1,
                "dry_overcast_cloud_fraction": 0.8,
                "clear_dry_cloud_fraction": 0.2,
            },
            "target_shares": dict(zip(SPLITS, TARGET_SHARES.tolist())),
            "grouping": "Whole Monday-Sunday weeks; weeks sharing measured rain, heavy-rain, dry-overcast, snow, or low-power runs are merged",
            "embargo_hours": EMBARGO_HOURS,
            "snow_handling": "Only observed snow day 2025-12-12 is forced to test as a descriptive held-out case, not a statistically independent snow benchmark",
            "stratification_variables": list(BALANCE_COLUMNS),
            "outcome_used_to_optimize_split": True,
            "outcome_stratification_caveat": "The <=-85 dBm received-power proxy and >=1% of seconds below -5 dB in coverage-qualified fast windows are used only to balance this supplementary challenge split. They are not calibrated atmospheric attenuation or an unbiased future-period estimate.",
            "fast_distribution_candidate_caveat": "Coverage-only candidate, not formally accepted until missingness and strict-window sensitivity are audited",
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nProposal written to {args.output}")


if __name__ == "__main__":
    main()
