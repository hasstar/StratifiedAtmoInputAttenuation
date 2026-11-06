"""Create a weather-event-aware, chronological split of verified Revision_003.

Only the split changes. All measurement targets and ERA5 predictors are copied
unchanged from Revision_003. The boundaries were chosen from ERA5 weather
episodes, never from received power or fast-fading outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
DATASETS = PROJECT / "Outputs" / "Datasets"
SOURCE = DATASETS / "Revision_003"
FAST = DATASETS / "Fast_Variation_Revision_001"
DESTINATION = DATASETS / "Revision_004"
TZ = "Asia/Shanghai"
SPLIT_COLUMN = "split_event_chrono_24h"
SPLITS = ("train", "validation", "test", "embargo", "snow_case")
TRAIN_END = pd.Timestamp("2026-06-20 00:00:00", tz=TZ)
VALIDATION_START = pd.Timestamp("2026-06-21 00:00:00", tz=TZ)
VALIDATION_END = pd.Timestamp("2026-07-15 00:00:00", tz=TZ)
TEST_START = pd.Timestamp("2026-07-16 00:00:00", tz=TZ)
SNOW_CASE_START = pd.Timestamp("2025-12-12 00:00:00", tz=TZ)
SNOW_CASE_END = SNOW_CASE_START + pd.Timedelta(days=1)
WEATHER_EVENTS = ("rain", "heavy_rain", "dry_overcast", "clear_dry", "snow")
COUNT_EVENTS = WEATHER_EVENTS + ("low_power_le_minus85_dbm",)
TABLE_PREFIXES = (
    "model_table_primary_p11",
    "model_table_baseline_l5",
    "model_table_baseline_p0_broad",
    "model_table_baseline_surface_local",
)
FEATURE_SIDECARS = (
    "feature_sets.csv",
    "feature_set_summary.csv",
    "feature_groups.csv",
    "subset_catalog.csv",
    "subset_counts.csv",
    "comparison_registry.csv",
)
SOURCE_FILES = (
    "validation_report.json",
    "time_splits.csv",
    "model_table_primary_p11_all_measurements.parquet",
    "model_table_primary_p11_high_coverage_95.parquet",
    "era5_hourly_pressure_levels.parquet",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def assign_split(hours: pd.Series) -> pd.Series:
    """Assign one complete chronological interval per split, plus snow case."""
    require(hours.dt.tz is not None, "Expected timezone-aware Beijing hours")
    hours = hours.dt.tz_convert(TZ)
    values = np.select(
        [
            (hours >= SNOW_CASE_START) & (hours < SNOW_CASE_END),
            hours < TRAIN_END,
            hours < VALIDATION_START,
            hours < VALIDATION_END,
            hours < TEST_START,
        ],
        ["snow_case", "train", "embargo", "validation", "embargo"],
        default="test",
    )
    return pd.Series(values, index=hours.index, name=SPLIT_COLUMN)


def load_time_splits() -> pd.DataFrame:
    source_report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    require(source_report["status"] == "VERIFIED", "Revision_003 is not verified")
    frame = pd.read_csv(SOURCE / "time_splits.csv")
    frame["hour_bjt"] = pd.to_datetime(frame["hour_bjt"]).dt.tz_convert(TZ)
    frame["hour_utc"] = pd.to_datetime(frame["hour_utc"])
    require(len(frame) == 3869 and frame["hour_bjt"].is_unique, "Source hour count/key changed")
    frame[SPLIT_COLUMN] = assign_split(frame["hour_bjt"])
    require(set(frame[SPLIT_COLUMN]) == set(SPLITS), "A new split category is absent")
    return frame


def weather_labels() -> pd.DataFrame:
    era = pd.read_parquet(
        SOURCE / "era5_hourly_pressure_levels.parquet",
        columns=[
            "hour_bjt",
            "surface__total_precipitation",
            "surface__snowfall",
            "surface__total_cloud_cover",
        ],
    )
    require(era["hour_bjt"].is_unique, "ERA5 hour key is not unique")
    clouds = era[["hour_bjt", "surface__total_cloud_cover"]].copy()
    accumulated = era[
        ["hour_bjt", "surface__total_precipitation", "surface__snowfall"]
    ].copy()
    # ERA5 accumulated values stamped t+1 describe the measured hour [t,t+1).
    accumulated["hour_bjt"] -= pd.Timedelta(hours=1)
    weather = clouds.merge(accumulated, on="hour_bjt", how="left", validate="one_to_one")
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
    weather = weather.sort_values("hour_bjt").reset_index(drop=True)
    gap = weather["hour_bjt"].diff().ne(pd.Timedelta(hours=1))
    for event in WEATHER_EVENTS:
        active = weather[event]
        starts = active & (~active.shift(fill_value=False) | gap)
        weather[f"{event}_run_id"] = starts.cumsum().where(active).astype("Int64")
    return weather[
        ["hour_bjt", "weather_known", *WEATHER_EVENTS,
         *(f"{event}_run_id" for event in WEATHER_EVENTS)]
    ]


def make_event_sidecar(splits: pd.DataFrame) -> pd.DataFrame:
    targets = pd.read_parquet(
        SOURCE / "model_table_primary_p11_all_measurements.parquet",
        columns=["hour_bjt", "received_power_dbm"],
    )
    require(len(targets) == len(splits), "Target and split hours differ")
    frame = splits[
        ["hour_bjt", "measurement_high_coverage_95", "split_strict_24h", SPLIT_COLUMN]
    ].merge(targets, on="hour_bjt", how="left", validate="one_to_one")
    frame = frame.merge(weather_labels(), on="hour_bjt", how="left", validate="one_to_one")
    require(frame["received_power_dbm"].notna().all(), "A measured target is missing")
    for event in ("weather_known", *WEATHER_EVENTS):
        frame[event] = frame[event].fillna(False).astype(bool)
    frame["low_power_le_minus85_dbm"] = frame["received_power_dbm"] <= -85.0
    frame = frame.sort_values("hour_bjt").reset_index(drop=True)
    low = frame["low_power_le_minus85_dbm"]
    starts = low & (
        ~low.shift(fill_value=False)
        | frame["hour_bjt"].diff().ne(pd.Timedelta(hours=1))
    )
    frame["low_power_le_minus85_dbm_run_id"] = starts.cumsum().where(low).astype("Int64")
    # The target-derived low-power flag is for post hoc audit only, never splitting.
    return frame.drop(columns="received_power_dbm")


def count_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, scoped in (
        ("all_measurements", events),
        ("high_coverage_95", events.loc[events["measurement_high_coverage_95"]]),
    ):
        for scheme, split_column in (
            ("original_chronological", "split_strict_24h"),
            ("event_aware_chronological", SPLIT_COLUMN),
        ):
            for split in SPLITS:
                group = scoped.loc[scoped[split_column] == split]
                if group.empty:
                    continue
                for event in COUNT_EVENTS:
                    hit = group.loc[group[event]]
                    rows.append(
                        {
                            "scope": scope,
                            "scheme": scheme,
                            "split": split,
                            "event": event,
                            "hours": len(hit),
                            "calendar_days": hit["hour_bjt"].dt.normalize().nunique(),
                            "contiguous_event_runs": hit[f"{event}_run_id"].nunique(),
                            "total_split_hours": len(group),
                        }
                    )
    return pd.DataFrame(rows)


def split_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, scoped in (
        ("all_measurements", events),
        ("high_coverage_95", events.loc[events["measurement_high_coverage_95"]]),
    ):
        for split in SPLITS:
            group = scoped.loc[scoped[SPLIT_COLUMN] == split]
            rows.append(
                {
                    "scope": scope,
                    "split": split,
                    "hours": len(group),
                    "first_hour_bjt": group["hour_bjt"].min(),
                    "last_hour_bjt": group["hour_bjt"].max(),
                    "weather_known_hours": int(group["weather_known"].sum()),
                }
            )
    return pd.DataFrame(rows)


def validate_design(events: pd.DataFrame) -> None:
    high = events.loc[events["measurement_high_coverage_95"]]
    sizes = high[SPLIT_COLUMN].value_counts().to_dict()
    require(
        sizes == {"train": 2526, "validation": 442, "test": 489,
                  "embargo": 48, "snow_case": 24},
        f"Unexpected high-coverage split sizes: {sizes}",
    )
    heavy = high.loc[high["heavy_rain"], SPLIT_COLUMN].value_counts().to_dict()
    require(heavy == {"train": 17, "validation": 26, "test": 25},
            f"Unexpected high-coverage heavy-rain counts: {heavy}")
    require(set(events.loc[events["snow"], SPLIT_COLUMN]) == {"snow_case"},
            "Snow hours must be reserved for descriptive case analysis")
    primary = events.loc[events[SPLIT_COLUMN].isin(("train", "validation", "test"))]
    train_end = primary.loc[primary[SPLIT_COLUMN] == "train", "hour_bjt"].max()
    val_start = primary.loc[primary[SPLIT_COLUMN] == "validation", "hour_bjt"].min()
    val_end = primary.loc[primary[SPLIT_COLUMN] == "validation", "hour_bjt"].max()
    test_start = primary.loc[primary[SPLIT_COLUMN] == "test", "hour_bjt"].min()
    require(train_end + pd.Timedelta(hours=24) < val_start,
            "Train-to-validation gap is not greater than 24 hours")
    require(val_end + pd.Timedelta(hours=24) < test_start,
            "Validation-to-test gap is not greater than 24 hours")
    for event in ("rain", "heavy_rain", "dry_overcast", "snow"):
        active = primary.loc[primary[event]]
        if active.empty:
            continue
        require(
            active.groupby(f"{event}_run_id")[SPLIT_COLUMN].nunique().max() <= 1,
            f"A {event} episode crosses formal splits",
        )


def write_tables(directory: Path, splits: pd.DataFrame) -> None:
    lookup = splits.set_index("hour_bjt")[SPLIT_COLUMN]
    for prefix in TABLE_PREFIXES:
        for scope in ("all_measurements", "high_coverage_95"):
            stem = f"{prefix}_{scope}"
            table = pd.read_parquet(SOURCE / f"{stem}.parquet")
            require(table["hour_bjt"].is_unique, f"Duplicate source hour in {stem}")
            new_split = table["hour_bjt"].map(lookup)
            require(new_split.notna().all(), f"Unmapped hour in {stem}")
            table.insert(table.columns.get_loc("received_power_dbm"), SPLIT_COLUMN,
                         new_split.to_numpy())
            table.to_parquet(directory / f"{stem}.parquet", index=False, compression="zstd")
            table.to_csv(directory / f"{stem}.csv", index=False, float_format="%.17g")


def write_missingness(directory: Path, splits: pd.DataFrame) -> None:
    groups = pd.read_csv(SOURCE / "feature_groups.csv")
    levels = groups.loc[groups["group_kind"] == "pressure_level"]
    lookup = splits.set_index("hour_bjt")[SPLIT_COLUMN]
    rows = []
    for scope in ("all_measurements", "high_coverage_95"):
        table = pd.read_parquet(SOURCE / f"model_table_primary_p11_{scope}.parquet")
        table[SPLIT_COLUMN] = table["hour_bjt"].map(lookup)
        for split, frame in table.groupby(SPLIT_COLUMN, sort=False):
            for level, group in levels.groupby("pressure_level_hpa", sort=False):
                complete = frame[group["column"].tolist()].notna().all(axis=1)
                rows.append(
                    {
                        "scope": scope,
                        "split": split,
                        "pressure_level_hpa": int(level),
                        "hours": len(frame),
                        "all_16_present_hours": int(complete.sum()),
                        "at_least_one_missing_hours": int((~complete).sum()),
                    }
                )
    pd.DataFrame(rows).to_csv(directory / "missingness_by_level_split.csv", index=False)


def write_fast_sidecars(directory: Path, splits: pd.DataFrame, events: pd.DataFrame) -> None:
    codes = {"train": 0, "validation": 1, "test": 2, "embargo": 3, "snow_case": 4}
    pd.DataFrame([{"split": name, "code": code} for name, code in codes.items()]).to_csv(
        directory / "fast_split_codes.csv", index=False
    )
    lookup = splits[["hour_bjt", SPLIT_COLUMN]].copy()
    lookup["hour_bjt"] = lookup["hour_bjt"].dt.tz_localize(None)
    lookup["split_event_chrono_code"] = lookup[SPLIT_COLUMN].map(codes).astype("int8")
    lookup.to_parquet(directory / "fast_hour_split_lookup.parquet", index=False, compression="zstd")
    columns = [
        "window_start_bjt", "window_minutes", "window_coverage_fraction",
        "high_coverage_fraction", "continuous_segment_count",
        "fade_below_minus5db_fraction",
    ]
    fast = pd.read_parquet(FAST / "fast_window_statistics.parquet", columns=columns)
    fast["hour_bjt"] = fast["window_start_bjt"].dt.floor("h")
    require(
        (fast["window_start_bjt"] + pd.to_timedelta(fast["window_minutes"], unit="m")
         <= fast["hour_bjt"] + pd.Timedelta(hours=1)).all(),
        "A fast window crosses an hourly split key",
    )
    fast = fast.merge(lookup, on="hour_bjt", how="left", validate="many_to_one")
    require(fast[SPLIT_COLUMN].notna().all(), "Unmapped fast window")
    weather = events[["hour_bjt", *WEATHER_EVENTS]].copy()
    weather["hour_bjt"] = weather["hour_bjt"].dt.tz_localize(None)
    fast = fast.merge(weather, on="hour_bjt", how="left", validate="many_to_one")
    fast["distribution_candidate"] = (
        (fast["window_coverage_fraction"] >= 0.95)
        & (fast["high_coverage_fraction"] >= 0.90)
    )
    fast["continuity_eligible"] = fast["distribution_candidate"] & (
        fast["continuous_segment_count"] == 1
    )
    fast["sustained_minus5db"] = fast["fade_below_minus5db_fraction"] >= 0.01
    fast.drop(columns="fade_below_minus5db_fraction").to_parquet(
        directory / "fast_window_split.parquet", index=False, compression="zstd"
    )
    rows = []
    primary = fast.loc[fast["window_minutes"] == 15]
    for split in SPLITS:
        for rule in ("distribution_candidate", "continuity_eligible"):
            group = primary.loc[(primary[SPLIT_COLUMN] == split) & primary[rule]]
            rows.append(
                {
                    "split": split,
                    "window_rule": rule,
                    "windows": len(group),
                    **{event: int(group[event].sum()) for event in WEATHER_EVENTS},
                    "sustained_minus5db_windows": int(group["sustained_minus5db"].sum()),
                    "sustained_minus5db_days": group.loc[
                        group["sustained_minus5db"], "window_start_bjt"
                    ].dt.normalize().nunique(),
                }
            )
    pd.DataFrame(rows).to_csv(directory / "fast_window_counts.csv", index=False)


def write_readme(directory: Path, counts: pd.DataFrame, summary: pd.DataFrame) -> None:
    high = counts.loc[
        (counts["scope"] == "high_coverage_95")
        & (counts["scheme"] == "event_aware_chronological")
        & (counts["event"] == "heavy_rain")
    ].set_index("split")
    hours = summary.loc[summary["scope"] == "high_coverage_95"].set_index("split")
    readme = f"""# Revision_004：强雨事件兼顾的顺时数据划分

本版本只改变训练／验证／测试的**时间归属**，不重算、不覆盖 Revision_003 的 11 气压面特征或接收功率。正式主表为 `model_table_primary_p11_high_coverage_95.parquet`，使用 `{SPLIT_COLUMN}`；旧的 `split_strict_24h` 仍保留供时间迁移敏感性分析。所有原始 ERA5、H5、Run_011 及 Revision_003 文件均保持不变。

| 类别 | 高覆盖小时 | 强雨小时 | 用途 |
| --- | ---: | ---: | --- |
| 训练 | {hours.loc['train', 'hours']:,} | {high.loc['train', 'hours']} | 拟合、训练内部时间折筛选 |
| 验证 | {hours.loc['validation', 'hours']:,} | {high.loc['validation', 'hours']} | 选定方案及阈值 |
| 测试 | {hours.loc['test', 'hours']:,} | {high.loc['test', 'hours']} | 最后一次评估 |
| 24 小时隔离 | {hours.loc['embargo', 'hours']:,} | 0 | 不训练、不评价 |
| 雪日个案 | {hours.loc['snow_case', 'hours']:,} | 0 | 单独描述，不混入训练或正式测试 |

强雨指 ERA5 小时降水量 ≥1 mm 且雪量 <0.1 mm 水当量；它是天气代理标签，不是地面雨量计实况。训练结束于 2026-06-19 的最后实测小时；2026-06-20 全天隔离；验证从 2026-06-21 到 2026-07-14；2026-07-15 全天隔离；测试从 2026-07-16 到 2026-08-20。唯一有实测降雪的 2025-12-12 全天为 `snow_case`，不作雪天总体精度结论。

这些边界在建模前依据**ERA5 强雨事件分布**固定，不使用接收功率或快变衰落结果来调划分。`event_counts.csv` 列出全部／高覆盖两种口径的事件小时数、日数和连续事件段数；强雨连续段数为训练 8、验证 6、测试 8。同一天的多个连续段可能属于同一场天气过程，不能把每小时或每段都直接当作独立重复。`hourly_event_labels.csv` 包含天气标签和仅供事后审计的低功率代理；事件标签及低功率代理都不是模型输入。

`time_splits.csv` 是全部 3,869 个实测小时的划分侧表。四种输入配置、每种两种覆盖口径均有新的 CSV/Parquet 模型表；正式 11 气压面表保留 364 个候选特征。`subset_catalog.csv` 原样保留 2,048 种组合的清单；这些组合尚未训练。`missingness_by_level_split.csv` 给出新划分下各气压面的可用性，尤其需检查 1000 hPa。

快变 100 ms／1 s 原始派生表没有被复制或改写。用 `fast_hour_split_lookup.parquet` 按无时区标注的北京时间 `hour_bjt` 与快变记录关联；`fast_window_split.parquet` 给 95,968 个现有非重叠窗口附上新划分，`fast_window_counts.csv` 汇总 15 分钟窗口。快变窗口合格性规则未变。旧快变表里的旧划分编码不可再当作本版本划分。

这种方案既保留完整的未来时段测试，又有意让三个集合都包含强雨。其总体测试误差反映的是本次明确设计的后期时段，而不是未经挑选的任意时间切分；原 Revision_003 顺时划分应作为更大跨时期迁移的补充检查。任何缺失填补、标准化和特征选择只能在训练折拟合，验证用于定方案，测试不得反选特征或组合。
"""
    (directory / "README.md").write_text(readme, encoding="utf-8")


def write_package(output: Path, splits: pd.DataFrame, events: pd.DataFrame) -> None:
    splits.to_csv(output / "time_splits.csv", index=False)
    events.to_csv(output / "hourly_event_labels.csv", index=False)
    counts = count_events(events)
    summary = split_summary(events)
    counts.to_csv(output / "event_counts.csv", index=False)
    summary.to_csv(output / "split_summary.csv", index=False)
    for filename in FEATURE_SIDECARS:
        shutil.copyfile(SOURCE / filename, output / filename)
    write_tables(output, splits)
    write_missingness(output, splits)
    write_fast_sidecars(output, splits, events)
    write_readme(output, counts, summary)
    manifest = {
        "version": "Revision_004",
        "status": "BUILT_AWAITING_INDEPENDENT_VALIDATION",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(SOURCE),
        "source_validation_status": "VERIFIED",
        "source_sha256": {name: sha256(SOURCE / name) for name in SOURCE_FILES},
        "fast_source_sha256": sha256(FAST / "fast_window_statistics.parquet"),
        "split_column": SPLIT_COLUMN,
        "split_design_uses_target": False,
        "boundaries_bjt": {
            "train_end_exclusive": TRAIN_END.isoformat(),
            "validation_start_inclusive": VALIDATION_START.isoformat(),
            "validation_end_exclusive": VALIDATION_END.isoformat(),
            "test_start_inclusive": TEST_START.isoformat(),
            "snow_case_start_inclusive": SNOW_CASE_START.isoformat(),
            "snow_case_end_exclusive": SNOW_CASE_END.isoformat(),
        },
        "weather_thresholds": {
            "rain_mm_per_hour": 0.1,
            "heavy_rain_mm_per_hour": 1.0,
            "snow_mm_water_equivalent_per_hour": 0.1,
            "dry_overcast_cloud_fraction": 0.8,
            "clear_dry_cloud_fraction": 0.2,
        },
        "weather_label_timing": "ERA5 precipitation/snowfall at t+1 labels measurement hour [t,t+1); cloud at t",
        "snow_case": "2025-12-12 full Beijing day excluded from formal fit/evaluation",
        "primary_model_table": "model_table_primary_p11_high_coverage_95.parquet",
        "primary_target": "received_power_dbm",
        "pressure_feature_set": "S_plus_B_plus_P11_primary",
        "pressure_subset_catalog_size": 2048,
        "models_trained": 0,
        "source_data_modified": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [f"{sha256(path)}  {path.name}" for path in sorted(output.iterdir()) if path.is_file()]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write Revision_004; otherwise inspect only")
    args = parser.parse_args()
    splits = load_time_splits()
    events = make_event_sidecar(splits)
    validate_design(events)
    print(split_summary(events).to_string(index=False))
    print("\nHeavy-rain event counts:")
    print(count_events(events).query("scope == 'high_coverage_95' and scheme == 'event_aware_chronological' and event == 'heavy_rain'").to_string(index=False))
    if not args.write:
        return
    require(not DESTINATION.exists(), f"Destination already exists: {DESTINATION}")
    with tempfile.TemporaryDirectory(prefix="revision004_build_", dir=DATASETS) as temp:
        package = Path(temp) / "package"
        package.mkdir()
        write_package(package, splits, events)
        package.rename(DESTINATION)
    print(f"\nDataset written to {DESTINATION}")


if __name__ == "__main__":
    main()
