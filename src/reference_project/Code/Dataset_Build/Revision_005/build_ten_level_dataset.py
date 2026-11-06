"""Derive the 10-pressure-level hourly package from verified Revision_004."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
DATASETS = PROJECT / "Outputs" / "Datasets"
SOURCE = DATASETS / "Revision_004"
DESTINATION = DATASETS / "Revision_005"
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
OLD_SET = "S_plus_B_plus_P11_primary"
NEW_SET = "S_plus_B_plus_P10_primary"
META = (
    "hour_bjt", "hour_utc", "split_blocked", "split_strict_24h",
    "split_event_chrono_24h", "received_power_dbm",
)
SCOPES = ("all_measurements", "high_coverage_95")
COPY_SIDECARS = (
    "time_splits.csv", "hourly_event_labels.csv", "event_counts.csv",
    "split_summary.csv", "fast_hour_split_lookup.parquet",
    "fast_split_codes.csv", "fast_window_counts.csv", "fast_window_split.parquet",
)
SOURCE_FILES = (
    "validation_report.json", "feature_sets.csv", "feature_groups.csv",
    "model_table_primary_p11_all_measurements.parquet",
    "model_table_primary_p11_high_coverage_95.parquet", *COPY_SIDECARS,
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_features() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[int, list[str]]]:
    sets = pd.read_csv(SOURCE / "feature_sets.csv")
    groups = pd.read_csv(SOURCE / "feature_groups.csv")
    source_sets = {
        name: part.sort_values("feature_order")["column"].tolist()
        for name, part in sets.groupby("feature_set", sort=False)
    }
    require(len(source_sets["S_local"]) == 117, "S_local != 117")
    require(len(source_sets["S_plus_B"]) == 188, "S_plus_B != 188")
    require(len(source_sets[OLD_SET]) == 364, "Source primary != 364")
    require(groups["column"].is_unique, "Source feature groups have duplicate columns")
    pressure = groups.loc[groups["group_kind"] == "pressure_level"].copy()
    pressure["pressure_level_hpa"] = pressure["pressure_level_hpa"].astype(int)
    require(set(pressure["pressure_level_hpa"]) == set((1000, *LEVELS)),
            "Source pressure levels are unexpected")
    by_level = {
        level: pressure.loc[pressure["pressure_level_hpa"] == level]
        .sort_values("feature_order")["column"].tolist()
        for level in LEVELS
    }
    removed = pressure.loc[pressure["pressure_level_hpa"] == 1000, "column"].tolist()
    require(len(removed) == 16 and all(c.startswith("p1000__") for c in removed),
            "1000 hPa is not exactly 16 columns")
    suffixes = [[c.split("__", 1)[1] for c in by_level[level]] for level in LEVELS]
    require(all(len(s) == 16 and s == suffixes[0] for s in suffixes),
            "The ten levels do not share an identical 16-variable schema")
    base = source_sets["S_plus_B"]
    primary = [*base, *(c for level in LEVELS for c in by_level[level])]
    require(len(primary) == 348 and len(primary) == len(set(primary)),
            "New primary count or uniqueness failed")
    require(set(primary) == set(source_sets[OLD_SET]) - set(removed),
            "New primary is not exactly old primary minus 1000 hPa")
    require(set(source_sets["S_local"]) <= set(base), "S_local is not inside S_plus_B")
    return sets, groups, {
        "S_local": source_sets["S_local"], "S_plus_B": base, NEW_SET: primary,
    }, by_level


def write_feature_sidecars(
    output: Path, old_sets: pd.DataFrame, old_groups: pd.DataFrame,
    features: dict[str, list[str]], by_level: dict[int, list[str]],
) -> None:
    parts = [old_sets.loc[old_sets["feature_set"] == name].copy()
             for name in ("S_local", "S_plus_B")]
    old_primary = old_sets.loc[old_sets["feature_set"] == OLD_SET].set_index("column")
    primary = old_primary.loc[features[NEW_SET]].reset_index()
    primary["feature_set"] = NEW_SET
    primary["feature_order"] = range(1, 349)
    primary["source_subgroup"] = "All single-level variables plus 10 pressure levels"
    primary["representation"] = "All single-level variables plus 10 pressure levels"
    primary["dataset_role"] = "formal 10-pressure-level primary"
    parts.append(primary.loc[:, old_sets.columns])
    new_sets = pd.concat(parts, ignore_index=True)
    new_sets.to_csv(output / "feature_sets.csv", index=False)
    (new_sets.groupby("feature_set", sort=False).size()
     .rename("feature_count").reset_index()
     .to_csv(output / "feature_set_summary.csv", index=False))

    new_groups = old_groups.set_index("column").loc[features[NEW_SET]].reset_index()
    new_groups["feature_order"] = range(1, 349)
    new_groups = new_groups.drop(columns="included_in_primary_p11")
    new_groups["included_in_primary_p10"] = True
    new_groups.to_csv(output / "feature_groups.csv", index=False)

    masks = []
    pairs = []
    for mask in range(1 << len(LEVELS)):
        included = [level for bit, level in enumerate(LEVELS) if mask & (1 << bit)]
        excluded = [level for level in LEVELS if level not in included]
        masks.append({
            "subset_id": f"P10_{mask:04X}",
            "mask_decimal": mask,
            "mask_binary": format(mask, "010b"),
            "n_pressure_levels": len(included),
            "retained_levels_hpa": ";".join(map(str, included)),
            "removed_levels_hpa": ";".join(map(str, excluded)),
            "n_pressure_predictors": 16 * len(included),
            "n_predictors_with_S_plus_B": 188 + 16 * len(included),
            "n_predictors_with_S_local": 117 + 16 * len(included),
            "is_formal_primary": mask == 1023,
        })
        for bit, level in enumerate(LEVELS):
            if not mask & (1 << bit):
                pairs.append({
                    "pressure_level_hpa": level,
                    "subset_without_id": f"P10_{mask:04X}",
                    "subset_with_id": f"P10_{mask | (1 << bit):04X}",
                    "other_level_count": mask.bit_count(),
                    "added_predictors": len(by_level[level]),
                })
    pd.DataFrame(masks).to_csv(output / "subset_catalog.csv", index=False)
    pd.DataFrame({
        "n_pressure_levels": range(11),
        "subset_count": [math.comb(10, n) for n in range(11)],
    }).to_csv(output / "subset_counts.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "layer_pair_catalog.csv", index=False)
    pd.DataFrame([
        ("primary_10_pressure", NEW_SET, 348, "formal full-profile primary"),
        ("no_pressure_broad", "S_plus_B", 188,
         "mask zero; fixed base for all 1024 combinations"),
        ("strict_local_surface", "S_local", 117,
         "separate near-surface control, not the combination base"),
    ], columns=["comparison_id", "feature_set", "n_predictors", "role"]).assign(
        model_table_prefix="model_table_primary_p10"
    )[["comparison_id", "model_table_prefix", "feature_set",
       "n_predictors", "role"]].to_csv(output / "comparison_registry.csv", index=False)


def write_tables(output: Path, primary: list[str],
                 by_level: dict[int, list[str]]) -> dict[str, int]:
    old_sets = pd.read_csv(SOURCE / "feature_sets.csv")
    old_primary = old_sets.loc[old_sets["feature_set"] == OLD_SET, "column"].tolist()
    expected = set(META) | set(old_primary)
    hours = {}
    missingness = []
    for scope in SCOPES:
        source = pd.read_parquet(SOURCE / f"model_table_primary_p11_{scope}.parquet")
        require(len(source.columns) == len(expected) and set(source.columns) == expected,
                f"Unexpected source schema for {scope}")
        require(source["hour_bjt"].is_unique, f"Duplicate hour in {scope}")
        frame = source.loc[:, [*META, *primary]]
        require(frame["received_power_dbm"].notna().all(), f"Missing target in {scope}")
        stem = f"model_table_primary_p10_{scope}"
        frame.to_parquet(output / f"{stem}.parquet", index=False, compression="zstd")
        frame.to_csv(output / f"{stem}.csv", index=False, float_format="%.17g")
        hours[scope] = len(frame)
        for split, section in frame.groupby("split_event_chrono_24h", sort=False):
            for level in LEVELS:
                present = section.loc[:, by_level[level]].notna().all(axis=1)
                missingness.append({
                    "scope": scope, "split": split, "pressure_level_hpa": level,
                    "hours": len(section), "all_16_present_hours": int(present.sum()),
                    "at_least_one_missing_hours": int((~present).sum()),
                })
    require(hours == {"all_measurements": 3869, "high_coverage_95": 3529},
            f"Unexpected hour counts: {hours}")
    pd.DataFrame(missingness).to_csv(
        output / "missingness_by_level_split.csv", index=False
    )
    return hours


def write_readme(output: Path, hours: dict[str, int]) -> None:
    description = f"""# Revision_005：十气压面小时模型数据集

从已验证的 Revision_004 派生。只删除 1000 hPa 的 16 个气象预测列，不插值、不重算实测接收功率；此前数据和 ERA5、H5、Run_011 源文件均不修改。

## 主表

- model_table_primary_p10_high_coverage_95.parquet：{hours["high_coverage_95"]:,} 个高覆盖实测小时，是正式模型比较表。
- model_table_primary_p10_all_measurements.parquet：{hours["all_measurements"]:,} 个全部实测小时，用于覆盖率敏感性分析。
- 同名 CSV 仅便于查看；建模首选 Parquet，避免 CSV 再解析时区和浮点数。
- 每行是一小时。前五列是时间与划分，第六列 received_power_dbm 是目标，随后 348 列是同期气象候选输入：188 列 S_plus_B 和 10 个气压面各 16 列。气压面顺序是 925、850、700、650、500、300、200、30、5、1 hPa。
- 事件标签、低功率代理、历史功率、同刻频谱指标均不在模型输入列内。缺失值保持来源原状；本版不做填补、标准化或特征选择。

## 划分和事件

time_splits.csv、hourly_event_labels.csv、event_counts.csv、split_summary.csv 及四个快变划分侧表从 Revision_004 原样复制。正式划分使用 split_event_chrono_24h；高覆盖小时仍为训练 2,526、验证 442、测试 489、隔离 48、雪日个案 24。隔离和雪日个案不参与正式拟合与评分。事件标签仅用于分层评价；低功率代理从目标派生，只用于事后审计。

## 压力层组合

subset_catalog.csv 共 1,024 行，列出十层的全部二进制组合。最低位对应 925 hPa，依次是 850、700、650、500、300、200、30、5、1 hPa。P10_0000 仅用 188 个固定基础参数；P10_03FF 使用全部 348 个参数。每个组合都是列选择清单，并非一张独立模型表，也不表示模型已训练。

feature_sets.csv 列出正式输入顺序；feature_groups.csv 给出各层的 16 个列名及垂直廓线顺序。layer_pair_catalog.csv 给每层列出 512 对“其他层不变，仅加入该层”的对照，共 5,120 对。subset_counts.csv 给出每个层数的组合数。comparison_registry.csv 说明全十层、零压力层和严格近地表三个输入配置均从同一主表选列。

旧版五层表来自 11 层来源，本版不把它冒充为十层同源对照；若后续需要五层对照，应从这十层单独派生。

## 验证边界

run_manifest.json 记录源文件和构建脚本 SHA-256；checksums.sha256 记录构建产物校验和。独立检查见 validation_report.json 和 VALIDATION.md。数据构建验证不等于模型训练或任何气压面贡献已获证实。
"""
    (output / "README.md").write_text(description, encoding="utf-8")


def build() -> None:
    require(SOURCE.is_dir(), "Revision_004 is missing")
    require(not DESTINATION.exists(), "Revision_005 exists; refusing to overwrite")
    source_report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    require(source_report.get("status") == "VERIFIED", "Revision_004 is not verified")
    before = {name: sha256(SOURCE / name) for name in SOURCE_FILES}
    old_sets, old_groups, features, by_level = read_features()
    stage = Path(tempfile.mkdtemp(prefix="_Revision_005_stage_", dir=DATASETS))
    print(f"STAGE {stage}", flush=True)
    try:
        write_feature_sidecars(stage, old_sets, old_groups, features, by_level)
        hours = write_tables(stage, features[NEW_SET], by_level)
        for name in COPY_SIDECARS:
            shutil.copyfile(SOURCE / name, stage / name)
        write_readme(stage, hours)
        after = {name: sha256(SOURCE / name) for name in SOURCE_FILES}
        require(before == after, "Revision_004 source changed during construction")
        manifest = {
            "version": "Revision_005",
            "status": "BUILT_AWAITING_INDEPENDENT_VALIDATION",
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(SOURCE),
            "source_validation_status": "VERIFIED",
            "source_sha256": before,
            "builder_sha256": sha256(Path(__file__)),
            "primary_model_table": "model_table_primary_p10_high_coverage_95.parquet",
            "primary_target": "received_power_dbm",
            "pressure_feature_set": NEW_SET,
            "base_feature_set": "S_plus_B",
            "base_predictors": 188,
            "pressure_levels_hpa": list(LEVELS),
            "pressure_predictors_per_level": 16,
            "primary_predictors": 348,
            "subset_catalog_size": 1024,
            "layer_pair_catalog_size": 5120,
            "hours": hours,
            "split_column": "split_event_chrono_24h",
            "missing_value_policy": "preserve source NaN; train-fold-only preprocessing later",
            "models_trained": 0,
            "source_data_modified": False,
        }
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksums = [f"{sha256(p)}  {p.name}"
                     for p in sorted(stage.iterdir()) if p.is_file()]
        (stage / "checksums.sha256").write_text(
            "\n".join(checksums) + "\n", encoding="utf-8"
        )
        require(not DESTINATION.exists(), "Revision_005 appeared during build")
        stage.rename(DESTINATION)
    except Exception:
        print(f"FAILED: staging files kept at {stage}", flush=True)
        raise
    print(f"BUILT {DESTINATION}")
    print(f"HOURS {hours}; PREDICTORS 348; SUBSETS 1024; PAIRS 5120")
    print("No models were trained.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    require(SOURCE.is_dir(), "Revision_004 is missing")
    report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    require(report.get("status") == "VERIFIED", "Revision_004 is not verified")
    _, _, features, by_level = read_features()
    print(f"BASE {len(features['S_plus_B'])}; LEVELS {list(by_level)}; "
          f"PRIMARY {len(features[NEW_SET])}; SUBSETS {1 << len(LEVELS)}")
    if args.write:
        build()


if __name__ == "__main__":
    main()
