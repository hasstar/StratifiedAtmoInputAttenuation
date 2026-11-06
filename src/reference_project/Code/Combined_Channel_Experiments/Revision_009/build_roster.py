"""Read-only reuse of frozen scores; no model, fitting, prediction or network calls."""
from __future__ import annotations

import hashlib
import json
import re
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OLD = ROOT / "Outputs/Experiments/Combined_Channel_Revision_008"
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_009"
LOCAL = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
UK = ROOT / "Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet"
SCREEN = ROOT / "Code/Hourly_Attenuation_Experiments/Revision_014/models.py"
AUDIT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_006/MODEL_PROVENANCE.md"
REGISTRY = ROOT / "Code/Combined_Channel_Experiments/Revision_002/MODEL_REGISTRY.md"
SCREENSHOT = Path("C:/Users/SDR_HH/AppData/Local/Temp/codex-clipboard-76aaaf8c-1e11-4a4c-9ec0-76002afe3174.png")
STOP = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json"
AUTOMATION = Path("C:/Users/SDR_HH/.codex/automations/tvt/automation.toml")

KEEP = {
    "ours_frozen_top10": "本文冻结 Top10",
    "bai2022_mlp_adapted": "Bai2022 MLP（既有 LOCAL 适配）",
    "bai2022_lstm_t0_adapted": "Bai2022 LSTM-T0（既有 LOCAL 适配）",
    "tabm2025_official_budgeted": "TabM2025（既有 LOCAL 适配）",
}
REMOVE = {
    "train_mean": "常数诊断，不属于本次 ITU/论文主对照",
    "ridge": "通用基线，不属于本次指定论文对照",
    "elastic_net": "原十模型筛选已经使用",
    "svr_rbf": "原十模型筛选已经使用",
    "random_forest": "原十模型筛选已经使用",
    "xgboost": "原十模型筛选已经使用",
    "tabpfn_v2_author_frozen": "既有结果依赖 LOCAL 带标签推理上下文，不满足零示例要求",
    "tabicl_v2_author_frozen": "既有结果依赖 LOCAL 带标签推理上下文，不满足零示例要求",
}
RAIN_ONLY = {"itu838_geometric_proxy", "itu618_hourly_proxy"}
COHORTS = {
    "LOCAL_test": "LOCAL 测试集",
    "UK_full_common": "英国两站合并",
    "UK_chilbolton_full_common": "Chilbolton",
    "UK_sparsholt_full_common": "Sparsholt",
}
FIELDS = {
    "surface__2m_temperature": ("2 m 温度", "K"),
    "surface__2m_dewpoint_temperature": ("2 m 露点", "K"),
    "surface__surface_pressure": ("地表气压", "Pa -> hPa"),
    "surface__total_column_water_vapour": ("整层水汽", "kg m-2"),
    "surface__total_column_cloud_liquid_water": ("整层云液态水", "kg m-2"),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def put(name, value):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as fh:
        if isinstance(value, str):
            fh.write(value)
        else:
            json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")


def protected():
    delivery = read_json(OLD / "completion.json")
    assert delivery["verification"] == "PASS" and delivery["execution_lock"] == "RELEASED"
    anchors = dict(delivery["artifacts"])
    original = read_json(OLD / "sources_before.json")
    for path, digest in original.items():
        assert path not in anchors or anchors[path] == digest
        anchors[path] = digest
    manifest = read_json(LOCAL / "manifest.json")
    anchors[str(LOCAL / "test.parquet")] = manifest["output_sha256"]["test.parquet"]
    for path in [LOCAL / "manifest.json", OLD / "completion.json", SCREEN, AUDIT, REGISTRY, UK, SCREENSHOT,
                 ROOT / "Tasks/T012_Combined_Channel_And_External_Validation.md",
                 ROOT / "Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md", *CODE.glob("*")]:
        if path.is_file():
            key = str(path)
            current = sha(path)
            assert key not in anchors or anchors[key] == current
            anchors[key] = current
    for key, expected in anchors.items():
        path = Path(key)
        if not path.is_absolute():
            path = ROOT / path
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Protected SHA mismatch: {key}")
    assert STOP.is_file()
    assert re.search(r'^status\s*=\s*"PAUSED"\s*$', AUTOMATION.read_text(encoding="utf-8"), re.M)
    return anchors, len(original)


def rank_subset(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["family"] in KEEP:
            clone = dict(row)
            clone["source_rank"] = clone.pop("rank")
            clone["name"] = KEEP[clone["family"]]
            groups[clone["cohort"], clone["event"]].append(clone)
    result = []
    for group in sorted(groups):
        batch = groups[group]
        assert len(batch) == len(KEEP) and len({x["n"] for x in batch}) == 1
        for row in sorted(batch, key=lambda x: (x["rmse_db_mean"], x["family"])):
            row["rank"] = 1 + sum(x["rmse_db_mean"] < row["rmse_db_mean"] for x in batch)
            result.append(row)
    return result


def inventory(local, uk, cloud):
    result = []
    for key, (name, unit) in FIELDS.items():
        entry = {"field": key, "name": name, "required_units": unit,
                 "unit_status": "requirements_only_not_source_metadata_certification"}
        for label, frame in (("LOCAL_test", local), ("UK_expanded", uk), ("UK_cloud", cloud)):
            present = key in frame.columns
            values = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float) if present else np.array([])
            finite = np.isfinite(values)
            entry[label] = {"rows": len(frame), "present": present,
                            "finite_count": int(finite.sum()),
                            "min": float(values[finite].min()) if finite.any() else None,
                            "max": float(values[finite].max()) if finite.any() else None}
        result.append(entry)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    started = utc()
    try:
        before, old_count = protected()
        put("sources_before.json", before)
        rows = read_json(OLD / "family_rankings.json")
        assert {r["family"] for r in rows} == set(KEEP) | set(REMOVE) | RAIN_ONLY
        selected = rank_subset(rows)
        primary = [r for r in selected if r["cohort"] in COHORTS]
        assert len(primary) == 32
        assert all(r["event"] in ("rain", "dry_overcast") for r in selected)
        for r in selected:
            src = next(s for s in rows if (s["cohort"], s["event"], s["family"]) == (r["cohort"], r["event"], r["family"]))
            for key in ("n", "n_runs", "rmse_db_mean", "rmse_db_sd"):
                assert src[key] == r[key]
        roster = {
            "status": "ADOPTED_ROSTER_COMBINED_ITU_PENDING",
            "retained_frozen": [{"family": k, "name": v, "inference_labelled_context": False,
                                 "author_pretrained_checkpoint": False,
                                 "prior_LOCAL_training": True} for k, v in KEEP.items()],
            "removed_from_main": [{"family": k, "reason": v} for k, v in REMOVE.items()],
            "component_diagnostics_only": sorted(RAIN_ONLY),
            "planned_combined_ITU": {"status": "NOT_COMPUTED", "training": False,
                                     "components": ["P676_gas", "P840_cloud_fog", "P838_rain_with_fixed_path"],
                                     "scintillation": "separate_definition_and_input_gate_not_assumed_zero",
                                     "blockers": ["UK_five_fields_not_in_full_evaluation_tables",
                                                  "clear_reference_physical_mapping_not_established",
                                                  "fixed_version_formula_implementation_not_verified"]},
            "model_selection_reason": "user_method_categories_and_no_labelled_context_not_test_performance",
            "new_training": 0, "new_prediction": 0, "meteorological_requests": 0,
        }
        put("roster.json", roster)
        put("paper_family_rankings.json", primary)
        put("paper_ML_sensitivity_rankings.json", [r for r in selected if r["cohort"] not in COHORTS])
        put("rain_only_component_diagnostics.json", [r for r in rows if r["family"] in RAIN_ONLY])
        local = pd.read_parquet(LOCAL / "test.parquet")
        uk = pd.read_parquet(UK)
        cloud = pd.read_parquet(OLD / "cloud_all.parquet")
        assert len(local) == 687 and len(uk) == len(cloud) == 5664
        fields = inventory(local, uk, cloud)
        assert all(r["LOCAL_test"]["finite_count"] == 687 for r in fields)
        assert all(not r["UK_expanded"]["present"] and not r["UK_cloud"]["present"] for r in fields)
        put("physical_input_inventory.json", {"fields": fields, "UK_columns": list(uk.columns),
                                             "UK_cloud_columns": list(cloud.columns),
                                             "scope": "these_frozen_evaluation_tables_only_not_all_disk_files",
                                             "extra_download_authorized": False})
        coverage = read_json(OLD / "coverage_by_station.json")
        assert sum(c["dates"] for c in coverage) == 236
        assert sum(c["station_hours"] for c in coverage) == 5664
        put("coverage_inherited.json", coverage)
        report = ["# 新版主对照名单与已有结果 — Revision 009", "",
                  "名单调整及旧结果整理已完成；组合 ITU 尚未计算，以下不是最终包含组合 ITU 的排行榜。",
                  "零训练、零新推理、零气象下载。原权重、预测和所有历史结果保留。", "",
                  "## 采用的比较方法", "",
                  "本文冻结 Top10；Bai2022 MLP、Bai2022 LSTM-T0、TabM2025 的既有 LOCAL 适配版；待补齐的组合 ITU。",
                  "后三项学习对照不需要推理例题，但已经在之前获授权的阶段用 LOCAL 训练，不能称作者原始权重。",
                  "Bai 两项对应论文结构，不是新增算法家族，也不是原论文输入/时间分辨率的完整复现。", "",
                  "ElasticNet、SVR、RF、XGBoost 移出；Ridge、训练均值也不再作为主对照。TabPFN/TabICL 因依赖带标签推理上下文移出，不进行伪零示例替代。",
                  "旧 P.838/P.618 雨衰结果仅存入 rain_only_component_diagnostics.json，不改名成组合基线。", "",
                  "## 口径不变", "",
                  "英国全量仍为 Chilbolton 143 日 + Sparsholt 93 日，共 236 站点日/5664 站点小时；含原日期缺口，不是连续236天或退回39天。",
                  "以下复用原共同有效母集5538小时中的雨/阴分组，LOCAL仍是687小时测试集。仅学习模型5593小时母集的既有敏感性结果另存。",
                  "均为小时 RMSE（dB）；3种子分别评分后取均值与样本SD，不取最优种子。名次仅在当前4个冻结学习方法内比较。", ""]
        for cohort, label in COHORTS.items():
            report += [f"## {label}", "", "| 方法 | 雨天 RMSE（排名） | 干燥阴天 RMSE（排名） |",
                       "|---|---:|---:|"]
            for family, name in KEEP.items():
                cells = []
                for event in ("rain", "dry_overcast"):
                    row = next(r for r in primary if r["cohort"] == cohort and r["event"] == event and r["family"] == family)
                    sd = "" if row["rmse_db_sd"] is None else f" ± {row['rmse_db_sd']:.4f}"
                    cells.append(f"{row['rmse_db_mean']:.4f}{sd}（{row['rank']}）")
                report.append(f"| {name} | {' | '.join(cells)} |")
            ns = [next(r["n"] for r in primary if r["cohort"] == cohort and r["event"] == e) for e in ("rain", "dry_overcast")]
            report += ["", f"相同评分小时：雨 {ns[0]}；阴 {ns[1]}。", ""]
        report += ["## 组合 ITU：尚待输入与目标基准核验", "",
                   "拟加入气体 P.676 + 云雾 P.840 + 雨 P.838/固定路径几何。图示 P.618 组合原本针对超越概率；小时形式须另标近似，闪烁定义未满足时不伪造完整四分量。",
                   "LOCAL测试的2米温度、露点、地表气压、整层水汽、整层云液态水均687/687有限；这只是字段清点，不是物理公式QA。",
                   "这五项在英国全量评价表/新增TCC表中不存在；TCC不能替代液态水。不把未核验的其它磁盘缓存当作全量输入。",
                   "需要用户明确允许固定5664小时的有界字段补齐；不得恢复旧持续下载。此外需建立物理损耗到既有相对晴空标签的映射，禁止按英国标签拟合偏置。",
                   "详见 [公式审计](../../../Code/Combined_Channel_Experiments/Revision_009/FORMULA_AUDIT.md) 和 [协议](../../../Code/Combined_Channel_Experiments/Revision_009/PROTOCOL.md)。", "",
                   "## 来源与可追溯性", "",
                   "数值逐字段复用 Revision008 family_rankings；source_rank保留旧14家族名次。数据键和样本门槛未变化。",
                   "Bai论文：[TAP2022 DOI](https://doi.org/10.1109/TAP.2021.3137285)；TabM：[作者代码](https://github.com/yandex-research/tabm)、[论文](https://arxiv.org/abs/2410.24210)。",
                   "模型来源及既有适配限制沿用 Revision006 MODEL_PROVENANCE 和 Revision002 MODEL_REGISTRY；本轮未另训作者模型或下载新的权重。",
                   "旧两条雨衰值和被移出的方法仍保留，不因名次而销毁历史证据。本轮不宣称统计显著性或完成全部 T012。", ""]
        put("REPORT.md", "\n".join(report))
        for key, expected in before.items():
            path = Path(key)
            if not path.is_absolute():
                path = ROOT / path
            assert sha(path) == expected, key
        put("sources_after.json", before)
        artifacts = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
        put("build_receipt.json", {"status": "ROSTER_AND_REUSE_PASS_COMBINED_ITU_PENDING",
                                   "started_utc": started, "finished_utc": utc(),
                                   "original_guarded_sources": old_count, "guard_entries": len(before),
                                   "primary_rows": len(primary), "all_retained_rows": len(selected),
                                   "new_training": 0, "new_prediction": 0, "meteorological_requests": 0,
                                   "historical_inputs_unchanged": True, "combined_ITU_computed": False,
                                   "t012_complete": False, "artifacts": artifacts})
        print(json.dumps({"status": "BUILT_PENDING_INDEPENDENT_CHECK", "primary_rows": len(primary),
                          "guard_entries": len(before), "output": str(OUT)}, ensure_ascii=False))
    except Exception as exc:
        put("failure.json", {"utc": utc(), "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
