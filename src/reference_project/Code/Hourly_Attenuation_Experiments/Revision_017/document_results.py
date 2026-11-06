"""Generate an evidence-linked Chinese summary only after both QA gates pass."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_experiment as stage

OUT = stage.OUT
LAYERS = OUT / "layers"


def read_json(path: Path) -> dict:
    stage.require(path.exists(), f"Required output missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def report() -> Path:
    features = read_json(stage.SCREEN)
    model_qa = read_json(stage.MODEL_QA)
    selection = read_json(OUT / "model_selection.json")
    layer_qa = read_json(LAYERS / "final_qa.json")
    layer_choice = read_json(LAYERS / "layer_selection.json")
    test = read_json(LAYERS / "final_test.json")
    layer_manifest = read_json(LAYERS / "run_manifest.json")
    stage.require(model_qa["status"] == layer_qa["status"] == "PASS" and
                  model_qa["candidate_trials"] == 40 and
                  model_qa["seed_validation_fits"] == 30 and
                  layer_qa["validated_subsets"] == 1024 and
                  layer_qa["validated_layer_pairs"] == 5120 and
                  layer_manifest["feature_fingerprint"] == features["fingerprint"] and
                  test["fingerprint"] == layer_qa["fingerprint"],
                  "Do not document partial or fingerprint-mismatched results")
    layer_summary = pd.read_csv(LAYERS / "layer_summary.csv")
    stage.require(len(layer_summary) == 10, "Expected ten layer contributions")
    layer_control = read_json(LAYERS / "subsets" / "mask_0000.json")
    layer_selected = read_json(
        LAYERS / "subsets" / f"mask_{layer_choice['selected_mask_train_only']:04d}.json")
    layer_all = read_json(LAYERS / "subsets" / "mask_1023.json")
    lines = [
        "# Revision 017：相关性去冗余后的小时衰减实验结果",
        "",
        "本报告仅在十模型及全部 1024 组合、687 小时测试的核验均通过后生成。"
        "原始 H5、ERA5、Run_011、数据集及旧实验均未修改。",
        "",
        "## 数据与特征工程",
        "",
        f"训练/验证/测试分别为 2404/343/{test['n_test']} 个质量合格小时；目标为"
        "同小时相对晴空衰减 `clear_sky_attenuation_db`（dB）。",
        f"仅利用训练小时的 ERA5 输入列做筛查：348 → {features['n_retained']} 列，"
        f"其中基础 {features['retained_base_count']} 列、十气压面"
        f" {features['n_retained']-features['retained_base_count']} 列。",
        f"剔除原因：缺失率 >50% {features['n_dropped_missing']} 列、恒定/空值"
        f" {features['n_dropped_constant']} 列、成对 `|Spearman ρ|≥0.95`"
        f" {features['n_dropped_correlated']} 列。筛后最大成对 `|ρ|`"
        f" 为 {features['max_remaining_absolute_spearman']:.4f}；"
        "这是两两秩相关约束，并非统计独立的证明。",
        "",
        "| 气压面 (hPa) | 保留气象列数 |",
        "|---:|---:|",
    ]
    for pressure, count in features["retained_per_pressure_level"].items():
        lines.append(f"| {pressure} | {count} |")
    lines += [
        "",
        "完整列名及去留原因见 `feature_selection.json` 和 `feature_audit.csv`；"
        "并未利用目标值或天气事件选列。被筛除列在垂直网络的固定结构中仅为"
        "训练折处理后的中性常数槽位，不含额外观测信息。",
        "",
        "## 十模型比较：343 小时验证",
        "",
        "四套固定候选配置各自保存逐轮轨迹；最佳配置按验证总体 RMSE 选出，"
        "固定该轮数后三种子重新拟合。主排名按验证总体 RMSE 与单条推理延迟"
        "（对数、分别 min-max 归一化）各 50% 计算；雨天包含全部强雨。",
        "",
        "| 排名 | 模型 | 验证总体 RMSE (dB) | 雨天 RMSE (dB) | 干燥阴天 RMSE (dB) | 单条推理 (ms) | 得分 | 轮数 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selection["model_rows"]:
        lines.append(
            f"| {row['rank_50_50']} | {row['name']} | {fmt(row['overall_rmse_mean'])} | "
            f"{fmt(row['rain_rmse_mean'])} | {fmt(row['dry_overcast_rmse_mean'])} | "
            f"{fmt(row['latency_median_ms'], 4)} | {fmt(row['score_50_50'], 4)} | "
            f"{row['formal_rounds'] if row['formal_rounds'] is not None else '求解器收敛'} |"
        )
    lines += [
        "",
        f"等权规则胜出：**{selection['primary_winner']}**；仅按总体 RMSE 胜出："
        f"**{selection['rmse_only_winner']}**。"
        "所有参数、三个种子的逐小时预测、逐轮训练/验证曲线以及结束原因"
        "保存在 `trials/`、`validation/`、`model_selection.json`。",
        "",
        "## 十层 1024 组合：验证层贡献",
        "",
        "所有组合固定上述获胜学习器的家族、配置与轮数；基础气象模型采用"
        "筛后基础列，气压面组合分别重新训练分层残差头。每层在 512 个"
        "‘其他层’搭配中对照有/无该层，以下正值表示加入该层后 RMSE 降低，"
        "负值表示升高；这些搭配高度相关，不能当作 512 次独立重复。",
        "",
        "| 气压面 (hPa) | 总体平均改善 (dB) | 雨天平均改善 (dB) | 干燥阴天平均改善 (dB) | 总体改善比例 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in layer_summary.itertuples(index=False):
        lines.append(
            f"| {row.level_hpa} | {fmt(row.overall_mean_improvement_db, 4)} | "
            f"{fmt(row.rain_mean_improvement_db, 4)} | "
            f"{fmt(row.dry_overcast_mean_improvement_db, 4)} | "
            f"{fmt(100*row.overall_fraction_improved, 1)}% |"
        )
    lines += [
        "",
        f"训练内部按日分组评价选出的掩码为 {layer_choice['selected_mask_train_only']}，"
        f"包含气压面 {layer_choice['selected_levels_hpa']}；"
        f"残差权重 α={layer_choice['selected_alpha']}，"
        f"正式残差树数 {layer_choice['selected_formal_rounds']}。"
        "1024 个组合的完整特征名单、参数、每折逐轮曲线、alpha 选择、"
        "逐小时验证预测及各事件指标见 `layers/subsets/`；"
        "汇总与配对结果见 `layers/subset_summary.csv`、"
        "`layers/layer_pairwise.csv`、`layers/layer_summary.csv`。",
        "",
        "同一固定种子的分层方案验证对照如下（与上面的三种子十模型均值口径不同）：",
        "",
        "| 分层方案 | 验证总体 RMSE (dB) | 雨天 RMSE (dB) | 干燥阴天 RMSE (dB) |",
        "|---|---:|---:|---:|",
    ]
    for label, record in (("基础无层", layer_control),
                          ("训练内部所选两层", layer_selected),
                          ("全部十层残差", layer_all)):
        metrics = record["validation_metrics"]
        lines.append(f"| {label} | {fmt(metrics['overall']['rmse'])} | "
                     f"{fmt(metrics['rain']['rmse'])} | "
                     f"{fmt(metrics['dry_overcast']['rmse'])} |")
    lines += [
        "",
        "所选两层在训练内部 CV 中优于基础方案，但在验证集总体与干燥阴天"
        "没有优于基础方案；验证集雨天则略有改善。这是过拟合/分布差异"
        "诊断，不据此改选掩码。",
        "",
        "## 预定 687 小时测试：所有选择已冻结后计算预测和评分",
        "",
        "| 方案 | 总体 RMSE (dB) | 雨天 RMSE (dB) | 干燥阴天 RMSE (dB) | 总体 MAE (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metric in test["metrics"].items():
        lines.append(
            f"| {label} | {fmt(metric['overall']['rmse'])} | "
            f"{fmt(metric['rain']['rmse'])} | {fmt(metric['dry_overcast']['rmse'])} | "
            f"{fmt(metric['overall']['mae'])} |"
        )
    base = test["metrics"]["Base188"]
    direct = test["metrics"]["Direct348"]
    selected = test["metrics"]["SelectedResidual"]
    lines += [
        "",
        f"直接使用筛后全部特征的 XGBoost 相比筛后基础特征方案，总体 RMSE "
        f"降低 {base['overall']['rmse']-direct['overall']['rmse']:.3f} dB，"
        f"雨天降低 {base['rain']['rmse']-direct['rain']['rmse']:.3f} dB，"
        f"干燥阴天降低 {base['dry_overcast']['rmse']-direct['dry_overcast']['rmse']:.3f} dB。"
        "这是锁定方案后的描述性对照，不据此重新选模；差异是否稳定仍需"
        "跨时段或异站验证。",
        f"训练内部选出的两层残差方案相对基础方案，测试总体 RMSE "
        f"增加 {selected['overall']['rmse']-base['overall']['rmse']:.3f} dB，"
        f"雨天增加 {selected['rain']['rmse']-base['rain']['rmse']:.3f} dB，"
        f"干燥阴天增加 {selected['dry_overcast']['rmse']-base['dry_overcast']['rmse']:.3f} dB；"
        "因此本版不能声称所选分层残差组合改善了保留集预测。",
        "",
        "内部名称 `Base188` 表示原始基础 188 槽位中仅使用筛后保留列；"
        "`Direct348` 表示原始 348 槽位中仅使用筛后保留列，平面模型实际"
        f"输入 {features['n_retained']} 列。上述名称不表示筛除列参与训练。",
        "",
        "本地测试集在更早实验中已经被查看；本版选模过程中另有一次只读的测试标签"
        "分布和事件计数审计（详见 `Code/Hourly_Attenuation_Experiments/Revision_017/"
        "TEST_ACCESS_NOTE.md`），但没有据此调整冻结的方案。随机逐小时划分也可能使相邻小时"
        "跨集合；因此这些是锁定本版决策后的保留集指标，不能称为全新盲测或"
        "独立跨时段/异站泛化。测试结果没有被用于反向挑选模型或层。",
        "",
        "## 核验与复现",
        "",
        f"特征筛查指纹：`{features['fingerprint']}`；十模型指纹："
        f"`{model_qa['fingerprint']}`；分层指纹：`{layer_qa['fingerprint']}`。",
        "十模型 QA、1024 组合 QA 和最终测试 QA 均为 PASS；"
        "详细实验顺序见 `Code/Hourly_Attenuation_Experiments/Revision_017/PROTOCOL.md`"
        " 与同目录 `CONTINUATION_RUNBOOK.md`。",
        "",
    ]
    path = OUT / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(report())
