"""Write a complete Chinese experiment summary only after every QA gate passes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_experiment as stage

OUT = stage.OUT
LAYERS = OUT / "layers"


def read_json(path: Path) -> dict:
    stage.require(path.exists(), f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    feature = read_json(stage.SCREEN)
    feature_qa = read_json(OUT / "feature_selection_qa.json")
    model_qa = read_json(stage.MODEL_QA)
    model = read_json(OUT / "model_selection.json")
    layer_qa = read_json(LAYERS / "final_qa.json")
    choice = read_json(LAYERS / "layer_selection.json")
    test = read_json(LAYERS / "final_test.json")
    manifest = read_json(LAYERS / "run_manifest.json")
    stage.require(feature_qa["status"] == model_qa["status"] == layer_qa["status"] == "PASS",
                  "At least one experiment QA gate has not passed")
    stage.require(feature["fingerprint"] == feature_qa["fingerprint"] ==
                  model_qa["feature_fingerprint"] == manifest["feature_fingerprint"] and
                  model_qa["candidate_trials"] == 40 and
                  model_qa["seed_validation_fits"] == 30 and
                  layer_qa["validated_subsets"] == 1024 and
                  layer_qa["validated_layer_pairs"] == 5120 and
                  layer_qa["fingerprint"] == test["fingerprint"],
                  "Incomplete or fingerprint-mismatched records")
    stage.require(len(model["model_rows"]) == 10 and test["n_test"] == 687 and
                  len(pd.read_csv(LAYERS / "layer_pairwise.csv")) == 5120,
                  "Expected complete model, test, and pair records")
    layer_summary = pd.read_csv(LAYERS / "layer_summary.csv")
    stage.require(len(layer_summary) == 10, "Expected one summary per pressure level")
    records = {
        "基础无层": read_json(LAYERS / "subsets" / "mask_0000.json"),
        "训练内部所选": read_json(
            LAYERS / "subsets" / f"mask_{choice['selected_mask_train_only']:04d}.json"),
        "全部十层": read_json(LAYERS / "subsets" / "mask_1023.json"),
    }

    lines = [
        "# Revision 018：0.90 高相关去冗余后的小时衰减实验",
        "",
        "本报告在特征筛选、十模型、1024 组合和最终测试 QA 均为 PASS 后生成。"
        "原始 H5、ERA5、Run_011、数据集和旧实验版本未修改。",
        "",
        "## 训练内特征工程",
        "",
        "目标为同小时 `clear_sky_attenuation_db`。训练、验证、测试分别为"
        " 2404、343、687 个质量合格小时。仅训练集的 348 个 ERA5 输入列参与筛选，"
        "不使用目标、事件、验证或测试值。先剔除高缺失与恒定列，再按成对"
        " `|Spearman rho|≥0.90` 建冗余图，以训练期缺失率低者优先、原列序"
        "打破平局，贪心保留无冗余边的列。每对至少有 200 个共同有效小时。",
        "",
        f"348 → **{feature['n_retained']}** 列，其中基础 {feature['retained_base_count']} "
        f"列、十气压面 {feature['n_retained']-feature['retained_base_count']} 列。"
        f"原始可定义高相关对 {feature['n_high_correlation_pairs_initial']} 对；"
        f"删除高缺失 {feature['n_dropped_missing']} 列、恒定/空值"
        f" {feature['n_dropped_constant']} 列、相关冗余"
        f" {feature['n_dropped_correlated']} 列。"
        f"筛后最高成对 `|rho|={feature['max_remaining_absolute_spearman']:.5f}`，"
        "所有可计算的保留列对均低于 0.90；这不等于完全统计独立。",
        "",
        "| 气压面 (hPa) | 保留输入列数 |",
        "|---:|---:|",
    ]
    for pressure, count in feature["retained_per_pressure_level"].items():
        lines.append(f"| {pressure} | {count} |")
    lines += [
        "",
        "完整保留列、每列去留原因及代表列见 `feature_selection.json`、"
        "`feature_audit.csv`；全部高相关对见 `high_correlation_pairs.csv`。"
        "垂直网络保留原固定槽位，但筛除槽位不包含观测信息。",
        "",
        "## 十模型：343 小时验证",
        "",
        "十类模型各试 4 套固定候选参数；验证总体 RMSE 选配置和最早最佳轮，"
        "再用 17、29、43 三个种子固定轮数重新拟合。模型主排名是验证总体"
        " RMSE 与对数单条推理延迟各 50%（分别 min-max 归一化）。"
        "天气事件只用于报告，雨天包含全部强雨。",
        "",
        "| 等权排名 | 模型 | 验证总体 RMSE | 雨天 RMSE | 干燥阴天 RMSE | 单条延迟 ms | 结束轮 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in model["model_rows"]:
        lines.append(
            f"| {row['rank_50_50']} | {row['name']} | "
            f"{fmt(row['overall_rmse_mean'])} | {fmt(row['rain_rmse_mean'])} | "
            f"{fmt(row['dry_overcast_rmse_mean'])} | "
            f"{fmt(row['latency_median_ms'], 4)} | "
            f"{row['formal_rounds'] if row['formal_rounds'] is not None else '求解器收敛'} |"
        )
    winner = next(row for row in model["model_rows"]
                  if row["name"] == model["primary_winner"])
    lines += [
        "",
        f"冻结规则选出 **{model['primary_winner']}**，其候选参数为"
        f" `{json.dumps(winner['best_config'], ensure_ascii=False)}`，"
        f"轮数 {winner['formal_rounds']}。纯总体 RMSE 排名第一为"
        f" **{model['rmse_only_winner']}**。"
        "每配置逐轮曲线、三种子固定轮数的逐小时验证预测、完整参数与停止原因"
        "存于 `trials/`、`validation/` 和 `model_selection.json`。",
        "",
        "## 十层 1024 组合",
        "",
        "固定获胜学习器，筛后基础列先建折外预测，再按每个气压面组合重新训练"
        "分层残差头。以下每层统计其在 512 个其他层搭配中的平均验证 RMSE"
        " 改善；正值表示加层后误差降低，负值表示升高。这些搭配相互相关，"
        "不作为 512 次独立样本。",
        "",
        "| 气压面 hPa | 总体平均改善 dB | 雨天平均改善 dB | 干燥阴天平均改善 dB | 总体改善比例 |",
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
        f"训练内部选择掩码 {choice['selected_mask_train_only']}，气压面"
        f" {choice['selected_levels_hpa']}，alpha={choice['selected_alpha']}，"
        f"残差正式轮数 {choice['selected_formal_rounds']}。"
        "全部 1024 掩码的精确特征名单、参数、三折逐轮曲线、训练内部得分、"
        "343 小时验证预测及事件指标位于 `layers/subsets/`；"
        "汇总表位于 `layers/subset_summary.csv`、`layers/layer_pairwise.csv`、"
        "`layers/layer_summary.csv`。",
        "",
        "| 分层方案 | 验证总体 RMSE | 雨天 RMSE | 干燥阴天 RMSE |",
        "|---|---:|---:|---:|",
    ]
    for label, record in records.items():
        metric = record["validation_metrics"]
        lines.append(
            f"| {label} | {fmt(metric['overall']['rmse'])} | "
            f"{fmt(metric['rain']['rmse'])} | "
            f"{fmt(metric['dry_overcast']['rmse'])} |"
        )
    lines += [
        "",
        "## 冻结方案后的 687 小时测试",
        "",
        "| 方案 | 总体 RMSE | 雨天 RMSE | 干燥阴天 RMSE | 总体 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metric in test["metrics"].items():
        lines.append(
            f"| {label} | {fmt(metric['overall']['rmse'])} | "
            f"{fmt(metric['rain']['rmse'])} | "
            f"{fmt(metric['dry_overcast']['rmse'])} | "
            f"{fmt(metric['overall']['mae'])} |"
        )
    base = test["metrics"]["Base188"]
    direct = test["metrics"]["Direct348"]
    selected = test["metrics"]["SelectedResidual"]
    lines += [
        "",
        f"筛后直接全特征模型相对筛后基础方案，测试总体/雨天/干燥阴天"
        f" RMSE 差值（基础减直接）分别为"
        f" {fmt(base['overall']['rmse']-direct['overall']['rmse'], 4)} /"
        f" {fmt(base['rain']['rmse']-direct['rain']['rmse'], 4)} /"
        f" {fmt(base['dry_overcast']['rmse']-direct['dry_overcast']['rmse'], 4)} dB。"
        f"训练内部所选残差方案相对基础方案的测试总体 RMSE 差值"
        f"（基础减残差）为"
        f" {fmt(base['overall']['rmse']-selected['overall']['rmse'], 4)} dB。"
        "所有差值只作预定方案评价，不用于回选模型或层。",
        "",
        f"内部名称 `Base188` 表示原 188 槽位中的筛后基础 {feature['retained_base_count']}"
        f" 列；`Direct348` 表示原 348 槽位中的筛后 {feature['n_retained']} 列。"
        "名称不表示被筛除列参与训练。",
        "",
        "这不是全新盲测：同一测试集在历史实验中已被查看，随机逐小时划分"
        "也可能使相邻小时跨集合。此结果不能替代独立跨时段或异站验证。",
        "",
        "## QA 与复现",
        "",
        f"特征筛查指纹 `{feature['fingerprint']}`；十模型指纹"
        f" `{model_qa['fingerprint']}`；分层指纹 `{layer_qa['fingerprint']}`。"
        "全部 40 候选、30 次三种子拟合、1024 掩码、5120 配对和 687"
        " 小时测试逐项复算通过。协议与断点续跑说明在"
        " `Code/Hourly_Attenuation_Experiments/Revision_018/`。",
        "",
    ]
    path = OUT / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
