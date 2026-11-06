"""Readable reports and compact all-round Parquet exports from independently QA-passed results."""

from __future__ import annotations

import argparse
import json

import pandas as pd

import common as c


def effective_parameters(row: dict) -> dict:
    name = row["name"]
    cfg = dict(row["best_config"])
    rounds = row["formal_rounds"]
    if name == "elastic_net":
        return {**cfg, "max_iter": 20000, "tol": 1e-4, "solver": "sklearn coordinate descent"}
    if name == "svr_rbf":
        return {**cfg, "kernel": "rbf", "solver": "libsvm"}
    if name == "random_forest":
        return {**cfg, "n_estimators": rounds, "n_jobs": 2, "random_state_seeds": list(c.SEEDS)}
    if name == "xgboost":
        return {**cfg, "n_estimators": rounds, "objective": "reg:squarederror",
                "tree_method": "hist", "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_lambda": 1.0, "n_jobs": 2, "random_state_seeds": list(c.SEEDS)}
    if name == "lightgbm":
        return {**cfg, "n_estimators": rounds, "max_depth": -1,
                "min_child_samples": 20, "reg_lambda": 1.0, "n_jobs": 2,
                "random_state_seeds": list(c.SEEDS)}
    if name == "catboost":
        return {**cfg, "iterations": rounds, "loss_function": "RMSE",
                "thread_count": 2, "random_seed_values": list(c.SEEDS)}
    return {**cfg, "epochs": rounds, "optimizer": "AdamW", "batch_size": 128,
            "target_standardization": "training mean/std only", "seeds": list(c.SEEDS),
            "architecture": {"mlp": "348-64-32-1 ReLU Dropout",
                             "transformer": "10 tokens x16, 2 attention layers, 4 heads, width32 + base branch",
                             "gcn": "10 neighboring pressure nodes, 2 graph-convolution layers, width32 + base branch",
                             "bilstm": "vertical 10-step bidirectional 1-layer LSTM, 32 units/direction + base branch"}[name]}


def model_report() -> None:
    qa = json.loads((c.OUT / "model_selection_qa.json").read_text(encoding="utf-8"))
    selection = json.loads((c.OUT / "model_selection.json").read_text(encoding="utf-8"))
    c.require(qa["status"] == "PASS" and qa["fingerprint"] == selection["fingerprint"],
              "Ten-model QA gate failed")
    parameters = {}
    trace_rows = []
    weather_rows = []
    lines = ["# Revision 014：十模型训练和验证报告", "",
             "同一批 2404 训练小时、343 正式验证小时，均输入 348 个同小时大气参数。四套预置参数和停止轮次由正式验证集选择；每类再以固定配置、固定轮数和三个种子从头训练并在验证集比较。测试未用于选择。", "",
             "主选择规则：验证 RMSE 与已训练模型单条端到端推理延迟各占 50%，在十模型范围内分别归一化后相加。雨和强雨不参与主分数。", "",
             "| 综合排名 | 模型 | 最佳配置编号 | 候选轮数上限 | 正式轮数 | 候选最佳验证 RMSE | 三种子验证 RMSE | 验证 MAE | 单条延迟 ms | 雨 RMSE | 强雨 RMSE | 干燥阴天 RMSE | 仅RMSE排名 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selection["model_rows"]:
        name = row["name"]
        parameters[name] = effective_parameters(row)
        rounds = "不适用" if row["formal_rounds"] is None else str(row["formal_rounds"])
        best_trial = c.read_record(c.OUT / "trials" /
                                   f"{name}_{row['best_config_id']:02d}.json",
                                   selection["fingerprint"])
        max_rounds = "不适用" if best_trial["candidate_max_rounds"] is None else str(best_trial["candidate_max_rounds"])
        lines.append(f"| {row['rank_50_50']} | {name} | {row['best_config_id']} | {max_rounds} | {rounds} | "
                     f"{row['candidate_validation_rmse_db']:.4f} | {row['overall_rmse_mean']:.4f} | "
                     f"{row['overall_mae_mean']:.4f} | {row['latency_median_ms']:.4f} | "
                     f"{row['rain_rmse_mean']:.4f} | {row['heavy_rain_rmse_mean']:.4f} | "
                     f"{row['dry_overcast_rmse_mean']:.4f} | {row['rank_rmse_only']} |")
        for config_id in range(4):
            trial = c.read_record(c.OUT / "trials" / f"{name}_{config_id:02d}.json",
                                  selection["fingerprint"])
            for tick in trial["round_history"]:
                trace_rows.append({"stage": "candidate_validation", "model": name,
                                   "config_id": config_id, "seed": 17,
                                   "fold": None, "round": tick["round"],
                                   "train_rmse_db": tick["train_rmse_db"],
                                   "holdout_rmse_db": tick["holdout_rmse_db"],
                                   "selected_round": trial["formal_rounds"]})
        for seed in c.SEEDS:
            formal = c.read_record(c.OUT / "validation" / f"{name}_seed{seed}.json",
                                   selection["fingerprint"])
            for group in c.GROUPS:
                event = formal["metrics"][group]
                weather_rows.append({"model": name, "seed": seed, "weather_group": group,
                                     "n": event["n"], "days": event["days"],
                                     "rmse_db": event["rmse"], "mae_db": event["mae"],
                                     "bias_db": event["bias"]})
            for tick in formal["round_history"]:
                trace_rows.append({"stage": "formal_train", "model": name,
                                   "config_id": formal["config_id"], "seed": seed,
                                   "fold": None, "round": tick["round"],
                                   "train_rmse_db": tick["train_rmse_db"],
                                   "holdout_rmse_db": None,
                                   "selected_round": formal["formal_rounds"]})
    c.prior.atomic_json(c.OUT / "training_parameters.json", {
        "fingerprint": selection["fingerprint"], "models": parameters,
        "note": "Elastic Net/SVR have solver iterations, not tree/epoch curves"})
    pd.DataFrame(trace_rows).to_parquet(c.OUT / "ten_model_rounds.parquet", index=False)
    pd.DataFrame(weather_rows).to_csv(c.OUT / "ten_model_weather_metrics.csv", index=False)
    lines.extend(["", f"验证阶段综合胜者：**{selection['primary_winner']}**；仅 RMSE 胜者：**{selection['rmse_only_winner']}**。", "",
                  "## 十模型各天气的正式验证结果", "",
                  "以下 RMSE 是三个固定种子的算术平均，单位 dB；括号内是该类天气的验证小时数。逐种子的 RMSE、MAE、偏差和天数见 `ten_model_weather_metrics.csv`。雨包含强雨。", "",
                  "| 模型 | 晴朗干燥 | 雨 | 强雨 | 干燥阴天 | 雪 | >5 dB 深衰落 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in selection["model_rows"]:
        name = row["name"]
        cells = []
        for group in ("clear_dry", "rain", "heavy_rain", "dry_overcast", "snow", "deep_fade_gt5"):
            items = [item for item in weather_rows if item["model"] == name and item["weather_group"] == group]
            values = [item["rmse_db"] for item in items if item["rmse_db"] is not None]
            cells.append("—" if not values else f"{sum(values) / len(values):.4f} ({items[0]['n']})")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## 每类模型的实际训练参数", "",
                  "下表为四套候选中选中的那一套及固定轮数；完整结构定义和全部候选参数仍以协议、`training_parameters.json` 和逐配置 JSON 为准。Elastic Net/SVR 的数字是求解器迭代，不与树数/epoch 混同。", "",
                  "| 模型 | 实际参数 | 种子17求解器迭代 |",
                  "|---|---|---:|"])
    for row in selection["model_rows"]:
        name = row["name"]
        formal = c.read_record(c.OUT / "validation" / f"{name}_seed17.json",
                               selection["fingerprint"])
        params = json.dumps(parameters[name], ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"))
        solver = (str(formal["solver_iterations"])
                  if name in ("elastic_net", "svr_rbf") and formal["solver_iterations"] is not None
                  else "—")
        lines.append(f"| {name} | `{params}` | {solver} |")
    lines.extend(["",
                  "## 停止轮次依据", "",
                  "每类四套配置以种子 17 在完整训练集训练到预设上限，逐轮查看正式验证 RMSE，选最小值的最早轮，再对所选配置以 17/29/43 三个种子定长重训。正式训练在选定轮数结束，因为验证曲线在候选范围内于该轮最小；若所选轮次紧贴或达到上限，只能说是所试范围的最优轮，不能宣称已收敛。验证集用于调参和选轮，所以验证分数有选择偏差；最终测试不能反向改变决策。随机森林也是先训练到上限再选累计树数，不称为节约计算的真实早停。Elastic Net/SVR 的内部优化不是可比的 epoch，保存求解器迭代数及最终指标。", "",
                  "准确的停止依据记录在 `formal_rounds_reason`、验证逐轮曲线及冻结协议中；固定轮数重训不是再次利用测试集挑轮次。", "",
                  "`training_parameters.json` 列出每类模型的实际超参数；`ten_model_rounds.parquet` 汇总候选配置每轮训练/正式验证 RMSE 与定长重训的训练曲线。原始逐配置记录在 `trials/`；正式训练的三个种子在 `validation/`，其中还保存求解器迭代数、343 条逐小时预测、各天气指标和 300 次单条延迟。", "",
                  "正式验证雨 25 小时，其中强雨 5 小时；雪只有 1 小时，不能用于稳定排名。随机逐小时划分可能共享同一天气过程；本地测试此前已被查看，后续测试不能称全新盲测。", ""])
    (c.OUT / "MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "MODEL_REPORT_WRITTEN", "winner": selection["primary_winner"],
                      "round_rows": len(trace_rows)}, ensure_ascii=False))


def final_report() -> None:
    qa = json.loads((c.OUT / "layers" / "final_qa.json").read_text(encoding="utf-8"))
    c.require(qa["status"] == "PASS" and qa["stage"] == "all", "Final QA gate failed")
    selection = json.loads((c.OUT / "model_selection.json").read_text(encoding="utf-8"))
    layer_choice = json.loads((c.OUT / "layers" / "layer_selection.json").read_text(encoding="utf-8"))
    test = json.loads((c.OUT / "layers" / "final_test.json").read_text(encoding="utf-8"))
    cache = json.loads((c.OUT / "layers" / "baseline_cache.json").read_text(encoding="utf-8"))
    selected = json.loads((c.OUT / "layers" / "subsets" /
                           f"mask_{layer_choice['selected_mask_train_only']:04d}.json").read_text(encoding="utf-8"))
    full = json.loads((c.OUT / "layers" / "subsets" / "mask_1023.json").read_text(encoding="utf-8"))
    direct = json.loads((c.OUT / "validation" /
                         f"{selection['primary_winner']}_seed17.json").read_text(encoding="utf-8"))
    c.require(qa["fingerprint"] == layer_choice["fingerprint"] == test["fingerprint"],
              "Layer/test fingerprints differ")
    lines = ["# Revision 014：十模型与十层残差贡献完整结果", "",
             f"十模型验证选择：**{selection['primary_winner']}**（按预定 50% RMSE＋50% 单条延迟）。训练内部选出的残差层掩码：**{layer_choice['selected_mask_train_only']}**；气压面：**{layer_choice['selected_levels_hpa']} hPa**；残差系数 **{layer_choice['selected_alpha']}**；正式残差树数 **{layer_choice['selected_formal_rounds']}**。", "",
             "训练 2404、验证 343、测试 687 小时；测试只在十模型和 1024 个组合的训练/验证方案及核验完成后评分。该测试集历史上已被查看，不能称新的完全盲测。", "",
             "## 锁定方案的正式验证结果", "",
             "下表是预先指定的四个对照，不是从 1024 组中按验证成绩挑最好的一组。单位为 dB；雨包含强雨。", "",
             "| 方法 | 总体 n / RMSE / MAE | 晴朗干燥 n / RMSE | 雨 n / RMSE | 强雨 n / RMSE | 干燥阴天 n / RMSE | 雪 n / RMSE | >5 dB 深衰落 n / RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    validation_metrics = {"Base188": cache["full_train"]["base_validation_metrics"],
                          "SelectedResidual": selected["validation_metrics"],
                          "Full10Residual": full["validation_metrics"],
                          "Direct348": direct["metrics"]}
    def metric_row(arm, met):
        def fmt(group):
            item = met[group]
            value = "—" if item["rmse"] is None else f"{item['rmse']:.4f}"
            return f"{item['n']} / {value}"
        return (f"| {arm} | {met['overall']['n']} / {met['overall']['rmse']:.4f} / "
                f"{met['overall']['mae']:.4f} | {fmt('clear_dry')} | {fmt('rain')} | "
                f"{fmt('heavy_rain')} | {fmt('dry_overcast')} | {fmt('snow')} | "
                f"{fmt('deep_fade_gt5')} |")
    for arm in ("Base188", "SelectedResidual", "Full10Residual", "Direct348"):
        lines.append(metric_row(arm, validation_metrics[arm]))
    lines.extend(["",
             "## 最终测试总体与天气结果", "",
             "下表单位为 dB；雨包含强雨。测试雪只有 2 小时，仅作个案。", "",
             "| 方法 | 总体 n / RMSE / MAE | 晴朗干燥 n / RMSE | 雨 n / RMSE | 强雨 n / RMSE | 干燥阴天 n / RMSE | 雪 n / RMSE | >5 dB 深衰落 n / RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm in ("Base188", "SelectedResidual", "Full10Residual", "Direct348"):
        lines.append(metric_row(arm, test["metrics"][arm]))
    selected_fold_rounds = [fold["selected_round"] for fold in selected["folds"]]
    full_fold_rounds = [fold["selected_round"] for fold in full["folds"]]
    base_cv = json.loads((c.OUT / "layers" / "subsets" /
                          "mask_0000.json").read_text(encoding="utf-8"))["cv_pooled_rmse_db"]
    lines.extend(["", "## 分层残差的停止与泛化核对", "",
                  f"训练内部按日分组留出中，所选 925＋300 hPa 组合三次各自的最佳树数为 {selected_fold_rounds}，取中位数 {selected['formal_rounds']} 作为最终训练树数；全十层的三次最佳树数为 {full_fold_rounds}，最终树数 {full['formal_rounds']}。每次最佳树数是在预设最多 120 棵内由该次留出残差 RMSE 的最小值决定；随后按训练内部衰减 RMSE 选残差系数。", "",
                  f"所选残差组合将训练内部交叉验证 RMSE 从 Base188 的 {base_cv:.4f} 降到 {selected['cv_pooled_rmse_db']:.4f} dB，但正式验证由 {validation_metrics['Base188']['overall']['rmse']:.4f} 升到 {validation_metrics['SelectedResidual']['overall']['rmse']:.4f} dB，最终测试由 {test['metrics']['Base188']['overall']['rmse']:.4f} 升到 {test['metrics']['SelectedResidual']['overall']['rmse']:.4f} dB。因此本次实验不支持“加入分层残差必然提高未见小时精度”；直接以 348 个气象参数拟合的模型在本次固定测试中的总体 RMSE 更低。", ""])
    layer_summary = pd.read_csv(c.OUT / "layers" / "layer_summary.csv")
    lines.extend(["", "## 每层在 512 种其他层搭配中的验证贡献", "",
                  "数值为未加入该层 RMSE 减去加入该层 RMSE；正数表示该搭配下加入该层较好。基于验证集的 512 个搭配不是 512 个独立实验，不作独立样本显著性检验。", "",
                  "| 气压面 hPa | 平均差 dB | 中位差 dB | 改善比例 | 雨平均差 dB | 强雨平均差 dB | 干燥阴天平均差 dB |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in layer_summary.itertuples(index=False):
        lines.append(f"| {row.level_hpa} | {row.overall_mean_improvement_db:.4f} | "
                     f"{row.overall_median_improvement_db:.4f} | "
                     f"{row.overall_fraction_improved:.1%} | {row.rain_mean_improvement_db:.4f} | "
                     f"{row.heavy_rain_mean_improvement_db:.4f} | "
                     f"{row.dry_overcast_mean_improvement_db:.4f} |")
    rows = []
    for mask in range(1024):
        item = json.loads((c.OUT / "layers" / "subsets" / f"mask_{mask:04d}.json").read_text(encoding="utf-8"))
        for fold in item["folds"]:
            for tick in fold["round_history"]:
                rows.append({"mask": mask, "stage": "train_internal_cv", "fold": fold["fold_id"],
                             "round": tick["round"],
                             "train_residual_rmse_db": tick["train_residual_rmse_db"],
                             "holdout_residual_rmse_db": tick["holdout_residual_rmse_db"],
                             "selected_round": fold["selected_round"]})
        for tick in item["final_train_round_history"]:
            rows.append({"mask": mask, "stage": "formal_train", "fold": None,
                         "round": tick["round"],
                         "train_residual_rmse_db": tick["train_residual_rmse_db"],
                         "holdout_residual_rmse_db": None,
                         "selected_round": item["formal_rounds"]})
    pd.DataFrame(rows).to_parquet(c.OUT / "layers" / "layer_rounds.parquet", index=False)
    lines.extend(["", "## 参数、轮次与审计位置", "",
                  "`MODEL_REPORT.md` 与 `training_parameters.json` 记录十模型每个胜出配置、正式验证选轮及定长重训规则；`ten_model_rounds.parquet` 保存十模型每轮记录。`layers/subsets/mask_XXXX.json` 对每组保存完整特征名、三次训练内部按日分组留出的逐树误差、每次最佳树数、正式树数、修正系数、验证逐小时预测及天气指标；`layers/layer_rounds.parquet` 汇总 1024 组全部逐树曲线。", "",
                  "本版没有根据测试结果回头改变模型或分层组合。随机逐小时划分可能使同一天气过程跨集合；测试曾在历史实验中被查看，不支持声称全新盲测或跨地点泛化。435 个质量不合格小时不在回归表中，可能包含真实深衰落。", ""])
    (c.OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "FINAL_REPORT_WRITTEN", "layer_round_rows": len(rows)},
                     ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("model", "final"))
    stage = parser.parse_args().stage
    if stage == "model":
        model_report()
    else:
        final_report()


if __name__ == "__main__":
    main()
