"""Delivery-only renderer: no ML imports, dataset reads, fitting or selection.

Added after the frozen scientific run and independent delivery audit. It only
exports audited JSON evidence into readable tables; it is not modeling code.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced"
GROUPS = ("overall", "clear_dry", "rain", "dry_overcast", "other")
LABELS = dict(zip(GROUPS, ("总体", "晴天", "雨天", "干燥阴天", "其他天气")))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_csv(name, rows):
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(x):
    return "—" if x is None else str(x)


def main():
    m = read(OUT / "run_manifest.json")
    report = read(OUT / "final_report.json")
    plans = read(OUT / "selection.json")["plans"]
    delivery = read(OUT / "delivery_qa.json")
    assert delivery["status"] == "PASS" and report["fingerprint"] == m["fingerprint"]
    assert read(OUT / "active_worker.json")["status"] == "EXITED"
    for name, expected in m["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in delivery["output_sha256"].items():
        assert digest(OUT / name) == expected, name

    # Historical report is already among the frozen source files.
    old = read(ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_026_Ten_Model_Test/final_report.json")
    winner = next(r for r in report["model_rows"] if r["model"] == report["full_overall_winner"])
    prior = next(r for r in old["model_rows"] if r["model"] == winner["model"])
    means, parameters, curves, intervals, feature_lists = [], [], [], [], []
    for row in report["model_rows"]:
        name = row["model"]
        for case in ("base", "full"):
            for group in GROUPS:
                entry = row["metrics"][group]
                means.append(dict(rank_full_overall=row["full_overall_rmse_rank"], model=name,
                                  case=case, weather=group, n_hours=entry["n_hours"], **entry[case]))
            plan = plans[case][name]
            trial = read(OUT / "trials" / f"{case}_{name}_config{plan['config_id']}.json")
            for j, feature in enumerate(trial["feature_names"], 1):
                feature_lists.append(dict(model=name, case=case, index=j, feature=feature))
            for h in trial["round_history"]:
                curves.append(dict(stage="selected_search", case=case, model=name, seed=17,
                                   config_id=plan["config_id"], **h))
            for seed in m["seeds"]:
                training = read(OUT / "training" / f"{case}_{name}_seed{seed}.json")
                meta, search_meta = training["fit_metadata"], trial["fit_metadata"]
                parameters.append(dict(case=case, model=name, seed=seed,
                                       n_features=len(training["feature_names"]), config_id=plan["config_id"],
                                       weight_policy=plan["choice"]["weight_policy"],
                                       gated_fusion=plan["choice"]["gated_fusion"],
                                       search_actual_end=search_meta["trained_rounds"],
                                       search_best_round=search_meta["selected_round"],
                                       search_solver_iterations=search_meta["solver_iterations"],
                                       search_termination_reason=search_meta["selection_reason"],
                                       formal_frozen_rounds=plan["formal_rounds"],
                                       formal_actual_end=meta["trained_rounds"],
                                       formal_solver_iterations=meta["solver_iterations"],
                                       formal_termination_reason=meta["selection_reason"],
                                       choice_json=json.dumps(meta["effective_choice"], ensure_ascii=False, sort_keys=True),
                                       warnings_json=json.dumps(meta["warnings"], ensure_ascii=False),
                                       model_sha256=training["model_sha256"]))
                for h in training["round_history"]:
                    assert h["validation_objective_db"] is None
                    curves.append(dict(stage="formal_refit", case=case, model=name, seed=seed,
                                       config_id=plan["config_id"], **h))
        for group in m["acceptance_groups"]:
            ci = row["metrics"][group]["date_bootstrap"]
            intervals.append(dict(model=name, weather=group, n_dates=ci["n_dates"],
                                  base_minus_full_rmse_db=row["metrics"][group]["base_minus_full_rmse_db"],
                                  percentile95_lower_db=ci["percentile_95_db"][0],
                                  percentile95_upper_db=ci["percentile_95_db"][1],
                                  bonferroni_lower_db=ci["bonferroni_three_endpoint_bounds_db"][0],
                                  bonferroni_upper_db=ci["bonferroni_three_endpoint_bounds_db"][1]))

    assert len(means) == 100 and len(parameters) == 60 and len(intervals) == 30
    assert len(feature_lists) == 10 * (50 + 156)
    dump_csv("weather_mean_metrics.csv", means)
    dump_csv("training_parameters.csv", parameters)
    dump_csv("selected_and_formal_rounds.csv", curves)
    dump_csv("paired_date_intervals.csv", intervals)
    dump_csv("complete_feature_lists.csv", feature_lists)

    lines = ["# Revision 028 详细结果与训练说明", "",
             f"冻结指纹：{m['fingerprint']}。独立交付审计PASS；本轮预定均值验收：{report['acceptance']}。", "",
             "## 1. 可以下什么结论", "",
             "先按十个全层模型的测试总体RMSE均值排名，再检查第一名与本轮独立调参基础输入的配对结果；没有先筛选有利模型。",
             f"第一名为{winner['model_label']}。这里每个指标是种子17/29/43分别计算后取平均，不是先平均预测的集成指标。",
             "本轮MET是预定点估计条件通过，不代表每个种子、所有新数据或所有历史基准均改善，也不代表确定的物理因果贡献。",
             "本轮基础和全层参数独立选择，因此它是相同预算下两套完整训练方案的比较，不是固定一切参数后仅改变输入列数的消融。", "",
             "## 2. 获胜模型五种天气的完整测试指标", "",
             "同一687测试小时：晴天322、雨天51、干燥阴天158、其他156。雨天含全部雨天，不拆强雨。",
             "RMSE、MAE单位dB，越小越好；Bias=平均(预测−实测)，越接近0越好。", "",
             "|范围|小时数|基础RMSE|全层RMSE|RMSE降低|基础MAE|全层MAE|基础Bias|全层Bias|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        z = winner["metrics"][group]
        b, f = z["base"], z["full"]
        decrease = 100 * (1 - f["rmse_db"] / b["rmse_db"])
        lines.append(f"|{LABELS[group]}|{z['n_hours']}|{b['rmse_db']:.6f}|{f['rmse_db']:.6f}|{decrease:.2f}%|"
                     f"{b['mae_db']:.6f}|{f['mae_db']:.6f}|{b['bias_db']:+.6f}|{f['bias_db']:+.6f}|")
    lines += ["", "## 3. 不确定性与旧版比较", "",
              "日期配对区间对基础减全层RMSE计算：正数表示全层较好。4096次重采样整个观察日期，三个种子固定不重采样。",
              "这些条件区间不修正历史测试访问、测试排名选择、多日依赖或同日训练重叠。", "",
              "|范围|配对日期数|基础−全层|95%日期区间|三端点Bonferroni区间|三种子改善情况|",
              "|---|---:|---:|---|---|---|"]
    for group in m["acceptance_groups"]:
        z = winner["metrics"][group]
        ci = z["date_bootstrap"]
        seed_summary = "; ".join(f"{s}:{'改善' if d > 0 else '恶化'}" for s, d in z["seed_deltas_db"].items())
        lines.append(f"|{LABELS[group]}|{ci['n_dates']}|{z['base_minus_full_rmse_db']:.6f}|"
                     f"[{ci['percentile_95_db'][0]:.6f}, {ci['percentile_95_db'][1]:.6f}]|"
                     f"[{ci['bonferroni_three_endpoint_bounds_db'][0]:.6f}, {ci['bonferroni_three_endpoint_bounds_db'][1]:.6f}]|{seed_summary}|")
    lines += ["", "雨天均值改善，但种子29恶化且日期区间跨0；不能称雨天稳定显著改善。", "",
              "下表是新旧全156列BiLSTM的三个种子均值，不与旧单种子17混比。负的RMSE变化表示新结果较好。", "",
              "|范围|旧Rev026全层RMSE|本轮全层RMSE|本轮相对旧版变化|", "|---|---:|---:|---:|"]
    for group in GROUPS:
        previous = prior["metrics"][group]["rmse_db_mean"]
        current = winner["metrics"][group]["full"]["rmse_db"]
        lines.append(f"|{LABELS[group]}|{previous:.6f}|{current:.6f}|{100*(current/previous-1):+.2f}%|")
    lines += ["", "因此，本轮配对改善不等于旧版雨天成绩进一步改善。旧版雨天/晴天负向变化完整保留。", "",
              "## 4. 十模型排名与全部天气结果", "",
              "排名为全层总体RMSE均值的纯精度排名，不采用50/50延迟评分。下表包括全部负结果。", "",
              "|全层排名|模型|输入|范围|小时数|RMSE/dB|MAE/dB|Bias/dB|",
              "|---:|---|---|---|---:|---:|---:|---:|"]
    for z in means:
        lines.append(f"|{z['rank_full_overall']}|{z['model']}|{50 if z['case']=='base' else 156}|{LABELS[z['weather']]}|"
                     f"{z['n_hours']}|{z['rmse_db']:.6f}|{z['mae_db']:.6f}|{z['bias_db']:+.6f}|")
    lines += ["", "## 5. 配置、最佳轮和实际结束轮", "",
              "2404小时仅用于拟合；343验证小时选择配置与轮数。每个输入池、每模型有8个候选位置（4普通、4天气感知扩展），合计160候选拟合。",
              "统一验证目标为sqrt((总体MSE+雨天MSE+干燥阴天MSE)/3)。天气标记只作权重/评价，不作为输入。",
              "随后冻结20计划，每个按17/29/43重拟合：60个正式模型、合计220次实际研究拟合。正式阶段不使用验证或测试来决定停止。",
              "神经网络搜索最多160轮，验证目标连续20轮未改善结束并恢复最佳轮。树模型先完成预设上限，再从已有树前缀选择最佳轮，不冒称提前结束。",
              "Elastic Net/SVR是求解器，不虚构神经训练轮次，实际求解迭代见training_parameters.csv。",
              "正式逐轮CSV中的验证列为空是有意隔离，搜索阶段逐轮验证曲线完整保留。", "",
              "|输入|模型|候选ID(0起)|权重|门控|搜索实际结束|搜索最佳轮|正式固定轮|完整搜索配置|",
              "|---|---|---:|---|---|---|---|---|---|"]
    for case in ("base", "full"):
        for row in report["model_rows"]:
            name = row["model"]
            plan = plans[case][name]
            trial = read(OUT / "trials" / f"{case}_{name}_config{plan['config_id']}.json")
            meta = trial["fit_metadata"]
            choice = json.dumps(plan["choice"], ensure_ascii=False, sort_keys=True)
            lines.append(f"|{50 if case=='base' else 156}|{row['model_label']}|{plan['config_id']}|"
                         f"{plan['choice']['weight_policy']}|{plan['choice']['gated_fusion']}|"
                         f"{value(meta['trained_rounds'])}|{value(meta['selected_round'])}|{value(plan['formal_rounds'])}|{choice}|")
    lines += ["", "本轮全层BiLSTM自然选中普通未门控配置：53轮、学习率0.001、dropout0.2、weight_decay0.001。",
              "基础BiLSTM选中天气加权配置：46轮、学习率0.001、dropout0.3、weight_decay0.01；不是强制分层有效。",
              "优化器AdamW、batch128，目标均值/标准差、填补和缩放只来自训练数据。各参数理由见PARAMETERS.md，门控为可学习直接融合而非残差头。", "",
              "## 6. 保存与审计", "",
              "训练、验证、测试精确小时互不相交。已有测试历史访问保持审计，156个测试日期均与训练日期重叠，不称全新盲测或时间外推结论。",
              "原始H5/ERA5、数据集、此前版本及慢变/快变冻结包均未修改。所有本轮失败/警告/负结果保留。", "",
              "- trials/：160候选完整逐轮曲线与逐小时训练/验证预测。",
              "- training/、models/、test/：60份正式记录、模型和测试预测。",
              "- per_seed_weather_metrics.csv：300个种子/输入/天气行；hourly_predictions.csv：41220逐小时行。",
              "- weather_mean_metrics.csv：100个模型/输入/天气均值行；training_parameters.csv：60个正式训练元数据行。",
              "- selected_and_formal_rounds.csv：20个获选搜索历史及60正式历史的全部逐轮数据；其他候选曲线仍在trials/。",
              "- complete_feature_lists.csv：2060个模型/输入保留参数名；paired_date_intervals.csv：30个配对区间行。",
              "- final_report.json、delivery_qa.json：独立复核全排名、均值、60种区间端点组和验收。",
              "- 本渲染器只整理已审计结果，科学代码与计划指纹未变；零次新拟合，零次测试后方案调整。", ""]
    with (OUT / "DETAILED_RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))

    # Round-trip scalar checks against already independently audited evidence.
    with (OUT / "weather_mean_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        stored = list(csv.DictReader(stream))
    assert len(stored) == len(means)
    for actual, expected in zip(stored, means):
        for metric in ("rmse_db", "mae_db", "bias_db"):
            assert float(actual[metric]) == expected[metric]
    with (OUT / "training_parameters.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 60
    with (OUT / "selected_and_formal_rounds.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == len(curves)
    files = ["DETAILED_RESULTS.md", "weather_mean_metrics.csv", "training_parameters.csv",
             "selected_and_formal_rounds.csv", "paired_date_intervals.csv", "complete_feature_lists.csv"]
    qa = dict(status="PASS", fingerprint=m["fingerprint"], modeling_source_files_unchanged=104,
              delivery_only_renderer_sha256=digest(Path(__file__)), weather_rows=100,
              parameter_rows=60, round_rows=len(curves), feature_list_rows=2060, interval_rows=30,
              source_metric_scalar_checks=300, new_fit_calls=0, post_test_plan_changes=0,
              output_sha256={f: digest(OUT / f) for f in files},
              utc=datetime.now(timezone.utc).isoformat())
    with (OUT / "detailed_report_qa.json").open("x", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
