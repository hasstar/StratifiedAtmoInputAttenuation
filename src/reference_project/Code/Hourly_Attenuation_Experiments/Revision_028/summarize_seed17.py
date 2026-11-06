"""Read existing frozen evidence and deliver a seed17-only slow-model report.

No model imports, dataset reads, training, inference or configuration selection.
Original three-seed scientific artifacts are neither modified nor replaced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced"
DEST = SOURCE / "Seed_017_Report"
GROUPS = ("overall", "clear_dry", "rain", "dry_overcast", "other")
ZH = dict(zip(GROUPS, ("总体", "晴天", "雨天", "干燥阴天", "其他天气")))
SEED = 17


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csv_write(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (DEST / name).open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def metric_table(lines, rows, title):
    lines.extend([f"## {title}", "", "单位dB；全部为单个seed=17模型，不做种子平均。", "",
                  "|集合|模型|输入列数|范围|小时数|日期数|RMSE|MAE|Bias|",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|"])
    for r in rows:
        split_name={"train":"训练", "validation":"验证", "test":"测试"}[r["split"]]
        lines.append(f"|{split_name}|{r['model_label']}|{r['n_features']}|{ZH[r['weather']]}|{r['n_hours']}|"
                     f"{r['n_dates']}|{r['rmse_db']:.6f}|{r['mae_db']:.6f}|{r['bias_db']:+.6f}|")
    lines.append("")


def main():
    assert not DEST.exists(), "Refusing to overwrite an existing seed17 report"
    m = load(SOURCE / "run_manifest.json")
    selection = load(SOURCE / "selection.json")
    delivery = load(SOURCE / "delivery_qa.json")
    old_report = load(SOURCE / "final_report.json")
    assert delivery["status"] == "PASS" and m["search_seed"] == SEED
    assert load(SOURCE / "active_worker.json")["status"] == "EXITED"
    for name, expected in m["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in delivery["output_sha256"].items():
        assert sha(SOURCE / name) == expected, name

    labels = {r["model"]: r["model_label"] for r in old_report["model_rows"]}
    source_hashes, metrics, parameters, hours, rounds = {}, [], [], [], []
    records, training_records = {}, {}
    max_metric_error = 0.0
    for case in ("base", "full"):
        features = m["base_names"] if case == "base" else m["retained_names"]
        for name in m["models"]:
            train_path = SOURCE / "training" / f"{case}_{name}_seed17.json"
            test_path = SOURCE / "test" / f"{case}_{name}_seed17.json"
            train, test = load(train_path), load(test_path)
            records[case, name], training_records[case, name] = test, train
            for p in (train_path, test_path):
                source_hashes[str(p.relative_to(SOURCE))] = sha(p)
            assert train["seed"] == test["seed"] == SEED
            assert train["fingerprint"] == test["fingerprint"] == m["fingerprint"]
            assert train["feature_names"] == test["feature_names"] == features
            assert train["plan"] == test["plan"] == selection["plans"][case][name]
            assert not train["test_read_for_fit"] and not train["validation_changed_formal_fit"]
            assert test["fit_calls_during_evaluation"] == 0
            model_path = SOURCE / "models" / f"{case}_{name}_seed17.joblib"
            assert sha(model_path) == train["model_sha256"] == test["model_sha256"]
            source_hashes[str(model_path.relative_to(SOURCE))] = sha(model_path)
            for group in GROUPS:
                mask = [True] * len(test["truth_db"]) if group == "overall" else test["weather_masks"][group]
                e = [p - y for p, y, yes in zip(test["prediction_db"], test["truth_db"], mask) if yes]
                raw = test["metrics"][group]
                values = dict(rmse_db=math.sqrt(math.fsum(x*x for x in e)/len(e)),
                              mae_db=math.fsum(abs(x) for x in e)/len(e),
                              bias_db=math.fsum(e)/len(e))
                assert len(e) == raw["n_hours"]
                for key, actual in values.items():
                    delta = abs(actual - raw[key]); max_metric_error = max(max_metric_error, delta)
                    assert delta < 1e-12
                for split, z in (("train", train["train_metrics"][group]),
                                 ("validation", train["validation_metrics"][group]), ("test", raw)):
                    metrics.append(dict(split=split, case=case, model=name, model_label=labels[name],
                                        seed=SEED, n_features=len(features), weather=group, **z))
            for i, h in enumerate(test["hour_bjt"]):
                hours.append(dict(case=case, model=name, seed=SEED, hour_bjt=h,
                                  truth_db=test["truth_db"][i], prediction_db=test["prediction_db"][i],
                                  error_db=test["prediction_db"][i]-test["truth_db"][i],
                                  **{g:bool(test["weather_masks"][g][i]) for g in GROUPS[1:]}))
            plan = selection["plans"][case][name]
            chosen = load(SOURCE / "trials" / f"{case}_{name}_config{plan['config_id']}.json")
            meta = train["fit_metadata"]
            parameters.append(dict(case=case, model=name, model_label=labels[name], seed=SEED,
                                   n_features=len(features), config_id=plan["config_id"],
                                   effective_choice=json.dumps(meta["effective_choice"], sort_keys=True),
                                   search_actual_end=chosen["fit_metadata"]["trained_rounds"],
                                   search_best_round=chosen["fit_metadata"]["selected_round"],
                                   search_reason=chosen["fit_metadata"]["selection_reason"],
                                   formal_actual_end=meta["trained_rounds"],
                                   formal_solver_iterations=meta["solver_iterations"],
                                   formal_reason=meta["selection_reason"],
                                   formal_fit_seconds=meta["fit_seconds"],
                                   warnings=json.dumps(meta["warnings"]), model_sha256=train["model_sha256"]))
            for h in train["round_history"]:
                assert h["validation_objective_db"] is None
                rounds.append(dict(stage="formal_refit", case=case, model=name, seed=SEED,
                                   config_id=plan["config_id"], **h))
            for config_id in range(8):
                p = SOURCE / "trials" / f"{case}_{name}_config{config_id}.json"
                trial = load(p); source_hashes[str(p.relative_to(SOURCE))] = sha(p)
                assert trial["fingerprint"] == m["fingerprint"]
                for h in trial["round_history"]:
                    rounds.append(dict(stage="candidate_search", case=case, model=name, seed=SEED,
                                       config_id=config_id, **h))

    ranking = sorted(m["models"], key=lambda n:(records["full",n]["metrics"]["overall"]["rmse_db"],
                                               records["full",n]["metrics"]["overall"]["mae_db"],n))
    assert len(metrics) == 300 and len(parameters) == 20 and len(hours) == 13740
    for name in m["models"]:
        assert records["base", name]["hour_bjt"] == records["full", name]["hour_bjt"]
        assert records["base", name]["truth_db"] == records["full", name]["truth_db"]
    lines = ["# Revision 028 慢变结果：固定seed=17专用汇总", "",
             "本报告只提取已有seed=17记录，没有重新训练、推理或改变配置，不包含种子平均。",
             "它是原三种子实验的单种子子集，不替换原冻结协议及三种子验收。后续用户已要求统一固定seed=17。", "",
             f"原科学指纹：{m['fingerprint']}。原独立QA均PASS，本报告另核验300项测试指标。", "",
             "## 1. 数据与指标口径", "",
             "训练2404小时、验证343小时、测试687小时。目标clear_sky_attenuation_db，单位dB，是小时相对晴空衰减，不是秒级快变。",
             "基础50列；全层为基础50+气压面106，共156有效气象参数。基础池含柱积分参数，不是全部只在地面的参数。",
             "训练只使用训练小时；验证选择各输入池/学习器的配置与最佳轮；冻结后测试评分，不用测试反向调参。",
             "天气标签只用于权重及统计，不是输入。十模型各输入池都有8个候选；seed17子集为20个正式模型。",
             "RMSE、MAE越小越好，Bias=平均(预测−实测)，越接近0越好。模型排名只用全层总体测试RMSE，不用推理延迟。",
             "测试晴天322、雨天51、干燥阴天158、其他156小时；雨天含全部雨天。", "",
             "## 2. 全层十模型测试总体排名", "",
             "|排名|模型|RMSE/dB|MAE/dB|Bias/dB|", "|---:|---|---:|---:|---:|"]
    for i, name in enumerate(ranking,1):
        z=records["full",name]["metrics"]["overall"]
        lines.append(f"|{i}|{labels[name]}|{z['rmse_db']:.6f}|{z['mae_db']:.6f}|{z['bias_db']:+.6f}|")
    lines += ["", f"全层总体第一名为{labels[ranking[0]]}。总体第一不表示每一种天气都在十模型中第一。", "",
              "## 3. BiLSTM基础与全层测试对照", "",
              "|范围|小时数|基础RMSE|全层RMSE|降低比例|基础MAE|全层MAE|基础Bias|全层Bias|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        b=records["base","bilstm"]["metrics"][g]; f=records["full","bilstm"]["metrics"][g]
        lines.append(f"|{ZH[g]}|{b['n_hours']}|{b['rmse_db']:.6f}|{f['rmse_db']:.6f}|{100*(1-f['rmse_db']/b['rmse_db']):.2f}%|"
                     f"{b['mae_db']:.6f}|{f['mae_db']:.6f}|{b['bias_db']:+.6f}|{f['bias_db']:+.6f}|")
    lines += ["", "本轮seed17的全部天气类别均改善，但基础与全层分别调参，是两套完整方案的比较，不是只改变列数的固定超参数消融。", ""]
    for case in ("full", "base"):
        selected=[r for name in ranking for r in metrics if r["split"]=="test" and r["case"]==case and r["model"]==name]
        metric_table(lines, selected, "4. 全层156列：十模型全部天气指标" if case=="full" else "5. 基础50列：十模型全部天气指标")
    metric_table(lines, [r for r in metrics if r["model"]=="bilstm" and r["split"]!="test"],
                 "6. BiLSTM训练/验证各天气指标（集合名另见CSV）")
    lines += ["训练/验证各15组指标在seed17_all_split_weather_metrics.csv中由split明确标识；训练分数不用于最终模型排名。", "",
              "## 7. 全部训练配置、轮数及结束原因", "",
              "神经网络搜索上限160轮、patience20，连续20轮验证目标未改善停止搜索并恢复最佳轮。正式重训按冻结轮数结束，不再依据验证或测试停止。",
              "树模型先训到搜索上限，再选最低验证目标的树前缀；搜索实际结束与正式树数不同。Elastic/SVR报告求解迭代，不虚构训练轮。",
              "优化器、架构及工程参数理由见Code/Hourly_Attenuation_Experiments/Revision_028/PARAMETERS.md。", "",
              "|模型|输入列数|搜索实际结束|最佳轮|正式结束|求解迭代|完整正式配置|",
              "|---|---:|---|---|---|---|---|"]
    for z in parameters:
        fmt=lambda x:"—" if x is None else str(x)
        lines.append(f"|{z['model_label']}|{z['n_features']}|{fmt(z['search_actual_end'])}|{fmt(z['search_best_round'])}|"
                     f"{fmt(z['formal_actual_end'])}|{fmt(z['formal_solver_iterations'])}|{z['effective_choice']}|")
    lines += ["", "BiLSTM基础：学习率0.001、dropout0.3、weight_decay0.01、天气加权、最佳46轮、搜索66轮结束、正式46轮。",
              "BiLSTM全层：学习率0.001、dropout0.2、weight_decay0.001、普通等权、最佳53轮、搜索73轮结束、正式53轮。",
              "AdamW,batch128,hidden32双向，垂直序列轴是气压面，不是未来小时；填补、缩放及目标标准化来自训练数据。", "",
              "## 8. 分层组合与RFE的版本边界", "",
              "Revision028未在本轮53轮模型上执行1024组分层重训或156→1 RFE，因此不存在当前模型的逐层贡献或RFE路径。",
              "旧Revision025的seed17、62轮BiLSTM已完成1024组直接重训及156→1 RFE，各QA PASS；不可把其数值当作当前53轮模型的分析。",
              f"旧版独立报告：{(ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_025_BiLSTM/RESULTS.md').as_posix()}。", "",
              "## 9. 历史同种子比较与解释界限", "",
              "以下只比较旧版和本轮seed17全层模型，不掺三种子均值。正的RMSE变化表示本轮更差。", "",
              "|范围|旧版62轮全层RMSE|当前53轮全层RMSE|变化|", "|---|---:|---:|---:|"]
    prior=load(ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_026_Ten_Model_Test/final_report.json")
    prior=next(r for r in prior["model_rows"] if r["model"]=="bilstm")
    for g in GROUPS:
        prev=prior["metrics"][g]["seed17"]["rmse_db"]; current=records["full","bilstm"]["metrics"][g]["rmse_db"]
        lines.append(f"|{ZH[g]}|{prev:.6f}|{current:.6f}|{100*(current/prev-1):+.2f}%|")
    lines += ["", "原三种子日期区间不是seed17单模型区间，本报告不挪用它，也不声称固定种子消除了数据不确定性。",
              "历史测试访问、156个测试日期与训练同日重叠仍保留审计；本报告不称新盲测。", "",
              "## 10. 全部证据文件", "",
              "- seed17_all_split_weather_metrics.csv：20模型×3集合×5范围=300行，RMSE/MAE/Bias齐全；其中测试100行。",
              "- seed17_training_parameters.csv：20个正式模型完整配置、警告、轮数/迭代和源模型指纹。",
              "- seed17_hourly_test_predictions.csv：13740行已保存逐小时测试预测，未重新推理。",
              "- seed17_all_rounds.csv：160搜索候选及20个正式模型的全部逐轮结果。",
              "- seed17_feature_lists.csv：50/156两套完整有效参数名单；模型文件仍在原models目录。",
              "- seed17_report_qa.json：300项测试指标独立复算、源/输出指纹和无新训练审计。", ""]

    DEST.mkdir()
    csv_write("seed17_all_split_weather_metrics.csv", metrics)
    csv_write("seed17_training_parameters.csv", parameters)
    csv_write("seed17_hourly_test_predictions.csv", hours)
    csv_write("seed17_all_rounds.csv", rounds)
    csv_write("seed17_feature_lists.csv", [dict(case=c,index=i,feature=n) for c in ("base","full")
              for i,n in enumerate(m["base_names"] if c=="base" else m["retained_names"],1)])
    (DEST/"SEED17_SLOW_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    for name, expected in delivery["output_sha256"].items():
        assert sha(SOURCE/name)==expected, "Original artifact changed"
    # A round-trip verifies the unrounded metric export; no averages are applied.
    with (DEST/"seed17_all_split_weather_metrics.csv").open(encoding="utf-8-sig",newline="") as f:
        reread=list(csv.DictReader(f))
    assert len(reread)==300 and all(r["seed"]=="17" for r in reread)
    for a,b in zip(reread,metrics):
        assert a["split"]==b["split"] and a["model"]==b["model"] and a["case"]==b["case"]
        for k in ("rmse_db","mae_db","bias_db"): assert float(a[k])==b[k]
    files=[p for p in DEST.iterdir() if p.is_file()]
    qa=dict(status="PASS", seed=SEED, parent_fingerprint=m["fingerprint"],
            formal_models=20, test_weather_rows=100, all_split_weather_rows=300,
            test_metric_checks=300, max_independent_test_metric_error_db=max_metric_error,
            csv_scalar_roundtrip_checks=900, hourly_test_rows=len(hours), round_rows=len(rounds),
            new_fit_calls=0,new_model_inference_calls=0,original_deliverables_preserved=True,
            fixed_model_seed_not_seed_average=True, renderer_sha256=sha(Path(__file__)),
            source_sha256=source_hashes, output_sha256={p.name:sha(p) for p in files},
            utc=datetime.now(timezone.utc).isoformat())
    (DEST/"seed17_report_qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in qa.items() if k not in ("source_sha256","output_sha256")},indent=2))


if __name__=="__main__": main()
