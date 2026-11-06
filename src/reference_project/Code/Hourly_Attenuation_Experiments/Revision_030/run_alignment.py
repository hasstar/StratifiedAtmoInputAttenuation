"""Read-only lineage alignment. No model fitting, inference, or source mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
R28 = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced"
R29 = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE"
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_030_Original_Training_Aligned"
FP28 = "168ec778ecbdff46a91e5edfd4704b26f4c11ca81e1ffbf948416001677eab3a"
FP29 = "cdda7a1eaa7e14477a8edef39702d5b5e34e03f2a29a337d01b91c88ba57b4cd"
WEATHER = ["overall", "clear_dry", "rain", "dry_overcast", "other"]
WNAME = dict(zip(WEATHER, ["总体", "晴天", "雨天（含全部强雨）", "干燥阴天", "其他天气"]))
LABEL = {"elastic_net": "ElasticNet", "svr_rbf": "SVR-RBF", "random_forest": "RandomForest",
         "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "mlp": "MLP",
         "transformer": "Transformer", "gcn": "GCN", "bilstm": "BiLSTM"}
SOURCES: dict[str, str] = {}
TOL = 1e-12


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def track(path: Path, expected: str | None = None) -> str:
    path = path.resolve()
    assert path.is_relative_to(ROOT) and not path.is_relative_to(OUT), str(path)
    value = sha(path)
    key = relative(path)
    if key in SOURCES:
        assert SOURCES[key] == value, f"Source changed: {key}"
    if expected is not None:
        assert value == expected, f"Frozen hash mismatch: {key}"
    SOURCES[key] = value
    return value


def read(path: Path) -> dict:
    track(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    assert sha(path) == SOURCES[relative(path)]
    return value


def compare_metrics(actual: dict, stored: dict) -> None:
    for field in ("n_hours", "n_dates"):
        assert actual[field] == stored[field], (field, actual, stored)
    for field in ("rmse_db", "mae_db", "bias_db"):
        assert abs(actual[field] - stored[field]) <= TOL, (field, actual, stored)


def metrics(record: dict) -> dict:
    hours, truth, pred = record["hour_bjt"], record["truth_db"], record["prediction_db"]
    assert len(hours) == len(truth) == len(pred) and len(set(hours)) == len(hours)
    assert all(math.isfinite(x) for x in truth + pred)
    masks = record["weather_masks"]
    assert set(masks) == set(WEATHER[1:])
    for mask in masks.values():
        assert len(mask) == len(hours) and all(type(x) is bool for x in mask)
    assert all(sum(masks[w][i] for w in WEATHER[1:]) == 1 for i in range(len(hours)))
    result = {}
    for weather in WEATHER:
        indices = list(range(len(hours))) if weather == "overall" else [i for i, take in enumerate(masks[weather]) if take]
        assert indices
        errors = [pred[i] - truth[i] for i in indices]
        n = len(errors)
        result[weather] = {"n_hours": n, "n_dates": len({hours[i][:10] for i in indices}),
                           "rmse_db": math.sqrt(math.fsum(x * x for x in errors) / n),
                           "mae_db": math.fsum(abs(x) for x in errors) / n,
                           "bias_db": math.fsum(errors) / n}
        compare_metrics(result[weather], record["metrics"][weather])
    return result


def same_samples(first: dict, second: dict, predictions: bool = False) -> None:
    for field in ("hour_bjt", "truth_db", "weather_masks"):
        assert first[field] == second[field], field
    if predictions:
        assert first["prediction_db"] == second["prediction_db"]


def write(name: str, value: object) -> str:
    path = (OUT / name).resolve()
    assert path.is_relative_to(OUT) and path.parent == OUT.resolve()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return sha(path)


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["|" + "|".join(str(value) for value in row) + "|" for row in rows])


def build() -> None:
    assert not OUT.exists(), "Output directory already exists; no overwriting or automatic rerun."
    manifests = [read(R28 / "run_manifest.json"), read(R29 / "run_manifest.json")]
    for manifest, fingerprint in zip(manifests, [FP28, FP29]):
        assert manifest["fingerprint"] == fingerprint
        for name, digest in manifest["source_sha256"].items():
            track(ROOT / name.replace("\\", "/"), digest)
    m28, m29 = manifests
    assert (m28["n_train"], m28["n_validation"], m28["n_test"]) == (2404, 343, 687)
    assert (m29["layer_states"], m29["rfe_states"], m29["epochs"], m29["model_seed"]) == (1024, 156, 53, 17)
    for name in ["selection_qa.json", "train_qa.json", "independent_qa.json", "delivery_qa.json", "detailed_report_qa.json"]:
        qa = read(R28 / name)
        assert qa["status"] == "PASS" and qa["fingerprint"] == FP28
        for filename, digest in qa.get("output_sha256", {}).items():
            track(R28 / filename.replace("\\", "/"), digest)
    single_qa = read(R28 / "Seed_017_Report/seed17_report_qa.json")
    assert single_qa["status"] == "PASS" and single_qa["parent_fingerprint"] == FP28 and single_qa["seed"] == 17
    for name, digest in single_qa["source_sha256"].items():
        track(R28 / name.replace("\\", "/"), digest)
    for name, digest in single_qa["output_sha256"].items():
        track(R28 / "Seed_017_Report" / name.replace("\\", "/"), digest)
    for name in ["synthetic_qa.json", "anchor_qa.json"]:
        qa = read(R29 / name)
        assert qa["status"] == "PASS"
        for filename, digest in qa["code_sha256"].items():
            track(ROOT / "Code/Hourly_Attenuation_Experiments/Revision_029" / filename, digest)
        if name == "synthetic_qa.json":
            assert qa["research_samples_used"] == 0 and qa["synthetic_fit_calls"] == 19
        else:
            assert qa["epochs"] == 53 and qa["seed"] == 17 and qa["official_holdout_samples_used"] == 0
            assert qa["train_prediction_max_difference_db"] == qa["round_rmse_max_difference_db"] == 0
            assert qa["network_weights_equal"] is True and qa["preprocessing_equal"] is True
    for name in ["train_qa.json", "evaluation_qa.json", "report_qa.json"]:
        qa = read(R29 / name)
        assert qa["status"] == "PASS" and qa["fingerprint"] == FP29
    completion = read(R29 / "completion.json")
    assert completion["status"] == "COMPLETE" and completion["fingerprint"] == FP29
    for field, name in [("train_qa_sha256", "train_qa.json"), ("evaluation_qa_sha256", "evaluation_qa.json"),
                        ("report_qa_sha256", "report_qa.json"), ("final_report_sha256", "final_report.json"), ("results_sha256", "RESULTS.md")]:
        track(R29 / name, completion[field])
    original_selection = read(R28 / "selection.json")
    selection29 = read(R29 / "selection.json")
    assert selection29["fingerprint"] == FP29
    old_report29 = read(R29 / "final_report.json")
    parameters, original, original_raw = [], [], {}
    reference = None
    for case in ["base", "full"]:
        names = m28["base_names"] if case == "base" else m28["retained_names"]
        for model in m28["models"]:
            stem = f"{case}_{model}_seed17"
            train = read(R28 / "training" / f"{stem}.json")
            test = read(R28 / "test" / f"{stem}.json")
            assert train["fingerprint"] == test["fingerprint"] == FP28
            assert train["case"] == test["case"] == case and train["model"] == test["model"] == model
            assert train["seed"] == test["seed"] == 17
            assert train["feature_names"] == test["feature_names"] == names
            assert train["plan"] == test["plan"] == original_selection["plans"][case][model]
            assert train["selection_sha256"] == SOURCES[relative(R28 / "selection.json")]
            model_hash = track(R28 / "models" / f"{stem}.joblib", train["model_sha256"])
            assert model_hash == test["model_sha256"] and test["fit_calls_during_evaluation"] == 0
            assert train["test_read_for_fit"] is False and train["validation_changed_formal_fit"] is False
            assert len(train["train_hour_bjt"]) == 2404 and len(train["validation_hour_bjt"]) == 343 and len(test["hour_bjt"]) == 687
            pools = [set(train["train_hour_bjt"]), set(train["validation_hour_bjt"]), set(test["hour_bjt"])]
            assert all(len(pool) == size for pool, size in zip(pools, [2404, 343, 687]))
            assert all(not pools[i] & pools[j] for i, j in [(0, 1), (0, 2), (1, 2)])
            if reference is None:
                reference = test
            same_samples(reference, test)
            score = metrics(test)
            original.append({"case": case, "model": model, "seed": 17, "feature_names": names,
                             "source_record": relative(R28 / "test" / f"{stem}.json"), "metrics": score})
            original_raw[(case, model)] = test
            parameters.append({"case": case, "model": model, "seed": 17, "n_features": len(names),
                               "original_frozen_plan": train["plan"], "fit_metadata": train["fit_metadata"],
                               "model_sha256": model_hash, "training_record": relative(R28 / "training" / f"{stem}.json"),
                               "round_history": train["round_history"]})
    assert len(original) == len(parameters) == 20
    base_train = next(row for row in parameters if row["case"] == "base" and row["model"] == "bilstm")
    full_train = next(row for row in parameters if row["case"] == "full" and row["model"] == "bilstm")
    assert base_train["original_frozen_plan"]["formal_rounds"] == 46
    assert full_train["original_frozen_plan"]["formal_rounds"] == 53
    base_choice = base_train["fit_metadata"]["effective_choice"]
    full_choice = full_train["fit_metadata"]["effective_choice"]
    assert base_choice["weight_policy"] == "balanced" and base_choice["gated_fusion"] is True
    assert full_choice["weight_policy"] == "uniform" and full_choice["gated_fusion"] is False
    assert len(base_train["round_history"]) == 46 and len(full_train["round_history"]) == 53
    rank = sorted([row for row in original if row["case"] == "full"], key=lambda row: (row["metrics"]["overall"]["rmse_db"], row["model"]))
    ranking = [{"rank": i + 1, "model": row["model"], "metrics": row["metrics"]} for i, row in enumerate(rank)]
    comparison = []
    original_base = metrics(original_raw[("base", "bilstm")])
    original_full = metrics(original_raw[("full", "bilstm")])
    for weather in WEATHER:
        a, b = original_base[weather], original_full[weather]
        comparison.append({"weather": weather, "base_original_46": a, "full_original_53": b,
                           "rmse_difference_base_minus_full_db": a["rmse_db"] - b["rmse_db"],
                           "rmse_reduction_percent": 100 * (a["rmse_db"] - b["rmse_db"]) / a["rmse_db"]})

    fixed, by_kind, raw_highlights = [], {"layer": {}, "rfe": {}}, {}
    for kind, keys, prefix in [("layer", range(1024), "mask"), ("rfe", range(1, 157), "k")]:
        files = sorted((R29 / kind / "evaluation").glob("*.json"))
        assert len(files) == len(keys)
        for key in keys:
            stem = f"{prefix}_{key:04d}"
            path = R29 / kind / "evaluation" / f"{stem}.json"
            record = read(path)
            assert record["fingerprint"] == FP29 and record["kind"] == kind and record["key"] == key
            assert record["seed"] == 17 and record["epochs"] == 53 and record["fit_calls_during_evaluation"] == 0
            assert record["selection_sha256"] == SOURCES[relative(R29 / "selection.json")]
            assert set(record["feature_names"]) <= set(m28["retained_names"])
            assert len(set(record["feature_names"])) == len(record["feature_names"])
            track(R29 / kind / "models" / f"{stem}.joblib", record["model_sha256"])
            scores = {}
            for split, size in [("validation", 343), ("test", 687)]:
                assert len(record["results"][split]["hour_bjt"]) == size
                if split == "test":
                    same_samples(reference, record["results"][split])
                scores[split] = metrics(record["results"][split])
            row = {"kind": kind, "key": key, "seed": 17, "epochs": 53,
                   "feature_names": record["feature_names"], "metrics": scores, "source_record": relative(path)}
            fixed.append(row)
            by_kind[kind][key] = row
            if (kind, key) in [("layer", 0), ("layer", 1023), ("layer", 64), ("rfe", 156), ("rfe", 5), ("rfe", 10), ("rfe", 1)]:
                raw_highlights[(kind, key)] = record
                state = read(R29 / kind / "states" / f"{stem}.json")
                assert state["fingerprint"] == FP29 and state["official_holdouts_used_for_training_or_deletion"] is False
                assert state["feature_names"] == record["feature_names"]
                assert state["full_fit"]["fit_metadata"]["choice"] == full_choice
                assert len(state["full_fit"]["round_history"]) == 53 and len(state["folds"]) == 3
    for kind, key in [("layer", 1023), ("rfe", 156)]:
        same_samples(original_raw[("full", "bilstm")], raw_highlights[(kind, key)]["results"]["test"], predictions=True)
        assert by_kind[kind][key]["feature_names"] == m28["retained_names"]
    assert by_kind["layer"][0]["feature_names"] == m28["base_names"]
    effects = []
    for bit, level in enumerate(m29["levels_hpa"]):
        absent = [mask for mask in range(1024) if not (mask & (1 << bit))]
        assert len(absent) == 512
        for weather in WEATHER:
            deltas = [by_kind["layer"][mask]["metrics"]["test"][weather]["rmse_db"] -
                      by_kind["layer"][mask | (1 << bit)]["metrics"]["test"][weather]["rmse_db"] for mask in absent]
            ordered = sorted(deltas)
            effects.append({"level_hpa": level, "weather": weather, "contexts": 512,
                            "mean_rmse_improvement_db": math.fsum(deltas) / 512,
                            "median_rmse_improvement_db": (ordered[255] + ordered[256]) / 2,
                            "fraction_helpful": sum(value > 0 for value in deltas) / 512,
                            "full_context_rmse_improvement_db": by_kind["layer"][1023 ^ (1 << bit)]["metrics"]["test"][weather]["rmse_db"] - original_full[weather]["rmse_db"]})
    for actual, stored in zip(effects, old_report29["layer_effects"]):
        assert actual["level_hpa"] == stored["level_hpa"] and actual["weather"] == stored["weather"]
        for field in ["mean_rmse_improvement_db", "median_rmse_improvement_db", "fraction_helpful", "full_context_rmse_improvement_db"]:
            assert abs(actual[field] - stored[field]) <= TOL
    survival = []
    for k in range(1, 157):
        current = set(by_kind["rfe"][k]["feature_names"])
        assert len(current) == k
        removed = current if k == 1 else current - set(by_kind["rfe"][k - 1]["feature_names"])
        assert len(removed) == 1 and (k == 1 or set(by_kind["rfe"][k - 1]["feature_names"]) < current)
        name = next(iter(removed))
        survival.append({"survival_rank": k, "feature_name": name, "type": "pressure" if name.startswith("p") else "base"})
    assert [r["feature_name"] for r in survival[:10]] == [r["feature_name"] for r in old_report29["top10_survival"]]
    for kind in ["layer", "rfe"]:
        selected = selection29["selections"][kind]
        assert selected["official_holdouts_used_for_selection"] is False
        assert selected["feature_names"] == by_kind[kind][selected["selected_key"]]["feature_names"]

    baseline_trace = [{"id": "rev028_original_base46", "source": original_raw[("base", "bilstm")]["model_sha256"],
                       "choice": base_choice, "metrics": original_base},
                      {"id": "rev029_uniform_base53", "source": raw_highlights[("layer", 0)]["model_sha256"],
                       "choice": full_choice, "metrics": by_kind["layer"][0]["metrics"]["test"]},
                      {"id": "rev028_original_full53", "source": original_raw[("full", "bilstm")]["model_sha256"],
                       "choice": full_choice, "metrics": original_full}]
    summary = {"scope": "report_alignment_no_new_training", "seed": 17, "seed_average": False,
               "original_primary_comparison": comparison, "original_full_model_ranking": ranking,
               "baseline_trace": baseline_trace, "fixed53_layer_rfe_selection_unchanged": selection29,
               "fixed53_highlights": [by_kind[kind][key] for kind, key in raw_highlights],
               "layer_effects": effects, "rfe_survival_rank": survival,
               "new_fit_calls": 0, "new_model_inference_calls": 0, "new_hyperparameter_searches": 0,
               "new_inferential_tests": 0, "historical_test_access": True,
               "source_model_replay_qa_reused_not_rerun": True}
    claims = [
        {"id": "C1", "claim": "原主比较绑定Rev028基础46轮/全层53轮，seed17，原配置不变。", "status": "SUPPORTED", "evidence": [relative(R28 / "selection.json"), relative(R28 / "training/base_bilstm_seed17.json"), relative(R28 / "training/full_bilstm_seed17.json")]},
        {"id": "C2", "claim": "BiLSTM在十个原全层seed17模型中总体测试RMSE排名第一。", "status": "SUPPORTED", "evidence": "original_full_model_ranking", "scope": "同一原687测试小时，不保证新数据排名"},
        {"id": "C3", "claim": "原主比较的全层RMSE在全部五评价范围均低于原基础。", "status": "SUPPORTED", "evidence": "original_primary_comparison", "scope": "各输入池独立验证选参的完整方案比较，不是特征单因素消融或显著性结论"},
        {"id": "C4", "claim": "Rev029固定53轮基础雨天RMSE低于全层；该负结果仍有效。", "status": "SUPPORTED", "evidence": "baseline_trace", "scope": "不得将此基线误写成Rev028原基础46轮"},
        {"id": "C5", "claim": "1024层组合、RFE删除路径与one-SE选择未改。", "status": "SUPPORTED", "evidence": [relative(R29 / "train_qa.json"), relative(R29 / "selection.json"), "rfe_survival_rank"], "scope": "本次只从保存预测复算；原模型/PFI重放证据来自原QA"},
        {"id": "C6", "claim": "所有超参数相同、各天气加层都必然改善。", "status": "NOT_SUPPORTED", "evidence": "baseline_trace"},
        {"id": "C7", "claim": "本次获得新的训练改善模型或新的盲测证据。", "status": "NOT_SUPPORTED", "evidence": "scope/new_fit_calls/historical_test_access"},
        {"id": "C8", "claim": "RFE小模型已保留全层精度，或雨天改善已证明显著。", "status": "NOT_SUPPORTED", "evidence": "fixed53_highlights/new_inferential_tests"},
    ]
    assert ranking[0]["model"] == "bilstm"
    assert all(row["rmse_difference_base_minus_full_db"] > 0 for row in comparison)
    assert baseline_trace[1]["metrics"]["rain"]["rmse_db"] < original_full["rain"]["rmse_db"]
    code_sha = {path.name: sha(path) for path in CODE.iterdir() if path.name in ["PROTOCOL.md", "PARAMETERS.md", "CONTINUATION_RUNBOOK.md", "run_alignment.py", "verify_alignment.py"]}
    assert len(code_sha) == 5
    for key, digest in SOURCES.items():
        assert sha(ROOT / key) == digest, f"Source changed before delivery: {key}"
    OUT.mkdir(parents=True)
    source_hash = write("source_snapshot.json", {"source_sha256": dict(sorted(SOURCES.items())), "count": len(SOURCES)})
    manifest = {"version": "rev030-original-training-aligned-report-v1", "type": "report_only",
                "parent_fingerprints": [FP28, FP29], "code_sha256": code_sha, "source_snapshot_sha256": source_hash,
                "source_files": len(SOURCES), "new_fit_calls": 0, "new_model_inference_calls": 0,
                "official_test_sample_count": 687, "seed": 17, "primary_base_rounds": 46, "primary_full_rounds": 53,
                "fixed_configuration_analysis_rounds": 53, "created_utc": datetime.now(timezone.utc).isoformat()}
    write("run_manifest.json", manifest)
    write("training_lineage.json", parameters)
    write("original_seed17_test_metrics.json", original)
    write("fixed53_all_state_metrics.json", fixed)
    write("analysis_summary.json", summary)
    write("claim_audit.json", claims)
    write("RESULTS.md", render(summary))
    write("build_receipt.json", {"status": "BUILT_AWAITING_INDEPENDENT_QA", "original_test_weather_rows": 100,
                                 "fixed53_states": 1180, "metric_scalar_checks": 35700,
                                 "sources_unchanged": len(SOURCES), "new_fit_calls": 0, "new_model_inference_calls": 0})
    print(json.dumps({"status": "BUILT_AWAITING_INDEPENDENT_QA", "sources": len(SOURCES), "new_fit_calls": 0, "new_model_inference_calls": 0}))


def render(summary: dict) -> str:
    primary_rows = [[WNAME[row["weather"]], row["base_original_46"]["n_hours"],
                     f'{row["base_original_46"]["rmse_db"]:.6f}', f'{row["full_original_53"]["rmse_db"]:.6f}',
                     f'{row["rmse_reduction_percent"]:.2f}%'] for row in summary["original_primary_comparison"]]
    ranking_rows = [[row["rank"], LABEL[row["model"]]] + [f'{row["metrics"][w]["rmse_db"]:.6f}' for w in WEATHER]
                    for row in summary["original_full_model_ranking"]]
    trace_rows = [[row["id"], row["choice"]["estimator"]["epochs"], row["choice"]["weight_policy"],
                   str(row["choice"]["gated_fusion"]), f'{row["metrics"]["overall"]["rmse_db"]:.6f}',
                   f'{row["metrics"]["rain"]["rmse_db"]:.6f}', f'{row["metrics"]["dry_overcast"]["rmse_db"]:.6f}']
                  for row in summary["baseline_trace"]]
    highlight_rows = [[row["kind"], row["key"], len(row["feature_names"])] +
                      [f'{row["metrics"]["test"][w]["rmse_db"]:.6f}' for w in WEATHER]
                      for row in summary["fixed53_highlights"]]
    parameter_rows = [["输入列数", "50", "156"], ["固定正式轮数", "46", "53"], ["种子", "17", "17"],
                      ["学习率", "0.001", "0.001"], ["dropout", "0.3", "0.2"], ["weight_decay", "0.01", "0.001"],
                      ["训练权重", "天气加权", "统一权重"], ["门控融合", "有", "无"],
                      ["配置/轮数来源", "基础池验证独立选定", "全层池验证独立选定"]]
    top_rows = [[row["survival_rank"], row["feature_name"], row["type"]] for row in summary["rfe_survival_rank"][:10]]
    link28 = (R28 / "Seed_017_Report/SEED17_SLOW_RESULTS.md").as_posix()
    link29 = (R29 / "RESULTS.md").as_posix()
    return "\n\n".join([
        "# 原始训练对齐后的慢变结果（seed=17）",
        "## 这次调整完成了什么",
        "只修正基线来源与报告口径；没有新增训练、模型推理、调参或选种子，原代码、模型、预测与报告未被覆盖。主比较使用原Rev028基础46轮和原全层53轮；Rev029固定53轮特征分析作为另一种比较完整保留。新版本是报告重分析包，不是新模型。",
        "## 1. 原主模型对照：同一687小时测试集",
        "以下RMSE单位dB，唯一种子17，不是三种子平均。雨天合并全部雨天，包括强雨。相对降低=(基础RMSE−全层RMSE)/基础RMSE。",
        table(["范围", "小时数", "原基础50列/46轮", "原全层156列/53轮", "RMSE降低"], primary_rows),
        "主对照中，全层方案的总体、晴天、雨天、干燥阴天和其他天气RMSE均低于原基础方案。雨天的原比较是0.463024→0.376102 dB；不是0.283988→0.376102 dB。",
        "## 2. 原十个全层模型的seed17排名",
        "仅按各全层模型总体测试RMSE排序，未先筛选有利模型。BiLSTM总体第一；全部天气的RMSE如下，MAE和平均Bias完整保存于original_seed17_test_metrics.json。",
        table(["排名", "模型", "总体", "晴天", "雨天", "干燥阴天", "其他"], ranking_rows),
        "## 3. 原训练设置不变，但两输入池并非同配置",
        table(["参数", "原基础模型", "原全层模型"], parameter_rows),
        "两池原来各有十模型×8候选的相同搜索预算，按343验证小时的总体/雨天/干燥阴天MSE各1/3目标独立选定参数和轮数；正式模型只训练2404训练小时，随后加载保存模型评价。上表比较的是两套独立验证选参的完整建模方案。不能把全部改善归因于增加气压面输入这一单因素，也不能写成其他训练参数完全相同。",
        "## 4. 0.2840到底来自哪里",
        table(["精确模型身份", "轮数", "权重", "门控", "总体RMSE", "雨天RMSE", "干燥阴天RMSE"], trace_rows),
        "0.283988 dB来自Rev029 mask0：基础50列套用全层统一53轮配置后另行训练的模型。它不是Rev028原基础模型，也没有覆盖原基础模型。这一固定配置补充对照的雨天负结果有效且保留；错误在于此前把它不加限定地称为‘基础’，从而与原主对照混用。",
        "## 5. 1024种气压面组合与156→1列RFE：保持原53轮结果",
        "Rev029已完整完成1024种层组合和156个RFE状态，主流水线4720次从头拟合、155次自然单列删除、36735次折外置换评价。全部模型唯一seed17；三个日期隔离折不是三个随机种子。训练/评价/交付的原独立QA均PASS，本次不重跑、不替换状态。",
        "mask1023与RFE k156的特征和每个测试小时预测与原Rev028全层锚点完全相同。层配对始终比较同一53轮配置的两个状态，不能把原基础46轮模型塞入mask0来改5120个配对。RFE删除只使用训练折置换重要性，训练内one-SE选择mask64（仅200hPa，62列）和k5不变。",
        table(["路径", "状态", "列数", "总体", "晴天", "雨天", "干燥阴天", "其他"], highlight_rows),
        "上述mask64/k5是训练内冻结选择，不是测试上选出的最优配置；其测试表现不及全层锚点，不能称压缩后已保留全层精度。k10/k1仅作原预定描述性展示，不重新选择k。完整1180状态、验证/测试各天气RMSE/MAE/Bias与名单另存fixed53_all_state_metrics.json；全部负贡献保留于analysis_summary.json。",
        "RFE自然生存前10列如下；‘生存排名’不是物理因果强度排名，也未设分层列配额。",
        table(["生存排名", "列名", "类别"], top_rows),
        "## 6. 论文中应使用的表述",
        "‘基础输入和全层输入分别采用相同候选预算，在验证集上独立确定模型配置。固定种子17后，全层BiLSTM在十个全层模型中获得最低总体测试RMSE（0.264652 dB）；相较原基础BiLSTM，总体、雨天和干燥阴天RMSE分别从0.427789、0.463024和0.604702 dB降至0.264652、0.376102和0.398341 dB。另以全层配置固定53轮开展气压面组合及递归特征删除分析，逐状态重新训练，并完整保留正、负贡献。’",
        "主比较不要写‘仅增加分层输入、其他所有参数不变’；固定配置分析不能写‘各天气必然改善’。本次不新增显著性检验，不将原三种子区间改称seed17区间，也不将Rev029基线的区间移植给Rev028原基础46轮。历史测试访问与同日训练重叠仍在，不称全新盲测。",
        "## 7. 保存与核验",
        "run_manifest.json记录两个父版本指纹和新代码哈希，source_snapshot.json保存所有使用的旧源哈希；training_lineage.json保存原20个seed17模型的冻结参数、实际轮数、全部原逐轮记录和模型SHA256。claim_audit.json逐条限制结论。独立QA从旧预测用另一套数值实现复算，PASS与completion.json齐全才算本次报告调整完成。新增fit=0，模型推理=0。原各自日期不确定性文件保持原用途，没有新生成或移植区间。",
        f"原训练与各分组详细结果：[Rev028 seed17原报告]({link28})。固定53轮全部分层/RFE及原65个日期区间：[Rev029原报告]({link29})。这两份报告只读保留，不因本次口径调整改变其原始内容。",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "status"])
    command = parser.parse_args().command
    if command == "build":
        build()
    else:
        result = {"scope": "report_only", "training_workers_started": 0, "output_exists": OUT.exists()}
        for name in ["alignment_qa.json", "completion.json"]:
            if (OUT / name).is_file():
                result[name] = json.loads((OUT / name).read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
