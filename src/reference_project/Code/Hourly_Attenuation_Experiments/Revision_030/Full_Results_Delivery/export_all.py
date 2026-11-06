"""Complete, read-only export of Revision 029. No training, inference or selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
SOURCE = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE"
ANCHOR = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced"
ALIGNED = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_030_Original_Training_Aligned"
OUT = ALIGNED / "Full_Results"
THREAD = "01a0bda1-92d1-7832-9508-60f4e0ec1634"
XLSX = OUT / "outputs" / THREAD / "ALL_LAYER1024_RFE156_RESULTS.xlsx"
ARCHIVE = OUT / "ALL_LAYER1024_RFE156_FULL_PACKAGE.zip"
FP = "cdda7a1eaa7e14477a8edef39702d5b5e34e03f2a29a337d01b91c88ba57b4cd"
WEATHER = ["overall", "clear_dry", "rain", "dry_overcast", "other"]
CN = dict(zip(WEATHER, ["总体", "晴天", "雨天", "干燥阴天", "其他天气"]))
SPLITS = ["test", "validation", "train"]
SCN = {"test": "测试", "validation": "验证", "train": "训练"}
TABLES = {"all_state_weather_metrics.csv": 17700, "all_feature_lists.csv": 1180,
          "all_rounds.csv": 250160, "layer_pairs_all_weather.csv": 25600,
          "layer_contributions_all_weather.csv": 50, "rfe_importance.csv": 12245,
          "rfe_survival_rank.csv": 156}


def hash_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    assert path.resolve().is_relative_to(OUT.resolve())
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_text(path: Path, value: str) -> None:
    assert path.resolve().is_relative_to(OUT.resolve())
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def save_csv(name: str, headers: list[str], rows: list[list]) -> None:
    path = OUT / "csv" / name
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def numeric(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        result = int(value)
        return str(result) if abs(result) > 2 ** 53 else result
    except ValueError:
        try:
            result = float(value)
            assert math.isfinite(result)
            return result
        except ValueError:
            return value


def csv_matrix(path: Path) -> tuple[list[str], list[list]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.reader(handle))
    text_fields = {"kind", "split", "weather", "type", "feature_name", "levels_hpa", "feature_names_json", "fold_delta_rmse_db_json"}
    return source[0], [[(value or None) if header in text_fields else numeric(value)
                        for header, value in zip(source[0], row)] for row in source[1:]]


def source_files() -> list[tuple[Path, str]]:
    files = [(p, "Original_Rev029/" + p.relative_to(SOURCE).as_posix()) for p in SOURCE.iterdir() if p.is_file()]
    for directory in ["layer", "rfe", "fit_events", "failures"]:
        files.extend((p, "Original_Rev029/" + p.relative_to(SOURCE).as_posix())
                     for p in (SOURCE / directory).rglob("*") if p.is_file())
    old_code = ROOT / "Code/Hourly_Attenuation_Experiments/Revision_029"
    files.extend((p, "Original_Rev029_Code/" + p.relative_to(old_code).as_posix()) for p in old_code.iterdir() if p.is_file())
    for case in ["base", "full"]:
        for directory, extension in [("training", "json"), ("test", "json"), ("models", "joblib")]:
            path = ANCHOR / directory / f"{case}_bilstm_seed17.{extension}"
            files.append((path, "Original_Rev028_Anchors/" + path.relative_to(ANCHOR).as_posix()))
    for name in ["run_manifest.json", "selection.json", "train_qa.json", "independent_qa.json", "delivery_qa.json"]:
        files.append((ANCHOR / name, "Original_Rev028_Anchors/" + name))
    return sorted(files, key=lambda item: item[1])


def old_gates() -> dict:
    manifest = load(SOURCE / "run_manifest.json")
    assert manifest["fingerprint"] == FP
    for name in ["train_qa.json", "evaluation_qa.json", "report_qa.json"]:
        record = load(SOURCE / name)
        assert record["status"] == "PASS" and record["fingerprint"] == FP
    complete = load(SOURCE / "completion.json")
    assert complete["status"] == "COMPLETE"
    for key, filename in [("train_qa_sha256", "train_qa.json"), ("evaluation_qa_sha256", "evaluation_qa.json"),
                          ("report_qa_sha256", "report_qa.json"), ("final_report_sha256", "final_report.json"), ("results_sha256", "RESULTS.md")]:
        assert hash_file(SOURCE / filename) == complete[key]
    assert load(ALIGNED / "alignment_qa.json")["status"] == "PASS"
    return manifest


def prepare() -> None:
    assert not OUT.exists(), "Do not overwrite an existing delivery."
    manifest = old_gates()
    OUT.mkdir(parents=True)
    (OUT / "csv").mkdir()
    (OUT / "previews").mkdir()
    XLSX.parent.mkdir(parents=True)
    table_counts, tables = {}, {}
    for name, expected in TABLES.items():
        src, dest = SOURCE / name, OUT / "csv" / name
        shutil.copy2(src, dest)
        assert hash_file(src) == hash_file(dest)
        if name == "all_rounds.csv":
            with dest.open(encoding="utf-8-sig", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
        else:
            headers, matrix = csv_matrix(dest)
            count = len(matrix)
            tables[name] = (headers, matrix)
        assert count == expected, (name, count, expected)
        table_counts[name] = count
    headers, matrix = tables["all_state_weather_metrics.csv"]
    weather_records = [dict(zip(headers, row)) for row in matrix]
    metric_lookup = {(row["kind"], row["key"], row["split"], row["weather"]): row for row in weather_records}
    assert len(metric_lookup) == 17700
    states, feature_rows, fit_rows, fit_weather_rows, deletions, importances = [], [], [], [], [], []
    validated_states, fit_count = 0, 0
    start = time.monotonic()
    for kind, keys, prefix in [("layer", range(1024), "mask"), ("rfe", range(156, 0, -1), "k")]:
        for key in keys:
            path = SOURCE / kind / "states" / f"{prefix}_{key:04d}.json"
            state = load(path)
            assert state["fingerprint"] == FP and state["seed"] == 17 and state["epochs"] == 53
            assert state["kind"] == kind and state["key"] == key
            assert len(state["feature_names"]) == state["n_features"]
            assert len(state["folds"]) == 3 and state["official_holdouts_used_for_training_or_deletion"] is False
            scores = [fold["holdout"]["metrics"]["overall"]["rmse_db"] for fold in state["folds"]]
            assert abs(statistics.fmean(scores) - state["cv_mean_rmse_db"]) < 1e-12
            assert abs(statistics.stdev(scores) / math.sqrt(3) - state["cv_se_rmse_db"]) < 1e-12
            states.append({name: state[name] for name in ["kind", "key", "n_features", "n_base", "n_pressure", "levels_hpa", "cv_mean_rmse_db", "cv_se_rmse_db", "feature_names", "removed_feature_next"]})
            for index, name in enumerate(state["feature_names"], 1):
                category = "pressure" if name.startswith("p") else "base"
                level = int(name.split("__")[0][1:]) if category == "pressure" else None
                feature_rows.append([kind, key, index, name, category, level])
            fits = [("formal", state["full_fit"])] + [("internal_fold", row) for row in state["folds"]]
            for role, fitted in fits:
                metadata = fitted["fit_metadata"]
                assert metadata["trained_rounds"] == 53 and metadata["seed"] == 17
                assert len(fitted["round_history"]) == 53
                assert metadata["sample_weight_policy"] == "uniform" and metadata["official_holdout_used_for_fit"] is False
                if role == "internal_fold":
                    assert set(fitted["fit_days"]).isdisjoint(fitted["hold_days"])
                fold = int(fitted["fold"])
                model_path = SOURCE / kind / ("models" if role == "formal" else "fold_models") / (
                    f"{prefix}_{key:04d}.joblib" if role == "formal" else f"{prefix}_{key:04d}_fold{fold}.joblib")
                assert hash_file(model_path) == fitted["model_sha256"], f"Saved model differs from original state: {model_path}"
                fit_rows.append([kind, key, role, fold, 17, 53, len(fitted["training"]["hour_bjt"]),
                                 len((fitted.get("holdout") or {}).get("hour_bjt", [])), metadata["fit_seconds"],
                                 fitted["model_sha256"], fitted["model_weights_sha256"], fitted["fit_receipt"],
                                 json.dumps(metadata["choice"], ensure_ascii=False),
                                 json.dumps(fitted.get("fit_days", [])), json.dumps(fitted.get("hold_days", []))])
                for partition in ["training", "holdout"]:
                    data = fitted.get(partition) or {}
                    for weather, score in data.get("metrics", {}).items():
                        fit_weather_rows.append([kind, key, role, fold, partition, weather,
                                                 score["n_hours"], score["n_dates"], score["rmse_db"], score["mae_db"], score["bias_db"]])
                fit_count += 1
            if kind == "rfe":
                importance = state["permutation_importance"]
                for item in importance:
                    assert abs(statistics.fmean(item["fold_delta_rmse_db"]) - item["mean_delta_rmse_db"]) < 1e-12
                    importances.append([key, item["feature_name"], item["mean_delta_rmse_db"],
                                        *item["fold_delta_rmse_db"], *item["fold_permuted_rmse_db"],
                                        item["feature_name"] == state["removed_feature_next"]])
                if key > 1:
                    removed = min(importance, key=lambda item: (item["mean_delta_rmse_db"], item["feature_name"]))
                    assert removed["feature_name"] == state["removed_feature_next"]
                    deletions.append([157 - key, key, key - 1, removed["feature_name"], removed["mean_delta_rmse_db"], *removed["fold_delta_rmse_db"]])
                else:
                    assert not importance and state["removed_feature_next"] is None
            validated_states += 1
            if validated_states % 128 == 0 or validated_states == 1180:
                print(f"states {validated_states}/1180; fit records {fit_count}; elapsed {time.monotonic()-start:.1f}s", flush=True)
    assert len(states) == 1180 and fit_count == 4720 and len(feature_rows) == 117718
    assert len(importances) == 12245 and len(deletions) == 155 and len(fit_weather_rows) == 41300
    state_lookup = {(row["kind"], row["key"]): row for row in states}
    for k in range(156, 1, -1):
        before, after = state_lookup[("rfe", k)], state_lookup[("rfe", k - 1)]
        assert set(before["feature_names"]) - set(after["feature_names"]) == {before["removed_feature_next"]}
    selection = load(SOURCE / "selection.json")
    assert selection["fingerprint"] == FP
    wide_headers = ["mask或k", "特征总数", "气压面(hPa)", "训练内CV_RMSE(dB)", "训练内CV_SE(dB)", "训练内oneSE选中"]
    for split in SPLITS:
        for metric in ["rmse_db", "mae_db", "bias_db"]:
            wide_headers += [f"{SCN[split]}_{CN[weather]}_{metric.replace('_db', '')}(dB)" for weather in WEATHER]
    wide_headers += ["基础列数", "气压面列数", "seed", "轮数", "下一删除列"]
    wide_by_kind = {}
    for kind in ["layer", "rfe"]:
        rows = []
        for state in states:
            if state["kind"] != kind:
                continue
            key = state["key"]
            row = [key, state["n_features"], ",".join(map(str, state["levels_hpa"])), state["cv_mean_rmse_db"], state["cv_se_rmse_db"],
                   key == selection["selections"][kind]["selected_key"]]
            for split in SPLITS:
                for metric in ["rmse_db", "mae_db", "bias_db"]:
                    row += [metric_lookup[(kind, key, split, weather)][metric] for weather in WEATHER]
            row += [state["n_base"], state["n_pressure"], 17, 53, state["removed_feature_next"]]
            assert len(row) == len(wide_headers)
            rows.append(row)
        wide_by_kind[kind] = rows
        save_csv(f"{kind}_all_states_wide.csv", wide_headers, rows)
        make_markdown(kind, states, metric_lookup)
    feature_headers = ["kind", "key", "feature_position", "feature_name", "feature_type", "level_hpa"]
    fit_headers = ["kind", "key", "fit_role", "fold", "seed", "epochs", "fit_hours", "internal_holdout_hours",
                   "fit_seconds", "model_sha256", "network_weights_sha256", "fit_receipt", "choice_json", "fit_dates_json", "holdout_dates_json"]
    fit_weather_headers = ["kind", "key", "fit_role", "fold", "partition", "weather", "n_hours", "n_dates", "rmse_db", "mae_db", "bias_db"]
    importance_headers = ["k", "feature_name", "mean_delta_rmse_db", "fold1_delta_rmse_db", "fold2_delta_rmse_db", "fold3_delta_rmse_db",
                          "fold1_permuted_rmse_db", "fold2_permuted_rmse_db", "fold3_permuted_rmse_db", "deleted_next"]
    deletion_headers = ["deletion_step", "k_before", "k_after", "deleted_feature", "mean_delta_rmse_db", "fold1_delta_rmse_db", "fold2_delta_rmse_db", "fold3_delta_rmse_db"]
    save_csv("complete_feature_membership.csv", feature_headers, feature_rows)
    save_csv("all_4720_fit_records.csv", fit_headers, fit_rows)
    save_csv("all_fit_and_internal_holdout_weather_metrics.csv", fit_weather_headers, fit_weather_rows)
    save_csv("rfe_all_importance_expanded.csv", importance_headers, importances)
    save_csv("rfe_all_155_deletions.csv", deletion_headers, deletions)
    intervals = load(SOURCE / "date_intervals.json")
    interval_rows = [[key, value["rmse_improvement_db"], *value["percentile95_db"], value["n_dates"], value["replicates"], str(value["bootstrap_seed"]), value["unit"]]
                     for key, value in intervals["comparisons"].items()]
    assert len(interval_rows) == 65 and intervals["not_used_to_change_model_or_feature_selection"] is True
    interval_headers = ["comparison_weather", "rmse_improvement_db", "percentile95_lower_db", "percentile95_upper_db", "n_dates", "replicates", "bootstrap_seed", "resampling_unit"]
    save_csv("all_65_existing_date_intervals.csv", interval_headers, interval_rows)
    prediction_counts = {"validation": 0, "test": 0}
    for split in ["validation", "test"]:
        with (OUT / "csv" / f"all_{split}_hourly_predictions.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kind", "key", "hour_bjt", "weather", "truth_db", "prediction_db", "error_prediction_minus_truth_db"])
            for state in states:
                kind, key = state["kind"], state["key"]
                prefix = "mask" if kind == "layer" else "k"
                evaluated = load(SOURCE / kind / "evaluation" / f"{prefix}_{key:04d}.json")
                assert evaluated["fingerprint"] == FP and evaluated["fit_calls_during_evaluation"] == 0
                data = evaluated["results"][split]
                for i, hour in enumerate(data["hour_bjt"]):
                    weather = [w for w in WEATHER[1:] if data["weather_masks"][w][i]]
                    assert len(weather) == 1
                    truth, prediction = data["truth_db"][i], data["prediction_db"][i]
                    writer.writerow([kind, key, hour, weather[0], truth, prediction, prediction - truth])
                    prediction_counts[split] += 1
    assert prediction_counts == {"validation": 404740, "test": 810660}
    notes = [
        ["内容", "范围 / 定义"],
        ["分层状态", "mask0至mask1023全部1024种组合，每状态均从头训练53轮"],
        ["RFE状态", "k156至k1全部156个状态，155次单列自然删除，无特征配额"],
        ["指标", "训练/验证/测试；总体/晴天/雨天/干燥阴天/其他；RMSE、MAE、平均Bias，单位dB"],
        ["种子与折", "唯一seed17。三个训练内日期隔离折不是三个随机种子"],
        ["基础身份", "本分层mask0是53轮统一配置，不是Rev028原基础46轮加权门控模型"],
        ["固定配置", "lr0.001，dropout0.2，weight_decay0.001，uniform，无gated_fusion；保持原训练"],
        ["训练内选择", "mask64，仅200hPa，62列；RFE k5。测试未用于重新选层或k"],
        ["主流水线拟合", "4720个模型，4096分层拟合与624 RFE拟合；不含原纯合成和锚点预检"],
        ["完整逐轮曲线", "250160行另存CSV；全部正式及折模型各53轮，不只末轮"],
        ["完整特征名单", "本簿117718条状态/特征记录；另存原1180行JSON名单与CSV"],
        ["正式验证逐小时预测", "404740行另存CSV；所有1180模型×343小时"],
        ["正式测试逐小时预测", "810660行另存CSV；所有1180模型×687小时"],
        ["训练及折内逐小时预测", "完整保存在压缩包Original_Rev029下1180个states JSON，见full_fit.training及folds[].training/holdout"],
        ["原日期区间", "仅原65个预定配对区间，不为每个状态新增区间，不移植给原基础46轮"],
        ["精度", "Excel/CSV保留原双精度数值；Markdown显示6位小数"],
        ["天气", "雨天包含全部强雨；保留其他天气；不删不利样本"],
        ["历史测试", "历史测试已访问，测试日期与训练日期重叠；不称新盲测"],
        ["层贡献符号", "RMSE(不含该层)-RMSE(含该层)，正值改善，负值恶化；512个背景配对平均"],
        ["RFE重要性符号", "训练折置换该列后的RMSE-原RMSE；三个折平均最小者删除，保留负值"],
        ["生存排序", "最后留下排1；倒数删除排2；最先删除排156。不是物理因果强度"],
        ["结果来源", "Rev029冻结指纹"], ["fingerprint", FP],
        ["新增拟合/推理", "0 / 0；本次只导出原记录，不改变模型与路径"],
    ]
    sheet_specs = [
        {"name": "分层组合1024", "headers": wide_headers, "rows": wide_by_kind["layer"], "source": "Rev029 states与all_state_weather_metrics.csv"},
        {"name": "RFE全路径156", "headers": wide_headers, "rows": wide_by_kind["rfe"], "source": "Rev029 RFE states与all_state_weather_metrics.csv"},
        {"name": "天气指标17700", "headers": headers, "rows": matrix, "source": "Rev029 all_state_weather_metrics.csv"},
        {"name": "层贡献50", "headers": tables["layer_contributions_all_weather.csv"][0], "rows": tables["layer_contributions_all_weather.csv"][1], "source": "Rev029 layer_contributions_all_weather.csv"},
        {"name": "层配对25600", "headers": tables["layer_pairs_all_weather.csv"][0], "rows": tables["layer_pairs_all_weather.csv"][1], "source": "Rev029 layer_pairs_all_weather.csv"},
        {"name": "重要性12245", "headers": importance_headers, "rows": importances, "source": "Rev029所有RFE states的permutation_importance"},
        {"name": "删除顺序155", "headers": deletion_headers, "rows": deletions, "source": "Rev029所有RFE states的removed_feature_next"},
        {"name": "生存排序156", "headers": tables["rfe_survival_rank.csv"][0], "rows": tables["rfe_survival_rank.csv"][1], "source": "Rev029 rfe_survival_rank.csv"},
        {"name": "完整特征117718", "headers": feature_headers, "rows": feature_rows, "source": "Rev029所有1180 states的完整feature_names"},
        {"name": "拟合记录4720", "headers": fit_headers, "rows": fit_rows, "source": "Rev029所有states的full_fit与folds"},
        {"name": "折内天气41300", "headers": fit_weather_headers, "rows": fit_weather_rows, "source": "Rev029所有正式训练/折拟合与折外检查天气指标"},
        {"name": "日期区间65", "headers": interval_headers, "rows": interval_rows, "source": "Rev029 date_intervals.json，只沿用原区间"},
        {"name": "说明", "headers": notes[0], "rows": notes[1:], "source": "Rev029原PROTOCOL/PARAMETERS/QA及Rev030对齐报告"},
    ]
    save_json(OUT / "workbook_data.json", {"xlsx_path": str(XLSX), "preview_dir": str(OUT / "previews"), "sheets": sheet_specs})
    save_json(OUT / "state_metadata.json", states)
    inventory = []
    for number, (path, archive_name) in enumerate(source_files(), 1):
        inventory.append({"source": path.relative_to(ROOT).as_posix(), "archive_name": archive_name, "bytes": path.stat().st_size, "sha256": hash_file(path)})
        if number % 2048 == 0:
            print(f"source fingerprints {number}", flush=True)
    save_json(OUT / "source_archive_inventory.json", {"files": inventory, "count": len(inventory), "source_fingerprint": FP})
    save_json(OUT / "content_qa.json", {"status": "PASS", "scope": "complete_export_of_existing_results_not_new_experiment",
              "layer_states": 1024, "rfe_states": 156, "formal_models": 1180, "fold_models": 3540,
              "fit_records": 4720, "feature_membership_rows": 117718, "fit_weather_rows": 41300,
              "rfe_deletions": 155, "importance_rows": 12245, "original_csv_rows": table_counts,
              "hourly_predictions": prediction_counts, "existing_date_intervals": 65,
              "new_training_fits": 0, "new_model_inference_calls": 0, "utc": datetime.now(timezone.utc).isoformat()})
    save_text(OUT / "全部结果索引.md", make_index())
    print(json.dumps({"status": "PREPARED", "states": 1180, "feature_rows": 117718, "raw_files": len(inventory), "new_fit_calls": 0}, ensure_ascii=False), flush=True)


def make_markdown(kind: str, states: list[dict], lookup: dict) -> None:
    filename = "1024种气压面组合_全部指标.md" if kind == "layer" else "156至1列RFE_全部指标.md"
    title = "1024种气压面组合：全部结果" if kind == "layer" else "156→1列RFE：全部结果"
    lines = [f"# {title}", "", "唯一seed17，固定53轮。逐状态从头训练，全部状态保留。单位dB；Bias为预测减实测的平均。雨天包含全部强雨。", "",
             "本文件完整列出训练、验证、测试的五天气RMSE/MAE/Bias，没有只选前几名。Markdown显示6位小数；CSV/Excel保留完整数值。完整名单、每轮曲线、全部逐小时预测、模型与参数见同目录全部结果索引。", "",
             "mask0是本固定53轮实验的基础模型，不是Rev028原基础46轮模型。one-SE选择仍由训练折决定；下表不用于测试回调。", ""]
    chosen = [state for state in states if state["kind"] == kind]
    for split in SPLITS:
        lines += [f"## {SCN[split]}集全部{len(chosen)}个状态", ""]
        labels = ["mask或k", "列数", "气压面(hPa)"] + [f"{CN[w]}_{metric}" for metric in ["RMSE", "MAE", "Bias"] for w in WEATHER]
        lines += ["|" + "|".join(labels) + "|", "|" + "|".join(["---"] * len(labels)) + "|"]
        for state in chosen:
            key = state["key"]
            row = [str(key), str(state["n_features"]), ",".join(map(str, state["levels_hpa"])) or "无"]
            for metric in ["rmse_db", "mae_db", "bias_db"]:
                row += [f'{lookup[(kind,key,split,w)][metric]:.6f}' for w in WEATHER]
            lines.append("|" + "|".join(row) + "|")
        lines.append("")
    save_text(OUT / filename, "\n".join(lines))


def make_index() -> str:
    book = XLSX.relative_to(OUT).as_posix()
    return f"""# 1024种气压面组合与156→1列RFE：全部结果索引

全部1024个mask、全部156个k均保留，唯一seed17、统一53轮。这里是原Rev029完整结果的交付，不是新训练。主流水线4720次真实拟合，正式1180模型、训练折3540模型；原数据、模型和结果只读不变。

## 直接阅读

- [全量Excel]({book})：全部状态、三集合五天气指标、层贡献及全部配对、完整置换重要性/删除/生存链、全部特征、全部拟合参数及折内天气指标、原日期区间。
- [1024种组合全部指标](1024种气压面组合_全部指标.md)：1024状态×训练/验证/测试×五天气×RMSE/MAE/Bias，一个都不省略。
- [156→1列RFE全部指标](156至1列RFE_全部指标.md)：156状态×训练/验证/测试×五天气×RMSE/MAE/Bias，一个都不省略。
- [完整结果包](ALL_LAYER1024_RFE156_FULL_PACKAGE.zip)：上述交付、全部CSV、全部原states/evaluation、1180正式模型与3540折模型、原代码/协议/参数/QA/日志/失败审计、原Rev028基础和全层锚点。

## 所有表与明细

|内容|完整记录数|文件|
|---|---:|---|
|气压面组合宽表|1024|[layer_all_states_wide.csv](csv/layer_all_states_wide.csv)|
|RFE路径宽表|156|[rfe_all_states_wide.csv](csv/rfe_all_states_wide.csv)|
|三集合五天气指标|17700|[all_state_weather_metrics.csv](csv/all_state_weather_metrics.csv)|
|所有每轮曲线|250160|[all_rounds.csv](csv/all_rounds.csv)|
|每状态完整名单|1180|[all_feature_lists.csv](csv/all_feature_lists.csv)|
|逐列展开的状态名单|117718|[complete_feature_membership.csv](csv/complete_feature_membership.csv)|
|五天气全部层配对|25600|[layer_pairs_all_weather.csv](csv/layer_pairs_all_weather.csv)|
|五天气每层汇总贡献|50|[layer_contributions_all_weather.csv](csv/layer_contributions_all_weather.csv)|
|全部RFE置换重要性|12245|[rfe_all_importance_expanded.csv](csv/rfe_all_importance_expanded.csv)|
|每一次删除|155|[rfe_all_155_deletions.csv](csv/rfe_all_155_deletions.csv)|
|完整生存排序|156|[rfe_survival_rank.csv](csv/rfe_survival_rank.csv)|
|全部拟合记录/配置|4720|[all_4720_fit_records.csv](csv/all_4720_fit_records.csv)|
|正式训练及折内拟合/检查天气指标|41300|[all_fit_and_internal_holdout_weather_metrics.csv](csv/all_fit_and_internal_holdout_weather_metrics.csv)|
|所有验证逐小时预测|404740|[all_validation_hourly_predictions.csv](csv/all_validation_hourly_predictions.csv)|
|所有测试逐小时预测|810660|[all_test_hourly_predictions.csv](csv/all_test_hourly_predictions.csv)|
|原预定成对日期区间|65|[all_65_existing_date_intervals.csv](csv/all_65_existing_date_intervals.csv)|

## 训练与折内逐小时预测、完整模型

完整包`Original_Rev029/layer/states/mask_0000.json`至`mask_1023.json`及`Original_Rev029/rfe/states/k_0156.json`至`k_0001.json`保留每次正式拟合和三个日期隔离折的全部实测/预测、天气掩码、指标、参数、每轮曲线、完整名单和模型哈希。每状态读取`full_fit.training`、`folds[0..2].training`与`folds[0..2].holdout`即可获得全部训练/折内检查逐小时记录；没有用平均曲线替代逐小时记录。

模型原文件在`Original_Rev029/layer/models`、`layer/fold_models`、`rfe/models`、`rfe/fold_models`，全部4720个；正式验证/测试原预测在两路径`evaluation`中。原`fit_events`全部4720条及`failures`也保留。纯合成接口临时目录不是主实验结果，未重复打包；其正式预检QA已保留。原H5/ERA5/数据集不复制、不修改，冻结manifest仍记录它们的原哈希。

拟合记录表的日期列表对应原`fit_days`/`hold_days`字段。原正式`full_fit`不单独保存这两个字段，该栏`[]`表示未单独记录，不是零个训练日期；完整真实日期由`full_fit.training.hour_bjt`提供（总体训练指标仍为162日期、2404小时）。三个内部折的日期名单按原字段完整保留。

## 口径

本实验所有子集固定全层53轮配置：seed17、lr0.001、dropout0.2、weight_decay0.001、uniform、无门控。三个训练内日期折不是三个种子。mask0的雨天0.283988dB是本固定配置基础模型；原Rev028基础46轮雨天0.463024dB是另外模型，不能偷换进本层配对。全层mask1023/RFE k156与原Rev028全层锚点一致。

训练内one-SE选择mask64（仅200hPa，62列）与k5保持原样，不根据下表测试成绩改选。全部负贡献和压缩后性能损失保留。层贡献为不含层RMSE减含层RMSE；正改善、负恶化。RFE重要性为训练折置换后的RMSE减原RMSE，三个折均值最小者删除。生存1是最后保留，不是物理因果强度。

测试仍是687小时（晴天322、雨天51、干燥阴天158、其他156），验证343小时，训练2404小时。历史测试访问及同日训练重叠仍在，不称新盲测。仅沿用原65个日期区间，没有为每状态另造区间或把固定53轮区间移植给原46轮模型。

## 核验

原训练/评价/交付QA均PASS。本次读取原记录，不调用fit/predict。原7张CSV逐字节保留，新增展开表核对数量、名单、每次删除及全部拟合配置；Excel逐单元格及归档逐文件SHA256核验完成后，见`delivery_qa.json`。来源逐文件清单是`source_archive_inventory.json`。CSV/Excel保持原双精度，Markdown仅显示6位小数。
"""


def verify_tabular() -> dict:
    states = load(OUT / "state_metadata.json")
    counts = {}
    for kind, expected in [("layer", 1024), ("rfe", 156)]:
        name = "1024种气压面组合_全部指标.md" if kind == "layer" else "156至1列RFE_全部指标.md"
        lines = (OUT / name).read_text(encoding="utf-8").splitlines()
        numbered = [line for line in lines if line.startswith("|") and line.split("|")[1].isdigit()]
        assert len(numbered) == expected * 3
        for block in range(3):
            keys = [int(line.split("|")[1]) for line in numbered[block*expected:(block+1)*expected]]
            assert keys == list(range(1024)) if kind == "layer" else keys == list(range(156, 0, -1))
        counts[kind + "_markdown_state_rows"] = len(numbered)
    with (OUT / "csv/complete_feature_membership.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for state in states:
            for position, name in enumerate(state["feature_names"], 1):
                row = next(reader)
                assert row["kind"] == state["kind"] and int(row["key"]) == state["key"]
                assert int(row["feature_position"]) == position and row["feature_name"] == name
                assert row["feature_type"] == ("pressure" if name.startswith("p") else "base")
                count += 1
        assert next(reader, None) is None and count == 117718
        counts["feature_rows_verified"] = count
    for split, expected in [("validation", 404740), ("test", 810660)]:
        with (OUT / "csv" / f"all_{split}_hourly_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            count = 0
            for state in states:
                prefix = "mask" if state["kind"] == "layer" else "k"
                source = load(SOURCE / state["kind"] / "evaluation" / f"{prefix}_{state['key']:04d}.json")["results"][split]
                for index, hour in enumerate(source["hour_bjt"]):
                    row = next(reader)
                    assert row["kind"] == state["kind"] and int(row["key"]) == state["key"] and row["hour_bjt"] == hour
                    assert float(row["truth_db"]) == source["truth_db"][index]
                    assert float(row["prediction_db"]) == source["prediction_db"][index]
                    assert float(row["error_prediction_minus_truth_db"]) == source["prediction_db"][index] - source["truth_db"][index]
                    assert source["weather_masks"][row["weather"]][index] is True
                    count += 1
            assert next(reader, None) is None and count == expected
            counts[split + "_prediction_rows_verified"] = count
        print(f"independently verified {split}: {count} exact source predictions", flush=True)
    result = {"status": "PASS", **counts, "new_training_fits": 0, "new_model_inference_calls": 0}
    save_json(OUT / "tabular_content_qa.json", result)
    print(json.dumps(result), flush=True)
    return result


def verify_xlsx() -> dict:
    payload = load(OUT / "workbook_data.json")
    import openpyxl  # Reading/verification only; never authoring.
    workbook = openpyxl.load_workbook(XLSX, read_only=True, data_only=False)
    assert workbook.sheetnames == [sheet["name"] for sheet in payload["sheets"]]
    cells = 0
    for spec in payload["sheets"]:
        sheet = workbook[spec["name"]]
        expected = [spec.get("display_headers", spec["headers"])] + spec["rows"]
        actual = sheet.iter_rows(min_row=6, max_row=6 + len(spec["rows"]), min_col=1, max_col=len(spec["headers"]), values_only=True)
        rows = 0
        for want, got in zip(expected, actual):
            assert len(want) == len(got)
            for a, b in zip(want, got):
                if a is None or a == "":
                    assert b in (None, "")
                elif isinstance(a, bool):
                    assert type(b) is bool and a == b
                elif isinstance(a, (float, int)):
                    assert isinstance(b, (float, int)) and not isinstance(b, bool)
                    assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (spec["name"], rows, a, b)
                else:
                    assert a == b, (spec["name"], rows, a, b)
                cells += 1
            rows += 1
        assert rows == len(expected)
    workbook.close()
    result = {"status": "PASS", "sheets": len(payload["sheets"]), "cells_independently_verified": cells,
              "xlsx_sha256": hash_file(XLSX), "reading_engine": "openpyxl read_only; not used for authoring"}
    save_json(OUT / "xlsx_content_qa.json", result)
    return result


def archive_originals() -> None:
    assert not ARCHIVE.exists(), "Do not overwrite an existing archive."
    assert load(OUT / "content_qa.json")["status"] == "PASS"
    inventory = load(OUT / "source_archive_inventory.json")
    original_files = inventory["files"]
    assert len(original_files) == inventory["count"]
    start = time.monotonic()
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i, row in enumerate(original_files, 1):
            path = ROOT / row["source"]
            assert path.stat().st_size == row["bytes"] and hash_file(path) == row["sha256"], f"Old source changed: {path}"
            archive.write(path, row["archive_name"])
            if i % 1024 == 0:
                print(f"archived {i}/{len(original_files)} original files; elapsed {time.monotonic()-start:.1f}s", flush=True)
    save_json(OUT / "archive_originals_receipt.json", {"status": "ORIGINALS_ARCHIVED_AWAITING_DELIVERY_QA",
              "files": len(original_files), "archive_bytes_before_append": ARCHIVE.stat().st_size,
              "utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "ORIGINALS_ARCHIVED", "files": len(original_files), "seconds": time.monotonic()-start}), flush=True)


def pack() -> None:
    assert ARCHIVE.exists() and not (OUT / "delivery_qa.json").exists()
    assert load(OUT / "archive_originals_receipt.json")["status"] == "ORIGINALS_ARCHIVED_AWAITING_DELIVERY_QA"
    assert load(OUT / "content_qa.json")["status"] == "PASS"
    assert load(OUT / "workbook_build_qa.json")["status"] == "PASS"
    assert load(OUT / "tabular_content_qa.json")["status"] == "PASS"
    xlsx_qa = verify_xlsx()
    inventory = load(OUT / "source_archive_inventory.json")
    original_files = inventory["files"]
    assert len(original_files) == inventory["count"]
    deliveries = [(path, "Deliverables/" + path.relative_to(OUT).as_posix()) for path in (OUT / "csv").iterdir() if path.is_file()]
    for path in [XLSX, OUT / "全部结果索引.md", OUT / "1024种气压面组合_全部指标.md", OUT / "156至1列RFE_全部指标.md",
                 OUT / "content_qa.json", OUT / "tabular_content_qa.json", OUT / "xlsx_content_qa.json", OUT / "source_archive_inventory.json", OUT / "state_metadata.json"]:
        deliveries.append((path, "Deliverables/" + path.relative_to(OUT).as_posix()))
    expected = {row["archive_name"]: row["sha256"] for row in original_files}
    expected.update({name: hash_file(path) for path, name in deliveries})
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        assert set(archive.namelist()) == {row["archive_name"] for row in original_files}
    with zipfile.ZipFile(ARCHIVE, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path, name in deliveries:
            archive.write(path, name)
    models, states, evaluations, fits = 0, 0, 0, 0
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        assert set(archive.namelist()) == set(expected) and len(archive.namelist()) == len(expected)
        for i, item in enumerate(archive.infolist(), 1):
            value = hashlib.sha256()
            with archive.open(item, "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            assert value.hexdigest() == expected[item.filename], f"Archive changed/truncated: {item.filename}"
            if item.filename.startswith("Original_Rev029/"):
                models += item.filename.endswith(".joblib")
                states += "/states/" in item.filename and item.filename.endswith(".json")
                evaluations += "/evaluation/" in item.filename and item.filename.endswith(".json")
                fits += item.filename.startswith("Original_Rev029/fit_events/") and item.filename.endswith(".json")
            if i % 2048 == 0:
                print(f"verified archive SHA256 {i}/{len(expected)}", flush=True)
    assert (models, states, evaluations, fits) == (4720, 1180, 1180, 4720)
    for row in original_files:
        assert hash_file(ROOT / row["source"]) == row["sha256"], f"Old source changed after export: {row['source']}"
    result = {"status": "PASS", "scope": "all_1024_layer_and_156_rfe_results", "original_files_unchanged": len(original_files),
              "archive_entries_sha256_verified": len(expected), "archived_main_models": models, "archived_states": states,
              "archived_evaluations": evaluations, "archived_fit_events": fits, "xlsx_cells_verified": xlsx_qa["cells_independently_verified"],
              "layer_states": 1024, "rfe_states": 156, "all_rounds": 250160, "weather_metric_rows": 17700,
              "feature_membership_rows": 117718, "layer_pair_weather_rows": 25600, "importance_rows": 12245,
              "new_training_fits": 0, "new_model_inference_calls": 0,
              "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": hash_file(ARCHIVE),
              "xlsx_sha256": xlsx_qa["xlsx_sha256"], "utc": datetime.now(timezone.utc).isoformat()}
    save_json(OUT / "delivery_qa.json", result)
    save_json(OUT / "completion.json", {"status": "COMPLETE", "scope": "complete_existing_result_delivery_not_new_training",
                                       "delivery_qa_sha256": hash_file(OUT / "delivery_qa.json"), "archive_sha256": result["archive_sha256"],
                                       "xlsx_sha256": result["xlsx_sha256"], "utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "archive-originals", "verify-tabular", "pack"])
    command = parser.parse_args().command
    if command == "prepare":
        prepare()
    elif command == "archive-originals":
        archive_originals()
    elif command == "verify-tabular":
        verify_tabular()
    else:
        pack()
