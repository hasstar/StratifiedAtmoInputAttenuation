"""Render already verified JSON results and preserve calendar/model provenance."""
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from collections import Counter
import hashlib
import json
import re
import traceback

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_036"
def rev(n):
    return ROOT / f"Outputs/Experiments/Combined_Channel_Revision_{n:03d}"
def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def put(name, value):
    with (OUT/name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
def md(name, value):
    with (OUT/name).open("x", encoding="utf-8") as f:
        f.write(value.rstrip()+"\n")
def link(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")

NAMES = {"ours_frozen_top10": "本文冻结Top10", "bai2022_lstm_t0_adapted": "Bai LSTM LOCAL适配",
         "tabm2025_official_budgeted": "TabM LOCAL适配", "itu_combined_geometric_approx": "组合ITU近似"}
SHORT = dict(zip(("ours", "bai_lstm", "tabm", "itu"), NAMES.values()))
COHORTS = {"LOCAL_liquid_input_valid_supplement": "LOCAL液态雨输入有效补充（684小时母集）",
           "LOCAL_legacy_687_reference": "LOCAL原687小时历史参考（含3小时输入异常）",
           "UK_full_common": "英国两站合并", "UK_chilbolton_full_common": "Chilbolton", "UK_sparsholt_full_common": "Sparsholt"}
EVENTS = {"rain": "雨小时", "dry_overcast": "干燥阴小时"}
DC = {"LOCAL_liquid_input_valid_supplement": COHORTS["LOCAL_liquid_input_valid_supplement"],
      "LOCAL_legacy_687_reference": COHORTS["LOCAL_legacy_687_reference"], "UK": "英国两站合并", "chilbolton": "Chilbolton", "sparsholt": "Sparsholt"}
def fmt(mean, deviation=None):
    return f"{mean:.6f}" + (f" ± {deviation:.6f}" if deviation is not None else "")
def hm(r, m):
    return fmt(r[m+"_mean"], r[m+"_sd"])


def main():
    qa = read(OUT/"verification.json")
    assert qa["status"] == "PASS_FINAL_NUMERIC_CONSOLIDATION"
    sources = read(OUT/"sources_before.json")
    assert all(sha(p) == h for p, h in sources.items())
    old_models = read(rev(2)/"source_sha256_before.json")
    assert all(sha(p) == h for p, h in old_models.items())
    more = {p: h for p, h in old_models.items() if p not in sources}
    for p in CODE.glob("*"):
        if p.is_file() and str(p) not in sources:
            more[str(p)] = sha(p)
    put("additional_source_pins.json", more)
    union = {**sources, **old_models, **more}
    model_path = ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/rfe/models/k_0010.joblib"
    assert sha(model_path) == "b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5"
    models = {"proposed": {"path": str(model_path), "sha256": sha(model_path)}, "adapted": []}
    for m in read(rev(2)/"frozen_selection.json")["models"]:
        if m["family"] not in ("bai2022_lstm_t0_adapted", "tabm2025_official_budgeted"):
            continue
        path = rev(2)/m["path"]
        assert sha(path) == m["sha256"]
        models["adapted"].append({"id": m["id"], "path": str(path), "sha256": m["sha256"], "seed": m["seed"]})
    assert len(models["adapted"]) == 6
    models["frozen_preprocessing"] = {"path": str(rev(2)/"preprocessing.joblib"), "sha256": sha(rev(2)/"preprocessing.joblib")}
    models["selection_and_original_input_pins"] = {"path": str(rev(2)/"source_sha256_before.json"), "sha256": sha(rev(2)/"source_sha256_before.json"), "verified_entries": len(old_models), "newly_flattened_entries": len(set(old_models)-set(sources))}
    put("model_source_index.json", models)

    coverage = read(OUT/"coverage_by_date.json")
    dates = {(r["station"], r["date"]) for r in coverage}
    planpath = ROOT/"Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/plan.json"
    plan = read(planpath)
    native = {(r["station"], r["date_utc"]) for r in plan["native_days"]}
    actual = set()
    for s in ("chilbolton", "sparsholt"):
        for p in (ROOT/"外部样例数据CEDA/GBS_20p7GHz"/s).rglob("*.nc"):
            match = re.fullmatch("cfarr-gbs_"+s+r"_(\d{8})\.nc", p.name)
            if match:
                day = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
                assert (s,day) not in actual
                actual.add((s,day))
    assert actual == native and len(native) == 1127
    calendar = []
    for station, c in plan["scope_counts"].items():
        day, last = date.fromisoformat(c["first"]), date.fromisoformat(c["last"])
        while day <= last:
            k = station, day.isoformat()
            state = "COMPLETE_WEATHER_INCLUDED" if k in dates else "NATIVE_PRESENT_WEATHER_NOT_COMPLETE_NOT_SCORED" if k in native else "NOT_LISTED_IN_FROZEN_NATIVE_CALENDAR"
            calendar.append({"station": station, "date_utc": k[1], "native_file_present": k in native, "in_completed_weather_cohort": k in dates, "status": state})
            day += timedelta(days=1)
    counts = Counter(r["status"] for r in calendar)
    assert counts == {"COMPLETE_WEATHER_INCLUDED": 236, "NATIVE_PRESENT_WEATHER_NOT_COMPLETE_NOT_SCORED": 891, "NOT_LISTED_IN_FROZEN_NATIVE_CALENDAR": 5}
    put("native_calendar_ledger.json", {"source": str(planpath), "source_sha256": sha(planpath), "counts": dict(counts),
        "scope": "Historical per-station first-to-last native span only; not a weather acquisition authorization or a new score cohort", "rows": calendar})

    hourly = read(OUT/"hourly_family_metrics.json")
    lines = ["# 四类模型小时排名（完整表）", "", "指标单位dB，Bias=预测−实测。Bai/TabM的±是17/29/43三种子指标的样本SD，不是置信区间；先逐run评分再汇总，不是预测集成。按未舍入RMSE均值排名。以下每个分组内键集合相同，不跨掩码排名。日期数指该类合格小时涉及的日期，不代表整日天气。", "", "[评分键与全部实际日期](sample_membership.json) · [全精度家族指标](hourly_family_metrics.json) · [80行逐run指标](hourly_run_metrics.json)"]
    for cohort, title in COHORTS.items():
        lines += ["", "## "+title, ""]
        for event, etitle in EVENTS.items():
            rows = sorted([r for r in hourly if r["cohort"] == cohort and r["event"] == event], key=lambda r:r["rank"])
            date_text = "；".join(f"{s} {len(ds)}日（{ds[0]}—{ds[-1]}，非连续全覆盖）" for s,ds in rows[0]["dates_by_station"].items())
            lines += ["### "+etitle, "", f"同样本 n={rows[0]['n']} 小时。实际覆盖：{date_text}。", "", "| 排名 | 模型 | RMSE | MAE | Bias | n |", "|---:|---|---:|---:|---:|---:|"]
            for r in rows:
                lines.append(f"| {r['rank']} | {NAMES[r['family']]} | {hm(r,'rmse_db')} | {hm(r,'mae_db')} | {hm(r,'bias_db')} | {r['n']} |")
    md("HOURLY_RANKINGS.md", "\n".join(lines))

    dyn = read(OUT/"dynamic_family_metrics.json")
    lines = ["# 共同冻结快变：边缘分布与时间结构单列结果", "", "这是本工程混合方案：四类各自冻结小时中心＋相同既有条件快变CDF。不是四种作者原版秒级模型，不是E-SST/MTS，也不是对应真实时刻的逐秒确定性预测。所有结果描述性报告，不检验显著性。", "", "## 边缘分布评分", "", "沿用201中点分位网格CRPS（dB，越小越好），401网格为固定数值敏感性；覆盖率为0—1比例。LOCAL按合格100ms记录计权，UK原生有效秒等权，再按原合同聚合小时/日期。日期等权：先在同日内平均站点小时，再平均日期；小时等权：直接平均小时。UK池化日期不按站点重复计。两个权重分别排名。±为三种子指标样本SD。", "", "LOCAL原209个雨阴小时保留为历史参考；输入有效补充为207小时（雨51、阴156），只从017冻结逐小时分数重聚合。异常两小时未补零；3个输入异常中的另1个是other，本来不进入此评分。", "", "[动态掩码核验](dynamic_mask_audit.json) · [全部7项指标和SD](dynamic_family_metrics.json) · [逐run全精度结果](dynamic_run_metrics.json)"]
    for weight, label in (("date_equal", "日期等权"), ("hour_equal", "小时等权")):
        lines += ["", "## "+label, ""]
        for cohort, title in DC.items():
            lines += ["### "+title, "", "| 天气 | n小时/日期 | 模型 | CRPS201 | CRPS401 | 中央90%覆盖 | 中央98%覆盖 | 排名 |", "|---|---:|---|---:|---:|---:|---:|---:|"]
            for event, etitle in EVENTS.items():
                for r in sorted((x for x in dyn if x["cohort"]==cohort and x["event"]==event and x["weighting"]==weight), key=lambda r:r["rank_crps201"]):
                    lines.append(f"| {etitle} | {r['n_station_hours']}/{r['n_dates']} | {SHORT[r['family']]}＋冻结快变 | {fmt(r['crps201_db'],r['crps201_db_sd'])} | {fmt(r['crps401_db'],r['crps401_db_sd'])} | {r['central90_coverage']:.6f} | {r['central98_coverage']:.6f} | {r['rank_crps201']} |")
            lines.append("")
    lines += ["## 时间结构：英国两站", "", "复用018已验收3027站点小时、32次固定随机重复，16次为敏感性；iid与block_0256沿用相同边缘和冻结256秒块长，不再挑种子。慢变常量在增量与中心化ACF中抵消，因此不能按本表给四类慢变模型排优劣。预设时滞1/2/5/10/30/60/120/300/600/900/1800秒，配对不跨缺口或小时。两个Sparsholt小时在1800秒无有效配对，两个候选共4条空滞后原样保留；这4个候选小时实际用10个时滞，其余6050条用11个。", ""]
    temporal = read(OUT/"temporal_reuse.json")["rows"]
    for weight, label in (("date_equal", "日期等权"), ("hour_equal", "小时等权")):
        lines += ["### "+label+"变差评分（dB²，越低越好）", "", "| 范围 | 天气 | 小时/日期 | iid | block_0256 |", "|---|---|---:|---:|---:|"]
        for station in ("pooled", "chilbolton", "sparsholt"):
            for event, etitle in EVENTS.items():
                rows = {r["candidate"]:r for r in temporal if r["station"]==station and r["event"]==event and r["weighting"]==weight}
                a,b = rows["iid"],rows["block_0256"]
                lines.append(f"| {dict(pooled='英国合并',chilbolton='Chilbolton',sparsholt='Sparsholt')[station]} | {etitle} | {a['n_station_hours']}/{a['n_dates']} | {a['vs32']:.6f} | {b['vs32']:.6f} |")
        lines.append("")
    lines += ["雨小时block_0256的变差评分较低，干燥阴小时iid较低；不据此重新选择方案。11个滞后的ACF、增量与配对数共66594行见[完整原生滞后记录](../Combined_Channel_Revision_018/attempt_003/lag_diagnostics.parquet)，[既有ACF展示和独立逐秒QA说明](../Combined_Channel_Revision_018/REPORT.md)仍保留，旧报告的七类阻塞范围由本版四类决定覆盖。", "", "LOCAL既有快变生成器时间诊断见[冻结快变交付](../Fast_Fading_Revision_002/final_evaluation/delivery/FINAL_RESULTS_ZH.md)，它是既有生成器验证背景；其中旧慢变组合不是本版四类Top10主表。当前LOCAL四类比较为上面的边缘评分，不新增模型特有时序排序。原生秒和配对相互依赖，不当作独立气候样本。中央覆盖低于名义水平的组别保留原值，不宣称所有组校准良好。"]
    md("DYNAMIC_RESULTS.md", "\n".join(lines))

    lines = ["# 日期、缺失与异常台账", "", "评分只使用既有完成日期，没有扩大采集范围。全母集保留6351小时：LOCAL687、Chilbolton3432、Sparsholt2232；对应392个站点日期（156/143/93）。LOCAL是非连续测试小时，不是156个整天。", "", "| 站点 | 源小时 | 标签合格 | 历史共同 | 液态雨有效共同 | 日期数 |", "|---|---:|---:|---:|---:|---:|"]
    for s in ("LOCAL", "chilbolton", "sparsholt"):
        rows = [r for r in coverage if r["station"] == s]
        vals = [sum(r[k] for r in rows) for k in ("source_hours", "label_eligible_hours", "legacy_common_hours", "liquid_common_hours")]
        lines.append(f"| {s} | "+" | ".join(map(str, vals))+f" | {len(rows)} |")
    lines += ["", "英国标签缺失Chilbolton24、Sparsholt13小时；原生覆盖不足95%为26、8小时；标签合格但液态雨负差为31、24小时。5593标签合格再排除55小时，形成5538物理共同小时。各原因、天气other、日期和未评分值都在[全小时账本](all_hour_ledger.json)，未补零。", "", "LOCAL原687表含3小时旧雨量截零异常，仅作历史参考。684补充严格要求t+1h累计tp、sf有限非负且1000×tp−1000×sf非负。3个异常完整保存于[异常表](local_excluded_hours.json)；其中2个干燥阴小时影响动态补充。108小时旧零度层缺失的冻结回退没有重估，也未被此单项液态掩码全剔除，因此684不是无任何近似的物理全集。", "", "## 历史日历中的缺日与气象未完成", "", "[原生日历账本](native_calendar_ledger.json)按冻结旧计划的各站起止范围逐日列出：1127个实际原生文件日，其中236日满足当前气象完成条件，891日未进入完整天气比较；另有5个日期不在冻结原生日历且无对应本地文件。891日不自动称原生数据缺失。这个历史日历不是新的待采任务，不延长天气授权。12月的部分天气记录不冒充完成日期。", "", "未列入原生日历的日期："]
    for r in calendar:
        if not r["native_file_present"]:
            lines.append(f"- {r['station']}：{r['date_utc']}")
    lines += ["", "## 全部已完成/原测试实际日期", "", "以下保留所有日期，包括没有雨/阴评分小时的日期。[逐日计数](coverage_by_date.json)区分原始天气分类与各掩码可评分小时；[评分成员表](sample_membership.json)再列每一比较分组的完整UTC键与实际日期。LOCAL日期为北京时间，英国为UTC。"]
    for s in ("LOCAL", "chilbolton", "sparsholt"):
        ds = sorted(d for station,d in dates if station==s)
        lines += ["", "### "+s+f"（{len(ds)}日期）", "", "、".join(ds)]
    md("COVERAGE_AND_MISSING.md", "\n".join(lines))

    winners = []
    for cohort in ("LOCAL_liquid_input_valid_supplement", "UK_full_common", "UK_chilbolton_full_common", "UK_sparsholt_full_common"):
        for event in EVENTS:
            r = next(x for x in hourly if x["cohort"]==cohort and x["event"]==event and x["rank"]==1)
            winners.append({"cohort":cohort,"event":event,"family":r["family"],"rmse_db_mean":r["rmse_db_mean"],"n":r["n"]})
    put("claims.json", {"status":"MACHINE_EVIDENCE_VERIFIED_NOT_HUMAN_SUBMISSION_APPROVAL", "claims": [
        {"id":"C01","claim":"Four-family hourly metrics and descriptive ranks","sources":["hourly_family_metrics.json","sample_membership.json","verification.json"],"facts":winners},
        {"id":"C02","claim":"Full mother set and explicit LOCAL legal-input supplement","sources":["all_hour_ledger.json","coverage_by_date.json","local_excluded_hours.json","native_calendar_ledger.json","verification.json"]},
        {"id":"C03","claim":"Frozen common-fast marginal and temporal comparisons are separate","sources":["dynamic_family_metrics.json","dynamic_mask_audit.json","temporal_reuse.json","verification.json"]},
        {"id":"C04","claim":"Model exclusions follow user decisions and trial deadline, not scores","sources":["../Combined_Channel_Revision_035/MODEL_SCOPE.json","../Combined_Channel_Revision_035/REPORT.md"]}], "human_manuscript_review":"PENDING_IF_USED_FOR_SUBMISSION"})
    lines = ["# 四类模型对比：最终中文报告", "", "当前四类范围的实验比较与独立数值QA已完成：本文冻结Top10、Bai LSTM LOCAL适配、TabM LOCAL适配、组合ITU近似。结果、全部日期和异常台账已合并到Revision036；最终文件终检以[final_verification.json](final_verification.json)为准。没有新增训练、权重调整、模型推理、随机轨迹或天气请求。", "", "## 直接结论", "", "LOCAL雨小时TabM误差最低，干燥阴小时本文模型最低。英国合并雨小时TabM最低，干燥阴小时ITU近似最低；两站情况不同，不能只报合并值。本文并非所有组别第一。下表只摘录各组RMSE最低者，四类全部RMSE、MAE、Bias、种子SD和排名见[完整小时表](HOURLY_RANKINGS.md)。[证据C01](claims.json)", "", "| 同样本范围 | 天气 | 最低RMSE模型 | RMSE均值(dB) | n小时 |", "|---|---|---|---:|---:|"]
    for r in winners:
        lines.append(f"| {COHORTS[r['cohort']]} | {EVENTS[r['event']]} | {NAMES[r['family']]} | {r['rmse_db_mean']:.6f} | {r['n']} |")
    lines += ["", "## 样本范围与解释", "", "LOCAL原687小时、156个日期完整保留。旧ITU输入有3小时负液态差被历史代码截零：687口径单列历史参考；684小时输入有效补充是明确同样本对比，未静默删除原母集。雨51小时/26日期不变，干燥阴从158小时/70日期变为156小时/69日期。英国保留全部Chilbolton143日和Sparsholt93日：5664气象完整小时、5593标签合格、5538物理共同小时，不回退39日。[完整范围、异常与日期](COVERAGE_AND_MISSING.md)；[证据C02](claims.json)。", "", "雨/干燥阴是小时分类，不是把含部分雨小时的整日标为雨天。英国总降水水当量≥0.1mm且雪水当量<0.1mm为雨；二者<0.1mm且总云量≥0.8为干燥阴，other保留。小时降水结束时刻为t+1h；LOCAL沿用冻结分类。两站合并按小时汇总，不是平均站点RMSE。多种子用17/29/43逐run指标后均值与样本SD；时间/空间相关样本不作独立重复，排名不等于显著性。", "", "## 动态比较另列", "", "[独立动态结果](DYNAMIC_RESULTS.md)包含共同冻结快变的边缘CRPS、覆盖率，以及英国既定iid/block_0256时间结构。LOCAL补充动态只重聚合已存分数，剔除2个液态输入异常干燥阴小时，保留原表。英国3027雨阴小时时间诊断没有重生成轨迹。慢变常量在增量/中心化ACF中抵消，时间结构表不能给四类小时模型排优劣。[证据C03](claims.json)", "", "## 模型身份和去留", "", "Bai LSTM、TabM使用既有LOCAL适配权重，不是作者原始预训练权重。ITU是固定物理气体变化＋单份云液水＋几何雨衰近似，不是完整ITU概率模型、不含闪烁；晴空参考、归一化和快变CDF未重估。本文Top10也只复用已有冻结版本。输入、单位、几何与代码/权重来源见[四类输入来源矩阵](MODEL_INPUT_MATRIX.md)。", "", "E-SST因两小时试下载净新增仅888/421104所需数值、剩余工期不支持英国输入和计算，按用户条件授权退出正式表；LOCAL028/029及下载记录保留作历史补充。RMOP、NWP–PRM已按用户决定排除，MTS保持未计算。排除不是实验完成，更不是用0分参与排名；本次完成的是最终约定的四类比较，不是原七类完整复现。[证据C04](claims.json)", "", "## 交付核验", "", f"独立实现重算80个小时run指标、40个家族行、160个动态run汇总、80个动态家族行和24个时间结构汇总。小时差异最大{qa['max_absolute_errors']['hourly_run_metrics']:.3g}dB，动态汇总最大{qa['max_absolute_errors']['dynamic_run_aggregation']:.3g}；均属浮点舍入。独立指另一算术实现，不是外部人工同行评审。[QA](QA.md)、[执行及修正记录](COMMANDS.md)、[机器核验](verification.json)、[接续入口](HANDOFF.md)。", "", "统计核验技能用于保持掩码、嵌套样本与种子SD口径；科学写作技能把结论绑定到证据并区分实验交付与作者投稿批准。本文稿件若使用这些结果，仍需作者确认措辞；这不是数据计算的剩余阻塞。"]
    md("REPORT.md", "\n".join(lines))
    assert all(sha(p) == h for p, h in union.items())
    put("publication_build.json", {"status":"REPORTS_RENDERED_PENDING_FILE_QA", "utc":datetime.now(timezone.utc).isoformat(),
        "source_pins_checked":len(union), "original_model_pins_checked":len(old_models), "original_model_new_flattened_pins":len(set(old_models)-set(sources)),
        "model_checkpoints":7, "native_calendar_rows":len(calendar), "native_calendar_counts":dict(counts),
        "raw_calendar_matches_existing_local_files":True, "new_remote_calls":0, "new_prediction_calls":0, "new_trajectories":0,
        "rendered_sha256":{p.name:sha(p) for p in OUT.glob("*.md")}, "publisher_sha256":sha(__file__)})
    print("REPORTS_RENDERED_PENDING_FILE_QA")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        put("publication_failure_"+stamp+".json", {"traceback":traceback.format_exc()})
        raise
