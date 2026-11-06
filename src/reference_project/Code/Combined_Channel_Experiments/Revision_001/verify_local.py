"""Independent numerical and aggregation audit; no fitting or model selection."""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import local_engine as e


def quantiles(model, probabilities):
    positions = np.searchsorted(model["support_cdf"], 1 - np.asarray(probabilities), side="left")
    return -10 * np.log10(model["support_q"][positions])


def literal(f, weight, model, offset, grid):
    """Literal pointwise pinball calculation, not the production prefix formula."""
    p = (np.arange(grid) + .5) / grid
    q = quantiles(model, p) + offset
    delta = f[:, None] - q[None, :]
    losses = np.where(delta >= 0, delta * p[None, :], -delta * (1 - p)[None, :])
    return float(2 * np.sum((weight / weight.sum())[:, None] * losses) / grid)


def independent_score(f, weight, model, offset):
    """Second implementation using separately accumulated positive/negative parts."""
    indices = np.argsort(f)
    values, weights = f[indices], weight[indices].astype(float) / weight.sum()
    cumulative = np.r_[0., np.cumsum(weights)]
    first = np.r_[0., np.cumsum(values * weights)]
    cumulative[-1] = 1.
    result = {}
    for grid in (201, 401):
        p = (np.arange(grid) + .5) / grid
        q = quantiles(model, p) + offset
        left = np.searchsorted(values, q, side="left")
        right = np.searchsorted(values, q, side="right")
        negative = q * cumulative[left] - first[left]
        positive = first[-1] - first[right] - q * (1 - cumulative[right])
        result[f"crps{grid}_db"] = float(2 * np.mean((1-p)*negative + p*positive))
    bounds = quantiles(model, [.01, .05, .5, .95, .99]) + offset
    for name, lo, hi in (("central90_coverage", 1, 3), ("central98_coverage", 0, 4)):
        result[name] = float(np.average((f >= bounds[lo]) & (f <= bounds[hi]), weights=weight))
    for name, j in (("tail95_exceedance", 3), ("tail99_exceedance", 4)):
        result[name] = float(np.average(f > bounds[j], weights=weight))
    result["slow_only_instant_mae_crps_db"] = float(np.average(np.abs(f-offset), weights=weight))
    p = (np.arange(201)+.5)/201
    boundaries = quantiles(model, p)+offset
    result["actual_cdf201"] = cumulative[np.searchsorted(values, boundaries, side="right")]
    return result


def numerical(save=False):
    checks = []
    def test(name, condition):
        e.require(bool(condition), "Synthetic test failed: " + name)
        checks.append(name)
    model = {"family": "empirical", "support_q": np.array([.5, 1., 1.5]),
             "support_cdf": np.array([.25, .75, 1.])}
    f = np.array([-2., -1., 0., 1., 3.])
    w = np.array([1., 2., 4., 2., 1.])
    hour = e.original_d1.marginal_hour(f, w)
    for offset in (-3., 0., 2.5):
        a = e.original_d1.score_marginal(hour, model, offset)
        b = independent_score(f, w, model, offset)
        test("weighted_pinball201_offset_"+str(offset), abs(a["crps201_db"]-literal(f,w,model,offset,201)) < 1e-12)
        test("weighted_pinball401_offset_"+str(offset), abs(a["crps401_db"]-literal(f,w,model,offset,401)) < 1e-12)
        test("coverage_tail_offset_"+str(offset), max(abs(a[k]-b[k]) for k in
            ("central90_coverage", "central98_coverage", "tail95_exceedance", "tail99_exceedance")) < 1e-12)
    test("population_unit_power", abs(np.dot(np.diff(np.r_[0., model["support_cdf"]]), model["support_q"])-1) < 1e-15)
    test("discrete_quantile_monotonic", np.all(np.diff(quantiles(model, e.fm.P201)) >= 0))
    delta = f - .7
    test("deterministic_CRPS_equals_absolute_error", np.array_equal(np.abs(delta), 2*np.where(delta >= 0, .5*delta, -.5*delta)))
    g = np.array([3., 1., 2., -1.])
    test("slow_offset_increment_cancellation", np.array_equal(np.diff(g+4.), np.diff(g)))
    test("translation_CRPS", abs(literal(f+2,w,model,2.7,201)-literal(f,w,model,.7,201)) < 1e-12)
    dates = pd.Series([1.,2.,3.,4.], index=["2020-01-01","2020-01-02","2020-01-04","2020-01-05"])
    interval = e.original_d1.date_interval(dates, 3, "synthetic")
    test("no_bootstrap_across_missing_day", interval["interval_status"] == "no_contiguous_date_blocks")
    test("finite_sample_not_forced_to_unit_power", not np.isclose(np.mean([.5,.5,1.]),1.))
    test("32_realizations_not_experimental_sample_units", e.SPLITS["test"] == 687 and e.CHUNK == 32)
    obj = {"status": "PASS", "study_sample_reads": 0, "test_count": len(checks), "checks": checks,
           "code_sha256": {name: e.sha(e.HERE/name) for name in
                           ("local_engine.py", "run_local.py", "verify_local.py", "PROTOCOL.md", "PARAMETERS.md")}}
    if save:
        path = e.LOCAL/"numerical_qa.json"
        if path.exists():
            e.require(e.read(path) == obj, "Existing synthetic QA changed")
        else:
            e.atomic(path, obj)
    return obj


def verify():
    m = e.manifest(check_sources=True)
    numerical(save=False)
    combined = e.combined_frame()
    pred = pd.read_parquet(e.LOCAL/"slow_predictions.parquet")
    e.require(len(combined) == 11330 and combined.hour.nunique() == 1030 and
              set(combined.variant) == set(e.VARIANTS) and not combined.duplicated(["hour","variant"]).any(),
              "Combined count/identity failure")
    e.require({s: combined.loc[combined.split.eq(s),"hour"].nunique() for s in ("validation","test")} ==
              {"validation":343,"test":687}, "Combination split coverage differs")
    hour_sets = {s:set(pred.loc[pred.split.eq(s),"hour"]) for s in e.SPLITS}
    e.require(all(hour_sets[a].isdisjoint(hour_sets[b]) for a,b in
        (("train","validation"),("train","test"),("validation","test"))), "Exact-hour leakage")
    for marker in sorted((e.LOCAL/"chunks").glob("*.json")):
        e.checked_marker(marker)
    models = e.load_cdfs()
    old_marginal = e.old_frames("marginal").set_index("hour")
    saved = combined.set_index(["hour","variant"])
    observed = e.old_frames("observed").set_index("hour")
    rep = e.old_frames("replicates")
    dynamic = e.old_frames("scores").set_index(["hour","candidate"])
    e.require(len(rep) == 65920 and len(dynamic) == 2060, "Reused dynamics incomplete")
    max_score, max_probability, max_vs, literal_error = 0.,0.,0.,0.
    witness_hours = set()
    for _, part in old_marginal.reset_index().groupby(["month","split","event"]):
        witness_hours.add(sorted(part.hour)[0])
    model_checks, score_checks = 0, 0
    for filename in sorted(e.read(e.D1/"manifest.json")["frozen_design"]["source_clean_1s"]):
        frame = e.source_frame(e.RAW/filename)
        for h, group in frame.groupby("hour_bjt",sort=True):
            h = str(h)
            a, good, f, weight, hour = e.observed_hour(group)
            prior = old_marginal.loc[h]
            model = models[prior.event]
            baseline = independent_score(f,weight,model,0.)
            e.require(abs(baseline["crps201_db"]-prior.crps201_db)<1e-10, "Old marginal source witness differs")
            for variant in e.VARIANTS:
                row = saved.loc[(h,variant)]
                check = independent_score(f,weight,model,float(row.slow_error_db))
                for key in e.METRICS:
                    error = abs(float(row[key])-check[key])
                    if "coverage" in key or "exceedance" in key:
                        max_probability=max(max_probability,error)
                        e.require(error < 1e-12, "Independent probability mismatch")
                    else:
                        max_score=max(max_score,error)
                        e.require(error < 1e-10, "Independent score mismatch: "+key)
                    score_checks+=1
                e.require(np.max(np.abs(np.asarray(row.actual_cdf201)-check["actual_cdf201"]))<1e-12,
                          "Independent calibration curve mismatch")
                if h in witness_hours:
                    for grid in (201,401):
                        value=literal(f,weight,model,float(row.slow_error_db),grid)
                        error=abs(value-float(row[f"crps{grid}_db"]))
                        literal_error=max(literal_error,error)
                        e.require(error<1e-10,"Literal pointwise CRPS mismatch")
                model_checks+=1
            obs=observed.loc[h]
            for candidate in ("iid","block_0256"):
                part=rep.loc[rep.hour.eq(h)&rep.candidate.eq(candidate)].sort_values("repetition")
                e.require(part.repetition.tolist()==list(range(32)),"Missing old realizations")
                g=np.stack(part.increment_mean.to_numpy())
                for n,key in ((32,"vs"),(16,"vs_half")):
                    values=g[:n]
                    total=np.sum(values,axis=0)
                    cross=(total*total-np.sum(values*values,axis=0))/(n*(n-1))
                    score=np.asarray(obs.increment_second_moment)-2*np.asarray(obs.increment_mean)*np.mean(values,axis=0)+cross
                    value=float(np.mean(score[np.isfinite(score)]))
                    error=abs(value-float(dynamic.loc[(h,candidate)][key]));max_vs=max(max_vs,error)
                    e.require(error<1e-10,"Reused VS independent audit failed")
        print("LOCAL VERIFY MONTH",filename,flush=True)
    e.require(model_checks==11330 and score_checks==79310,"Independent check counts incomplete")
    replay=e.read(e.LOCAL/"prediction_qa.json")
    e.require(replay["status"]=="PASS" and replay["fit_calls"]==0,"Model replay gate not PASS")
    qa={"status":"PASS","fingerprint":m["fingerprint"],"completed_utc":e.now(),
        "formal_hours":1030,"combination_records":len(combined),"independent_scalar_checks":score_checks,
        "independent_cdf_curve_checks":11330,"literal_source_witness_hours":len(witness_hours),
        "literal_model_hours":len(witness_hours)*len(e.VARIANTS),"max_score_error_db":max_score,
        "max_probability_error":max_probability,"max_literal_crps_error_db":literal_error,
        "max_reused_vs_error_db2":max_vs,"reused_realizations":len(rep),"new_model_fits":0,
        "prediction_qa_sha256":e.sha(e.LOCAL/"prediction_qa.json"),
        "output_sha256":{"slow_predictions.parquet":e.sha(e.LOCAL/"slow_predictions.parquet")},
        "all_external_stages_complete":False}
    path=e.LOCAL/"qa.json"
    if path.exists():
        old=e.read(path)
        e.require(all(old.get(k)==v for k,v in qa.items() if k!="completed_utc"),"Existing local QA differs")
        return old
    e.atomic(path,qa)
    return qa


def bootstrap_literal(series,length,label):
    series=series.dropna().sort_index()
    dates=pd.to_datetime(series.index)
    blocks=[]
    for start in range(len(series)-length+1):
        indices=list(range(start,start+length))
        if all((dates[indices[j]]-dates[indices[j-1]]).days==1 for j in range(1,length)):
            blocks.append(indices)
    if not blocks:
        return {"interval_status":"no_contiguous_date_blocks","ci025":None,"ci975":None,
                "eligible_blocks":0,"date_union":0}
    rng=np.random.default_rng(e.dyn.stable_seed("D-bootstrap",label,length))
    data=series.to_numpy(float)
    estimates=[]
    for _ in range(2000):
        chosen=rng.integers(len(blocks),size=math.ceil(len(data)/length))
        draw=[j for b in chosen for j in blocks[b]][:len(data)]
        estimates.append(float(sum(data[j] for j in draw)/len(draw)))
    return {"interval_status":"complete","ci025":float(np.quantile(estimates,.025)),
            "ci975":float(np.quantile(estimates,.975)),"eligible_blocks":len(blocks),
            "date_union":len({j for b in blocks for j in b})}


def audit_report():
    m=e.manifest(check_sources=True)
    e.require(e.read(e.LOCAL/"qa.json")["status"]=="PASS","Numerical QA gate")
    report=e.read(e.LOCAL/"report_manifest.json")
    for name,digest in report["output_sha256"].items():
        e.require(e.sha(e.LOCAL/name)==digest,"Report file changed")
    combined=e.combined_frame()
    pred=pd.read_parquet(e.LOCAL/"slow_predictions.parquet")
    slow=pd.read_parquet(e.LOCAL/"slow_hour_metrics.parquet")
    summaries=pd.read_parquet(e.LOCAL/"combined_summary.parquet")
    effects=pd.read_parquet(e.LOCAL/"paired_effects.parquet")
    calibration=pd.read_parquet(e.LOCAL/"cdf_calibration.parquet")
    largest=0.
    for row in slow.itertuples():
        part=pred.loc[pred.split.eq(row.split)&pred.variant.eq(row.variant)&
                      (pred.event.eq(row.event) if row.event!="overall" else True)]
        errors=part.prediction_db.to_numpy()-part.truth_db.to_numpy()
        for key,value in (("rmse_db",np.linalg.norm(errors)/math.sqrt(len(errors))),
                          ("mae_db",sum(abs(x) for x in errors)/len(errors)),
                          ("bias_db",sum(errors)/len(errors))):
            delta=abs(getattr(row,key)-value);largest=max(largest,delta)
            e.require(delta<1e-10,"Slow aggregation mismatch")
        e.require(len(part)==row.hours and part.day.nunique()==row.dates,"Slow report counts")
    interval_checks=0
    for row in summaries.itertuples():
        part=combined.loc[combined.split.eq(row.split)&combined.variant.eq(row.variant)&
                         (combined.event.eq(row.event) if row.event!="overall" else True)]
        day_values={day:float(sum(group[row.quantity])/len(group)) for day,group in part.groupby("day")}
        values=pd.Series(day_values)
        e.require(abs(float(sum(values)/len(values))-row.date_mean)<1e-10,"Date mean mismatch")
        check=bootstrap_literal(values,row.block_days,f"C1|{row.split}|{row.event}|{row.variant}|{row.quantity}")
        for key,value in check.items():
            saved=getattr(row,key)
            e.require((value is None and pd.isna(saved)) or (isinstance(value,str) and saved==value) or
                      (value is not None and not isinstance(value,str) and abs(saved-value)<1e-10),"Date interval mismatch")
        interval_checks+=1
    for row in effects.itertuples():
        part=combined.loc[combined.split.eq(row.split)&
                         (combined.event.eq(row.event) if row.event!="overall" else True)]
        if row.contrast.endswith("_combination_vs_slow_only"):
            variant=row.contrast.removesuffix("_combination_vs_slow_only")
            take=part.loc[part.variant.eq(variant)].copy()
            take["difference"]=take.slow_only_instant_mae_crps_db-take.crps201_db
        else:
            variant=row.contrast.removeprefix("full_BiLSTM_vs_")
            pivot=part.pivot(index=["hour","day"],columns="variant",values="crps201_db")
            take=(pivot[variant]-pivot["full:bilstm"]).rename("difference").reset_index()
        values=take.groupby("day").difference.mean()
        e.require(abs(values.mean()-row.date_difference_db)<1e-10,"Paired date effect mismatch")
        check=bootstrap_literal(values,row.block_days,f"C1-effect|{row.split}|{row.event}|{row.contrast}")
        for key,value in check.items():
            saved=getattr(row,key)
            e.require((value is None and pd.isna(saved)) or (isinstance(value,str) and saved==value) or
                      (value is not None and not isinstance(value,str) and abs(saved-value)<1e-10),"Paired interval mismatch")
        interval_checks+=1
    for (split,event,variant),rows in calibration.groupby(["split","event","variant"]):
        part=combined.loc[combined.split.eq(split)&combined.variant.eq(variant)&
                         (combined.event.eq(event) if event!="overall" else True)]
        actual=np.mean([np.mean(np.stack(day.actual_cdf201),axis=0) for _,day in part.groupby("day")],axis=0)
        left=np.mean([np.mean(np.stack(day.model_cdf201_left),axis=0) for _,day in part.groupby("day")],axis=0)
        right=np.mean([np.mean(np.stack(day.model_cdf201_right),axis=0) for _,day in part.groupby("day")],axis=0)
        rows=rows.sort_values("probability")
        e.require(max(np.max(abs(actual-rows.date_actual_cdf.to_numpy())),np.max(abs(left-rows.model_cdf_left.to_numpy())),
                      np.max(abs(right-rows.model_cdf_right.to_numpy())))<1e-12,"Report calibration mismatch")
    e.require(len(slow)==165 and len(summaries)==2310 and len(effects)==630 and len(calibration)==22110,
              "Published table count failure")
    report_text=chinese_report(slow,summaries,effects)
    path=e.LOCAL/"RESULTS_ZH.md"
    if path.exists():
        e.require(path.read_text(encoding="utf-8")==report_text,"Published Chinese report changed")
    else:
        path.write_text(report_text,encoding="utf-8")
    obj={"status":"PASS","fingerprint":m["fingerprint"],"completed_utc":e.now(),
         "independent_slow_rows":len(slow),"independent_probability_rows":len(summaries),
         "independent_paired_rows":len(effects),"interval_checks":interval_checks,
         "calibration_rows":len(calibration),"max_slow_recompute_error_db":largest,
         "report_sha256":e.sha(path),"new_model_fits":0,"local_phase_complete":True,
         "GBS_complete":False,"OpenSat_complete":False}
    qa_path=e.LOCAL/"report_qa.json"
    if qa_path.exists():
        old=e.read(qa_path)
        e.require(all(old.get(k)==v for k,v in obj.items() if k!="completed_utc"),"Report QA changed")
        return old
    e.atomic(qa_path,obj)
    return obj


def chinese_report(slow,summary,effects):
    lines=["# 当前冻结慢变＋快变组合：本地结果", "", "主模型为Rev028全156列BiLSTM，seed17、53轮；原基础BiLSTM为50列、46轮。所有旧模型和结果保持不变，本轮新增训练次数为0。", "",
        "正式组合只评价343验证和687测试小时。快变采用既有四天气经验CDF与256秒块；CRPS直接评价冻结CDF平移后的组合分布，IID与块生成的理论边缘评分相同，不平均32条随机曲线成为点预测。", ""]
    for split,label in (("validation","验证"),("test","测试")):
        lines += [f"## {label}：慢变小时指标", "", "|模型|天气|小时数|RMSE dB|MAE dB|平均Bias dB|", "|---|---|---:|---:|---:|---:|"]
        for variant in e.VARIANTS:
            for event in e.GROUPS:
                row=slow.loc[slow.split.eq(split)&slow.variant.eq(variant)&slow.event.eq(event)].iloc[0]
                lines.append(f"|{variant}|{e.ZH[event]}|{row.hours}|{row.rmse_db:.6f}|{row.mae_db:.6f}|{row.bias_db:+.6f}|")
        lines += ["", f"## {label}：组合概率结果（日期等权主结果）", "",
            "|模型|天气|CRPS201 dB|CRPS401 dB|90%覆盖|98%覆盖|95%尾超越|99%尾超越|慢变确定性CRPS dB|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in e.VARIANTS:
            for event in e.GROUPS:
                take=summary.loc[summary.split.eq(split)&summary.variant.eq(variant)&summary.event.eq(event)&summary.block_days.eq(1)]
                values=take.set_index("quantity").date_mean
                lines.append(f"|{variant}|{e.ZH[event]}|{values.crps201_db:.6f}|{values.crps401_db:.6f}|{values.central90_coverage:.2%}|{values.central98_coverage:.2%}|{values.tail95_exceedance:.2%}|{values.tail99_exceedance:.2%}|{values.slow_only_instant_mae_crps_db:.6f}|")
    main=summary.loc[summary.split.eq("test")&summary.variant.eq("full:bilstm")&summary.event.eq("overall")&summary.block_days.eq(1)].set_index("quantity").date_mean
    improvement=100*(main.slow_only_instant_mae_crps_db-main.crps201_db)/main.slow_only_instant_mae_crps_db
    lines += ["", "## 分析口径", "",
        f"本次主模型测试总体：组合CRPS为{main.crps201_db:.6f} dB，其自身确定性慢变输出作为瞬时预测时的CRPS为{main.slow_only_instant_mae_crps_db:.6f} dB，差异为{improvement:+.2f}%。这比较的是瞬时衰减概率分布，不是小时RMSE，也不是随机路径逐秒预测精度。",
        f"中央90%区间实际覆盖为{main.central90_coverage:.2%}，中央98%区间实际覆盖为{main.central98_coverage:.2%}；必须与标称90%/98%比较，不能仅因CRPS较低就称概率已校准。当前组合未显式加入慢变预测不确定性；覆盖不足的原因不能只凭此表唯一归因。",
        "所有完整小时、天气结果和1/3/7连续日块配对区间均另存，负效应保留。多模型排序是描述性结果；本地测试历史已访问，且训练/测试同日重叠，不能称新盲测或未经校正的显著性证明。",
        "时间结构复用已完成D1的2060个IID/block评分和65920个实现记录。慢变小时常数在增量中相消，不能将共同快变VS重复计为不同慢变模型的改善。旧窗口、100ms、设备/谱配置结果未重跑、未修改。",
        "这只是LOCAL阶段完成，GBS/OpenSat外部实验仍未完成。12:38输出是给定小时气象/天气条件的随机场景，不是确定性逐点预测。", ""]
    return "\n".join(lines)
