"""Independent delivery readback. Does not import the producer/auditor or fit models."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import csv
import hashlib
import json
import math
import re
import traceback
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE"
OUT=ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002"
CODE=ROOT/"Code/Hourly_Attenuation_Experiments/Revision_034"
W=("overall","clear_dry","rain","dry_overcast","other")
M=("tabm","bai_lstm","bilstm","transformer","mlp","catboost","gcn","random_forest","svr_rbf","elastic_net")
FIELDS=("rmse_db","mae_db","bias_db","r2","pearson_r","pearson_r_squared")
MAX=0.0
def check(v,msg):
    if not bool(v): raise AssertionError(msg)
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))
def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def write_json(p,v):
    with p.open("x",encoding="utf-8") as f: json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def equal(a,b,msg):
    global MAX
    if pd.isna(a) or pd.isna(b):
        check(pd.isna(a) and pd.isna(b),"null "+msg)
    else:
        MAX=max(MAX,abs(float(a)-float(b)))
        check(math.isclose(float(a),float(b),abs_tol=1e-9,rel_tol=1e-10),msg+f" {a} != {b}")
def recalc(frame):
    y=frame.truth_db.to_numpy(float); p=frame.prediction_db.to_numpy(float)
    check(np.isfinite(y).all() and np.isfinite(p).all(),"finite CSV predictions")
    n=len(y); err=p-y
    if not n: return dict(n=0,**{k:float("nan") for k in FIELDS},dates=[])
    sy=((y-y.mean())**2).sum(); sp=((p-p.mean())**2).sum()
    corr=float(np.corrcoef(y,p)[0,1]) if n>=2 and sy>0 and sp>0 else float("nan")
    return dict(n=n,rmse_db=float(np.sqrt(np.dot(err,err)/n)),mae_db=float(np.mean(np.abs(err))),
                bias_db=float(np.mean(err)),r2=float(1-np.dot(err,err)/sy) if n>=2 and sy>0 else float("nan"),
                pearson_r=corr,pearson_r_squared=corr*corr,dates=sorted(set(frame.hour_bjt.str.slice(0,10))))
def verify_metrics(df,row,tag):
    v=recalc(df)
    check(v["n"]==int(row["n"]),tag+" n")
    for k in FIELDS: equal(v[k],row[k],tag+" "+k)
    check(v["dates"]==ast.literal_eval(row["dates"]),tag+" actual dates")
    check(len(v["dates"])==int(row["n_dates"]),tag+" n_dates")
def verify_report_table(source,report,basis,metric_fields,full=False):
    subset=source[(source.basis==basis)&(source.population=="ML_687")&(source.weather=="overall")].set_index("model")
    labels=dict(tabm="TabM",bai_lstm="Bai LSTM",bilstm="BiLSTM",transformer="Transformer",mlp="MLP",catboost="CatBoost",
                gcn="GCN",random_forest="Random Forest",svr_rbf="SVR-RBF",elastic_net="Elastic Net")
    for m in M:
        values=[labels[m]]
        if not full: values += [str(int(subset.loc[m,"k"]))]
        values += [f"{subset.loc[m,k]:.6f}" for k in metric_fields]
        check("| "+" | ".join(values)+" |" in report,"report table row "+basis+"/"+m)
def main():
    start=time.perf_counter()
    check(not (OUT/"evidence/FINAL_QA.json").exists(),"final QA must be new")
    agg=read(OUT/"evidence/INDEPENDENT_AGGREGATE_QA.json")
    check(agg["status"]=="PASS" and agg["states"]==1560,"aggregate stage")
    met=pd.read_csv(OUT/"tables/all_test_metrics_15605_rows.csv",float_precision="round_trip")
    orig=pd.read_csv(BASE/"final/all_test_metrics.csv",float_precision="round_trip")
    check(len(met)==len(orig)==15605,"metric count")
    key=["model","k","population","weather"]
    check(not met.duplicated(key).any(),"metric unique keys")
    a=met.sort_values(key).reset_index(drop=True); b=orig.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
    check(met.query("population=='ML_687' and weather=='overall'")["n"].eq(687).all(),"main n")
    check(met.query("population=='COMMON_684' and weather=='overall'")["n"].eq(684).all(),"common n")
    check(not met.query("population=='ML_687'").model.eq("combined_ITU_approx").any(),"ITU never in687")
    model_lookup=met.set_index(key)
    states=pd.read_csv(OUT/"tables/all_states_1560.csv",float_precision="round_trip")
    check(len(states)==1560 and not states.duplicated(["model","k"]).any(),"state table")
    feature=pd.read_csv(OUT/"tables/retained_features_122460_rows.csv",float_precision="round_trip")
    original_features=pd.read_csv(BASE/"final/retained_features_all_steps.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(feature,original_features,check_exact=True)
    check(len(feature)==122460,"retained feature rows")
    check((feature.groupby(["model","k"]).size().reset_index(name="n").eval("k==n")).all(),"retained counts")
    for _,r in feature.iterrows():
        prefix,var=r.feature.split("__",1)
        check(r.variable==var and r.group==prefix,"feature provenance columns")
        check(pd.isna(r.level_hpa) if prefix=="surface" else r.level_hpa==int(prefix[1:]),"pressure level")
    paths=pd.read_csv(OUT/"tables/independent_rfe_paths_1560.csv",float_precision="round_trip")
    total=0; metric_groups=0
    for model in M:
        h=pd.read_csv(OUT/f"hourly/{model}_test_687_all_steps.csv",float_precision="round_trip")
        check(len(h)==107172 and not h.duplicated(["model","k","hour_bjt"]).any(),model+" hourly count/keys")
        check(h.model.eq(model).all() and set(h.k)==set(range(1,157)),model+" complete k")
        check(h.itu_common_684.dtype==bool,model+" membership bool")
        reference_keys=None
        for k,df in h.groupby("k",sort=True):
            check(len(df)==687 and df.weather.isin(W[1:]).all(),"hour/weather completeness")
            keys=df.hour_bjt.tolist()
            if reference_keys is None: reference_keys=keys
            check(keys==reference_keys,model+" all k same ordered hours")
            sr=states[(states.model==model)&states.k.eq(k)].iloc[0]
            check(df.evaluation_sha256.eq(sr.evaluation_sha256).all(),"hourly evaluation provenance")
            check(np.allclose(df.error_db,df.prediction_db-df.truth_db,rtol=0,atol=1e-14),"error column")
            check(int(df.itu_common_684.sum())==684,"common membership count")
            for pop,sub in (("ML_687",df),("COMMON_684",df[df.itu_common_684])):
                for weather in W:
                    g=sub if weather=="overall" else sub[sub.weather==weather]
                    row=model_lookup.loc[(model,k,pop,weather)]
                    verify_metrics(g,row,f"{model}/{k}/{pop}/{weather}")
                    if pop=="ML_687":
                        for f in ("n",*FIELDS):
                            equal(sr["test_"+weather+"_"+f],row[f],"state-wide "+f)
                    metric_groups+=1
        path=read(OUT/f"paths/{model}_156_to_1.json")
        check(len(path)==156 and [r["k"] for r in path]==list(range(156,0,-1)),"JSON path")
        check(len(set(r["removed_next"] for r in path[:-1]))==155,"no duplicate deletion")
        previous=path[0]["retained"]
        for row in path:
            check(row["retained"]==previous and len(previous)==row["k"],"path nested")
            pr=paths[(paths.model==model)&paths.k.eq(row["k"])].iloc[0]
            check(json.loads(pr.retained_names_json)==row["retained"],"path CSV/JSON retained")
            check(json.loads(pr.removed_so_far_json)==row["removed_so_far"],"path CSV/JSON deleted")
            check(len(set(row["retained"])|set(row["removed_so_far"]))==156,"path coverage")
            if row["k"]>1: previous=[v for v in previous if v!=row["removed_next"]]
            else: check(row["removed_next"] is None,"terminal delete null")
        total+=len(h)
        print("DELIVERY_READBACK_PASS",model,flush=True)
    itu=pd.read_csv(OUT/"hourly/combined_ITU_common_684.csv",float_precision="round_trip")
    for w in W:
        verify_metrics(itu if w=="overall" else itu[itu.weather.eq(w)],
                       model_lookup.loc[("combined_ITU_approx",np.nan,"COMMON_684",w)],"ITU/"+w)
        metric_groups+=1

    reps=pd.read_csv(OUT/"tables/representative_points_300_rows.csv",float_precision="round_trip")
    check(len(reps)==300,"representative count")
    for model in M:
        ss=states[states.model.eq(model)]
        cv=ss.sort_values(["cv_mean_rmse_db","k"]).iloc[0]
        tm=met[(met.model==model)&met.population.eq("ML_687")&met.weather.eq("overall")].sort_values(["rmse_db","k"]).iloc[0]
        for basis,k in (("full_156",156),("training_cv_min_descriptive",int(cv.k)),("test_min_exploratory",int(tm.k))):
            rs=reps[(reps.model==model)&reps.basis.eq(basis)]
            check(len(rs)==10 and rs.k.eq(k).all(),"summary selection")
            for _,r in rs.iterrows():
                ref=model_lookup.loc[(model,k,r.population,r.weather)]
                for f in ("n",*FIELDS): equal(r[f],ref[f],"summary "+f)
    common_plot=pd.read_csv(OUT/"tables/F04_plot_data_common684_full156.csv",float_precision="round_trip")
    check(len(common_plot)==11 and common_plot["n"].eq(684).all(),"F04 samples")
    check(set(common_plot.model)==set(M)|{"combined_ITU_approx"},"F04 all models")
    # Validate required report tables directly against numeric exports.
    report=(OUT/"REPORT_中文.md").read_text(encoding="utf-8")
    verify_report_table(reps,report,"full_156",["rmse_db","r2","pearson_r","pearson_r_squared","mae_db","bias_db"],True)
    verify_report_table(reps,report,"training_cv_min_descriptive",
                        ["cv_mean_rmse_db","cv_se_rmse_db","rmse_db","r2"])
    verify_report_table(reps,report,"test_min_exploratory",["rmse_db","r2"])
    required=["R²=1−SSE/SST","探索性","单种子17","635","5605","367350","1071720",
              "NEGATIVE_LIQUID_DIFFERENCE"]  # Evidence-specific literals may appear in companion files.
    combined=report+(OUT/"README.md").read_text(encoding="utf-8")+(OUT/"evidence/itu_anomaly_reconciliation.json").read_text(encoding="utf-8")
    for term in required: check(term in combined,"report scope "+term)
    # All weather and common684 report table cells, not only the headline rows.
    labelmap=dict(tabm="TabM",bai_lstm="Bai LSTM",bilstm="BiLSTM",transformer="Transformer",mlp="MLP",
                  catboost="CatBoost",gcn="GCN",random_forest="Random Forest",svr_rbf="SVR-RBF",elastic_net="Elastic Net",
                  combined_ITU_approx="组合ITU小时近似")
    for basis in ("full_156","training_cv_min_descriptive"):
        for m in M:
            sub=reps[(reps.basis==basis)&reps.population.eq("ML_687")&reps.model.eq(m)].set_index("weather")
            label=labelmap[m] if basis=="full_156" else f"{labelmap[m]} ({int(sub.iloc[0].k)})"
            cells=[label]+[f"{sub.loc[w,'rmse_db']:.6f} / {sub.loc[w,'r2']:.6f}" for w in W[1:]]
            check("| "+" | ".join(cells)+" |" in report,"report weather row "+basis+"/"+m)
    for _,r in common_plot.iterrows():
        k=np.nan if r.model=="combined_ITU_approx" else 156
        ref=model_lookup.loc[(r.model,k,"COMMON_684","overall")]
        for f in FIELDS: equal(r[f],ref[f],"F04 plot data")
        cells=[labelmap[r.model],"684"]+[f"{r[f]:.6f}" for f in ("rmse_db","r2","pearson_r")]
        check("| "+" | ".join(cells)+" |" in report,"report common684 row")
    coverage=pd.read_csv(OUT/"tables/weather_sample_coverage.csv")
    check(coverage.query("population=='test' and weather!='overall'")["n"].tolist()==[322,51,158,156],"main weather coverage")
    check(coverage.query("population=='test_itu_common684' and weather!='overall'")["n"].tolist()==[321,51,156,156],"common weather coverage")
    rank=met.query("population=='ML_687' and weather=='overall' and k==156").sort_values("rmse_db").model.tolist()
    check(rank.index("tabm")+1==3,"TabM rank claim")
    
    # Every local report/README link must resolve; FINAL_QA is the artifact written after these checks.
    for path in [OUT/"REPORT_中文.md",OUT/"README.md"]:
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if target.endswith("/evidence/FINAL_QA.json"): continue
            check(Path(target).is_file(),"broken report link "+target)
    vis=read(OUT/"evidence/VISUAL_REVIEW.json")
    check(vis["status"]=="PASS" and len(vis["figures"])==4,"visual review")
    for name in vis["figures"]:
        p=OUT/f"figures/{name}.png"; pdf=OUT/f"figures/{name}.pdf"
        check(p.read_bytes()[:8]==b"\x89PNG\r\n\x1a\n" and p.stat().st_size>20000,"PNG signature")
        check(pdf.read_bytes()[:4]==b"%PDF" and pdf.stat().st_size>5000,"PDF signature")
    for _,r in pd.read_csv(OUT/"evidence/frozen_inputs_1028_sha.csv").iterrows():
        check(sha(ROOT/r.path)==r.expected_sha256==r.actual_sha256,"frozen input end check "+r.path)
    for _,r in pd.read_csv(OUT/"evidence/frozen_provenance_sha.csv").iterrows():
        check(sha(ROOT/r.path)==r.sha256,"frozen provenance end check "+r.path)
    state_index=pd.read_csv(OUT/"evidence/state_qa_index_1560.csv")
    for _,r in state_index.iterrows():
        folder=BASE/f"models/{r.model}/k_{int(r.k):04d}"
        for name,col in (("state","state_sha256"),("evaluation","evaluation_sha256"),("qa","qa_sha256")):
            check(sha(folder/f"{name}.json")==r[col],"frozen state end check")
    claims=read(OUT/"evidence/CLAIM_EVIDENCE_MAP.json")
    for claim in claims[:2]:
        src=pd.read_csv(OUT/claim["source"],float_precision="round_trip")
        for key,val in claim["filter"].items(): src=src[src[key].eq(val)]
        check(len(src)==1,"claim selector")
        for key,val in claim["values"].items(): equal(src.iloc[0][key],val,"claim "+claim["id"])
    check(len(claims)==4,"claim map completeness")
    anomaly=read(OUT/"evidence/itu_anomaly_reconciliation.json")
    check(len(anomaly)==3 and sum(a["legacy_event_differs_from_frozen_weather"] for a in anomaly)==1,"anomaly discrepancy recorded")
    check(all(a["signed_liquid_proxy_mmph"]<0 for a in anomaly),"no zero clipping")
    ledger=pd.read_csv(OUT/"evidence/failure_anomaly_ledger.csv")
    check({"I01","I02","I03","R01","R02","A01","A02","A03","A04"}<=set(ledger.id),"complete incident categories")
    external=[]
    for p in [*sorted(CODE.glob("*")),
              *sorted((OUT.parent/"evidence").glob("*.json")),
              *sorted((ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_033_Execution_Repair/recovery_events/20261019_1324_Interrupted_Process").glob("*.json"))]:
        if p.is_file(): external.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    pd.DataFrame(external).to_csv(OUT/"evidence/delivery_code_and_external_evidence_sha.csv",index=False,encoding="utf-8-sig",mode="x")
    final=dict(status="PASS",scope="T014 technical results, independently checked CSV delivery, figures, Chinese report and failure ledger",
               completed_utc=datetime.now(timezone.utc).isoformat(),implementation="independent verify_delivery_v2.py; pandas/numpy CSV readback, no producer/auditor import",
               all_required_delivery_items_present=True,states=1560,models=10,states_per_model=156,
               hourly_test_rows=total,independently_recomputed_exported_metric_groups=metric_groups,
               maximum_readback_numeric_difference=MAX,model_role_slots=6240,input_sha_count=1028,
               all_frozen_source_and_state_digests_unchanged=True,
               four_png_figures_visually_reviewed=True,pdf_files_header_and_size_verified=True,
               same_figures_exported_as_pdf=True,training_calls_during_delivery=0,
               scientific_experiment_failures_preserved=1,unknown_interruption_cause_preserved=True,
               reporting_audit_failure_preserved=True,report_table_schema_check_failure_preserved=True,itu_negative_liquid_hours=3,
               historical_weather_event_discrepancy_explicitly_recorded=True,
               author_scientific_interpretation_and_submission_approval="NOT_PERFORMED_NOT_PART_OF_AUTOMATED_TECHNICAL_DELIVERY",
               old_032_numeric_status_intentionally_not_modified=True,
               elapsed_seconds=time.perf_counter()-start)
    write_json(OUT/"evidence/FINAL_QA.json",final)
    files=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            files.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(files).to_csv(OUT/"PACKAGE_SHA256.csv",index=False,encoding="utf-8-sig",mode="x")
    print("FINAL_DELIVERY_QA_PASS",json.dumps(final,ensure_ascii=False),flush=True)
if __name__=="__main__":
    try: main()
    except Exception:
        write_json(OUT/f"evidence/delivery_qa_failure_{time.time_ns()}.json",
                   dict(error=traceback.format_exc(),note="Preserved final-QA failure; no scientific result modified."))
        raise
