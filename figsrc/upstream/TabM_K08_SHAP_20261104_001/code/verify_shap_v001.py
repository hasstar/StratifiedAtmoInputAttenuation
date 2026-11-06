"""Independent readback, scalar identities and actual-model path replay."""
from pathlib import Path
import sys,json,hashlib,math,importlib.util
import numpy as np
import pandas as pd
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.dont_write_bytecode=True
assert not (ROOT/"qa/independent_qa.json").exists(),"Preserve earlier QA."
checks=[]
def check(name,ok,detail=None):
    checks.append({"name":name,"pass":bool(ok),"detail":detail})
def read(p): return json.loads(p.read_text("utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=read(ROOT/"manifest.json")
for p,digest in manifest["source_sha256"].items(): check("unchanged:"+str(Path(p).name),sha(p)==digest)
check("scientific_completion",read(ROOT/"progress.json")["status"]=="NUMERICS_COMPLETE_VISUAL_REVIEW_PENDING")
hours=pd.read_csv(ROOT/"results/test_hours.csv")
check("687_unique_test_hours",len(hours)==687 and hours.hour_bjt.is_unique)
spec=importlib.util.spec_from_file_location("t038_qa_runtime",PROJECT/"Code/Hourly_Attenuation_Experiments/Revision_032/runtime.py")
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r);r.torch.set_num_threads(2)
check("analytic_method_validation",read(ROOT/"qa/method_qa.json")["pass"])
rankings=[]
for model in ("full","base"):
    ident=read(ROOT/"results"/f"{model}_identity.json")
    a=np.load(ROOT/"results"/f"{model}_shap.npz")
    raw=pd.read_csv(ROOT/"results"/f"{model}_test_inputs.csv")
    bgraw=pd.read_csv(ROOT/"results"/f"{model}_background_inputs.csv")
    proc=np.load(ROOT/"results"/f"{model}_processed_inputs.npz")
    names=ident["features"];prep=ident["preprocess"]
    check(model+":k8",len(names)==8)
    check(model+":Monte_Carlo_precision",read(ROOT/"qa"/f"{model}_numerical.json")["precision_screen_pass"])
    check(model+":training_background",len(bgraw)==64 and set(bgraw.hour_bjt).isdisjoint(set(hours.hour_bjt)))
    values=raw[names].to_numpy(float);values[~np.isfinite(values)]=np.nan
    filled=np.where(np.isnan(values),np.asarray(prep["median"]),values)
    converted=((filled-np.asarray(prep["mean"]))/np.asarray(prep["scale"])).astype("float32")
    check(model+":frozen_transform",np.allclose(converted,proc["x"],atol=5e-6,rtol=0))
    check(model+":baseline_mean",abs(math.fsum(proc["fz"])/64-float(a["baseline_db"]))<1e-12)
    delta=[abs(math.fsum(row)+float(a["baseline_db"])-pred) for row,pred in zip(a["phi"],a["prediction_db"])]
    check(model+":scalar_additivity",max(delta)<1e-10,max(delta))
    reps=[]
    for rep in range(2):
        files=sorted((ROOT/"results"/model/f"rep_{rep}").glob("reference_*.npz"))
        check(model+f":all64_references_rep{rep}",len(files)==64)
        arr=np.stack([np.load(p)["phi"] for p in files])
        summed=np.array([[math.fsum(arr[:,h,j])/64 for j in range(len(names))] for h in range(687)])
        check(model+f":rep{rep}_chunk_readback",np.max(abs(summed-a["replicates"][rep]))<1e-12)
        reps.append(summed)
    avg=(reps[0]+reps[1])/2
    check(model+":final_replicate_mean",np.max(abs(avg-a["phi"]))<1e-12)
    table=pd.read_csv(ROOT/"results"/f"{model}_shap_values.csv")
    check(model+":csv_roundtrip",list(table.hour_bjt)==list(hours.hour_bjt) and np.allclose(table[names],avg,atol=1e-14,rtol=0))
    summary=pd.read_csv(ROOT/"results"/f"{model}_shap_summary.csv")
    for group in ("overall","rain","dry_overcast","clear_dry","other"):
        mask=np.ones(687,dtype=bool) if group=="overall" else hours[group].to_numpy(bool)
        for j,feature in enumerate(names):
            row=summary[(summary.weather==group)&(summary.feature==feature)].iloc[0]
            expected=math.fsum(abs(float(x)) for x in avg[mask,j])/sum(mask)
            check(model+":"+group+":"+feature,abs(expected-row.mean_abs_shap_db)<1e-14 and row.n==int(sum(mask)))
    # Independent loop over scalar hybrid rows on actual saved models.
    bundle=r.load_fit({"model_file":ident["model"],"model_sha256":ident["model_sha256"],"format":"joblib"})
    replay=r.predict_x(bundle,proc["x"])
    check(model+":original_predictor_replay",np.max(abs(replay-a["prediction_db"]))<1e-6)
    chunk=np.load(ROOT/"results"/model/"rep_0/reference_000.npz")
    for h in (0,343,686):
        order=chunk["orders"][h].astype(int).tolist()
        result=np.zeros(len(names));start=proc["bg"][0]
        for sequence in (order,list(reversed(order))):
            masks=[start.copy()]
            for j in sequence:
                nxt=masks[-1].copy();nxt[j]=proc["x"][h,j];masks.append(nxt)
            outputs=r.predict_x(bundle,np.stack(masks))
            for step,j in enumerate(sequence): result[j]+=(outputs[step+1]-outputs[step])/2
        check(model+f":actual_model_scalar_path_{h}",np.max(abs(result-chunk["phi"][h]))<2e-6,float(np.max(abs(result-chunk["phi"][h]))))
    ranks=summary[summary.weather=="overall"].sort_values("mean_abs_shap_db",ascending=False)
    rankings.append({"model":model,"ranked_features":ranks.feature.tolist(),"mean_abs_shap_db":ranks.mean_abs_shap_db.tolist(),
                     "pressure_level_attribution_share":float(ranks.loc[ranks.feature.str.startswith("p"),"mean_abs_shap_db"].sum()/ranks.mean_abs_shap_db.sum())})
report={"utc":datetime.now(timezone.utc).isoformat(),"passed":sum(x["pass"] for x in checks),"total":len(checks),
        "all_pass":all(x["pass"] for x in checks),"checks":checks,"ranking_summary":rankings,
        "scope":"Numerical consistency of post-hoc model attributions; not validation of a causal atmospheric mechanism."}
(ROOT/"qa/independent_qa.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:report[k] for k in ("passed","total","all_pass","ranking_summary")},ensure_ascii=False,indent=2))
if not report["all_pass"]: print([x for x in checks if not x["pass"]]);raise SystemExit(1)
