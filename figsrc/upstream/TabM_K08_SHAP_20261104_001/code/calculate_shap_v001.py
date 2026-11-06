"""Marginal permutation SHAP for two frozen TabM k8 fits.
No fit/optimization calls. Full paired order paths preserve local additivity.
Executed source is frozen; use a new numbered script for any repair.
"""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,time,traceback,msvcrt
from datetime import datetime,timezone
sys.dont_write_bytecode=True
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
HERE=Path(__file__).resolve()
OUT=HERE.parents[1]
ROOT=OUT.parents[1]
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text("utf-8-sig"))
def emit(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(v):
    p=OUT/"progress.json";q=OUT/"progress.tmp"
    q.write_text(json.dumps(dict(utc=now(),**v),indent=2),encoding="utf-8")
    os.replace(q,p)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr

def paired_path(predict,x,z,perm,fx,fz):
    """One permutation and its reverse for one reference row."""
    n,d=x.shape;blocks=[];orders=[perm,perm[::-1]]
    for order in orders:
        h=np.broadcast_to(z,(n,d)).copy()
        for j in order[:-1]:
            h[:,j]=x[:,j];blocks.append(h.copy())
    middle=predict(np.concatenate(blocks,axis=0)).reshape(2,d-1,n)
    phi=np.zeros((n,d),dtype=float)
    for side,order in enumerate(orders):
        vals=np.vstack([np.full(n,fz),middle[side],fx])
        for step,j in enumerate(order): phi[:,j]+=.5*(vals[step+1]-vals[step])
    assert np.max(np.abs(phi.sum(axis=1)-(fx-fz)))<1e-7
    return phi

def algebra_checks():
    import itertools, math
    rng=np.random.default_rng(60);x=rng.normal(size=(7,4));z=rng.normal(size=4)
    w=np.array([1.,-2.,.3,2.7]);f=lambda a: 1.+a@w+.6*a[:,0]*a[:,1]
    exact=w*(x-z);pair=.3*(x[:,0]-z[0])*(x[:,1]+z[1]);pair1=.3*(x[:,1]-z[1])*(x[:,0]+z[0])
    exact[:,0]+=pair;exact[:,1]+=pair1
    maxerr=0.
    for perm in itertools.permutations(range(4)):
        v=paired_path(f,x,z,np.array(perm),f(x),float(f(z[None])[0]))
        maxerr=max(maxerr,float(np.max(abs(v-exact))))
    assert maxerr<1e-12
    g=lambda a: np.sin(a[:,0]+a[:,1]*a[:,2])+a[:,3]**2
    enumerated=np.zeros_like(x)
    for j in range(4):
        others=[k for k in range(4) if k!=j]
        for size in range(4):
            wt=math.factorial(size)*math.factorial(3-size)/math.factorial(4)
            for subset in itertools.combinations(others,size):
                h=np.broadcast_to(z,x.shape).copy()
                if subset: h[:,list(subset)]=x[:,list(subset)]
                before=g(h);h[:,j]=x[:,j]
                enumerated[:,j]+=wt*(g(h)-before)
    sampled=np.mean([paired_path(g,x,z,np.array(q),g(x),float(g(z[None])[0]))
                     for q in itertools.permutations(range(4))],axis=0)
    err=float(np.max(abs(enumerated-sampled)));assert err<1e-12
    return {"additive_pairwise_all_orders_max_error":maxerr,"nonlinear_exhaustive_game_max_error":err,"passed":True}

def paths(predict,x,z,orders,fx,fz):
    n,d=x.shape;row=np.arange(n);blocks=[]
    for order in (orders,orders[:,::-1]):
        h=np.broadcast_to(z,x.shape).copy()
        for s in range(d-1):
            j=order[:,s];h[row,j]=x[row,j];blocks.append(h.copy())
    middle=predict(np.concatenate(blocks)).reshape(2,d-1,n)
    phi=np.zeros_like(x,dtype=float)
    for side,order in enumerate((orders,orders[:,::-1])):
        y=np.vstack([np.full(n,fz),middle[side],fx])
        for s in range(d): phi[row,order[:,s]]+=.5*(y[s+1]-y[s])
    assert np.max(abs(phi.sum(1)-(fx-fz)))<1e-7
    return phi
def method_checks():
    import itertools
    rng=np.random.default_rng(732);x=rng.normal(size=(11,4));z=rng.normal(size=4);w=np.array([.5,-2.,1.,3.])
    f=lambda a: a@w+2*a[:,0]*a[:,3]
    exact=(x-z)*w;exact[:,0]+=(x[:,0]-z[0])*(x[:,3]+z[3]);exact[:,3]+=(x[:,3]-z[3])*(x[:,0]+z[0])
    orders=np.argsort(rng.random(x.shape),axis=1)
    actual=paths(f,x,z,orders,f(x),float(f(z[None])[0]))
    err=float(np.max(abs(actual-exact)));assert err<1e-12
    # Check per-row path kernel against the independently coded scalar/global-order implementation.
    g=lambda a: np.sin(a[:,0]+a[:,1]*a[:,2])+a[:,3]**3
    a=paths(g,x,z,orders,g(x),float(g(z[None])[0]))
    b=np.vstack([paired_path(g,x[i:i+1],z,orders[i],g(x[i:i+1]),float(g(z[None])[0])) for i in range(len(x))])
    diff=float(np.max(abs(a-b)));assert diff<1e-12
    return {"analytic_pairwise_error":err,"independent_path_implementation_error":diff,"v001_exhaustive_check":algebra_checks(),"pass":True}
def main():
    assert not (OUT/"manifest.json").exists(),"Do not overwrite executed analysis."
    (OUT/"results").mkdir(exist_ok=True)
    (OUT/"qa").mkdir(exist_ok=True)
    emit(OUT/"qa/algebra_checks.json",algebra_checks())
    emit(OUT/"qa/method_qa.json",method_checks())
    runtime=ROOT/"Code/Hourly_Attenuation_Experiments/Revision_032/runtime.py"
    r=module("t038_frozen_runtime",runtime)
    r.torch.set_num_threads(2);r.torch.set_num_interop_threads(2)
    r.torch.use_deterministic_algorithms(True)
    frames=r.frames();train=frames["train"];test=frames["test"]
    assert len(train)==2404 and len(test)==687
    assert set(train.hour_bjt).isdisjoint(set(test.hour_bjt))
    bg_idx=np.sort(np.random.default_rng(20261103).choice(len(train),64,replace=False))
    models={};source_files={str(HERE):sha(HERE),str(OUT/"PROTOCOL.md"):sha(OUT/"PROTOCOL.md")}
    for name,folder in [("full","Hourly_Attenuation_Revision_032_Multi_Model_RFE"),("base","Hourly_Attenuation_Revision_041_TabM_Surface_RFE")]:
        d=ROOT/"Outputs/Experiments"/folder
        state=d/"models/tabm/k_0008/state.json";ev=d/"models/tabm/k_0008/evaluation.json";qp=d/"models/tabm/k_0008/qa.json"
        s=read(state);e=read(ev);q=read(qp)
        assert q["status"]=="PASS" and q["state_sha256"]==sha(state) and q["evaluation_sha256"]==sha(ev)
        b=r.load_fit(s["full_fit"]);b["net"].eval()
        for param in b["net"].parameters(): param.requires_grad_(False)
        x=r.transformed(b,test);bg=r.transformed(b,train.iloc[bg_idx])
        fx=r.predict_x(b,x);fz=r.predict_x(b,bg)
        rec=e["results"]["test"]
        assert list(test.hour_bjt.astype(str))==rec["hour_bjt"]
        replay=float(np.max(abs(fx-np.asarray(rec["prediction_db"]))))
        assert replay<1e-6,(name,replay)
        assert s["feature_names"]==b["retained_names"] and len(b["retained_names"])==8
        for p in (state,ev,qp,ROOT/s["full_fit"]["model_file"],d/"manifest.json"):
            source_files[str(p)]=sha(p)
        models[name]=(b,x,bg,fx,fz)
        emit(OUT/"results"/(name+"_identity.json"),{
            "features":b["retained_names"],"model":s["full_fit"]["model_file"],"model_sha256":s["full_fit"]["model_sha256"],
            "preprocess":b["preprocess"],"background_mean_db":float(fz.mean()),"replay_max_delta_db":replay,
            "test_metrics":{k:v for k,v in rec["metrics"]["overall"].items() if k not in ("dates",)},
            "training_calls":0})
        cols=["hour_bjt",r.TARGET,"clear_dry","rain","dry_overcast","other",*b["retained_names"]]
        test[cols].to_csv(OUT/"results"/(name+"_test_inputs.csv"),index=False)
        train.iloc[bg_idx][["hour_bjt",*b["retained_names"]]].to_csv(OUT/"results"/(name+"_background_inputs.csv"),index=False)
        np.savez_compressed(OUT/"results"/(name+"_processed_inputs.npz"),x=x,bg=bg,fx=fx,fz=fz,background_indices=bg_idx)
    # Pin imported project sources and actual input files, without calling any old queue.
    for m in list(sys.modules.values()):
        file=getattr(m,"__file__",None)
        if file:
            p=Path(file)
            if p.suffix==".py" and p.is_relative_to(ROOT) and p.exists(): source_files[str(p)]=sha(p)
    for p in [r.P29/"run_manifest.json",r.c.FEATURE_SOURCE,*r.DATA.glob("*.parquet")]:
        source_files[str(p)]=sha(p)
    manifest={"created_utc":now(),"source_sha256":source_files,"background_seed":20261103,
              "background_indices":bg_idx.tolist(),"permutation_seeds":[20261105,20261106],
              "n_test":687,"n_background":64,"n_permutation_replicates":2,
              "paths_per_test_hour":256,"output_units":"dB",
              "versions":{name:__import__(name).__version__ for name in ("numpy","pandas","torch","scipy")},
              "method":"marginal Shapley Monte Carlo, independent per-hour per-reference permutation and reverse"}
    emit(OUT/"manifest.json",manifest)
    test[["hour_bjt",r.TARGET,"clear_dry","rain","dry_overcast","other"]].to_csv(OUT/"results/test_hours.csv",index=False)
    for name,(b,x,bg,fx,fz) in models.items():
        reps=[]
        for rep,seed in enumerate(manifest["permutation_seeds"]):
            rg=np.random.default_rng(seed)
            # Same permutations across both models; columns retain their own identity/order.
            orders=np.argsort(rg.random((64,len(test),8)),axis=2)
            total=np.zeros((687,8));out=OUT/"results"/name/f"rep_{rep}";out.mkdir(parents=True)
            for bi,order in enumerate(orders):
                if (OUT/"STOP_REQUESTED").exists(): raise InterruptedError("T038 STOP_REQUESTED")
                assert psutil.virtual_memory().available>1024**3
                tick=time.monotonic()
                phi=paths(lambda a:r.predict_x(b,a),x,bg[bi],order,fx,float(fz[bi]))
                total+=phi
                np.savez_compressed(out/f"reference_{bi:03d}.npz",phi=phi,orders=order.astype(np.uint8),background_index=int(bg_idx[bi]))
                update({"status":"RUNNING","model":name,"replicate":rep+1,"references_complete":bi+1,"references_total":64,
                        "last_reference_seconds":time.monotonic()-tick,"rss_bytes":psutil.Process().memory_info().rss})
                if (bi+1)%8==0: print(name,rep+1,bi+1,"/64",flush=True)
            value=total/64
            assert np.max(abs(value.sum(axis=1)+fz.mean()-fx))<1e-7
            reps.append(value)
            np.savez_compressed(OUT/"results"/f"{name}_rep_{rep}.npz",phi=value,orders=orders)
        reps=np.stack(reps);phi=reps.mean(axis=0)
        np.savez_compressed(OUT/"results"/f"{name}_shap.npz",phi=phi,replicates=reps,baseline_db=float(fz.mean()),prediction_db=fx)
        frame=pd.DataFrame(phi,columns=b["retained_names"]);frame.insert(0,"hour_bjt",test.hour_bjt.astype(str).to_numpy())
        frame.to_csv(OUT/"results"/f"{name}_shap_values.csv",index=False)
        a=np.mean(abs(reps[0]),axis=0);c=np.mean(abs(reps[1]),axis=0);m=np.mean(abs(phi),axis=0)
        diag={"replicate_global_rank_spearman":float(spearmanr(a,c).statistic),
              "global_absolute_attribution_relative_L1_difference":float(abs(a-c).sum()/((a+c)/2).sum()),
              "replicate_top3":[np.argsort(-v)[:3].tolist() for v in (a,c)],
              "additivity_max_error_db":float(np.max(abs(phi.sum(axis=1)+fz.mean()-fx))),
              "interpretation":"Monte Carlo agreement, not a statistical confidence interval."}
        diag["precision_screen_pass"] = bool(diag["global_absolute_attribution_relative_L1_difference"] <= .05 and diag["replicate_global_rank_spearman"] >= .95 and set(diag["replicate_top3"][0]) == set(diag["replicate_top3"][1]))
        emit(OUT/"qa"/f"{name}_numerical.json",diag)
        rows=[]
        for group in ("overall","rain","dry_overcast","clear_dry","other"):
            mask=np.ones(687,dtype=bool) if group=="overall" else test[group].to_numpy(bool)
            for j,feature in enumerate(b["retained_names"]):
                rows.append({"model":name,"weather":group,"n":int(mask.sum()),"feature":feature,
                             "mean_abs_shap_db":float(np.mean(abs(phi[mask,j]))),"mean_shap_db":float(np.mean(phi[mask,j])),
                             "mean_abs_rep0_db":float(np.mean(abs(reps[0,mask,j]))),"mean_abs_rep1_db":float(np.mean(abs(reps[1,mask,j])))})
        pd.DataFrame(rows).to_csv(OUT/"results"/f"{name}_shap_summary.csv",index=False)
    unchanged={p:sha(p)==digest for p,digest in source_files.items()}
    assert all(unchanged.values())
    emit(OUT/"qa/source_unchanged.json",{"checked_at_utc":now(),"all_unchanged":True,"checks":unchanged})
    update({"status":"NUMERICS_COMPLETE_VISUAL_REVIEW_PENDING","training_calls":0,"models":2})
    print("NUMERICS_COMPLETE",flush=True)

if __name__=="__main__":
    lock=(OUT/"writer.lock").open("a+b")
    lock.seek(0)
    if lock.read(1)==b"": lock.write(b"0");lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    emit(OUT/"launch.json",{"utc":now(),"pid":os.getpid(),"create_time":psutil.Process().create_time(),"command":sys.argv,"threads":2,"cpu_only":True})
    try: main()
    except BaseException as exc:
        emit(OUT/("failure_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+".json"),
             {"utc":now(),"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        update({"status":"FAILED_REQUIRES_INSPECTION","error_type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
