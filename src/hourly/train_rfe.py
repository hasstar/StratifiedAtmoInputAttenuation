"""Independent train-only RFE; each state saves models, folds and all metrics.

Uses the unchanged scientific implementations delivered under reference_project.
No original CLI is launched and no reference output/stop marker is modified.
The default full run refits models; --mode frozen is a separate verification path.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s

def fit(r,runner,name,names,train,hold,out,fold):
    out.mkdir(parents=True,exist_ok=True)
    ident=r.fp(dict(model=name,names=names,fold=fold,seed=17,
                    train=train.hour_bjt.astype(str).tolist(),config=r.config(name)))
    if (out/'fit_identity.json').exists():
        assert s.read(out/'fit_identity.json')['identity']==ident,'Changed fitting inputs/configuration'
    else:s.write(out/'fit_identity.json',dict(identity=ident))
    if (out/'result.json').exists():
        ref=s.read(out/'result.json');bundle=r.load_fit(ref)
        r.check_prep(bundle,train,names,name)
        return ref,bundle
    start=time.perf_counter()
    if name in r.NN:
        bundle,history,meta=runner.train_neural(name,names,train,hold,out,ident)
    else:
        x,bundle=r.prep_fit(name,train,names);cfg=r.config(name)
        model,history,_,_,meta=r.engine.fit(name,cfg['choice'],17,r.r29.active(names),x,train,fixed_rounds=cfg['formal_rounds'])
        bundle['model']=model
    if not meta.get('solver_converged',True):raise RuntimeError('Solver did not converge; inspect this run')
    model_file=out/'model.joblib';r.write_bundle(model_file,bundle)
    ref=dict(model_file=s.rel(model_file),model_sha256=s.sha(model_file),format='joblib',
             model=name,k=len(names),fold=fold,seed=17,feature_names=names,
             config=r.config(name),round_history=history,fit_metadata=meta,
             elapsed_seconds=time.perf_counter()-start)
    b=r.load_fit(ref);r.check_prep(b,train,names,name)
    ref['training']=r.record(train,r.predict(b,train))
    ref['holdout']=None if hold is None else r.record(hold,r.predict(b,hold))
    s.write(out/'result.json',ref)
    return ref,b

def execute(out,models,start_k=None,stop_k=1,pools=('full156','base50'),full_fit_only=False,resume=False):
    r=s.hourly();data=r.frames();r.OUT=out
    r.progress=lambda **kw: print(json.dumps(dict(utc=s.now(),**kw)),flush=True)
    r.stop_check=lambda: (_ for _ in ()).throw(InterruptedError('Current run STOP_REQUESTED')) if (out/'STOP_REQUESTED').exists() else None
    sys.modules['runtime']=r
    runner=s.load_module('supp_original_training',s.original('Code/Hourly_Attenuation_Experiments/Revision_032/run.py'))
    folds=r.r29.folds(data['train']);full_names=r.key_names()
    protocol=dict(primary_seed=17,models=models,pools=pools,
            start_k=start_k,stop_k=stop_k,full_fit_only=full_fit_only,
            training_rows=2404,validation_rows=343,test_rows=687,
            folds=[{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in f.items()} for f in folds],
            configuration=s.config()['hourly'],reference_results_overwritten=False)
    protocol=json.loads(json.dumps(protocol))
    if resume:
        assert s.read(out/'protocol.json')==protocol,'Resume protocol/configuration mismatch'
    else:
        s.write(out/'environment.json',s.environment());s.write(out/'protocol.json',protocol)
    for pool in pools:
        for name in models:
            if pool=='base50' and name!='tabm':continue
            names=[n for n in full_names if pool=='full156' or n.startswith('surface__')]
            if start_k is not None:
                exp='Hourly_Attenuation_Revision_032_Multi_Model_RFE' if pool=='full156' else 'Hourly_Attenuation_Revision_041_TabM_Surface_RFE'
                reference=s.original(f'Outputs/Experiments/{exp}/models/{name}/k_{start_k:04d}/state.json')
                names=s.read(reference)['feature_names']
            for k in range(len(names),stop_k-1,-1):
                r.stop_check();folder=out/pool/name/f'k_{k:04d}';folder.mkdir(parents=True,exist_ok=resume)
                if (folder/'state.json').exists():
                    state=s.read(folder/'state.json');assert state['feature_names']==names
                    if not (folder/'evaluation.json').exists():
                        b=r.load_fit(state['full_fit'])
                        evaluation={split:r.record(data[split],r.predict(b,data[split]))for split in ('validation','test')}
                        s.write(folder/'evaluation.json',dict(results=evaluation,state_sha256=s.sha(folder/'state.json')))
                    assert s.read(folder/'evaluation.json')['state_sha256']==s.sha(folder/'state.json')
                    if full_fit_only:break
                    names=[n for n in names if n!=state['removed_feature_next']]
                    print('RESUMED_COMPLETE_STATE',pool,name,k,flush=True);continue
                roles=[];deltas=[];permutations=[]
                if not full_fit_only:
                    for f in folds:
                        tr=data['train'].iloc[f['fit']];hold=data['train'].iloc[f['hold']]
                        ref,b=fit(r,runner,name,names,tr,hold,folder/f'fold{f["fold"]}',f['fold']);roles.append(ref)
                        if k>1:
                            x=r.transformed(b,hold);values=[]
                            for feature in names:
                                changed=x.copy();col=r.column_index(b,feature)
                                ix=r.np.random.default_rng(r.permutation_seed(k,f['fold'],feature)).permutation(len(x))
                                changed[:,col]=x[ix,col];values.append(r.predict_x(b,changed))
                            matrix=r.np.stack(values);p=folder/f'permutation_fold{f["fold"]}.npz'
                            r.np.savez_compressed(p,prediction_db=matrix)
                            permutations.append(dict(path=s.rel(p),sha256=s.sha(p)))
                            deltas.append(r.np.sqrt(r.np.mean((matrix-hold[r.TARGET].to_numpy()[None,:])**2,axis=1))-ref['holdout']['metrics']['overall']['rmse_db'])
                full,b=fit(r,runner,name,names,data['train'],None,folder/'full',0)
                importance=[];remove=None
                if deltas:
                    values=r.np.stack(deltas)
                    importance=[dict(feature_name=n,fold_delta_rmse_db=values[:,i].tolist(),mean_delta_rmse_db=float(values[:,i].mean()))for i,n in enumerate(names)]
                    remove=min(importance,key=lambda a:(a['mean_delta_rmse_db'],a['feature_name']))['feature_name']
                state=dict(model=name,pool=pool,k=k,feature_names=names,removed_feature_next=remove,
                           full_fit=full,folds=roles,permutations=permutations,permutation_importance=importance,
                           seed=17,official_test_used_for_deletion=False,
                           starting_subset_from_reference=start_k is not None)
                # The removal decision is recorded before evaluation on validation/test.
                s.write(folder/'state.json',state)
                evaluation={split:r.record(data[split],r.predict(b,data[split]))for split in ('validation','test')}
                s.write(folder/'evaluation.json',dict(results=evaluation,state_sha256=s.sha(folder/'state.json')))
                print(json.dumps(dict(stage='STATE_COMPLETE',pool=pool,model=name,k=k,
                       test={key:value for key,value in evaluation['test']['metrics']['overall'].items()
                             if key in ('n','rmse_db','mae_db','r2','bias_db')},removed_next=remove)),flush=True)
                if full_fit_only:break
                if k>1:names=[n for n in names if n!=remove]
                gc.collect()
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--models',nargs='+')
    ap.add_argument('--pools',nargs='+',choices=['full156','base50'],default=['full156','base50'])
    ap.add_argument('--start-k',type=int)
    ap.add_argument('--stop-k',type=int,default=1)
    ap.add_argument('--full-fit-only',action='store_true',help='Bounded fitting verification, not full RFE')
    ap.add_argument('--resume',action='store_true',help='Resume this new run only, after the previous process has exited; validate identities before using checkpoints')
    a=ap.parse_args();out=a.output or s.path('new_runs')/s.timestamp()/'hourly'
    out.mkdir(parents=True,exist_ok=a.resume)
    with s.run_lock('hourly training','hourly.lock'):
        execute(out,a.models or list(s.config()['hourly']['models']),a.start_k,a.stop_k,a.pools,a.full_fit_only,a.resume)
if __name__=='__main__':main()
