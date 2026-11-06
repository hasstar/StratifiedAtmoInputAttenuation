"""Single local worker for independent multi-family RFE. No network operations."""
from __future__ import annotations
import argparse, copy, gc, math, os, sys, time, traceback
import runtime as r
from runtime import np, torch

def reuse_ref(name,k,fold,names,train,hold):
    old=None; origin=None; fmt='joblib'; expected_train=None; expected_hold=None
    if name=='bilstm':
        origin=r.P29/'rfe/states'/f'k_{k:04d}.json'; old=r.read(origin)
        r.require(old['feature_names']==names and old['seed']==17 and old['epochs']==53,'BiLSTM old path mismatch')
        role=old['full_fit'] if fold==0 else old['folds'][fold-1]
        path=r.r29.model_path('rfe',k,fold); expected_train=role['training']; expected_hold=role['holdout']
        history=role['round_history']; metadata=role['fit_metadata']; digest=role['model_sha256']
    elif name=='tabm' and k==156:
        state=r.read(r.P31/'layer/states/1023.json'); role=state['full_fit'] if fold==0 else state['folds'][fold-1]
        origin=r.P31/role['result_file']; old=r.read(origin)
        r.require(r.sha(origin)==role['result_sha256'] and old['feature_names']==names and old['selected_epoch']==54,'TabM source mismatch')
        path=r.P31/old['model_file']; expected_train=old['training']; expected_hold=old['holdout'] if fold else None
        history=old['round_history']; metadata=dict(selected_epoch=54,trained_epochs=old['trained_epochs']); digest=old['model_sha256']; fmt='tabm31'
    elif name in r.legacy.MODEL_NAMES and k==156 and fold==0:
        origin=r.P28/'training'/f'full_{name}_seed17.json'; old=r.read(origin)
        r.require(old['feature_names']==names and old['plan']==r.config(name) and old['seed']==17,'Full156 prior config mismatch')
        path=r.P28/'models'/f'full_{name}_seed17.joblib'; digest=old['model_sha256']
        expected_train=dict(hour_bjt=old['train_hour_bjt'],prediction_db=old['train_prediction_db'])
        history=old['round_history']; metadata=old['fit_metadata']
    else: return None
    r.require(r.sha(path)==digest,'Old model content changed')
    r.require(expected_train['hour_bjt']==train.hour_bjt.astype(str).tolist(),'Reuse fitting keys')
    if hold is not None: r.require(expected_hold['hour_bjt']==hold.hour_bjt.astype(str).tolist(),'Reuse hold keys')
    ref=dict(model_file=r.rel(path),model_sha256=digest,format=fmt,reused=True,new_fit=False,
             source_file=r.rel(origin),source_sha256=r.sha(origin),round_history=history,fit_metadata=metadata)
    b=r.load_fit(ref); r.check_prep(b,train,names,name)
    pred=r.predict(b,train)
    r.require(np.allclose(pred,expected_train['prediction_db'],rtol=0,atol=2e-6),'Prior fitting prediction replay')
    if hold is not None: r.require(np.allclose(r.predict(b,hold),expected_hold['prediction_db'],rtol=0,atol=2e-6),'Prior fold prediction replay')
    return ref

def train_neural(name,names,train,hold,directory,identity,search=False):
    x,bundle=r.prep_fit(name,train,names); net=r.make_nn(name,names)
    y=train[r.TARGET].to_numpy(float)
    if 'preprocess' not in bundle: bundle['preprocess']=dict(y_mean=float(y.mean()),y_scale=max(float(y.std()),1e-6))
    prep=bundle['preprocess']; bundle['net']=net
    cfg=r.config(name)
    if name=='bai_lstm':
        optim=torch.optim.RMSprop(net.parameters(),lr=1e-4,alpha=.9,eps=1e-6,momentum=0)
        batch=128; limit=1000 if search else cfg['epochs']; policy='uniform'
    else:
        e=cfg['config'] if name=='tabm' else cfg['choice']['estimator']
        optim=torch.optim.AdamW(net.parameters(),lr=e['lr'],weight_decay=e['weight_decay'])
        batch=256 if name=='tabm' else 128; limit=cfg['epochs'] if name=='tabm' else cfg['formal_rounds']
        policy='uniform' if name=='tabm' else cfg['choice']['weight_policy']
    r.require(limit is not None,'Epochs not frozen')
    xx=torch.as_tensor(np.asarray(x,np.float32)); yy=torch.as_tensor((y-prep['y_mean'])/prep['y_scale'],dtype=torch.float32)
    ww=torch.as_tensor(r.engine.weights(train,policy),dtype=torch.float32)
    gen=torch.Generator().manual_seed(17); history=[]; first=1; stale=0; best=math.inf; selected=0; best_weights=None
    recovery=directory/'recovery.pt'; elapsed=0.; start=time.perf_counter()
    if recovery.exists():
        cp=r.c.load_tensor(recovery); r.require(cp['identity']==identity,'Recovery identity mismatch')
        net.load_state_dict(cp['weights']); optim.load_state_dict(cp['optimizer']); gen.set_state(cp['generator']); torch.set_rng_state(cp['rng'])
        history=cp['history']; first=cp['epoch']+1; stale=cp['stale']; best=cp['best']; selected=cp['selected']; best_weights=cp['best_weights']; elapsed=cp['seconds']
    xh=None if hold is None else r.transformed(bundle,hold)
    for epoch in range(first,limit+1):
        r.stop_check(); net.train(); losses=[]
        for ix in torch.randperm(len(xx),generator=gen).split(batch):
            optim.zero_grad(set_to_none=True); out=net(xx[ix])
            err=out-yy[ix,None] if name=='tabm' else out-yy[ix]
            loss=(err.square()*(ww[ix,None] if name=='tabm' else ww[ix])).mean()
            r.require(bool(torch.isfinite(loss)),'Nonfinite neural loss'); loss.backward(); optim.step(); losses.append(float(loss.detach()))
        pred=r.predict_x(bundle,x); hp=None if hold is None else r.predict_x(bundle,xh)
        history.append(dict(epoch=epoch,loss_standardized=float(np.mean(losses)),train=r.scores(train,pred),holdout=None if hold is None else r.scores(hold,hp)))
        if search:
            score=history[-1]['holdout']['overall']['rmse_db']
            if score<best-1e-12: best=score; selected=epoch; best_weights=copy.deepcopy(net.state_dict()); stale=0
            else: stale+=1
        r.progress(stage='BAI_FULL156_EPOCH_SELECTION' if search else 'TRAINING',model=name,k=len(names),job=directory.name,epoch=epoch,epochs=limit)
        if epoch%5==0 or epoch==limit or (search and stale>=5):
            r.c.save_tensor(recovery,dict(identity=identity,epoch=epoch,weights=net.state_dict(),optimizer=optim.state_dict(),generator=gen.get_state(),rng=torch.get_rng_state(),history=history,stale=stale,best=best,selected=selected,best_weights=best_weights,seconds=elapsed+time.perf_counter()-start),intermediate=True)
        if search and stale>=5: break
    if search:
        r.require(best_weights is not None,'No Bai validation checkpoint'); net.load_state_dict(best_weights)
    else: selected=limit
    bundle.pop('net'); bundle['nn_weights']=net.state_dict()
    return bundle,history,dict(selected_epoch=selected,trained_epochs=len(history),fit_seconds=elapsed+time.perf_counter()-start,
                              weight_policy=policy,seed=17,search=search,validation_selection='overall RMSE, earliest min, patience5' if search else None,solver_converged=True)

def fit_role(name,k,fold,names,train,hold,m,search=False):
    directory=r.OUT/'fits'/name/(f'k_{k:04d}_fold{fold}' if not search else 'full156_epoch_selection')
    directory.mkdir(parents=True,exist_ok=True)
    identity=r.fp(dict(manifest=m['fingerprint'],model=name,k=k,fold=fold,names=names,train=train.hour_bjt.astype(str).tolist(),
                       hold=None if hold is None else hold.hour_bjt.astype(str).tolist(),config=r.config(name) if not search else 'bai_fixed_search_policy',search=search))
    path=directory/'result.json'
    if path.exists():
        ref=r.read(path); r.require(ref['identity']==identity,'Job identity mismatch'); r.require(r.sha(r.ROOT/ref['model_file'])==ref['model_sha256'],'Existing fit changed'); return ref
    r.stop_check(); r.progress(stage='FIT_START',model=name,k=k,fold=fold,search=search)
    ref=None if search else reuse_ref(name,k,fold,names,train,hold)
    if ref is None and name=='bai_lstm' and k==156 and fold==0 and not search:
        source=r.read(r.OUT/'fits/bai_lstm/full156_epoch_selection/result.json')
        r.require(source['feature_names']==names and source['fit_metadata']['selected_epoch']==r.config(name)['epochs'],'Bai selected reuse')
        ref={key:source[key] for key in ('model_file','model_sha256','format','round_history','fit_metadata')}
        ref.update(reused=True,new_fit=False,source_file=r.rel(r.OUT/'fits/bai_lstm/full156_epoch_selection/result.json'),source_sha256=r.sha(r.OUT/'fits/bai_lstm/full156_epoch_selection/result.json'))
    if ref is None:
        attempt=time.time_ns(); r.emit(directory/f'attempt_{attempt}.json',dict(identity=identity,utc=r.now(),pid=os.getpid(),search=search))
        if name in r.NN: bundle,history,meta=train_neural(name,names,train,hold,directory,identity,search)
        else:
            x,bundle=r.prep_fit(name,train,names); cfg=r.config(name)
            model,history,_,_,meta=r.engine.fit(name,cfg['choice'],17,r.r29.active(names),x,train,fixed_rounds=cfg['formal_rounds'])
            bundle['model']=model
        r.require(meta.get('solver_converged',True),'Solver convergence failure; preserve attempt, no silent retry')
        modelpath=directory/f'model_{attempt}.joblib'; r.write_bundle(modelpath,bundle)
        ref=dict(model_file=r.rel(modelpath),model_sha256=r.sha(modelpath),format='joblib',reused=False,new_fit=True,round_history=history,fit_metadata=meta)
    b=r.load_fit(ref); r.check_prep(b,train,names,name)
    ref.update(identity=identity,manifest=m['fingerprint'],model=name,k=k,fold=fold,seed=17,feature_names=names,
               train_hour_sha256=r.fp(train.hour_bjt.astype(str).tolist()),training=r.record(train,r.predict(b,train)),
               holdout=None if hold is None else r.record(hold,r.predict(b,hold)),completed_utc=r.now())
    r.emit(path,ref); r.append(r.OUT/'fit_ledger.jsonl',dict(model=name,k=k,fold=fold,reused=ref['reused'],model_file=ref['model_file'],model_sha256=ref['model_sha256'],utc=r.now()))
    return ref

def permutation_matrix(name,k,fold,names,hold,ref,folder):
    path=folder/f'permutation_fold{fold}.npz'; receipt=path.with_suffix('.json')
    ident=r.fp(dict(model_sha=ref['model_sha256'],k=k,fold=fold,names=names,hours=hold.hour_bjt.astype(str).tolist(),namespace=r.PERM_NAMESPACE))
    if receipt.exists():
        meta=r.read(receipt); r.require(meta['identity']==ident and r.sha(path)==meta['sha256'],'Permutation saved digest')
        with np.load(path,allow_pickle=False) as z: return z['prediction_db'].copy(),meta
    b=r.load_fit(ref); x=r.transformed(b,hold); values=[]
    for i,feature in enumerate(names):
        r.stop_check(); changed=x.copy(); col=r.column_index(b,feature)
        changed[:,col]=x[np.random.default_rng(r.permutation_seed(k,fold,feature)).permutation(len(x)),col]
        values.append(r.predict_x(b,changed))
        if i%25==0: r.progress(stage='PERMUTATION',model=name,k=k,fold=fold,feature=i+1,features=len(names))
    matrix=np.stack(values)
    if path.exists():
        with np.load(path,allow_pickle=False) as z: r.require(np.array_equal(z['prediction_db'],matrix),'Orphan permutation differs; retained for inspection')
    else:
        with path.open('xb') as stream: np.savez_compressed(stream,prediction_db=matrix)
    meta=dict(identity=ident,path=r.rel(path),sha256=r.sha(path),shape=list(matrix.shape),namespace=r.PERM_NAMESPACE)
    r.emit(receipt,meta); return matrix,meta

def process_state(name,k,names,data,m):
    import verify
    folder=r.state_dir(name,k); folder.mkdir(parents=True,exist_ok=True); statefile=folder/'state.json'
    if (folder/'qa.json').exists():
        qa=r.read(folder/'qa.json'); r.require(qa['status']=='PASS' and qa['state_sha256']==r.sha(statefile) and qa['evaluation_sha256']==r.sha(folder/'evaluation.json'),'Completed state digest')
        state=r.read(statefile); r.require(state['feature_names']==names,'RFE path mismatch'); return state
    train=data['train']; folds=r.r29.folds(train)
    if not statefile.exists():
        roles=[]; permrefs=[]; deltas=[]
        for f in folds:
            fit=train.iloc[f['fit']]; hold=train.iloc[f['hold']]
            ref=fit_role(name,k,f['fold'],names,fit,hold,m); roles.append(ref)
            if k>1:
                matrix,pm=permutation_matrix(name,k,f['fold'],names,hold,ref,folder); permrefs.append(pm)
                errors=matrix-hold[r.TARGET].to_numpy(float)[None,:]
                deltas.append(np.sqrt(np.mean(errors**2,axis=1))-ref['holdout']['metrics']['overall']['rmse_db'])
        full=fit_role(name,k,0,names,train,None,m)
        importance=[]; remove=None
        if k>1:
            vals=np.stack(deltas)
            importance=[dict(feature_name=n,fold_delta_rmse_db=vals[:,i].tolist(),mean_delta_rmse_db=float(vals[:,i].mean())) for i,n in enumerate(names)]
            remove=min(importance,key=lambda z:(z['mean_delta_rmse_db'],z['feature_name']))['feature_name']
        if name=='bilstm':
            prior=r.read(r.P29/'rfe/states'/f'k_{k:04d}.json')
            r.require(prior['removed_feature_next']==remove,'Independent BiLSTM RFE replay differs from prior path')
        cv=[x['holdout']['metrics']['overall']['rmse_db'] for x in roles]
        state=dict(manifest=m['fingerprint'],model=name,k=k,feature_names=names,seed=17,config=r.config(name),
                   folds=roles,full_fit=full,permutations=permrefs,permutation_importance=importance,removed_feature_next=remove,
                   cv_mean_rmse_db=float(np.mean(cv)),cv_se_rmse_db=float(np.std(cv,ddof=1)/math.sqrt(3)),
                   official_test_used_for_deletion=False,completed_utc=r.now())
        r.emit(statefile,state)
    else:
        state=r.read(statefile); r.require(state['manifest']==m['fingerprint'] and state['feature_names']==names,'Partial state identity')
    # Official holdouts are evaluated only AFTER the training-only deletion is committed.
    evalpath=folder/'evaluation.json'
    if not evalpath.exists():
        b=r.load_fit(state['full_fit']); results={split:r.record(data[split],r.predict(b,data[split])) for split in ('validation','test')}
        mask=r.legal_mask(data['test']); results['test_itu_common684']=r.record(data['test'].loc[mask],np.asarray(results['test']['prediction_db'])[mask])
        r.emit(evalpath,dict(state_sha256=r.sha(statefile),model=name,k=k,results=results,training_calls=0,utc=r.now()))
    verify.state_qa(folder,data,m)
    r.progress(stage='STATE_QA_PASS',model=name,k=k,removed_next=state['removed_feature_next'])
    gc.collect(); return state

def itu(data,m):
    import verify
    p=r.OUT/'itu_baseline.json'
    if p.exists():
        r.require((r.OUT/'itu_qa.json').exists() and r.read(r.OUT/'itu_qa.json')['status']=='PASS','ITU unfinished QA'); return
    source=r.ROOT/'Outputs/Experiments/Combined_Channel_Revision_011/predictions_local.parquet'
    a=r.pd.read_parquet(source).set_index('hour_bjt').loc[data['test'].hour_bjt]
    r.require(np.allclose(a.truth_db,data['test'][r.TARGET],rtol=0,atol=1e-12),'ITU target alignment')
    mask=r.legal_mask(data['test']); pred=a.itu_combined_geometric_approx.to_numpy(float)
    result=r.record(data['test'].loc[mask],pred[mask]); verify.record_qa(result,data['test'].loc[mask])
    r.emit(p,dict(name='combined_ITU_corrected_hourly_geometric_approx',source=r.rel(source),source_sha256=r.sha(source),manifest=m['fingerprint'],results=result,
                  all_hours=687,scored_hours=int(mask.sum()),missing=r.read(r.ROOT/'Outputs/Experiments/Combined_Channel_Revision_027/excluded_hours.json'),
                  fitted_label_offset=False,not_full_itu_probability_model=True,utc=r.now()))
    r.emit(r.OUT/'itu_qa.json',dict(status='PASS',result_sha256=r.sha(p),n=684,independent_metrics=True,utc=r.now()))

def run():
    import msvcrt, psutil
    r.OUT.mkdir(parents=True,exist_ok=True)
    lock=(r.OUT/'writer.lock').open('a+b'); lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('Another T014 worker owns the lock')
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command=psutil.Process().cmdline(),utc=r.now())
    r.update(r.OUT/'active_worker.json',dict(identity,status='RUNNING'))
    try:
        m=r.manifest(); r.require(r.read(r.OUT/'preflight_qa.json')['status']=='PASS','Preflight gate'); data=r.frames(); itu(data,m)
        if not (r.OUT/'bai_epoch_selection.json').exists():
            ref=fit_role('bai_lstm',156,0,m['feature_names'],data['train'],data['validation'],m,search=True)
            r.emit(r.OUT/'bai_epoch_selection.json',dict(selected_epoch=ref['fit_metadata']['selected_epoch'],model_sha256=ref['model_sha256'],source_result=r.rel(r.OUT/'fits/bai_lstm/full156_epoch_selection/result.json'),test_used=False,utc=r.now()))
        import verify
        verify.bai_selection(data)
        names={name:list(m['feature_names']) for name in r.MODELS}
        for k in range(156,0,-1):
            for name in r.MODELS:
                r.stop_check(); state=process_state(name,k,names[name],data,m)
                if k>1: names[name]=[n for n in names[name] if n!=state['removed_feature_next']]
        import report
        report.build(data,m)
        r.update(r.OUT/'status.json',dict(status='NUMERICS_COMPLETE_REPORT_PENDING',utc=r.now(),note='Final human-visible consistency review and delivery still required; do not pause tvt early.'))
    except BaseException as error:
        status='STOPPED_USER_REQUEST' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED_REQUIRES_INSPECTION'
        r.emit(r.OUT/'failures'/f'{time.time_ns()}.json',dict(identity,status=status,error=repr(error),traceback=traceback.format_exc(),utc=r.now()))
        r.update(r.OUT/'status.json',dict(status=status,error=repr(error),utc=r.now())); raise
    finally:
        r.update(r.OUT/'active_worker.json',dict(identity,status='EXITED',exit_utc=r.now())); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['run']); parser.parse_args(); run()
