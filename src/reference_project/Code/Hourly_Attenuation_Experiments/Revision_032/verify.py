"""Independent scalar metric arithmetic and lineage/weight replay (no fitting)."""
from __future__ import annotations
import math
import runtime as r

def scalar_metrics(y,p):
    y,p=list(map(float,y)),list(map(float,p)); n=len(y)
    assert len(p)==n and all(math.isfinite(x) for x in y+p)
    if not n: return dict(n=0,rmse_db=None,mae_db=None,bias_db=None,r2=None,pearson_r=None,pearson_r_squared=None)
    error=[a-b for a,b in zip(p,y)]; sse=math.fsum(e*e for e in error)
    ym=math.fsum(y)/n; pm=math.fsum(p)/n
    sy=math.fsum((v-ym)**2 for v in y); sp=math.fsum((v-pm)**2 for v in p)
    rr=math.fsum((a-ym)*(b-pm) for a,b in zip(y,p))/math.sqrt(sy*sp) if n>1 and sy>0 and sp>0 else None
    return dict(n=n,rmse_db=math.sqrt(sse/n),mae_db=math.fsum(abs(e) for e in error)/n,bias_db=math.fsum(error)/n,
                r2=1-sse/sy if n>1 and sy>0 else None,pearson_r=rr,pearson_r_squared=rr*rr if rr is not None else None)

def equal(a,b,message,tol=1e-9):
    if a is None or b is None: r.require(a is b,message)
    else: r.require(math.isclose(a,b,rel_tol=1e-10,abs_tol=tol),message+f': {a} vs {b}')

def record_qa(rec,data):
    r.require(rec['hour_bjt']==data.hour_bjt.astype(str).tolist(),'QA hour keys/order')
    r.require(rec['truth_db']==data[r.TARGET].to_numpy(float).tolist(),'QA target')
    for g in r.WEATHER[1:]: r.require(rec['weather_masks'][g]==data[g].astype(bool).tolist(),'QA weather masks')
    dates=data.hour_bjt.dt.strftime('%Y-%m-%d').tolist()
    for g in r.WEATHER:
        mask=[True]*len(data) if g=='overall' else rec['weather_masks'][g]
        indices=[i for i,v in enumerate(mask) if v]
        want=scalar_metrics([rec['truth_db'][i] for i in indices],[rec['prediction_db'][i] for i in indices])
        for key,value in want.items(): equal(value,rec['metrics'][g][key],f'QA metric {g}/{key}')
        ds=sorted({dates[i] for i in indices})
        r.require(ds==rec['metrics'][g]['dates'] and len(ds)==rec['metrics'][g]['n_dates'],'QA dates')
    return True

def state_qa(folder,data,m):
    path=folder/'qa.json'
    if path.exists():
        q=r.read(path); r.require(q['status']=='PASS' and q['state_sha256']==r.sha(folder/'state.json') and q['evaluation_sha256']==r.sha(folder/'evaluation.json'),'QA hash'); return
    s=r.read(folder/'state.json'); e=r.read(folder/'evaluation.json'); k=s['k']; names=s['feature_names']; model=s['model']
    r.require(s['manifest']==m['fingerprint'] and len(names)==k and len(set(names))==k and set(names)<=set(m['feature_names']),'QA feature identity')
    r.require(s['official_test_used_for_deletion'] is False and e['training_calls']==0,'QA leakage flag')
    r.require(e['state_sha256']==r.sha(folder/'state.json'),'QA scoring follows frozen state')
    train=data['train']; folds=r.r29.folds(train); delta=[]; replay_max=0.; matrix_replay_count=0
    for f,role in zip(folds,s['folds']):
        fit=train.iloc[f['fit']]; hold=train.iloc[f['hold']]
        r.require(set(fit.hour_bjt.dt.date).isdisjoint(hold.hour_bjt.dt.date),'QA date leakage')
        r.require(role['fold']==f['fold'] and role['train_hour_sha256']==r.fp(fit.hour_bjt.astype(str).tolist()),'QA fit keys')
        record_qa(role['training'],fit); record_qa(role['holdout'],hold)
        b=r.load_fit(role); r.check_prep(b,fit,names,model)
        for rec,df in ((role['training'],fit),(role['holdout'],hold)):
            err=float(r.np.max(abs(r.predict(b,df)-r.np.asarray(rec['prediction_db'])))); replay_max=max(replay_max,err)
            r.require(err<2e-6,'QA fold weight replay')
        if k>1:
            pm=s['permutations'][f['fold']-1]; pp=r.ROOT/pm['path']; r.require(r.sha(pp)==pm['sha256'],'QA permutation digest')
            with r.np.load(pp,allow_pickle=False) as z: matrix=z['prediction_db']
            r.require(matrix.shape==(k,len(hold)) and r.np.isfinite(matrix).all(),'QA permutation shape')
            baseline=scalar_metrics(role['holdout']['truth_db'],role['holdout']['prediction_db'])['rmse_db']
            delta.append([scalar_metrics(role['holdout']['truth_db'],row)['rmse_db']-baseline for row in matrix])
            # All permutation scores independently recomputed; deterministic first/middle/last predictions also replayed.
            x=r.transformed(b,hold)
            for j in sorted({0,k//2,k-1}):
                feature=names[j]; column=r.column_index(b,feature); changed=x.copy()
                seed=int.from_bytes(__import__('hashlib').sha256(f'{r.PERM_NAMESPACE}|{k}|{f["fold"]}|{feature}'.encode()).digest()[:8],'big')
                changed[:,column]=x[r.np.random.default_rng(seed).permutation(len(hold)),column]
                err=float(r.np.max(abs(r.predict_x(b,changed)-matrix[j]))); replay_max=max(replay_max,err); matrix_replay_count+=1
                r.require(err<2e-6,'QA permutation prediction replay')
    if k>1:
        averages=[math.fsum(d[j] for d in delta)/3 for j in range(k)]
        for j,item in enumerate(s['permutation_importance']):
            r.require(item['feature_name']==names[j],'QA permutation name order')
            equal(averages[j],item['mean_delta_rmse_db'],'QA mean permutation delta',1e-8)
            for f in range(3): equal(delta[f][j],item['fold_delta_rmse_db'][f],'QA fold permutation delta',1e-8)
        # Use committed double-precision means for exact tie handling after validating independent arithmetic.
        chosen=min(s['permutation_importance'],key=lambda d:(d['mean_delta_rmse_db'],d['feature_name']))['feature_name']
        r.require(chosen==s['removed_feature_next'],'QA RFE minimum deletion')
    else: r.require(s['removed_feature_next'] is None and not s['permutation_importance'],'QA terminal state')
    full=s['full_fit']; b=r.load_fit(full); r.check_prep(b,train,names,model); record_qa(full['training'],train)
    for split,rec in e['results'].items():
        df=data['test'].loc[r.legal_mask(data['test'])] if split=='test_itu_common684' else data[split]
        record_qa(rec,df); err=float(r.np.max(abs(r.predict(b,df)-r.np.asarray(rec['prediction_db']))))
        replay_max=max(replay_max,err); r.require(err<2e-6,'QA formal holdout weight replay')
    err=float(r.np.max(abs(r.predict(b,train)-r.np.asarray(full['training']['prediction_db'])))); replay_max=max(replay_max,err)
    r.require(err<2e-6,'QA full fitting replay')
    r.emit(path,dict(status='PASS',manifest=m['fingerprint'],model=model,k=k,state_sha256=r.sha(folder/'state.json'),evaluation_sha256=r.sha(folder/'evaluation.json'),
                     independent_metric_implementation='scalar math.fsum; separate from numpy producer',independent_permutation_metric_count=3*k if k>1 else 0,
                     permutation_prediction_replay_count=matrix_replay_count,all_four_models_replayed=True,preprocessors_refit_for_audit_only=True,
                     training_calls_during_qa=0,maximum_prediction_replay_delta_db=replay_max,utc=r.now()))

def bai_selection(data):
    p=r.OUT/'bai_epoch_selection_qa.json'
    if p.exists(): r.require(r.read(p)['status']=='PASS','Bai selection QA'); return
    selection=r.read(r.OUT/'bai_epoch_selection.json'); source=r.read(r.ROOT/selection['source_result'])
    history=source['round_history']; epoch=min(history,key=lambda x:(x['holdout']['overall']['rmse_db'],x['epoch']))['epoch']
    r.require(epoch==selection['selected_epoch'] and selection['test_used'] is False,'Bai earliest validation minimum')
    record_qa(source['training'],data['train']); record_qa(source['holdout'],data['validation'])
    b=r.load_fit(source); r.check_prep(b,data['train'],source['feature_names'],'bai_lstm')
    r.require(r.np.allclose(r.predict(b,data['validation']),source['holdout']['prediction_db'],rtol=0,atol=2e-6),'Bai search weight replay')
    r.emit(p,dict(status='PASS',selected_epoch=epoch,selection_sha256=r.sha(r.OUT/'bai_epoch_selection.json'),source_sha256=r.sha(r.ROOT/selection['source_result']),utc=r.now()))
