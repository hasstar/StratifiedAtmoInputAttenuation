"""Independent scalar audits of all layer/RFE artifacts. Never fits a model."""
from __future__ import annotations

import math
import statistics

import core as c
import train_search as t
import verify_search as q


def close(a,b,message):
    c.require(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),message)


def refs_valid(value):
    for name,digest in value['verified_files'].items():
        c.require(c.sha(c.OUT/name)==digest,'Verified artifact changed: '+name)


def verify_fit(ref,names,fit,hold,m):
    path=c.OUT/ref['result_file']
    c.require(c.sha(path)==ref['result_sha256'],'Result SHA differs')
    record=c.read(path)
    c.require(record['config']==m['config'] and record['feature_names']==names and record['selected_epoch']==m['fixed_epochs'], 'Fit architecture/epoch/names')
    c.require(record['seed']==17 and not record['test_used_for_fit_or_selection'],'Seed/test fit guard')
    c.require(record['training']['hour_bjt']==fit.hour_bjt.astype(str).tolist(),'Wrong fit keys')
    c.require(record['model_sha256']==ref['model_sha256'] and record['model_file']==ref['model_file'],'Wrong checkpoint reference')
    c.require(c.sha(c.OUT/ref['model_file'])==ref['model_sha256'],'Checkpoint SHA differs')
    if hold is not None:
        c.require(record['holdout']['hour_bjt']==hold.hour_bjt.astype(str).tolist(),'Wrong held-out keys')
    if record['search']:
        choice=c.read(c.OUT/'search_selection.json')
        c.require(ref['fold']==0 and names==m['feature_names'] and ref['job_key']==choice['selected_job'] and ref['reused'], 'Only exact full156 anchor may reuse search')
    else:
        c.require(record['trained_epochs']==m['fixed_epochs'] and record['stop_reason']=='fixed_frozen_epochs', 'Fixed epoch contract')
    target=c.OUT/'pipeline_fit_qa'/f'{ref["job_key"]}.json'
    if target.exists():
        saved=c.read(target); refs_valid(saved)
        c.require(saved['pipeline_fingerprint']==m['fingerprint'],'Fit QA source differs')
    else:
        replay=q.verify_job(record,fit,hold)
        c.emit(target,dict(status='PASS',pipeline_fingerprint=m['fingerprint'],
            verified_files={ref['result_file']:ref['result_sha256'],ref['model_file']:ref['model_sha256']},
            checkpoint_replay_max_db=replay,train_only_preprocessing='independent median/fmean/pstdev',utc=c.now()))
    return record,target


def verify_permutations(meta,record,names,hold,k,fold,m):
    import pipeline as p
    path=c.OUT/meta['file']
    c.require(c.sha(path)==meta['sha256'] and meta['feature_names']==names,'Permutation source differs')
    c.require(meta['hour_sha256']==c.fingerprint(hold.hour_bjt.astype(str).tolist()),'Permutation keys')
    with c.np.load(path,allow_pickle=False) as z:
        prediction=z['prediction_db'].copy()
    c.require(prediction.shape==(k,len(hold)) and c.np.isfinite(prediction).all(),'Permutation prediction array')
    net,bundle=t.load_model(record)
    x=c.transform(hold,names,bundle['preprocess'])
    y=[float(v) for v in hold[c.TARGET]]
    out=[]
    for j,name in enumerate(names):
        # Construct seed independently rather than call the production seed helper.
        token='|'.join([m['fingerprint'],str(k),str(fold),name]).encode()
        seed=int.from_bytes(c.hashlib.sha256(token).digest()[:8],byteorder='big')
        order=c.np.random.default_rng(seed).permutation(len(hold))
        changed=x.copy(); changed[:,j]=x[order,j]
        replay=c.predict(net,changed,bundle['preprocess'])
        c.require(float(c.np.max(c.np.abs(replay-prediction[j])))<1e-8,'Permutation weight replay differs')
        rmse=math.sqrt(math.fsum((float(a)-b)**2 for a,b in zip(prediction[j],y))/len(y))
        out.append(rmse)
    return out


def verify_state(state,train,fold_list,source,m):
    import pipeline as p
    kind,key=state['kind'],state['key']
    path=p.path_state(kind,key)
    target=c.OUT/kind/'qa'/f'{key:04d}.json'
    if target.exists():
        value=c.read(target)
        c.require(value['status']=='PASS' and value['pipeline_fingerprint']==m['fingerprint'],'State QA contract')
        refs_valid(value)
        return value
    c.require(state['pipeline_fingerprint']==m['fingerprint'] and state['epochs']==m['fixed_epochs'] and state['seed']==17,'State contract')
    c.require(not state['official_holdouts_used_for_training_or_deletion'],'Official label leakage')
    names=state['feature_names']
    c.require(len(names)==state['n_features']==len(set(names)),'State feature identity')
    if kind=='layer':
        expected=[n for n in source['feature_names'] if n.startswith('surface__') or
                  any((key//(2**i))%2==1 and n.startswith(f'p{level}__') for i,level in enumerate(m['levels_hpa']))]
        c.require(names==expected,'Independent bitmask feature mapping differs')
    else:
        c.require(kind=='rfe' and len(names)==key,'RFE state width')
    files={str(path.relative_to(c.OUT)):c.sha(path)}
    metrics=[]; perm=[]
    c.require(len(state['folds'])==3,'Three folds required')
    for ref,f in zip(state['folds'],fold_list):
        c.require(ref['fold']==f['fold'] and ref['fit_days']==f['fit_days'] and ref['hold_days']==f['hold_days'],'Fold date identity')
        c.require(set(ref['fit_days']).isdisjoint(ref['hold_days']),'Fold date leakage')
        for field in ('fit_index_sha256','hold_index_sha256'):
            c.require(ref[field]==f[field],'Fold index differs')
        fit,hold=train.iloc[f['fit']],train.iloc[f['hold']]
        record,fitqa=verify_fit(ref,names,fit,hold,m)
        independent=q.check_record(record['holdout'],hold)
        q.compare_metrics(independent,ref['metrics'])
        metrics.append(independent['overall']['rmse_db'])
        files.update({ref['result_file']:ref['result_sha256'],ref['model_file']:ref['model_sha256'],str(fitqa.relative_to(c.OUT)):c.sha(fitqa)})
        if kind=='rfe' and key>1:
            values=verify_permutations(ref['permutations'],record,names,hold,key,f['fold'],m)
            perm.append(values)
            meta=ref['permutations']
            files[meta['file']]=meta['sha256']
    _,fitqa=verify_fit(state['full_fit'],names,train,None,m)
    full=state['full_fit']
    files.update({full['result_file']:full['result_sha256'],full['model_file']:full['model_sha256'],str(fitqa.relative_to(c.OUT)):c.sha(fitqa)})
    close(statistics.fmean(metrics),state['cv_mean_rmse_db'],'Mean CV RMSE')
    close(statistics.stdev(metrics)/math.sqrt(3),state['cv_se_rmse_db'],'CV standard error')
    if perm:
        c.require([r['feature_name'] for r in state['permutation_importance']]==names,'Permutation feature order')
        independent_importance=[]
        for j,(name,item) in enumerate(zip(names,state['permutation_importance'])):
            delta=[perm[f][j]-metrics[f] for f in range(3)]
            for f in range(3):
                close(perm[f][j],item['fold_permuted_rmse_db'][f],'Permuted RMSE')
                close(delta[f],item['fold_delta_rmse_db'][f],'Permutation delta')
            average=statistics.fmean(delta)
            close(average,item['mean_delta_rmse_db'],'Permutation mean delta')
            independent_importance.append((average,name))
        c.require(min(independent_importance)[1]==state['removed_feature_next'],'RFE minimum deletion differs')
    else:
        c.require(state['permutation_importance']==[] and state['removed_feature_next'] is None,'Unexpected permutation/deletion')
    value=dict(status='PASS',pipeline_fingerprint=m['fingerprint'],kind=kind,key=key,verified_files=files,
               independent_scalar_metrics=True,all_weight_and_permutation_replay=True,
               auditor='same agent, independently coded numerical audit; not external peer review',utc=c.now())
    c.emit(target,value)
    return value


def verify_complete_training(source,m):
    import pipeline as p
    target=c.OUT/'pipeline_training_qa.json'
    if target.exists():
        refs_valid(c.read(target)); return
    train=c.frame('train',source); fold_list=p.folds(train)
    files={}; names=list(source['feature_names']); roles=0; perms=0; unique_jobs=set(); new_calls=0
    for kind in ('layer','rfe'):
        for key in p.state_keys(kind):
            row=c.read(p.path_state(kind,key))
            if kind=='rfe':
                c.require(row['feature_names']==names,'RFE deletion chain differs')
            verify_state(row,train,fold_list,source,m)
            qp=c.OUT/kind/'qa'/f'{key:04d}.json'
            files[str(qp.relative_to(c.OUT))]=c.sha(qp)
            for ref in [*row['folds'],row['full_fit']]:
                roles+=1; unique_jobs.add(ref['job_key']); new_calls+=int(ref['new_fit_call'])
            if kind=='rfe' and key>1:
                perms+=3*key
                names=[n for n in names if n!=row['removed_feature_next']]
    c.require(roles==4720 and perms==36735 and len(names)==1,'Incomplete full experiment')
    c.emit(target,dict(status='PASS',pipeline_fingerprint=m['fingerprint'],verified_files=files,fit_roles=roles,
        unique_weight_sets=len(unique_jobs),roles_recorded_as_new_fits=new_calls,
        rfe_permutation_evaluations=perms,layer_states=1024,rfe_states=156,test_used_for_selection=False,utc=c.now()))


def verify_selection(m):
    import pipeline as p
    saved=c.read(c.OUT/'pipeline_selection.json')
    for kind in ('layer','rfe'):
        states=[c.read(p.path_state(kind,k)) for k in p.state_keys(kind)]
        best=min(states,key=lambda r:(r['cv_mean_rmse_db'],r['key']))
        limit=best['cv_mean_rmse_db']+best['cv_se_rmse_db']
        allowed=[r for r in states if r['cv_mean_rmse_db']<=limit]
        chosen=min(allowed,key=lambda r:(len(r['levels_hpa']) if kind=='layer' else len(r['feature_names']),r['cv_mean_rmse_db'],r['key']))
        actual=saved['selections'][kind]
        c.require(actual['key']==chosen['key'] and actual['feature_names']==chosen['feature_names'] and not actual['official_holdouts_used'],'One-SE rule differs')
        close(actual['one_se_threshold_db'],limit,'One-SE threshold')
    path=c.OUT/'pipeline_selection_qa.json'
    if path.exists():
        refs_valid(c.read(path))
    else:
        c.emit(path,dict(status='PASS',verified_files={n:c.sha(c.OUT/n) for n in ('pipeline_selection.json','pipeline_training_qa.json')},utc=c.now()))


def verify_evaluation(path,state,sets,m):
    target=path.parent.parent/'evaluation_qa'/path.name
    if target.exists():
        refs_valid(c.read(target)); return
    row=c.read(path); ref=state['full_fit']
    c.require(row['pipeline_fingerprint']==m['fingerprint'] and row['model_sha256']==ref['model_sha256'] and row['new_fit_calls']==0,'Evaluation model contract')
    record=c.read(c.OUT/ref['result_file'])
    net,bundle=t.load_model(record)
    maximum=0.
    for split,data in sets.items():
        q.check_record(row[split],data)
        replay=c.predict(net,c.transform(data,bundle['names'],bundle['preprocess']),bundle['preprocess'])
        delta=float(c.np.max(c.np.abs(replay-row[split]['prediction_db'])))
        c.require(delta<1e-8,'Evaluation checkpoint replay differs'); maximum=max(maximum,delta)
    c.emit(target,dict(status='PASS',verified_files={str(path.relative_to(c.OUT)):c.sha(path),ref['model_file']:ref['model_sha256']},
                       max_prediction_replay_error_db=maximum,utc=c.now()))


def verify_complete_evaluation(source,m):
    import pipeline as p
    target=c.OUT/'pipeline_evaluation_qa.json'
    if target.exists():
        refs_valid(c.read(target)); return
    files={}; index={}
    curve=c.read(c.OUT/'all_state_metric_curves.json')
    c.require(len(curve)==1180,'Missing metric curve states')
    for r in curve:
        key=(r['kind'],r['key']); c.require(key not in index,'Duplicate curve state'); index[key]=r
    for kind in ('layer','rfe'):
        for key in p.state_keys(kind):
            path=c.OUT/kind/'evaluation_qa'/f'{key:04d}.json'
            refs_valid(c.read(path)); files[str(path.relative_to(c.OUT))]=c.sha(path)
            state=c.read(p.path_state(kind,key)); summary=index[(kind,key)]
            c.require(summary['n_features']==state['n_features'] and summary['levels_hpa']==state['levels_hpa'],'Curve feature mapping')
            close(summary['cv_mean_rmse_db'],state['cv_mean_rmse_db'],'Curve CV mean')
            close(summary['cv_se_rmse_db'],state['cv_se_rmse_db'],'Curve CV SE')
            prediction=c.read(c.OUT/kind/'evaluation'/path.name)
            for split in ('validation','test'):
                q.compare_metrics(q.scores(prediction[split]),index[(kind,key)][split])
    pairs=c.read(c.OUT/'layer_context_pairs.json')
    c.require(len(pairs)==5120 and len({(r['level_hpa'],r['without_mask']) for r in pairs})==5120,'Layer context pairs incomplete')
    for row in pairs:
        bit=m['levels_hpa'].index(row['level_hpa']); lo=row['without_mask']; hi=row['with_mask']
        c.require(not lo & (1<<bit) and hi==lo|(1<<bit),'Layer pair mapping')
        a,b=index[('layer',lo)],index[('layer',hi)]
        close(a['cv_mean_rmse_db']-b['cv_mean_rmse_db'],row['cv_rmse_improvement_db'],'Pair CV delta')
        for split in ('validation','test'):
            for group in c.WEATHER:
                x,y=a[split][group]['rmse_db'],b[split][group]['rmse_db']
                actual=row[f'{split}_rmse_improvement_db'][group]
                if x is None or y is None:
                    c.require(actual is None,'Empty group must stay uncomputed')
                else:
                    close(x-y,actual,'Pair endpoint delta')
    for name in ('all_state_metric_curves.json','layer_context_pairs.json','pipeline_selection_qa.json'):
        files[name]=c.sha(c.OUT/name)
    c.emit(target,dict(status='PASS',pipeline_fingerprint=m['fingerprint'],verified_files=files,
        independent_metric_curves=1180,layer_context_pairs=5120,n_test=687,final_report_review_pending=True,utc=c.now()))
