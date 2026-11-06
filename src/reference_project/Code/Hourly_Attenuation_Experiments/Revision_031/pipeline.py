"""Full layer/RFE refits after the frozen full-156 search; strictly local."""
from __future__ import annotations

import argparse
import gc
import math
import shutil
import statistics
import time

import core as c
import train_search as t
import verify_search as q

PIPE_SOURCES = ('pipeline.py', 'verify_pipeline.py', 'queue_pipeline.py', 'PIPELINE_PROTOCOL.md')


def folds(train):
    from sklearn.model_selection import GroupKFold
    groups = train.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    result=[]
    for number,(fit,hold) in enumerate(GroupKFold(3).split(train, groups=groups),1):
        fitdays,holddays=sorted(set(groups[fit])),sorted(set(groups[hold]))
        c.require(set(fitdays).isdisjoint(holddays), 'Date overlap in internal fold')
        result.append(dict(fold=number,fit=fit,hold=hold,fit_days=fitdays,hold_days=holddays,
                           fit_index_sha256=c.fingerprint(fit.tolist()),hold_index_sha256=c.fingerprint(hold.tolist())))
    c.require(sorted(c.np.concatenate([f['hold'] for f in result]).tolist())==list(range(len(train))), 'Fold partition')
    return result


def manifest(source, create=False):
    path=c.OUT/'pipeline_manifest.json'
    if path.exists():
        m=c.read(path)
        c.require(c.fingerprint({k:v for k,v in m.items() if k not in ('fingerprint','created_utc')})==m['fingerprint'],'Pipeline manifest changed')
        c.require(m['search_source_fingerprint']==source['fingerprint'],'Different search source')
        for name,digest in m['source_sha256'].items():
            c.require(c.sha(c.ROOT/name)==digest,'Pipeline input/source changed: '+name)
        return m
    c.require(create, 'Pipeline manifest missing')
    c.source_manifest()
    q.verify(source); q.verify_full156(source)
    choice=c.read(c.OUT/'search_selection.json')
    train=c.frame('train',source)
    files=[c.HERE/name for name in PIPE_SOURCES]
    files += [c.OUT/name for name in ('source_manifest.json','search_selection.json','search_qa.json','full156_qa.json')]
    files += [c.OUT/name for name in c.read(c.OUT/'search_qa.json')['verified_files']]
    m=dict(version='rev031-all-layer-rfe-v1',search_source_fingerprint=source['fingerprint'],
           config=choice['config'],fixed_epochs=choice['fixed_epochs'],seed=17,
           feature_names=source['feature_names'],levels_hpa=list(c.LEVELS),layer_states=1024,rfe_states=156,
           fold_method='GroupKFold(3), Beijing calendar date; original training hours only',
           folds=[{k:v for k,v in f.items() if k not in ('fit','hold')} for f in folds(train)],
           permutation_rule='SHA256(fingerprint|k|fold|feature), first 8 bytes big-endian',
           fit_roles=4720,rfe_permutation_evaluations=36735,unchanged_exact_fit_reuse=True,
           source_sha256={str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(set(files))},
           new_network_calls=0,zero_timing_pilot=True)
    m['fingerprint']=c.fingerprint(m); m['created_utc']=c.now()
    c.emit(path,m)
    return m


def path_state(kind,key):
    return c.OUT/kind/'states'/f'{key:04d}.json'


def state_keys(kind):
    return [1023,*range(1023)] if kind=='layer' else list(range(156,0,-1))


def permutation_seed(m,k,fold,name):
    return int.from_bytes(c.hashlib.sha256(f'{m["fingerprint"]}|{k}|{fold}|{name}'.encode()).digest()[:8],'big')


def obtain_fit(names, fold, train, hold, source, m):
    if fold==0 and names==source['feature_names']:
        job=c.read(c.OUT/'search_selection.json')['selected_job']
        record=c.read(c.OUT/'jobs'/job/'result.json')
        reused=True
        reason='same full156 training path and selected-epoch weights from formal search'
    else:
        job=f'subset_{c.fingerprint(names)}_fold_{fold}'
        reused=(c.OUT/'jobs'/job/'result.json').exists()
        reason='exact feature/config/epoch/split/seed match' if reused else None
        record=t.fit_job(job,m['config'],names,train,source,hold,fixed_epochs=m['fixed_epochs'])
    c.require(record['feature_names']==names and record['config']==m['config'] and record['selected_epoch']==m['fixed_epochs'], 'Fit contract differs')
    c.require(record['training']['hour_bjt']==train.hour_bjt.astype(str).tolist(),'Fit training keys differ')
    if hold is not None:
        c.require(record['holdout']['hour_bjt']==hold.hour_bjt.astype(str).tolist(),'Fit hold keys differ')
    file=c.OUT/'jobs'/job/'result.json'
    ref=dict(fold=fold,job_key=job,result_file=str(file.relative_to(c.OUT)),result_sha256=c.sha(file),
             model_file=record['model_file'],model_sha256=record['model_sha256'],
             new_fit_call=not reused,reused=reused,reuse_reason=reason)
    return record,ref


def permute_predictions(k,fold,names,hold,record,m):
    path=c.OUT/'rfe'/'permutations'/f'k{k:04d}_fold{fold}.npz'
    receipt=path.with_suffix('.json')
    identity=c.fingerprint(dict(pipeline=m['fingerprint'],k=k,fold=fold,names=names,
                                model=record['model_sha256'],hours=hold.hour_bjt.astype(str).tolist()))
    if receipt.exists():
        meta=c.read(receipt)
        c.require(meta['identity']==identity and c.sha(path)==meta['sha256'],'Permutation artifact differs')
        with c.np.load(path,allow_pickle=False) as z:
            array=z['prediction_db'].copy()
        c.require(array.shape==(k,len(hold)) and c.np.isfinite(array).all(),'Permutation shape')
        return array,meta
    net,bundle=t.load_model(record)
    x=c.transform(hold,names,bundle['preprocess'])
    values=[]
    for column,name in enumerate(names):
        changed=x.copy()
        order=c.np.random.default_rng(permutation_seed(m,k,fold,name)).permutation(len(hold))
        changed[:,column]=x[order,column]
        values.append(c.predict(net,changed,bundle['preprocess']))
    array=c.np.stack(values)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with c.np.load(path,allow_pickle=False) as z:
            c.require(c.np.array_equal(z['prediction_db'],array),'Orphan permutation differs; preserve and inspect')
    else:
        with path.open('xb') as stream:
            c.np.savez_compressed(stream,prediction_db=array)
    meta=dict(identity=identity,file=str(path.relative_to(c.OUT)),sha256=c.sha(path),
              k=k,fold=fold,feature_names=names,hour_sha256=c.fingerprint(hold.hour_bjt.astype(str).tolist()),
              utc=c.now(),new_fit_calls=0)
    c.emit(receipt,meta)
    return array,meta


def make_state(kind,key,names,train,fold_list,source,m):
    path=path_state(kind,key)
    if path.exists():
        row=c.read(path)
        c.require(row['pipeline_fingerprint']==m['fingerprint'] and row['kind']==kind and row['key']==key and row['feature_names']==names,'State identity differs')
        return row
    c.require(not (c.OUT/'pipeline_evaluation_opened.json').exists(),'No new layer/RFE fit after test opened')
    c.require(shutil.disk_usage(c.OUT).free>=8*1024**3,'Less than 8 GiB free; stopping safely')
    importance=[dict(feature_name=n,fold_delta_rmse_db=[],fold_permuted_rmse_db=[]) for n in names] if kind=='rfe' and key>1 else []
    fold_rows=[]
    for f in fold_list:
        fit,hold=train.iloc[f['fit']],train.iloc[f['hold']]
        record,ref=obtain_fit(names,f['fold'],fit,hold,source,m)
        ref.update({k:v for k,v in f.items() if k not in ('fit','hold')})
        ref['metrics']=record['holdout']['metrics']
        if importance:
            prediction,meta=permute_predictions(key,f['fold'],names,hold,record,m)
            y=hold[c.TARGET].to_numpy(float)
            score=c.np.sqrt(c.np.mean((prediction-y[None,:])**2,axis=1))
            baseline=record['holdout']['metrics']['overall']['rmse_db']
            for item,value in zip(importance,score):
                item['fold_permuted_rmse_db'].append(float(value))
                item['fold_delta_rmse_db'].append(float(value-baseline))
            ref['permutations']=meta
        fold_rows.append(ref)
        gc.collect()
    for item in importance:
        item['mean_delta_rmse_db']=float(c.np.mean(item['fold_delta_rmse_db']))
    removed=min(importance,key=lambda r:(r['mean_delta_rmse_db'],r['feature_name']))['feature_name'] if importance else None
    _,full=obtain_fit(names,0,train,None,source,m)
    cv=c.np.asarray([f['metrics']['overall']['rmse_db'] for f in fold_rows])
    row=dict(pipeline_fingerprint=m['fingerprint'],kind=kind,key=key,feature_names=names,n_features=len(names),
             levels_hpa=[p for p in c.LEVELS if any(n.startswith(f'p{p}__') for n in names)],
             seed=17,epochs=m['fixed_epochs'],folds=fold_rows,full_fit=full,
             cv_mean_rmse_db=float(cv.mean()),cv_se_rmse_db=float(cv.std(ddof=1)/c.np.sqrt(3)),
             permutation_importance=importance,removed_feature_next=removed,
             official_holdouts_used_for_training_or_deletion=False,completed_utc=c.now())
    c.emit(path,row)
    return row


def train_paths(source,m):
    import verify_pipeline as audit
    train=c.frame('train',source)
    fold_list=folds(train)
    for kind in ('layer','rfe'):
        names=list(source['feature_names'])
        for number,key in enumerate(state_keys(kind),1):
            current=c.names_for_mask(key,source['feature_names']) if kind=='layer' else names
            row=make_state(kind,key,current,train,fold_list,source,m)
            audit.verify_state(row,train,fold_list,source,m)
            c.progress('layer_rfe_state_verified',kind=kind,key=key,n_features=len(current),
                       completed_states=number,total_states=len(state_keys(kind)),
                       cv_rmse_db=row['cv_mean_rmse_db'],removed_next=row['removed_feature_next'])
            if kind=='rfe' and key>1:
                c.require(row['removed_feature_next'] in names,'Missing RFE deletion')
                names=[n for n in names if n!=row['removed_feature_next']]


def freeze_selection(source,m):
    import verify_pipeline as audit
    audit.verify_complete_training(source,m)
    selected={}
    for kind in ('layer','rfe'):
        rows=[c.read(path_state(kind,k)) for k in state_keys(kind)]
        best=min(rows,key=lambda r:(r['cv_mean_rmse_db'],r['key']))
        limit=best['cv_mean_rmse_db']+best['cv_se_rmse_db']
        eligible=[r for r in rows if r['cv_mean_rmse_db']<=limit]
        chosen=min(eligible,key=lambda r:(len(r['levels_hpa']) if kind=='layer' else r['n_features'],r['cv_mean_rmse_db'],r['key']))
        selected[kind]=dict(key=chosen['key'],feature_names=chosen['feature_names'],levels_hpa=chosen['levels_hpa'],
                            minimum_cv_key=best['key'],one_se_threshold_db=limit,selected_cv_rmse_db=chosen['cv_mean_rmse_db'],
                            official_holdouts_used=False)
    path=c.OUT/'pipeline_selection.json'
    value=dict(pipeline_fingerprint=m['fingerprint'],selections=selected,
               train_qa_sha256=c.sha(c.OUT/'pipeline_training_qa.json'))
    if path.exists():
        c.require(c.read(path)==value,'One-SE selection changed')
    else:
        c.emit(path,value)
    audit.verify_selection(m)
    return value


def evaluate(source,m):
    import verify_pipeline as audit
    c.source_manifest(); manifest(source)
    c.require(c.read(c.OUT/'pipeline_selection_qa.json')['status']=='PASS','Selection QA gate')
    marker=c.OUT/'pipeline_evaluation_opened.json'
    if not marker.exists():
        c.emit(marker,dict(utc=c.now(),selection_sha256=c.sha(c.OUT/'pipeline_selection.json'),
                           rule='No fitting or reselection after this marker; descriptive curves only'))
    sets={split:c.frame(split,source,allow_test=True) for split in ('validation','test')}
    for kind in ('layer','rfe'):
        for key in state_keys(kind):
            state=c.read(path_state(kind,key))
            path=c.OUT/kind/'evaluation'/f'{key:04d}.json'
            if not path.exists():
                record=c.read(c.OUT/state['full_fit']['result_file'])
                net,bundle=t.load_model(record)
                value=dict(pipeline_fingerprint=m['fingerprint'],kind=kind,key=key,state_sha256=c.sha(path_state(kind,key)),
                           model_sha256=record['model_sha256'],new_fit_calls=0)
                for split,data in sets.items():
                    pred=c.predict(net,c.transform(data,bundle['names'],bundle['preprocess']),bundle['preprocess'])
                    value[split]=c.prediction_record(data,pred)
                c.emit(path,value)
            audit.verify_evaluation(path,state,sets,m)
            c.progress('layer_rfe_evaluated',kind=kind,key=key)
    descriptive_tables(m)
    audit.verify_complete_evaluation(source,m)


def descriptive_tables(m):
    def difference(a,b):
        return None if a is None or b is None else a-b
    curve=[]
    for kind in ('layer','rfe'):
        for key in state_keys(kind):
            state=c.read(path_state(kind,key)); evaluation=c.read(c.OUT/kind/'evaluation'/f'{key:04d}.json')
            curve.append(dict(kind=kind,key=key,n_features=state['n_features'],levels_hpa=state['levels_hpa'],
                              cv_mean_rmse_db=state['cv_mean_rmse_db'],cv_se_rmse_db=state['cv_se_rmse_db'],
                              validation=evaluation['validation']['metrics'],test=evaluation['test']['metrics']))
    layers={r['key']:r for r in curve if r['kind']=='layer'}
    pairs=[]
    for bit,level in enumerate(c.LEVELS):
        for mask in range(1024):
            if mask & (1<<bit):
                continue
            without,with_=layers[mask],layers[mask|(1<<bit)]
            pairs.append(dict(level_hpa=level,without_mask=mask,with_mask=mask|(1<<bit),
                cv_rmse_improvement_db=without['cv_mean_rmse_db']-with_['cv_mean_rmse_db'],
                validation_rmse_improvement_db={g:difference(without['validation'][g]['rmse_db'],with_['validation'][g]['rmse_db']) for g in c.WEATHER},
                test_rmse_improvement_db={g:difference(without['test'][g]['rmse_db'],with_['test'][g]['rmse_db']) for g in c.WEATHER}))
    for name,value in [('all_state_metric_curves.json',curve),('layer_context_pairs.json',pairs)]:
        path=c.OUT/name
        if path.exists():
            c.require(c.read(path)==value,'Descriptive output differs')
        else:
            c.emit(path,value)


def run():
    source=c.source_manifest()
    m=manifest(source,create=True)
    if not (c.OUT/'pipeline_evaluation_opened.json').exists():
        train_paths(source,m)
        freeze_selection(source,m)
    else:
        c.require(c.read(c.OUT/'pipeline_selection_qa.json')['status']=='PASS','Missing frozen selection QA')
    evaluate(source,m)
    c.progress('layer_rfe_numerics_complete',final_delivery_complete=False,
               remaining='Review Chinese report, historical BiLSTM comparison, uncertainty and figure/table delivery; do not claim final completion')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('resume',))
    parser.parse_args()
    with c.writer_lock('layer-rfe'):
        run()
