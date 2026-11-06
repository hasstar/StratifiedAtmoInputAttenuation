"""Audit v002: copy station masks before filtering, and replay external outputs.

Original unexecuted verifier is retained unchanged. The boolean-mask copy fixes
a potential pandas backing-array mutation detected during review, not a model,
data-selection, metric or training change. No training functions are called.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_002'
DATA=ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003'
R29=ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
EXT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert read(OUT/'completion.json')['protected_sources_unchanged']
    assert not (OUT/'independent_qa.json').exists(), 'Never overwrite QA'
    expected=read(OUT/'source_sha256_before.json')
    assert expected==read(OUT/'source_sha256_after.json')
    for path,sha in expected.items():
        assert digest(path)==sha, path
    frozen=read(OUT/'frozen_selection.json')
    assert frozen['all_training_complete'] and not frozen['test_or_external_used']
    models=frozen['models']
    pre=joblib.load(OUT/'preprocessing.joblib')
    assert digest(OUT/'preprocessing.joblib')==read(OUT/'run_manifest.json')['preprocessing_sha256']
    names=read(R29/'rfe/states/k_0010.json')['feature_names']
    assert pre['feature_names']==names and pre['fit_split']=='train'
    train=pd.read_parquet(DATA/'train.parquet').sort_values('hour_bjt')
    assert pre['fit_hours']==[str(h) for h in train.hour_bjt]
    x=train[names].to_numpy(dtype=float); x[~np.isfinite(x)]=np.nan
    med=np.nanmedian(x,axis=0)
    assert np.array_equal(pre['imputer'].statistics_,med)
    filled=np.where(np.isfinite(x),x,med)
    assert np.allclose(pre['scaler'].mean_,filled.mean(0),atol=1e-12,rtol=1e-13)
    assert np.allclose(pre['scaler'].var_,filled.var(0),atol=1e-12,rtol=1e-13)
    assert pre['scaler'].n_samples_seen_==2404
    assert pre['target_mean']==train.clear_sky_attenuation_db.mean()
    assert pre['target_scale']==train.clear_sky_attenuation_db.to_numpy().std()
    receipts=[]
    for row in models:
        if row['model_type']=='constant':
            continue
        assert digest(OUT/row['path'])==row['sha256']
        receipt=read(OUT/'training'/f"{row['id']}.json")
        assert receipt['train_hours']==2404 and not receipt['test_or_external_used']
        if row['model_type']=='torch':
            best=min(receipt['history'],key=lambda r:r['validation_rmse_db'])
            assert best['epoch']==receipt['best_epoch']
            assert abs(best['validation_rmse_db']-receipt['validation_rmse_db'])<1e-12
            assert len(receipt['history'])==receipt['epochs_run']
        else:
            selection=read(OUT/'selection'/f"{row['family']}.json")
            assert selection['selected']==min(selection['candidates'],key=lambda r:(r['validation_rmse_db'],r['index']))['index']
            assert selection['selection_split']=='validation' and not selection['test_used']
        receipts.append(row['id'])
    tables={s:pd.read_parquet(OUT/'predictions'/f'local_{s}.parquet') for s in ('validation','test')}
    ext=pd.read_parquet(OUT/'predictions/gbs39.parquet')
    old=read(R29/'rfe/evaluation/k_0010.json')
    for sp,table in tables.items():
        src=pd.read_parquet(DATA/f'{sp}.parquet').sort_values('hour_bjt')
        assert table.hour_bjt.equals(src.hour_bjt.reset_index(drop=True))
        assert np.array_equal(table.truth_db,src.clear_sky_attenuation_db)
        p=pd.Series(old['results'][sp]['prediction_db'],index=pd.to_datetime(old['results'][sp]['hour_bjt']))
        assert np.array_equal(p.reindex(table.hour_bjt),table.ours_frozen_top10)
        assert not set(table.hour_bjt).intersection(set(train.hour_bjt))
    oldext=pd.read_parquet(EXT/'evaluation_002/hourly_predictions.parquet').sort_values(['station','hour_utc']).reset_index(drop=True)
    ext=ext.sort_values(['station','hour_utc']).reset_index(drop=True)
    assert ext[['station','hour_utc']].equals(oldext[['station','hour_utc']])
    assert np.array_equal(ext.ours_frozen_top10,oldext.prediction_db)
    assert np.array_equal(ext.truth_db,oldext.truth_db,equal_nan=True)
    assert np.array_equal(ext.primary_score_eligible,oldext.primary_score_eligible)
    assert ext.primary_score_eligible.sum()==1847
    metric_deltas=[]
    for row in read(OUT/'metrics.json'):
        cohort=row['cohort']
        if cohort.startswith('LOCAL_'):
            table=tables[cohort[6:]]
            mask=np.ones(len(table),dtype=bool) if row['group']=='overall' else table[row['group']].to_numpy(dtype=bool)
        else:
            table=ext
            mask=table.primary_score_eligible.to_numpy(dtype=bool,copy=True)
            if cohort!='GBS39_pooled':
                mask &= table.station.eq(cohort[6:]).to_numpy()
        y=table.truth_db.to_numpy(dtype=float)[mask]
        p=table[row['model']].to_numpy(dtype=float)[mask]
        assert len(y)==row['n'] and np.isfinite(y).all() and np.isfinite(p).all()
        errors=p-y
        calc={'rmse_db':np.sqrt(np.dot(errors,errors)/len(errors)),
              'mae_db':np.sum(np.abs(errors))/len(errors),'bias_db':np.sum(errors)/len(errors)}
        for key,value in calc.items():
            delta=abs(float(value)-row[key]); metric_deltas.append(delta)
            assert delta<1e-9,(cohort,row['model'],key,delta)
    # Replay all new checkpoints, without fit, to check persistence and row ordering.
    import run_comparison as inference
    inference.set_seed(17)
    replay=[]
    for sp,table in tables.items():
        f=pd.read_parquet(DATA/f'{sp}.parquet').sort_values('hour_bjt')
        raw=f[names].to_numpy(dtype=float); raw[~np.isfinite(raw)]=np.nan
        z=pre['scaler'].transform(pre['imputer'].transform(raw))
        for row in models:
            p=inference.predict_record(row,z,pre)
            difference=float(np.max(np.abs(p-table[row['id']].to_numpy())))
            assert difference<1e-8,(sp,row['id'],difference)
            replay.append(dict(split=sp,model=row['id'],maximum_abs_delta_db=difference))
    source=read(EXT/'local_top10_acquisition.json')['rows']
    raw=np.array([[r['values'][n] for n in names] for r in source],dtype=float)
    z=pre['scaler'].transform(pre['imputer'].transform(raw))
    keys=pd.DataFrame({'station':[r['station'] for r in source],
                       'hour_utc':pd.to_datetime([r['utc'] for r in source],utc=True)})
    for row in models:
        keys['replayed']=inference.predict_record(row,z,pre)
        joined=ext[['station','hour_utc',row['id']]].merge(keys,on=['station','hour_utc'],validate='one_to_one')
        difference=float(np.max(np.abs(joined.replayed-joined[row['id']])))
        assert len(joined)==1872 and difference<1e-8,(row['id'],difference)
        replay.append(dict(split='GBS39',model=row['id'],maximum_abs_delta_db=difference))
    test=tables['test']
    dates=pd.to_datetime(test.hour_bjt).dt.strftime('%Y-%m-%d').to_numpy()
    levels,ids=np.unique(dates,return_inverse=True)
    rng=np.random.default_rng(20261016)
    draws=rng.multinomial(len(levels),np.ones(len(levels))/len(levels),size=2000)
    n=np.bincount(ids,minlength=len(levels))
    y=test.truth_db.to_numpy(); ours=(test.ours_frozen_top10.to_numpy()-y)**2
    a=np.bincount(ids,weights=ours,minlength=len(levels))
    bootstrap=[]
    compare=[r['id'] for r in models]+['physics_geom_circular','physics_p618_hourly_circular']
    for name in compare:
        sq=(test[name].to_numpy()-y)**2
        b=np.bincount(ids,weights=sq,minlength=len(levels))
        delta=np.sqrt((draws@b)/(draws@n))-np.sqrt((draws@a)/(draws@n))
        lo,hi=np.quantile(delta,[.025,.975])
        bootstrap.append(dict(model=name,n_days=len(levels),draws=2000,
                              delta_rmse_model_minus_ours_db=float(np.sqrt(sq.mean())-np.sqrt(ours.mean())),
                              ci95_db=[float(lo),float(hi)],negative_favors_comparator=True))
    save(OUT/'day_cluster_bootstrap.json',bootstrap)
    frame=pd.DataFrame(read(OUT/'metrics.json'))
    families={'ours_frozen_top10':'ours_frozen_top10',**{r['id']:r['family'] for r in models}}
    frame['family']=frame.model.map(families).fillna(frame.model)
    agg=frame.groupby(['cohort','comparison_group','group','family'],sort=False).agg(
        runs=('model','count'),n=('n','first'),rmse_mean_db=('rmse_db','mean'),
        rmse_sd_db=('rmse_db','std'),mae_mean_db=('mae_db','mean'),bias_mean_db=('bias_db','mean')).reset_index()
    agg.to_csv(OUT/'family_summary.csv',index=False)
    save(OUT/'independent_qa.json',dict(status='PASS',verified_utc=datetime.now(timezone.utc).isoformat(),
                                      verifier_sha256=digest(__file__),protected_sources=len(expected),
                                      train_only_preprocessing=True,unchanged_old_predictions=True,
                                      original_masks_and_keys=True,checkpoint_replay=replay,
                                      metric_rows=len(read(OUT/'metrics.json')),maximum_metric_delta=max(metric_deltas),
                                      epoch_selection_validated=True,bootstrap_not_used_for_selection=True,
                                      no_fit_in_audit=True,external_new_days_included=False))
    print(agg[(agg.cohort=='LOCAL_test') & (agg['group']=='overall')].to_string(index=False))
    print(agg[agg.cohort=='GBS39_pooled'].to_string(index=False))
    print('INDEPENDENT_QA_PASS')

if __name__=='__main__':
    main()
