"""Frozen checkpoint inference on new UK hours; reuse all previous predictions."""
from __future__ import annotations
import hashlib
import traceback
import joblib
import numpy as np
import pandas as pd
import torch
from common import *

def inference_setup():
    offline();torch.set_num_threads(2)
    comparison=load_module('rev002_frozen_inference',REV2/'run_comparison.py')
    sys.path.insert(0,str(ROOT/'Code/Hourly_Attenuation_Experiments/Revision_014'))
    import models
    require(Path(models.__file__).resolve()==(ROOT/'Code/Hourly_Attenuation_Experiments/Revision_014/models.py').resolve(),'Frozen class identity')
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    classes=[SimpleImputer,StandardScaler,models.TorchRegressor,comparison.Ridge,comparison.ElasticNet,
             comparison.SVR,comparison.RandomForestRegressor,comparison.xgboost.XGBRegressor]
    for cls in classes:
        for attr in ('fit','partial_fit','fit_transform'):
            if hasattr(cls,attr):setattr(cls,attr,deny)
    torch.Tensor.backward=deny
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD,torch.optim.RMSprop):cls.step=deny
    pre=joblib.load(OLD/'preprocessing.joblib')
    require(sha(OLD/'preprocessing.joblib')==read(OLD/'run_manifest.json')['preprocessing_sha256'],'Comparator preprocessing SHA')
    require(pre['feature_names']==FEATURES and pre['fit_split']=='train' and len(pre['fit_hours'])==2404,'Preprocessing train-only provenance')
    registry=read(OLD/'frozen_selection.json')['models']
    require(len(registry)==19,'Fixed comparator count')
    return comparison,pre,registry

def weights_sha(model):
    h=hashlib.sha256()
    for name,array in sorted(model.net.state_dict().items()):
        h.update(name.encode());h.update(array.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def proposed(frame):
    path=R29/'rfe/models/k_0010.joblib'
    require(sha(path)=='b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5','Proposed checkpoint SHA')
    bundle=joblib.load(path);model=bundle['model'];state=read(R29/'rfe/states/k_0010.json');manifest=read(R29/'run_manifest.json')
    require(bundle['retained_names']==FEATURES==state['feature_names'] and bundle['kind']=='rfe' and bundle['key']==10 and bundle['fold']==0,'Natural Top10 identity')
    require(bundle['fingerprint']==manifest['fingerprint'] and bundle['model_name']=='bilstm' and bundle['seed']==17 and manifest['n_train']==2404 and manifest['epochs']==53,'Proposed frozen provenance')
    require(len(bundle['positions'])==model.n_features==268 and tuple(bundle['active_positions'])==(1,3,4,8,9),'Frozen structural layout')
    require(not model.net.training,'Frozen network evaluation mode')
    # Use the established digest implementation; no old evaluation entrypoint is called.
    old=load_module('proposed_digest_only',REV1/'evaluate_gbs_frozen_top10_v002.py')
    wb,pb=old.weights_sha(model),old.preprocessing_sha(bundle)
    require(wb==state['full_fit']['model_weights_sha256'],'Proposed weight identity')
    raw=np.full((len(frame),len(manifest['original_names'])),np.nan);slots=[]
    for name in FEATURES:
        p=manifest['original_names'].index(name);raw[:,p]=frame[name];slots.append(bundle['positions'].index(p))
    native=raw[:,bundle['positions']];inactive=[i for i in range(268) if i not in slots]
    require(np.isfinite(native[:,slots]).all() and np.isnan(native[:,inactive]).all(),'Only actual10 inputs')
    imputed=bundle['imputer'].transform(native);scaled=bundle['scaler'].transform(imputed)
    require(np.isfinite(scaled).all() and np.array_equal(imputed[:,slots],native[:,slots]) and np.all(scaled[:,inactive]==0),'Frozen transformation/layout')
    with torch.inference_mode():pred=model.predict(scaled)
    require(np.isfinite(pred).all() and len(pred)==len(frame),'Proposed finite shape')
    require(old.weights_sha(model)==wb and old.preprocessing_sha(bundle)==pb,'Frozen in-memory state unchanged')
    return pred,{'checkpoint_sha256':sha(path),'weights_sha256':wb,'preprocessing_sha256':pb,
                 'n_actual_features':10,'native_structural_slots':268,'masked_slots':258,
                 'feature_z_diagnostics':{n:{'minimum':float(scaled[:,j].min()),'maximum':float(scaled[:,j].max()),'abs_above5':int((np.abs(scaled[:,j])>5).sum())} for n,j in zip(FEATURES,slots)}}

def scores(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    require(np.isfinite(e).all() and len(e)>0,'Common finite score rows')
    variance=float(np.sum((y-np.mean(y))**2))
    return {'n':len(e),'rmse_db':float(np.sqrt(np.mean(e**2))),'mae_db':float(np.mean(np.abs(e))),
            'bias_db':float(np.mean(e)),'r2':float(1-np.sum(e**2)/variance) if variance>0 else None}

def family(name,registry):
    return 'ours_frozen_top10' if name=='ours_frozen_top10' else next(r['family'] for r in registry if r['id']==name)

def cohort_masks(frame):
    eligible=frame.primary_score_eligible.to_numpy(bool);allrows=np.ones(len(frame),bool)
    selectors={'UK_all':allrows,'UK_original39':frame.period.eq('original39').to_numpy(),
               'UK_extension2003':frame.period.eq('extension2003').to_numpy()}
    for station in ('chilbolton','sparsholt'):
        selectors['UK_'+station]=frame.station.eq(station).to_numpy()
        selectors['UK_'+station+'_extension2003']=frame.station.eq(station).to_numpy() & frame.period.eq('extension2003').to_numpy()
    for cohort,selection in selectors.items():
        yield cohort,'overall',eligible&selection
        yield cohort,'all_finite_sensitivity',np.isfinite(frame.truth_db)&selection
        for band in ('negative_difference','below_0.1','0.1_to_1','at_least_1'):
            yield cohort,'rain_'+band,eligible&selection&frame.rain_band.eq(band).to_numpy()
    for (station,month),group in frame.groupby(['station',frame.hour_utc.dt.strftime('%Y-%m')]):
        mask=np.zeros(len(frame),bool);mask[group.index]=True
        yield 'UK_'+station+'_month_'+month,'overall',mask&eligible

def summarize(frame,local,models,registry):
    metrics=[];distributions=[]
    for cohort,group,mask in cohort_masks(frame):
        n=int(np.count_nonzero(mask))
        if not n:continue
        d=frame.loc[mask];y=d.truth_db.to_numpy(float)
        for name in models:
            metrics.append({'cohort':cohort,'group':group,'model':name,'family':family(name,registry),'comparison':'common_top10_approximate_transfer',**scores(y,d[name].to_numpy(float))})
        if group=='overall':
            distributions.append({'cohort':cohort,'n':len(d),'dates':int(d.hour_utc.dt.date.nunique()),'station_days':int(d.groupby('station').hour_utc.apply(lambda x:x.dt.date.nunique()).sum()),'min_db':float(y.min()),'median_db':float(np.median(y)),'p95_db':float(np.quantile(y,.95)),'max_db':float(y.max()),'fraction_abs_le_0.5':float((np.abs(y)<=.5).mean()),'fraction_ge_3':float((y>=3).mean()),'mean_db':float(y.mean())})
    localnames=models+['physics_geom_circular','physics_p618_hourly_circular']
    for group in ('overall','rain','heavy_rain','dry_overcast','clear_dry','other'):
        mask=np.ones(len(local),bool) if group=='overall' else ~(local.rain|local.dry_overcast|local.clear_dry) if group=='other' else local[group].to_numpy(bool)
        d=local.loc[mask]
        for name in localnames:
            metrics.append({'cohort':'LOCAL_test','group':group,'model':name,'family':family(name,registry) if name in models else name,'comparison':'common_top10' if name in models else 'additional_inputs_rain_only_proxy',**scores(d.truth_db.to_numpy(float),d[name].to_numpy(float))})
    table=pd.DataFrame(metrics);table.to_csv(OUT/'metrics.csv',index=False)
    summary=table.groupby(['cohort','group','comparison','family'],sort=False).agg(runs=('model','size'),n=('n','first'),rmse_mean_db=('rmse_db','mean'),rmse_seed_sd_db=('rmse_db',lambda x:float(x.std(ddof=1)) if len(x)>1 else 0.),mae_mean_db=('mae_db','mean'),bias_mean_db=('bias_db','mean')).reset_index()
    summary.to_csv(OUT/'family_summary.csv',index=False)
    emit(OUT/'target_distributions.json',distributions)
    return table

def bootstrap(frame,models):
    result=[]
    for cohort,group,mask in cohort_masks(frame):
        if group!='overall' or '_month_' in cohort:continue
        d=frame.loc[mask];dates=d.hour_utc.dt.strftime('%Y-%m-%d');unique=sorted(dates.unique())
        count=np.array([(dates==day).sum() for day in unique])
        e=d[models].to_numpy(float)-d.truth_db.to_numpy(float)[:,None]
        sums=np.stack([(e[dates.eq(day).to_numpy()]**2).sum(0) for day in unique])
        rng=np.random.default_rng(20261016)
        draws=rng.integers(0,len(unique),size=(2000,len(unique)))
        boot=np.sqrt(sums[draws].sum(1)/count[draws].sum(1)[:,None]);point=np.sqrt(sums.sum(0)/count.sum())
        for reference in ('ours_frozen_top10','train_mean'):
            j=models.index(reference)
            for k,name in enumerate(models):
                if name==reference:continue
                delta=boot[:,k]-boot[:,j]
                result.append({'cohort':cohort,'model':name,'reference':reference,'n_dates':len(unique),'draws':2000,'delta_rmse_db':float(point[k]-point[j]),'unadjusted_ci95_db':[float(x) for x in np.quantile(delta,[.025,.975])],'negative_favors_model':True})
    emit(OUT/'date_cluster_bootstrap.json',{'method':'paired UTC-date cluster; both stations resampled together; exploratory unadjusted intervals; fixed seed20261016','results':result})

def main():
    with lock():
        require(not (OUT/'evaluation_complete.json').exists(),'Immutable inference already complete')
        gate=read(OUT/'dataset_complete.json');require(gate['status']=='PASS' and sha(OUT/'expanded_dataset.parquet')==gate['dataset_sha256'],'Independent dataset gate')
        guard=read(OUT/'source_sha256_used.json');protected_verify(guard)
        frame=pd.read_parquet(OUT/'expanded_dataset.parquet');require(len(frame)==5664,'Fixed cohort')
        comparison,pre,registry=inference_setup();prehash=joblib.hash(pre)
        models=['ours_frozen_top10']+[r['id'] for r in registry]
        old=pd.read_parquet(OLD/'predictions/gbs39.parquet').set_index(['station','hour_utc'])
        overlap=frame.period.eq('original39');new=~overlap
        prior=old.loc[pd.MultiIndex.from_frame(frame.loc[overlap,['station','hour_utc']])]
        require(len(prior)==1872 and np.array_equal(prior.primary_score_eligible,frame.loc[overlap,'primary_score_eligible']),'Old39 eligibility unchanged')
        require(np.allclose(prior.truth_db,frame.loc[overlap,'truth_db'],equal_nan=True,atol=1e-12,rtol=0),'Old39 labels unchanged')
        table=read(EXT/'validation_gbs_48h_v010/local_top10_acquisition.json')
        oldraw={(r['station'],r['utc_epoch_seconds']):r['values'] for r in table['rows']}
        for row in frame.loc[overlap].itertuples(index=False):
            require(all(getattr(row,n)==oldraw[row.station,row.utc_epoch_seconds][n] for n in FEATURES),'Old39 actual inputs unchanged')
        emit(OUT/'evaluation_launch.json',{'started':now(),'pid':os.getpid(),'new_hours':int(new.sum()),'reused_old39_hours':int(overlap.sum()),'model_ids':models,'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'new_fit_calls':0})
        x=pre['scaler'].transform(pre['imputer'].transform(frame.loc[new,FEATURES].to_numpy(float)))
        require(np.isfinite(x).all(),'Finite selected inputs')
        for row in registry:
            pred=comparison.predict_record(row,x,pre)
            frame[row['id']]=np.nan;frame.loc[new,row['id']]=pred;frame.loc[overlap,row['id']]=prior[row['id']].to_numpy()
            print('frozen_inference',row['id'],len(pred),flush=True)
        pred,compatibility=proposed(frame.loc[new])
        frame['ours_frozen_top10']=np.nan;frame.loc[new,'ours_frozen_top10']=pred;frame.loc[overlap,'ours_frozen_top10']=prior.ours_frozen_top10.to_numpy()
        require(np.isfinite(frame[models]).all().all() and joblib.hash(pre)==prehash,'Finite predictions/frozen preprocessing')
        emit(OUT/'compatibility_gate.json',compatibility)
        frame.to_parquet(OUT/'predictions_expanded.parquet',index=False)
        local=pd.read_parquet(OLD/'predictions/local_test.parquet')
        table=summarize(frame,local,models,registry);bootstrap(frame,models)
        protected_verify(guard)
        emit(OUT/'evaluation_complete.json',{'status':'PENDING_INDEPENDENT_VERIFICATION','completed':now(),'new_fit_calls':0,'new_downloads':0,'new_prediction_rows_per_model':int(new.sum()),'old39_rows_reused':int(overlap.sum()),'local_rows_reused':len(local),'scored_uk_hours':int(frame.primary_score_eligible.sum()),'models':models,'predictions_sha256':sha(OUT/'predictions_expanded.parquet'),'metrics_rows':len(table),'preprocessing_hash_before_after':prehash,'protected_sources':len(guard)})
        print(read(OUT/'evaluation_complete.json'),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        emit(OUT/f'failure_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',{'time':now(),'traceback':traceback.format_exc()})
        raise
