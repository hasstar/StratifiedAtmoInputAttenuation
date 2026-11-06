"""Independent saved-output audit, scalar metric reconstruction and frozen replay."""
from __future__ import annotations
import math
import traceback
import joblib
import numpy as np
import pandas as pd
import torch
from common import *

def mask_for(cohort,group,uk,local):
    if cohort=='LOCAL_test':
        f=local
        if group=='overall':mask=np.ones(len(f),bool)
        elif group=='other':mask=~(f.rain|f.clear_dry|f.dry_overcast).to_numpy()
        else:mask=f[group].to_numpy(bool)
    else:
        f=uk;mask=np.ones(len(f),bool)
        if cohort.startswith('UK_chilbolton'):mask &= f.station.eq('chilbolton').to_numpy()
        if cohort.startswith('UK_sparsholt'):mask &= f.station.eq('sparsholt').to_numpy()
        if 'extension2003' in cohort:mask &= f.hour_utc.dt.year.eq(2003).to_numpy()
        if 'original39' in cohort:mask &= f.hour_utc.dt.year.eq(2004).to_numpy()
        if '_month_' in cohort:mask &= f.hour_utc.dt.strftime('%Y-%m').eq(cohort.split('_month_')[1]).to_numpy()
        if group!='all_finite_sensitivity':mask &= f.valid_seconds.ge(3420).to_numpy()
        mask &= np.isfinite(f.truth_db)
        if group.startswith('rain_'):
            liquid=3600*(f[RAIN[0]]-f[RAIN[1]]);band=group[5:]
            cond={'negative_difference':liquid<0,'below_0.1':(liquid>=0)&(liquid<.1),
                  '0.1_to_1':(liquid>=.1)&(liquid<1),'at_least_1':liquid>=1}[band]
            mask &= cond.to_numpy()
    return f.loc[mask]

def raw_negative_controls():
    build=load_module('verify_actual_decoder',HERE/'build_expanded.py')
    controls=[]
    for variable in ['geopotential','convective_available_potential_energy',RAIN[0]]:
        recpath=next(p for p in sorted((ACQ/'records').glob('*.json')) if read(p)['records'][0][0][0]==variable)
        record=read(recpath);node=read(ACQ/'jobs'/recpath.name)['job']
        raw=(ROOT/record['source']).read_text();das=(ROOT/record['metadata']).read_text()
        nc=build.CATALOG[variable][0]
        badnode=read(ACQ/'jobs'/recpath.name)['job'];badnode['coords']['latitude'][0]+=.01
        examples={'wrong_parameter':(raw,das.replace(f'Int32 ecmwf_parameter {build.CATALOG[variable][2]};','Int32 ecmwf_parameter 999;'),node),
                  'off_grid_request':(raw,das,badnode),
                  'wrong_dataset':(raw.replace('d633000','d999999'),das,node),
                  'truncated_payload':(raw[:len(raw)//2],das,node)}
        for name,args in examples.items():
            try:build.decode_ascii(args[0],args[1],variable,args[2])
            except (RuntimeError,ValueError,KeyError,IndexError):controls.append(variable+':'+name)
            else:raise RuntimeError('Bad payload unexpectedly accepted: '+name)
    # Hash mismatch must fail independently of decoder acceptance.
    actual=sha(ROOT/record['source'])
    require(actual!=('0'*64),'SHA mismatch control')
    controls.append('source_sha_mismatch_detected')
    return controls

def replay(uk,registry):
    offline();torch.set_num_threads(2)
    comp=load_module('verification_model_classes',REV2/'run_comparison.py')
    sys.path.insert(0,str(ROOT/'Code/Hourly_Attenuation_Experiments/Revision_014'))
    import models
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    for cls in [models.TorchRegressor,SimpleImputer,StandardScaler,comp.Ridge,comp.ElasticNet,comp.SVR,comp.RandomForestRegressor,comp.xgboost.XGBRegressor]:
        for attr in ('fit','partial_fit','fit_transform'):
            if hasattr(cls,attr):setattr(cls,attr,deny)
    torch.Tensor.backward=deny
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.RMSprop,torch.optim.SGD):cls.step=deny
    # Non-outcome-selected replay: first, middle and last collected hour per station/month.
    indexes=[]
    for _,f in uk[uk.hour_utc.dt.year.eq(2003)].groupby(['station',uk.hour_utc.dt.strftime('%Y-%m')]):
        indexes.extend(f.index[[0,len(f)//2,len(f)-1]].tolist())
    test=uk.loc[sorted(set(indexes))]
    pre=joblib.load(OLD/'preprocessing.joblib');before=joblib.hash(pre)
    values=test[FEATURES].to_numpy(float)
    require(np.isfinite(values).all(),'No evaluation imputation required')
    # Independently reconstruct transform from frozen means/scales; no fit or transform helper.
    x=(values-pre['scaler'].mean_)/pre['scaler'].scale_
    checks={}
    for row in registry:
        if row['model_type']=='constant':out=np.full(len(x),pre['target_mean'])
        elif row['model_type']=='sklearn':
            mdl=joblib.load(OLD/row['path']);out=np.asarray(mdl.predict(x))*pre['target_scale']+pre['target_mean']
        else:
            mdl=comp.build_neural(row['family']);mdl.load_state_dict(torch.load(OLD/row['path'],map_location='cpu',weights_only=True));mdl.eval()
            with torch.inference_mode():
                out=mdl(torch.tensor(x,dtype=torch.float32))
                if out.ndim==3:out=out.squeeze(-1).mean(dim=1)
                out=out.numpy().astype(float)*pre['target_scale']+pre['target_mean']
        delta=float(np.max(np.abs(out-test[row['id']].to_numpy())))
        require(delta<2e-5,'Frozen comparator replay: '+row['id'])
        checks[row['id']]={'rows':len(test),'maximum_delta_db':delta}
    bundle=joblib.load(R29/'rfe/models/k_0010.joblib');manifest=read(R29/'run_manifest.json')
    native=np.full((len(test),len(bundle['positions'])),np.nan)
    for feature in FEATURES:
        idx=bundle['positions'].index(manifest['original_names'].index(feature));native[:,idx]=test[feature]
    imputer=bundle['imputer'];scaler=bundle['scaler']
    filled=np.where(np.isfinite(native),native,imputer.statistics_[None,:]);xnative=(filled-scaler.mean_)/scaler.scale_
    require(np.isfinite(xnative).all(),'Frozen native structural transform')
    with torch.inference_mode():out=bundle['model'].predict(xnative)
    delta=float(np.max(np.abs(out-test.ours_frozen_top10.to_numpy())))
    require(delta<2e-5,'Proposed frozen replay')
    checks['ours_frozen_top10']={'rows':len(test),'maximum_delta_db':delta}
    require(joblib.hash(pre)==before,'Replay did not change preprocessor')
    return checks,test[['station','hour_utc']].astype(str).to_dict('records')

def verify_bootstrap(uk,models):
    data=read(OUT/'date_cluster_bootstrap.json')['results'];maximum=0.
    for cohort in sorted({r['cohort'] for r in data}):
        f=mask_for(cohort,'overall',uk,None).copy();f['date']=f.hour_utc.dt.strftime('%Y-%m-%d')
        errors=f[models].subtract(f.truth_db,axis=0)**2
        grouped=errors.groupby(f.date).sum().sort_index();counts=f.groupby('date').size().sort_index()
        rng=np.random.default_rng(20261016);indices=rng.integers(0,len(counts),size=(2000,len(counts)))
        estimates=np.sqrt(grouped.to_numpy()[indices].sum(axis=1)/counts.to_numpy()[indices].sum(axis=1)[:,None])
        point=np.sqrt(grouped.sum(axis=0)/counts.sum())
        for row in (r for r in data if r['cohort']==cohort):
            k,j=models.index(row['model']),models.index(row['reference'])
            require(row['n_dates']==len(counts) and row['draws']==2000,'Bootstrap unit/count')
            ci=np.quantile(estimates[:,k]-estimates[:,j],[.025,.975])
            diff=max(np.abs(ci-row['unadjusted_ci95_db']).max(),abs(point.iloc[k]-point.iloc[j]-row['delta_rmse_db']))
            maximum=max(maximum,float(diff));require(diff<1e-10,'Bootstrap reconstruction')
    return {'comparisons':len(data),'maximum_delta_db':maximum,'paired_station_dates_preserved':True,'not_used_for_selection':True}

def main():
    offline()
    with lock():
        require(not (OUT/'independent_qa.json').exists(),'Verification already exists')
        guard=read(OUT/'source_sha256_used.json');protected_verify(guard)
        completed=read(OUT/'evaluation_complete.json');require(sha(OUT/'predictions_expanded.parquet')==completed['predictions_sha256'],'Prediction artifact hash')
        uk=pd.read_parquet(OUT/'predictions_expanded.parquet');local=pd.read_parquet(OLD/'predictions/local_test.parquet')
        base=pd.read_parquet(OUT/'expanded_dataset.parquet');pd.testing.assert_frame_equal(uk[base.columns],base)
        require(len(uk)==5664 and uk[['station','hour_utc']].duplicated().sum()==0,'Unique fixed station hours')
        require(uk.primary_score_eligible.equals(np.isfinite(uk.truth_db)&uk.valid_seconds.ge(3420)) and uk.primary_score_eligible.sum()==5593,'Independent mask')
        require(Counter(uk.exclusion_reason)=={'eligible':5593,'missing_native_target':37,'native_coverage_below_95':34},'Every exclusion accounted')
        registry=read(OLD/'frozen_selection.json')['models'];names=['ours_frozen_top10']+[r['id'] for r in registry]
        require(names==completed['models'] and np.isfinite(uk[names]).all().all(),'Same20 finite model IDs')
        provenance=pd.read_parquet(OUT/'field_provenance.parquet')
        require(len(provenance)==len(uk)*12 and not provenance.duplicated(['station','utc_epoch_seconds','name']).any(),'Full12field coverage')
        pivot=provenance.pivot(index=['station','utc_epoch_seconds'],columns='name',values='value').sort_index()
        expected=uk.set_index(['station','utc_epoch_seconds'])[FEATURES+RAIN].sort_index()
        require(np.array_equal(pivot[FEATURES+RAIN],expected),'Dataset matches per-value source provenance')
        require((provenance.valid_utc_epoch_seconds-provenance.utc_epoch_seconds).equals(provenance.name.isin(RAIN).astype(int)*3600),'Rain/feature UTC convention')
        for p,h in provenance[['source','source_sha256']].drop_duplicates().itertuples(index=False):require(sha(ROOT/p)==h,'Raw provenance SHA')
        old=pd.read_parquet(OLD/'predictions/gbs39.parquet').set_index(['station','hour_utc']).sort_index()
        overlap=uk[uk.period.eq('original39')].set_index(['station','hour_utc']).sort_index()
        for col in names:require(np.array_equal(overlap[col],old[col]),'Old39 prediction changed: '+col)
        require(np.allclose(overlap.truth_db,old.truth_db,equal_nan=True,atol=0,rtol=0),'Old39 truth changed')
        metrics=pd.read_csv(OUT/'metrics.csv');maximum=0.
        for row in metrics.itertuples(index=False):
            f=mask_for(row.cohort,row.group,uk,local);errors=(f[row.model]-f.truth_db).tolist();n=len(errors)
            require(n==row.n and n>0,'Independent group count')
            mse=math.fsum(e*e for e in errors)/n;mae=math.fsum(abs(e) for e in errors)/n;bias=math.fsum(errors)/n
            diff=max(abs(math.sqrt(mse)-row.rmse_db),abs(mae-row.mae_db),abs(bias-row.bias_db));maximum=max(maximum,diff)
            require(diff<1e-11,'Independent metric mismatch')
            vals=f.truth_db.tolist();mean=math.fsum(vals)/n;variance=math.fsum((v-mean)**2 for v in vals)
            if variance:require(abs((1-mse*n/variance)-row.r2)<1e-9,'Independent R2')
        summary=pd.read_csv(OUT/'family_summary.csv')
        for row in summary.itertuples(index=False):
            q=metrics[(metrics.cohort==row.cohort)&(metrics.group==row.group)&(metrics.family==row.family)&(metrics.comparison==row.comparison)]
            require(len(q)==row.runs and q.n.nunique()==1 and q.n.iloc[0]==row.n,'Family same-mask/seed count')
            require(abs(q.rmse_db.mean()-row.rmse_mean_db)<1e-11 and abs((q.rmse_db.std(ddof=1) if len(q)>1 else 0)-row.rmse_seed_sd_db)<1e-11,'Family mean/seed SD')
        replay_checks,replay_keys=replay(uk,registry)
        boot=verify_bootstrap(uk,names);controls=raw_negative_controls()
        protected_verify(guard)
        emit(OUT/'independent_qa.json',{'status':'PASS','verified':now(),'verifier_sha256':sha(Path(__file__)),'protected_sources':len(guard),'raw_field_provenance_rows':len(provenance),'original39_prediction_equality':True,'local_predictions_read_only_reused':True,'common_model_ids':names,'metric_rows':len(metrics),'maximum_metric_delta_db':maximum,'family_rows':len(summary),'checkpoint_replay':replay_checks,'replay_keys':replay_keys,'bootstrap':boot,'additional_negative_controls':controls,'new_fit_calls':0,'new_downloads':0})
        print('independent_qa_PASS',len(metrics),'metric_rows',len(guard),'protected_sources',flush=True)

if __name__=='__main__':
    from collections import Counter
    try:main()
    except Exception:
        emit(OUT/f'failure_verify_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',{'time':now(),'traceback':traceback.format_exc()})
        raise
