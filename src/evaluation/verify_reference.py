"""Independent metric/RFE-path audit and replay of all 100 frozen UK states."""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s
import numpy as np
import pandas as pd

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;n=len(y)
    var=np.sum((y-y.mean())**2) if n else 0
    return dict(n=n,rmse_db=float(np.sqrt(np.mean(e*e))) if n else None,
      mae_db=float(np.mean(abs(e))) if n else None,bias_db=float(e.mean()) if n else None,
      r2=float(1-np.sum(e*e)/var) if n>1 and var>0 else None)

def audit_states(out):
    rows=[];features=[];checks=[];largest=0.;state_count=0
    models=list(s.config()['hourly']['models'])
    for pool,experiment,start,names in [('full156','Hourly_Attenuation_Revision_032_Multi_Model_RFE',156,models),('base50','Hourly_Attenuation_Revision_041_TabM_Surface_RFE',50,['tabm'])]:
        for model in names:
            expected=None
            for k in range(start,0,-1):
                folder=s.original(f'Outputs/Experiments/{experiment}/models/{model}/k_{k:04d}')
                st=s.read(folder/'state.json');ev=s.read(folder/'evaluation.json')
                assert ev['state_sha256']==s.sha(folder/'state.json')
                assert st['k']==k and len(st['feature_names'])==len(set(st['feature_names']))==k
                if expected is not None:assert st['feature_names']==expected
                if k>1:
                    imp=st['permutation_importance']
                    assert len(imp)==k and all(len(z['fold_delta_rmse_db'])==3 for z in imp)
                    for z in imp:assert abs(np.mean(z['fold_delta_rmse_db'])-z['mean_delta_rmse_db'])<1e-12
                    removed=min(imp,key=lambda z:(z['mean_delta_rmse_db'],z['feature_name']))['feature_name']
                    assert removed==st['removed_feature_next']
                    expected=[x for x in st['feature_names'] if x!=removed]
                else:assert st['removed_feature_next'] is None
                for split,record in ev['results'].items():
                    y=np.asarray(record['truth_db']);p=np.asarray(record['prediction_db'])
                    assert len(y)==len(p)==len(record['hour_bjt']) and np.isfinite(y).all() and np.isfinite(p).all()
                    for weather,stored in record['metrics'].items():
                        mask=np.ones(len(y),bool) if weather=='overall' else np.asarray(record['weather_masks'][weather],bool)
                        actual=metrics(y[mask],p[mask]);delta=0.
                        for key,val in actual.items():
                            if val is None:assert stored[key] is None
                            else:delta=max(delta,abs(val-stored[key]))
                        assert delta<1e-10;largest=max(largest,delta)
                        rows.append(dict(pool=pool,model=model,k=k,split=split,weather=weather,**actual))
                for i,name in enumerate(st['feature_names'],1):
                    features.append(dict(pool=pool,model=model,k=k,input_order=i,feature=name,
                      removed_next=name==st['removed_feature_next'],state_file=s.rel(folder/'state.json')))
                checks.append(dict(pool=pool,model=model,k=k,state=s.rel(folder/'state.json'),
                    full_model=st['full_fit']['model_file'].replace('\\','/'),full_model_sha256=st['full_fit']['model_sha256'],
                    removed_feature_next=st['removed_feature_next'],folds=len(st['folds'])))
                state_count+=1
            print('AUDITED',pool,model,flush=True)
    assert state_count==1610
    pd.DataFrame(rows).to_csv(out/'all_local_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(features).to_csv(out/'all_retained_and_removed_features.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'all_states_and_models.csv',index=False)
    result=dict(status='PASS',states=state_count,fit_roles=sum(x['folds']+1 for x in checks),metric_rows=len(rows),maximum_metric_error=largest,
                audit='Recomputed from saved predictions and checked every RFE transition; this does not rerun 6440 training roles.')
    s.write(out/'all_state_audit.json',result);return result

def infer_uk(out,hourly_run=None):
    r=s.hourly();folder=s.original('Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE')
    data=pd.read_parquet(folder/'inputs/uk_inputs_38.parquet')
    weather=pd.read_parquet(s.original('Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet'))
    weather=data[['station','utc_epoch_seconds']].merge(weather,on=['station','utc_epoch_seconds'],validate='one_to_one')
    states=s.read(folder/'frozen_states_100.json');reference=pd.read_parquet(folder/'evaluation/all_hourly_predictions.parquet')
    frames=r.frames();local=frames['test'];legacy=data.reindex(columns=r.old_manifest()['original_names'])
    rows=[];outputs=[];checks=[];start=time.perf_counter()
    for st in states:
        if hourly_run:
            local_state=Path(hourly_run)/'full156'/st['model']/f'k_{st["k"]:04d}'/'state.json'
            record=s.read(local_state);ref=record['full_fit'];features=record['feature_names']
        else:ref=st;features=st['feature_names']
        if not set(features).issubset(data.columns):raise RuntimeError('A newly trained path requires UK fields absent from the frozen 38-field input. Inspect; do not impute an unacquired field.')
        assert np.isfinite(data[features].to_numpy(float)).all()
        b=r.load_fit(ref);assert b['retained_names']==features
        lp=r.predict(b,local);p=r.predict(b,data if b.get('type')=='numeric' else legacy)
        oldlocal=s.read(s.original(st['state_file']).parent/'evaluation.json')['results']['test']
        expected=reference[(reference.model==st['model'])&(reference.k==st['k'])].set_index(['station','utc_epoch_seconds']).loc[pd.MultiIndex.from_frame(data[['station','utc_epoch_seconds']])]
        ld=float(np.max(abs(lp-np.asarray(oldlocal['prediction_db']))));ud=float(np.max(abs(p-expected.prediction_db.to_numpy())))
        if not hourly_run:assert max(ld,ud)<1e-5
        output=data[['station','utc_epoch_seconds','hour_utc','truth_db','primary_score_eligible']].copy()
        output['prediction_db']=p;output['model']=st['model'];output['k']=st['k'];output['event']=weather.event.to_numpy();outputs.append(output)
        for site in ('ALL','chilbolton','sparsholt'):
            for event in ('overall','rain','dry_overcast','other'):
                mask=data.primary_score_eligible.to_numpy().copy()
                if site!='ALL':mask&=data.station.eq(site).to_numpy()
                if event!='overall':mask&=weather.event.eq(event).to_numpy()
                rows.append(dict(model=st['model'],k=st['k'],station=site,weather=event,**metrics(data.truth_db.to_numpy()[mask],p[mask])))
        checks.append(dict(model=st['model'],k=st['k'],local_max_abs_db=ld,uk_max_abs_db=ud))
        print('INFERRED',len(checks),st['model'],st['k'],flush=True);del b;gc.collect()
    pd.concat(outputs,ignore_index=True).to_parquet(out/'all_hourly_predictions.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'uk_metrics.csv',index=False,float_format='%.17g')
    s.write(out/'uk_inference_qa.json',dict(status='PASS_FROZEN_REPLAY' if not hourly_run else 'NEW_MODEL_INFERENCE_COMPLETE',
        states=len(checks),scored_hours=5593,training_calls=0,preprocessing_fit_calls=0,elapsed_seconds=time.perf_counter()-start,
        checks=checks,maximum_reference_error=max(max(x['local_max_abs_db'],x['uk_max_abs_db'])for x in checks)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--skip-uk',action='store_true');ap.add_argument('--skip-state-audit',action='store_true')
    ap.add_argument('--hourly-run',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not a.skip_state_audit:audit_states(a.out)
    if not a.skip_uk:infer_uk(a.out,a.hourly_run)
if __name__=='__main__':main()
