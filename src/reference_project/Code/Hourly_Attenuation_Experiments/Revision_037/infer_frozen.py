"""Read-only 100-checkpoint inference; every checkpoint first replays LOCAL test."""
import sys, gc, importlib.util, time
import numpy as np
import pandas as pd
from freeze import *
from transport import stage_lock

def runtime():
    p=ROOT/'Code/Hourly_Attenuation_Experiments/Revision_032/runtime.py'
    spec=importlib.util.spec_from_file_location('t015_frozen_runtime32',p)
    r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
    return r

def metric(y,p):
    e=p-y;n=len(y);sy=float(np.sum((y-y.mean())**2)) if n else 0
    reason='N_LT_2' if n<2 else 'ZERO_TARGET_VARIANCE' if sy==0 else None
    return dict(n=n,rmse_db=float(np.sqrt(np.mean(e*e))) if n else None,
        r2=1-float(np.sum(e*e))/sy if reason is None else None,r2_null_reason=reason,
        mae_db=float(np.mean(abs(e))) if n else None,bias_db=float(e.mean()) if n else None)

def main():
    with stage_lock('frozen_inference'):
        need(read(OUT/'inputs/independent_input_qa.json')['status']=='PASS','Independent input gate')
        folder=OUT/'evaluation';folder.mkdir(exist_ok=False);r=runtime();tic=time.monotonic()
        states=read(OUT/'frozen_states_100.json');data=pd.read_parquet(OUT/'inputs/uk_inputs_38.parquet')
        weather=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet')
        weather=data[['station','utc_epoch_seconds']].merge(weather,on=['station','utc_epoch_seconds'],validate='one_to_one')
        need(np.array_equal(weather.primary_score_eligible,data.primary_score_eligible),'Weather scoring key alignment')
        eligible=data.primary_score_eligible.to_numpy(bool);need(eligible.sum()==5593,'5593 scoring hours')
        truth=data.truth_db.to_numpy(float);need(np.isfinite(truth[eligible]).all(),'Scored labels finite')
        original=r.old_manifest()['original_names']
        # Absent structural slots are unretained, not missing retained meteorology.
        legacy_frame=data.reindex(columns=original)
        frames=r.frames();local=frames['test'];need(len(local)==687,'Frozen LOCAL replay cohort')
        rows=[];predictions=[];checks=[];features=[]
        for i,st in enumerate(states):
            need(sha(ROOT/st['state_file'])==st['state_sha256'] and sha(ROOT/st['model_file'])==st['model_sha256'],'Checkpoint integrity')
            names=st['feature_names'];need(np.isfinite(data[names].to_numpy(float)).all(),'Unacquired retained field forbidden')
            bundle=r.load_fit(st);need(bundle['retained_names']==names,'Checkpoint retained order')
            old_eval=read((ROOT/st['state_file']).parent/'evaluation.json');ref=old_eval['results']['test']
            need(ref['hour_bjt']==local.hour_bjt.astype(str).tolist(),'LOCAL replay keys')
            np.testing.assert_allclose(local[r.TARGET].to_numpy(float),ref['truth_db'],rtol=0,atol=1e-12)
            local_pred=r.predict(bundle,local);delta=float(np.max(np.abs(local_pred-np.asarray(ref['prediction_db']))))
            need(delta<=1e-5,'Frozen LOCAL prediction replay differs')
            ukframe=data if bundle.get('type')=='numeric' else legacy_frame
            pred=r.predict(bundle,ukframe);need(pred.shape==(5664,) and np.isfinite(pred).all(),'Complete finite UK prediction')
            output=data[['station','utc_epoch_seconds','truth_db','primary_score_eligible']].copy()
            output['event']=weather.event.to_numpy();output['prediction_db']=pred;output['model']=st['model'];output['k']=st['k']
            predictions.append(output)
            for weather_name in ['overall','rain','dry_overcast','other']:
                mask=eligible.copy()
                if weather_name!='overall':mask&=weather.event.eq(weather_name).to_numpy()
                for station in ['ALL','chilbolton','sparsholt']:
                    group=mask if station=='ALL' else mask&data.station.eq(station).to_numpy()
                    dates=sorted(set(pd.to_datetime(data.loc[group,'utc_epoch_seconds'],unit='s',utc=True).dt.strftime('%Y-%m-%d')))
                    station_dates=sorted(set(data.loc[group,'station']+'|'+pd.to_datetime(data.loc[group,'utc_epoch_seconds'],unit='s',utc=True).dt.strftime('%Y-%m-%d')))
                    rows.append(dict(model=st['model'],k=st['k'],station=station,weather=weather_name,**metric(truth[group],pred[group]),
                        dates_utc=';'.join(dates),station_dates_utc=';'.join(station_dates),n_dates=len(dates),n_station_dates=len(station_dates),
                        model_sha256=st['model_sha256'],input_sha256=sha(OUT/'inputs/uk_inputs_38.parquet'),seed=17,training_calls=0))
            for rank,name in enumerate(names,1):features.append(dict(model=st['model'],k=st['k'],input_order=rank,feature=name,model_sha256=st['model_sha256']))
            checks.append(dict(model=st['model'],k=st['k'],local_replay_max_abs_db=delta,model_sha256=st['model_sha256'],status='REPLAY_PASS_INFERENCE_DONE_PENDING_QA'))
            output.to_parquet(folder/f'{st["model"]}_k{st["k"]:02d}_predictions.parquet',index=False)
            emit(folder/f'{st["model"]}_k{st["k"]:02d}_receipt.json',checks[-1])
            print('INFERRED',i+1,st['model'],st['k'],'LOCAL_REPLAY_MAX_DB',delta,flush=True)
            del bundle;gc.collect()
        detail=pd.DataFrame(rows);detail.to_csv(folder/'metrics_all_weather_station.csv',index=False,encoding='utf-8-sig',float_format='%.17g',mode='x')
        main=detail[detail.station.eq('ALL')&detail.weather.isin(['overall','rain','dry_overcast'])]
        table=main.pivot(index=['model','k'],columns='weather',values=['r2','rmse_db','n'])
        table.columns=[f'{weather}_{value}' for value,weather in table.columns];table=table.reset_index().sort_values(['model','k'])
        need(len(table)==100 and np.isfinite(table.select_dtypes('number')).all().all(),'100 complete states')
        table.to_csv(folder/'UK_100_states_six_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.17g',mode='x')
        pd.DataFrame(features).to_csv(folder/'retained_inputs_all_100_states.csv',index=False,encoding='utf-8-sig',mode='x')
        pd.concat(predictions,ignore_index=True).to_parquet(folder/'all_hourly_predictions.parquet',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(folder/'all_hourly_predictions.csv.gz',index=False,compression='gzip',float_format='%.17g',mode='x')
        emit(folder/'inference_summary.json',dict(status='INFERENCE_DONE_PENDING_INDEPENDENT_QA',utc=now(),states=100,training_calls=0,
            preprocessing_fit_calls=0,local_replay_count=100,uk_hours_per_state=5664,scoring_hours_per_state=5593,
            elapsed_seconds=time.monotonic()-tic,checkpoints=checks,source_sha256={rel(p):sha(p) for p in [Path(__file__),OUT/'inputs/uk_inputs_38.parquet',OUT/'inputs/independent_input_qa.json']}))
if __name__=='__main__':main()
