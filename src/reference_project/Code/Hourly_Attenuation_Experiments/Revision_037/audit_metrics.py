"""Independent scalar metrics and full protected-file post-inference audit."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from freeze import *
from transport import stage_lock

def reference(y,p):
    y=list(map(float,y));p=list(map(float,p));n=len(y)
    if not n:return dict(n=0,rmse_db=None,r2=None,mae_db=None,bias_db=None)
    mean=math.fsum(y)/n;errors=[a-b for a,b in zip(p,y)]
    sse=math.fsum(e*e for e in errors);sst=math.fsum((a-mean)**2 for a in y)
    return dict(n=n,rmse_db=math.sqrt(sse/n),r2=1-sse/sst if n>=2 and sst>0 else None,
                mae_db=math.fsum(abs(e) for e in errors)/n,bias_db=math.fsum(errors)/n)

def main():
    with stage_lock('independent_numerical_qa'):
        folder=OUT/'evaluation';summary=read(folder/'inference_summary.json')
        need(summary['states']==100 and summary['training_calls']==summary['preprocessing_fit_calls']==0,'Frozen inference accounting')
        pred=pd.read_parquet(folder/'all_hourly_predictions.parquet');source=pd.read_parquet(OUT/'inputs/uk_inputs_38.parquet')
        weather=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet')
        names=read(OUT/'frozen_states_100.json');detail=pd.read_csv(folder/'metrics_all_weather_station.csv',float_precision='round_trip')
        table=pd.read_csv(folder/'UK_100_states_six_metrics.csv',float_precision='round_trip')
        need(len(pred)==566400 and len(table)==100 and len(detail)==1200,'Complete state/metric counts')
        need(not pred.duplicated(['model','k','station','utc_epoch_seconds']).any(),'Duplicate predictions')
        checked=0;max_error=0;datechecks=0
        for st in names:
            part=pred[pred.model.eq(st['model'])&pred.k.eq(st['k'])].reset_index(drop=True)
            pd.testing.assert_frame_equal(part[source[['station','utc_epoch_seconds','truth_db','primary_score_eligible']].columns],
                source[['station','utc_epoch_seconds','truth_db','primary_score_eligible']],check_exact=True)
            event=source[['station','utc_epoch_seconds']].merge(weather,on=['station','utc_epoch_seconds'],validate='one_to_one').event
            need(np.array_equal(event,part.event),'Frozen weather labels changed')
            need(np.isfinite(part.prediction_db).all(),'Nonfinite predictions')
            receipt=read(folder/f'{st["model"]}_k{st["k"]:02d}_receipt.json')
            need(receipt['local_replay_max_abs_db']<=1e-5 and receipt['model_sha256']==st['model_sha256'],'LOCAL replay receipt')
            need(sha(ROOT/st['model_file'])==st['model_sha256'],'Frozen inference checkpoint SHA')
            for row in detail[detail.model.eq(st['model'])&detail.k.eq(st['k'])].itertuples(index=False):
                group=part[part.primary_score_eligible]
                if row.station!='ALL':group=group[group.station.eq(row.station)]
                if row.weather!='overall':group=group[group.event.eq(row.weather)]
                expect=reference(group.truth_db,group.prediction_db)
                for key,value in expect.items():
                    got=getattr(row,key)
                    if value is None:need(pd.isna(got),'Expected null metric')
                    else:
                        err=abs(got-value);max_error=max(max_error,err)
                        need(err<=1e-10*(1+abs(value)),'Independent metric disagreement');checked+=1
                dates=sorted({str(pd.Timestamp(t,unit='s',tz='UTC').date()) for t in group.utc_epoch_seconds})
                stationdates=sorted({s+'|'+str(pd.Timestamp(t,unit='s',tz='UTC').date()) for s,t in zip(group.station,group.utc_epoch_seconds)})
                need(row.dates_utc==';'.join(dates) and row.station_dates_utc==';'.join(stationdates),'Date coverage disagreement')
                need(row.n_dates==len(dates) and row.n_station_dates==len(stationdates),'Date count disagreement');datechecks+=1
                if row.station=='ALL' and row.weather!='other':
                    rr=table[table.model.eq(st['model'])&table.k.eq(st['k'])];need(len(rr)==1,'Wide table unique state')
                    for key in ['r2','rmse_db','n']:
                        need(float(rr.iloc[0][f'{row.weather}_{key}'])==float(getattr(row,key)),'Six-metric table mapping')
        retained=pd.read_csv(folder/'retained_inputs_all_100_states.csv')
        need(len(retained)==550,'550 retained input rows')
        for st in names:
            values=retained[retained.model.eq(st['model'])&retained.k.eq(st['k'])].sort_values('input_order').feature.tolist()
            need(values==st['feature_names'],'Independent RFE retained order')
        guards=read(OUT/'protected_sha256.json')
        for i,(p,digest) in enumerate(guards.items()):
            need(sha(ROOT/p)==digest,'Historical protected file changed '+p)
            if i%1000==0:print('PROTECTED_SHA_RECHECKED',i,flush=True)
        emit(folder/'independent_numeric_qa.json',dict(status='PASS',utc=now(),states=100,metrics_checked=checked,
            coverage_rows_checked=datechecks,maximum_metric_abs_difference=max_error,protected_files_checked=len(guards),
            all_models_inputs_unchanged=True,training_calls=0,preprocessing_fit_calls=0,
            input_sha256=sha(OUT/'inputs/uk_inputs_38.parquet'),script_sha256=sha(Path(__file__)),
            numeric_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
        print('INDEPENDENT_NUMERIC_QA_PASS',flush=True)
if __name__=='__main__':main()
