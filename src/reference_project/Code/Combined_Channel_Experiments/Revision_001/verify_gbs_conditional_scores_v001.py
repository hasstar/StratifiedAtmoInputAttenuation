"""Literal native-second pinball/boolean replay, separate from prefix evaluator."""
from pathlib import Path
import hashlib, json, math, time
from collections import defaultdict
import numpy as np
import pandas as pd
import netCDF4
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT=BASE/'conditional_evaluation_independent_qa_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def same(a,b,tol=1e-10):
    assert (a is None and b is None) or (a is not None and b is not None and np.max(np.abs(np.asarray(a)-np.asarray(b)))<tol)
def avg(xs):return None if not xs else math.fsum(xs)/len(xs)
def mask_event(event,heavy,name): return name=='overall' or (heavy if name=='heavy_rain_subset' else event==name)
def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    primary=BASE/'conditional_evaluation_001'
    protected=read(primary/'protected_inputs.json');assert all(sha(ROOT/p)==h for p,h in protected.items())
    w=pd.read_parquet(BASE/'conditional_fast_001/weather_labels.parquet')
    p=pd.read_parquet(BASE/'evaluation_002/hourly_predictions.parquet')
    joined=p.merge(w,on=['station','hour_utc'],validate='one_to_one')
    hourly=read(primary/'weather_hourly_metrics.json');weather_checks=0
    for analysis,eligible in [('primary','primary_score_eligible'),('all_finite_diagnostic','finite_target_diagnostic_eligible')]:
        for row in hourly[analysis]:
            g=[r for r in joined.itertuples() if getattr(r,eligible) and (row['station']=='pooled' or r.station==row['station']) and mask_event(r.event,r.heavy_rain,row['event'])]
            e=[r.prediction_db-r.truth_db for r in g]
            assert row['n_station_hours']==len(g) and row['n_dates']==len({r.hour_utc.date() for r in g})
            same(row['rmse_db'],None if not e else math.sqrt(math.fsum(x*x for x in e)/len(e)));same(row['mae_db'],avg([abs(x) for x in e]));same(row['bias_db'],avg(e));weather_checks+=1
    quantiles=read(primary/'frozen_quantiles.json');package=read(ROOT/'Outputs/Experiments/Fast_Fading_Revision_002/final_generator/model.json')
    for event,c in package['components'].items():
        with np.load(ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'/c['cdf_file_relative_to_revision_output']) as a:
            # Locate probability in cumulative intervals by lower bound, preserving atoms.
            for n in [201,401]:
                targets=1-(np.arange(n)+.5)/n
                idx=np.searchsorted(a['cdf'],targets,side='left')
                expected=-10*np.log10(a['q'][idx]);same(expected,quantiles[event][str(n)],1e-13)
    scores=pd.read_parquet(primary/'probability_hourly.parquet').set_index(['station','hour_utc'])
    maximum=0.;seconds=0;hours=0;literal_rows=[]
    for file_index,(source,g) in enumerate(joined.groupby('source_file',sort=True)):
        path=ROOT/'外部样例数据CEDA/GBS_20p7GHz'/source
        with netCDF4.Dataset('literal.nc',memory=path.read_bytes()) as ds:
            ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];t=np.asarray(ds.variables['time'][:]);y=np.asarray(v[:],float)
            valid=np.isfinite(y)&(y!=v._FillValue)&(y>=v.valid_min)&(y<=v.valid_max)&np.isfinite(t)&(t>=1)&(t<=86400)
            hour_bins=np.floor((t-1)/3600)
            for r in g.itertuples():
                if not (r.finite_target_diagnostic_eligible and r.weather_known):
                    assert (r.station,r.hour_utc) not in scores.index;continue
                observed=y[valid&(hour_bins==r.hour_utc.hour)];old=scores.loc[(r.station,r.hour_utc)];assert len(observed)==old.n_seconds
                fresh={'station':r.station,'hour_utc':r.hour_utc,'event':r.event,'heavy_rain':r.heavy_rain,'primary':r.primary_score_eligible,'n_seconds':len(observed)}
                for label,shift in [('combined',r.prediction_db),('oracle',r.truth_db)]:
                    for n in [201,401]:
                        thresholds=np.asarray(quantiles[r.event][str(n)])+shift
                        probs=(np.arange(n)+.5)/n
                        # Explicit pointwise pinball, unlike the first implementation's sorted cumulative formula.
                        delta=observed[:,None]-thresholds[None,:]
                        literal=float(2*np.maximum(probs*delta,(probs-1)*delta).mean())
                        key=f'{label if label=="combined" else "oracle_fast"}_crps{n}_db'
                        maximum=max(maximum,abs(literal-old[key]));same(literal,old[key]);fresh[key]=literal
                    boundary=np.asarray(quantiles[r.event]['bounds'])+shift
                    for level,lo,hi in [(90,1,3),(98,0,4)]:
                        key=f'{label}_central{level}_coverage';v=float(np.count_nonzero((observed>=boundary[lo])&(observed<=boundary[hi]))/len(observed));same(v,old[key]);fresh[key]=v
                    for level,hi in [(95,3),(99,4)]:
                        key=f'{label}_tail{level}_exceedance';v=float(np.count_nonzero(observed>boundary[hi])/len(observed));same(v,old[key]);fresh[key]=v
                    thresholds=np.asarray(quantiles[r.event]['201'])+shift
                    empirical=np.mean(observed[:,None]<=thresholds[None,:],axis=0);same(empirical,old[f'{label}_actual_cdf201'])
                baseline=math.fsum(abs(float(v)-r.prediction_db) for v in observed)/len(observed)
                same(baseline,old.deterministic_slow_instant_mae_db);fresh['deterministic_slow_instant_mae_db']=baseline
                literal_rows.append(fresh);seconds+=len(observed);hours+=1
        if (file_index+1)%13==0:print(json.dumps({'files_checked':file_index+1,'hours_checked':hours,'max_crps_delta_db':maximum}),flush=True)
    summary=read(primary/'probability_summary.json');summary_checks=0
    cols=['combined_crps201_db','combined_crps401_db','oracle_fast_crps201_db','oracle_fast_crps401_db','deterministic_slow_instant_mae_db','combined_central90_coverage','combined_central98_coverage','combined_tail95_exceedance','combined_tail99_exceedance','oracle_central90_coverage','oracle_central98_coverage']
    for analysis in ['primary','all_finite_diagnostic']:
        for old in summary[analysis]:
            selected=[r for r in literal_rows if (analysis!='primary' or r['primary']) and (old['station']=='pooled' or r['station']==old['station']) and mask_event(r['event'],r['heavy_rain'],old['event'])]
            assert old['n_station_hours']==len(selected) and old['n_seconds']==sum(r['n_seconds'] for r in selected)
            assert old['n_dates']==len({r['hour_utc'].date() for r in selected})
            for col in cols:
                if old['weighting']=='hour_equal':value=avg([r[col] for r in selected])
                else:
                    per_day=defaultdict(list)
                    for r in selected:per_day[r['hour_utc'].date()].append(r[col])
                    value=avg([avg(v) for v in per_day.values()])
                same(value,old[col])
            summary_checks+=1
    assert hours==1853 and sum(r['primary'] for r in literal_rows)==1844
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    result={'status':'PASS_WEATHER_AND_FROZEN_MARGINAL_EVALUATION','weather_summary_rows_checked':weather_checks,'probability_summary_rows_checked':summary_checks,
       'literal_crps201_and401_hours':hours,'primary_hours':1844,'native_seconds_replayed':seconds,'max_absolute_crps_difference_db':maximum,
       'calibration_boolean_replay_all_hours':True,'all_hourly_cdf201_replayed':True,'source_sha_unchanged':True,'fit_calls':0,'new_downloads':0,
       'not_a_dynamic_or_physical_measurement_equivalence_qa':True,'verifier_sha256':sha(__file__),'elapsed_s':time.monotonic()-start,
       'input_outputs_sha256':{p.name:sha(p) for p in primary.iterdir() if p.is_file()}}
    with (OUT/'qa.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
