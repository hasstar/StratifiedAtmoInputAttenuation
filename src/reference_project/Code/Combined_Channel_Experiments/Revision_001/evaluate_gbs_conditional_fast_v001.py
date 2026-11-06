"""Frozen external marginal probabilities and weather groups; no training imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, traceback
import netCDF4
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT=BASE/'conditional_evaluation_001'
FAST=ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
EVENTS=['overall','clear_dry','rain','heavy_rain_subset','dry_overcast','other','unknown']

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def emit(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def qf(q,cdf,p):return -10*np.log10(q[np.searchsorted(cdf,1-p,side='left')])
def pinball_mean(y,threshold,p):
    require(np.isfinite(y).all() and len(y)>0,'Invalid native score array')
    ys=np.sort(y);cs=np.r_[0.,np.cumsum(ys)];pos=np.searchsorted(ys,threshold,side='right')
    return p*(cs[-1]/len(y)-threshold)+threshold*pos/len(y)-cs[pos]/len(y)
def group_mask(df,event):
    return np.ones(len(df),bool) if event=='overall' else df.heavy_rain if event=='heavy_rain_subset' else df.event.eq(event)
def slow_metrics(frame):
    rows=[]
    for site in ['pooled','chilbolton','sparsholt']:
        s=frame if site=='pooled' else frame[frame.station.eq(site)]
        for event in EVENTS:
            g=s[group_mask(s,event)];e=(g.prediction_db-g.truth_db).to_numpy(float)
            rows.append({'station':site,'event':event,'n_station_hours':len(g),'n_dates':int(g.hour_utc.dt.date.nunique()),
                'n_unique_utc_hours':int(g.hour_utc.nunique()),'rmse_db':float(np.sqrt(np.mean(e*e))) if len(e) else None,
                'mae_db':float(np.mean(np.abs(e))) if len(e) else None,'bias_db':float(np.mean(e)) if len(e) else None})
    return rows
def summary(frame):
    cols=['combined_crps201_db','combined_crps401_db','oracle_fast_crps201_db','oracle_fast_crps401_db',
          'deterministic_slow_instant_mae_db','combined_central90_coverage','combined_central98_coverage',
          'combined_tail95_exceedance','combined_tail99_exceedance','oracle_central90_coverage','oracle_central98_coverage']
    rows=[]
    for site in ['pooled','chilbolton','sparsholt']:
        s=frame if site=='pooled' else frame[frame.station.eq(site)]
        for event in EVENTS:
            g=s[group_mask(s,event)]
            for weighting in ['hour_equal','date_equal']:
                data=g[cols] if weighting=='hour_equal' else g.assign(date=g.hour_utc.dt.strftime('%Y-%m-%d')).groupby('date')[cols].mean()
                m={c:None if not len(data) else float(data[c].mean()) for c in cols}
                rows.append({'station':site,'event':event,'weighting':weighting,'n_station_hours':len(g),'n_dates':int(g.hour_utc.dt.date.nunique()),
                    'n_seconds':int(g.n_seconds.sum()),**m})
    return rows
def score(y,model,shift):
    out={}
    for n in [201,401]:
        p=(np.arange(n)+.5)/n;loss=pinball_mean(y,model[n]+shift,p)
        require(np.min(loss)>-1e-10,'Negative pinball numerical failure')
        out[f'crps{n}_db']=float(2*loss.mean())
    bounds=model['bounds']+shift;ys=np.sort(y)
    right=np.searchsorted(ys,bounds,side='right')/len(y);left=np.searchsorted(ys,bounds,side='left')/len(y)
    out.update(central90_coverage=float(right[3]-left[1]),central98_coverage=float(right[4]-left[0]),
               tail95_exceedance=float(1-right[3]),tail99_exceedance=float(1-right[4]),
               boundary_quantiles_db=bounds.tolist(),actual_cdf201=(np.searchsorted(ys,model[201]+shift,side='right')/len(y)).tolist())
    return out
def controls():
    y=np.array([-2.,0.,1.,4.]);p=(np.arange(201)+.5)/201;t=np.zeros(201)
    require(abs(2*pinball_mean(y,t,p).mean()-np.abs(y).mean())<1e-12,'Degenerate CRPS')
    direct=np.mean(np.maximum(p[None,:]*(y[:,None]-t[None,:]),(p[None,:]-1)*(y[:,None]-t[None,:])),axis=0)
    require(np.max(np.abs(pinball_mean(y,t,p)-direct))<1e-12,'Pinball literal check')
    require(np.max(np.abs(pinball_mean(y+5,t+5,p)-pinball_mean(y,t,p)))<1e-12,'Translation invariance')
    bounds=np.array([-1.,-1.,0.,1.,1.]);m={201:np.zeros(201),401:np.zeros(401),'bounds':bounds}
    require(score(np.array([-1.,0.,1.]),m,0)['central90_coverage']==1.,'Inclusive interval atoms')
    emit('controls.json',{'status':'PASS','control_count':4})
def main():
    OUT.mkdir(exist_ok=False);controls();protected={}
    def pin(p,expect=None):
        h=sha(p);require(expect is None or h==expect,'Input SHA mismatch: '+str(p));protected[str(Path(p).relative_to(ROOT))]=h;return Path(p)
    qa=read(pin(BASE/'conditional_fast_independent_qa_001/qa.json'))
    require(qa['status']=='PASS_WEATHER_AND_FROZEN_MARGINAL_TRANSFER_GATE','Independent compatibility gate')
    pin(BASE/'conditional_fast_001/PROTOCOL.md');pin(Path(__file__))
    pred=pd.read_parquet(pin(BASE/'evaluation_002/hourly_predictions.parquet','c58822f6db3eea41e706b632610fdb1fddded0c9b401f48e72fdbf16e15061d8'))
    weather=pd.read_parquet(pin(BASE/'conditional_fast_001/weather_labels.parquet',qa['weather_labels_sha256']))
    joined=pred.merge(weather,on=['station','hour_utc'],how='left',validate='one_to_one')
    require(len(joined)==1872 and joined.event.notna().all(),'Join scope')
    emit('weather_hourly_metrics.json',{'status':'PENDING_INDEPENDENT_EVALUATION_QA','primary':slow_metrics(joined[joined.primary_score_eligible]),
        'all_finite_diagnostic':slow_metrics(joined[joined.finite_target_diagnostic_eligible]),'definition':'Existing weather proxy thresholds; unknown separate; rain includes heavy subset; no new predictions'})
    joined.loc[~(joined.finite_target_diagnostic_eligible&joined.weather_known)].to_parquet(OUT/'unscored_probability_hours.parquet',index=False)
    package=read(pin(FAST/'final_generator/model.json','b524c268054992953b63197eccd76888fde4de2e12a3d9c1ac6d3295647be714'))
    models={}
    for event,c in package['components'].items():
        with np.load(pin(FAST/c['cdf_file_relative_to_revision_output'],c['cdf_sha256'])) as a:
            q,cdf=a['q'],a['cdf'];m={n:qf(q,cdf,(np.arange(n)+.5)/n) for n in [201,401]}
            m['bounds']=qf(q,cdf,np.array([.01,.05,.5,.95,.99]));fs=-10*np.log10(q[::-1]);fc=np.r_[0.,np.cumsum(np.diff(np.r_[0.,cdf])[::-1])];fc[-1]=1
            m['model_cdf201_left']=fc[np.searchsorted(fs,m[201],side='left')].tolist()
            m['model_cdf201_right']=fc[np.searchsorted(fs,m[201],side='right')].tolist();models[event]=m
    emit('frozen_quantiles.json',{e:{str(k):v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()} for e,m in models.items()})
    source_evidence={e['source']:e['sha256'] for e in read(pin(BASE/'evaluation_002/native_label_source_evidence.json'))}
    rows=[]
    for source,g in joined.groupby('source_file',sort=True):
        path=ROOT/'外部样例数据CEDA/GBS_20p7GHz'/source;pin(path,source_evidence[str(path.relative_to(ROOT))])
        with netCDF4.Dataset('native.nc',memory=path.read_bytes()) as ds:
            ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];t=np.asarray(ds.variables['time'][:],float);a=np.asarray(v[:],float)
            valid=np.isfinite(a)&(a!=v._FillValue)&(a>=v.valid_min)&(a<=v.valid_max)&np.isfinite(t)&(t>0)&(t<=86400)
            require(float(v.integration_period)==1,'Not native 1s')
            for r in g.itertuples(index=False):
                if not (r.finite_target_diagnostic_eligible and r.weather_known):continue
                h=r.hour_utc.hour;mask=valid&(t>h*3600)&(t<=(h+1)*3600);y=a[mask]
                require(len(y)==r.valid_seconds,'Native quality count changed')
                qobs=np.exp(-(y-r.truth_db)*math.log(10)/10)
                require(abs(float(qobs.mean())-1)<1e-10,'Frozen native hourly target mismatch')
                combined=score(y,models[r.event],r.prediction_db);oracle=score(y,models[r.event],r.truth_db)
                row={'station':r.station,'hour_utc':r.hour_utc,'event':r.event,'heavy_rain':r.heavy_rain,'primary':r.primary_score_eligible,
                    'n_seconds':len(y),'truth_hour_db':r.truth_db,'prediction_hour_db':r.prediction_db,'observed_power_ratio_mean':float(qobs.mean()),
                    'deterministic_slow_instant_mae_db':float(np.abs(y-r.prediction_db).mean())}
                row.update({'combined_'+k:v for k,v in combined.items()})
                row.update({'oracle_fast_'+k:v for k,v in oracle.items() if k.startswith('crps')})
                row.update({'oracle_'+k:v for k,v in oracle.items() if not k.startswith('crps')})
                rows.append(row)
    frame=pd.DataFrame(rows).sort_values(['station','hour_utc']).reset_index(drop=True)
    require(len(frame)==qa['all_finite_known_hours'] and frame.primary.sum()==qa['primary_known_hours'],'Score coverage differs from gate')
    frame.to_parquet(OUT/'probability_hourly.parquet',index=False)
    emit('probability_summary.json',{'status':'PENDING_INDEPENDENT_EVALUATION_QA','primary':summary(frame[frame.primary]),'all_finite_diagnostic':summary(frame),
        'primary_grouping':'Native >=95% valid seconds AND complete existing diagnostic weather',
        'not_local_measurement_equivalence':True,'frozen_weather_conditional_model_no_adaptation':True,
        'iid_block_same_marginal_scores':True,'dynamic_evaluation':'NOT_EVALUATED_HERE','new_fit_calls':0,'new_downloads':0})
    require(all(sha(ROOT/p)==h for p,h in protected.items()),'Protected files changed')
    emit('protected_inputs.json',protected)
    emit('completion.json',{'status':'PASS_PROBABILITY_COMPUTATION_PENDING_INDEPENDENT_QA','completed_utc':datetime.now(timezone.utc).isoformat(),
        'primary_station_hours':int(frame.primary.sum()),'all_finite_station_hours':len(frame),'primary_valid_seconds':int(frame.loc[frame.primary,'n_seconds'].sum()),
        'new_fit_calls':0,'new_predictions':0,'new_downloads':0,'new_reference_fits':0,'new_cdf_fits':0,'inputs_unchanged':True,
        'outputs_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'status':'PASS_PROBABILITY_COMPUTATION_PENDING_INDEPENDENT_QA','primary_hours':int(frame.primary.sum()),
        'primary_pooled_date_equal':next(r for r in summary(frame[frame.primary]) if r['station']=='pooled' and r['event']=='overall' and r['weighting']=='date_equal')}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():emit('failure.json',{'status':'FAIL_PRESERVED','error':str(e),'traceback':traceback.format_exc()})
        raise
