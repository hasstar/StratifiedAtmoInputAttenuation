"""Offline common-frozen-marginal experiment; no model/imported legacy entrypoint."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, math, os, sys, traceback, msvcrt
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import netCDF4
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_017'
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_012'
FAST=ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
RAW=ROOT/'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s'
GBS=ROOT/'外部样例数据CEDA/GBS_20p7GHz'
CLEAR=-76.19362933508154
FAMILIES={'ours':['ours_frozen_top10'],
 'bai_lstm':[f'bai2022_lstm_t0_adapted_seed{s}' for s in (17,29,43)],
 'tabm':[f'tabm2025_official_budgeted_seed{s}' for s in (17,29,43)],
 'itu':['itu_combined_geometric_approx']}
RUNS={r:f for f,rs in FAMILIES.items() for r in rs}
METRICS=['crps201_db','crps401_db','central90_coverage','central98_coverage',
         'tail95_exceedance','tail99_exceedance','slow_only_crps_db']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def need(x,s):
    if not x:raise ValueError(s)
def emit(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def pinball(y,w,t,p):
    need(len(y)>0 and y.shape==w.shape and np.isfinite(y).all() and np.isfinite(w).all() and (w>0).all(),'invalid observations/weights')
    need(np.isfinite(t).all() and np.isfinite(p).all() and ((p>0)&(p<1)).all(),'invalid quantiles')
    order=np.argsort(y,kind='stable');ys=y[order];ws=w[order]/w.sum()
    cw=np.r_[0.,np.cumsum(ws)];cwy=np.r_[0.,np.cumsum(ws*ys)];cw[-1]=1.
    i=np.searchsorted(ys,t,side='right')
    return p*(cwy[-1]-t)+t*cw[i]-cwy[i]
def score(y,w,model,shift):
    result={}
    for n in (201,401):
        v=2*pinball(y,w,model[n]+shift,(np.arange(n)+.5)/n).mean()
        need(v>=-1e-10,'negative CRPS');result[f'crps{n}_db']=float(v)
    b=model['bounds']+shift
    for pct,lo,hi in [(90,1,3),(98,0,4)]:result[f'central{pct}_coverage']=float(np.average((y>=b[lo])&(y<=b[hi]),weights=w))
    for pct,i in [(95,3),(99,4)]:result[f'tail{pct}_exceedance']=float(np.average(y>b[i],weights=w))
    result['slow_only_crps_db']=float(np.average(np.abs(y-shift),weights=w))
    return result
def controls():
    y=np.array([-2.,0.,0.,3.]);w=np.array([1.,2.,4.,3.]);p=(np.arange(201)+.5)/201
    t=np.linspace(-3.,4.,201);delta=y[:,None]-t
    literal=2*np.average(np.maximum(p*delta,(p-1)*delta),axis=0,weights=w).mean()
    need(abs(2*pinball(y,w,t,p).mean()-literal)<1e-12,'weighted literal')
    need(np.max(np.abs(pinball(y+7,w,t+7,p)-pinball(y,w,t,p)))<1e-12,'translation')
    need(abs(2*pinball(y,w,np.zeros(201),p).mean()-np.average(abs(y),weights=w))<1e-12,'constant distribution')
    for bady,badw in [(np.array([np.nan]),np.ones(1)),(np.ones(1),np.zeros(1)),(np.ones(1),-np.ones(1))]:
        try:pinball(bady,badw,t,p)
        except ValueError:pass
        else:raise AssertionError('negative control accepted')
    m={201:np.zeros(201),401:np.zeros(401),'bounds':np.array([-2.,-2.,0.,3.,3.])}
    need(score(y,w,m,0)['central90_coverage']==1,'inclusive atoms')
    emit('tests.json',{'status':'PASS','tests':7,'study_rows_used':0})
def sources():
    protected=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_016/sources_before.json')
    need(all(sha(p)==h for p,h in protected.items()),'old protection changed')
    def pin(p,expected=None):
        p=Path(p);h=sha(p);need(expected is None or h==expected,'source SHA '+str(p))
        need(str(p) not in protected or protected[str(p)]==h,'conflicting source SHA')
        protected[str(p)]=h
    for directory in [ROOT/'Outputs/Experiments/Combined_Channel_Revision_016',HERE]:
        for p in directory.iterdir():
            if p.is_file():pin(p)
    for p in [OUT/'launch.json',BASE/'selected_local.parquet',BASE/'selected_uk.parquet',BASE/'all_hour_ledger.json',BASE/'final_verification.json',
              ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet',ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json',
              FAST/'final_evaluation/manifest.json',FAST/'final_evaluation/qa.json',FAST/'final_generator/qa.json']:
        pin(p)
    old=read(FAST/'final_evaluation/manifest.json');qa=read(FAST/'final_evaluation/qa.json')
    need(qa['status']=='PASS' and sha(FAST/'final_evaluation/manifest.json')==qa['manifest_sha256'],'fast source QA')
    for name,h in old['frozen_design']['source_clean_1s'].items():pin(RAW/name,h)
    gqa=read(FAST/'final_generator/qa.json');need(gqa['status']=='PASS','generator QA')
    pin(FAST/'final_generator/model.json',gqa['model_sha256'])
    package=read(FAST/'final_generator/model.json')
    for c in package['components'].values():pin(FAST/c['cdf_file_relative_to_revision_output'],c['cdf_sha256'])
    for r in read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json'):pin(ROOT/r['source'],r['sha256'])
    emit('sources_before.json',protected)
    return protected,old,package
def quantiles(package):
    models={};qa={}
    for event in ('rain','dry_overcast'):
        c=package['components'][event]
        with np.load(FAST/c['cdf_file_relative_to_revision_output']) as a:q=a['q'];cdf=a['cdf']
        weights=np.diff(np.r_[0.,cdf]);moment=float(np.dot(weights,q))
        need(np.isfinite(q).all() and (q>0).all() and (np.diff(q)>=0).all() and (weights>0).all() and abs(cdf[-1]-1)<1e-12 and abs(moment-1)<1e-10,'frozen CDF integrity')
        m={n:-10*np.log10(q[np.searchsorted(cdf,1-(np.arange(n)+.5)/n,side='left')]) for n in (201,401)}
        m['bounds']=-10*np.log10(q[np.searchsorted(cdf,1-np.array([.01,.05,.5,.95,.99]),side='left')]);models[event]=m
        qa[event]={'source_sha256':c['cdf_sha256'],'support_size':len(q),'unit_power_moment':moment}
    emit('quantiles.json',{e:{str(k):v.tolist() for k,v in m.items()} for e,m in models.items()});emit('cdf_qa.json',qa)
    return models
def inputs():
    local=pd.read_parquet(BASE/'selected_local.parquet');uk=pd.read_parquet(BASE/'selected_uk.parquet')
    raw=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet')
    uk=uk.merge(raw[['station','hour_utc','source_file','valid_seconds','primary_score_eligible','truth_db']].rename(columns={'truth_db':'native_truth_db'}),on=['station','hour_utc'],validate='one_to_one')
    need(len(local)==687 and len(uk)==5664 and np.allclose(uk.truth_db,uk.native_truth_db,atol=0,rtol=0,equal_nan=True),'source cohort/target')
    for f in (local,uk):
        need(not f.duplicated(['station','time']).any(),'duplicate station-hour')
        f['hybrid_score_eligible']=f.common_score_eligible&f.event.isin(['rain','dry_overcast'])
        need(np.isfinite(f.loc[f.hybrid_score_eligible,list(RUNS)].to_numpy()).all(),'incomplete frozen predictions')
    need(local.hybrid_score_eligible.sum()==209 and uk.hybrid_score_eligible.sum()==3027,'fixed mask changed')
    ledger=read(BASE/'all_hour_ledger.json');by={(r.station,r.time):bool(r.hybrid_score_eligible) for f in (local,uk) for r in f.itertuples()}
    for r in ledger:
        r['hybrid_score_eligible']=by[(r['station'],r['time'])]
        r['hybrid_exclusion_reason']=r['exclusion_reason'] if not r['common_score_eligible'] else ('' if r['hybrid_score_eligible'] else 'not_requested_weather')
    emit('all_hour_ledger.json',ledger)
    return local,uk
def append_scores(out,row,y,w,model):
    for run,family in RUNS.items():
        shift=float(row[run]);out.append({'station':row['station'],'time':row['time'],'date':row['date'],'event':row['event'],
          'family':family,'run':run,'n_seconds':len(y),'native_weight_total':int(w.sum()),'prediction_hour_db':shift,'truth_hour_db':float(row['truth_db']),**score(y,w,model,shift)})
def process_local(local,old,models):
    wanted=local[local.hybrid_score_eligible].set_index('hour_bjt');out=[];seen=set();audit=[]
    for name in sorted(old['frozen_design']['source_clean_1s']):
        f=pd.read_parquet(RAW/name,filters=[('split','==','test')]);f=f[f.hour_bjt.isin(wanted.index)]
        for h,g in f.groupby('hour_bjt',sort=True):
            r=wanted.loc[h];need(h not in seen,'duplicate local source');seen.add(h)
            slots=(g.second_bjt-g.hour_bjt).dt.total_seconds().to_numpy()
            need(np.isfinite(slots).all() and (slots==np.floor(slots)).all() and ((slots>=0)&(slots<3600)).all() and len(set(slots))==len(slots),'local time slots')
            need(g.hour_mean_power_w.nunique()==1,'frozen hourly base')
            base=float(g.hour_mean_power_w.iloc[0]);truth=CLEAR-10*math.log10(1000*base)
            need(abs(truth-r.truth_db)<1e-8,'local frozen truth')
            event='rain' if bool(g.rain.iloc[0]) else 'dry_overcast' if bool(g.dry_overcast.iloc[0]) else 'other'
            need(event==r.event and g.rain.nunique()==1 and g.dry_overcast.nunique()==1,'local weather')
            p=g.received_power_w.to_numpy(float);cnt=g.quality_zero_records.to_numpy(float)
            good=np.isfinite(p)&(p>0)&(cnt>=9)
            need(good.any() and np.array_equal(g.primary_usable_1s.to_numpy(bool),good),'frozen local second gate')
            y=CLEAR-10*np.log10(1000*p[good]);w=cnt[good]
            need(np.allclose(y-truth,g.fast_residual_db.to_numpy(float)[good],rtol=0,atol=1e-10),'local residual identity')
            append_scores(out,r,y,w,models[event]);audit.append({'station':'LOCAL','time':r.time,'n_seconds':len(y),'weight':int(w.sum()),'source':str((RAW/name).relative_to(ROOT))})
        print(json.dumps({'stage':'local','file':name,'hours':len(seen)}),flush=True)
    need(len(seen)==209,'missing local score hours');return out,audit
def process_uk(uk,models):
    out=[];audit=[];f=uk[uk.hybrid_score_eligible]
    for i,(name,g) in enumerate(f.groupby('source_file',sort=True)):
        with netCDF4.Dataset('native.nc',memory=(GBS/name).read_bytes()) as ds:
            ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];tv=ds.variables['time']
            t=np.asarray(tv[:],float);a=np.asarray(v[:],float)
            need(v.units=='dB' and float(v.integration_period)==1 and 'end of integration' in tv.long_name.lower(),'GBS definition')
            need(abs(float(ds.variables['frequency'][...])-20.7)<1e-4,'GBS frequency')
            day=pd.Timestamp(g.hour_utc.iloc[0]).normalize()
            origin=f'seconds since {day.year}-{day.month}-{day.day} 0:0:0.0 +0:00'
            import re
            mm=re.match(r'seconds since (\d+)-(\d+)-(\d+) 0:0:0\.0+ \+0:00$',tv.units)
            need(mm is not None and tuple(map(int,mm.groups()))==(day.year,day.month,day.day),'GBS UTC origin')
            valid=np.isfinite(t)&(t>0)&(t<=86400)&np.isfinite(a)&(a!=v._FillValue)&(a>=v.valid_min)&(a<=v.valid_max)
            need((t[valid]==np.rint(t[valid])).all() and len(set(t[valid]))==int(valid.sum()),'GBS unique seconds')
            for _,r in g.iterrows():
                h=r.hour_utc.hour;y=a[valid&(t>3600*h)&(t<=3600*(h+1))]
                need(len(y)==r.valid_seconds and len(y)>=3420,'GBS valid coverage')
                truth=-10*np.log10(np.mean(10**(-y/10)));need(abs(truth-r.truth_db)<1e-10,'GBS frozen truth')
                append_scores(out,r,y,np.ones(len(y)),models[r.event]);audit.append({'station':r.station,'time':r.time,'n_seconds':len(y),'weight':len(y),'source':str((GBS/name).relative_to(ROOT))})
        if (i+1)%20==0:print(json.dumps({'stage':'uk','files':i+1,'hours':len(audit)}),flush=True)
    need(len(audit)==3027,'missing UK score hours');return out,audit
def summarize(frame):
    runs=[];families=[]
    for cohort in ['LOCAL','UK','chilbolton','sparsholt']:
        sub=frame[frame.station.ne('LOCAL') if cohort=='UK' else frame.station.eq(cohort)]
        for event in ['rain','dry_overcast']:
            part=sub[sub.event.eq(event)]
            for weighting in ['date_equal','hour_equal']:
                cell=[]
                for run,g in part.groupby('run',sort=True):
                    data=g.groupby('date')[METRICS].mean() if weighting=='date_equal' else g[METRICS]
                    r={'cohort':cohort,'event':event,'weighting':weighting,'run':run,'family':RUNS[run],
                     'n_station_hours':len(g),'n_dates':g.date.nunique(),'n_seconds':int(g.n_seconds.sum()),'date_min':g.date.min(),'date_max':g.date.max(),
                     **{m:float(data[m].mean()) for m in METRICS}}
                    runs.append(r);cell.append(r)
                for fam in FAMILIES:
                    rs=[r for r in cell if r['family']==fam];need(len(rs)==len(FAMILIES[fam]),'run completeness')
                    rec={k:rs[0][k] for k in ['cohort','event','weighting','n_station_hours','n_dates','n_seconds','date_min','date_max']}
                    rec.update(family=fam,n_frozen_runs=len(rs))
                    for m in METRICS:
                        a=np.array([r[m] for r in rs]);rec[m]=float(a.mean());rec[m+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
                    families.append(rec)
                group=families[-4:]
                for r in group:r['rank_crps201']=1+sum(x['crps201_db']<r['crps201_db'] for x in group)
    return runs,families
def main():
    with (OUT/'execution.lock').open('xb') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            emit('started.json',{'utc':now(),'pid':os.getpid(),'command':sys.argv,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'netCDF4':netCDF4.__version__})
            controls();protected,old,package=sources();models=quantiles(package);local,uk=inputs()
            l,la=process_local(local,old,models);u,ua=process_uk(uk,models);frame=pd.DataFrame(l+u)
            need(len(frame)==3236*8 and not frame.duplicated(['station','time','run']).any(),'score count/uniqueness')
            need(np.isfinite(frame[METRICS].to_numpy()).all(),'nonfinite scores')
            frame.to_parquet(OUT/'hourly_scores.parquet',index=False);emit('native_hour_audit.json',la+ua)
            runs,families=summarize(frame);emit('run_metrics.json',runs);emit('family_metrics.json',families)
            need(all(sha(p)==h for p,h in protected.items()),'source changed during computation')
            emit('calculation_complete.json',{'status':'COMPUTED_PENDING_INDEPENDENT_QA','utc':now(),'station_hours':3236,'run_hour_rows':len(frame),'native_seconds':sum(r['n_seconds'] for r in la+ua),
               'new_training':0,'new_hourly_predictions':0,'new_cdf_fits':0,'new_reference_fits':0,'weather_requests':0,'stochastic_realizations':0,'full_seven_complete':False,
               'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['execution.lock'] and p.suffix!='.log'}})
        except BaseException:
            emit('failure.json',{'utc':now(),'traceback':traceback.format_exc()});raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            emit('calculation_lock_release.json',{'utc':now(),'status':'RELEASED'})
    print('COMPUTED; independent QA required',flush=True)
if __name__=='__main__':main()
