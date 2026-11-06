"""Independent energy-form score replay. Does not import common_fast.py."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
from bisect import bisect_left
import json,math,hashlib,os,msvcrt,traceback
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import netCDF4
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_017'
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_012'
FAST=ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def same(a,b,tol=2e-10):
    assert abs(float(a)-float(b))<tol,(a,b)
def energy(y,w,q):
    x=np.sort(q);n=len(x);cs=np.r_[0.,np.cumsum(x)];k=np.searchsorted(x,y,side='right')
    distance=((2*k-n)*y+cs[-1]-2*cs[k])/n
    spread=math.fsum((2*j+1-n)*float(v) for j,v in enumerate(x))/(n*n)
    return float(np.dot(w,distance)/math.fsum(w))-spread
def independent_tests():
    y=np.array([-5.,0.,0.,4.]);w=np.array([1.,3.,2.,4.]);x=np.array([-2.,-2.,0.,5.,5.])
    brute=math.fsum(float(wi)*math.fsum(abs(float(yi)-float(xx)) for xx in x)/len(x) for yi,wi in zip(y,w))/sum(w)
    brute-=math.fsum(abs(float(a)-float(b)) for a in x for b in x)/(2*len(x)**2)
    same(energy(y,w,x),brute,1e-12);same(energy(y+9,w,x+9),brute,1e-12)
    same(energy(y,w,np.zeros(201)),np.average(abs(y),weights=w),1e-12)
    return 3
def main():
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            c=read(OUT/'calculation_complete.json');assert c['status']=='COMPUTED_PENDING_INDEPENDENT_QA'
            assert all(sha(OUT/p)==h for p,h in c['output_sha256'].items())
            protected=read(OUT/'sources_before.json');assert all(sha(p)==h for p,h in protected.items())
            tests=independent_tests();package=read(FAST/'final_generator/model.json');savedq=read(OUT/'quantiles.json');models={}
            for e in ['rain','dry_overcast']:
                comp=package['components'][e]
                with np.load(FAST/comp['cdf_file_relative_to_revision_output']) as a:
                    q=a['q'];cdf=a['cdf'];m={}
                    for n in [201,401]:m[str(n)]=np.array([-10*math.log10(float(q[bisect_left(cdf,1-(i+.5)/n)])) for i in range(n)])
                    m['bounds']=np.array([-10*math.log10(float(q[bisect_left(cdf,1-p)])) for p in [.01,.05,.5,.95,.99]])
                for k,v in m.items():assert np.max(abs(v-np.array(savedq[e][k])))<1e-13
                models[e]=m
            local=pd.read_parquet(BASE/'selected_local.parquet');uk=pd.read_parquet(BASE/'selected_uk.parquet')
            newledger=read(OUT/'all_hour_ledger.json');oldledger=read(BASE/'all_hour_ledger.json');assert len(newledger)==len(oldledger)==6351
            for a,b in zip(newledger,oldledger):
                assert all(a[k]==v for k,v in b.items())
                should=b['common_score_eligible'] and b['event'] in ['rain','dry_overcast']
                assert a['hybrid_score_eligible']==should
                assert a['hybrid_exclusion_reason']==(b['exclusion_reason'] if not b['common_score_eligible'] else '' if should else 'not_requested_weather')
            expected={(r.station,r.time) for f in [local,uk] for r in f.itertuples() if r.common_score_eligible and r.event in ['rain','dry_overcast']}
            assert len(expected)==3236
            scores=pd.read_parquet(OUT/'hourly_scores.parquet');assert len(scores)==25888 and not scores.duplicated(['station','time','run']).any()
            assert set(zip(scores.station,scores.time))==expected
            byhour={(s,t):g.set_index('run') for (s,t),g in scores.groupby(['station','time'])}
            audit={(r['station'],r['time']):r for r in read(OUT/'native_hour_audit.json')};assert set(audit)==expected
            runs=['ours_frozen_top10']+[f'bai2022_lstm_t0_adapted_seed{s}' for s in [17,29,43]]+[f'tabm2025_official_budgeted_seed{s}' for s in [17,29,43]]+['itu_combined_geometric_approx']
            families={r:('ours' if r==runs[0] else 'bai_lstm' if r.startswith('bai') else 'tabm' if r.startswith('tabm') else 'itu') for r in runs}
            fresh=[];seen=set();maxdiff=0.;maxnative=0.;seconds=0
            def replay(row,y,w,source):
                nonlocal maxdiff,seconds
                key=(row['station'],row['time']);assert key in expected and key not in seen;seen.add(key)
                old=byhour[key];assert set(old.index)==set(runs)
                weight=math.fsum(w);assert audit[key]['source']==source and audit[key]['n_seconds']==len(y) and audit[key]['weight']==int(weight)
                seconds+=len(y);m=models[row['event']]
                for run in runs:
                    prior=old.loc[run];shift=float(row[run]);same(shift,prior.prediction_hour_db,1e-14)
                    assert prior.n_seconds==len(y) and prior.native_weight_total==int(weight) and prior.event==row['event'] and prior.family==families[run]
                    same(prior.truth_hour_db,row['truth_db'],1e-12)
                    v={f'crps{n}_db':energy(y,w,m[str(n)]+shift) for n in [201,401]}
                    b=m['bounds']+shift
                    for pct,i,j in [(90,1,3),(98,0,4)]:v[f'central{pct}_coverage']=math.fsum(w[(y>=b[i])&(y<=b[j])])/weight
                    for pct,j in [(95,3),(99,4)]:v[f'tail{pct}_exceedance']=math.fsum(w[y>b[j]])/weight
                    v['slow_only_crps_db']=math.fsum(float(wi)*abs(float(yi)-shift) for yi,wi in zip(y,w))/weight
                    for k,value in v.items():same(value,prior[k]);maxdiff=max(maxdiff,abs(value-prior[k]))
                    fresh.append({'station':row['station'],'time':row['time'],'date':row['date'],'event':row['event'],'run':run,'family':families[run],'n_seconds':len(y),**v})
            wanted=local[local.common_score_eligible&local.event.isin(['rain','dry_overcast'])].set_index('hour_bjt')
            fm=read(FAST/'final_evaluation/manifest.json')
            for name in sorted(fm['frozen_design']['source_clean_1s']):
                path=ROOT/'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s'/name
                f=pd.read_parquet(path,filters=[('split','==','test')]);f=f[f.hour_bjt.isin(wanted.index)]
                for h,g in f.groupby('hour_bjt'):
                    row=wanted.loc[h];assert g.second_bjt.nunique()==len(g) and (g.second_bjt.dt.floor('h')==h).all()
                    assert g.hour_mean_power_w.nunique()==1 and g.weather_stratum.nunique()==1
                    p=g.received_power_w.to_numpy(float);counts=g.quality_zero_records.to_numpy(int)
                    keep=(counts>=9)&np.isfinite(p)&(p>0);assert np.array_equal(keep,g.primary_usable_1s)
                    event='rain' if bool(g.rain.iloc[0]) else 'dry_overcast' if bool(g.dry_overcast.iloc[0]) else 'other';assert event==row.event
                    truth=-76.19362933508154-10*math.log10(float(g.hour_mean_power_w.iloc[0]))-30
                    same(truth,row.truth_db,1e-8);maxnative=max(maxnative,abs(truth-row.truth_db))
                    y=-76.19362933508154-10*np.log10(p[keep])-30
                    replay(row,y,counts[keep].astype(float),str(path.relative_to(ROOT)))
            raw=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet').set_index(['station','hour_utc'])
            us=uk[uk.common_score_eligible&uk.event.isin(['rain','dry_overcast'])].copy()
            us['source_file']=[raw.loc[(r.station,r.hour_utc),'source_file'] for r in us.itertuples()]
            for i,(rel,g) in enumerate(us.groupby('source_file',sort=True)):
                path=ROOT/'外部样例数据CEDA/GBS_20p7GHz'/rel
                with netCDF4.Dataset('replay.nc',memory=path.read_bytes()) as ds:
                    ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];tv=ds.variables['time'];t=np.asarray(tv[:],float);a=np.asarray(v[:],float)
                    assert v.units=='dB' and float(v.integration_period)==1 and 'end of integration' in tv.long_name.lower()
                    day=pd.Timestamp(g.hour_utc.iloc[0]).normalize()
                    origin=tv.units.split(' since ')[1].split(' ')[0];assert pd.Timestamp(origin).date()==day.date()
                    valid=np.isfinite(t)&(t>=1)&(t<=86400)&np.isfinite(a)&(a!=v._FillValue)&(a>=v.valid_min)&(a<=v.valid_max)
                    assert (t[valid]==np.floor(t[valid])).all() and len(set(t[valid]))==sum(valid)
                    bins=np.floor((t-1)/3600)
                    for _,row in g.iterrows():
                        y=a[valid&(bins==row.hour_utc.hour)];oldraw=raw.loc[(row.station,row.hour_utc)]
                        assert len(y)==oldraw.valid_seconds and len(y)>=3420
                        truth=-10*math.log10(math.fsum(math.exp(-float(z)*math.log(10)/10) for z in y)/len(y))
                        same(truth,row.truth_db,1e-10);maxnative=max(maxnative,abs(truth-row.truth_db))
                        replay(row,y,np.ones(len(y)),str(path.relative_to(ROOT)))
                if (i+1)%40==0:print(json.dumps({'independent_files':i+1,'hours':len(seen),'max_score_delta':maxdiff}),flush=True)
            assert seen==expected and seconds==c['native_seconds']
            metrics=['crps201_db','crps401_db','central90_coverage','central98_coverage','tail95_exceedance','tail99_exceedance','slow_only_crps_db']
            checked_runs=0
            for r in read(OUT/'run_metrics.json'):
                sub=[x for x in fresh if x['event']==r['event'] and x['run']==r['run'] and (x['station']!='LOCAL' if r['cohort']=='UK' else x['station']==r['cohort'])]
                assert len(sub)==r['n_station_hours'] and len({x['date'] for x in sub})==r['n_dates'] and sum(x['n_seconds'] for x in sub)==r['n_seconds']
                assert min(x['date'] for x in sub)==r['date_min'] and max(x['date'] for x in sub)==r['date_max']
                for m in metrics:
                    if r['weighting']=='hour_equal':v=math.fsum(x[m] for x in sub)/len(sub)
                    else:
                        days=defaultdict(list)
                        for x in sub:days[x['date']].append(x[m])
                        v=math.fsum(math.fsum(xs)/len(xs) for xs in days.values())/len(days)
                    same(v,r[m]);maxdiff=max(maxdiff,abs(v-r[m]))
                checked_runs+=1
            runstats=read(OUT/'run_metrics.json');famstats=read(OUT/'family_metrics.json')
            for r in famstats:
                sub=[x for x in runstats if all(x[k]==r[k] for k in ['cohort','event','weighting','family'])]
                assert len(sub)==r['n_frozen_runs']
                for m in metrics:
                    mean=math.fsum(x[m] for x in sub)/len(sub);same(mean,r[m])
                    if len(sub)>1:same(math.sqrt(math.fsum((x[m]-mean)**2 for x in sub)/(len(sub)-1)),r[m+'_sd'])
                    else:assert r[m+'_sd'] is None
                peers=[x for x in famstats if all(x[k]==r[k] for k in ['cohort','event','weighting'])]
                assert r['rank_crps201']==1+sum(x['crps201_db']<r['crps201_db'] for x in peers)
            assert all(sha(p)==h for p,h in protected.items())
            emit('independent_qa.json',{'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'implementation':'energy form, independent native readers, boolean coverage, math.fsum aggregates; same agent not external reviewer',
             'tests':tests,'station_hours_replayed':len(seen),'run_hours_replayed':len(fresh),'seconds_replayed':seconds,'run_summaries':checked_runs,'family_summaries':len(famstats),'ledger_rows':len(newledger),
             'max_score_delta':maxdiff,'max_native_target_delta_db':maxnative,'source_sha_unchanged':True,'protected_paths':len(protected),'full_seven_complete':False,
             'verified_outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['execution.lock'] and p.suffix!='.log'}})
        except BaseException:
            emit('verification_failure.json',{'traceback':traceback.format_exc()});raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    print('INDEPENDENT_QA_PASS',flush=True)
if __name__=='__main__':main()
