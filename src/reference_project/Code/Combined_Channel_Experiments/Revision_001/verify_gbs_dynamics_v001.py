"""Independent native temporal-score arithmetic and frozen generation replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,time,traceback
import numpy as np
import pandas as pd
import netCDF4
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
SRC=BASE/'dynamic_evaluation_002';OUT=BASE/'dynamic_independent_qa_001'
FAST=ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
LAGS=(1,2,5,10,30,60,120,300,600,900,1800)
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def emit(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def equal(a,b,tol=1e-10):return bool(np.allclose(a,b,rtol=0,atol=tol,equal_nan=True))
def pairs(valid):
    # Independent contiguous-run enumeration; no cumulative bad-mask implementation.
    ix=np.flatnonzero(valid);parts=np.split(ix,np.where(np.diff(ix)!=1)[0]+1)
    return [np.concatenate([run[:-lag] for run in parts if len(run)>lag]) if any(len(run)>lag for run in parts) else np.array([],int) for lag in LAGS]
def means(f,indices):
    m1=[];m2=[];acf=[]
    for lag,start in zip(LAGS,indices):
        if len(start)==0:m1.append(np.nan);m2.append(np.nan);acf.append(np.nan);continue
        a,b=f[start],f[start+lag];d=np.abs(b-a)
        m1.append(math.fsum(map(float,d))/len(d));m2.append(math.fsum(map(float,d*d))/len(d))
        ac=a-float(a.mean());bc=b-float(b.mean());den=math.sqrt(float(ac@ac)*float(bc@bc))
        acf.append(float(ac@bc)/den if den>0 and len(a)>=3 else np.nan)
    return np.array(m1),np.array(m2),np.array(acf)
def score(m1,m2,gen):
    r=len(gen);use=np.isfinite(m1)&np.isfinite(m2);g=gen[:,use].T;need(np.isfinite(g).all(),'Generated moments')
    cross=(g[:,:,None]*g[:,None,:])[:,~np.eye(r,dtype=bool)].mean(axis=1)
    avg=np.array([math.fsum(map(float,v))/r for v in g]);v=m2[use]-2*m1[use]*avg+cross
    p=m2[use]-2*m1[use]*avg+avg*avg
    return float(v.mean()),float(p.mean())
def load_day(path):
    with netCDF4.Dataset('raw',memory=path.read_bytes()) as ds:
        ds.set_auto_mask(False);tv=ds['time'];v=ds['clearsky_attenuation'];t=tv[:].astype(float);a=v[:].astype(float)
        need(float(v.integration_period)==1,'Native second scale');fill=float(tv._FillValue)
        validtime=np.isfinite(t)&(t!=fill)&(t>0)&(t<=86400)
        need(np.equal(t[validtime],np.floor(t[validtime])).all() and len(set(t[validtime]))==validtime.sum(),'Duplicate/noninteger valid times')
        valid=validtime&np.isfinite(a)&(a>=float(v.valid_min))&(a<=float(v.valid_max))&(a!=float(v._FillValue))
        dense=np.full(86400,np.nan);dense[t[valid].astype(int)-1]=a[valid]
    return dense.reshape(24,3600),int((~validtime).sum())
def stable_seed(*parts):
    b='|'.join(map(str,(20261007,)+parts)).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little')%(2**63)
def libraries():
    package=read(FAST/'final_generator/model.json');models={};pools={}
    for event,c in package['components'].items():
        with np.load(FAST/c['cdf_file_relative_to_revision_output']) as f:models[event]=(f['q'],f['cdf'])
        with np.load(FAST/'final_generator'/c['library_file']) as f:
            entries=[]
            for h,q,n in zip(f['hours'],f['q'],f['records']):
                # Literal 256-second sliding window against frozen LOCAL quality.
                start=np.flatnonzero(np.lib.stride_tricks.sliding_window_view(n>=9,256).all(axis=1))
                if len(start):entries.append((str(h),q.copy(),start))
            need(len(entries)==c['eligible_hours'] and sum(len(x[2]) for x in entries)==c['legal_start_count'],'Independent library enumeration')
            pools[event]=entries
    return models,pools
def generate(event,key,rep,candidate,models,pools):
    q,cdf=models[event];u=np.random.default_rng(stable_seed('marginal',-1,key,rep)).random(3600)
    values=q[np.minimum(np.searchsorted(cdf,u,side='left'),len(q)-1)]
    if candidate=='iid':return values
    rng=np.random.default_rng(stable_seed('dependence',candidate,-1,key,rep));template=np.empty(3600);lib=pools[event]
    for pos in range(0,3600,256):
        _,train,starts=lib[int(rng.integers(len(lib)))];begin=starts[int(rng.integers(len(starts)))];n=min(256,3600-pos)
        template[pos:pos+n]=train[begin:begin+n]
    order=np.lexsort((rng.random(3600),template));result=np.empty(3600);result[order]=np.sort(values)
    return np.roll(result,int(rng.integers(3600)))
def main():
    starttime=time.monotonic();OUT.mkdir(exist_ok=False)
    complete=read(SRC/'completion.json');need(complete['status']=='COMPUTED_PENDING_INDEPENDENT_QA','Computation incomplete')
    for p,h in complete['outputs_sha256'].items():need(sha(SRC/p)==h,'Completed output changed')
    protected=read(SRC/'protected_inputs.json')
    for p,h in protected.items():need(sha(ROOT/p)==h,'Protected input changed')
    source_evidence={x['source']:x['sha256'] for x in read(BASE/'evaluation_002/native_label_source_evidence.json')}
    predictions=pd.read_parquet(BASE/'evaluation_002/hourly_predictions.parquet').merge(pd.read_parquet(BASE/'conditional_fast_001/weather_labels.parquet'),on=['station','hour_utc'],validate='one_to_one')
    selected=predictions[predictions.finite_target_diagnostic_eligible&predictions.weather_known].sort_values(['station','hour_utc'])
    replay={(g.iloc[0].station,g.iloc[0].hour_utc.isoformat()) for _,g in selected.groupby(['station','event'],sort=True)}
    models,pools=libraries();hourly=pd.read_parquet(SRC/'hourly.parquet').set_index(['station','hour_utc','candidate'])
    expected=selected.set_index(['station','hour_utc']);need(len(hourly)==3706 and len(expected)==1853,'Scope rows')
    n_hours=n_pairs=n_replay=invalid_times=0;maxscore=0.;maxmoment=0.;replayed=[]
    test=np.ones(3600,bool);test[2]=False
    need(0 not in pairs(test)[1] and 3 in pairs(test)[0],'Independent gap control')
    g=np.array([[1.,2.],[3.,4.],[5.,6.]])
    explicit=np.mean(np.array([14.,21.])-2*np.array([2.,3.])*g.mean(axis=0)+np.array([sum(g[i,k]*g[j,k] for i in range(3) for j in range(3) if i!=j)/6 for k in range(2)]))
    need(abs(score(np.array([2.,3.]),np.array([14.,21.]),g)[0]-explicit)<1e-12,'Cross realization control')
    for item in read(SRC/'shards.json'):
        path=SRC/item['file'];need(sha(path)==item['sha256'],'Shard SHA')
        with np.load(path) as z:
            meta=json.loads(str(z['metadata']));gm=z['generated_mean'];ga=z['generated_acf']
            raw=ROOT/'外部样例数据CEDA/GBS_20p7GHz'/meta[0]['source_file'];need(sha(raw)==source_evidence[str(raw.relative_to(ROOT))],'Raw SHA')
            day,ninvalid=load_day(raw);invalid_times+=ninvalid
            for i,m in enumerate(meta):
                ts=pd.Timestamp(m['hour_utc']);r=expected.loc[(m['station'],ts)];y=day[ts.hour];idx=pairs(np.isfinite(y));a,b,acf=means(y,idx)
                need(np.isfinite(y).sum()==r.valid_seconds,'Native seconds changed')
                need(equal(a,z['observed_mean'][i]) and equal(b,z['observed_second'][i]) and equal(acf,z['observed_acf'][i],1e-8),'Independent observed moments')
                need(np.array_equal([len(x) for x in idx],z['pair_counts'][i]),'Gap-safe pairs')
                n_pairs+=sum(map(len,idx));n_hours+=1
                for j,c in enumerate(('iid','block_0256')):
                    row=hourly.loc[(m['station'],ts,c)]
                    need(row.event==r.event and row.primary==r.primary_score_eligible and row.heavy_rain==r.heavy_rain,'Hourly metadata')
                    for count in (16,32):
                        vs,plugin=score(a,b,gm[i,j,:count]);delta=abs(vs-row[f'vs{count}']);maxscore=max(maxscore,delta)
                        need(delta<1e-9 and abs(plugin-row[f'plugin{count}'])<1e-9,'Independent score')
                    if (m['station'],m['hour_utc']) in replay:
                        for rep in range(32):
                            q=generate(m['event'],m['key'],rep,c,models,pools);im,_,ia=means(-10*np.log10(q),idx)
                            maxmoment=max(maxmoment,float(np.nanmax(np.abs(im-gm[i,j,rep]))))
                            need(equal(im,gm[i,j,rep],1e-10) and equal(ia,ga[i,j,rep],1e-8),'Independent generator replay')
                        n_replay+=32
                if (m['station'],m['hour_utc']) in replay:replayed.append(m)
        print(json.dumps({'verified_hours':n_hours,'total':1853}),flush=True)
    frame=hourly.reset_index();summaries=read(SRC/'summary.json')['rows']
    for row in summaries:
        g=frame[frame.candidate.eq(row['candidate'])]
        if row['subset']=='primary':g=g[g.primary]
        if row['station']!='pooled':g=g[g.station.eq(row['station'])]
        if row['event']=='heavy_rain_subset':g=g[g.heavy_rain]
        elif row['event']!='overall':g=g[g.event.eq(row['event'])]
        need(len(g)==row['n_station_hours'] and g.date.nunique()==row['n_dates'],'Summary denominators')
        for col in ('vs32','vs16','plugin32','plugin16'):
            vals=g[col].tolist() if row['weighting']=='hour_equal' else [math.fsum(x[col])/len(x) for _,x in g.groupby('date')]
            expected_value=None if not vals else math.fsum(vals)/len(vals)
            need((expected_value is None and row[col] is None) or (expected_value is not None and abs(row[col]-expected_value)<1e-9),'Summary arithmetic')
    need(n_hours==1853 and n_replay==len(replay)*64 and len(replayed)==len(replay),'QA coverage')
    for p,h in protected.items():need(sha(ROOT/p)==h,'Post-QA source changed')
    emit('qa.json',{'status':'PASS_EXTERNAL_FROZEN_TEMPORAL_EVALUATION','completed_utc':datetime.now(timezone.utc).isoformat(),
        'hours':n_hours,'primary_hours':1844,'literal_gap_safe_pairs':n_pairs,'invalid_time_records_retained_as_missing':invalid_times,
        'summary_rows':len(summaries),'independently_regenerated_hours':len(replayed),'independently_regenerated_realizations':n_replay,
        'replay_selection':'Earliest known finite hour per station and weather, fixed before scoring','replayed':replayed,
        'max_score_difference_db2':maxscore,'max_replayed_increment_difference_db':maxmoment,
        'new_fit_calls':0,'new_downloads':0,'old_sources_unchanged':True,'elapsed_s':time.monotonic()-starttime,
        'verifier_sha256':sha(Path(__file__)),'source_completion_sha256':sha(SRC/'completion.json'),
        'not_slow_prediction_gain':True,'not_measurement_equivalence_or_calibration_proof':True})
    print(json.dumps({'status':'PASS_EXTERNAL_FROZEN_TEMPORAL_EVALUATION','hours':n_hours,'max_score_difference_db2':maxscore,'independent_realizations':n_replay}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():emit('failure.json',{'error':str(e),'traceback':traceback.format_exc()})
        raise
