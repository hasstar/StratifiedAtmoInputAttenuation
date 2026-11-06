"""External native-second temporal scores using frozen generators only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, sys, time, traceback
import numpy as np
import pandas as pd
import netCDF4

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT=BASE/'dynamic_evaluation_001'
FAST=ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
CODE=ROOT/'Code/Fast_Fading_Experiments/Revision_002'
LAGS=(1,2,5,10,30,60,120,300,600,900,1800)
CANDIDATES=('iid','block_0256')
EVENTS=('overall','clear_dry','rain','heavy_rain_subset','dry_overcast','other','unknown')
MODULE_SHA={'run_dynamics.py':'c4cca85f4ac652b5aa050cb0dcdfffd122ead6caf16ffa59a15ddeda73df64ca',
 'freeze_generator.py':'85b99eb393836aa8f4fbefe6cfd737dadef6e5b04e3b6ffef40ed20c0f2c6c25',
 'fit_marginals.py':'22487203b1799558d8e7fef2b219baa29b3fa59090d0d8cd83321fe992c6f6e8',
 'window_diagnostics.py':'7d18d63e79c14c14f90e4aaa703d642fe4a290d6e5189f4ed22c1e94670ea70c'}
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def emit(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def forbidden(*args,**kwargs):raise RuntimeError('Forbidden fitting/original-output entry point')
def load_frozen():
    for p,h in MODULE_SHA.items():require(sha(CODE/p)==h,'Frozen module changed '+p)
    sys.path.insert(0,str(CODE))
    import freeze_generator as fg
    import run_dynamics as dyn
    import fit_marginals as fm
    import window_diagnostics as wd
    # Existing module imports only define functions/classes/constants. Deny mutators in memory.
    for mod in (fg,dyn,fm,wd):
        for name in list(vars(mod)):
            if (name in ('run','main','upstream','verify','core_design','read_train','write_json','worker_lock','estimate_phi')
                or name.startswith(('fit','freeze','atomic_'))):setattr(mod,name,forbidden)
    saved,models,pools=fg.load_package()
    require(saved['candidate']=='block_0256' and saved['block_length_s']==256,'Frozen candidate')
    libraries={e:dyn.block_library(pool,256) for e,pool in pools.items()}
    for e,lib in libraries.items():
        c=saved['components'][e]
        require(len(lib)==c['eligible_hours'] and sum(x['n_starts'] for x in lib)==c['legal_start_count'],'Library mismatch')
    require(tuple(dyn.LAGS)==LAGS and dyn.SEED==20261007,'Frozen lag/seed mismatch')
    return dyn,saved,models,libraries
def native(source):
    path=ROOT/'外部样例数据CEDA/GBS_20p7GHz'/source
    with netCDF4.Dataset('native.nc',memory=path.read_bytes()) as ds:
        ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];t=np.asarray(ds.variables['time'][:],float);a=np.asarray(v[:],float)
        require(float(v.integration_period)==1 and np.all(t==np.round(t)) and len(np.unique(t))==len(t),'Native time grid')
        good=np.isfinite(a)&(a!=v._FillValue)&(a>=v.valid_min)&(a<=v.valid_max)&np.isfinite(t)&(t>0)&(t<=86400)
        day=np.full(86400,np.nan);day[t[good].astype(int)-1]=a[good]
    return day.reshape(24,3600)
def matrix_increments(f,masks):
    means=np.full((len(f),len(LAGS)),np.nan);corr=means.copy()
    for j,(lag,mask) in enumerate(zip(LAGS,masks)):
        if not mask.any():continue
        a,b=f[:,:-lag][:,mask],f[:,lag:][:,mask]
        means[:,j]=np.abs(b-a).mean(axis=1)
        a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
        den=np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1));ok=(den>0)&(mask.sum()>=3)
        corr[ok,j]=np.sum(a[ok]*b[ok],axis=1)/den[ok]
    return means,corr
def summarize(frame):
    rows=[];cols=['vs32','vs16','plugin32','plugin16']
    for subset in ('primary','all_finite_diagnostic'):
        base=frame[frame.primary] if subset=='primary' else frame
        for station in ('pooled','chilbolton','sparsholt'):
            site=base if station=='pooled' else base[base.station.eq(station)]
            for event in EVENTS:
                g=site if event=='overall' else site[site.heavy_rain] if event=='heavy_rain_subset' else site[site.event.eq(event)]
                for candidate in CANDIDATES:
                    c=g[g.candidate.eq(candidate)]
                    for weighting in ('hour_equal','date_equal'):
                        a=c[cols] if weighting=='hour_equal' else c.groupby('date')[cols].mean()
                        rows.append(dict(subset=subset,station=station,event=event,candidate=candidate,weighting=weighting,
                            n_station_hours=len(c),n_dates=int(c.date.nunique()),n_scoreable_hours=int(c.vs32.notna().sum()),
                            **{k:None if a[k].notna().sum()==0 else float(a[k].mean()) for k in cols}))
    return rows
def controls(dyn):
    valid=np.ones(3600,bool);valid[2]=False;m=dyn.pair_masks(valid)
    require(not m[1][0] and not m[2][0] and m[0][3],'Gap crossing control')
    f=np.arange(3600,dtype=float);full=dyn.pair_masks(np.ones(3600,bool));o=dyn.increments(f,full)
    require(np.array_equal(o[0],np.array(LAGS)),'Known increments')
    require(np.allclose(dyn.increments(f+21.3,full)[0],o[0],rtol=0,atol=1e-12),'Constant cancellation')
    g=np.tile(o[0],(32,1));score=dyn.score_from_moments(o[0],o[1],g)
    require(abs(score['vs'])<1e-10,'Deterministic variogram')
    got,_=matrix_increments(np.array([f,f+2]),full)
    require(np.array_equal(got,np.tile(LAGS,(2,1))),'Vectorized increments')
    emit('controls.json',{'status':'PASS','count':5,'synthetic_controls_not_observations':True})
def main():
    start=time.monotonic();require((OUT/'PROTOCOL.md').exists(),'Missing frozen protocol')
    require(not (OUT/'protected_inputs.json').exists(),'Run already started; inspect before resumption')
    protected=read(BASE/'conditional_evaluation_001/protected_inputs.json')
    def pin(p,expect=None):
        h=sha(p);require(expect is None or h==expect,'SHA '+str(p));protected[str(p.relative_to(ROOT))]=h
    for p,h in list(protected.items()):require(sha(ROOT/p)==h,'Old protected input changed')
    pin(Path(__file__));pin(OUT/'PROTOCOL.md')
    qa=BASE/'conditional_evaluation_independent_qa_001/qa.json';pin(qa)
    require(read(qa)['status']=='PASS_WEATHER_AND_FROZEN_MARGINAL_EVALUATION','Independent gate')
    for p,h in MODULE_SHA.items():pin(CODE/p,h)
    dyn,saved,models,libraries=load_frozen();controls(dyn)
    for c in saved['components'].values():pin(FAST/'final_generator'/c['library_file'],c['library_sha256'])
    rows=pd.read_parquet(BASE/'evaluation_002/hourly_predictions.parquet').merge(pd.read_parquet(BASE/'conditional_fast_001/weather_labels.parquet'),on=['station','hour_utc'],validate='one_to_one')
    selected=rows[rows.finite_target_diagnostic_eligible&rows.weather_known].copy()
    require(len(selected)==1853 and selected.primary_score_eligible.sum()==1844,'Scope')
    rows.loc[~rows.index.isin(selected.index)].to_parquet(OUT/'unscored_hours.parquet',index=False)
    emit('protected_inputs.json',protected);(OUT/'days').mkdir(exist_ok=False)
    scored=[];shards=[];number=0
    for source,group in selected.groupby('source_file',sort=True):
        day=native(source);group=group.sort_values('hour_utc');obs1=[];obs2=[];obsa=[];counts=[];gm=[];ga=[];metas=[];meta=[]
        for r in group.itertuples(index=False):
            y=day[r.hour_utc.hour];valid=np.isfinite(y);require(valid.sum()==r.valid_seconds,'Quality count')
            f=y-r.truth_db;masks=dyn.pair_masks(valid);m1,m2,acf=dyn.increments(f,masks)
            require(np.isfinite(m1).any(),'No lag available; preserve failure, not fill')
            key='GBS-external-v010-'+r.station+'-'+r.hour_utc.isoformat()
            qs=[];hourmeta=[]
            for c in CANDIDATES:
                seq=[];cm=[]
                for rep in range(32):
                    q,item=dyn.generate(c,models[r.event],libraries[r.event] if c=='block_0256' else None,-1,key,rep)
                    seq.append(q);cm.append(item)
                qs.append(np.array(seq));hourmeta.append(cm)
            require(np.array_equal(np.sort(qs[0],axis=1),np.sort(qs[1],axis=1)),'Common marginal mismatch')
            hg=[];ha=[]
            for c,q in zip(CANDIDATES,qs):
                g,a=matrix_increments(-10*np.log10(q),masks);hg.append(g);ha.append(a)
                s32=dyn.score_from_moments(m1,m2,g);s16=dyn.score_from_moments(m1,m2,g[:16])
                scored.append(dict(station=r.station,hour_utc=r.hour_utc,date=r.hour_utc.strftime('%Y-%m-%d'),event=r.event,
                    heavy_rain=bool(r.heavy_rain),primary=bool(r.primary_score_eligible),candidate=c,n_seconds=int(valid.sum()),
                    n_lags=s32['lags_used'],vs32=s32['vs'],vs16=s16['vs'],plugin32=s32['vs_plugin'],plugin16=s16['vs_plugin']))
            obs1.append(m1);obs2.append(m2);obsa.append(acf);counts.append([int(m.sum()) for m in masks]);gm.append(hg);ga.append(ha);metas.append(hourmeta)
            meta.append(dict(station=r.station,hour_utc=r.hour_utc.isoformat(),source_file=source,event=r.event,key=key))
            number+=1
        shard=OUT/'days'/(group.iloc[0].station+'_'+group.iloc[0].hour_utc.strftime('%Y%m%d')+'.npz')
        with shard.open('xb') as out:np.savez_compressed(out,observed_mean=obs1,observed_second=obs2,observed_acf=obsa,pair_counts=counts,
            generated_mean=gm,generated_acf=ga,metadata=np.array(json.dumps(meta)),generation_metadata=np.array(json.dumps(metas)))
        shards.append({'file':str(shard.relative_to(OUT)),'sha256':sha(shard),'hours':len(group)})
        print(json.dumps({'station_hours_finished':number,'total':1853,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    frame=pd.DataFrame(scored);require(len(frame)==3706,'Output row count');frame.to_parquet(OUT/'hourly.parquet',index=False)
    emit('summary.json',{'status':'COMPUTED_PENDING_INDEPENDENT_QA','score_unit':'dB squared','rows':summarize(frame),
        'constant_slow_cancels':True,'new_fit_calls':0,'new_downloads':0,'new_slow_predictions':0,'not_deterministic_seconds_forecast':True})
    emit('shards.json',shards)
    require(all(sha(ROOT/p)==h for p,h in protected.items()),'Protected source changed')
    emit('completion.json',{'status':'COMPUTED_PENDING_INDEPENDENT_QA','completed_utc':datetime.now(timezone.utc).isoformat(),
        'station_hours':1853,'primary_station_hours':1844,'simulated_realizations':1853*32*2,'elapsed_s':time.monotonic()-start,
        'common_marginal_multiset_checks':1853*32,'source_sha_unchanged':True,'new_fit_calls':0,'new_downloads':0,
        'outputs_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'status':'COMPUTED_PENDING_INDEPENDENT_QA','overall_primary_date_equal':[x for x in summarize(frame) if x['subset']=='primary' and x['station']=='pooled' and x['event']=='overall' and x['weighting']=='date_equal']}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():emit('failure.json',{'error':str(e),'traceback':traceback.format_exc()})
        raise
