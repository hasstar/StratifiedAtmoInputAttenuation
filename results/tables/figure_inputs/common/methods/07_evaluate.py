"""T016 frozen ensemble evaluation. New outputs only; single resumable worker."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse, time, traceback, platform
import pandas as pd
import psutil
from common import *

def controls(dyn):
    rng=np.random.default_rng(3);x=rng.normal(size=(32,19));y=rng.normal(size=19)
    a,b=crps(x,y)
    direct=np.abs(x-y).mean(0)-np.abs(x[:,None,:]-x[None,:,:]).mean((0,1))/2
    need(np.max(np.abs(a-direct))<1e-12,'CRPS brute-force control')
    need(np.allclose(crps(x+5,y+5)[0],a,rtol=0,atol=1e-12),'CRPS translation')
    need(np.allclose(crps(x[::-1],y)[0],a,rtol=0,atol=1e-12),'CRPS permutation')
    need(np.allclose(crps(np.ones((32,19))*3,y)[0],np.abs(3-y),rtol=0,atol=1e-12),'Degenerate CRPS')
    v=np.ones(3600,dtype=bool);v[100]=False
    for lag,mask in zip(LAGS,dyn.pair_masks(v)):
        literal=np.array([v[i:i+lag+1].all() for i in range(3600-lag)])
        need(np.array_equal(mask,literal),'Full interval gap mask')
    return {'status':'PASS','crps_brute_force_max_error':float(np.max(np.abs(a-direct))),
            'degenerate_translation_permutation_gap_controls':True}

def prepare_inputs(saved):
    pins={}
    def pin(p,expected=None):
        h=sha(p);need(expected is None or h==expected,'Source SHA mismatch '+str(p));pins[relative(p)]=h
    for p in (CODE/'common.py',CODE/'evaluate.py',CODE/'PROTOCOL.md',SLOW/'evaluation.json',SLOW/'state.json',UK/'evaluation/all_hourly_predictions.parquet',UK/'inputs/uk_inputs_38.parquet',UK/'inputs/independent_input_qa.json',UK/'evaluation/independent_numeric_qa.json',WEATHER,FAST/'final_generator/model.json',FAST/'final_generator/qa.json'):
        pin(p)
    need(read(FAST/'final_generator/qa.json')['status']=='PASS','Generator QA')
    for p in (UK/'inputs/independent_input_qa.json',UK/'evaluation/independent_numeric_qa.json'):
        need(read(p)['status'].startswith('PASS'),'UK QA gate '+str(p))
    state=read(SLOW/'state.json');ev=read(SLOW/'evaluation.json')
    need(ev['state_sha256']==sha(SLOW/'state.json') and state['k']==10 and state['model']=='tabm','Slow state identity')
    pin(ROOT/state['full_fit']['model_file'],state['full_fit']['model_sha256'])
    for c in saved['components'].values():
        pin(FAST/c['cdf_file_relative_to_revision_output'],c['cdf_sha256'])
        pin(FAST/'final_generator'/c['library_file'],c['library_sha256'])
    for name in ('run_dynamics.py','freeze_generator.py','fit_marginals.py','window_diagnostics.py'):
        pin(ROOT/'Code/Fast_Fading_Experiments/Revision_002'/name)
    pin(ROOT/'Code/Combined_Channel_Experiments/Revision_001/evaluate_gbs_dynamics_v002.py')
    originals=ROOT/'最终提交的修改材料'
    for p in originals.iterdir():
        if p.is_file():pin(p)
    clean_paths=sorted((ROOT/'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s').glob('*.parquet'))
    for p in clean_paths:pin(p)
    u=pd.read_parquet(UK/'inputs/uk_inputs_38.parquet')
    pred=pd.read_parquet(UK/'evaluation/all_hourly_predictions.parquet',filters=[('model','==','tabm'),('k','==',10)])
    pred=pred[['station','utc_epoch_seconds','prediction_db','event','truth_db','primary_score_eligible']]
    u=u.merge(pred,on=['station','utc_epoch_seconds'],validate='one_to_one',suffixes=('','_prediction'))
    need(len(u)==5664 and u.primary_score_eligible.sum()==5593,'UK cohort')
    need(np.allclose(u.truth_db,u.truth_db_prediction,equal_nan=True,rtol=0,atol=1e-12),'UK label join')
    need(np.array_equal(u.primary_score_eligible,u.primary_score_eligible_prediction),'UK eligibility join')
    w=pd.read_parquet(WEATHER)[['station','utc_epoch_seconds','tcc','total_water_mm','snow_water_mm']]
    u=u.merge(w,on=['station','utc_epoch_seconds'],validate='one_to_one',suffixes=('','_weather'))
    need(np.isfinite(u[['tcc','total_water_mm','snow_water_mm']].to_numpy()).all(),'Missing weather')
    need(u.tcc.between(0,1).all(),'TCC range')
    need(np.array_equal(u.total_water_mm,u.total_water_mm_weather) and np.array_equal(u.snow_water_mm,u.snow_water_mm_weather),'Weather source join')
    p=u.total_water_mm;s=u.snow_water_mm;c=u.tcc
    u['generator_event']=np.select([(p>=.1)&(s<.1),(p<.1)&(s<.1)&(c>=.8),(p<.1)&(s<.1)&(c<=.2)],['rain','dry_overcast','clear_dry'],default='other')
    for event in ('rain','dry_overcast'):
        need(np.array_equal(u.event.eq(event),u.generator_event.eq(event)),'UK weather mask changed')
    sources_path=ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json';pin(sources_path)
    receipts=read(sources_path)
    known={x['source'].replace('\\','/').split('GBS_20p7GHz/')[-1]:x['sha256'] for x in receipts}
    need(len(known)==236 and len(u.source_file.unique())==236,'UK daily scope')
    for src in u.source_file.unique():
        src=src.replace('\\','/');need(src in known,'Native receipt missing');pin(GBS/src,known[src])
    return pins,clean_paths,ev,u

def iterate(clean_paths,ev,u,native):
    slow={split:{str(pd.Timestamp(h)): (float(y),float(p)) for h,y,p in zip(ev['results'][split]['hour_bjt'],ev['results'][split]['truth_db'],ev['results'][split]['prediction_db'])} for split in ('validation','test')}
    seen={'validation':set(),'test':set()}
    cols=['second_bjt','hour_bjt','quality_zero_records','received_power_w','hour_mean_power_w','fast_residual_db','split','rain','dry_overcast','clear_dry']
    for path in clean_paths:
        df=pd.read_parquet(path,columns=cols,filters=[('split','in',['validation','test'])])
        for hour,g in df.groupby('hour_bjt',sort=True):
            split=g.split.iloc[0];h=str(hour);need(h in slow[split] and h not in seen[split],'LOCAL identity');seen[split].add(h)
            y,pred=slow[split][h];base=float(g.hour_mean_power_w.iloc[0])
            need(abs(y-(-76.19362933508154-(10*np.log10(base)+30)))<1e-8,'LOCAL slow/fast truth mismatch')
            pos=((g.second_bjt-hour).dt.total_seconds()).to_numpy(dtype=int)
            need(len(pos)==len(set(pos)) and np.all((pos>=0)&(pos<3600)),'LOCAL second grid')
            f=np.full(3600,np.nan);records=np.zeros(3600,dtype=np.int16)
            records[pos]=g.quality_zero_records.to_numpy();f[pos]=g.fast_residual_db.to_numpy()
            valid=(records>=9)&np.isfinite(f)
            expected=-10*np.log10(g.received_power_w.to_numpy()/base)
            need(np.allclose(expected,g.fast_residual_db,equal_nan=True,rtol=0,atol=1e-10),'LOCAL residual definition')
            f[~valid]=np.nan
            event='rain' if bool(g.rain.iloc[0]) else 'dry_overcast' if bool(g.dry_overcast.iloc[0]) else 'clear_dry' if bool(g.clear_dry.iloc[0]) else 'other'
            meta={'dataset':'LOCAL','split':split,'station':'local','hour':hour.isoformat(),'date':str(hour.date()),'event':event,'original_event':event,'truth_db':y,'prediction_db':pred,'key':'D-final-'+h,'source_file':relative(path)}
            yield meta,f,valid,records
    need({k:len(v) for k,v in seen.items()}=={'validation':343,'test':687},'LOCAL coverage')
    for src,group in u[u.primary_score_eligible].groupby('source_file',sort=True):
        day=native(src)
        for r in group.sort_values('utc_epoch_seconds').itertuples(index=False):
            h=pd.Timestamp(r.hour_utc);a=day[h.hour];valid=np.isfinite(a)
            need(valid.sum()==r.valid_seconds and valid.sum()>=3420,'UK second quality')
            recomputed=-10*np.log10(np.mean(10**(-a[valid]/10)))
            need(abs(recomputed-r.truth_db)<1e-10,'UK power mean truth')
            meta={'dataset':'UK','split':'external','station':r.station,'hour':h.isoformat(),'date':str(h.date()),'event':r.generator_event,'original_event':r.event,'truth_db':float(r.truth_db),'prediction_db':float(r.prediction_db),'key':'T016-UK-'+r.station+'-'+h.isoformat(),'source_file':src}
            yield meta,a-r.truth_db,valid,valid.astype(np.int16)

def dynamics(dyn,f,valid,g,thresholds):
    masks=dyn.pair_masks(valid);obs1,obs2,obsacf=dyn.increments(f,masks)
    genacf=[];geninc=[]
    for lag,mask in zip(LAGS,masks):
        left=g[:,:-lag][:,mask];right=g[:,lag:][:,mask]
        if left.shape[1]<3:
            genacf.append(np.full(32,np.nan));geninc.append(np.full(32,np.nan));continue
        ac=left-left.mean(1)[:,None];bc=right-right.mean(1)[:,None]
        den=np.sqrt(np.sum(ac*ac,1)*np.sum(bc*bc,1))
        genacf.append(np.divide(np.sum(ac*bc,1),den,out=np.full(32,np.nan),where=den>0));geninc.append(np.abs(left-right).mean(1))
    genacf=np.array(genacf).T;geninc=np.array(geninc).T
    stats={'observed_sd_db':float(np.std(f[valid])),'generated_sd_db':np.std(g[:,valid],axis=1),
        'observed_acf':obsacf,'generated_acf':genacf,'pair_counts':[int(x.sum()) for x in masks],
        'observed_increment_mean':obs1,'observed_increment_second':obs2,'generated_increment_mean':geninc}
    need(np.isfinite(obs1).any(),'No valid lag')
    vs=dyn.score_from_moments(obs1,obs2,geninc)['vs']
    for tag,t in zip(('95','99'),thresholds):
        stats['observed_tail'+tag]=dyn.episode_arrays(f,valid,t)
        stats['generated_tail'+tag]=[dyn.episode_arrays(x,valid,t) for x in g]
    return stats,vs

def evaluate_hour(meta,f,valid,records,dyn,models,libraries,fm):
    event=meta['event'];q=[];seeds=[]
    for rep in range(32):
        vals,record=dyn.generate('block_0256',models[event],libraries[event],-1,meta['key'],rep)
        q.append(vals);seeds.append(record)
    q=np.array(q);g=-10*np.log10(q)
    fast,fair_fast=crps(g,f)
    error=meta['prediction_db']-meta['truth_db']
    combined,fair_combined=crps(g,f-error)
    slow=np.abs(f-error)
    for v in (fast,fair_fast,combined,fair_combined,slow):v[~valid]=np.nan
    thresholds=fm.quantile_f(models[event],np.array([.95,.99]))
    stats,vs=dynamics(dyn,f,valid,g,thresholds)
    lo,hi=np.quantile(g,[.05,.95],axis=0)
    row={**meta,'n_seconds':int(valid.sum()),'n_missing_seconds':int((~valid).sum()),
        'fast_crps_db':float(np.mean(fast[valid])),'fast_fair_crps_db':float(np.mean(fair_fast[valid])),
        'fast_zero_crps_db':float(np.mean(np.abs(f[valid]))),
        'combined_crps_db':float(np.mean(combined[valid])),'combined_fair_crps_db':float(np.mean(fair_combined[valid])),
        'slow_only_crps_db':float(np.mean(slow[valid])),
        'combined_minus_slow_db':float(np.mean(combined[valid]-slow[valid])),
        'observed_sd_db':stats['observed_sd_db'],'generated_sd_mean_db':float(np.mean(stats['generated_sd_db'])),
        'sd_absolute_error_db':float(np.mean(np.abs(stats['generated_sd_db']-stats['observed_sd_db']))),
        'acf_absolute_error':float(np.nanmean(np.abs(stats['generated_acf']-stats['observed_acf']))),
        'vs_db2':vs,'fast_coverage90':float(np.mean(((f>=lo)&(f<=hi))[valid])),
        'combined_coverage90':float(np.mean(((f-error>=lo)&(f-error<=hi))[valid])),
        'generated_mean_q':float(q.mean()),'generated_drift_db_mean':float(np.mean(-10*np.log10(q.mean(1))))}
    for tag in ('95','99'):
        obs=stats['observed_tail'+tag];gen=stats['generated_tail'+tag]
        row['observed_duration'+tag+'_mean_s']=float(np.mean(obs['complete_fades_s'])) if obs['complete_fades_s'] else np.nan
        repmeans=[np.mean(x['complete_fades_s']) for x in gen if x['complete_fades_s']]
        row['generated_duration'+tag+'_mean_s']=float(np.mean(repmeans)) if repmeans else np.nan
        row['observed_duration'+tag+'_complete_count']=len(obs['complete_fades_s'])
        row['observed_duration'+tag+'_censored_count']=obs['censored_fades']
        row['generated_duration'+tag+'_nonempty_replicates']=len(repmeans)
        row['generated_duration'+tag+'_censored_mean']=float(np.mean([x['censored_fades'] for x in gen]))
    return row,dict(observed_fast_db=f,valid=valid,records=records,generated_fast_db=g,
        fast_crps_db=fast,combined_crps_db=combined,slow_crps_db=slow,
        fair_fast_crps_db=fair_fast,fair_combined_crps_db=fair_combined,
        metadata=np.array(json.dumps(clean(meta),ensure_ascii=False)),seeds=np.array(json.dumps(seeds)),
        diagnostics=np.array(json.dumps(clean(stats),allow_nan=False)),row=np.array(json.dumps(clean(row),allow_nan=False)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--limit',type=int);args=parser.parse_args()
    with lock():
        need(not (OUT/'STOP_REQUESTED').exists(),'New version stop marker exists')
        process=psutil.Process();runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write(OUT/'runs'/f'{runid}.json',{'started_utc':now(),'pid':process.pid,'create_time':process.create_time(),'command':process.cmdline(),'resume':args.resume,'cpu_threads':2,'python':platform.python_version(),'numpy':np.__version__,'available_ram':psutil.virtual_memory().available,'disk_free':psutil.disk_usage(str(ROOT.anchor)).free})
        need(psutil.disk_usage(str(ROOT.anchor)).free>15_000_000_000,'Insufficient disk reserve')
        mod,dyn,saved,models,libraries,fm=legacy()
        control=controls(dyn)
        pins,paths,ev,u=prepare_inputs(saved)
        manifest={'sources':pins,'expected_hours':6623,'local_validation':343,'local_test':687,'uk_primary':5593,'generated_members':32,'training_calls':0,'new_preprocessing_fit':0,'cdf_fits':0,'model_inference_calls':0}
        if args.resume:
            need(read(OUT/'manifest.json')==manifest,'Resume manifest/source mismatch')
        else:
            need(not (OUT/'manifest.json').exists(),'Existing run requires inspected --resume')
            write(OUT/'manifest.json',manifest);write(OUT/'controls.json',control)
            u.loc[~u.primary_score_eligible,['station','hour_utc','valid_seconds','exclusion_reason']].to_csv(OUT/'excluded_uk_hours.csv',index=False)
            u[['station','hour_utc','event','generator_event','tcc','total_water_mm','snow_water_mm','primary_score_eligible']].to_csv(OUT/'uk_generator_weather.csv',index=False)
        count=0;computed=0;rows=[];start=time.monotonic();shards=OUT/'hours';shards.mkdir(exist_ok=True)
        for meta,f,valid,records in iterate(paths,ev,u,mod.native):
            need(not (OUT/'STOP_REQUESTED').exists(),'User stop marker')
            ident=hashlib.sha256((meta['dataset']+'|'+meta['split']+'|'+meta['station']+'|'+meta['hour']).encode()).hexdigest()[:20]
            path=shards/(ident+'.npz');receipt=shards/(ident+'.json')
            if receipt.exists():
                rec=read(receipt);need(rec['sha256']==sha(path) and rec['identity']==meta,'Checkpoint mismatch')
                row=rec['row']
            else:
                need(not path.exists(),'Unreceipted artifact; inspect before retry '+str(path))
                row,arrays=evaluate_hour(meta,f,valid,records,dyn,models,libraries,fm)
                with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
                write(receipt,{'identity':meta,'row':clean(row),'sha256':sha(path),'completed_utc':now()});computed+=1
            count+=1;rows.append(row)
            if count%12==0 or count==1:
                status={'state':'RUNNING','updated_utc':now(),'pid':process.pid,'create_time':process.create_time(),'completed_hours':count,'total_hours':6623,'computed_this_run':computed,'current':meta,'elapsed_s_this_run':round(time.monotonic()-start,3),'rss_bytes':process.memory_info().rss,'cpu_s':sum(process.cpu_times()[:2]),'disk_free':psutil.disk_usage(str(ROOT.anchor)).free}
                write(OUT/'status.json',status,replace=True);print(json.dumps({k:v for k,v in status.items() if k not in ('current',)},ensure_ascii=False),flush=True)
            if args.limit and computed>=args.limit:
                write(OUT/'runs'/f'{runid}_exit.json',{'state':'PAUSED_AFTER_BOUNDED_CHECK','completed_hours':count,'utc':now()});return
        need(count==6623 and len(rows)==6623,'Final hour count')
        frame=pd.DataFrame(rows);need(not frame.duplicated(['dataset','split','station','hour']).any(),'Duplicate hour')
        for src,h in pins.items():need(sha(ROOT/src)==h,'Post-run source change '+src)
        frame.to_csv(OUT/'hourly_results.csv',index=False,encoding='utf-8-sig')
        frame.to_parquet(OUT/'hourly_results.parquet',index=False)
        write(OUT/'computation_complete.json',{'state':'COMPUTED_PENDING_INDEPENDENT_QA','utc':now(),'hours':count,'realizations':count*32,'sources_unchanged':True,'new_fits':0,'elapsed_s_this_run':time.monotonic()-start})
        write(OUT/'status.json',{'state':'COMPUTED_PENDING_INDEPENDENT_QA','completed_hours':count,'total_hours':6623,'updated_utc':now()},replace=True)
        write(OUT/'runs'/f'{runid}_exit.json',{'state':'COMPUTED_PENDING_INDEPENDENT_QA','utc':now()})

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'failures'/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json',{'state':'FAILED_REQUIRES_INSPECTION','utc':now(),'traceback':traceback.format_exc()})
        write(OUT/'status.json',{'state':'FAILED_REQUIRES_INSPECTION','updated_utc':now()},replace=True)
        raise
