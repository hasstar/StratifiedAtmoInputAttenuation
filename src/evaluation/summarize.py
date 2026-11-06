"""Recompute manuscript summaries and figure inputs from a new numerical run.

The diagnostic definitions and date-cluster bootstrap match the archived study.
Numerical figures consume these outputs, rather than silently using old curves.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fast'))
import evaluate as fast

def temporal(folder, dest):
    dense=s.load_module('supp_dense_acf',s.path('figure_sources')/'common/derive_dense_acf.py')
    sums=defaultdict(lambda:np.zeros((5,1800)))
    durations=defaultdict(lambda:dict(obs=np.zeros(3601),gen=np.zeros(3601),n=0))
    files=sorted((folder/'hours').glob('*.npz'));max_error=0.
    for index,p in enumerate(files):
        assert s.sha(p)==s.read(p.with_suffix('.json'))['sha256']
        with np.load(p,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));d=json.loads(str(z['diagnostics']))
            valid=z['valid'].astype(bool)
            values=np.vstack([z['observed_fast_db'][None,:],z['generated_fast_db']]).astype(float)
        c,n=dense.calculate(values,valid)
        old=np.vstack([np.asarray(d['observed_acf'],float)[None,:],np.asarray(d['generated_acf'],float)])
        actual=c[:,np.asarray(s.config()['fast']['lags_seconds'])-1]
        assert np.array_equal(np.isnan(actual),np.isnan(old)),p.name
        error=float(np.nanmax(abs(actual-old)));max_error=max(max_error,error)
        assert error<2e-7,(p.name,error)
        count=np.isfinite(c[1:]).sum(0)
        gm=np.divide(np.nansum(c[1:],axis=0),count,out=np.full(1800,np.nan),where=count>0)
        obs=c[0];good=np.isfinite(obs)&np.isfinite(gm)
        groups=[('LOCAL_'+meta['split'],'local')] if meta['dataset']=='LOCAL' else [('UK',meta['station']),('UK','pooled')]
        for cohort,station in groups:
            for event in ('overall',meta['event']):
                a=sums[cohort,station,event]
                a[0,good]+=obs[good];a[1,good]+=gm[good];a[2,good]+=1
                a[3,good]+=abs(obs[good]-gm[good]);a[4,good]+=n[good]
        for tag in ('95','99'):
            od=np.asarray(d['observed_tail'+tag]['complete_fades_s'],int)
            gd=[np.asarray(v['complete_fades_s'],int)for v in d['generated_tail'+tag] if v['complete_fades_s']]
            if not len(od) or not gd:continue
            oh=np.bincount(od,minlength=3601)/len(od)
            gh=np.mean([np.bincount(g,minlength=3601)/len(g) for g in gd],axis=0)
            for cohort,station in groups:
                for event in ('overall',meta['event']):
                    a=durations[cohort,station,event,tag];a['obs']+=oh;a['gen']+=gh;a['n']+=1
        if (index+1)%500==0:print('TEMPORAL_SUMMARY',index+1,'/',len(files),flush=True)
    rows=[]
    for (cohort,station,event),v in sorted(sums.items()):
        for j in range(1800):
            n=int(v[2,j]);rows.append(dict(cohort=cohort,station=station,event=event,lag_seconds=j+1,
                hours_with_defined_acf=n,observed_acf=v[0,j]/n if n else np.nan,
                generated_acf_mean=v[1,j]/n if n else np.nan,acf_mae=v[3,j]/n if n else np.nan,valid_pair_count=int(v[4,j])))
    acf=pd.DataFrame(rows);acf.to_csv(dest/'Fig10_dense_acf_1_to_1800.csv',index=False)
    acf[acf.lag_seconds.isin(s.config()['fast']['lags_seconds'])].to_csv(dest/'frozen_exports/fast_acf.csv',index=False)
    rows=[]
    for (cohort,station,event,tag),v in sorted(durations.items()):
        for kind in ('obs','gen'):
            mass=v[kind]/v['n'];cdf=np.cumsum(mass)
            for seconds in np.flatnonzero(mass):rows.append(dict(cohort=cohort,station=station,event=event,threshold='F'+tag,
                kind=kind,duration_s=int(seconds),mass=mass[seconds],cdf=cdf[seconds],paired_hours=v['n']))
    pd.DataFrame(rows).to_csv(dest/'Fig10_duration_distributions.csv',index=False)
    return dict(hours=len(files),dense_lags=1800,max_saved_lag_error=max_error)

def hourly_tables(folder,dest):
    rows=[];features=[]
    for p in sorted(folder.glob('*/*/k_*/evaluation.json')):
        state=s.read(p.with_name('state.json'));ev=s.read(p)
        for split,value in ev['results'].items():
            for weather,metrics in value['metrics'].items():
                rows.append(dict(pool=state['pool'],model=state['model'],k=state['k'],split=split,weather=weather,**metrics))
        features.append(dict(pool=state['pool'],model=state['model'],k=state['k'],
              retained_features=json.dumps(state['feature_names']),removed_next=state['removed_feature_next']))
    frame=pd.DataFrame(rows);frame.to_csv(dest/'all_new_hourly_metrics.csv',index=False)
    pd.DataFrame(features).to_csv(dest/'all_new_feature_paths.csv',index=False)
    local=frame[(frame.pool=='full156')&(frame.split=='test')].copy();local['population']='ML_687'
    local.to_csv(dest/'figure_inputs/common/frozen_exports/local_metrics.csv',index=False)
    tab=frame[frame.model=='tabm'];keys=['k','split','weather','n']
    full=tab[tab.pool=='full156'];base=tab[tab.pool=='base50'];cols=['rmse_db','r2','mae_db','bias_db']
    both=full[keys+cols].merge(base[keys+cols],on=keys,suffixes=('_all','_surface'),validate='one_to_one')
    for col in cols:
        both['all_'+col]=both.pop(col+'_all');both['surface_'+col]=both.pop(col+'_surface')
        both['surface_minus_all_'+col]=both['surface_'+col]-both['all_'+col]
    both.to_csv(dest/'figure_inputs/common/frozen_exports/tabm_same_k.csv',index=False)
    comparison_tables(local,both,dest)

def comparison_tables(local,both,dest):
    names=list(s.config()['hourly']['models']);others=[n for n in names if n!='tabm']
    local=local.copy()
    if 'population' in local:local=local[local.population=='ML_687']
    if 'split' in local:local=local[local.split=='test']
    local=local[local.model.isin(names)]
    assert not local.duplicated(['model','k','weather']).any(),'Comparison table input has duplicate states'
    chosen=[156,*range(10,0,-1)];events=['overall','rain','dry_overcast']
    rmse=[];r2=[]
    for weather in events:
        for k in chosen:
            part=local[(local.weather==weather)&(local.k==k)].set_index('model')
            if 'tabm' not in part.index:continue
            a=dict(Weather=weather,k=k,TabM=part.loc['tabm','rmse_db'])
            b=dict(Weather=weather,k=k,TabM=part.loc['tabm','r2'])
            for name in others:
                a[name]=100*(part.loc[name,'rmse_db']-a['TabM'])/part.loc[name,'rmse_db'] if name in part.index else np.nan
                b[name]=b['TabM']-part.loc[name,'r2'] if name in part.index else np.nan
            rmse.append(a);r2.append(b)
    for name,rows in [('RMSE_reduction',rmse),('R2_difference',r2)]:
        f=pd.DataFrame(rows);f.to_csv(dest/f'Table04_{name}.csv',index=False)
        f.to_latex(dest/f'Table04_{name}.tex',index=False,float_format='%.4f',escape=True)
    q=both[(both.split=='test')&both.weather.isin(events)&both.k.isin([50,25,*range(10,0,-1)])]
    q=q[['weather','k','all_rmse_db','surface_rmse_db','all_r2','surface_r2']]
    q.to_csv(dest/'Table05_Independent_Paths.csv',index=False)
    q.to_latex(dest/'Table05_Independent_Paths.tex',index=False,float_format='%.4f',escape=True)

def readable_tables(summary, dest):
    """Separate scores from matched-hour process statistics; never print 42 columns."""
    labels={'LOCAL_validation':'Local validation','LOCAL_test':'Local test'}
    frame=summary.copy()
    frame['Data']=[labels.get(c,'UK '+st.title()) for c,st in zip(frame.cohort,frame.station)]
    frame['Weather']=frame.event.map({'overall':'Overall','rain':'Rain','dry_overcast':'Overcast',
                                    'clear_dry':'Clear/dry','other':'Other'})
    specifications=[
      ('Amplitude', ['Data','Weather','hours','fast_zero_crps_db','fast_crps_db','observed_sd_db',
                     'generated_sd_mean_db','sd_absolute_error_db'],
       ['Data','Weather','Hours','Zero CRPS','Generated CRPS','Observed SD','Generated SD','SD MAE'],
       'Second-scale amplitude scores and fluctuation intensity (dB).'),
      ('Durations', ['Data','Weather','duration95_matched_hours','observed_duration95_matched_mean_s',
                      'generated_duration95_matched_mean_s','duration99_matched_hours',
                      'observed_duration99_matched_mean_s','generated_duration99_matched_mean_s'],
       ['Data','Weather','Pairs F95','Observed F95','Generated F95','Pairs F99','Observed F99','Generated F99'],
       'Mean complete fade durations (s); observed and generated means use the same paired hours.')]
    for name,cols,heads,caption in specifications:
        table=frame[cols].copy();table.columns=heads
        table.to_latex(dest/f'Table08_{name}.tex',index=False,float_format='%.4f',longtable=True,
                       escape=True,caption=caption,label='tab:fast_'+name.lower())
    table=frame[frame.event.eq('overall')].copy()
    table['95% CI']=[f'[{lo:.4f}, {hi:.4f}]'for lo,hi in zip(table.delta_cluster_ci95_low,table.delta_cluster_ci95_high)]
    table=table[['Data','hours','slow_only_crps_db','combined_crps_db','combined_improvement_percent','95% CI']]
    table.columns=['Data','Hours','Hourly only','Combined','Reduction (%)','95% CI']
    table.to_latex(dest/'Table09_Combined_CRPS.tex',index=False,float_format='%.4f',escape=True,
                   caption='Second-scale CRPS (dB). Confidence intervals describe combined minus hourly-only CRPS.',
                   label='tab:combined_crps')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fast',type=Path,required=True)
    ap.add_argument('--hourly',type=Path);ap.add_argument('--uk',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--skip-dense',action='store_true',help='Tables only; omit new temporal figure inputs')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    # The imported module only contributes aggregation functions, never its CLI.
    fast.numeric_module()
    mod=s.load_module('supp_fast_summary',s.original('Code/Fast_Fading_Experiments/Revision_003/summarize_and_plot.py'))
    mod.OUT=a.fast
    frame=pd.read_parquet(a.fast/'hourly_results.parquet');summary=mod.aggregate(frame)
    assert len(frame)==6623 and len(summary)==25
    summary.to_csv(a.out/'fast_and_combined_by_weather.csv',index=False,float_format='%.17g')
    readable_tables(summary,a.out)
    inputs=a.out/'figure_inputs/common'
    shutil.copytree(s.path('figure_inputs')/'common',inputs)
    frame.to_csv(inputs/'frozen_exports/fast_hourly.csv',index=False)
    summary.to_csv(inputs/'frozen_exports/fast_summary.csv',index=False)
    if a.hourly:hourly_tables(a.hourly,a.out)
    if a.uk:shutil.copyfile(a.uk/'uk_metrics.csv',inputs/'frozen_exports/uk_metrics.csv')
    examples=[]
    for ds,st in [('LOCAL','local'),('UK','chilbolton'),('UK','sparsholt')]:
        g=frame[(frame.dataset==ds)&(frame.station==st)&(frame.event=='rain')]
        if ds=='LOCAL':g=g[g.split=='test']
        r=g.sort_values('hour').iloc[0];key='|'.join(str(r[x])for x in ('dataset','split','station','hour'))
        ident=hashlib.sha256(key.encode()).hexdigest()[:20]
        shutil.copyfile(a.fast/'hours'/(ident+'.npz'),inputs/'examples'/f'{ds}_{st}.npz');examples.append(r.to_dict())
    pd.DataFrame(examples).to_csv(inputs/'FigS01_examples.csv',index=False)
    temporal_report={} if a.skip_dense else temporal(a.fast,inputs)
    s.write(a.out/'summary_qa.json',dict(status='PASS',hours=len(frame),groups=len(summary),
        source_fast=s.rel(a.fast),source_hourly=s.rel(a.hourly)if a.hourly else 'frozen reference',
        temporal=temporal_report,elapsed_seconds=time.perf_counter()-start,
        static_assets='Measurement, geometry, architecture, ITU and SHAP remain publication reference assets.'))
    print(summary[summary.event.eq('overall')][['cohort','station','hours','fast_crps_db','combined_crps_db']].to_string(index=False),flush=True)
if __name__=='__main__':main()
