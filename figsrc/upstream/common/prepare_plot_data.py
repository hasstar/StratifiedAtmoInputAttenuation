"""Prepare reproducible plot inputs from frozen experiments; no model fitting/calls.

New duration ECDFs give every paired hour equal weight and, inside a generated
hour, give every nonempty realization equal weight. They are descriptive, with
censored events excluded consistently with Fast003. Original files are read-only.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='2'
import argparse,hashlib,json,shutil,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'最终提交的修改材料/论文图表_20261030_001'
SOURCES={
 'local_metrics.csv':'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002/tables/all_test_metrics_15605_rows.csv',
 'local_states.csv':'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002/tables/all_states_1560.csv',
 'itu_predictions.csv':'最终提交的修改材料/ITU组合模型_20261027_001/01_结果/ITU_共同684小时_逐小时预测.csv',
 'tabm_same_k.csv':'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/same_k_comparison.csv',
 'tabm_two_pool_metrics.csv':'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/all_states_metrics.csv',
 'tabm_retained_features.csv':'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/retained_features.csv',
 'uk_metrics.csv':'Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE/evaluation/metrics_all_weather_station.csv',
 'fast_summary.csv':'最终提交的修改材料/补充材料_20261024_001/快变与TabM组合_分天气汇总.csv',
 'fast_acf.csv':'最终提交的修改材料/补充材料_20261024_001/快变_ACF分天气汇总.csv',
 'fast_hourly.csv':'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble/hourly_results.csv',
 'block_selection.csv':'Outputs/Experiments/Fast_Fading_Revision_002/dynamics/r16/candidate_ranking.csv',
 'clean_hourly.csv':'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s_hourly_summary.csv',
}
DOCS=[
'Materials/Original_Submission/English_2026_TVT.pdf',
'Code/Hourly_Attenuation_Experiments/Revision_023/PROTOCOL.md',
'Code/Hourly_Attenuation_Experiments/Revision_032/PROTOCOL.md',
'Code/Hourly_Attenuation_Experiments/Revision_034/build_figures.py',
'Code/Hourly_Attenuation_Experiments/Revision_037/PROTOCOL.md',
'Code/Hourly_Attenuation_Experiments/Revision_041/PROTOCOL.md',
'Code/Fast_Fading_Experiments/Revision_003/PROTOCOL.md',
'Code/Fast_Fading_Experiments/Revision_003/evaluate.py',
'Code/Fast_Fading_Experiments/Revision_003/common.py',
'Code/Fast_Fading_Experiments/Revision_003/summarize_and_plot.py',
'Code/Fast_Fading_Experiments/Revision_002/MODEL_PLAN.md',
'Code/Fast_Fading_Experiments/Revision_002/DYNAMICS_PLAN.md',
'Outputs/Experiments/Fast_Fading_Revision_002/dynamics/r16/DYNAMICS_RESULTS.md',
'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble/independent_numeric_qa.json',
'Outputs/Datasets/Signal_Detectability_Revision_003/README.md',
'Outputs/Datasets/Signal_Detectability_Revision_003/manifest.json',
'最终提交的修改材料/ITU组合模型_20261027_001/00_阅读说明.md',
'最终提交的修改材料/TabM原论文对照与任务适配说明_20261024_001.md',
'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/NUMERICAL_QA.json',
]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()

def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--skip-durations',action='store_true');ap.add_argument('--resume-preparation',action='store_true');a=ap.parse_args()
 out=a.output.resolve();data=out/'04_Source_Data';snap=data/'frozen_exports';snap.mkdir(parents=True,exist_ok=a.resume_preparation)
 qa=out/'05_QA';qa.mkdir(exist_ok=True)
 registry=[]
 def track(p,role,copy_to=None):
  p=Path(p);rec={'original_path':str(p.resolve()),'project_relative':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'role':role}
  if copy_to:
   copy_to.parent.mkdir(parents=True,exist_ok=True)
   if copy_to.exists():assert a.resume_preparation and sha(copy_to)==rec['sha256']
   else:shutil.copy2(p,copy_to)
   assert sha(copy_to)==rec['sha256'];rec['snapshot']=copy_to.relative_to(out).as_posix()
  registry.append(rec);return rec
 def csv(frame,name):frame.to_csv(data/name,index=False,encoding='utf-8-sig')
 for name,rel in SOURCES.items():track(ROOT/rel,name,snap/name)
 for i,rel in enumerate(DOCS):
  p=ROOT/rel
  if p.exists():track(p,'upstream method / QA',data/'methods'/f'{i:02d}_{p.name}')
 status_p=ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/hourly_sample_status.parquet';track(status_p,'local weather and split')
 status=pd.read_parquet(status_p)
 status['event']=np.select([status.rain,status.dry_overcast,status.clear_dry],['rain','dry_overcast','clear_dry'],default='other')
 local=[]
 for split in ('train','validation','test'):
  p=ROOT/f'Outputs/Datasets/Signal_Detectability_Revision_003/{split}.parquet';track(p,'local official labels '+split)
  d=pd.read_parquet(p,columns=['hour_bjt','hour_utc','clear_sky_attenuation_db']);d['split']=split
  d=d.merge(status[['hour_bjt','event']],on='hour_bjt',validate='one_to_one');local.append(d)
 local=pd.concat(local).sort_values('hour_bjt');local['power_dbm']=-76.19362933508154-local.clear_sky_attenuation_db
 csv(local,'Fig03_all_local_hours.csv')
 hourly=pd.read_csv(snap/'fast_hourly.csv');summary=pd.read_csv(snap/'fast_summary.csv')
 rows=[]
 for split in ('train','validation','test'):
  g=local[local.split.eq(split)];dates=pd.to_datetime(g.hour_bjt).dt.strftime('%Y-%m-%d')
  rows.append(dict(dataset='LOCAL',subset=split,station='local',hours=len(g),start=dates.min(),end=dates.max(),dates=dates.nunique(),timezone='BJT',**{e:int(g.event.eq(e).sum()) for e in ('rain','dry_overcast','clear_dry','other')}))
 for station in ('chilbolton','sparsholt','pooled'):
  g=hourly[hourly.dataset.eq('UK')];g=g if station=='pooled' else g[g.station.eq(station)]
  rows.append(dict(dataset='UK',subset='external',station=station,hours=len(g),start=g.date.min(),end=g.date.max(),dates=g.date.nunique(),timezone='UTC',**{e:int(g.event.eq(e).sum()) for e in ('rain','dry_overcast','clear_dry','other')}))
 csv(pd.DataFrame(rows),'Table02_datasets.csv')
 eligible=pd.read_csv(snap/'clean_hourly.csv').query('primary_seconds==3600').sort_values('hour_bjt')
 selected=eligible.iloc[np.random.default_rng(20261030).integers(len(eligible))]
 hour=pd.Timestamp(selected.hour_bjt).tz_convert('Asia/Shanghai').as_unit('ns');month=hour.strftime('%Y-%m');p=ROOT/f'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s/clean_1s_{month}.parquet';track(p,'Fig03 random-hour raw seconds')
 sec=pd.read_parquet(p,filters=[('hour_bjt','=',hour)])
 assert len(sec)==3600 and sec.primary_usable_1s.all()
 sec['power_dbm']=10*np.log10(sec.received_power_w)+30;sec['second_in_hour']=(sec.second_bjt-hour).dt.total_seconds()
 csv(sec[['second_bjt','second_in_hour','power_dbm','normalized_power_ratio','primary_usable_1s']],'Fig03_selected_seconds.csv')
 js(data/'Fig03_selection.json',dict(seed=20261030,rule='Uniform choice among all LOCAL hours with 3600 primary valid seconds; sorted chronologically; no outcome-based reselection.',candidate_hours=len(eligible),selected_hour=str(hour),event=str(selected.weather_stratum)))
 print('Selected hour',str(hour),flush=True)
 manifest_p=ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE/manifest.json';track(manifest_p,'Table03 frozen model configs')
 configs=json.loads(manifest_p.read_text(encoding='utf-8'))['model_configs']
 bai_p=ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE/models/bai_lstm/k_0156/state.json';track(bai_p,'Table03 selected Bai epoch')
 configs['bai_lstm']=json.loads(bai_p.read_text())['config'];js(data/'Table03_model_configs.json',configs)
 # A prerecorded example for each target cohort, chosen by earliest rainy hour.
 selected_examples=[]
 for ds,split,st in [('LOCAL','test','local'),('UK','external','chilbolton'),('UK','external','sparsholt')]:
  g=hourly[(hourly.dataset==ds)&(hourly.event=='rain')]
  if ds=='LOCAL':g=g[g.split.eq(split)]
  else:g=g[g.station.eq(st)]
  r=g.sort_values('hour').iloc[0];key='|'.join(str(r[x]) for x in ('dataset','split','station','hour'))
  p=ROOT/'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble/hours'/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.npz')
  track(p,'FigS01 saved example',data/'examples'/f'{ds}_{st}.npz');selected_examples.append(r.to_dict())
 csv(pd.DataFrame(selected_examples),'FigS01_examples.csv')
 if not a.skip_durations:
  accum=defaultdict(lambda:dict(obs=np.zeros(3601),gen=np.zeros(3601),n=0,obs_count=0,gen_count=0))
  receipts=[];thresholds={};science=ROOT/'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble'
  for index,r in enumerate(hourly.to_dict('records')):
   key='|'.join(str(r[x]) for x in ('dataset','split','station','hour'));p=science/'hours'/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.npz')
   receipt=json.loads(p.with_suffix('.json').read_text());digest=sha(p);assert digest==receipt['sha256'],str(p)
   receipts.append({'path':p.relative_to(ROOT).as_posix(),'sha256':digest,'key':key})
   with np.load(p,allow_pickle=False) as z:d=json.loads(str(z['diagnostics']))
   cohort='LOCAL_'+r['split'] if r['dataset']=='LOCAL' else 'UK'
   stations=['local'] if r['dataset']=='LOCAL' else [r['station'],'pooled']
   for tag in ('95','99'):
    od=d['observed_tail'+tag];obs=np.asarray(od['complete_fades_s'],dtype=int)
    gens=[np.asarray(v['complete_fades_s'],dtype=int) for v in d['generated_tail'+tag] if v['complete_fades_s']]
    thresholds[(r['event'],tag)]=od['threshold_db']
    if len(obs)==0 or not gens:continue
    om=np.bincount(obs,minlength=3601)/len(obs)
    gm=np.mean([np.bincount(g,minlength=3601)/len(g) for g in gens],axis=0)
    assert np.isclose(np.dot(np.arange(3601),om),r['observed_duration'+tag+'_mean_s'])
    assert np.isclose(np.dot(np.arange(3601),gm),r['generated_duration'+tag+'_mean_s'])
    for st in stations:
     for event in ('overall',r['event']):
      acc=accum[(cohort,st,event,tag)];acc['obs']+=om;acc['gen']+=gm;acc['n']+=1;acc['obs_count']+=len(obs);acc['gen_count']+=sum(len(g) for g in gens)
   if (index+1)%500==0:print('Verified stored-hour diagnostics',index+1,'/',len(hourly),flush=True)
  hist=[];qa_rows=[]
  for (cohort,st,event,tag),v in accum.items():
   s=summary[(summary.cohort==cohort)&(summary.station==st)&(summary.event==event)].iloc[0]
   assert v['n']==s['duration'+tag+'_matched_hours']
   for kind in ('obs','gen'):
    mass=v[kind]/v['n'];mean=float(np.dot(np.arange(3601),mass));field=('observed' if kind=='obs' else 'generated')+'_duration'+tag+'_matched_mean_s'
    assert np.isclose(mean,s[field],atol=1e-10,rtol=1e-10)
    cdf=np.cumsum(mass);ix=np.flatnonzero(mass)
    for duration in ix:hist.append(dict(cohort=cohort,station=st,event=event,threshold='F'+tag,kind=kind,duration_s=int(duration),mass=float(mass[duration]),cdf=float(cdf[duration]),paired_hours=v['n']))
    qa_rows.append(dict(cohort=cohort,station=st,event=event,threshold='F'+tag,kind=kind,mean_duration_s=mean,reported_mean_s=float(s[field]),paired_hours=v['n'],cdf_final=float(cdf[-1])))
  csv(pd.DataFrame(hist),'Fig10_duration_distributions.csv');js(qa/'duration_distribution_qa.json',dict(status='PASS',groups=qa_rows,stored_hours=len(receipts)))
  js(qa/'Fast003_hour_receipts.json',receipts);js(data/'fast_thresholds.json',[dict(event=k[0],threshold='F'+k[1],db=v) for k,v in sorted(thresholds.items())])
 js(qa/'SOURCE_MANIFEST.json',registry)
 print('PREPARED',len(registry),'source files; output',out,flush=True)

if __name__=='__main__':main()
