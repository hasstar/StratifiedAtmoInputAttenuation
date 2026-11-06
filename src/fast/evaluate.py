"""Regenerate 32 curves/hour and evaluate local/UK fast and combined signals."""
from __future__ import annotations
import argparse,hashlib,json,sys,types,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s
import numpy as np
import pandas as pd
import generator

def clean(x):
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def need(ok,message):
    if not ok:raise RuntimeError(message)
def crps(g,y):
    m=len(g);ordered=np.sort(g,axis=0)
    spread=np.sum((2*np.arange(1,m+1)-m-1)[:,None]*ordered,axis=0)/(m*m)
    first=np.mean(abs(g-y),axis=0)
    return first-spread,first-spread*m/(m-1)
def relative(p):
    try:return 'Outputs/'+Path(p).relative_to(s.path('reference_results')/'Outputs').as_posix()
    except ValueError:return s.rel(p)
def numeric_module():
    """Import original evaluation equations with portable I/O helpers and no lock CLI."""
    c=types.ModuleType('common')
    for k,v in dict(np=np,Path=Path,json=json,read=s.read,sha=s.sha,need=need,clean=clean,crps=crps,
                    relative=relative,ROOT=s.ProjectRoot(),LAGS=tuple(s.config()['fast']['lags_seconds'])).items():setattr(c,k,v)
    sys.modules['common']=c
    return s.load_module('supp_fast_numeric',s.original('Code/Fast_Fading_Experiments/Revision_003/evaluate.py'))
def native(source):
    import netCDF4
    p=s.path('ceda_raw')/str(source).replace('\\','/')
    with netCDF4.Dataset('native.nc',memory=p.read_bytes())as ds:
        ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];t=np.asarray(ds.variables['time'][:],float);a=np.asarray(v[:],float)
        valid=np.isfinite(a)&(a!=v._FillValue)&(a>=v.valid_min)&(a<=v.valid_max)&np.isfinite(t)&(t>0)&(t<=86400)
        day=np.full(86400,np.nan);day[t[valid].astype(int)-1]=a[valid]
    return day.reshape(24,3600)
def inputs(hourly_run=None,uk_predictions=None):
    slow=s.original('Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE/models/tabm/k_0010/evaluation.json')
    if hourly_run:slow=Path(hourly_run)/'full156/tabm/k_0010/evaluation.json'
    ev=s.read(slow);uk=s.original('Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE')
    u=pd.read_parquet(uk/'inputs/uk_inputs_38.parquet')
    predictions=uk_predictions or uk/'evaluation/all_hourly_predictions.parquet'
    pred=pd.read_parquet(predictions,filters=[('model','==','tabm'),('k','==',10)])
    pred=pred[['station','utc_epoch_seconds','prediction_db','event','truth_db','primary_score_eligible']]
    u=u.merge(pred,on=['station','utc_epoch_seconds'],validate='one_to_one',suffixes=('','_prediction'))
    w=pd.read_parquet(s.original('Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet'))[['station','utc_epoch_seconds','tcc','total_water_mm','snow_water_mm']]
    u=u.merge(w,on=['station','utc_epoch_seconds'],validate='one_to_one',suffixes=('','_weather'))
    p=u.total_water_mm;sn=u.snow_water_mm;c=u.tcc
    u['generator_event']=np.select([(p>=.1)&(sn<.1),(p<.1)&(sn<.1)&(c>=.8),(p<.1)&(sn<.1)&(c<=.2)],['rain','dry_overcast','clear_dry'],default='other')
    paths=sorted(s.original('Outputs/Experiments/Fast_Fading_Revision_001/clean_1s').glob('*.parquet'))
    return paths,ev,u
def run(out,generator_dir=None,hourly_run=None,uk_predictions=None,limit=None):
    dyn,fm,models,libraries,saved=generator.load_new(generator_dir) if generator_dir else generator.load_frozen()
    ev=numeric_module();out.mkdir(parents=True,exist_ok=False);(out/'hours').mkdir()
    control=ev.controls(dyn);s.write(out/'controls.json',control)
    paths,slow,u=inputs(hourly_run,uk_predictions);rows=[];start=time.perf_counter()
    for meta,f,valid,records in ev.iterate(paths,slow,u,native):
        row,arrays=ev.evaluate_hour(meta,f,valid,records,dyn,models,libraries,fm)
        ident=hashlib.sha256((meta['dataset']+'|'+meta['split']+'|'+meta['station']+'|'+meta['hour']).encode()).hexdigest()[:20]
        file=out/'hours'/(ident+'.npz');np.savez_compressed(file,**arrays)
        s.write(file.with_suffix('.json'),dict(identity=meta,row=clean(row),sha256=s.sha(file)))
        rows.append(row)
        if len(rows)%25==0:print('GENERATED',len(rows),'of',6623,flush=True)
        if limit and len(rows)>=limit:break
    pd.DataFrame(rows).to_parquet(out/'hourly_results.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'hourly_results.csv',index=False,float_format='%.17g')
    if not limit:assert len(rows)==6623
    s.write(out/'computation.json',dict(status='BOUNDED_CHECK' if limit else 'FULL_GENERATION_COMPLETE',
        hours=len(rows),members=32,elapsed_seconds=time.perf_counter()-start,
        training_calls=0,cdf_refits=0,generator_source=str(generator_dir)if generator_dir else 'frozen reference',
        hourly_source=str(hourly_run)if hourly_run else 'frozen local k10'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--generator',type=Path)
    ap.add_argument('--hourly-run',type=Path);ap.add_argument('--uk-predictions',type=Path);ap.add_argument('--limit',type=int)
    a=ap.parse_args();run(a.out,a.generator,a.hourly_run,a.uk_predictions,a.limit)
if __name__=='__main__':main()
