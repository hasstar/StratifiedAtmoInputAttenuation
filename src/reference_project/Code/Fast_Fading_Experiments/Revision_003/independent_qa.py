"""Separate CDF-integral CRPS audit and native-second reconstruction, no scoring import."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,time,traceback
import pandas as pd
import netCDF4
from common import ROOT,OUT,CODE,GBS,LAGS,sha,read,write,now,need,clean,legacy
import numpy as np

def integral_crps(x,y):
    x=np.sort(x,axis=0);m=len(x)
    width=np.diff(x,axis=0)
    below=np.minimum(np.maximum(y-x[:-1],0),width)
    p=(np.arange(1,m)/m)[:,None]
    return np.maximum(x[0]-y,0)+np.maximum(y-x[-1],0)+np.sum(p*p*below+(1-p)**2*(width-below),axis=0)

class NativeCheck:
    def __init__(self):self.days={};self.local={}
    def get(self,meta):
        if meta['dataset']=='LOCAL':
            file=meta['source_file']
            if file not in self.local:
                df=pd.read_parquet(ROOT/file,columns=['hour_bjt','second_bjt','quality_zero_records','received_power_w','hour_mean_power_w'],filters=[('split','in',['validation','test'])])
                d={}
                for h,g in df.groupby('hour_bjt'):
                    pos=((g.second_bjt-h).dt.total_seconds()).to_numpy(dtype=int)
                    p=np.full(3600,np.nan);cnt=np.zeros(3600,int)
                    p[pos]=g.received_power_w;cnt[pos]=g.quality_zero_records
                    valid=(cnt>=9)&np.isfinite(p)&(p>0)
                    y=-10*np.log10(p/float(g.hour_mean_power_w.iloc[0]));y[~valid]=np.nan
                    d[h.isoformat()]=(y,valid)
                self.local[file]=d
            return self.local[file][meta['hour']]
        file=meta['source_file']
        if file not in self.days:
            with netCDF4.Dataset('qa_native',memory=(GBS/file).read_bytes()) as ds:
                ds.set_auto_mask(False);var=ds.variables['clearsky_attenuation']
                t=np.asarray(ds.variables['time'][:],float);a=np.asarray(var[:],float)
                ok=np.isfinite(a)&np.isfinite(t)&(t>0)&(t<=86400)&(a!=var._FillValue)&(a>=var.valid_min)&(a<=var.valid_max)
                day=np.full(86400,np.nan)
                for sec,val in zip(t[ok].astype(int),a[ok]):day[sec-1]=val
                self.days[file]=day
        h=pd.Timestamp(meta['hour']).hour;obs=self.days[file][h*3600:(h+1)*3600]
        return obs-meta['truth_db'],np.isfinite(obs)

def episodes(y,valid,threshold):
    lengths=[];censored=0;begin=None
    for i in range(len(y)+1):
        state=i<len(y) and valid[i] and y[i]>threshold
        if state and begin is None:begin=i
        if not state and begin is not None:
            left=begin==0 or not valid[begin-1]
            right=i==len(y) or not valid[i]
            if left or right:censored+=1
            else:lengths.append(i-begin)
            begin=None
    return lengths,censored

def check_one(receipt,native,deep):
    rec=read(receipt);path=receipt.with_suffix('.npz');need(sha(path)==rec['sha256'],'Hour artifact SHA')
    with np.load(path,allow_pickle=False) as z:
        import json
        meta=json.loads(str(z['metadata']));row=json.loads(str(z['row']));stats=json.loads(str(z['diagnostics']))
        need(meta==rec['identity'] and row==rec['row'],'Hour identity/row')
        g=z['generated_fast_db'];y=z['observed_fast_db'];v=z['valid'];need(g.shape==(32,3600) and np.isfinite(g).all(),'Ensemble dimensions/finiteness')
        observed,valid=native.get(meta)
        need(np.array_equal(valid,v) and np.allclose(observed[v],y[v],rtol=0,atol=1e-9),'Native observation reconstruction')
        delta=meta['prediction_db']-meta['truth_db']
        fast=integral_crps(g[:,v],y[v]);combined=integral_crps(g[:,v],y[v]-delta)
        maxerr=max(np.max(np.abs(fast-z['fast_crps_db'][v])),np.max(np.abs(combined-z['combined_crps_db'][v])))
        need(maxerr<2e-11,'Independent integral CRPS mismatch')
        need(fast.min()>=-1e-12 and combined.min()>=-1e-12,'Negative empirical CRPS')
        for key,expected in [('fast_crps_db',fast.mean()),('combined_crps_db',combined.mean()),('slow_only_crps_db',np.abs(y[v]-delta).mean()),('combined_minus_slow_db',(combined-np.abs(y[v]-delta)).mean())]:
            need(abs(row[key]-expected)<2e-11,'Hourly mean mismatch '+key)
        for key in ('fast_crps_db','combined_crps_db','slow_crps_db'):
            need(np.isnan(z[key][~v]).all(),'Missing seconds must not score')
        need(row['n_seconds']==int(v.sum()),'Effective second count')
        need(abs(row['observed_sd_db']-np.std(y[v]))<1e-11,'Observed SD')
        need(np.allclose(np.std(g[:,v],axis=1),stats['generated_sd_db'],rtol=0,atol=1e-11),'Per-realization SD')
        if deep:
            for i,lag in enumerate(LAGS):
                pairs=np.array([v[t:t+lag+1].all() for t in range(3600-lag)])
                need(pairs.sum()==stats['pair_counts'][i],'Independent gap-pair count')
                for rep in (None,0,31):
                    seq=y if rep is None else g[rep];left=seq[:-lag][pairs];right=seq[lag:][pairs]
                    actual=stats['observed_acf'][i] if rep is None else stats['generated_acf'][rep][i]
                    expected=np.corrcoef(left,right)[0,1] if len(left)>=3 and np.std(left)>0 and np.std(right)>0 else np.nan
                    need((actual is None and not np.isfinite(expected)) or (actual is not None and abs(actual-expected)<2e-10),'Independent ACF')
            for tag in ('95','99'):
                for rep in (None,0,31):
                    obj=stats['observed_tail'+tag] if rep is None else stats['generated_tail'+tag][rep]
                    lengths,cens=episodes(y if rep is None else g[rep],v,obj['threshold_db'])
                    need(lengths==obj['complete_fades_s'] and cens==obj['censored_fades'],'Independent fade/censoring')
        return {'status':'PASS','hour_sha256':rec['sha256'],'hour_identity':meta,'n_seconds':int(v.sum()),'max_crps_error':float(maxerr),'deep_dynamics_checked':deep,'utc':now()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--follow',action='store_true');args=parser.parse_args()
    dest=OUT/'independent_hour_qa';dest.mkdir(exist_ok=True);native=NativeCheck();deepgroups=set();idle=0
    while True:
        pending=[p for p in (OUT/'hours').glob('*.json') if not (dest/p.name).exists()]
        for p in pending:
            m=read(p)['identity'];key=(m['dataset'],m['split'],m['station'],m['event'])
            deep=key not in deepgroups
            qa=check_one(p,native,deep);write(dest/p.name,qa);deepgroups.add(key)
        done=list(dest.glob('*.json'))
        write(OUT/'independent_qa_status.json',{'state':'AUDITING','verified_hours':len(done),'total_hours':6623,'utc':now()},replace=True)
        if (OUT/'computation_complete.json').exists() and len(done)==6623:break
        if not args.follow:print('Partial independent QA',len(done),flush=True);return
        status=read(OUT/'status.json')
        need(status['state']!='FAILED_REQUIRES_INSPECTION','Computation failed: preserve partial audit')
        idle=idle+1 if not pending else 0
        need(idle<120,'No new computed hours for10min; inspection required')
        time.sleep(5)
    records=[read(p) for p in done];manifest=read(OUT/'manifest.json')
    for p,h in manifest['sources'].items():need(sha(ROOT/p)==h,'Protected source changed '+p)
    frame=pd.read_csv(OUT/'hourly_results.csv');need(len(frame)==6623,'Result count')
    need(frame.groupby(['dataset','split']).size().to_dict()=={('LOCAL','validation'):343,('LOCAL','test'):687,('UK','external'):5593},'Split counts')
    # Verify generator replay for each observed dataset/split/station/weather combination.
    mod,dyn,saved,models,libraries,fm=legacy();replays=0
    for rec in records:
        if not rec['deep_dynamics_checked']:continue
        m=rec['hour_identity'];ident=__import__('hashlib').sha256((m['dataset']+'|'+m['split']+'|'+m['station']+'|'+m['hour']).encode()).hexdigest()[:20]
        with np.load(OUT/'hours'/(ident+'.npz')) as z:
            for rep in (0,31):
                q,_=dyn.generate('block_0256',models[m['event']],libraries[m['event']],-1,m['key'],rep)
                need(np.array_equal(-10*np.log10(q),z['generated_fast_db'][rep]),'Frozen generator replay');replays+=1
    result={'status':'PASS','utc':now(),'verified_hours':len(records),'verified_second_scores_each_kind':sum(r['n_seconds'] for r in records),'verified_score_kinds':['fast','combined'],'source_native_reconstructed_all_hours':True,'max_crps_error':max(r['max_crps_error'] for r in records),'deep_dynamics_hours':sum(r['deep_dynamics_checked'] for r in records),'generator_replayed_realizations':replays,'sources_unchanged':True,'new_fits':0,'auditor_sha256':sha(CODE/'independent_qa.py')}
    write(OUT/'independent_numeric_qa.json',result);write(OUT/'independent_qa_status.json',result,replace=True);print(result,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        from datetime import datetime,timezone
        write(OUT/'failures'/('independent_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),{'utc':now(),'traceback':traceback.format_exc()});raise
