"""Derive integer-lag ACF(1..1800 s) from saved Fast003 curves, no generation.
Pearson correlations use only pairs whose entire intervening interval is valid.
FFT cross-products are computed separately inside each continuous valid run;
prefix sums supply the pair-specific means/variances. Every old saved lag is
checked against the frozen per-hour diagnostic before aggregation.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
LAGS=np.arange(1,1801);OLD=np.array([1,2,5,10,30,60,120,300,600,900,1800])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def calculate(y,valid):
 # Center once for numerical conditioning; pairwise Pearson is shift invariant.
 y=y-y[:,valid].mean(1)[:,None]
 edges=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(np.int8)))
 n=np.zeros(1800);cross=np.zeros((len(y),1800));sl=cross.copy();sr=cross.copy();ql=cross.copy();qr=cross.copy()
 for a,b in edges.reshape(-1,2):
  z=y[:,a:b];length=b-a;cap=min(1800,length-1)
  if cap<1:continue
  lag=LAGS[:cap];n[:cap]+=length-lag
  sums=np.c_[np.zeros(len(y)),np.cumsum(z,axis=1)];sqs=np.c_[np.zeros(len(y)),np.cumsum(z*z,axis=1)]
  sl[:,:cap]+=sums[:,length-lag];sr[:,:cap]+=sums[:,[-1]]-sums[:,lag]
  ql[:,:cap]+=sqs[:,length-lag];qr[:,:cap]+=sqs[:,[-1]]-sqs[:,lag]
  nf=1<<int(2*length-1).bit_length();ft=np.fft.rfft(z,n=nf,axis=1)
  cross[:,:cap]+=np.fft.irfft(ft*np.conj(ft),n=nf,axis=1)[:,1:cap+1]
 with np.errstate(divide='ignore',invalid='ignore'):
  cov=cross-sl*sr/n;vl=np.maximum(ql-sl*sl/n,0);vr=np.maximum(qr-sr*sr/n,0);den=np.sqrt(vl*vr)
  acf=np.divide(cov,den,out=np.full_like(cov,np.nan),where=(den>0)&(n>=3))
 return acf,n.astype(int)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=ROOT);ap.add_argument('--package',type=Path,default=ROOT/'最终提交的修改材料/论文图表_20261030_001');a=ap.parse_args()
 p=a.package;src=a.project/'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble/hours';qa=p/'05_QA';files=sorted(src.glob('*.npz'));assert len(files)==6623
 receipts=json.loads((qa/'Fast003_hour_receipts.json').read_text(encoding='utf8'))
 # Exact shape of this manifest is checked rather than inferred from a PID/job.
 receiptmap={Path(r['path']).name:r['sha256'] for r in receipts}
 agg=defaultdict(lambda:np.zeros((5,1800)));maxerr=0.;started=time.time();counts=defaultdict(int)
 for i,f in enumerate(files):
  assert sha(f)==receiptmap[f.name],f'SHA changed: {f.name}'
  with np.load(f,allow_pickle=False) as z:
   meta=json.loads(str(z['metadata']));diag=json.loads(str(z['diagnostics']));valid=z['valid'].astype(bool)
   values=np.vstack([z['observed_fast_db'][None,:],z['generated_fast_db']]).astype(float)
  c,n=calculate(values,valid);old=np.vstack([np.array(diag['observed_acf'],dtype=float)[None,:],np.array(diag['generated_acf'],dtype=float)])
  known=c[:,OLD-1];assert np.array_equal(np.isnan(known),np.isnan(old)),f'NaN pattern {f.name}'
  error=np.nanmax(np.abs(known-old));maxerr=max(maxerr,float(error));assert error<2e-7,(f.name,error)
  assert np.array_equal(n[OLD-1],np.array(diag['pair_counts'])),f'Pair counts {f.name}'
  with np.errstate(invalid='ignore'):
   finite=np.isfinite(c[1:]);den=finite.sum(0);gm=np.divide(np.nansum(c[1:],axis=0),den,out=np.full(1800,np.nan),where=den>0)
  obs=c[0];good=np.isfinite(obs)&np.isfinite(gm)
  groups=[('LOCAL_'+meta['split'],'local')] if meta['dataset']=='LOCAL' else [('UK',meta['station']),('UK','pooled')]
  for cohort,station in groups:
   for event in [meta['event'],'overall']:
    key=(cohort,station,event);r=agg[key];r[0,good]+=obs[good];r[1,good]+=gm[good];r[2,good]+=1;r[3,good]+=np.abs(obs[good]-gm[good]);r[4,good]+=n[good];counts[key]+=1
  if (i+1)%500==0:print(f'ACF {i+1}/{len(files)}; max saved-lag discrepancy={maxerr:.3g}',flush=True)
 rows=[]
 for (cohort,station,event),v in sorted(agg.items()):
  for j,lag in enumerate(LAGS):
   count=int(v[2,j]);rows.append(dict(cohort=cohort,station=station,event=event,lag_seconds=int(lag),hours_with_defined_acf=count,observed_acf=v[0,j]/count if count else np.nan,generated_acf_mean=v[1,j]/count if count else np.nan,acf_mae=v[3,j]/count if count else np.nan,valid_pair_count=int(v[4,j])))
 frame=pd.DataFrame(rows);old=pd.read_csv(p/'04_Source_Data/frozen_exports/fast_acf.csv')
 merged=old.merge(frame,on=['cohort','station','event','lag_seconds'],suffixes=('_old','_dense'),validate='one_to_one')
 for col in ['observed_acf','generated_acf_mean','acf_mae']:assert np.allclose(merged[col+'_old'],merged[col+'_dense'],atol=1e-9,rtol=0,equal_nan=True),col
 for col in ['hours_with_defined_acf','valid_pair_count']:assert np.array_equal(merged[col+'_old'],merged[col+'_dense']),col
 frame.to_csv(p/'04_Source_Data/Fig10_dense_acf_1_to_1800.csv',index=False)
 report={'status':'PASS','hours':len(files),'lag_range':[1,1800],'rows':len(frame),'saved_lag_checks_per_hour':11,'max_per_realization_old_lag_discrepancy':maxerr,'aggregate_275_old_rows_match':True,'all_input_sha_verified':True,'elapsed_seconds':time.time()-started,'method':'FFT within complete valid runs; pair-specific Pearson means/variances; no interpolation across lags and no pair bridging missing seconds; equal paired-hour aggregation.','implementation_sha256':sha(Path(__file__))}
 (qa/'dense_acf_qa.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
