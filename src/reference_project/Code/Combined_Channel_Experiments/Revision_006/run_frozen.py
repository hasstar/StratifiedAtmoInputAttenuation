"""Local-only author-pretrained inference. No parameter-training implementation."""
from __future__ import annotations
import argparse, hashlib, json, os, socket, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_006'
DATA=ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003'
OLD=ROOT/'Outputs/Experiments/Combined_Channel_Revision_002'
EXP=ROOT/'Outputs/Experiments/Combined_Channel_Revision_003'
PHY=ROOT/'Outputs/Experiments/Combined_Channel_Revision_004'
R29=ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
STOP=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
sys.path.insert(0,str(HERE/'vendor'))
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',
                  TABPFN_ALLOW_CPU_LARGE_DATASET='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',
                  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
import pandas as pd
import torch
import sklearn
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from threadpoolctl import threadpool_limits

TARGET='clear_sky_attenuation_db'
SPECS={
 'tabpfn_v2_author_frozen':dict(file='tabpfn-v2-regressor.ckpt',sha256='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736',bytes=44390977,version='2.0.9'),
 'tabicl_v2_author_frozen':dict(file='tabicl-regressor-v2-20260212.ckpt',sha256='0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a',bytes=114324594,version='2.2.0')}
COUNTERS=dict(gradient_calls=0,optimizer_constructions=0,network_calls=0,context_preparations=0)
LOADS=[]

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def emit(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False,default=str);f.write('\n')
def need(ok,msg):
 if not ok: raise RuntimeError(msg)
def arraysha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()
def parameter_hash(net):
 h=hashlib.sha256()
 for k,v in sorted(net.named_parameters()):
  a=v.detach().cpu().contiguous().numpy()
  h.update(k.encode());h.update(arraysha(a).encode())
 return h.hexdigest()
def block(kind):
 def fail(*a,**k):
  COUNTERS[kind]+=1
  raise RuntimeError('Forbidden operation: '+kind)
 return fail
def guards():
 torch.autograd.backward=block('gradient_calls');torch.autograd.grad=block('gradient_calls')
 torch.Tensor.backward=block('gradient_calls');torch.optim.Optimizer.__init__=block('optimizer_constructions')
 socket.create_connection=block('network_calls');socket.socket.connect=block('network_calls');socket.socket.connect_ex=block('network_calls')
 original=torch.nn.Module.load_state_dict
 def frozen_load(self,*args,**kwargs):
  result=original(self,*args,**kwargs)
  self.requires_grad_(False)
  LOADS.append(dict(type=type(self).__name__,parameters=sum(v.numel() for v in self.parameters()),sha256=parameter_hash(self)))
  return result
 torch.nn.Module.load_state_dict=frozen_load

def source_paths():
 paths={STOP,R29/'rfe/models/k_0010.joblib',R29/'rfe/states/k_0010.json',R29/'run_manifest.json'}
 paths.update(DATA/f'{s}.parquet' for s in ('train','validation','test'))
 paths.update((DATA/'hourly_sample_status.parquet',DATA/'manifest.json'))
 for directory in (OLD,EXP,PHY,ROOT/'Outputs/Experiments/Combined_Channel_Revision_005',HERE/'weights',HERE/'vendor'):
  paths.update(p for p in directory.rglob('*') if p.is_file() and p.suffix not in ('.lock','.pyc'))
 for revision in ('Revision_002','Revision_003','Revision_004','Revision_005'):
  paths.update(p for p in (HERE.parent/revision).rglob('*') if p.is_file() and p.suffix!='.pyc')
 return sorted(paths)

def prepare():
 need(not (OUT/'preflight.json').exists(),'Preflight already exists')
 need(sha(STOP)=='0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a','Stop flag changed')
 packages=[]
 for meta in sorted((OUT/'acquisition_002/references').glob('*_pypi.json')):
  d=read(meta);name=d['info']['name'];version=d['info']['version']
  for w in d['urls']:
   path=HERE/'artifacts'/w['filename']
   if not path.exists():continue
   need(sha(path)==w['digests']['sha256'] and path.stat().st_size==w['size'],'Wheel SHA/size')
   with zipfile.ZipFile(path) as z:
    for member in z.infolist():
     if member.is_dir():continue
     need(sha(HERE/'vendor'/member.filename)==hashlib.sha256(z.read(member)).hexdigest(),'Changed author/package bytes '+member.filename)
   packages.append(dict(package=name,version=version,sha256=sha(path),bytes=path.stat().st_size,unchanged_members=len(z.infolist())))
 need(len(packages)==5,'Expected isolated five wheels')
 weights=[]
 for model,spec in SPECS.items():
  p=HERE/'weights'/spec['file']
  need(sha(p)==spec['sha256'] and p.stat().st_size==spec['bytes'],'Weight SHA/size')
  unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(p);need(not unsafe,'Unsafe checkpoint globals')
  d=torch.load(p,map_location='cpu',weights_only=True)
  need(set(d)=={'state_dict','config'},'Unexpected checkpoint structure')
  weights.append(dict(model=model,**spec,unsafe_globals=unsafe,config=d['config']))
 existing=read(OLD/'frozen_selection.json')['models'];old_audit=[]
 for row in existing:
  if 'path' in row:
   p=OLD/row['path'];need(sha(p)==row['sha256'],'Historical weight altered')
   old_audit.append(dict(model=row['id'],sha256=sha(p),provenance='our_LOCAL_trained_historical_comparator',action='PRESERVED_NO_RETRAINING',author_weight_restored=False))
 sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
 emit(OUT/'source_sha256_before.json',sources)
 features=read(R29/'rfe/states/k_0010.json')['feature_names']
 need(len(features)==10 and len(set(features))==10,'Natural Top10 order')
 emit(OUT/'preflight.json',dict(utc=now(),packages=packages,weights=weights,features=features,source_count=len(sources),historical_models=old_audit,
    training_allowed=False,weight_updates_allowed=False,context='2404 original LOCAL train hours only',query_chunk_size=512,torch_threads=4,
    overrides={'both':'CPU; n_jobs=1; explicit pinned model path; no network; no gradient',
               'tabicl':'allow_auto_download=False; otherwise defaults',
               'tabpfn':'TABPFN_ALLOW_CPU_LARGE_DATASET=1 is documented speed-warning override, not a model/training limit change; otherwise defaults'},
    selected_from_scores=False))
 print(json.dumps(dict(preflight='PASS',sources=len(sources),historical_models=len(old_audit))),flush=True)

def frames(features):
 all_frames={}
 status=pd.read_parquet(DATA/'hourly_sample_status.parquet')
 for split,n in [('train',2404),('validation',343),('test',687)]:
  f=pd.read_parquet(DATA/f'{split}.parquet',columns=['hour_bjt',*features,TARGET]).sort_values('hour_bjt').reset_index(drop=True)
  need(len(f)==n and f.hour_bjt.is_unique and np.isfinite(f[TARGET]).all(),'LOCAL schema/target')
  check=f[['hour_bjt']].merge(status[['hour_bjt','split','use_for_regression']],on='hour_bjt',validate='one_to_one')
  need(check.split.eq(split).all() and check.use_for_regression.all(),'LOCAL split gate')
  all_frames[split]=f
 for a,b in [('train','test'),('train','validation'),('test','validation')]:
  need(not set(all_frames[a].hour_bjt)&set(all_frames[b].hour_bjt),'LOCAL overlap')
 uk=pd.read_parquet(EXP/'expanded_dataset.parquet')
 need(len(uk)==5664 and uk.primary_score_eligible.sum()==5593,'UK fixed cohort')
 need(not uk.duplicated(['station','utc_epoch_seconds']).any(),'UK duplicate keys')
 for f in [*all_frames.values(),uk]: need(np.isfinite(f[features].to_numpy()).all(),'Nonfinite feature')
 all_frames['uk']=uk
 return all_frames

def run(model):
 import msvcrt
 directory=OUT/model
 directory.mkdir(exist_ok=False)
 lock=(OUT/'inference.lock').open('a+b');lock.seek(0)
 if lock.read(1)==b'':lock.write(b'0');lock.flush()
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 start=time.monotonic()
 try:
  pre=read(OUT/'preflight.json')
  baseline=read(OUT/'source_sha256_before.json')
  for p,digest in baseline.items():need(sha(ROOT/p)==digest,'Pre-inference source drift '+p)
  emit(directory/'launch.json',dict(utc=now(),pid=os.getpid(),command=sys.argv,code_sha256=sha(__file__),preflight_sha256=sha(OUT/'preflight.json'),model=model))
  torch.set_num_threads(4);torch.set_num_interop_threads(1)
  guards()
  features=pre['features'];f=frames(features)
  train_x=f['train'][features].to_numpy(dtype=np.float64);train_y=f['train'][TARGET].to_numpy(dtype=np.float64)
  weights=HERE/'weights'/SPECS[model]['file']
  kwargs=dict(model_path=str(weights),device='cpu',n_jobs=1)
  estimator=TabPFNRegressor(**kwargs) if model.startswith('tabpfn') else TabICLRegressor(**kwargs,allow_auto_download=False)
  emit(directory/'fixed_settings.json',dict(utc=now(),parameters=estimator.get_params(deep=False),context_x_sha256=arraysha(train_x),context_y_sha256=arraysha(train_y),context_keys_sha256=arraysha(f['train'].hour_bjt.astype('int64').to_numpy()),feature_names=features))
  with threadpool_limits(limits=4),torch.inference_mode():
   COUNTERS['context_preparations']+=1
   estimator.fit(train_x,train_y)
   net=estimator.model_
   need(not net.training and all(not p.requires_grad for p in net.parameters()),'Model is not frozen eval')
   original=torch.load(weights,map_location='cpu',weights_only=True)['state_dict']
   for name,p in net.named_parameters():need(name in original and torch.equal(p.cpu(),original[name]),'Author parameter mismatch '+name)
   del original
   before=parameter_hash(net)
   need(any(row['sha256']==before and row['parameters']>0 for row in LOADS),'Parameter change during context preparation')
   emit(directory/'context_ready.json',dict(utc=now(),elapsed_seconds=time.monotonic()-start,learned_parameters_sha256=before,loads=LOADS,counters=COUNTERS,parameters=sum(p.numel() for p in net.parameters()),context_rows=len(train_y)))
   print(model+' CONTEXT_READY',flush=True)
   pilot=f['validation'].iloc[:16][features].to_numpy(dtype=np.float64)
   t=time.monotonic();p16=np.asarray(estimator.predict(pilot),dtype=float)
   p8=np.r_[estimator.predict(pilot[:8]),estimator.predict(pilot[8:])]
   delta=float(np.max(np.abs(p16-p8)))
   need(p16.shape==(16,) and np.isfinite(p16).all(),'Pilot output')
   need(np.allclose(p16,p8,rtol=1e-5,atol=1e-5),'Query chunk non-equivalence')
   emit(directory/'pilot.json',dict(utc=now(),validation_hours=f['validation'].iloc[:16].hour_bjt.astype(str).tolist(),n=16,seconds=time.monotonic()-t,chunk_max_abs_delta_db=delta,success=True,no_parameter_selection=True,predictions=p16.tolist()))
   print(model+' PILOT_PASS seconds='+str(round(time.monotonic()-t,2)),flush=True)
   for label,frame,keys in [('local_test',f['test'],['hour_bjt']),('uk_expanded',f['uk'],['station','utc_epoch_seconds'])]:
    predictions=[]
    for offset in range(0,len(frame),512):
     segment=frame.iloc[offset:offset+512]
     p=np.asarray(estimator.predict(segment[features].to_numpy(dtype=np.float64)),dtype=np.float64)
     need(p.shape==(len(segment),) and np.isfinite(p).all(),'Prediction shape/finite')
     part=segment[keys].copy();part[model]=p
     part.to_parquet(directory/f'{label}_chunk_{offset:05}.parquet',index=False)
     predictions.extend(p.tolist())
     with (directory/'progress.ndjson').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(utc=now(),dataset=label,done=len(predictions),total=len(frame),elapsed_seconds=time.monotonic()-start))+'\n')
     print(model+' '+label+' '+str(len(predictions))+'/'+str(len(frame)),flush=True)
    result=frame[keys].copy();result[model]=predictions
    result.to_parquet(directory/(label+'.parquet'),index=False)
    need(parameter_hash(net)==before,'Learned parameters changed')
   need(all(COUNTERS[k]==0 for k in ('gradient_calls','optimizer_constructions','network_calls')),'Forbidden operation attempted')
   need(sha(weights)==SPECS[model]['sha256'],'Checkpoint bytes changed')
   for p,digest in baseline.items():need(sha(ROOT/p)==digest,'Post-inference source drift '+p)
   emit(directory/'complete.json',dict(utc=now(),model=model,elapsed_seconds=time.monotonic()-start,counters=COUNTERS,
       original_weight_sha256=sha(weights),parameter_sha256_before=before,parameter_sha256_after=parameter_hash(net),
       original_parameters_exact_match=True,old_sources_unchanged=len(baseline),local_predictions=687,uk_predictions=5664,
       training_steps=0,finetune_steps=0,optimizer_steps=0,author_source_modified=False,sklearn=sklearn.__version__,torch=torch.__version__))
  print(model+' COMPLETE',flush=True)
 except Exception as e:
  emit(directory/'failure.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc(),counters=COUNTERS))
  raise
 finally:
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--model',choices=list(SPECS));a=p.parse_args()
 if a.action=='prepare':prepare()
 else:
  need(a.model is not None,'Specify model');run(a.model)
