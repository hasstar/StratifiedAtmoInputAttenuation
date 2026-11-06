"""Execution-only adapter; original frozen T014 scientific functions stay unchanged."""
from __future__ import annotations
import argparse, ctypes, json, os, shutil, sys, threading, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parent/'Revision_032'))
import runtime as r
import run as original
AUDIT=r.ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_033_Execution_Repair'

def sharing(error):
    return isinstance(error,PermissionError) and getattr(error,'winerror',None) in (5,32,33)

def atomic_state(path,value,replace_fn=os.replace,sleep_fn=time.sleep,attempts=14):
    """Only execution JSON. Keep failed temporary artifacts for inspection."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.r033.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    for attempt in range(attempts):
        try:
            replace_fn(temp,path)
            return
        except OSError as error:
            if not sharing(error) or attempt==attempts-1: raise
            sleep_fn(min(.05*2**attempt,.5))

def record_deferred(obj,phase,error):
    payload=dict(progress=obj,phase=phase,error=repr(error),reason='Non-scientific telemetry deferred after bounded Windows sharing retries',utc=r.now())
    path=AUDIT/'deferred_progress'/f'{time.time_ns()}.json'
    r.emit(path,payload)
    print('TELEMETRY_DEFERRED',phase,str(path),flush=True)

def progress(**kw):
    obj=dict(utc=r.now(),pid=os.getpid(),**kw)
    try: atomic_state(r.OUT/'progress.json',obj)
    except OSError as error:
        if not sharing(error): raise
        record_deferred(obj,'progress_replace',error)
    # The event is unique even if the convenient latest-progress view is delayed.
    for attempt in range(14):
        try:
            stream=(r.OUT/'progress_events.jsonl').open('a',encoding='utf-8')
            break
        except OSError as error:
            if not sharing(error): raise
            if attempt==13:
                record_deferred(obj,'event_open',error)
                return
            time.sleep(min(.05*2**attempt,.5))
    with stream:
        stream.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()

def update(path,value):
    atomic_state(path,value)
    if Path(path).name=='active_worker.json' and value.get('status')=='RUNNING':
        atomic_state(r.OUT/'status.json',dict(status='RUNNING',execution_revision='033',pid=value['pid'],utc=r.now()))

def tests():
    folder=AUDIT/'tests'; folder.mkdir(parents=True,exist_ok=False)
    target=folder/'state.json'; atomic_state(target,dict(version=1))
    r.require(r.read(target)==dict(version=1),'Initial JSON write')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    kernel.CreateFileW.restype=ctypes.c_void_p; kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateFileW(str(target),0x80000000,1|2,None,3,0x80,None)
    r.require(handle not in (None,ctypes.c_void_p(-1).value),'Windows synthetic read handle')
    legacy_failed=False
    try:
        try: r.c.update(target,dict(version=2))
        except OSError as e:
            r.require(sharing(e),'Expected Windows replace sharing error'); legacy_failed=True
        r.require(legacy_failed and r.read(target)==dict(version=1),'Reproduce old error without changing old target')
        closer=threading.Thread(target=lambda:(time.sleep(.2),kernel.CloseHandle(handle)))
        closer.start(); atomic_state(target,dict(version=3)); closer.join(); handle=None
        r.require(r.read(target)==dict(version=3),'New atomic retry succeeds after handle release')
    finally:
        if handle is not None: kernel.CloseHandle(handle)
    def denied(*args):
        e=PermissionError(13,'synthetic sharing'); e.winerror=5; raise e
    try: atomic_state(target,dict(version=4),replace_fn=denied,sleep_fn=lambda x:None,attempts=2)
    except PermissionError: pass
    else: raise RuntimeError('Persistent denial was swallowed')
    r.require(r.read(target)==dict(version=3),'Persistent failure preserves previous state')
    def other_error(*args): raise OSError(22,'synthetic non-sharing failure')
    try: atomic_state(target,dict(version=5),replace_fn=other_error,sleep_fn=lambda x:None)
    except OSError as e: r.require(e.errno==22,'Non-sharing error fidelity')
    else: raise RuntimeError('Unrelated error swallowed')
    # Explicitly exercise diagnostic preservation for a permanently deferred view.
    record_deferred(dict(stage='SYNTHETIC_TEST_ONLY'),'synthetic_persistent_lock',PermissionError(13,'synthetic'))
    return dict(status='PASS',synthetic_only=True,research_fit_calls=0,legacy_windows_error_reproduced=True,retry_after_real_handle_release=True,
                persistent_failure_preserves_target=True,unrelated_errors_propagate=True,deferred_diagnostics_saved=True)

def checkpoint_identity():
    m=r.manifest(); data=r.frames(); previous=r.read(r.state_dir('tabm',155)/'state.json')
    names=[n for n in previous['feature_names'] if n!=previous['removed_feature_next']]
    f=r.r29.folds(data['train'])[0]; fit=data['train'].iloc[f['fit']]; hold=data['train'].iloc[f['hold']]
    identity=r.fp(dict(manifest=m['fingerprint'],model='tabm',k=154,fold=1,names=names,
                       train=fit.hour_bjt.astype(str).tolist(),hold=hold.hour_bjt.astype(str).tolist(),config=r.config('tabm'),search=False))
    path=r.OUT/'fits/tabm/k_0154_fold1/recovery.pt'; cp=r.c.load_tensor(path)
    r.require(cp['identity']==identity and cp['epoch']==35 and len(cp['history'])==35,'Exact checkpoint identity and epoch')
    r.require(all(k in cp for k in ('rng','generator','optimizer','weights')),'Checkpoint optimizer/RNG completeness')
    b=r.make_nn('tabm',names); b.load_state_dict(cp['weights'],strict=True)
    return dict(path=r.rel(path),sha256=r.sha(path),identity=identity,epoch=35,history_length=35,source_fingerprint=m['fingerprint'])

def prepare():
    AUDIT.mkdir(parents=True,exist_ok=True)
    r.require(not (AUDIT/'repair_manifest.json').exists(),'Repair already frozen')
    qa=tests(); cp=checkpoint_identity()
    evidence=AUDIT/'preserved_before_resume'; evidence.mkdir(exist_ok=False)
    targets=[r.OUT/'active_worker.json',r.OUT/'status.json',r.OUT/'progress.json',r.OUT/'.progress.json.9228.tmp',r.ROOT/cp['path'],
             r.OUT/'logs/20261019_023401_stderr.log',*sorted((r.OUT/'failures').glob('*.json'))]
    copied={}
    for index,source in enumerate(targets):
        dest=evidence/f'{index:02d}_{source.name}'; shutil.copy2(source,dest)
        r.require(r.sha(source)==r.sha(dest),'Preserved evidence SHA'); copied[r.rel(source)]=dict(copy=r.rel(dest),sha256=r.sha(source))
    committed={}
    for file in (r.OUT/'models').glob('*/k_*/qa.json'):
        q=r.read(file); r.require(q['status']=='PASS','Prior QA gate')
        for name,key in (('state.json','state_sha256'),('evaluation.json','evaluation_sha256')):
            p=file.parent/name; r.require(r.sha(p)==q[key],'Prior committed data digest'); committed[r.rel(p)]=q[key]
        committed[r.rel(file)]=r.sha(file)
    r.require(len(committed)==60,'Exactly20 accepted states at recovery boundary')
    sources={r.rel(p):r.sha(p) for p in HERE.iterdir() if p.is_file()}
    m=dict(kind='execution_only_windows_state_retry',sources=sources,base_manifest_sha256=r.sha(r.OUT/'manifest.json'),checkpoint=cp,
           preserved=copied,prior_committed_sha256=committed,scientific_contract_changed=False,utc=r.now())
    r.emit(AUDIT/'repair_manifest.json',m)
    r.emit(AUDIT/'QA.json',dict(qa,checkpoint_identity_verified=True,all_1028_pinned_sources_unchanged=True,prior_20_states_verified=True,repair_manifest_sha256=r.sha(AUDIT/'repair_manifest.json'),utc=r.now()))
    print('EXECUTION_REPAIR_PREPARED',r.sha(AUDIT/'repair_manifest.json'),flush=True)

def run():
    m=r.read(AUDIT/'repair_manifest.json'); q=r.read(AUDIT/'QA.json')
    r.require(q['status']=='PASS' and q['repair_manifest_sha256']==r.sha(AUDIT/'repair_manifest.json'),'Repair QA gate')
    for file,digest in m['sources'].items(): r.require(r.sha(r.ROOT/file)==digest,'Repair source changed')
    r.require(r.sha(r.OUT/'manifest.json')==m['base_manifest_sha256'],'Scientific base manifest changed')
    for file,digest in m['prior_committed_sha256'].items(): r.require(r.sha(r.ROOT/file)==digest,'Accepted prior state changed')
    r.manifest()  # every original input/code SHA still checked
    r.update=update; r.progress=progress
    original.run()  # original OS lock, fitting, selection, evaluation, QA and reporting

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run']); args=parser.parse_args()
    prepare() if args.command=='prepare' else run()
