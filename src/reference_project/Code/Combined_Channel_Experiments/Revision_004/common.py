"""Versioned, append-only arithmetic/verification helpers; no model imports."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_004'
OLD = ROOT/'Outputs/Experiments/Combined_Channel_Revision_003'
REV2 = ROOT/'Outputs/Experiments/Combined_Channel_Revision_002'
ACQ = ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011'
AUTOMATION = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
INDEXES = [ROOT/'Tasks/T012_Combined_Channel_And_External_Validation.md',
           ROOT/'Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md']

def require(ok, message):
    if not ok: raise RuntimeError(message)

def now(): return datetime.now(timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def newpath(path):
    p=Path(path)
    require(p.resolve().is_relative_to(OUT.resolve()),'Output outside Revision004')
    require(not p.exists(),'Refusing overwrite: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def emit(path,obj):
    with newpath(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def deny(*args,**kwargs): raise RuntimeError('Offline arithmetic only; network forbidden')
def offline():
    socket.socket.connect=deny;socket.create_connection=deny

@contextlib.contextmanager
def lock():
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'evaluation.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def protected_verify(guard):
    for name,digest in guard.items(): require(sha(ROOT/name)==digest,'Protected source changed: '+name)

def snapshot():
    guard=read(OLD/'source_sha256_used.json')
    protected_verify(guard)
    for directory in (OLD,ROOT/'Code/Combined_Channel_Experiments/Revision_003',
                      ROOT/'Code/Hourly_Attenuation_Experiments/Revision_016',
                      ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline'):
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix!='.lock' and '__pycache__' not in p.parts:
                key=str(p.relative_to(ROOT));digest=sha(p)
                require(key not in guard or guard[key]==digest,'Previous guard conflict')
                guard[key]=digest
    for p in (AUTOMATION,ACQ/'STOP_REQUESTED.json'):
        guard[str(p)]=sha(p)
    require('PAUSED' in AUTOMATION.read_text(encoding='utf-8'),'tvt must remain paused')
    for p in INDEXES:
        with newpath(OUT/'document_snapshots/before'/p.relative_to(ROOT)).open('xb') as f:f.write(p.read_bytes())
    emit(OUT/'source_sha256_before.json',guard)
    emit(OUT/'index_sha256_before.json',{str(p.relative_to(ROOT)):sha(p) for p in INDEXES})
    emit(OUT/'launch.json',{'time':now(),'pid':os.getpid(),'code_sha256':{p.name:sha(p) for p in HERE.iterdir() if p.is_file()},'guard_count':len(guard),'weather_downloads':0,'training_calls':0,'ml_prediction_calls':0,'automation_sha256':sha(AUTOMATION),'stop_sha256':sha(ACQ/'STOP_REQUESTED.json')})
    return guard
