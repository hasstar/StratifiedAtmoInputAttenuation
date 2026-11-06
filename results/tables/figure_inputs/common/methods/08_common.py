"""Isolated T016 utilities; no import-time writes or model fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, contextlib, msvcrt, importlib.util, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT/'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble'
DELIVERY = ROOT/'最终提交的修改材料/补充材料_20261024_001'
FAST = ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
RFE = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002'
SLOW = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE/models/tabm/k_0010'
UK = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE'
WEATHER = ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet'
GBS = ROOT/'外部样例数据CEDA/GBS_20p7GHz'
LAGS = (1,2,5,10,30,60,120,300,600,900,1800)

def now(): return datetime.now(timezone.utc).isoformat()
def need(ok, message):
    if not ok: raise RuntimeError(message)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def relative(p): return str(Path(p).resolve().relative_to(ROOT)).replace('\\','/')
def write(p, data, replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
@contextlib.contextmanager
def lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'execution.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
def legacy():
    p=ROOT/'Code/Combined_Channel_Experiments/Revision_001/evaluate_gbs_dynamics_v002.py'
    spec=importlib.util.spec_from_file_location('t016_readonly_frozen_loader',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    dyn,saved,models,libraries=mod.load_frozen()
    import fit_marginals as fm
    return mod,dyn,saved,models,libraries,fm
def crps(g,y):
    m=len(g);s=np.sort(g,axis=0)
    spread=np.sum((2*np.arange(1,m+1)-m-1)[:,None]*s,axis=0)/(m*m)
    first=np.mean(np.abs(g-y),axis=0)
    return first-spread, first-spread*m/(m-1)
