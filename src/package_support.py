"""Package paths, configuration, provenance and immutable reference adapters.

The distinct module name avoids the legacy scientific module named support.
"""
from __future__ import annotations
import contextlib, hashlib, importlib.util, json, os, platform, subprocess, sys, types
from datetime import datetime, timezone
from pathlib import Path
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

def now(): return datetime.now(timezone.utc).isoformat()
def config(): return yaml.safe_load((PACKAGE/'config.yaml').read_text('utf-8'))
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,default=str)
def rel(p): return Path(p).absolute().relative_to(PACKAGE).as_posix()
def timestamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def path(name):
    override=os.environ.get('TVT_FIGURE_INPUTS') if name=='figure_inputs' else None
    p=Path(override).absolute() if override else PACKAGE/config()['paths'][name]
    if override:p.relative_to(PACKAGE)
    return p
def original(relative):
    s=str(relative).replace('\\','/')
    if s.startswith('Code/'): return PACKAGE/'src/reference_project'/s
    if s.startswith('Outputs/'): return path('reference_results')/s
    prefix='外部样例数据CEDA/GBS_20p7GHz/'
    if s.startswith(prefix): return path('ceda_raw')/s[len(prefix):]
    return PACKAGE/'results/logs/provenance/project'/s
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

class ProjectRoot:
    """Resolve historical project-relative paths without OS links or privileges."""
    def __truediv__(self, value): return original(value)
    def __fspath__(self):return str(PACKAGE/'src/reference_project')
    def __str__(self):return self.__fspath__()
    @property
    def anchor(self):return PACKAGE.anchor
    def resolve(self):return self

def adapt_paths(module):
    """Rebind path constants in memory; the shipped source bytes stay unchanged."""
    prefix=PACKAGE/'src/reference_project';seen=set()
    def visit(mod):
        if id(mod) in seen:return
        seen.add(id(mod))
        for name,value in list(vars(mod).items()):
            if isinstance(value,types.ModuleType):
                f=getattr(value,'__file__',None)
                if f and str(prefix).lower() in str(f).lower():visit(value)
            elif isinstance(value,Path):
                try:relative=value.relative_to(prefix)
                except ValueError:continue
                if str(relative)=='.':setattr(mod,name,ProjectRoot())
                elif relative.parts[0]!='Code':setattr(mod,name,original(relative.as_posix()))
    visit(module);return module

def hourly():
    # The historical archive keeps vendor Python sources for provenance, but
    # platform-specific extensions come from the pinned installed environment.
    # Import them before historical modules prepend their vendor directories.
    for dependency in ('catboost','xgboost','lightgbm'):
        __import__(dependency)
    r=adapt_paths(load_module('supp_hourly_runtime',original('Code/Hourly_Attenuation_Experiments/Revision_032/runtime.py')))
    frozen_config=config()['hourly']['models']
    if config()['primary_seed']!=17 or config()['cpu_threads']!=2:
        raise ValueError('This reproduction preserves seed 17 and two CPU threads. A changed experimental protocol must be registered separately.')
    r.c.CONFIGS[0]=dict(frozen_config['tabm']['config'])
    r.config=lambda name:frozen_config[name]
    saved_loader=r.load_fit
    def load_fit(ref):
        ref=dict(ref);s=str(ref['model_file']).replace('\\','/')
        p=Path(s)
        if not p.is_absolute():p=original(s) if s.startswith('Outputs/') else PACKAGE/s
        ref['model_file']=str(p)
        return saved_loader(ref)
    r.load_fit=load_fit
    return r

@contextlib.contextmanager
def run_lock(label,filename='execution.lock'):
    p=path('logs')/filename;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def environment():
    import importlib.metadata as m
    packages=config()['environment']['packages']
    versions={}
    for name in packages:
        try:versions[name]=m.version(name)
        except m.PackageNotFoundError:versions[name]=None
    return dict(python=sys.version,platform=platform.platform(),packages=versions)
