"""T014 primitives; pinned legacy constructors are read-only, never their CLIs."""
from __future__ import annotations
import copy, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

r29 = module('t014_legacy29', HERE.parent / 'Revision_029/runtime.py')
c = module('t014_tabm31', HERE.parent / 'Revision_031/core.py')
np, pd, torch, joblib = r29.np, r29.pd, r29.torch, r29.joblib
engine, legacy = r29.engine, r29.engine.legacy
read, sha, fp, emit, update, append, require, now = c.read, c.sha, c.fingerprint, c.emit, c.update, c.append, c.require, c.now
DATA, TARGET, WEATHER, LEVELS = c.DATA, c.TARGET, c.WEATHER, c.LEVELS
MODELS = ('tabm','bai_lstm','bilstm','transformer','mlp','catboost','gcn','random_forest','svr_rbf','elastic_net')
STRUCTURED = ('bilstm','transformer','gcn')
NN = ('tabm','bai_lstm',*legacy.NEURAL_MODELS)
P28, P29, P31 = r29.PARENT, r29.OUT, c.OUT
PERM_NAMESPACE = 'cdda7a1eaa7e14477a8edef39702d5b5e34e03f2a29a337d01b91c88ba57b4cd'

def rel(path): return str(Path(path).resolve().relative_to(ROOT))
def key_names(): return read(c.FEATURE_SOURCE)['feature_names_retained']
def old_manifest(): return read(P29 / 'run_manifest.json')
def config(name):
    if name == 'tabm': return dict(config=c.CONFIGS[0], epochs=54, weight_policy='uniform')
    if name == 'bai_lstm':
        p = OUT / 'bai_epoch_selection.json'
        return dict(epochs=read(p)['selected_epoch'] if p.exists() else None, max_epochs=1000,
                    patience=5, batch=128, lr=1e-4, alpha=.9, eps=1e-6, weight_policy='uniform')
    return read(P28 / 'selection.json')['plans']['full'][name]

def manifest(verify=True):
    m = read(OUT / 'manifest.json')
    require(fp({k:v for k,v in m.items() if k not in ('fingerprint','created_utc')}) == m['fingerprint'], 'Manifest digest')
    if verify:
        for path, digest in m['source_sha256'].items():
            require(sha(ROOT/path) == digest, 'Changed pinned input/source: '+path)
    return m

def stop_check():
    if (OUT / 'STOP_REQUESTED').exists(): raise InterruptedError('T014 STOP_REQUESTED')
    require(__import__('shutil').disk_usage(OUT).free > 3*1024**3, 'Disk safety threshold 3GiB')

def progress(**kw):
    obj = dict(utc=now(), pid=os.getpid(), **kw)
    update(OUT/'progress.json', obj)
    append(OUT/'progress_events.jsonl', obj)

def frames():
    m = old_manifest()
    result = {split:r29.frame(split,m) for split in ('train','validation','test')}
    for split, data in result.items():
        require(len(data)==c.N_ROWS[split] and data.hour_bjt.is_unique, 'Dataset keys')
        # Legacy frame already carries mutually exclusive weather flags.
        if 'other' not in data: data['other'] = ~data[['clear_dry','rain','dry_overcast']].any(axis=1)
        require(data.hour_bjt.dt.tz is not None and str(data.hour_bjt.dt.tz) in ('Asia/Shanghai','UTC+08:00','pytz.FixedOffset(480)'), 'Beijing timezone')
    return result

def metric(y,p):
    y,p = np.asarray(y,float),np.asarray(p,float)
    require(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(), 'Metrics nonfinite/shape')
    n=len(y)
    if n==0: return dict(n=0,rmse_db=None,mae_db=None,bias_db=None,r2=None,pearson_r=None,pearson_r_squared=None,r2_reason='EMPTY',pearson_reason='EMPTY')
    e=p-y; sy=float(np.sum((y-y.mean())**2)); sp=float(np.sum((p-p.mean())**2))
    valid=n>=2 and sy>0
    rv=float(np.sum((y-y.mean())*(p-p.mean()))/math.sqrt(sy*sp)) if valid and sp>0 else None
    return dict(n=n,rmse_db=float(np.sqrt(np.mean(e**2))),mae_db=float(np.mean(abs(e))),bias_db=float(np.mean(e)),
                r2=float(1-np.sum(e**2)/sy) if valid else None,pearson_r=rv,pearson_r_squared=None if rv is None else rv*rv,
                r2_reason=None if valid else 'N_LT_2_OR_CONSTANT_TARGET',pearson_reason=None if rv is not None else 'N_LT_2_OR_CONSTANT_VECTOR')

def scores(data,p):
    days=data.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    y=data[TARGET].to_numpy(float); p=np.asarray(p,float)
    result={}
    for group in WEATHER:
        mask=np.ones(len(data),bool) if group=='overall' else data[group].to_numpy(bool)
        result[group]=dict(metric(y[mask],p[mask]),dates=sorted(set(days[mask])),n_dates=len(set(days[mask])))
    return result

def record(data,p):
    return dict(hour_bjt=data.hour_bjt.astype(str).tolist(),truth_db=data[TARGET].to_numpy(float).tolist(),
                prediction_db=np.asarray(p,float).tolist(),weather_masks={w:data[w].astype(bool).tolist() for w in WEATHER[1:]},metrics=scores(data,p))

def legal_mask(data):
    bad=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_027/excluded_hours.json')
    keys={pd.Timestamp(x['hour_bjt']) for x in bad}
    return ~data.hour_bjt.isin(keys).to_numpy()

def prep_fit(name,train,names):
    if name in ('tabm','bai_lstm'):
        x,prep=c.fit_preprocess(train,names)
        return x,dict(type='numeric',preprocess=prep,retained_names=names,model_name=name,
                      fit_hour_sha256=fp(train.hour_bjt.astype(str).tolist()))
    m=old_manifest(); raw=r29.raw(train,names,m)
    if name in STRUCTURED:
        positions=list(range(188))
        for i in r29.active(names): positions.extend(range(188+16*i,188+16*(i+1)))
    else: positions=[m['original_names'].index(n) for n in names]
    imputer=r29.v.SimpleImputer(strategy='median',keep_empty_features=True)
    x=imputer.fit_transform(raw[:,positions]); scaler=None
    if name in legacy.SCALED_MODELS:
        scaler=r29.v.StandardScaler(); x=scaler.fit_transform(x)
    return x,dict(type='legacy',positions=positions,imputer=imputer,scaler=scaler,retained_names=names,
                  active_positions=r29.active(names),model_name=name,seed=17,fit_hour_sha256=fp(train.hour_bjt.astype(str).tolist()))

def transformed(bundle,data):
    if bundle.get('type')=='numeric': return c.transform(data,bundle['retained_names'],bundle['preprocess'])
    raw=r29.raw(data,bundle['retained_names'],old_manifest())
    x=bundle['imputer'].transform(raw[:,bundle['positions']])
    return bundle['scaler'].transform(x) if bundle['scaler'] is not None else x

def column_index(bundle,feature):
    if bundle.get('type')=='numeric': return bundle['retained_names'].index(feature)
    return bundle['positions'].index(old_manifest()['original_names'].index(feature))

class BaiNet(torch.nn.Module):
    def __init__(self,k):
        super().__init__(); self.lstm=torch.nn.LSTM(k,32,batch_first=True)
        self.head=torch.nn.Sequential(torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,32),torch.nn.ReLU(),
                                      torch.nn.Linear(32,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        for obj in self.modules():
            if isinstance(obj,torch.nn.Linear): torch.nn.init.xavier_uniform_(obj.weight); torch.nn.init.zeros_(obj.bias)
        for key,param in self.lstm.named_parameters():
            if 'weight' in key: torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
                if 'bias_ih' in key:
                    with torch.no_grad(): param[32:64].fill_(1.)
    def forward(self,x): return self.head(self.lstm(x[:,None,:])[0][:,-1,:]).flatten()

def make_nn(name,names,choice=None):
    torch.manual_seed(17); torch.set_num_threads(2)
    if name=='tabm': return c.AtmosphericTabM(names,c.CONFIGS[0])
    if name=='bai_lstm': return BaiNet(len(names))
    choice=choice or config(name)['choice']; cfg=choice['estimator']
    if name=='mlp': return legacy.FlatMLP(len(names),cfg['dropout'])
    cls=engine.GatedProfileNet if choice['gated_fusion'] else legacy.VerticalProfileNet
    return cls(name,r29.active(names),cfg['dropout'])

def predict_x(bundle,x):
    if 'model' in bundle: return np.asarray(bundle['model'].predict(x),float)
    net=bundle['net']; net.eval(); values=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):
            y=net(torch.as_tensor(np.asarray(x[start:start+256],np.float32)))
            if bundle['model_name']=='tabm': y=y.mean(dim=1)
            values.append(y.numpy().astype(float))
    return np.concatenate(values)*bundle['preprocess']['y_scale']+bundle['preprocess']['y_mean']

def predict(bundle,data): return predict_x(bundle,transformed(bundle,data))

def load_fit(ref):
    path=ROOT/ref['model_file']; require(sha(path)==ref['model_sha256'],'Model SHA')
    if ref['format']=='tabm31':
        b=c.load_tensor(path); net=c.AtmosphericTabM(b['names'],b['config']); net.load_state_dict(b['weights'])
        return dict(type='numeric',retained_names=b['names'],preprocess=b['preprocess'],net=net,model_name='tabm')
    b=joblib.load(path)
    if 'nn_weights' in b:
        net=make_nn(b['model_name'],b['retained_names']); net.load_state_dict(b.pop('nn_weights')); b['net']=net
    return b

def write_bundle(path,b):
    require(not path.exists(),'Refuse model overwrite'); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    require(not temp.exists(),'Orphan model temp preserved; inspect')
    joblib.dump(b,temp,compress=3); os.replace(temp,path)

def permutation_seed(k,fold,name):
    return int.from_bytes(hashlib.sha256(f'{PERM_NAMESPACE}|{k}|{fold}|{name}'.encode()).digest()[:8],'big')

def state_dir(name,k): return OUT/'models'/name/f'k_{k:04d}'

def check_prep(bundle,train,names,name):
    require(bundle['retained_names']==names,'Retained names differ')
    x,new=prep_fit(name,train,names)
    oldx=transformed(bundle,train)
    require(x.shape==oldx.shape and np.allclose(x,oldx,rtol=0,atol=1e-7),'Fitting-only preprocessing replay')
    if new['type']=='numeric':
        for key in ('median','mean','scale','y_mean','y_scale'):
            require(np.allclose(np.asarray(new['preprocess'][key]),np.asarray(bundle['preprocess'][key]),rtol=0,atol=1e-10),'Preprocessing parameters '+key)
    else:
        require(new['positions']==bundle['positions'],'Structured/flat positions')
        require(np.allclose(new['imputer'].statistics_,bundle['imputer'].statistics_,rtol=0,atol=1e-10,equal_nan=True),'Imputer fit')
        if new['scaler'] is not None:
            require(np.allclose(new['scaler'].mean_,bundle['scaler'].mean_,rtol=0,atol=1e-10),'Scaler mean')
            require(np.allclose(new['scaler'].scale_,bundle['scaler'].scale_,rtol=0,atol=1e-10),'Scaler scale')
    return x
