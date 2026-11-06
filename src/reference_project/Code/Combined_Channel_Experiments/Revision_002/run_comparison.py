"""Isolated, CPU-only, finite comparator batch. Never invokes an old entrypoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_002'
DATA = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003'
R29 = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
EXT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
STOP = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
sys.path.insert(0, str(HERE / 'vendor'))
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from tabm import TabM

TARGET = 'clear_sky_attenuation_db'
SEEDS = (17, 29, 43)
GRIDS = {
    'ridge': [{'alpha': a} for a in (.1, 1., 10.)],
    'elastic_net': [{'alpha': a, 'l1_ratio': l, 'max_iter': 30000, 'tol': 1e-6}
                    for a, l in ((.001, .1), (.01, .5), (.1, .9))],
    'svr_rbf': [{'C': c, 'epsilon': .05, 'gamma': 'scale'} for c in (1., 10., 100.)],
    'random_forest': [dict(n_estimators=400, max_features=f, min_samples_leaf=l,
                           n_jobs=2) for f, l in ((1., 1), (.7, 2), (.7, 5))],
    'xgboost': [dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=.8,
                     colsample_bytree=.8, reg_lambda=1., objective='reg:squarederror',
                     tree_method='hist', n_jobs=2)
                for n, d, lr in ((400, 3, .03), (600, 4, .03), (800, 3, .02))],
}
NEURAL = {
    'bai2022_mlp_adapted': dict(max_epochs=1000, patience=5, batch_size=128,
                               optimizer='RMSProp', lr=1e-4),
    'bai2022_lstm_t0_adapted': dict(max_epochs=1000, patience=5, batch_size=128,
                                   optimizer='RMSProp', lr=1e-4),
    'tabm2025_official_budgeted': dict(max_epochs=200, patience=20, batch_size=256,
                                      optimizer='AdamW', lr=.002),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def emit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


class BaiNetwork(nn.Module):
    def __init__(self, kind, n_features=10):
        super().__init__()
        self.kind = kind
        self.recurrent = nn.LSTM(n_features, 32, batch_first=True) if 'lstm' in kind else None
        dims = [32, 16, 32, 64, 1] if self.recurrent is not None else [n_features, 16, 32, 64, 32, 16, 1]
        modules = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            modules.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                modules.append(nn.ReLU())
        self.head = nn.Sequential(*modules)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if self.recurrent is not None:
            for name, p in self.recurrent.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            # Keras-style forget bias; exactly one effective bias unit, not two.
            with torch.no_grad():
                self.recurrent.bias_ih_l0[32:64].fill_(1.)

    def forward(self, x):
        if self.recurrent is not None:
            x = self.recurrent(x[:, None, :])[0][:, -1, :]
        return self.head(x).squeeze(-1)


def build_neural(name):
    if name == 'tabm2025_official_budgeted':
        return TabM.make(n_num_features=10, d_out=1, k=32, n_blocks=3,
                         d_block=256, dropout=.1)
    return BaiNetwork(name)


def member_loss(output, target):
    if output.ndim == 3:
        require(output.shape[2] == 1, 'TabM output must be scalar')
        return (output.squeeze(-1) - target[:, None]).square().mean()
    return (output - target).square().mean()


def neural_predict(model, x):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            p = model(torch.as_tensor(x[start:start+256], dtype=torch.float32))
            if p.ndim == 3:
                p = p.squeeze(-1).mean(1)
            result.append(p.numpy().astype(np.float64))
    return np.concatenate(result)


def numeric(frame, names):
    x = frame[names].to_numpy(dtype=np.float64, copy=True)
    x[~np.isfinite(x)] = np.nan
    return x


def align_saved(hours, predictions, frame):
    p = pd.Series(np.asarray(predictions, dtype=float), index=pd.to_datetime(hours))
    require(p.index.is_unique and len(p) == len(frame), 'Saved prediction key count')
    result = p.reindex(pd.DatetimeIndex(frame.hour_bjt)).to_numpy()
    require(np.isfinite(result).all(), 'Saved prediction key/finite mismatch')
    return result


def local_frame(split, names):
    frame = pd.read_parquet(DATA / f'{split}.parquet', columns=['hour_bjt', *names, TARGET])
    status = pd.read_parquet(DATA / 'hourly_sample_status.parquet',
                             columns=['hour_bjt', 'split', 'use_for_regression', 'clear_dry',
                                      'rain', 'heavy_rain', 'dry_overcast'])
    frame = frame.merge(status, on='hour_bjt', validate='one_to_one')
    frame = frame.sort_values('hour_bjt').reset_index(drop=True)
    require(len(frame) == {'train':2404, 'validation':343, 'test':687}[split], 'LOCAL count')
    require(frame.split.eq(split).all() and frame.use_for_regression.eq(True).all(), 'LOCAL split/quality')
    require(frame.hour_bjt.is_unique and np.isfinite(frame[TARGET]).all(), 'LOCAL key/label')
    return frame


def protected_inputs():
    paths = [DATA / f'{sp}.parquet' for sp in ('train','validation','test')]
    paths += [DATA/'hourly_sample_status.parquet', DATA/'manifest.json', STOP,
              R29/'rfe/states/k_0010.json', R29/'rfe/models/k_0010.joblib',
              R29/'rfe/evaluation/k_0010.json', R29/'run_manifest.json',
              EXT/'local_top10_acquisition.json', EXT/'scope_manifest.json',
              EXT/'field_audit_001/actual_field_qa.json',
              EXT/'evaluation_002/hourly_predictions.parquet',
              EXT/'evaluation_002/read_only_inputs.json',
              ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/final_test.json']
    paths += list(HERE.glob('*.py')) + list(HERE.glob('*.md'))
    paths += [p for p in (HERE/'vendor').iterdir() if p.is_file()]
    original = read(EXT/'evaluation_002/read_only_inputs.json')
    all_hashes = {str(Path(p).resolve()): digest for p,digest in original.items()}
    for p in paths:
        require(p.exists(), 'Missing source: '+str(p))
        all_hashes.setdefault(str(p.resolve()), sha(p))
    for p, digest in all_hashes.items():
        require(sha(p) == digest, 'Protected source SHA mismatch: '+p)
    require(sha(R29/'rfe/models/k_0010.joblib') ==
            'b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5', 'Frozen model mismatch')
    vend = read(HERE/'vendor/source_manifest.json')
    for row in vend['files']:
        require(sha(HERE/'vendor'/row['local']) == row['sha256'], 'Official source changed')
    return all_hashes


def self_tests():
    tests = []
    for name in NEURAL:
        set_seed(17)
        net = build_neural(name).eval()
        x = np.arange(50, dtype=np.float32).reshape(5,10) / 50
        a = neural_predict(net, x)
        set_seed(17)
        b = neural_predict(build_neural(name).eval(), x)
        require(a.shape == (5,) and np.isfinite(a).all() and np.array_equal(a,b), 'Shape/determinism '+name)
        tests.append(dict(name=name, shape_determinism='PASS', parameters=sum(p.numel() for p in net.parameters())))
    pred = torch.tensor([[[-1.],[1.]]])
    require(member_loss(pred,torch.tensor([0.])).item() == 1., 'Wrong ensemble training loss')
    require(pred.mean().square().item() == 0., 'Ensemble-loss negative control')
    a = np.array([[1.,np.nan],[3.,2.],[5.,4.]])
    imp = SimpleImputer(strategy='median').fit(a)
    sc = StandardScaler().fit(imp.transform(a))
    old = sc.mean_.copy()
    sc.transform(imp.transform([[1000.,1000.]]))
    require(np.array_equal(old,sc.mean_) and np.allclose(imp.statistics_,[3.,3.]), 'Train-only preprocessing control')
    f = pd.DataFrame({'hour_bjt':pd.to_datetime(['2020-01-01','2020-01-02'])})
    require(np.array_equal(align_saved(['2020-01-02','2020-01-01'],[2.,1.],f),[1.,2.]), 'Key alignment')
    try:
        align_saved(['2020-01-02','2020-01-02'],[2.,1.],f)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Duplicate-key negative control failed')
    return dict(status='PASS', tests=tests, ensemble_member_loss='PASS',
                train_only_scaler='PASS', shuffled_keys='PASS', duplicate_key_rejected='PASS',
                no_real_data_fit=True)


def classic_model(name, params, seed):
    classes = dict(ridge=Ridge, elastic_net=ElasticNet, svr_rbf=SVR,
                   random_forest=RandomForestRegressor, xgboost=xgboost.XGBRegressor)
    args = dict(params)
    if name in ('random_forest','xgboost'):
        args['random_state'] = seed
    return classes[name](**args)


def fit_neural(name, seed, xt, yt, xv, yv, yscale):
    set_seed(seed)
    net = build_neural(name)
    cfg = NEURAL[name]
    opt = (torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=.9, eps=1e-6, momentum=0.)
           if cfg['optimizer'] == 'RMSProp' else
           torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0003))
    xx = torch.tensor(xt, dtype=torch.float32)
    yy = torch.tensor(yt, dtype=torch.float32)
    best, best_epoch, state, stale = float('inf'), 0, None, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg['max_epochs']+1):
        net.train()
        order = torch.randperm(len(xx))
        loss_sum = 0.
        for indices in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss = member_loss(net(xx[indices]), yy[indices])
            require(torch.isfinite(loss).item(), name+' nonfinite training loss')
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(indices)
        vp = neural_predict(net, xv)
        mse = float(np.mean((vp-yv)**2))
        require(np.isfinite(mse), name+' nonfinite validation')
        history.append(dict(epoch=epoch, train_member_mse_scaled=loss_sum/len(xx),
                            validation_rmse_db=mse**.5*yscale))
        if mse < best:
            best, best_epoch, state, stale = mse, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
        if epoch % 20 == 0:
            print(json.dumps(dict(event='epoch',model=name,seed=seed,epoch=epoch,
                                   best_epoch=best_epoch,validation_rmse_db=best**.5*yscale)),flush=True)
        if stale >= cfg['patience']:
            break
    require(state is not None, 'No finite neural checkpoint')
    net.load_state_dict(state)
    net.eval()
    ident = f'{name}_seed{seed}'
    path = OUT/'models'/f'{ident}.pt'
    torch.save(state, path)
    receipt = dict(id=ident, family=name, seed=seed, model_type='torch', params=cfg,
                   architecture=str(net), parameter_count=sum(p.numel() for p in net.parameters()),
                   epochs_run=epoch, best_epoch=best_epoch, validation_rmse_db=best**.5*yscale,
                   stop_reason='patience' if stale >= cfg['patience'] else 'budget_cap',
                   history=history, fit_seconds=time.perf_counter()-start, train_hours=len(xt),
                   test_or_external_used=False, path=str(path.relative_to(OUT)), sha256=sha(path))
    emit(OUT/'training'/f'{ident}.json',receipt)
    print(json.dumps(dict(event='trained',model=ident,epochs=epoch,best_epoch=best_epoch,
                          validation_rmse_db=receipt['validation_rmse_db'])),flush=True)
    return {k:v for k,v in receipt.items() if k not in ('history','architecture')}


def fit_all(xt, yt, xv, yv, yscale):
    registry = [dict(id='train_mean',family='train_mean',seed=None,model_type='constant',value_scaled=0.)]
    for name, grid in GRIDS.items():
        candidates, fitted = [], []
        for i, params in enumerate(grid):
            start = time.perf_counter()
            model = classic_model(name,params,17)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(xt,yt)
            vp = model.predict(xv)
            require(np.isfinite(vp).all(), 'Invalid validation '+name)
            score = float(np.mean((vp-yv)**2)**.5*yscale)
            candidates.append(dict(index=i, params=params, validation_rmse_db=score,
                                   fit_seconds=time.perf_counter()-start,
                                   warnings=[str(w.message) for w in caught]))
            fitted.append(model)
        choice = min(range(len(candidates)),key=lambda i:(candidates[i]['validation_rmse_db'],i))
        emit(OUT/'selection'/f'{name}.json',dict(selected=choice, candidates=candidates,
                                               selection_split='validation',seed=17,test_used=False))
        seeds = SEEDS if name in ('random_forest','xgboost') else (17,)
        for seed in seeds:
            start = time.perf_counter()
            model = fitted[choice] if seed == 17 else classic_model(name,grid[choice],seed).fit(xt,yt)
            ident = f'{name}_seed{seed}'
            path = OUT/'models'/f'{ident}.joblib'
            joblib.dump(model,path,compress=3)
            receipt = dict(id=ident,family=name,seed=seed,model_type='sklearn',params=grid[choice],
                           selected_candidate=choice,validation_rmse_db=float(np.mean((model.predict(xv)-yv)**2)**.5*yscale),
                           fit_seconds=candidates[choice]['fit_seconds'] if seed == 17 else time.perf_counter()-start,
                           train_hours=len(xt),test_or_external_used=False,path=str(path.relative_to(OUT)),sha256=sha(path))
            emit(OUT/'training'/f'{ident}.json',receipt)
            registry.append(receipt)
        print(json.dumps(dict(event='selected',model=name,candidate=choice)),flush=True)
    for name in NEURAL:
        for seed in SEEDS:
            registry.append(fit_neural(name,seed,xt,yt,xv,yv,yscale))
    return registry


def predict_record(row, x, pre):
    if row['model_type'] == 'constant':
        p = np.zeros(len(x))
    elif row['model_type'] == 'sklearn':
        require(sha(OUT/row['path']) == row['sha256'],'New checkpoint SHA')
        p = joblib.load(OUT/row['path']).predict(x)
    else:
        require(sha(OUT/row['path']) == row['sha256'],'New checkpoint SHA')
        model = build_neural(row['family'])
        model.load_state_dict(torch.load(OUT/row['path'], map_location='cpu', weights_only=True))
        p = neural_predict(model,x)
    p = np.asarray(p,dtype=float) * pre['target_scale'] + pre['target_mean']
    require(p.shape == (len(x),) and np.isfinite(p).all(),'Invalid prediction '+row['id'])
    return p


def metrics(y,p):
    error = np.asarray(p,dtype=float)-np.asarray(y,dtype=float)
    require(np.isfinite(error).all() and len(error)>0,'Metric finite/count')
    return dict(n=len(error),rmse_db=float(np.mean(error**2)**.5),
                mae_db=float(np.mean(np.abs(error))),bias_db=float(np.mean(error)))


def evaluate(registry, pre, frames, names):
    ours = read(R29/'rfe/evaluation/k_0010.json')
    all_scores = []
    for split in ('validation','test'):
        f = frames[split]
        x = pre['scaler'].transform(pre['imputer'].transform(numeric(f,names)))
        result = f[['hour_bjt',TARGET,'rain','heavy_rain','clear_dry','dry_overcast']].rename(columns={TARGET:'truth_db'}).copy()
        old = ours['results'][split]
        require(np.allclose(align_saved(old['hour_bjt'],old['truth_db'],f),f[TARGET],atol=1e-12,rtol=0),'Frozen target equality')
        result['ours_frozen_top10'] = align_saved(old['hour_bjt'],old['prediction_db'],f)
        for row in registry:
            result[row['id']] = predict_record(row,x,pre)
        common = ['ours_frozen_top10']+[r['id'] for r in registry]
        if split == 'test':
            physical = read(ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/final_test.json')
            require(np.allclose(align_saved(physical['hour_bjt'],physical['truth_db'],f),f[TARGET],atol=1e-12,rtol=0),'ITU target equality')
            for name in ('physics_geom_circular','physics_p618_hourly_circular'):
                result[name] = align_saved(physical['hour_bjt'],physical['results'][name]['prediction_db'],f)
            common += ['physics_geom_circular','physics_p618_hourly_circular']
        for name in common:
            for group in ('overall','rain','dry_overcast','clear_dry'):
                mask = np.ones(len(result),dtype=bool) if group == 'overall' else result[group].to_numpy(dtype=bool)
                all_scores.append(dict(cohort='LOCAL_'+split,model=name,group=group,
                                       comparison_group='different_input_rain_physics' if name.startswith('physics') else 'common_top10',
                                       **metrics(result.truth_db[mask],result[name][mask])))
        result.to_parquet(OUT/'predictions'/f'local_{split}.parquet',index=False)
    qa = read(EXT/'field_audit_001/actual_field_qa.json')
    require(qa['status']=='PASS' and qa['scope_sha256']==sha(EXT/'scope_manifest.json'),'External independent field gate')
    table = read(EXT/'local_top10_acquisition.json')
    require(table['feature_names']==names,'External feature order')
    rows = table['rows']
    features = pd.DataFrame([r['values'] for r in rows])
    xx = numeric(features,names)
    require(xx.shape==(1872,10) and np.isfinite(xx).all(),'External actual Top10 complete')
    x = pre['scaler'].transform(pre['imputer'].transform(xx))
    output = pd.DataFrame({'station':[r['station'] for r in rows],
                           'hour_utc':pd.to_datetime([r['utc'] for r in rows],utc=True)})
    for row in registry:
        output[row['id']] = predict_record(row,x,pre)
    saved = pd.read_parquet(EXT/'evaluation_002/hourly_predictions.parquet').rename(columns={'prediction_db':'ours_frozen_top10'})
    result = output.merge(saved,on=['station','hour_utc'],how='outer',validate='one_to_one',indicator=True)
    require(result._merge.eq('both').all() and len(result)==1872,'External exact cohort identity')
    result = result.drop(columns='_merge').sort_values(['station','hour_utc'])
    eligible = result.primary_score_eligible.to_numpy(dtype=bool)
    require(eligible.sum()==1847,'External primary sample count')
    for station in ['pooled',*sorted(result.station.unique())]:
        mask = eligible & (True if station=='pooled' else result.station.eq(station).to_numpy())
        for name in ['ours_frozen_top10']+[r['id'] for r in registry]:
            all_scores.append(dict(cohort='GBS39_'+station,model=name,group='eligible_all',
                                   comparison_group='common_top10_approximate_transfer',
                                   **metrics(result.truth_db[mask],result[name][mask])))
    result.to_parquet(OUT/'predictions/gbs39.parquet',index=False)
    emit(OUT/'metrics.json',all_scores)
    pd.DataFrame(all_scores).to_csv(OUT/'metrics.csv',index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('selftest','run'))
    args = parser.parse_args()
    set_seed(17)
    if args.stage == 'selftest':
        print(json.dumps(self_tests(),indent=2))
        return
    require(not OUT.exists(),'This output revision already exists; do not overwrite/retrain')
    require(STOP.exists(),'Weather stop marker must remain present')
    OUT.mkdir(parents=True)
    for d in ('models','training','selection','predictions'):
        (OUT/d).mkdir()
    import msvcrt
    with (OUT/'experiment.lock').open('xb+') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        started = now()
        emit(OUT/'launch.json',dict(pid=os.getpid(),started_utc=started,command=sys.argv,
                                    protocol_sha256=sha(HERE/'PROTOCOL.md'),weather_downloads_stopped=True))
        try:
            tests = self_tests()
            emit(OUT/'selftest.json',tests)
            protected = protected_inputs()
            emit(OUT/'source_sha256_before.json',protected)
            names = read(R29/'rfe/states/k_0010.json')['feature_names']
            frames = {sp:local_frame(sp,names) for sp in ('train','validation','test')}
            for a,b in [('train','validation'),('train','test'),('validation','test')]:
                require(set(frames[a].hour_bjt).isdisjoint(frames[b].hour_bjt),'Split hour overlap')
            train, val = frames['train'],frames['validation']
            rawtrain = numeric(train,names)
            require(np.isfinite(rawtrain).any(0).all(),'Empty train feature')
            imputer = SimpleImputer(strategy='median',keep_empty_features=True).fit(rawtrain)
            xti = imputer.transform(rawtrain)
            scaler = StandardScaler().fit(xti)
            y = train[TARGET].to_numpy(dtype=float)
            pre = dict(feature_names=names,imputer=imputer,scaler=scaler,
                       target_mean=float(y.mean()),target_scale=float(y.std()),
                       fit_split='train',fit_hours=[str(h) for h in train.hour_bjt])
            require(pre['target_scale']>0,'Constant training target')
            joblib.dump(pre,OUT/'preprocessing.joblib',compress=3)
            config = dict(created_utc=now(),protocol_sha256=sha(HERE/'PROTOCOL.md'),
                           feature_names=names,classical_grids=GRIDS,neural=NEURAL,seeds=list(SEEDS),
                           source_count=len(protected),preprocessing_sha256=sha(OUT/'preprocessing.joblib'),
                           python=sys.version,platform=platform.platform(),versions=dict(numpy=np.__version__,
                           pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__),
                           train_hours=2404,validation_hours=343,test_hours=687,
                           test_previously_seen=True,new_target_reference_fit=False,external_fit=False)
            emit(OUT/'run_manifest.json',config)
            xt=scaler.transform(xti)
            xv=scaler.transform(imputer.transform(numeric(val,names)))
            yt=(y-pre['target_mean'])/pre['target_scale']
            yv=(val[TARGET].to_numpy()-pre['target_mean'])/pre['target_scale']
            registry=fit_all(xt,yt,xv,yv,pre['target_scale'])
            emit(OUT/'frozen_selection.json',dict(frozen_utc=now(),models=registry,
                                                all_training_complete=True,test_or_external_used=False))
            print('ALL_FITS_FROZEN; STARTING_HELDOUT_EVALUATION',flush=True)
            evaluate(registry,pre,frames,names)
            after={p:sha(p) for p in protected}
            require(after==protected,'Protected input modified')
            emit(OUT/'source_sha256_after.json',after)
            emit(OUT/'completion.json',dict(status='COMPLETED_PENDING_INDEPENDENT_QA',started_utc=started,
                                            completed_utc=now(),new_models=len(registry)-1,
                                            protected_sources_unchanged=True,weather_stop_preserved=True))
            print('BATCH_COMPLETE_PENDING_INDEPENDENT_QA',flush=True)
        except BaseException:
            emit(OUT/f'failure_{time.time_ns()}.json',dict(utc=now(),traceback=traceback.format_exc()))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__ == '__main__':
    main()
