"""Frozen single-seed adapter: current Rev028 numerics, new refitted subsets."""
from __future__ import annotations

import contextlib
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('frozen28_adapter', HERE.parent / 'Revision_028/run_weather.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
v, engine = w.v, w.e
np, pd, joblib, torch = v.np, v.pd, v.joblib, v.torch
read, sha, fp, atomic, require, now = v.read, v.sha, v.fp, v.atomic_json, v.require, v.now
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
PARENT = w.OUT
TARGET = 'clear_sky_attenuation_db'
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
WEATHER = v.WEATHER
LABELS = v.LABELS
SEED, EPOCHS = 17, 53
SOURCE_FILES = ('PROTOCOL.md', 'PARAMETERS.md', 'runtime.py', 'run_analysis.py', 'verify_independent.py', 'make_report.py')


def code_hashes():
    return {name: sha(HERE / name) for name in SOURCE_FILES if name.endswith('.py')}


def source_inputs():
    parent = w.manifest()
    choice = read(PARENT / 'selection.json')['plans']['full']['bilstm']
    require(choice['formal_rounds'] == 53 and choice['config_id'] == 2, 'Current full53 plan differs')
    require(choice['choice'] == {'estimator': {'epochs': 160, 'lr': .001, 'dropout': .2, 'weight_decay': .001},
                                  'override': {}, 'weight_policy': 'uniform', 'gated_fusion': False}, 'Full model config differs')
    require(read(PARENT / 'independent_qa.json')['status'] == 'PASS' and read(PARENT / 'delivery_qa.json')['status'] == 'PASS', 'Parent gates')
    return parent, choice


def build_manifest():
    parent, plan = source_inputs()
    sources = dict(parent['source_sha256'])
    paths = [PARENT / 'run_manifest.json', PARENT / 'selection.json', PARENT / 'independent_qa.json',
             PARENT / 'delivery_qa.json', PARENT / 'training/full_bilstm_seed17.json',
             PARENT / 'models/full_bilstm_seed17.joblib', PARENT / 'test/full_bilstm_seed17.json']
    paths += [HERE / s for s in SOURCE_FILES]
    paths += [v.DATA / s for s in ('train.parquet', 'validation.parquet', 'test.parquet', 'hourly_sample_status.parquet')]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = sha(path)
    payload = {'version': 'rev029-current53-single17-direct-refits-v1', 'source_sha256': sources,
               'parent_fingerprint': parent['fingerprint'], 'anchor_plan': plan,
               'original_names': parent['original_names'], 'retained_names': parent['retained_names'],
               'base_names': parent['base_names'], 'target': TARGET, 'levels_hpa': list(LEVELS),
               'n_train': 2404, 'n_validation': 343, 'n_test': 687, 'model_seed': SEED, 'epochs': EPOCHS,
               'fold_method': 'GroupKFold(n_splits=3), group=Beijing calendar date',
               'layer_states': 1024, 'rfe_states': 156, 'main_fit_calls': 4720,
               'formal_models': 1180, 'fold_models': 3540, 'rfe_permutation_evaluations': 36735,
               'bootstrap_replicates': 4096, 'bootstrap_seed': 20261012,
               'historical_test_access': True, 'package_versions': parent['package_versions']}
    payload['fingerprint'] = fp(payload)
    return payload


def manifest(create=False):
    path = OUT / 'run_manifest.json'
    if path.exists():
        saved = read(path)
        payload = {k: val for k, val in saved.items() if k not in ('fingerprint', 'created_utc')}
        require(fp(payload) == saved['fingerprint'], 'Manifest self digest')
        for name, digest in saved['source_sha256'].items():
            require(sha(ROOT / name) == digest, f'Frozen source changed: {name}')
        return saved
    require(create, 'Run prepare after successful preflight')
    for name in ('synthetic_qa.json', 'anchor_qa.json'):
        q = read(OUT / name)
        require(q['status'] == 'PASS' and q['runtime_sha256'] == sha(HERE / 'runtime.py') and
                q['code_sha256'] == code_hashes(), f'Preflight gate {name}')
    payload = build_manifest()
    payload['created_utc'] = now()
    atomic(path, payload)
    return payload


def state_path(kind, key):
    return OUT / kind / 'states' / f'{"mask" if kind == "layer" else "k"}_{key:04d}.json'


def model_path(kind, key, fold=0):
    stem = f'{"mask" if kind == "layer" else "k"}_{key:04d}'
    return OUT / kind / ('fold_models' if fold else 'models') / f'{stem}{f"_fold{fold}" if fold else ""}.joblib'


def evaluation_path(kind, key):
    return OUT / kind / 'evaluation' / state_path(kind, key).name


def keys(kind):
    return list(range(1024)) if kind == 'layer' else list(range(156, 0, -1))


def active(names):
    return tuple(i for i, level in enumerate(LEVELS) if any(n.startswith(f'p{level}__') for n in names))


def names_for_mask(mask, retained):
    return [n for n in retained if n.startswith('surface__') or any(mask & (1 << i) and n.startswith(f'p{p}__') for i, p in enumerate(LEVELS))]


def frame(split, m):
    return v.data_frame(split, m, events=True)


def folds(train):
    from sklearn.model_selection import GroupKFold
    groups = train.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    result = []
    for number, (fit, hold) in enumerate(GroupKFold(n_splits=3).split(train, groups=groups), 1):
        fit_days, hold_days = sorted(set(groups[fit])), sorted(set(groups[hold]))
        require(set(fit_days).isdisjoint(hold_days), 'Date leakage in internal fold')
        result.append({'fold': number, 'fit': fit, 'hold': hold, 'fit_days': fit_days, 'hold_days': hold_days,
                       'fit_index_sha256': fp(fit.tolist()), 'hold_index_sha256': fp(hold.tolist())})
    require(sorted(np.concatenate([f['hold'] for f in result]).tolist()) == list(range(len(train))), 'Incomplete fold coverage')
    return result


def raw(train, names, m):
    return v.raw(train, dict(m, retained_names=names))


def preprocess(train, names, m):
    values = raw(train, names, m)
    positions = list(range(188))
    for i in active(names):
        positions.extend(range(188 + 16 * i, 188 + 16 * (i + 1)))
    imputer = v.SimpleImputer(strategy='median', keep_empty_features=True)
    xf = imputer.fit_transform(values[:, positions])
    scaler = v.StandardScaler()
    xf = scaler.fit_transform(xf)
    return xf, {'positions': positions, 'imputer': imputer, 'scaler': scaler,
                'retained_names': names, 'active_positions': active(names), 'model_name': 'bilstm', 'seed': SEED}


def fit_fixed(xf, train, positions, choice, xh=None, hold=None, epochs=EPOCHS):
    """Same Rev028 optimization; internal holdout logs only, never stopping."""
    require((xh is None) == (hold is None), 'Diagnostic holdout mismatch')
    cfg = copy.deepcopy(choice['estimator'])
    cfg['epochs'] = epochs
    require(choice['weight_policy'] == 'uniform' and not choice['gated_fusion'] and not choice['override'], 'Frozen direct/uniform policy')
    y = train[TARGET].to_numpy(float)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    model = engine.legacy.TorchRegressor('bilstm', cfg, SEED, xf.shape[1], positions)
    model.net = engine.legacy.VerticalProfileNet('bilstm', positions, cfg['dropout'])
    model.target_mean = float(np.mean(y))
    model.target_scale = max(float(np.std(y)), 1e-6)
    xx = torch.as_tensor(np.asarray(xf, np.float32))
    yy = torch.as_tensor(np.asarray((y - model.target_mean) / model.target_scale, np.float32))
    ww = torch.ones(len(train), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        model.net.train()
        order = torch.randperm(len(xx), generator=generator)
        for ix in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(ww[ix] * (model.net(xx[ix]) - yy[ix]) ** 2)
            loss.backward()
            optimizer.step()
        model.net.eval()
        pred = model.predict(xf)
        row = engine.curve_row(epoch, train, pred)
        row['train_weather_metrics'] = metrics(train, pred)
        row['internal_holdout_metrics'] = None if hold is None else metrics(hold, model.predict(xh))
        history.append(row)
    metadata = {'trained_rounds': epochs, 'selected_round': epochs, 'seed': SEED,
                'stop_reason': 'fixed 53-round full-model budget frozen in Rev028; no new early stopping',
                'fit_seconds': time.perf_counter() - start, 'target_mean': model.target_mean,
                'target_scale': model.target_scale, 'choice': dict(choice, estimator=cfg),
                'sample_weight_policy': 'uniform', 'official_holdout_used_for_fit': False}
    require(np.isfinite(pred).all(), 'Nonfinite trained prediction')
    return model, history, metadata


def metrics(data, pred):
    truth = data[TARGET].to_numpy(float)
    pred = np.asarray(pred, float)
    require(pred.shape == truth.shape and np.isfinite(pred).all(), 'Invalid prediction')
    days = data.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    result = {}
    for group in WEATHER:
        mask = np.ones(len(data), bool) if group == 'overall' else data[group].to_numpy(bool)
        error = pred[mask] - truth[mask]
        result[group] = {'n_hours': int(mask.sum()), 'n_dates': int(len(set(days[mask]))),
                         'rmse_db': None if not len(error) else float(np.sqrt(np.mean(error ** 2))),
                         'mae_db': None if not len(error) else float(np.mean(np.abs(error))),
                         'bias_db': None if not len(error) else float(np.mean(error))}
    return result


def prediction_record(data, pred):
    return {'hour_bjt': data.hour_bjt.astype(str).tolist(), 'truth_db': data[TARGET].to_numpy(float).tolist(),
            'prediction_db': np.asarray(pred, float).tolist(),
            'weather_masks': {g: data[g].astype(bool).tolist() for g in WEATHER if g != 'overall'},
            'metrics': metrics(data, pred)}


def permutation_seed(m, k, fold, name):
    token = f'{m["fingerprint"]}|{k}|{fold}|{name}'
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big')


def weights_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_model(path, bundle):
    v.save_model(path, bundle)
    return sha(path)


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def csv_file(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    count = 0
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    return {'rows': count, 'sha256': sha(path)}


@contextlib.contextmanager
def writer_lock(stage):
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'writer.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('An existing Rev029 writer holds the OS lock') from exc
        set_stage(stage)
        failed = False
        try:
            yield
        except Exception:
            failed = True
            atomic(OUT / 'failures' / f'{time.time_ns()}_{stage}.json',
                   {'pid': os.getpid(), 'stage': stage, 'traceback': traceback.format_exc(), 'utc': now()})
            raise
        finally:
            atomic(OUT / 'active_worker.json', {'pid': os.getpid(), 'stage': stage,
                   'status': 'FAILED' if failed else 'EXITED', 'finished_utc': now()})
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def set_stage(stage):
    atomic(OUT / 'active_worker.json', {'pid': os.getpid(), 'stage': stage, 'command': sys.argv,
                                      'status': 'RUNNING', 'utc': now()})


def os_lock_available():
    import msvcrt
    path = OUT / 'writer.lock'
    if not path.exists():
        return True
    with path.open('r+b') as f:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    return True


def progress(stage, extra=None):
    lines = ['# Revision 029 进度', '', f'更新UTC：{now()}', '', f'当前阶段：{stage}。', '',
             '单一种子17、统一53轮；旧模型/数据只读。未完成最终QA与报告前不得称全部完成。']
    if extra:
        lines += ['', json.dumps(extra, ensure_ascii=False)]
    atomic_text(HERE / 'PROGRESS.md', '\n'.join(lines) + '\n')
