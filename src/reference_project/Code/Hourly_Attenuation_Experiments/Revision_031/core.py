"""Isolated TabM research primitives. No network or legacy training entrypoints."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_031_TabM_Main'
DATA = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003'
FEATURE_SOURCE = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation/feature_selection.json'
VENDOR = ROOT / 'Code/Combined_Channel_Experiments/Revision_002/vendor'
sys.path.insert(0, str(VENDOR))
import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
from torch import nn
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings

TARGET = 'clear_sky_attenuation_db'
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
WEATHER = ('overall', 'clear_dry', 'rain', 'dry_overcast', 'other')
N_ROWS = {'train': 2404, 'validation': 343, 'test': 687}
SEED = 17
CONFIGS = [
    dict(id=0, name='official_base', width=256, blocks=3, k=32, dropout=.1, lr=.002, weight_decay=.01, embedding=0, gate=False),
    dict(id=1, name='compact_regularized', width=128, blocks=3, k=16, dropout=.2, lr=.001, weight_decay=.01, embedding=0, gate=False),
    dict(id=2, name='compact_embedding', width=128, blocks=3, k=16, dropout=.2, lr=.001, weight_decay=.01, embedding=8, gate=False),
    dict(id=3, name='compact_group_gate', width=128, blocks=3, k=16, dropout=.2, lr=.001, weight_decay=.01, embedding=0, gate=True),
    dict(id=4, name='compact_embedding_group_gate', width=128, blocks=3, k=16, dropout=.2, lr=.001, weight_decay=.01, embedding=8, gate=True),
]
SOURCE_NAMES = ('core.py', 'train_search.py', 'verify_search.py', 'self_tests.py', 'PROTOCOL.md')


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def emit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def update(path, value):
    """Only mutable execution state, never a committed experimental result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def save_tensor(path, value, intermediate=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(intermediate or not path.exists(), 'Refuse to overwrite committed weights: ' + str(path))
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def load_tensor(path):
    # Only this project's trusted local checkpoints, pinned by manifests/results.
    return torch.load(path, map_location='cpu', weights_only=False)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def names_for_mask(mask, names):
    return [name for name in names if name.startswith('surface__') or
            any(mask & (1 << index) and name.startswith(f'p{level}__') for index, level in enumerate(LEVELS))]


def group_ids(names):
    groups = [name.split('__')[0] for name in names]
    allowed = ['surface', *[f'p{p}' for p in LEVELS]]
    require(set(groups) <= set(allowed), 'Unrecognized atmospheric feature group')
    active = [g for g in allowed if g in groups]
    return [active.index(g) for g in groups], active


class AtmosphericTabM(nn.Module):
    def __init__(self, names, cfg):
        super().__init__()
        indices, groups = group_ids(names)
        self.register_buffer('group_index', torch.tensor(indices, dtype=torch.long))
        self.gate = None
        if cfg['gate']:
            self.gate = nn.Sequential(nn.Linear(len(names), 32), nn.ReLU(), nn.Linear(32, len(groups)))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        embedding = LinearReLUEmbeddings(len(names), cfg['embedding']) if cfg['embedding'] else None
        self.tabm = TabM.make(n_num_features=len(names), d_out=1, k=cfg['k'],
                             n_blocks=cfg['blocks'], d_block=cfg['width'],
                             dropout=cfg['dropout'], num_embeddings=embedding)

    def forward(self, x):
        if self.gate is not None:
            weights = 2 * torch.sigmoid(self.gate(x))
            x = x * weights[:, self.group_index]
        return self.tabm(x).squeeze(-1)


def numeric(data, names):
    values = data[names].to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def fit_preprocess(data, names):
    values = numeric(data, names)
    nonempty = np.isfinite(values).any(axis=0)
    medians = np.zeros(len(names), dtype=np.float64)
    medians[nonempty] = np.nanmedian(values[:, nonempty], axis=0)
    filled = np.where(np.isnan(values), medians, values)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-12] = 1.
    truth = data[TARGET].to_numpy(float)
    prep = dict(median=medians.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                all_missing=[n for n, okay in zip(names, nonempty) if not okay],
                y_mean=float(truth.mean()), y_scale=max(float(truth.std()), 1e-6),
                fitted_hour_sha256=fingerprint(data.hour_bjt.astype(str).tolist()))
    return transform(data, names, prep), prep


def transform(data, names, prep):
    values = numeric(data, names)
    values = np.where(np.isnan(values), np.asarray(prep['median']), values)
    values = (values - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    require(np.isfinite(values).all(), 'Nonfinite transformed input')
    return values.astype(np.float32)


def frame(split, manifest, allow_test=False):
    require(split != 'test' or allow_test, 'Test data gate is closed')
    data = pd.read_parquet(DATA / f'{split}.parquet', columns=['hour_bjt', *manifest['feature_names'], TARGET])
    # Filter metadata to this split before inspecting weather or quality membership.
    status = pd.read_parquet(DATA / 'hourly_sample_status.parquet', columns=[
        'hour_bjt', 'split', 'use_for_regression', 'clear_dry', 'rain', 'heavy_rain', 'dry_overcast'])
    status = status.loc[status['split'].eq(split)]
    data = data.merge(status, on='hour_bjt', how='left', validate='one_to_one')
    require(len(data) == N_ROWS[split] and data.hour_bjt.is_unique, 'Invalid hour keys/count')
    require(data.split.eq(split).all() and data.use_for_regression.eq(True).all(), 'Invalid split/quality')
    require(data[['clear_dry','rain','heavy_rain','dry_overcast']].notna().all().all(), 'Missing weather flags')
    membership = data[['clear_dry','rain','dry_overcast']].astype(int).sum(axis=1)
    require((membership <= 1).all() and (~data.heavy_rain | data.rain).all(), 'Weather flag conflict')
    data['other'] = membership.eq(0)
    require(np.isfinite(data[TARGET]).all(), 'Invalid target')
    return data.sort_values('hour_bjt').reset_index(drop=True)


def metrics(data, prediction):
    truth = data[TARGET].to_numpy(float)
    prediction = np.asarray(prediction, float)
    require(prediction.shape == truth.shape and np.isfinite(prediction).all(), 'Invalid predictions')
    days = data.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    result = {}
    for group in WEATHER:
        mask = np.ones(len(data), bool) if group == 'overall' else data[group].to_numpy(bool)
        error = prediction[mask] - truth[mask]
        result[group] = dict(n=int(mask.sum()), n_dates=len(set(days[mask])),
                             rmse_db=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                             mae_db=float(np.mean(np.abs(error))) if len(error) else None,
                             bias_db=float(np.mean(error)) if len(error) else None)
    return result


def objective(scores):
    parts = [scores[g]['rmse_db'] for g in ('overall', 'rain', 'dry_overcast')]
    require(all(v is not None and math.isfinite(v) for v in parts), 'Missing validation endpoint')
    return math.sqrt(sum(v*v for v in parts) / 3)


def predict(net, x, prep):
    net.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            values.append(net(torch.as_tensor(x[start:start+256])).mean(dim=1).numpy().astype(float))
    return np.concatenate(values) * prep['y_scale'] + prep['y_mean']


def prediction_record(data, prediction):
    return dict(hour_bjt=data.hour_bjt.astype(str).tolist(), truth_db=data[TARGET].to_numpy(float).tolist(),
                prediction_db=np.asarray(prediction, float).tolist(),
                weather_masks={g:data[g].astype(bool).tolist() for g in WEATHER[1:]}, metrics=metrics(data, prediction))


def source_manifest(create=False):
    path = OUT / 'source_manifest.json'
    if path.exists():
        manifest = read(path)
        require(fingerprint({k:v for k,v in manifest.items() if k not in ('fingerprint', 'created_utc')}) == manifest['fingerprint'], 'Source manifest content changed')
        for name, digest in manifest['source_sha256'].items():
            require(sha(ROOT / name) == digest, 'Frozen source changed: ' + name)
        return manifest
    require(create, 'Prepare the source manifest first')
    screen = read(FEATURE_SOURCE)
    names = screen['feature_names_retained']
    parent = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/run_manifest.json'
    require(names == read(parent)['retained_names'], 'Rev023/Rev029 names disagree')
    require(len(names) == len(set(names)) == 156 and sum(n.startswith('surface__') for n in names) == 50, 'Feature pool changed')
    require(len({tuple(names_for_mask(i,names)) for i in range(1024)}) == 1024, 'Invalid layer partition')
    for item in read(VENDOR / 'source_manifest.json')['files']:
        require(sha(VENDOR / item['local']) == item['sha256'], 'Official vendor changed')
    paths = [FEATURE_SOURCE, parent, *[DATA / f'{split}.parquet' for split in N_ROWS],
             DATA / 'hourly_sample_status.parquet', DATA / 'manifest.json']
    paths += list(VENDOR.glob('*')) + [HERE / name for name in SOURCE_NAMES]
    paths += [ROOT / 'Outputs/Experiments/Combined_Channel_Revision_036/final_verification.json',
              ROOT / 'Outputs/Experiments/Combined_Channel_Revision_036/manifest.json',
              ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json']
    # Known BiLSTM anchor paths are taken from the frozen Rev029 manifest, not guessed.
    for name in read(parent)['source_sha256']:
        if name.endswith(('models/full_bilstm_seed17.joblib', 'test/full_bilstm_seed17.json',
                          'training/full_bilstm_seed17.json')):
            paths.append(ROOT / name)
    paths = sorted(set(paths))
    for p in paths:
        require(p.is_file(), 'Required source missing: ' + str(p))
    payload = dict(version='tabm-main-rev031-v1', feature_names=names, levels_hpa=list(LEVELS),
                   configs=CONFIGS, seed=SEED, max_epochs=160, patience=20, batch_size=256,
                   selection_objective='sqrt((MSE_overall+MSE_rain+MSE_dry_overcast)/3)',
                   n_rows=N_ROWS, target=TARGET, historical_test_access=True,
                   no_timing_pilot=True, new_weather_calls=0,
                   source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
                   versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 pandas=pd.__version__, sklearn=sklearn.__version__, platform=platform.platform()))
    payload['fingerprint'] = fingerprint(payload)
    payload['created_utc'] = now()
    emit(path, payload)
    return payload


def progress(stage, **kwargs):
    value = dict(stage=stage, pid=os.getpid(), utc=now(), **kwargs)
    update(OUT / 'progress.json', value)
    append(OUT / 'progress_events.jsonl', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


@contextlib.contextmanager
def writer_lock(stage):
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'writer.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Existing T013 worker owns the OS lock') from exc
        started = dict(pid=os.getpid(), process_created=psutil.Process().create_time(),
                       command=sys.argv, stage=stage, started_utc=now(), status='RUNNING')
        update(OUT / 'active_worker.json', started)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            emit(OUT / 'failures' / f'{time.time_ns()}_{stage}.json',
                 dict(utc=now(), stage=stage, traceback=traceback.format_exc()))
            raise
        finally:
            update(OUT / 'active_worker.json', dict(started, status='FAILED' if failed else 'EXITED', finished_utc=now()))
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
