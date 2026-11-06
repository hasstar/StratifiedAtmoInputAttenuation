"""Reconstruct frozen Revision 023 models, then score test without fitting."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
LEGACY = HERE.parent / 'Revision_014'
sys.path.insert(0, str(LEGACY))
import common as c
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PARENT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation'
DATA = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003'
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_026_Ten_Model_Test'
MANIFEST = OUT / 'run_manifest.json'
STRUCTURED = frozenset(('bilstm', 'transformer', 'gcn'))
WEATHER = ('overall', 'clear_dry', 'rain', 'dry_overcast', 'other')
LABELS = {'overall': '总体', 'clear_dry': '晴天', 'rain': '雨天',
          'dry_overcast': '干燥阴天', 'other': '其他天气'}
MODEL_LABELS = {'elastic_net': 'Elastic Net', 'svr_rbf': 'SVR-RBF',
                'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
                'lightgbm': 'LightGBM', 'catboost': 'CatBoost', 'mlp': 'MLP',
                'transformer': 'Transformer', 'gcn': 'GCN', 'bilstm': 'BiLSTM'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    c.prior.atomic_json(Path(path), value)


def versions():
    return {name: str(__import__(name).__version__) for name in
            ('numpy', 'pandas', 'sklearn', 'torch', 'xgboost', 'lightgbm', 'catboost', 'joblib')}


def source_plan():
    parent = read(PARENT / 'run_manifest.json')
    for name, digest in parent['source_hashes'].items():
        require(sha(ROOT / name) == digest, f'Frozen parent source changed: {name}')
    for name in ('feature_selection_qa.json', 'model_selection_qa.json'):
        require(read(PARENT / name)['status'] == 'PASS', f'Parent QA not PASS: {name}')
    selection = read(PARENT / 'model_selection.json')
    require(selection['fingerprint'] == parent['fingerprint'], 'Parent selection fingerprint differs')
    screen = read(PARENT / 'feature_selection.json')
    names = screen['feature_names_original']
    kept = screen['feature_names_retained']
    require(len(names) == 348 and len(kept) == 156 and len(set(kept)) == 156,
            'Frozen feature list changed')
    plans = {row['name']: {key: row[key] for key in
                         ('best_config_id', 'best_config', 'formal_rounds', 'formal_rounds_reason')}
             for row in selection['model_rows']}
    require(set(plans) == set(c.MODEL_NAMES), 'Ten model families not present')
    paths = [ROOT / name for name in parent['source_hashes']]
    paths += [PARENT / 'run_manifest.json', PARENT / 'model_selection.json',
              PARENT / 'model_selection_qa.json', DATA / 'test.parquet',
              HERE / 'PROTOCOL.md', HERE / 'run_test_benchmark.py',
              HERE / 'verify_independent.py']
    paths += sorted((PARENT / 'trials').glob('*.json'))
    paths += sorted((PARENT / 'validation').glob('*.json'))
    paths += [LEGACY / '_vendor/catboost/_catboost.pyd',
              LEGACY / '_vendor/lightgbm/bin/lib_lightgbm.dll']
    payload = {'version': 'rev026-frozen-ten-model-test-v1',
               'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
               'parent_model_fingerprint': parent['fingerprint'],
               'feature_fingerprint': screen['fingerprint'],
               'original_names': names, 'retained_names': kept,
               'models': list(c.MODEL_NAMES), 'seeds': list(c.SEEDS), 'plans': plans,
               'target': c.TARGET, 'n_train': 2404, 'n_validation': 343, 'n_test': 687,
               'ranking': 'mean seed overall test RMSE; supplementary RMSE/log-latency 50/50',
               'test_used_for_fitting_or_retuning': False,
               'test_used_for_family_ranking_per_user': True,
               'historical_test_access': True,
               'validation_reproduction_max_abs_tolerance_db': 1e-6,
               'package_versions': versions(), 'python': sys.version,
               'platform': platform.platform()}
    payload['fingerprint'] = fp(payload)
    return payload


def manifest(create=False):
    candidate = source_plan()
    if MANIFEST.exists():
        old = read(MANIFEST)
        require(old['fingerprint'] == candidate['fingerprint'], 'Frozen new manifest mismatch')
        return old
    require(create, 'Run prepare first')
    candidate['created_utc'] = now()
    atomic_json(MANIFEST, candidate)
    return candidate


@contextlib.contextmanager
def lock(stage):
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'writer.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError('A matching Revision 026 writer is already active') from e
        atomic_json(OUT / 'active_worker.json', {'pid': os.getpid(), 'stage': stage,
                    'command': sys.argv, 'started_utc': now(), 'status': 'RUNNING'})
        try:
            yield
        except Exception:
            atomic_json(OUT / 'failures' / f'{time.time_ns()}_{stage}.json',
                        {'stage': stage, 'pid': os.getpid(), 'traceback': traceback.format_exc(),
                         'failed_utc': now()})
            raise
        finally:
            atomic_json(OUT / 'active_worker.json', {'pid': os.getpid(), 'stage': stage,
                        'finished_utc': now(), 'status': 'EXITED'})
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def data_frame(split, m, events=False):
    columns = ['hour_bjt', *m['original_names'], c.TARGET]
    frame = pd.read_parquet(DATA / f'{split}.parquet', columns=columns)
    require(len(frame) == m[f'n_{split}'] and frame.hour_bjt.is_unique,
            f'Invalid {split} partition')
    require(np.isfinite(frame[c.TARGET].to_numpy(dtype=float)).all(), 'Invalid target')
    if events:
        status = pd.read_parquet(DATA / 'hourly_sample_status.parquet',
                    columns=['hour_bjt', 'split', 'use_for_regression', 'clear_dry', 'rain',
                             'heavy_rain', 'dry_overcast'])
        frame = frame.merge(status, on='hour_bjt', how='left', validate='one_to_one')
        require(frame.split.eq(split).all() and frame.use_for_regression.eq(True).all(),
                'Status partition/quality mismatch')
        require(frame[['clear_dry', 'rain', 'dry_overcast', 'heavy_rain']].notna().all().all(),
                'Missing weather status')
        membership = frame[['clear_dry', 'rain', 'dry_overcast']].astype(int).sum(axis=1)
        require((membership <= 1).all() and (~frame.heavy_rain | frame.rain).all(),
                'Weather groups overlap or strong rain missing from rain')
        frame['other'] = membership.eq(0)
    return frame.sort_values('hour_bjt').reset_index(drop=True)


def raw(frame, m):
    x = frame[m['original_names']].to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    keep = set(m['retained_names'])
    x[:, [i for i, name in enumerate(m['original_names']) if name not in keep]] = np.nan
    return x


def fit_preprocess(x, name, m):
    indices = list(range(348)) if name in STRUCTURED else [
        m['original_names'].index(s) for s in m['retained_names']]
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    xf = imputer.fit_transform(x[:, indices])
    scaler = StandardScaler() if name in c.SCALED_MODELS else None
    if scaler is not None:
        xf = scaler.fit_transform(xf)
    return xf, {'imputer': imputer, 'scaler': scaler, 'positions': indices}


def transform(bundle, values):
    x = bundle['imputer'].transform(values[:, bundle['positions']])
    if bundle['scaler'] is not None:
        x = bundle['scaler'].transform(x)
    return x


def predict(bundle, values):
    return np.asarray(bundle['model'].predict(transform(bundle, values)), dtype=float).reshape(-1)


def save_model(path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    joblib.dump(bundle, temp, compress=3)
    os.replace(temp, path)


def record_path(folder, name, seed):
    return OUT / folder / f'{name}_seed{seed}.json'


def model_path(name, seed):
    return OUT / 'models' / f'{name}_seed{seed}.joblib'


def refit(m):
    require(not (OUT / 'test').exists(), 'Refitting is prohibited after test stage exists')
    train = data_frame('train', m)
    validation = data_frame('validation', m)
    require(set(train.hour_bjt).isdisjoint(set(validation.hour_bjt)), 'Train/validation hour overlap')
    xt, xv = raw(train, m), raw(validation, m)
    yf = train[c.TARGET].to_numpy(dtype=float)
    for name in m['models']:
        plan = m['plans'][name]
        for seed in m['seeds']:
            path = record_path('training', name, seed)
            if path.exists():
                old = read(path)
                require(old['fingerprint'] == m['fingerprint'] and
                        sha(model_path(name, seed)) == old['model_sha256'], 'Resume record mismatch')
                continue
            expected = read(PARENT / 'validation' / f'{name}_seed{seed}.json')
            require(expected['config'] == plan['best_config'] and
                    expected['formal_rounds'] == plan['formal_rounds'] and
                    expected['fingerprint'] == m['parent_model_fingerprint'], 'Frozen parameters differ')
            xf, bundle = fit_preprocess(xt, name, m)
            model, history, _, meta = c.fit_logged(name, plan['best_config'], seed,
                tuple(range(10)), xf, yf, fixed_rounds=plan['formal_rounds'])
            bundle.update({'model': model, 'model_name': name, 'seed': seed,
                           'fingerprint': m['fingerprint'], 'retained_names': m['retained_names'],
                           'config': c.model_config(name, plan['best_config'], plan['formal_rounds'])})
            val_pred = predict(bundle, xv)
            old_pred = np.asarray(expected['validation_prediction_db'], dtype=float)
            require(validation.hour_bjt.dt.strftime('%Y-%m-%dT%H:%M:%S%z').tolist() ==
                    expected['validation_hour_bjt'], 'Validation order differs')
            delta = float(np.max(np.abs(val_pred - old_pred)))
            require(delta <= m['validation_reproduction_max_abs_tolerance_db'],
                    f'Old validation predictions not reproduced: {name}/{seed}: {delta}')
            require(len(history) == len(expected['round_history']), 'Fixed history length differs')
            history_delta = float(max(abs(left['train_rmse_db'] - right['train_rmse_db'])
                for left, right in zip(history, expected['round_history'])))
            require(history_delta <= 1e-6, 'Old training curve not reproduced')
            save_model(model_path(name, seed), bundle)
            loaded = joblib.load(model_path(name, seed))
            loaded_delta = float(np.max(np.abs(predict(loaded, xv) - val_pred)))
            require(loaded_delta <= 1e-10, 'Saved/reloaded estimator changed')
            record = {'fingerprint': m['fingerprint'], 'model': name, 'seed': seed,
                'source_config_id': plan['best_config_id'], 'source_config': plan['best_config'],
                'effective_config': bundle['config'], 'formal_rounds': plan['formal_rounds'],
                'rounds_reason': plan['formal_rounds_reason'], 'fit_metadata': meta,
                'round_history': history, 'feature_names': m['retained_names'],
                'n_train': 2404, 'n_validation': 343, 'test_read_during_fit': False,
                'validation_used_to_change_fit': False, 'new_searches': 0,
                'train_hour_bjt': train.hour_bjt.astype(str).tolist(),
                'train_prediction_db': predict(bundle, xt).tolist(),
                'validation_hour_bjt': validation.hour_bjt.astype(str).tolist(),
                'validation_prediction_db': val_pred.tolist(),
                'validation_old_max_abs_difference_db': delta,
                'train_history_max_abs_difference_db': history_delta,
                'saved_reload_max_abs_difference_db': loaded_delta,
                'model_sha256': sha(model_path(name, seed)), 'completed_utc': now()}
            atomic_json(path, record)
            print(f'RECONSTRUCTED {name} seed={seed} old_validation_max_delta={delta:.3g}', flush=True)
            del bundle, loaded, model
            gc.collect()


def verify_refit(m, write=True):
    errors = []
    for name in m['models']:
        for seed in m['seeds']:
            row = read(record_path('training', name, seed))
            require(row['fingerprint'] == m['fingerprint'] and row['model'] == name and
                    row['seed'] == seed and row['feature_names'] == m['retained_names'] and
                    len(row['train_prediction_db']) == 2404 and
                    len(row['validation_prediction_db']) == 343 and
                    row['effective_config'] == c.model_config(name, m['plans'][name]['best_config'],
                                                             m['plans'][name]['formal_rounds']) and
                    not row['test_read_during_fit'] and not row['validation_used_to_change_fit'] and
                    sha(model_path(name, seed)) == row['model_sha256'], 'Reconstruction QA mismatch')
            require(row['validation_old_max_abs_difference_db'] <= 1e-6 and
                    row['saved_reload_max_abs_difference_db'] <= 1e-10, 'Reproduction QA failed')
            errors.append(row['validation_old_max_abs_difference_db'])
    qa = {'status': 'PASS', 'fingerprint': m['fingerprint'], 'reconstructed_models': 30,
          'training_records': 30, 'new_configuration_searches': 0,
          'test_read_for_fitting': False, 'validation_max_abs_reproduction_error_db': max(errors),
          'verified_utc': now()}
    if write:
        atomic_json(OUT / 'reconstruction_qa.json', qa)
    return qa


def weather_metrics(frame, pred):
    truth = frame[c.TARGET].to_numpy(dtype=float)
    days = frame.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    require(np.isfinite(pred).all() and len(pred) == len(frame), 'Invalid test prediction')
    out = {}
    for group in WEATHER:
        mask = np.ones(len(frame), dtype=bool) if group == 'overall' else frame[group].to_numpy(dtype=bool)
        error = pred[mask] - truth[mask]
        out[group] = {'n_hours': int(mask.sum()), 'n_dates': int(len(set(days[mask]))),
                      'rmse_db': float(np.sqrt(np.mean(error ** 2))),
                      'mae_db': float(np.mean(np.abs(error))), 'bias_db': float(np.mean(error))}
    return out


def latency(bundle, x):
    positions = np.linspace(0, len(x) - 1, 100, dtype=int)
    for index in positions[:20]:
        predict(bundle, x[index:index + 1])
    times = []
    for _ in range(3):
        for index in positions:
            start = time.perf_counter_ns()
            output = predict(bundle, x[index:index + 1])
            duration = time.perf_counter_ns() - start
            require(output.shape == (1,) and np.isfinite(output).all(), 'Invalid timed prediction')
            times.append(duration)
    return {'timed_calls': 300, 'warmup_calls': 20, 'samples_ns': times,
            'median_ms': float(np.median(times) / 1e6),
            'p95_ms': float(np.percentile(times, 95) / 1e6),
            'definition': 'single input row: frozen feature slice, median imputation, scaling, prediction'}


def evaluate(m):
    qa = read(OUT / 'reconstruction_qa.json')
    require(qa['status'] == 'PASS' and qa['fingerprint'] == m['fingerprint'], 'Reconstruction gate not passed')
    verify_refit(m, write=False)
    test = data_frame('test', m, events=True)
    require([weather_metrics(test, np.zeros(len(test)))[g]['n_hours'] for g in WEATHER] ==
            [687, 322, 51, 158, 156], 'Unexpected weather sample counts')
    train_dates = set(data_frame('train', m).hour_bjt.dt.strftime('%Y-%m-%d'))
    x = raw(test, m)
    for name in m['models']:
        for seed in m['seeds']:
            path = record_path('test', name, seed)
            if path.exists():
                old = read(path)
                require(old['fingerprint'] == m['fingerprint'] and
                        old['model_sha256'] == sha(model_path(name, seed)), 'Test resume mismatch')
                continue
            bundle = joblib.load(model_path(name, seed))
            prediction = predict(bundle, x)
            row = {'fingerprint': m['fingerprint'], 'model': name, 'seed': seed,
                   'model_sha256': sha(model_path(name, seed)), 'feature_names': m['retained_names'],
                   'effective_config': bundle['config'], 'fit_calls_during_evaluation': 0,
                   'hour_bjt': test.hour_bjt.astype(str).tolist(),
                   'truth_db': test[c.TARGET].to_numpy(dtype=float).tolist(),
                   'prediction_db': prediction.tolist(),
                   'weather_masks': {g: test[g].astype(bool).tolist() for g in WEATHER if g != 'overall'},
                   'metrics': weather_metrics(test, prediction), 'latency': latency(bundle, x),
                   'scored_utc': now()}
            atomic_json(path, row)
            print(f'TEST_SCORED {name} seed={seed} RMSE={row["metrics"]["overall"]["rmse_db"]:.6f}', flush=True)
            del bundle
            gc.collect()
    atomic_json(OUT / 'test_scoring_receipt.json', {'fingerprint': m['fingerprint'],
                'test_model_records': 30, 'hour_predictions': 20610, 'fit_calls': 0,
                'test_dates': len(set(test.hour_bjt.dt.strftime('%Y-%m-%d'))),
                'test_dates_overlapping_train': len(train_dates & set(test.hour_bjt.dt.strftime('%Y-%m-%d'))),
                'historical_test_access': True, 'completed_utc': now()})


def family_rows(m):
    result = []
    for name in m['models']:
        items = [read(record_path('test', name, seed)) for seed in m['seeds']]
        row = {'model': name, 'model_label': MODEL_LABELS[name],
               'formal_rounds': m['plans'][name]['formal_rounds'],
               'effective_config': items[0]['effective_config'], 'metrics': {}}
        for group in WEATHER:
            values = [item['metrics'][group] for item in items]
            summary = {'n_hours': values[0]['n_hours'], 'n_dates': values[0]['n_dates']}
            for metric in ('rmse_db', 'mae_db', 'bias_db'):
                scores = [v[metric] for v in values]
                summary[f'{metric}_mean'] = float(np.mean(scores))
                summary[f'{metric}_sd_seed'] = float(np.std(scores, ddof=1))
            summary['seed17'] = values[0]
            row['metrics'][group] = summary
        row['latency_median_ms'] = float(np.median([i['latency']['median_ms'] for i in items]))
        result.append(row)
    rmses = np.array([r['metrics']['overall']['rmse_db_mean'] for r in result])
    logtime = np.log([r['latency_median_ms'] for r in result])
    rnorm = np.zeros(10) if np.ptp(rmses) == 0 else (rmses - rmses.min()) / np.ptp(rmses)
    lnorm = np.zeros(10) if np.ptp(logtime) == 0 else (logtime - logtime.min()) / np.ptp(logtime)
    for i, row in enumerate(result):
        row.update({'normalized_test_rmse': float(rnorm[i]), 'normalized_log_latency': float(lnorm[i]),
                    'score_50_50': float((rnorm[i] + lnorm[i]) / 2)})
    rmse_sorted = sorted(result, key=lambda r: (r['metrics']['overall']['rmse_db_mean'],
                        r['metrics']['overall']['mae_db_mean'], r['model']))
    for i, row in enumerate(rmse_sorted, 1):
        row['rank_test_rmse'] = i
    balanced = sorted(result, key=lambda r: (r['score_50_50'], r['metrics']['overall']['rmse_db_mean'],
                       r['metrics']['overall']['mae_db_mean'], r['model']))
    for i, row in enumerate(balanced, 1):
        row['rank_test_50_50'] = i
    return rmse_sorted


def report(m):
    qa = read(OUT / 'independent_qa.json')
    require(qa['status'] == 'PASS' and qa['fingerprint'] == m['fingerprint'], 'Independent QA not passed')
    rows = family_rows(m)
    result = {'status': 'COMPLETE', 'fingerprint': m['fingerprint'], 'source_revision': '023',
              'n_features': 156, 'n_test_hours': 687, 'seeds': m['seeds'],
              'aggregation': 'arithmetic mean of seed-specific hour-equal metrics; not ensemble',
              'primary_test_rmse_winner': rows[0]['model'],
              'supplementary_test_50_50_winner': min(rows, key=lambda r: r['rank_test_50_50'])['model'],
              'model_rows': rows, 'source_preservation_verified': True,
              'reconstruction_qa': read(OUT / 'reconstruction_qa.json'),
              'independent_qa': qa, 'test_scoring_receipt': read(OUT / 'test_scoring_receipt.json'),
              'prior_two_model_table': {'bilstm_seed17': qa['bilstm_seed17_reference'],
                                      'separate_layer_svr_C10': qa['separate_svr_reference']},
              'test_based_family_ranking': True, 'fresh_blind_test': False, 'completed_utc': now()}
    atomic_json(OUT / 'final_report.json', result)
    summary, detailed, per_seed, hourly = [], [], [], []
    for row in rows:
        flat = {key: row[key] for key in ('rank_test_rmse', 'rank_test_50_50', 'model',
                                        'model_label', 'latency_median_ms', 'score_50_50', 'formal_rounds')}
        for group in WEATHER:
            for metric in ('rmse_db', 'mae_db', 'bias_db'):
                flat[f'{group}_{metric}'] = row['metrics'][group][f'{metric}_mean']
            detailed.append({'model': row['model'], 'weather': group, 'weather_label': LABELS[group],
                             **row['metrics'][group], 'seed17': json.dumps(row['metrics'][group]['seed17'])})
        summary.append(flat)
        for seed in m['seeds']:
            record = read(record_path('test', row['model'], seed))
            for group in WEATHER:
                per_seed.append({'model': row['model'], 'seed': seed, 'weather': group,
                                 **record['metrics'][group]})
            for i, hour in enumerate(record['hour_bjt']):
                hourly.append({'model': row['model'], 'seed': seed, 'hour_bjt': hour,
                               'truth_db': record['truth_db'][i], 'prediction_db': record['prediction_db'][i],
                               'error_db': record['prediction_db'][i] - record['truth_db'][i],
                               **{g: record['weather_masks'][g][i] for g in WEATHER if g != 'overall'}})
    pd.DataFrame(summary).to_csv(OUT / 'test_model_ranking.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(detailed).to_csv(OUT / 'weather_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_seed).to_csv(OUT / 'test_metrics_per_seed.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(hourly).to_csv(OUT / 'test_hourly_predictions.csv', index=False, encoding='utf-8-sig')
    lines = ['# 十模型测试结果（Revision 026）', '',
             '同一687测试小时，全部156列输入。参数与轮数继承Rev023，不根据测试回调。',
             '指标为三个种子各自评分后取平均，不是平均预测形成的集成模型。', '',
             '|测试RMSE排名|模型|总体|晴天|雨天|干燥阴天|其他天气|50/50排名|单条延迟ms|',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        scores = '|'.join(f'{row["metrics"][g]["rmse_db_mean"]:.6f}' for g in WEATHER)
        lines.append(f'|{row["rank_test_rmse"]}|{row["model_label"]}|{scores}|'
                     f'{row["rank_test_50_50"]}|{row["latency_median_ms"]:.6f}|')
    lines += ['', 'RMSE、MAE、Bias完整分天气数值见weather_metrics.csv；三种子见test_metrics_per_seed.csv。',
              '旧表BiLSTM 0.253694 dB来自seed17；本表主指标是三个种子平均。旧表SVR C=10为分层阶段另行调参，'
              '本表SVR C=1是十模型阶段冻结配置，二者不可混用。',
              '原记录与数据指纹未改变，重建30权重，测试阶段拟合次数0。测试已有历史访问，'
              '本次按用户要求进行测试集排名，不能称为全新独立盲测。所有156测试日期均与训练日期重叠。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'rmse_winner': result['primary_test_rmse_winner'],
                      'balanced_winner': result['supplementary_test_50_50_winner']}, indent=2), flush=True)


def status():
    return {'manifest_exists': MANIFEST.exists(),
            'fingerprint': read(MANIFEST)['fingerprint'] if MANIFEST.exists() else None,
            'training_records': len(list((OUT / 'training').glob('*.json'))),
            'saved_models': len(list((OUT / 'models').glob('*.joblib'))),
            'test_records': len(list((OUT / 'test').glob('*.json'))),
            'reconstruction_qa': read(OUT / 'reconstruction_qa.json') if (OUT / 'reconstruction_qa.json').exists() else None,
            'independent_qa': read(OUT / 'independent_qa.json').get('status') if (OUT / 'independent_qa.json').exists() else None,
            'active_worker': read(OUT / 'active_worker.json') if (OUT / 'active_worker.json').exists() else None,
            'complete': (OUT / 'final_report.json').exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'refit', 'verify-refit', 'evaluate', 'report', 'status'))
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return
    with lock(args.command):
        m = manifest(create=args.command == 'prepare')
        if args.command == 'prepare':
            print(json.dumps({'status': 'FROZEN', 'fingerprint': m['fingerprint'], 'models': m['models']}, indent=2))
        elif args.command == 'refit':
            refit(m)
        elif args.command == 'verify-refit':
            print(json.dumps(verify_refit(m), indent=2))
        elif args.command == 'evaluate':
            evaluate(m)
        else:
            report(m)


if __name__ == '__main__':
    main()
