"""Independent source-to-prediction scoring and immutability audit; no fitting."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_026_Ten_Model_Test'
DATA = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003'

import numpy as np
import pandas as pd


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    m = read(OUT / 'run_manifest.json')
    for name, digest in m['source_sha256'].items():
        require(sha(ROOT / name) == digest, f'Source modified: {name}')
    require(read(OUT / 'reconstruction_qa.json')['status'] == 'PASS', 'Refit gate failed')
    table = pd.read_parquet(DATA / 'test.parquet', columns=['hour_bjt', m['target']])
    labels = pd.read_parquet(DATA / 'hourly_sample_status.parquet',
        columns=['hour_bjt', 'split', 'use_for_regression', 'clear_dry', 'rain', 'heavy_rain', 'dry_overcast'])
    table = table.merge(labels, on='hour_bjt', how='left', validate='one_to_one').sort_values('hour_bjt')
    table = table.reset_index(drop=True)
    truth = table[m['target']].to_numpy(dtype=float)
    require(len(table) == 687 and table.hour_bjt.is_unique and table.split.eq('test').all() and
            table.use_for_regression.eq(True).all(), 'Test partition mismatch')
    weather = {'overall': np.ones(len(table), dtype=bool)}
    for group in ('clear_dry', 'rain', 'dry_overcast'):
        weather[group] = table[group].to_numpy(dtype=bool)
    membership = weather['clear_dry'].astype(int) + weather['rain'].astype(int) + weather['dry_overcast'].astype(int)
    require((membership <= 1).all(), 'Weather not exclusive')
    weather['other'] = membership == 0
    require([int(mask.sum()) for mask in weather.values()] == [687, 322, 51, 158, 156], 'Weather counts differ')
    require((~table.heavy_rain | table.rain).all(), 'Rain did not merge strong rain')
    dates = table.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    checks, errors, output_hashes = 0, [], {}
    for family in m['models']:
        for seed in m['seeds']:
            path = OUT / 'test' / f'{family}_seed{seed}.json'
            row = read(path)
            train = read(OUT / 'training' / f'{family}_seed{seed}.json')
            require(row['fingerprint'] == train['fingerprint'] == m['fingerprint'] and
                    row['model'] == family and row['seed'] == seed and
                    row['hour_bjt'] == table.hour_bjt.astype(str).tolist() and
                    row['truth_db'] == truth.tolist() and row['feature_names'] == m['retained_names'] and
                    row['effective_config'] == train['effective_config'] and
                    row['fit_calls_during_evaluation'] == 0 and not train['test_read_during_fit'] and
                    not train['validation_used_to_change_fit'] and
                    sha(OUT / 'models' / f'{family}_seed{seed}.joblib') == row['model_sha256'] == train['model_sha256'],
                    'Identity/weights/config/source leak audit failed')
            require(set(row['hour_bjt']).isdisjoint(set(train['train_hour_bjt'])) and
                    set(row['hour_bjt']).isdisjoint(set(train['validation_hour_bjt'])), 'Exact hour leakage')
            pred = np.asarray(row['prediction_db'], dtype=float)
            require(len(pred) == 687 and np.isfinite(pred).all(), 'Invalid predictions')
            for group, mask in weather.items():
                if group != 'overall':
                    require(row['weather_masks'][group] == mask.tolist(), 'Weather label mismatch')
                residual = pred[mask] - truth[mask]
                calculated = {'rmse_db': float(np.linalg.norm(residual) / np.sqrt(len(residual))),
                              'mae_db': float(sum(abs(residual)) / len(residual)),
                              'bias_db': float(sum(residual) / len(residual))}
                saved = row['metrics'][group]
                require(saved['n_hours'] == int(mask.sum()) and
                        saved['n_dates'] == len(set(dates[mask])), 'Hours/dates mismatch')
                for metric, value in calculated.items():
                    difference = abs(saved[metric] - value)
                    require(difference < 1e-12, f'Metric recomputation failed: {family}/{seed}/{group}/{metric}')
                    errors.append(difference)
                    checks += 1
            times = row['latency']['samples_ns']
            require(len(times) == 300 and min(times) > 0 and
                    abs(np.median(times) / 1e6 - row['latency']['median_ms']) < 1e-12 and
                    abs(np.percentile(times, 95) / 1e6 - row['latency']['p95_ms']) < 1e-12,
                    'Inference latency does not recompute')
            output_hashes[str(path.relative_to(ROOT))] = sha(path)
            output_hashes[str((OUT / 'models' / f'{family}_seed{seed}.joblib').relative_to(ROOT))] = row['model_sha256']
    require(len(list((OUT / 'training').glob('*.json'))) == 30 and
            len(list((OUT / 'models').glob('*.joblib'))) == 30 and
            len(list((OUT / 'test').glob('*.json'))) == 30, 'Record count mismatch')
    old_bi_path = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_025_BiLSTM/layer/evaluation/mask_1023.json'
    old_bi = read(old_bi_path)
    require(old_bi['key'] == 1023 and not old_bi['model_refitted_during_evaluation'], 'Old Bi reference not frozen')
    old_bi_test = old_bi['results']['test']
    old_bi_hourly = old_bi_test['hourly_predictions']
    new_bi = read(OUT / 'test/bilstm_seed17.json')
    require(old_bi_hourly['hour_bjt'] == new_bi['hour_bjt'] and
            old_bi_hourly['truth_db'] == new_bi['truth_db'], 'Bi reference hours/truth differ')
    bi_delta = float(np.max(np.abs(np.asarray(old_bi_hourly['prediction_db']) -
                                  np.asarray(new_bi['prediction_db']))))
    require(bi_delta <= 1e-6, 'New Bi seed17 does not reproduce full-input old table')
    old_svr_path = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation/layer_test/mask_1023.json'
    old_svr = read(old_svr_path)
    require(old_svr['mask'] == 1023 and old_svr['hour_bjt'] == new_bi['hour_bjt'] and
            old_svr['truth_db'] == truth.tolist(), 'Separate old SVR reference differs')
    summary = {'status': 'PASS', 'fingerprint': m['fingerprint'],
        'source_fingerprints_checked': len(m['source_sha256']), 'source_files_unchanged': True,
        'metric_checks': checks, 'max_abs_metric_difference': max(errors),
        'models': 30, 'test_records': 30, 'hourly_predictions': 20610,
        'model_seed_weather_records': 150, 'fit_calls_during_test': 0,
        'output_sha256': output_hashes,
        'bilstm_seed17_reference': {'source': str(old_bi_path.relative_to(ROOT)),
            'sha256': sha(old_bi_path), 'max_abs_prediction_difference_db': bi_delta,
            'overall_rmse_db': new_bi['metrics']['overall']['rmse_db']},
        'separate_svr_reference': {'source': str(old_svr_path.relative_to(ROOT)),
            'sha256': sha(old_svr_path), 'ranked_in_this_benchmark': False,
            'config': {'C': 10.0, 'epsilon': 0.1, 'gamma': 'scale'},
            'old_overall_rmse_db': old_svr['metrics']['overall']['rmse']}}
    # No old path is written; the only new artifact is this independent QA.
    from datetime import datetime, timezone
    summary['verified_utc'] = datetime.now(timezone.utc).isoformat()
    path = OUT / 'independent_qa.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'output_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
