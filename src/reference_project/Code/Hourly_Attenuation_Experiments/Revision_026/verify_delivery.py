"""Read-only replay of all saved estimators and delivered result tables."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'Revision_014'))
import models  # frozen class definitions needed to deserialize saved estimators
import joblib
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_026_Ten_Model_Test'


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
    report = read(OUT / 'final_report.json')
    require(report['status'] == 'COMPLETE' and report['independent_qa']['status'] == 'PASS', 'Missing report QA')
    data = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003'
    test = pd.read_parquet(data / 'test.parquet').sort_values('hour_bjt').reset_index(drop=True)
    x = test[m['original_names']].to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    keep = set(m['retained_names'])
    x[:, [i for i, name in enumerate(m['original_names']) if name not in keep]] = np.nan
    truth = test[m['target']].to_numpy(dtype=float)
    prediction_csv = pd.read_csv(OUT / 'test_hourly_predictions.csv')
    per_seed_csv = pd.read_csv(OUT / 'test_metrics_per_seed.csv')
    weather_csv = pd.read_csv(OUT / 'weather_metrics.csv')
    ranking_csv = pd.read_csv(OUT / 'test_model_ranking.csv')
    require(len(prediction_csv) == 20610 and len(per_seed_csv) == 150 and
            len(weather_csv) == 50 and len(ranking_csv) == 10, 'Delivered CSV rows differ')
    family_metrics = {}
    differences = []
    checks = 0
    for family in m['models']:
        family_metrics[family] = {}
        for seed in m['seeds']:
            record = read(OUT / 'test' / f'{family}_seed{seed}.json')
            bundle = joblib.load(OUT / 'models' / f'{family}_seed{seed}.joblib')
            processed = bundle['imputer'].transform(x[:, bundle['positions']])
            if bundle['scaler'] is not None:
                processed = bundle['scaler'].transform(processed)
            replay = np.asarray(bundle['model'].predict(processed), dtype=float).reshape(-1)
            difference = float(np.max(np.abs(replay - np.asarray(record['prediction_db']))))
            require(difference < 1e-10, f'Saved weight replay failed: {family}/{seed}')
            differences.append(difference)
            subset = prediction_csv[(prediction_csv.model == family) & (prediction_csv.seed == seed)]
            require(subset.hour_bjt.tolist() == record['hour_bjt'] and
                    np.max(np.abs(subset.truth_db.to_numpy() - truth)) < 1e-12 and
                    np.max(np.abs(subset.prediction_db.to_numpy() - replay)) < 1e-10 and
                    np.max(np.abs(subset.error_db.to_numpy() - (replay - truth))) < 1e-10,
                    'Per-hour CSV replay mismatch')
            for group in ('overall', 'clear_dry', 'rain', 'dry_overcast', 'other'):
                mask = np.ones(len(truth), dtype=bool) if group == 'overall' else np.asarray(record['weather_masks'][group])
                residual = replay[mask] - truth[mask]
                scores = [float(np.linalg.norm(residual) / np.sqrt(len(residual))),
                          float(np.mean(np.abs(residual))), float(np.mean(residual))]
                family_metrics[family].setdefault(group, []).append(scores)
                subset_score = per_seed_csv[(per_seed_csv.model == family) & (per_seed_csv.seed == seed) &
                                           (per_seed_csv.weather == group)].iloc[0]
                for index, metric in enumerate(('rmse_db', 'mae_db', 'bias_db')):
                    require(abs(subset_score[metric] - scores[index]) < 1e-10, 'Seed metric CSV mismatch')
                    checks += 1
            del bundle
    for family_row in report['model_rows']:
        family = family_row['model']
        for group, values in family_metrics[family].items():
            matrix = np.array(values)
            means, deviations = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
            csv_row = weather_csv[(weather_csv.model == family) & (weather_csv.weather == group)].iloc[0]
            for index, metric in enumerate(('rmse_db', 'mae_db', 'bias_db')):
                require(abs(family_row['metrics'][group][metric + '_mean'] - means[index]) < 1e-10 and
                        abs(csv_row[metric + '_mean'] - means[index]) < 1e-10 and
                        abs(csv_row[metric + '_sd_seed'] - deviations[index]) < 1e-10,
                        'Family mean/SD CSV/report mismatch')
                checks += 3
    names = [row['model'] for row in report['model_rows']]
    recomputed_order = sorted(names, key=lambda family: (
        np.mean(family_metrics[family]['overall'], axis=0)[0],
        np.mean(family_metrics[family]['overall'], axis=0)[1], family))
    require(names == recomputed_order == ranking_csv.model.tolist(), 'Primary test ranking differs')
    times = np.array([row['latency_median_ms'] for row in report['model_rows']])
    rmses = np.array([np.mean(family_metrics[name]['overall'], axis=0)[0] for name in names])
    logs = np.log(times)
    balanced_score = 0.5 * ((rmses-rmses.min())/np.ptp(rmses) + (logs-logs.min())/np.ptp(logs))
    balanced_order = sorted(range(10), key=lambda i: (balanced_score[i],rmses[i],
                            np.mean(family_metrics[names[i]]['overall'], axis=0)[1],names[i]))
    for rank, index in enumerate(balanced_order, 1):
        require(report['model_rows'][index]['rank_test_50_50'] == rank and
                abs(report['model_rows'][index]['score_50_50'] - balanced_score[index]) < 1e-12,
                'Supplementary 50/50 rank differs')
    for name, digest in m['source_sha256'].items():
        require(sha(ROOT / name) == digest, f'Frozen source changed: {name}')
    files = ['final_report.json', 'RESULTS.md', 'test_model_ranking.csv', 'weather_metrics.csv',
             'test_metrics_per_seed.csv', 'test_hourly_predictions.csv', 'Ten_Model_Test_Results.xlsx']
    from datetime import datetime, timezone
    result = {'status': 'PASS', 'fingerprint': m['fingerprint'],
              'loaded_model_replays': 30, 'test_replay_fits': 0,
              'max_abs_saved_weight_prediction_difference_db': max(differences),
              'csv_numeric_checks': checks, 'weather_rows': 50, 'seed_weather_rows': 150,
              'predictions': 20610, 'primary_ranking_independently_recomputed': True,
              'balanced_ranking_independently_recomputed': True,
              'all_frozen_sources_unchanged': True,
              'delivered_sha256': {name: sha(OUT / name) for name in files},
              'verified_utc': datetime.now(timezone.utc).isoformat()}
    (OUT / 'delivery_qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'delivered_sha256'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
