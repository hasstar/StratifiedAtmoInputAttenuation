"""Independent source/metric/replay audit of Revision 027 paired outputs."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired27', HERE / 'run_pair.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
np, pd = p.np, p.pd


def verify():
    m = p.manifest()
    p.require(p.read(p.OUT / 'fit_qa.json')['status'] == 'PASS', 'Fit QA absent')
    source_count = 0
    for name, digest in m['source_sha256'].items():
        p.require(p.sha(p.ROOT / name) == digest, f'Changed source {name}')
        source_count += 1
    frame = p.v.data_frame('test', m, events=True)
    train = p.v.data_frame('train', m)
    val = p.v.data_frame('validation', m)
    hours = [set(f.hour_bjt.astype(str)) for f in (train, val, frame)]
    p.require(all(hours[i].isdisjoint(hours[j]) for i in range(3) for j in range(i)), 'Exact-hour leakage')
    truth = frame[m['target']].to_numpy(dtype=float)
    xbase, xfull = p.v.raw(frame, p.case_plan(m, 'base')), p.v.raw(frame, m)
    checks, replay_max = 0, 0.
    per_seed, hourly = [], []
    for name in m['models']:
        for seed in m['seeds']:
            training = p.read(p.record_path('training', name, seed))
            row = p.read(p.record_path('evaluation', name, seed))
            p.require(row['fingerprint'] == training['fingerprint'] == m['fingerprint'] and
                      row['model'] == name and row['seed'] == seed and row['fit_calls_during_evaluation'] == 0 and
                      row['base_feature_names'] == m['base_names'] and row['full_feature_names'] == m['retained_names'] and
                      row['effective_config'] == training['effective_config'], 'Pair identity/feature/config error')
            p.require(row['hour_bjt'] == frame.hour_bjt.astype(str).tolist() and
                      np.array_equal(row['truth_db'], truth), 'Truth or ordering differs')
            paths = {'base': p.model_path(name, seed), 'full': p.PARENT / 'models' / f'{name}_seed{seed}.joblib'}
            for case, path in paths.items():
                p.require(p.sha(path) == row[f'{case}_model_sha256'], 'Model digest changed')
                bundle = p.joblib.load(path)
                pred = np.asarray(row[f'{case}_prediction_db'], dtype=float)
                actual = p.v.predict(bundle, xbase if case == 'base' else xfull)
                delta = float(np.max(np.abs(actual - pred)))
                p.require(delta <= 1e-10, f'Replay differs {case}/{name}/{seed}')
                replay_max = max(replay_max, delta)
                for group in p.v.WEATHER:
                    mask = np.ones(len(frame), bool) if group == 'overall' else frame[group].to_numpy(dtype=bool)
                    if group != 'overall':
                        p.require(np.array_equal(mask, row['weather_masks'][group]), 'Weather mask differs')
                    err = pred[mask] - truth[mask]
                    values = {'rmse_db': float(np.sqrt(np.dot(err, err) / len(err))),
                              'mae_db': float(sum(abs(e) for e in err) / len(err)),
                              'bias_db': float(sum(err) / len(err))}
                    p.require(row[f'{case}_metrics'][group]['n_hours'] == int(mask.sum()), 'Weather count differs')
                    for metric, value in values.items():
                        p.require(abs(value - row[f'{case}_metrics'][group][metric]) <= 1e-12, 'Metric recomputation failed')
                        checks += 1
                    per_seed.append({'model': name, 'seed': seed, 'input': case, 'weather': group,
                                     'n_hours': int(mask.sum()), **values})
                for i, h in enumerate(row['hour_bjt']):
                    hourly.append({'model': name, 'seed': seed, 'input': case, 'hour_bjt': h,
                                   'truth_db': float(truth[i]), 'prediction_db': float(pred[i]),
                                   'error_db': float(pred[i] - truth[i]),
                                   **{g: bool(row['weather_masks'][g][i]) for g in p.v.WEATHER if g != 'overall'}})
                del bundle
    p.require(len(list((p.OUT / 'models').glob('*.joblib'))) == 30 and
              len(list((p.OUT / 'training').glob('*.json'))) == 30 and
              len(list((p.OUT / 'evaluation').glob('*.json'))) == 30 and
              len(per_seed) == 300 and len(hourly) == 41220 and checks == 900, 'Complete count failed')
    pd.DataFrame(per_seed).to_csv(p.OUT / 'per_seed_weather_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(hourly).to_csv(p.OUT / 'paired_hourly_predictions.csv', index=False, encoding='utf-8-sig')
    # Independence audit of fixed-setting refits: raw train only, no eval labels for optimization.
    old = p.read(p.PARENT / 'final_report.json')
    full_names = [r['model'] for r in old['model_rows']]
    full_order = sorted(m['models'], key=lambda name: (
        np.mean([r['rmse_db'] for r in per_seed if r['model'] == name and r['input'] == 'full' and r['weather'] == 'overall']),
        np.mean([r['mae_db'] for r in per_seed if r['model'] == name and r['input'] == 'full' and r['weather'] == 'overall']), name))
    p.require(full_order == full_names, 'Unchanged full model ranking not reproduced')
    result = {'status': 'PASS', 'fingerprint': m['fingerprint'], 'frozen_sources_unchanged': source_count,
              'new_models': 30, 'reused_models': 30, 'paired_evaluations': 30, 'metric_checks': checks,
              'paired_hourly_predictions': len(hourly), 'seed_input_weather_rows': len(per_seed),
              'max_prediction_replay_difference_db': replay_max, 'full_ranking_identical_to_rev026': True,
              'test_fit_calls': 0, 'test_dates_overlapping_train': len(set(frame.hour_bjt.dt.strftime('%Y-%m-%d')) &
                  set(train.hour_bjt.dt.strftime('%Y-%m-%d'))), 'historical_test_access': True, 'utc': p.now()}
    p.atomic(p.OUT / 'independent_qa.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with p.lock('independent-verify'):
        verify()
