"""Train-only base-input refits paired with the frozen ten full-input models."""
from __future__ import annotations

import argparse
import contextlib
import gc
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
spec = importlib.util.spec_from_file_location('frozen26', HERE.parent / 'Revision_026/run_test_benchmark.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
c, np, pd, joblib = v.c, v.np, v.pd, v.joblib
OUT = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_027_Paired_Base_Full'
PARENT = v.OUT
OLD_BASE = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_025_BiLSTM/layer/evaluation/mask_0000.json'
read, sha, fp, now, require, atomic = v.read, v.sha, v.fp, v.now, v.require, v.atomic_json


def source_plan():
    parent = read(PARENT / 'run_manifest.json')
    for name, digest in parent['source_sha256'].items():
        require(sha(ROOT / name) == digest, f'Parent input changed: {name}')
    for name in ('reconstruction_qa.json', 'independent_qa.json', 'delivery_qa.json'):
        require(read(PARENT / name)['status'] == 'PASS', f'Parent QA missing: {name}')
    names = [s for s in parent['retained_names'] if s.startswith('surface__')]
    require(len(names) == 50, 'Base list is not 50')
    paths = [ROOT / s for s in parent['source_sha256']]
    paths += [PARENT / 'run_manifest.json', PARENT / 'final_report.json',
              PARENT / 'independent_qa.json', PARENT / 'reconstruction_qa.json',
              PARENT / 'delivery_qa.json', OLD_BASE, HERE / 'PROTOCOL.md',
              HERE / 'run_pair.py', HERE / 'verify_pair.py']
    paths += sorted((PARENT / 'models').glob('*.joblib'))
    paths += sorted((PARENT / 'test').glob('*.json'))
    obj = {'version': 'rev027-fixed-setting-paired-three-seed-v1',
           'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
           'parent_fingerprint': parent['fingerprint'], 'original_names': parent['original_names'],
           'retained_names': parent['retained_names'], 'base_names': names,
           'models': parent['models'], 'seeds': parent['seeds'], 'plans': parent['plans'],
           'target': parent['target'], 'n_train': 2404, 'n_validation': 343, 'n_test': 687,
           'acceptance_groups': ['overall', 'rain', 'dry_overcast'],
           'bootstrap_replicates': 4096, 'bootstrap_seed': 20261011,
           'bootstrap_unit': 'observed calendar date; paired hour multiplicities, fixed model seeds',
           'package_versions': v.versions(), 'historical_test_access': True,
           'test_used_for_fit': False, 'new_configuration_searches': 0}
    obj['fingerprint'] = fp(obj)
    return obj


def manifest(create=False):
    candidate = source_plan()
    path = OUT / 'run_manifest.json'
    if path.exists():
        saved = read(path)
        require(saved['fingerprint'] == candidate['fingerprint'], 'Frozen source mismatch')
        return saved
    require(create, 'prepare is required')
    candidate['created_utc'] = now()
    atomic(path, candidate)
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
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        atomic(OUT / 'active_worker.json', {'pid': os.getpid(), 'stage': stage, 'status': 'RUNNING',
                                           'command': sys.argv, 'started_utc': now()})
        try:
            yield
        except Exception:
            atomic(OUT / 'failures' / f'{time.time_ns()}_{stage}.json',
                   {'pid': os.getpid(), 'stage': stage, 'traceback': traceback.format_exc(), 'utc': now()})
            raise
        finally:
            atomic(OUT / 'active_worker.json', {'pid': os.getpid(), 'stage': stage, 'status': 'EXITED',
                                               'finished_utc': now()})
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def case_plan(m, case):
    result = dict(m)
    result['retained_names'] = m['base_names'] if case == 'base' else m['retained_names']
    return result


def model_path(name, seed):
    return OUT / 'models' / f'{name}_seed{seed}.joblib'


def record_path(folder, name, seed):
    return OUT / folder / f'{name}_seed{seed}.json'


def fit_base(m):
    require(not (OUT / 'evaluation').exists(), 'No new fit after test stage')
    base = case_plan(m, 'base')
    train, val = v.data_frame('train', m), v.data_frame('validation', m)
    require(set(train.hour_bjt).isdisjoint(set(val.hour_bjt)), 'Partition overlap')
    xt, xv = v.raw(train, base), v.raw(val, base)
    truth = train[c.TARGET].to_numpy(dtype=float)
    for name in m['models']:
        plan = m['plans'][name]
        for seed in m['seeds']:
            path = record_path('training', name, seed)
            if path.exists():
                row = read(path)
                require(row['fingerprint'] == m['fingerprint'] and sha(model_path(name, seed)) == row['model_sha256'],
                        'Resume mismatch')
                continue
            # Structured base networks have no pressure token, but preserve the 188 base slots.
            if name in v.STRUCTURED:
                indices = list(range(188))
                imputer = v.SimpleImputer(strategy='median', keep_empty_features=True)
                xf = imputer.fit_transform(xt[:, indices])
                scaler = v.StandardScaler()
                xf = scaler.fit_transform(xf)
                bundle = {'positions': indices, 'imputer': imputer, 'scaler': scaler}
            else:
                xf, bundle = v.fit_preprocess(xt, name, base)
            model, history, _, meta = c.fit_logged(name, plan['best_config'], seed, (), xf, truth,
                                                  fixed_rounds=plan['formal_rounds'])
            bundle.update({'model': model, 'model_name': name, 'seed': seed,
                           'fingerprint': m['fingerprint'], 'retained_names': m['base_names'],
                           'config': c.model_config(name, plan['best_config'], plan['formal_rounds'])})
            pred = v.predict(bundle, xv)
            delta = None
            if name == 'bilstm' and seed == 17:
                old = read(OLD_BASE)['results']['validation']['hourly_predictions']
                require(val.hour_bjt.astype(str).tolist() == old['hour_bjt'], 'Old base validation order differs')
                delta = float(np.max(np.abs(pred - np.asarray(old['prediction_db']))))
                require(delta <= 1e-6, 'Legacy base seed17 not reproduced')
            v.save_model(model_path(name, seed), bundle)
            loaded = joblib.load(model_path(name, seed))
            reload_delta = float(np.max(np.abs(v.predict(loaded, xv) - pred)))
            require(reload_delta <= 1e-10, 'Saved model changes prediction')
            row = {'fingerprint': m['fingerprint'], 'case': 'base', 'model': name, 'seed': seed,
                   'feature_names': m['base_names'], 'effective_config': bundle['config'],
                   'formal_rounds': plan['formal_rounds'], 'fit_metadata': meta, 'round_history': history,
                   'n_train': 2404, 'n_validation': 343, 'test_read_for_fit': False,
                   'validation_used_to_change_fit': False, 'new_searches': 0,
                   'train_hour_bjt': train.hour_bjt.astype(str).tolist(),
                   'train_prediction_db': v.predict(bundle, xt).tolist(),
                   'validation_hour_bjt': val.hour_bjt.astype(str).tolist(),
                   'validation_prediction_db': pred.tolist(),
                   'validation_metrics': v.weather_metrics(v.data_frame('validation', m, events=True), pred),
                   'old_base_seed17_validation_max_delta_db': delta,
                   'reload_max_delta_db': reload_delta, 'model_sha256': sha(model_path(name, seed)), 'utc': now()}
            atomic(path, row)
            print(f'BASE_FIT {name} seed={seed} train_rounds={plan["formal_rounds"]}', flush=True)
            del model, bundle, loaded
            gc.collect()


def verify_fit(m, write=True):
    errors = []
    for name in m['models']:
        for seed in m['seeds']:
            row = read(record_path('training', name, seed))
            require(row['fingerprint'] == m['fingerprint'] and row['feature_names'] == m['base_names'] and
                    row['effective_config'] == c.model_config(name, m['plans'][name]['best_config'],
                                                              m['plans'][name]['formal_rounds']) and
                    not row['test_read_for_fit'] and not row['validation_used_to_change_fit'] and
                    len(row['train_prediction_db']) == 2404 and len(row['validation_prediction_db']) == 343 and
                    sha(model_path(name, seed)) == row['model_sha256'] and row['reload_max_delta_db'] <= 1e-10,
                    f'Fit QA failed: {name}/{seed}')
            require(all(h['holdout_rmse_db'] is None for h in row['round_history']), 'Holdout used during fixed fit')
            errors.append(row['reload_max_delta_db'])
    result = {'status': 'PASS', 'fingerprint': m['fingerprint'], 'base_models': 30, 'training_records': 30,
              'test_read_for_fit': False, 'configuration_searches': 0, 'max_reload_delta_db': max(errors), 'utc': now()}
    if write:
        atomic(OUT / 'fit_qa.json', result)
    return result


def evaluate(m):
    require(read(OUT / 'fit_qa.json')['status'] == 'PASS', 'Fit QA not PASS')
    verify_fit(m, write=False)
    frame = v.data_frame('test', m, events=True)
    base_x = v.raw(frame, case_plan(m, 'base'))
    full_x = v.raw(frame, m)
    require([v.weather_metrics(frame, np.zeros(len(frame)))[g]['n_hours'] for g in v.WEATHER] ==
            [687, 322, 51, 158, 156], 'Weather counts differ')
    for name in m['models']:
        for seed in m['seeds']:
            path = record_path('evaluation', name, seed)
            if path.exists():
                require(read(path)['fingerprint'] == m['fingerprint'], 'Evaluation resume mismatch')
                continue
            base = joblib.load(model_path(name, seed))
            full_path = PARENT / 'models' / f'{name}_seed{seed}.joblib'
            full = joblib.load(full_path)
            old = read(PARENT / 'test' / f'{name}_seed{seed}.json')
            pred, full_pred = v.predict(base, base_x), v.predict(full, full_x)
            require(frame.hour_bjt.astype(str).tolist() == old['hour_bjt'] and
                    frame[c.TARGET].to_numpy().tolist() == old['truth_db'], 'Pair hour/truth mismatch')
            full_delta = float(np.max(np.abs(full_pred - np.asarray(old['prediction_db']))))
            require(full_delta <= 1e-10, 'Frozen full prediction changed')
            base_delta = None
            if name == 'bilstm' and seed == 17:
                reference = read(OLD_BASE)['results']['test']['hourly_predictions']
                require(reference['hour_bjt'] == old['hour_bjt'], 'Legacy base test hour mismatch')
                base_delta = float(np.max(np.abs(pred - np.asarray(reference['prediction_db']))))
                require(base_delta <= 1e-6, 'Legacy base test seed17 changed')
            row = {'fingerprint': m['fingerprint'], 'model': name, 'seed': seed,
                   'base_feature_names': m['base_names'], 'full_feature_names': m['retained_names'],
                   'effective_config': base['config'], 'fit_calls_during_evaluation': 0,
                   'base_model_sha256': sha(model_path(name, seed)), 'full_model_sha256': sha(full_path),
                   'full_reference_path': str((PARENT / 'test' / f'{name}_seed{seed}.json').relative_to(ROOT)),
                   'full_reference_sha256': sha(PARENT / 'test' / f'{name}_seed{seed}.json'),
                   'full_max_replay_delta_db': full_delta, 'old_base_seed17_test_delta_db': base_delta,
                   'hour_bjt': old['hour_bjt'], 'truth_db': old['truth_db'], 'weather_masks': old['weather_masks'],
                   'base_prediction_db': pred.tolist(), 'full_prediction_db': old['prediction_db'],
                   'base_metrics': v.weather_metrics(frame, pred), 'full_metrics': old['metrics'], 'utc': now()}
            atomic(path, row)
            print(f'PAIRED {name} seed={seed} base_RMSE={row["base_metrics"]["overall"]["rmse_db"]:.6f}'
                  f' full_RMSE={row["full_metrics"]["overall"]["rmse_db"]:.6f}', flush=True)
            del base, full
            gc.collect()


def date_interval(items, group, m):
    rows = items[0]
    mask = np.ones(687, dtype=bool) if group == 'overall' else np.array(rows['weather_masks'][group], bool)
    days = np.array([s[:10] for s in rows['hour_bjt']])[mask]
    dates, day_ix = np.unique(days, return_inverse=True)
    n = np.bincount(day_ix).astype(float)
    truth = np.array(rows['truth_db'])[mask]
    sums = {}
    for case in ('base', 'full'):
        sums[case] = np.array([np.bincount(day_ix,
            weights=(np.array(r[f'{case}_prediction_db'])[mask] - truth) ** 2,
            minlength=len(dates)) for r in items])
    # Multinomial counts are exactly the multiplicities of whole-date draws with replacement.
    rng = np.random.default_rng(m['bootstrap_seed'])
    count = rng.multinomial(len(dates), np.full(len(dates), 1 / len(dates)), size=m['bootstrap_replicates'])
    denominator = count @ n
    delta = (np.sqrt((count @ sums['base'].T) / denominator[:, None]) -
             np.sqrt((count @ sums['full'].T) / denominator[:, None])).mean(axis=1)
    return {'n_dates': len(dates), 'replicates': len(delta), 'unit': 'calendar date',
            'percentile_95_db': np.quantile(delta, [.025, .975]).tolist(),
            'bonferroni_three_endpoint_bounds_db': np.quantile(delta, [.05 / 6, 1 - .05 / 6]).tolist(),
            'fixed_seeds_not_resampled': m['seeds'], 'draw_sha256': fp(count.tolist())}


def summaries(m):
    rows = []
    for name in m['models']:
        items = [read(record_path('evaluation', name, seed)) for seed in m['seeds']]
        row = {'model': name, 'model_label': v.MODEL_LABELS[name], 'metrics': {}}
        for group in v.WEATHER:
            group_row = {}
            for case in ('base', 'full'):
                group_row[case] = {metric: float(np.mean([r[f'{case}_metrics'][group][metric] for r in items]))
                                  for metric in ('rmse_db', 'mae_db', 'bias_db')}
                group_row[case]['n_hours'] = items[0][f'{case}_metrics'][group]['n_hours']
            deltas = [r['base_metrics'][group]['rmse_db'] - r['full_metrics'][group]['rmse_db'] for r in items]
            group_row.update({'base_minus_full_rmse_db': float(np.mean(deltas)),
                              'seed_deltas_db': dict(zip(map(str, m['seeds']), deltas)),
                              'seed_delta_sd_db': float(np.std(deltas, ddof=1)),
                              'improved_seed_count': int(sum(d > 0 for d in deltas)),
                              'improved_mean': group_row['full']['rmse_db'] < group_row['base']['rmse_db']})
            if group in m['acceptance_groups']:
                group_row['date_bootstrap'] = date_interval(items, group, m)
            row['metrics'][group] = group_row
        row['all_three_mean_improvements'] = all(row['metrics'][g]['improved_mean'] for g in m['acceptance_groups'])
        rows.append(row)
    rows.sort(key=lambda r: (r['metrics']['overall']['full']['rmse_db'],
                             r['metrics']['overall']['full']['mae_db'], r['model']))
    for index, row in enumerate(rows, 1):
        row['full_overall_rmse_rank'] = index
    return rows


def report(m):
    qa = read(OUT / 'independent_qa.json')
    require(qa['status'] == 'PASS' and qa['fingerprint'] == m['fingerprint'], 'Independent QA gate')
    rows = summaries(m)
    winner = rows[0]
    result = {'stage_status': 'COMPLETE', 'fingerprint': m['fingerprint'], 'seeds': m['seeds'],
              'full_overall_winner': winner['model'], 'acceptance': 'MET' if winner['all_three_mean_improvements'] else 'NOT_MET',
              'acceptance_is_mean_metric_not_every_seed': True, 'model_rows': rows,
              'new_base_fits': 30, 'reused_full_weights': 30, 'new_tuning_trials': 0,
              'test_based_family_ranking': True, 'historical_test_access': True, 'fresh_blind_test': False,
              'uncertainty_is_conditional_exploratory': True, 'qa': qa, 'utc': now()}
    atomic(OUT / 'final_report.json', result)
    lines = ['# Revision 027 三种子基础／全层配对结果', '',
             '同一687测试小时；各种子独立计算指标后取平均，非集成预测。',
             f'全层总体第一名：{winner["model_label"]}；三个均值验收：{result["acceptance"]}。', '',
             '|全层排名|模型|总体基础→全层|雨天基础→全层|干燥阴天基础→全层|三项均值改善|',
             '|---:|---|---|---|---|---|']
    for row in rows:
        cells = [f'{row["metrics"][g]["base"]["rmse_db"]:.6f} → {row["metrics"][g]["full"]["rmse_db"]:.6f}'
                 for g in m['acceptance_groups']]
        lines.append(f'|{row["full_overall_rmse_rank"]}|{row["model_label"]}|{"|".join(cells)}|'
                     f'{"是" if row["all_three_mean_improvements"] else "否"}|')
    lines += ['', '三种子均值改善不代表每个种子都改善；种子17的雨天负结果保持在报告中。',
              '配置和轮数继承全层验证选型，是固定设置的输入消融，不是各输入池独立最优调参。',
              '训练阶段不读取测试目标，评价零拟合。测试历史访问及同日重叠保持审计；日期重采样区间仅为条件探索性区间。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'winner': winner['model'], 'acceptance': result['acceptance']}, indent=2))


def self_test():
    # Synthetic numbers only: aggregation must not become ensemble RMSE.
    errors = np.array([[1., -1.], [-1., 1.], [1., -1.]])
    mean_metric = np.sqrt((errors ** 2).mean(axis=1)).mean()
    require(mean_metric == 1 and np.sqrt((errors.mean(axis=0) ** 2).mean()) < mean_metric,
            'Seed metric/ensemble distinction failed')
    values = [{'full': .1, 'improved': False}, {'full': .2, 'improved': True}]
    winner = min(values, key=lambda r: r['full'])
    require(not winner['improved'], 'Do not filter winner by favorable effect')
    rng = np.random.default_rng(5)
    multiplicities = rng.multinomial(3, [1/3] * 3, size=50)
    require((multiplicities.sum(axis=1) == 3).all(), 'Date resampling multiplicities invalid')
    synthetic = {'hour_bjt': [f'2000-01-{1 + i // 229:02d}T{i % 24:02d}:00:00+08:00' for i in range(687)],
                 'truth_db': [0.] * 687, 'base_prediction_db': [1.] * 687,
                 'full_prediction_db': [.5] * 687, 'weather_masks': {'rain': [True] * 687}}
    interval = date_interval([synthetic] * 3, 'overall',
                             {'bootstrap_seed': 5, 'bootstrap_replicates': 128, 'seeds': [17, 29, 43]})
    require(interval['percentile_95_db'] == [.5, .5] and
            interval['bonferroni_three_endpoint_bounds_db'] == [.5, .5], 'Paired bootstrap constant difference failed')
    negative = dict(synthetic, base_prediction_db=[.25] * 687)
    reverse = date_interval([negative] * 3, 'rain',
                            {'bootstrap_seed': 5, 'bootstrap_replicates': 128, 'seeds': [17, 29, 43]})
    require(reverse['percentile_95_db'] == [-.25, -.25], 'Negative effects must remain negative')
    require(all([True, True, True]) and not all([True, False, True]), 'All three endpoints required')
    print(json.dumps({'status': 'PASS', 'synthetic_checks': 6, 'research_samples_read': 0}))


def status():
    m = read(OUT / 'run_manifest.json') if (OUT / 'run_manifest.json').exists() else {}
    return {'fingerprint': m.get('fingerprint'),
            'training_records': len(list((OUT / 'training').glob('*.json'))),
            'new_models': len(list((OUT / 'models').glob('*.joblib'))),
            'evaluation_records': len(list((OUT / 'evaluation').glob('*.json'))),
            'fit_qa': read(OUT / 'fit_qa.json')['status'] if (OUT / 'fit_qa.json').exists() else None,
            'independent_qa': read(OUT / 'independent_qa.json')['status'] if (OUT / 'independent_qa.json').exists() else None,
            'active_worker': read(OUT / 'active_worker.json') if (OUT / 'active_worker.json').exists() else None,
            'report_complete': (OUT / 'final_report.json').exists()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('self-test', 'prepare', 'fit-base', 'verify-fit', 'evaluate', 'report', 'status'))
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test()
    elif args.command == 'status':
        print(json.dumps(status(), indent=2))
    else:
        with lock(args.command):
            m = manifest(create=args.command == 'prepare')
            if args.command == 'prepare':
                print(json.dumps({'status': 'FROZEN', 'fingerprint': m['fingerprint']}))
            elif args.command == 'fit-base':
                fit_base(m)
            elif args.command == 'verify-fit':
                print(json.dumps(verify_fit(m), indent=2))
            elif args.command == 'evaluate':
                evaluate(m)
            else:
                report(m)


if __name__ == '__main__':
    main()
