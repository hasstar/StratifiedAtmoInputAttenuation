"""Resumable current53 BiLSTM layer and column refits; official evaluation last."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import time

import runtime as r


def check_existing(kind, key, names, m):
    state = r.read(r.state_path(kind, key))
    r.require(state['fingerprint'] == m['fingerprint'] and state['kind'] == kind and state['key'] == key and
              state['feature_names'] == names and state['seed'] == 17 and state['epochs'] == 53, 'Checkpoint identity')
    r.require(r.sha(r.model_path(kind, key)) == state['full_fit']['model_sha256'], 'Formal checkpoint digest')
    for row in state['folds']:
        r.require(r.sha(r.model_path(kind, key, row['fold'])) == row['model_sha256'], 'Fold checkpoint digest')
    return state


def fit_record(kind, key, fold_id, data, names, m, hold=None):
    xf, bundle = r.preprocess(data, names, m)
    xh = None if hold is None else r.v.transform(bundle, r.raw(hold, names, m))
    model, history, metadata = r.fit_fixed(xf, data, bundle['active_positions'], m['anchor_plan']['choice'], xh, hold)
    bundle.update(model=model, fingerprint=m['fingerprint'], kind=kind, key=key, fold=fold_id,
                  choice=metadata['choice'], training_hour_sha256=r.fp(data.hour_bjt.astype(str).tolist()))
    target = r.model_path(kind, key, fold_id)
    digest = r.save_model(target, bundle)
    saved = r.joblib.load(target)
    prediction = model.predict(xf)
    replay = r.v.predict(saved, r.raw(data, names, m))
    r.require(float(r.np.max(r.np.abs(prediction - replay))) < 1e-10, 'Model save/reload mismatch')
    receipt = r.OUT / 'fit_events' / f'{time.time_ns()}_{kind}_{key}_{fold_id}.json'
    r.atomic(receipt, {'fingerprint': m['fingerprint'], 'kind': kind, 'key': key, 'fold': fold_id,
                      'seed': 17, 'epochs': 53, 'model_sha256': digest,
                      'training_hour_sha256': bundle['training_hour_sha256'], 'n_fit': len(data), 'utc': r.now()})
    record = {'fold': fold_id, 'model_sha256': digest, 'model_weights_sha256': r.weights_digest(model),
              'fit_receipt': str(receipt.relative_to(r.OUT)), 'fit_metadata': metadata, 'round_history': history,
              'training': r.prediction_record(data, prediction),
              'holdout': None if hold is None else r.prediction_record(hold, model.predict(xh))}
    return record, bundle, xf, xh


def make_state(kind, key, names, train, fold_list, m):
    fold_records = []
    importance = [{'feature_name': name, 'fold_delta_rmse_db': [], 'fold_permuted_rmse_db': []} for name in names] if kind == 'rfe' and key > 1 else []
    for fold in fold_list:
        fit, hold = train.iloc[fold['fit']], train.iloc[fold['hold']]
        row, bundle, xf, xh = fit_record(kind, key, fold['fold'], fit, names, m, hold)
        baseline = row['holdout']['metrics']['overall']['rmse_db']
        row.update({k: fold[k] for k in ('fit_days', 'hold_days', 'fit_index_sha256', 'hold_index_sha256')})
        if importance:
            y = hold[r.TARGET].to_numpy(float)
            for item in importance:
                original_index = m['original_names'].index(item['feature_name'])
                slot = bundle['positions'].index(original_index)
                changed = xh.copy()
                rng = r.np.random.default_rng(r.permutation_seed(m, key, fold['fold'], item['feature_name']))
                changed[:, slot] = xh[rng.permutation(len(hold)), slot]
                permuted = bundle['model'].predict(changed)
                score = float(r.np.sqrt(r.np.mean((permuted - y) ** 2)))
                item['fold_permuted_rmse_db'].append(score)
                item['fold_delta_rmse_db'].append(score - baseline)
        fold_records.append(row)
        del bundle, xf, xh
        gc.collect()
    for item in importance:
        item['mean_delta_rmse_db'] = float(r.np.mean(item['fold_delta_rmse_db']))
    removed = min(importance, key=lambda i: (i['mean_delta_rmse_db'], i['feature_name']))['feature_name'] if importance else None
    full, bundle, _, _ = fit_record(kind, key, 0, train, names, m)
    if len(names) == 156:
        anchor = r.read(r.PARENT / 'training/full_bilstm_seed17.json')
        delta = float(r.np.max(r.np.abs(r.np.asarray(anchor['train_prediction_db']) - full['training']['prediction_db'])))
        r.require(delta < 1e-8, 'Current53 full train prediction anchor failed')
        parent_model = r.joblib.load(r.PARENT / 'models/full_bilstm_seed17.joblib')['model']
        r.require(r.weights_digest(parent_model) == full['model_weights_sha256'], 'Current53 full weights anchor failed')
        full['rev028_anchor_max_delta_db'] = delta
    cv = r.np.asarray([row['holdout']['metrics']['overall']['rmse_db'] for row in fold_records])
    state = {'fingerprint': m['fingerprint'], 'kind': kind, 'key': key, 'seed': 17, 'epochs': 53,
             'feature_names': names, 'n_features': len(names), 'n_base': sum(n.startswith('surface__') for n in names),
             'n_pressure': sum(not n.startswith('surface__') for n in names), 'active_positions': list(r.active(names)),
             'levels_hpa': [r.LEVELS[i] for i in r.active(names)], 'folds': fold_records, 'full_fit': full,
             'cv_mean_rmse_db': float(cv.mean()), 'cv_se_rmse_db': float(cv.std(ddof=1) / r.np.sqrt(3)),
             'permutation_importance': importance, 'removed_feature_next': removed,
             'official_holdouts_used_for_training_or_deletion': False, 'completed_utc': r.now()}
    r.atomic(r.state_path(kind, key), state)
    del bundle
    gc.collect()
    return state


def run_path(kind, m):
    r.require(not (r.OUT / 'holdout_opened.json').exists(), 'No training after official evaluation opened')
    r.set_stage(f'{kind}-run')
    r.progress(f'{kind}训练运行中')
    train = r.frame('train', m)
    fold_list = r.folds(train)
    names = m['retained_names'].copy()
    order = [1023, *range(1023)] if kind == 'layer' else r.keys(kind)
    for key in order:
        current = r.names_for_mask(key, m['retained_names']) if kind == 'layer' else names
        if r.state_path(kind, key).exists():
            state = check_existing(kind, key, current, m)
        else:
            state = make_state(kind, key, current, train, fold_list, m)
            print(json.dumps({'stage': kind, 'completed_key': key, 'features': len(current),
                              'cv_rmse_db': state['cv_mean_rmse_db'], 'removed_next': state['removed_feature_next']}, ensure_ascii=False), flush=True)
        if kind == 'rfe' and key > 1:
            r.require(state['removed_feature_next'] in names, 'RFE deletion missing')
            names = [n for n in names if n != state['removed_feature_next']]
    r.progress(f'{kind}训练状态齐全，等待独立QA', {'states': len(order), 'committed_fits': len(order) * 4})


def freeze_training(m):
    import verify_independent as q
    r.set_stage('verify-train')
    qa = q.verify_training(m)
    selections = {}
    for kind in ('layer', 'rfe'):
        rows = [r.read(r.state_path(kind, k)) for k in r.keys(kind)]
        best = min(rows, key=lambda i: (i['cv_mean_rmse_db'], i['key']))
        threshold = best['cv_mean_rmse_db'] + best['cv_se_rmse_db']
        eligible = [i for i in rows if i['cv_mean_rmse_db'] <= threshold]
        selected = min(eligible, key=lambda i: (len(i['levels_hpa']) if kind == 'layer' else i['n_features'], i['cv_mean_rmse_db'], i['key']))
        selections[kind] = {'selected_key': selected['key'], 'feature_names': selected['feature_names'],
                            'levels_hpa': selected['levels_hpa'], 'minimum_cv_key': best['key'],
                            'one_se_threshold_db': threshold, 'selected_cv_rmse_db': selected['cv_mean_rmse_db'],
                            'official_holdouts_used_for_selection': False}
    result = {'fingerprint': m['fingerprint'], 'selections': selections,
              'train_qa_sha256': r.sha(r.OUT / 'train_qa.json'), 'frozen_utc': r.now()}
    path = r.OUT / 'selection.json'
    if path.exists():
        saved = r.read(path)
        r.require(saved['fingerprint'] == m['fingerprint'] and saved['selections'] == selections, 'Training selection changed')
        return saved
    r.atomic(path, result)
    r.progress('两路径训练独立QA PASS、训练内方案冻结；尚未进行本轮官方评价', selections)
    return result


def evaluate(m):
    r.require(r.manifest()['fingerprint'] == m['fingerprint'], 'Frozen sources changed before official evaluation')
    r.set_stage('evaluate')
    qa, chosen = r.read(r.OUT / 'train_qa.json'), r.read(r.OUT / 'selection.json')
    r.require(qa['status'] == 'PASS' and qa['fingerprint'] == chosen['fingerprint'] == m['fingerprint'], 'Training QA gate')
    r.require(qa['qa_input_sha256'] == __import__('verify_independent').training_inventory(m), 'Verified training artifacts changed')
    __import__('verify_independent').check_selection(m)
    opened = r.OUT / 'holdout_opened.json'
    if not opened.exists():
        r.atomic(opened, {'fingerprint': m['fingerprint'], 'selection_sha256': r.sha(r.OUT / 'selection.json'),
                         'zero_new_fits_required': True, 'opened_utc': r.now()})
    frames = {split: r.frame(split, m) for split in ('validation', 'test')}
    train_days = set(r.frame('train', m).hour_bjt.dt.strftime('%Y-%m-%d'))
    test_days = set(frames['test'].hour_bjt.dt.strftime('%Y-%m-%d'))
    r.require([len(frames['test']) if g == 'overall' else int(frames['test'][g].sum()) for g in r.WEATHER] == [687, 322, 51, 158, 156], 'Test group counts')
    optimizer = r.torch.optim.AdamW
    def forbid(*args, **kwargs):
        raise RuntimeError('Fitting/optimizer construction forbidden in evaluation')
    original_fit = r.fit_fixed
    r.fit_fixed = forbid
    r.torch.optim.AdamW = forbid
    try:
        for kind in ('layer', 'rfe'):
            for key in r.keys(kind):
                target = r.evaluation_path(kind, key)
                state = r.read(r.state_path(kind, key))
                if target.exists():
                    old = r.read(target)
                    r.require(old['fingerprint'] == m['fingerprint'] and old['model_sha256'] == state['full_fit']['model_sha256'], 'Evaluation checkpoint changed')
                    continue
                bundle = r.joblib.load(r.model_path(kind, key))
                results = {split: r.prediction_record(data, r.v.predict(bundle, r.raw(data, state['feature_names'], m))) for split, data in frames.items()}
                r.atomic(target, {'fingerprint': m['fingerprint'], 'kind': kind, 'key': key, 'seed': 17, 'epochs': 53,
                                  'feature_names': state['feature_names'], 'model_sha256': state['full_fit']['model_sha256'],
                                  'results': results, 'fit_calls_during_evaluation': 0, 'selection_sha256': r.sha(r.OUT / 'selection.json'),
                                  'scored_utc': r.now()})
                print(json.dumps({'stage': 'evaluate', 'kind': kind, 'key': key, 'test_rmse_db': results['test']['metrics']['overall']['rmse_db']}), flush=True)
    finally:
        r.fit_fixed = original_fit
        r.torch.optim.AdamW = optimizer
    r.atomic(r.OUT / 'evaluation_receipt.json', {'fingerprint': m['fingerprint'], 'models_scored': 1180,
             'validation_predictions': 404740, 'test_predictions': 810660, 'fit_calls': 0,
             'test_dates': len(test_days), 'test_dates_overlapping_training': len(test_days & train_days), 'utc': r.now()})


def synthetic():
    parent, choice = r.source_inputs()
    rng = r.np.random.default_rng(20261012)
    data = r.pd.DataFrame(rng.normal(size=(64, 348)), columns=parent['original_names'])
    data['hour_bjt'] = r.pd.date_range('2020-01-01', periods=64, freq='D')
    data[r.TARGET] = rng.normal(size=64)
    for i, group in enumerate(r.WEATHER[1:]):
        data[group] = r.np.arange(64) % 4 == i
    checks = []
    all_names = parent['retained_names']
    for mask in (0, 1, 1023):
        names = r.names_for_mask(mask, all_names)
        xf, bundle = r.preprocess(data, names, parent)
        model, history, meta = r.fit_fixed(xf, data, bundle['active_positions'], choice['choice'], epochs=3)
        ref, old_history, p, _, _ = r.engine.fit('bilstm', choice['choice'], 17, bundle['active_positions'], xf, data, fixed_rounds=3)
        r.require(r.np.max(r.np.abs(p - model.predict(xf))) < 1e-10 and r.weights_digest(ref) == r.weights_digest(model), 'Frozen training loop equivalence')
        r.require(len(history) == 3 and len(names) == (50 if mask == 0 else 156 if mask == 1023 else len(names)), 'Synthetic state layout')
        altered = data.copy()
        excluded = [n for n in parent['original_names'] if n not in names]
        altered[excluded] += 99999
        r.require(r.np.array_equal(r.v.transform(bundle, r.raw(altered, names, parent)), xf), 'Excluded input affects preprocessing')
        checks.append({'mask': mask, 'layout_width': xf.shape[1], 'features': len(names), 'loop_equal': True})
    name = all_names[-1]
    xf, bundle = r.preprocess(data, [name], parent)
    r.require(xf.shape[1] == 204 and len(bundle['active_positions']) == 1, 'k1 pressure-only layout')
    fold_list = r.folds(data)
    fit, hold = data.iloc[fold_list[0]['fit']], data.iloc[fold_list[0]['hold']]
    xf, b = r.preprocess(fit, all_names, parent)
    xh = r.v.transform(b, r.raw(hold, all_names, parent))
    model, history, _ = r.fit_fixed(xf, fit, b['active_positions'], choice['choice'], xh, hold, epochs=2)
    r.require(all(h['internal_holdout_metrics'] is not None and h['validation_objective_db'] is None for h in history), 'Diagnostic-only fold curves')
    record = r.prediction_record(data, data[r.TARGET].to_numpy(float) + .5)
    r.require(all(abs(record['metrics'][g]['rmse_db'] - .5) < 1e-12 and abs(record['metrics'][g]['bias_db'] - .5) < 1e-12 for g in r.WEATHER), 'Weather metric/Bias meaning')
    r.require(sum(range(2, 157)) * 3 == 36735 and len({tuple(r.names_for_mask(mask, all_names)) for mask in range(1024)}) == 1024, 'Exhaustive state/importance budget')
    # Exercise checkpoint/replay/PFI plumbing before freezing or real main fits.
    import verify_independent as independent
    main_output = r.OUT
    api_output = main_output / f'synthetic_api_{time.time_ns()}'
    local_manifest = dict(parent, anchor_plan=choice, levels_hpa=list(r.LEVELS),
                          fingerprint=r.fp({'synthetic': True, 'runner_sha256': r.sha(r.HERE / 'run_analysis.py')}))
    small_names = [all_names[0], all_names[1], all_names[-1]]
    try:
        r.OUT = api_output
        for k in (3, 2, 1):
            state = make_state('rfe', k, small_names, data, fold_list, local_manifest)
            for rec, definition in zip(state['folds'], fold_list):
                independent.check_fit(rec, data.iloc[definition['fit']], data.iloc[definition['hold']], small_names, local_manifest, 'rfe', k)
            independent.check_fit(state['full_fit'], data, None, small_names, local_manifest, 'rfe', k)
            r.require(len(state['permutation_importance']) == (k if k > 1 else 0), 'Synthetic RFE importance count')
            replay = check_existing('rfe', k, small_names, local_manifest)
            r.require(replay['full_fit']['model_sha256'] == state['full_fit']['model_sha256'], 'Synthetic checkpoint replay')
            if k > 1:
                r.require(state['removed_feature_next'] in small_names, 'Synthetic deletion membership')
                small_names = [n for n in small_names if n != state['removed_feature_next']]
            else:
                r.require(state['removed_feature_next'] is None, 'Synthetic k1 termination')
    finally:
        r.OUT = main_output
    q = {'status': 'PASS', 'research_samples_used': 0, 'checks': checks, 'synthetic_fit_calls': 19,
         'synthetic_saved_fold_models': 9, 'synthetic_saved_formal_models': 3,
         'synthetic_api_source': str(api_output.relative_to(r.ROOT)),
         'date_fold_and_metric_checks': True, 'runtime_sha256': r.sha(r.HERE / 'runtime.py'),
         'code_sha256': r.code_hashes(), 'utc': r.now()}
    r.atomic(r.OUT / 'synthetic_qa.json', q)
    return q


def anchor():
    parent, plan = r.source_inputs()
    train = r.frame('train', parent)
    xf, bundle = r.preprocess(train, parent['retained_names'], parent)
    _, _, legacy_xf, _, old_bundle = r.w.preprocess(train, train, parent, 'bilstm', 'full')
    r.require(r.np.array_equal(xf, legacy_xf), 'Full preprocessing not identical')
    model, history, meta = r.fit_fixed(xf, train, bundle['active_positions'], plan['choice'])
    prediction = model.predict(xf)
    saved = r.read(r.PARENT / 'training/full_bilstm_seed17.json')
    original = r.joblib.load(r.PARENT / 'models/full_bilstm_seed17.joblib')
    r.require(saved['train_hour_bjt'] == train.hour_bjt.astype(str).tolist(), 'Anchor hours differ')
    difference = float(r.np.max(r.np.abs(prediction - saved['train_prediction_db'])))
    round_difference = max(abs(a['train_rmse_db'] - b['train_rmse_db']) for a, b in zip(history, saved['round_history']))
    r.require(difference < 1e-8 and round_difference < 1e-8 and r.weights_digest(model) == r.weights_digest(original['model']), 'Current53 anchor mismatch')
    r.require(r.np.array_equal(bundle['imputer'].statistics_, original['imputer'].statistics_) and
              r.np.array_equal(bundle['scaler'].mean_, original['scaler'].mean_) and
              r.np.array_equal(bundle['scaler'].scale_, original['scaler'].scale_), 'Anchor preprocessing parameters differ')
    q = {'status': 'PASS', 'real_fit_calls': 1, 'train_hours': 2404, 'official_holdout_samples_used': 0,
         'runtime_sha256': r.sha(r.HERE / 'runtime.py'), 'code_sha256': r.code_hashes(), 'train_prediction_max_difference_db': difference,
         'round_rmse_max_difference_db': round_difference, 'network_weights_equal': True,
         'preprocessing_equal': True, 'epochs': 53, 'seed': 17, 'fit_metadata': meta, 'utc': r.now()}
    r.atomic(r.OUT / 'anchor_qa.json', q)
    return q


def status(deep=False):
    result = {'manifest_exists': (r.OUT / 'run_manifest.json').exists(), 'os_lock_available': r.os_lock_available(),
              'active_worker': r.read(r.OUT / 'active_worker.json') if (r.OUT / 'active_worker.json').exists() else None}
    if result['manifest_exists']:
        m = r.manifest() if deep else r.read(r.OUT / 'run_manifest.json')
        result.update(fingerprint=m['fingerprint'], frozen_source_hashes_checked=deep)
    for kind in ('layer', 'rfe'):
        result[kind] = {folder: len(list((r.OUT / kind / folder).glob('*.json' if folder in ('states', 'evaluation') else '*.joblib'))) for folder in ('states', 'models', 'fold_models', 'evaluation')}
        paths = sorted((r.OUT / kind / 'states').glob('*.json'), key=lambda p: p.stat().st_mtime)
        result[kind]['last_state'] = None if not paths else {'file': paths[-1].name, 'sha256': r.sha(paths[-1]), 'key': r.read(paths[-1])['key']}
    result['fit_events'] = len(list((r.OUT / 'fit_events').glob('*.json')))
    result['qa'] = {n: r.read(r.OUT / f'{n}.json').get('status') if (r.OUT / f'{n}.json').exists() else None for n in ('synthetic_qa', 'anchor_qa', 'train_qa', 'evaluation_qa', 'report_qa')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-test', 'anchor', 'prepare', 'layer-run', 'rfe-run', 'verify-train', 'evaluate', 'verify-evaluation', 'report', 'all', 'status'))
    parser.add_argument('--deep', action='store_true')
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(args.deep), indent=2, ensure_ascii=False))
        return
    with r.writer_lock(args.command):
        if args.command == 'self-test':
            print(json.dumps(synthetic(), ensure_ascii=False)); return
        if args.command == 'anchor':
            print(json.dumps(anchor(), ensure_ascii=False)); return
        m = r.manifest(create=args.command == 'prepare')
        if args.command == 'prepare':
            print(json.dumps({'status': 'FROZEN', 'fingerprint': m['fingerprint']})); return
        if args.command in ('layer-run', 'rfe-run'):
            run_path(args.command.split('-')[0], m); return
        if args.command == 'verify-train':
            freeze_training(m)
            return
        if args.command == 'all' and not (r.OUT / 'selection.json').exists():
            if args.command == 'all':
                run_path('layer', m)
                run_path('rfe', m)
            freeze_training(m)
        if args.command in ('all', 'evaluate'):
            evaluate(m)
        if args.command in ('all', 'verify-evaluation'):
            import verify_independent as q
            r.set_stage('verify-evaluation')
            print(json.dumps(q.verify_evaluation(m), ensure_ascii=False), flush=True)
        if args.command in ('all', 'report'):
            import make_report as report
            r.set_stage('report')
            print(json.dumps(report.generate(m), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
