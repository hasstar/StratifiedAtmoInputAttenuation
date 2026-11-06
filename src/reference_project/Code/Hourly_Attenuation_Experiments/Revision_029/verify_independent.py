"""Independent prediction, PFI deletion, count and delivery recomputation."""
from __future__ import annotations

import csv
import hashlib
import json

import runtime as r


def close(a, b, message, atol=1e-10):
    if a is None or b is None:
        r.require(a is b, message)
    else:
        r.require(abs(float(a) - float(b)) <= atol, message)


def score_arrays(y, p, mask):
    error = r.np.asarray(p, float)[mask] - r.np.asarray(y, float)[mask]
    return {'rmse_db': None if not len(error) else float(r.np.sqrt(r.np.dot(error, error) / len(error))),
            'mae_db': None if not len(error) else float(r.np.sum(r.np.abs(error)) / len(error)),
            'bias_db': None if not len(error) else float(r.np.sum(error) / len(error))}


def check_prediction(row, data, bundle=None, names=None, m=None):
    r.require(row['hour_bjt'] == data.hour_bjt.astype(str).tolist() and
              row['truth_db'] == data[r.TARGET].to_numpy(float).tolist(), 'Prediction source hours/targets differ')
    y, p = r.np.asarray(row['truth_db']), r.np.asarray(row['prediction_db'])
    r.require(len(p) == len(data) and r.np.isfinite(p).all(), 'Incomplete/nonfinite predictions')
    dates = data.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    for group in r.WEATHER:
        mask = r.np.ones(len(data), bool) if group == 'overall' else data[group].to_numpy(bool)
        if group != 'overall':
            r.require(row['weather_masks'][group] == mask.tolist(), 'Weather source masks differ')
        saved = row['metrics'][group]
        r.require(saved['n_hours'] == int(mask.sum()) and saved['n_dates'] == len(set(dates[mask])), 'Weather sample/date count')
        for key, number in score_arrays(y, p, mask).items():
            close(number, saved[key], f'Independent {group}/{key} score differs')
    if bundle is not None:
        replay = r.v.predict(bundle, r.raw(data, names, m))
        r.require(r.np.max(r.np.abs(replay - p)) < 1e-8, 'Saved model replay differs')
    return p


def check_model(bundle, data, names, m, kind, key, fold, metadata):
    expected_positions = list(range(188))
    expected_active = []
    for i, level in enumerate(m['levels_hpa']):
        if any(name.startswith(f'p{level}__') for name in names):
            expected_active.append(i)
            expected_positions += list(range(188 + i * 16, 188 + (i + 1) * 16))
    r.require(bundle['positions'] == expected_positions and list(bundle['active_positions']) == expected_active and
              bundle['retained_names'] == names and bundle['fingerprint'] == m['fingerprint'] and
              (bundle['kind'], bundle['key'], bundle['fold'], bundle['seed']) == (kind, key, fold, 17), 'Model layout/identity')
    r.require(tuple(bundle['model'].net.active_positions) == tuple(expected_active), 'Unused pressure token in network')
    r.require(bundle['training_hour_sha256'] == r.fp(data.hour_bjt.astype(str).tolist()), 'Model fitting-hour source')
    close(bundle['model'].target_mean, float(data[r.TARGET].mean()), 'Target centering source')
    close(bundle['model'].target_scale, max(float(r.np.std(data[r.TARGET].to_numpy(float))), 1e-6), 'Target scaling source')
    values = r.raw(data, names, m)[:, expected_positions]
    imputer = r.v.SimpleImputer(strategy='median', keep_empty_features=True)
    filled = imputer.fit_transform(values)
    scaler = r.v.StandardScaler().fit(filled)
    r.require(r.np.array_equal(imputer.statistics_, bundle['imputer'].statistics_) and
              r.np.array_equal(scaler.mean_, bundle['scaler'].mean_) and
              r.np.array_equal(scaler.scale_, bundle['scaler'].scale_), 'Preprocessing not fit exclusively to recorded rows')
    r.require(metadata['seed'] == 17 and metadata['trained_rounds'] == metadata['selected_round'] == 53 and
              metadata['choice'] == dict(m['anchor_plan']['choice'], estimator=dict(m['anchor_plan']['choice']['estimator'], epochs=53)) and
              not metadata['official_holdout_used_for_fit'] and metadata['sample_weight_policy'] == 'uniform', 'Training policy/rounds')


def training_inventory(m):
    items = {}
    for kind in ('layer', 'rfe'):
        for key in r.keys(kind):
            state_file = r.state_path(kind, key)
            state = r.read(state_file)
            paths = [state_file, r.model_path(kind, key)]
            paths += [r.model_path(kind, key, f['fold']) for f in state['folds']]
            paths += [r.OUT / rec['fit_receipt'] for rec in [*state['folds'], state['full_fit']]]
            for path in paths:
                items[str(path.relative_to(r.OUT))] = r.sha(path)
    return r.fp(items)


def verify_training(m):
    r.require(r.manifest()['fingerprint'] == m['fingerprint'], 'Frozen sources changed before training QA')
    train = r.frame('train', m)
    fold_list = r.folds(train)
    date_groups = train.hour_bjt.dt.strftime('%Y-%m-%d').to_numpy()
    n_perm = 0
    for kind in ('layer', 'rfe'):
        names = m['retained_names'].copy()
        for key in r.keys(kind):
            state = r.read(r.state_path(kind, key))
            current = r.names_for_mask(key, m['retained_names']) if kind == 'layer' else names
            r.require(state['fingerprint'] == m['fingerprint'] and state['feature_names'] == current and
                      (state['kind'], state['key'], state['seed'], state['epochs']) == (kind, key, 17, 53) and
                      not state['official_holdouts_used_for_training_or_deletion'], 'Training state identity')
            r.require(state['n_features'] == len(current) and state['n_base'] + state['n_pressure'] == len(current) and
                      state['active_positions'] == list(r.active(current)) and len(state['folds']) == 3, 'Feature counts')
            if kind == 'layer':
                r.require(state['n_base'] == 50 and state['levels_hpa'] == [r.LEVELS[i] for i in range(10) if key & (1 << i)], 'Layer combination changed')
            cv = []
            for recorded, definition in zip(state['folds'], fold_list):
                fit, hold = train.iloc[definition['fit']], train.iloc[definition['hold']]
                r.require(recorded['fit_days'] == definition['fit_days'] and recorded['hold_days'] == definition['hold_days'] and
                          recorded['fit_index_sha256'] == definition['fit_index_sha256'] and recorded['hold_index_sha256'] == definition['hold_index_sha256'] and
                          set(date_groups[definition['fit']]).isdisjoint(set(date_groups[definition['hold']])), 'Fold source/date isolation')
                check_fit(recorded, fit, hold, current, m, kind, key)
                baseline = score_arrays(recorded['holdout']['truth_db'], recorded['holdout']['prediction_db'], r.np.ones(len(hold), bool))['rmse_db']
                cv.append(baseline)
                if kind == 'rfe' and key > 1:
                    bundle = r.joblib.load(r.model_path(kind, key, recorded['fold']))
                    values = r.v.transform(bundle, r.raw(hold, current, m))
                    for item in state['permutation_importance']:
                        column = bundle['positions'].index(m['original_names'].index(item['feature_name']))
                        token = f'{m["fingerprint"]}|{key}|{recorded["fold"]}|{item["feature_name"]}'
                        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big')
                        order = r.np.random.default_rng(seed).permutation(len(hold))
                        changed = values.copy()
                        changed[:, column] = values[order, column]
                        p = bundle['model'].predict(changed)
                        observed = score_arrays(hold[r.TARGET].to_numpy(float), p, r.np.ones(len(hold), bool))['rmse_db']
                        number = recorded['fold'] - 1
                        close(observed, item['fold_permuted_rmse_db'][number], 'Independent permuted RMSE')
                        close(observed - baseline, item['fold_delta_rmse_db'][number], 'Independent PFI delta')
                        n_perm += 1
            check_fit(state['full_fit'], train, None, current, m, kind, key)
            close(state['cv_mean_rmse_db'], r.np.mean(cv), 'CV mean')
            close(state['cv_se_rmse_db'], r.np.std(cv, ddof=1) / r.np.sqrt(3), 'CV heuristic SE')
            if len(current) == 156:
                parent = r.read(r.PARENT / 'training/full_bilstm_seed17.json')
                r.require(r.np.max(r.np.abs(r.np.asarray(state['full_fit']['training']['prediction_db']) - parent['train_prediction_db'])) < 1e-8, 'Full anchor prediction')
                r.require(state['full_fit']['model_weights_sha256'] == r.weights_digest(r.joblib.load(r.PARENT / 'models/full_bilstm_seed17.joblib')['model']), 'Full anchor weights')
            if kind == 'rfe':
                importance = state['permutation_importance']
                r.require([i['feature_name'] for i in importance] == (current if key > 1 else []), 'RFE importance column list')
                for i in importance:
                    r.require(len(i['fold_delta_rmse_db']) == len(i['fold_permuted_rmse_db']) == 3, 'PFI fold count')
                    close(i['mean_delta_rmse_db'], sum(i['fold_delta_rmse_db']) / 3, 'PFI mean')
                removed = None if key == 1 else min(importance, key=lambda i: (i['mean_delta_rmse_db'], i['feature_name']))['feature_name']
                r.require(state['removed_feature_next'] == removed, 'RFE deletion rule differs')
                if removed is not None:
                    names = [name for name in names if name != removed]
            if key % 25 == 0 or key in (1, 1023):
                print(json.dumps({'stage': 'independent-training-qa', 'kind': kind, 'key': key, 'permutations_replayed': n_perm}), flush=True)
        count = 1024 if kind == 'layer' else 156
        r.require(len(list((r.OUT / kind / 'states').glob('*.json'))) == count and
                  len(list((r.OUT / kind / 'models').glob('*.joblib'))) == count and
                  len(list((r.OUT / kind / 'fold_models').glob('*.joblib'))) == count * 3, 'Training file inventory differs')
    r.require(n_perm == 36735, 'Total independent PFI replay count')
    q = {'status': 'PASS', 'fingerprint': m['fingerprint'], 'layer_states': 1024, 'rfe_states': 156,
         'formal_models': 1180, 'fold_models': 3540, 'committed_fits': 4720, 'actual_fit_events': len(list((r.OUT / 'fit_events').glob('*.json'))),
         'rfe_single_deletions': 155, 'permutations_independently_replayed': n_perm,
         'official_holdout_samples_used_for_training_or_selection': 0, 'qa_input_sha256': training_inventory(m), 'utc': r.now()}
    path = r.OUT / 'train_qa.json'
    if path.exists():
        old = r.read(path)
        r.require(old['fingerprint'] == m['fingerprint'] and old['qa_input_sha256'] == q['qa_input_sha256'], 'Previously verified training changed')
        return old
    r.atomic(path, q)
    return q


def check_fit(row, fit, hold, names, m, kind, key):
    fold = row['fold']
    path = r.model_path(kind, key, fold)
    r.require(r.sha(path) == row['model_sha256'], 'Model file SHA changed')
    receipt = r.read(r.OUT / row['fit_receipt'])
    r.require(receipt['fingerprint'] == m['fingerprint'] and receipt['model_sha256'] == row['model_sha256'] and
              (receipt['kind'], receipt['key'], receipt['fold'], receipt['seed'], receipt['epochs'], receipt['n_fit']) == (kind, key, fold, 17, 53, len(fit)), 'New fit receipt')
    bundle = r.joblib.load(path)
    check_model(bundle, fit, names, m, kind, key, fold, row['fit_metadata'])
    r.require(r.weights_digest(bundle['model']) == row['model_weights_sha256'], 'Model weights fingerprint')
    check_prediction(row['training'], fit, bundle, names, m)
    if hold is not None:
        check_prediction(row['holdout'], hold, bundle, names, m)
    else:
        r.require(row['holdout'] is None, 'Official holdout in formal training record')
    history = row['round_history']
    r.require(len(history) == 53 and [h['round'] for h in history] == list(range(1, 54)), 'Training history count')
    for h in history:
        r.require(h['validation_objective_db'] is None and h['validation_rmse_db'] is None and
                  r.np.isfinite(h['train_rmse_db']) and (h['internal_holdout_metrics'] is None) == (hold is None), 'Round/holdout logging policy')
    for group in r.WEATHER:
        for key_name in ('rmse_db', 'mae_db', 'bias_db'):
            close(history[-1]['train_weather_metrics'][group][key_name], row['training']['metrics'][group][key_name], 'Last epoch training metric')
            if hold is not None:
                close(history[-1]['internal_holdout_metrics'][group][key_name], row['holdout']['metrics'][group][key_name], 'Last epoch fold metric')


def verify_evaluation(m):
    r.require(r.manifest()['fingerprint'] == m['fingerprint'], 'Frozen sources changed before final QA')
    train_qa, selection = r.read(r.OUT / 'train_qa.json'), r.read(r.OUT / 'selection.json')
    r.require(train_qa['status'] == 'PASS' and train_qa['fingerprint'] == selection['fingerprint'] == m['fingerprint'] and
              train_qa['qa_input_sha256'] == training_inventory(m), 'Training changed during evaluation')
    check_selection(m)
    frames = {split: r.frame(split, m) for split in ('train', 'validation', 'test')}
    sets = {s: set(f.hour_bjt) for s, f in frames.items()}
    r.require(sets['train'].isdisjoint(sets['validation'] | sets['test']) and sets['validation'].isdisjoint(sets['test']), 'Exact-hour leakage')
    rows = 0
    for kind in ('layer', 'rfe'):
        for key in r.keys(kind):
            result = r.read(r.evaluation_path(kind, key))
            state = r.read(r.state_path(kind, key))
            r.require(result['fingerprint'] == m['fingerprint'] and result['feature_names'] == state['feature_names'] and
                      result['model_sha256'] == state['full_fit']['model_sha256'] and
                      result['fit_calls_during_evaluation'] == 0 and result['selection_sha256'] == r.sha(r.OUT / 'selection.json') and
                      result['scored_utc'] >= selection['frozen_utc'] and
                      (result['kind'], result['key'], result['seed'], result['epochs']) == (kind, key, 17, 53), 'Evaluation identity/order')
            bundle = r.joblib.load(r.model_path(kind, key))
            for split in ('validation', 'test'):
                prediction = check_prediction(result['results'][split], frames[split], bundle, state['feature_names'], m)
                if len(state['feature_names']) == 156:
                    anchor = r.read(r.PARENT / ('training/full_bilstm_seed17.json' if split == 'validation' else 'test/full_bilstm_seed17.json'))
                    old = anchor['validation_prediction_db'] if split == 'validation' else anchor['prediction_db']
                    r.require(r.np.max(r.np.abs(prediction - old)) < 1e-8, 'Current53 official prediction anchor')
            rows += 1
        r.require(len(list((r.OUT / kind / 'evaluation').glob('*.json'))) == len(r.keys(kind)), 'Evaluation file count')
    q = {'status': 'PASS', 'fingerprint': m['fingerprint'], 'models_replayed': rows,
         'weather_metric_values_recomputed': rows * 2 * 5 * 3, 'validation_predictions': rows * 343, 'test_predictions': rows * 687,
         'test_training_fits': 0, 'test_selection_reads': 0, 'historical_test_access': True, 'utc': r.now()}
    r.atomic(r.OUT / 'evaluation_qa.json', q)
    return q


def check_selection(m):
    selection = r.read(r.OUT / 'selection.json')
    for kind in ('layer', 'rfe'):
        rows = [r.read(r.state_path(kind, key)) for key in r.keys(kind)]
        best = sorted(rows, key=lambda row: (row['cv_mean_rmse_db'], row['key']))[0]
        threshold = best['cv_mean_rmse_db'] + best['cv_se_rmse_db']
        eligible = [row for row in rows if row['cv_mean_rmse_db'] <= threshold]
        picked = sorted(eligible, key=lambda row: (len(row['levels_hpa']) if kind == 'layer' else row['n_features'],
                                                 row['cv_mean_rmse_db'], row['key']))[0]
        saved = selection['selections'][kind]
        r.require(saved['selected_key'] == picked['key'] and saved['minimum_cv_key'] == best['key'] and
                  saved['feature_names'] == picked['feature_names'] and saved['levels_hpa'] == picked['levels_hpa'] and
                  not saved['official_holdouts_used_for_selection'], 'Independent training-only selection')
        close(threshold, saved['one_se_threshold_db'], 'Independent one-SE threshold')


def bootstrap_verify(a, b, group, name, m):
    y = r.np.asarray(a['truth_db'])
    keep = r.np.ones(len(y), bool) if group == 'overall' else r.np.asarray(a['weather_masks'][group], bool)
    dates = r.np.asarray([h[:10] for h in a['hour_bjt']])[keep]
    unique, ids = r.np.unique(dates, return_inverse=True)
    n = r.np.bincount(ids).astype(float)
    ea = r.np.bincount(ids, weights=(r.np.asarray(a['prediction_db'])[keep] - y[keep]) ** 2)
    eb = r.np.bincount(ids, weights=(r.np.asarray(b['prediction_db'])[keep] - y[keep]) ** 2)
    seed = int.from_bytes(hashlib.sha256(f'{m["bootstrap_seed"]}|{name}|{group}'.encode()).digest()[:8], 'big')
    draw = r.np.random.default_rng(seed).integers(0, len(unique), size=(m['bootstrap_replicates'], len(unique)))
    score = r.np.sqrt(ea[draw].sum(1) / n[draw].sum(1)) - r.np.sqrt(eb[draw].sum(1) / n[draw].sum(1))
    return r.np.quantile(score, [.025, .975]).tolist()


def verify_delivery(m, table_sources, comparisons):
    checks = 0
    for filename, source in table_sources.items():
        with (r.OUT / filename).open(encoding='utf-8-sig', newline='') as f:
            actual = csv.DictReader(f)
            n = 0
            for expected in source:
                got = next(actual, None)
                r.require(got is not None and set(got) == set(expected), f'CSV structure {filename}')
                for field, value in expected.items():
                    if isinstance(value, float):
                        close(float(got[field]), value, f'CSV numeric {filename}/{field}')
                    else:
                        r.require(got[field] == ('' if value is None else str(value)), f'CSV value {filename}/{field}')
                n += 1
            r.require(next(actual, None) is None, f'Extra CSV rows {filename}')
            checks += n
    intervals = r.read(r.OUT / 'date_intervals.json')
    r.require(len(intervals['comparisons']) == 65, 'Date interval comparison count')
    for name, a, b in comparisons:
        for group in r.WEATHER:
            observed = bootstrap_verify(a, b, group, name, m)
            saved = intervals['comparisons'][f'{name}/{group}']
            r.require(r.np.max(r.np.abs(r.np.asarray(observed) - saved['percentile95_db'])) < 1e-10, 'Independent date interval recomputation')
            close(a['metrics'][group]['rmse_db'] - b['metrics'][group]['rmse_db'], saved['rmse_improvement_db'], 'Date effect sign')
    q = {'status': 'PASS', 'fingerprint': m['fingerprint'], 'csv_rows_verified': checks,
         'date_intervals_independently_recomputed': 65, 'all_sources_and_predictions_verified': True, 'utc': r.now()}
    r.atomic(r.OUT / 'report_qa.json', q)
    return q
