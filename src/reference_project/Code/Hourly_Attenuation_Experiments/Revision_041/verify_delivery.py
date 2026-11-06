"""Independent standard-library CSV readback, lineage and metric audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def metrics(y, p):
    y, p = list(map(float, y)), list(map(float, p))
    require(len(y) == len(p) and all(math.isfinite(x) for x in y + p), 'Finite aligned metric inputs')
    n = len(y)
    if not n:
        return dict(n=0, rmse_db=None, mae_db=None, bias_db=None, r2=None, pearson_r=None, pearson_r_squared=None)
    delta = [a-b for a, b in zip(p, y)]
    sse = math.fsum(x*x for x in delta)
    ym, pm = math.fsum(y)/n, math.fsum(p)/n
    sy, sp = math.fsum((x-ym)**2 for x in y), math.fsum((x-pm)**2 for x in p)
    corr = math.fsum((a-ym)*(b-pm) for a, b in zip(y, p))/math.sqrt(sy*sp) if n > 1 and sy > 0 and sp > 0 else None
    return dict(n=n, rmse_db=math.sqrt(sse/n), mae_db=math.fsum(abs(x) for x in delta)/n,
                bias_db=math.fsum(delta)/n, r2=1-sse/sy if n > 1 and sy > 0 else None,
                pearson_r=corr, pearson_r_squared=None if corr is None else corr*corr)


def close(value, stored, message):
    if value is None:
        require(stored in ('', None), message + ': undefined must remain blank')
        return 0.0
    require(stored not in ('', None), message + ': missing value')
    error = abs(float(stored)-value)
    require(math.isclose(value, float(stored), rel_tol=1e-10, abs_tol=1e-9), message)
    return error


def verify(directory, root):
    directory, root = Path(directory), Path(root)
    index = read(directory / 'source_index.json')
    require(len(index['states']) == 206, '206 reference plus new states')
    for name, expected in index['files_sha256'].items():
        require(digest(directory / name) == expected, 'Delivery file changed: ' + name)
    state_lookup = {}
    for entry in index['states']:
        key = entry['pool'], entry['k']
        require(key not in state_lookup, 'Duplicate state')
        state_lookup[key] = entry
        for part in ('state', 'evaluation', 'qa'):
            require(digest(root / entry[part + '_path']) == entry[part + '_sha256'], 'Source ' + part + ' SHA')
        q = read(root / entry['qa_path'])
        require(q['status'] == 'PASS' and q['state_sha256'] == entry['state_sha256']
                and q['evaluation_sha256'] == entry['evaluation_sha256'], 'Source state QA identity')
    require({k for pool, k in state_lookup if pool == 'surface50'} == set(range(1, 51)), 'Surface50 coverage')
    require({k for pool, k in state_lookup if pool == 'all156'} == set(range(1, 157)), 'All156 coverage')

    grouped = defaultdict(list)
    hours = rows(directory / 'hourly_predictions.csv')
    require(len(hours) == 206 * 1030, 'Hourly row coverage')
    for row in hours:
        grouped[(row['pool'], int(row['k']), row['split'])].append(row)
    del hours
    for (pool, k), entry in state_lookup.items():
        evaluation = read(root / entry['evaluation_path'])['results']
        for split, count in (('validation', 343), ('test', 687)):
            block = grouped[(pool, k, split)]
            source = evaluation[split]
            require(len(block) == count and len({x['hour_bjt'] for x in block}) == count, 'Hour uniqueness/count')
            require([x['hour_bjt'] for x in block] == source['hour_bjt'], 'Exported key/order')
            for i, row in enumerate(block):
                require(float(row['truth_db']) == source['truth_db'][i] and float(row['prediction_db']) == source['prediction_db'][i], 'Prediction/truth exact round trip')
                weather = [w for w, mask in source['weather_masks'].items() if mask[i]]
                require(weather == [row['weather']], 'Exported weather identity')

    metric_rows = rows(directory / 'all_states_metrics.csv')
    require(len(metric_rows) == 206 * 2 * 5, 'Metric coverage')
    metric_lookup = {}
    max_error = 0.0
    for row in metric_rows:
        key = row['pool'], int(row['k']), row['split'], row['weather']
        require(key not in metric_lookup, 'Duplicate metric')
        metric_lookup[key] = row
        block = grouped[key[:3]]
        selected = [v for v in block if key[3] == 'overall' or v['weather'] == key[3]]
        y, p = [float(v['truth_db']) for v in selected], [float(v['prediction_db']) for v in selected]
        for name, value in metrics(y, p).items():
            max_error = max(max_error, close(value, row[name], 'CSV scalar ' + name))
        require(int(row['n_dates']) == len({v['hour_bjt'][:10] for v in selected}), 'Date coverage')
        for values, field in ((y, 'truth_std_db'), (p, 'prediction_std_db')):
            mean = math.fsum(values)/len(values)
            max_error = max(max_error, close(math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values)), row[field], 'Population std'))
        require(int(row['prediction_unique_values']) == len(set(p)), 'Prediction variability count')
        require(row['state_sha256'] == state_lookup[key[:2]]['state_sha256'], 'Metric source identity')

    comparison = rows(directory / 'same_k_comparison.csv')
    require(len(comparison) == 50 * 2 * 5, 'Same-k comparison coverage')
    seen = set()
    for row in comparison:
        key = int(row['k']), row['split'], row['weather']
        require(key not in seen, 'Duplicate comparison row')
        seen.add(key)
        left, right = metric_lookup[('surface50', *key)], metric_lookup[('all156', *key)]
        require(int(row['n']) == int(left['n']) == int(right['n']), 'Matched comparison population')
        for metric in ('rmse_db', 'r2', 'mae_db', 'bias_db'):
            a = None if left[metric] == '' else float(left[metric])
            b = None if right[metric] == '' else float(right[metric])
            close(a, row['surface_' + metric], 'Surface comparison')
            close(b, row['all_' + metric], 'All comparison')
            close(None if a is None or b is None else a-b, row['surface_minus_all_' + metric], 'Paired metric difference')

    paths = rows(directory / 'rfe_paths.csv')
    retained = rows(directory / 'retained_features.csv')
    require(len(paths) == 206 and len(retained) == 12246 + 1275, 'Feature-path row counts')
    retained_lookup = defaultdict(list)
    for row in retained:
        retained_lookup[(row['pool'], int(row['k']))].append(row['feature_name'])
    for pool, count in (('all156', 156), ('surface50', 50)):
        ordered = sorted([p for p in paths if p['pool'] == pool], key=lambda p: -int(p['k']))
        require([int(p['k']) for p in ordered] == list(range(count, 0, -1)), 'Full ordered path')
        removed = []
        for i, row in enumerate(ordered):
            k = int(row['k'])
            names = json.loads(row['retained_names_json'])
            require(names == retained_lookup[(pool, k)] and len(names) == k, 'Expanded retained names')
            require(json.loads(row['removed_so_far_json']) == removed, 'Removed-so-far sequence')
            source = read(root / state_lookup[(pool, k)]['state_path'])
            require(names == source['feature_names'] and row['removed_next'] == (source['removed_feature_next'] or ''), 'Path source identity')
            if pool == 'surface50':
                require(all(n.startswith('surface__') for n in names), 'No pressure feature in surface path')
            if k > 1:
                removed.append(row['removed_next'])
                require(json.loads(ordered[i+1]['retained_names_json']) == [n for n in names if n != removed[-1]], 'One feature removed per transition')
            else:
                require(row['removed_next'] == '', 'Terminal state has no next deletion')
    roles = rows(directory / 'surface_model_roles.csv')
    require(len(roles) == 200 and sum(int(x['reused']) for x in roles) == 4, '200 roles, 4 reuses')
    for role in roles:
        require(digest(root / role['model_file']) == role['model_sha256'], 'Role weight digest')
    importance = rows(directory / 'surface_permutation_importance.csv')
    require(len(importance) == 1274, '1274 feature importances / 3822 fold scores')
    for row in importance:
        vals = [float(row['fold' + str(i) + '_delta_rmse_db']) for i in (1, 2, 3)]
        close(math.fsum(vals)/3, row['mean_delta_rmse_db'], 'Permutation mean')
    return dict(status='PASS', implementation='stdlib CSV readback + independent math.fsum, no producer/runtime import',
                states=206, new_surface_states=50, metric_groups=len(metric_rows), same_k_groups=len(comparison),
                hourly_rows=206*1030, surface_role_slots=200, reused_roles=4, new_fits=196,
                maximum_metric_readback_difference=max_error, actual_visual_review=False,
                training_calls_during_delivery=0)
