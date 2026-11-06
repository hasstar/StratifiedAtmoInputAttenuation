"""Freeze the new experiment; no research model fitting or timing pilot."""
from __future__ import annotations

import ast
import csv
import importlib.metadata
import platform
import sys
import time
import traceback
import runtime as r


def reference_files():
    paths = {r.P32 / 'manifest.json', r.TASK_PATH, r.OUT / 'HANDOFF.md'}
    old = r.read(r.P32 / 'manifest.json')
    for name, expected in old['source_sha256'].items():
        normal = name.replace('\\', '/')
        if normal.startswith(('Code/', 'Outputs/Datasets/')):
            path = r.ROOT / name
            r.require(r.sha(path) == expected, 'Prior frozen source/input digest ' + name)
            paths.add(path)
    paths.update(r.HERE.glob('*.py'))
    paths.update(r.HERE.glob('*.md'))
    paths.update(r.P28 / x for x in ('selection.json', 'run_manifest.json'))
    paths.add(r.P29 / 'run_manifest.json')
    paths.update(r.P31 / x for x in ('source_manifest.json', 'search_selection.json', 'search_qa.json', 'full156_qa.json',
                                    'layer/states/0000.json', 'layer/qa/0000.json'))
    source31 = r.read(r.P31 / 'source_manifest.json')
    for name, expected in source31['source_sha256'].items():
        if name.replace('\\', '/').startswith('Code/'):
            r.require(r.sha(r.ROOT / name) == expected, 'Rev031 source digest ' + name)
            paths.add(r.ROOT / name)
    qa0 = r.read(r.P31 / 'layer/qa/0000.json')
    r.require(qa0['status'] == 'PASS', 'Existing surface starting-state QA')
    for name, expected in qa0['verified_files'].items():
        path = r.P31 / name
        r.require(r.sha(path) == expected, 'Existing start-state provenance ' + name)
        paths.add(path)
    paths.add(r.ROOT / 'Outputs/Experiments/Combined_Channel_Revision_027/excluded_hours.json')
    qa34 = r.DELIVERY34 / 'evidence/FINAL_QA.json'
    table = r.DELIVERY34 / 'tables/all_states_1560.csv'
    r.require(r.read(qa34)['status'] == 'PASS', 'Official full-pool delivery QA')
    paths.update((qa34, table))
    with table.open(encoding='utf-8-sig', newline='') as stream:
        records = [row for row in csv.DictReader(stream) if row['model'] == 'tabm']
    r.require(sorted(int(row['k']) for row in records) == list(range(1, 157)), 'All156 reference coverage')
    for row in records:
        folder = (r.ROOT / row['state_path']).parent
        for filename, column in (('state.json', 'state_sha256'), ('evaluation.json', 'evaluation_sha256'), ('qa.json', 'qa_sha256')):
            path = folder / filename
            r.require(r.sha(path) == row[column], 'Reference delivery state digest ' + str(path))
            paths.add(path)
    old_digests = {name.replace('\\', '/'): value for name, value in old['source_sha256'].items()}
    for path in paths:
        expected = old_digests.get(r.rel(path).replace('\\', '/'))
        if expected is not None:
            r.require(r.sha(path) == expected, 'Historical manifest dependency changed: ' + str(path))
    return paths


def synthetic(auditor):
    for path in r.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    cases = [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1]), ([1, 1], [2, 2]),
             ([2], [1]), ([], []), ([1, 2, 3], [2, 2, 2])]
    for truth, pred in cases:
        for key, value in auditor.scalar_metrics(truth, pred).items():
            auditor.equal(value, r.metric(truth, pred)[key], 'Synthetic metric ' + key)
    r.require(r.metric([1, 2, 3], [3, 2, 1])['r2'] == -3, 'R2 must not be squared correlation')
    for names in (r.surface_names(), [r.surface_names()[0]]):
        net = r.make_nn('tabm', names)
        pred = net(r.torch.zeros(4, len(names)))
        r.require(tuple(pred.shape) == (4, 32), 'TabM synthetic output shape')
        pred.square().mean().backward()
        r.require(r.torch.isfinite(pred).all() and any(p.grad is not None for p in net.parameters()), 'Synthetic gradients')
    # Reporting scalar implementation is also tested independently of the producer.
    import verify_delivery
    for truth, pred in cases:
        for key, value in verify_delivery.metrics(truth, pred).items():
            auditor.equal(value, r.metric(truth, pred)[key], 'Delivery synthetic metric ' + key)
    return dict(status='PASS', metric_edge_cases=len(cases), synthetic_network_widths=[50, 1],
                research_model_fit_calls=0, timing_pilot=False)


def prepare():
    with r.writer_lock():
        r.require(not (r.OUT / 'manifest.json').exists(), 'Manifest already frozen: do not prepare again')
        resources = r.resource_check()
        runner, auditor = r.load_reference_functions()
        tests = synthetic(auditor)
        paths = reference_files()
        data = r.frames()
        names = r.surface_names()
        r.require(len(names) == 50 and len(r.key_names()) == 156, '50/156 candidate counts')
        r.require([name for name in r.key_names() if not name.startswith('surface__')]
                  and all(name.startswith('surface__') for name in names), 'Surface prefix scope')
        folds = r.r29.folds(data['train'])
        oldstate = r.read(r.P32 / 'models/tabm/k_0156/state.json')
        for f, role in zip(folds, oldstate['folds']):
            r.require(role['training']['hour_bjt'] == data['train'].iloc[f['fit']].hour_bjt.astype(str).tolist(), 'Same full-pool fitting dates/keys')
            r.require(role['holdout']['hour_bjt'] == data['train'].iloc[f['hold']].hour_bjt.astype(str).tolist(), 'Same full-pool holdout dates/keys')
        for group, count in zip(r.WEATHER, (687, 322, 51, 158, 156)):
            r.require((len(data['test']) if group == 'overall' else int(data['test'][group].sum())) == count, 'Test weather count ' + group)
        from run_surface import reuse_start
        reused = []
        for f in folds:
            ref = reuse_start('tabm', 50, f['fold'], names, data['train'].iloc[f['fit']], data['train'].iloc[f['hold']])
            reused.append(dict(fold=f['fold'], model_file=ref['model_file'], model_sha256=ref['model_sha256']))
        ref = reuse_start('tabm', 50, 0, names, data['train'], None)
        reused.append(dict(fold=0, model_file=ref['model_file'], model_sha256=ref['model_sha256']))
        hashes = {r.rel(path): r.sha(path) for path in sorted(paths)}
        payload = dict(version='T017-tabm-surface50-rfe-v1', models=['tabm'], feature_names=names,
                       reference_feature_names=r.key_names(), surface_definition='single-level/non-pressure base variables, including column integrals',
                       target=r.TARGET, source_sha256=hashes, model_configs={'tabm': r.config('tabm')}, model_seed=17,
                       folds=[{key: value for key, value in f.items() if key not in ('fit', 'hold')} for f in folds],
                       rfe_states_per_model=50, total_states=50, fit_roles=200, reused_start_roles=4, expected_new_fits=196,
                       permutation_evaluations=3822, permutation_namespace=r.PERM_NAMESPACE,
                       environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(), threads=2,
                                        packages={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'torch', 'scikit-learn', 'joblib', 'psutil', 'matplotlib', 'seaborn')}),
                       historical_test_access=True, no_weather_network=True, no_uk_refit=True,
                       reference_delivery=r.rel(r.DELIVERY34), representative_selection='minimum training CV mean RMSE; smaller k for exact ties')
        payload['fingerprint'] = r.fp(payload)
        payload['created_utc'] = r.now()
        r.emit(r.OUT / 'synthetic_qa.json', dict(tests, utc=r.now()))
        r.emit(r.OUT / 'resource_check_before_start.json', resources)
        r.emit(r.OUT / 'reuse_start_qa.json', dict(status='PASS', roles=reused, exact_feature_config_seed_epoch_hour_and_preprocessor_match=True,
               saved_predictions_replayed=True, research_model_fit_calls=0, utc=r.now()))
        r.emit(r.OUT / 'manifest.json', payload)
        r.emit(r.OUT / 'preflight_qa.json', dict(status='PASS', manifest=payload['fingerprint'], source_count=len(hashes),
               train_n=2404, validation_n=343, test_n=687, surface_features=50, folds=3, reused_roles=4,
               research_model_fit_calls=0, timing_pilot=False, utc=r.now()))
        r.emit(r.OUT / 'status.json', dict(status='PREPARED_NOT_STARTED', utc=r.now()))
        print('PREPARED', payload['fingerprint'], 'sources', len(hashes), 'reused starting roles', len(reused), flush=True)


if __name__ == '__main__':
    try:
        prepare()
    except BaseException as error:
        r.emit(r.OUT / 'failures' / (str(time.time_ns()) + '_prepare.json'),
               dict(status='FAILED_REQUIRES_INSPECTION', error=repr(error), traceback=traceback.format_exc(), utc=r.now()))
        raise
