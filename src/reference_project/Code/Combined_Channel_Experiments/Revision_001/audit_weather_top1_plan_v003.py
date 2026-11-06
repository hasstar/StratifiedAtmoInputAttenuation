"""Separate read-only reconstruction of Top1 date/field/ID migration coverage."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top1_v003'
EXT = DEST.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit():
    p, old, prior = read(DEST / 'plan.json'), read(EXT / 'weather_plan.json'), read(EXT / 'weather_v002/plan.json')
    expected = {n['id']: n for n in old['jobs'] if 'pressure_level' in n['request']}
    check({n['id']: n for n in p['parents']} == expected, 'Source pressure months changed')
    check(len(expected) == 33 and sum(n['expected_hours'] for n in expected.values()) == 22824, 'Date scope differs')
    coverage = {pid: Counter() for pid in expected}
    inherited = {a['id']: a for a in p['adopted']}
    for item in p['reused']:
        node, parent = item['job'], expected[item['parent_id']]
        check(digest(ROOT / item['target']) == item['sha256'], 'Reused payload SHA differs')
        req = node['request']
        check('geopotential' in req['variable'] and '1' in req['pressure_level'], 'Reused file lacks Top1')
        check(all(req[k] == v for k, v in parent['request'].items() if k != 'day'), 'Reused geographic/time scope differs')
        coverage[parent['id']].update((d, t) for d in req['day'] for t in req['time'])
    for node in p['jobs']:
        req, parent = node['request'], expected[node['parent_id']]
        if node['id'] in inherited:
            item = inherited[node['id']]
            before = read(ROOT / item['source_state'])
            after = read(DEST / 'jobs' / (node['id'] + '.json'))
            check(node == before['job'] == after['job'], 'Migrated request not exactly the same')
            check(item['request_id'] == before['request_id'] == after['request_id'], 'Migrated remote ID changed')
            check(digest(ROOT / item['source_state']) == item['source_state_sha256'], 'Old state was modified')
        else:
            check(req['variable'] == ['geopotential'] and req['pressure_level'] == ['1'], 'New POST scope exceeds Top1')
            check(all(req[k] == v for k, v in parent['request'].items() if k not in ('day', 'variable', 'pressure_level')),
                  'New request changed source hour/area/month/options')
        check(node['expected_hours'] == len(req['day']) * len(req['time']), 'Hours inconsistent')
        coverage[parent['id']].update((d, t) for d in req['day'] for t in req['time'])
    for pid, parent in expected.items():
        check(coverage[pid] == Counter((d, t) for d in parent['request']['day'] for t in parent['request']['time']),
              'Coverage missing/duplicated: ' + pid)
    for name, sha in {**prior['protected_files'], **p['protected_files']}.items():
        check(digest(ROOT / name) == sha, 'Protected source changed: ' + name)
    ids = [a['request_id'] for a in p['adopted']]
    check(len(ids) == len(set(ids)) == 2, 'Inherited remote slots differ')
    model = read(DEST / 'model_inventory.json')
    check(model['feature_names'] == ['p1__geopotential'] and model['new_fit_calls'] == 0, 'Model gate differs')
    check(digest(ROOT / model['checkpoint']) == model['checkpoint_sha256'], 'Model checkpoint changed')
    report = {'status': 'PASS', 'scope': 'Independent Top1 plan/date/field/provenance audit, not full payload field QA',
        'plan_sha256': digest(DEST / 'plan.json'), 'plan_fingerprint': p['fingerprint'],
        'auditor_sha256': digest(Path(__file__)), 'source_months': len(expected), 'request_hours': 22824,
        'reused_pressure_payloads': len(p['reused']), 'same_remote_ids_adopted': ids,
        'new_single_field_requests': len(p['jobs']) - len(ids),
        'old_and_new_protected_files_verified': len({**prior['protected_files'], **p['protected_files']}),
        'network_requests': 0, 'training_calls': 0, 'old_top10_completion_claimed': False}
    target = DEST / 'plan_independent_qa.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit()
