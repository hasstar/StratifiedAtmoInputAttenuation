"""User-authorized RFE survivor first. Single field; frozen model, zero training.

Reuse v002 transport/recovery primitives in this process with an isolated DEST.
The v002 source, plan, states, logs and payloads are never modified or rerun.
Already-submitted v002 IDs occupy the same global two remote slots until done.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
import weather_download_v002 as transport

HERE, ROOT, EXTERNAL = transport.HERE, transport.ROOT, transport.EXTERNAL
OLD = EXTERNAL / 'weather_v002'
DEST = EXTERNAL / 'weather_top1_v003'
PLAN = DEST / 'plan.json'
FEATURE = 'p1__geopotential'
CODE = Path(__file__)
TEST = HERE / 'test_weather_download_top1_v003.py'
READ, SHA, FP, REQUIRE = transport.read, transport.sha, transport.fingerprint, transport.require
# This changes only imported runtime globals in the new process, not any file.
transport.DEST = DEST


def reduced_request(parent, days):
    request = copy.deepcopy(parent['request'])
    request.update(variable=['geopotential'], pressure_level=['1'], day=list(days))
    return request


def coverage_contract(plan):
    parents = {p['id']: p for p in plan['parents']}
    coverage = {pid: Counter() for pid in parents}
    adopted = {a['id']: a for a in plan['adopted']}
    REQUIRE(len(adopted) == len(plan['adopted']) <= 2, 'Duplicate or excessive inherited IDs')
    REQUIRE(len({a['request_id'] for a in plan['adopted']}) == len(adopted), 'Remote ID reused twice')
    REQUIRE(len({n['id'] for n in plan['jobs']}) == len(plan['jobs']), 'Duplicate new job ID')
    for item in plan['reused']:
        node = item['job']
        parent = parents[node['parent_id']]
        REQUIRE(node['collection_id'] == parent['collection_id'], 'Reused collection changed')
        REQUIRE({k: v for k, v in node['request'].items() if k != 'day'} ==
                {k: v for k, v in parent['request'].items() if k != 'day'}, 'Reused fields differ')
        REQUIRE(node['expected_hours'] == len(node['request']['day']) * len(parent['request']['time']),
                'Reused hour count differs')
        coverage[parent['id']].update(node['request']['day'])
    for node in plan['jobs']:
        parent = parents[node['parent_id']]
        REQUIRE(node['collection_id'] == parent['collection_id'], 'Collection changed')
        REQUIRE(node['request']['day'], 'Empty dates')
        if node['id'] in adopted:
            REQUIRE(node == adopted[node['id']]['original_job'], 'Inherited exact request changed')
            REQUIRE({k: v for k, v in node['request'].items() if k != 'day'} ==
                    {k: v for k, v in parent['request'].items() if k != 'day'}, 'Inherited fields changed')
        else:
            REQUIRE(node['request'] == reduced_request(parent, node['request']['day']),
                    'New request not single geopotential level 1 with original scope')
        REQUIRE(node['expected_hours'] == len(node['request']['day']) * len(parent['request']['time']),
                'Wrong request hour count')
        coverage[parent['id']].update(node['request']['day'])
    for pid, parent in parents.items():
        REQUIRE(coverage[pid] == Counter(parent['request']['day']), 'Missing/duplicated dates: ' + pid)
    return {'parent_months': len(parents), 'expected_request_hours': sum(p['expected_hours'] for p in parents.values()),
            'new_single_field_requests': len(plan['jobs']) - len(adopted), 'inherited_remote_ids': len(adopted),
            'reused_pressure_payloads': len(plan['reused']), 'date_hour_area_exact': True,
            'new_request_variables': ['geopotential'], 'new_request_pressure_levels': ['1'],
            'new_surface_or_diagnostic_requests': 0, 'old_top10_completion_claimed': False}


def build_plan():
    REQUIRE(not transport.lock_held(), 'An old weather writer still holds the OS lock')
    old_plan = transport.load_plan()  # Read-only original v002 SHA checks.
    REQUIRE(old_plan['fingerprint'] == '248cd58b1dc1fc7ae0baadcb1da39adb827daeb81b220335a5b9f363113158a0',
            'Unexpected v002 input plan')
    REQUIRE(not list((OLD / 'payloads').glob('*.part')), 'Old partial requires explicit diagnosis')
    original = READ(EXTERNAL / 'weather_plan.json')
    parents = [p for p in original['jobs'] if 'pressure_level' in p['request']]
    parent_map = {p['id']: p for p in parents}
    used = {pid: set() for pid in parent_map}
    reused, adopted, nodes = [], [], []
    candidates = []
    for state_path in sorted((EXTERNAL / 'weather_jobs').glob('*.json')):
        saved = READ(state_path)
        if state_path.stem.startswith('SMOKE_'):
            REQUIRE(saved['status'] == 'VERIFIED', 'Unresolved smoke request')
            continue  # Preflight payloads are never counted as formal coverage.
        original_job = saved.get('job')
        if not original_job:
            original_job = next(p for p in original['jobs'] if state_path.stem == p['id'])
        if 'pressure_level' not in original_job['request']:
            REQUIRE(saved['status'] == 'VERIFIED', 'Unresolved old surface request')
            continue
        REQUIRE(saved['request_fingerprint'] == FP(original_job), 'Original saved request changed')
        candidates.append((state_path, {**original_job, 'parent_id': original_job['id']}, saved,
                           EXTERNAL / original_job['target']))
    for state_path in sorted((OLD / 'jobs').glob('*.json')):
        saved = READ(state_path)
        REQUIRE(saved.get('status') in ('VERIFIED', 'WAITING_REMOTE', 'SUBMITTED', 'RETRY_SAME_ID'),
                'Unresolved v002 state; diagnose before migration')
        node = saved['job']
        REQUIRE(saved['request_fingerprint'] == FP(node), 'v002 saved request changed')
        REQUIRE(node['parent_id'] in parent_map, 'Unresolved non-pressure request needs separate handling')
        candidates.append((state_path, node, saved, OLD / node['target']))
    for state_path, node, saved, payload in candidates:
        pid = node['parent_id']
        days = set(node['request']['day'])
        REQUIRE(not (used[pid] & days), 'Duplicate inherited/reused dates')
        REQUIRE(saved.get('request_id'), 'Saved remote ID missing')
        used[pid].update(days)
        if saved['status'] == 'VERIFIED':
            REQUIRE(SHA(payload) == saved['sha256'], 'Existing payload SHA differs')
            reused.append({'parent_id': pid, 'job': node, 'source_state': str(state_path.relative_to(ROOT)),
                'target': str(payload.relative_to(ROOT)), 'sha256': saved['sha256'],
                'bytes': payload.stat().st_size, 'request_id': saved['request_id']})
        else:
            adopted.append({'id': node['id'], 'original_job': copy.deepcopy(node),
                'request_id': saved['request_id'], 'source_state': str(state_path.relative_to(ROOT)),
                'source_state_sha256': SHA(state_path)})
            nodes.append(copy.deepcopy(node))
    for parent in parents:
        days = [day for day in parent['request']['day'] if day not in used[parent['id']]]
        if not days:
            continue
        identity = parent['id'] + '__top1'
        nodes.append({'id': identity, 'parent_id': parent['id'], 'collection_id': parent['collection_id'],
            'request': reduced_request(parent, days), 'expected_hours': len(days) * len(parent['request']['time']),
            'target': 'payloads/' + identity + '.download'})
    inventory = READ(DEST / 'model_inventory.json')
    REQUIRE(inventory['feature_names'] == [FEATURE] and inventory['new_fit_calls'] == 0, 'Model inventory gate')
    protected = {str(p.relative_to(ROOT)): SHA(p) for p in EXTERNAL.rglob('*')
                 if p.is_file() and not p.is_relative_to(DEST) and p != transport.LOCK}
    protected[str((HERE / 'weather_download_v002.py').relative_to(ROOT))] = SHA(HERE / 'weather_download_v002.py')
    protected.update(inventory['source_sha256'])
    for p in (DEST / 'model_inventory.json', DEST / 'handoff.json'):
        protected[str(p.relative_to(ROOT))] = SHA(p)
    plan = {'version': 'top1-v003', 'authorization': '2026-10-14 latest user: RFE final single feature first',
        'feature_names': [FEATURE], 'code_sha256': SHA(CODE), 'test_sha256': SHA(TEST),
        'transport_sha256': SHA(HERE / 'weather_download_v002.py'),
        'original_plan_sha256': SHA(EXTERNAL / 'weather_plan.json'),
        'previous_plan_sha256': SHA(OLD / 'plan.json'), 'parents': parents,
        'reused': reused, 'adopted': adopted, 'jobs': nodes, 'protected_files': protected,
        'max_in_flight_requests': 2, 'initial_batching': 'remaining dates within each original month',
        'fallback': 'Explicit oversize rejection of NEW single-field request permits date-only bisection',
        'model_training_started': False, 'scope': 'Top1 input acquisition; not full Top10 or full external evaluation'}
    coverage_contract(plan)
    plan['fingerprint'] = FP(plan)
    return plan


def load_plan():
    plan = READ(PLAN)
    REQUIRE(plan['fingerprint'] == FP({k: v for k, v in plan.items() if k != 'fingerprint'}), 'Plan digest changed')
    REQUIRE(plan['code_sha256'] == SHA(CODE) and plan['test_sha256'] == SHA(TEST), 'Frozen Top1 source changed')
    REQUIRE(plan['transport_sha256'] == SHA(HERE / 'weather_download_v002.py'), 'Frozen transport changed')
    for name, expected in plan['protected_files'].items():
        REQUIRE(SHA(ROOT / name) == expected, 'Protected source changed: ' + name)
    coverage_contract(plan)
    return plan


def prepare(plan):
    transport.immutable(PLAN, plan)
    for item in plan['adopted']:
        node = item['original_job']
        saved = READ(ROOT / item['source_state'])
        REQUIRE(saved['request_id'] == item['request_id'], 'Inherited remote ID differs')
        transport.immutable(transport.state_path(node), {**saved, 'plan_fingerprint': plan['fingerprint'],
            'migrated_from': item['source_state'], 'migrated_from_sha256': item['source_state_sha256']})


def runtime_nodes(plan):
    # Never submit full-field children of a legacy request if its remote result fails.
    for item in plan['adopted']:
        state = transport.state(item['original_job'])
        REQUIRE(state.get('request_id') == item['request_id'], 'Inherited ID lost')
        REQUIRE(state.get('status') != 'SPLIT_SIZE_REJECTION', 'Inherited request rejected: diagnose before a Top1 replacement')
    return transport.leaves(plan)


def progress(plan, verify=False):
    parents = {p['id']: p for p in plan['parents']}
    covered = {pid: Counter() for pid in parents}
    for item in plan['reused']:
        if verify:
            REQUIRE(SHA(ROOT / item['target']) == item['sha256'], 'Reused pressure payload changed')
        covered[item['parent_id']].update(item['job']['request']['day'])
    pairs = [(n, transport.state(n)) for n in runtime_nodes(plan)]
    adopted_ids = {a['id'] for a in plan['adopted']}
    for node, saved in pairs:
        if saved.get('status') == 'VERIFIED':
            if verify:
                REQUIRE(SHA(DEST / node['target']) == saved['sha256'], 'New payload changed')
            covered[node['parent_id']].update(node['request']['day'])
    for pid, dates in covered.items():
        REQUIRE(not (dates - Counter(parents[pid]['request']['day'])), 'Duplicate/excess completed date')
    pending = [{'id': n['id'], 'request_id': s.get('request_id'), 'status': s.get('status'),
        'remote_status': s.get('remote_status'), 'last_checked_utc': s.get('last_checked_utc'),
        'inherited': n['id'] in adopted_ids} for n, s in pairs if s and s.get('status') != 'VERIFIED']
    return {'plan_fingerprint': plan['fingerprint'], 'feature_names': [FEATURE],
        'top1_parent_months_total': len(parents),
        'top1_parent_months_complete': sum(covered[p] == Counter(v['request']['day']) for p, v in parents.items()),
        'top1_expected_request_hours': sum(p['expected_hours'] for p in parents.values()),
        'top1_covered_request_hours': sum(sum(v.values()) * len(parents[p]['request']['time']) for p, v in covered.items()),
        'reused_pressure_payloads': len(plan['reused']), 'inherited_remote_ids': len(adopted_ids),
        'inherited_downloads_verified': sum(n['id'] in adopted_ids and s.get('status') == 'VERIFIED' for n, s in pairs),
        'new_top1_batches_verified': sum(n['id'] not in adopted_ids and s.get('status') == 'VERIFIED' for n, s in pairs),
        'remaining_unsubmitted': sum(not s for _, s in pairs), 'pending_or_blocked': pending,
        'model_training_started': False, 'old_top10_complete': False}


def guard_states(pairs):
    REQUIRE(not any(s.get('status') in ('SUBMITTING', 'AMBIGUOUS_SUBMISSION') for _, s in pairs),
            'Submission ambiguous: reconcile exact official request before POST')
    REQUIRE(not any(s.get('status', '').startswith('BLOCKED') for _, s in pairs),
            'Blocked request needs diagnosis; do not assume slot is free')
    active = [n for n, s in pairs if s.get('request_id') and s.get('status') != 'VERIFIED']
    REQUIRE(len(active) <= 2, 'More than two active remote IDs')
    return active


def cycle(plan, client):
    pairs = [(n, transport.state(n)) for n in runtime_nodes(plan)]
    guard_states(pairs)
    for node, saved in pairs:
        if saved.get('request_id') and saved.get('status') != 'VERIFIED':
            transport.poll(node, client)
    pairs = [(n, transport.state(n)) for n in runtime_nodes(plan)]
    active = guard_states(pairs)
    for node, saved in pairs:
        if len(active) >= 2:
            break
        if not saved:
            REQUIRE(node['request']['variable'] == ['geopotential'] and node['request']['pressure_level'] == ['1'],
                    'New POST must contain ONLY geopotential at 1hPa')
            transport.submit(node, plan, client)
            current = transport.state(node)
            guard_states([(node, current)])
            if current.get('request_id'):
                active.append(node)
    return progress(plan)


def run():
    plan = load_plan()
    qa = READ(DEST / 'preflight_qa.json')
    REQUIRE(qa['status'] == 'PASS' and qa['code_sha256'] == SHA(CODE) and qa['test_sha256'] == SHA(TEST),
            'Matching offline QA absent')
    progress(plan, verify=True)
    c = transport.client()
    while True:
        report = cycle(plan, c)
        transport.atomic(DEST / 'progress.json', {'checked_utc': transport.now(), **report})
        if report['top1_parent_months_complete'] == report['top1_parent_months_total']:
            load_plan()
            progress(plan, verify=True)
            transport.immutable(DEST / 'download_qa.json', {'status': 'PASS_DOWNLOAD_READABILITY_ONLY',
                'completed_utc': transport.now(), **report, 'full_field_QA_complete': False,
                'external_evaluation_complete': False})
            print('TOP1 ALL REQUESTED INPUT COVERAGE DOWNLOADED; NO TRAINING', flush=True)
            return
        time.sleep(30)


def status():
    plan = load_plan()
    return {**progress(plan, verify=True), 'os_lock_held': transport.lock_held(),
        'active_worker': READ(DEST / 'active_worker.json') if (DEST / 'active_worker.json').exists() else None,
        'download_qa': READ(DEST / 'download_qa.json') if (DEST / 'download_qa.json').exists() else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'status'))
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(), indent=2), flush=True)
    elif args.command == 'prepare':
        candidate = build_plan()
        with transport.writer('top1_prepare'):
            prepare(candidate)
        print(json.dumps(progress(candidate, verify=True), indent=2), flush=True)
    else:
        with transport.writer('top1_run'):
            run()
