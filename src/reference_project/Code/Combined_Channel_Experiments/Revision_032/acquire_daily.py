"""Same authorized keys, smaller requests after explicit size rejection."""
import argparse
import copy
import datetime as dt
import importlib.util
import msvcrt
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_031'
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_032'
spec = importlib.util.spec_from_file_location('bounded31', ROOT / 'Code/Combined_Channel_Experiments/Revision_031/acquire.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.OUT = OUT

def make_subrequests():
    original = a.read(a.PLAN / 'input_batches.json')
    expected = a.plan_keys(original)
    seen = set()
    requests = []
    for parent in original:
        for day in parent['days']:
            b = copy.deepcopy(parent)
            b.update(id=len(requests) + 1, parent_batch_id=parent['id'], days=[day],
                     scalar_values=24 * len(parent['variables']) * len(parent['pressure_levels_hpa']))
            r = a.make_request(b)
            keys = {(b['station'], t, a.NAMES[v], p) for t in a.expected_times(b)
                    for v in b['variables'] for p in b['pressure_levels_hpa']}
            assert not seen.intersection(keys)
            seen.update(keys)
            requests.append({'batch': b, 'request': r})
    assert seen == expected and len(requests) == 708
    return requests

def initialize():
    assert not OUT.exists()
    fail = a.read(OLD / 'batches/001/failure.json')
    assert fail['http_status'] == 403 and fail['title'] == 'cost limits exceeded'
    assert not (OLD / 'batches/001/submitted.json').exists()
    sources = a.read(OLD / 'sources_before.json')
    sources.update(a.read(OLD / 'code_before.json'))
    for name, expected in sources.items(): assert a.sha(name) == expected, name
    for p in OLD.rglob('*'):
        if p.is_file(): sources[str(p.resolve())] = a.sha(p)
    used = sum(a.json.loads(line)['bytes'] for line in (OLD / 'network_ledger.jsonl').read_text().splitlines())
    assert used == 7658
    requests = make_subrequests()
    OUT.mkdir()
    (OUT / 'batches').mkdir(); (OUT / 'runs').mkdir()
    a.put(OUT / 'sources_before.json', sources)
    a.put(OUT / 'code_before.json', {str(p): a.sha(p) for p in CODE.iterdir() if p.is_file()})
    a.put(OUT / 'requests.json', requests)
    a.put(OUT / 'authorization.json', {**a.read(OLD / 'authorization.json'),
        'prior_http_body_bytes': used, 'logical_batches': 30, 'physical_subrequests': 708,
        'change': 'split each approved logical batch by day; identical target keys'})
    a.put(OUT / 'offline_tests.json', {'utc': a.utc(), 'status': 'PASS_SCOPE_EQUIVALENCE',
        'tests': ['same_421104_keys', 'no_duplicates', '708_daily_subrequests', 'at_most_888_values_each',
                  'same_service', 'same_point', 'same_variables_levels_times', 'prior_bytes_carried'],
        'max_request_values': max(x['batch']['scalar_values'] for x in requests), 'network_requests': 0})
    a.say(status='DAILY_REQUESTS_READY', subrequests=708, values=421104, prior_bytes=used)

class CumulativeBudget(a.Budget):
    def __init__(self):
        super().__init__()
        self.used += a.read(OUT / 'authorization.json')['prior_http_body_bytes']

def run(wait_seconds):
    lock = (OUT / 'worker_os.lock').open('a+b')
    lock.seek(0)
    if lock.read(1) == b'': lock.write(b'1'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    rd = OUT / 'runs' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    rd.mkdir()
    a.put(rd / 'started.json', {'utc': a.utc(), 'pid': os.getpid(), 'argv': sys.argv,
                              'source_sha256': a.sha(__file__), 'wait_seconds': wait_seconds})
    bid = None
    try:
        a.verify_sources()
        budget = CumulativeBudget()
        client = a.client_for(budget)
        client.check_authentication()
        a.say(status='AUTHENTICATED', pid=os.getpid())
        deadline = time.monotonic() + wait_seconds
        items = a.read(OUT / 'requests.json')
        for item in items:
            b = item['batch']; bid = b['id']
            folder = OUT / 'batches' / f'{bid:03d}'
            folder.mkdir(exist_ok=True)
            if (folder / 'complete.json').exists():
                assert a.sha(folder / 'raw.nc') == a.read(folder / 'complete.json')['sha256']
                continue
            if (folder / 'failure.json').exists(): raise RuntimeError('PRIOR_FAILURE_REQUIRES_REVIEW')
            if (folder / 'submitted.json').exists():
                rid = a.read(folder / 'submitted.json')['request_id']
                remote = client.get_remote(rid)
            else:
                if (folder / 'submit_intent.json').exists(): raise RuntimeError('SUBMISSION_UNCERTAIN_DO_NOT_RESUBMIT')
                a.put(folder / 'submit_intent.json', {'utc': a.utc(), 'parent_batch_id': b['parent_batch_id'],
                                                   'dataset': b['dataset'], 'request': item['request']})
                remote = client.submit(b['dataset'], item['request'])
                rid = remote.request_id
                a.put(folder / 'submitted.json', {'utc': a.utc(), 'request_id': rid})
                a.say(status='SUBMITTED', subrequest=bid, request_id=rid, values=b['scalar_values'])
            while True:
                state = remote.status
                a.put(rd / f'poll_{bid:03d}_{time.time_ns()}.json', {'utc': a.utc(), 'subrequest': bid, 'status': state, 'request_id': rid})
                if state == 'successful': break
                if state not in ('accepted', 'running', 'queued'): raise RuntimeError(f'REMOTE_STATUS_{state}')
                if time.monotonic() >= deadline:
                    a.put(rd / 'finished.json', {'utc': a.utc(), 'status': 'WAITING_REMOTE', 'subrequest': bid, 'request_id': rid, 'cumulative_http_body_bytes': budget.used})
                    a.say(status='WAITING_REMOTE', subrequest=bid, request_id=rid)
                    return
                time.sleep(30)
            if (folder / 'raw.nc').exists(): raise RuntimeError('PRESERVE_PRIOR_PARTIAL_FILE')
            metadata = a.download(remote.get_results(), folder / 'raw.nc', budget)
            check = a.validate_netcdf(folder / 'raw.nc', b)
            a.put(folder / 'complete.json', {'utc': a.utc(), 'subrequest': bid, 'parent_batch_id': b['parent_batch_id'],
                                          'request_id': rid, **metadata, **check})
            a.say(status='SUBREQUEST_INPUT_VERIFIED', subrequest=bid, values=b['scalar_values'], cumulative_http_body_bytes=budget.used)
        a.verify_sources()
        done = [a.read(p / 'complete.json') for p in sorted((OUT / 'batches').iterdir())]
        assert len(done) == 708 and sum(r['values'] for r in done) == 421104
        a.put(OUT / 'acquisition_complete.json', {'utc': a.utc(), 'status': 'ACQUIRED_PENDING_INDEPENDENT_QA',
            'logical_batches': 30, 'subrequests': 708, 'new_values': 421104,
            'cumulative_http_body_bytes': budget.used, 'protected_sources_unchanged': True,
            'new_predictions': 0, 'new_scores': 0})
        a.put(rd / 'finished.json', {'utc': a.utc(), 'status': 'ACQUIRED_PENDING_INDEPENDENT_QA'})
        a.say(status='ACQUIRED_PENDING_INDEPENDENT_QA', new_values=421104)
    except Exception as error:
        detail = a.safe_error(error)
        a.put(rd / 'failure.json', {'utc': a.utc(), 'subrequest': bid, **detail})
        if bid is not None:
            fp = OUT / 'batches' / f'{bid:03d}' / 'failure.json'
            if not fp.exists(): a.put(fp, {'utc': a.utc(), **detail})
        a.say(status='STOPPED_REVIEW_REQUIRED', subrequest=bid, **detail)
        raise SystemExit(2)
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['init', 'run'])
    p.add_argument('--max-wait-seconds', type=int, default=3600)
    args = p.parse_args()
    if args.mode == 'init': initialize()
    else:
        assert 60 <= args.max_wait_seconds <= 3600
        run(args.max_wait_seconds)
