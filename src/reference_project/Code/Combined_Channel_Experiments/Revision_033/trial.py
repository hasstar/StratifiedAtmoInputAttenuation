"""Bounded continuation of the existing 032 collection, never a second worker."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_033'
BASE = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_032'
CHANGE = ROOT / 'Outputs/Automation_Changes/TVT_Comparison_20261018_002'
TVT = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            digest.update(block)
    return digest.hexdigest()


def put(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return dt.datetime.now(dt.timezone.utc)


def wait_limit(cutoff, moment):
    """Keep a minute of margin; the independent guard enforces absolute cutoff."""
    return min(3600, max(0, int((cutoff - moment).total_seconds()) - 60))


def initialize():
    assert not (OUT / 'sources_before.json').exists()
    policy = read(OUT / 'trial_policy.json')
    before = sha(CHANGE / 'automation_before.toml')
    after = sha(CHANGE / 'automation_after.toml')
    assert before == policy['before_tvt_sha256'] and after == sha(TVT)
    config = tomllib.loads(TVT.read_text(encoding='utf-8-sig'))
    assert config['id'] == 'tvt' and config['status'] == 'ACTIVE'
    assert config['target_thread_id'] == '01a0bda1-92d1-7832-9508-60f4e0ec1634'
    assert config['rrule'] == 'RRULE:FREQ=MINUTELY;INTERVAL=15'
    assert '2026-10-18T08:50:00Z' in config['prompt']
    put(CHANGE / 'automation_change.json', {'utc': now().isoformat(), 'automation_id': config['id'],
        'status': config['status'], 'before_sha256': before, 'after_sha256': after,
        'same_thread': True, 'same_fifteen_minute_interval': True,
        'verified_on_disk': True, 'cutoff_utc': policy['cutoff_utc']})
    protected = read(BASE / 'sources_before.json')
    protected.update(read(BASE / 'code_before.json'))
    exceptions = []
    for name, expected in list(protected.items()):
        actual = sha(name)
        if Path(name).resolve() == TVT.resolve():
            assert expected == before and actual == after
            exceptions.append({'path': name, 'old_sha256': expected, 'new_sha256': actual,
                               'reason': 'explicit human two-hour trial and deadline'})
            protected[name] = actual
        else:
            assert actual == expected, name
    assert len(exceptions) == 1
    for name in ['sources_before.json', 'code_before.json', 'requests.json', 'authorization.json',
                 'offline_tests.json', 'HANDOFF.md', 'QA.md']:
        path = BASE / name
        protected[str(path)] = sha(path)
    for path in CHANGE.iterdir():
        if path.is_file():
            protected[str(path.resolve())] = sha(path)
    protected[str(OUT / 'trial_policy.json')] = sha(OUT / 'trial_policy.json')
    put(OUT / 'source_exceptions.json', exceptions)
    protected[str(OUT / 'source_exceptions.json')] = sha(OUT / 'source_exceptions.json')
    put(OUT / 'sources_before.json', protected)
    put(OUT / 'code_before.json', {str(path): sha(path) for path in CODE.iterdir() if path.is_file()})
    put(OUT / 'initialization.json', {'utc': now().isoformat(), 'status': 'PASS_CONTROL_INITIALIZATION',
        'protected_sources': len(protected), 'intentional_control_changes': exceptions,
        'python': sys.version, 'executable': sys.executable, 'cutoff_utc': policy['cutoff_utc'],
        'network_requests': 0, 'new_predictions': 0, 'new_scores': 0})
    print(json.dumps({'status': 'PASS_CONTROL_INITIALIZATION', 'protected_sources': len(protected),
                      'tvt_sha256': after}))


def verify_sources():
    for filename in ['sources_before.json', 'code_before.json']:
        for name, expected in read(OUT / filename).items():
            assert sha(name) == expected, f'protected source changed: {name}'


def resume():
    policy = read(OUT / 'trial_policy.json')
    cutoff = dt.datetime.fromisoformat(policy['cutoff_utc'].replace('Z', '+00:00'))
    limit = wait_limit(cutoff, now())
    if limit < 60 or (OUT / 'trial_closed.json').exists():
        print(json.dumps({'status': 'TRIAL_CLOSED_NO_NETWORK', 'cutoff': policy['cutoff_utc']}))
        return
    verify_sources()
    assert (OUT / 'guard_started.json').exists(), 'DEADLINE_GUARD_NOT_STARTED'
    # Guard holds a Windows exclusive file handle for its entire lifetime.
    try:
        with (OUT / 'deadline_guard.lock').open('r+b'):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError('DEADLINE_GUARD_NOT_HOLDING_LOCK')
    previous = max((BASE / 'runs').iterdir(), key=lambda path: path.name)
    assert not (previous / 'failure.json').exists(), 'PRIOR_FAILURE_REQUIRES_REVIEW'
    assert (previous / 'finished.json').exists(), 'PRIOR_WORKER_NOT_CONFIRMED_FINISHED'
    assert read(previous / 'finished.json')['status'] == 'WAITING_REMOTE'
    spec = importlib.util.spec_from_file_location('trial_collector32',
        ROOT / 'Code/Combined_Channel_Experiments/Revision_032/acquire_daily.py')
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    # Exact approved config change is audited above; every other source still matches.
    collector.a.verify_sources = verify_sources
    record = OUT / ('resume_' + now().strftime('%Y%m%dT%H%M%S_%f') + '.json')
    put(record, {'utc': now().isoformat(), 'pid': os.getpid(), 'argv': sys.argv,
        'source_sha256': sha(__file__), 'cutoff_utc': policy['cutoff_utc'],
        'wait_seconds': limit, 'previous_run': str(previous), 'requests_reused': str(BASE / 'requests.json'),
        'network_ledger_reused': str(BASE / 'network_ledger.jsonl')})
    collector.run(limit)


def tests():
    class Limits(unittest.TestCase):
        def test_fixed_two_hour_window(self):
            p = read(OUT / 'trial_policy.json')
            self.assertEqual((dt.datetime.fromisoformat(p['cutoff_utc']) -
                              dt.datetime.fromisoformat(p['start_utc'])).total_seconds(), 7200)

        def test_one_hour_cap(self):
            t = now()
            self.assertEqual(wait_limit(t + dt.timedelta(hours=2), t), 3600)

        def test_remaining_margin(self):
            t = now()
            self.assertEqual(wait_limit(t + dt.timedelta(minutes=10), t), 540)

        def test_deadline_closed(self):
            t = now()
            self.assertEqual(wait_limit(t, t), 0)
            self.assertEqual(wait_limit(t, t + dt.timedelta(hours=1)), 0)

        def test_no_auto_extension(self):
            self.assertFalse(read(OUT / 'trial_policy.json')['automatic_extension_allowed'])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Limits))
    put(OUT / 'unit_tests.json', {'utc': now().isoformat(), 'status': 'PASS' if result.wasSuccessful() else 'FAIL',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'network_requests': 0, 'source_sha256': sha(__file__)})
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['initialize', 'resume', 'test', 'verify'])
    mode = parser.parse_args().mode
    if mode == 'initialize':
        initialize()
    elif mode == 'resume':
        resume()
    elif mode == 'test':
        tests()
    else:
        verify_sources()
        print('PASS_PROTECTED_SOURCES')
