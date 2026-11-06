"""Offline scheduler and inherited transport tests; no live network requests."""
import argparse
import ast
import io
import json
import tempfile
import unittest
import uuid
from unittest.mock import Mock, patch
import requests
from scheduler_common import *
import retry_scheduler as scheduling
from retry_scheduler import Journal, RetryScheduler, CDSTicker
from acquire_v004 import FetcherV4
import ncar_transport_v003 as producer
import test_transport_v003 as inherited


class FakeClock:
    def __init__(self): self.value = 1000.0
    def time(self): return self.value
    def sleep(self, seconds): self.value += seconds


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='scheduler_fixture_', dir=OUT_V4 / 'synthetic_tests')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.journal_path = self.folder / 'journal.jsonl'
        self.clock = FakeClock()
        self.journal = Journal(self.journal_path, 'SYNTHETIC_TEST_SCOPE')
        self.url = 'https://tds.gdex.ucar.edu/thredds/dodsC/files/d633000/test.nc.ascii?test[0]'

    def scheduler(self, journal=None, **kw):
        defaults = dict(guard=lambda: None, clock=self.clock.time, sleep=self.clock.sleep)
        defaults.update(kw)
        return RetryScheduler(journal or self.journal, producer.TransientReadFailure, **defaults)

    def transient(self): return producer.TransientReadFailure('Bounded transient HTTP 503')

    def test_retry_recovery_cooldown_and_serial_cds(self):
        calls, cds_times, active = [], [], [False]
        def cds():
            self.assertFalse(active[0])
            cds_times.append(self.clock.time())
            return dict(status='accepted')
        ticker = CDSTicker(cds, clock=self.clock.time)
        def get():
            self.assertFalse(active[0])
            active[0] = True
            calls.append(self.clock.time())
            active[0] = False
            if len(calls) < 3: raise self.transient()
            return 'VALID'
        self.assertEqual(self.scheduler(tick=ticker).call(self.url, get), 'VALID')
        self.assertEqual(calls, [1000, 1060, 1240])
        self.assertEqual(cds_times, [1000, 1060, 1120, 1180, 1240])
        self.assertTrue(all(b - a >= 60 for a, b in zip(cds_times, cds_times[1:])))

    def test_four_attempt_limit_persists_after_restart(self):
        get = Mock(side_effect=self.transient())
        with self.assertRaisesRegex(RuntimeError, 'Four bounded'):
            self.scheduler().call(self.url, get)
        self.assertEqual(get.call_count, 4)
        self.assertEqual(self.clock.time(), 1840)
        reloaded = Journal(self.journal_path, 'SYNTHETIC_TEST_SCOPE')
        with self.assertRaisesRegex(RuntimeError, 'terminal'):
            self.scheduler(reloaded).call(self.url, get)
        self.assertEqual(get.call_count, 4)
        self.assertEqual(reloaded.transients, 4)

    def test_retry_wait_and_remaining_budget_survive_restart(self):
        get = Mock(side_effect=self.transient())
        with self.assertRaises(KeyboardInterrupt):
            self.scheduler(sleep=Mock(side_effect=KeyboardInterrupt)).call(self.url, get)
        self.assertEqual(get.call_count, 1)
        reloaded = Journal(self.journal_path, 'SYNTHETIC_TEST_SCOPE')
        with self.assertRaisesRegex(RuntimeError, 'Four bounded'):
            self.scheduler(reloaded).call(self.url, get)
        self.assertEqual(get.call_count, 4)
        self.assertEqual(self.clock.time(), 1840)

    def test_unknown_inflight_not_retried_but_verified_cache_reconciles(self):
        get = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt): self.scheduler().call(self.url, get)
        reloaded = Journal(self.journal_path, 'SYNTHETIC_TEST_SCOPE')
        with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
            self.scheduler(reloaded).call(self.url, get)
        self.assertEqual(get.call_count, 1)
        validated_cache = Mock(return_value='VERIFIED')
        self.assertEqual(self.scheduler(reloaded).call(self.url, validated_cache, cache_exists=True), 'VERIFIED')
        self.assertEqual(next(iter(reloaded.entries.values()))['status'], 'SUCCEEDED')

    def test_daily_failure_once_then_original_fallback_no_reset(self):
        get = Mock(side_effect=self.transient())
        for journal in (self.journal, None):
            if journal is None: journal = Journal(self.journal_path, 'SYNTHETIC_TEST_SCOPE')
            with self.assertRaises(producer.TransientReadFailure):
                self.scheduler(journal).call(self.url, get, daily=True)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(self.clock.time(), 1000)

    def test_no_validation_or_rate_limit_retry(self):
        for i, error in enumerate((RuntimeError('Invalid numeric coordinate/SHA'), producer.TransientReadFailure('Bounded transient HTTP 429'))):
            get = Mock(side_effect=error)
            with self.assertRaises(RuntimeError): self.scheduler().call(self.url + str(i), get)
            self.assertEqual(get.call_count, 1)
        self.assertEqual(self.clock.time(), 1000)

    def test_stop_during_cooldown(self):
        get = Mock(side_effect=self.transient())
        def guard(): need(self.clock.time() < 1005, 'SYNTHETIC_STOP')
        with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC_STOP'):
            self.scheduler(guard=guard).call(self.url, get)
        self.assertEqual(get.call_count, 1)

    def test_total_transient_budget(self):
        get = Mock(side_effect=self.transient())
        with patch.object(scheduling, 'MAX_TRANSIENTS', 1):
            with self.assertRaisesRegex(RuntimeError, 'total transient ceiling'):
                self.scheduler().call(self.url, get)
        self.assertEqual(get.call_count, 1)

    def test_journal_truncation_tamper_and_scope_rejected(self):
        self.scheduler().call(self.url, Mock(return_value='VALID'))
        original_bytes = self.journal_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'scope'):
            Journal(self.journal_path, 'WRONG_SCOPE')
        for index, data in enumerate((original_bytes[:-1], original_bytes.replace(b'SUCCEEDED', b'CORRUPTED'))):
            fixture = self.folder / f'bad_journal_{index}.jsonl'
            fixture.write_bytes(data)
            with self.assertRaises(RuntimeError): Journal(fixture, 'SYNTHETIC_TEST_SCOPE')

    def test_sealed_fetcher_single_physical_attempt_and_cache_integrity(self):
        n = read(ACQ / 'ncar_plan.json')['nodes'][0]
        path = self.folder / 'payload.ascii'
        fetcher = FetcherV4(self.scheduler(), lambda **kw: None)
        class Response:
            status_code, headers = 200, {'Content-Length': '4'}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def iter_content(self, size): return iter([b'GOOD'])
        with patch.object(producer, 'ACQ', self.folder), patch.object(producer, 'OUT', self.folder):
            with patch.object(fetcher.s, 'get', side_effect=[requests.ReadTimeout('SYNTHETIC'), Response()]) as get:
                result = fetcher.fetch_raw(n['url'], path, lambda text: text, 180, 2)
                self.assertEqual(result[0], 'GOOD')
                self.assertEqual(get.call_count, 2)
                self.assertEqual(self.clock.time(), 1060)
                self.assertEqual(result[2]['transport'], 'v003')
            with patch.object(fetcher.s, 'get', side_effect=AssertionError('No network allowed')) as get:
                self.assertEqual(fetcher.fetch_raw(n['url'], path, lambda text: text, 180, 2)[0], 'GOOD')
                self.assertEqual(get.call_count, 0)
                path.write_bytes(b'EVIL')  # isolated synthetic fixture only
                with self.assertRaisesRegex(RuntimeError, 'integrity'):
                    fetcher.fetch_raw(n['url'], path, lambda text: text, 180, 2)
                self.assertEqual(get.call_count, 0)

    def test_cds_completed_ticker_stops(self):
        step = Mock(return_value=dict(status='ALL_DOWNLOADED_NOT_YET_INPUT_QA', done=12))
        tick = CDSTicker(step, clock=self.clock.time)
        tick()
        self.clock.sleep(3600)
        tick()
        self.assertEqual(step.call_count, 1)

    def test_all_new_modules_parse(self):
        for path in CODE_V4.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal-qa', action='store_true')
    args = parser.parse_args()
    (OUT_V4 / 'synthetic_tests').mkdir(parents=True, exist_ok=True)
    if args.seal_qa: check_guards(require_tests=False)
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(SchedulerTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(inherited.RepairTests),
    ])
    stream = io.StringIO()
    with patch.object(inherited, 'REPAIR_OUT', OUT_V4):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    record = dict(utc=now(), status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), network_requests=0, training_calls=0,
        full_input_qa=False, synthetic_fixtures_isolated=True, test_log=stream.getvalue(),
        all_parent_layouts=2012, minimum_existing_real_payloads=509,
        scheduler_sources={rel(p): sha(p) for p in CODE_V4.iterdir() if p.is_file()})
    if args.seal_qa: record['source_manifest_sha256'] = sha(MANIFEST_V4)
    emit(OUT_V4 / 'test_runs' / (uuid.uuid4().hex + '.json'), record)
    if args.seal_qa and result.wasSuccessful(): emit(OUT_V4 / 'offline_qa.json', record)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
