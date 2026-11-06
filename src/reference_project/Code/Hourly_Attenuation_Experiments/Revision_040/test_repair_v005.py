"""Offline tests: inherited transport plus CDS isolation, budgets, secrets and relay gates."""
import argparse
import ast
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
import uuid
import warnings
from unittest.mock import Mock, patch
import requests
from repair_common_v005 import *
import cds_isolated_v005 as cds
import cds_scheduler_v005 as cs
import test_scheduler_v004 as inherited_scheduler
import test_transport_v003 as inherited_transport
from retry_scheduler import RetryScheduler
from ncar_transport_v003 import TransientReadFailure
from acquire_v004 import partial_audit
from audit_transport_v003 import audit_all
from continue_delivery_v005 import exact_command, validate_success


class RepairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='v5_', dir=OUT_V5 / 'synthetic_tests')
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.clock = inherited_scheduler.FakeClock()
        self.scope = 'SYNTHETIC_CDS_ONLY'
        self.log = self.folder / 'cds.jsonl'

    def ticker(self, operation, **kw):
        defaults = dict(clock=self.clock.time, guard=lambda: None)
        defaults.update(kw)
        return cs.PersistentCDSTicker(Journal(self.log, self.scope), operation, **defaults)

    def transient(self):
        return cds.CDSTransientError(cds.failure_record(requests.exceptions.ProxyError('synthetic connection'), 'POLL_GET', 'fixture'))

    def fixture(self, submitted=True):
        plan = copy.deepcopy(read(ACQ / 'cds_three/plan.json'))
        emit(self.folder / 'plan.json', plan)
        job = plan['jobs'][0]
        batch = self.folder / 'jobs' / job['id']
        if submitted:
            emit(batch / 'intent.json', dict(request=job['request'], request_sha256=job['request_sha256'],
                 collection_id=cds.COLLECTION, plan_sha256=sha(self.folder / 'plan.json')))
            emit(batch / 'submitted.json', dict(request_id='synthetic-new-request-1234', request_sha256=job['request_sha256']))
        return job, batch

    def call_step(self, client):
        with patch.object(cds, 'DEST', self.folder), patch.object(cds, 'quiet_client', return_value=client), patch.object(cds, 'stopped'):
            return cds.step()

    def remote(self, status='accepted'):
        return Mock(request_id='synthetic-new-request-1234', collection_id=cds.COLLECTION, status=status)

    def test_transient_cds_does_not_abort_ncar_and_never_overlaps(self):
        calls = []
        def operation():
            calls.append('CDS_START')
            calls.append('CDS_END')
            raise self.transient()
        tick = self.ticker(operation)
        ncar = Mock(side_effect=lambda: calls.append('NCAR') or 'VERIFIED')
        j = Journal(self.folder / 'ncar.jsonl', 'NCAR_FIXTURE')
        scheduler = RetryScheduler(j, TransientReadFailure, tick=tick, guard=lambda: None,
            clock=self.clock.time, sleep=self.clock.sleep)
        self.assertEqual(scheduler.call('synthetic-url', ncar), 'VERIFIED')
        self.assertEqual(calls, ['CDS_START', 'CDS_END', 'NCAR'])
        self.assertEqual(tick.state['status'], 'CDS_RETRY_WAIT_NCAR_CAN_CONTINUE')

    def test_cds_retry_delays_and_exhaustion_persist(self):
        operation = Mock(side_effect=self.transient())
        expected = [(1000, 1060), (1060, 1240), (1240, 1840), (1840, 0)]
        for instant, next_time in expected:
            self.clock.value = instant
            tick = self.ticker(operation)
            tick()
            self.assertEqual(tick.state['next_allowed'], next_time)
        self.assertEqual(tick.state['status'], cs.BLOCKED)
        self.clock.sleep(10000)
        self.ticker(operation)()
        self.assertEqual(operation.call_count, 4)
        self.assertEqual(tick.state['consecutive_failures'], 4)
        self.assertEqual(tick.state['total_transients'], 4)

    def test_success_resets_consecutive_but_not_total_and_preserves_poll_interval(self):
        operation = Mock(side_effect=[self.transient(), dict(status='accepted', done=5), self.transient()])
        self.ticker(operation)()
        self.clock.sleep(59)
        self.ticker(operation)()
        self.assertEqual(operation.call_count, 1)
        self.clock.sleep(1)
        tick = self.ticker(operation)
        tick()
        self.assertEqual((tick.state['consecutive_failures'], tick.state['total_transients']), (0, 1))
        self.clock.sleep(60)
        tick = self.ticker(operation)
        tick()
        self.assertEqual((tick.state['consecutive_failures'], tick.state['total_transients']), (1, 2))
        self.assertEqual(tick.state['next_allowed'], self.clock.time() + 60)

    def test_global_cds_budget_cannot_reset_across_relay(self):
        j = Journal(self.log, self.scope)
        j.record('CDS', 'FIXTURE', dict(status='CDS_WAIT', consecutive_failures=0, total_transients=63,
            operations=100, next_allowed=0, observation=None))
        operation = Mock(side_effect=self.transient())
        tick = self.ticker(operation)
        tick()
        self.assertEqual(tick.state['status'], cs.BLOCKED)
        self.ticker(operation)()
        self.assertEqual(operation.call_count, 1)

    def test_cds_completed_state_never_polls_again(self):
        operation = Mock(return_value=dict(status=cs.DONE, done=12))
        self.ticker(operation)()
        self.clock.sleep(3600)
        self.ticker(operation)()
        self.assertEqual(operation.call_count, 1)

    def test_unknown_inflight_cds_operation_requires_inspection(self):
        operation = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt): self.ticker(operation)()
        with self.assertRaisesRegex(RuntimeError, 'Unresolved'): self.ticker(operation)
        self.assertEqual(operation.call_count, 1)

    def test_hard_cds_error_stops_before_ncar(self):
        tick = self.ticker(Mock(side_effect=RuntimeError('synthetic identity mismatch')))
        physical = Mock()
        scheduler = RetryScheduler(Journal(self.folder / 'ncar.jsonl', 'NCAR'), TransientReadFailure,
            tick=tick, guard=lambda: None, clock=self.clock.time, sleep=self.clock.sleep)
        with self.assertRaises(cds.CDSSafeError): scheduler.call('synthetic', physical)
        physical.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'fatal'): self.ticker(Mock())

    def test_cds_error_classification_is_narrow(self):
        for exc in [requests.ReadTimeout(), requests.ConnectTimeout(), requests.ConnectionError(), requests.exceptions.ProxyError('connect failed')]:
            self.assertTrue(cds.failure_record(exc, 'POLL_GET', 'fixture')['transient'])
            self.assertFalse(cds.failure_record(exc, 'SUBMIT', 'fixture')['transient'])
        for exc in [requests.exceptions.SSLError(), requests.exceptions.ProxyError('407 Proxy Authentication Required'), RuntimeError(), ValueError()]:
            self.assertFalse(cds.failure_record(exc, 'POLL_GET', 'fixture')['transient'])
        for status in (400, 401, 403, 404, 408, 429, 500, 502, 503, 504):
            response = requests.Response()
            response.status_code = status
            exc = requests.HTTPError('synthetic', response=response)
            self.assertEqual(cds.failure_record(exc, 'POLL_GET', 'fixture')['transient'], status in cds.TRANSIENT_HTTP)

    def test_warning_and_error_text_never_exposed(self):
        sentinel = 'SYNTHETIC_SECRET_SENTINEL'
        def noisy_client():
            warnings.warn(sentinel)
            return 'CLIENT'
        with warnings.catch_warnings(record=True) as outside, patch.object(cds, 'client', noisy_client):
            self.assertEqual(cds.quiet_client(), 'CLIENT')
        self.assertEqual(outside, [])
        self.fixture()
        c = Mock()
        c.get_remote.side_effect = requests.exceptions.ProxyError(sentinel)
        with self.assertRaises(cds.CDSTransientError) as caught: self.call_step(c)
        self.assertNotIn(sentinel, str(caught.exception))
        for path in self.folder.rglob('*.json'):
            self.assertNotIn(sentinel, path.read_text(encoding='utf-8'))
        c.submit.assert_not_called()

    def test_only_registered_id_is_polled(self):
        job, _ = self.fixture()
        c = Mock()
        c.get_remote.return_value = self.remote()
        result = self.call_step(c)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['job'], job['id'])
        c.get_remote.assert_called_once_with('synthetic-new-request-1234')
        c.submit.assert_not_called()

    def test_error_receipt_write_failure_still_suppresses_raw_error(self):
        sentinel = 'SYNTHETIC_PRIVATE_ERROR'
        with patch.object(cds, 'emit', side_effect=OSError(sentinel)):
            with self.assertRaises(cds.CDSSafeError) as caught:
                cds.raise_safe(requests.ConnectionError(sentinel), 'POLL_GET', 'fixture', self.folder)
        self.assertNotIsInstance(caught.exception, cds.CDSTransientError)
        self.assertNotIn(sentinel, str(caught.exception))
        self.assertEqual(caught.exception.record['phase'], 'LOCAL_ERROR_RECEIPT')

    def test_ambiguous_post_is_terminal_and_not_repeated(self):
        _, batch = self.fixture(submitted=False)
        c = Mock()
        c.submit.side_effect = requests.ReadTimeout('synthetic')
        with self.assertRaises(cds.CDSSafeError): self.call_step(c)
        self.assertTrue((batch / 'intent.json').exists())
        self.assertTrue((batch / 'failure.json').exists())
        with self.assertRaises(RuntimeError): self.call_step(c)
        self.assertEqual(c.submit.call_count, 1)

    def test_successful_new_post_registered_once(self):
        _, batch = self.fixture(submitted=False)
        c = Mock()
        c.submit.return_value = self.remote()
        c.get_remote.return_value = self.remote()
        self.assertEqual(self.call_step(c)['status'], 'SUBMITTED_NEW_BOUNDED_JOB')
        self.assertTrue((batch / 'submitted.json').exists())
        self.assertEqual(self.call_step(c)['status'], 'accepted')
        self.assertEqual(c.submit.call_count, 1)

    def test_wrong_remote_identity_unknown_status_and_remote_failure_stop(self):
        self.fixture()
        for changes in [dict(request_id='wrong-id'), dict(collection_id='wrong-collection'), dict(status='unexpected')]:
            c = Mock()
            c.get_remote.return_value = self.remote()
            for key, value in changes.items(): setattr(c.get_remote.return_value, key, value)
            with self.assertRaises(cds.CDSSafeError): self.call_step(c)
            c.get_remote.return_value.download.assert_not_called()
        c = Mock()
        c.get_remote.return_value = self.remote('failed')
        with self.assertRaises(cds.CDSSafeError): self.call_step(c)

    def test_corrupt_payload_not_counted_or_retried(self):
        _, batch = self.fixture()
        c = Mock()
        remote = c.get_remote.return_value = self.remote('successful')
        remote.download.side_effect = lambda path: Path(path).write_bytes(b'SYNTHETIC_INVALID_PAYLOAD')
        with patch('assemble_inputs.read_cds', side_effect=RuntimeError('synthetic invalid fields')):
            with self.assertRaises(cds.CDSSafeError): self.call_step(c)
        self.assertFalse((batch / 'download.json').exists())
        self.assertEqual(remote.download.call_count, 1)

    def test_validated_payload_receipt_is_append_only_and_not_full_input_qa(self):
        job, batch = self.fixture()
        c = Mock()
        remote = c.get_remote.return_value = self.remote('successful')
        remote.download.side_effect = lambda path: Path(path).write_bytes(b'SYNTHETIC_TEST_ONLY')
        with patch('assemble_inputs.read_cds', return_value=([None] * (3 * len(job['epochs'])), {})):
            result = self.call_step(c)
        self.assertEqual(result['status'], 'DOWNLOADED_NOT_YET_INPUT_QA')
        receipt = read(batch / 'download.json')
        self.assertEqual(sha(self.folder / receipt['file']), receipt['sha256'])

    def test_inherited_ncar_retry_budget_is_unchanged(self):
        base_path, delta_path = self.folder / 'base.jsonl', self.folder / 'delta.jsonl'
        base = Journal(base_path, 'BASE')
        url = 'synthetic-url'
        key = hashlib.sha256(url.encode()).hexdigest()
        base.record(key, 'RETRY_WAIT', dict(status='RETRY_WAIT', attempts=1, next_allowed=1060))
        pinned = sha(base_path)
        j = InheritedJournal(base_path, 'BASE', pinned, delta_path, 'DELTA')
        get = Mock(side_effect=TransientReadFailure('Bounded transient HTTP 503'))
        scheduler = RetryScheduler(j, TransientReadFailure, guard=lambda: None,
            clock=self.clock.time, sleep=self.clock.sleep)
        with self.assertRaisesRegex(RuntimeError, 'Four bounded'): scheduler.call(url, get)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(j.transients, 4)
        self.assertEqual(sha(base_path), pinned)
        again = InheritedJournal(base_path, 'BASE', pinned, delta_path, 'DELTA')
        self.assertEqual(again.transients, 4)
        self.assertEqual(again.entries[key]['attempts'], 4)
        with self.assertRaises(RuntimeError): InheritedJournal(base_path, 'BASE', 'wrong', delta_path, 'DELTA')

    def test_inherited_terminal_state_cannot_be_reset(self):
        base_path = self.folder / 'base.jsonl'
        base = Journal(base_path, 'BASE')
        base.record('key', 'DAILY_FALLBACK', dict(status='FALLBACK', attempts=1, next_allowed=0))
        j = InheritedJournal(base_path, 'BASE', sha(base_path), self.folder / 'delta.jsonl', 'DELTA')
        with self.assertRaisesRegex(RuntimeError, 'reset'): j.record('key', 'NEW', dict(status='NEW', attempts=0))
        with self.assertRaisesRegex(RuntimeError, 'terminal'): j.record('key', 'NEW', dict(status='NEW', attempts=1))

    def test_cds_journal_scope_and_tamper_rejected(self):
        self.ticker(Mock(return_value=dict(status='accepted')))()
        with self.assertRaises(RuntimeError): Journal(self.log, 'WRONG')
        raw = self.log.read_bytes()
        self.log.write_bytes(raw[:-1])  # isolated synthetic fixture only
        with self.assertRaises(RuntimeError): Journal(self.log, self.scope)

    def test_exact_worker_command_and_success_gates(self):
        run = 'a' * 32
        exact_command(['python', str(CODE_V5 / 'acquire_v005.py'), '--run-id', run], run)
        for command in [['python', str(CODE_V5 / 'acquire_v005.py'), '--run-id', 'b' * 32],
                        ['python', str(v4.CODE_V4 / 'acquire_v004.py'), '--run-id', run]]:
            with self.assertRaises(RuntimeError): exact_command(command, run)
        result = dict(status='SUCCESS', pid=123, create_time=1000, run_id=run,
            completion_manifest_sha256='DONE', source_manifest_sha256='MANIFEST')
        done = dict(nodes=2012, plan_sha256=PLAN_SHA, transport='v005', run_id=run, pid=123, create_time=1000,
            scheduler_manifest_sha256='MANIFEST', transport_qa_sha256='QA')
        qa = dict(status='PASS', nodes=2012, scheduler_manifest_sha256='MANIFEST', source_manifest_sha256=v4.UPSTREAM_SHA)
        validate_success(result, done, qa, 123, 1000, run, 'MANIFEST', 'DONE', 'QA')
        for bad in [dict(result, status='FAILED_REQUIRES_INSPECTION'), dict(result, create_time=1001), dict(result, run_id='wrong')]:
            with self.assertRaises(RuntimeError): validate_success(bad, done, qa, 123, 1000, run, 'MANIFEST', 'DONE', 'QA')
        with self.assertRaises(RuntimeError): validate_success(result, dict(done, nodes=517), qa, 123, 1000, run, 'MANIFEST', 'DONE', 'QA')

    def test_all_new_modules_parse(self):
        for path in CODE_V5.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal-qa', action='store_true')
    args = parser.parse_args()
    (OUT_V5 / 'synthetic_tests').mkdir(parents=True, exist_ok=True)
    if args.seal_qa:
        check_guards(require_tests=False)
    else:
        v4.check_guards()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (
        inherited_scheduler.SchedulerTests, inherited_transport.RepairTests, RepairTests)])
    stream = io.StringIO()
    with patch.object(inherited_scheduler, 'OUT_V4', OUT_V5), patch.object(inherited_transport, 'REPAIR_OUT', OUT_V5), \
         patch.object(requests.Session, 'request', side_effect=AssertionError('Offline test attempted network')):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    actual = audit_all(require_complete=False)
    partial = partial_audit(read(ACQ / 'ncar_plan.json'), actual['nodes'])
    cds_cache = cds_cache_audit()
    need(actual['nodes'] >= 517 and cds_cache['downloaded'] >= 5, 'Available real input regression')
    record = dict(utc=now(), status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), network_requests=0, training_calls=0,
        full_input_qa=False, synthetic_fixtures_isolated=True, test_log=stream.getvalue(),
        all_parent_layouts=2012, actual_available_transport=actual, partial=partial, actual_cds=cds_cache,
        repair_sources={rel(p): sha(p) for p in CODE_V5.iterdir() if p.is_file()})
    if args.seal_qa:
        record['source_manifest_sha256'] = sha(MANIFEST_V5)
    path = OUT_V5 / 'test_runs' / (uuid.uuid4().hex + '.json')
    emit(path, record)
    if args.seal_qa and result.wasSuccessful(): emit(OUT_V5 / 'offline_qa.json', record)
    print(json.dumps(dict(status=record['status'], tests=result.testsRun, record=rel(path), sha256=sha(path))), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
