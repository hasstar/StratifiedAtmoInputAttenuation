"""Offline Top1 scope and two-slot migration tests; reuse v002 recovery tests."""
import ast
import copy
import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weather_download_top1_v003 as w
from test_weather_download_v002 import RecoveryTests


def fixture():
    parent = {'id': 'source', 'collection_id': 'pressure', 'expected_hours': 8,
        'request': {'day': ['01', '02', '03', '04'], 'time': ['00:00', '01:00'], 'year': ['2022'],
            'month': ['08'], 'area': [1, 0, 0, 0], 'pressure_level': ['1', '5'],
            'variable': ['geopotential', 'u_component_of_wind'], 'product_type': ['reanalysis'],
            'data_format': 'netcdf', 'download_format': 'unarchived'}}
    def node(name, days, single):
        req = w.reduced_request(parent, days) if single else {**copy.deepcopy(parent['request']), 'day': days}
        return {'id': name, 'parent_id': 'source', 'collection_id': 'pressure', 'request': req,
            'expected_hours': 2 * len(days), 'target': 'payloads/' + name + '.download'}
    reused, inherited, new = node('done', ['01'], False), node('inherit', ['02'], False), node('new', ['03', '04'], True)
    plan = {'parents': [parent], 'reused': [{'job': reused, 'parent_id': 'source'}],
        'adopted': [{'id': 'inherit', 'request_id': 'official-id', 'original_job': copy.deepcopy(inherited)}],
        'jobs': [inherited, new], 'fingerprint': 'synthetic'}
    return plan


class Top1Tests(unittest.TestCase):
    def test_scope_exact(self):
        self.assertEqual(w.coverage_contract(fixture())['expected_request_hours'], 8)

    def test_extra_variable_rejected(self):
        p = fixture(); p['jobs'][-1]['request']['variable'].append('relative_humidity')
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_other_level_rejected(self):
        p = fixture(); p['jobs'][-1]['request']['pressure_level'] = ['5']
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_changed_region_rejected(self):
        p = fixture(); p['jobs'][-1]['request']['area'] = [1, 1, 0, 0]
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_changed_hours_rejected(self):
        p = fixture(); p['jobs'][-1]['request']['time'] = ['00:00']
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_missing_day_rejected(self):
        p = fixture(); p['jobs'][-1]['request']['day'] = ['03']; p['jobs'][-1]['expected_hours'] = 2
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_duplicated_day_rejected(self):
        p = fixture(); p['jobs'][-1]['request']['day'] = ['02', '03', '04']; p['jobs'][-1]['expected_hours'] = 6
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_inherited_request_must_stay_exact(self):
        p = fixture(); p['jobs'][0]['request']['variable'] = ['geopotential']
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_duplicate_remote_id_rejected(self):
        p = fixture(); p['adopted'].append(copy.deepcopy(p['adopted'][0]))
        with self.assertRaises(RuntimeError): w.coverage_contract(p)

    def test_ambiguous_never_frees_slot(self):
        with self.assertRaises(RuntimeError):
            w.guard_states([({}, {'status': 'AMBIGUOUS_SUBMISSION'})])

    def test_access_block_never_frees_slot(self):
        with self.assertRaises(RuntimeError):
            w.guard_states([({}, {'status': 'BLOCKED_REMOTE_ACCESS', 'request_id': 'saved'})])

    def test_three_active_rejected(self):
        with self.assertRaises(RuntimeError):
            w.guard_states([({}, {'status': 'WAITING_REMOTE', 'request_id': str(i)}) for i in range(3)])

    def test_verified_slot_is_released(self):
        self.assertEqual(len(w.guard_states([({}, {'status': 'VERIFIED', 'request_id': 'done'})])), 0)

    def test_two_inherited_ids_prevent_post(self):
        p = fixture(); nodes = [p['jobs'][0], copy.deepcopy(p['jobs'][0]), p['jobs'][1]]
        nodes[1]['id'] = 'inherit2'
        def state(node):
            return {} if node['id'] == 'new' else {'status': 'WAITING_REMOTE', 'request_id': node['id']}
        with patch.object(w, 'runtime_nodes', return_value=nodes), patch.object(w.transport, 'state', side_effect=state), \
             patch.object(w.transport, 'poll') as poll, patch.object(w.transport, 'submit') as submit, \
             patch.object(w, 'progress', return_value={}):
            w.cycle(p, SimpleNamespace())
            self.assertEqual(poll.call_count, 2); submit.assert_not_called()

    def test_inherited_failed_split_not_resubmitted(self):
        with patch.object(w.transport, 'state', return_value={'request_id': 'official-id', 'status': 'SPLIT_SIZE_REJECTION'}):
            with self.assertRaises(RuntimeError): w.runtime_nodes(fixture())

    def test_inherited_id_loss_rejected(self):
        with patch.object(w.transport, 'state', return_value={'status': 'WAITING_REMOTE'}):
            with self.assertRaises(RuntimeError): w.runtime_nodes(fixture())

    def test_no_training_calls_or_model_imports(self):
        tree = ast.parse(w.CODE.read_text(encoding='utf-8'))
        denied = {'fit', 'fit_transform', 'partial_fit', 'backward', 'train'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, denied)
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name in ('torch', 'sklearn', 'local_engine', 'external_prepare') for a in node.names))


if __name__ == '__main__':
    plan = w.build_plan()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(Top1Tests)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = unittest.TextTestRunner(stream=out, verbosity=2).run(suite)
    if not result.wasSuccessful():
        w.transport.failure('TOP1_OFFLINE_PREFLIGHT_FAILED', tests_run=result.testsRun, transcript=out.getvalue())
        print(out.getvalue()); raise SystemExit(1)
    for name, expected in plan['protected_files'].items():
        w.REQUIRE(w.SHA(w.ROOT / name) == expected, 'Old source changed during tests')
    report = {'status': 'PASS', 'completed_utc': w.transport.now(), 'code_sha256': w.SHA(w.CODE),
        'test_sha256': w.SHA(w.TEST), 'tests_run': result.testsRun, 'contract': w.coverage_contract(plan),
        'old_protected_files_unchanged': len(plan['protected_files']), 'network_requests': 0,
        'new_fit_calls': 0, 'transcript': out.getvalue()}
    w.transport.immutable(w.DEST / 'preflight_qa.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'transcript'}, indent=2))
