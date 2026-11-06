"""Offline repair tests. Synthetic fixtures are isolated, NEVER acquisition inputs."""
import argparse
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import requests
import numpy as np
from common import *
import audit_transport_v003 as audit
import ncar_transport_v003 as producer


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read(ACQ / 'ncar_plan.json')
        need(sha(ACQ / 'ncar_plan.json') == PLAN_SHA, 'Test uses immutable plan')

    def test_all_2012_layouts_and_exact_children(self):
        for n in self.plan['nodes']:
            shape = tuple(map(len, n['axes']))
            # SYNTHETIC TEST VALUES, in memory only.
            synthetic = np.arange(np.prod(shape), dtype=float).reshape(shape) * .125
            children = [child_node(n, i) for i in range(len(n['epochs']))]
            self.assertEqual([c['epochs'][0] for c in children], n['epochs'])
            for c in children:
                self.assertEqual(c['axes'][1:], n['axes'][1:])
                self.assertEqual(c['levels'], n['levels'])
                self.assertEqual(c['url_path'], n['url_path'])
            text = producer.composite_ascii(n, [synthetic[i:i+1] for i in range(shape[0])])
            np.testing.assert_array_equal(audit.verified_array(text, n), synthetic)

    def test_existing_real_parent_roundtrip(self):
        count = 0
        for n in self.plan['nodes']:
            path = ACQ / 'ncar_values' / (n['id'] + '.ascii')
            if not path.exists():
                continue
            real = audit.verified_array(path.read_text(encoding='utf-8'), n)
            derived = producer.composite_ascii(n, [real[i:i+1] for i in range(len(real))])
            das = (ACQ / 'ncar_metadata' / (original.fp(n['url_path'])[:24] + '.das')).read_text(encoding='utf-8')
            np.testing.assert_array_equal(original.decode(derived, das, n), real.ravel())
            np.testing.assert_array_equal(audit.verified_array(derived, n), real)
            count += 1
        self.assertGreaterEqual(count, 480)

    def test_wrong_time_grid_variable_and_truncation_rejected(self):
        n = self.plan['nodes'][0]
        raw = (ACQ / 'ncar_values' / (n['id'] + '.ascii')).read_text(encoding='utf-8')
        for field in ['epochs', 'latitude_indices', 'longitude_indices']:
            bad = copy.deepcopy(n)
            bad[field][0] += 1
            with self.assertRaises((RuntimeError, AssertionError)):
                audit.verified_array(raw, bad)
        bad = copy.deepcopy(n)
        bad['variable'] = 'INCORRECT_PARAMETER'
        with self.assertRaises((RuntimeError, AssertionError)):
            audit.verified_array(raw, bad)
        with self.assertRaises((RuntimeError, AssertionError, IndexError)):
            audit.verified_array(raw[:len(raw)//2], n)

    def test_missing_hour_rejected_before_composite(self):
        n = next(n for n in self.plan['nodes'] if len(n['epochs']) == 24)
        a = np.zeros((1, *map(len, n['axes'][1:])))
        with self.assertRaises(RuntimeError):
            producer.composite_ascii(n, [a] * 23)

    def test_full_provenance_missing_duplicate_tamper_parameter(self):
        n = next(n for n in self.plan['nodes'][:480] if len(n['epochs']) == 24)
        test_root = REPAIR_OUT / 'synthetic_tests'
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='isolated_fixture_', dir=test_root) as directory:
            tmp = Path(directory)
            emit(tmp / 'TEST_FIXTURE_ONLY.json', dict(synthetic=True, purpose='Offline rejection tests; never scientific input'))
            actual_das = ACQ / 'ncar_metadata' / (original.fp(n['url_path'])[:24] + '.das')
            fake_das = tmp / 'ncar_metadata' / actual_das.name
            fake_das.parent.mkdir()
            fake_das.write_bytes(actual_das.read_bytes())
            dr = read(actual_das.with_name(actual_das.name + '.receipt.json'))
            dr['synthetic_test_fixture'] = True
            emit(fake_das.with_name(fake_das.name + '.receipt.json'), dr)
            refs, arrays = [], []
            for i in range(24):
                child = child_node(n, i)
                a = np.full((1, *map(len, n['axes'][1:])), i + .25)
                arrays.append(a)
                path = tmp / 'ncar_hourly_v003' / n['id'] / f't_{n["axes"][0][i]:06d}.ascii'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(producer.composite_ascii(child, [a]), encoding='utf-8')
                emit(path.with_name(path.name + '.receipt.json'), dict(url=child['url'], validation='PASS', status=200,
                     method='GET', bytes=path.stat().st_size, sha256=sha(path), synthetic_test_fixture=True))
                refs.append(dict(source=rel(path), sha256=sha(path), bytes=path.stat().st_size, time_position=i))
            parent = tmp / 'ncar_values' / (n['id'] + '.ascii')
            parent.parent.mkdir()
            parent.write_text(producer.composite_ascii(n, arrays), encoding='utf-8')
            receipt = dict(url=n['url'], validation='PASS', status=None, method='DERIVED_TIME_CONCAT', source_kind='DERIVED_NCAR_ASCII',
                           bytes=parent.stat().st_size, sha256=sha(parent), plan_sha256=PLAN_SHA, parent_url_logical_only=True,
                           fragments=refs, synthetic_test_fixture=True)
            rp = parent.with_name(parent.name + '.receipt.json')
            with patch.object(audit, 'ACQ', tmp):
                rp.write_text(json.dumps(receipt), encoding='utf-8')
                self.assertEqual(audit.audit_parent(n)['hourly_fragments'], 24)
                for refs_bad in [refs[:-1], [refs[0]] * 24]:
                    rp.write_text(json.dumps(dict(receipt, fragments=refs_bad)), encoding='utf-8')
                    with self.assertRaises((RuntimeError, AssertionError)):
                        audit.audit_parent(n)
                rp.write_text(json.dumps(receipt), encoding='utf-8')
                p0 = ROOT / refs[0]['source']
                baseline = p0.read_bytes()
                p0.write_bytes(baseline + b'X')
                with self.assertRaises(RuntimeError):
                    audit.audit_parent(n)
                p0.write_bytes(baseline)
                # Changed DAS parameter plus updated SHA must still be rejected.
                das = fake_das.read_text(encoding='utf-8')
                param = int(n['spec'].split('_')[1])
                import re
                fake_das.write_text(re.sub(r'(Int32 ecmwf_parameter\s+)' + str(param) + r'\s*;', r'\g<1>999;', das), encoding='utf-8')
                dr.update(sha256=sha(fake_das), bytes=fake_das.stat().st_size)
                fake_das.with_name(fake_das.name + '.receipt.json').write_text(json.dumps(dr), encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    audit.audit_parent(n)

    def test_bounded_timeout_attempts_and_no_validation_retry(self):
        test_root = REPAIR_OUT / 'synthetic_tests'
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mock_transport_', dir=test_root) as directory:
            tmp = Path(directory)
            n = self.plan['nodes'][0]
            f = producer.FetcherV3()
            with patch.object(producer, 'ACQ', tmp), patch.object(producer, 'OUT', tmp), patch.object(producer.time, 'sleep'):
                with patch.object(f.s, 'get', side_effect=requests.ReadTimeout('SYNTHETIC TEST TIMEOUT')) as get:
                    with self.assertRaises(producer.TransientReadFailure):
                        f.fetch_raw(n['url'], tmp/'value.ascii', lambda s: None, 180, 2)
                    self.assertEqual(get.call_count, 2)
                    self.assertFalse((tmp/'value.ascii').exists())
                class Response:
                    status_code = 200
                    headers = {'Content-Length': '4'}
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def iter_content(self, size): return iter([b'BAD!'])
                def reject(text): raise RuntimeError('SYNTHETIC invalid payload')
                with patch.object(f.s, 'get', return_value=Response()) as get:
                    with self.assertRaisesRegex(RuntimeError, 'invalid payload'):
                        f.fetch_raw(n['url'], tmp/'invalid.ascii', reject, 180, 2)
                    self.assertEqual(get.call_count, 1)
                    self.assertFalse((tmp/'invalid.ascii').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal-qa', action='store_true')
    args = parser.parse_args()
    if args.seal_qa:
        check_guards(require_tests=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairTests))
    if args.seal_qa:
        emit(REPAIR_OUT / 'offline_qa.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL', utc=now(),
             tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), source_manifest_sha256=sha(MANIFEST),
             all_parent_layouts=2012, minimum_existing_real_payloads=480, synthetic_fixtures_isolated=True,
             network_requests=0, training_calls=0, full_input_qa=False))
    sys.exit(0 if result.wasSuccessful() else 1)
