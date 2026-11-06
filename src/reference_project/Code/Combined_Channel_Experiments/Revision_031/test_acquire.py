"""Offline safety tests; no client instantiation or network."""
import copy
import json
import unittest
import acquire as a

class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batches = a.read(a.PLAN / 'input_batches.json')

    def test_exact_keys(self):
        keys = a.plan_keys(self.batches)
        self.assertEqual(len(keys), 421104)
        for station, count in [('chilbolton', 256152), ('sparsholt', 164952)]:
            self.assertEqual(sum(k[0] == station for k in keys), count)

    def test_same_hour_scope(self):
        expected = {(r['station'], r['utc_epoch_seconds']) for r in a.read(a.PLAN / 'scope_hours.json')}
        actual = {(s, t) for s, t, v, p in a.plan_keys(self.batches)}
        self.assertEqual(actual, expected)
        self.assertEqual(len(expected), 5664)

    def test_single_points(self):
        for b in self.batches:
            r = a.make_request(b)
            self.assertEqual(r['area'][:2], r['area'][2:])
            self.assertEqual(r['product_type'], ['reanalysis'])
            self.assertEqual(r['download_format'], 'unarchived')

    def test_foreign_dataset_denied(self):
        b = copy.deepcopy(self.batches[0]); b['dataset'] = 'other'
        with self.assertRaises(AssertionError): a.make_request(b)

    def test_other_point_denied(self):
        b = copy.deepcopy(self.batches[0]); b['grid_latitude'] += 0.25
        with self.assertRaises(AssertionError): a.make_request(b)

    def test_rain_not_requested(self):
        self.assertTrue(all('precipitation' not in str(a.make_request(b)) for b in self.batches))

    def test_extra_variable_denied(self):
        b = copy.deepcopy(self.batches[0]); b['variables'] = ['total_precipitation']
        with self.assertRaises(AssertionError): a.make_request(b)

    def test_duplicate_denied(self):
        with self.assertRaises(AssertionError): a.plan_keys(self.batches + [self.batches[0]])

    def test_hosts(self):
        self.assertEqual(a.permitted_url('https://cds.climate.copernicus.eu/api', True), 'cds.climate.copernicus.eu')
        for url in ['http://cds.climate.copernicus.eu/api', 'https://cds.climate.copernicus.eu.evil.org',
                    'https://user:pw@cds.climate.copernicus.eu', 'https://evil.org']:
            with self.assertRaises(RuntimeError): a.permitted_url(url, True)

    def test_data_no_api_credentials_other_hosts(self):
        a.permitted_url('https://object-store.os-api.cci1.ecmwf.int/result')
        with self.assertRaises(RuntimeError): a.permitted_url('https://object-store.os-api.cci1.ecmwf.int/result', True)

    def test_budget_boundaries(self):
        b = object.__new__(a.Budget); b.used = a.BODY_CAP - 1
        b.require(1)
        with self.assertRaises(RuntimeError): b.require(2)
        with self.assertRaises(RuntimeError): b.require(-1)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SafetyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if a.OUT.exists():
        a.put(a.OUT / 'offline_tests.json', {'utc': a.utc(), 'passed': result.wasSuccessful(),
                                          'tests': result.testsRun, 'network_requests': 0})
    raise SystemExit(0 if result.wasSuccessful() else 1)
