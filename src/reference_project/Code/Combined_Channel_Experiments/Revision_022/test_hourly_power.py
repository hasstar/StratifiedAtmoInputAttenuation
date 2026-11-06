"""Synthetic boundary cases, exclusively written results, no real labels."""
import json
import math
import unittest
from pathlib import Path
from hourly_power import hourly_power_mean, geometric_height_m, ContractError, InsufficientContextError

P = dict(wind_speed=10., height_above_antenna=3., elevation=45., k=.1, alpha=1.,
    timestep_seconds=3600, rain_unit='mm/h', wind_unit='m/s',
    height_unit='km_above_antenna', angle_unit='degree')


class BoundaryTests(unittest.TestCase):
    def result(self, rain, anchor=0, **updates):
        return hourly_power_mean(rain, [i * 3600 for i in range(len(rain))], anchor, **{**P, **updates})

    def test_uniform(self):
        x = self.result([4., 4.])
        self.assertAlmostEqual(x['power_mean_attenuation_db'], .4 * 3 / math.sin(math.pi / 4), 12)

    def test_all_zero(self):
        self.assertAlmostEqual(self.result([0., 0.])['power_mean_attenuation_db'], 0, 14)

    def test_zero_height(self):
        self.assertEqual(self.result([9.], height_above_antenna=0)['power_mean_attenuation_db'], 0)

    def test_vertical(self):
        self.assertAlmostEqual(self.result([4.], elevation=90)['power_mean_attenuation_db'], 1.2, 14)

    def test_zero_coefficient(self):
        self.assertAlmostEqual(self.result([4., 9.], k=0)['power_mean_attenuation_db'], 0, 14)

    def test_power_not_db_mean(self):
        r = self.result([0., 50.])
        self.assertLess(r['power_mean_attenuation_db'], r['arithmetic_mean_attenuation_db'])
        self.assertGreater(r['power_mean_attenuation_db'], r['anchor_attenuation_db'])

    def test_full_hour_missing_context(self):
        with self.assertRaises(InsufficientContextError): self.result([4.])

    def test_near_vertical(self):
        a = self.result([4., 4.], elevation=89.999)
        self.assertAlmostEqual(a['power_mean_attenuation_db'], 1.2 / math.sin(math.radians(89.999)), 10)

    def test_large_attenuation_no_underflow(self):
        a = self.result([10., 10.], k=100.)
        self.assertAlmostEqual(a['power_mean_attenuation_db'], 3000 / math.sin(math.pi / 4), 10)

    def test_epoch_offset_invariant(self):
        r = [1., 20., 0.]
        x = self.result(r)
        y = hourly_power_mean(r, [1800000000 + i * 3600 for i in range(3)], 0, **P)
        self.assertEqual(x, y)

    def test_nonzero_anchor(self):
        a = self.result([0., 1., 20., 0.], anchor=1)
        b = self.result([1., 20., 0.])
        for k in a: self.assertAlmostEqual(a[k], b[k], 12)

    def test_boundary_context_exact(self):
        r = self.result([1., 2.], wind_speed=1., height_above_antenna=3.6, elevation=45.)
        self.assertTrue(math.isfinite(r['power_mean_attenuation_db']))

    def test_invalid_rates(self):
        for rate in [-1., math.nan, math.inf, True]:
            with self.subTest(rate=rate), self.assertRaises(ContractError): self.result([rate, 0.])

    def test_units(self):
        for name, value in [('rain_unit', 'm/h'), ('wind_unit', 'km/h'), ('height_unit', 'm'),
                            ('angle_unit', 'radian'), ('timestep_seconds', 60)]:
            with self.subTest(name=name), self.assertRaises(ContractError): self.result([1., 2.], **{name: value})

    def test_invalid_geometry(self):
        for name, value in [('wind_speed', 0), ('wind_speed', -1), ('height_above_antenna', -1),
                            ('elevation', 0), ('elevation', 91), ('k', -1), ('alpha', 0)]:
            with self.subTest(name=name,value=value), self.assertRaises(ContractError): self.result([1., 2.], **{name: value})

    def test_missing_hour(self):
        with self.assertRaises(ContractError): hourly_power_mean([1., 2.], [0, 7200], 0, **P)

    def test_conversion(self):
        for h in [-100., 0., 1000., 6000.]:
            alt = geometric_height_m(h, earth_radius_m=6371229.)
            self.assertAlmostEqual(alt * 6371229 / (6371229 + alt), h, 10)

    def test_conversion_domain(self):
        for h, r in [(1., 0.), (1., -1.), (6371229., 6371229.), (math.nan, 6371229.)]:
            with self.subTest(h=h,r=r), self.assertRaises(ContractError): geometric_height_m(h, earth_radius_m=r)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests))
    out = Path(__file__).resolve().parents[3] / 'Outputs/Experiments/Combined_Channel_Revision_022/unit_tests.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump({'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
            'failures': [(str(t), trace) for t, trace in result.failures],
            'errors': [(str(t), trace) for t, trace in result.errors],
            'scope': 'synthetic_boundary_cases_only'}, f, indent=2)
    raise SystemExit(not result.wasSuccessful())
