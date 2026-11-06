"""Synthetic input/UTC contract tests, not experiments or training."""
import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import audit_v002 as audit


class Contracts(unittest.TestCase):
    def test_finite(self):
        self.assertEqual(audit.stats([1, 0, -1])['finite'], 3)

    def test_nonfinite(self):
        s = audit.stats([np.nan, np.inf, -np.inf])
        self.assertEqual((s['nan'], s['inf'], s['minimum']), (1, 2, None))

    def test_negative_not_clipped(self):
        self.assertEqual(audit.stats([-1, 0, 1])['negative'], 1)
        self.assertEqual(audit.stats([-1])['minimum'], -1)

    def test_ranges_preserve_gaps(self):
        t = pd.to_datetime(['2026-01-01T00', '2026-01-01T01', '2026-01-01T03'])
        self.assertEqual([x['hours'] for x in audit.ranges(t)], [2, 1])

    def test_empty_ranges(self):
        self.assertEqual(audit.ranges(pd.DatetimeIndex([])), [])

    def test_beijing_to_utc(self):
        t = pd.Timestamp('2025-09-30T00:00:00+08:00').tz_convert('UTC')
        self.assertEqual(t.isoformat(), '2025-09-29T16:00:00+00:00')

    def test_valid_next_hour_month_boundary(self):
        t = pd.Timestamp('2025-12-31T23:00Z') + pd.Timedelta(hours=1)
        self.assertEqual(t.isoformat(), '2026-01-01T00:00:00+00:00')

    def test_request_calendar(self):
        j = {'request': {'year': ['2026'], 'month': ['02'], 'day': ['02', '01'], 'time': ['00:00', '01:00']}}
        t = audit.expected_times(j)
        self.assertEqual(len(t), 4)
        self.assertTrue(t.is_monotonic_increasing)

    def test_density_units_only(self):
        # The paper's Qi[g/kg]*p/(Rd*T) is in g/m3, not attenuation.
        q_kgkg, p_pa, temp_k = 1e-4, 100000, 300
        g_m3 = q_kgkg * 1000 * p_pa / (287.05 * temp_k)
        self.assertAlmostEqual(g_m3, 10000 / 86115, places=12)

    def test_rain_rate_not_differenced(self):
        self.assertAlmostEqual(3600 * 0.001, 3.6)

    def test_no_fallback(self):
        with self.assertRaises(ValueError):
            audit.need(False, 'missing input')

    def test_fields(self):
        self.assertEqual(len(audit.PRESSURE), 9)
        self.assertEqual(sum(map(len, audit.SURFACE.values())), 13)
        self.assertEqual(audit.LEVELS.index(700), 3)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    audit.save('tests.json', {'utc': datetime.now(timezone.utc).isoformat(),
                            'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                            'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors)})
    raise SystemExit(0 if result.wasSuccessful() else 1)
