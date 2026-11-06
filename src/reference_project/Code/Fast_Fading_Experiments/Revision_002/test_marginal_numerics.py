"""Independent probability and scoring checks before full marginal CV."""

import unittest

import numpy as np
from scipy.integrate import quad

from fit_marginals import Hour, fit, quantile_f, rice_logpdf


def small_hour():
    f = np.array([-1., 0., 2.])
    weight = np.array([.25, .5, .25])
    q = 10 ** (-f / 10)
    return Hour("2026-10-06 12:00:00+08:00", "2026-10-06", "rain", 0, q, weight, f,
                np.r_[0., np.cumsum(weight)], np.r_[0., np.cumsum(weight * f)], 40)


class ProbabilityChecks(unittest.TestCase):
    def test_weighted_pinball_known_values(self):
        result = small_hour().pinball(np.array([0., 1.]), np.array([.25, .75]))
        np.testing.assert_allclose(result, [.3125, .4375], rtol=0, atol=1e-14)

    def test_rician_pdf_integrates_to_probability_and_unit_power(self):
        b = 3.
        density = lambda q: float(np.exp(rice_logpdf(np.array([q]), b)[0]))
        mass = quad(density, 0, np.inf, epsabs=1e-10)[0]
        power_mean = quad(lambda q: q * density(q), 0, np.inf, epsabs=1e-10)[0]
        self.assertAlmostEqual(mass, 1., places=9)
        self.assertAlmostEqual(power_mean, 1., places=9)
        np.testing.assert_allclose(rice_logpdf(np.array([.3, 1.4]), 0), [-.3, -1.4], atol=1e-14)

    def test_fitted_laws_have_unit_power_and_monotone_db_quantiles(self):
        hour = small_hour()
        for family in ("lognormal", "gamma", "rice", "empirical"):
            model = fit(family, [hour])
            self.assertAlmostEqual(model["expected_q"], 1., places=10)
            values = quantile_f(model, np.array([.01, .5, .99]))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((np.diff(values) >= 0).all())
        empirical = fit("empirical", [hour])
        self.assertAlmostEqual(float(np.dot(hour.weight, hour.q / empirical["power_normalizer"])), 1., places=12)


if __name__ == "__main__":
    unittest.main()
