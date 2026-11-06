"""Synthetic, source-independent tests before freezing the C2 worker code."""
from __future__ import annotations

import time
import unittest

import numpy as np

import run_dynamics as d


def example_model(phi: float = .4) -> dict:
    # Discrete mass .2*.25 + .65*1 + .15*2 = exactly one.
    return {"support_q": np.array([.25, 1., 2.]), "support_cdf": np.array([.2, .85, 1.]),
            "power_normalizer": 1., "ar": {"phi": phi}, "tail_thresholds_db": [3., 5.]}


def source_hour(name: str, good: np.ndarray | None = None) -> d.Series:
    good = np.ones(d.SIZE, dtype=bool) if good is None else good
    q = .3 + np.exp(.5 * np.sin(np.arange(d.SIZE) / 18))
    records = np.where(good, 10, 0).astype(np.uint8)
    q[~good] = np.nan
    return d.Series(name, "2020-01-01", "rain", 0, q, records)


class DynamicsNumerics(unittest.TestCase):
    def test_inverse_step_mass_and_power(self):
        model = example_model()
        u = (np.arange(20000) + .5) / 20000
        q = d.inverse_q(model, u)
        self.assertTrue(np.array_equal(np.unique(q), model["support_q"]))
        self.assertAlmostEqual(float(q.mean()), 1., places=13)
        self.assertTrue(np.allclose(d.inverse_q(model, np.array([0., .2, .2001, .85, 1.])), [.25, .25, 1., 1., 2.]))

    def test_midpoint_ties(self):
        model = example_model()
        z, clipped = d.midpoint_z(model, np.array([.25, 1., 2.]))
        self.assertTrue(np.allclose(d.ndtr(z), [.1, .525, .925], atol=1e-14))
        self.assertEqual(clipped, 0)
        extreme, clipped = d.midpoint_z(model, np.array([.01, 3.]))
        self.assertTrue(np.isfinite(extreme).all())
        self.assertEqual(clipped, 2)

    def test_gap_masks_entire_interval(self):
        valid = np.ones(d.SIZE, dtype=bool)
        valid[100] = False
        for lag, mask in zip(d.LAGS, d.pair_masks(valid)):
            literal = np.array([valid[i:i + lag + 1].all() for i in range(d.SIZE - lag)])
            self.assertTrue(np.array_equal(mask, literal))
        self.assertFalse(d.pair_masks(valid)[1][99])  # Both endpoints valid, interior missing.

    def test_valid_run_endpoints(self):
        self.assertTrue(np.array_equal(d.valid_runs(np.array([1, 1, 0, 1, 0, 1, 1], bool)), [[0, 2], [3, 4], [5, 7]]))
        self.assertEqual(d.valid_runs(np.zeros(5, bool)).shape, (0, 2))

    def test_ar_recursion_and_stationary_initialization(self):
        phi = -.65
        seed = 71
        innovations = np.random.default_rng(seed).standard_normal(100)
        expected = innovations.copy()
        for i in range(1, len(expected)):
            expected[i] = phi * expected[i - 1] + np.sqrt(1 - phi ** 2) * innovations[i]
        actual = d.ar_normal(phi, np.random.default_rng(seed), size=100)
        self.assertTrue(np.allclose(actual, expected, atol=1e-14))
        ensemble = np.stack([d.ar_normal(.93, np.random.default_rng(i), 32) for i in range(5000)])
        self.assertLess(float(np.max(np.abs(ensemble.mean(axis=0)))), .055)
        self.assertLess(float(np.max(np.abs(ensemble.var(axis=0) - 1))), .075)

    def test_phi_hour_balanced_not_gap_bridged(self):
        good = np.ones(d.SIZE, bool)
        good[120:400] = False
        hours = [source_hour("hour_a"), source_hour("hour_b", good)]
        model = example_model()
        result = d.estimate_phi(model, hours)
        numerator = denominator = 0.
        for h in hours:
            z = np.full(d.SIZE, np.nan)
            z[h.valid], _ = d.midpoint_z(model, h.q[h.valid])
            pairs = [(z[i], z[i + 1]) for i in range(d.SIZE - 1) if h.valid[i:i + 2].all()]
            numerator += np.mean([a * b for a, b in pairs])
            denominator += np.mean([(a * a + b * b) / 2 for a, b in pairs])
        self.assertAlmostEqual(result["phi_raw"], numerator / denominator, places=13)
        self.assertEqual(result["hours_with_pairs"], 2)

    def test_block_eligibility_sampling(self):
        good = np.zeros(d.SIZE, bool)
        good[20:100] = True; good[200:290] = True
        hours = [source_hour("legal", good)]
        library = d.block_library(hours, 64)
        self.assertEqual(library[0]["n_starts"], (80 - 64 + 1) + (90 - 64 + 1))
        template, provenance = d.block_template(library, 64, np.random.default_rng(701))
        self.assertTrue(np.isfinite(template).all())
        for piece in provenance:
            a = piece["start_s"]
            self.assertTrue(good[a:a + 64].all())
        self.assertEqual(d.block_library(hours, 91), [])

    def test_rank_reordering_preserves_full_power_sample(self):
        q = np.array([.25, 2., 1., 1., .25, 2.])
        reordered, phase = d.rank_reorder(q, np.array([3., 2., 2., 1., 4., 0.]), np.random.default_rng(17))
        self.assertTrue(np.array_equal(np.sort(q), np.sort(reordered)))
        self.assertAlmostEqual(float(q.mean()), float(reordered.mean()), places=14)
        self.assertIn(phase, range(len(q)))

    def test_common_draws_reproducibility_and_unit_power(self):
        model = example_model()
        library = d.block_library([source_hour("synthetic")], 64)
        all_means = {c: [] for c in ("iid", "gaussian_ar1", "block_0064")}
        first, _ = d.generate("block_0064", model, library, 0, "test", 0)
        repeated, _ = d.generate("block_0064", model, library, 0, "test", 0)
        self.assertTrue(np.array_equal(first, repeated))
        for r in range(300):
            samples = {c: d.generate(c, model, library, 0, "test", r)[0] for c in all_means}
            self.assertTrue(np.array_equal(np.sort(samples["iid"]), np.sort(samples["block_0064"])))
            for c, q in samples.items():
                all_means[c].append(q.mean())
        for means in all_means.values():
            self.assertLess(abs(float(np.mean(means)) - 1), .012)

    def test_literal_vs_cross_product_correction(self):
        observed = [np.array([.4, .7, 1.3]), np.array([2., 3., 4., 1.])]
        generated = np.array([[.5, 2.], [1., 1.5], [.8, 3.], [.3, 2.3]])
        result = d.score_from_moments(np.array([x.mean() for x in observed]),
                                      np.array([(x ** 2).mean() for x in observed]), generated)
        scores = []
        for j, x in enumerate(observed):
            cross = np.mean([generated[r, j] * generated[s, j] for r in range(4) for s in range(4) if r != s])
            scores.append(np.mean(x ** 2 - 2 * x * generated[:, j].mean() + cross))
        self.assertAlmostEqual(result["vs"], float(np.mean(scores)), places=13)
        self.assertGreaterEqual(result["vs_plugin"], result["vs"])
        # A finite-MC unbiased estimate can be negative: it must not be clamped.
        negative = d.score_from_moments(np.array([1.]), np.array([1.]), np.array([[0.], [2.]]))
        self.assertEqual(negative["vs"], -1.)

    def test_episodes_do_not_cross_gaps_or_endpoints(self):
        f = np.array([2., 2., 0., 2., 2., 0., 0., 2., 2., 0.])
        valid = np.ones(len(f), bool)
        valid[7] = False
        result = d.episode_arrays(f, valid, 1.)
        self.assertEqual(result["complete_fades_s"], [2])
        self.assertEqual(result["complete_interfade_s"], [1])
        self.assertEqual(result["censored_fades"], 2)
        self.assertEqual(result["observed_seconds"], 9)

    def test_candidate_budget(self):
        self.assertEqual(len(d.CANDIDATES), 14)
        self.assertEqual(len(d.CANDIDATES) * 5, 70)
        self.assertEqual(len(d.LAGS), 11)
        self.assertEqual(d.stable_seed("same"), d.stable_seed("same"))
        self.assertNotEqual(d.stable_seed("different"), d.stable_seed("same"))


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DynamicsNumerics)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    d.DYN.mkdir(parents=True, exist_ok=True)
    d.write_json(d.DYN / "numerical_tests_qa.json", {
        "status": "PASS" if result.wasSuccessful() else "FAIL", "completed_utc": d.now(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "elapsed_seconds": time.perf_counter() - started,
        "script_sha256": d.sha256(d.HERE / "run_dynamics.py"),
        "test_script_sha256": d.sha256(d.HERE / "test_dynamics_numerics.py"),
        "scope": "Synthetic only; no research source or validation/test data read",
        "finite_sample_tolerances": "MC check .012 on 300x3600 unit-power draws; AR N(0,1) .055 mean/.075 variance on 5000 realizations",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)
