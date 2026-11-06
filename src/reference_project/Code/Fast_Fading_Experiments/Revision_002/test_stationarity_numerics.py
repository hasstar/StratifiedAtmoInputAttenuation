"""Synthetic-only C3 numerical checks. No research residuals are read."""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

import run_dynamics as dyn
import run_stationarity as st
import freeze_generator as fg
from window_diagnostics import OUT, now, sha256, write_json


class NumericalChecks(unittest.TestCase):
    def test_known_linear_power_drift(self):
        q = np.r_[np.ones(30), np.full(30, 2.)]
        result = st.endpoint_bank(q, np.full(60, 10), 0, 60, 9)[0]
        self.assertAlmostEqual(result[0], 10 * np.log10(2), places=12)
        np.testing.assert_allclose(result[2:5], 10 * np.log10(2), atol=1e-12)
        self.assertTrue(np.isnan(result[5]))

    def test_masked_vectorization_matches_literal(self):
        rng = np.random.default_rng(2102)
        q = np.exp(rng.normal(0, .3, (7, 3600)))
        records = rng.integers(8, 11, 3600)
        for minimum in (9, 10):
            for start, width in ((60, 60), (300, 600), (0, 3600)):
                result = st.endpoint_bank(q, records, start, width, minimum)
                for r in range(7):
                    halves, means, sd, quantiles, corr = [], [], [], [], []
                    for a, b in ((start, start + width // 2), (start + width // 2, start + width)):
                        mask = records[a:b] >= minimum
                        raw = q[r, a:b]
                        f = -10 * np.log10(raw)
                        means.append(np.average(raw[mask], weights=records[a:b][mask]))
                        sd.append(np.std(f[mask], ddof=1))
                        quantiles.append(np.quantile(f[mask], [.05, .5, .95]))
                        pair = mask[:-1] & mask[1:]
                        corr.append(np.corrcoef(f[:-1][pair], f[1:][pair])[0, 1] if pair.sum() >= 3 else np.nan)
                    expected = np.r_[abs(10 * np.log10(means[0] / means[1])), abs(sd[0] - sd[1]),
                                     abs(quantiles[0] - quantiles[1]), abs(corr[0] - corr[1])]
                    np.testing.assert_allclose(result[r], expected, atol=2e-14, rtol=0)

    def test_gaps_have_no_endpoint_effect(self):
        rng = np.random.default_rng(23)
        q = np.exp(rng.normal(size=(2, 120)))
        n = np.full(120, 10)
        n[[11, 24, 83]] = 0
        first = st.endpoint_bank(q, n, 0, 120, 9)
        q[:, n == 0] = 1e100
        second = st.endpoint_bank(q, n, 0, 120, 9)
        np.testing.assert_array_equal(first, second)

    def test_reference_tail_probability_literal(self):
        rng = np.random.default_rng(12)
        bank = rng.exponential(size=(st.REPS, 6))
        obs = np.array([2., 3., 1., 5., 6., 1.])
        for half in (False, True):
            n = st.BANK // 2 if half else st.BANK
            a, b = bank[:n], bank[st.BANK:st.BANK + n]
            center, scale = a.mean(axis=0), a.std(axis=0, ddof=1)
            t = max((obs - center) / scale)
            reference = ((b - center) / scale).max(axis=1)
            result = st.compatibility(obs, bank, half)
            self.assertEqual(result["p"], (1 + sum(x >= t for x in reference)) / (n + 1))
            self.assertEqual(result["compatible"], result["p"] > .05)
            self.assertGreater(result["p"], 0)

    def test_equal_ties_and_zero_scale(self):
        bank = np.zeros((st.REPS, 6))
        self.assertEqual(st.compatibility(np.zeros(6), bank)["p"], 1.)
        result = st.compatibility(np.ones(6), bank)
        self.assertEqual(result["p"], 1 / 65)
        self.assertTrue(result["observed_t_positive_infinity"])
        self.assertFalse(result["compatible"])

    def test_no_fake_lag1_for_constants(self):
        obs = np.zeros(6)
        obs[-1] = np.nan
        result = st.compatibility(obs, np.zeros((st.REPS, 6)))
        self.assertEqual(result["status"], "incomplete_endpoints")
        self.assertIsNone(result["compatible"])
        self.assertIsNone(result["p"])

    def test_library_start_count_and_gap_boundaries(self):
        q = 1 + .2 * np.sin(np.arange(3600) / 12)
        records = np.full(3600, 10, dtype=np.uint8)
        records[1300:1320] = 0
        h = dyn.Series("synthetic", "synthetic", "other", 0, q, records)
        library = dyn.block_library([h], 256)
        literal = sum((records[s:s + 256] >= 9).all() for s in range(3600 - 256 + 1))
        self.assertEqual(library[0]["n_starts"], literal)
        _, pieces = dyn.block_template(library, 256, np.random.default_rng(42))
        self.assertEqual(sum(x["length_used_s"] for x in pieces), 3600)
        for piece in pieces:
            self.assertTrue((records[piece["start_s"]:piece["start_s"] + 256] >= 9).all())

    def test_final_rank_reorder_and_repeatability(self):
        model = {"support_q": np.array([.5, 1.5]), "support_cdf": np.array([.5, 1.])}
        h = dyn.Series("synthetic", "synthetic", "other", 0,
                       1 + .2 * np.sin(np.arange(3600) / 12), np.full(3600, 10, dtype=np.uint8))
        library = dyn.block_library([h], 256)
        first, _ = dyn.generate("iid", model, None, -1, "test-case", 0)
        second, meta = dyn.generate("block_0256", model, library, -1, "test-case", 0)
        again, repeated = dyn.generate("block_0256", model, library, -1, "test-case", 0)
        np.testing.assert_array_equal(np.sort(first), np.sort(second))
        np.testing.assert_array_equal(second, again)
        self.assertEqual(meta, repeated)
        self.assertEqual(np.dot(np.diff(np.r_[0., model["support_cdf"]]), model["support_q"]), 1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NumericalChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        dest = OUT / "stationarity"
        dest.mkdir(exist_ok=True)
        qa = {"status": "PASS", "completed_utc": now(), "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "script_sha256": sha256(Path(st.__file__)), "builder_sha256": sha256(Path(fg.__file__)),
              "test_script_sha256": sha256(Path(__file__)),
              "plan_sha256": sha256(st.HERE / "STATIONARITY_PLAN.md"),
              "research_data_reads": 0, "final_validation_test_reads": 0}
        path = dest / "numerical_tests_qa.json"
        if path.exists():
            old = dyn.read_json(path)
            if any(old.get(k) != v for k, v in qa.items() if k != "completed_utc"):
                raise RuntimeError("Existing numerical QA differs; do not silently overwrite frozen code")
        else:
            write_json(path, qa)
    raise SystemExit(0 if result.wasSuccessful() else 1)
