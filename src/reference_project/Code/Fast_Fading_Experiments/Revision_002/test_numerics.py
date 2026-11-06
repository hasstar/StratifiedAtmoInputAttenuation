"""Known-power and missing-slot checks for the window experiment."""

import unittest

import numpy as np
import pandas as pd

from window_diagnostics import calculate_hour, longest_missing


def example(power):
    hour = pd.Timestamp("2026-05-14 12:00:00", tz="Asia/Shanghai")
    baseline = float(np.mean(power))
    return pd.DataFrame({
        "second_bjt": pd.date_range(hour, periods=3600, freq="s"),
        "hour_bjt": hour, "received_power_w": power, "hour_mean_power_w": baseline,
        "fast_residual_db": 10 * np.log10(baseline / power),
        "quality_zero_records": 10, "k11_records": 10, "k101_records": 0,
        "rain": True, "dry_overcast": False, "clear_dry": False,
        "weather_stratum": "rain_nonheavy",
    })


class WindowChecks(unittest.TestCase):
    def test_constant_complete_hour(self):
        rows = calculate_hour(example(np.full(3600, 2e-11)))
        self.assertEqual(len(rows), 460)
        for row in rows:
            self.assertAlmostEqual(row["half_power_drift_db"], 0, places=12)
            self.assertAlmostEqual(row["half_sd_difference_db"], 0, places=12)
            self.assertAlmostEqual(row["local_shape_wasserstein_db"], 0, places=12)
            self.assertEqual(row["equipment_epoch"], "transition_day")

    def test_known_factor_two_drift_is_not_shape_change(self):
        power = np.r_[np.full(1800, 2e-11), np.full(1800, 1e-11)]
        rows = [x for x in calculate_hour(example(power)) if x["window_minutes"] == 60]
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertAlmostEqual(row["half_power_drift_db"], 10 * np.log10(2), places=12)
            self.assertAlmostEqual(row["local_shape_wasserstein_db"], 0, places=12)

    def test_missing_edges_are_counted_and_never_interpolated(self):
        self.assertEqual(longest_missing(np.array([False, False, True, True, False])), 2)
        frame = example(np.full(3600, 2e-11))
        frame.loc[[0, 1, 2], "quality_zero_records"] = 8
        rows = [x for x in calculate_hour(frame) if x["window_minutes"] == 1 and x["slot"] == 0]
        self.assertEqual([x["quality_gate"] for x in rows], ["primary90_gap5"])
        self.assertEqual(rows[0]["n_seconds_first"], 27)
        self.assertEqual(rows[0]["longest_missing_first_s"], 3)


if __name__ == "__main__":
    unittest.main()
