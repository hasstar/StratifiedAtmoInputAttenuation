"""Offline positive/negative controls for the independent GBS actual-field audit."""
import json
import math
import unittest
from pathlib import Path
import numpy as np
import audit_gbs_48h_actual_fields_v001 as a

ASCII = '''Dataset {
 Grid {
 ARRAY:
 Float32 O3[time = 2][level = 1][latitude = 1][longitude = 1];
 MAPS:
 Int32 time[time = 2];
 Float64 level[level = 1];
 Float64 latitude[latitude = 1];
 Float64 longitude[longitude = 1];
 } O3;
} files/g/d633000/e5.oper.an.pl/200402/e5.oper.an.pl.128_203_o3.ll025sc.2004021900_2004021923.nc;
---------------------------------------------
O3.O3[2][1][1][1]
[0][0][0], 1.1E-5
[1][0][0], 1.2E-5
O3.time[2]
912816, 912817
O3.level[1]
5.0
O3.latitude[1]
50.5
O3.longitude[1]
358.25
'''
DAS = '''Attributes {
    O3 {
        String units "kg kg**-1";
        String short_name "o3";
        Int32 ecmwf_parameter 203;
        Float32 missing_value 9.999E20;
        String rda_dataset "ds633.0";
    }
    level {
        String units "hPa";
    }
    latitude {
        String units "degrees_north";
    }
    longitude {
        String units "degrees_east";
    }
    time {
        String units "hours since 1900-01-01 00:00:00";
        String calendar "gregorian";
    }
}
'''


class Controls(unittest.TestCase):
    def parse(self, text=ASCII, das=DAS):
        return a.parse_ascii(text, das, 'ozone_mass_mixing_ratio')

    def test_valid_ascii(self):
        data, ev = self.parse()
        self.assertEqual(len(data), 2)
        self.assertIn(('ozone_mass_mixing_ratio', 5, 1077148800, 202, -7), data)
        self.assertEqual(ev['actual_decoded_values'], 2)

    def test_parameter_id_rejected(self):
        with self.assertRaises(ValueError): self.parse(das=DAS.replace('203;', '131;'))

    def test_unit_rejected(self):
        with self.assertRaises(ValueError): self.parse(das=DAS.replace('kg kg**-1', 'K'))

    def test_pressure_units_rejected(self):
        with self.assertRaises(ValueError): self.parse(das=DAS.replace('"hPa"', '"Pa"'))

    def test_dataset_identity_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('128_203_o3', '128_131_u'))

    def test_missing_array_row_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('[1][0][0], 1.2E-5\n', ''))

    def test_duplicate_array_row_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('[1][0][0]', '[0][0][0]'))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('1.1E-5', 'NaN'))

    def test_missing_sentinel_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('1.1E-5', '9.999E20'))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('O3.O3[2]', 'O3.O3[3]'))

    def test_duplicate_times_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('912817', '912816'))

    def test_non_grid_coordinate_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('50.5\n', '50.51\n'))

    def test_unexpected_layer_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII.replace('5.0\n', '4.0\n'))

    def test_extra_payload_rejected(self):
        with self.assertRaises(ValueError): self.parse(text=ASCII + 'extra\n')

    def test_wrong_utc_fails_required_identity(self):
        actual, _ = self.parse(text=ASCII.replace('912816, 912817', '912818, 912819'))
        wanted, _ = self.parse()
        with self.assertRaises(ValueError): a.compare_required(actual, wanted)

    def test_wrong_grid_fails_required_identity(self):
        actual, _ = self.parse(text=ASCII.replace('50.5\n', '50.75\n'))
        wanted, _ = self.parse()
        with self.assertRaises(ValueError): a.compare_required(actual, wanted)

    def test_wrong_value_rejected(self):
        actual, _ = self.parse(text=ASCII.replace('1.1E-5', '1.7E-5'))
        wanted, _ = self.parse()
        with self.assertRaises(ValueError): a.compare_required(actual, wanted)

    def test_exact_value_pass(self):
        actual, _ = self.parse()
        self.assertEqual(a.compare_required(actual, dict(actual)), 0)

    def test_nearest_ties_north_east(self):
        self.assertEqual(a.nearest(50.625, -1.875), [203, -7])

    def test_longitude_wrap(self):
        self.assertEqual(a.exact_grid([358.25], True), [-7])
        self.assertEqual(a.nearest(0, 180), [0, -720])

    def test_fractional_hour_rejected(self):
        with self.assertRaises(ValueError): a.utc_seconds([0.5], 'hours since 1900-01-01', 'gregorian')

    def test_geometry_independent_ray(self):
        o = np.array(a.TO_XYZ.transform(0., 0., 100.))
        r = a.intersection(o, np.array([1., 0., 0.]), 1000.)
        self.assertAlmostEqual(r['slant'], 900., places=5)
        self.assertEqual(r['cell'], [0, 0])

    def test_layer_below_antenna_rejected(self):
        o = np.array(a.TO_XYZ.transform(0., 0., 100.))
        with self.assertRaises(ValueError): a.intersection(o, np.array([1., 0., 0.]), 50.)

    def test_nonunit_ray_rejected(self):
        o = np.array(a.TO_XYZ.transform(0., 0., 100.))
        with self.assertRaises(ValueError): a.intersection(o, np.array([2., 0., 0.]), 1000.)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
              'failures': [str(x) for x in result.failures], 'errors': [str(x) for x in result.errors],
              'audit_code_sha256': a.sha(a.__file__), 'test_code_sha256': a.sha(__file__),
              'new_fit_calls': 0, 'predictions_executed': 0, 'network_calls': 0}
    a.emit('negative_controls.json', report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
