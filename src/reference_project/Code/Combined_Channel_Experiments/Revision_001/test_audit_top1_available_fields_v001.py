"""Offline positive/negative checks for Top1 field identity and exact axes."""
import unittest
from contextlib import contextmanager

import netCDF4
import numpy as np

import audit_top1_available_fields_v001 as audit

REQUEST = {"variable": ["geopotential"], "pressure_level": ["1", "5"],
           "year": ["2004"], "month": ["02"], "day": ["18"],
           "time": ["00:00", "01:00"], "area": [51.25, -1.5, 51.0, -1.5]}


@contextmanager
def synthetic():
    ds = netCDF4.Dataset("top1_unit_test.nc", "w", diskless=True, persist=False)
    try:
        for key, n in (("valid_time", 2), ("pressure_level", 2), ("latitude", 2), ("longitude", 1)):
            ds.createDimension(key, n)
        tv = ds.createVariable("valid_time", "f8", ("valid_time",))
        tv.units = "hours since 2004-02-18 00:00:00"
        tv.standard_name = "time"
        tv.calendar = "proleptic_gregorian"
        tv[:] = [0, 1]
        for key, values, unit in (("pressure_level", [5, 1], "hPa"),
                                   ("latitude", [51.25, 51], "degrees_north"),
                                   ("longitude", [-1.5], "degrees_east")):
            var = ds.createVariable(key, "f8", (key,))
            var.units = unit
            var[:] = values
        z = ds.createVariable("z", "f8", ("valid_time", "pressure_level", "latitude", "longitude"))
        z.units = "m**2 s**-2"
        z.standard_name = "geopotential"
        z.GRIB_paramId = 129
        z.GRIB_stepType = "instant"
        z[:] = np.arange(8).reshape(2, 2, 2, 1) + 400000
        yield ds
    finally:
        ds.close()


class FieldTests(unittest.TestCase):
    def test_selects_one_hpa_by_coordinate_not_first_index(self):
        with synthetic() as ds:
            row = audit.inspect_dataset(ds, REQUEST)
            self.assertEqual(row["pressure_index_in_payload"], 1)
            self.assertEqual(row["minimum"], 400002)
            self.assertEqual(row["grid_values"], 4)

    def test_rejects_height_metres_instead_of_geopotential(self):
        with synthetic() as ds:
            ds["z"].units = "m"
            with self.assertRaisesRegex(ValueError, "unit"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_pressure_unit_pa(self):
        with synthetic() as ds:
            ds["pressure_level"].units = "Pa"
            with self.assertRaisesRegex(ValueError, "Pressure unit"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_utc_shift(self):
        with synthetic() as ds:
            ds["valid_time"][:] = [8, 9]
            with self.assertRaisesRegex(ValueError, "UTC"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_duplicate_hour(self):
        with synthetic() as ds:
            ds["valid_time"][:] = [0, 0]
            with self.assertRaisesRegex(ValueError, "UTC"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_wrong_coordinate(self):
        with synthetic() as ds:
            ds["longitude"][:] = [-1.25]
            with self.assertRaisesRegex(ValueError, "Coordinate values"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_missing_one_hpa(self):
        with synthetic() as ds:
            ds["pressure_level"][:] = [5, 2]
            with self.assertRaisesRegex(ValueError, "levels"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_wrong_identity(self):
        with synthetic() as ds:
            ds["z"].standard_name = "geopotential_height"
            with self.assertRaisesRegex(ValueError, "Not geopotential"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_noninstant_field(self):
        with synthetic() as ds:
            ds["z"].GRIB_stepType = "accum"
            with self.assertRaisesRegex(ValueError, "instantaneous"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_nan(self):
        with synthetic() as ds:
            ds["z"][0, 1, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_mask(self):
        with synthetic() as ds:
            ds["z"][0, 1, 0, 0] = netCDF4.default_fillvals["f8"]
            with self.assertRaisesRegex(ValueError, "Missing"):
                audit.inspect_dataset(ds, REQUEST)

    def test_rejects_archive_escape(self):
        for name in ("../x.nc", "/x.nc", "C:/x.nc", "a/../../x.nc"):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                audit.safe_member(name)
        audit.safe_member("inside/x.nc")


if __name__ == "__main__":
    unittest.main()
