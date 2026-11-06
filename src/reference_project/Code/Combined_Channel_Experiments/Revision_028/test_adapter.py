"""Deterministic synthetic boundaries; no site data or fitted parameters."""
import math
import unittest
import numpy as np
from adapter import gate, record_attenuations, power_mean_db


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(height_km=2., wind_mps=10., elevation_deg=45., k=.1, alpha=1.2)
        self.offsets = np.array([0, 103, 900001, 3599999], dtype=np.int64)

    def contract(self, **changes):
        args = dict(candidates_m=[2000.], up_crossings=0, exact_zeros=0, bad_pairs=0,
                    wind_mps=10., elevation_deg=45., rain=[0.,0.,0.], event_ids=[-1,-1,-1], events=[], anchor=0)
        args.update(changes)
        return gate(**args)

    def test_constant(self):
        a = record_attenuations([4.,4.], self.offsets, **self.p)
        np.testing.assert_allclose(a, .1*4**1.2*2/math.sin(math.pi/4), rtol=0, atol=1e-12)

    def test_dry(self):
        a = record_attenuations([0.,0.], self.offsets, **self.p)
        self.assertTrue((a == 0).all()); self.assertEqual(power_mean_db(a), 0)

    def test_boundary_overlap(self):
        p = dict(self.p, alpha=1.)
        a = record_attenuations([0.,10.], np.array([3400000,3500000]), **p)
        np.testing.assert_allclose(a, [0., math.sqrt(2)], atol=1e-12)

    def test_power_not_db_mean(self):
        self.assertAlmostEqual(power_mean_db([0.,10.]), -10*math.log10(.55), places=13)
        self.assertLess(power_mean_db([0.,10.]), 5.)

    def test_large_loss_stability(self):
        self.assertAlmostEqual(power_mean_db([10000.,10010.]), 10000-10*math.log10(.55), places=10)

    def test_zero_height(self):
        self.assertTrue((record_attenuations([4.],self.offsets,**dict(self.p,height_km=0)) == 0).all())

    def test_vertical(self):
        a = record_attenuations([4.],self.offsets,**dict(self.p,elevation_deg=90))
        np.testing.assert_allclose(a,.1*4**1.2*2)

    def test_negative_and_nan_rain(self):
        for r in ([-1e-20,0.], [np.nan,0.]):
            with self.assertRaises(ValueError): record_attenuations(r,self.offsets,**self.p)

    def test_full_hour_context(self):
        with self.assertRaises(ValueError): record_attenuations([0.],np.array([0]),**self.p)

    def test_timestamp_contract(self):
        for t in (np.array([]), np.array([0.,1.]), np.array([0,0]), np.array([1,0]),np.array([-1]),np.array([3600000])):
            with self.assertRaises(ValueError): record_attenuations([0.,0.],t,**self.p)

    def test_parameter_contract(self):
        for change in ({'wind_mps':0},{'height_km':-1},{'elevation_deg':0},{'elevation_deg':91},{'k':-1},{'alpha':0}):
            with self.assertRaises(ValueError): record_attenuations([0.,0.],self.offsets,**dict(self.p,**change))

    def test_empty_mean(self):
        for a in ([], [np.nan]):
            with self.assertRaises(ValueError): power_mean_db(a)

    def test_dry_gate(self):
        g = self.contract(); self.assertEqual(g['regime'],'dry_window'); self.assertEqual(g['reasons'],[])

    def test_height_gates(self):
        for change, reason in [({'candidates_m':[]},'ZERO_LEVEL_NO_DOWN_CROSSING'),
            ({'candidates_m':[1000.,2000.]},'ZERO_LEVEL_MULTIPLE_CROSSINGS'),
            ({'up_crossings':1},'ZERO_LEVEL_MULTIPLE_CROSSINGS'),
            ({'exact_zeros':1},'ZERO_LEVEL_EXACT_ZERO_AMBIGUITY'),
            ({'bad_pairs':1},'ZERO_LEVEL_NONASCENDING_PROFILE'),
            ({'candidates_m':[-300.]},'EQUIVALENT_LAYER_BELOW_ANTENNA')]:
            self.assertIn(reason,self.contract(**change)['reasons'])

    def test_wind_gate(self):
        self.assertIn('WIND_INVALID',self.contract(wind_mps=0)['reasons'])

    def test_missing_context_gate(self):
        self.assertIn('MISSING_FORWARD_CONTEXT',self.contract(anchor=2)['reasons'])

    def test_invalid_context_gate(self):
        self.assertIn('INVALID_LIQUID_RAIN_CONTEXT',self.contract(rain=[0.,np.nan,0.])['reasons'])

    def test_censored_gate(self):
        g = self.contract(rain=[1.,0.,0.],event_ids=[0,-1,-1],events=[{'complete':False,'hourly_peak_type_candidate':'stratiform'}])
        self.assertIn('CENSORED_WET_EVENT',g['reasons'])

    def test_mixed_gate(self):
        es = [{'complete':True,'hourly_peak_type_candidate':x} for x in ['stratiform','convective']]
        g = self.contract(rain=[1.,20.,0.],event_ids=[0,1,-1],events=es)
        self.assertIn('MIXED_RAIN_REGIMES',g['reasons'])

    def test_convective_height(self):
        g = self.contract(rain=[20.,0.,0.],event_ids=[0,-1,-1],events=[{'complete':True,'hourly_peak_type_candidate':'convective'}])
        self.assertAlmostEqual(g['height_km'],1.8825)
        self.assertLess(g['used_lookahead_s'],g['conservative_lookahead_s'])


if __name__ == '__main__': unittest.main()
