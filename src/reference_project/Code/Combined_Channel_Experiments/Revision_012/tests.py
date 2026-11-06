"""Synthetic arithmetic/identity contract tests; no scientific runs or weights."""
import json
import math
import unittest
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from audit import score, unique, FAMILIES, BLOCKED, MODELS, OUT, offline

class Contracts(unittest.TestCase):
    def test_identity(self): self.assertEqual(score([1,2],[1,2])['rmse_db'],0)
    def test_signed_bias(self): self.assertEqual(score([0,0],[-2,1])['bias_db'],-.5)
    def test_rmse(self): self.assertAlmostEqual(score([0,0],[3,4])['rmse_db'],math.sqrt(12.5))
    def test_mae(self): self.assertEqual(score([0,0],[-3,4])['mae_db'],3.5)
    def test_empty(self):
        with self.assertRaises(ValueError): score([],[])
    def test_unequal(self):
        with self.assertRaises(ValueError): score([0],[0,1])
    def test_nan_target(self):
        with self.assertRaises(ValueError): score([np.nan],[1])
    def test_nan_prediction(self):
        with self.assertRaises(ValueError): score([1],[np.nan])
    def test_infinity(self):
        with self.assertRaises(ValueError): score([1],[np.inf])
    def test_matrix(self):
        with self.assertRaises(ValueError): score([[1]],[[1]])
    def test_duplicate(self):
        with self.assertRaises(ValueError): unique(pd.DataFrame({'station':['a','a'],'t':[1,1]}),['station','t'])
    def test_distinct_station(self): unique(pd.DataFrame({'station':['a','b'],'t':[1,1]}),['station','t'])
    def test_roster(self):
        self.assertEqual(len(FAMILIES)+len(BLOCKED),7)
        self.assertEqual(len(MODELS),8)
        self.assertFalse(any('mlp' in m or 'tabpfn' in m or 'tabicl' in m for m in MODELS))
    def test_seed_policy(self):
        for name in ('bai2022_lstm_t0_adapted','tabm2025_official_budgeted'):
            self.assertEqual([m.rsplit('seed',1)[1] for m in FAMILIES[name][1]],['17','29','43'])

if __name__=='__main__':
    offline()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    with (OUT/'test_results.json').open('x',encoding='utf-8') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'success':result.wasSuccessful(),
                   'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                   'data':'Synthetic fixtures only; no fitting/inference/network.'},f,indent=2)
    if not result.wasSuccessful(): raise SystemExit(1)
