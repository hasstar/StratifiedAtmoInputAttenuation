"""Synthetic boundary controls, not model evaluation."""
import json
import sys
import unittest
from pathlib import Path
from inventory import valid_rain,rain_events,runs


def rows(values):
    return [{'utc_epoch_seconds':t*3600,'liquid_mm_per_h':v,'rain_valid':v>=0} for t,v in values]


class Checks(unittest.TestCase):
    def test_rate_conversion(self):self.assertEqual(valid_rain(.001,.00025),(2.7,True))
    def test_negative_not_clipped(self):self.assertEqual(valid_rain(0,.001),(-3.6,False))
    def test_both_nonnegative(self):self.assertFalse(valid_rain(-.001,-.002)[1])
    def test_complete(self):
        e,m=rain_events(rows([(0,0),(1,2),(2,1),(3,0)]))
        self.assertTrue(e[0]['two_sided_complete']);self.assertEqual(e[0]['hours'],2);self.assertEqual(len(m),2)
    def test_absent_boundary(self):
        e,_=rain_events(rows([(0,1),(1,0)]));self.assertEqual(e[0]['left'],'ABSENT');self.assertFalse(e[0]['two_sided_complete'])
    def test_invalid_not_dry(self):
        e,_=rain_events(rows([(0,-1),(1,1),(2,0)]));self.assertEqual(e[0]['left'],'INVALID_RAIN')
    def test_gap_splits(self):
        e,_=rain_events(rows([(0,0),(1,1),(3,1),(4,0)]));self.assertEqual(len(e),2);self.assertTrue(all(not x['two_sided_complete'] for x in e))
    def test_exact_zero(self):
        e,_=rain_events(rows([(0,0),(1,1e-15),(2,0)]));self.assertEqual(len(e),1)
    def test_dry_empty(self):self.assertEqual(rain_events(rows([(0,0)])),([],{}));self.assertEqual(runs([]),[])
    def test_segments(self):self.assertEqual([r['hours'] for r in runs([0,3600,10800])],[2,1])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:
            json.dump({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
                'failures':[str(x) for x in result.failures],'errors':[str(x) for x in result.errors]},f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
