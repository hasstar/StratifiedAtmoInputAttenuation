import json
import sys
import unittest
from pathlib import Path
import numpy as np
from sampling_support import deduplicate, physical_quality_mask, utc_milliseconds, within_hour_offsets


class SamplingTests(unittest.TestCase):
    def test_irregular_milliseconds_preserved(self):
        t,q=deduplicate(np.array([581,682,783]),np.array([0,16,32]))
        np.testing.assert_array_equal(t,[581,682,783])
        np.testing.assert_array_equal(within_hour_offsets(t,0),[581,682,783])
        np.testing.assert_array_equal(q,[0,16,32])

    def test_duplicate_or_and_sort(self):
        t,q=deduplicate(np.array([200,100,200,100,99]),np.array([16,0,32,8,0]))
        np.testing.assert_array_equal(t,[99,100,200])
        np.testing.assert_array_equal(q,[0,8,48])

    def test_physical_mask_keeps_16_32(self):
        np.testing.assert_array_equal(physical_quality_mask([0,1,2,4,8,16,32,48,49]),
            [True,False,False,False,False,True,True,True,False])

    def test_bjt_hour_rollover(self):
        t=utc_milliseconds(np.array([8*3600000-1,8*3600000,9*3600000-1]))
        np.testing.assert_array_equal(t,[-1,0,3599999])
        np.testing.assert_array_equal(within_hour_offsets(t[:1],-3600),[3599999])
        np.testing.assert_array_equal(within_hour_offsets(t[1:],0),[0,3599999])

    def test_empty(self):
        t,q=deduplicate(np.array([],dtype='i8'),np.array([],dtype='u2'))
        self.assertEqual(t.size,0);self.assertEqual(q.size,0)

    def test_single(self):
        t,q=deduplicate(np.array([101]),np.array([32]))
        self.assertEqual(t.tolist(),[101]);self.assertEqual(q.tolist(),[32])

    def test_shape_rejection(self):
        with self.assertRaises(ValueError):deduplicate(np.array([1,2]),np.array([0]))
        with self.assertRaises(ValueError):deduplicate(np.array([[1]]),np.array([[0]]))

    def test_noninteger_rejection(self):
        with self.assertRaises(TypeError):utc_milliseconds(np.array([0.1]))
        with self.assertRaises(TypeError):deduplicate(np.array([1.1]),np.array([0]))

    def test_quality_range_rejection(self):
        for q in [-1,65536]:
            with self.assertRaises(ValueError):deduplicate(np.array([1]),np.array([q]))

    def test_wrong_hour_and_order_rejection(self):
        for t in [[-1],[3600000],[2,1],[1,1]]:
            with self.assertRaises(ValueError):within_hour_offsets(np.array(t),0)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests))
    if len(sys.argv)==2:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:
            json.dump({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
                'failures':[(str(t),s) for t,s in result.failures],
                'errors':[(str(t),s) for t,s in result.errors]},f,indent=2)
    sys.exit(0 if result.wasSuccessful() else 1)
