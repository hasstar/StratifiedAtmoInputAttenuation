"""Synthetic controls only; no study labels or fitting."""
import unittest
import numpy as np
from core import score, frozen_quantiles


class Controls(unittest.TestCase):
    def setUp(self):
        self.y=np.array([-2.,0.,0.,3.]); self.w=np.array([1.,2.,4.,3.])
        self.model={n:np.linspace(-3,4,n) for n in (201,401)}
        self.model['bounds']=np.array([-2.,-1.,0.,2.,3.])

    def test_literal_pinball(self):
        got=score(self.y,self.w,self.model,.2)
        for n in (201,401):
            p=(np.arange(n)+.5)/n; diff=self.y[:,None]-self.model[n]-.2
            expected=2*np.average(np.maximum(p*diff,(p-1)*diff),axis=0,weights=self.w).mean()
            self.assertAlmostEqual(got[f'crps{n}_db'],expected,places=12)

    def test_translation(self):
        a=score(self.y,self.w,self.model,0); b=score(self.y+7,self.w,self.model,7)
        for k in a:self.assertAlmostEqual(a[k],b[k],places=12)

    def test_constant_distribution(self):
        m={n:np.zeros(n) for n in (201,401)}; m['bounds']=np.zeros(5)
        got=score(self.y,self.w,m,0)
        for n in (201,401):self.assertAlmostEqual(got[f'crps{n}_db'],np.average(abs(self.y),weights=self.w),places=12)

    def test_weight_scale(self):
        a=score(self.y,self.w,self.model,0); b=score(self.y,9*self.w,self.model,0)
        for k in a:self.assertAlmostEqual(a[k],b[k],places=12)

    def test_inclusive_interval_strict_tail(self):
        m={n:np.zeros(n) for n in (201,401)};m['bounds']=np.array([-2.,-2.,0.,3.,3.])
        got=score(self.y,self.w,m,0)
        self.assertEqual(got['central90_coverage'],1.)
        self.assertEqual(got['tail95_exceedance'],0.)

    def test_invalid_observation(self):
        for y in (np.array([np.nan]),np.array([np.inf])):
            with self.assertRaises(ValueError):score(y,np.ones(1),self.model,0)

    def test_invalid_weight(self):
        for w in (np.array([0.]),np.array([-1.]),np.array([np.nan])):
            with self.assertRaises(ValueError):score(np.ones(1),w,self.model,0)

    def test_invalid_shift(self):
        with self.assertRaises(ValueError):score(self.y,self.w,self.model,float('nan'))

    def test_frozen_cdf(self):
        m=frozen_quantiles(np.ones(2),np.array([.25,1.]))
        for v in m.values():self.assertTrue(np.array_equal(v,np.zeros_like(v)))

    def test_refuse_changed_moment(self):
        with self.assertRaises(ValueError):frozen_quantiles([1,2],[.5,1])

    def test_refuse_invalid_cdf(self):
        for q,c in [([0,1],[.5,1]),([1,1],[.5,.4]),([1,1],[0,1]),([1,1],[.5,.9])]:
            with self.assertRaises(ValueError):frozen_quantiles(q,c)

    def test_hour_constant_temporal_cancellation(self):
        x=np.array([-.7,.2,1.3,-.3,.5,.1,1.1]); keep=np.array([1,1,0,1,1,1,1],bool)
        for lag in (1,2):
            valid=np.array([keep[i:i+lag+1].all() for i in range(len(x)-lag)])
            a=x[:-lag][valid];b=x[lag:][valid]
            for shift in (-10.,-.2,0.,7.):
                np.testing.assert_allclose((b+shift)-(a+shift),b-a,rtol=0,atol=4e-15)
                if len(a)>=3:
                    self.assertAlmostEqual(np.corrcoef(a,b)[0,1],np.corrcoef(a+shift,b+shift)[0,1],places=12)


if __name__=='__main__':unittest.main()
