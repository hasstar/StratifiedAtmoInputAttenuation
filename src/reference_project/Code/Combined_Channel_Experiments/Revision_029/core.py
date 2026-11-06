"""Read-only reuse of the sealed Revision017 scalar-shift marginal scorer."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
LEGACY=ROOT/'Code/Combined_Channel_Experiments/Revision_017/common_fast.py'
spec=importlib.util.spec_from_file_location('frozen017_marginal',LEGACY)
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)  # no legacy entry point called
score=legacy.score
METRICS=tuple(legacy.METRICS)
FAMILIES={**legacy.FAMILIES,'esst':['esst_hourly_adaptation']}
RUNS={r:f for f,runs in FAMILIES.items() for r in runs}


def frozen_quantiles(q,cdf):
    q=np.asarray(q,dtype=float); cdf=np.asarray(cdf,dtype=float)
    if (q.ndim!=1 or q.shape!=cdf.shape or not len(q) or
        not np.isfinite(q).all() or not np.isfinite(cdf).all() or
        (q<=0).any() or (np.diff(q)<0).any() or
        (np.diff(np.r_[0.,cdf])<=0).any() or abs(cdf[-1]-1.)>1e-12 or
        abs(float(np.dot(q,np.diff(np.r_[0.,cdf])))-1.)>1e-10):
        raise ValueError('invalid frozen power CDF; never normalize or refit')
    def quant(p):
        return -10*np.log10(q[np.searchsorted(cdf,1-np.asarray(p),side='left')])
    return {201:quant((np.arange(201)+.5)/201),
            401:quant((np.arange(401)+.5)/401),
            'bounds':quant([.01,.05,.5,.95,.99])}
