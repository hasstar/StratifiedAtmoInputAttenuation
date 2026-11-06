"""Verify the included scientific inputs and freeze the run's input identities.

This stage starts from the delivered quality-screened measurements and ERA5
exports. It does not silently reclassify hours or redo historical calibration.
"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    source=s.original('Outputs/Datasets/Signal_Detectability_Revision_003');manifest=s.read(source/'manifest.json')
    checks=[];sets={}
    for split,n in s.config()['hourly']['splits'].items():
        p=source/(split+'.parquet');assert s.sha(p)==manifest['output_sha256'][p.name]
        df=pd.read_parquet(p);assert len(df)==n and df.hour_bjt.is_unique
        assert np.isfinite(df.clear_sky_attenuation_db).all();sets[split]=set(df.hour_bjt.astype(str))
        checks.append(dict(input=s.rel(p),sha256=s.sha(p),hours=n))
    assert sets['train'].isdisjoint(sets['test']) and sets['train'].isdisjoint(sets['validation']) and sets['test'].isdisjoint(sets['validation'])
    feature_source=s.original('Outputs/Experiments/Hourly_Attenuation_Revision_023_Physical_Decorrelation/feature_selection.json')
    features=s.read(feature_source)['feature_names_retained'];assert len(features)==156
    assert sum(x.startswith('surface__')for x in features)==50
    uk=s.original('Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE/inputs/uk_inputs_38.parquet')
    df=pd.read_parquet(uk);assert len(df)==5664 and int(df.primary_score_eligible.sum())==5593
    daily=s.read(s.original('Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json'))
    for row in daily:
        rel=row['source'].replace('\\','/').split('GBS_20p7GHz/')[-1];p=s.path('ceda_raw')/rel
        assert s.sha(p)==row['sha256']
    assert len(daily)==236
    s.write(a.out/'input_qa.json',dict(status='PASS',local=checks,uk_input=s.rel(uk),uk_sha256=s.sha(uk),uk_valid_hours=5593,
        ceda_days_verified=236,full_candidates=156,base_candidates=50,split_seed=manifest['random_seed'],
        model_seed=17,split_method=manifest['split_method'],scope='Input identity, row keys, target finiteness and raw CEDA checks; original ERA5 field QA is separately preserved.',
        temporal_independence_claimed=False))
    print('INPUTS_VERIFIED: 3434 local hours, 5593 eligible UK station-hours, 236 raw CEDA days',flush=True)
if __name__=='__main__':main()
