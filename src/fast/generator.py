"""Weather-conditioned marginal fitting, date-fold selection and frozen generation."""
from __future__ import annotations
import argparse,hashlib,json,sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s
import numpy as np
import pandas as pd

def modules():
    directory=s.original('Code/Fast_Fading_Experiments/Revision_002')
    sys.path.insert(0,str(directory))
    dyn=s.adapt_paths(s.load_module('run_dynamics',directory/'run_dynamics.py'))
    import freeze_generator as fg
    s.adapt_paths(fg)
    return dyn,dyn.fm,fg

def load_frozen():
    dyn,fm,fg=modules();saved,models,pools=fg.load_package()
    libraries={w:dyn.block_library(v,s.config()['fast']['block_seconds'])for w,v in pools.items()}
    return dyn,fm,models,libraries,saved

def load_new(folder):
    dyn,fm,_=modules();saved=s.read(folder/'generator.json');models={};libraries={}
    for event,info in saved['components'].items():
        with np.load(folder/info['support_file'])as z:models[event]=dict(family='empirical',support_q=z['q'],support_cdf=z['cdf'])
        with np.load(folder/info['library_file'])as z:
            pool=[dyn.Series(str(h),str(h)[:10],event,-1,q,n)for h,q,n in zip(z['hours'],z['q'],z['records'])]
        libraries[event]=dyn.block_library(pool,saved['block_seconds'])
    return dyn,fm,models,libraries,saved

def fit(out,with_cv=True):
    dyn,fm,_=modules();cfg=s.config()['fast'];reference=s.original('Outputs/Experiments/Fast_Fading_Revision_002')
    design=s.read(reference/'dynamics/design_manifest.json')['frozen_design']
    series=dyn.load_series(design)
    assert len(series)==2404 and dyn.SEED==s.config()['historical_seeds']['fast_dynamics_namespace']
    out.mkdir(parents=True,exist_ok=False)
    s.write(out/'configuration.json',dict(config=cfg,source_design=design,training_hours=[h.hour for h in series],
              validation_test_used=False,cv_recomputed=with_cv))
    winner='block_%04d'%cfg['block_seconds']
    if with_cv:
        dyn.DYN=out/'cross_validation';dyn.DYN.mkdir();(dyn.DYN/'folds').mkdir()
        trials=dyn.DYN/f'r{cfg["selection_replicates"]:02d}';trials.mkdir()
        signature=s.sha(out/'configuration.json')
        for fold in range(cfg['group_folds']):
            models,pools=dyn.fit_context(fold,series,signature)
            held=[h for h in series if h.fold==fold]
            for candidate in cfg['candidates']:
                dyn.candidate_fold(candidate,fold,cfg['selection_replicates'],models,pools,held,signature)
        all_scores=pd.concat([pd.read_parquet(p)for p in (trials/'chunks').rglob('*_scores.parquet')],ignore_index=True)
        all_scores.to_parquet(out/'cv_hour_scores.parquet',index=False)
        date_scores=all_scores.groupby(['candidate','day'],as_index=False)[['vs','vs_half']].mean()
        ranking=date_scores.groupby('candidate',as_index=False)[['vs','vs_half']].mean().sort_values(['vs','candidate'])
        ranking.to_csv(out/'cv_candidate_ranking.csv',index=False)
        winner=str(ranking.iloc[0].candidate)
        s.write(out/'selection.json',dict(winner=winner,half_budget_winner=str(ranking.sort_values(['vs_half','candidate']).iloc[0].candidate),
                   expected_reference_winner='block_0256',matches_reference=winner=='block_0256',selection_only=True))
        if not winner.startswith('block_'):
            raise RuntimeError('Recomputed selected family differs from the published block family. Inspect before combination; no forced substitution.')
    components={}
    for event in cfg['weather']:
        pool=[h for h in series if h.event==event]
        model=fm.fit('empirical',[h.as_marginal_hour()for h in pool])
        support=event+'_support.npz';library=event+'_train_series.npz'
        np.savez_compressed(out/support,q=model['support_q'],cdf=model['support_cdf'])
        np.savez_compressed(out/library,hours=np.array([h.hour for h in pool]),q=np.stack([h.q for h in pool]),records=np.stack([h.records for h in pool]))
        components[event]=dict(support_file=support,library_file=library,support_sha256=s.sha(out/support),
                 library_sha256=s.sha(out/library),hours=len(pool),training_mean_q=model['training_mean_q'],power_normalizer=model['power_normalizer'])
        print('FITTED_GENERATOR',event,len(pool),flush=True)
    s.write(out/'generator.json',dict(candidate=winner,block_seconds=int(winner.split('_')[1]),components=components,
                training_hours=2404,cv_recomputed=with_cv,primary_model_seed=17,
                stochastic_seed_namespace=dyn.SEED,amplitude_normalization='One common training-distribution power normalization; no per-realization normalization'))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--skip-cv',action='store_true',help='Refit the published empirical generator only; keep the previously selected 256-s length')
    a=ap.parse_args();fit(a.out,not a.skip_cv)
if __name__=='__main__':main()
