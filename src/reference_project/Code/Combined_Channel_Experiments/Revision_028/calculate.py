"""One-shot approved LOCAL calculation, frozen before outcomes are loaded."""
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from adapter import gate, record_attenuations, power_mean_db
from test_adapter import AdapterTests

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_028'
EXP = ROOT/'Outputs/Experiments'
ESST = 'esst_hourly_adaptation'
FAMILIES = {
    'ours_frozen_top10':['ours_frozen_top10'],
    'bai2022_lstm_t0_adapted':[f'bai2022_lstm_t0_adapted_seed{s}' for s in (17,29,43)],
    'tabm2025_official_budgeted':[f'tabm2025_official_budgeted_seed{s}' for s in (17,29,43)],
    'itu_combined_geometric_approx':['itu_combined_geometric_approx'],
    ESST:[ESST],
}
EVENTS = ('rain','dry_overcast')


def now(): return datetime.now(timezone.utc).isoformat()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def emit(name, data):
    with (OUT/name).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write('\n')


def epoch(s): return int(pd.Timestamp(s).timestamp())


def iso(s): return datetime.fromtimestamp(int(s),timezone.utc).isoformat()


def load_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def guard():
    previous=EXP/'Combined_Channel_Revision_027'
    hashes=read(previous/'sources_before.json')
    assert sha(previous/'delivery_manifest.json')==read(previous/'final_verification.json')['manifest_sha256']
    for rel,item in read(previous/'delivery_manifest.json')['files'].items():
        assert sha(ROOT/rel)==item['sha256'],rel
    for directory in (previous,ROOT/'Code/Combined_Channel_Experiments/Revision_027'):
        for p in directory.rglob('*'):
            if p.is_file(): hashes[str(p)]=sha(p)
    required=[EXP/f'Combined_Channel_Revision_{v:03d}'/name for v,name in [
        (21,'zero_crossing_diagnostics.npz'),(21,'rain_context_ledger.json'),(21,'rain_events.json'),
        (22,'derived_inputs.npz'),(22,'coefficient_parameters.json'),(23,'sampling_support.npz'),
        (23,'ADAPTATION_PROPOSAL.md'),(11,'fixed_reference_parameters.json')]]
    for p in required:
        existing=next((h for q,h in hashes.items() if Path(q).resolve()==p.resolve()),None)
        assert existing is not None and sha(p)==existing,str(p)
    assert all(sha(p)==h for p,h in hashes.items()),'frozen inputs changed'
    emit('sources_before.json',hashes)
    emit('code_before.json',{str(p.relative_to(ROOT)):sha(p) for p in CODE.iterdir() if p.is_file()})
    return hashes


def metrics(y,p):
    assert y.shape==p.shape and len(y) and np.isfinite(y).all() and np.isfinite(p).all()
    e=p-y
    return {'rmse_db':float(np.sqrt(np.mean(e*e))), 'mae_db':float(np.mean(np.abs(e))), 'bias_db':float(np.mean(e))}


def main():
    emit('execution.json',dict(utc=now(),pid=os.getpid(),argv=sys.argv,executable=sys.executable,
        python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
        pyarrow=pyarrow.__version__,mode='approved_LOCAL_hourly_engineering_adaptation',network=False,
        new_training=0,weather_requests=0,logical_lock='LOCKED'))
    hashes=guard()
    proposal=EXP/'Combined_Channel_Revision_023/ADAPTATION_PROPOSAL.md'
    emit('approval.json',dict(recorded_utc=now(),answer='接受该工程适配，继续测试和计算',
        thread='01a0bda1-92d1-7832-9508-60f4e0ec1634',
        question_item_id='["request_user_input_async","call_786d30ae00d041e08ba941cf14c355bf",0]',
        proposal_path=str(proposal),proposal_sha256=sha(proposal),approved=True,
        common_vertical_datum='accepted engineering approximation, not measured equivalence',
        scope='LOCAL only; UK weather, RMOP/PRM gates and MTS user decision unchanged'))
    with (OUT/'PROTOCOL.md').open('x',encoding='utf-8') as f: f.write((CODE/'PROTOCOL.md').read_text(encoding='utf-8'))
    log=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests)
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    emit('unit_tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),log=log.getvalue()))
    assert result.wasSuccessful(),log.getvalue()
    spec=importlib.util.spec_from_file_location('frozen020',ROOT/'Code/Combined_Channel_Experiments/Revision_020/esst_hourly_kernel.py')
    legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    cross=load_npz(EXP/'Combined_Channel_Revision_021/zero_crossing_diagnostics.npz')
    inputs=load_npz(EXP/'Combined_Channel_Revision_022/derived_inputs.npz')
    sampling=load_npz(EXP/'Combined_Channel_Revision_023/sampling_support.npz')
    context=read(EXP/'Combined_Channel_Revision_021/rain_context_ledger.json')
    events=read(EXP/'Combined_Channel_Revision_021/rain_events.json')
    ctime=np.array([epoch(r['time_utc']) for r in context],dtype=np.int64)
    rain=np.array([np.nan if r['site_rain_mm_per_h'] is None else r['site_rain_mm_per_h'] for r in context])
    event_ids=np.array([r['event_id'] for r in context])
    assert len(ctime)==7823 and np.all(np.diff(ctime)==3600)
    assert all(epoch(r['rate_valid_end_utc'])==int(t)+3600 for r,t in zip(context,ctime))
    assert np.array_equal(cross['campaign_epoch_s'],inputs['campaign_epoch_s'])
    assert np.array_equal(cross['test_member'],inputs['test_member'])
    keys=sampling['hour_epoch_s']; assert len(keys)==687 and np.all(np.diff(keys)>0)
    assert np.array_equal(keys,inputs['campaign_epoch_s'][inputs['test_member']])
    assert (sampling['quality'] & 15 == 0).all()
    assert len(sampling['offset_ms'])==24196167 and sampling['indptr'][-1]==len(sampling['offset_ms'])
    coeff=next(r for r in read(EXP/'Combined_Channel_Revision_022/coefficient_parameters.json') if r['site']=='LOCAL')
    assert coeff['frequency_ghz']==19.13 and coeff['polarization']=='circular'
    angle,k,alpha=(coeff[n] for n in ('elevation_deg','k','alpha'))
    parameters=dict(frequency_ghz=19.13,polarization='circular',elevation_deg=angle,k=k,alpha=alpha,
        antenna_height_m_amsl=117.5,earth_radius_m=6371229.,time_step_seconds=3600,
        stratiform_layer_addition_m=360.,convective_peak_strictly_greater_mmph=10.,
        parameter_selection='frozen approved definitions, not label-dependent')
    emit('parameters.json',parameters)
    # Only components, keys and historical event classification; no truth or other predictions.
    cols=['hour_bjt','event','common_score_eligible','gas_absolute_db','gas_reference_db','gas_relative_db','cloud_db']
    base=pd.read_parquet(EXP/'Combined_Channel_Revision_011/predictions_local.parquet',columns=cols)
    assert base.hour_bjt.is_unique and base.hour_bjt.dt.tz is not None and len(base)==687
    base['epoch_s']=base.hour_bjt.astype('int64')//10**9
    base=base.set_index('epoch_s').loc[keys].reset_index()
    assert base.common_score_eligible.all()
    ref=read(EXP/'Combined_Channel_Revision_011/fixed_reference_parameters.json')['LOCAL']
    assert ref['height_km']==0.0 and ref['frequency_GHz']==19.13 and ref['elevation_deg']==angle
    assert np.array_equal(base.gas_reference_db.to_numpy(),np.full(687,ref['gas_absolute_db']))
    assert np.array_equal(base.gas_relative_db.to_numpy(),(base.gas_absolute_db-base.gas_reference_db).to_numpy())
    campaign_index=np.searchsorted(inputs['campaign_epoch_s'],keys)
    wind_index=np.searchsorted(inputs['raw_epoch_s'],keys)
    assert np.array_equal(inputs['raw_epoch_s'][wind_index],keys)
    context_index=np.searchsorted(ctime,keys); assert np.array_equal(ctime[context_index],keys)
    ledger=[]
    for i,t in enumerate(keys):
        j,w,a=int(campaign_index[i]),int(wind_index[i]),int(context_index[i])
        candidates=inputs['candidate_geometric_m'][j,4]
        candidates=candidates[np.isfinite(candidates)]
        assert len(candidates)==int(cross['warm_to_cold_count'][j,4])
        g=gate(candidates,int(cross['cold_to_warm_count'][j,4]),int(cross['exact_zero_count'][j,4]),
            int(cross['nonascending_pair_count'][j,4]),float(inputs['wind700_speed_site_mps'][w]),
            angle,rain,event_ids,events,a)
        comp=base.iloc[i]
        if not np.isfinite([comp.gas_relative_db,comp.cloud_db]).all() or comp.cloud_db<0:
            g['reasons'].append('FROZEN_GAS_CLOUD_INVALID')
        lo,hi=map(int,sampling['indptr'][i:i+2])
        if hi<=lo: g['reasons'].append('NO_VALID_SAMPLING_SUPPORT')
        ledger.append(dict(hour_epoch_s=int(t),hour_utc=iso(t),date_bjt=comp.hour_bjt.date().isoformat(),
            event=comp.event,station='LOCAL',candidate_geometric_m=candidates.tolist(),
            up_crossings=int(cross['cold_to_warm_count'][j,4]),exact_zero_count=int(cross['exact_zero_count'][j,4]),
            wind700_mps=float(inputs['wind700_speed_site_mps'][w]),context_start_index=a,
            valid_record_count=hi-lo,eligible=not g['reasons'],**g))
    emit('eligibility_ledger.json',ledger)
    mask=np.array([r['eligible'] for r in ledger])
    emit('mask_frozen_before_predictions.json',dict(utc=now(),mother_hours=687,eligible_hours=int(mask.sum()),
        ledger_sha256=sha(OUT/'eligibility_ledger.json'),truth_loaded=False,old_predictions_loaded=False,
        reasons=dict(Counter(x for r in ledger for x in r['reasons'])),protocol_sha256=sha(CODE/'PROTOCOL.md')))
    print(json.dumps({'phase':'MASK_FROZEN','eligible':int(mask.sum()),'reasons':dict(Counter(x for r in ledger for x in r['reasons']))}),flush=True)
    predictions=[]; max_anchor=0.; sample_count=0
    for i,row in enumerate(ledger):
        p=dict(hour_epoch_s=row['hour_epoch_s'],hour_utc=row['hour_utc'],eligible=row['eligible'],
            rain_power_mean_db=None,rain_arithmetic_mean_db=None,rain_anchor_db=None,
            rain_min_db=None,rain_max_db=None,total_relative_db=None,
            gas_relative_db=float(base.iloc[i].gas_relative_db),cloud_db=float(base.iloc[i].cloud_db))
        if row['eligible']:
            a,n=row['context_start_index'],row['context_bins']; rates=rain[a:a+n]
            lo,hi=map(int,sampling['indptr'][i:i+2]); offsets=sampling['offset_ms'][lo:hi]
            arguments=dict(height_km=row['height_km'],wind_mps=row['wind700_mps'],elevation_deg=angle,k=k,alpha=alpha)
            losses=record_attenuations(rates,offsets,**arguments)
            anchor=float(record_attenuations(rates,np.array([0],dtype=np.int64),**arguments)[0])
            oldanchor=legacy.attenuation_at_bin_start(rates.tolist(),list(range(0,n*3600,3600)),0,
                wind_speed=arguments['wind_mps'],height_above_antenna=arguments['height_km'],
                elevation=angle,k=k,alpha=alpha,timestep_seconds=3600,rain_unit='mm/h',
                wind_unit='m/s',height_unit='km_above_antenna',angle_unit='degree')
            max_anchor=max(max_anchor,abs(anchor-oldanchor)); assert abs(anchor-oldanchor)<2e-10
            rain_mean=power_mean_db(losses)
            assert float(losses.min())-2e-10<=rain_mean<=float(losses.mean())+2e-10
            p.update(rain_power_mean_db=rain_mean,rain_arithmetic_mean_db=float(losses.mean()),
                rain_anchor_db=anchor,rain_min_db=float(losses.min()),rain_max_db=float(losses.max()),
                total_relative_db=p['gas_relative_db']+p['cloud_db']+rain_mean)
            sample_count+=len(losses)
        predictions.append(p)
    emit('predictions_local.json',predictions)
    emit('prediction_complete.json',dict(utc=now(),mother_hours=687,computed_hours=int(mask.sum()),
        evaluated_unique_records=sample_count,legacy020_max_anchor_difference_db=max_anchor,
        gas_reference_db=ref['gas_absolute_db'],truth_loaded=False,prediction_sha256=sha(OUT/'predictions_local.json')))
    # Comparison mask additionally excludes the three historical negative-input ITU hours.
    legacy_mask={epoch(r['hour_utc']):r['supplement_eligible'] for r in read(EXP/'Combined_Channel_Revision_027/all_hour_ledger.json')}
    assert set(legacy_mask)==set(map(int,keys))
    common=mask & np.array([legacy_mask[int(t)] for t in keys])
    emit('comparison_mask.json',dict(utc=now(),cohort='LOCAL_ESST_common_input_valid',mother_hours=687,
        esst_eligible_hours=int(mask.sum()),common_hours=int(common.sum()),truth_loaded=False,
        eligible_hour_epoch_s=[int(t) for t in keys[common]],
        additional_legacy_itu_rejection_keys=[int(t) for t in keys[mask & ~common]],
        rule='E-SST qualified AND Revision027 signed liquid ITU input valid; no outcome filtering'))
    names=[n for f,rs in FAMILIES.items() if f!=ESST for n in rs]
    frozen=pd.read_parquet(EXP/'Combined_Channel_Revision_012/selected_local.parquet',columns=['hour_bjt','truth_db',*names])
    assert len(frozen)==687 and frozen.hour_bjt.is_unique
    frozen['epoch_s']=frozen.hour_bjt.astype('int64')//10**9
    frozen=frozen.set_index('epoch_s').loc[keys]
    assert np.isfinite(frozen[['truth_db',*names]].to_numpy()).all()
    original=pd.read_parquet(EXP/'Combined_Channel_Revision_011/predictions_local.parquet',columns=['hour_bjt','truth_db',*names])
    original['epoch_s']=original.hour_bjt.astype('int64')//10**9
    assert np.array_equal(original.set_index('epoch_s').loc[keys,['truth_db',*names]],frozen[['truth_db',*names]])
    frozen[ESST]=[np.nan if r['total_relative_db'] is None else r['total_relative_db'] for r in predictions]
    runs=[]; ranking=[]; coverage=[]; native=[]
    for event in EVENTS:
        eventmask=base.event.eq(event).to_numpy(); selected=common & eventmask
        dates=sorted(set(base.loc[selected,'hour_bjt'].dt.date.astype(str)))
        selected_keys=[int(t) for t in keys[selected]]
        coverage.append(dict(event=event,mother_n=int(eventmask.sum()),esst_eligible_n=int((mask & eventmask).sum()),
            common_n=int(selected.sum()),dates_bjt=dates,hour_epoch_s=selected_keys,
            missing_esst_keys=[int(t) for t in keys[eventmask & ~mask]],
            excluded_comparison_keys=[int(t) for t in keys[eventmask & ~common]]))
        own=mask & eventmask
        native.append(dict(event=event,n=int(own.sum()),rank=None,not_ranked_against_other_masks=True,
            metrics=metrics(frozen.truth_db.to_numpy()[own],frozen[ESST].to_numpy()[own]) if own.any() else None))
        group=[]
        for family,run_names in FAMILIES.items():
            values=[]
            for name in run_names:
                value=metrics(frozen.truth_db.to_numpy()[selected],frozen[name].to_numpy()[selected]) if selected.any() else None
                runs.append(dict(cohort='LOCAL_ESST_common_input_valid',event=event,family=family,run=name,n=int(selected.sum()),metrics=value))
                if value: values.append(value)
            means={m:float(np.mean([x[m] for x in values])) if values else None for m in ('rmse_db','mae_db','bias_db')}
            sds={m:float(np.std([x[m] for x in values],ddof=1)) if len(values)>1 else None for m in means}
            group.append(dict(cohort='LOCAL_ESST_common_input_valid',event=event,family=family,
                n=int(selected.sum()),n_dates=len(dates),dates_bjt=dates,seeds=len(run_names),status='COMPUTED' if values else 'NO_ELIGIBLE_SAMPLES',
                metrics_mean=means,metrics_sd=sds,rank=None))
        for row in group:
            if row['metrics_mean']['rmse_db'] is not None:
                row['rank']=1+sum(q['metrics_mean']['rmse_db']<row['metrics_mean']['rmse_db'] for q in group)
        ranking.extend(group)
        for family in ('rmop_atmospheric_branch','nwp_prm_mts_chain'):
            ranking.append(dict(cohort='LOCAL_ESST_common_input_valid',event=event,family=family,
                n=None,n_dates=None,dates_bjt=None,seeds=None,status='BLOCKED_INPUT_OR_IMPLEMENTATION',
                metrics_mean={m:None for m in ('rmse_db','mae_db','bias_db')},metrics_sd={m:None for m in ('rmse_db','mae_db','bias_db')},rank=None))
    emit('run_metrics.json',runs); emit('seven_family_rankings.json',ranking)
    emit('date_coverage.json',coverage); emit('esst_own_coverage_metrics_unranked.json',native)
    emit('coverage_summary.json',dict(mother_hours=687,computed_hours=int(mask.sum()),missing_hours=int((~mask).sum()),
        common_hours=int(common.sum()),evaluated_unique_records=sample_count,
        reasons=dict(Counter(x for r in ledger for x in r['reasons'])),
        regime_counts=dict(Counter(r['regime'] for r in ledger if r['eligible'])),
        mother_dates_bjt=sorted(set(base.hour_bjt.dt.date.astype(str))),
        computed_dates_bjt=sorted(set(base.loc[mask,'hour_bjt'].dt.date.astype(str))),
        negative_total_relative_hours=sum(r['total_relative_db'] is not None and r['total_relative_db']<0 for r in predictions)))
    emit('model_status.json',dict(local_esst='COMPUTED_ENGINEERING_ADAPTATION_PENDING_QA',uk_esst='BLOCKED_MISSING_INPUT_NO_DOWNLOAD_AUTHORIZATION',
        rmop='BLOCKED_INPUT_AND_IMPLEMENTATION',nwp_prm='BLOCKED_INPUT_AND_IMPLEMENTATION',
        nwp_mts='NOT_COMPUTED_USER_DECISION',all_seven_complete=False,new_training=0,weather_downloads=0,
        old_results_overwritten=False,new_fast_dynamics=0))
    assert all(sha(p)==h for p,h in hashes.items()),'protected source changed after computation'
    emit('calculation_complete.json',dict(status='PASS_PRODUCER_ONLY',utc=now(),protected_sources=len(hashes),changed_sources=0,
        computed_hours=int(mask.sum()),common_hours=int(common.sum()),unit_tests=result.testsRun))
    print(json.dumps(read(OUT/'coverage_summary.json'),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
