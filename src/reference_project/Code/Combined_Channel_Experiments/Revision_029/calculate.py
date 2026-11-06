"""LOCAL-only incremental marginal calculation. No fitting or random generation."""
import os
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import math
import sys
import traceback
import unittest
import numpy as np
import pandas as pd
from core import score, frozen_quantiles, FAMILIES, RUNS, METRICS
from test_core import Controls

ROOT=Path(__file__).resolve().parents[3]; CODE=Path(__file__).parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_029'
EXP=ROOT/'Outputs/Experiments'
OLD=EXP/'Combined_Channel_Revision_017'; ESST=EXP/'Combined_Channel_Revision_028'
BASE=EXP/'Combined_Channel_Revision_012'; FAST=EXP/'Fast_Fading_Revision_002'
RAW=EXP/'Fast_Fading_Revision_001/clean_1s'
CLEAR=-76.19362933508154


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def need(x,msg):
    if not x:raise ValueError(msg)


def guard_sources():
    protected=read(ESST/'sources_before.json')
    need(len(protected)==4721,'028 protected source count')
    need(all(sha(p)==h for p,h in protected.items()),'changed historical source')
    def pin(p,expected=None):
        p=Path(p).resolve(); h=sha(p)
        need(expected is None or h==expected,'SHA mismatch '+str(p))
        need(str(p) not in protected or protected[str(p)]==h,'conflicting SHA')
        protected[str(p)]=h
    for version in ('017','018','028'):
        directory=EXP/('Combined_Channel_Revision_'+version)
        m=read(directory/'delivery_manifest.json'); final=read(directory/'final_verification.json')
        need(final['status'].startswith('PASS'),'unverified previous version')
        expected=final.get('manifest_sha256',final.get('delivery_manifest_sha256'))
        pin(directory/'delivery_manifest.json',expected)
        pin(directory/'final_verification.json')
        for rel,item in m['files'].items():pin(ROOT/rel,item['sha256'] if isinstance(item,dict) else item)
    for p in ESST.iterdir():
        if p.is_file():pin(p)
    fm=read(FAST/'final_evaluation/manifest.json'); package=read(FAST/'final_generator/model.json')
    for name,h in fm['frozen_design']['source_clean_1s'].items():pin(RAW/name,h)
    for p in [FAST/'final_generator/model.json',FAST/'final_generator/qa.json',
              FAST/'final_evaluation/manifest.json',BASE/'selected_local.parquet',
              OLD/'hourly_scores.parquet',OLD/'quantiles.json',OLD/'native_hour_audit.json']:
        pin(p)
    for e in ('rain','dry_overcast'):
        comp=package['components'][e];pin(FAST/comp['cdf_file_relative_to_revision_output'],comp['cdf_sha256'])
    code={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in CODE.iterdir() if p.is_file()}
    emit('sources_before.json',protected);emit('code_before.json',code)
    return protected,code,fm,package


def freeze_cohort():
    # This read intentionally excludes target and model-value columns.
    meta=pd.read_parquet(BASE/'selected_local.parquet',columns=['hour_bjt','station','time','date','event','common_score_eligible'])
    meta['hour_epoch_s']=[int(h.timestamp()) for h in meta.hour_bjt]
    need(len(meta)==687 and meta.hour_epoch_s.is_unique,'mother keys')
    eligibility={r['hour_epoch_s']:r for r in read(ESST/'eligibility_ledger.json')}
    comparison=read(ESST/'comparison_mask.json'); common=set(comparison['eligible_hour_epoch_s'])
    need(len(common)==634 and comparison['truth_loaded'] is False,'028 frozen mask')
    ledger=[]
    for r in meta.itertuples():
        old=eligibility[r.hour_epoch_s]
        need(old['event']==r.event and old['date_bjt']==r.date and r.station=='LOCAL','UTC/date/weather mapping')
        target=r.event in ('rain','dry_overcast')
        use=bool(r.common_score_eligible and r.hour_epoch_s in common and target)
        reasons=[]
        if not old['eligible']:reasons.extend(old['reasons'])
        elif r.hour_epoch_s not in common:reasons.append('legacy_itu_input_invalid')
        if not r.common_score_eligible:reasons.append('old_common_score_ineligible')
        if not target:reasons.append('not_requested_weather')
        ledger.append(dict(station='LOCAL',hour_epoch_s=r.hour_epoch_s,hour_utc=old['hour_utc'],
            time=r.time,date=r.date,event=r.event,esst_eligible=old['eligible'],
            common_input_eligible=r.hour_epoch_s in common,target_weather=target,
            score_eligible=use,exclusion_reasons=reasons,esst_exclusion_reasons=old['reasons'],
            esst_hourly_valid_representatives=old['valid_record_count']))
    keys={r['hour_epoch_s'] for r in ledger if r['score_eligible']}
    need(len(keys)==200,'expected input-only 200-hour cohort')
    need(Counter(r['event'] for r in ledger if r['score_eligible'])=={'rain':50,'dry_overcast':150},'weather counts')
    emit('all_local_hour_ledger.json',ledger)
    emit('score_mask.json',dict(utc=now(),cohort='LOCAL_ESST_common_frozen_fast',mother_hours=687,
        common_input_hours=634,score_hours=200,eligible_hour_epoch_s=sorted(keys),
        new_score_target_loaded=False,prior_results_visible=True,rule='028 common input mask AND 012 rain/dry_overcast AND old common gate'))
    return meta,ledger,keys


def summarize(frame):
    runrows=[]; familyrows=[]
    for event in ('rain','dry_overcast'):
        part=frame[frame.event.eq(event)]
        for weighting in ('date_equal','hour_equal'):
            cell=[]
            for run,g in part.groupby('run',sort=True):
                data=g.groupby('date')[list(METRICS)].mean() if weighting=='date_equal' else g[list(METRICS)]
                row=dict(cohort='LOCAL_ESST_common_frozen_fast',event=event,weighting=weighting,
                    run=run,family=RUNS[run],n_station_hours=len(g),n_dates=int(g.date.nunique()),
                    n_seconds=int(g.n_seconds.sum()),native_weight_total=int(g.native_weight_total.sum()),
                    actual_dates=sorted(g.date.unique().tolist()),date_min=g.date.min(),date_max=g.date.max(),
                    **{m:float(data[m].mean()) for m in METRICS})
                cell.append(row);runrows.append(row)
            group=[]
            for family,names in FAMILIES.items():
                items=[r for r in cell if r['family']==family]
                need({r['run'] for r in items}==set(names),'all frozen runs required')
                row={k:items[0][k] for k in ['cohort','event','weighting','n_station_hours','n_dates','n_seconds','native_weight_total','actual_dates','date_min','date_max']}
                row.update(family=family,n_frozen_runs=len(items),status='COMPUTED')
                for m in METRICS:
                    a=np.array([r[m] for r in items]);row[m]=float(a.mean())
                    row[m+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
                group.append(row)
            for row in group:
                row['rank_crps201']=1+sum(peer['crps201_db']<row['crps201_db'] for peer in group)
            familyrows.extend(group)
    return runrows,familyrows


def main():
    emit('execution.json',dict(utc=now(),pid=os.getpid(),command=sys.argv,cwd=str(ROOT),python=sys.version,
        numpy=np.__version__,pandas=pd.__version__,threads=2,new_training=0,new_weather_requests=0,
        new_hourly_predictions=0,new_random_realizations=0,new_cdf_fits=0,
        preflight=dict(process_utc='2026-10-18T05:20:05.4118692Z',related_workers=[],
            free_bytes=165010489344,code_and_output_new=True,previous_locks='AVAILABLE_UNCHANGED',
            other_unregistered_writers='UNKNOWN; source SHA guards and isolated output')))
    with (OUT/'PROTOCOL.md').open('x',encoding='utf-8') as f:f.write((CODE/'PROTOCOL.md').read_text(encoding='utf-8'))
    protected,code,frozen_manifest,package=guard_sources()
    stream=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    emit('unit_tests.json',dict(status='PASS' if tests.wasSuccessful() else 'FAIL',tests=tests.testsRun,
        output=stream.getvalue(),study_rows_used=0))
    need(tests.wasSuccessful(),'synthetic controls failed')
    meta,ledger,keys=freeze_cohort()
    predictions={r['hour_epoch_s']:r for r in read(ESST/'predictions_local.json')}
    selected=pd.read_parquet(BASE/'selected_local.parquet')
    selected['hour_epoch_s']=[int(h.timestamp()) for h in selected.hour_bjt]
    selected=selected[selected.hour_epoch_s.isin(keys)].copy()
    selected['esst_hourly_adaptation']=[predictions[int(h)]['total_relative_db'] for h in selected.hour_epoch_s]
    need(np.isfinite(selected[list(RUNS)].to_numpy()).all(),'nonfinite fixed hourly shifts')
    for r in selected.itertuples():
        p=predictions[r.hour_epoch_s]
        need(p['eligible'] and int(pd.Timestamp(p['hour_utc']).timestamp())==r.hour_epoch_s,'028 key/eligibility')
    old_scores=pd.read_parquet(OLD/'hourly_scores.parquet')
    reused=old_scores[old_scores.station.eq('LOCAL')&old_scores.time.isin(selected.time)].copy()
    need(len(reused)==1600 and not reused.duplicated(['time','run']).any(),'old scores unique/full')
    wanted=selected.set_index('hour_bjt'); prior={(r.time,r.run):r for r in reused.itertuples()}
    old_audit={r['time']:r for r in read(OLD/'native_hour_audit.json') if r['station']=='LOCAL'}
    old_quantiles=read(OLD/'quantiles.json'); models={};cdf_check={}
    for event in ('rain','dry_overcast'):
        comp=package['components'][event]
        with np.load(FAST/comp['cdf_file_relative_to_revision_output']) as a:
            model=frozen_quantiles(a['q'],a['cdf']); moment=float(np.dot(a['q'],np.diff(np.r_[0.,a['cdf']])))
            support=len(a['q'])
        for k,v in model.items():need(np.array_equal(v,np.asarray(old_quantiles[event][str(k)])),'frozen quantiles changed')
        models[event]=model;cdf_check[event]=dict(sha256=comp['cdf_sha256'],support=support,unit_power_moment=moment,changed=False)
    emit('cdf_compatibility.json',dict(status='PASS',components=cdf_check,new_fits=0,
        slow_component='028 hourly constant, not its internal representative-point trajectory',
        temporal_claim='constant cancels in within-hour increments and centered ACF; no new real-data temporal scores'))
    additions=[];audit=[];seen=set(); bykey={r['hour_epoch_s']:r for r in ledger}
    for name in sorted(frozen_manifest['frozen_design']['source_clean_1s']):
        path=RAW/name; f=pd.read_parquet(path,filters=[('split','==','test')])
        f=f[f.hour_bjt.isin(wanted.index)]
        for h,g in f.groupby('hour_bjt',sort=True):
            g=g.sort_values('second_bjt'); row=wanted.loc[h]
            need(h not in seen,'duplicate native hour');seen.add(h)
            slots=(g.second_bjt-g.hour_bjt).dt.total_seconds().to_numpy()
            need(np.isfinite(slots).all() and (slots==np.floor(slots)).all() and ((slots>=0)&(slots<3600)).all() and len(set(slots))==len(slots),'native second time')
            need(g.hour_mean_power_w.nunique()==1 and g.rain.nunique()==1 and g.dry_overcast.nunique()==1,'fixed hourly source metadata')
            truth=CLEAR-10*math.log10(1000*float(g.hour_mean_power_w.iloc[0]))
            need(abs(truth-row.truth_db)<1e-8,'hourly target identity; no refitting')
            event='rain' if bool(g.rain.iloc[0]) else 'dry_overcast' if bool(g.dry_overcast.iloc[0]) else 'other'
            need(event==row.event,'weather condition mismatch')
            p=g.received_power_w.to_numpy(float);w=g.quality_zero_records.to_numpy(float)
            keep=np.isfinite(p)&(p>0)&(w>=9)
            need(np.array_equal(keep,g.primary_usable_1s.to_numpy(bool)) and keep.any(),'native second mask')
            y=CLEAR-10*np.log10(1000*p[keep]);weights=w[keep]
            need(np.allclose(y-row.truth_db,g.fast_residual_db.to_numpy(float)[keep],rtol=0,atol=1e-10),'native residual identity')
            native_count=int(weights.sum());native_seconds=int(keep.sum());origin=str(path.relative_to(ROOT))
            source_old=old_audit[row.time]
            need(source_old['n_seconds']==native_seconds and source_old['weight']==native_count and source_old['source']==origin,'old native audit')
            for run in RUNS:
                if run=='esst_hourly_adaptation':continue
                old=prior[(row.time,run)]
                need(old.prediction_hour_db==float(row[run]) and old.truth_hour_db==float(row.truth_db),'frozen score/prediction mismatch')
                need(old.n_seconds==native_seconds and old.native_weight_total==native_count and old.event==event and old.date==row.date and old.family==RUNS[run],'old score mask identity')
            shift=float(row.esst_hourly_adaptation)
            additions.append(dict(station='LOCAL',time=row.time,date=row.date,event=event,family='esst',
                run='esst_hourly_adaptation',n_seconds=native_seconds,native_weight_total=native_count,
                prediction_hour_db=shift,truth_hour_db=float(row.truth_db),**score(y,weights,models[event],shift)))
            sig=np.column_stack((slots[keep],weights)).astype('<i8').tobytes()
            audit.append(dict(station='LOCAL',hour_epoch_s=int(row.hour_epoch_s),time=row.time,date=row.date,event=event,
                source=origin,source_sha256=sha(path),n_present_seconds=len(g),n_valid_seconds=native_seconds,
                native_weight_total=native_count,second_mask_sha256=hashlib.sha256(sig).hexdigest(),
                esst_hourly_valid_representatives=bykey[int(row.hour_epoch_s)]['esst_hourly_valid_representatives'],
                min_valid_slot=int(slots[keep].min()),max_valid_slot=int(slots[keep].max()),
                target_identity_abs_error_db=abs(truth-row.truth_db)))
        print(json.dumps({'stage':'local_native','file':name,'hours':len(seen)}),flush=True)
    need(len(seen)==200 and len(additions)==200,'missing new scores')
    frame=pd.concat([reused,pd.DataFrame(additions)],ignore_index=True)
    frame=frame.sort_values(['time','run'],kind='stable')
    need(len(frame)==1800 and np.isfinite(frame[list(METRICS)].to_numpy()).all(),'new complete score table')
    emit('hourly_scores.json',frame.to_dict('records'));emit('native_support_audit.json',audit)
    runrows,familyrows=summarize(frame)
    emit('run_metrics.json',runrows);emit('family_metrics.json',familyrows)
    seven=list(familyrows)
    for event in ('rain','dry_overcast'):
        for weighting in ('date_equal','hour_equal'):
            for family,status in [('rmop','BLOCKED_INPUT_AND_IMPLEMENTATION'),('nwp_chain','BLOCKED_PRM_INPUT_AND_CONFIGURATION')]:
                seven.append(dict(cohort='LOCAL_ESST_common_frozen_fast',event=event,weighting=weighting,family=family,
                    status=status,n_station_hours=None,n_dates=None,n_seconds=None,n_frozen_runs=None,
                    rank_crps201=None,**{m:None for m in METRICS},nwp_mts='NOT_COMPUTED_USER_DECISION' if family=='nwp_chain' else None))
    emit('seven_family_metrics.json',seven)
    days=[]
    for date in sorted({r['date'] for r in ledger}):
        rows=[r for r in ledger if r['date']==date]
        days.append(dict(date=date,mother_hours=len(rows),esst_eligible_hours=sum(r['esst_eligible'] for r in rows),
            missing_esst_hours=sum(not r['esst_eligible'] for r in rows),
            weather={e:dict(mother=sum(r['event']==e for r in rows),scored=sum(r['event']==e and r['score_eligible'] for r in rows)) for e in ('rain','dry_overcast','other')}))
    emit('coverage_by_date.json',days)
    summary={}
    for e in ('rain','dry_overcast','other'):
        rows=[r for r in ledger if r['event']==e]; chosen=[r for r in rows if r['score_eligible']]
        summary[e]=dict(mother_hours=len(rows),esst_available_hours=sum(r['esst_eligible'] for r in rows),
            score_hours=len(chosen),score_dates=sorted({r['date'] for r in chosen}),
            excluded_hour_epoch_s=[r['hour_epoch_s'] for r in rows if not r['score_eligible']],
            missing_esst_reasons=dict(Counter(reason for r in rows for reason in r['esst_exclusion_reasons'])))
    emit('coverage_summary.json',dict(mother_hours=687,mother_dates=len(days),esst_available_hours=634,
        esst_missing_hours=53,score_hours=200,score_dates=len({r['date'] for r in ledger if r['score_eligible']}),
        valid_seconds=sum(r['n_valid_seconds'] for r in audit),native_weight_total=sum(r['native_weight_total'] for r in audit),
        esst_hourly_representatives=sum(r['esst_hourly_valid_representatives'] for r in audit),by_weather=summary))
    emit('reuse_audit.json',dict(status='PASS',old_scores_source=str((OLD/'hourly_scores.parquet').relative_to(ROOT)),
        old_scores_sha256=sha(OLD/'hourly_scores.parquet'),reused_run_hour_rows=1600,new_run_hour_rows=200,
        old_values_unchanged=True,new_hourly_predictions=0,new_stochastic_realizations=0,
        local_hourly_source=str((ESST/'predictions_local.json').relative_to(ROOT)),
        local_hourly_sha256=sha(ESST/'predictions_local.json')))
    emit('model_status.json',dict(local_esst_hourly='COMPUTED_PARTIAL_COVERAGE_ENGINEERING_ADAPTATION',
        local_esst_marginal='COMPUTED_PENDING_INDEPENDENT_QA',uk_esst='BLOCKED_EXISTING_LOCAL_INPUT_INVENTORY',
        rmop='BLOCKED_INPUT_AND_IMPLEMENTATION',nwp_prm='BLOCKED_PRM_INPUT_AND_CONFIGURATION',
        nwp_mts='NOT_COMPUTED_USER_DECISION',all_seven_complete=False))
    need(all(sha(p)==h for p,h in protected.items()),'protected source changed during calculation')
    need(all(sha(ROOT/p)==h for p,h in code.items()),'code changed during calculation')
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix not in ('.log','.lock')}
    emit('calculation_complete.json',dict(status='COMPUTED_PENDING_INDEPENDENT_QA',utc=now(),
        score_hours=200,new_esst_score_rows=200,reused_score_rows=1600,run_summaries=len(runrows),
        family_summaries=len(familyrows),seven_family_summary_rows=len(seven),protected_sources=len(protected),
        all_seven_complete=False,nwp_mts='NOT_COMPUTED_USER_DECISION',output_sha256=files))


if __name__=='__main__':
    try:main()
    except BaseException:
        emit('calculation_failure.json',dict(utc=now(),traceback=traceback.format_exc()));raise
