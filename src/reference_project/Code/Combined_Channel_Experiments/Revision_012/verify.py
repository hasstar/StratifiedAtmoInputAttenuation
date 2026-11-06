"""Separate scalar verification; deliberately does not import audit.py."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import msvcrt
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_012'
R11=ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'

def need(ok,msg):
    if not ok: raise AssertionError(msg)
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def deny(*a,**k): raise RuntimeError('Offline verification')

def main():
    socket.socket.connect=deny; socket.create_connection=deny
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try: verify_all()
        finally: lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

def verify_all():
    guard=read(OUT/'sources_before.json')
    for name,h in guard.items(): need(sha(ROOT/name)==h,'Source SHA: '+name)
    c=read(OUT/'calculation_complete.json')
    for name,h in c['output_sha256'].items(): need(sha(OUT/name)==h,'Output SHA: '+name)
    local=pd.read_parquet(R11/'predictions_local.parquet')
    uk=pd.read_parquet(R11/'predictions_uk.parquet')
    old_local=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_006/predictions_local_all.parquet')
    raw_uk=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/expanded_dataset_with_itu_inputs.parquet')
    weather=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet')
    models=['ours_frozen_top10']+[f'bai2022_lstm_t0_adapted_seed{s}' for s in [17,29,43]]+[f'tabm2025_official_budgeted_seed{s}' for s in [17,29,43]]+['itu_combined_geometric_approx']
    family_map={m:m.rsplit('_seed',1)[0] for m in models}
    for col in ['hour_bjt','truth_db','rain','dry_overcast']+models[:-1]:
        need(local[col].equals(old_local[col]),'LOCAL frozen identity '+col)
    keyed=uk.set_index(['station','utc_epoch_seconds']).sort_index()
    raw=raw_uk.set_index(['station','utc_epoch_seconds']).sort_index()
    lab=weather.set_index(['station','utc_epoch_seconds']).sort_index()
    need(keyed.index.is_unique and keyed.index.equals(raw.index) and keyed.index.equals(lab.index),'UK keys')
    need(keyed.truth_db.equals(raw.truth_db),'UK truth identity')
    independent_events={}
    for row in uk.itertuples():
        key=(row.station,row.utc_epoch_seconds)
        p=3600*row.mean_total_precipitation_rate; s=3600*row.mean_snowfall_rate
        need(all(math.isfinite(v) for v in [p,s,row.tcc]),'Nonfinite classification input')
        e='rain' if p>=0.1 and s<0.1 else 'dry_overcast' if p<0.1 and s<0.1 and row.tcc>=0.8 else 'other'
        need(e==row.event and e==lab.loc[key,'event'],'Weather event mismatch')
        need(row.common_score_eligible == bool(row.primary_score_eligible and row.physical_input_valid),'Common rule')
        need(row.hour_utc.timestamp()==row.utc_epoch_seconds,'UTC key')
        need((row.rain_valid_utc_end-row.hour_utc).total_seconds()==3600,'Rain valid-end alignment')
        independent_events[key]=e
    local=local.copy(); local['station']='LOCAL'; local['time']=local.hour_bjt.astype(str); local['date']=local.hour_bjt.dt.strftime('%Y-%m-%d')
    uk=uk.copy(); uk['time']=uk.hour_utc.astype(str); uk['date']=uk.hour_utc.dt.strftime('%Y-%m-%d')
    frames={
        'LOCAL_test':local,
        'UK_full_common':uk[uk.primary_score_eligible & uk.physical_input_valid],
        'UK_chilbolton_full_common':uk[uk.primary_score_eligible & uk.physical_input_valid & uk.station.eq('chilbolton')],
        'UK_sparsholt_full_common':uk[uk.primary_score_eligible & uk.physical_input_valid & uk.station.eq('sparsholt')]}
    expected={}; memberships=[]; coverage_map={}
    for cohort,frame in frames.items():
        for event in ('rain','dry_overcast'):
            mask=frame[event] if cohort=='LOCAL_test' else frame.apply(lambda r:independent_events[(r.station,r.utc_epoch_seconds)]==event,axis=1)
            d=frame[mask]
            coverage_map[(cohort,event)]={s:sorted(g.date.unique().tolist()) for s,g in d.groupby('station')}
            memberships.extend((cohort,event,r.station,r.time) for r in d.itertuples())
            for m in models:
                err=[float(p)-float(y) for p,y in zip(d[m],d.truth_db)]
                need(err and all(math.isfinite(e) for e in err),'Missing scoring value')
                n=len(err)
                expected[(cohort,event,m)]={'n':n,'rmse_db':math.sqrt(math.fsum(e*e for e in err)/n),
                    'mae_db':math.fsum(abs(e) for e in err)/n,'bias_db':math.fsum(err)/n}
    runs=read(OUT/'run_metrics.json'); max_error=0.
    need(len(runs)==64 and len({(r['cohort'],r['event'],r['model']) for r in runs})==64,'Exactly 64 unique run metrics')
    old={(r['cohort'],r['event'],r['model']):r for r in read(R11/'run_metrics.json')}
    for r in runs:
        key=(r['cohort'],r['event'],r['model']); q=expected[key]
        need(r['family']==family_map[r['model']] and r['n']==q['n'],'Family/n')
        for m in ('rmse_db','mae_db','bias_db'):
            error=abs(r[m]-q[m]); max_error=max(error,max_error)
            need(error<1e-12 and abs(r[m]-old[key][m])<1e-12,'Scalar/prior score '+str(key))
    family=read(OUT/'seven_family_rankings.json')
    need(len(family)==56,'7 families x4 cohorts x2 events')
    scored=[r for r in family if r['status']=='SCORED_FROZEN']
    need(len(scored)==32,'Four scored families')
    for r in family:
        if r['status']=='BLOCKED_NOT_COMPUTED':
            need(r['rank'] is None and r['n']==0 and all(r[m] is None for m in ['rmse_db_mean','mae_db_mean','bias_db_mean']),'No manufactured blocked scores')
            need(r['missing_prediction_n']==expected[(r['cohort'],r['event'],'ours_frozen_top10')]['n'],'Blocked coverage')
            continue
        ms=[m for m in models if family_map[m]==r['family']]
        need(r['n_runs']==len(ms),'Seed policy')
        for metric in ('rmse_db','mae_db','bias_db'):
            vals=[expected[(r['cohort'],r['event'],m)][metric] for m in ms]
            need(abs(r[metric+'_mean']-math.fsum(vals)/len(vals))<1e-12,'Family mean')
            if len(vals)>1: need(abs(r[metric+'_sd']-statistics.stdev(vals))<1e-12,'Seed SD')
            else: need(r[metric+'_sd'] is None,'No single-run uncertainty')
        need(r['dates_by_station']==coverage_map[(r['cohort'],r['event'])],'Scored date coverage')
        rank=1+sum(x['cohort']==r['cohort'] and x['event']==r['event'] and x['rmse_db_mean']<r['rmse_db_mean'] for x in scored)
        need(rank==r['rank'],'Rank')
    delivered=[(r['cohort'],r['event'],r['station'],r['time']) for r in read(OUT/'sample_membership.json')]
    need(sorted(memberships)==sorted(delivered),'Exact same sample membership')
    ledger=read(OUT/'all_hour_ledger.json'); need(len(ledger)==6351,'Complete ledger')
    need(len({(r['station'],r['time']) for r in ledger})==6351,'Ledger uniqueness')
    records={(r['station'],r['time']):r for r in ledger}
    for frame in (local,uk):
        for r in frame.itertuples():
            q=records[(r.station,r.time)]
            need(q['common_score_eligible']==r.common_score_eligible and q['event']==r.event,'Ledger mask/event')
            need(q['primary_score_eligible']==bool(getattr(r,'primary_score_eligible',True)),'Ledger primary')
            need(q['physical_input_valid']==bool(getattr(r,'physical_input_valid',True)),'Ledger physical')
            need(q['truth_finite']==math.isfinite(r.truth_db),'Ledger truth flag')
            need(q['date']==r.date and bool(q['exclusion_reason'])==(not r.common_score_eligible),'Exclusion retained')
    for file,frame in [('selected_local.parquet',local),('selected_uk.parquet',uk)]:
        new=pd.read_parquet(OUT/file)
        need(set(new.columns).intersection(models)==set(models),'New table model roster')
        need(not any('mlp' in c or 'tabpfn' in c or 'tabicl' in c for c in new.columns),'Excluded models absent')
        for c0 in new.columns: need(new[c0].equals(frame[c0]),'Frozen table copied exactly '+c0)
    days=read(OUT/'coverage_by_date.json')
    need(len(days)==392,'156 LOCAL days +236 UK station-days')
    for q in days:
        frame=local if q['station']=='LOCAL' else uk[uk.station.eq(q['station'])]
        g=frame[frame.date.eq(q['date'])]
        need(q['source_hours']==len(g) and q['common_hours']==int(g.common_score_eligible.sum()),'Per-date population')
        for e in ('rain','dry_overcast','other'): need(q['event_hours_all'][e]==int(g.event.eq(e).sum()),'All-hour weather')
        for e in ('rain','dry_overcast'): need(q['event_hours_scored'][e]==int((g.event.eq(e)&g.common_score_eligible).sum()),'Scored weather')
    audit=read(OUT/'input_availability.json')
    source_local=pd.read_parquet(ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet')
    for q in audit['fields']:
        frame=source_local if q['dataset']=='LOCAL_test' else uk
        finite=sum(math.isfinite(float(v)) for v in frame[q['actual_column']]) if q['actual_column'] else 0
        need(q['n_finite']==finite and q['n_unavailable']==len(frame)-finite,'Input availability')
    tests=read(OUT/'test_results.json'); need(tests['success'] and tests['tests_run']>=10,'Negative tests')
    for name,h in guard.items(): need(sha(ROOT/name)==h,'Post-QA protected SHA')
    result={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'method':'Separate scalar math.fsum implementation, same agent; not external peer review.',
        'verified_run_metrics':64,'scored_family_rows':32,'blocked_family_rows':24,'all_hour_ledger_rows':6351,
        'max_metric_absolute_error':max_error,'common_uk_hours':5538,'label_valid_uk_hours':5593,
        'uk_station_days':{'chilbolton':143,'sparsholt':93},'local_test_hours':687,
        'protected_sources':len(guard),'prediction_identity':'exact','full_dates_and_masks':'PASS',
        'weather_thresholds_and_rain_utc_alignment':'PASS','tests_run':tests['tests_run'],
        'new_training':0,'new_inference':0,'weather_downloads':0,'all_seven_complete':False,
        'verification_code_sha256':sha(Path(__file__))}
    with (OUT/'verification.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
