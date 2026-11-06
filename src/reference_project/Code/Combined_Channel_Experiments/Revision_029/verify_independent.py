"""Independent Arrow native reader + energy CRPS + scalar aggregation.

No imports of Revision029 core/calculate or Revision017 score implementation.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from bisect import bisect_left
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib
import json
import math
import sys
import traceback
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'Outputs/Experiments';OUT=EXP/'Combined_Channel_Revision_029'
ESST=EXP/'Combined_Channel_Revision_028';OLD=EXP/'Combined_Channel_Revision_017'
FAST=EXP/'Fast_Fading_Revision_002';RAW=EXP/'Fast_Fading_Revision_001/clean_1s'
METRICS=('crps201_db','crps401_db','central90_coverage','central98_coverage',
         'tail95_exceedance','tail99_exceedance','slow_only_crps_db')
FAMS={'ours':['ours_frozen_top10'],'bai_lstm':[f'bai2022_lstm_t0_adapted_seed{s}' for s in (17,29,43)],
      'tabm':[f'tabm2025_official_budgeted_seed{s}' for s in (17,29,43)],
      'itu':['itu_combined_geometric_approx'],'esst':['esst_hourly_adaptation']}
RUNS={r:f for f,rs in FAMS.items() for r in rs}
TOL=2e-10


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def same(x,y,tol=TOL):
    if not math.isfinite(float(x)) or not math.isfinite(float(y)) or abs(float(x)-float(y))>tol:
        raise AssertionError((x,y,tol))


def energy(y,w,atoms):
    # Sorted model atoms, not production's sorted observations/pinball prefix.
    x=np.sort(atoms);n=len(x);prefix=np.r_[0.,np.cumsum(x)]
    idx=np.searchsorted(x,y,side='right')
    distance=((2*idx-n)*y+prefix[-1]-2*prefix[idx])/n
    first=math.fsum(float(wi)*float(di) for wi,di in zip(w,distance))/math.fsum(w)
    spread=math.fsum((2*i+1-n)*float(v) for i,v in enumerate(x))/(n*n)
    return first-spread


def controls():
    y=np.array([-5.,0.,0.,4.]);w=np.array([1.,3.,2.,4.]);a=np.array([-2.,-2.,0.,5.,5.])
    direct=math.fsum(float(wi)*math.fsum(abs(float(yi)-float(z)) for z in a)/len(a) for yi,wi in zip(y,w))/math.fsum(w)
    direct-=math.fsum(abs(float(x)-float(z)) for x in a for z in a)/(2*len(a)**2)
    same(energy(y,w,a),direct,1e-12)
    same(energy(y+7,w,a+7),direct,1e-12)
    same(energy(y,w,np.zeros(201)),math.fsum(float(wi)*abs(float(yi)) for yi,wi in zip(y,w))/math.fsum(w),1e-12)
    return 3


def main():
    complete=read(OUT/'calculation_complete.json')
    assert complete['status']=='COMPUTED_PENDING_INDEPENDENT_QA'
    assert read(OUT/'calculate.exit.json')['exit_code']==0
    for p,h in complete['output_sha256'].items():assert sha(OUT/p)==h,p
    protected=read(OUT/'sources_before.json');code=read(OUT/'code_before.json')
    assert all(sha(p)==h for p,h in protected.items())
    assert all(sha(ROOT/p)==h for p,h in code.items())
    control_count=controls()
    native=pq.read_table(EXP/'Combined_Channel_Revision_012/selected_local.parquet').to_pylist()
    base={int(r['hour_bjt'].timestamp()):r for r in native}
    eligibility={r['hour_epoch_s']:r for r in read(ESST/'eligibility_ledger.json')}
    preds={r['hour_epoch_s']:r for r in read(ESST/'predictions_local.json')}
    mask=set(read(ESST/'comparison_mask.json')['eligible_hour_epoch_s'])
    assert len(base)==687 and len(mask)==634
    expected={h for h,r in base.items() if h in mask and r['event'] in ('rain','dry_overcast') and r['common_score_eligible']}
    assert len(expected)==200 and set(read(OUT/'score_mask.json')['eligible_hour_epoch_s'])==expected
    ledger=read(OUT/'all_local_hour_ledger.json');assert len(ledger)==687
    assert {r['hour_epoch_s'] for r in ledger}==set(base)
    for r in ledger:
        h=r['hour_epoch_s'];b=base[h];e=eligibility[h]
        assert int(datetime.fromisoformat(r['hour_utc']).timestamp())==h
        assert datetime.fromtimestamp(h,timezone(timedelta(hours=8))).date().isoformat()==r['date']==b['date']
        assert r['time']==b['time'] and r['event']==b['event']==e['event'] and r['station']=='LOCAL'
        assert r['esst_eligible']==e['eligible'] and r['esst_exclusion_reasons']==e['reasons']
        assert r['esst_hourly_valid_representatives']==e['valid_record_count']
        assert r['common_input_eligible']==(h in mask) and r['score_eligible']==(h in expected)
        reasons=list(e['reasons']) if not e['eligible'] else (['legacy_itu_input_invalid'] if h not in mask else [])
        if not b['common_score_eligible']:reasons.append('old_common_score_ineligible')
        if b['event'] not in ('rain','dry_overcast'):reasons.append('not_requested_weather')
        assert r['exclusion_reasons']==reasons
        assert r['target_weather']==(b['event'] in ('rain','dry_overcast'))
        assert preds[h]['eligible']==e['eligible']
        if not e['eligible']:assert preds[h]['total_relative_db'] is None
    package=read(FAST/'final_generator/model.json');models={};max_quantile=0.
    frozen_saved=read(OLD/'quantiles.json')
    for event in ('rain','dry_overcast'):
        c=package['components'][event]
        assert sha(FAST/c['cdf_file_relative_to_revision_output'])==c['cdf_sha256']
        with np.load(FAST/c['cdf_file_relative_to_revision_output']) as z:
            q=z['q'].tolist();cdf=z['cdf'].tolist()
        assert all(x>0 and math.isfinite(x) for x in q) and all(q[i]<=q[i+1] for i in range(len(q)-1))
        weights=[cdf[0]]+[cdf[i]-cdf[i-1] for i in range(1,len(cdf))]
        assert all(w>0 for w in weights);same(cdf[-1],1,1e-12)
        same(math.fsum(w*x for w,x in zip(weights,q)),1,1e-10)
        def quant(p):return -10*math.log10(q[bisect_left(cdf,1-p)])
        model={n:np.array([quant((i+.5)/n) for i in range(n)]) for n in (201,401)}
        model['bounds']=np.array([quant(p) for p in (.01,.05,.5,.95,.99)])
        for k,a in model.items():
            delta=float(np.max(abs(a-np.array(frozen_saved[event][str(k)]))));same(delta,0,1e-13)
            max_quantile=max(max_quantile,delta)
        models[event]=model
    scores=read(OUT/'hourly_scores.json');assert len(scores)==1800
    indexed={(int(datetime.fromisoformat(r['time']).timestamp()),r['run']):r for r in scores}
    assert len(indexed)==1800 and set(indexed)=={(h,run) for h in expected for run in RUNS}
    previous=pq.read_table(OLD/'hourly_scores.parquet',filters=[('station','==','LOCAL')]).to_pylist()
    reused=0
    for row in previous:
        h=int(datetime.fromisoformat(row['time']).timestamp())
        if h in expected:
            new=indexed[h,row['run']]
            assert new==row,'historical row altered'
            reused+=1
    assert reused==1600
    audit={r['hour_epoch_s']:r for r in read(OUT/'native_support_audit.json')}
    assert set(audit)==expected
    manifest=read(FAST/'final_evaluation/manifest.json')
    fresh=[];seen=set();total_seconds=0;total_weight=0;max_score=0.;max_truth=0.
    columns=['second_bjt','hour_bjt','received_power_w','hour_mean_power_w','quality_zero_records',
             'primary_usable_1s','rain','dry_overcast','fast_residual_db']
    for name,hsh in sorted(manifest['frozen_design']['source_clean_1s'].items()):
        path=RAW/name;assert sha(path)==hsh
        table=pq.read_table(path,columns=columns,filters=[('split','==','test')])
        assert table.schema.field('second_bjt').type.unit=='ns' and table.schema.field('second_bjt').type.tz=='Asia/Shanghai'
        assert table.schema.field('hour_bjt').type.unit=='ns' and table.schema.field('hour_bjt').type.tz=='Asia/Shanghai'
        hours=table['hour_bjt'].cast(pa.int64()).to_numpy()//10**9
        tns=table['second_bjt'].cast(pa.int64()).to_numpy()
        arrays={k:table[k].to_numpy() for k in columns if k not in ('hour_bjt','second_bjt')}
        for h in sorted(set(hours)&expected):
            h=int(h);assert h not in seen;seen.add(h);b=base[h]
            inds=np.flatnonzero(hours==h);inds=inds[np.argsort(tns[inds],kind='stable')]
            ts=tns[inds];assert np.all(ts%10**9==0) and len(np.unique(ts))==len(ts)
            slots=ts//10**9-h;assert np.all((slots>=0)&(slots<3600))
            p=arrays['received_power_w'][inds].astype(float);counts=arrays['quality_zero_records'][inds].astype(float)
            assert np.all(counts==np.floor(counts))
            keep=(counts>=9)&np.isfinite(p)&(p>0)
            assert keep.any() and np.array_equal(keep,arrays['primary_usable_1s'][inds])
            y=-76.19362933508154-10*np.log10(p[keep])-30;w=counts[keep];weight=math.fsum(w)
            hourpower=arrays['hour_mean_power_w'][inds];assert len(np.unique(hourpower))==1
            truth=-76.19362933508154-10*math.log10(float(hourpower[0]))-30
            same(truth,b['truth_db'],1e-8);max_truth=max(max_truth,abs(truth-b['truth_db']))
            np.testing.assert_allclose(y-b['truth_db'],arrays['fast_residual_db'][inds][keep],rtol=0,atol=1e-10)
            rr=arrays['rain'][inds];dd=arrays['dry_overcast'][inds]
            assert len(np.unique(rr))==len(np.unique(dd))==1
            event='rain' if bool(rr[0]) else 'dry_overcast' if bool(dd[0]) else 'other'
            assert event==b['event']
            au=audit[h];assert au['n_valid_seconds']==len(y) and au['native_weight_total']==int(weight)
            assert au['n_present_seconds']==len(inds) and au['esst_hourly_valid_representatives']==eligibility[h]['valid_record_count']
            sig=np.asarray(list(zip(slots[keep],w)),dtype='<i8').tobytes()
            assert au['second_mask_sha256']==hashlib.sha256(sig).hexdigest()
            assert au['source']==str(path.relative_to(ROOT)) and au['source_sha256']==hsh
            assert au['min_valid_slot']==int(min(slots[keep])) and au['max_valid_slot']==int(max(slots[keep]))
            total_seconds+=len(y);total_weight+=int(weight)
            for run,family in RUNS.items():
                shift=float(preds[h]['total_relative_db'] if family=='esst' else b[run])
                saved=indexed[h,run]
                assert saved['prediction_hour_db']==shift and saved['truth_hour_db']==b['truth_db']
                assert saved['family']==family and saved['event']==event and saved['date']==b['date']
                assert saved['n_seconds']==len(y) and saved['native_weight_total']==int(weight)
                model=models[event];v={f'crps{n}_db':energy(y,w,model[n]+shift) for n in (201,401)}
                bounds=model['bounds']+shift
                for pct,i,j in [(90,1,3),(98,0,4)]:v[f'central{pct}_coverage']=math.fsum(w[(y>=bounds[i])&(y<=bounds[j])])/weight
                for pct,j in [(95,3),(99,4)]:v[f'tail{pct}_exceedance']=math.fsum(w[y>bounds[j]])/weight
                v['slow_only_crps_db']=math.fsum(float(wi)*abs(float(yi)-shift) for yi,wi in zip(y,w))/weight
                for m,value in v.items():same(value,saved[m]);max_score=max(max_score,abs(value-saved[m]))
                fresh.append(dict(event=event,date=b['date'],run=run,family=family,n_seconds=len(y),native_weight_total=int(weight),**v))
        print(json.dumps(dict(independent_file=name,checked_hours=len(seen),max_score_delta=max_score)),flush=True)
    assert seen==expected
    summaries=read(OUT/'run_metrics.json');assert len(summaries)==36
    independently_aggregated={};max_aggregate=0.
    for row in summaries:
        sub=[x for x in fresh if x['run']==row['run'] and x['event']==row['event']]
        dates=sorted({x['date'] for x in sub})
        assert row['cohort']=='LOCAL_ESST_common_frozen_fast' and row['family']==RUNS[row['run']]
        assert row['n_station_hours']==len(sub) and row['n_dates']==len(dates) and row['actual_dates']==dates
        assert row['date_min']==min(dates) and row['date_max']==max(dates)
        assert row['n_seconds']==sum(x['n_seconds'] for x in sub) and row['native_weight_total']==sum(x['native_weight_total'] for x in sub)
        vals={}
        for m in METRICS:
            if row['weighting']=='hour_equal':value=math.fsum(x[m] for x in sub)/len(sub)
            else:
                assert row['weighting']=='date_equal';days=defaultdict(list)
                for x in sub:days[x['date']].append(x[m])
                value=math.fsum(math.fsum(a)/len(a) for a in days.values())/len(days)
            same(value,row[m]);max_aggregate=max(max_aggregate,abs(value-row[m]));vals[m]=value
        independently_aggregated[row['event'],row['weighting'],row['run']]=vals
    fam=read(OUT/'family_metrics.json');assert len(fam)==20
    for row in fam:
        members=FAMS[row['family']];assert row['n_frozen_runs']==len(members) and row['status']=='COMPUTED'
        refs=[s for s in summaries if s['event']==row['event'] and s['weighting']==row['weighting'] and s['family']==row['family']]
        for k in ('n_station_hours','n_dates','n_seconds','native_weight_total','actual_dates','date_min','date_max'):
            assert row[k]==refs[0][k]
        for m in METRICS:
            a=[independently_aggregated[row['event'],row['weighting'],run][m] for run in members]
            mean=math.fsum(a)/len(a);same(mean,row[m]);max_aggregate=max(max_aggregate,abs(mean-row[m]))
            if len(a)>1:same(math.sqrt(math.fsum((x-mean)**2 for x in a)/(len(a)-1)),row[m+'_sd'])
            else:assert row[m+'_sd'] is None
        peers=[x for x in fam if x['event']==row['event'] and x['weighting']==row['weighting']]
        independent_means={x['family']:math.fsum(independently_aggregated[x['event'],x['weighting'],run]['crps201_db'] for run in FAMS[x['family']])/len(FAMS[x['family']]) for x in peers}
        assert row['rank_crps201']==1+sum(v<independent_means[row['family']] for v in independent_means.values())
    seven=read(OUT/'seven_family_metrics.json');assert len(seven)==28
    assert [r for r in seven if r['status']=='COMPUTED']==fam
    for r in seven:
        if r['status']!='COMPUTED':
            assert r['family'] in ('rmop','nwp_chain') and all(r[m] is None for m in METRICS)
            assert all(r[k] is None for k in ('n_station_hours','n_dates','n_seconds','n_frozen_runs','rank_crps201'))
            if r['family']=='nwp_chain':assert r['nwp_mts']=='NOT_COMPUTED_USER_DECISION'
    coverage=read(OUT/'coverage_summary.json');days=read(OUT/'coverage_by_date.json')
    assert len(days)==156 and sum(r['mother_hours'] for r in days)==687
    assert coverage['valid_seconds']==total_seconds and coverage['native_weight_total']==total_weight
    assert coverage['esst_hourly_representatives']==sum(eligibility[h]['valid_record_count'] for h in expected)
    assert coverage['mother_hours']==687 and coverage['esst_available_hours']==634 and coverage['esst_missing_hours']==53
    assert coverage['score_dates']==len({base[h]['date'] for h in expected})
    for d in days:
        rows=[r for r in ledger if r['date']==d['date']]
        assert d['mother_hours']==len(rows) and d['esst_eligible_hours']==sum(r['esst_eligible'] for r in rows)
        assert d['missing_esst_hours']==sum(not r['esst_eligible'] for r in rows)
        for e,v in d['weather'].items():
            assert v['mother']==sum(r['event']==e for r in rows)
            assert v['scored']==sum(r['event']==e and r['score_eligible'] for r in rows)
    for e,row in coverage['by_weather'].items():
        rows=[r for r in ledger if r['event']==e];chosen=[r for r in rows if r['score_eligible']]
        assert row['mother_hours']==len(rows) and row['score_hours']==len(chosen)
        assert row['esst_available_hours']==sum(r['esst_eligible'] for r in rows)
        assert row['score_dates']==sorted({r['date'] for r in chosen})
        assert row['excluded_hour_epoch_s']==[r['hour_epoch_s'] for r in rows if not r['score_eligible']]
        assert row['missing_esst_reasons']==dict(Counter(x for r in rows for x in r['esst_exclusion_reasons']))
    assert read(OUT/'model_status.json')['nwp_mts']=='NOT_COMPUTED_USER_DECISION'
    assert all(sha(p)==h for p,h in protected.items()) and all(sha(ROOT/p)==h for p,h in code.items())
    assert all(sha(OUT/p)==h for p,h in complete['output_sha256'].items())
    emit('independent_qa.json',dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        implementation='Arrow native seconds; independent bisect quantiles; energy CRPS; math.fsum aggregation; same agent, not external reviewer',
        controls=control_count,replayed_hours=len(seen),replayed_run_hours=len(fresh),native_seconds=total_seconds,
        native_weight_total=total_weight,unchanged_reused_rows=reused,mother_ledger_hours=len(ledger),
        run_summaries=len(summaries),computed_family_summaries=len(fam),seven_family_summary_rows=len(seven),
        max_quantile_delta_db=max_quantile,max_native_hour_target_delta_db=max_truth,
        max_score_delta=max_score,max_aggregate_delta=max_aggregate,tolerance=TOL,
        protected_source_files=len(protected),protected_sources_unchanged=True,
        code_files=len(code),all_seven_complete=False,nwp_mts='NOT_COMPUTED_USER_DECISION',
        verifier_sha256=sha(__file__),calculation_complete_sha256=sha(OUT/'calculation_complete.json'),
        verified_output_sha256=complete['output_sha256']))


if __name__=='__main__':
    try:main()
    except BaseException:
        emit('verification_failure.json',dict(utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
