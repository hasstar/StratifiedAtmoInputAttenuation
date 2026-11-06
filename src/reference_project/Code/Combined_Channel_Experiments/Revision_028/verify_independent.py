"""Independent horizontal-distance overlap and math.fsum QA; no producer imports."""
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_028'
EXP=ROOT/'Outputs/Experiments'
ERRORS=Counter()
CHECKS=Counter()


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def data(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def epoch(s): return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())


def need(ok,label):
    CHECKS[label]+=1
    if not ok: raise AssertionError(label)


def number(a,b,label,tol=2e-10):
    difference=abs(float(a)-float(b))
    need(math.isfinite(difference) and difference<=tol,label)
    ERRORS[label]=max(ERRORS[label],difference)


def horizontal_losses(rates,milliseconds,H,v,theta,k,alpha):
    """Direct intersection of horizontal rain cells, no prefix antiderivative."""
    offsets=np.asarray(milliseconds,dtype=np.float64)
    need(offsets.size>0 and (offsets>=0).all() and (offsets<3600000).all(),'timestamp_domain')
    if H==0: return np.zeros(len(offsets))
    if theta==90: return np.full(len(offsets),k*rates[0]**alpha*H)
    angle=math.radians(theta)
    distance=H/math.tan(angle)
    cell_width=3.6*v
    origin=offsets*v/1e6
    answer=np.zeros(len(origin))
    for j,R in enumerate(rates):
        # max(0,...) is the exact length of an empty geometric intersection,
        # not clipping meteorological inputs or attenuation predictions.
        overlap=np.maximum(0.,np.minimum(origin+distance,(j+1)*cell_width)-np.maximum(origin,j*cell_width))
        answer+=overlap*(k*float(R)**alpha)/math.cos(angle)
    return answer


def reduce_power(a):
    m=float(np.min(a))
    mean=math.fsum(map(float,np.power(10.,-(a-m)/10.)))/len(a)
    return m-10.*math.log10(mean)


def score(y,p):
    need(len(y)>0 and len(y)==len(p),'metric_shape')
    err=[a-b for a,b in zip(p,y)]
    return {'rmse_db':math.sqrt(math.fsum(x*x for x in err)/len(err)),
        'mae_db':math.fsum(abs(x) for x in err)/len(err),'bias_db':math.fsum(err)/len(err)}


def main():
    sources=read(OUT/'sources_before.json'); codes=read(OUT/'code_before.json')
    for p,h in sources.items(): need(sha(p)==h,'source_before')
    for p,h in codes.items(): need(sha(ROOT/p)==h,'code_before')
    approval=read(OUT/'approval.json')
    need(approval['approved'] and approval['answer']=='接受该工程适配，继续测试和计算','human_approval')
    need(sha(approval['proposal_path'])==approval['proposal_sha256'],'exact_approved_proposal')
    need(read(OUT/'unit_tests.json')['status']=='PASS','unit_tests')
    cross=data(EXP/'Combined_Channel_Revision_021/zero_crossing_diagnostics.npz')
    derived=data(EXP/'Combined_Channel_Revision_022/derived_inputs.npz')
    sampling=data(EXP/'Combined_Channel_Revision_023/sampling_support.npz')
    raw_context=read(EXP/'Combined_Channel_Revision_021/rain_context_ledger.json')
    old_events=read(EXP/'Combined_Channel_Revision_021/rain_events.json')
    ledger=read(OUT/'eligibility_ledger.json'); predicted=read(OUT/'predictions_local.json')
    keys=list(map(int,sampling['hour_epoch_s']))
    need(len(keys)==len(set(keys))==len(ledger)==len(predicted)==687,'mother687_unique')
    need(keys==list(map(int,cross['campaign_epoch_s'][cross['test_member']])),'original_test_keys')
    sample_meta=read(EXP/'Combined_Channel_Revision_023/hourly_sampling_ledger.json')
    rain=[]; context_keys=[]
    weights=list(map(float,cross['weights_NW_NE_SW_SE']))
    for record in raw_context:
        ts=epoch(record['time_utc']); context_keys.append(ts)
        need(epoch(record['rate_valid_end_utc'])==ts+3600,'rain_interval_t_plus_1')
        vals=record['liquid_rain_corners_mm_per_h']
        ok=all(v is not None and math.isfinite(v) and v>=0 for v in vals)
        need(ok==record['all_corners_nonnegative'],'raw_corner_validity')
        value=math.fsum(w*v for w,v in zip(weights,vals)) if ok else None
        if ok: number(value,record['site_rain_mm_per_h'],'rain_reinterpolated',2e-12)
        else: need(record['site_rain_mm_per_h'] is None,'invalid_rain_kept_null')
        rain.append(value)
    need(len(rain)==7823 and all(b-a==3600 for a,b in zip(context_keys,context_keys[1:])),'full_context_axis')
    event_ids=[-1]*len(rain); events=[]; start=None
    for i,r in enumerate(rain+[None]):
        wet=r is not None and r>0
        if wet and start is None: start=i
        if not wet and start is not None:
            end=i-1; peak=max(rain[start:i]); left=start==0 or rain[start-1]!=0
            right=i==len(rain) or rain[i]!=0
            event=dict(complete=not left and not right,kind='convective' if peak>10 else 'stratiform')
            eid=len(events); old=old_events[eid]
            need(old['event_id']==eid and old['hours']==i-start and old['complete']==event['complete'],'event_identity_and_censoring')
            need(epoch(old['start_utc'])==context_keys[start] and epoch(old['last_bin_start_utc'])==context_keys[end],'event_time_bounds')
            number(peak,old['peak_hourly_mm_per_h'],'event_peak',2e-12)
            need(event['kind']==old['hourly_peak_type_candidate'],'hourly_type_threshold')
            for j in range(start,i): event_ids[j]=eid
            events.append(event); start=None
    need(len(events)==len(old_events)==253,'all_events_checked')
    need(event_ids==[r['event_id'] for r in raw_context],'all_context_event_ids')
    # Explicit >10 threshold controls, separate from real-data outcomes.
    need(['convective' if r>10 else 'stratiform' for r in [0.,10.,10.0000001]]==['stratiform','stratiform','convective'],'synthetic_threshold_control')
    cmp_index={t:i for i,t in enumerate(map(int,cross['campaign_epoch_s']))}
    wind_index={t:i for i,t in enumerate(map(int,derived['raw_epoch_s']))}
    rain_index={t:i for i,t in enumerate(context_keys)}
    components={int(r['hour_bjt'].timestamp()):r for r in pq.read_table(EXP/'Combined_Channel_Revision_011/predictions_local.parquet',
        columns=['hour_bjt','event','gas_relative_db','cloud_db','gas_absolute_db','gas_reference_db']).to_pylist()}
    ref=read(EXP/'Combined_Channel_Revision_011/fixed_reference_parameters.json')['LOCAL']
    cfg=read(OUT/'parameters.json')
    frozen_cfg=next(c for c in read(EXP/'Combined_Channel_Revision_022/coefficient_parameters.json') if c['site']=='LOCAL')
    for name in ('frequency_ghz','elevation_deg','polarization','k','alpha'): need(cfg[name]==frozen_cfg[name],'frozen_parameter_'+name)
    need(cfg['antenna_height_m_amsl']==117.5 and cfg['time_step_seconds']==3600 and cfg['earth_radius_m']==6371229.,'fixed_adaptation_constants')
    theta,k,alpha=(cfg[x] for x in ('elevation_deg','k','alpha'))
    eligible=[]; rain_answers={}; total_answers={}; records_evaluated=0; reasons_total=Counter(); regimes=Counter()
    for i,t in enumerate(keys):
        row,p=ledger[i],predicted[i]; j=cmp_index[t]; w=wind_index[t]; a=rain_index[t]
        need(row['hour_epoch_s']==p['hour_epoch_s']==t and epoch(row['hour_utc'])==t,'output_key_alignment')
        need(row['event']==components[t]['event'],'frozen_event_labels')
        cands=[]
        for g in cross['candidate_geopotential_m'][j,4]:
            if math.isfinite(float(g)): cands.append(6371229.*float(g)/(6371229.-float(g)))
        need(len(cands)==int(cross['warm_to_cold_count'][j,4]),'candidate_count')
        need(len(cands)==len(row['candidate_geometric_m']),'output_candidate_count')
        for x,y in zip(cands,row['candidate_geometric_m']): number(x,y,'converted_height_m')
        u=math.fsum(c*b for c,b in zip(weights,derived['wind700_u_corners_mps'][w]))
        vcomp=math.fsum(c*b for c,b in zip(weights,derived['wind700_v_corners_mps'][w]))
        speed=math.sqrt(u*u+vcomp*vcomp); number(speed,row['wind700_mps'],'wind_reinterpolated_mps',2e-12)
        reasons=[]; n=None; H=None; regime=None; window=None; used=None; ids=[]
        if len(cands)==0: reasons.append('ZERO_LEVEL_NO_DOWN_CROSSING')
        if len(cands)+int(cross['cold_to_warm_count'][j,4])>1: reasons.append('ZERO_LEVEL_MULTIPLE_CROSSINGS')
        if int(cross['exact_zero_count'][j,4]): reasons.append('ZERO_LEVEL_EXACT_ZERO_AMBIGUITY')
        if int(cross['nonascending_pair_count'][j,4]): reasons.append('ZERO_LEVEL_NONASCENDING_PROFILE')
        if not math.isfinite(speed) or speed<=0: reasons.append('WIND_INVALID')
        if not reasons:
            strat=(cands[0]+360.-117.5)/1000.
            if strat<0: reasons.append('EQUIVALENT_LAYER_BELOW_ANTENNA')
            else:
                window=0. if theta==90 else strat*1000./(speed*math.tan(math.radians(theta)))
                n=math.ceil((3600.+window)/3600.)
                if a<0 or a+n>len(rain): reasons.append('MISSING_FORWARD_CONTEXT')
                elif any(r is None or r<0 or not math.isfinite(r) for r in rain[a:a+n]): reasons.append('INVALID_LIQUID_RAIN_CONTEXT')
                else:
                    ids=sorted({event_ids[q] for q in range(a,a+n) if rain[q]>0})
                    if any(not events[e]['complete'] for e in ids): reasons.append('CENSORED_WET_EVENT')
                    kinds={events[e]['kind'] for e in ids}
                    if len(kinds)>1: reasons.append('MIXED_RAIN_REGIMES')
                    if not reasons:
                        regime=next(iter(kinds)) if kinds else 'dry_window'
                        H=(cands[0]-117.5+(0. if regime=='convective' else 360.))/1000.
                        if H<0: reasons.append('EQUIVALENT_LAYER_BELOW_ANTENNA'); H=None; regime=None
                        else: used=0. if theta==90 else H*1000./(speed*math.tan(math.radians(theta)))
        b=components[t]
        number(b['gas_reference_db'],ref['gas_absolute_db'],'fixed_gas_reference',0.)
        number(b['gas_absolute_db']-b['gas_reference_db'],b['gas_relative_db'],'relative_gas_arithmetic',0.)
        if not all(math.isfinite(b[x]) for x in ('gas_relative_db','cloud_db')) or b['cloud_db']<0: reasons.append('FROZEN_GAS_CLOUD_INVALID')
        lo,hi=map(int,sampling['indptr'][i:i+2]); count=hi-lo
        if count<=0: reasons.append('NO_VALID_SAMPLING_SUPPORT')
        need(count==row['valid_record_count']==sample_meta[i]['physical_valid_records'],'original_sampling_count')
        need(not np.any(sampling['quality'][lo:hi] & 15),'quality_mask_retains_16_32')
        need(reasons==row['reasons'],'independent_exclusion_reasons')
        need(n==row['context_bins'] and ids==row['wet_event_ids'],'independent_context_membership')
        need(regime==row['regime'],'independent_selected_regime')
        for name,x in [('height_km',H),('conservative_lookahead_s',window),('used_lookahead_s',used)]:
            if x is None: need(row[name] is None,'null_'+name)
            else: number(x,row[name],name,2e-9 if name.endswith('_s') else 2e-10)
        need(row['eligible']==p['eligible']==(not reasons),'independent_eligibility')
        number(p['gas_relative_db'],b['gas_relative_db'],'inherited_gas_component',0.)
        number(p['cloud_db'],b['cloud_db'],'inherited_cloud_component',0.)
        eligible.append(not reasons); reasons_total.update(reasons)
        if reasons:
            need(all(p[name] is None for name in ('rain_power_mean_db','rain_arithmetic_mean_db','rain_anchor_db','rain_min_db','rain_max_db','total_relative_db')),'missing_not_zero')
            continue
        regimes[regime]+=1
        offsets=sampling['offset_ms'][lo:hi]
        need((np.diff(offsets.astype(np.int64))>0).all(),'unique_exact_offsets')
        A=horizontal_losses(rain[a:a+n],offsets,H,speed,theta,k,alpha)
        Ap=reduce_power(A); Adb=math.fsum(map(float,A))/len(A)
        anchor=float(horizontal_losses(rain[a:a+n],[0],H,speed,theta,k,alpha)[0])
        expected={'rain_power_mean_db':Ap,'rain_arithmetic_mean_db':Adb,'rain_anchor_db':anchor,
            'rain_min_db':float(A.min()),'rain_max_db':float(A.max()),'total_relative_db':math.fsum([b['gas_relative_db'],b['cloud_db'],Ap])}
        for name,x in expected.items(): number(p[name],x,name)
        need(expected['rain_min_db']-2e-10<=Ap<=Adb+2e-10,'power_mean_bounds')
        rain_answers[t]=Ap; total_answers[t]=expected['total_relative_db']; records_evaluated+=len(A)
    need(sum(eligible)==read(OUT/'prediction_complete.json')['computed_hours'],'computed_count')
    need(records_evaluated==read(OUT/'prediction_complete.json')['evaluated_unique_records'],'all_records_recomputed')
    prior_mask={epoch(r['hour_utc']):r['supplement_eligible'] for r in read(EXP/'Combined_Channel_Revision_027/all_hour_ledger.json')}
    common=[t for t,ok in zip(keys,eligible) if ok and prior_mask[t]]
    need(common==read(OUT/'comparison_mask.json')['eligible_hour_epoch_s'],'independent_same_sample_mask')
    masks=read(OUT/'mask_frozen_before_predictions.json')
    need(sha(OUT/'eligibility_ledger.json')==masks['ledger_sha256'] and not masks['truth_loaded'],'mask_seal')
    need(epoch(masks['utc'])<=epoch(read(OUT/'prediction_complete.json')['utc']),'mask_before_prediction_time')
    old_names=['ours_frozen_top10']+[f'bai2022_lstm_t0_adapted_seed{s}' for s in [17,29,43]]+[f'tabm2025_official_budgeted_seed{s}' for s in [17,29,43]]+['itu_combined_geometric_approx']
    frozen={int(r['hour_bjt'].timestamp()):r for r in pq.read_table(EXP/'Combined_Channel_Revision_012/selected_local.parquet',columns=['hour_bjt','truth_db',*old_names]).to_pylist()}
    source11={int(r['hour_bjt'].timestamp()):r for r in pq.read_table(EXP/'Combined_Channel_Revision_011/predictions_local.parquet',columns=['hour_bjt','truth_db',*old_names]).to_pylist()}
    need(set(frozen)==set(source11)==set(keys),'011_012_key_identity')
    need(all(frozen[t][n]==source11[t][n] for t in keys for n in ['truth_db',*old_names]),'011_012_prediction_identity')
    family_runs={
        'ours_frozen_top10':['ours_frozen_top10'],
        'bai2022_lstm_t0_adapted':[f'bai2022_lstm_t0_adapted_seed{s}' for s in [17,29,43]],
        'tabm2025_official_budgeted':[f'tabm2025_official_budgeted_seed{s}' for s in [17,29,43]],
        'itu_combined_geometric_approx':['itu_combined_geometric_approx'],
        'esst_hourly_adaptation':['esst_hourly_adaptation']}
    published_runs={(r['event'],r['run']):r for r in read(OUT/'run_metrics.json')}
    ranks=read(OUT/'seven_family_rankings.json'); coverage=read(OUT/'date_coverage.json')
    own_metrics={r['event']:r for r in read(OUT/'esst_own_coverage_metrics_unranked.json')}
    for event in ['rain','dry_overcast']:
        subset=[t for t in common if components[t]['event']==event]
        own=[t for t,ok in zip(keys,eligible) if ok and components[t]['event']==event]
        dates=sorted({datetime.fromtimestamp(t,timezone(timedelta(hours=8))).date().isoformat() for t in subset})
        c=next(r for r in coverage if r['event']==event)
        need(c['hour_epoch_s']==subset and c['dates_bjt']==dates and c['esst_eligible_n']==len(own),'exact_event_dates_and_keys')
        need(own_metrics[event]['n']==len(own) and own_metrics[event]['rank'] is None,'own_coverage_not_mixed_rank')
        if own:
            own_score=score([frozen[t]['truth_db'] for t in own],[total_answers[t] for t in own])
            for metric,x in own_score.items(): number(own_metrics[event]['metrics'][metric],x,'native_'+metric,1e-12)
        expected_means={}
        for family,names in family_runs.items():
            values=[]
            for name in names:
                value=score([frozen[t]['truth_db'] for t in subset],
                    [total_answers[t] if name=='esst_hourly_adaptation' else frozen[t][name] for t in subset]) if subset else None
                row=published_runs[event,name]
                need(row['family']==family and row['n']==len(subset),'run_scope')
                if value:
                    for metric,x in value.items(): number(row['metrics'][metric],x,'metric_'+metric,1e-12)
                    values.append(value)
                else: need(row['metrics'] is None,'empty_run_null')
            row=next(r for r in ranks if r['event']==event and r['family']==family)
            need(row['n']==len(subset) and row['n_dates']==len(dates) and row['dates_bjt']==dates and row['seeds']==len(names),'family_membership_dates_seeds')
            for metric in ['rmse_db','mae_db','bias_db']:
                if values:
                    mean=math.fsum(v[metric] for v in values)/len(values)
                    number(row['metrics_mean'][metric],mean,'mean_'+metric,1e-12)
                    if len(values)>1:
                        sd=math.sqrt(math.fsum((v[metric]-mean)**2 for v in values)/(len(values)-1))
                        number(row['metrics_sd'][metric],sd,'sd_'+metric,1e-12)
                    else: need(row['metrics_sd'][metric] is None,'single_run_sd_null')
                    if metric=='rmse_db': expected_means[family]=mean
                else: need(row['metrics_mean'][metric] is None,'empty_family_null')
        for row in [r for r in ranks if r['event']==event]:
            if row['family'] in expected_means:
                expected_rank=1+sum(v<expected_means[row['family']] for v in expected_means.values())
                need(row['rank']==expected_rank,'unrounded_competition_rank')
            else: need(row['rank'] is None and all(x is None for x in row['metrics_mean'].values()),'blocked_not_zero_rank')
    need(len(ranks)==14 and len(published_runs)==18,'seven_family_two_events_nine_runs')
    summary=read(OUT/'coverage_summary.json')
    need(summary['reasons']==dict(reasons_total) and summary['regime_counts']==dict(regimes),'coverage_and_reason_counts')
    need(summary['computed_hours']==sum(eligible) and summary['common_hours']==len(common),'coverage_counts')
    status=read(OUT/'model_status.json')
    need(status['nwp_mts']=='NOT_COMPUTED_USER_DECISION' and not status['all_seven_complete'],'remaining_model_states')
    for p,h in sources.items(): need(sha(p)==h,'source_after')
    for p,h in codes.items(): need(sha(ROOT/p)==h,'code_after')
    output=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),implementation='independent horizontal kilometre overlap for every retained timestamp; scalar event reconstruction; PyArrow joins and math.fsum metrics; no producer or legacy calculation imports',
        python=sys.version,protected_sources=len(sources),changed_sources=0,checked_mother_hours=687,
        checked_rain_context_hours=7823,checked_events=253,computed_hours=sum(eligible),common_hours=len(common),
        independently_evaluated_records=records_evaluated,maximum_differences=dict(ERRORS),checks=dict(CHECKS),
        predeclared_tolerances=dict(attenuation_db=2e-10,metrics_db=1e-12),independent_person_review=False)
    with (OUT/'independent_qa.json').open('x',encoding='utf-8') as f: json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k not in ('checks','python')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
