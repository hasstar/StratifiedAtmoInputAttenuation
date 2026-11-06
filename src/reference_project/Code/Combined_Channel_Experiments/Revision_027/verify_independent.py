"""Independent scalar/Arrow audit; does not import production or legacy code."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import math
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_027'
P11 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_011'
P12 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'
FAMILIES = {
    'ours_frozen_top10': ('ours_frozen_top10',),
    'bai2022_lstm_t0_adapted': tuple('bai2022_lstm_t0_adapted_seed' + str(s) for s in [17, 29, 43]),
    'tabm2025_official_budgeted': tuple('tabm2025_official_budgeted_seed' + str(s) for s in [17, 29, 43]),
    'itu_combined_geometric_approx': ('itu_combined_geometric_approx',),
}
BLOCKED = {'esst_hourly_adaptation', 'rmop_atmospheric_branch', 'nwp_prm_mts_chain'}
RUNS = sum(FAMILIES.values(), ())
CHECKS = []
DELTAS = []
BJT = timezone(timedelta(hours=8))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(value, message):
    if not value:
        raise AssertionError(message)
    CHECKS.append(message)


def key(dt):
    assert dt.tzinfo is not None
    return dt.astimezone(timezone.utc)


def rows(path, columns):
    records = pq.read_table(path, columns=columns).to_pylist()
    result = {key(r['hour_bjt']): r for r in records}
    assert len(result) == len(records), 'duplicate original keys'
    return result


def near(a, b, context):
    if a is None or b is None:
        assert a is None and b is None, context
        return
    assert math.isfinite(a) and math.isfinite(b), context
    delta = abs(a-b)
    assert delta < 1e-12, (context, a, b)
    DELTAS.append(delta)


def scalar(y, p):
    assert len(y) == len(p) and len(y) > 0
    differences = [float(b)-float(a) for a, b in zip(y, p)]
    assert all(math.isfinite(x) for x in differences)
    return {'rmse_db': math.sqrt(math.fsum(x*x for x in differences)/len(y)),
            'mae_db': math.fsum(abs(x) for x in differences)/len(y),
            'bias_db': math.fsum(differences)/len(y)}


def main():
    assert not (OUT/'independent_qa.json').exists(), 'one shot verifier'
    assert not (OUT/'failure.json').exists(), 'preserved production failure blocks publication'
    guard = read(OUT/'sources_before.json')
    changed = [p for p, h in guard.items() if sha(p) != h]
    check(not changed, 'all protected source SHA unchanged before audit')
    check(all(sha(ROOT/p) == h for p, h in read(OUT/'code_before.json').items()), 'production-time source and protocol frozen')
    base = rows(P11/'predictions_local.parquet', ['hour_bjt', 'event', 'rain', 'dry_overcast', 'common_score_eligible', 'truth_db', 'itu838_geometric_proxy', *RUNS])
    chosen = rows(P12/'selected_local.parquet', ['hour_bjt', 'truth_db', *RUNS])
    weather = rows(ROOT/'Outputs/Datasets/Revision_002/era5_hourly_pressure_levels.parquet', ['hour_bjt', 'surface__total_precipitation', 'surface__snowfall'])
    test = rows(ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet', ['hour_bjt', 'surface__zero_degree_level', 'surface__mean_total_precipitation_rate', 'surface__mean_snowfall_rate'])
    check(len(base) == 687 and set(base) == set(chosen) == set(test), 'original 687 distinct UTC test keys preserved')
    check(all(r['common_score_eligible'] for r in base.values()), 'legacy mask includes all 687')
    check(all(base[k][c] == chosen[k][c] for k in base for c in ['truth_db', *RUNS]), '011 and 012 target/eight frozen predictions identical by key')
    check(all(math.isfinite(base[k][c]) for k in base for c in ['truth_db', *RUNS]), 'no silent outcome/run-specific shrinking')
    hist = read(ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/final_test.json')
    hmap = {key(datetime.fromisoformat(k)): i for i, k in enumerate(hist['hour_bjt'])}
    check(len(hmap) == 687 and set(hmap) == set(base), 'legacy rain result keys identical')
    with (ROOT/'Outputs/Datasets/Revision_002/data_dictionary.csv').open(encoding='utf-8-sig', newline='') as f:
        dictionary = list(csv.reader(f))
    check(any(r[0] == 'era5_hourly_pressure_levels' and 'surface__total_precipitation' in r and 'm' in r for r in dictionary), 'total precipitation unit m')
    check(any(r[0] == 'era5_hourly_pressure_levels' and 'surface__snowfall' in r and 'm of water equivalent' in r for r in dictionary), 'snow unit water equivalent m')
    ledger = read(OUT/'all_hour_ledger.json')
    lmap = {datetime.fromisoformat(r['hour_utc']): r for r in ledger}
    check(len(ledger) == len(lmap) == 687 and set(lmap) == set(base), 'full ledger preserves mother set')
    valid_keys, rejected, missing_h0, alternative_negative = set(), [], 0, 0
    for k, original in base.items():
        source = weather[k + timedelta(hours=1)]
        total = float(source['surface__total_precipitation'])*1000
        snow = float(source['surface__snowfall'])*1000
        finite = math.isfinite(total) and math.isfinite(snow)
        valid = finite and total >= 0 and snow >= 0 and total >= snow
        raw = total-snow
        reasons = []
        if not finite: reasons.append('NONFINITE_PRECIPITATION_INPUT')
        if total < 0: reasons.append('NEGATIVE_TOTAL_PRECIPITATION')
        if snow < 0: reasons.append('NEGATIVE_SNOWFALL')
        if total < snow: reasons.append('NEGATIVE_LIQUID_DIFFERENCE')
        actual = lmap[k]
        assert actual['source_valid_utc'] == (k+timedelta(hours=1)).isoformat()
        assert actual['hour_bjt'] == k.astimezone(BJT).isoformat()
        assert actual['date_bjt'] == k.astimezone(BJT).date().isoformat()
        assert actual['event'] == original['event']
        assert actual['supplement_eligible'] == valid and actual['exclusion_reasons'] == reasons
        assert actual['total_mm_in_one_hour'] == total and actual['snow_water_equivalent_mm_in_one_hour'] == snow
        assert actual['signed_liquid_proxy_mmph'] == raw and actual['legacy_eligible'] is True
        assert original['rain'] == (finite and total >= .1 and snow < .1)
        expected_event = 'rain' if original['rain'] else 'dry_overcast' if original['dry_overcast'] else 'other'
        assert original['event'] == expected_event
        hi = hmap[k]
        assert hist['aligned_liquid_rain_mmph'][hi] == max(raw, 0.)
        assert hist['results']['physics_geom_circular']['prediction_db'][hi] == original['itu838_geometric_proxy']
        h = test[k]['surface__zero_degree_level']
        fallback = h is None or not math.isfinite(h)
        assert actual['legacy_zero_height_fallback_used'] == fallback
        missing_h0 += fallback
        alternative_negative += test[k]['surface__mean_total_precipitation_rate'] < test[k]['surface__mean_snowfall_rate']
        if valid:
            valid_keys.add(k)
        else:
            rejected.append(actual)
    check(len(valid_keys) == 684 and len(rejected) == 3, 'independent source-based 684/3 mask reconstructed')
    check({r['hour_utc'] for r in rejected} == {r['hour_utc'] for r in read(OUT/'excluded_hours.json')}, 'three exact excluded hours preserved')
    check(sum(r['event'] == 'dry_overcast' for r in rejected) == 2 and sum(r['event'] == 'other' for r in rejected) == 1, 'excluded events two overcast one other')
    check(missing_h0 == 108 and alternative_negative == 5, 'fallback and alternate mean-rate diagnostic not used as mask')
    contract = read(OUT/'mask_contract.json')
    check(contract['protocol_sha256'] == sha(ROOT/'Code/Combined_Channel_Experiments/Revision_027/PROTOCOL.md') and contract['ledger_sha256'] == sha(OUT/'all_hour_ledger.json'), 'mask contract and ledger SHA frozen')
    run_rows = read(OUT/'run_metrics.json')
    family_rows = read(OUT/'seven_family_rankings.json')
    membership = read(OUT/'sample_membership.json')
    coverage = {(r['cohort'], r['event']): r for r in read(OUT/'date_coverage.json')}
    old = {(r['event'], r['family']): r for r in read(P12/'seven_family_rankings.json') if r['cohort'] == 'LOCAL_test'}
    check(len(run_rows) == 32 and len(family_rows) == 28, 'exact two separately identified mask tables')
    for cohort, eligible in [('LOCAL_legacy_687_reference', set(base)), ('LOCAL_liquid_input_valid_supplement', valid_keys)]:
        for event in ('rain', 'dry_overcast'):
            keys = sorted(k for k in eligible if base[k]['event'] == event)
            expected_n = 51 if event == 'rain' else 158 if cohort == 'LOCAL_legacy_687_reference' else 156
            check(len(keys) == expected_n, cohort+'/'+event+' count')
            dates = sorted({k.astimezone(BJT).date().isoformat() for k in keys})
            assert membership[cohort+'/'+event] == [k.isoformat() for k in keys]
            assert coverage[(cohort,event)] == {'cohort':cohort, 'event':event,'n':len(keys),'n_dates':len(dates),'dates_bjt':dates}
            group = [r for r in family_rows if r['cohort'] == cohort and r['event'] == event]
            assert len(group) == 7 and {r['family'] for r in group} == set(FAMILIES) | BLOCKED
            computed = {}
            for family, names in FAMILIES.items():
                perrun = []
                for run in names:
                    reference = scalar([base[k]['truth_db'] for k in keys], [base[k][run] for k in keys])
                    found = [r for r in run_rows if r['cohort'] == cohort and r['event'] == event and r['run'] == run]
                    assert len(found) == 1 and found[0]['family'] == family and found[0]['n'] == len(keys)
                    for metric, value in reference.items(): near(value, found[0][metric], run+'/'+metric)
                    perrun.append(reference)
                row = next(r for r in group if r['family'] == family)
                assert row['n'] == len(keys) and row['n_runs'] == len(names) and row['n_dates'] == len(dates) and row['dates_bjt'] == dates
                for metric in ('rmse_db','mae_db','bias_db'):
                    vals = [r[metric] for r in perrun]
                    mean = math.fsum(vals)/len(vals)
                    sd = math.sqrt(math.fsum((v-mean)**2 for v in vals)/(len(vals)-1)) if len(vals)>1 else None
                    near(mean, row[metric+'_mean'], family+'/'+metric+'/mean')
                    near(sd, row[metric+'_sd'], family+'/'+metric+'/sd')
                    if cohort == 'LOCAL_legacy_687_reference':
                        near(mean, old[(event,family)][metric+'_mean'], '012 legacy metric identity')
                        near(sd, old[(event,family)][metric+'_sd'], '012 legacy sd identity')
                computed[family] = math.fsum(r['rmse_db'] for r in perrun)/len(perrun)
            for row in group:
                if row['family'] in BLOCKED:
                    assert all(row.get(f) is None for f in ('n','n_runs','rank','numeric_results'))
                    assert row['eligible_cohort_hours'] == len(keys)
                else:
                    assert row['rank'] == 1+sum(v<computed[row['family']] for v in computed.values())
                    if cohort == 'LOCAL_legacy_687_reference':
                        assert row['rank'] == old[(event,row['family'])]['rank'] and dates == old[(event,row['family'])]['dates_by_station']['LOCAL']
    check(True, 'all run/family metrics SD ranks dates independently reproduced; blocked rows null')
    status = read(OUT/'model_status.json')
    check(status['formal_family_count'] == 7 and len(status['models']) == 7 and not status['current_required_work_complete'] and not status['all_seven_original_models_complete'], 'seven-family boundary and incomplete status')
    mts = next(r for r in status['models'] if r['id'] == 'nwp_chain')['branches']['mts_seconds']
    check(mts['status'] == 'NOT_COMPUTED_USER_DECISION' and mts['numeric_results'] is None and mts['rank'] is None and not mts['awaiting_user_materials'] and not mts['required_for_current_delivery'], 'MTS left uncomputed by decision')
    check(status['data_scope']['uk_weather_hours'] == 5664 and status['data_scope']['uk_common_physics_mask'] == 5538 and status['data_scope']['uk_eligible_targets'] == 5593 and status['data_scope']['uk_days'] == {'chilbolton':143,'sparsholt':93}, 'UK scope unchanged not recomputed')
    check(read(OUT/'test_results.json')['status'] == 'PASS' and read(OUT/'test_results.json')['count'] == 16, 'sixteen production boundary tests pass')
    changed = [p for p,h in guard.items() if sha(p) != h]
    check(not changed, 'all protected sources unchanged after audit')
    result = {'utc':datetime.now(timezone.utc).isoformat(), 'argv':sys.argv, 'status':'PASS',
        'method':'PyArrow UTC dictionary joins; scalar math.fsum; no producer/legacy module imports',
        'protected_sources':len(guard),'source_changes':changed,'checks':CHECKS,
        'numeric_comparisons':len(DELTAS),'max_abs_metric_difference':max(DELTAS),
        'mother_hours':687,'supplement_hours':684,'excluded_hours':3,'run_rows':32,'family_rows':28,
        'UK_scores_recomputed':False,'new_predictions':0,'new_training':0,'new_weather_requests':0,
        'MTS':'NOT_COMPUTED_USER_DECISION','external_peer_review':False}
    with (OUT/'independent_qa.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
