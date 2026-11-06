"""One-shot offline rescore of frozen predictions; no model/legacy imports."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_027'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_026'
P11 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_011'
P12 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'
WX = ROOT / 'Outputs/Datasets/Revision_002/era5_hourly_pressure_levels.parquet'
TEST = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet'
H16 = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/final_test.json'
FAMILIES = {
    'ours_frozen_top10': ['ours_frozen_top10'],
    'bai2022_lstm_t0_adapted': [f'bai2022_lstm_t0_adapted_seed{s}' for s in (17, 29, 43)],
    'tabm2025_official_budgeted': [f'tabm2025_official_budgeted_seed{s}' for s in (17, 29, 43)],
    'itu_combined_geometric_approx': ['itu_combined_geometric_approx'],
}
BLOCKED = ['esst_hourly_adaptation', 'rmop_atmospheric_branch', 'nwp_prm_mts_chain']
RUNS = [r for runs in FAMILIES.values() for r in runs]
EVENTS = ('rain', 'dry_overcast')
TP, SF = 'surface__total_precipitation', 'surface__snowfall'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def eligibility(total, snow):
    total, snow = np.asarray(total, dtype=float), np.asarray(snow, dtype=float)
    return np.isfinite(total) & np.isfinite(snow) & (total >= 0) & (snow >= 0) & ((total - snow) >= 0)


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    require(truth.shape == prediction.shape and truth.ndim == 1, 'metric shapes')
    require(len(truth) > 0 and np.isfinite(truth).all() and np.isfinite(prediction).all(), 'invalid metric group')
    error = prediction - truth
    return {'rmse_db': float(np.sqrt(np.mean(error ** 2))),
            'mae_db': float(np.mean(np.abs(error))), 'bias_db': float(np.mean(error))}


def unique_keys(frame):
    require(frame.hour_bjt.notna().all() and frame.hour_bjt.is_unique, 'duplicate/missing hour key')
    require(frame.hour_bjt.dt.tz is not None, 'timezone required')


def selftest():
    checks = []
    for title, total, snow, expected in [
        ('dry zero accepted', [0], [0], [True]),
        ('wet positive accepted', [2], [.5], [True]),
        ('snow equals total accepted', [2], [2], [True]),
        ('negative liquid rejected', [1], [2], [False]),
        ('tiny negative not clipped', [0], [1e-15], [False]),
        ('negative total rejected', [-1], [0], [False]),
        ('negative snow rejected', [1], [-1], [False]),
        ('nonfinite rejected', [np.nan, 1, np.inf], [0, np.inf, 1], [False] * 3),
    ]:
        require(eligibility(total, snow).tolist() == expected, title)
        checks.append(title)
    require(metrics([1, 2], [2, 1]) == {'rmse_db': 1., 'mae_db': 1., 'bias_db': 0.}, 'metric signs')
    checks.append('known metrics and bias sign')
    for title, y, p in [('empty group fails', [], []), ('NaN fails', [0], [np.nan]),
                        ('shape mismatch fails', [0], [0, 1])]:
        try:
            metrics(y, p)
        except ValueError:
            checks.append(title)
        else:
            raise AssertionError(title)
    for title, keys in [('duplicate key fails', pd.to_datetime(['2020-01-01T00:00:00Z'] * 2)),
                        ('naive timestamp fails', pd.to_datetime(['2020-01-01']))]:
        try:
            unique_keys(pd.DataFrame({'hour_bjt': keys}))
        except ValueError:
            checks.append(title)
        else:
            raise AssertionError(title)
    require(pd.Timestamp('2025-12-12T02:00:00+08:00').tz_convert('UTC') == pd.Timestamp('2025-12-11T18:00:00Z'), 'UTC conversion')
    checks.append('BJT UTC previous day conversion')
    require(1 + sum(x < 2 for x in [1, 1, 2, 3]) == 3, 'competition rank')
    checks.append('competition ties')
    put('test_results.json', {'status': 'PASS', 'count': len(checks), 'checks': checks})


def protect():
    guard = read(PREV / 'sources_before.json')
    require(all(sha(p) == h for p, h in guard.items()), 'prior protected source changed')
    m = read(PREV / 'delivery_manifest.json')
    require(sha(PREV / 'delivery_manifest.json') == read(PREV / 'final_verification.json')['manifest_sha256'], '026 seal')
    for name, item in m['files'].items():
        path = ROOT / name
        require(sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], name)
    extra = [WX, TEST, H16, ROOT / 'Outputs/Datasets/Revision_002/data_dictionary.csv',
             ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_016/run_itu_baseline.py',
             ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_016/PROTOCOL.md']
    for directory in [PREV, ROOT / 'Code/Combined_Channel_Experiments/Revision_026', P11, P12]:
        extra.extend(p for p in directory.rglob('*') if p.is_file())
    for path in extra:
        h = sha(path)
        require(str(path) not in guard or guard[str(path)] == h, 'extra source changed')
        guard[str(path)] = h
    put('sources_before.json', guard)
    put('code_before.json', {str(p.relative_to(ROOT)): sha(p) for p in CODE.iterdir() if p.is_file()})
    return guard


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'scoring_run').mkdir(exist_ok=False)
    put('execution.json', {'utc': utc(), 'pid': os.getpid(), 'argv': sys.argv,
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__,
        'thread': '01a0bda1-92d1-7832-9508-60f4e0ec1634', 'lock': 'LOCKED', 'network': False})
    guard = protect()
    selftest()
    # No outcome columns are read until the signed-input mask has been saved.
    base = pd.read_parquet(P11 / 'predictions_local.parquet', columns=['hour_bjt', 'event', 'rain', 'dry_overcast', 'common_score_eligible'])
    weather = pd.read_parquet(WX, columns=['hour_bjt', TP, SF])
    for frame in (base, weather):
        unique_keys(frame)
    require(len(base) == 687 and base.common_score_eligible.all(), 'original mother scope')
    weather['source_valid_time'] = weather.hour_bjt
    weather['hour_bjt'] = weather.hour_bjt - pd.Timedelta(hours=1)
    frame = base.merge(weather, on='hour_bjt', how='left', validate='one_to_one', indicator=True)
    require(frame['_merge'].eq('both').all(), 'missing shifted weather key')
    total, snow = frame[TP].to_numpy() * 1000., frame[SF].to_numpy() * 1000.
    liquid = total - snow
    valid = eligibility(total, snow)
    require(np.array_equal(np.isfinite(total) & np.isfinite(snow) & (total >= .1) & (snow < .1), frame.rain), 'rain label alignment')
    diagnostic = pd.read_parquet(TEST, columns=['hour_bjt', 'surface__zero_degree_level', 'surface__mean_total_precipitation_rate', 'surface__mean_snowfall_rate'])
    unique_keys(diagnostic)
    require(set(diagnostic.hour_bjt) == set(base.hour_bjt), '003 test scope')
    diagnostic = frame[['hour_bjt']].merge(diagnostic, on='hour_bjt', how='left', validate='one_to_one')
    missing_h0 = ~np.isfinite(diagnostic.surface__zero_degree_level.to_numpy())
    alt_rate = (diagnostic.surface__mean_total_precipitation_rate - diagnostic.surface__mean_snowfall_rate) * 3600
    ledger = []
    for i, row in frame.iterrows():
        reasons = []
        if not np.isfinite(total[i]) or not np.isfinite(snow[i]): reasons.append('NONFINITE_PRECIPITATION_INPUT')
        if total[i] < 0: reasons.append('NEGATIVE_TOTAL_PRECIPITATION')
        if snow[i] < 0: reasons.append('NEGATIVE_SNOWFALL')
        if liquid[i] < 0: reasons.append('NEGATIVE_LIQUID_DIFFERENCE')
        require(bool(valid[i]) == (not reasons), 'reason mask')
        ledger.append({'station': 'LOCAL', 'hour_utc': row.hour_bjt.tz_convert('UTC').isoformat(),
            'hour_bjt': row.hour_bjt.isoformat(), 'date_bjt': row.hour_bjt.date().isoformat(),
            'source_valid_utc': row.source_valid_time.tz_convert('UTC').isoformat(), 'event': row.event,
            'total_mm_in_one_hour': float(total[i]), 'snow_water_equivalent_mm_in_one_hour': float(snow[i]),
            'signed_liquid_proxy_mmph': float(liquid[i]), 'legacy_eligible': True,
            'supplement_eligible': bool(valid[i]), 'exclusion_reasons': reasons,
            'legacy_zero_height_fallback_used': bool(missing_h0[i])})
    require(int(valid.sum()) == 684 and len(ledger) == 687 and int(missing_h0.sum()) == 108, 'unexpected input census')
    put('all_hour_ledger.json', ledger)
    exclusions = [r for r in ledger if not r['supplement_eligible']]
    put('excluded_hours.json', exclusions)
    put('mask_contract.json', {'frozen_utc_before_outcome_read': utc(), 'protocol_sha256': sha(CODE / 'PROTOCOL.md'),
        'full_mother_hours': 687, 'supplement_hours': 684, 'excluded_hours': 3,
        'rule': 'finite(tp,sf) & tp>=0 & sf>=0 & (1000*tp-1000*sf)>=0 at source valid=t+1h',
        'excluded_by_event': {e: sum(r['event'] == e for r in exclusions) for e in ('rain', 'dry_overcast', 'other')},
        'alternate_current_t_mean_rate_negative_count_not_used': int((alt_rate < 0).sum()),
        'missing_zero_height_count_not_a_new_exclusion': 108, 'no_outcome_dependent_filter': True,
        'ledger_sha256': sha(OUT / 'all_hour_ledger.json'), 'unit_metadata': '002/data_dictionary.csv: tp=m, sf=m of water equivalent'})
    # Frozen prediction/target values are only used from this point onward.
    values = pd.read_parquet(P11 / 'predictions_local.parquet', columns=['hour_bjt', 'truth_db', 'itu838_geometric_proxy', *RUNS])
    unique_keys(values)
    frame = frame.merge(values, on='hour_bjt', how='left', validate='one_to_one')
    require(np.isfinite(frame[['truth_db', *RUNS]].to_numpy(dtype=float)).all(), 'missing outcome/run; cannot shrink mask')
    selected = pd.read_parquet(P12 / 'selected_local.parquet', columns=['hour_bjt', 'truth_db', *RUNS])
    unique_keys(selected)
    selected = frame[['hour_bjt']].merge(selected, on='hour_bjt', validate='one_to_one')
    require(np.array_equal(selected[['truth_db', *RUNS]], frame[['truth_db', *RUNS]]), '011/012 frozen identity')
    hist = read(H16)
    hist_map = {pd.Timestamp(k): i for i, k in enumerate(hist['hour_bjt'])}
    require(len(hist_map) == 687 and set(hist_map) == set(frame.hour_bjt), '016 key set')
    idx = [hist_map[k] for k in frame.hour_bjt]
    require(np.array_equal(np.maximum(liquid, 0), np.asarray(hist['aligned_liquid_rain_mmph'])[idx]), 'legacy clipped source identity')
    require(np.array_equal(frame.itu838_geometric_proxy, np.asarray(hist['results']['physics_geom_circular']['prediction_db'])[idx]), '016 to 011 rain prediction identity')
    put('trace_checks.json', {'status': 'PASS', 'legacy_clipped_input_matches_016': True,
        'frozen_016_rain_prediction_matches_011': True, 'frozen_011_012_targets_and_eight_runs_equal': True,
        'rain_event_keys_match_shifted_accumulation': True, 'source_interval': '[t,t+1h)',
        'frequency_ghz': 19.13, 'polarization': 'circular', 'target': 'clear-sky-relative attenuation dB',
        'historical_h0_fallback_m_not_refitted': hist['h0_train_median_m'], 'new_physical_predictions': 0})
    runs, families, memberships, coverage = [], [], {}, []
    old = {(r['event'], r['family']): r for r in read(P12 / 'seven_family_rankings.json') if r['cohort'] == 'LOCAL_test'}
    for cohort, eligible in [('LOCAL_legacy_687_reference', np.ones(687, dtype=bool)), ('LOCAL_liquid_input_valid_supplement', valid)]:
        for event in EVENTS:
            mask = eligible & frame.event.eq(event).to_numpy()
            subset = frame.loc[mask]
            dates = sorted({k.date().isoformat() for k in subset.hour_bjt})
            ids = sorted(k.tz_convert('UTC').isoformat() for k in subset.hour_bjt)
            memberships[cohort + '/' + event] = ids
            coverage.append({'cohort': cohort, 'event': event, 'n': len(subset), 'n_dates': len(dates), 'dates_bjt': dates})
            group = []
            for family, names in FAMILIES.items():
                scored = [metrics(subset.truth_db, subset[name]) for name in names]
                for name, value in zip(names, scored):
                    runs.append({'cohort': cohort, 'event': event, 'family': family, 'run': name, 'n': len(subset), **value})
                aggregate = {'cohort': cohort, 'event': event, 'family': family, 'status': 'SCORED_FROZEN',
                    'n': len(subset), 'n_runs': len(names), 'n_dates': len(dates), 'dates_bjt': dates}
                for metric in ('rmse_db', 'mae_db', 'bias_db'):
                    array = [r[metric] for r in scored]
                    aggregate[metric + '_mean'] = float(np.mean(array))
                    aggregate[metric + '_sd'] = float(np.std(array, ddof=1)) if len(array) > 1 else None
                group.append(aggregate)
            for row in group:
                row['rank'] = 1 + sum(other['rmse_db_mean'] < row['rmse_db_mean'] for other in group)
                if cohort == 'LOCAL_legacy_687_reference':
                    ref = old[(event, row['family'])]
                    require(row['n'] == ref['n'] and row['rank'] == ref['rank'] and dates == ref['dates_by_station']['LOCAL'], 'original table membership/rank')
                    for metric in ('rmse_db', 'mae_db', 'bias_db'):
                        for suffix in ('_mean', '_sd'):
                            k = metric + suffix
                            require((row[k] is None and ref[k] is None) or (row[k] is not None and ref[k] is not None and abs(row[k]-ref[k]) < 1e-12), 'original metric differs')
                families.append(row)
            for family in BLOCKED:
                families.append({'cohort': cohort, 'event': event, 'family': family, 'status': 'BLOCKED_NOT_COMPUTED',
                    'n': None, 'n_runs': None, 'rank': None, 'numeric_results': None, 'eligible_cohort_hours': len(subset)})
    put('run_metrics.json', runs)
    put('seven_family_rankings.json', families)
    put('sample_membership.json', memberships)
    put('date_coverage.json', coverage)
    status = read(PREV / 'model_status.json')
    status.update(revision='027', scope='LOCAL_SIGNED_LIQUID_INPUT_SUPPLEMENT', new_scores=16,
        supplementary_family_metric_rows=8, legacy_reference_run_rows_reverified=16,
        supplementary_score_source='seven_family_rankings.json')
    status['data_scope'].update(LOCAL_supplement_eligible_hours=684, LOCAL_supplement_excluded_hours=3,
        new_keywise_reconstruction=True, changed=False)
    for model in status['models'][:4]:
        model['supplement_scores_source'] = 'seven_family_rankings.json'
    put('model_status.json', status)
    changed = [p for p, h in guard.items() if sha(p) != h]
    require(not changed, 'source changed during scoring')
    put('calculation_complete.json', {'utc': utc(), 'status': 'PASS_PENDING_INDEPENDENT_QA',
        'full_LOCAL_hours': 687, 'supplement_eligible': 684, 'excluded': 3, 'run_metric_rows': len(runs),
        'supplement_run_rows': 16, 'family_rows': len(families), 'protected_sources': len(guard),
        'source_changes': changed, 'new_predictions': 0, 'new_training': 0, 'new_weather_requests': 0,
        'MTS': 'NOT_COMPUTED_USER_DECISION', 'worker_lock': 'RELEASED', 'delivery_lock': 'AWAITING_QA_AND_SEAL'})
    print(json.dumps(read(OUT / 'calculation_complete.json'), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        if OUT.is_dir() and not (OUT / 'failure.json').exists():
            put('failure.json', {'utc': utc(), 'error': error, 'preserve_all_outputs': True})
        raise
