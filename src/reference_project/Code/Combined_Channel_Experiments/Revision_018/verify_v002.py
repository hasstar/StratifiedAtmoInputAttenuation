"""Independent all-hour native, temporal arithmetic and bounded generator replay QA."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib, hashlib, importlib.util, json, math, msvcrt, time, traceback
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_018'
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010/dynamic_evaluation_002'
FAST = ROOT / 'Outputs/Experiments/Fast_Fading_Revision_002'
LAGS = (1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1800)
INDEPENDENT = ROOT / 'Code/Combined_Channel_Experiments/Revision_001/verify_gbs_dynamics_v001.py'


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def equal(a, b, tol=1e-9):
    return bool(np.allclose(a, b, rtol=0, atol=tol, equal_nan=True))


def average(values):
    v = [float(x) for x in values if np.isfinite(x)]
    return math.fsum(v) / len(v) if v else np.nan


def forbid(*args, **kwargs):
    raise RuntimeError('Historical main/write forbidden')


@contextlib.contextmanager
def locks():
    handles = []
    try:
        for p in (OUT / 'execution.lock', FAST / 'worker.lock'):
            f = p.open('r+b'); handles.append(f)
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    finally:
        for f in reversed(handles):
            f.close()
        emit('verification_lock_release.json', {'utc': datetime.now(timezone.utc).isoformat(), 'closed_handles': len(handles)})


def check_units_origin(path, timestamp):
    import re
    import netCDF4
    with netCDF4.Dataset('independent_units.nc', memory=path.read_bytes()) as ds:
        tv, v = ds['time'], ds['clearsky_attenuation']
        need(str(v.units) == 'dB' and float(v.integration_period) == 1, 'Independent native units')
        need('end of integration' in str(tv.long_name).lower(), 'Independent interval convention')
        need(abs(float(ds['frequency'][...]) - 20.7) < 1e-4, 'Independent native GHz')
        parsed = re.fullmatch(r'seconds since (\d+)-(\d+)-(\d+) 0:0:0\.0+ \+0:00', str(tv.units))
        need(parsed is not None and tuple(map(int, parsed.groups())) == (timestamp.year, timestamp.month, timestamp.day), 'Independent UTC origin')


def main():
    begin = time.monotonic()
    need(not (OUT / 'independent_qa.json').exists(), 'QA already complete')
    with locks():
        complete = read(OUT / 'calculation_complete.json')
        need(complete['status'] == 'COMPUTED_PENDING_INDEPENDENT_QA', 'Complete gate')
        for p, h in complete['outputs_sha256'].items():
            need(sha(ROOT / p) == h, 'Calculation output changed ' + p)
        protected = read(OUT / 'sources_before.json')
        for p, h in protected.items():
            need(sha(p) == h, 'Protected input changed ' + p)
        need(sha(INDEPENDENT) == '8c9737f0312faae62bdb5df8fc08a2883dacd7a8d29524d78aed968681e134fd', 'Verifier pin')
        spec = importlib.util.spec_from_file_location('independent_temporal_primitives', INDEPENDENT)
        independent = importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)
        independent.main = independent.emit = forbid
        # Three independent controls, including explicit interval holes and cross-replication product.
        mask = np.ones(3600, bool); mask[2] = False
        need(0 not in independent.pairs(mask)[1] and 3 in independent.pairs(mask)[0], 'Independent gap control')
        g = np.array([[0.], [2.]])
        need(abs(independent.score(np.ones(1), np.ones(1), g)[0] + 1) < 1e-12, 'Negative corrected estimate')
        idx = independent.pairs(np.ones(3600, bool))
        obs = np.arange(3600, dtype=float)
        need(equal(independent.means(obs, idx)[0], LAGS) and
             equal(independent.means(obs + 19, idx)[0], LAGS), 'Constant and known-increment control')
        current = pd.read_parquet(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012/selected_uk.parquet')
        raw_frame = pd.read_parquet(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet')
        ledger = pd.read_parquet(OUT / 'all_uk_hour_ledger.parquet')
        keys = ['station', 'hour_utc']
        need(len(current) == len(ledger) == 5664 and not ledger.duplicated(keys).any(), 'Complete ledger')
        a, b = current.sort_values(keys).reset_index(drop=True), ledger.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b[current.columns], check_exact=True)
        expected = current[current.common_score_eligible & current.event.isin(['rain', 'dry_overcast'])].merge(
            raw_frame[['station', 'hour_utc', 'source_file', 'valid_seconds', 'primary_score_eligible', 'liquid_negative']],
            on=keys, validate='one_to_one').sort_values(keys)
        need(len(expected) == 3027 and expected.event.value_counts().to_dict() == {'dry_overcast': 2292, 'rain': 735}, 'Selected keys')
        selected_keys = set(map(tuple, expected[keys].itertuples(index=False, name=None)))
        need(set(map(tuple, ledger.loc[ledger.temporal_eligible, keys].itertuples(index=False, name=None))) == selected_keys, 'Ledger eligibility')
        need(expected.primary_score_eligible.all() and not expected.liquid_negative.any(), 'Quality/common cohort exclusion')
        need(ledger.groupby('station').date.nunique().to_dict() == {'chilbolton': 143, 'sparsholt': 93}, 'Full date counts')
        expected_excl = ['outside_rain_or_dry_overcast' if r.event not in ('rain', 'dry_overcast') else
                         'outside_existing_physical_common_cohort' if not r.common_score_eligible else ''
                         for r in ledger.itertuples(index=False)]
        need(expected_excl == ledger.temporal_exclusion.tolist(), 'Exclusion ledger')
        cov = pd.read_parquet(OUT / 'coverage_by_date.parquet').set_index(['station', 'utc_date'])
        for (s, d), grp in ledger.groupby(['station', 'date']):
            c = cov.loc[(s, d)]
            need(c.acquired_hours == len(grp) == 24 and c.common_hours == grp.common_score_eligible.sum()
                 and c.temporal_hours == grp.temporal_eligible.sum(), 'Date coverage')
        old_meta = {}
        for item in read(OLD / 'shards.json'):
            with np.load(OLD / item['file']) as z:
                for i, m in enumerate(json.loads(str(z['metadata']))):
                    old_meta[(m['station'], m['hour_utc'])] = (item, i, m)
        index = read(OUT / 'combined_shard_index.json')
        ix = {(x['station'], x['hour_utc']): x for x in index}
        need(len(ix) == len(index) == 3027 and
             {(s, pd.Timestamp(h)) for s, h in ix} == selected_keys, 'Combined index exact scope')
        mode_counts = {}
        independently_new = []
        for r in expected.itertuples(index=False):
            k = (r.station, r.hour_utc.isoformat()); e = ix[k]; old = old_meta.get(k)
            mode = 'new_hour' if old is None else 'reuse_unchanged' if old[2]['event'] == r.event else 'current_weather_changed'
            need(e['mode'] == mode and e['event'] == r.event and e['source_file'] == r.source_file, 'Current event/source')
            need(e['key'] == 'GBS-external-v010-' + r.station + '-' + r.hour_utc.isoformat(), 'Stable key')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if mode == 'reuse_unchanged':
                need((ROOT / e['shard']).resolve() == (OLD / old[0]['file']).resolve() and e['row'] == old[1]
                     and e['shard_sha256'] == old[0]['sha256'], 'Direct unchanged shard reuse')
            else:
                need((ROOT / e['shard']).resolve().is_relative_to((OUT / 'days').resolve()), 'New shard scope')
                independently_new.append(r)
        need(mode_counts == read(OUT / 'reuse_plan.json')['counts'], 'Reuse counts')
        need(mode_counts == {'new_hour': 2094, 'reuse_unchanged': 932, 'current_weather_changed': 1}, 'Frozen count contract')
        new_frame = pd.DataFrame(independently_new)
        expected_replay = {(g.iloc[0].station, g.iloc[0].hour_utc.isoformat()) for _, g in new_frame.groupby(['station', 'event'])}
        expected_replay.add(('sparsholt', '2004-03-15T08:00:00+00:00'))
        replay = {(x['station'], x['hour_utc']) for x in read(OUT / 'replay_plan.json')}
        need(replay == expected_replay and all(ix[k]['mode'] != 'reuse_unchanged' for k in replay), 'Predeclared new-only replay')
        hourly = pd.read_parquet(OUT / 'hourly_scores.parquet').set_index(keys + ['candidate'])
        lag = pd.read_parquet(OUT / 'lag_diagnostics.parquet').set_index(keys + ['candidate', 'lag_s'])
        need(len(hourly) == 6054 and hourly.index.is_unique and len(lag) == 66594 and lag.index.is_unique, 'Metric scope')
        models, pools = independent.libraries()
        source_evidence = {x['source']: x['sha256'] for x in read(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json')}
        seen_shards = {}
        n_hours = n_pairs = n_replay = invalid_times = valid_seconds = 0
        max_score = max_moment = max_acf = 0.
        for source, group in expected.groupby('source_file', sort=True):
            raw = ROOT / '外部样例数据CEDA/GBS_20p7GHz' / source
            need(sha(raw) == source_evidence[str(raw.relative_to(ROOT))], 'Native source SHA')
            check_units_origin(raw, group.hour_utc.iloc[0])
            day, nbadtime = independent.load_day(raw); invalid_times += nbadtime
            for r in group.itertuples(index=False):
                k = (r.station, r.hour_utc.isoformat()); e = ix[k]; shard = ROOT / e['shard']
                if e['shard'] not in seen_shards:
                    need(sha(shard) == e['shard_sha256'], 'Shard SHA')
                    with np.load(shard) as z:
                        seen_shards[e['shard']] = {x: z[x].copy() for x in z.files if x != 'generation_metadata'}
                z = seen_shards[e['shard']]; i = e['row']
                meta = json.loads(str(z['metadata']))[i]
                need(all(meta[f] == e[f] for f in ['station', 'hour_utc', 'source_file', 'event', 'key']), 'Stored metadata')
                y = day[r.hour_utc.hour]; indices = independent.pairs(np.isfinite(y))
                o1, o2, acf = independent.means(y, indices)
                ns = int(np.isfinite(y).sum()); valid_seconds += ns
                linear_mean = average(np.exp(-math.log(10.) * y / 10.))
                power_truth = -10. * math.log(linear_mean) / math.log(10.)
                need(ns == r.valid_seconds and abs(power_truth - r.truth_db) < 1e-9, 'Native power-mean target/seconds')
                need(equal(o1, z['observed_mean'][i]) and equal(o2, z['observed_second'][i])
                     and equal(acf, z['observed_acf'][i], 1e-8), 'Independent raw moments')
                need(np.array_equal([len(q) for q in indices], z['pair_counts'][i]), 'Literal interval pairs')
                n_pairs += sum(map(len, indices)); n_hours += 1
                gm, ga = z['generated_mean'][i], z['generated_acf'][i]
                need(gm.shape == ga.shape == (2, 32, 11), 'Realization shape')
                for j, candidate in enumerate(('iid', 'block_0256')):
                    h = hourly.loc[(r.station, r.hour_utc, candidate)]
                    need(h.event == r.event and h.calculation_mode == e['mode'] and h.n_seconds == ns
                         and h.n_lags == np.isfinite(o1).sum(), 'Hourly metadata')
                    for count in (16, 32):
                        vs, plugin = independent.score(o1, o2, gm[j, :count])
                        max_score = max(max_score, abs(vs - h[f'vs{count}']), abs(plugin - h[f'plugin{count}']))
                        need(equal([vs, plugin], [h[f'vs{count}'], h[f'plugin{count}']]), 'Independent score')
                    for l, lag_s in enumerate(LAGS):
                        a = lag.loc[(r.station, r.hour_utc, candidate, lag_s)]
                        need(a.n_pairs == len(indices[l]) and a.event == r.event and a.calculation_mode == e['mode'], 'Lag metadata')
                        need(equal([a.observed_increment_db, a.observed_acf, a.generated_increment_db, a.generated_acf],
                                   [o1[l], acf[l], average(gm[j, :, l]), average(ga[j, :, l])], 1e-8), 'Lag metrics')
                        need(a.generated_acf_finite_reps == np.isfinite(ga[j, :, l]).sum(), 'Missing ACF count')
                    if k in replay:
                        for rep in range(32):
                            q = independent.generate(r.event, e['key'], rep, candidate, models, pools)
                            im, _, ia = independent.means(-10 * np.log10(q), indices)
                            max_moment = max(max_moment, float(np.nanmax(np.abs(im - gm[j, rep]))))
                            if np.isfinite(ia).any():
                                max_acf = max(max_acf, float(np.nanmax(np.abs(ia - ga[j, rep]))))
                            need(equal(im, gm[j, rep], 1e-10) and equal(ia, ga[j, rep], 1e-8), 'Independent frozen generator')
                            n_replay += 1
            print(json.dumps({'verified_hours': n_hours, 'total': 3027, 'elapsed_s': round(time.monotonic()-begin, 1)}), flush=True)
        def check_summary(records, table, columns):
            for row in records:
                group = table[table.candidate.eq(row['candidate']) & table.event.eq(row['event'])]
                if row['station'] != 'pooled':
                    group = group[group.station.eq(row['station'])]
                if 'lag_s' in row:
                    group = group[group.lag_s.eq(row['lag_s'])]
                need(len(group) == row['n_station_hours'] and group.date.nunique() == row['n_dates'], 'Summary denominators')
                need(group.hour_utc.min().isoformat() == row['start_utc'] and group.hour_utc.max().isoformat() == row['end_utc'], 'Summary UTC range')
                for col in columns:
                    vals = group[col] if row['weighting'] == 'hour_equal' else [average(g[col]) for _, g in group.groupby('date')]
                    value = average(vals)
                    need(row[col + '_finite_hours'] == group[col].notna().sum(), 'Summary finite denominators')
                    need((not np.isfinite(value) and row[col] is None) or equal(value, row[col]), 'Independent summary')
        score_summary = read(OUT / 'summary.json')['rows']; lag_summary = read(OUT / 'lag_summary.json')
        need(len(score_summary) == 24 and len(lag_summary) == 264, 'Summary shape')
        check_summary(score_summary, hourly.reset_index(), ['vs32', 'vs16', 'plugin32', 'plugin16'])
        check_summary(lag_summary, lag.reset_index(), ['observed_increment_db', 'generated_increment_db', 'observed_acf', 'generated_acf'])
        need(n_hours == 3027 and n_replay == len(replay) * 64, 'QA accounting')
        for p, h in protected.items():
            need(sha(p) == h, 'Post-QA protected source')
        emit('independent_qa.json', {'status': 'PASS_EXTENDED_FROZEN_TEMPORAL_DIAGNOSTIC',
             'utc': datetime.now(timezone.utc).isoformat(), 'all_ledger_hours': 5664, 'verified_hours': n_hours,
             'valid_native_seconds': valid_seconds, 'gap_safe_pairs': n_pairs, 'invalid_time_records_on_used_source_days': invalid_times,
             'reused_hours': 932, 'new_generation_hours': 2095, 'weather_condition_changes': 1,
             'independent_replay_hours': len(replay), 'independent_replay_realizations': n_replay,
             'score_summary_rows': len(score_summary), 'lag_summary_rows': len(lag_summary), 'independent_controls': 3,
             'max_score_difference_db2': max_score, 'max_replay_increment_difference_db': max_moment,
             'max_replay_acf_difference': max_acf, 'protected_sources_unchanged': len(protected),
             'verifier_sha256': sha(Path(__file__)), 'source_completion_sha256': sha(OUT / 'calculation_complete.json'),
             'method': 'Independent interval enumeration, explicit cross-replicate product, independent generator replay; not external peer review',
             'new_training_calls': 0, 'new_weather_downloads': 0, 'new_slow_predictions': 0,
             'all_seven_complete': False, 'full_MTS_complete': False, 'elapsed_s': time.monotonic() - begin})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (OUT / 'verification_failure.json').exists():
            emit('verification_failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
