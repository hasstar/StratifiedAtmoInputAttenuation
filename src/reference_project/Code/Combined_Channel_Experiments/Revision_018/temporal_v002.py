"""Extend frozen UK temporal diagnostics, reusing unchanged old hours. No fitting."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib, hashlib, importlib.util, json, msvcrt, sys, time, traceback
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_018'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_017'
BASE = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OLD = BASE / 'dynamic_evaluation_002'
FAST = ROOT / 'Outputs/Experiments/Fast_Fading_Revision_002'
LEGACY = ROOT / 'Code/Combined_Channel_Experiments/Revision_001/evaluate_gbs_dynamics_v002.py'
LEGACY_SHA = '856967bece532192c918f3cb39e8bdb06d88e5b6a83115f993cb94cecf53ac37'
CANDIDATES = ('iid', 'block_0256')
LAGS = (1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1800)


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def parquet(name, frame):
    with (OUT / name).open('xb') as stream:
        frame.to_parquet(stream, index=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def forbid(*args, **kwargs):
    raise RuntimeError('Historical mutation / main entry point forbidden')


@contextlib.contextmanager
def locks():
    handles = []
    try:
        for p, mode in ((OUT / 'execution.lock', 'x+b'), (FAST / 'worker.lock', 'r+b')):
            f = p.open(mode)
            handles.append(f)
            if mode == 'x+b':
                f.write(b'0'); f.flush()
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    finally:
        for f in reversed(handles):
            f.close()
        emit('calculation_lock_release.json', {'utc': utc(), 'closed_handles': len(handles)})


def protected_sources():
    protected = {}
    def pin(p, expected=None):
        p = Path(p).resolve()
        h = sha(p)
        need(expected is None or h == expected, 'SHA mismatch: ' + str(p))
        need(str(p) not in protected or protected[str(p)] == h, 'Conflicting pin')
        protected[str(p)] = h
    for p, h in read(PREV / 'sources_before.json').items():
        pin(p, h)
    for p, h in read(PREV / 'delivery_manifest.json')['files'].items():
        pin(ROOT / p, h)
    for p, h in read(OLD / 'protected_inputs.json').items():
        pin(ROOT / p, h)
    for name in ('sources_before.json', 'delivery_manifest.json', 'final_verification.json'):
        pin(PREV / name)
    for name in ('protected_inputs.json', 'completion.json', 'hourly.parquet', 'shards.json'):
        pin(OLD / name)
    old_qa = BASE / 'dynamic_independent_qa_001/qa.json'
    pin(old_qa)
    need(read(old_qa)['status'] == 'PASS_EXTERNAL_FROZEN_TEMPORAL_EVALUATION', 'Old independent gate')
    for item in read(OLD / 'shards.json'):
        pin(OLD / item['file'], item['sha256'])
    for p in Path(__file__).parent.glob('*.py'):
        pin(p)
    pin(OUT / 'PROTOCOL.md')
    pin(LEGACY, LEGACY_SHA)
    for p in ('Outputs/Experiments/Combined_Channel_Revision_012/selected_uk.parquet',
              'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet',
              'Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json',
              'Code/Combined_Channel_Experiments/Revision_001/verify_gbs_dynamics_v001.py'):
        pin(ROOT / p)
    for item in read(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003/native_label_sources.json'):
        pin(ROOT / item['source'], item['sha256'])
    saved = read(FAST / 'final_generator/model.json')
    pin(FAST / 'final_generator/model.json')
    for c in saved['components'].values():
        pin(FAST / c['cdf_file_relative_to_revision_output'], c['cdf_sha256'])
        pin(FAST / 'final_generator' / c['library_file'], c['library_sha256'])
    return protected


def old_index():
    index, arrays = {}, {}
    for item in read(OLD / 'shards.json'):
        path = OLD / item['file']
        with np.load(path) as z:
            arrays[str(path)] = {k: z[k].copy() for k in
                ('observed_mean', 'observed_second', 'observed_acf', 'pair_counts', 'generated_mean', 'generated_acf')}
            meta = json.loads(str(z['metadata']))
        for i, m in enumerate(meta):
            key = (m['station'], pd.Timestamp(m['hour_utc']))
            need(key not in index, 'Duplicate old key')
            index[key] = dict(path=str(path), row=i, sha256=item['sha256'], metadata=m)
    return index, arrays


def controls(dyn, legacy):
    valid = np.ones(3600, bool); valid[2] = False
    masks = dyn.pair_masks(valid)
    need(not masks[1][0] and masks[0][3], 'Gap control')
    full = dyn.pair_masks(np.ones(3600, bool)); f = np.arange(3600, dtype=float)
    a, b, acf = dyn.increments(f, full)
    need(np.array_equal(a, LAGS), 'Known increments')
    need(np.allclose(dyn.increments(f + 21.3, full)[0], a, atol=1e-12, rtol=0), 'Slow cancellation')
    need(abs(dyn.score_from_moments(a, b, np.tile(a, (32, 1)))['vs']) < 1e-10, 'Known score')
    need(np.isnan(dyn.increments(np.ones(3600), full)[2]).all(), 'Zero variance ACF missing')
    g, _ = legacy.matrix_increments(np.array([f, f + 2]), full)
    need(np.array_equal(g, np.tile(LAGS, (2, 1))), 'Vector increments')
    s = dyn.score_from_moments(np.ones(11), np.ones(11), np.tile([[0.], [2.]], (1, 11)))
    need(s['vs'] < 0, 'Unbiased negative estimate retained')
    power_target = -10 * np.log10(np.mean(10 ** (-np.array([0., 10.]) / 10)))
    need(abs(power_target - (-10 * np.log10(.55))) < 1e-12 and abs(power_target - 5.) > 2., 'Power-domain target, not mean dB')
    emit('tests.json', {'status': 'PASS', 'synthetic_controls': 8, 'not_observational_results': True})


def summaries(frame, columns):
    rows = []
    for station in ('pooled', 'chilbolton', 'sparsholt'):
        site = frame if station == 'pooled' else frame[frame.station.eq(station)]
        for event in ('rain', 'dry_overcast'):
            for candidate in CANDIDATES:
                group = site[site.event.eq(event) & site.candidate.eq(candidate)]
                need(len(group) > 0, 'Empty intended stratum')
                for weighting in ('date_equal', 'hour_equal'):
                    ag = group.groupby('date')[columns].mean() if weighting == 'date_equal' else group[columns]
                    row = dict(station=station, event=event, candidate=candidate, weighting=weighting,
                               n_station_hours=len(group), n_dates=int(group.date.nunique()),
                               start_utc=group.hour_utc.min().isoformat(), end_utc=group.hour_utc.max().isoformat())
                    for c in columns:
                        row[c] = None if ag[c].notna().sum() == 0 else float(ag[c].mean())
                        row[c + '_finite_hours'] = int(group[c].notna().sum())
                    rows.append(row)
    return rows


def checked_native(legacy, source, group):
    import re
    import netCDF4
    path = ROOT / '外部样例数据CEDA/GBS_20p7GHz' / source
    with netCDF4.Dataset('unit_utc_check.nc', memory=path.read_bytes()) as ds:
        tv, v = ds.variables['time'], ds.variables['clearsky_attenuation']
        need(v.units == 'dB' and float(v.integration_period) == 1, 'Native dB / one second')
        need('end of integration' in tv.long_name.lower(), 'Integration end convention')
        need(abs(float(ds.variables['frequency'][...]) - 20.7) < 1e-4, '20.7 GHz native carrier')
        mm = re.fullmatch(r'seconds since (\d+)-(\d+)-(\d+) 0:0:0\.0+ \+0:00', tv.units)
        ts = group.hour_utc.iloc[0]
        need(mm is not None and tuple(map(int, mm.groups())) == (ts.year, ts.month, ts.day), 'Native UTC date')
    return legacy.native(source)


def main():
    begin = time.monotonic()
    need((OUT / 'PROTOCOL.md').is_file() and not (OUT / 'started.json').exists(), 'One-shot protocol gate')
    with locks():
        protected = protected_sources()
        emit('sources_before.json', protected)
        emit('started.json', {'utc': utc(), 'python': sys.version, 'numpy': np.__version__,
                              'pandas': pd.__version__, 'command': sys.argv,
                              'owner': '/root; same tvt; Revision018 only'})
        spec = importlib.util.spec_from_file_location('frozen_external_temporal', LEGACY)
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        legacy.main = legacy.emit = legacy.controls = forbid
        dyn, saved, models, libraries = legacy.load_frozen()
        controls(dyn, legacy)
        current = pd.read_parquet(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012/selected_uk.parquet')
        native = pd.read_parquet(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet')
        cols = ['station', 'hour_utc', 'source_file', 'valid_seconds', 'truth_db', 'target_status',
                'primary_score_eligible', 'exclusion_reason', 'liquid_negative']
        frame = current.merge(native[cols].rename(columns={'truth_db': 'native_truth_db'}),
                              on=['station', 'hour_utc'], validate='one_to_one')
        need(len(frame) == 5664 and np.allclose(frame.truth_db, frame.native_truth_db, equal_nan=True,
                                               rtol=0, atol=1e-12), 'Full cohort join')
        need(str(frame.hour_utc.dt.tz) == 'UTC', 'UTC')
        frame['temporal_eligible'] = frame.common_score_eligible & frame.event.isin(('rain', 'dry_overcast'))
        frame['temporal_exclusion'] = np.where(~frame.event.isin(('rain', 'dry_overcast')), 'outside_rain_or_dry_overcast',
                                     np.where(~frame.common_score_eligible, 'outside_existing_physical_common_cohort', ''))
        selected = frame[frame.temporal_eligible].sort_values(['station', 'hour_utc']).copy()
        need(len(selected) == 3027 and selected.event.eq('rain').sum() == 735, 'Selected scope')
        need(selected.primary_score_eligible.all(), 'Quality primary')
        oi, oa = old_index()
        modes, changes = {}, []
        for r in selected.itertuples(index=False):
            key = (r.station, r.hour_utc); item = oi.get(key)
            mode = 'new_hour' if item is None else ('reuse_unchanged' if item['metadata']['event'] == r.event else 'current_weather_changed')
            modes[key] = mode
            if item is not None:
                need(item['metadata']['source_file'] == r.source_file, 'Old raw source identity')
            if mode == 'current_weather_changed':
                changes.append(dict(station=r.station, hour_utc=r.hour_utc.isoformat(),
                                    old_event=item['metadata']['event'], current_event=r.event))
        mode_counts = pd.Series(list(modes.values())).value_counts().to_dict()
        need(mode_counts == {'new_hour': 2094, 'reuse_unchanged': 932, 'current_weather_changed': 1}, 'Frozen reuse scope')
        need(changes == [dict(station='sparsholt', hour_utc='2004-03-15T08:00:00+00:00',
                             old_event='rain', current_event='dry_overcast')], 'Known source-version change')
        frame['calculation_mode'] = [modes.get((r.station, r.hour_utc), 'not_scored') for r in frame.itertuples(index=False)]
        parquet('all_uk_hour_ledger.parquet', frame)
        coverage = frame.assign(utc_date=frame.hour_utc.dt.strftime('%Y-%m-%d')).groupby(['station', 'utc_date']).agg(
            acquired_hours=('hour_utc', 'size'), common_hours=('common_score_eligible', 'sum'),
            temporal_hours=('temporal_eligible', 'sum')).reset_index()
        parquet('coverage_by_date.parquet', coverage)
        emit('reuse_plan.json', {'counts': mode_counts, 'changed_weather': changes,
                               'new_generation_hours': 2095, 'unchanged_historical_hours_not_regenerated': 932})
        # Replays fixed here, before any scores are computed; changed-condition hour explicitly included.
        new_sel = selected[[modes[(r.station, r.hour_utc)] != 'reuse_unchanged' for r in selected.itertuples(index=False)]]
        replay = {(g.iloc[0].station, g.iloc[0].hour_utc.isoformat()) for _, g in new_sel.groupby(['station', 'event'])}
        replay.update((x['station'], x['hour_utc']) for x in changes)
        emit('replay_plan.json', [{'station': s, 'hour_utc': h} for s, h in sorted(replay)])
        (OUT / 'days').mkdir(exist_ok=False)
        hourly, lag_rows, index, raw_audit, shards = [], [], [], [], []
        nnew = nreuse = nchecks = 0
        for source, group in selected.groupby('source_file', sort=True):
            day = checked_native(legacy, source, group); group = group.sort_values('hour_utc')
            batch = {k: [] for k in ('observed_mean', 'observed_second', 'observed_acf', 'pair_counts',
                                     'generated_mean', 'generated_acf', 'metadata', 'generation_metadata')}
            target = OUT / 'days' / (group.iloc[0].station + '_' + group.iloc[0].hour_utc.strftime('%Y%m%d') + '.npz')
            new_entries = []
            for r in group.itertuples(index=False):
                k = (r.station, r.hour_utc); mode = modes[k]; y = day[r.hour_utc.hour]; valid = np.isfinite(y)
                need(valid.sum() == r.valid_seconds and valid.sum() >= 3420, 'Native quality count')
                need(abs(float(-10 * np.log10(np.mean(10 ** (-y[valid] / 10)))) - r.truth_db) < 1e-9, 'Frozen native linear-power hourly target')
                masks = dyn.pair_masks(valid)
                m1, m2, acf = dyn.increments(y - r.truth_db, masks)
                pc = np.array([m.sum() for m in masks], dtype=int)
                need(np.isfinite(m1).any(), 'No scoreable lag')
                key = 'GBS-external-v010-' + r.station + '-' + r.hour_utc.isoformat()
                meta = dict(station=r.station, hour_utc=r.hour_utc.isoformat(), source_file=source,
                            event=r.event, key=key)
                if mode == 'reuse_unchanged':
                    item = oi[k]; i = item['row']; data = oa[item['path']]
                    for name, value in [('observed_mean', m1), ('observed_second', m2), ('observed_acf', acf)]:
                        need(np.allclose(value, data[name][i], equal_nan=True, rtol=0, atol=1e-9), 'Old observed identity ' + name)
                    need(np.array_equal(pc, data['pair_counts'][i]) and item['metadata']['key'] == key, 'Old mask/key identity')
                    gm, ga = data['generated_mean'][i], data['generated_acf'][i]
                    entry = dict(meta, mode=mode, shard=str(Path(item['path']).relative_to(ROOT)),
                                 row=i, shard_sha256=item['sha256'])
                    nreuse += 1
                else:
                    qs, generation_meta = [], []
                    for candidate in CANDIDATES:
                        result = [dyn.generate(candidate, models[r.event], libraries[r.event] if candidate == 'block_0256' else None,
                                               -1, key, rep) for rep in range(32)]
                        qs.append(np.array([q for q, _ in result])); generation_meta.append([m for _, m in result])
                    need(all(np.isfinite(q).all() and (q > 0).all() for q in qs), 'Finite positive power')
                    need(np.array_equal(np.sort(qs[0], axis=1), np.sort(qs[1], axis=1)), 'Common marginal multiset')
                    nchecks += 32
                    generated = [legacy.matrix_increments(-10 * np.log10(q), masks) for q in qs]
                    gm = np.array([x[0] for x in generated]); ga = np.array([x[1] for x in generated])
                    i = len(batch['metadata'])
                    for name, value in [('observed_mean', m1), ('observed_second', m2), ('observed_acf', acf),
                                        ('pair_counts', pc), ('generated_mean', gm), ('generated_acf', ga),
                                        ('metadata', meta), ('generation_metadata', generation_meta)]:
                        batch[name].append(value)
                    entry = dict(meta, mode=mode, shard=str(target.relative_to(ROOT)), row=i)
                    new_entries.append(entry); nnew += 1
                index.append(entry)
                raw_audit.append(dict(meta, mode=mode, valid_seconds=int(valid.sum()), invalid_seconds=int((~valid).sum()),
                                      n_lags=int(np.isfinite(m1).sum()), total_gap_safe_pairs=int(pc.sum())))
                for j, candidate in enumerate(CANDIDATES):
                    scores = [dyn.score_from_moments(m1, m2, gm[j, :n]) for n in (16, 32)]
                    base = dict(station=r.station, hour_utc=r.hour_utc, date=r.hour_utc.strftime('%Y-%m-%d'),
                                event=r.event, candidate=candidate, calculation_mode=mode)
                    hourly.append(dict(base, n_seconds=int(valid.sum()), n_lags=scores[1]['lags_used'],
                                       vs16=scores[0]['vs'], vs32=scores[1]['vs'],
                                       plugin16=scores[0]['vs_plugin'], plugin32=scores[1]['vs_plugin']))
                    for z, lag in enumerate(LAGS):
                        finite_acf = ga[j, :, z][np.isfinite(ga[j, :, z])]
                        g = gm[j, :, z]
                        lag_rows.append(dict(base, lag_s=lag, n_pairs=int(pc[z]),
                                             observed_increment_db=float(m1[z]), observed_acf=float(acf[z]),
                                             generated_increment_db=float(g.mean()) if np.isfinite(g).all() else np.nan,
                                             generated_acf=float(finite_acf.mean()) if len(finite_acf) else np.nan,
                                             generated_acf_finite_reps=len(finite_acf)))
            if batch['metadata']:
                with target.open('xb') as stream:
                    np.savez_compressed(stream, **{k: np.array(json.dumps(v)) if k in ('metadata', 'generation_metadata') else np.array(v)
                                                   for k, v in batch.items()})
                h = sha(target)
                for entry in new_entries:
                    entry['shard_sha256'] = h
                shards.append({'path': str(target.relative_to(ROOT)), 'sha256': h, 'hours': len(batch['metadata'])})
            print(json.dumps({'utc': utc(), 'new_hours': nnew, 'reused_hours': nreuse, 'total': 3027,
                              'elapsed_s': round(time.monotonic() - begin, 1)}), flush=True)
        need(nnew == 2095 and nreuse == 932 and nchecks == 67040, 'Final generation accounting')
        hf, lf = pd.DataFrame(hourly), pd.DataFrame(lag_rows)
        need(len(hf) == 6054 and len(lf) == 66594, 'Output shape')
        parquet('hourly_scores.parquet', hf); parquet('lag_diagnostics.parquet', lf)
        emit('combined_shard_index.json', index); emit('new_shards.json', shards); emit('native_hour_audit.json', raw_audit)
        emit('summary.json', {'status': 'COMPUTED_PENDING_INDEPENDENT_QA', 'score_unit': 'dB squared',
                             'rows': summaries(hf, ['vs32', 'vs16', 'plugin32', 'plugin16'])})
        lag_summary = []
        for lag, g in lf.groupby('lag_s'):
            lag_summary.extend([dict(x, lag_s=int(lag)) for x in summaries(g, ['observed_increment_db', 'generated_increment_db',
                                                                            'observed_acf', 'generated_acf'])])
        emit('lag_summary.json', lag_summary)
        need(all(sha(p) == h for p, h in protected.items()), 'Source changed during calculation')
        files = {str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file()
                 and p.name not in ('execution.lock',) and not p.name.endswith('.log')}
        emit('calculation_complete.json', {'status': 'COMPUTED_PENDING_INDEPENDENT_QA', 'utc': utc(),
             'station_hours': 3027, 'new_generation_hours': nnew, 'reused_hours': nreuse,
             'new_realizations': nnew * 64, 'common_marginal_checks': nchecks, 'protected_sources': len(protected),
             'new_training_calls': 0, 'new_weather_downloads': 0, 'new_slow_predictions': 0,
             'full_MTS_complete': False, 'seven_families_complete': False,
             'elapsed_s': time.monotonic() - begin, 'outputs_sha256': files})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (OUT / 'calculation_failure.json').exists():
            emit('calculation_failure.json', {'utc': utc(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
