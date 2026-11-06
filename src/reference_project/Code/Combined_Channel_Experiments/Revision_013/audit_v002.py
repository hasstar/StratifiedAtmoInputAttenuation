"""One-shot LOCAL raw meteorology QA. No network, model, labels, or source writes."""
import hashlib
import json
import msvcrt
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_013/attempt_002'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'
RAW = Path('E:/0_Workflow_System/01_PhD/Data/era5_campaign_20250930_20260820/raw')
STOP = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
STOP_SHA = '0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a'
LEVELS = [1000, 925, 850, 700, 650, 500, 300, 200, 30, 5, 1]
PRESSURE = {
    'temperature': ('t', 130, 'K'),
    'specific_humidity': ('q', 133, 'kg kg**-1'),
    'geopotential': ('z', 129, 'm**2 s**-2'),
    'u_component_of_wind': ('u', 131, 'm s**-1'),
    'v_component_of_wind': ('v', 132, 'm s**-1'),
    'specific_cloud_liquid_water_content': ('clwc', 246, 'kg kg**-1'),
    'specific_cloud_ice_water_content': ('ciwc', 247, 'kg kg**-1'),
    'specific_rain_water_content': ('crwc', 75, 'kg kg**-1'),
    'specific_snow_water_content': ('cswc', 76, 'kg kg**-1'),
}
SURFACE = {
    'surface_instant': [('deg0l', 228024, 'm'), ('ptype', 260015, '(Code table 4.201)'),
                        ('t2m', 167, 'K'), ('d2m', 168, 'K'), ('sp', 134, 'Pa'),
                        ('tclw', 78, 'kg m**-2'), ('tcc', 164, '(0 - 1)')],
    'surface_mean_rate': [('avg_tprate', 235055, 'kg m**-2 s**-1'),
                          ('avg_cpr', 235030, 'kg m**-2 s**-1'),
                          ('avg_lsprate', 235029, 'kg m**-2 s**-1')],
    'surface_supp_mean_rate_01': [('avg_tsrwe', 235031, 'kg m**-2 s**-1'),
                                  ('avg_csfr', 235056, 'kg m**-2 s**-1'),
                                  ('avg_lssfr', 235057, 'kg m**-2 s**-1')],
}


def need(value, message):
    if not value:
        raise ValueError(message)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    good = a[finite]
    return {'size': int(a.size), 'finite': int(finite.sum()),
            'nan': int(np.isnan(a).sum()), 'inf': int(np.isinf(a).sum()),
            'negative': int((good < 0).sum()), 'zero': int((good == 0).sum()),
            'minimum': float(good.min()) if good.size else None,
            'maximum': float(good.max()) if good.size else None}


def ranges(times):
    times = pd.DatetimeIndex(times).sort_values().unique()
    if not len(times):
        return []
    cuts = np.flatnonzero(np.diff(times.asi8) != 3600 * 10**9) + 1
    chunks = np.split(np.arange(len(times)), cuts)
    return [{'start_utc': times[c[0]].isoformat() + 'Z',
             'end_utc_inclusive': times[c[-1]].isoformat() + 'Z', 'hours': int(len(c))}
            for c in chunks]


def expected_times(job):
    r = job['request']
    need(len(r['year']) == len(r['month']) == 1, 'Unexpected request calendar')
    return pd.DatetimeIndex([f'{r["year"][0]}-{int(r["month"][0]):02d}-{int(d):02d}T{t}:00'
                             for d in r['day'] for t in r['time']]).sort_values()


def check_meta(ds, key, param, units, pressure, expected):
    need(key in ds, 'Missing variable: ' + key)
    v = ds[key]
    dims = ('valid_time', 'pressure_level', 'latitude', 'longitude') if pressure else ('valid_time', 'latitude', 'longitude')
    need(v.dims == dims, 'Variable dimension order: ' + key)
    need(int(v.attrs['GRIB_paramId']) == param, 'Parameter identity: ' + key)
    need(v.attrs['units'] == units, 'Units: ' + key)
    need(v.attrs['GRIB_stepType'] == ('avg' if key.startswith('avg_') else 'instant'), 'Step type: ' + key)
    need(pd.DatetimeIndex(ds.valid_time.values).equals(expected), 'UTC calendar mismatch: ' + key)
    need(ds.valid_time.encoding['units'].startswith('seconds since 1970-01-01'), 'Unreviewed time epoch')
    need(np.array_equal(ds.latitude.values, [40, 39.75, 39.5, 39.25] if pressure else [40, 39.75]), 'Latitude grid')
    need(np.array_equal(ds.longitude.values, [116.25, 116.5]), 'Longitude grid')
    if pressure:
        need(ds.pressure_level.attrs['units'] == 'hPa', 'Pressure units')
        need(np.array_equal(ds.pressure_level.values, LEVELS), 'Pressure levels')
    a = v.values.astype(np.float64)
    need(not np.any(np.abs(a[np.isfinite(a)]) >= 1e30), 'Undecoded GRIB fill value: ' + key)
    return a


def load_protection():
    guards = {}
    for name in ['sources_before.json', 'publication_sources.json', 'delivery_manifest.json']:
        for path, digest in read(PREV / name).items():
            p = str((ROOT / path).resolve())
            need(p not in guards or guards[p] == digest, 'Guard conflict: ' + p)
            guards[p] = digest
    for directory in [PREV, ROOT / 'Code/Combined_Channel_Experiments/Revision_012', CODE]:
        for p in directory.iterdir():
            if p.is_file():
                key = str(p.resolve())
                current = sha(p)
                need(key not in guards or guards[key] == current, 'Changed frozen file: ' + key)
                guards[key] = current
    for p in [ROOT / 'Tasks/T012_Combined_Channel_And_External_Validation.md',
              ROOT / 'Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md',
              Path('E:/0_Workflow_System/README.md'), STOP]:
        guards[str(p.resolve())] = sha(p)
    need(sha(STOP) == STOP_SHA, 'Weather STOP changed')
    for p, digest in guards.items():
        need(sha(p) == digest, 'Frozen source mismatch: ' + p)
    return guards


def main():
    need(read(OUT / 'tests.json')['status'] == 'PASS', 'Unit tests required')
    need(not (OUT / 'audit_complete.json').exists(), 'One-shot output already exists')
    guards = load_protection()
    manifest_path = RAW / 'download_manifest.json'
    manifest = read(manifest_path)
    jobs = []
    for job in manifest['jobs']:
        if job['dataset'] == 'reanalysis-era5-pressure-levels' and job['request']['variable'][0] in PRESSURE:
            need(len(job['request']['variable']) == 1, 'Unexpected pressure group')
            specs = [PRESSURE[job['request']['variable'][0]]]
            pressure = True
        else:
            prefix = job['job_id'].rsplit('_', 2)[0]
            if prefix not in SURFACE:
                continue
            specs, pressure = SURFACE[prefix], False
        path = RAW / Path(job['target'].replace('\\', '/')).name
        need(path.is_file(), 'Source absent: ' + str(path))
        jobs.append((job, path, specs, pressure))
    need(len(jobs) == 144, 'Expected nine pressure and three surface groups x twelve files')
    raw_guard = {str(p): sha(p) for _, p, _, _ in jobs}
    raw_guard[str(manifest_path)] = sha(manifest_path)
    test_path = PREV / 'selected_local.parquet'
    selected = pd.read_parquet(test_path, columns=['time'])
    test_times = pd.DatetimeIndex(pd.to_datetime(selected['time'], utc=True)).tz_localize(None)
    need(len(test_times) == 687 and test_times.is_unique, 'Original test time keys')
    start = pd.Timestamp(manifest['start_utc']).tz_convert('UTC').tz_localize(None)
    end = pd.Timestamp(manifest['end_utc']).tz_convert('UTC').tz_localize(None)
    campaign = pd.date_range(start, end, freq='h')
    need(len(campaign) == 7800, 'Campaign hours')
    need(test_times.isin(campaign).all(), 'Test key outside campaign')
    with (OUT / 'execution.lock').open('x+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            save('launch.json', {'utc': utcnow(), 'pid': os.getpid(), 'python': sys.version,
                                'executable': sys.executable, 'platform': platform.platform(),
                                'numpy': np.__version__, 'pandas': pd.__version__, 'xarray': xr.__version__,
                                'command': sys.argv, 'scope': 'offline input QA only', 'lock': 'acquired'})
            save('protected_sources_before.json', guards)
            save('raw_sources_before.json', raw_guard)
            chunks, clock = defaultdict(list), defaultdict(list)
            inventory = []
            for job, path, specs, pressure in jobs:
                expected = expected_times(job)
                with xr.open_dataset(path, engine='netcdf4') as ds:
                    record = {'path': str(path), 'sha256': raw_guard[str(path)],
                              'bytes': path.stat().st_size, 'job_id': job['job_id'],
                              'calendar_utc': ranges(expected), 'pressure': pressure,
                              'time_encoding': str(ds.valid_time.encoding.get('units')),
                              'latitude': ds.latitude.values.tolist(), 'longitude': ds.longitude.values.tolist(),
                              'pressure_hpa': ds.pressure_level.values.tolist() if pressure else None,
                              'expver': sorted(set(map(str, ds.expver.values))) if 'expver' in ds else [], 'variables': {}}
                    for key, param, units in specs:
                        present = key in ds
                        if not present:
                            need(job['job_id'] == 'surface_instant_20260701_20260731' and key in {'deg0l', 'ptype'},
                                 'Unexpected absent source variable: ' + job['job_id'] + ':' + key)
                            # Explicit missing-parameter ledger placeholders, never observations or imputation.
                            a = np.full(ds['t2m'].shape, np.nan, dtype=np.float64)
                        else:
                            a = check_meta(ds, key, param, units, pressure, expected)
                        record['variables'][key] = dict(stats(a), parameter_id=param, units=units,
                                                       long_name=ds[key].attrs.get('long_name') if present else None,
                                                       step_type=ds[key].attrs['GRIB_stepType'] if present else None,
                                                       source_present=present,
                                                       source_status='PRESENT' if present else 'ABSENT_PARAMETER',
                                                       nan_ledger_placeholders_only=not present)
                        chunks[key].append(a)
                        clock[key].append(expected.values)
                    inventory.append(record)
            save('file_inventory.json', inventory)
            arrays, times, summaries = {}, {}, {}
            reference_times = pd.DatetimeIndex(np.sort(np.concatenate(clock['t'])))
            for key in chunks:
                t = np.concatenate(clock[key]); order = np.argsort(t)
                need(len(t) == len(np.unique(t)), 'Duplicate UTC: ' + key)
                times[key] = pd.DatetimeIndex(t[order])
                arrays[key] = np.concatenate(chunks[key])[order]
                need(times[key].equals(reference_times), 'Cross-field time mismatch: ' + key)
                indices = times[key].get_indexer(campaign)
                need((indices >= 0).all(), 'Missing campaign clock: ' + key)
                a = arrays[key][indices]
                valid_hours = np.isfinite(a).reshape(len(a), -1).all(1)
                summaries[key] = {'raw': stats(arrays[key]), 'campaign': stats(a),
                                  'raw_utc_ranges': ranges(times[key]),
                                  'campaign_all_grid_finite_hours': int(valid_hours.sum()),
                                  'campaign_incomplete_utc_ranges': ranges(campaign[~valid_hours])}
            save('field_summary.json', summaries)
            base_times = times['t']; now = base_times.get_indexer(campaign)
            nxt = base_times.get_indexer(campaign + pd.Timedelta(hours=1))
            need((nxt >= 0).all(), 'No valid t+1h boundary input')
            h0_ok = np.isfinite(arrays['deg0l'][now]).reshape(7800, -1).all(1)
            wind = np.hypot(arrays['u'][:, LEVELS.index(700)], arrays['v'][:, LEVELS.index(700)])
            wind_ok = np.isfinite(wind[now]).reshape(7800, -1).all(1)
            rate_keys = [k for k in arrays if k.startswith('avg_')]
            rain_ok = np.logical_and.reduce([np.isfinite(arrays[k][nxt]).reshape(7800, -1).all(1) for k in rate_keys])
            liquid = arrays['avg_tprate'][nxt] - arrays['avg_tsrwe'][nxt]
            liquid_ok = np.isfinite(liquid).reshape(7800, -1).all(1) & (liquid.reshape(7800, -1) >= 0).all(1)
            profile_ok = np.logical_and.reduce([np.isfinite(arrays[s[0]][now]).reshape(7800, -1).all(1) for s in PRESSURE.values()])
            ledger = [{'time_utc': t.isoformat() + 'Z', 'original_test_key': bool(t in test_times),
                       'h0_all_corners_finite': bool(h0_ok[i]), 'wind700_all_grid_finite': bool(wind_ok[i]),
                       'six_rates_valid_t_plus_1h_finite': bool(rain_ok[i]),
                       'total_minus_snow_nonnegative_all_corners': bool(liquid_ok[i]),
                       'nine_pressure_fields_all_grid_finite': bool(profile_ok[i]),
                       'esst_input_coavailability_only': bool(h0_ok[i] and wind_ok[i] and rain_ok[i] and liquid_ok[i])}
                      for i, t in enumerate(campaign)]
            save('hourly_input_ledger.json', ledger)
            z_height = arrays['z'][now] / 9.80665
            pressure_pa = np.array(LEVELS)[None, :, None, None] * 100
            sp = arrays['sp'][now][:, None, :, :]
            partition = (arrays['avg_cpr'][nxt] + arrays['avg_lsprate'][nxt] - arrays['avg_tprate'][nxt]) * 3600
            diagnostics = {
                'wind700_m_per_s': stats(wind[now]), 'wind700_exact_zero_grid_values': int((wind[now] == 0).sum()),
                'geopotential_height_m': stats(z_height),
                'geopotential_height_nonincreasing_adjacent_pairs': int((np.diff(z_height, axis=1) <= 0).sum()),
                'geopotential_height_below_station_117_5m_grid_values': int((z_height < 117.5).sum()),
                'common_surface_grid_columns': 4, 'pressure_only_columns_without_surface_pressure': 4,
                'pressure_exceeds_surface_pressure_common_grid_values': int((pressure_pa > sp).sum()),
                'liquid_total_minus_snow_mm_hour': stats(liquid * 3600),
                'convective_plus_large_scale_minus_total_mm_hour': stats(partition),
                'partition_abs_error_gt_1e_minus_5_mm_hour': int((np.abs(partition) > 1e-5).sum()),
                'h0_file_reference': 'geometric height above ground, not AMSL',
                'station_within_both_grids': 39.75 <= 39.95051 <= 40 and 116.25 <= 116.3369 <= 116.5,
                'slant_path_coverage_verified': False, 'graupel_in_manifest': any('graupel' in str(j) for j in manifest['jobs']),
                'no_clipping_imputation_or_interpolation': True,
            }
            save('geometry_diagnostics.json', diagnostics)
            for path, digest in {**guards, **raw_guard}.items():
                need(sha(path) == digest, 'Source changed during QA: ' + path)
            summary = {'status': 'INPUT_AUDIT_COMPLETED_WITH_LIMITATIONS', 'utc': utcnow(),
                       'raw_files': len(jobs), 'variables': len(arrays), 'campaign_hours': len(campaign),
                       'raw_hours_per_variable': len(base_times), 'original_test_hours': len(test_times),
                       'campaign_utc_ranges': ranges(campaign), 'source_shas_unchanged': True,
                       'metadata_calendar_grid_checks': 'PASS_FOR_PRESENT_FIELDS',
                       'absent_parameter_files': [dict(path=r['path'], variable=k, parameter_id=v['parameter_id']) for r in inventory for k,v in r['variables'].items() if not v['source_present']],
                       'ledger_counts': {k: int(sum(r[k] for r in ledger)) for k in ledger[0] if k != 'time_utc'},
                       'test_ledger_counts': {k: int(sum(r[k] for r in ledger if r['original_test_key'])) for k in ledger[0] if k != 'time_utc'},
                       'all_seven_complete': False, 'new_training': 0, 'new_model_inference': 0,
                       'weather_requests': 0, 'new_attenuation_scores': 0,
                       'warning': 'Finite raw pressure grids are not a verified vertical/slant physical input. Coavailability is not E-SST eligibility.'}
            save('audit_complete.json', summary)
            print(json.dumps(summary, ensure_ascii=True, indent=2))
        except Exception as exc:
            save('failure.json', {'utc': utcnow(), 'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            save('lock_release.json', {'utc': utcnow(), 'pid': os.getpid(), 'state': 'RELEASED'})


if __name__ == '__main__':
    main()
