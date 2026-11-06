"""Actual LOCAL weather diagnostics only; never computes attenuation or scores."""
import hashlib
import json
import msvcrt
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
RAW = Path('E:/0_Workflow_System/01_PhD/Data/era5_campaign_20250930_20260820/raw')
LEVELS = np.array([1000, 925, 850, 700, 650, 500, 300, 200, 30, 5, 1])
SPECS = {
    't': ('pressure_temperature_', 130, 'K', True),
    'z': ('pressure_geopotential_', 129, 'm**2 s**-2', True),
    'sp': ('surface_instant_', 134, 'Pa', False),
    'deg0l': ('surface_instant_', 228024, 'm', False),
    'avg_tprate': ('surface_mean_rate_', 235055, 'kg m**-2 s**-1', False),
    'avg_tsrwe': ('surface_supp_mean_rate_01_', 235031, 'kg m**-2 s**-1', False),
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def safe(x):
    return float(x) if np.isfinite(x) else None


def iso(s):
    return datetime.fromtimestamp(int(s), timezone.utc).isoformat().replace('+00:00', 'Z')


def load_field(key, records):
    prefix, param, unit, pressure = SPECS[key]
    files = [r for r in records if r['manifest_job_id'] and r['file'].startswith(prefix)]
    assert len(files) == 12, (key, len(files))
    chunks, clocks, missing = [], [], []
    for r in files:
        with xr.open_dataset(RAW / r['file'], engine='netcdf4') as ds:
            ts = ds.valid_time.values.astype('datetime64[s]').astype(np.int64)
            assert ds.valid_time.encoding['units'].startswith('seconds since 1970-01-01')
            assert np.all(np.diff(ts) == 3600)
            assert np.array_equal(ds.latitude.values, [40, 39.75, 39.5, 39.25] if pressure else [40, 39.75])
            assert np.array_equal(ds.longitude.values, [116.25, 116.5])
            dims = ('valid_time', 'pressure_level', 'latitude', 'longitude') if pressure else ('valid_time', 'latitude', 'longitude')
            if pressure:
                assert np.array_equal(ds.pressure_level.values, LEVELS)
                assert ds.pressure_level.attrs['units'] == 'hPa'
            if key not in ds:
                assert key == 'deg0l', (key, r['file'])
                a = np.full((len(ts), 2, 2), np.nan)
                missing.append(r['file'])
            else:
                v = ds[key]
                assert v.dims == dims
                assert int(v.attrs['GRIB_paramId']) == param and v.attrs['units'] == unit
                assert v.attrs['GRIB_stepType'] == ('avg' if key.startswith('avg_') else 'instant')
                a = v.sel(latitude=[40, 39.75], longitude=[116.25, 116.5]).values.astype(np.float64)
                assert not np.any(np.abs(a[np.isfinite(a)]) >= 1e30)
            chunks.append(a); clocks.append(ts)
    clock = np.concatenate(clocks)
    order = np.argsort(clock)
    clock = clock[order]
    assert len(clock) == 7824 and np.all(np.diff(clock) == 3600)
    return clock, np.concatenate(chunks)[order], missing


def crossings(t, h, valid):
    """Vectorized adjacent-layer diagnostic, shapes time x location x level."""
    pair = valid[..., :-1] & valid[..., 1:]
    dh = h[..., 1:] - h[..., :-1]
    ascending = dh > 0
    lower, upper = t[..., :-1], t[..., 1:]
    down = pair & ascending & (lower > 273.15) & (upper < 273.15)
    up = pair & ascending & (lower < 273.15) & (upper > 273.15)
    fraction = np.divide(273.15 - lower, upper - lower,
                         out=np.full_like(lower, np.nan), where=down)
    candidate = np.where(down, h[..., :-1] + fraction * dh, np.nan)
    return {'candidate_geopotential_m': candidate,
            'bracket_width_geopotential_m': np.where(down, dh, np.nan),
            'warm_to_cold_count': down.sum(axis=-1), 'cold_to_warm_count': up.sum(axis=-1),
            'exact_zero_count': ((t == 273.15) & valid).sum(axis=-1),
            'nonascending_pair_count': (pair & ~ascending).sum(axis=-1),
            'valid_level_count': valid.sum(axis=-1)}


def wet_events(times, rain):
    wet = np.isfinite(rain) & (rain > 0)
    starts = np.flatnonzero(wet & ~np.r_[False, wet[:-1]])
    ends = np.flatnonzero(wet & ~np.r_[wet[1:], False])
    event_id = np.full(len(times), -1, dtype=np.int32)
    events = []
    for eid, (start, end) in enumerate(zip(starts, ends, strict=True)):
        left = start == 0 or not (np.isfinite(rain[start - 1]) and rain[start - 1] == 0)
        right = end == len(times) - 1 or not (np.isfinite(rain[end + 1]) and rain[end + 1] == 0)
        peak = float(rain[start:end + 1].max())
        events.append({'event_id': eid, 'start_utc': iso(times[start]),
                       'last_bin_start_utc': iso(times[end]), 'hours': int(end - start + 1),
                       'left_censored': bool(left), 'right_censored': bool(right),
                       'complete': not (left or right), 'peak_hourly_mm_per_h': peak,
                       'hourly_peak_type_candidate': 'convective' if peak > 10 else 'stratiform',
                       'minute_peak_type_verified': False})
        event_id[start:end + 1] = eid
    return event_id, events


def main():
    assert read(OUT / 'inventory_qa.json')['status'] == 'PASS'
    guards = read(OUT / 'sources_before.json')
    for p in CODE.iterdir():
        if p.is_file():
            guards[str(p.resolve())] = sha(p)
    for p, digest in guards.items():
        assert sha(p) == digest, p
    with (OUT / 'diagnostics.lock').open('x+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        emit('diagnostic_execution.json', {'utc': datetime.now(timezone.utc).isoformat(),
             'pid': os.getpid(), 'executable': sys.executable, 'argv': sys.argv,
             'pandas': pd.__version__, 'numpy': np.__version__, 'xarray': xr.__version__})
        emit('computation_sources_before.json', guards)
        records = read(OUT / 'header_inventory.json')
        manifest = read(RAW / 'download_manifest.json')
        station = manifest['campaign']['station']
        time, data, missing = None, {}, {}
        for key in SPECS:
            clock, a, absent = load_field(key, records)
            assert time is None or np.array_equal(clock, time)
            time = clock; data[key] = a; missing[key] = absent
        start = int(pd.Timestamp(manifest['start_utc']).timestamp())
        end = int(pd.Timestamp(manifest['end_utc']).timestamp())
        campaign = np.arange(start, end + 3600, 3600, dtype=np.int64)
        assert len(campaign) == 7800
        indices = np.searchsorted(time, campaign)
        assert np.array_equal(time[indices], campaign)
        test = pd.read_parquet(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012/selected_local.parquet', columns=['time'])
        test_seconds = pd.to_datetime(test.time, utc=True).astype('int64').to_numpy() // 10**9
        assert len(np.unique(test_seconds)) == len(test_seconds) == 687
        test_mask = np.isin(campaign, test_seconds)
        assert test_mask.sum() == 687
        fn = (station['latitude_deg_n'] - 39.75) / .25
        fe = (station['longitude_deg_e'] - 116.25) / .25
        weights = np.array([fn * (1 - fe), fn * fe, (1 - fn) * (1 - fe), (1 - fn) * fe])
        assert np.all(weights > 0) and abs(weights.sum() - 1) < 1e-14
        tc = data['t'][indices].reshape(7800, 11, 4).transpose(0, 2, 1)
        hc = (data['z'][indices] / 9.80665).reshape(7800, 11, 4).transpose(0, 2, 1)
        spc = data['sp'][indices].reshape(7800, 4)
        vc = np.isfinite(tc) & np.isfinite(hc) & (LEVELS[None, None, :] <= spc[:, :, None] / 100)
        # A station-level interpolation is not accepted from below-ground corner values.
        vs = vc.all(axis=1)
        ts = np.where(vs, np.einsum('tcl,c->tl', tc, weights), np.nan)
        hs = np.where(vs, np.einsum('tcl,c->tl', hc, weights), np.nan)
        temperatures = np.concatenate([tc, ts[:, None, :]], axis=1)
        heights = np.concatenate([hc, hs[:, None, :]], axis=1)
        valid = np.concatenate([vc, vs[:, None, :]], axis=1)
        cross = crossings(temperatures, heights, valid)
        # All raw one-hour rain intervals having the following valid-time endpoint.
        context_time = time[:-1]
        raw_rain = ((data['avg_tprate'][1:] - data['avg_tsrwe'][1:]) * 3600).reshape(-1, 4)
        rain_valid = np.isfinite(raw_rain).all(axis=1) & (raw_rain >= 0).all(axis=1)
        site_rain = np.where(rain_valid, raw_rain @ weights, np.nan)
        context_idx = np.searchsorted(context_time, campaign)
        assert np.array_equal(context_time[context_idx], campaign)
        ids, events = wet_events(context_time, site_rain)
        dg = data['deg0l'][indices].reshape(7800, 4)
        ledger, context_ledger = [], []
        for i, s in enumerate(context_time):
            r = raw_rain[i]
            context_ledger.append({'time_utc': iso(s), 'rate_valid_end_utc': iso(s + 3600),
                                   'campaign_member': bool(start <= s <= end),
                                   'liquid_rain_corners_mm_per_h': [safe(x) for x in r],
                                   'all_corners_nonnegative': bool(rain_valid[i]),
                                   'site_rain_mm_per_h': safe(site_rain[i]), 'event_id': int(ids[i])})
        for i, s in enumerate(campaign):
            c = context_idx[i]; eid = int(ids[c]); event = events[eid] if eid >= 0 else None
            ledger.append({'time_utc': iso(s), 'test_member': bool(test_mask[i]),
                           'instant_valid_utc': iso(s), 'rate_valid_end_utc': iso(s + 3600),
                           'deg0l_corners_m_agl': [safe(x) for x in dg[i]],
                           'deg0l_all_corners_finite': bool(np.isfinite(dg[i]).all()),
                           'deg0l_zero_corner_count': int((dg[i] == 0).sum()),
                           'site_rain_mm_per_h': safe(site_rain[c]),
                           'all_corners_liquid_nonnegative': bool(rain_valid[c]),
                           'wet_event_id': eid, 'wet_event_complete': event['complete'] if event else None,
                           'station_crossings': int(cross['warm_to_cold_count'][i, 4]),
                           'station_exact_zero_levels': int(cross['exact_zero_count'][i, 4]),
                           'used_for_model': False})
        arrays = {'campaign_epoch_s': campaign, 'test_member': test_mask, 'pressure_hpa': LEVELS,
                  'weights_NW_NE_SW_SE': weights, 'valid_above_surface': valid,
                  **cross}
        with (OUT / 'zero_crossing_diagnostics.npz').open('xb') as f:
            np.savez_compressed(f, **arrays)
        emit('hourly_input_ledger.json', ledger)
        emit('rain_context_ledger.json', context_ledger)
        emit('rain_events.json', events)
        names = ['NW', 'NE', 'SW', 'SE', 'station_bilinear_profile']
        summary = {'status': 'COMPUTED_PENDING_INDEPENDENT_QA', 'campaign_hours': 7800, 'test_keys': 687,
                   'labels_loaded': False, 'new_predictions': 0, 'new_scores': 0,
                   'height_output_kind': 'candidate_geopotential_metres_not_geometric_altitude',
                   'selected_zero_crossing_branch': None, 'missing_field_files': missing,
                   'station_corner_weights': dict(zip(names[:4], weights.tolist())),
                   'station_profile_requires_all_four_corners_above_model_surface': True,
                   'profiles': {}, 'rain_context': {'raw_instant_hours': len(time),
                   'valid_end_paired_hours': len(context_time),
                   'buffer_hours_before_campaign': int((context_time < start).sum()),
                   'buffer_hours_after_campaign': int((context_time > end).sum()),
                   'wet_segments': len(events), 'complete_wet_segments': sum(e['complete'] for e in events),
                   'censored_wet_segments': sum(not e['complete'] for e in events)}}
        for j, name in enumerate(names):
            c = cross['warm_to_cold_count'][:, j]
            vals = cross['candidate_geopotential_m'][:, j]
            finite = vals[np.isfinite(vals)]
            widths = cross['bracket_width_geopotential_m'][:, j]
            summary['profiles'][name] = {
                'no_strict_down_crossing_hours': int((c == 0).sum()),
                'one_strict_down_crossing_hours': int((c == 1).sum()),
                'multiple_strict_down_crossing_hours': int((c > 1).sum()),
                'test_no_crossing_hours': int(((c == 0) & test_mask).sum()),
                'test_one_crossing_hours': int(((c == 1) & test_mask).sum()),
                'test_multiple_crossing_hours': int(((c > 1) & test_mask).sum()),
                'exact_zero_level_count': int(cross['exact_zero_count'][:, j].sum()),
                'nonascending_pair_count': int(cross['nonascending_pair_count'][:, j].sum()),
                'candidate_min_m': float(finite.min()) if len(finite) else None,
                'candidate_max_m': float(finite.max()) if len(finite) else None,
                'largest_bracket_m': float(np.nanmax(widths))}
        for label, rows in [('campaign', ledger), ('test', [r for r in ledger if r['test_member']])]:
            summary[label] = {'n': len(rows),
                'rain_invalid_hours': sum(not r['all_corners_liquid_nonnegative'] for r in rows),
                'deg0l_missing_hours': sum(not r['deg0l_all_corners_finite'] for r in rows),
                'deg0l_any_zero_hours': sum(r['deg0l_zero_corner_count'] > 0 for r in rows),
                'wet_hours': sum(r['wet_event_id'] >= 0 for r in rows),
                'complete_wet_event_hours': sum(r['wet_event_complete'] is True for r in rows),
                'censored_wet_event_hours': sum(r['wet_event_complete'] is False for r in rows),
                'dry_hours': sum(r['site_rain_mm_per_h'] == 0 for r in rows)}
        emit('diagnostic_summary.json', summary)
        changed = [p for p, digest in guards.items() if sha(p) != digest]
        assert not changed, changed
        emit('diagnostic_source_guard.json', {'status': 'PASS', 'count': len(guards), 'changes': changed,
                                             'utc': datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
