"""Append-only input conversion and synthetic hourly-operator execution."""
import ast
import hashlib
import importlib.util
import itertools
import json
import math
import msvcrt
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from hourly_power import geometric_height_m, hourly_power_mean, InsufficientContextError

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_022'
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
RAW = Path('E:/0_Workflow_System/01_PhD/Data/era5_campaign_20250930_20260820/raw')
RE = 6371229.0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def emit(name, data):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def guard_sources():
    guards = read(OLD / 'sources_after_final.json')
    for path, item in read(OLD / 'delivery_manifest.json')['files'].items():
        guards[str((ROOT / path).resolve())] = item['sha256']
    guards[str((OLD / 'delivery_manifest.json').resolve())] = read(OLD / 'final_verification.json')['manifest_sha256']
    more = [OLD / 'final_verification.json',
        ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_016/PROTOCOL.md',
        ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_016/run_itu_baseline.py',
        ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/run_manifest.json',
        ROOT / 'Code/Combined_Channel_Experiments/Revision_004/physics.py',
        ROOT / 'Code/Combined_Channel_Experiments/Revision_004/common.py',
        ROOT / 'Outputs/Experiments/Combined_Channel_Revision_004/native_metadata_gate.json',
        ROOT / 'Outputs/Experiments/Combined_Channel_Revision_004/references/P838_3.pdf',
        ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/plan.json']
    for p in more:
        if str(p.resolve()) not in guards:
            guards[str(p.resolve())] = sha(p)
    for p, digest in guards.items():
        assert sha(p) == digest, p
    return guards


def build_inputs():
    with np.load(OLD / 'zero_crossing_diagnostics.npz', allow_pickle=False) as f:
        prior = {k: f[k] for k in f.files}
    candidates = prior['candidate_geopotential_m']
    converted = np.full_like(candidates, np.nan)
    finite = np.isfinite(candidates)
    converted[finite] = [geometric_height_m(float(h), earth_radius_m=RE) for h in candidates[finite]]
    assert int(finite.sum()) == 33514
    records = read(OLD / 'header_inventory.json')
    weights = prior['weights_NW_NE_SW_SE']
    values = {}
    clock = None
    used = []
    for var, parameter, prefix in [('u', 131, 'pressure_u_component_of_wind_'),
                                  ('v', 132, 'pressure_v_component_of_wind_')]:
        files = [r for r in records if r['file'].startswith(prefix) and r['manifest_job_id']]
        assert len(files) == 12
        timestamps, chunks = [], []
        for record in files:
            path = RAW / record['file']
            assert sha(path) == record['sha256']
            with xr.open_dataset(path, engine='netcdf4') as ds:
                a = ds[var]
                assert a.dims == ('valid_time', 'pressure_level', 'latitude', 'longitude')
                assert int(a.GRIB_paramId) == parameter and a.units == 'm s**-1'
                assert a.GRIB_stepType == 'instant' and a.GRIB_typeOfLevel == 'isobaricInhPa'
                assert ds.pressure_level.attrs['units'] == 'hPa'
                assert np.array_equal(ds.pressure_level.values, prior['pressure_hpa'])
                assert np.array_equal(ds.latitude.values, [40, 39.75, 39.5, 39.25])
                assert np.array_equal(ds.longitude.values, [116.25, 116.5])
                assert ds.valid_time.encoding['units'].startswith('seconds since 1970-01-01')
                t = ds.valid_time.values.astype('datetime64[s]').astype('int64')
                x = a.sel(pressure_level=700, latitude=[40, 39.75], longitude=[116.25, 116.5]).values
                timestamps.append(t); chunks.append(x.reshape(len(t), 4).astype(float))
            used.append({'file': str(path), 'sha256': record['sha256'], 'parameter': parameter,
                         'unit': 'm s**-1', 'level_hpa': 700})
        t = np.concatenate(timestamps); order = np.argsort(t); t = t[order]
        assert len(t) == 7824 and np.all(np.diff(t) == 3600)
        assert clock is None or np.array_equal(clock, t)
        clock = t; values[var] = np.concatenate(chunks)[order]
        assert np.isfinite(values[var]).all()
    uc, vc = values['u'], values['v']
    u, v = uc @ weights, vc @ weights
    speed = np.hypot(u, v)
    speed_before_interpolation = np.hypot(uc, vc) @ weights
    idx = np.searchsorted(clock, prior['campaign_epoch_s'])
    assert np.array_equal(clock[idx], prior['campaign_epoch_s'])
    assert np.all(prior['valid_above_surface'][:, :, 3])
    with (OUT / 'derived_inputs.npz').open('xb') as f:
        np.savez_compressed(f, candidate_geometric_m=converted,
            campaign_epoch_s=prior['campaign_epoch_s'], test_member=prior['test_member'],
            raw_epoch_s=clock, wind700_u_corners_mps=uc, wind700_v_corners_mps=vc,
            wind700_u_site_mps=u, wind700_v_site_mps=v, wind700_speed_site_mps=speed,
            wind700_mean_corner_speed_mps=speed_before_interpolation,
            weights_NW_NE_SW_SE=weights)
    ledger = []
    for j, old in enumerate(read(OLD / 'hourly_input_ledger.json')):
        q = idx[j]
        heights = converted[j, 4][np.isfinite(converted[j, 4])]
        ledger.append({'time_utc': old['time_utc'], 'test_member': old['test_member'],
            'wind700_u_mps': float(u[q]), 'wind700_v_mps': float(v[q]),
            'wind700_speed_mps': float(speed[q]), 'wind_positive': bool(speed[q] > 0),
            'station_geometric_zero_candidates_m': heights.tolist(),
            'candidate_count': len(heights), 'used_for_model': False,
            'site_rain_mm_per_h': old['site_rain_mm_per_h'],
            'wet_event_complete': old['wet_event_complete']})
    emit('hourly_input_ledger.json', ledger)
    test = prior['test_member'].astype(bool)
    emit('input_summary.json', {'status': 'EXECUTED_PENDING_INDEPENDENT_QA',
        'campaign_hours': len(idx), 'test_members': int(test.sum()), 'raw_wind_hours': len(clock),
        'wind_source_files': len(used), 'wind_raw_corner_components': int(uc.size + vc.size),
        'wind_campaign_site_min_max_mps': [float(speed[idx].min()), float(speed[idx].max())],
        'wind_test_site_min_max_mps': [float(speed[idx][test].min()), float(speed[idx][test].max())],
        'wind_nonpositive_campaign': int((speed[idx] <= 0).sum()),
        'interpolate_components_vs_magnitude_max_difference_mps': float(np.max(np.abs(speed - speed_before_interpolation))),
        'geometric_candidates': int(finite.sum()), 'retained_null_positions': int((~finite).sum()),
        'max_geometric_minus_geopotential_m': float(np.max(converted[finite] - candidates[finite])),
        'earth_radius_m': RE, 'height_conversion_source': 'ECMWF spherical approximation',
        'antenna_common_vertical_datum_verified': False, 'selected_height_branch': None,
        'model_height_produced': False, 'new_predictions': 0, 'labels_loaded': False,
        'all_test_members_retained': True})
    emit('wind_source_files.json', used)


def coefficient_parameters():
    # Import the already verified pure physics module; never call its I/O or
    # attenuation functions, and never import the earlier training entrypoint.
    sys.path.insert(0, str(ROOT / 'Code/Combined_Channel_Experiments/Revision_004'))
    spec = importlib.util.spec_from_file_location('frozen_p838004',
        ROOT / 'Code/Combined_Channel_Experiments/Revision_004/physics.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    old = read(ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/run_manifest.json')
    assert old['frequency_and_circular_polarization_confirmed_by_user_2026_09_30'] is True
    assert old['frequency_ghz'] == 19.13 and old['main_polarization'] == 'circular'
    native = read(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_004/native_metadata_gate.json')
    assert native['status'] == 'PASS' and native['files'] == 236
    plan = read(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/plan.json')
    settings = [('LOCAL', old['frequency_ghz'], old['elevation_deg'],
        'historical protocol and frozen manifest record human confirmation 2026-09-30')]
    settings += [(r['id'], 20.7, r['elevation_deg'], 'native source matched LHCP, Revision004 gate 236 files') for r in plan['links']]
    result = []
    for site, frequency, angle, provenance in settings:
        k, alpha = mod.coefficients(frequency, angle, 'circular')
        result.append({'site': site, 'frequency_ghz': frequency, 'elevation_deg': angle,
            'polarization': 'circular', 'p838_tau_deg': 45.0, 'k': k, 'alpha': alpha,
            'unit_gamma': 'dB/km at R in mm/h', 'provenance': provenance,
            'geometry': 'existing frozen nominal geometry, not hourly measured ephemeris',
            'new_attenuation_prediction': False})
    emit('coefficient_parameters.json', result)


def synthetic_cases():
    rainsets = [[0.] * 8, [4.] * 8, [0., 0., 8., 8., 0., 0., 0., 0.],
        [0., 20., 0., 4., 15., 0., 8., 0.], [1., 2., 3., 4., 5., 6., 7., 8.],
        [30., 20., 10., 5., 2., 1., 0., 0.]]
    cases = []
    for case_id, (storm, height, wind, elevation, coeff, anchor) in enumerate(itertools.product(
            range(6), [.1, 3., 8.], [.25, 7., 30.], [5., 42.96904244342862, 89., 90.],
            [(.08, 1.1), (.11, .95)], [0, 3])):
        p = dict(wind_speed=wind, height_above_antenna=height, elevation=elevation,
            k=coeff[0], alpha=coeff[1], timestep_seconds=3600, rain_unit='mm/h',
            wind_unit='m/s', height_unit='km_above_antenna', angle_unit='degree')
        item = {'id': case_id, 'data_kind': 'SYNTHETIC_TEST_ONLY_NOT_MEASUREMENTS',
                'rain': rainsets[storm], 'anchor': anchor, 'parameters': p}
        try:
            item['result'] = hourly_power_mean(rainsets[storm], list(range(0, 8 * 3600, 3600)), anchor, **p)
            item['status'] = 'NUMERIC'
        except InsufficientContextError:
            item['result'] = None; item['status'] = 'REFUSED_INCOMPLETE_CONTEXT'
        cases.append(item)
    emit('synthetic_cases.json', cases)
    emit('synthetic_summary.json', {'cases': len(cases),
        'numeric': sum(c['status'] == 'NUMERIC' for c in cases),
        'refused_missing_context': sum(c['result'] is None for c in cases),
        'random_numbers_or_seeds_used': False, 'new_real_predictions': 0,
        'status': 'EXECUTED_PENDING_INDEPENDENT_QA'})


def main():
    guards = guard_sources()
    assert not (OUT / 'execution.json').exists()
    with (OUT / 'execution.lock').open('x+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        emit('execution.json', {'utc': now(), 'pid': os.getpid(), 'executable': sys.executable,
            'argv': sys.argv, 'python': platform.python_version(), 'numpy': np.__version__,
            'xarray': xr.__version__, 'mode': 'offline_input_conversion_and_synthetic_test_only'})
        emit('sources_before.json', guards)
        emit('code_before.json', {str(p): sha(p) for p in CODE.iterdir() if p.is_file()})
        build_inputs(); coefficient_parameters(); synthetic_cases()
        for path, digest in guards.items():
            assert sha(path) == digest, path
        emit('execution_complete.json', {'utc': now(), 'status': 'EXECUTED_PENDING_QA',
            'protected_sources': len(guards), 'changed_sources': 0, 'worker_exiting': True})
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    print(json.dumps({'status': 'EXECUTED_PENDING_QA', 'sources': len(guards)}))


if __name__ == '__main__':
    main()
