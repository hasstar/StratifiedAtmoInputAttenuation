"""Independent netCDF4/scalar/Decimal/QUADPACK verification; no core imports."""
import ast
import hashlib
import json
import math
import sys
import traceback
import warnings
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import netCDF4
import numpy as np
import scipy
from scipy.integrate import quad, IntegrationWarning

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_022'
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
TOL = 2e-10


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def checked_close(a, b, message, *, atol=TOL, rtol=0):
    assert math.isfinite(float(a)) and math.isfinite(float(b)), message
    assert abs(a - b) <= atol + rtol * abs(b), (message, a, b, abs(a - b))


def verify_synthetic():
    rows = read(OUT / 'synthetic_cases.json')
    count = refused = 0
    max_errors = {'power_mean_db': 0., 'db_mean': 0., 'anchor_db': 0.}
    differing_means = 0
    c = math.log(10) / 10
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        for row in rows:
            assert row['data_kind'] == 'SYNTHETIC_TEST_ONLY_NOT_MEASUREMENTS'
            p, rain = row['parameters'], row['rain']
            a = row['anchor']
            theta = p['elevation'] * math.pi / 180
            if p['elevation'] == 90 or p['height_above_antenna'] == 0:
                width = 0.
            else:
                width = p['height_above_antenna'] / (p['wind_speed'] / 1000 * math.tan(theta))
            if (a + 1) * 3600 + width > len(rain) * 3600:
                assert row['status'] == 'REFUSED_INCOMPLETE_CONTEXT' and row['result'] is None
                refused += 1; continue
            assert row['status'] == 'NUMERIC'
            gamma = [p['k'] * r ** p['alpha'] for r in rain]

            def point(x):
                if p['elevation'] == 90:
                    return gamma[a] * p['height_above_antenna']
                if p['height_above_antenna'] == 0:
                    return 0.
                # Direct overlap in horizontal kilometres, not the producer's
                # seconds integral and stable exponential antiderivative.
                start = (a * 3600 + x) * p['wind_speed'] / 1000
                stop = start + p['height_above_antenna'] / math.tan(theta)
                parts = []
                for j, g in enumerate(gamma):
                    lower = j * 3600 * p['wind_speed'] / 1000
                    upper = (j + 1) * 3600 * p['wind_speed'] / 1000
                    length = min(stop, upper) - max(start, lower)
                    if length > 0: parts.append(g * length / math.cos(theta))
                return math.fsum(parts)

            points = sorted({0., 3600.} | {j * 3600 - a * 3600 - width
                for j in range(len(rain) + 1) if 0 < j * 3600 - a * 3600 - width < 3600})
            low = min(point(t) for t in points)
            power_parts, db_parts = [], []
            for begin, end in zip(points, points[1:]):
                power_parts.append(quad(lambda x: math.exp(-c * (point(x) - low)),
                    begin, end, epsabs=1e-10, epsrel=1e-12, limit=100)[0])
                db_parts.append(quad(point, begin, end, epsabs=1e-10, epsrel=1e-12, limit=100)[0])
            effective = low - math.log(math.fsum(power_parts) / 3600) / c
            db_mean = math.fsum(db_parts) / 3600
            answer = row['result']
            for key, expected, error_key in [('power_mean_attenuation_db', effective, 'power_mean_db'),
                    ('arithmetic_mean_attenuation_db', db_mean, 'db_mean'),
                    ('anchor_attenuation_db', point(0), 'anchor_db')]:
                checked_close(answer[key], expected, ('synthetic', row['id'], key))
                max_errors[error_key] = max(max_errors[error_key], abs(answer[key] - expected))
            assert effective >= low - TOL and effective <= db_mean + TOL
            assert answer['hour_duration_seconds'] == 3600
            differing_means += abs(effective - db_mean) > 1e-6
            count += 1
    assert count + refused == len(rows) == 864
    return {'cases': len(rows), 'numeric': count, 'refused_context': refused,
        'max_absolute_errors_db': max_errors, 'power_and_db_mean_differ_cases_gt_1e_6_db': differing_means,
        'tolerance_db': TOL, 'reference': 'direct horizontal-box integration and adaptive QUADPACK quadrature'}


def verify_inputs():
    with np.load(OUT / 'derived_inputs.npz', allow_pickle=False) as f: actual = {k:f[k] for k in f.files}
    with np.load(OLD / 'zero_crossing_diagnostics.npz', allow_pickle=False) as f: old = {k:f[k] for k in f.files}
    before, after = old['candidate_geopotential_m'], actual['candidate_geometric_m']
    assert np.array_equal(np.isfinite(before), np.isfinite(after))
    maximum = 0.
    with localcontext() as ctx:
        ctx.prec = 40
        r = Decimal(6371229)
        for h, result in zip(before[np.isfinite(before)], after[np.isfinite(after)], strict=True):
            hd = Decimal.from_float(float(h))
            expected = float(r * hd / (r - hd))
            checked_close(float(result), expected, 'geometric height')
            maximum = max(maximum, abs(result - expected))
    wind = {'u': {}, 'v': {}}
    for item in read(OUT / 'wind_source_files.json'):
        assert sha(item['file']) == item['sha256']
        name = 'u' if item['parameter'] == 131 else 'v'
        with netCDF4.Dataset(item['file']) as ds:
            var = ds.variables[name]
            assert int(var.GRIB_paramId) == item['parameter'] and var.units == 'm s**-1'
            assert var.GRIB_stepType == 'instant' and var.GRIB_typeOfLevel == 'isobaricInhPa'
            lat, lon = list(ds['latitude'][:]), list(ds['longitude'][:])
            level = list(ds['pressure_level'][:]).index(700.)
            assert ds['pressure_level'].units == 'hPa'
            timevar = ds['valid_time']
            assert timevar.units.startswith('seconds since 1970-01-01')
            assert timevar.calendar in ('proleptic_gregorian', 'gregorian', 'standard')
            vals = var[:, level, :, :]
            assert not np.ma.getmaskarray(vals).any()
            for i, sec in enumerate(timevar[:]):
                t = int(sec)
                assert t not in wind[name]
                wind[name][t] = [float(vals[i, lat.index(y), lon.index(x)])
                    for y, x in [(40.,116.25),(40.,116.5),(39.75,116.25),(39.75,116.5)]]
    station = read(Path(read(OUT / 'wind_source_files.json')[0]['file']).parent / 'download_manifest.json')['campaign']['station']
    north = (station['latitude_deg_n'] - 39.75) / .25
    east = (station['longitude_deg_e'] - 116.25) / .25

    def bilinear(c):
        top = c[0] + east * (c[1] - c[0])
        bottom = c[2] + east * (c[3] - c[2])
        return bottom + north * (top - bottom)

    worst = 0.
    epochs = sorted(wind['u'])
    assert epochs == sorted(wind['v']) == actual['raw_epoch_s'].tolist()
    assert len(epochs) == 7824 and all(b-a == 3600 for a,b in zip(epochs,epochs[1:]))
    by_time = {}
    for i, sec in enumerate(epochs):
        u0, v0 = wind['u'][sec], wind['v'][sec]
        assert u0 == actual['wind700_u_corners_mps'][i].tolist()
        assert v0 == actual['wind700_v_corners_mps'][i].tolist()
        u, v = bilinear(u0), bilinear(v0)
        speed = math.sqrt(u * u + v * v)
        corner_mean = bilinear([math.sqrt(x*x+y*y) for x,y in zip(u0,v0,strict=True)])
        for k, expected in [('wind700_u_site_mps',u),('wind700_v_site_mps',v),
                ('wind700_speed_site_mps',speed),('wind700_mean_corner_speed_mps',corner_mean)]:
            got = float(actual[k][i]); checked_close(got, expected, k, atol=2e-12,rtol=2e-12)
            worst = max(worst, abs(got-expected))
        by_time[sec] = (u, v, speed)
    rows = read(OUT / 'hourly_input_ledger.json')
    oldrows = read(OLD / 'hourly_input_ledger.json')
    assert len(rows) == len(oldrows) == 7800
    tests = 0
    for i, (row, before_row) in enumerate(zip(rows, oldrows,strict=True)):
        assert row['time_utc'] == before_row['time_utc']
        sec = int(datetime.fromisoformat(row['time_utc'].replace('Z','+00:00')).timestamp())
        assert sec == int(old['campaign_epoch_s'][i])
        assert row['test_member'] == before_row['test_member'] == bool(old['test_member'][i])
        assert row['used_for_model'] is False
        for k,e in zip(('wind700_u_mps','wind700_v_mps','wind700_speed_mps'),by_time[sec],strict=True):
            checked_close(row[k], e, ('ledger',k),atol=2e-12,rtol=2e-12)
        assert row['station_geometric_zero_candidates_m'] == after[i,4][np.isfinite(after[i,4])].tolist()
        assert row['candidate_count'] == before_row['station_crossings']
        assert row['site_rain_mm_per_h'] == before_row['site_rain_mm_per_h']
        assert row['wet_event_complete'] == before_row['wet_event_complete']
        tests += row['test_member']
    assert tests == 687
    return {'wind_files':24, 'raw_wind_hours': len(epochs), 'corner_components_exact': 8*len(epochs),
        'campaign_ledger_rows':len(rows),'test_members':tests,'geometric_candidates':int(np.isfinite(before).sum()),
        'preserved_nan_positions':int(np.isnan(before).sum()), 'max_height_error_m':float(maximum),
        'max_wind_error_mps':worst,'test_keys_and_metadata_exact':True}


def verify_coefficients():
    path = ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_016/run_itu_baseline.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tables = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COEFFICIENT_TABLES' for t in node.targets):
            tables = ast.literal_eval(node.value)
    assert tables is not None
    rows = read(OUT / 'coefficient_parameters.json'); maxerr = 0.
    assert [r['site'] for r in rows] == ['LOCAL','chilbolton','sparsholt']
    for row in rows:
        x = math.log10(row['frequency_ghz']); base = {}
        for name, (a,b,c,m,d) in tables.items():
            value = math.fsum(aa * math.exp(-((x-bb)/cc)**2) for aa,bb,cc in zip(a,b,c,strict=True))+m*x+d
            base[name] = 10**value if name.startswith('k_') else value
        expectedk = (base['k_h']+base['k_v'])/2
        expecteda = (base['k_h']*base['alpha_h']+base['k_v']*base['alpha_v'])/(base['k_h']+base['k_v'])
        for key, expected in [('k',expectedk),('alpha',expecteda)]:
            checked_close(row[key],expected,key,atol=2e-12,rtol=2e-12)
            maxerr = max(maxerr,abs(row[key]-expected))
        assert row['p838_tau_deg'] == 45 and row['new_attenuation_prediction'] is False
    old = read(ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/run_manifest.json')
    assert old['main_polarization'] == 'circular' and old['frequency_ghz'] == 19.13
    assert old['frequency_and_circular_polarization_confirmed_by_user_2026_09_30'] is True
    return {'sites':3,'max_coefficient_abs_error':maxerr,
        'historical_human_confirmation_record_verified':True,'device_remeasurement_performed':False,
        'old_training_entrypoint_executed':False}


def main():
    assert not (OUT / 'independent_qa.json').exists()
    report = {'utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL',
        'implementation':'independent netCDF4, scalar bilinear, Decimal40, QUADPACK; no producer imports',
        'scipy':scipy.__version__,'netCDF4':netCDF4.__version__,'python':sys.version}
    try:
        assert read(OUT/'unit_tests.json')['status']=='PASS'
        report['synthetic'] = verify_synthetic()
        report['real_inputs'] = verify_inputs()
        report['coefficients'] = verify_coefficients()
        guards = read(OUT/'sources_before.json')
        for path,digest in {**guards,**read(OUT/'code_before.json')}.items():
            assert sha(path)==digest,path
        report['protected_sources_unchanged']=len(guards)
        report['status']='PASS'
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        with (OUT/'independent_qa.json').open('x',encoding='utf-8') as f:
            json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
