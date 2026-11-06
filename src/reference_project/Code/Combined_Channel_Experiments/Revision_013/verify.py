"""Independent netCDF4/scalar re-read of Revision013 input QA; no audit import."""
import hashlib
import json
import math
import msvcrt
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_013/attempt_002'
EXPECTED = {
    't': (130, 'K'), 'q': (133, 'kg kg**-1'), 'z': (129, 'm**2 s**-2'),
    'u': (131, 'm s**-1'), 'v': (132, 'm s**-1'),
    'clwc': (246, 'kg kg**-1'), 'ciwc': (247, 'kg kg**-1'),
    'crwc': (75, 'kg kg**-1'), 'cswc': (76, 'kg kg**-1'),
    'deg0l': (228024, 'm'), 'ptype': (260015, '(Code table 4.201)'),
    't2m': (167, 'K'), 'd2m': (168, 'K'), 'sp': (134, 'Pa'),
    'tclw': (78, 'kg m**-2'), 'tcc': (164, '(0 - 1)'),
    'avg_tprate': (235055, 'kg m**-2 s**-1'), 'avg_cpr': (235030, 'kg m**-2 s**-1'),
    'avg_lsprate': (235029, 'kg m**-2 s**-1'), 'avg_tsrwe': (235031, 'kg m**-2 s**-1'),
    'avg_csfr': (235056, 'kg m**-2 s**-1'), 'avg_lssfr': (235057, 'kg m**-2 s**-1')}


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            b = f.read(1048576)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scalar_stats(a):
    values = list(map(float, np.asarray(a).reshape(-1)))
    usable = [x for x in values if math.isfinite(x)]
    return {'size': len(values), 'finite': len(usable),
            'nan': sum(math.isnan(x) for x in values), 'inf': sum(math.isinf(x) for x in values),
            'negative': sum(x < 0 for x in usable), 'zero': sum(x == 0 for x in usable),
            'minimum': min(usable) if usable else None, 'maximum': max(usable) if usable else None}


def verify_statistics(a, recorded, label):
    measured = scalar_stats(a)
    for key, value in measured.items():
        require(value == recorded[key], f'{label}: {key}: {value} != {recorded[key]}')


def ranges_of_missing(all_times, flags):
    blocks = []
    for t, finite in zip(all_times, flags):
        if finite:
            continue
        if blocks and t == blocks[-1][-1] + timedelta(hours=1):
            blocks[-1].append(t)
        else:
            blocks.append([t])
    return [{'start_utc': b[0].isoformat() + 'Z', 'end_utc_inclusive': b[-1].isoformat() + 'Z', 'hours': len(b)} for b in blocks]


def main():
    require(not (OUT / 'verification.json').exists(), 'One-shot verification')
    inventory, summary, ledger = read('file_inventory.json'), read('field_summary.json'), read('hourly_input_ledger.json')
    guards = {**read('protected_sources_before.json'), **read('raw_sources_before.json')}
    for p, h in guards.items():
        require(digest(p) == h, 'Source changed: ' + p)
    blocks, clocks = defaultdict(list), defaultdict(list)
    missing = []
    file_variables = 0
    epoch = datetime(1970, 1, 1)
    for item in inventory:
        filename = Path(item['path']).name
        lo, hi = re.search(r'_(\d{8})_(\d{8})\.nc$', filename).groups()
        first = datetime.strptime(lo, '%Y%m%d'); last = datetime.strptime(hi, '%Y%m%d') + timedelta(hours=23)
        expected_times = [first + timedelta(hours=i) for i in range(int((last-first).total_seconds()/3600)+1)]
        with netCDF4.Dataset(item['path'], 'r') as nc:
            tv = nc.variables['valid_time']
            native_times = netCDF4.num2date(tv[:], tv.units, calendar=getattr(tv, 'calendar', 'standard'),
                                          only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in native_times]
            require(times == expected_times, 'Independent filename-to-UTC calendar')
            require(list(nc.variables['latitude'][:]) == ([40.,39.75,39.5,39.25] if item['pressure'] else [40.,39.75]), 'Grid latitude')
            require(list(nc.variables['longitude'][:]) == [116.25,116.5], 'Grid longitude')
            if item['pressure']:
                require(list(nc.variables['pressure_level'][:]) == [1000,925,850,700,650,500,300,200,30,5,1], 'Levels')
            for k, rec in item['variables'].items():
                file_variables += 1
                present = k in nc.variables
                require(present == rec['source_present'], 'Parameter presence flag')
                if present:
                    v = nc.variables[k]
                    require((int(v.GRIB_paramId), v.units) == EXPECTED[k], 'Independent identity mapping: ' + k)
                    require(v.GRIB_stepType == ('avg' if k.startswith('avg_') else 'instant'), 'Step type')
                    a = np.ma.filled(v[:], np.nan).astype(float)
                    require(rec['source_status'] == 'PRESENT' and not rec['nan_ledger_placeholders_only'], 'Presence claim')
                else:
                    require(filename == 'surface_instant_20260701_20260731.nc' and k in {'deg0l','ptype'}, 'Unreviewed missing variable')
                    require(rec['long_name'] is None and rec['step_type'] is None, 'Absent metadata fabricated')
                    require(rec['source_status'] == 'ABSENT_PARAMETER' and rec['nan_ledger_placeholders_only'], 'Missing disclosure')
                    a = np.full((len(times), 2, 2), math.nan)
                    missing.append((filename,k))
                verify_statistics(a, rec, filename + ':' + k)
                blocks[k].append(a); clocks[k].extend(times)
    values, time_index = {}, {}
    campaign = [datetime(2025,9,29,16)+timedelta(hours=i) for i in range(7800)]
    full_campaign_arrays = {}
    for k, b in blocks.items():
        require(len(clocks[k]) == len(set(clocks[k])) == 7824, 'Clock unique count')
        order = sorted(range(len(clocks[k])), key=clocks[k].__getitem__)
        t = [clocks[k][i] for i in order]
        values[k] = np.concatenate(b)[order]
        time_index[k] = {d:i for i,d in enumerate(t)}
        a = values[k][[time_index[k][d] for d in campaign]]
        full_campaign_arrays[k] = a
        verify_statistics(values[k], summary[k]['raw'], k+' raw')
        verify_statistics(a, summary[k]['campaign'], k+' campaign')
        flags = [all(math.isfinite(float(x)) for x in row.flat) for row in a]
        require(sum(flags) == summary[k]['campaign_all_grid_finite_hours'], 'Field finite hours')
        require(ranges_of_missing(campaign, flags) == summary[k]['campaign_incomplete_utc_ranges'], 'Missing range identity')
    test = pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_012/selected_local.parquet', columns=['time'])
    test_set = {x.to_pydatetime().replace(tzinfo=None) for x in pd.to_datetime(test['time'],utc=True)}
    require(len(test_set) == 687, 'Original test keys')
    pressure_keys = ['t','q','z','u','v','clwc','ciwc','crwc','cswc']
    rate_keys = ['avg_tprate','avg_cpr','avg_lsprate','avg_tsrwe','avg_csfr','avg_lssfr']
    checked = []
    for i, t in enumerate(campaign):
        finite = lambda k,d: all(math.isfinite(float(x)) for x in values[k][time_index[k][d]].flat)
        h0 = finite('deg0l',t)
        w = all(math.isfinite(float(x)) for k in ['u','v'] for x in values[k][time_index[k][t],3].flat)
        rain = all(finite(k,t+timedelta(hours=1)) for k in rate_keys)
        a = values['avg_tprate'][time_index['avg_tprate'][t+timedelta(hours=1)]]
        b = values['avg_tsrwe'][time_index['avg_tsrwe'][t+timedelta(hours=1)]]
        liquid = all(math.isfinite(float(x-y)) and x>=y for x,y in zip(a.flat,b.flat))
        item = {'time_utc':t.isoformat()+'Z','original_test_key':t in test_set,
                'h0_all_corners_finite':h0,'wind700_all_grid_finite':w,
                'six_rates_valid_t_plus_1h_finite':rain,'total_minus_snow_nonnegative_all_corners':liquid,
                'nine_pressure_fields_all_grid_finite':all(finite(k,t) for k in pressure_keys),
                'esst_input_coavailability_only':h0 and w and rain and liquid}
        require(item == ledger[i], 'Independent per-hour ledger: '+str(t))
        checked.append(item)
    require(len(ledger) == 7800, 'Ledger length')
    complete = read('audit_complete.json')
    for key in ledger[0]:
        if key == 'time_utc':
            continue
        require(sum(r[key] for r in checked) == complete['ledger_counts'][key], 'Ledger total')
        require(sum(r[key] for r in checked if r['original_test_key']) == complete['test_ledger_counts'][key], 'Test ledger total')
    g = read('geometry_diagnostics.json')
    z = full_campaign_arrays['z'] / 9.80665
    below = sum(float(x)<117.5 for x in z.flat)
    require(below == g['geopotential_height_below_station_117_5m_grid_values'], 'Height mask')
    plevels = [1000,925,850,700,650,500,300,200,30,5,1]
    subground = sum(p*100>float(s) for row in full_campaign_arrays['sp'] for s in row.flat for p in plevels)
    require(subground == g['pressure_exceeds_surface_pressure_common_grid_values'], 'Below surface pressure')
    require(int((np.diff(z,axis=1)<=0).sum()) == g['geopotential_height_nonincreasing_adjacent_pairs'], 'Height ordering')
    require(not complete['all_seven_complete'] and complete['new_training'] == complete['new_model_inference'] == complete['weather_requests'] == 0, 'Scope')
    for p,h in guards.items():
        require(digest(p) == h, 'Source changed at verification end')
    with (OUT/'execution.lock').open('r+b') as f:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    result = {'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
              'scope':'input QA, not model implementation or scores',
              'method':'independent netCDF4/scalar statistics and filename-derived UTC; no production audit import',
              'files':len(inventory),'variable_file_checks':file_variables,'hourly_ledger_rows':len(checked),
              'absent_parameters_verified':missing,'protected_source_count':len(guards),
              'data_sources_unchanged':True,'os_lock':'acquired and released',
              'no_training_no_inference_no_weather_requests':True,
              'seven_model_completion':False,'verification_script_sha256':digest(__file__),
              'checked_outputs_sha256':{n:digest(OUT/n) for n in ['file_inventory.json','field_summary.json','hourly_input_ledger.json','geometry_diagnostics.json','audit_complete.json']}}
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
