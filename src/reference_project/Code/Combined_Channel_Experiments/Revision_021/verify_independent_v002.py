"""Independent netCDF4 + scalar recomputation; imports no production audit module."""
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
RAW = Path('E:/0_Workflow_System/01_PhD/Data/era5_campaign_20250930_20260820/raw')
P = [1000, 925, 850, 700, 650, 500, 300, 200, 30, 5, 1]
counts = Counter()
errors = {'candidate_geopotential_m': 0., 'bracket_width_geopotential_m': 0., 'rain_mm_per_h': 0.}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def require(value, category):
    if not value:
        raise AssertionError(category)
    counts[category] += 1


def number(actual, expected, kind, atol, rtol=0.):
    a = float('nan') if actual is None else float(actual)
    if math.isnan(expected):
        require(math.isnan(a), kind + '_null_mask')
    else:
        require(math.isfinite(a), kind + '_finite')
        diff = abs(a - expected)
        errors[kind] = max(errors.get(kind, 0.), diff)
        require(diff <= atol + rtol * abs(expected), kind + '_value')


def epoch(text):
    return int(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp())


def load(key, prefix):
    jobs = [j for j in manifest['jobs'] if Path(j['target'].replace('\\', '/')).name.startswith(prefix)]
    require(len(jobs) == 12, 'monthly_file_count')
    result = {}
    for job in jobs:
        name = Path(job['target'].replace('\\', '/')).name
        with netCDF4.Dataset(RAW / name, 'r') as d:
            var = d.variables.get(key)
            stamps = np.asarray(d.variables['valid_time'][:], dtype=np.int64)
            require(d.variables['valid_time'].units.startswith('seconds since 1970-01-01'), 'time_units')
            if var is None:
                require(key == 'deg0l' and name == 'surface_instant_20260701_20260731.nc', 'explicit_absent_field')
                a = np.full((len(stamps), 2, 2), np.nan)
            else:
                a = np.ma.filled(var[:], np.nan).astype(float)
                a = a[:, :, :2, :] if 'pressure_level' in var.dimensions else a[:, :2, :]
            for i, stamp in enumerate(stamps):
                require(int(stamp) not in result, 'no_duplicate_source_hour')
                result[int(stamp)] = a[i]
    return result


manifest = read(RAW / 'download_manifest.json')
headers = read(OUT / 'header_inventory.json')
require(len(headers) == 340, 'all_header_records')
terrain = []
for record in headers:
    with netCDF4.Dataset(RAW / record['file'], 'r') as d:
        require(dict((k, len(v)) for k, v in d.dimensions.items()) == record['sizes'], 'header_dimensions')
        for item in record['variables']:
            v = d.variables[item['key']]
            require(list(v.dimensions) == item['dims'], 'header_variable_dimensions')
            for k, value in item['attrs'].items():
                require(getattr(v, k) == value, 'header_variable_metadata')
        for key, v in d.variables.items():
            description = (key + ' ' + getattr(v, 'long_name', '') + ' ' + getattr(v, 'standard_name', '')).lower()
            if 'pressure_level' not in v.dimensions and (getattr(v, 'GRIB_paramId', -1) == 129 or
                    any(s in description for s in ('orograph', 'surface_geopotential', 'surface_altitude', 'surface elevation'))):
                terrain.append([record['file'], key])
require(terrain == [], 'bounded_surface_terrain_absence')
require(read(OUT / 'unit_checks.json')['status'] == 'PASS', 'boundary_tests_pass')

T = load('t', 'pressure_temperature_')
Z = load('z', 'pressure_geopotential_')
SP = load('sp', 'surface_instant_')
D0 = load('deg0l', 'surface_instant_')
TP = load('avg_tprate', 'surface_mean_rate_')
SF = load('avg_tsrwe', 'surface_supp_mean_rate_01_')
for source in (Z, SP, D0, TP, SF):
    require(sorted(source) == sorted(T), 'source_clock_identity')
station = manifest['campaign']['station']
lat, lon = station['latitude_deg_n'], station['longitude_deg_e']
coords = [(40., 116.25), (40., 116.5), (39.75, 116.25), (39.75, 116.5)]
w = [(1 - abs(lat - y) / .25) * (1 - abs(lon - x) / .25) for y, x in coords]
require(abs(math.fsum(w) - 1) < 1e-14, 'geometric_weights_sum')
with np.load(OUT / 'zero_crossing_diagnostics.npz', allow_pickle=False) as archive:
    npz = {key: archive[key] for key in archive.files}
for a, b in zip(npz['weights_NW_NE_SW_SE'], w, strict=True):
    number(a, b, 'weight', 2e-12)
ledger = read(OUT / 'hourly_input_ledger.json')
require(len(ledger) == 7800, 'campaign_ledger_size')
test = pd.read_parquet(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012/selected_local.parquet', columns=['time'])
test_set = {epoch(str(s)) for s in test.time}
require(len(test_set) == 687, 'original_test_key_count')
candidate_total = 0
for i, row in enumerate(ledger):
    stamp = epoch(row['time_utc'])
    require(stamp == epoch(manifest['start_utc']) + 3600 * i, 'continuous_campaign_time')
    require(stamp == int(npz['campaign_epoch_s'][i]), 'npz_time_identity')
    require(epoch(row['rate_valid_end_utc']) == stamp + 3600, 'rate_end_alignment')
    require(epoch(row['instant_valid_utc']) == stamp, 'instant_alignment')
    require(row['test_member'] == (stamp in test_set) == bool(npz['test_member'][i]), 'test_membership')
    corners_t = [[float(T[stamp][k, c // 2, c % 2]) for k in range(11)] for c in range(4)]
    corners_h = [[float(Z[stamp][k, c // 2, c % 2]) / 9.80665 for k in range(11)] for c in range(4)]
    corners_valid = [[math.isfinite(corners_t[c][k]) and math.isfinite(corners_h[c][k])
                      and P[k] * 100 <= float(SP[stamp][c // 2, c % 2]) for k in range(11)] for c in range(4)]
    station_valid = [all(corners_valid[c][k] for c in range(4)) for k in range(11)]
    station_t = [math.fsum(w[c] * corners_t[c][k] for c in range(4)) if station_valid[k] else math.nan for k in range(11)]
    station_h = [math.fsum(w[c] * corners_h[c][k] for c in range(4)) if station_valid[k] else math.nan for k in range(11)]
    for c, (ts, hs, valid) in enumerate(zip(corners_t + [station_t], corners_h + [station_h], corners_valid + [station_valid], strict=True)):
        ndown = nup = nzero = nbad = 0
        require(np.array_equal(npz['valid_above_surface'][i, c], valid), 'above_surface_mask')
        for k in range(11):
            nzero += int(valid[k] and ts[k] == 273.15)
            if k == 10:
                break
            candidate = width = math.nan
            if valid[k] and valid[k + 1]:
                if hs[k + 1] <= hs[k]:
                    nbad += 1
                elif ts[k] > 273.15 > ts[k + 1]:
                    ndown += 1
                    width = hs[k + 1] - hs[k]
                    # Independent two-point barycentric expression.
                    candidate = (hs[k] * (ts[k + 1] - 273.15) + hs[k + 1] * (273.15 - ts[k])) / (ts[k + 1] - ts[k])
                elif ts[k] < 273.15 < ts[k + 1]:
                    nup += 1
            number(npz['candidate_geopotential_m'][i, c, k], candidate, 'candidate_geopotential_m', 2e-10)
            number(npz['bracket_width_geopotential_m'][i, c, k], width, 'bracket_width_geopotential_m', 2e-10)
        for name, value in [('warm_to_cold_count', ndown), ('cold_to_warm_count', nup),
                            ('exact_zero_count', nzero), ('nonascending_pair_count', nbad),
                            ('valid_level_count', sum(valid))]:
            require(int(npz[name][i, c]) == value, 'profile_' + name)
        candidate_total += ndown
        if c == 4:
            require(row['station_crossings'] == ndown and row['station_exact_zero_levels'] == nzero, 'station_ledger_crossings')
    dg = D0[stamp].reshape(4)
    for a, b in zip(row['deg0l_corners_m_agl'], dg, strict=True):
        number(a, float(b), 'deg0l_native_m', 0.)
    require(row['deg0l_all_corners_finite'] == bool(np.isfinite(dg).all()), 'deg0l_finite_mask')
    require(row['deg0l_zero_corner_count'] == int((dg == 0).sum()), 'deg0l_zero_semantics')

context = read(OUT / 'rain_context_ledger.json')
events = read(OUT / 'rain_events.json')
time_keys = sorted(T)
require(len(context) == len(time_keys) - 1 == 7823, 'all_available_rain_intervals')
rain = []
for i, row in enumerate(context):
    stamp = epoch(row['time_utc'])
    require(stamp == time_keys[i] and epoch(row['rate_valid_end_utc']) == stamp + 3600, 'context_time_mapping')
    values = [(float(TP[stamp + 3600][c // 2, c % 2]) - float(SF[stamp + 3600][c // 2, c % 2])) * 3600 for c in range(4)]
    ok = all(math.isfinite(x) and x >= 0 for x in values)
    expected = math.fsum(w[c] * values[c] for c in range(4)) if ok else math.nan
    require(row['all_corners_nonnegative'] == ok, 'rain_negative_or_missing_mask')
    for a, b in zip(row['liquid_rain_corners_mm_per_h'], values, strict=True):
        number(a, b, 'rain_mm_per_h', 2e-12, 2e-12)
    number(row['site_rain_mm_per_h'], expected, 'rain_mm_per_h', 2e-12, 2e-12)
    rain.append(expected)
# Independent sequential state machine, not vector edge detection.
segments = []; segment = []
for i, value in enumerate(rain + [math.nan]):
    if math.isfinite(value) and value > 0:
        segment.append(i)
    elif segment:
        segments.append(segment); segment = []
require(len(segments) == len(events), 'event_count')
ids = [-1] * len(rain)
for eid, segment in enumerate(segments):
    a, b = segment[0], segment[-1]
    e = events[eid]
    left = a == 0 or not (math.isfinite(rain[a - 1]) and rain[a - 1] == 0)
    right = b == len(rain) - 1 or not (math.isfinite(rain[b + 1]) and rain[b + 1] == 0)
    require(e['event_id'] == eid and e['hours'] == len(segment), 'event_length_id')
    require(epoch(e['start_utc']) == time_keys[a] and epoch(e['last_bin_start_utc']) == time_keys[b], 'event_utc_bounds')
    require(e['left_censored'] == left and e['right_censored'] == right and e['complete'] == (not left and not right), 'event_censoring')
    peak = max(rain[k] for k in segment)
    number(e['peak_hourly_mm_per_h'], peak, 'rain_mm_per_h', 2e-12, 2e-12)
    require(e['hourly_peak_type_candidate'] == ('convective' if peak > 10 else 'stratiform'), 'hourly_peak_type_only')
    for k in segment:
        ids[k] = eid
for i, row in enumerate(context):
    require(row['event_id'] == ids[i], 'context_event_membership')
context_map = {epoch(r['time_utc']): r for r in context}
for row in ledger:
    cr = context_map[epoch(row['time_utc'])]
    eid = cr['event_id']
    require(row['wet_event_id'] == eid and row['all_corners_liquid_nonnegative'] == cr['all_corners_nonnegative'], 'campaign_context_join')
    require(row['wet_event_complete'] == (events[eid]['complete'] if eid >= 0 else None), 'campaign_event_completeness')
    number(row['site_rain_mm_per_h'], math.nan if cr['site_rain_mm_per_h'] is None else cr['site_rain_mm_per_h'], 'rain_mm_per_h', 2e-12, 2e-12)

intersections = {}
for name, subset in [('campaign_wet', [r for r in ledger if r['wet_event_id'] >= 0]),
                     ('test_wet', [r for r in ledger if r['test_member'] and r['wet_event_id'] >= 0]),
                     ('test_deg0l_missing', [r for r in ledger if r['test_member'] and not r['deg0l_all_corners_finite']])]:
    intersections[name] = {'hours': len(subset), 'station_no_crossing': sum(r['station_crossings'] == 0 for r in subset),
                           'station_one_crossing': sum(r['station_crossings'] == 1 for r in subset),
                           'station_multiple_crossings': sum(r['station_crossings'] > 1 for r in subset),
                           'model_eligibility': 'NOT_ESTABLISHED'}
guards = read(OUT / 'computation_sources_before.json')
for path, digest in guards.items():
    require(sha(path) == digest, 'protected_source_sha')
result = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(), 'executable': sys.executable,
          'netCDF4': netCDF4.__version__, 'implementation': 'netCDF4 direct reads, scalar barycentric zero crossing, sequential event state machine; no production import',
          'checked_headers': len(headers), 'checked_profiles': len(ledger) * 5,
          'checked_adjacent_layer_positions': len(ledger) * 5 * 10, 'numeric_crossing_candidates': candidate_total,
          'checked_context_hours': len(context), 'checked_events': len(events),
          'checks_by_category': dict(counts), 'max_absolute_errors': errors,
          'predeclared_tolerances': {'height_m_atol': 2e-10, 'rain_rtol_and_atol': 2e-12},
          'new_diagnostic_intersections_not_eligibility': intersections,
          'labels_read': False, 'new_scores': 0, 'protected_sources': len(guards)}
with (OUT / 'independent_qa.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({k: v for k, v in result.items() if k != 'checks_by_category'}, ensure_ascii=False))
