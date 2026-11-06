"""Independent, offline audit of the frozen reduced GBS scope.

No imports of downloader/old decoder/old geometry. No model loading, fit,
predict, network access or input writes. Original NetCDF/OPeNDAP ASCII are
decoded directly; pyproj + scipy solve the declared WGS84 ray independently.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
from pyproj import Transformer
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT = BASE / 'field_audit_001'
SCOPE_SHA = '43864c01e313b768ca21a8cbda396cda7a9a3dfbab71975f524a80089333f052'
SCOPE_FP = 'b701f2481acdef06fcfa1bbe48dbd584e32261fd9231801357671c693f10d047'
CATALOG = {
    'geopotential': ('Z', 'z', 129, {'m**2 s**-2', 'm2 s-2'}),
    'u_component_of_wind': ('U', 'u', 131, {'m s**-1', 'm s-1'}),
    'ozone_mass_mixing_ratio': ('O3', 'o3', 203, {'kg kg**-1', 'kg kg-1'}),
    'relative_humidity': ('R', 'r', 157, {'%'}),
    'convective_available_potential_energy': ('CAPE', 'cape', 59, {'J kg**-1', 'J kg-1', 'm**2 s**-2'}),
    'maximum_2m_temperature_since_previous_post_processing': ('MX2T', 'mx2t', 201, {'K'}),
}
FEATURES = [
    'surface__convective_available_potential_energy',
    'surface__maximum_2m_temperature_since_previous_post_processing',
    'p850__ozone_mass_mixing_ratio', 'p850__relative_humidity',
    'p650__u_component_of_wind', 'p500__ozone_mass_mixing_ratio',
    'p5__ozone_mass_mixing_ratio', 'p5__u_component_of_wind',
    'p1__geopotential', 'p1__u_component_of_wind',
]
FIXED = {5: 35979.045999395494, 500: 5579.330236996233,
         650: 3592.7169282522495, 850: 1457.634631232037}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
TO_XYZ = Transformer.from_crs(4979, 4978, always_xy=True)
TO_GEO = Transformer.from_crs(4978, 4979, always_xy=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def local_path(value):
    p = (ROOT / value).resolve()
    require(p.is_relative_to(ROOT), f'Outside project: {value}')
    return p


def exact_grid(values, longitude=False):
    a = np.asarray(values, dtype=float)
    require(np.isfinite(a).all(), 'Non-finite coordinate')
    if longitude:
        a = (a + 180.) % 360. - 180.
    else:
        require(np.all(np.abs(a) <= 90), 'Latitude range')
    q = np.rint(a * 4).astype(int)
    require(np.all(np.abs(a * 4 - q) < 1e-8), 'Not a published quarter-degree grid')
    require(len(set(q.tolist())) == len(q), 'Duplicate grid coordinate')
    return q.tolist()


def utc_seconds(values, units, calendar):
    require(calendar in ('gregorian', 'standard', 'proleptic_gregorian'), 'Unsupported calendar')
    m = re.fullmatch(r'(seconds|hours) since (\d{4}-\d{2}-\d{2})(?: (\d{2}:\d{2}:\d{2}))?', units)
    require(m is not None, f'Unsupported time units: {units}')
    origin = datetime.fromisoformat(m[2] + 'T' + (m[3] or '00:00:00')).replace(tzinfo=timezone.utc)
    raw = np.asarray(values, dtype=float)
    require(np.isfinite(raw).all(), 'Non-finite time')
    t = raw * (3600 if m[1] == 'hours' else 1) + (origin - EPOCH).total_seconds()
    require(np.all(np.abs(t - np.rint(t)) < 1e-7), 'Nonintegral UTC seconds')
    t = np.rint(t).astype(np.int64).tolist()
    require(len(set(t)) == len(t), 'Duplicate UTC time')
    require(all(v % 3600 == 0 for v in t), 'Not exact hourly UTC')
    return t


def check_identity(variable, parameter_id, units, short_name):
    _, short, pid, accepted = CATALOG[variable]
    require(int(parameter_id) == pid, f'Parameter identity mismatch: {variable}')
    require(units in accepted, f'Unit mismatch: {variable}: {units}')
    require(short_name.lower() == short, f'Short-name mismatch: {variable}')


def das_blocks(text):
    result = {}
    for name, body in re.findall(r'^    (\w+) \{(.*?)^    \}', text, re.M | re.S):
        require(name not in result, f'Duplicate DAS block: {name}')
        attrs = {}
        for kind, attr, raw in re.findall(r'^\s+(String|Int\d+|Float\d+) (\w+) (.*?);\s*$', body, re.M):
            require(attr not in attrs, f'Duplicate DAS attribute: {attr}')
            attrs[attr] = json.loads(raw) if kind == 'String' else float(raw)
        result[name] = attrs
    return result


def parse_ascii(text, das, variable):
    """Decode DAP ASCII without the acquisition parser or saved records."""
    nc, short, pid, _ = CATALOG[variable]
    blocks = das_blocks(das)
    require(nc in blocks, 'Missing actual parameter DAS block')
    check_identity(variable, blocks[nc]['ecmwf_parameter'], blocks[nc]['units'], blocks[nc]['short_name'])
    require(blocks[nc].get('rda_dataset') == 'ds633.0', 'Unexpected NCAR dataset')
    require(blocks['level']['units'] == 'hPa', 'Pressure units not hPa')
    require(blocks['latitude']['units'] == 'degrees_north' and blocks['longitude']['units'] == 'degrees_east', 'Coordinate units')
    header, body = re.split(r'\n-{10,}\s*\n', text, maxsplit=1)
    declaration = re.findall(r'Float32\s+' + re.escape(nc) + r'((?:\[\w+\s*=\s*\d+\])+);', header)
    require(len(declaration) == 1, 'Ambiguous array declaration')
    dimensions = re.findall(r'\[(\w+)\s*=\s*(\d+)\]', declaration[0])
    require([v[0] for v in dimensions] == ['time', 'level', 'latitude', 'longitude'], 'Actual pressure axes/order')
    shape = tuple(int(v[1]) for v in dimensions)
    require(all(v > 0 for v in shape), 'Empty ASCII axis')
    dataset = re.search(r'\}\s*(files/g/d633000/[^;]+);\s*$', header)
    require(dataset is not None and f'.128_{pid:03d}_{short}.' in dataset[1], 'Dataset parameter filename identity')
    lines = [x.strip() for x in body.splitlines() if x.strip()]
    actual_shape = f'{nc}.{nc}' + ''.join(f'[{n}]' for n in shape)
    require(lines[0] == actual_shape, 'Actual array shape differs from DDS')
    arr = np.full(shape, np.nan)
    seen = set()
    idx = 1
    while idx < len(lines) and lines[idx].startswith('['):
        m = re.fullmatch(r'((?:\[\d+\]){3}),\s*(.*)', lines[idx])
        require(m is not None, 'Malformed indexed ASCII row')
        key = tuple(int(x) for x in re.findall(r'\[(\d+)\]', m[1]))
        require(key not in seen, 'Duplicate ASCII row')
        require(all(0 <= k < n for k, n in zip(key, shape)), 'ASCII index outside shape')
        vals = [float(x.strip()) for x in m[2].split(',')]
        require(len(vals) == shape[3], 'ASCII longitude count')
        arr[key] = vals
        seen.add(key)
        idx += 1
    require(len(seen) == math.prod(shape[:3]), 'Missing ASCII array rows')
    require(np.isfinite(arr).all(), 'Non-finite ASCII value')
    for k in ('_FillValue', 'missing_value'):
        if k in blocks[nc]:
            require(not np.any(np.isclose(arr, blocks[nc][k], rtol=1e-6, atol=0)), 'ASCII missing sentinel')
    maps = {}
    for (axis, n) in dimensions:
        require(idx + 1 < len(lines) and lines[idx] == f'{nc}.{axis}[{n}]', f'Missing actual map: {axis}')
        vals = [float(x.strip()) for x in lines[idx + 1].split(',')]
        require(len(vals) == int(n) and len(set(vals)) == len(vals), f'Map count/duplicates: {axis}')
        maps[axis] = vals
        idx += 2
    require(idx == len(lines), 'Trailing or duplicate ASCII data')
    times = utc_seconds(maps['time'], blocks['time']['units'], blocks['time'].get('calendar', 'gregorian'))
    levels = maps['level']
    require(all(math.isfinite(x) and x in (1, 5, 500, 650, 850) for x in levels), 'Unexpected pressure level')
    lats = exact_grid(maps['latitude'])
    lons = exact_grid(maps['longitude'], True)
    decoded = {}
    for at in np.ndindex(shape):
        key = (variable, int(levels[at[1]]), times[at[0]], lats[at[2]], lons[at[3]])
        require(key not in decoded, 'Duplicate actual sample')
        decoded[key] = float(arr[at])
    return decoded, {'layout': 'NCAR_OPeNDAP_pressure_ASCII', 'shape': list(shape),
                     'dataset_path': dataset[1], 'variable': variable, 'parameter_id': pid,
                     'units': blocks[nc]['units'], 'actual_decoded_values': len(decoded),
                     'pressure_levels_hpa': levels, 'latitude': maps['latitude'], 'longitude': maps['longitude']}


@contextmanager
def nc_members(path):
    opened = []
    archive = None
    try:
        if zipfile.is_zipfile(path):
            archive = zipfile.ZipFile(path)
            for n in sorted(archive.namelist()):
                if n.endswith('.nc'):
                    opened.append((n, netCDF4.Dataset(n, memory=archive.read(n))))
        else:
            opened.append((Path(path).name, netCDF4.Dataset(path, mode='r')))
        require(bool(opened), 'No NetCDF payload')
        yield opened
    finally:
        for _, ds in opened:
            ds.close()
        if archive is not None:
            archive.close()


def parse_netcdf(path, wanted):
    decoded, evidence = {}, []
    with nc_members(path) as members:
        for member, ds in members:
            for variable in sorted({k[0] for k in wanted}):
                short = CATALOG[variable][1]
                if short not in ds.variables:
                    continue
                v = ds.variables[short]
                check_identity(variable, v.getncattr('GRIB_paramId'), v.getncattr('units'), v.getncattr('GRIB_shortName'))
                pressure = 'pressure_level' in v.dimensions
                dims = ('valid_time', 'pressure_level', 'latitude', 'longitude') if pressure else ('valid_time', 'latitude', 'longitude')
                require(v.dimensions == dims, f'Unexpected actual NetCDF dimensions: {v.dimensions}')
                require(v.getncattr('GRIB_typeOfLevel') == ('isobaricInhPa' if pressure else 'surface'), 'NetCDF level type')
                require(v.getncattr('GRIB_stepType') == ('max' if short == 'mx2t' else 'instant'), 'NetCDF time statistic')
                time = ds.variables['valid_time']
                times = utc_seconds(time[:], time.units, time.calendar)
                lat, lon = ds.variables['latitude'], ds.variables['longitude']
                require(lat.units == 'degrees_north' and lon.units == 'degrees_east', 'NetCDF coordinate units')
                lats, lons = exact_grid(lat[:]), exact_grid(lon[:], True)
                if pressure:
                    level = ds.variables['pressure_level']
                    require(level.units == 'hPa', 'NetCDF pressure units')
                    levels = np.asarray(level[:], dtype=float).tolist()
                    require(len(set(levels)) == len(levels) and all(math.isfinite(p) for p in levels), 'NetCDF duplicate/nonfinite level')
                else:
                    levels = [0]
                require(v.shape == tuple(len(x) for x in ([times, levels, lats, lons] if pressure else [times, lats, lons])), 'NetCDF shape/count')
                maps = [{k: i for i, k in enumerate(a)} for a in (times, levels, lats, lons)]
                arr = v[:]
                found = 0
                for key in sorted(wanted):
                    if key[0] != variable:
                        continue
                    parts = (key[2], key[1], key[3], key[4])
                    if not all(k in d for k, d in zip(parts, maps)):
                        continue
                    indices = tuple(d[k] for k, d in zip(parts, maps))
                    if not pressure:
                        indices = (indices[0], indices[2], indices[3])
                    value = arr[indices]
                    require(not np.ma.is_masked(value) and np.isfinite(value), f'Missing required NetCDF sample: {key}')
                    require(key not in decoded, f'Duplicate required NetCDF sample across members: {key}')
                    decoded[key] = float(value)
                    found += 1
                evidence.append({'member': member, 'variable': variable, 'parameter_id': int(v.GRIB_paramId),
                                 'units': v.units, 'shape': list(v.shape), 'step_type': v.GRIB_stepType,
                                 'pressure_levels_hpa': levels, 'required_values_found': found})
    return decoded, {'layout': 'CDS_NetCDF', 'fields': evidence}


def compare_required(actual, wanted):
    require(set(wanted).issubset(actual), f'Missing required raw keys: {len(set(wanted) - set(actual))}')
    delta = 0.
    for key, value in wanted.items():
        require(math.isfinite(value) and math.isfinite(actual[key]), 'Required value not finite')
        diff = abs(actual[key] - value)
        require(diff <= 1e-12 * max(1., abs(value)), f'Raw versus table value mismatch: {key}: {actual[key]} != {value}')
        delta = max(delta, diff)
    return delta


def nearest(lat, lon):
    require(math.isfinite(lat) and math.isfinite(lon) and abs(lat) <= 90, 'Invalid intersection coordinate')
    return [math.floor(lat * 4 + .5), (math.floor(lon * 4 + .5) + 720) % 1440 - 720]


def ray(link):
    o = np.array(TO_XYZ.transform(link['longitude_deg_e'], link['latitude_deg_n'], link['height_amsl_m']))
    s = np.array(TO_XYZ.transform(link['satellite_longitude_deg_e'], 0., link['nominal_geo_altitude_km'] * 1000))
    d = (s - o) / np.linalg.norm(s - o)
    require(np.linalg.norm(o - link['origin_ecef_m']) < 1e-6, 'Origin differs from independent WGS84')
    require(np.linalg.norm(d - link['direction_ecef']) < 1e-12, 'Direction differs from nominal station-satellite ray')
    return o, d


def intersection(o, d, height):
    require(np.isfinite(o).all() and np.isfinite(d).all() and abs(np.linalg.norm(d) - 1) < 1e-12, 'Invalid ray vector')
    require(math.isfinite(height) and height > TO_GEO.transform(*o)[2], 'Layer is not above antenna')
    def residual(t):
        return TO_GEO.transform(*(o + t * d))[2] - height
    require(residual(500000) > 0, 'Forward height not bracketed')
    distance = brentq(residual, 0., 500000., xtol=1e-8)
    lon, lat, actual_height = TO_GEO.transform(*(o + distance * d))
    require(distance > 0 and abs(actual_height - height) < .001, 'Forward intersection residual')
    return {'cell': nearest(lat, lon), 'lat': lat, 'lon': lon, 'height': height,
            'slant': distance, 'residual': actual_height - height}


def check_geometry(scope, table_rows, units):
    station = {x['id']: x for x in scope['links']}
    rays = {k: ray(v) for k, v in station.items()}
    record_map = {}
    surface_map = {}
    for unit in units:
        for r in unit['records']:
            hours = r['hours_utc_epoch']
            require(len(set(hours)) == len(hours), 'Duplicate unit hours')
            if 'level_hpa' not in r:
                for t in hours:
                    key = (r['link_id'], t)
                    require(key not in surface_map, 'Duplicate surface unit')
                    surface_map[key] = r['surface_cell']
            else:
                require(len(hours) == len(r['rows']), 'Geometry row count')
                for t, row in zip(hours, r['rows']):
                    key = (r['link_id'], t, r['level_hpa'])
                    require(key not in record_map, 'Duplicate pressure unit')
                    record_map[key] = row
    max_residual = max_slant_diff = max_coordinate_diff = 0.
    count = 0
    records = []
    for row in table_rows:
        sid, t = row['station'], row['utc_epoch_seconds']
        link = station[sid]
        cell = nearest(link['latitude_deg_n'], link['longitude_deg_e'])
        require(surface_map[(sid, t)] == cell, 'Surface not station-nearest grid')
        for name in FEATURES[:2]:
            require(row['provenance'][name]['key'][3:] == cell, 'Surface provenance grid mismatch')
        for level in (1, 5, 500, 650, 850):
            old = record_map[(sid, t, level)]
            if level == 1:
                z = row['values']['p1__geopotential']
                H = z / 9.80665
                require(0 < H < 6371229, 'Invalid geopotential-to-height input')
                height = 6371229 * H / (6371229 - H)
                require(old['status'] == 'STABLE_NEAREST' and 1 <= old['iteration'] <= 8, 'Invalid dynamic 1hPa status')
                require(abs(old['z'] - z) < 1e-6, 'Geometry z differs from actual input z')
            else:
                height = FIXED[level]
                require(scope['fixed_heights'][str(level)]['geometric_height_m'] == height, 'Fixed height changed')
                require(old['status'] == 'STABLE_FIXED_HEIGHT_NEAREST', 'Invalid fixed-height status')
            result = intersection(*rays[sid], height)
            require(old['cell'] == result['cell'], 'Recorded intersection nearest-grid mismatch')
            coord_diff = max(abs(old['intersection_lat'] - result['lat']), abs(old['intersection_lon'] - result['lon']))
            slant_diff = abs(old['slant_m'] - result['slant'])
            require(coord_diff < 1e-7 and slant_diff < .01 and abs(old['height_m'] - height) < .001, 'Stored intersection not independently reproduced')
            for name in FEATURES:
                if name.startswith(f'p{level}__'):
                    require(row['provenance'][name]['key'][3:] == result['cell'], 'Pressure field not at LOS-nearest grid')
            max_residual = max(max_residual, abs(result['residual']))
            max_slant_diff = max(max_slant_diff, slant_diff)
            max_coordinate_diff = max(max_coordinate_diff, coord_diff)
            count += 1
            records.append({'station': sid, 'utc_epoch_seconds': t, 'pressure_hpa': level, **result})
    require(count == 9360 and len(surface_map) == 1872 and len(record_map) == 9360, 'Geometry scope cardinality')
    return {'status': 'PASS', 'pressure_intersections_checked': count, 'surface_station_hours_checked': 1872,
            'max_independent_height_residual_m': max_residual, 'max_original_slant_difference_m': max_slant_diff,
            'max_original_lat_lon_difference_deg': max_coordinate_diff,
            'implementation': 'pyproj EPSG4979/4978 + scipy brentq, no acquisition geometry imports',
            'qualification': 'Verifies declared nominal GEO/WGS84/AMSL-as-geodetic approximation, not precise ephemeris/geoid truth'}, records


def emit(name, data):
    p = OUT / name
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def main():
    OUT.mkdir(exist_ok=True)
    require(not (OUT / 'actual_field_qa.json').exists() and not (OUT / 'failure.json').exists(), 'Audit result already frozen; use a new audit version')
    started = datetime.now(timezone.utc).isoformat()
    guarded = {}
    def protect(path, expected=None):
        p = Path(path)
        digest = sha(p)
        require(expected is None or digest == expected, f'Input SHA mismatch: {p}')
        guarded[str(p.relative_to(ROOT))] = digest
        return p
    try:
        scope = read(protect(BASE / 'scope_manifest.json', SCOPE_SHA))
        require(scope['fingerprint'] == SCOPE_FP and fp({k: v for k, v in scope.items() if k != 'fingerprint'}) == SCOPE_FP, 'Scope fingerprint mismatch')
        require(scope['stations'] == ['chilbolton', 'sparsholt'] and scope['feature_names'] == FEATURES and scope['auxiliary_pairs'] == [], 'Frozen scope mismatch')
        ref = scope['local_acquisition_table']
        table = read(protect(local_path(ref['path']), ref['sha256']))
        require(table['feature_names'] == FEATURES and len(table['rows']) == 1872, 'Actual table schema/count')
        expected_hours = []
        for day in scope['dates_utc']:
            midnight = int((datetime.fromisoformat(day).replace(tzinfo=timezone.utc) - EPOCH).total_seconds())
            expected_hours.extend(midnight + h * 3600 for h in range(24))
        require(len(scope['dates_utc']) == 39 and len(set(expected_hours)) == 936 and expected_hours == scope['hours_utc_epoch_seconds'], 'Calendar exact scope')
        expected_rows = set(itertools.product(scope['stations'], expected_hours))
        seen = set()
        required = defaultdict(dict)
        all_keys = set()
        for row in table['rows']:
            identity = (row['station'], row['utc_epoch_seconds'])
            require(identity not in seen, 'Duplicate station hour')
            seen.add(identity)
            require(int(datetime.fromisoformat(row['utc'].replace('Z', '+00:00')).timestamp()) == row['utc_epoch_seconds'], 'Table UTC text identity')
            require(set(row['values']) == set(FEATURES) and set(row['provenance']) == set(FEATURES), 'Missing/extra feature')
            for name in FEATURES:
                p = row['provenance'][name]
                key = tuple(p['key'])
                prefix, variable = name.split('__')
                require(key[:3] == (variable, 0 if prefix == 'surface' else int(prefix[1:]), identity[1]), 'Feature/pressure/UTC identity')
                value = row['values'][name]
                require(math.isfinite(value), 'Non-finite table value')
                require(key not in required[p['source']] or required[p['source']][key] == value, 'Conflicting repeated source sample')
                required[p['source']][key] = value
                all_keys.add(key)
        require(seen == expected_rows and len(all_keys) == 11240 and len(required) == 161, 'Fixed coverage count/identity')
        require(set(required) == set(scope['source_sha256']), 'Manifest actual-source set differs')
        file_evidence = []
        source_counts = Counter()
        max_value_difference = 0.
        for number, (source, wanted) in enumerate(sorted(required.items()), 1):
            path = protect(local_path(source), scope['source_sha256'][source])
            if path.suffix == '.ascii':
                side = read(protect(path.parent.parent / 'records' / (path.stem + '.json')))
                require(side['source_sha256'] == sha(path) and local_path(side['source']) == path, 'Sidecar/source binding')
                metadata = protect(local_path(side['metadata']), side['metadata_sha256'])
                variables = {k[0] for k in wanted}
                require(len(variables) == 1, 'Unexpected NCAR multi-parameter file')
                actual, evidence = parse_ascii(path.read_text(encoding='utf-8'), metadata.read_text(encoding='utf-8'), next(iter(variables)))
                source_type = 'NCAR'
            else:
                actual, evidence = parse_netcdf(path, wanted)
                source_type = 'CDS'
            diff = compare_required(actual, wanted)
            max_value_difference = max(max_value_difference, diff)
            source_counts[source_type] += len(wanted)
            file_evidence.append({'source': source, 'sha256': sha(path), 'bytes': path.stat().st_size,
                                  'source_type': source_type, 'required_unique_grid_field_hours': len(wanted),
                                  'max_raw_vs_table_difference': diff, **evidence})
            if number % 40 == 0:
                print(f'Independently decoded {number}/{len(required)} actual files', flush=True)
        require(dict(source_counts) == {'CDS': 7680, 'NCAR': 3560}, 'Source-specific unique value counts')
        units = [read(protect(local_path(r['path']), r['sha256'])) for r in scope['selected_units']]
        geometry, points = check_geometry(scope, table['rows'], units)
        for path, digest in guarded.items():
            require(sha(local_path(path)) == digest, f'Input changed during audit: {path}')
        sources_ref = emit('actual_source_evidence.json', file_evidence)
        geometry_ref = emit('independent_intersections.json', points)
        guard_ref = emit('read_only_input_sha256.json', guarded)
        report = {'version': 'gbs_actual_fields_001', 'status': 'PASS', 'started_utc': started,
                  'completed_utc': datetime.now(timezone.utc).isoformat(), 'scope_sha256': SCOPE_SHA, 'scope_fingerprint': SCOPE_FP,
                  'stations': scope['stations'], 'dates': 39, 'station_hours': 1872, 'station_field_hours': 18720,
                  'unique_grid_field_hours': 11240, 'source_unique_grid_field_hours': dict(source_counts),
                  'actual_files': 161, 'required_missing_values': 0, 'required_nonfinite_values': 0,
                  'max_raw_vs_table_difference': max_value_difference, 'geometry': geometry,
                  'source_evidence': sources_ref, 'intersection_evidence': geometry_ref, 'read_only_sha_evidence': guard_ref,
                  'audit_code_sha256': sha(__file__), 'new_downloads': 0, 'new_fit_calls': 0, 'predictions_executed': 0,
                  'cross_platform_bitwise_equivalence': 'NOT_CLAIMED_OR_REQUIRED',
                  'model_input_target_compatibility': 'PENDING', 'evaluation_complete': False,
                  'temperature_semantics': 'paramId 201, stepType=max: maximum since previous post-processing, not daily maximum; selected source is CDS valid_time',
                  'sampling_qualification': 'Four fixed heights + forward LOS nearest published grid; 1hPa actual Top10 z; differs from original bilinear model sampling'}
        emit('actual_field_qa.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'station_hours', 'unique_grid_field_hours', 'actual_files', 'max_raw_vs_table_difference')}), flush=True)
    except Exception as e:
        emit('failure.json', {'status': 'FAIL', 'started_utc': started, 'exception': repr(e), 'traceback': traceback.format_exc(),
                             'inputs_hashed': guarded, 'new_fit_calls': 0, 'predictions_executed': 0})
        raise


if __name__ == '__main__':
    main()
