"""Read-only source geometry and conservative download envelopes; never fit.

The envelope is NOT a pressure-level sampling location. Actual hourly features
must use the frozen ray/isobaric intersection and bilinear sampling algorithm.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import netCDF4
import weather_download_v002 as io

sys.dont_write_bytecode = True
HERE, ROOT = io.HERE, io.ROOT
DEST = io.EXTERNAL / 'weather_top10_los_v004'
LOS = ROOT.parent / '3_TAP_被拒/scripts/prepare_era5_los_csv.py'
SITE = ROOT.parent / '3_TAP_被拒/config/site_bjtu.json'
LOS_SHA = 'ae1e6d54c1a903bd3e372ff7f0ce99cfd83b704dca4410d4827529bc5ed18b39'
GRID = .25
PADDING = .5
HEIGHT_CEILING_M = 80000.0


def original():
    io.require(io.sha(LOS) == LOS_SHA, 'Frozen original LOS implementation changed')
    spec = importlib.util.spec_from_file_location('frozen_los_readonly', LOS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def east_longitude(value, units):
    io.require(units in ('degrees_east', 'degrees_west'), 'Unknown longitude convention')
    return ((float(value) * (-1 if units == 'degrees_west' else 1) + 180) % 360) - 180


def enu_basis(lat, lon):
    p, l = np.deg2rad([lat, lon])
    return (np.array([-np.sin(l), np.cos(l), 0.]),
            np.array([-np.sin(p)*np.cos(l), -np.sin(p)*np.sin(l), np.cos(p)]),
            np.array([np.cos(p)*np.cos(l), np.cos(p)*np.sin(l), np.sin(p)]))


def ray_from_angles(lat, lon, height, azimuth, elevation, module):
    io.require(0 <= azimuth < 360 and 0 < elevation <= 90, 'Invalid pointing angles')
    east, north, up = enu_basis(lat, lon)
    az, el = np.deg2rad([azimuth, elevation])
    direction = np.cos(el)*np.sin(az)*east + np.cos(el)*np.cos(az)*north + np.sin(el)*up
    return module._geodetic_to_ecef(lat, lon, height), direction / np.linalg.norm(direction)


def ray_angles(direction, lat, lon):
    east, north, up = enu_basis(lat, lon)
    return (float(np.rad2deg(np.arctan2(direction @ east, direction @ north)) % 360),
            float(np.rad2deg(np.arcsin(direction @ up))))


def nominal_ray(lat, lon, height, sat_lon, module):
    config = {'station': {'latitude_deg_n': lat, 'longitude_deg_e': lon,
                         'antenna_phase_center_m_amsl': height},
              'satellite': {'nominal_longitude_deg_e': sat_lon, 'nominal_altitude_km': 35786.0}}
    # Adapter keys exactly match the original nested campaign configuration.
    return module._ray_geometry(config)


def envelope_points(origin, direction, module):
    """Sample a straight outward ray to 80 km geometric height, not standard levels."""
    lo, hi = 0., 500000.
    io.require(module._ecef_to_geodetic((origin + hi*direction)[None, :])[2][0] > HEIGHT_CEILING_M,
               'Ray does not reach configured ceiling within 500 km')
    for _ in range(60):
        mid = (lo + hi)/2
        h = module._ecef_to_geodetic((origin + mid*direction)[None, :])[2][0]
        if h < HEIGHT_CEILING_M: lo = mid
        else: hi = mid
    slants = np.linspace(0, hi, 257)
    lat, lon, height = module._ecef_to_geodetic(origin[None, :] + slants[:, None]*direction)
    io.require(np.isfinite([lat, lon, height]).all() and np.all(np.diff(height) > 0), 'Non-forward geometry')
    return lat, lon, height, hi


def area(points):
    lat, lon = points
    return [math.ceil((float(np.max(lat))+PADDING)/GRID)*GRID,
            math.floor((float(np.min(lon))-PADDING)/GRID)*GRID,
            math.floor((float(np.min(lat))-PADDING)/GRID)*GRID,
            math.ceil((float(np.max(lon))+PADDING)/GRID)*GRID]


def build():
    module = original()
    meta_path = io.EXTERNAL / 'metadata.json'
    meta = io.read(meta_path)
    gbs_root = ROOT / '外部样例数据CEDA/GBS_20p7GHz'
    manifest = gbs_root / 'source_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    dates = set(meta['GBS']['common_dates'])
    io.require(len(dates) == 399, 'Frozen GBS date scope changed')
    selected = [r for r in rows if r['date_utc'] in dates]
    io.require(len(selected) == 1197, 'Expected 399 dates x 3 GBS stations')
    fields = ['latitude_receiver','longitude_receiver','receiver_altitude_amsl','satellite_position','elevation_angle']
    grouped, source_hashes, gbs_audit = {}, {}, []
    for row in selected:
        path = gbs_root / row['relative_path']
        data = path.read_bytes()
        io.require(len(data) == int(row['bytes']) and hashlib.md5(data).hexdigest() == row['md5'],
                   'Original GBS byte/MD5 mismatch: ' + row['relative_path'])
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        with netCDF4.Dataset('geometry.nc', memory=data) as ds:
            vals = {n: float(ds[n][:]) for n in fields}
            units = {n: str(ds[n].units) for n in fields}
        io.require(units['latitude_receiver'] == 'degrees_north' and units['receiver_altitude_amsl'] == 'm'
                   and units['elevation_angle'] == 'degrees', 'GBS coordinate units')
        coords = (vals['latitude_receiver'], east_longitude(vals['longitude_receiver'], units['longitude_receiver']),
                  vals['receiver_altitude_amsl'], east_longitude(vals['satellite_position'], units['satellite_position']),
                  vals['elevation_angle'])
        io.require(np.isfinite(coords).all(), 'Nonfinite GBS geometry')
        grouped.setdefault(row['site'], set()).add(coords)
        gbs_audit.append({'site':row['site'], 'date_utc':row['date_utc'], 'file':str(path.relative_to(ROOT)),
                          'geometry':list(coords), 'units':units})
    links = []
    for site, entries in sorted(grouped.items()):
        io.require(len(entries) == 1, 'Time-varying GBS geometry requires date-resolved adapter')
        lat, lon, height, sat_lon, coarse_elev = next(iter(entries))
        origin, direction = nominal_ray(lat, lon, height, sat_lon, module)
        az, elev = ray_angles(direction, lat, lon)
        links.append({'source':'GBS', 'id':site, 'region':'GBS_Dundee' if site=='dundee' else 'GBS_England',
                      'latitude_deg_n':lat,'longitude_deg_e':lon,'height_amsl_m':height,
                      'satellite_longitude_deg_e':sat_lon,'nominal_geo_altitude_km':35786.,
                      'azimuth_deg_true_north_clockwise':az,'elevation_deg':elev,
                      'source_coarse_elevation_deg':coarse_elev,
                      'geometry_basis':'receiver coordinates + source satellite 23 W + nominal GEO orbit; no ephemeris',
                      'origin_ecef_m':origin.tolist(),'direction_ecef':direction.tolist()})
    op = ROOT / 'Outputs/Datasets/External_OpenSat4Weather_001/extracted/sml_data_2022.nc'
    with netCDF4.Dataset(op) as ds:
        names = ['site_0_lat','site_0_lon','site_0_alt','satellite_azimuth','satellite_elevation']
        values = {n:np.asarray(ds[n][:], float) for n in names}
        attrs = {n:io.attrs(ds[n]) for n in names}
        ids, sats = np.asarray(ds['sml_id'][:]).tolist(), np.asarray(ds['satellite'][:]).tolist()
    io.require(len(ids)==len(set(ids))==215 and all(np.isfinite(v).all() for v in values.values()), 'OpenSat geometry invalid')
    pointing_differences = []
    for k, ident in enumerate(ids):
        lat, lon, height, az, elev = [float(values[n][k]) for n in names]
        origin, direction = ray_from_angles(lat, lon, height, az, elev, module)
        # Independent semantic check: documented satellite labels must agree with
        # north-clockwise azimuth, not mathematical east-origin or reverse direction.
        io.require(sats[k] in ('Astra 19','Astra 23.5','HotBird 13E'), 'Unrecognized OpenSat satellite label')
        sat_lon = {'Astra 19':19.2, 'Astra 23.5':23.5, 'HotBird 13E':13.}[sats[k]]
        _, nominal = nominal_ray(lat, lon, height, sat_lon, module)
        delta = float(np.rad2deg(np.arccos(np.clip(direction @ nominal, -1, 1))))
        pointing_status = 'PASS_NOMINAL_SANITY_CHECK' if delta < .5 else 'PENDING_POINTING_METADATA_REVIEW'
        pointing_differences.append(delta)
        links.append({'source':'OpenSat','id':str(ident),'region':'OpenSat_France','satellite':str(sats[k]),
                      'latitude_deg_n':lat,'longitude_deg_e':lon,'height_amsl_m':height,
                      'azimuth_deg_true_north_clockwise':az,'elevation_deg':elev,
                      'geometry_basis':'published per-link azimuth/elevation; metadata nominal GEO comparison only a sanity check',
                      'nominal_crosscheck_angle_difference_deg':delta,'pointing_check':pointing_status,
                      'nominal_crosscheck_direction_ecef':nominal.tolist(),
                      'origin_ecef_m':origin.tolist(),'direction_ecef':direction.tolist()})
    envelopes = {}
    for region in ('GBS_England','GBS_Dundee','OpenSat_France'):
        subset = [r for r in links if r['region']==region]
        all_lat, all_lon = [], []
        for row in subset:
            lat, lon, height, slant = envelope_points(np.array(row['origin_ecef_m']), np.array(row['direction_ecef']), module)
            row['ray_to_80km'] = {'end_latitude_deg_n':float(lat[-1]), 'end_longitude_deg_e':float(lon[-1]),
                                 'end_height_m':float(height[-1]), 'slant_m':slant,
                                 'lat_min_max':[float(lat.min()),float(lat.max())],
                                 'lon_min_max':[float(lon.min()),float(lon.max())]}
            all_lat.extend(lat.tolist()); all_lon.extend(lon.tolist())
            if row.get('pointing_check') == 'PENDING_POINTING_METADATA_REVIEW':
                lat2, lon2, _, _ = envelope_points(np.array(row['origin_ecef_m']), np.array(row['nominal_crosscheck_direction_ecef']), module)
                all_lat.extend(lat2.tolist()); all_lon.extend(lon2.tolist())
        pressure = area((all_lat, all_lon))
        surface = area(([r['latitude_deg_n'] for r in subset], [r['longitude_deg_e'] for r in subset]))
        envelopes[region] = {'pressure_area_NWSE':pressure,'surface_area_NWSE':surface,'link_count':len(subset)}
    for p in [meta_path, manifest, op, op.parent/'readme.txt', Path(__file__), LOS, SITE]:
        source_hashes[str(p)] = io.sha(p)
    result = {'status':'PASS_DOWNLOAD_ENVELOPE_WITH_POINTING_GAP','created_utc':io.now(),'links':links,'regions':envelopes,
              'GBS_selected_file_count':1197,'GBS_geometry_source_records':gbs_audit,
              'OpenSat_pointing_nominal_max_difference_deg':max(pointing_differences),
              'pointing_review_required_ids':[r['id'] for r in links if r.get('pointing_check') == 'PENDING_POINTING_METADATA_REVIEW'],
              'pointing_gap_policy':'Do not relax 0.5 degree check or change published pointing. Envelope includes both published and nominal rays for flagged links; their prediction remains pending geometry review.',
              'OpenSat_geometry_variable_attributes':attrs,'grid_deg':GRID,'padding_deg':PADDING,
              'forward_geometric_height_ceiling_m':HEIGHT_CEILING_M,'samples_per_ray':257,
              'height_note':'Use AMSL antenna height as in frozen original LOS implementation; no new geoid/ephemeris estimation. Nominal geometry, not surveyed satellite orbit.',
              'actual_feature_definition':'hourly dynamic forward ray / ERA5 isobaric geopotential surface intersection, residual <=5m; bilinear sample at intersection; invalid or below-antenna levels retain NaN and validity flag',
              'scope_note':'80km + 0.5deg padding is a conservative acquisition envelope, not a replacement for hourly intersections. Reject out-of-envelope solutions; no silent extrapolation or automatic scope expansion.',
              'full_field_QA_complete':False,'external_prediction_executed':False,'new_fit_calls':0,
              'source_sha256':source_hashes}
    result['fingerprint'] = io.fingerprint(result)
    io.DEST = DEST
    io.immutable(DEST/'geometry_manifest.json', result)
    print(json.dumps({k:result[k] for k in ('status','regions','GBS_selected_file_count','OpenSat_pointing_nominal_max_difference_deg','fingerprint')},indent=2))


if __name__ == '__main__': build()
