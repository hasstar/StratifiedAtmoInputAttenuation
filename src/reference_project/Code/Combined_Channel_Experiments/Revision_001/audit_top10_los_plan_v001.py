"""Independent plan/geometry readback: no downloader contract helpers, no network."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import weather_download_v002 as io
ROOT,HERE=io.ROOT,io.HERE
DEST=io.EXTERNAL/'weather_top10_los_v004'


def inverse_independent(xyz):
    # Bowring closed-form geodetic inverse, independent of the source 7-iteration
    # implementation. Used only to check acquisition envelopes, not extract data.
    a=6378137.;f=1/298.257223563;b=a*(1-f);e2=f*(2-f);ep2=(a*a-b*b)/(b*b)
    x,y,z=np.asarray(xyz).T;p=np.hypot(x,y);theta=np.arctan2(z*a,p*b)
    lat=np.arctan2(z+ep2*b*np.sin(theta)**3,p-e2*a*np.cos(theta)**3)
    lon=np.arctan2(y,x);normal=a/np.sqrt(1-e2*np.sin(lat)**2)
    return np.rad2deg(lat),np.rad2deg(lon),p/np.cos(lat)-normal


def main():
    plan=io.read(DEST/'plan.json');original=io.read(io.EXTERNAL/'weather_plan.json');g=io.read(DEST/'geometry_manifest.json')
    inventory=io.read(DEST/'model_inventory.json');old={n['id']:n for n in original['jobs']}
    io.require(plan['fingerprint']==io.fingerprint({k:v for k,v in plan.items() if k!='fingerprint'}),'Digest mismatch')
    io.require(set(inventory['feature_names'])==set(plan['feature_names']) and len(plan['feature_names'])==10,'Model order coverage')
    pressure_pairs={'p1__geopotential','p1__u_component_of_wind','p5__ozone_mass_mixing_ratio','p5__u_component_of_wind',
                    'p500__ozone_mass_mixing_ratio','p650__u_component_of_wind','p850__relative_humidity','p850__ozone_mass_mixing_ratio'}
    io.require(pressure_pairs=={n for n in plan['feature_names'] if n.startswith('p')},'Pressure predictor pairs differ')
    counts=Counter(); hour_total=0
    for parent in plan['parents']:
        source=old[parent['source_parent_id']];r=parent['request'];s=source['request']
        for key in set(s)-{'area','variable','pressure_level'}:io.require(r[key]==s[key],'Calendar or product field changed: '+key)
        north,west,south,east=r['area'];io.require(north>south and east>west,'Degenerate spatial axes')
        io.require(all(abs(v*4-round(v*4))<1e-12 for v in r['area']),'Grid alignment')
        region=next(k for k in g['regions'] if parent['id'].startswith(k+'_'))
        pressure='pressure_level' in r
        io.require(r['area']==g['regions'][region]['pressure_area_NWSE' if pressure else 'surface_area_NWSE'],'Geometry/request area differs')
        io.require(r['variable']==(['geopotential','u_component_of_wind','ozone_mass_mixing_ratio','relative_humidity'] if pressure else
                   ['convective_available_potential_energy','maximum_2m_temperature_since_previous_post_processing']),'Unexpected variable')
        if pressure:io.require(r['pressure_level']==['1','5','500','650','850'],'Geopotential support levels missing')
        nodes=[n for n in plan['jobs'] if n['role']=='los_top10' and n['parent_id']==parent['id']]
        actual=Counter()
        for node in nodes:
            io.require({k:v for k,v in node['request'].items() if k!='day'}=={k:v for k,v in r.items() if k!='day'},'Child non-date field differs')
            actual.update(node['request']['day'])
            io.require(node['expected_hours']==len(node['request']['day'])*len(r['time']),'Wrong child hours')
        io.require(actual==Counter(s['day']),'Missing/duplicate original dates')
        hour_total+=sum(n['expected_hours'] for n in nodes);counts[region]+=1
    io.require(len(plan['parents'])==77 and hour_total==sum(p['expected_hours'] for p in original['jobs']),'Parent/hour union changed')
    for a in plan['adopted']:
        source=ROOT/a['source_state'];saved=io.read(source);node=a['job'];migrated=io.read(DEST/'jobs'/f"{node['id']}.json")
        io.require(io.sha(source)==a['source_state_sha256'],'Legacy state changed')
        io.require(node['request']==saved['job']['request'] and node['collection_id']==saved['job']['collection_id'],'Legacy request changed')
        io.require(migrated['request_id']==saved['request_id']==a['request_id'] and migrated['counted_as_LOS_coverage'] is False,'Legacy ID migration/counting')
    dense=0;minimum_margin=100.;origin_max_error=0.
    for link in g['links']:
        lat0,lon0,h0=inverse_independent([link['origin_ecef_m']])
        origin_max_error=max(origin_max_error,abs(float(h0[0])-link['height_amsl_m']))
        io.require(abs(lat0[0]-link['latitude_deg_n'])<1e-7 and abs(lon0[0]-link['longitude_deg_e'])<1e-7 and origin_max_error<.02,'Origin coordinate mismatch')
        direction=np.array(link['direction_ecef']);io.require(abs(np.linalg.norm(direction)-1)<1e-12,'Nonunit ray')
        slants=np.linspace(0,link['ray_to_80km']['slant_m'],4097)
        rays=[direction]
        if link.get('pointing_check')=='PENDING_POINTING_METADATA_REVIEW':rays.append(np.array(link['nominal_crosscheck_direction_ecef']))
        for ray in rays:
            lat,lon,height=inverse_independent(np.array(link['origin_ecef_m'])[None,:]+slants[:,None]*ray)
            north,west,south,east=g['regions'][link['region']]['pressure_area_NWSE']
            margin=min(float((north-lat).min()),float((lat-south).min()),float((lon-west).min()),float((east-lon).min()))
            io.require(margin>.24 and np.all(np.diff(height)>0),'Ray envelope/bilinear support insufficient')
            minimum_margin=min(minimum_margin,margin);dense+=len(slants)
        # Original vertical-column support must demonstrably be rejected.
        if link['id']=='chilbolton':
            io.require(abs(link['ray_to_80km']['end_longitude_deg_e']-link['longitude_deg_e'])>.2,'Accidental vertical ray')
    io.require(g['pointing_review_required_ids']==['FRA-00609'],'Pointing gap lost')
    geometry_sources=0
    for name,digest in g['source_sha256'].items():
        io.require(io.sha(ROOT/name)==digest,'Geometry source changed: '+name);geometry_sources+=1
    for name,digest in plan['protected_files'].items():io.require(io.sha(ROOT/name)==digest,'Old artifact changed: '+name)
    report={'status':'PASS','checked_utc':io.now(),'plan_fingerprint':plan['fingerprint'],
            'code_sha256':io.sha(HERE/'weather_download_top10_los_v004.py'),'auditor_sha256':io.sha(Path(__file__)),
            'original_calendar_parents':77,'request_hours':hour_total,'regions':dict(counts),
            'independent_dense_ray_points':dense,'minimum_grid_margin_deg':minimum_margin,
            'independent_origin_max_height_error_m':origin_max_error,'geometry_sources_sha_verified':geometry_sources,
            'protected_files_sha_verified':len(plan['protected_files']),'legacy_ids_retained':len(plan['adopted']),
            'old_grid_coverage_counted':0,'pointing_review_required_ids':g['pointing_review_required_ids'],
            'new_fit_calls':0,'network_requests':0,'full_field_QA_complete':False,
            'scope':'Plan, migration, source geometry and acquisition-envelope QA only; not hourly field/intersection or external-model QA.'}
    io.DEST=DEST;io.immutable(DEST/'plan_independent_qa.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
