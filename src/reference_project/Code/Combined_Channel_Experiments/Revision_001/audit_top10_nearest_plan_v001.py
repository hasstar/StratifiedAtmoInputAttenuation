"""Independent plan/initial ray/packing audit; not downloaded-field QA.

Recomputes geodetic origins and seed intersections without production geometry.
Checks exact sample expansion of packing using a separate implementation.
"""
from __future__ import annotations
import math
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import weather_download_top10_nearest_v005 as w
import nearest_los_core_v001 as c


def xyz(lat,lon,h):
    a=6378137.;e2=6.6943799901413165e-3
    lat,lon=math.radians(lat),math.radians(lon);N=a/math.sqrt(1-e2*math.sin(lat)**2)
    return np.array([(N+h)*math.cos(lat)*math.cos(lon),(N+h)*math.cos(lat)*math.sin(lon),
                     (N*(1-e2)+h)*math.sin(lat)])


def lla(p):
    a=6378137.;e2=6.6943799901413165e-3; x,y,z=p
    rho=math.hypot(x,y);lat=math.atan2(z,rho*(1-e2))
    for _ in range(15):
        N=a/math.sqrt(1-e2*math.sin(lat)**2)
        lat=math.atan2(z+e2*N*math.sin(lat),rho)
    N=a/math.sqrt(1-e2*math.sin(lat)**2)
    return math.degrees(lat),math.degrees(math.atan2(y,x)),rho/math.cos(lat)-N


def ray_cell(link,height):
    p=np.array(link['origin_ecef_m']);v=np.array(link['direction_ecef'])
    lo,hi=0.,500000.
    for _ in range(55):
        s=(lo+hi)/2
        if lla(p+s*v)[2]<height:lo=s
        else:hi=s
    lat,lon,h=lla(p+(lo+hi)/2*v)
    return (int(math.floor(lat*4+.5)),int(math.floor(lon*4+.5))),abs(h-height)


def audit():
    plan=w.load_plan();original=w.R(w.EXTERNAL/'weather_plan.json')
    assert plan['parents']==original['jobs'] and len(plan['parents'])==77
    assert sum(p['expected_hours'] for p in plan['parents'])==45663
    assert plan['max_in_flight_requests']==2 and plan['new_fit_calls']==plan['predictions_executed']==0
    assert plan['pointing_review_required_ids']==['FRA-00609']
    assert len(plan['links'])==218 and len({x['id'] for x in plan['links']})==218
    expected={1:{'geopotential','u_component_of_wind'},5:{'ozone_mass_mixing_ratio','u_component_of_wind'},
              500:{'ozone_mass_mixing_ratio'},650:{'u_component_of_wind'},850:{'ozone_mass_mixing_ratio','relative_humidity'}}
    assert {int(k):set(v) for k,v in plan['pressure_pairs'].items()}==expected
    assert plan['auxiliary_pairs']==[['geopotential',v] for v in (5,500,650,850)]
    for item in plan['adopted']:
        old=w.R(w.ROOT/item['source_state'])
        assert old['request_id']==item['request_id'] and old['job']['request']==item['job']['request']
        assert old['job']['collection_id']==item['job']['collection_id']
    max_res=0.;initial_cells=Counter();initial_samples=Counter();initial_requests=Counter()
    for link in plan['links']:
        assert np.linalg.norm(xyz(link['latitude_deg_n'],link['longitude_deg_e'],link['height_amsl_m'])-
                              np.array(link['origin_ecef_m']))<.001
        for level,h in c.SEED_HEIGHT.items():
            cell,res=ray_cell(link,h);max_res=max(max_res,res)
            assert cell==c.seed_cell(link,level)
            initial_cells[(link['region'],level,cell)]+=1
    for parent in plan['parents']:
        links=[x for x in plan['links'] if x['region']==w.region(parent)]
        hours=c.parent_hours(parent);pressure='pressure_level' in parent['request'];demand=set()
        for link in links:
            if pressure:
                for level in expected:
                    cell,_=ray_cell(link,c.SEED_HEIGHT[level])
                    demand.update(('geopotential',level,t,*cell) for t in hours)
            else:
                cell=(int(math.floor(link['latitude_deg_n']*4+.5)),int(math.floor(link['longitude_deg_e']*4+.5)))
                demand.update((v,0,t,*cell) for t in hours for v in c.SURFACE)
        requests=c.demands_to_requests(parent,demand); expanded=Counter()
        for r in requests:
            assert 'grid' not in r and len(r.get('pressure_level',['0']))==1
            n,west,s,east=[int(round(v*4)) for v in r['area']]
            for day in r['day']:
                for t in r['time']:
                    epoch=int(datetime(int(r['year'][0]),int(r['month'][0]),int(day),int(t[:2]),tzinfo=timezone.utc).timestamp())
                    for i in range(s,n+1):
                        for j in range(west,east+1):
                            for v in r['variable']:expanded[(v,int(r.get('pressure_level',['0'])[0]),epoch,i,j)]+=1
        assert expanded==Counter(demand)
        label='initial_z_seed' if pressure else 'surface_exact'
        initial_requests[label]+=len(requests);initial_samples[label]+=len(demand)
    geom=w.R(w.OLD/'geometry_manifest.json')
    for path,digest in geom['source_sha256'].items():assert w.SHA(w.ROOT/path)==digest
    result={'status':'PASS','checked_utc':w.io.now(),'plan_fingerprint':plan['fingerprint'],
            'code_sha256':w.SHA(w.CODE),'auditor_sha256':w.SHA(__file__),
            'original_calendar_parents':77,'request_hours':45663,'pressure_request_hours':22824,
            'links':218,'independent_seed_intersections':218*5,'max_height_residual_m':max_res,
            'unique_region_level_seed_cells':len(initial_cells), 'initial_request_estimate_before_cache':dict(initial_requests),
            'initial_grid_field_hours_before_cache':dict(initial_samples),'pressure_pairs_before':20,'pressure_pairs_after':12,
            'geometry_source_hashes_checked':len(geom['source_sha256']),'protected_hashes_checked':len(plan['protected_files']),
            'inherited_ids_unchanged':len(plan['adopted']), 'network_requests':0,'new_fit_calls':0,
            'full_field_QA_complete':False,
            'scope':'Static scope, original geometry, nearest seed locations and exact packing. Actual hourly intersections/fields still need independent QA. Initial request count is NOT total; subsequent z corrections and direct target fields add requests.'}
    w.io.immutable(w.DEST/'plan_independent_qa.json',result)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':audit()
