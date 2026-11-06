"""Independent calendar, fixed-atmosphere LOS and exact packing audit.

No remote operations, model load or training. Rounded published atmosphere table
bases and pressure bisection independently check production analytic inversion.
"""
from __future__ import annotations
import math
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import weather_download_top10_fixedheight_v007 as w
import fixedheight_los_core_v001 as c


def xyz(lat,lon,h):
    a=6378137.;e2=6.6943799901413165e-3
    lat,lon=math.radians(lat),math.radians(lon);N=a/math.sqrt(1-e2*math.sin(lat)**2)
    return np.array([(N+h)*math.cos(lat)*math.cos(lon),(N+h)*math.cos(lat)*math.sin(lon),
                     (N*(1-e2)+h)*math.sin(lat)])


def lla(p):
    a=6378137.;e2=6.6943799901413165e-3;x,y,z=p
    rho=math.hypot(x,y);lat=math.atan2(z,rho*(1-e2))
    for _ in range(15):
        N=a/math.sqrt(1-e2*math.sin(lat)**2)
        lat=math.atan2(z+e2*N*math.sin(lat),rho)
    N=a/math.sqrt(1-e2*math.sin(lat)**2)
    return math.degrees(lat),math.degrees(math.atan2(y,x)),rho/math.cos(lat)-N


def ray_cell(link,height):
    p=np.array(link['origin_ecef_m']);v=np.array(link['direction_ecef'])
    assert height>link['height_amsl_m'] and abs(np.linalg.norm(v)-1)<1e-8
    lo,hi=0.,500000.
    for _ in range(55):
        distance=(lo+hi)/2
        if lla(p+distance*v)[2]<height:lo=distance
        else:hi=distance
    lat,lon,h=lla(p+(lo+hi)/2*v)
    assert (lo+hi)/2>0
    return (int(math.floor(lat*4+.5)),int(math.floor(lon*4+.5))),abs(h-height)


def table_height(p_hpa):
    # NASA/TM-2005-213659 Appendix A rounded table, independent base pressures.
    h0,t0,p0,lapse,upper=(32000.,228.65,868.02,.0028,47000.) if p_hpa==5 else (0.,288.15,101325.,-.0065,11000.)
    lo,hi=h0,upper
    for _ in range(60):
        H=(lo+hi)/2
        pressure=p0*math.exp(-9.80665*28.9644/(8314.32*lapse)*math.log((t0+lapse*(H-h0))/t0))
        if pressure>p_hpa*100:lo=H
        else:hi=H
    H=(lo+hi)/2
    return 6356660.*H/(6356660.-H)


def audit(plan=None,save=True):
    plan=w.load_plan() if plan is None else plan
    original=w.R(w.EXTERNAL/'weather_plan.json');geom=w.R(w.OLD/'geometry_manifest.json')
    assert plan['parents']==original['jobs'] and len(plan['parents'])==77
    assert sum(p['expected_hours'] for p in plan['parents'])==45663
    assert plan['max_in_flight_requests']==2 and plan['new_fit_calls']==plan['predictions_executed']==0
    assert plan['pointing_review_required_ids']==['FRA-00609']
    assert len(plan['links'])==218 and len({x['id'] for x in plan['links']})==218
    expected={1:{'geopotential','u_component_of_wind'},5:{'ozone_mass_mixing_ratio','u_component_of_wind'},
              500:{'ozone_mass_mixing_ratio'},650:{'u_component_of_wind'},850:{'ozone_mass_mixing_ratio','relative_humidity'}}
    assert {int(k):set(v) for k,v in plan['pressure_pairs'].items()}==expected
    assert plan['auxiliary_pairs']==[] and plan['original_calendar_requests_are_not_submission_templates']
    height_errors={}
    for p in (5,500,650,850):
        stored=plan['fixed_heights'].get(str(p),plan['fixed_heights'].get(p))
        height_errors[p]=abs(table_height(p)-stored['geometric_height_m'])
        assert height_errors[p]<.5
    for item in plan['adopted']:
        old=w.R(w.ROOT/item['source_state'])
        assert old['request_id']==item['request_id'] and old['job']['request']==item['job']['request']
        assert old['job']['collection_id']==item['job']['collection_id']
        assert item['job']['download_policy']=='poll_only_retired_auxiliary'
    cells={};max_res=0.;invalid=[];valid_fixed=0
    for link in plan['links']:
        assert np.linalg.norm(xyz(link['latitude_deg_n'],link['longitude_deg_e'],link['height_amsl_m'])-
                              np.array(link['origin_ecef_m']))<.001
        cell,res=ray_cell(link,max(48000.,float(link['height_amsl_m'])+1.))
        assert cell==c.seed_cell(link,1);cells[(link['id'],1)]=cell;max_res=max(max_res,res)
        for p in (5,500,650,850):
            record=c.fixed_location(link,p)
            stored=plan['fixed_heights'].get(str(p),plan['fixed_heights'].get(p))
            height=stored['geometric_height_m']
            if height<=link['height_amsl_m']:
                assert record['status']=='INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA'
                invalid.append({'link_id':link['id'],'level_hpa':p,'antenna_height_m':link['height_amsl_m'],'fixed_height_m':height})
                continue
            cell,res=ray_cell(link,height);max_res=max(max_res,res)
            assert list(cell)==record['cell'] and abs(record['height_residual_m'])<.001
            assert record['fixed_height_approximation'] and not record['actual_hourly_pressure_height_verified']
            n,west,s,east=geom['regions'][link['region']]['pressure_area_NWSE']
            assert s<=cell[0]/4<=n and west<=cell[1]/4<=east
            cells[(link['id'],p)]=cell;valid_fixed+=1
    assert invalid==[{'link_id':'FRA-00289','level_hpa':850,'antenna_height_m':1622.0,
                     'fixed_height_m':c.FIXED_HEIGHTS[850]['geometric_height_m']}]
    sample_counts=Counter();request_counts=Counter()
    for parent in plan['parents']:
        links=[x for x in plan['links'] if x['region']==w.region(parent)]
        # Independent reconstruction of original UTC masks, not a continuous fill.
        r=parent['request']
        hours=[int(datetime(int(r['year'][0]),int(r['month'][0]),int(day),int(t[:2]),tzinfo=timezone.utc).timestamp())
               for day in r['day'] for t in r['time']]
        assert hours==c.parent_hours(parent) and len(hours)==parent['expected_hours']
        pressure='pressure_level' in r;demand=set()
        for link in links:
            if pressure:
                # Only initial 1hPa z plus fixed-layer direct targets. 1hPa u and
                # later z cell corrections are deliberately not estimated here.
                demand.update(('geopotential',1,t,*cells[(link['id'],1)]) for t in hours)
                for p in (5,500,650,850):
                    if (link['id'],p) in cells:
                        demand.update((v,p,t,*cells[(link['id'],p)]) for t in hours for v in expected[p])
            else:
                cell=(int(math.floor(link['latitude_deg_n']*4+.5)),int(math.floor(link['longitude_deg_e']*4+.5)))
                demand.update((v,0,t,*cell) for t in hours for v in c.SURFACE)
        requests=c.demands_to_requests(parent,demand);expanded=Counter()
        for request in requests:
            assert 'grid' not in request and len(request.get('pressure_level',['0']))==1
            level=int(request.get('pressure_level',['0'])[0])
            assert all(v in (expected[level] if level else c.SURFACE) for v in request['variable'])
            n,west,s,east=[int(round(v*4)) for v in request['area']]
            for day in request['day']:
                for t in request['time']:
                    epoch=int(datetime(int(request['year'][0]),int(request['month'][0]),int(day),int(t[:2]),tzinfo=timezone.utc).timestamp())
                    for i in range(s,n+1):
                        for j in range(west,east+1):
                            for v in request['variable']:expanded[(v,level,epoch,i,j)]+=1
        assert expanded==Counter(demand)
        label='initial_1hPa_z_plus_fixed_targets' if pressure else 'surface_exact'
        request_counts[label]+=len(requests);sample_counts[label]+=len(demand)
    for path,digest in geom['source_sha256'].items():assert w.SHA(w.ROOT/path)==digest
    for path,digest in plan['protected_files'].items():assert w.SHA(w.ROOT/path)==digest
    result={'status':'PASS','checked_utc':w.io.now(),'plan_fingerprint':plan['fingerprint'],
            'code_sha256':w.SHA(w.CODE),'auditor_sha256':w.SHA(__file__),
            'original_calendar_parents':77,'request_hours':45663,'pressure_request_hours':22824,
            'links':218,'independent_1hPa_seed_intersections':218,'independent_valid_fixed_intersections':valid_fixed,
            'invalid_fixed_intersections_retained':invalid,'max_height_residual_m':max_res,
            'standard_height_table_difference_m':height_errors,'initial_request_estimate_before_cache':dict(request_counts),
            'initial_grid_field_hours_before_cache':dict(sample_counts),'pressure_pairs_before':12,'pressure_pairs_after':8,
            'surface_pairs':2,'new_auxiliary_pairs':0,
            'geometry_source_hashes_checked':len(geom['source_sha256']),'protected_hashes_checked':len(plan['protected_files']),
            'inherited_poll_only_ids_unchanged':len(plan['adopted']),'network_requests':0,'new_fit_calls':0,
            'full_field_QA_complete':False,
            'scope':'Static scope, fixed atmosphere heights, forward LOS, 1hPa seeds, exact packing. Four fixed heights are NOT observed hourly isobars. Estimates before cache exclude subsequent 1hPa corrections/u and are NOT final download totals.'}
    if save:w.io.immutable(w.DEST/'plan_independent_qa.json',result)
    print(__import__('json').dumps(result,indent=2))
    return result


if __name__=='__main__':audit()
