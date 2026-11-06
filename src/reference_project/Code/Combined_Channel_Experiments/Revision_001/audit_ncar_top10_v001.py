"""Separate offline plan/calendar/sampling audit; not full downloaded-field QA."""
import json
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import subprocess
import sys
import weather_download_top10_ncar_v008 as d
import ncar_top10_transport_v001 as t

def audit(plan):
    old=d.R(d.PREVIOUS/'plan.json')
    preserved=['parents','links','feature_names','natural_rank_top10','fixed_heights','seed_heights_m','pressure_pairs','auxiliary_pairs','pointing_review_required_ids','model_inventory']
    for name in preserved:d.require(plan[name]==old[name],'Scientific scope changed: '+name)
    d.require(len(plan['parents'])==77 and sum(p['expected_hours'] for p in plan['parents'])==45663,'Calendar differs')
    d.require(sum(p['expected_hours'] for p in plan['parents'] if 'pressure_level' in p['request'])==22824,'Pressure calendar differs')
    d.require(len(plan['links'])==218 and plan['auxiliary_pairs']==[],'Geometry/auxiliary scope changed')
    d.require(plan['remote_base']==t.BASE and 'adopted' not in plan,'Unexpected old transport adoption')
    statuses=Counter();source_dates=Counter();unique_demands=set();node_count=0
    for link in plan['links']:
        for level in (5,500,650,850):
            point=d.core.fixed_location(link,level);statuses[point['status']]+=1
    for parent in plan['parents']:
        # Calendar count independently reconstructed with stdlib datetimes.
        req=parent['request'];hours={int(datetime(int(req['year'][0]),int(req['month'][0]),int(day),int(tm[:2]),tzinfo=timezone.utc).timestamp()) for day in req['day'] for tm in req['time']}
        d.require(len(hours)==parent['expected_hours'] and hours==set(d.core.parent_hours(parent)),'Calendar extraction mismatch')
        source_dates[d.legacy.region(parent)]+=len(hours)
        # Audit representative first/last/exact-gap time masks at actual initial/fixed cells.
        chosen=sorted(hours); chosen=sorted({chosen[0],chosen[-1],chosen[len(chosen)//2]})
        demands=set()
        for link in [x for x in plan['links'] if x['region']==d.legacy.region(parent)]:
            if 'pressure_level' in req:
                for level,fields in d.core.LEVEL_VARS.items():
                    if level==1:cell=d.core.seed_cell(link,1)
                    else:
                        point=d.core.fixed_location(link,level)
                        if point['status']!='STABLE_FIXED_HEIGHT_NEAREST':continue
                        cell=point['cell']
                    demands.update((v,level,tm,*cell) for v in fields for tm in chosen)
            else:
                cell=d.core.grid_index(link['latitude_deg_n'],link['longitude_deg_e'])
                demands.update((v,0,tm,*cell) for v in d.core.SURFACE for tm in chosen)
        request_nodes=t.plan_requests(demands);node_count+=len(request_nodes)
        keys=[]
        for n in request_nodes:
            d.require(n['url'].startswith(t.BASE) and '.ascii?' in n['url'],'Non-NCAR request')
            # Independent axis expansion, with Gregorian epoch arithmetic.
            for coord in __import__('itertools').product(*(n['coords'][x] for x in n['dimensions'])):
                m=dict(zip(n['dimensions'],coord))
                dt=datetime(1900,1,1,tzinfo=timezone.utc)+timedelta(hours=m.get('time',m.get('forecast_initial_time',0))+m.get('forecast_hour',0))
                lon=m['longitude'];lon=lon-360 if lon>=180 else lon
                keys.append((n['variable'],n['level_hpa'],int(dt.timestamp()),int(m['latitude']*4),int(lon*4)))
        d.require(Counter(keys)==Counter(demands),'Independent archive-axis expansion mismatch')
        unique_demands.update(demands)
    protected=0
    for path,digest in plan['protected_files'].items():d.require(d.SHA(d.ROOT/path)==digest,'Old asset changed');protected+=1
    return {'status':'PASS','plan_fingerprint':plan['fingerprint'],'code_sha256':d.SHA(d.CODE),'preserved_scope_keys':preserved,
            'fixed_geometry_status_counts':dict(statuses),'calendar_region_hours':dict(source_dates),'representative_exact_requests_audited':node_count,
            'representative_unique_grid_field_hours':len(unique_demands),'old_protected_files_sha_verified':protected,
            'scope':'Offline calendar, frozen geometry and independent archive-axis/packing audit only. Actual full field and prediction QA pending.',
            'new_fit_calls':0,'predictions_executed':0}

def main():
    candidate=d.build_plan() if not d.PLAN.exists() else d.load_plan()
    cp=subprocess.run([sys.executable,'-B',str(d.HERE/'test_ncar_top10_v001.py'),'-v'],capture_output=True,text=True,encoding='utf-8')
    result={'status':'PASS' if cp.returncode==0 else 'FAIL','plan_fingerprint':candidate['fingerprint'],'code_sha256':d.SHA(d.CODE),
            'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr,'scope':'Offline transport/negative/recovery tests only'}
    qa=audit(candidate) if cp.returncode==0 else None
    if not qa or cp.returncode:
        d.io.immutable(d.DEST/'preflight_history'/(str(d.time.time_ns())+'_failed.json'),result)
        print(json.dumps(result));sys.exit(1)
    if d.PLAN.exists():
        d.io.immutable(d.DEST/'preflight_qa.json',result)
        d.io.immutable(d.DEST/'plan_independent_qa.json',qa)
    else:
        # Preserve first audit as candidate evidence; final QA binds the initialized plan.
        d.io.immutable(d.DEST/'candidate_preflight.json',result)
        d.io.immutable(d.DEST/'candidate_plan_audit.json',qa)
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
