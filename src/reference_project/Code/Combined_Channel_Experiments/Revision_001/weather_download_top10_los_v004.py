"""RFE Top10 acquisition with LOS support. Frozen v002 transport, new namespace.

No training or feature extraction here. Legacy Top1 IDs are drained unchanged,
never counted as complete LOS support. Only explicit new-request size failures
allow date-only bisection. All old artifacts are immutable inputs.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode = True
import weather_download_v002 as transport

HERE, ROOT, EXTERNAL = transport.HERE, transport.ROOT, transport.EXTERNAL
OLD = EXTERNAL / 'weather_top1_v003'
DEST = EXTERNAL / 'weather_top10_los_v004'
PLAN = DEST / 'plan.json'
CODE = Path(__file__)
TEST = HERE / 'test_weather_download_top10_los_v004.py'
AUDIT = HERE / 'audit_top10_los_plan_v001.py'
READ, SHA, FP, REQUIRE = transport.read, transport.sha, transport.fingerprint, transport.require
transport.DEST = DEST
LEVELS = ['1','5','500','650','850']
PVARS = ['geopotential','u_component_of_wind','ozone_mass_mixing_ratio','relative_humidity']
SVARS = ['convective_available_potential_energy','maximum_2m_temperature_since_previous_post_processing']
TOP10 = ['p1__geopotential','p1__u_component_of_wind','p5__ozone_mass_mixing_ratio',
         'p650__u_component_of_wind','surface__convective_available_potential_energy',
         'surface__maximum_2m_temperature_since_previous_post_processing','p850__relative_humidity',
         'p850__ozone_mass_mixing_ratio','p5__u_component_of_wind','p500__ozone_mass_mixing_ratio']
_split_original = transport.split_rejected


def split_guarded(node, saved, evidence):
    if node.get('role') == 'legacy_top1':
        transport.save(node, saved, status='BLOCKED_LEGACY_SIZE_DIAGNOSIS', failure=evidence)
    else:
        _split_original(node, saved, evidence)


transport.split_rejected = split_guarded


def region_for(parent):
    matches = [r for r in ('GBS_England','GBS_Dundee','OpenSat_France') if parent['id'].startswith(r+'_')]
    REQUIRE(len(matches)==1, 'Unrecognized source region')
    return matches[0]


def new_request(original, geometry):
    request = copy.deepcopy(original['request'])
    pressure = 'pressure_level' in request
    request['variable'] = PVARS.copy() if pressure else SVARS.copy()
    if pressure: request['pressure_level'] = LEVELS.copy()
    request['area'] = geometry['regions'][region_for(original)]['pressure_area_NWSE' if pressure else 'surface_area_NWSE']
    return request


def coverage_contract(plan):
    REQUIRE(plan['feature_names']==TOP10 and plan['max_in_flight_requests']==2, 'Top10 or slot contract changed')
    REQUIRE(len(plan['adopted'])<=2 and len({a['request_id'] for a in plan['adopted']})==len(plan['adopted']), 'Legacy remote ID duplicate/excess')
    REQUIRE(len({n['id'] for n in plan['jobs']})==len(plan['jobs']), 'Duplicate runtime ID')
    original={p['id']:p for p in plan['original_parents']}
    parents={p['id']:p for p in plan['parents']}
    REQUIRE(len(parents)==len(original)==77, 'Original calendar parent coverage changed')
    adopted={a['job']['id']:a for a in plan['adopted']}
    coverage={pid:Counter() for pid in parents}
    for parent in parents.values():
        source=original[parent['source_parent_id']]
        REQUIRE(parent['request']==new_request(source,plan['geometry']), 'Incorrect LOS area / Top10 variables / original times')
        REQUIRE(parent['expected_hours']==source['expected_hours'] and parent['collection_id']==source['collection_id'], 'Source calendar/collection changed')
        north,west,south,east=parent['request']['area']
        REQUIRE(north-south>=.5 and east-west>=.5, 'Insufficient bilinear support axes')
    for node in plan['jobs']:
        if node['id'] in adopted:
            item=adopted[node['id']]
            REQUIRE(node==item['job'] and node['role']=='legacy_top1', 'Adopted node altered')
            REQUIRE(node['request']==item['original_job']['request'] and node['collection_id']==item['original_job']['collection_id'], 'Inherited remote request altered')
            continue
        REQUIRE(node['role']=='los_top10' and node['parent_id'] in parents, 'Invalid new job parent/role')
        parent=parents[node['parent_id']]
        expected=copy.deepcopy(parent['request']); expected['day']=node['request']['day']
        REQUIRE(node['request']==expected and node['collection_id']==parent['collection_id'], 'New node differs beyond dates')
        REQUIRE(node['expected_hours']==len(node['request']['day'])*len(node['request']['time']), 'New node hour count wrong')
        REQUIRE(len(node['request']['day'])<= (7 if 'pressure_level' in expected else 31), 'Initial validated batching exceeded')
        coverage[parent['id']].update(node['request']['day'])
    for pid,parent in parents.items():
        REQUIRE(coverage[pid]==Counter(parent['request']['day']), 'Missing/duplicated original dates: '+pid)
    return {'parents':len(parents),'pressure_parent_months':sum('pressure_level' in p['request'] for p in parents.values()),
            'expected_request_hours':sum(p['expected_hours'] for p in parents.values()),
            'pressure_request_hours':sum(p['expected_hours'] for p in parents.values() if 'pressure_level' in p['request']),
            'initial_new_batches':len(plan['jobs'])-len(adopted),'adopted_remote_ids':len(adopted),
            'model_features':10,'auxiliary_geopotential_levels_hpa':[5,500,650,850],
            'surface_diagnostic_requests':0,'old_grid_files_counted_as_LOS_coverage':0,
            'new_fit_calls':0,'full_field_QA_complete':False}


def build_plan():
    REQUIRE(not transport.lock_held(), 'Old writer holds shared OS lock')
    REQUIRE(not list((OLD/'payloads').glob('*.part')), 'Old partial requires diagnosis before migration')
    geometry=READ(DEST/'geometry_manifest.json'); inventory=READ(DEST/'model_inventory.json')
    REQUIRE(geometry['fingerprint']==FP({k:v for k,v in geometry.items() if k!='fingerprint'}), 'Geometry digest changed')
    REQUIRE(inventory['natural_rank_top10']==TOP10 and inventory['new_fit_calls']==0, 'Existing k10 inventory gate')
    previous=READ(OLD/'plan.json'); original=READ(EXTERNAL/'weather_plan.json')
    REQUIRE(SHA(OLD/'plan.json')=='265fc2dbddd0f7b8215aad9f4d99518fd1fc1dc6727000ee14536172aebf06d3','Old plan changed')
    REQUIRE(previous['fingerprint']=='b80606aa0b5619ce8a447f4dabcc6cfabe6ca5aeb17b95c0a40451545fe70500','Wrong previous plan')
    adopted=[]
    for path in sorted((OLD/'jobs').glob('*.json')):
        saved=READ(path); node=saved['job']
        REQUIRE(saved['request_fingerprint']==FP(node), 'Old saved request fingerprint differs')
        if saved['status']=='VERIFIED':
            REQUIRE(SHA(OLD/node['target'])==saved['sha256'], 'Old verified payload changed')
            continue
        REQUIRE(saved['status'] in ('WAITING_REMOTE','SUBMITTED','RETRY_SAME_ID') and saved.get('request_id'),
                'Unresolved old submission/access state: reconcile first')
        new=copy.deepcopy(node)
        new.update(id='legacy__'+node['id'],parent_id='legacy__'+node['parent_id'],role='legacy_top1',
                   target='payloads/legacy__'+node['id']+'.download')
        adopted.append({'job':new,'original_job':node,'request_id':saved['request_id'],
                        'source_state':str(path.relative_to(ROOT)),'source_state_sha256':SHA(path)})
    parents=[]; batches={}
    for source in original['jobs']:
        identity=source['id']+'__los004'
        parent={'id':identity,'parent_id':identity,'source_parent_id':source['id'],'role':'los_top10',
                'collection_id':source['collection_id'],'request':new_request(source,geometry),
                'expected_hours':source['expected_hours'],'target':'payloads/'+identity+'.download'}
        parents.append(parent)
        days=parent['request']['day']; cap=7 if 'pressure_level' in parent['request'] else 31
        batches[identity]=[transport.child(parent,days[k:k+cap],'__b'+str(k//cap+1).zfill(3)) for k in range(0,len(days),cap)]
    # Fixed calendar order: first pressure batch then matching surface month,
    # enabling an early same-date Top10 trial, never selected by prediction score.
    ordered=[]; emitted=set()
    for parent in parents:
        if parent['id'] in emitted: continue
        chunks=batches[parent['id']]; ordered.append(chunks[0]); emitted.add(parent['id'])
        if 'pressure_level' in parent['request']:
            surface=parent['id'].replace('_pressure__los004','_surface__los004')
            if surface in batches and surface not in emitted:
                ordered.extend(batches[surface]); emitted.add(surface)
        ordered.extend(chunks[1:])
    protected={str(p.relative_to(ROOT)):SHA(p) for p in EXTERNAL.rglob('*')
               if p.is_file() and not p.is_relative_to(DEST) and p!=transport.LOCK}
    protected.update(inventory['source_sha256'])
    # Full 1197-file geometry inventory is retained; control hashes gate scheduler.
    for p in [HERE/'weather_download.py', HERE/'weather_download_v002.py', HERE/'weather_download_top1_v003.py',
              HERE/'top10_los_geometry_v003.py', HERE/'inventory_top10_model_v002.py',
              HERE/'test_weather_download_v002.py',DEST/'geometry_manifest.json',DEST/'model_inventory.json',DEST/'handoff.json']:
        protected[str(p.relative_to(ROOT))]=SHA(p)
    plan={'version':'top10-los-v004','authorization':'2026-10-15 latest human: natural Top10 and satellite-ground LOS pressure sampling; zero training',
          'feature_names':TOP10,'model_feature_order':inventory['feature_names'],
          'code_sha256':SHA(CODE),'test_sha256':SHA(TEST),'independent_audit_code_sha256':SHA(AUDIT),
          'transport_sha256':SHA(HERE/'weather_download_v002.py'),
          'geometry':{'regions':geometry['regions'],'manifest_fingerprint':geometry['fingerprint'],
                      'pointing_review_required_ids':geometry['pointing_review_required_ids']},
          'original_parents':original['jobs'],'parents':parents,'adopted':adopted,
          'jobs':[a['job'] for a in adopted]+ordered,'protected_files':protected,
          'max_in_flight_requests':2,'initial_pressure_date_cap':7,
          'field_packing':'CDS Cartesian product of 4 variable types x 5 levels. Only original 8 pressure feature pairs are model inputs. z at all levels is geometry support; remaining Cartesian companions never become predictors.',
          'surface_scope':'Two frozen Top10 fields only; original surface padding UTCs retained as provenance, not additional evaluation dates.',
          'geometry_scope':'Ray envelopes for acquisition only; actual hourly isobaric intersections required before prediction. FRA-00609 pointing discrepancy remains flagged.',
          'fallback':'Explicit NEW request size rejection only: date bisection to one day; legacy IDs never resubmitted or split.',
          'model_training_started':False,'scope':'Corrected Top10 LOS weather acquisition, not old Top10 completion or complete external evaluation'}
    coverage_contract(plan); plan['fingerprint']=FP(plan)
    return plan


def load_plan():
    plan=READ(PLAN)
    REQUIRE(plan['fingerprint']==FP({k:v for k,v in plan.items() if k!='fingerprint'}),'Plan digest changed')
    REQUIRE(plan['code_sha256']==SHA(CODE) and plan['test_sha256']==SHA(TEST) and plan['independent_audit_code_sha256']==SHA(AUDIT),'Frozen code changed')
    for name,digest in plan['protected_files'].items(): REQUIRE(SHA(ROOT/name)==digest,'Protected input changed: '+name)
    coverage_contract(plan)
    return plan


def prepare(plan):
    transport.immutable(PLAN,plan)
    for item in plan['adopted']:
        source=ROOT/item['source_state']; saved=READ(source); node=item['job']
        REQUIRE(SHA(source)==item['source_state_sha256'] and saved['request_id']==item['request_id'],'Legacy migration changed')
        copied={**saved,'job':node,'request_fingerprint':FP(node),'plan_fingerprint':plan['fingerprint'],
                'migrated_from':item['source_state'],'migrated_from_sha256':item['source_state_sha256'],
                'counted_as_LOS_coverage':False}
        transport.immutable(transport.state_path(node),copied)


def runtime_nodes(plan):
    for item in plan['adopted']:
        saved=transport.state(item['job'])
        REQUIRE(saved.get('request_id')==item['request_id'],'Legacy ID lost')
        REQUIRE(saved.get('status')!='SPLIT_SIZE_REJECTION','Legacy splitting prohibited')
    return transport.leaves(plan)


def guard_states(pairs):
    REQUIRE(not any(s.get('status') in ('SUBMITTING','AMBIGUOUS_SUBMISSION') for _,s in pairs),'Submission ambiguous: official reconciliation before POST')
    REQUIRE(not any(s.get('status','').startswith('BLOCKED') for _,s in pairs),'Blocked remote/HTTP/access state requires diagnosis; slot not assumed free')
    active=[n for n,s in pairs if s.get('request_id') and s.get('status')!='VERIFIED']
    REQUIRE(len(active)<=2,'More than two active remote IDs')
    REQUIRE(len({s.get('request_id') for _,s in pairs if s.get('request_id') and s.get('status')!='VERIFIED'})==len(active),'Duplicate active ID')
    return active


def progress(plan,verify=False):
    parents={p['id']:p for p in plan['parents']}; covered={pid:Counter() for pid in parents}
    pairs=[(n,transport.state(n)) for n in runtime_nodes(plan)]
    for node,saved in pairs:
        if saved.get('status')=='VERIFIED':
            if verify: REQUIRE(SHA(DEST/node['target'])==saved['sha256'],'Verified new payload changed')
            if node['role']=='los_top10': covered[node['parent_id']].update(node['request']['day'])
    for pid,dates in covered.items(): REQUIRE(not(dates-Counter(parents[pid]['request']['day'])),'Excess/duplicate verified dates')
    pending=[{'id':n['id'],'role':n['role'],'request_id':s.get('request_id'),'status':s.get('status'),
              'remote_status':s.get('remote_status'),'last_checked_utc':s.get('last_checked_utc')}
              for n,s in pairs if s and s.get('status')!='VERIFIED']
    return {'plan_fingerprint':plan['fingerprint'],'feature_names':TOP10,
            'los_parents_total':len(parents),'los_parents_complete':sum(covered[k]==Counter(p['request']['day']) for k,p in parents.items()),
            'los_expected_request_hours':sum(p['expected_hours'] for p in parents.values()),
            'los_downloaded_request_hours':sum(sum(v.values())*len(parents[k]['request']['time']) for k,v in covered.items()),
            'los_batches_verified':sum(s.get('status')=='VERIFIED' and n['role']=='los_top10' for n,s in pairs),
            'legacy_downloads_verified_not_LOS':sum(s.get('status')=='VERIFIED' and n['role']=='legacy_top1' for n,s in pairs),
            'remaining_unsubmitted':sum(not s for _,s in pairs),'runtime_leaf_batches':len(pairs),'pending_or_blocked':pending,
            'full_field_QA_complete':False,'external_evaluation_complete':False,'model_training_started':False,
            'old_top10_complete':False,'pointing_review_required_ids':plan['geometry']['pointing_review_required_ids']}


def cycle(plan,client):
    pairs=[(n,transport.state(n)) for n in runtime_nodes(plan)];guard_states(pairs)
    for node,saved in pairs:
        if saved.get('request_id') and saved.get('status')!='VERIFIED':transport.poll(node,client)
    pairs=[(n,transport.state(n)) for n in runtime_nodes(plan)];active=guard_states(pairs)
    for node,saved in pairs:
        if len(active)>=2:break
        if not saved:
            REQUIRE(node['role']=='los_top10','Legacy request must never be POSTed')
            transport.submit(node,plan,client);current=transport.state(node)
            guard_states([(node,current)])
            if current.get('request_id'):active.append(node)
    return progress(plan)


def run():
    plan=load_plan()
    for filename in ('preflight_qa.json','plan_independent_qa.json'):
        qa=READ(DEST/filename)
        REQUIRE(qa['status']=='PASS' and qa['plan_fingerprint']==plan['fingerprint'] and qa['code_sha256']==SHA(CODE),'Matching frozen QA absent: '+filename)
    progress(plan,verify=True);client=transport.client()
    while True:
        report=cycle(plan,client);transport.atomic(DEST/'progress.json',{'checked_utc':transport.now(),**report})
        if report['los_parents_complete']==report['los_parents_total'] and not report['pending_or_blocked']:
            load_plan();progress(plan,verify=True)
            transport.immutable(DEST/'download_qa.json',{'status':'PASS_DOWNLOAD_READABILITY_ONLY','completed_utc':transport.now(),**report})
            print('TOP10 LOS REQUEST COVERAGE DOWNLOADED; FULL FIELD QA/PREDICTION STILL REQUIRED',flush=True);return
        time.sleep(30)


def status():
    plan=load_plan()
    return {**progress(plan,verify=True),'os_lock_held':transport.lock_held(),
            'active_worker':READ(DEST/'active_worker.json') if (DEST/'active_worker.json').exists() else None}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','status'));args=parser.parse_args()
    if args.command=='status':print(json.dumps(status(),indent=2),flush=True)
    elif args.command=='prepare':
        candidate=build_plan()
        with transport.writer('top10_los_prepare'):prepare(candidate)
        print(json.dumps(coverage_contract(candidate),indent=2),flush=True)
    else:
        with transport.writer('top10_los_run'):run()
