"""Version 007: natural Top10 only, four fixed-height LOS nearest-grid layers.

No model import, training, feature synthesis, interpolation, remote cancellation,
or edits to any old run. Frozen v002 transport retains IDs and ambiguous outcomes.
Only initialize ONCE after offline review; run/status are the continuation entry.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import netCDF4
import weather_download_v002 as io
import fixedheight_los_core_v001 as core
sys.dont_write_bytecode = True
ROOT, HERE, EXTERNAL = io.ROOT, io.HERE, io.EXTERNAL
OLD = EXTERNAL/'weather_top10_los_v004'
PREVIOUS = EXTERNAL/'weather_top10_nearest_v006'
DEST = EXTERNAL/'weather_top10_fixedheight_v007'
io.DEST = DEST
PLAN = DEST/'plan.json'
CODE = Path(__file__).resolve()
TEST = HERE/'test_weather_download_top10_fixedheight_v007.py'
AUDIT = HERE/'audit_top10_fixedheight_plan_v001.py'
R, SHA, FP, require = io.read, io.sha, io.fingerprint, io.require
FIELDS = {'geopotential':(129,('m**2 s**-2','m2 s-2')),
          'u_component_of_wind':(131,('m s**-1','m s-1')),
          'ozone_mass_mixing_ratio':(203,('kg kg**-1','kg kg-1')),
          'relative_humidity':(157,('%',)),
          core.SURFACE[0]:(59,('J kg**-1','J kg-1','m**2 s**-2')),
          core.SURFACE[1]:(201,('K',))}
_split = io.split_rejected


def split_guarded(node, saved, evidence):
    if node['role'] == 'adopted':
        io.save(node, saved, status='BLOCKED_LEGACY_SIZE_DIAGNOSIS', failure=evidence)
    else:
        _split(node, saved, evidence)
io.split_rejected = split_guarded


def region(parent):
    matches = [r for r in ('GBS_England','GBS_Dundee','OpenSat_France') if parent['id'].startswith(r+'_')]
    require(len(matches)==1, 'Unknown parent region')
    return matches[0]


def build_plan():
    require(not io.lock_held(), 'Shared writer lock occupied')
    previous = R(PREVIOUS/'plan.json'); geom = R(OLD/'geometry_manifest.json'); inventory = R(OLD/'model_inventory.json')
    require(SHA(PREVIOUS/'plan.json')=='fecc5fe9119fa4354a0617def1e039f690578bbba01495390b7e1710857c44f2', 'Previous plan changed')
    require(geom['fingerprint']==FP({k:v for k,v in geom.items() if k!='fingerprint'}), 'Geometry manifest changed')
    adopted = []
    for path in sorted((PREVIOUS/'jobs').glob('*.json')):
        saved = R(path)
        if saved['status']=='VERIFIED':
            require(SHA(PREVIOUS/saved['job']['target'])==saved['sha256'], 'Old verified bytes changed')
            continue
        require(saved['status'] in ('WAITING_REMOTE','SUBMITTED','RETRY_SAME_ID') and saved.get('request_id'),
                'Unresolved old operation; official reconciliation required')
        require(saved['request_fingerprint']==FP(saved['job']), 'Old node fingerprint changed')
        node = copy.deepcopy(saved['job']); node.update(id='adopted007__'+saved['job']['id'], role='adopted')
        node['target'] = 'payloads/'+node['id']+'.download'
        req=node['request']
        require(req.get('pressure_level')==['5'] and req['variable']==['geopotential'], 'Unexpected inherited nonterminal request; reconcile scope')
        node['download_policy']='poll_only_retired_auxiliary'
        adopted.append({'job':node,'source_state':str(path.relative_to(ROOT)), 'source_state_sha256':SHA(path),
                        'request_id':saved['request_id']})
    require(len(adopted)<=2, 'Too many inherited requests')
    reusable = []; seen = set()
    for base, directory in [(EXTERNAL,'weather_jobs'),(EXTERNAL/'weather_v002','jobs'),
                            (EXTERNAL/'weather_top1_v003','jobs'),(OLD,'jobs'),(PREVIOUS,'jobs')]:
        for path in sorted((base/directory).glob('*.json')):
            saved = R(path)
            if saved.get('status') != 'VERIFIED':
                continue
            payload = base/saved['job']['target']
            require(SHA(payload)==saved['sha256'], 'Old data SHA differs')
            if saved['sha256'] not in seen:
                reusable.append({'path':str(payload.relative_to(ROOT)), 'sha256':saved['sha256'],
                                 'state':str(path.relative_to(ROOT)), 'state_sha256':SHA(path)})
                seen.add(saved['sha256'])
    # Full old trees are protected; own evolving output and shared byte-lock excluded.
    protected = {str(p.relative_to(ROOT)):SHA(p) for p in EXTERNAL.rglob('*')
                 if p.is_file() and not p.is_relative_to(DEST) and p!=io.LOCK}
    protected.update(previous['protected_files'])
    protected.update(previous['source_sha256'])
    dependencies = [CODE, TEST, AUDIT, HERE/'fixedheight_los_core_v001.py', HERE/'nearest_los_core_v002.py', HERE/'top10_los_geometry_v003.py',
                    HERE/'weather_download_v002.py', HERE/'preflight_top10_fixedheight_v001.py', core.geometry.LOS]
    source_sha = {str(p):SHA(p) for p in dependencies}
    parents = R(EXTERNAL/'weather_plan.json')['jobs']
    require(len(parents)==77 and sum(p['expected_hours'] for p in parents)==45663, 'Calendar scope changed')
    require(sum(p['expected_hours'] for p in parents if 'pressure_level' in p['request'])==22824, 'Pressure calendar changed')
    plan = {'version':'top10-fixedheight-v007', 'authorization':'Human explicitly removed four auxiliary z fields and accepted fixed standard-atmosphere LOS heights for those four layers; 2026-10-15',
            'parents':parents, 'links':geom['links'], 'adopted':adopted, 'reusable':reusable,
            'feature_names':inventory['feature_names'], 'natural_rank_top10':inventory['natural_rank_top10'],
            'model_inventory':str((OLD/'model_inventory.json').relative_to(ROOT)),
            'max_in_flight_requests':2, 'grid_degrees':.25, 'sampling':'coordinate-wise nearest 0.25-degree published grid; ties north/east; no spatial or temporal interpolation',
            'geometry':'1hPa uses its Top10 z hourly nearest-grid iteration; 5/500/650/850hPa use fixed standard-atmosphere heights with forward WGS84 LOS nearest grid, NOT actual hourly isobaric heights',
            'seed_heights_m':core.SEED_HEIGHT, 'pressure_pairs':{str(k):list(v) for k,v in core.LEVEL_VARS.items()},
            'auxiliary_pairs':[], 'fixed_heights':core.FIXED_HEIGHTS, 'standard_atmosphere_source':core.STANDARD_SOURCE,
            'original_calendar_requests_are_not_submission_templates':True,
            'inherited_auxiliary_policy':'poll original ID only; never download auxiliary payload or resubmit it',
            'scope':'33 pressure months/22824 hours plus original surface dates/padding. Dynamic exact-point waves, not old envelope coverage.',
            'wave_contract':'Only unsatisfied exact field/level/cell/hour demands; exact rectangles never add cells/hours/variables. One original parent at a time, fixed original order.',
            'pointing_review_required_ids':geom['pointing_review_required_ids'], 'source_sha256':source_sha,
            'protected_files':protected, 'new_fit_calls':0, 'predictions_executed':0}
    plan['fingerprint'] = FP(plan)
    return plan


def initialize(plan):
    require(not PLAN.exists(), 'Initialization cannot be rerun')
    io.immutable(PLAN, plan)
    for item in plan['adopted']:
        require(SHA(ROOT/item['source_state'])==item['source_state_sha256'], 'Adoption source changed')
        saved = R(ROOT/item['source_state']); node=item['job']
        io.immutable(io.state_path(node), {**saved, 'job':node, 'request_fingerprint':FP(node),
            'plan_fingerprint':plan['fingerprint'], 'migrated_from':item['source_state'], 'counted_as_new_nearest_coverage':False})
    return plan


def load_plan():
    plan = R(PLAN)
    require(plan['fingerprint']==FP({k:v for k,v in plan.items() if k!='fingerprint'}), 'Plan fingerprint differs')
    for path, digest in plan['source_sha256'].items():
        require(SHA(path)==digest, 'Frozen code changed: '+path)
    for path, digest in plan['protected_files'].items():
        require(SHA(ROOT/path)==digest, 'Protected old file changed: '+path)
    return plan


def waves(plan):
    result=[]
    for path in sorted((DEST/'waves').glob('*.json')):
        wave=R(path)
        require(wave['fingerprint']==FP({k:v for k,v in wave.items() if k!='fingerprint'}) and
                wave['plan_fingerprint']==plan['fingerprint'], 'Wave digest mismatch')
        for node in wave['jobs']:
            require(node['role']=='fixedheight007' and node['parent_id']==wave['parent_id'], 'Wave node lineage mismatch')
            core.validate_request(node['request'])
        result.append(wave)
    return result


def runtime_nodes(plan):
    for item in plan['adopted']:
        s=io.state(item['job']); require(s.get('request_id')==item['request_id'], 'Lost inherited ID')
        require(s.get('status')!='SPLIT_SIZE_REJECTION', 'Inherited split forbidden')
    jobs=[a['job'] for a in plan['adopted']]+[n for w in waves(plan) for n in w['jobs']]
    nodes=io.leaves({'jobs':jobs})
    for node in nodes:
        if node['role']=='fixedheight007': core.validate_request(node['request'])
    return nodes


def terminal(saved):
    return saved.get('status') in ('VERIFIED','RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD','RETIRED_REMOTE_TERMINAL_NO_RETRY')


def poll_retired(node, client):
    """Exact inherited auxiliary ID: GET status only, never get_results/download."""
    saved=io.state(node)
    require(node['role']=='adopted' and node.get('download_policy')=='poll_only_retired_auxiliary', 'Not a retired auxiliary node')
    if time.time()<saved.get('retry_after_epoch',0): return
    try:
        remote=client.get_remote(saved['request_id'])
        require(remote.collection_id==node['collection_id'], 'Retired collection mismatch')
        status=remote.status
        saved=io.save(node,saved,last_checked_utc=io.now(),remote_status=status)
        if status in ('accepted','running'):
            io.save(node,saved,status='WAITING_REMOTE',retry_count=0,retry_after_epoch=0)
        elif status=='successful':
            io.save(node,saved,status='RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD',payload_downloaded=False,
                    counted_as_new_coverage=False,terminal_utc=io.now())
            print('RETIRED AUXILIARY FINISHED WITHOUT DOWNLOAD',node['id'],flush=True)
        elif status in ('failed','rejected','dismissed','deleted'):
            evidence=io.failure('RETIRED_AUXILIARY_REMOTE_TERMINAL',node,request_id=saved['request_id'],remote_status=status)
            io.save(node,saved,status='RETIRED_REMOTE_TERMINAL_NO_RETRY',failure=evidence,
                    payload_downloaded=False,counted_as_new_coverage=False,terminal_utc=io.now())
        else:
            raise RuntimeError('Unknown retired remote status: '+status)
    except Exception as exc:
        evidence=io.failure('RETIRED_AUXILIARY_POLL_FAILED_ID_RETAINED',node,exc,request_id=saved['request_id'])
        code=getattr(getattr(exc,'response',None),'status_code',None)
        if code in (401,403,404,410):
            io.save(node,saved,status='BLOCKED_REMOTE_ACCESS',failure=evidence,http_status=code)
        else:
            retries=saved.get('retry_count',0)+1
            io.save(node,saved,status='RETRY_SAME_ID',failure=evidence,retry_count=retries,
                    retry_after_epoch=time.time()+min(900,60*2**min(retries-1,4)))


def guard(pairs):
    require(not any(s.get('status') in ('SUBMITTING','AMBIGUOUS_SUBMISSION') for _,s in pairs), 'Ambiguous submission; reconcile official requests before POST')
    require(not any(s.get('status','').startswith('BLOCKED') for _,s in pairs), 'Blocked state requires diagnosis; no slot release')
    active=[n for n,s in pairs if s.get('request_id') and not terminal(s)]
    ids=[s['request_id'] for n,s in pairs if s.get('request_id') and not terminal(s)]
    require(len(active)<=2 and len(set(ids))==len(ids), 'Two-slot/ID uniqueness violation')
    return active


def payloads(plan):
    result=[{'path':ROOT/x['path'],'sha256':x['sha256'],'node':None} for x in plan['reusable']]
    for node in runtime_nodes(plan):
        s=io.state(node)
        if s.get('status')=='VERIFIED':
            result.append({'path':DEST/node['target'],'sha256':s['sha256'],'node':node})
    return result


def datasets(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe archive')
                if name.endswith('.nc'):
                    with netCDF4.Dataset('memory.nc', memory=archive.read(name)) as ds:
                        yield ds
    else:
        with netCDF4.Dataset(path) as ds:
            yield ds


def load_cache(plan, parent):
    """Only read verified data. Decode CF UTC, grid, level and parameter identities.

    Source priority is deterministic old-then-new; never average duplicate values.
    Nonfinite values remain detectable and block the affected acquisition unit.
    """
    cache={}; hashes={}; allowed_hours=set(core.parent_hours(parent)); pressure='pressure_level' in parent['request']
    bounds=R(OLD/'geometry_manifest.json')['regions'][region(parent)]['pressure_area_NWSE' if pressure else 'surface_area_NWSE']
    n,w,s,e=bounds
    wanted = set(core.LEVEL_VARS) if pressure else {0}
    for item in payloads(plan):
        if item['node']:
            node_request=item['node']['request']
            if ('pressure_level' in node_request)!=pressure or not (set(core.parent_hours(item['node'])) & allowed_hours):
                continue
        path=item['path']; require(SHA(path)==item['sha256'], 'Payload hash mismatch')
        selected=set(); expected=set(core.expand_request(item['node']['request'])) if item['node'] and item['node']['role']=='fixedheight007' else None
        observed=set()
        for ds in datasets(path):
            require('valid_time' in ds.variables, 'UTC valid_time missing')
            tv=ds['valid_time']; require(getattr(tv,'calendar','standard') in ('standard','gregorian','proleptic_gregorian'), 'Unsupported calendar')
            times=netCDF4.num2date(tv[:],tv.units,calendar=getattr(tv,'calendar','standard'),only_use_cftime_datetimes=False)
            epochs=[int(datetime(t.year,t.month,t.day,t.hour,t.minute,t.second,tzinfo=timezone.utc).timestamp()) for t in times]
            require(len(set(epochs))==len(epochs) and all(t%3600==0 for t in epochs), 'Nonunique/nonhourly UTC')
            lat=np.asarray(ds['latitude'][:],float); lon=(np.asarray(ds['longitude'][:],float)+180)%360-180
            require(getattr(ds['latitude'],'units','')=='degrees_north' and getattr(ds['longitude'],'units','')=='degrees_east','Coordinate units differ')
            require(np.isfinite(lat).all() and np.isfinite(lon).all(), 'Nonfinite coordinates')
            require(np.all(np.abs(lat/.25-np.rint(lat/.25))<1e-7) and np.all(np.abs(lon/.25-np.rint(lon/.25))<1e-7), 'Not published quarter-degree grid')
            levels=np.asarray(ds['pressure_level'][:],float).tolist() if 'pressure_level' in ds.variables else [0.]
            if levels!=[0.]:
                require(ds['pressure_level'].units=='hPa', 'Pressure unit mismatch')
            require(len(set(lat))==len(lat) and len(set(lon))==len(lon) and len(set(levels))==len(levels), 'Duplicate coordinates/levels')
            for variable in ds.variables.values():
                param=getattr(variable,'GRIB_paramId',None)
                matches=[name for name,(pid,units) in FIELDS.items() if param==pid]
                if not matches:
                    require(expected is None or param is None, 'Unexpected new weather parameter')
                    continue
                name=matches[0]; require(variable.units in FIELDS[name][1], 'Field unit mismatch: '+name)
                dims=list(variable.dimensions); values=np.ma.filled(variable[:],np.nan)
                axes=['valid_time']+(['pressure_level'] if 'pressure_level' in dims else [])+['latitude','longitude']
                for axis in list(dims):
                    if axis not in axes:
                        idx=dims.index(axis); require(values.shape[idx]==1, 'Unexpected nonsingleton field axis')
                        values=np.take(values,0,axis=idx);dims.pop(idx)
                require(set(dims)==set(axes), 'Unsupported field layout')
                values=np.transpose(values,[dims.index(a) for a in axes])
                if 'pressure_level' not in axes: values=values[:,None,:,:]
                require(values.shape==(len(epochs),len(levels),len(lat),len(lon)), 'Field shape mismatch')
                for kt,epoch in enumerate(epochs):
                    relevant=epoch in allowed_hours
                    if not relevant and expected is None: continue
                    for kl,level_value in enumerate(levels):
                        level=int(level_value); require(level==level_value, 'Noninteger pressure level')
                        for ki,latitude in enumerate(lat):
                            for kj,longitude in enumerate(lon):
                                key=(name,level,epoch,round(latitude/.25),round(longitude/.25))
                                value=float(values[kt,kl,ki,kj])
                                if expected is not None:
                                    require(np.isfinite(value), 'Nonfinite newly downloaded required field')
                                    require(key not in observed, 'Duplicate new field across NetCDF members')
                                    observed.add(key)
                                if relevant and level in wanted and s<=latitude<=n and w<=longitude<=e:
                                    if key not in cache:
                                        cache[key]=(value,str(path.relative_to(ROOT))); selected.add(key)
        if expected is not None:
            require(observed==expected, 'New payload exact variable/level/grid/hour scope mismatch')
        if selected:
            hashes[str(path.relative_to(ROOT))]=item['sha256']
    return cache, hashes


def acquisition_needs(plan, parent, cache):
    hours=core.parent_hours(parent); links=[x for x in plan['links'] if x['region']==region(parent)]
    require(bool(links), 'No source links')
    records=[]; z_missing=set(); targets=set()
    if 'pressure_level' in parent['request']:
        for link in links:
            for level in core.LEVEL_VARS:
                rows, missing, needed=core.trace(link,level,hours,lambda *key:cache.get(key))
                records.append({'link_id':link['id'],'level_hpa':level,'hours_utc_epoch':hours,'rows':rows,
                                'prediction_geometry_review_pending':link['id'] in plan['pointing_review_required_ids']})
                z_missing.update(missing);targets.update(needed)
        if z_missing:
            return 'geopotential_location',z_missing,records
    else:
        for link in links:
            cell=core.grid_index(link['latitude_deg_n'],link['longitude_deg_e'])
            records.append({'link_id':link['id'],'surface_cell':list(cell),'hours_utc_epoch':hours})
            targets.update((v,0,t,*cell) for v in core.SURFACE for t in hours)
    missing={key for key in targets if key not in cache}
    require(all(np.isfinite(cache[k][0]) for k in targets if k in cache), 'Nonfinite required historical sample')
    return 'direct_top10_fields',missing,records


def advance(plan):
    """One frozen-calendar unit at a time. No score/model based date choice."""
    for parent in plan['parents']:
        done=DEST/'units'/(parent['id']+'.json')
        if done.exists():
            record=R(done)
            require(record['plan_fingerprint']==plan['fingerprint'] and record['fingerprint']==FP({k:v for k,v in record.items() if k!='fingerprint'}), 'Unit provenance differs')
            continue
        cache, hashes=load_cache(plan,parent)
        phase,demands,records=acquisition_needs(plan,parent,cache)
        if demands:
            requests=core.demands_to_requests(parent,demands)
            jobs=[]
            for request in requests:
                ident=parent['id']+'__fixedheight007__'+FP(request)[:20]
                require(not io.state_path({'id':ident}).exists(), 'Already requested exact samples missing; stop, do not duplicate POST')
                jobs.append({'id':ident,'parent_id':parent['id'],'role':'fixedheight007','collection_id':parent['collection_id'],
                             'request':request,'expected_hours':len(request['day'])*len(request['time']),
                             'target':'payloads/'+ident+'.download'})
            wave={'plan_fingerprint':plan['fingerprint'],'parent_id':parent['id'],'phase':phase,'jobs':jobs,
                  'needed_samples':len(demands),'demands_sha256':FP(sorted(demands)),'source_payloads':hashes,
                  'no_extra_cells_or_fields_or_hours':True,'model_fit_or_predict_calls':0}
            wave['fingerprint']=FP(wave)
            existing=waves(plan)
            io.immutable(DEST/'waves'/('%04d.json'%(len(existing)+1)),wave)
            print('EXACT GRID WAVE',parent['id'],phase,len(jobs),len(demands),flush=True)
            return False
        statuses=Counter(row['status'] for record in records for row in record.get('rows',[]))
        record={'plan_fingerprint':plan['fingerprint'],'parent_id':parent['id'],'records':records,'source_payloads':hashes,
                'geometry_status_counts':dict(statuses),'hour_scope':parent['expected_hours'],
                'scope':'Acquisition and fixed-height/1hPa-hourly nearest geometry records; independent field/model QA pending, four fixed layers are not actual hourly pressure surfaces',
                'invalid_rows_retained':sum(c for st,c in statuses.items() if st not in ('STABLE_NEAREST','STABLE_FIXED_HEIGHT_NEAREST')),
                'new_fit_calls':0,'predictions_executed':0}
        record['fingerprint']=FP(record);io.immutable(done,record)
        print('NEAREST ACQUISITION UNIT',parent['id'],dict(statuses),flush=True)
    return True


def progress(plan, verify=False):
    pairs=[(n,io.state(n)) for n in runtime_nodes(plan)]
    if verify:
        for n,s in pairs:
            if s.get('status')=='VERIFIED': require(SHA(DEST/n['target'])==s['sha256'], 'Verified payload changed')
    return {'plan_fingerprint':plan['fingerprint'],'calendar_parents_total':77,
            'acquisition_units_recorded':len(list((DEST/'units').glob('*.json'))),
            'waves':len(waves(plan)),'runtime_leaves':len(pairs),'verified_requests':sum(s.get('status')=='VERIFIED' for n,s in pairs),
            'retired_auxiliary_terminal_requests':sum(s.get('status','').startswith('RETIRED_REMOTE_') for n,s in pairs),
            'unsubmitted':sum(not s for n,s in pairs),
            'pending_or_blocked':[{'id':n['id'],'request_id':s.get('request_id'),'status':s.get('status'),
                                  'remote_status':s.get('remote_status'),'last_checked_utc':s.get('last_checked_utc')}
                                 for n,s in pairs if s and not terminal(s)],
            'full_field_independent_QA_complete':False,'external_evaluation_complete':False,'old_top10_complete':False,
            'new_fit_calls':0,'predictions_executed':0}


def cycle(plan, client):
    pairs=[(n,io.state(n)) for n in runtime_nodes(plan)];guard(pairs)
    for node,s in pairs:
        if s.get('request_id') and not terminal(s):
            if node.get('download_policy')=='poll_only_retired_auxiliary': poll_retired(node,client)
            else: io.poll(node,client)
    pairs=[(n,io.state(n)) for n in runtime_nodes(plan)];guard(pairs)
    new_work=any(n['role']=='fixedheight007' and not terminal(s) for n,s in pairs)
    complete=False
    if not new_work: complete=advance(plan)
    pairs=[(n,io.state(n)) for n in runtime_nodes(plan)];active=guard(pairs)
    for node,saved in pairs:
        if len(active)>=2: break
        if not saved:
            require(node['role']=='fixedheight007','Never POST inherited job')
            core.validate_request(node['request'])
            io.submit(node,plan,client);guard([(node,io.state(node))])
            if io.state(node).get('request_id'): active.append(node)
    return complete and not active


def run():
    plan=load_plan()
    for filename in ('preflight_qa.json','plan_independent_qa.json'):
        qa=R(DEST/filename)
        require(qa['status']=='PASS' and qa['plan_fingerprint']==plan['fingerprint'] and qa['code_sha256']==SHA(CODE), 'Matching frozen QA missing')
    progress(plan,True); client=io.client()
    while True:
        done=cycle(plan,client);io.atomic(DEST/'progress.json',{'checked_utc':io.now(),**progress(plan)})
        if done:
            load_plan();io.immutable(DEST/'download_qa.json',{'status':'ACQUISITION_RECORDED_INDEPENDENT_QA_PENDING',**progress(plan,True)})
            return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('initialize','run','status'));args=parser.parse_args()
    if args.command=='status':
        print(json.dumps({**progress(load_plan(),True),'os_lock_held':io.lock_held(),
                          'active_worker':R(DEST/'active_worker.json') if (DEST/'active_worker.json').exists() else None},indent=2))
    else:
        candidate=build_plan() if args.command=='initialize' else None
        with io.writer('top10_fixedheight_'+args.command):
            if args.command=='initialize':
                p=initialize(candidate);print(json.dumps({'plan_fingerprint':p['fingerprint'],'adopted':len(p['adopted']),'reusable':len(p['reusable'])}))
            else: run()

