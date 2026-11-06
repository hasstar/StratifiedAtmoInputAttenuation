"""T012 NCAR-only, exact Top10 subset acquisition; no model/fit/inference.

Initialize once, then run/status. Old frozen files are read-only. Every network
operation is an anonymous GET to the fixed official NCAR host, never CDS POST.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import requests
import weather_download_top10_fixedheight_v007 as legacy
import ncar_top10_transport_v001 as transport

sys.dont_write_bytecode=True
io=legacy.io; core=legacy.core
ROOT,HERE,EXTERNAL=legacy.ROOT,legacy.HERE,legacy.EXTERNAL
PREVIOUS=EXTERNAL/'weather_top10_fixedheight_v007'
DEST=EXTERNAL/'weather_top10_ncar_v008'
CODE=Path(__file__).resolve(); PLAN=DEST/'plan.json'
R,SHA,FP,require=io.read,io.sha,io.fingerprint,io.require
io.DEST=DEST
# The public NCAR route must not read CDS credentials even when logging an error.
io.safe_error=lambda exc: type(exc).__name__+': '+str(exc)[:2000]
legacy.payloads=lambda plan:[{'path':ROOT/x['path'],'sha256':x['sha256'],'node':None} for x in plan['reusable']]
META_LOCK=threading.Lock()
MAX_BODY=16*1024*1024

class HttpFailure(RuntimeError):
    def __init__(self,status,message,retry_after=0):
        super().__init__(message);self.status=status;self.retry_after=retry_after

def signed(obj):
    return {**obj,'fingerprint':FP(obj)}

def verified_record(path):
    obj=R(path);require(obj['fingerprint']==FP({k:v for k,v in obj.items() if k!='fingerprint'}),'Record fingerprint changed: '+str(path))
    return obj

def build_plan():
    require(not io.lock_held(),'Shared downloader lock occupied')
    require(SHA(PREVIOUS/'plan.json')=='369d29d0fce76aaddfc9b8efc1e02710077e74fa094b167b0a0e6d4d77aa51f9','Source plan changed')
    old=R(PREVIOUS/'plan.json')
    for path,digest in old['source_sha256'].items(): require(SHA(path)==digest,'Old frozen source changed')
    for path,digest in old['protected_files'].items(): require(SHA(ROOT/path)==digest,'Old protected source changed')
    plan=copy.deepcopy(old)
    plan.pop('fingerprint');plan.pop('adopted');plan.pop('inherited_auxiliary_policy')
    reusable=plan['reusable'];seen={x['sha256'] for x in reusable};retired=[]
    for p in sorted((PREVIOUS/'jobs').glob('*.json')):
        s=R(p)
        if s.get('status')=='VERIFIED':
            payload=PREVIOUS/s['job']['target'];require(SHA(payload)==s['sha256'],'Old completed payload changed')
            if s['sha256'] not in seen:
                reusable.append({'path':str(payload.relative_to(ROOT)),'sha256':s['sha256'],'state':str(p.relative_to(ROOT)),'state_sha256':SHA(p)})
                seen.add(s['sha256'])
        else:
            retired.append({'source_state':str(p.relative_to(ROOT)),'state_sha256':SHA(p),'request_id':s.get('request_id'),
                            'last_local_status':s.get('status'),'last_remote_status':s.get('remote_status'),
                            'policy':'No further local polling, result download, cancellation, split or resubmission; old CDS remote state is not NCAR activity'})
    protected={str(p.relative_to(ROOT)):SHA(p) for p in EXTERNAL.rglob('*') if p.is_file() and not p.is_relative_to(DEST) and p!=io.LOCK}
    protected.update(old['protected_files'])
    dependencies=[CODE,HERE/'ncar_top10_transport_v001.py',HERE/'test_ncar_top10_v001.py',HERE/'audit_ncar_top10_v001.py',
                  Path(legacy.__file__),HERE/'fixedheight_los_core_v001.py',HERE/'nearest_los_core_v002.py',HERE/'top10_los_geometry_v003.py',
                  HERE/'weather_download_v002.py',core.geometry.LOS]
    plan.update(version='top10-ncar-v008',authorization='2026-10-15 direct human: 全部改为NCAR下载; all missing Top10 uses NCAR; existing bytes retained',
                route='NCAR GDEX d633000 published quarter-degree files; constrained DAP2 GET only',remote_base=transport.BASE,
                predecessor_plan=str((PREVIOUS/'plan.json').relative_to(ROOT)),predecessor_plan_sha256=SHA(PREVIOUS/'plan.json'),
                retired_cds_jobs=retired,source_sha256={str(p):SHA(p) for p in dependencies},protected_files=protected,
                max_in_flight_requests=2,auxiliary_pairs=[],transport_policy={'connect_timeout_seconds':10,'read_timeout_seconds':180,
                'maximum_body_bytes':MAX_BODY,'retry':'Same deterministic GET, exponential backoff 30..300 seconds; no POST',
                'fatal_http':[400,401,403,404,410,413,422],'source_contract':'NCAR metadata/coords/valid time and exact sample identity checked; no cross-platform equality claimed'},
                wave_contract='Original parent order, exact missing samples only; single variable/layer per request, archive-file boundaries; forecast valid time = initialization + hour(1..12)')
    require(plan['auxiliary_pairs']==[] and len(plan['feature_names'])==10,'Top10 scope changed')
    return signed(plan)

def load_plan():
    plan=verified_record(PLAN)
    require(plan['remote_base']==transport.BASE and plan['max_in_flight_requests']==2 and plan['auxiliary_pairs']==[],'Route/scope changed')
    for path,digest in plan['source_sha256'].items():require(SHA(path)==digest,'Frozen source changed '+path)
    for path,digest in plan['protected_files'].items():require(SHA(ROOT/path)==digest,'Old protected file changed '+path)
    return plan

def waves(plan):
    result=[]
    for p in sorted((DEST/'waves').glob('*.json')):
        w=verified_record(p);require(w['plan_fingerprint']==plan['fingerprint'],'Wave plan mismatch')
        result.append(w)
    return result

def nodes(plan):
    return [n for wave in waves(plan) for n in wave['jobs']]

def state(node):
    p=DEST/'jobs'/(node['id']+'.json')
    if not p.exists():return {}
    s=R(p);require(s['request_fingerprint']==FP(node),'Job changed')
    return s

def save(node,plan,**updates):
    saved=state(node)
    saved.update(job=node,request_fingerprint=FP(node),plan_fingerprint=plan['fingerprint'],updated_utc=io.now(),**updates)
    io.atomic(DEST/'jobs'/(node['id']+'.json'),saved)
    return saved

def history(kind,node,detail):
    io.immutable(DEST/'failures'/(str(time.time_ns())+'_'+kind+'.json'),{'kind':kind,'utc':io.now(),'job_id':node['id'],**detail})

def preserve_partial(path,node,reason):
    if path.exists():
        digest=SHA(path);target=DEST/'partial_history'/(str(time.time_ns())+'_'+path.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        # Both resolved paths are in this version's isolated directory.
        require(path.resolve().is_relative_to(DEST.resolve()) and target.resolve().is_relative_to(DEST.resolve()),'Unsafe partial path')
        os.replace(path,target)
        io.immutable(target.with_suffix(target.suffix+'.json'),{'utc':io.now(),'job_id':node['id'],'reason':reason,'sha256':digest,'bytes':target.stat().st_size})

def get_body(url,partial,node):
    require(url.startswith(transport.BASE) and '\n' not in url,'Non-NCAR destination refused')
    preserve_partial(partial,node,'prior interrupted GET preserved before idempotent retry')
    partial.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with requests.get(url,headers={'Accept-Encoding':'identity','User-Agent':'T012-NCAR-exact-Top10/008'},stream=True,timeout=(10,180),allow_redirects=False) as r:
        if r.status_code!=200:
            delay=r.headers.get('Retry-After','0')
            raise HttpFailure(r.status_code,'NCAR HTTP '+str(r.status_code),int(delay) if delay.isdigit() else 0)
        require(r.headers.get('Content-Encoding','identity')=='identity','Unexpected compressed HTTP response')
        expected=int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None
        require(expected is None or 0<expected<=MAX_BODY,'Unexpected response size')
        total=0
        with partial.open('xb') as f:
            for block in r.iter_content(chunk_size=65536):
                total+=len(block);require(total<=MAX_BODY,'Response exceeded bounded subset limit')
                f.write(block)
            f.flush();os.fsync(f.fileno())
        require(total>0 and (expected is None or total==expected),'Truncated Content-Length')
    return {'http_status':200,'bytes':total,'content_length':expected,'seconds':time.monotonic()-started,'url':url,'sha256':SHA(partial)}

def metadata(node):
    path=DEST/'metadata'/(FP(node['file'])+'.das')
    with META_LOCK:
        if path.exists():
            stamp=R(path.with_suffix('.json'));require(SHA(path)==stamp['sha256'],'Metadata SHA changed')
        else:
            part=path.with_suffix('.das.part')
            receipt=get_body(node['metadata_url'],part,node)
            transport.metadata_contract(part.read_text(encoding='utf-8'),node)
            os.replace(part,path);io.immutable(path.with_suffix('.json'),receipt)
    text=path.read_text(encoding='utf-8');transport.metadata_contract(text,node)
    return path,text

def download(node,plan):
    before=state(node);attempt=int(before.get('attempts',0))+1
    save(node,plan,status='GET_IN_PROGRESS',attempts=attempt,started_utc=io.now())
    final=DEST/node['target'];part=final.with_suffix('.ascii.part')
    try:
        meta,das=metadata(node)
        if final.exists():
            # Successful rename before a crash can be recovered without another GET.
            receipt={'bytes':final.stat().st_size,'sha256':SHA(final),'seconds':None,'url':node['url'],'recovered_complete_file':True}
            source=final
        elif part.exists():
            try:
                transport.decode(part.read_text(encoding='utf-8'),das,node)
            except Exception:
                preserve_partial(part,node,'incomplete or invalid DAP response; exact same GET retry')
                receipt=get_body(node['url'],part,node)
            else:
                receipt={'bytes':part.stat().st_size,'sha256':SHA(part),'seconds':None,'url':node['url'],'recovered_complete_part':True}
            source=part
        else:
            receipt=get_body(node['url'],part,node);source=part
        rows,contract=transport.decode(source.read_text(encoding='utf-8'),das,node)
        if source==part:os.replace(part,final)
        record=signed({'request_fingerprint':FP(node),'plan_fingerprint':plan['fingerprint'],'source':str(final.relative_to(ROOT)),
                       'source_sha256':SHA(final),'metadata':str(meta.relative_to(ROOT)),'metadata_sha256':SHA(meta),'contract':contract,'records':rows})
        record_path=DEST/'records'/(node['id']+'.json');io.immutable(record_path,record)
        save(node,plan,status='VERIFIED_SUBSET',sha256=SHA(final),records_sha256=SHA(record_path),receipt=receipt,values=len(rows),completed_utc=io.now(),next_retry_epoch=0)
        print('NCAR SUBSET',node['id'],len(rows),receipt['bytes'],receipt['seconds'],flush=True)
    except Exception as exc:
        history('GET_FAILURE',node,{'attempt':attempt,'exception':io.safe_error(exc),'http_status':getattr(exc,'status',None)})
        transient=isinstance(exc,(requests.RequestException,)) or (isinstance(exc,HttpFailure) and exc.status in (408,425,429,500,502,503,504))
        delay=max(min(300,30*2**min(attempt-1,4)),min(3600,getattr(exc,'retry_after',0)))
        save(node,plan,status='RETRY_SAME_GET' if transient else 'BLOCKED_DOWNLOAD_INTEGRITY_OR_ACCESS',error=io.safe_error(exc),next_retry_epoch=time.time()+delay if transient else 0)

def read_rows(node,saved):
    path=DEST/node['target'];rp=DEST/'records'/(node['id']+'.json')
    require(SHA(path)==saved['sha256'] and SHA(rp)==saved['records_sha256'],'Completed subset changed')
    record=verified_record(rp)
    require(record['request_fingerprint']==FP(node) and record['source_sha256']==saved['sha256'],'Subset provenance changed')
    require(SHA(ROOT/record['metadata'])==record['metadata_sha256'],'Subset metadata changed')
    require(Counter(tuple(k) for k,v in record['records'])==Counter(transport.expected_keys(node)),'Canonical exact sample mismatch')
    return record['records']

def load_cache(plan,parent):
    cache,hashes=legacy.load_cache(plan,parent)
    hours=set(core.parent_hours(parent));pressure='pressure_level' in parent['request']
    for node in nodes(plan):
        if bool(node['level_hpa'])!=pressure:continue
        if not any(k[2] in hours for k in transport.expected_keys(node)):continue
        s=state(node)
        if s.get('status')!='VERIFIED_SUBSET':continue
        for key,value in read_rows(node,s):
            key=tuple(key)
            if key[2] in hours and key not in cache:cache[key]=(value,str((DEST/node['target']).relative_to(ROOT)))
        hashes[str((DEST/node['target']).relative_to(ROOT))]=s['sha256']
    return cache,hashes

def advance(plan):
    for parent in plan['parents']:
        done=DEST/'units'/(parent['id']+'.json')
        if done.exists():
            require(verified_record(done)['plan_fingerprint']==plan['fingerprint'],'Unit plan mismatch');continue
        cache,hashes=load_cache(plan,parent)
        phase,demands,records=legacy.acquisition_needs(plan,parent,cache)
        if demands:
            jobs=[]
            for node in transport.plan_requests(demands):
                ident='ncar008_'+FP(node)[:24]
                node.update(id=ident,parent_id=parent['id'],target='payloads/'+ident+'.ascii')
                require(not state(node),'Existing job needed but not in valid cache; no duplicate request')
                jobs.append(node)
            w=signed({'plan_fingerprint':plan['fingerprint'],'parent_id':parent['id'],'phase':phase,'jobs':jobs,
                      'needed_samples':len(demands),'demands_sha256':FP(sorted(demands)),'source_payloads':hashes,
                      'exact_scope_no_extras':True,'model_fit_or_predict_calls':0})
            io.immutable(DEST/'waves'/(f'{len(waves(plan))+1:04d}.json'),w)
            print('NCAR WAVE',parent['id'],phase,len(jobs),len(demands),flush=True)
            return False
        counts=Counter(row['status'] for rec in records for row in rec.get('rows',[]))
        io.immutable(done,signed({'plan_fingerprint':plan['fingerprint'],'parent_id':parent['id'],'records':records,'source_payloads':hashes,
            'geometry_status_counts':dict(counts),'hour_scope':parent['expected_hours'],
            'scope':'Acquisition recorded; independent field/model QA pending; fixed four heights are not actual hourly isobaric heights',
            'invalid_rows_retained':sum(c for st,c in counts.items() if st not in ('STABLE_NEAREST','STABLE_FIXED_HEIGHT_NEAREST')),
            'new_fit_calls':0,'predictions_executed':0}))
        print('NCAR ACQUISITION UNIT',parent['id'],dict(counts),flush=True)
    return True

def progress(plan,verify=False):
    pairs=[(n,state(n)) for n in nodes(plan)]
    if verify:
        for n,s in pairs:
            if s.get('status')=='VERIFIED_SUBSET':read_rows(n,s)
    return {'checked_utc':io.now(),'plan_fingerprint':plan['fingerprint'],'route':'NCAR_ONLY',
        'calendar_parents_total':len(plan['parents']),'acquisition_units_recorded':len(list((DEST/'units').glob('*.json'))),
        'waves':len(waves(plan)),'planned_subsets_in_materialized_waves':len(pairs),
        'verified_ncar_subsets':sum(s.get('status')=='VERIFIED_SUBSET' for n,s in pairs),
        'verified_unique_grid_field_hours':sum(s.get('values',0) for n,s in pairs if s.get('status')=='VERIFIED_SUBSET'),
        'bytes_downloaded':sum(s.get('receipt',{}).get('bytes',0) for n,s in pairs if s.get('status')=='VERIFIED_SUBSET'),
        'status_counts':dict(Counter(s.get('status','UNSTARTED') for n,s in pairs)),
        'pending_or_blocked':[{'id':n['id'],'variable':n['variable'],'level_hpa':n['level_hpa'],'status':s.get('status','UNSTARTED'),
                               'attempts':s.get('attempts',0),'error':s.get('error'),'next_retry_epoch':s.get('next_retry_epoch')} for n,s in pairs if s.get('status')!='VERIFIED_SUBSET'],
        'old_cds_remote_jobs_managed':False,'full_field_independent_QA_complete':False,'external_evaluation_complete':False,'new_fit_calls':0,'predictions_executed':0}

def run():
    plan=load_plan()
    for filename in ('preflight_qa.json','plan_independent_qa.json'):
        qa=R(DEST/filename)
        require(qa['status']=='PASS' and qa['plan_fingerprint']==plan['fingerprint'] and qa['code_sha256']==SHA(CODE),'Matching frozen QA missing')
    progress(plan,True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active={}
        while True:
            for future in list(active):
                if future.done():future.result();active.pop(future)
            current=nodes(plan); pending=[n for n in current if state(n).get('status')!='VERIFIED_SUBSET']
            blocked=[n for n in pending if state(n).get('status','').startswith('BLOCKED')]
            if blocked:
                io.atomic(DEST/'progress.json',progress(plan));raise RuntimeError('Blocked NCAR subset; inspect preserved failure; no old-route fallback')
            if not pending and not active:
                if advance(plan):
                    load_plan();io.immutable(DEST/'acquisition_complete.json',{'status':'ACQUISITION_RECORDED_INDEPENDENT_QA_PENDING',**progress(plan,True)})
                    io.atomic(DEST/'progress.json',progress(plan));return
                pending=[n for n in nodes(plan) if state(n).get('status')!='VERIFIED_SUBSET']
            running_ids={n['id'] for n in active.values()}
            for node in pending:
                if len(active)>=2:break
                s=state(node)
                if node['id'] not in running_ids and s.get('next_retry_epoch',0)<=time.time():
                    active[pool.submit(download,node,plan)]=node
            io.atomic(DEST/'progress.json',progress(plan))
            if active:wait(active,timeout=5,return_when=FIRST_COMPLETED)
            else:time.sleep(5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('initialize','run','status'));args=parser.parse_args()
    if args.command=='status':
        print(json.dumps({**progress(load_plan(),True),'os_lock_held':io.lock_held(),
                          'active_worker':R(DEST/'active_worker.json') if (DEST/'active_worker.json').exists() else None},indent=2));return
    candidate=build_plan() if args.command=='initialize' else None
    with io.writer('top10_ncar_'+args.command):
        if args.command=='initialize':
            require(not PLAN.exists(),'Initialization cannot be rerun');io.immutable(PLAN,candidate)
            print(json.dumps({'plan_fingerprint':candidate['fingerprint'],'reusable_payloads':len(candidate['reusable'])}))
        else:run()

if __name__=='__main__':main()
