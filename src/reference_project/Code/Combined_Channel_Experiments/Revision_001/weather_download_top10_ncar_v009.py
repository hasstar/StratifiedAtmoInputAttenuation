"""NCAR v009: immutable v008 reuse, Windows state concurrency repair.

No changes to frozen v008. All new output isolated, previous NCAR bytes reused.
"""
from __future__ import annotations
import copy
import threading
import time
from pathlib import Path
import weather_download_top10_ncar_v008 as app

PREV8=app.DEST
DEST=app.EXTERNAL/'weather_top10_ncar_v009'
CODE=Path(__file__).resolve()
STATE_LOCK=threading.RLock()
raw_read=app.R;raw_atomic=app.io.atomic;raw_state=app.state;raw_save=app.save
raw_cache=app.load_cache;raw_progress=app.progress
app.DEST=DEST;app.io.DEST=DEST;app.CODE=CODE;app.PLAN=DEST/'plan.json'

def sharing_retry(call):
    for attempt in range(100):
        try:return call()
        except PermissionError:
            if attempt==99:raise
            time.sleep(.02)

def read(path):
    with STATE_LOCK:return sharing_retry(lambda:raw_read(path))

def atomic(path,obj):
    with STATE_LOCK:return sharing_retry(lambda:raw_atomic(path,obj))

def state(node):
    with STATE_LOCK:return raw_state(node)

def save(node,plan,**updates):
    with STATE_LOCK:return raw_save(node,plan,**updates)

def safe_error(exc):
    return {'error_type':type(exc).__name__,'message':str(exc)[:2000]}

app.R=read;app.io.read=read;app.io.atomic=atomic;app.state=state;app.save=save;app.io.safe_error=safe_error

def old_records(item):
    node=item['node'];record_path=app.ROOT/item['record_path']
    app.require(app.SHA(record_path)==item['records_sha256'],'Old NCAR records changed')
    record=app.verified_record(record_path)
    app.require(record['request_fingerprint']==app.FP(node),'Old NCAR node changed')
    app.require(app.SHA(app.ROOT/record['source'])==item['sha256']==record['source_sha256'],'Old NCAR payload changed')
    app.require(app.SHA(app.ROOT/record['metadata'])==record['metadata_sha256'],'Old NCAR metadata changed')
    rows,contract=app.transport.decode((app.ROOT/record['source']).read_text(encoding='utf-8'),(app.ROOT/record['metadata']).read_text(encoding='utf-8'),node)
    app.require(rows==record['records'] and contract==record['contract'],'Old NCAR canonical decoding mismatch')
    return record

def build_plan():
    app.require(not app.io.lock_held(),'Shared downloader lock occupied')
    app.require(app.SHA(PREV8/'plan.json')=='7ad663b779f04c6fb87e08feb30c6d7fc3e16063d3b658d659589371baf77fa1','Frozen v008 plan changed')
    old=app.verified_record(PREV8/'plan.json')
    for path,digest in old['source_sha256'].items():app.require(app.SHA(path)==digest,'Old frozen code changed')
    for path,digest in old['protected_files'].items():app.require(app.SHA(app.ROOT/path)==digest,'Old protected file changed')
    plan=copy.deepcopy(old);plan.pop('fingerprint');reusable=[]
    for p in sorted((PREV8/'jobs').glob('*.json')):
        s=read(p)
        app.require(s['status']=='VERIFIED_SUBSET','Unexpected unfinished v008 job; manual migration diagnosis needed')
        item={'node':s['job'],'source_state':str(p.relative_to(app.ROOT)),'state_sha256':app.SHA(p),'sha256':s['sha256'],
              'record_path':str((PREV8/'records'/(s['job']['id']+'.json')).relative_to(app.ROOT)),
              'records_sha256':s['records_sha256'],'values':s['values'],'response_bytes':s['receipt']['bytes']}
        old_records(item);reusable.append(item)
    protected={str(p.relative_to(app.ROOT)):app.SHA(p) for p in app.EXTERNAL.rglob('*') if p.is_file() and not p.is_relative_to(DEST) and p!=app.io.LOCK}
    protected.update(old['protected_files'])
    sources=dict(old['source_sha256'])
    for p in (CODE,app.HERE/'test_ncar_recovery_v002.py',app.HERE/'audit_ncar_top10_v002.py'):
        sources[str(p)]=app.SHA(p)
    plan.update(version='top10-ncar-v009',ncar_reusable=reusable,protected_files=protected,source_sha256=sources,
                predecessor_ncar_plan=str((PREV8/'plan.json').relative_to(app.ROOT)),predecessor_ncar_sha256=app.SHA(PREV8/'plan.json'),
                local_state_policy='Reentrant lock serializes state read/write; transient sharing errors retry up to 2 seconds; safe_error mapping matches frozen writer')
    return app.signed(plan)

def load_cache(plan,parent):
    cache,hashes=raw_cache(plan,parent);hours=set(app.core.parent_hours(parent));pressure='pressure_level' in parent['request']
    for item in plan['ncar_reusable']:
        if bool(item['node']['level_hpa'])!=pressure:continue
        if not any(k[2] in hours for k in app.transport.expected_keys(item['node'])):continue
        record=old_records(item)
        for key,value in record['records']:
            key=tuple(key)
            if key[2] in hours and key not in cache:cache[key]=(value,record['source'])
        hashes[record['source']]=record['source_sha256']
    return cache,hashes

def progress(plan,verify=False):
    result=raw_progress(plan,verify)
    if verify:
        for item in plan['ncar_reusable']:old_records(item)
    result['reused_completed_ncar_subsets']=len(plan['ncar_reusable'])
    result['reused_completed_ncar_grid_field_hours']=sum(x['values'] for x in plan['ncar_reusable'])
    result['total_verified_ncar_grid_field_hours_including_migrated']=result['verified_unique_grid_field_hours']+result['reused_completed_ncar_grid_field_hours']
    return result

app.build_plan=build_plan;app.load_cache=load_cache;app.progress=progress

def __getattr__(name):return getattr(app,name)

if __name__=='__main__':app.main()
