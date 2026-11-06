"""Only three explicitly authorized CDS inputs, only newly registered requests."""
from __future__ import annotations
import sys, json, logging, re, uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from freeze import ROOT, OUT, CODE, read, sha, now, emit, need, rel
from transport import stage_lock

DEST=OUT/'acquisition/cds_three'
VARIABLES=['k_index','total_totals_index','precipitation_type']
COLLECTION='reanalysis-era5-single-levels'

def fingerprint(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def plan():
    p=DEST/'plan.json'
    if p.exists(): return read(p)
    old=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/plan.json')
    jobs=[]
    for n in old['nodes']:
        dates=sorted({datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%d') for t in n['epochs']})
        need(len(n['epochs'])==24*len(dates),'Full existing UTC days required')
        lat=n['latitude'];lon=(n['longitude_0_360']+180)%360-180
        request=dict(product_type=['reanalysis'],variable=VARIABLES,
            year=[dates[0][:4]],month=[dates[0][5:7]],day=[d[8:10] for d in dates],
            time=[f'{h:02d}:00' for h in range(24)],area=[lat,lon,lat,lon],
            data_format='netcdf',download_format='unarchived')
        jobs.append(dict(id=n['id'],station=n['station'],epochs=n['epochs'],latitude=lat,
                         longitude=lon,collection_id=COLLECTION,request=request,
                         request_sha256=fingerprint(request)))
    need(len(jobs)==12 and sum(len(j['epochs']) for j in jobs)==5664,'Exact CDS cohort')
    result=dict(created_utc=now(),authorization='Explicit user CDS approval for these three fields only',
        code_sha256=sha(Path(__file__)),jobs=jobs,old_remote_jobs_accessed=False)
    emit(p,result);return result

def client():
    import yaml
    cfg=yaml.safe_load((Path.home()/'.cdsapirc').read_text(encoding='utf-8'))
    need(isinstance(cfg,dict) and cfg.get('key'),'Existing credential unavailable')
    endpoint=str(cfg.get('url','https://cds.climate.copernicus.eu/api')).rstrip('/')
    need(urlparse(endpoint).scheme=='https' and urlparse(endpoint).hostname=='cds.climate.copernicus.eu',
         'Credential endpoint is not official CDS')
    if endpoint.endswith('/api/v2'):endpoint=endpoint[:-3]
    sys.path.insert(0,str(ROOT/'Code/Dataset_Build/_vendor_cdsapi'))
    from ecmwf.datastores import Client
    logging.disable(logging.CRITICAL)
    return Client(url=endpoint,key=str(cfg['key']),verify=True,timeout=45,progress=False,
        sleep_max=30,retry_after=30,maximum_tries=1,cleanup=False)

def safe_error(e):
    # Exception content may include auth headers or signed URLs. Do not retain it.
    return dict(error_type=type(e).__name__,message='CDS operation failed; no raw exception retained to protect authentication material')

def step():
    """One bounded state transition; never lists or accesses a pre-existing job."""
    need(not (OUT/'STOP_REQUESTED.json').exists(),'New acquisition stopped')
    p=plan(); active=[]; next_job=None; done=0
    for j in p['jobs']:
        folder=DEST/'jobs'/j['id'];done_path=folder/'download.json'
        if done_path.exists():
            r=read(done_path);need(sha(DEST/r['file'])==r['sha256'],'CDS cached bytes changed');done+=1;continue
        if (folder/'intent.json').exists():active.append(j)
        elif next_job is None:next_job=j
    need(len(active)<=1,'More than one new unresolved CDS job')
    if done==len(p['jobs']):return dict(status='ALL_DOWNLOADED_NOT_YET_INPUT_QA',done=done,total=12)
    j=active[0] if active else next_job;folder=DEST/'jobs'/j['id']
    if (folder/'failure.json').exists():raise RuntimeError('Recorded CDS failure requires inspection: '+j['id'])
    if active and not (folder/'submitted.json').exists():
        raise RuntimeError('Unacknowledged submission intent; no automatic POST retry: '+j['id'])
    c=client()
    if not active:
        emit(folder/'intent.json',dict(utc=now(),request=j['request'],request_sha256=j['request_sha256'],
                                     collection_id=COLLECTION,plan_sha256=sha(DEST/'plan.json')))
        try:
            remote=c.submit(COLLECTION,j['request'])
            ident=str(remote.request_id)
            need(re.fullmatch(r'[A-Za-z0-9-]{8,100}',ident) is not None,'Invalid new request ID')
            emit(folder/'submitted.json',dict(utc=now(),request_id=ident,request_sha256=j['request_sha256']))
            return dict(status='SUBMITTED_NEW_BOUNDED_JOB',job=j['id'],done=done,total=12)
        except Exception as e:
            emit(folder/'failure.json',dict(utc=now(),phase='SUBMIT',**safe_error(e)))
            raise RuntimeError('CDS submit failed or response unknown; no blind retry') from None
    saved=read(folder/'submitted.json');intent=read(folder/'intent.json')
    need(saved['request_sha256']==intent['request_sha256']==j['request_sha256'],'New request identity mismatch')
    try:
        remote=c.get_remote(saved['request_id'])
        need(remote.collection_id==COLLECTION,'New remote collection mismatch')
        status=str(remote.status)
        emit(folder/'observations'/f'{uuid.uuid4().hex}.json',dict(utc=now(),status=status))
        if status in ('failed','dismissed','deleted','cancelled'):
            emit(folder/'failure.json',dict(utc=now(),phase='REMOTE',status=status))
            raise RuntimeError('New bounded CDS request unsuccessful')
        if status!='successful':return dict(status=status,job=j['id'],done=done,total=12)
        part=folder/f'payload_{uuid.uuid4().hex}.download'
        remote.download(str(part))
        need(part.is_file() and part.stat().st_size>0,'Empty CDS payload')
        emit(folder/'download.json',dict(utc=now(),file=str(part.relative_to(DEST)),sha256=sha(part),
             bytes=part.stat().st_size,request_sha256=j['request_sha256'],request_id=saved['request_id']))
        return dict(status='DOWNLOADED_NOT_YET_INPUT_QA',job=j['id'],done=done+1,total=12)
    except Exception as e:
        emit(folder/'observations'/f'error_{uuid.uuid4().hex}.json',dict(utc=now(),phase='POLL_OR_DOWNLOAD',**safe_error(e)))
        raise RuntimeError('CDS state/download failed; saved new ID preserved; raw error suppressed') from None

if __name__=='__main__':
    with stage_lock('cds_three'):
        print(json.dumps(step(),ensure_ascii=False),flush=True)
