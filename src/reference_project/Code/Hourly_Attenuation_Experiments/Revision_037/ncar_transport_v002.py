"""Transport-only repair: allow slow exact pressure-level subset extraction.

Original 40-second read timeouts and every failed attempt remain preserved.
Same immutable request plan, fields, coordinates, dates, bytes cap and validator.
"""
from transport import *

class FetcherV2(Fetcher):
    def get(self,url,relative,validate):
        need(url.startswith(HOST) and not urlparse(url).username,'Public archive only')
        tail=url[len(HOST):]
        need(tail.startswith('dodsC/files/d633000/') and (tail.endswith('.das') or '.nc.ascii?' in tail),'Exact DAP only')
        need(not (OUT/'STOP_REQUESTED.json').exists(),'New batch stopped')
        p=ACQ/relative;receipt=p.with_name(p.name+'.receipt.json')
        if p.exists():
            r=read(receipt);need(r['url']==url and r['sha256']==sha(p),'Cached source identity')
            s=p.read_text(encoding='utf-8');return s,validate(s)
        need(not receipt.exists(),'Orphan receipt');p.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(1,3):
            ident=uuid.uuid4().hex;part=ACQ/'partial_history'/f'{p.name}.{ident}.part';part.parent.mkdir(parents=True,exist_ok=True)
            started=now();tic=time.monotonic()
            try:
                self.s.cookies.clear()
                with self.s.get(url,stream=True,allow_redirects=False,timeout=(10,180)) as r:
                    if r.status_code!=200:
                        emit(ACQ/'http_failures'/f'{ident}.json',dict(url=url,status=r.status_code,utc=now(),transport='v002'))
                        if (r.status_code in (408,429) or 500<=r.status_code<600) and attempt<2:
                            time.sleep(15);continue
                        raise RuntimeError('HTTP '+str(r.status_code)+' '+url)
                    count=0;length=r.headers.get('Content-Length')
                    with part.open('xb') as h:
                        for chunk in r.iter_content(65536):
                            count+=len(chunk);need(count<=20*1024**2 and time.monotonic()-tic<240,'Response byte/time limit');h.write(chunk)
                    need(count>0 and (length is None or count==int(length)),'Response completeness')
                    s=part.read_text(encoding='utf-8');v=validate(s);digest=sha(part);part.rename(p)
                    emit(receipt,dict(url=url,status=200,method='GET',started_utc=started,completed_utc=now(),bytes=count,
                        sha256=digest,content_length=None if length is None else int(length),source=rel(p),validation='PASS',attempt=attempt,
                        transport='v002',read_timeout_seconds=180))
                    print(json.dumps(dict(acquired=relative,bytes=count,utc=now())),flush=True);return s,v
            except requests.RequestException as e:
                emit(ACQ/'http_failures'/f'{ident}.json',dict(url=url,error=repr(e),utc=now(),attempt=attempt,transport='v002'))
                if attempt==2:raise
                time.sleep(15)
        raise RuntimeError('Bounded GET exhausted')

if __name__=='__main__':
    import ncar_inputs
    from cds_three import step
    with stage_lock('ncar_acquire_v002'):
        print('CDS',json.dumps(step()),flush=True)
        ncar_inputs.Fetcher=FetcherV2
        ncar_inputs.acquire(0)
