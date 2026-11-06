"""New bounded anonymous archive reader; never invokes historical downloaders."""
from __future__ import annotations
import contextlib, json, os, time, uuid, msvcrt
from pathlib import Path
from urllib.parse import urlparse
import requests
from freeze import ROOT, OUT, CODE, read, sha, now, emit, need, rel

ACQ=OUT/'acquisition'
HOST='https://tds.gdex.ucar.edu/thredds/'

@contextlib.contextmanager
def stage_lock(stage):
    with (OUT/'execution.lock').open('a+b') as f:
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        token=uuid.uuid4().hex
        emit(OUT/'runs'/f'{stage}_{token}.json',dict(stage=stage,utc=now(),pid=os.getpid(),
            code_sha256={p.name:sha(p) for p in CODE.glob('*.py')}))
        try:yield
        except Exception as e:
            emit(OUT/'failures'/f'{stage}_{token}.json',dict(stage=stage,utc=now(),error=repr(e)));raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

class Fetcher:
    def __init__(self):
        self.s=requests.Session();self.s.trust_env=False
        self.s.headers.update({'User-Agent':'T015-frozen-RFE-bounded-inputs/037','Accept-Encoding':'identity'})
    def get(self,url,relative,validate):
        need(url.startswith(HOST) and not urlparse(url).username,'Public archive only')
        tail=url[len(HOST):]
        need((tail.startswith('catalog/files/g/d633000/') and tail.endswith('/catalog.xml')) or
             (tail.startswith('dodsC/files/d633000/') and (tail.endswith('.das') or '.nc.ascii?' in tail)),
             'Only metadata and constrained ASCII subsets allowed')
        need(not (OUT/'STOP_REQUESTED.json').exists(),'New batch stopped')
        p=ACQ/relative; receipt=p.with_name(p.name+'.receipt.json')
        if p.exists():
            r=read(receipt);need(r['url']==url and r['sha256']==sha(p),'Cached source identity')
            text=p.read_text(encoding='utf-8');return text,validate(text)
        need(not receipt.exists(),'Orphan receipt')
        p.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(1,4):
            ident=uuid.uuid4().hex;part=ACQ/'partial_history'/f'{p.name}.{ident}.part'
            part.parent.mkdir(parents=True,exist_ok=True); started=now();tic=time.monotonic()
            try:
                self.s.cookies.clear()
                with self.s.get(url,stream=True,allow_redirects=False,timeout=(10,40)) as response:
                    status=response.status_code
                    if status!=200:
                        emit(ACQ/'http_failures'/f'{ident}.json',dict(url=url,status=status,utc=now()))
                        if (status in (408,429) or 500<=status<600) and attempt<3:
                            time.sleep(min(10*attempt,30));continue
                        raise RuntimeError('HTTP '+str(status)+' '+url)
                    count=0;length=response.headers.get('Content-Length')
                    with part.open('xb') as f:
                        for chunk in response.iter_content(65536):
                            count+=len(chunk);need(count<=20*1024**2 and time.monotonic()-tic<70,'Response size/time limit')
                            f.write(chunk)
                    need(count>0 and (length is None or count==int(length)),'Response byte completeness')
                    text=part.read_text(encoding='utf-8');result=validate(text)
                    digest=sha(part);part.rename(p)
                    emit(receipt,dict(url=url,status=200,method='GET',started_utc=started,completed_utc=now(),
                        bytes=count,sha256=digest,content_length=None if length is None else int(length),
                        source=rel(p),validation='PASS',attempt=attempt))
                    print(json.dumps(dict(acquired=relative,bytes=count,utc=now())),flush=True)
                    return text,result
            except requests.RequestException as e:
                emit(ACQ/'http_failures'/f'{ident}.json',dict(url=url,error=repr(e),utc=now(),attempt=attempt))
                if attempt==3:raise
                time.sleep(5*attempt)
        raise RuntimeError('Bounded GET exhausted')
