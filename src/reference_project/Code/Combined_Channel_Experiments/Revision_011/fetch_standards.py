"""Bounded anonymous retrieval of exact official reference URLs; no weather GETs."""
from support import *
import requests
import traceback
from urllib.parse import urlparse

SOURCES = {
 'P676_13.pdf': 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-13-202208-I!!PDF-E.pdf',
 'P676_13_part1.txt': 'https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020001TXTM.txt',
 'P676_13_part2.txt': 'https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020002TXTM.txt',
 'P840_9.pdf': 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-9-202308-I!!PDF-E.pdf',
 'P835_6.pdf': 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.835-6-201712-S!!PDF-E.pdf',
 'P453_14.pdf': 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-14-201908-I!!PDF-E.pdf',
}

def main():
    initialize()
    with lock():
        session = requests.Session()
        session.trust_env = False  # anonymous, no ambient .netrc credentials
        rows=[]
        for name,url in SOURCES.items():
            target = new(OUT/'references'/name)
            part = new(target.with_suffix(target.suffix+'.part'))
            try:
                with session.get(url,timeout=(10,90),stream=True,allow_redirects=False) as r:
                    need(r.status_code==200, f'{name}: HTTP {r.status_code}; no redirect/auth bypass')
                    size=0
                    with part.open('xb') as f:
                        for b in r.iter_content(65536):
                            size+=len(b); need(size<=16*1024*1024,'Reference exceeds size limit')
                            f.write(b)
                    advertised=r.headers.get('Content-Length')
                    if advertised and not r.headers.get('Content-Encoding'):
                        need(size==int(advertised), 'Content-Length mismatch')
                    if name.endswith('.pdf'):
                        with part.open('rb') as f: need(f.read(5)==b'%PDF-', 'Not a PDF')
                    row=dict(name=name,url=url,final_url=r.url,utc=utc(),http_status=r.status_code,
                             actual_bytes=size,content_length=advertised,headers=dict(r.headers),sha256=sha(part))
                part.rename(target)
                emit('references/'+name+'.receipt.json',row)
                rows.append(row); print(name,size,flush=True)
            except Exception:
                emit('failures/'+stamp()+'_'+name+'.json',dict(url=url,error=traceback.format_exc(),partial=str(part)))
                raise
        check(read(OUT/'sources_before.json'))
        emit('references_complete.json',dict(status='PASS',records=rows))

if __name__=='__main__': main()
