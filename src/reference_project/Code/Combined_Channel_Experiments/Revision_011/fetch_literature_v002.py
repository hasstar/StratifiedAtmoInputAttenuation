"""Second isolated attempt after sandbox DNS failure; preserve first receipts."""
from support import *
from fetch_literature import SOURCES
import requests
import traceback

def main():
    with lock():
        check(read(OUT/'sources_before.json'))
        session=requests.Session(); session.trust_env=False
        records=[]
        for name,url in SOURCES.items():
            dest=new(OUT/'literature/attempt002'/name)
            part=new(dest.with_suffix('.pdf.part'))
            row=dict(name=name,url=url,utc=utc(),prior_failure='sandbox DNS; retained in first receipt')
            try:
                with session.get(url,timeout=(10,90),stream=True,allow_redirects=False) as r:
                    row.update(http_status=r.status_code,headers=dict(r.headers),final_url=r.url)
                    need(r.status_code==200,'HTTP '+str(r.status_code)+'; no authentication/redirect workaround attempted')
                    size=0
                    with part.open('xb') as f:
                        for block in r.iter_content(65536):
                            size+=len(block)
                            need(size<=32*1024*1024,'32 MiB document cap')
                            f.write(block)
                    if r.headers.get('Content-Length') and not r.headers.get('Content-Encoding'):
                        need(size==int(r.headers['Content-Length']),'Content-Length mismatch')
                    with part.open('rb') as f: need(f.read(5)==b'%PDF-','Not a PDF')
                    row.update(status='SAVED_PDF_PENDING_CONTENT_QA',actual_bytes=size,sha256=sha(part))
                part.rename(dest)
            except Exception:
                row.update(status='NOT_SAVED',error=traceback.format_exc(),partial_retained=part.exists())
            emit('literature/attempt002/'+name+'.receipt.json',row)
            records.append(row); print(name,row['status'],row.get('actual_bytes'),row.get('http_status'),flush=True)
        check(read(OUT/'sources_before.json'))
        emit('literature_download_manifest_v002.json',dict(utc=utc(),records=records,
             supplementary_material_user_authorized=True,
             supplementary_status='No separate reproduction-related SI link found on the inspected source pages. Official ITU Part1/Part2 coefficient supplements already saved; not a claim that no SI exists elsewhere.'))

if __name__=='__main__': main()
