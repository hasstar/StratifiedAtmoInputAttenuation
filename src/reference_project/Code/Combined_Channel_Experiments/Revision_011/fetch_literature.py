"""Bounded, anonymous exact-source paper retrieval; no browser credentials."""
from support import *
import requests
import traceback

SOURCES = {
    'NASA_RP1082_1989.pdf':'https://ntrs.nasa.gov/api/citations/19890007689/downloads/19890007689.pdf',
    'Biscarini_2024_author.pdf':'https://iris.uniroma1.it/retrieve/954aaa86-90cf-46d8-a92e-df3e0be080e0/Biscarini_Model-chain_2024.pdf',
    'AlSaegh_2025.pdf':'https://www.nature.com/articles/s41598-025-20199-z.pdf',
}

def main():
    with lock():
        check(read(OUT/'sources_before.json'))
        session=requests.Session(); session.trust_env=False
        records=[]
        for name,url in SOURCES.items():
            dest=new(OUT/'literature'/name)
            part=new(dest.with_suffix('.pdf.part'))
            row=dict(name=name,url=url,utc=utc())
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
            emit('literature/'+name+'.receipt.json',row)
            records.append(row); print(name,row['status'],row.get('actual_bytes'),row.get('http_status'),flush=True)
        check(read(OUT/'sources_before.json'))
        emit('literature_download_manifest.json',dict(utc=utc(),records=records,
             supplementary_material_user_authorized=True,
             supplementary_status='No separate reproduction-related SI link found on the inspected source pages; official ITU Part1/Part2 coefficient supplements already saved. This is not a claim that no SI exists elsewhere.',
             browser_access='Chrome known IEEE tab could not attach debugger; no browser credentials accessed'))

if __name__=='__main__': main()
