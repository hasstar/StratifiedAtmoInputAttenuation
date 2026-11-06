"""Explicit optional retrieval of new official requests. No old job recovery.

Default is a dry run. Downloads go to a new incoming folder, never over the
delivered read-only raw data. Credentials and transient URLs are not logged.
"""
from pathlib import Path
from urllib.parse import urlsplit
import argparse,contextlib,hashlib,json,logging,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['cds_local','cds_uk','ceda_days','ncar_exact'],required=True)
    ap.add_argument('--download',action='store_true');ap.add_argument('--destination',type=Path)
    a=ap.parse_args();jobs=s.read(Path(__file__).parent/'requests'/(a.source+'.json'))
    if not a.download:
        print(json.dumps(dict(source=a.source,requests=len(jobs),network_requests=0,mode='DRY_RUN')));return
    if a.destination is None:raise ValueError('--destination must name a new incoming folder')
    dest=a.destination.absolute();raw=(s.PACKAGE/'data/raw').absolute()
    if dest==raw or raw in dest.parents:raise ValueError('Keep the delivered data/raw read-only; choose a new incoming folder')
    dest.mkdir(parents=True,exist_ok=False)
    logging.disable(logging.CRITICAL)
    import requests
    client=None
    if a.source.startswith('cds_'):
        import cdsapi
        with open(os.devnull,'w')as null,contextlib.redirect_stdout(null),contextlib.redirect_stderr(null):
            client=cdsapi.Client(url='https://cds.climate.copernicus.eu/api',quiet=True,debug=False,progress=False)
    report=[]
    for i,job in enumerate(jobs):
        target=dest/job['target'];target.resolve().relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True)
        try:
            if client is not None:
                with open(os.devnull,'w')as null,contextlib.redirect_stdout(null),contextlib.redirect_stderr(null):
                    client.retrieve(job['dataset'],job['request'],str(target))
            else:
                u=urlsplit(job['url']);allowed=('dap.ceda.ac.uk','tds.gdex.ucar.edu','tds.rda.ucar.edu','gdex.ucar.edu')
                if u.scheme!='https' or u.hostname not in allowed or u.username:raise ValueError('Unapproved data host')
                headers={}
                if u.hostname=='dap.ceda.ac.uk' and os.environ.get('CEDA_ACCESS_TOKEN'):
                    headers['Authorization']='Bearer '+os.environ['CEDA_ACCESS_TOKEN']
                with requests.get(job['url'],headers=headers,stream=True,timeout=(30,1200))as r:
                    r.raise_for_status()
                    if urlsplit(r.url).hostname not in allowed:raise ValueError('Unexpected redirect host')
                    with target.open('xb')as f:
                        for block in r.iter_content(1024*1024):f.write(block)
            digest=s.sha(target)
            if 'sha256'in job and digest!=job['sha256']:raise ValueError('Downloaded payload differs from frozen source; inspect rather than replace')
            report.append(dict(id=job['id'],target=job['target'],bytes=target.stat().st_size,sha256=digest))
            print('DOWNLOADED',i+1,'/',len(jobs),job['id'],flush=True)
        except Exception as error:
            s.write(dest/'download_failure.json',dict(id=job['id'],error_type=type(error).__name__,message='Inspect the official service response privately. No credential, request identifier or signed URL was logged.'))
            raise SystemExit('Download stopped. See download_failure.json; no automatic retry.')from None
    s.write(dest/'download_receipts.json',report)
if __name__=='__main__':main()
