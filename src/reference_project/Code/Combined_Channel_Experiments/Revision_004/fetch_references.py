"""One-time bounded retrieval of four public ITU method references, not weather."""
from common import *
import time
import traceback
import zipfile
from urllib.parse import urlparse
import requests

FILES={
 'P839_4_components.zip':'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-I!!ZIP-E.zip',
 'P839_4.pdf':'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-I!!PDF-E.pdf',
 'P838_3.pdf':'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf',
 'P618_14.pdf':'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-14-202308-I!!PDF-E.pdf'}

def main():
    with lock():
        guard=snapshot();records=[]
        session=requests.Session()
        for name,url in FILES.items():
            response=None
            for attempt in range(2):
                try:
                    response=session.get(url,stream=True,timeout=(10,60),allow_redirects=False)
                    require(response.status_code==200,f'Standard GET {response.status_code}: {url}')
                    break
                except (requests.Timeout,requests.ConnectionError):
                    if attempt:raise
                    time.sleep(2)
            require(response is not None,'No response')
            length=response.headers.get('Content-Length');limit=8*1024*1024
            require(length is None or int(length)<=limit,'Reference size cap')
            p=newpath(OUT/'references'/name);size=0
            with p.open('xb') as f:
                for chunk in response.iter_content(65536):
                    size+=len(chunk);require(size<=limit,'Reference body exceeds size cap');f.write(chunk)
            response.close()
            require(length is None or size==int(length),'Reference Content-Length mismatch')
            require(size>1000,'Unexpected short reference')
            if name.endswith('.zip'):
                with zipfile.ZipFile(p) as z:
                    require(z.testzip() is None,'ZIP CRC failure')
                    members=[{'name':i.filename,'size':i.file_size,'crc':i.CRC} for i in z.infolist()]
                    print('ZIP',members,flush=True)
            else:
                require(p.read_bytes()[:5]==b'%PDF-','Not a PDF');members=None
            record={'file':name,'url':url,'status':200,'bytes':size,'content_length':int(length) if length else None,'sha256':sha(p),'retrieved_utc':now(),'zip_members':members}
            emit(OUT/'references'/(name+'.receipt.json'),record);records.append(record)
            print('REFERENCE',name,size,flush=True)
        protected_verify(guard)
        emit(OUT/'references_complete.json',{'status':'PASS','records':records,'weather_gets':0,'reference_gets':len(records),'code_sha256':sha(Path(__file__))})

if __name__=='__main__':
    try:main()
    except Exception:
        emit(OUT/('failure_reference_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{'time':now(),'traceback':traceback.format_exc()});raise
