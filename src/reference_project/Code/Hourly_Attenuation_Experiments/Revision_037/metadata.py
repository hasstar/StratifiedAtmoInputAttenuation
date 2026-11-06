"""Discover missing-field files from actual public catalogs, save raw evidence."""
import sys, xml.etree.ElementTree as ET
from transport import *

MONTHS=['200308','200309','200310','200311','200402','200403']
GROUPS=['e5.oper.an.vinteg','e5.oper.fc.sfc.instan','e5.oper.an.pl']
def catalog(text):
    tree=ET.fromstring(text)
    result=[dict(x.attrib) for x in tree.iter() if x.tag.endswith('dataset') and x.attrib.get('urlPath')]
    need(bool(result),'Empty official catalog')
    need(all(x['urlPath'].startswith('files/d633000/') for x in result),'Unexpected catalog root')
    return result
def main():
    with stage_lock('catalogs'):
        f=Fetcher();index={}
        for group in GROUPS:
            for month in MONTHS:
                url=HOST+f'catalog/files/g/d633000/{group}/{month}/catalog.xml'
                _,entries=f.get(url,f'catalogs/{group}_{month}.xml',catalog)
                index[group+'/'+month]=entries
                if month=='200308':print(group,sorted({e['name'].split('.')[4] for e in entries}),flush=True)
        emit(ACQ/'catalog_index.json',index)
        print('CATALOG_DISCOVERY_PASS',flush=True)
if __name__=='__main__':main()
