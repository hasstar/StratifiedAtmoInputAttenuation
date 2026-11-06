"""Acquire actual official field headers before building bounded value requests."""
import re
from metadata import catalog
from transport import *

SPECS={
 'e5.oper.an.pl':['128_203_o3','128_157_r','128_133_q','128_130_t','128_129_z','128_060_pv','128_075_crwc'],
 'e5.oper.an.sfc':['128_187_mcc','128_079_tciw','128_206_tco3'],
 'e5.oper.an.vinteg':['162_071_viwve','162_077_vioze','162_078_viozn','162_089_vilwn']}
def main():
    with stage_lock('ncar_field_metadata'):
        idx=read(ACQ/'catalog_index.json');f=Fetcher();samples={}
        for group,specs in SPECS.items():
            entries=(catalog((ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/catalogs/200308.xml').read_text())
                     if group=='e5.oper.an.sfc' else idx[group+'/200308'])
            for spec in specs:
                entries_for_field=sorted([e for e in entries if '.'+spec+'.ll025sc.' in e['name']],key=lambda e:e['name'])
                need(bool(entries_for_field),'Missing catalog field '+spec)
                e=entries_for_field[0];url=HOST+'dodsC/'+e['urlPath']+'.das'
                text,_=f.get(url,'metadata_samples/'+spec+'.das',lambda s:need('Attributes {' in s,'Not DAS'))
                blocks=dict(re.findall(r'\b([A-Za-z_]\w*)\s*\{([^{}]*)\}',text,re.S))
                block=[(n,b) for n,b in blocks.items() if 'ecmwf_parameter' in b]
                need(len(block)==1,'Unique field metadata');n,b=block[0]
                samples[spec]=dict(url_path=e['urlPath'],variable=n,attributes=b)
                print(spec,n,b,flush=True)
        emit(ACQ/'metadata_samples/index.json',samples)
if __name__=='__main__':main()
