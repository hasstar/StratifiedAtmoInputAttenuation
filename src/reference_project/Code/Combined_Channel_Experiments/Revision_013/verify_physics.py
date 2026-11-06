"""Independent scalar verification of diagnostic calculations (not attenuation)."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import netCDF4
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_013/attempt_002'
def read(n): return json.loads((OUT/n).read_text(encoding='utf-8'))
def need(v,m):
    if not v: raise AssertionError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    need(not (OUT/'physics_verification.json').exists(),'one shot')
    want=['u','v','z','avg_tprate','avg_tsrwe','avg_cpr','avg_lsprate']
    pieces={k:[] for k in want}
    for rec in read('file_inventory.json'):
        keys=[k for k in rec['variables'] if k in want]
        if not keys: continue
        with netCDF4.Dataset(rec['path']) as ds:
            for k in keys:
                pieces[k].append((rec['calendar_utc'][0]['start_utc'],ds.variables[k][:].astype(float)))
    a={k:np.concatenate([v for _,v in sorted(pieces[k],key=lambda z:z[0])]) for k in want}
    # Full raw coverage starts 2025-09-29T00Z; campaign starts at +16h.
    now=slice(16,16+7800); valid_next=slice(17,17+7800)
    wind=[math.sqrt(float(u)*float(u)+float(v)*float(v)) for u,v in zip(a['u'][now,3].flat,a['v'][now,3].flat)]
    height=[float(x)/9.80665 for x in a['z'][now].flat]
    liquid=[3600*(float(x)-float(y)) for x,y in zip(a['avg_tprate'][valid_next].flat,a['avg_tsrwe'][valid_next].flat)]
    partition=[3600*(float(x)+float(y)-float(z)) for x,y,z in zip(a['avg_cpr'][valid_next].flat,a['avg_lsprate'][valid_next].flat,a['avg_tprate'][valid_next].flat)]
    g=read('geometry_diagnostics.json')
    max_error=0.
    for key,values in [('wind700_m_per_s',wind),('geopotential_height_m',height),('liquid_total_minus_snow_mm_hour',liquid),('convective_plus_large_scale_minus_total_mm_hour',partition)]:
        measured={'size':len(values),'finite':sum(math.isfinite(x) for x in values),
                  'nan':sum(math.isnan(x) for x in values),'inf':sum(math.isinf(x) for x in values),
                  'negative':sum(x<0 for x in values),'zero':sum(x==0 for x in values),
                  'minimum':min(values),'maximum':max(values)}
        for k,v in measured.items():
            if k in ['minimum','maximum']:
                e=abs(v-g[key][k]);max_error=max(e,max_error)
                need(e<1e-11,key+':'+k)
            else: need(v==g[key][k],key+':'+k)
    need(sum(abs(x)>1e-5 for x in partition)==g['partition_abs_error_gt_1e_minus_5_mm_hour'],'partition tolerance count')
    for p,h in read('raw_sources_before.json').items(): need(sha(p)==h,'raw source changed')
    report={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
            'scope':'independent scalar wind/height/rate-difference diagnostic checks, not model output',
            'max_extrema_difference':max_error,'diagnostic_sets':4,
            'weather_requests':0,'attenuation_predictions':0,'source_shas_unchanged':True,
            'script_sha256':sha(__file__),'geometry_diagnostics_sha256':sha(OUT/'geometry_diagnostics.json')}
    with (OUT/'physics_verification.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
