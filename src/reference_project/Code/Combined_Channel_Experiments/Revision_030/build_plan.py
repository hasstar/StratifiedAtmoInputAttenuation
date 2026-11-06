"""Offline authorization-plan builder. No requests, credentials or old imports."""
import hashlib
import json
import math
import msvcrt
import os
import platform
import socket
import sys
import traceback
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_030'
OLD = ROOT/'Outputs/Experiments/Combined_Channel_Revision_029'
R25 = ROOT/'Outputs/Experiments/Combined_Channel_Revision_025'
LEVELS = [1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,50,30,20,10,7,5,3,2,1]
GRID = {'chilbolton':(51.25,-1.5),'sparsholt':(51.0,-1.5)}

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def emit(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def deny(*a,**k): raise RuntimeError('No network in Revision030')
def checks(d):
    assert all(sha(p)==h for p,h in d.items()), 'Protected source changed'

def main():
    socket.socket.connect=deny; socket.create_connection=deny
    assert not OUT.exists(), 'One-shot new directory only'
    guard=read(OLD/'sources_before.json')
    assert len(guard)==4991
    assert sha(OLD/'delivery_manifest.json')==read(OLD/'final_verification.json')['manifest_sha256']
    assert read(OLD/'final_verification.json')['execution_lock']=='RELEASED'
    for p,m in read(OLD/'delivery_manifest.json')['files'].items():
        assert sha(ROOT/p)==m['sha256']; guard[str(ROOT/p)]=m['sha256']
    extra=[OLD/'delivery_manifest.json',OLD/'final_verification.json',
           ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/field_provenance.parquet']
    for p in extra: guard[str(p)]=sha(p)
    checks(guard)
    OUT.mkdir()
    with (OUT/'PROTOCOL.md').open('x',encoding='utf-8') as f:
        f.write((CODE/'PROTOCOL.md').read_text(encoding='utf-8'))
    emit('sources_before.json',guard)
    emit('code_before.json',{str(p):sha(p) for p in CODE.iterdir() if p.is_file()})
    emit('execution.json',{'utc':now(),'pid':os.getpid(),'command':sys.argv,'executable':sys.executable,
        'python':sys.version,'platform':platform.platform(),'netCDF4':netCDF4.__version__,'numpy':np.__version__,
        'weather_requests':0,'new_predictions':0,'training':0,'new_scores':0,'network_disabled':True})
    (OUT/'execution.lock').write_bytes(b'0')
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            ledger=read(R25/'hourly_input_ledger.json')
            keys={(q['station'],q['utc_epoch_seconds']) for q in ledger}
            assert len(keys)==len(ledger)==5664
            assert Counter(s for s,t in keys)=={'chilbolton':3432,'sparsholt':2232}
            rain=read(R25/'rain_context.json')
            assert len(rain)==7152
            for q in rain:
                assert (q['grid_latitude_quarter']/4,q['grid_longitude_quarter']/4)==GRID[q['station']]
                assert q['valid_end_epoch_seconds']==q['utc_epoch_seconds']+3600
            pressure=[q for q in pq.read_table(ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/field_provenance.parquet').to_pylist() if q['field']=='sp']
            assert {(q['station'],q['utc_epoch_seconds']) for q in pressure}==keys
            assert len(pressure)==5664
            for q in pressure:
                assert q['units']=='Pa' and math.isfinite(q['value']) and q['value']>0
                assert q['utc_epoch_seconds']==q['valid_utc_epoch_seconds']
                assert (q['latitude'],q['longitude_0_360']-360)==GRID[q['station']]
            reuse={}; candidates=[]
            for item in read(R25/'cds_headers.json'):
                for member in item['members']:
                    if 'z' not in member['variables'] or 'pressure_level' not in member['variables']['z']['dimensions']: continue
                    assert member['member'] is None, 'Unexpected zipped pressure candidate'
                    p=ROOT/item['path']; assert sha(p)==item['sha256']
                    candidates.append(item['path'])
                    with netCDF4.Dataset(str(p)) as ds:
                        v=ds['z']
                        assert v.dimensions==('valid_time','pressure_level','latitude','longitude')
                        assert v.units=='m**2 s**-2' and v.GRIB_paramId==129
                        assert v.GRIB_dataType=='an' and v.GRIB_stepType=='instant'
                        assert ds['valid_time'].units=='seconds since 1970-01-01'
                        lat=ds['latitude'][:].tolist(); lon=ds['longitude'][:].tolist()
                        for s,(y,x) in GRID.items():
                            if y not in lat or x not in lon: continue
                            for ti,t0 in enumerate(ds['valid_time'][:]):
                                t=int(t0)
                                if (s,t) not in keys: continue
                                for pi,p0 in enumerate(ds['pressure_level'][:]):
                                    level=int(p0)
                                    if level not in LEVELS: continue
                                    value=v[ti,pi,lat.index(y),lon.index(x)]
                                    if np.ma.is_masked(value) or not math.isfinite(float(value)): continue
                                    k=(s,t,level); value=float(value)
                                    if k in reuse: assert reuse[k]['value']==value, 'Conflicting old z values'
                                    else: reuse[k]={'station':s,'utc_epoch_seconds':t,'level_hpa':level,'value':value,'sources':[]}
                                    reuse[k]['sources'].append(item['path'])
            ncar_examined=0; ncar_matched=[]
            for item in read(R25/'ncar_record_inventory.json'):
                if item['variable'] not in ('geopotential','temperature','u_component_of_wind','v_component_of_wind'): continue
                ncar_examined+=1; obj=read(ROOT/item['path'])
                for k,value in obj['records']:
                    for s,(y,x) in GRID.items():
                        if (s,k[2]) not in keys or (k[3]/4,k[4]/4)!=(y,x): continue
                        if (k[0] in ('geopotential','temperature') and k[1] in LEVELS) or (k[0] in ('u_component_of_wind','v_component_of_wind') and k[1]==700):
                            ncar_matched.append([item['path'],k,value])
            assert not ncar_matched, 'Additional local reuse must be incorporated before requesting'
            assert len(reuse)==9360
            # Group exact same variable/level missing sets by station/month/days.
            groups=defaultdict(list)
            for s in GRID:
                times=sorted(t for st,t in keys if st==s)
                for t in times:
                    dt=datetime.fromtimestamp(t,timezone.utc)
                    missing_z=tuple(p for p in LEVELS if (s,t,p) not in reuse)
                    for variables,levels in [(('temperature',),tuple(LEVELS)),(('geopotential',),missing_z),(('u_component_of_wind','v_component_of_wind'),(700,))]:
                        if levels: groups[(s,dt.year,dt.month,variables,levels)].append(t)
            batches=[]
            for (s,year,month,variables,levels),times in sorted(groups.items()):
                days=sorted({datetime.fromtimestamp(t,timezone.utc).day for t in times})
                expected=[int(datetime(year,month,day,h,tzinfo=timezone.utc).timestamp()) for day in days for h in range(24)]
                assert times==expected, 'Batch would add or omit UTC hours'
                batches.append({'id':len(batches)+1,'station':s,'dataset':'reanalysis-era5-pressure-levels',
                    'product_type':'reanalysis','year':year,'month':month,'days':days,'hours_utc':list(range(24)),
                    'grid_latitude':GRID[s][0],'grid_longitude':GRID[s][1],
                    'variables':list(variables),'pressure_levels_hpa':list(levels),
                    'scalar_values':len(times)*len(variables)*len(levels),'authorized':False})
            requested=sum(b['scalar_values'] for b in batches)
            assert requested==421104 and len(batches)==30
            emit('scope_hours.json',[{'station':s,'utc_epoch_seconds':t} for s,t in sorted(keys)])
            emit('reused_geopotential.json',[reuse[k] for k in sorted(reuse)])
            emit('input_batches.json',batches)
            emit('reuse_check.json',{'cds_candidate_files':candidates,'ncar_relevant_records_examined':ncar_examined,
                'ncar_target_matches':ncar_matched,'reused_sp_hours':len(pressure),'existing_rain_context_hours':len(rain),
                'additional_rain_hours_requested':0,'old_rain_and_sp_grid_matched':True})
            summary={'status':'PROPOSED_NOT_AUTHORIZED','utc':now(),'station_days':236,'station_hours':len(keys),
                'levels_hpa':LEVELS,'required_t_values':5664*37,'required_z_values':5664*37,
                'required_wind_values':5664*2,'required_values':5664*76,'reused_z_values':len(reuse),
                'new_values_requested':requested,'batches':len(batches),'float32_numeric_bytes':requested*4,
                'wire_bytes_known':False,'proposed_total_transfer_cap_bytes':256*1024*1024,
                'new_context_hours_requested':0,'weather_requests':0,'new_predictions':0,'new_scores':0,
                'uk_adaptation_authorized':False,'mts_status':'NOT_COMPUTED_USER_DECISION','all_seven_complete':False,
                'by_station':{s:{'hours':sum(st==s for st,t in keys),'reused_z':sum(k[0]==s for k in reuse),
                    'new_values':sum(b['scalar_values'] for b in batches if b['station']==s)} for s in GRID}}
            emit('summary.json',summary)
            checks(guard)
            emit('build_complete.json',{'status':'PASS_PLAN_BUILD_ONLY','utc':now(),'sources_unchanged':len(guard)})
            print(json.dumps(summary,ensure_ascii=False))
        except BaseException:
            emit('build_failure.json',{'utc':now(),'traceback':traceback.format_exc()}); raise
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__': main()
