"""Independent offline set algebra and HDF5 value audit; no builder import."""
import calendar
import hashlib
import json
import math
import msvcrt
import os
import socket
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import h5py
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_030'
R25=ROOT/'Outputs/Experiments/Combined_Channel_Revision_025'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def deny(*a,**k):raise RuntimeError('Network prohibited')
def norm(x):return x.decode() if isinstance(x,bytes) else str(x)

def main():
    socket.socket.connect=deny;socket.create_connection=deny
    emit('qa_claim.json',{'utc':now(),'pid':os.getpid(),'argv':sys.argv,'h5py':h5py.__version__})
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            native=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/cohort_manifest.json')['native_days']
            assert len(native)==236
            keys=set()
            for row in native:
                y,m,d=map(int,row['date_utc'].split('-'))
                for hour in range(24):keys.add((row['station'],calendar.timegm((y,m,d,hour,0,0))))
            got={(x['station'],x['utc_epoch_seconds']) for x in read(OUT/'scope_hours.json')}
            assert got==keys and len(keys)==5664
            levels={1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000}
            grid={}
            for q in read(R25/'native_header_geometry.json'):
                g=(math.floor(q['latitude']*4+0.5)/4,math.floor((q['longitude']-360)*4+0.5)/4)
                if q['station'] in grid:assert grid[q['station']]==g
                grid[q['station']]=g
            assert grid=={'chilbolton':(51.25,-1.5),'sparsholt':(51.0,-1.5)}
            existing={}; occurrences=0; candidate_files=[]
            for q in read(R25/'cds_headers.json'):
                for m in q['members']:
                    # Detect new T/wind opportunities, then read every pressure z candidate independently.
                    assert 't' not in m['variables'] and 'v' not in m['variables']
                    if 'u' in m['variables']:assert 700 not in m['coordinates'].get('pressure_level',[])
                    if 'z' not in m['variables'] or 'pressure_level' not in m['variables']['z']['dimensions']:continue
                    assert m['member'] is None
                    path=ROOT/q['path'];assert sha(path)==q['sha256'];candidate_files.append(q['path'])
                    with h5py.File(path,'r') as f:
                        z=f['z'];assert norm(z.attrs['units'])=='m**2 s**-2'
                        assert norm(z.attrs['GRIB_dataType'])=='an' and norm(z.attrs['GRIB_stepType'])=='instant'
                        assert int(z.attrs['GRIB_paramId'][0])==129
                        lat=f['latitude'][:].tolist();lon=f['longitude'][:].tolist();times=f['valid_time'][:].tolist();ps=f['pressure_level'][:].tolist()
                        assert z.shape==(len(times),len(ps),len(lat),len(lon))
                        for yi,y in enumerate(lat):
                            for xi,x in enumerate(lon):
                                stations=[s for s,g in grid.items() if g==(y,x)]
                                for pi,p in enumerate(ps):
                                    if int(p) not in levels:continue
                                    vals=z[:,pi,yi,xi].tolist()
                                    for t,v in zip(times,vals,strict=True):
                                        if not math.isfinite(v):continue
                                        for s in stations:
                                            if (s,int(t)) not in keys:continue
                                            k=(s,int(t),'geopotential',int(p));occurrences+=1
                                            if k in existing:assert existing[k]==v
                                            existing[k]=v
            supplied={};count_sources=0
            for q in read(OUT/'reused_geopotential.json'):
                k=(q['station'],q['utc_epoch_seconds'],'geopotential',q['level_hpa'])
                assert k not in supplied;supplied[k]=q['value'];count_sources+=len(q['sources'])
            assert supplied==existing and count_sources==occurrences
            # NCAR canonical records have no additional requested coordinates/levels.
            ncar_relevant=0
            for q in read(R25/'ncar_record_inventory.json'):
                if q['variable'] not in ('geopotential','temperature','u_component_of_wind','v_component_of_wind'):continue
                ncar_relevant+=1
                for k,v in read(ROOT/q['path'])['records']:
                    wanted=(k[0] in ('geopotential','temperature') and k[1] in levels) or (k[0] in ('u_component_of_wind','v_component_of_wind') and k[1]==700)
                    if wanted:assert not any((s,k[2]) in keys and g==(k[3]/4,k[4]/4) for s,g in grid.items())
            needed={(s,t,v,p) for s,t in keys for v in ('temperature','geopotential') for p in levels}
            needed.update((s,t,v,700) for s,t in keys for v in ('u_component_of_wind','v_component_of_wind'))
            requested=set();batches=read(OUT/'input_batches.json')
            for b in batches:
                assert b['authorized'] is False and b['hours_utc']==list(range(24))
                assert (b['grid_latitude'],b['grid_longitude'])==grid[b['station']]
                entries=set()
                for day in b['days']:
                    for hour in b['hours_utc']:
                        t=calendar.timegm((b['year'],b['month'],day,hour,0,0))
                        for v in b['variables']:
                            for p in b['pressure_levels_hpa']:entries.add((b['station'],t,v,p))
                assert len(entries)==b['scalar_values'] and not requested.intersection(entries)
                requested.update(entries)
            assert requested==needed-set(existing)
            assert len(needed)==430464 and len(requested)==421104 and len(existing)==9360
            sp=[q for q in pq.read_table(ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/field_provenance.parquet').to_pylist() if q['field']=='sp']
            assert {(q['station'],q['utc_epoch_seconds']) for q in sp}==keys and len(sp)==5664
            for q in sp:
                assert q['units']=='Pa' and math.isfinite(q['value']) and q['value']>0
                assert q['valid_utc_epoch_seconds']==q['utc_epoch_seconds']
                assert (q['latitude'],(q['longitude_0_360']+180)%360-180)==grid[q['station']]
            rain=read(R25/'rain_context.json')
            assert len(rain)==7152
            for q in rain:
                assert (q['grid_latitude_quarter'],q['grid_longitude_quarter'])==tuple(int(4*x) for x in grid[q['station']])
                assert q['valid_end_epoch_seconds']-q['utc_epoch_seconds']==3600
            summary=read(OUT/'summary.json')
            assert summary['new_values_requested']==len(requested)
            assert set(summary['levels_hpa'])==levels and len(summary['levels_hpa'])==37
            assert summary['float32_numeric_bytes']==len(requested)*4
            for s in grid:
                assert summary['by_station'][s]['new_values']==sum(k[0]==s for k in requested)
            guards=read(OUT/'sources_before.json');code=read(OUT/'code_before.json')
            for p,h in {**guards,**code}.items():assert sha(p)==h
            emit('independent_qa.json',{'status':'PASS_PLAN_ONLY','utc':now(),'reviewer':'same agent, independent implementation; not external review',
                'native_dates':len(native),'hours':len(keys),'required_values':len(needed),'reused_z':len(existing),
                'checked_reuse_occurrences':occurrences,'new_values':len(requested),'batches':len(batches),
                'candidate_files':len(candidate_files),'ncar_relevant_records':ncar_relevant,'sp_hours':len(sp),
                'rain_context_hours':len(rain),'sources_unchanged':len(guards),'code_unchanged':len(code),
                'weather_requests':0,'new_predictions':0,'all_seven_complete':False})
            print('PASS_PLAN_ONLY: 430464 required - 9360 reused = 421104 proposed; 30 exact batches; 0 downloads')
        except BaseException:
            emit('qa_failure.json',{'utc':now(),'traceback':traceback.format_exc()});raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':main()
