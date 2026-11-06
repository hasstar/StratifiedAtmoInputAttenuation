"""Offline, immutable UK E-SST input census. No attenuation or label values."""
import hashlib
import json
import math
import msvcrt
import os
import platform
import re
import socket
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_024'
OLD = ROOT/'Outputs/Experiments/Combined_Channel_Revision_023'
EXT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external'
ACQ = EXT/'weather_gbs_extend_v011'
R3 = ROOT/'Outputs/Experiments/Combined_Channel_Revision_003'
R11 = ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'
RAIN = ('mean_total_precipitation_rate','mean_snowfall_rate')
NCAR_ROOTS = ('weather_top10_ncar_v008','weather_top10_ncar_v009','weather_gbs_extend_v011')
ALIASES = {'ozone_mass_mixing_ratio':'O3','u_component_of_wind':'U',
    'geopotential':'Z','relative_humidity':'R','mean_total_precipitation_rate':'MTPR',
    'mean_snowfall_rate':'MSR','convective_available_potential_energy':'CAPE',
    'maximum_2m_temperature_since_previous_post_processing':'MX2T'}


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def emit(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def deny(*a,**k): raise RuntimeError('Network forbidden in Revision024')


def valid_rain(total,snow):
    r=3600*total-3600*snow
    return r, bool(math.isfinite(total) and math.isfinite(snow) and total>=0 and snow>=0 and r>=0)


def rain_events(rows):
    """Exact positive wet runs; invalid/absent hours are never dry separators."""
    by={r['utc_epoch_seconds']:r for r in rows}
    wet=sorted(t for t,r in by.items() if r['rain_valid'] and r['liquid_mm_per_h']>0)
    groups=[]
    for t in wet:
        if not groups or groups[-1][-1]+3600!=t: groups.append([t])
        else: groups[-1].append(t)
    result=[]; mapping={}
    for i,ts in enumerate(groups):
        def boundary(t):
            r=by.get(t)
            if r is None: return 'ABSENT'
            if not r['rain_valid']: return 'INVALID_RAIN'
            assert r['liquid_mm_per_h']==0
            return 'ONE_HOUR_EXACT_ZERO'
        left,right=boundary(ts[0]-3600),boundary(ts[-1]+3600)
        e={'id':i,'first_utc_epoch_seconds':ts[0],'last_utc_epoch_seconds':ts[-1],
            'hours':len(ts),'left':left,'right':right,
            'two_sided_complete':left==right=='ONE_HOUR_EXACT_ZERO'}
        result.append(e)
        for t in ts: mapping[t]=e
    return result,mapping


def runs(times):
    groups=[]
    for t in sorted(times):
        if not groups or groups[-1][-1]+3600!=t:groups.append([t])
        else:groups[-1].append(t)
    return [{'start_utc':datetime.fromtimestamp(g[0],timezone.utc).isoformat(),
             'end_exclusive_utc':datetime.fromtimestamp(g[-1]+3600,timezone.utc).isoformat(),
             'hours':len(g)} for g in groups]


def collect():
    guards=read(OLD/'sources_before.json')
    assert len(guards)==1175
    assert sha(OLD/'delivery_manifest.json')==read(OLD/'final_verification.json')['manifest_sha256']
    for p,d in read(OLD/'delivery_manifest.json')['files'].items():
        assert sha(ROOT/p)==d['sha256']; guards[str(ROOT/p)]=d['sha256']
    paths={OLD/'delivery_manifest.json',OLD/'final_verification.json',ACQ/'plan.json',
        R3/'cohort_manifest.json',R3/'expanded_dataset.parquet',R3/'field_provenance.parquet',
        R11/'predictions_uk.parquet',
        ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/cloud_all.parquet',
        ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/raw_5fields.parquet',
        ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/field_summary.json'}
    downloads=sorted(EXT.rglob('*.download'))
    records=sorted(p for r in NCAR_ROOTS for p in (EXT/r/'records').glob('*.json'))
    sources=set()
    for p in records:
        d=read(p); paths.update([p,ROOT/d['source'],ROOT/d['metadata']]); sources.add((ROOT/d['source']).resolve())
    raw=set(p.resolve() for r in NCAR_ROOTS for p in (EXT/r/'payloads').glob('*.ascii'))
    assert sources==raw, 'Uncatalogued NCAR payloads require explicit inspection'
    paths.update(downloads)
    native=read(R3/'cohort_manifest.json')['native_days']
    for r in native:
        p=ROOT/r['path']; assert sha(p)==r['sha256']; paths.add(p)
    opensat=sorted((ROOT/'Outputs/Datasets/External_OpenSat4Weather_001/extracted').glob('*.nc'))
    assert len(opensat)==3; paths.update(opensat)
    for p in sorted(paths):
        h=sha(p)
        if str(p) in guards: assert guards[str(p)]==h
        guards[str(p)]=h
    assert {p:sha(p) for p in guards}==guards
    return guards,downloads,records,native,opensat


def nc_header(ds):
    v={k:{'dimensions':list(x.dimensions),'units':getattr(x,'units',None),
           'long_name':getattr(x,'long_name',None)} for k,x in ds.variables.items()}
    coords={}
    for k in ('valid_time','pressure_level','latitude','longitude'):
        if k in ds.variables:coords[k]=np.asarray(ds[k][:]).tolist()
    return {'dimensions':{k:len(x) for k,x in ds.dimensions.items()},'variables':v,'coordinates':coords}


def main():
    socket.socket.connect=deny; socket.create_connection=deny
    assert not OUT.exists(), 'Create-new only; never rerun this output'
    guards,downloads,records,native,opensat=collect()
    OUT.mkdir()
    emit('sources_before.json',guards)
    emit('code_before.json',{str(p):sha(p) for p in CODE.iterdir() if p.is_file()})
    (OUT/'PROTOCOL.md').open('x',encoding='utf-8').write((CODE/'PROTOCOL.md').read_text(encoding='utf-8'))
    emit('execution.json',{'utc':now(),'pid':os.getpid(),'argv':sys.argv,'executable':sys.executable,
        'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'pyarrow':pyarrow.__version__,'netCDF4':netCDF4.__version__,
        'preflight_cim_utc':'2026-10-17T17:28:49.5671489Z','matching_workers':0,
        'network_disabled':True,'training':0,'label_or_prediction_values_read':False})
    with (OUT/'execution.lock').open('xb') as f:f.write(b'0')
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            cds=[]
            for p in downloads:
                members=[]
                if zipfile.is_zipfile(p):
                    with zipfile.ZipFile(p) as z:
                        for n in z.namelist():
                            blob=z.read(n)
                            with netCDF4.Dataset('member.nc',memory=blob) as ds: h=nc_header(ds)
                            members.append({'member':n,'sha256':hashlib.sha256(blob).hexdigest(),**h})
                else:
                    with netCDF4.Dataset('payload.nc',memory=p.read_bytes()) as ds:h=nc_header(ds)
                    members.append({'member':None,**h})
                cds.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,'members':members})
            emit('cds_headers.json',cds)
            fields=defaultdict(set); record_ledger=[]; rain={}; rain_paths=defaultdict(set)
            for p in records:
                d=read(p); source=ROOT/d['source']; meta=ROOT/d['metadata']
                assert sha(source)==d['source_sha256'] and sha(meta)==d['metadata_sha256']
                pairs={(k[0],k[1]) for k,v in d['records']}; assert len(pairs)==1
                var,lev=next(iter(pairs)); fields[var].add(lev)
                raw=source.read_text(encoding='utf-8'); das=meta.read_text(encoding='utf-8')
                match=re.search(r'ARRAY:\s*Float32\s+(\w+)\[',raw)
                assert match and match[1]==ALIASES[var]
                assert f'String units "{d["contract"]["units"]}";' in das
                assert f'Int32 ecmwf_parameter {d["contract"]["parameter_id"]};' in das
                record_ledger.append({'path':str(p.relative_to(ROOT)),'source':d['source'],
                    'metadata':d['metadata'],'variable':var,'level_hpa':lev,
                    'units':d['contract']['units'],'values':len(d['records']),
                    'first_valid_epoch_s':min(k[2] for k,v in d['records']),
                    'last_valid_epoch_s':max(k[2] for k,v in d['records'])})
                if var in RAIN:
                    assert d['contract']['ecmwf_local_table']==235
                    assert d['contract']['units']=='kg m**-2 s**-1'
                    for k,v in d['records']:
                        key=tuple(k)
                        if key in rain:assert rain[key]==v
                        rain[key]=v;rain_paths[key].add(str(p.relative_to(ROOT)))
            emit('ncar_record_inventory.json',record_ledger)
            assert 'v_component_of_wind' not in fields and 'temperature' not in fields
            assert 700 not in fields['u_component_of_wind']
            special=[]
            for p in opensat:
                with netCDF4.Dataset('opensat.nc',memory=p.read_bytes()) as ds:
                    t=ds['time']; nums=t[:];dates=netCDF4.num2date([float(nums.min()),float(nums.max())],t.units,getattr(t,'calendar','standard'))
                    special.append({'path':str(p.relative_to(ROOT)),'variables':list(ds.variables),
                        'time_units':t.units,'first_time':str(dates[0]),'last_time':str(dates[1]),
                        'uk_cohort_time_overlap':False,'reason':'2022 dataset; UK cohort is 2003/2004'})
                    assert all(x.year==2022 for x in dates)
            emit('out_of_scope_assets.json',special)
            plan=read(ACQ/'plan.json'); links={r['id']:r for r in plan['links']}
            native_rows=[]; expected=set()
            for r in native:
                station=r['station'];day=datetime.fromisoformat(r['date_utc']).replace(tzinfo=timezone.utc)
                expected.update((station,int((day+timedelta(hours=h)).timestamp())) for h in range(24))
                with netCDF4.Dataset('native.nc',memory=(ROOT/r['path']).read_bytes()) as ds:
                    names=list(ds.variables)
                    assert set(names)=={'link_id','frequency','polarisation_transmit','polarisation_receive',
                        'latitude_receiver','longitude_receiver','satellite_position','ngr_receiver',
                        'elevation_angle','receiver_altitude_amsl','time','clearsky_attenuation'}
                    def scalar(n):return float(ds[n][...])
                    pol=ds['polarisation_transmit'][:].tobytes().decode().strip('\x00 ')
                    rr={'station':station,'date_utc':r['date_utc'],'source':r['path'],
                        'latitude':scalar('latitude_receiver'),'longitude':scalar('longitude_receiver'),
                        'height_amsl_m':scalar('receiver_altitude_amsl'),'frequency_ghz':scalar('frequency'),
                        'polarisation_transmit':pol,'variables':names}
                    assert abs(rr['frequency_ghz']-20.7)<1e-4
                    assert rr['height_amsl_m']==links[station]['height_amsl_m']
                    native_rows.append(rr)
            assert len(expected)==5664 and len(native_rows)==236
            emit('native_header_geometry.json',native_rows)
            old=pq.read_table(R3/'expanded_dataset.parquet',columns=['station','utc_epoch_seconds',*RAIN,'liquid_estimate_mm']).to_pylist()
            mask=pq.read_table(R11/'predictions_uk.parquet',columns=['station','utc_epoch_seconds','primary_score_eligible','common_score_eligible']).to_pylist()
            oldmap={(r['station'],r['utc_epoch_seconds']):r for r in old}
            maskmap={(r['station'],r['utc_epoch_seconds']):r for r in mask}
            assert len(old)==len(oldmap)==len(mask)==len(maskmap)==5664
            assert set(oldmap)==set(maskmap)==expected
            contexts=[]; events=[]; ledger=[]; summaries={}; maxdelta=0
            for sid,link in links.items():
                cell=(int(round(4*link['latitude_deg_n'])),int(round(4*link['longitude_deg_e'])))
                times=sorted({k[2]-3600 for k in rain if k[0] in RAIN and k[3:]==cell})
                rows=[]
                for t in times:
                    keys=[(v,0,t+3600,*cell) for v in RAIN]
                    assert all(k in rain for k in keys)
                    a,b=(rain[k] for k in keys); liquid,valid=valid_rain(a,b)
                    row={'station':sid,'utc_epoch_seconds':t,'valid_end_epoch_seconds':t+3600,
                        'grid_latitude_quarter':cell[0],'grid_longitude_quarter':cell[1],
                        'mtpr_kg_m2_s':a,'msr_kg_m2_s':b,'liquid_mm_per_h':liquid,
                        'rain_valid':valid,'in_evaluation_scope':(sid,t) in expected,
                        'record_sources':sorted(set().union(*(rain_paths[k] for k in keys)))}
                    rows.append(row)
                ev,emap=rain_events(rows); contexts.extend(rows)
                events.extend({'station':sid,**e} for e in ev)
                by={r['utc_epoch_seconds']:r for r in rows}
                for station,t in sorted(k for k in expected if k[0]==sid):
                    r=by[t]; original=oldmap[sid,t]; m=maskmap[sid,t]
                    for col,val in [(RAIN[0],r['mtpr_kg_m2_s']),(RAIN[1],r['msr_kg_m2_s']),('liquid_estimate_mm',r['liquid_mm_per_h'])]:
                        delta=abs(val-original[col]);assert delta<=1e-12;maxdelta=max(maxdelta,delta)
                    assert bool(m['common_score_eligible'])==(bool(m['primary_score_eligible']) and r['rain_valid'])
                    if not r['rain_valid']:status='INVALID_RAIN'
                    elif r['liquid_mm_per_h']==0:status='DRY_HOUR_NOT_PROOF_OF_DRY_FORWARD_PATH'
                    elif emap[t]['two_sided_complete']:status='WET_EVENT_TWO_SIDED_COMPLETE'
                    else:status='WET_EVENT_CENSORED'
                    ledger.append({'station':sid,'utc_epoch_seconds':t,
                        'date_utc':datetime.fromtimestamp(t,timezone.utc).date().isoformat(),
                        'rain_status':status,'rain_valid':r['rain_valid'],
                        'u700_status':'MISSING','v700_status':'MISSING','qualified_h0_status':'MISSING',
                        'event_id':emap[t]['id'] if t in emap else None,
                        'forward_context_sufficiency':'UNDETERMINED_WITHOUT_H_AND_V',
                        'e_sst_computable':False,'qualified_target':bool(m['primary_score_eligible']),
                        'existing_physics_common':bool(m['common_score_eligible'])})
                relevant=[r for r in ledger if r['station']==sid]
                summaries[sid]={'evaluation_hours':len(relevant),'evaluation_days':len({r['date_utc'] for r in relevant}),
                    'existing_rain_context_hours':len(rows),'outside_evaluation_rain_hours':sum(not r['in_evaluation_scope'] for r in rows),
                    'valid_rain_context_hours':sum(r['rain_valid'] for r in rows),
                    'evaluation_rain_status_counts':dict(Counter(r['rain_status'] for r in relevant)),
                    'all_context_events':len(ev),'all_context_complete_events':sum(e['two_sided_complete'] for e in ev),
                    'known_context_segments':runs(times),
                    'evaluation_segments':runs(r['utc_epoch_seconds'] for r in relevant),
                    'missing_u700_hours':len(relevant),'missing_v700_hours':len(relevant),'missing_qualified_h0_hours':len(relevant)}
            emit('rain_context.json',contexts); emit('rain_event_boundaries.json',events)
            emit('hourly_input_ledger.json',ledger)
            gap_days=[]
            for r in native:
                gap_days.append({'station':r['station'],'date_utc':r['date_utc'],'hours':24,
                    'missing_fields':['u700','v700','qualified_h0'],'missing_field_station_hours':72,
                    'additional_context_quantity':'UNDETERMINED_H_V_AND_EVENT_BOUNDARIES',
                    'weather_acquisition_authorized':False})
            emit('missing_fields_by_date.json',gap_days)
            auxiliary={}
            for revision,name in [('008','cloud_all.parquet'),('010','raw_5fields.parquet')]:
                p=ROOT/f'Outputs/Experiments/Combined_Channel_Revision_{revision}'/name
                tab=pq.read_table(p)
                assert len(tab)==5664
                keys=set(zip(tab['station'].to_pylist(),tab['utc_epoch_seconds'].to_pylist()))
                assert keys==expected
                auxiliary[revision]={'path':str(p.relative_to(ROOT)),'columns':tab.column_names,'hours':len(tab),'exact_key_match':True}
            emit('summary.json',{'utc':now(),'status':'PASS_INPUT_AUDIT_ONLY','cds_payloads':len(cds),
                'cds_netcdf_members':sum(len(x['members']) for x in cds),'ncar_record_files':len(records),
                'ncar_field_levels':{k:sorted(v) for k,v in fields.items()},'canonical_rain_values':len(rain),
                'stations':summaries,'evaluation_hours':len(ledger),'evaluation_station_days':len(native),
                'qualified_target_hours':sum(r['qualified_target'] for r in ledger),
                'existing_physics_common_hours':sum(r['existing_physics_common'] for r in ledger),
                'max_old_rain_delta_mm_per_h':maxdelta,'rain_tolerance_mm_per_h':1e-12,
                'minimum_missing_evaluation_field_station_hours':len(ledger)*3,
                'new_computable_esst_hours':0,'auxiliary_existing_fields':auxiliary,
                'actual_future_context_request_quantity':None,'weather_requests':0,
                'new_predictions':0,'new_scores':0,'all_seven_complete':False,
                'bounded_search_only':True,'mts_status':'NOT_COMPUTED_USER_DECISION'})
            after={p:sha(p) for p in guards}; assert after==guards; emit('sources_after.json',after)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    emit('execution_complete.json',{'utc':now(),'status':'PASS_INPUT_AUDIT_ONLY','pid':os.getpid(),'lock':'RELEASED'})
    print(json.dumps({'status':'PASS_INPUT_AUDIT_ONLY','protected_sources':len(guards),'summary':read(OUT/'summary.json')},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():
            import traceback
            emit('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json',{'utc':now(),'traceback':traceback.format_exc()})
        raise
