"""Independent input-only QA: h5py headers and native ASCII rain decoding.

Does not import inventory.py or any legacy download/evaluation entry point.
"""
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import h5py
import pyarrow.dataset as ds

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_025'
R3=ROOT/'Outputs/Experiments/Combined_Channel_Revision_003'
R11=ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(524288),b''):h.update(b)
    return h.hexdigest()


def main():
    assert not (OUT/'independent_qa.json').exists()
    headers=load(OUT/'cds_headers.json'); variable_counts=Counter(); pressure=set(); members=0
    for item in headers:
        p=ROOT/item['path']; assert digest(p)==item['sha256']
        for member in item['members']:
            if member['member'] is None:blob=p.read_bytes()
            else:
                with zipfile.ZipFile(p) as z:blob=z.read(member['member'])
                assert hashlib.sha256(blob).hexdigest()==member['sha256']
            with h5py.File(io.BytesIO(blob),'r') as f:
                visible={k for k in f.keys() if not k.startswith('_')}
                assert visible==set(member['variables'])
                for k,v in member['variables'].items():
                    units=f[k].attrs.get('units')
                    if isinstance(units,bytes):units=units.decode()
                    assert units==v['units'],(p,k,units,v['units'])
                    variable_counts[k]+=1
                for k,values in member['coordinates'].items():
                    assert f[k][...].tolist()==values,(p,k)
                pressure.update(member['coordinates'].get('pressure_level',[]));members+=1
    assert 700 not in pressure
    assert not set(variable_counts).intersection({'t','v','deg0l','h0'})

    inv=load(OUT/'ncar_record_inventory.json'); ncar_pairs=defaultdict(set)
    all_rain={}; decoded=0; files=0; sources=defaultdict(set)
    origin=datetime(1900,1,1,tzinfo=timezone.utc)
    for item in inv:
        text=(ROOT/item['source']).read_text(encoding='utf-8')
        dataset_header=text.split('---------------------------------------------')[0]
        nc_name=dataset_header.split('ARRAY:')[1].split()[1].split('[')[0]
        metadata=(ROOT/item['metadata']).read_text(encoding='utf-8')
        short=metadata.split('String short_name "',1)[1].split('"',1)[0]
        expected={'U':'u','Z':'z','R':'r','O3':'o3','CAPE':'cape','MX2T':'mx2t','MTPR':'mtpr','MSR':'msr'}
        assert expected[nc_name]==short
        ncar_pairs[item['variable']].add(item['level_hpa'])
        if nc_name not in ('MTPR','MSR'):continue
        files+=1
        assert 'String units "kg m**-2 s**-1";' in metadata
        assert 'Int32 ecmwf_local_table 235;' in metadata
        assert 'String units "hours since 1900-01-01 00:00:00";' in metadata
        assert 'String units "hour";' in metadata
        assert '.meanflux.' in dataset_header
        axes={}
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if line.startswith(nc_name+'.') and not line.startswith(nc_name+'.'+nc_name+'['):
                name=line.split('.',1)[1].split('[')[0]
                axes[name]=[float(x.strip()) for x in lines[i+1].split(',')]
        assert set(axes)=={'forecast_initial_time','forecast_hour','latitude','longitude'}
        expected_values=math.prod(len(a) for a in axes.values());count=0
        local={}
        for line in lines:
            if not line.startswith('['):continue
            idxpart,vals=line.split(',',1); idx=[int(x) for x in re.findall(r'\[(\d+)\]',idxpart)]
            assert len(idx)==3
            values=[float(v.strip()) for v in vals.split(',')]
            assert len(values)==len(axes['longitude'])
            fi,fh,la=idx
            end=origin+timedelta(hours=axes['forecast_initial_time'][fi]+axes['forecast_hour'][fh])
            assert end.minute==end.second==end.microsecond==0
            lat=axes['latitude'][la]
            for j,value in enumerate(values):
                lon=(axes['longitude'][j]+180)%360-180
                assert lat*4==round(lat*4) and lon*4==round(lon*4)
                key=(item['variable'],0,int(end.timestamp()),int(lat*4),int(lon*4))
                if key in all_rain:assert all_rain[key]==value
                assert key not in local
                all_rain[key]=value;local[key]=value;count+=1
                sources[key].add(item['path'])
        assert count==expected_values==item['values'];decoded+=count
        canonical=load(ROOT/item['path'])
        assert {tuple(k):v for k,v in canonical['records']}==local

    cohort=load(R3/'cohort_manifest.json')['native_days']; keys=set()
    for r in cohort:
        midnight=datetime.strptime(r['date_utc'],'%Y-%m-%d').replace(tzinfo=timezone.utc)
        for h in range(24):keys.add((r['station'],int(midnight.timestamp())+h*3600))
    old=ds.dataset(R3/'expanded_dataset.parquet').to_table(columns=['station','utc_epoch_seconds','mean_total_precipitation_rate','mean_snowfall_rate','liquid_estimate_mm']).to_pylist()
    masks=ds.dataset(R11/'predictions_uk.parquet').to_table(columns=['station','utc_epoch_seconds','primary_score_eligible','common_score_eligible']).to_pylist()
    old={ (r['station'],r['utc_epoch_seconds']):r for r in old }
    masks={ (r['station'],r['utc_epoch_seconds']):r for r in masks }
    assert len(keys)==5664 and keys==set(old)==set(masks)
    contexts=load(OUT/'rain_context.json');by={(r['station'],r['utc_epoch_seconds']):r for r in contexts}
    assert len(contexts)==len(by)==7152
    maxdelta=0
    for k,r in by.items():
        sid,t=k;lat,lon=r['grid_latitude_quarter'],r['grid_longitude_quarter']
        assert (lat,lon)==({'chilbolton':(205,-6),'sparsholt':(204,-6)}[sid])
        a=all_rain['mean_total_precipitation_rate',0,t+3600,lat,lon]
        b=all_rain['mean_snowfall_rate',0,t+3600,lat,lon]
        liquid=(a-b)*3600
        good=all(math.isfinite(v) for v in (a,b)) and min(a,b)>=0 and liquid>=0
        assert r['rain_valid']==good and r['in_evaluation_scope']==(k in keys)
        err=abs(liquid-r['liquid_mm_per_h']);assert err<=1e-12;maxdelta=max(maxdelta,err)
        assert a==r['mtpr_kg_m2_s'] and b==r['msr_kg_m2_s']
        wanted_sources=set(sources['mean_total_precipitation_rate',0,t+3600,lat,lon])|set(sources['mean_snowfall_rate',0,t+3600,lat,lon])
        assert set(r['record_sources'])==wanted_sources
        if k in keys:
            assert a==old[k]['mean_total_precipitation_rate'] and b==old[k]['mean_snowfall_rate']
            assert abs(liquid-old[k]['liquid_estimate_mm'])<=1e-12
            assert masks[k]['common_score_eligible']==(masks[k]['primary_score_eligible'] and good)
    # Independently assign each wet hour by looking backwards/forwards, not grouping runs.
    independently=[]; idx={}; events=[]
    for sid in ('chilbolton','sparsholt'):
        hourly={t:r for (s,t),r in by.items() if s==sid}
        wet={t for t,r in hourly.items() if r['rain_valid'] and r['liquid_mm_per_h']>0}
        starts=sorted(t for t in wet if t-3600 not in wet)
        for i,start in enumerate(starts):
            end=start
            while end+3600 in wet:end+=3600
            def boundary(t):
                if t not in hourly:return 'ABSENT'
                if not hourly[t]['rain_valid']:return 'INVALID_RAIN'
                assert hourly[t]['liquid_mm_per_h']==0
                return 'ONE_HOUR_EXACT_ZERO'
            l,r=boundary(start-3600),boundary(end+3600)
            event={'station':sid,'id':i,'first_utc_epoch_seconds':start,'last_utc_epoch_seconds':end,
                'hours':(end-start)//3600+1,'left':l,'right':r,'two_sided_complete':l==r=='ONE_HOUR_EXACT_ZERO'}
            events.append(event)
            for t in range(start,end+1,3600):idx[sid,t]=event
    assert events==load(OUT/'rain_event_boundaries.json')
    for k in sorted(keys):
        sid,t=k;r=by[k]; e=idx.get(k);m=masks[k]
        status=('INVALID_RAIN' if not r['rain_valid'] else 'DRY_HOUR_NOT_PROOF_OF_DRY_FORWARD_PATH'
                if r['liquid_mm_per_h']==0 else 'WET_EVENT_TWO_SIDED_COMPLETE'
                if e['two_sided_complete'] else 'WET_EVENT_CENSORED')
        independently.append({'station':sid,'utc_epoch_seconds':t,'date_utc':datetime.fromtimestamp(t,timezone.utc).date().isoformat(),
            'rain_status':status,'rain_valid':r['rain_valid'],'u700_status':'MISSING','v700_status':'MISSING',
            'qualified_h0_status':'MISSING','event_id':None if e is None else e['id'],
            'forward_context_sufficiency':'UNDETERMINED_WITHOUT_H_AND_V','e_sst_computable':False,
            'qualified_target':m['primary_score_eligible'],'existing_physics_common':m['common_score_eligible']})
    assert independently==load(OUT/'hourly_input_ledger.json')
    gaps=load(OUT/'missing_fields_by_date.json')
    assert len(gaps)==236 and sum(x['hours'] for x in gaps)==5664
    assert {(x['station'],x['date_utc']) for x in gaps}=={(x['station'],x['date_utc']) for x in cohort}
    assert sum(x['missing_field_station_hours'] for x in gaps)==16992
    summary=load(OUT/'summary.json')
    assert summary['ncar_field_levels']=={k:sorted(v) for k,v in ncar_pairs.items()}
    assert summary['qualified_target_hours']==sum(r['qualified_target'] for r in independently)==5593
    assert summary['existing_physics_common_hours']==sum(r['existing_physics_common'] for r in independently)==5538
    for sid,s in summary['stations'].items():
        sub=[r for r in independently if r['station']==sid]
        assert s['evaluation_rain_status_counts']==dict(Counter(r['rain_status'] for r in sub))
        assert s['existing_rain_context_hours']==sum(r['station']==sid for r in contexts)
        assert s['all_context_events']==sum(r['station']==sid for r in events)
        assert s['all_context_complete_events']==sum(r['station']==sid and r['two_sided_complete'] for r in events)
    guards=load(OUT/'sources_before.json')
    assert guards==load(OUT/'sources_after.json')=={p:digest(p) for p in guards}
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_INPUT_AUDIT_ONLY',
        'argv':sys.argv,'executable':sys.executable,'independent_of_producer_imports':True,
        'h5py_version':h5py.__version__,'cds_payloads':len(headers),'cds_members_checked':members,
        'cds_pressure_levels':sorted(pressure),'ncar_records_checked':len(inv),
        'rain_ascii_files_decoded':files,'raw_rain_values_decoded':decoded,
        'unique_raw_rain_keys':len(all_rain),'station_context_hours_checked':len(contexts),
        'evaluation_hours_exactly_checked':len(keys),'station_days_exactly_checked':len(gaps),
        'max_rain_delta_mm_per_h':maxdelta,'rain_tolerance_mm_per_h':1e-12,
        'all_event_boundaries_exact':True,'all_hourly_fields_exact':True,
        'protected_sources_unchanged':len(guards),'label_values_read':False,
        'real_esst_computation_qualified':False,'new_scores':0,'new_predictions':0}
    with (OUT/'independent_qa.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
