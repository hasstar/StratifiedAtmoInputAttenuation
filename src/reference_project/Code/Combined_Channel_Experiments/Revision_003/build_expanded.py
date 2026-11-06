"""Independent raw-byte field, geometry and native-target QA; no model imports."""
from __future__ import annotations
import csv
import hashlib
import math
import re
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import netCDF4
import numpy as np
import pandas as pd
from common import *

audit = load_module('independent_field_primitives', REV1/'audit_gbs_48h_actual_fields_v001.py')
CATALOG = dict(audit.CATALOG)
CATALOG.update({RAIN[0]:('MTPR','mtpr',55,('kg m**-2 s**-1',)),
                RAIN[1]:('MSR','msr',31,('kg m**-2 s**-1',))})
LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]

def decode_ascii(text, das, variable, node):
    """Separate decoder: parse actual DDS, maps, indexed values, DAS and URL axes."""
    nc, short, pid, units = CATALOG[variable]
    blocks = audit.das_blocks(das)
    attrs = blocks[nc]
    table = 235 if variable in RAIN else 128
    require(attrs['ecmwf_parameter']==pid and attrs['ecmwf_local_table']==table, 'Parameter/table identity')
    require(attrs['short_name']==short and attrs['units'] in units and attrs['rda_dataset']=='ds633.0', 'Short name/unit/dataset identity')
    require(blocks['latitude']['units']=='degrees_north' and blocks['longitude']['units']=='degrees_east', 'Coordinate unit')
    header, body = re.split(r'\n-{10,}\s*\n', text, maxsplit=1)
    m = re.findall(r'Float32\s+'+nc+r'((?:\[\w+\s*=\s*\d+\])+);', header)
    require(len(m)==1, 'One DDS array')
    dims = re.findall(r'\[(\w+)\s*=\s*(\d+)\]', m[0])
    axes, shape = [x[0] for x in dims], tuple(int(x[1]) for x in dims)
    expected_axes = {'pressure':['time','level','latitude','longitude'],
                     'surface':['time','latitude','longitude'],
                     'forecast':['forecast_initial_time','forecast_hour','latitude','longitude']}[node['layout']]
    require(axes==expected_axes==node['dimensions'] and all(n>0 for n in shape), 'DDS axes/order/empty')
    dataset = re.search(r'\}\s*(files/(?:g/)?d633000/[^;]+);\s*$', header)
    require(dataset is not None and dataset[1].split('d633000/',1)[1]==node['file'], 'Actual dataset path')
    require(f'.{table}_{pid:03d}_{short}.' in dataset[1], 'Filename parameter identity')
    lines = [s.strip() for s in body.splitlines() if s.strip()]
    require(lines[0]==f'{nc}.{nc}'+''.join(f'[{n}]' for n in shape), 'Array shape header')
    arr = np.full(shape, np.nan); seen=set(); i=1
    while i<len(lines) and lines[i].startswith('['):
        line = re.fullmatch(r'((?:\[\d+\]){'+str(len(shape)-1)+r'}),\s*(.*)', lines[i])
        require(line is not None, 'Indexed row format')
        index=tuple(int(v) for v in re.findall(r'\[(\d+)\]', line[1]))
        vals=[float(v.strip()) for v in line[2].split(',')]
        require(index not in seen and all(0<=k<n for k,n in zip(index,shape)), 'Duplicate/out-of-range indexed row')
        require(len(vals)==shape[-1], 'Longitude value count')
        arr[index]=vals;seen.add(index);i+=1
    require(len(seen)==math.prod(shape[:-1]) and np.isfinite(arr).all(), 'Incomplete/nonfinite data')
    for missing in ('_FillValue','missing_value'):
        require(missing not in attrs or not np.isclose(arr, attrs[missing],rtol=1e-6,atol=0).any(), 'Missing sentinel')
    maps={}
    for axis,n in dims:
        require(i+1<len(lines) and lines[i]==f'{nc}.{axis}[{n}]', 'Map header/order')
        vals=[float(v.strip()) for v in lines[i+1].split(',')]
        require(len(vals)==int(n) and len(set(vals))==len(vals) and np.isfinite(vals).all(), 'Map count/unique/finite')
        require(vals==node['coords'][axis], 'Actual map versus exact request')
        maps[axis]=vals;i+=2
    require(i==len(lines) and math.prod(shape)==node['expected_values'], 'Trailing data/count')
    require(node['nc_variable']==nc and len(node['axes'])==len(axes), 'Request variable/axes')
    date_match = re.search(r'\.(\d{10})_(\d{10})\.nc$', node['file'])
    require(date_match is not None, 'Dataset date range')
    first=datetime.strptime(date_match[1],'%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp()
    last=datetime.strptime(date_match[2],'%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp()
    temporal='forecast_initial_time' if node['layout']=='forecast' else 'time'
    times=audit.utc_seconds(maps[temporal],blocks[temporal]['units'],blocks[temporal].get('calendar','gregorian'))
    step=43200 if node['layout']=='forecast' else 3600
    require(times==[int(first+step*k) for k in node['axes'][0]] and max(times)<=last, 'Independent file/UTC index identity')
    lats=audit.exact_grid(maps['latitude']);lons=audit.exact_grid(maps['longitude'],True)
    require(maps['latitude']==[90-.25*k for k in node['axes'][-2]], 'Latitude published index')
    require(maps['longitude']==[.25*k for k in node['axes'][-1]], 'Longitude published index')
    constraint=nc+''.join(f'[{a[0]}]' if len(a)==1 else f'[{a[0]}:{a[1]-a[0]}:{a[-1]}]' for a in node['axes'])
    require(node['url']=='https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/'+node['file']+'.ascii?'+constraint, 'Exact anonymous request URL')
    decoded={}
    if node['layout']=='pressure':
        require(blocks['level']['units']=='hPa' and maps['level']==[LEVELS[k] for k in node['axes'][1]], 'Pressure identity')
        require(set(maps['level'])=={node['level_hpa']}, 'Requested pressure level')
    if node['layout']=='forecast':
        require(blocks['forecast_hour']['units']=='hour', 'Forecast lead units')
        leads=maps['forecast_hour']
        require(leads==[k+1 for k in node['axes'][1]] and all(1<=x<=12 and x==int(x) for x in leads), 'Forecast lead identity')
        require(all(datetime.fromtimestamp(t,timezone.utc).hour in (6,18) for t in times), 'Forecast cycle')
        require(('meanflux' if variable in RAIN else 'minmax') in node['file'], 'Mean-rate/max processing group')
    for index in np.ndindex(shape):
        if node['layout']=='pressure':
            key=(variable,int(maps['level'][index[1]]),times[index[0]],lats[index[2]],lons[index[3]])
        elif node['layout']=='surface':
            key=(variable,0,times[index[0]],lats[index[1]],lons[index[2]])
        else:
            key=(variable,0,int(times[index[0]]+3600*leads[index[1]]),lats[index[2]],lons[index[3]])
        require(key not in decoded, 'Duplicate actual key')
        decoded[key]=float(arr[index])
    return decoded, {'source_layout':node['layout'],'variable':variable,'parameter_table':table,
                     'parameter':pid,'units':attrs['units'],'shape':shape,'values':len(decoded)}

def select_scope(plan):
    parents={p['id']:p for p in plan['parents']}
    units={}; maps={kind:{} for kind in ('surface','rain','pressure')}
    for path in sorted((ACQ/'units').glob('*.json')):
        unit=read(path);signed_check(unit)
        require(unit['plan_fingerprint']==PLAN_FP, 'Unit plan fingerprint')
        parent=parents[unit['parent_id']];kind=parent['kind'];units[parent['id']]=unit
        for record in unit['records']:
            require(record['hours_utc_epoch']==parent['hours_utc_epoch'], 'Record versus parent exact hours')
            times=record.get('beacon_interval_start_utc_epoch',record['hours_utc_epoch'])
            if kind=='rain':require(times==[t-3600 for t in record['hours_utc_epoch']], 'Rain interval alignment')
            for j,t in enumerate(times):
                key=(record['link_id'],t,int(record.get('level_hpa',0)))
                require(key not in maps[kind], 'Duplicate station-hour unit')
                maps[kind][key]=(parent['id'],record['rows'][j] if kind=='pressure' else record['surface_cell'])
    complete=[]
    for link in plan['links']:
        station=link['id']
        for native in plan['native_days']:
            if native['station']!=station:continue
            t=int(pd.Timestamp(native['date_utc'],tz='UTC').timestamp())
            keys=[(station,t+h*3600) for h in range(24)]
            if all((s,e,0) in maps[k] for s,e in keys for k in ('surface','rain')) and all((s,e,l) in maps['pressure'] for s,e in keys for l in (1,5,500,650,850)):
                complete.append(native)
    counts=Counter(x['station'] for x in complete)
    require(counts=={'chilbolton':143,'sparsholt':93}, 'Stopped complete-day scope changed')
    return complete, maps, units, parents

def build_fields(plan, complete, maps, units, protect):
    requests=[];required=set();links={x['id']:x for x in plan['links']}
    for native in complete:
        sid=native['station'];day=int(pd.Timestamp(native['date_utc'],tz='UTC').timestamp())
        for t in range(day,day+86400,3600):
            for name in FEATURES+RAIN:
                prefix,var=name.split('__',1) if '__' in name else ('rain',name)
                kind='pressure' if prefix.startswith('p') else prefix
                level=int(prefix[1:]) if kind=='pressure' else 0
                parent,geo=maps[kind][sid,t,level]
                cell=geo['cell'] if kind=='pressure' else geo
                key=(var,level,t+3600 if kind=='rain' else t,*cell)
                requests.append((sid,t,name,parent,key));required.add(key)
    # Source ordering reproduces collection priority: old CDS, current waves, old NCAR.
    order=[x['path'] for x in plan['reusable']]
    for wave in sorted((ACQ/'waves').glob('*.json')):
        w=read(wave);signed_check(w)
        for node in w['jobs']:
            p=str((ACQ/node['target']).relative_to(ROOT))
            if p not in order:order.append(p)
    for item in plan['ncar_reusable']:
        p=read(ROOT/item['record_path'])['source']
        if p not in order:order.append(p)
    neededparents={r[3] for r in requests}
    source_hashes={p:h for name,u in units.items() if name in neededparents for p,h in u['source_payloads'].items()}
    require(set(source_hashes).issubset(order), 'Unknown payload source')
    decoded={};evidence=[];negative=[]
    for n,source in enumerate(p for p in order if p in source_hashes):
        path=protect(ROOT/source,source_hashes[source])
        if path.suffix=='.ascii':
            rp=path.parent.parent/'records'/f'{path.stem}.json'
            jp=path.parent.parent/'jobs'/f'{path.stem}.json'
            rec=read(protect(rp));signed_check(rec);job=read(protect(jp));node=job['job']
            require(job['status']=='VERIFIED_SUBSET' and job['sha256']==rec['source_sha256']==sha(path), 'Verified payload identity')
            require(job['records_sha256']==sha(rp) and rec['request_fingerprint']==fp(node)==job['request_fingerprint'], 'Record/request identity')
            receipt=job['receipt'];require(receipt['http_status']==200 and receipt['bytes']==path.stat().st_size and receipt['sha256']==sha(path), 'Actual response size/SHA')
            if receipt.get('content_length') is not None:require(receipt['content_length']==path.stat().st_size, 'Official length')
            require(receipt['url']==node['url'] and path.stat().st_size<=16*1024*1024, 'Receipt URL/body cap')
            meta=protect(ROOT/rec['metadata'],rec['metadata_sha256'])
            raw=path.read_text();das=meta.read_text()
            values,ev=decode_ascii(raw,das,node['variable'],node)
            canonical={tuple(k):v for k,v in rec['records']}
            require(len(canonical)==len(rec['records']) and set(values)==set(canonical), 'Canonical complete key identity')
            audit.compare_required(values,canonical)
            if node['variable'] not in {x['variable'] for x in negative}:
                tests={'wrong_unit':(raw,das.replace('String units "'+CATALOG[node['variable']][3][0]+'";','String units "BAD";',1),node),
                       'nonfinite':(re.sub(r'(\n\[0\](?:\[0\])*,\s*)[^,\n]+',r'\g<1>NaN',raw,count=1),das,node),
                       'wrong_utc_map':(raw,das,{**node,'coords':{**node['coords'],node['dimensions'][0]:[v+1 for v in node['coords'][node['dimensions'][0]]]}}),
                       'wrong_count':(raw,das,{**node,'expected_values':node['expected_values']+1})}
                for test,args in tests.items():
                    try:decode_ascii(args[0],args[1],node['variable'],args[2])
                    except (RuntimeError,ValueError,KeyError):pass
                    else:raise RuntimeError('Negative control not rejected: '+test)
                negative.append({'variable':node['variable'],'controls':list(tests),'status':'PASS'})
        else:
            values,ev=audit.parse_netcdf(path,{k:0. for k in required if k[0] not in RAIN})
        decoded[source]=values;evidence.append({'source':source,'sha256':sha(path),**ev})
        if n%100==0:print('raw_fields_decoded',n,flush=True)
    cache={}
    for parent in neededparents:
        d={}
        for source in order:
            if source in units[parent]['source_payloads']:
                for key,value in decoded[source].items():d.setdefault(key,(value,source))
        cache[parent]=d
    rows={};provenance=[]
    for sid,t,name,parent,key in requests:
        require(key in cache[parent], 'Missing complete-scope field: '+str((sid,t,name,key)))
        value,source=cache[parent][key];require(math.isfinite(value), 'Nonfinite field')
        rows.setdefault((sid,t),{'station':sid,'utc_epoch_seconds':t})[name]=value
        provenance.append({'station':sid,'utc_epoch_seconds':t,'name':name,'variable':key[0],
                           'level_hpa':key[1],'valid_utc_epoch_seconds':key[2],'latitude_quarter_index':key[3],
                           'longitude_quarter_index':key[4],'value':value,'source':source,'source_sha256':source_hashes[source]})
    frame=pd.DataFrame(rows.values()).sort_values(['station','utc_epoch_seconds']).reset_index(drop=True)
    require(len(frame)==5664 and len(provenance)==5664*12, 'Full station-hour field count')
    geometry=[];rays={s:audit.ray(x) for s,x in links.items()};maxdiff=0.
    for row in frame.itertuples(index=False):
        sid,t=row.station,row.utc_epoch_seconds;stationcell=audit.nearest(links[sid]['latitude_deg_n'],links[sid]['longitude_deg_e'])
        require(maps['surface'][sid,t,0][1]==maps['rain'][sid,t,0][1]==stationcell, 'Surface/rain nearest grid')
        for l in (1,5,500,650,850):
            prior=maps['pressure'][sid,t,l][1]
            if l==1:
                z=row.p1__geopotential;H=z/9.80665;height=6371229*H/(6371229-H)
                require(0<H<6371229 and prior['status']=='STABLE_NEAREST' and 1<=prior['iteration']<=8 and abs(prior['z']-z)<1e-6, 'Dynamic pressure height')
            else:
                height=audit.FIXED[l];require(prior['status']=='STABLE_FIXED_HEIGHT_NEAREST' and plan['fixed_heights'][str(l)]['geometric_height_m']==height, 'Fixed-height definition')
            geo=audit.intersection(*rays[sid],height)
            delta=max(abs(geo['lat']-prior['intersection_lat']),abs(geo['lon']-prior['intersection_lon']))
            require(geo['cell']==prior['cell'] and delta<1e-7 and abs(geo['slant']-prior['slant_m'])<.01 and abs(height-prior['height_m'])<.001, 'Independent forward LOS geometry')
            maxdiff=max(maxdiff,delta)
            geometry.append({'station':sid,'utc_epoch_seconds':t,'level_hpa':l,'latitude':geo['lat'],'longitude':geo['lon'],'height_m':height,'slant_m':geo['slant']})
    frame['hour_utc']=pd.to_datetime(frame.utc_epoch_seconds,unit='s',utc=True)
    frame['period']=np.where(frame.hour_utc.dt.year.eq(2004),'original39','extension2003')
    frame['total_water_mm']=3600*frame[RAIN[0]];frame['snow_water_mm']=3600*frame[RAIN[1]]
    frame['liquid_estimate_mm']=frame.total_water_mm-frame.snow_water_mm
    frame['liquid_negative']=frame.liquid_estimate_mm.lt(0)
    frame['rain_band']=np.select([frame.liquid_negative,frame.liquid_estimate_mm.lt(.1),frame.liquid_estimate_mm.lt(1)],['negative_difference','below_0.1','0.1_to_1'],default='at_least_1')
    pd.DataFrame(provenance).to_parquet(OUT/'field_provenance.parquet',index=False)
    pd.DataFrame(geometry).to_parquet(OUT/'geometry_checks.parquet',index=False)
    emit(OUT/'raw_source_evidence.json',evidence);emit(OUT/'negative_controls.json',{'status':'PASS','checks':negative,'tests':sum(len(x['controls']) for x in negative)})
    emit(OUT/'field_qa.json',{'status':'PASS','station_hours':len(frame),'required_field_values':len(provenance),'raw_sources':len(evidence),'pressure_intersections':len(geometry),'max_geometry_coordinate_difference_deg':maxdiff,'network_calls':0,'fit_calls':0,'prediction_calls':0})
    return frame

def native_audit(frame, plan, complete, protect):
    build=read(protect(ALIGN/'build_manifest.json'))
    labelpath=protect(ALIGN/'ceda_gbs_hourly_attenuation.parquet',build['output_sha256']['ceda_gbs_hourly_attenuation.parquet'])
    labels=pd.read_parquet(labelpath)
    require(not labels.duplicated(['site','hour_utc']).any(), 'Duplicate original label')
    selected=frame[['station','hour_utc']].rename(columns={'station':'site'}).merge(labels,on=['site','hour_utc'],how='left',validate='one_to_one')
    require(selected.source_file.notna().all(), 'Missing native source')
    with protect(GBS/'source_manifest.csv').open(encoding='utf-8-sig',newline='') as f:manifest={r['relative_path']:r for r in csv.DictReader(f)}
    links={r['id']:r for r in plan['links']};native_sha={x['path']:x['sha256'] for x in complete}
    checks=[];maxdelta=0.
    for rel,rows in selected.groupby('source_file',sort=True):
        path=protect(GBS/rel);require(sha(path)==native_sha[str(path.relative_to(ROOT))], 'Native source plan SHA')
        blob=path.read_bytes();entry=manifest[rel];link=links[entry['site']]
        require(len(blob)==int(entry['bytes']) and hashlib.md5(blob).hexdigest()==entry['md5'], 'Native length/MD5')
        require(len(rows)==24 and rows.site.nunique()==1 and rows.site.iloc[0]==entry['site'], 'Native date/site count')
        day=pd.Timestamp(entry['date_utc'],tz='UTC')
        with netCDF4.Dataset('native.nc',memory=blob) as ds:
            ds.set_auto_mask(False)
            def scalar(name,unit):
                v=ds.variables[name];require(v.units==unit,'Native scalar units');return float(v[...])
            require(abs(scalar('frequency','GHz')-20.7)<1e-4, 'Native carrier')
            require(scalar('latitude_receiver','degrees_north')==link['latitude_deg_n'] and (scalar('longitude_receiver','degrees_east')+180)%360-180==link['longitude_deg_e'], 'Native receiver coordinate')
            require(scalar('receiver_altitude_amsl','m')==link['height_amsl_m'] and -scalar('satellite_position','degrees_west')==link['satellite_longitude_deg_e'], 'Native geometry')
            signal,time=ds.variables['clearsky_attenuation'],ds.variables['time']
            require(signal.units=='dB' and float(signal.integration_period)==1 and 'relative to clear sky' in signal.long_name.lower(), 'Native target definition')
            origin=re.match(r'seconds since (\d+)-(\d+)-(\d+) 0:0:0\.0+ \+0:00$',time.units)
            require(origin is not None and tuple(map(int,origin.groups()))==(day.year,day.month,day.day) and 'end of integration' in time.long_name.lower(), 'Native UTC/integration end')
            t=np.asarray(time[:],float);y=np.asarray(signal[:],float)
            require(t.ndim==y.ndim==1 and t.shape==y.shape, 'Native shape')
            valid=np.isfinite(t)&(t>0)&(t<=86400)&np.isfinite(y)&(y!=signal._FillValue)&(y>=signal.valid_min)&(y<=signal.valid_max)
            require(np.all(t[valid]==np.rint(t[valid])) and len(set(t[valid]))==int(valid.sum()), 'Native seconds unique/integral')
            for row in rows.itertuples(index=False):
                h=int((row.hour_utc-day).total_seconds()/3600);require(0<=h<24,'Hour/date identity')
                values=y[valid&(t>h*3600)&(t<=(h+1)*3600)];n=len(values)
                require(n==int(row.valid_seconds)<=3600 and abs(row.coverage_fraction-n/3600)<1e-14 and bool(row.coverage_ge_95)==(n>=3420), 'Native coverage')
                if n:
                    answer=float(-10*np.log10(np.mean(np.exp(-values*math.log(10)/10))))
                    delta=abs(answer-row.clear_sky_attenuation_db)
                    require(math.isfinite(row.clear_sky_attenuation_db) and delta<1e-10 and row.target_status=='native_ceda_gbs', 'Independent native label')
                    maxdelta=max(maxdelta,delta)
                else:require(pd.isna(row.clear_sky_attenuation_db) and row.target_status=='no_native_attenuation','Missing target retained')
                require(row.clear_reference_method=='source_native_clear_sky_attenuation','No target reference change')
                require(int((values>=3).sum())==row.seconds_ge_3db and int((values>=10).sum())==row.seconds_ge_10db,'Native diagnostic counts')
        checks.append({'station':entry['site'],'date':entry['date_utc'],'source':str(path.relative_to(ROOT)),'sha256':sha(path),'valid_seconds':int(rows.valid_seconds.sum())})
    require(len(checks)==236, 'Native file count')
    columns=['site','hour_utc','clear_sky_attenuation_db','valid_seconds','coverage_fraction','coverage_ge_95','target_status','source_file']
    merged=frame.merge(selected[columns].rename(columns={'site':'station','clear_sky_attenuation_db':'truth_db'}),on=['station','hour_utc'],validate='one_to_one')
    merged['primary_score_eligible']=np.isfinite(merged.truth_db)&merged.coverage_ge_95
    merged['exclusion_reason']=np.select([~np.isfinite(merged.truth_db),~merged.coverage_ge_95],['missing_native_target','native_coverage_below_95'],default='eligible')
    emit(OUT/'native_label_sources.json',checks)
    counts={s:{'complete_days':int(f.hour_utc.dt.date.nunique()),'hours':len(f),'eligible_hours':int(f.primary_score_eligible.sum()),'exclusion_reasons':f.exclusion_reason.value_counts().to_dict()} for s,f in merged.groupby('station')}
    emit(OUT/'native_label_qa.json',{'status':'PASS','files':len(checks),'max_label_delta_db':maxdelta,'counts':counts,'reference_fits':0,'target_imputations':0})
    return merged

def main():
    offline()
    with lock():
        require(not (OUT/'dataset_complete.json').exists(), 'Immutable stage already complete')
        require((ACQ/'STOP_REQUESTED.json').exists() and sha(ACQ/'plan.json')==PLAN_SHA,'Stop/plan gate')
        plan=read(ACQ/'plan.json');signed_check(plan);require(plan['fingerprint']==PLAN_FP and plan['feature_names']==FEATURES,'Plan/features')
        guard,protect=source_snapshot(plan)
        emit(OUT/'build_launch.json',{'started':now(),'pid':os.getpid(),'protocol_sha256':sha(HERE/'PROTOCOL.md'),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'network_disabled':True})
        complete,maps,units,parents=select_scope(plan)
        emit(OUT/'cohort_manifest.json',{'created':now(),'selection':'all completed acquisition-unit intersections with full24hour native date','native_days':complete,'counts':dict(Counter(x['station'] for x in complete)),'target_or_outcome_selection':False})
        print('scope_frozen',len(complete),'station_days',flush=True)
        frame=build_fields(plan,complete,maps,units,protect)
        print('field_gate_pass',len(frame),flush=True)
        frame=native_audit(frame,plan,complete,protect)
        frame.to_parquet(OUT/'expanded_dataset.parquet',index=False)
        frame.groupby(['station','period',frame.hour_utc.dt.strftime('%Y-%m')]).agg(hours=('hour_utc','size'),eligible_hours=('primary_score_eligible','sum')).to_csv(OUT/'coverage_by_month.csv')
        emit(OUT/'source_sha256_used.json',guard)
        protected_verify(guard)
        emit(OUT/'dataset_complete.json',{'status':'PASS','completed':now(),'station_hours':len(frame),'primary_eligible':int(frame.primary_score_eligible.sum()),'dataset_sha256':sha(OUT/'expanded_dataset.parquet'),'protected_sources':len(guard),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'new_fits':0,'predictions':0,'downloads':0})
        print(read(OUT/'dataset_complete.json'),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():emit(OUT/f'failure_build_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',{'time':now(),'traceback':traceback.format_exc()})
        raise
