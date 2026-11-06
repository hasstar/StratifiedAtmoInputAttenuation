"""Exact 24-field UK completion, with frozen-cohort forward-LOS geometry.

DAP decoder adapted from the read-only Combined001 ncar_top10_transport_v001.
No model execution, downloads of global grids, or historical queue operations.
"""
from __future__ import annotations
import re, math, itertools, hashlib, argparse, time
from collections import defaultdict, Counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from pyproj import Transformer
from metadata import catalog
from transport import *

LEVELS=[1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]
E1900=int(datetime(1900,1,1,tzinfo=timezone.utc).timestamp())
SPECS={
 'ozone_mass_mixing_ratio':'128_203_o3','relative_humidity':'128_157_r',
 'specific_humidity':'128_133_q','temperature':'128_130_t','geopotential':'128_129_z',
 'potential_vorticity':'128_060_pv','specific_rain_water_content':'128_075_crwc',
 'medium_cloud_cover':'128_187_mcc','total_column_cloud_ice_water':'128_079_tciw',
 'total_column_ozone':'128_206_tco3',
 'vertical_integral_of_eastward_water_vapour_flux':'162_071_viwve',
 'vertical_integral_of_eastward_ozone_flux':'162_077_vioze',
 'vertical_integral_of_northward_ozone_flux':'162_078_viozn',
 'vertical_integral_of_northward_cloud_liquid_water_flux':'162_089_vilwn'}
EXPECTED_UNITS={'128_203_o3':'kg kg**-1','128_157_r':'%','128_133_q':'kg kg**-1',
 '128_130_t':'K','128_129_z':'m**2 s**-2','128_060_pv':'K m**2 kg**-1 s**-1',
 '128_075_crwc':'kg kg**-1','128_187_mcc':'(0-1)','128_079_tciw':'kg m**-2',
 '128_206_tco3':'kg m**-2',**{s:'kg m**-1 s**-1' for s in SPECS.values() if s.startswith('162_')}}
OLD=ROOT/'Outputs/Experiments/Combined_Channel_Revision_003'
DATA=ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/expanded_dataset_with_itu_inputs.parquet'
GEOM=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010/scope_manifest.json'

def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def attr(body,name):
    m=re.search(r'String '+name+r'\s+"([^"]+)";',body);need(m is not None,'Metadata missing '+name);return m[1]
def contract(text,spec,pressure):
    blocks=dict(re.findall(r'\b([A-Za-z_]\w*)\s*\{([^{}]*)\}',text,re.S))
    table,pid,short=spec.split('_');var=short.upper()
    need(var in blocks,'DAS variable identity '+spec);b=blocks[var]
    for key,want in [('ecmwf_local_table',int(table)),('ecmwf_parameter',int(pid))]:
        m=re.search(r'Int32 '+key+r'\s+(\d+)\s*;',b);need(m and int(m[1])==want,'DAS parameter identity '+spec)
    need(attr(b,'short_name')==short and attr(b,'units')==EXPECTED_UNITS[spec],'DAS name/unit '+spec)
    expected_group=('ERA5 atmospheric pressure level analysis [netCDF4]' if pressure else
        'ERA5 atmospheric vertical integrals (ECMWF Table 162) [netCDF4]' if table=='162' else
        'ERA5 atmospheric surface analysis [netCDF4]')
    need(attr(b,'rda_dataset_group')==expected_group,'DAS analysis group')
    need('0.25 degree x 0.25 degree' in attr(b,'grid_specification') and '721 x 1440' in attr(b,'grid_specification'),'DAS grid')
    units={'time':'hours since 1900-01-01 00:00:00','latitude':'degrees_north','longitude':'degrees_east'}
    if pressure:units['level']='hPa'
    for a,u in units.items():need(a in blocks and attr(blocks[a],'units')==u,'DAS coordinate units '+a)
    need(attr(blocks['time'],'calendar')=='gregorian','DAS calendar')
    fill=[float(v) for v in re.findall(r'(?:_FillValue|missing_value)\s+([-+\d.eE]+)\s*;',b)]
    return dict(variable=var,parameter=int(pid),table=int(table),units=EXPECTED_UNITS[spec],
        long_name=attr(b,'long_name'),missing_values=sorted(set(fill)))

def parse_ascii(body):
    pieces=re.split(r'\n-{5,}\n',body,maxsplit=1);need(len(pieces)==2,'Not DAP ASCII')
    lines=pieces[1].splitlines();sections={};pos=0
    heading=re.compile(r'^([\w.]+)((?:\[\d+\])+)$')
    while pos<len(lines):
        line=lines[pos].strip();pos+=1
        if not line:continue
        m=heading.fullmatch(line);need(m is not None,'Bad DAP heading')
        name=m[1];shape=tuple(int(x) for x in re.findall(r'\d+',m[2]))
        need(name not in sections and all(n>0 for n in shape),'Duplicate/empty DAP array')
        a=np.full(shape,np.nan);seen=np.zeros(shape,bool);one=[]
        while pos<len(lines):
            line=lines[pos].strip()
            if not line:pos+=1;break
            if heading.fullmatch(line):break
            pos+=1
            if len(shape)==1:
                need(not line.startswith('['),'Indexed coordinate');one.extend(float(x.strip()) for x in line.split(','))
            else:
                row=re.fullmatch(r'((?:\[\d+\])+),\s*(.*)',line);need(row is not None,'DAP row')
                ix=tuple(int(x) for x in re.findall(r'\d+',row[1]));v=[float(x.strip()) for x in row[2].split(',')]
                need(len(ix)==len(shape)-1 and len(v)==shape[-1],'DAP row dimensions')
                need(all(i<n for i,n in zip(ix,shape)) and not seen[ix].any(),'DAP bounds/duplicate')
                a[ix]=v;seen[ix]=True
        if len(shape)==1:
            need(len(one)==shape[0],'DAP vector length');a[:]=one;seen[:]=True
        need(seen.all(),'Truncated DAP response');sections[name]=a
    return sections

def height(level):
    wanted=level*100.;p=101325.;t=288.15;g=9.80665;r=8314.32;m=28.9644;earth=6356660.
    for lo,hi,lapse in zip((0,11000,20000,32000),(11000,20000,32000,47000),(-.0065,0,.001,.0028)):
        tn=t+lapse*(hi-lo)
        pn=p*math.exp(-g*m*(hi-lo)/(r*t)) if lapse==0 else p*(tn/t)**(-g*m/(r*lapse))
        if pn<=wanted<=p:
            h=lo-r*t/(g*m)*math.log(wanted/p) if lapse==0 else lo+t/lapse*((wanted/p)**(-r*lapse/(g*m))-1)
            return earth*h/(earth-h)
        p,t=pn,tn
    raise RuntimeError('Pressure outside fixed standard-height definition')

def geometry(data):
    old=pd.read_parquet(OLD/'geometry_checks.parquet');prov=pd.read_parquet(OLD/'field_provenance.parquet')
    want=sorted({int(f['feature'].split('__')[0][1:]) for f in read(OUT/'required_fields_38.json') if not f['available_all_5664'] and f['feature'].startswith('p')})
    xyz=Transformer.from_crs(4979,4978,always_xy=True);geo=Transformer.from_crs(4978,4979,always_xy=True)
    rows=[]
    for link in read(GEOM)['links']:
        station=link['id'];o=np.array(xyz.transform(link['longitude_deg_e'],link['latitude_deg_n'],link['height_amsl_m']))
        s=np.array(xyz.transform(link['satellite_longitude_deg_e'],0,link['nominal_geo_altitude_km']*1000))
        d=(s-o)/np.linalg.norm(s-o)
        need(np.linalg.norm(o-link['origin_ecef_m'])<1e-6 and np.linalg.norm(d-link['direction_ecef'])<1e-12,'Frozen station ray')
        times=data.loc[data.station==station,'utc_epoch_seconds'].astype(int).tolist()
        fixed={}
        for lev in want:
            if lev==1:continue
            h=height(lev);need(h>link['height_amsl_m'],'Level below station')
            dist=brentq(lambda a:geo.transform(*(o+a*d))[2]-h,0,500000,xtol=1e-8)
            lon,lat,actual=geo.transform(*(o+dist*d));need(abs(actual-h)<1e-5,'LOS height residual')
            fixed[lev]=(lat,lon,h,dist)
        old_s=old[old.station==station].set_index(['utc_epoch_seconds','level_hpa'])
        for t in times:
            for lev in want:
                if lev==1:
                    record=old_s.loc[(t,lev)];lat,lon,h,dist=(float(record[x]) for x in ['latitude','longitude','height_m','slant_m'])
                else:
                    lat,lon,h,dist=fixed[lev]
                    if (t,lev) in old_s.index:
                        record=old_s.loc[(t,lev)]
                        need(abs(record.latitude-lat)<1e-8 and abs(record.longitude-lon)<1e-8 and abs(record.height_m-h)<1e-5,'Legacy fixed-height replay')
                qi=math.floor(lat*4+.5);qj=(math.floor(lon*4+.5)+720)%1440-720
                rows.append(dict(station=station,utc_epoch_seconds=t,level_hpa=lev,latitude=lat,longitude=lon,
                    height_m=h,slant_m=dist,latitude_index=360-qi,longitude_index=qj%1440,
                    basis='FROZEN_HOURLY_GEOPOTENTIAL_LOS' if lev==1 else 'FIXED_STANDARD_ATMOSPHERE_LOS'))
    result=pd.DataFrame(rows)
    check=prov[prov.level_hpa>0].merge(result,on=['station','utc_epoch_seconds','level_hpa'],suffixes=('','_new'))
    need((check.latitude_quarter_index==360-check.latitude_index).all() and
         (check.longitude_quarter_index%1440==check.longitude_index).all(),'Nearest legacy cell mismatch')
    return result

def runs(values):
    a=sorted(set(values));out=[];i=0
    while i<len(a):
        if i+1==len(a):out.append([a[i]]);break
        delta=a[i+1]-a[i];j=i+2
        while j<len(a) and a[j]-a[j-1]==delta:j+=1
        out.append(a[i:j]);i=j
    return out
def expr(a):return f'[{a[0]}]' if len(a)==1 else f'[{a[0]}:{a[1]-a[0]}:{a[-1]}]'
def node_keys(n):
    for t,lev,la,lo in itertools.product(n['epochs'],n['levels'],n['latitude_indices'],n['longitude_indices']):
        yield (n['spec'],lev,t,la,lo)

def prepare():
    need(not (ACQ/'ncar_plan.json').exists(),'Plan already frozen')
    data=pd.read_parquet(DATA);geom=geometry(data)
    p=ACQ/'geometry.parquet';need(not p.exists(),'Geometry exists');geom.to_parquet(p,index=False)
    g=geom.set_index(['station','utc_epoch_seconds','level_hpa'])
    idx=read(ACQ/'catalog_index.json');months=sorted({datetime.fromtimestamp(t,timezone.utc).strftime('%Y%m') for t in data.utc_epoch_seconds})
    for month in months:idx['e5.oper.an.sfc/'+month]=catalog((ROOT/f'Outputs/Experiments/Combined_Channel_Revision_008/catalogs/{month}.xml').read_text())
    filemap={}
    for gm,entries in idx.items():
        if gm.startswith('e5.oper.fc'):continue
        for e in entries:
            m=re.search(r'\.((?:128|162)_\d{3}_\w+)\.ll025sc\.(\d{10})_(\d{10})\.nc$',e['name'])
            if m and m[1] in SPECS.values():
                start=int(datetime.strptime(m[2],'%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp())
                end=int(datetime.strptime(m[3],'%Y%m%d%H').replace(tzinfo=timezone.utc).timestamp())
                key=(gm.split('/')[0],m[1],m[2][:8] if '.pl/' in gm else m[2][:6])
                need(key not in filemap,'Ambiguous source catalog');filemap[key]=(e['urlPath'],start,end)
    records=[];demands=set();filegroups=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    surfaces={n['station']:(n['latitude_index'],n['longitude_index']) for n in read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/plan.json')['nodes']}
    fields=[r['feature'] for r in read(OUT/'required_fields_38.json') if not r['available_all_5664'] and r['feature'].split('__')[1] in SPECS]
    need(len(fields)==24,'24 NCAR fields')
    for row in data[['station','utc_epoch_seconds']].itertuples(index=False):
        station,t=row.station,int(row.utc_epoch_seconds);dt=datetime.fromtimestamp(t,timezone.utc)
        for feature in fields:
            prefix,v=feature.split('__');lev=int(prefix[1:]) if prefix.startswith('p') else 0;spec=SPECS[v]
            if lev:
                rr=g.loc[(station,t,lev)];la,lo=int(rr.latitude_index),int(rr.longitude_index);group='e5.oper.an.pl';keydate=dt.strftime('%Y%m%d')
            else:
                la,lo=surfaces[station];group='e5.oper.an.vinteg' if spec.startswith('162_') else 'e5.oper.an.sfc';keydate=dt.strftime('%Y%m')
            path,start,end=filemap[(group,spec,keydate)];need(start<=t<=end and (t-start)%3600==0,'Catalog UTC coverage')
            identity=(spec,lev,t,la,lo);demands.add(identity)
            records.append(dict(station=station,utc_epoch_seconds=t,feature=feature,spec=spec,level_hpa=lev,latitude_index=la,longitude_index=lo))
            filegroups[(spec,path,start,lev>0)][(la,lo)][lev].add((t-start)//3600)
    pd.DataFrame(records).to_parquet(ACQ/'ncar_demands.parquet',index=False)
    nodes=[]
    for (spec,path,start,pressure),cells in sorted(filegroups.items()):
        bylayout=defaultdict(set)
        for cell,levels in cells.items():
            bymask=defaultdict(list)
            for lev,times in levels.items():bymask[tuple(sorted(times))].append(lev)
            for times,levs in bymask.items():bylayout[(times,tuple(sorted(levs)))].add(cell)
        for (times,levs),points in bylayout.items():
            bylon=defaultdict(list)
            for la,lo in points:bylon[lo].append(la)
            levelruns=runs(LEVELS.index(l) for l in levs) if pressure else [[0]]
            for tt,ll,(lo,las) in itertools.product(runs(times),levelruns,sorted(bylon.items())):
                for aa in runs(las):
                    levels=[LEVELS[i] for i in ll] if pressure else [0];var=spec.split('_')[2].upper()
                    axes=[tt,ll,aa,[lo]] if pressure else [tt,aa,[lo]]
                    node=dict(spec=spec,url_path=path,start=start,pressure=pressure,axes=axes,
                        levels=levels,epochs=[start+3600*x for x in tt],latitude_indices=aa,longitude_indices=[lo],
                        variable=var,url=HOST+'dodsC/'+path+'.ascii?'+var+''.join(expr(a) for a in axes))
                    node['id']=fp(node)[:24];nodes.append(node)
    observed=Counter(k for n in nodes for k in node_keys(n));need(observed==Counter(demands),'Exact packing no added/omitted/duplicate grid samples')
    source_paths=[DATA,GEOM,OLD/'geometry_checks.parquet',OLD/'field_provenance.parquet',OUT/'required_fields_38.json',ACQ/'catalog_index.json',ACQ/'ncar_demands.parquet',ACQ/'geometry.parquet']
    source_paths+=list(CODE.glob('*.py'))+list(CODE.glob('*.md'))
    plan=dict(utc=now(),nodes=nodes,field_count=24,station_hour_values=len(records),unique_grid_values=len(demands),
        source_sha256={rel(p):sha(p) for p in source_paths},geometry='Frozen hourly LOS at 1hPa; fixed standard-height LOS at other required levels',
        no_training=True,exact_packing_qa='PASS')
    emit(ACQ/'ncar_plan.json',plan);print(json.dumps({k:v for k,v in plan.items() if k not in ('nodes','source_sha256')}),flush=True)
    print('REQUEST_NODES',len(nodes),'SOURCE_FILES',len(filegroups),flush=True)

def decode(text,das,n):
    meta=contract(das,n['spec'],n['pressure']);var=n['variable'];a=parse_ascii(text)
    need(n['url_path'] in text.split('--------------------------------')[0],'DAP source path identity')
    coords={'time':[(t-E1900)//3600 for t in n['epochs']],
            'latitude':[90-i*.25 for i in n['latitude_indices']],
            'longitude':[i*.25 for i in n['longitude_indices']]}
    if n['pressure']:coords['level']=n['levels']
    need(set(a)=={var+'.'+var}|{var+'.'+x for x in coords},'Unexpected DAP variables')
    for key,value in coords.items():need(np.array_equal(a[var+'.'+key],value),'DAP coordinate/time/level '+key)
    value=a[var+'.'+var]
    need(value.shape==tuple(len(x) for x in n['axes']),'DAP array shape')
    need(np.isfinite(value).all() and (np.abs(value)<1e19).all(),'NCAR nonfinite/fill')
    for f in meta['missing_values']:need(not (value==f).any(),'NCAR missing value')
    return value.ravel().tolist()

def acquire(limit):
    p=read(ACQ/'ncar_plan.json')
    for path,digest in p['source_sha256'].items():need(sha(ROOT/path)==digest,'Frozen acquisition source SHA '+path)
    f=Fetcher();done=0;tic=time.monotonic();cds_time=0
    from cds_three import step
    for n in p['nodes']:
        if limit and done>=limit:break
        fileid=fp(n['url_path'])[:24];das,_=f.get(HOST+'dodsC/'+n['url_path']+'.das',f'ncar_metadata/{fileid}.das',lambda s:contract(s,n['spec'],n['pressure']))
        text,values=f.get(n['url'],f'ncar_values/{n["id"]}.ascii',lambda s:decode(s,das,n))
        done+=1
        if done%10==0:
            state=dict(utc=now(),pid=os.getpid(),stage='ACQUIRING',validated_ncar_nodes=done,total_ncar_nodes=len(p['nodes']),
                current_file=n['url_path'],elapsed_seconds=time.monotonic()-tic)
            temp=OUT/'acquisition_progress.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');os.replace(temp,OUT/'acquisition_progress.json')
            print(json.dumps(state),flush=True)
        if time.monotonic()-cds_time>60:
            print('CDS',json.dumps(step()),flush=True);cds_time=time.monotonic()
    print('NCAR_VALIDATED_NODES',done,'OF',len(p['nodes']),flush=True)
    if done==len(p['nodes']):emit(ACQ/'ncar_download_complete.json',dict(utc=now(),nodes=done,plan_sha256=sha(ACQ/'ncar_plan.json'),status='DOWNLOADED_NOT_YET_INDEPENDENT_INPUT_QA'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','acquire']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    with stage_lock('ncar_'+args.action):
        if args.action=='plan':prepare()
        else:acquire(args.limit)
