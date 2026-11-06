"""Exact NCAR d633000 OPeNDAP subsets: Top10 only, no interpolation.

Pure request planning and strict ASCII/DAS decoding. HTTP and durable state live
in the versioned driver. No CDS request, remote write, model, or training calls.
"""
from __future__ import annotations
import calendar
import itertools
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import numpy as np
import fixedheight_los_core_v001 as core

BASE = 'https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/'
UTC = timezone.utc
EPOCH1900 = int(datetime(1900,1,1,tzinfo=UTC).timestamp())
LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]
FIELDS = {
    'geopotential': ('Z',129,'z','ll025sc',('m**2 s**-2','m2 s-2')),
    'u_component_of_wind': ('U',131,'u','ll025uv',('m s**-1','m s-1')),
    'ozone_mass_mixing_ratio': ('O3',203,'o3','ll025sc',('kg kg**-1','kg kg-1')),
    'relative_humidity': ('R',157,'r','ll025sc',('%',)),
    core.SURFACE[0]: ('CAPE',59,'cape','ll025sc',('J kg**-1','J kg-1','m**2 s**-2')),
    core.SURFACE[1]: ('MX2T',201,'mx2t','ll025sc',('K',)),
}

def check(ok, message):
    if not ok:
        raise ValueError(message)

def stamp(dt):
    return dt.strftime('%Y%m%d%H')

def locate(variable, level, epoch):
    """Map valid UTC to archived file/index, including MX2T month boundaries."""
    check(core.allowed_pair(variable, level), 'Variable/level outside natural Top10')
    check(isinstance(epoch,int) and epoch % 3600 == 0, 'Non-hourly UTC')
    dt = datetime.fromtimestamp(epoch,UTC)
    nc,pid,short,grid,_ = FIELDS[variable]
    if level:
        start = dt.replace(hour=0)
        end = start.replace(hour=23)
        group = 'e5.oper.an.pl'
        shape = 'pressure'
        temporal = (dt.hour,)
    elif nc == 'CAPE':
        start = dt.replace(day=1,hour=0)
        end = start.replace(day=calendar.monthrange(dt.year,dt.month)[1],hour=23)
        group = 'e5.oper.an.sfc'
        shape = 'surface'
        temporal = ((epoch-int(start.timestamp()))//3600,)
    else:
        # The archive has forecasts initialized at 06/18 UTC, steps 1..12.
        # Strictly preceding initialization gives exactly one valid-hour value.
        initial = 21600 + ((epoch-21600-3600)//43200)*43200
        initdt = datetime.fromtimestamp(initial,UTC)
        start = initdt.replace(day=1 if initdt.day<16 else 16,hour=6)
        if start.day == 1:
            end = start.replace(day=16)
        else:
            end = (start.replace(day=28)+timedelta(days=4)).replace(day=1)
        group = 'e5.oper.fc.sfc.minmax'
        shape = 'forecast'
        temporal = ((initial-int(start.timestamp()))//43200,(epoch-initial)//3600-1)
        check(0 <= temporal[1] < 12, 'Forecast lead outside 1..12')
    file = f'{group}/{start:%Y%m}/{group}.128_{pid:03d}_{short}.{grid}.{stamp(start)}_{stamp(end)}.nc'
    return file,shape,int(start.timestamp()),temporal

def runs(values):
    """Disjoint arithmetic runs, no unrequested gap members."""
    a=sorted(set(values)); out=[]; n=0
    while n<len(a):
        if n+1==len(a):
            out.append([a[n]]);break
        step=a[n+1]-a[n]; end=n+2
        while end<len(a) and a[end]-a[end-1]==step:
            end+=1
        out.append(a[n:end]);n=end
    return out

def expression(indices):
    check(bool(indices), 'Empty index')
    if len(indices)==1:
        return f'[{indices[0]}]'
    step=indices[1]-indices[0]
    check(step>0 and indices==list(range(indices[0],indices[-1]+1,step)), 'Nonuniform index list')
    return f'[{indices[0]}:{step}:{indices[-1]}]'

def expected_keys(node):
    var,level=node['variable'],node['level_hpa']
    c=node['coords']; shape=node['layout']
    if shape=='forecast':
        times=[int(EPOCH1900+3600*(a+b)) for a in c['forecast_initial_time'] for b in c['forecast_hour']]
    else:
        times=[int(EPOCH1900+3600*a) for a in c['time']]
    for t,lat,lon in itertools.product(times,c['latitude'],c['longitude']):
        yield (var,level,t,round(lat/.25),round(((lon+180)%360-180)/.25))

def make_node(variable,level,file,layout,start,temporal,lats,lons):
    nc=FIELDS[variable][0]
    if layout=='pressure':
        axes=[temporal[0],[LEVELS.index(level)],lats,lons]
        names=['time','level','latitude','longitude']
        coords={'time':[(start-EPOCH1900)//3600+t for t in temporal[0]],'level':[level]}
    elif layout=='surface':
        axes=[temporal[0],lats,lons];names=['time','latitude','longitude']
        coords={'time':[(start-EPOCH1900)//3600+t for t in temporal[0]]}
    else:
        axes=[temporal[0],temporal[1],lats,lons]
        names=['forecast_initial_time','forecast_hour','latitude','longitude']
        coords={'forecast_initial_time':[(start-EPOCH1900)//3600+12*t for t in temporal[0]],'forecast_hour':[t+1 for t in temporal[1]]}
    coords.update(latitude=[90-i*.25 for i in lats],longitude=[j*.25 for j in lons])
    url=BASE+file+'.ascii?'+nc+''.join(expression(a) for a in axes)
    return dict(variable=variable,level_hpa=level,nc_variable=nc,file=file,layout=layout,axes=axes,dimensions=names,coords=coords,url=url,
                metadata_url=BASE+file+'.das',expected_values=math.prod(len(a) for a in axes))

def plan_requests(demands):
    """Exact field/time/cell sets, split only by physical NCAR archive layout."""
    demands=set(demands); groups=defaultdict(lambda:defaultdict(set))
    for variable,level,t,i,j in sorted(demands):
        check(-360<=i<=360 and -720<=j<720, 'Invalid quarter-degree grid cell')
        file,layout,start,temporal=locate(variable,level,t)
        groups[(variable,level,file,layout,start)][(360-i,j%1440)].add(temporal)
    nodes=[]
    for (variable,level,file,layout,start),cells in sorted(groups.items()):
        by_mask=defaultdict(set)
        for cell,mask in cells.items(): by_mask[tuple(sorted(mask))].add(cell)
        for mask,points in sorted(by_mask.items()):
            # tiles works with integer coordinate indices regardless of convention.
            rectangles=core.tiles(points)
            for rectangle in rectangles:
                # Frozen tiles returns (maximum first, minimum second, minimum first, maximum second).
                bi,aj,ai,bj=rectangle
                latidx=list(range(ai,bi+1)); lonidx=list(range(aj,bj+1))
                if layout=='forecast':
                    init_leads=defaultdict(set)
                    for init,lead in mask: init_leads[init].add(lead)
                    lead_init=defaultdict(set)
                    for init,leads in init_leads.items(): lead_init[tuple(sorted(leads))].add(init)
                    temporals=[(a,b) for leads,inits in sorted(lead_init.items()) for a in runs(inits) for b in runs(leads)]
                else:
                    temporals=[(a,) for a in runs(t[0] for t in mask)]
                for temporal in temporals:
                    nodes.append(make_node(variable,level,file,layout,start,temporal,latidx,lonidx))
    observed=Counter(k for n in nodes for k in expected_keys(n))
    check(observed==Counter(demands), 'NCAR subset packing added/dropped/duplicated requested samples')
    return nodes

def parse_ascii(body):
    """Decode constrained DAP2 ASCII arrays, checking row indices and shapes."""
    pieces=re.split(r'\n-{5,}\n',body,maxsplit=1)
    check(len(pieces)==2, 'Not a DAP ASCII response')
    lines=pieces[1].splitlines(); sections={}; pos=0
    heading=re.compile(r'^([\w.]+)((?:\[\d+\])+)$')
    while pos<len(lines):
        line=lines[pos].strip();pos+=1
        if not line: continue
        match=heading.fullmatch(line);check(match is not None,'Unexpected DAP heading: '+line[:80])
        name=match[1]; shape=tuple(int(x) for x in re.findall(r'\d+',match[2]))
        check(name not in sections and all(n>0 for n in shape),'Duplicate/empty DAP array')
        array=np.full(shape,np.nan); occupied=np.zeros(shape,dtype=bool); one=[]
        while pos<len(lines):
            value_line=lines[pos].strip()
            if not value_line:
                pos+=1;break
            if heading.fullmatch(value_line): break
            pos+=1
            if len(shape)==1:
                check(not value_line.startswith('['),'Unexpected indexed coordinate line')
                one.extend(float(x.strip()) for x in value_line.split(','))
            else:
                row=re.fullmatch(r'((?:\[\d+\])+),\s*(.*)',value_line)
                check(row is not None,'Unindexed multidimensional data row')
                indices=tuple(int(x) for x in re.findall(r'\d+',row[1]))
                values=[float(x.strip()) for x in row[2].split(',')]
                check(len(indices)==len(shape)-1 and len(values)==shape[-1],'DAP row shape mismatch')
                check(all(i<n for i,n in zip(indices,shape)),'DAP row out of bounds')
                check(not occupied[indices].any(),'Duplicate DAP row')
                array[indices]=values;occupied[indices]=True
        if len(shape)==1:
            check(len(one)==shape[0],'Coordinate length mismatch');array[:]=one;occupied[:]=True
        check(occupied.all(),'Truncated DAP array')
        sections[name]=array
    return sections

def metadata_contract(body,node):
    nc=node['nc_variable']; match=re.search(r'\b'+nc+r'\s*\{([^{}]*)\}',body,re.S)
    check(match is not None,'NCAR variable metadata absent')
    text=match[1]
    pid=re.search(r'ecmwf_parameter\s+(\d+)\s*;',text)
    units=re.search(r'String units\s+"([^"]+)";',text)
    check(pid is not None and int(pid[1])==FIELDS[node['variable']][1],'Wrong NCAR parameter ID')
    check(units is not None and units[1] in FIELDS[node['variable']][4],'Wrong NCAR parameter units')
    for name in node['dimensions']:
        block=re.search(r'\b'+name+r'\s*\{([^{}]*)\}',body,re.S)
        check(block is not None,'Missing coordinate metadata '+name)
        u=re.search(r'String units\s+"([^"]+)";',block[1])
        allowed={'latitude':['degrees_north'],'longitude':['degrees_east'],'level':['hPa'],
                 'time':['hours since 1900-01-01 00:00:00'],'forecast_initial_time':['hours since 1900-01-01 00:00:00'],
                 'forecast_hour':['hour','hours']}
        check(u is not None and u[1] in allowed[name],'Unexpected time/coordinate units '+name)
    fill=re.findall(r'(?:_FillValue|missing_value)\s+([-+\d.eE]+)\s*;',text)
    return {'parameter_id':int(pid[1]),'units':units[1],'missing_values':sorted(set(float(x) for x in fill))}

def decode(body,das,node):
    contract=metadata_contract(das,node); data=parse_ascii(body);nc=node['nc_variable']
    wanted={nc+'.'+nc}|{nc+'.'+name for name in node['dimensions']}
    check(set(data)==wanted,'Unexpected/absent variables or coordinate maps')
    for name in node['dimensions']:
        actual=data[nc+'.'+name]
        check(actual.ndim==1 and np.array_equal(actual,np.asarray(node['coords'][name])),'Coordinate/time/level mismatch '+name)
    values=data[nc+'.'+nc]
    check(values.shape==tuple(len(a) for a in node['axes']),'Data shape differs from exact request')
    check(np.isfinite(values).all() and np.all(np.abs(values)<1e19),'Nonfinite/fill values in required samples')
    for fill in contract['missing_values']:
        check(not np.any(values==fill),'Missing NCAR data value')
    keys=list(expected_keys(node));flat=values.reshape(-1).tolist()
    check(len(keys)==len(flat)==node['expected_values'] and len(set(keys))==len(keys),'Field count/identity mismatch')
    return [[list(k),float(v)] for k,v in zip(keys,flat)],contract
