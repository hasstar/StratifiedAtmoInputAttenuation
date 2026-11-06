"""Bounded TCC completion on existing UK prediction keys. No model execution."""
from __future__ import annotations
import calendar
import contextlib
import hashlib
import itertools
import json
import math
import msvcrt
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_008'
R2 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_002'
R3 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003'
R6 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_006'
GBS = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
STOPDIR = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011'
AUTOMATION = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
HOST = 'https://tds.gdex.ucar.edu/thredds/'
EPOCH1900 = int(datetime(1900, 1, 1, tzinfo=timezone.utc).timestamp())
RAIN = ['mean_total_precipitation_rate', 'mean_snowfall_rate']

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def key(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

def emit(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

@contextlib.contextmanager
def lock():
    with (OUT / 'execution.lock').open('a+b') as f:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

def snapshot():
    selection = read(R2 / 'frozen_selection.json')
    rain = pd.read_parquet(R3 / 'field_provenance.parquet')
    rain = rain[rain.name.isin(RAIN)]
    prior = read(R6 / 'source_sha256_before.json')
    weights = {p: h for p, h in prior.items() if 'Revision_006\\weights\\' in p}
    need(len(weights) == 2, 'Expected two frozen author checkpoints')
    sources = [R6 / n for n in ['predictions_uk_all.parquet', 'predictions_local_all.parquet', 'delivery_manifest.json', 'metrics.json', 'source_sha256_before.json', 'independent_qa.json']]
    sources += [R3 / n for n in ['expanded_dataset.parquet', 'field_provenance.parquet', 'raw_source_evidence.json', 'field_qa.json', 'independent_qa.json', 'dataset_complete.json', 'source_sha256_used.json']]
    sources += [R2 / 'frozen_selection.json', GBS / 'scope_manifest.json', GBS / 'conditional_fast_001/weather_labels.parquet', ROOT / 'Outputs/Experiments/Combined_Channel_Revision_007/run_scores.json']
    sources += [STOPDIR / 'STOP_REQUESTED.json', STOPDIR / 'active_worker.json', AUTOMATION]
    sources += [R2 / m['path'] for m in selection['models'] if 'path' in m]
    sources += [ROOT / p for p in weights]
    sources += [ROOT / p for p in rain.source.unique()]
    sources += sorted(p for p in CODE.iterdir() if p.is_file())
    before = {key(p): sha(p) for p in sorted(set(sources))}
    manifest = read(R6 / 'delivery_manifest.json')
    for n in ['predictions_uk_all.parquet', 'predictions_local_all.parquet', 'metrics.json']:
        need(before[key(R6 / n)] == manifest[key(R6 / n)]['sha256'], 'Delivered prediction/metric SHA mismatch')
    need(before[key(R2 / 'frozen_selection.json')] == prior[key(R2 / 'frozen_selection.json')], 'Selection SHA mismatch')
    for m in selection['models']:
        if 'path' in m:
            need(before[key(R2 / m['path'])] == m['sha256'], 'Historical checkpoint changed')
    for p, h in weights.items():
        need(before[p] == h, 'Author weight changed')
    for p, rows in rain.groupby('source'):
        need(rows.source_sha256.nunique() == 1 and before[p] == rows.iloc[0].source_sha256, 'Rain payload changed')
    need(read(STOPDIR / 'active_worker.json')['status'] == 'EXITED_NORMAL', 'Old worker not exited')
    need(re.search(r'^status\s*=\s*"PAUSED"', AUTOMATION.read_text(encoding='utf-8'), re.M), 'Old automation not paused')
    need(read(R3 / 'field_qa.json')['status'] == 'PASS' and read(R3 / 'independent_qa.json')['status'] == 'PASS', 'Prior source QA missing')
    return before

def parse_ascii(body):
    pieces = re.split(r'\r?\n-{5,}\r?\n', body, maxsplit=1)
    need(len(pieces) == 2, 'Not DAP2 ASCII')
    lines = pieces[1].splitlines()
    heading = re.compile(r'^([\w.]+)((?:\[\d+\])+)$')
    sections = {}
    pos = 0
    while pos < len(lines):
        line = lines[pos].strip()
        pos += 1
        if not line:
            continue
        match = heading.fullmatch(line)
        need(match is not None, 'Invalid DAP array heading')
        name = match[1]
        shape = tuple(map(int, re.findall(r'\d+', match[2])))
        need(name not in sections and all(n > 0 for n in shape), 'Duplicate/empty array')
        array = np.full(shape, np.nan)
        occupied = np.zeros(shape, bool)
        one = []
        while pos < len(lines):
            line = lines[pos].strip()
            if not line:
                pos += 1
                break
            if heading.fullmatch(line):
                break
            pos += 1
            if len(shape) == 1:
                need(not line.startswith('['), 'Indexed 1D coordinate')
                one += [float(s.strip()) for s in line.split(',')]
            else:
                row = re.fullmatch(r'((?:\[\d+\])+),\s*(.*)', line)
                need(row is not None, 'Missing multidimensional row index')
                idx = tuple(map(int, re.findall(r'\d+', row[1])))
                values = [float(s.strip()) for s in row[2].split(',')]
                need(len(idx) == len(shape)-1 and len(values) == shape[-1], 'Wrong DAP row shape')
                need(all(i < n for i, n in zip(idx, shape)) and not occupied[idx].any(), 'Duplicate/out-of-range DAP row')
                array[idx] = values
                occupied[idx] = True
        if len(shape) == 1:
            need(len(one) == shape[0], 'Truncated coordinate')
            array[:] = one
            occupied[:] = True
        need(occupied.all() and np.isfinite(array).all(), 'Incomplete/nonfinite DAP array')
        sections[name] = array
    return sections

def contract(das):
    def block(name):
        m = re.search(r'\b' + name + r'\s*\{([^{}]*)\}', das, re.S)
        need(m is not None, 'Missing metadata block ' + name)
        return m[1]
    t = block('TCC')
    for pattern in [r'ecmwf_local_table\s+128\s*;', r'ecmwf_parameter\s+164\s*;', r'String units\s+"\(0-1\)";', r'String long_name\s+"Total cloud cover";']:
        need(re.search(pattern, t) is not None, 'Wrong TCC metadata: ' + pattern)
    for n, u in [('latitude','degrees_north'),('longitude','degrees_east'),('time','hours since 1900-01-01 00:00:00')]:
        need(re.search(r'String units\s+"' + re.escape(u) + r'";', block(n)) is not None, 'Wrong coordinate/UTC unit')
    need('gregorian' in block('time'), 'Wrong calendar')
    return {'table':128, 'parameter':164, 'units':'(0-1)', 'time':'instant UTC t', 'grid_step_deg':.25}

def decode(body, das, node):
    contract(das)
    need(node['url_path'] in body.split('--------------------------------')[0], 'Wrong source dataset in DAP header')
    data = parse_ascii(body)
    need(set(data) == {'TCC.TCC','TCC.time','TCC.latitude','TCC.longitude'}, 'Extra/missing variables')
    for axis, values in [('time', [(t-EPOCH1900)//3600 for t in node['epochs']]), ('latitude',[node['latitude']]), ('longitude',[node['longitude_0_360']])]:
        need(np.array_equal(data['TCC.'+axis], values), 'Coordinate/time mismatch ' + axis)
    a = data['TCC.TCC']
    need(a.shape == (len(node['epochs']), 1, 1), 'Wrong subset shape')
    need(((a >= 0) & (a <= 1)).all(), 'TCC fill/outside 0..1')
    return a.ravel().tolist()

class Fetcher:
    def __init__(self):
        self.s = requests.Session()
        self.s.trust_env = False
        self.s.headers.update({'Accept-Encoding':'identity', 'User-Agent':'T012-bounded-cloud-completion/008'})
        self.success = []

    def get(self, url, relative, validator):
        need(url.startswith(HOST+'catalog/files/g/d633000/e5.oper.an.sfc/') or url.startswith(HOST+'dodsC/files/d633000/e5.oper.an.sfc/'), 'URL outside scoped public archive')
        need(url.endswith('catalog.xml') or url.endswith('.das') or '.nc.ascii?TCC[' in url, 'Unconstrained/global request blocked')
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        need(not path.exists(), 'No overwrite/redundant retrieval')
        for attempt in range(1, 4):
            part = OUT / 'partial_history' / (path.name + f'.attempt{attempt}.part')
            part.parent.mkdir(parents=True, exist_ok=True)
            response = None
            started = utc()
            try:
                self.s.cookies.clear()
                response = self.s.get(url, timeout=(10,180), allow_redirects=False, stream=True)
                headers = dict(response.headers)
                if response.status_code != 200:
                    record = dict(url=url,started_utc=started,finished_utc=utc(),status=response.status_code,headers=headers,attempt=attempt)
                    emit('http/'+path.name+f'.attempt{attempt}.json',record)
                    if response.status_code in [408,429] or 500 <= response.status_code < 600:
                        if attempt == 3:
                            raise RuntimeError('Bounded HTTP retries exhausted')
                        delay = min(60, 5*attempt)
                        ra = response.headers.get('Retry-After')
                        if ra:
                            try:
                                delay = min(60,max(delay,float(ra)))
                            except ValueError:
                                delay = min(60,max(delay,(parsedate_to_datetime(ra)-datetime.now(timezone.utc)).total_seconds()))
                        response.close()
                        time.sleep(delay)
                        continue
                    raise RuntimeError('Nonretryable HTTP ' + str(response.status_code))
                length = headers.get('Content-Length')
                need(length is None or (0 < int(length) <= 16*1024*1024), 'Invalid/excessive advertised length')
                size = 0
                with part.open('xb') as f:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        need(size <= 16*1024*1024, 'Response exceeds 16MiB')
                        f.write(chunk)
                need(size > 0 and (length is None or size == int(length)), 'Content-Length/actual mismatch')
                body = part.read_text(encoding='utf-8')
                decoded = validator(body)
                digest = sha(part)
                part.rename(path)
                record = dict(url=url,request_method='GET',started_utc=started,finished_utc=utc(),status=200,headers=headers,attempt=attempt,actual_bytes=size,content_length=None if length is None else int(length),sha256=digest,path=key(path),validation='PASS')
                emit('http/'+path.name+'.json',record)
                self.success.append(record)
                print(json.dumps({'acquired':relative,'bytes':size,'utc':utc()}),flush=True)
                return body, decoded
            except (requests.RequestException, OSError) as exc:
                emit('failures/'+path.name+f'.attempt{attempt}.json',dict(utc=utc(),url=url,error=repr(exc),partial_exists=part.exists()))
                if attempt == 3:
                    raise
                time.sleep(5*attempt)
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError('No complete payload')

def acquire():
    need(not OUT.exists(), 'Output exists; no automatic rerun')
    OUT.mkdir(parents=True,exist_ok=False)
    started = utc()
    with lock():
        try:
            before = snapshot()
            emit('sources_before.json',before)
            uk = pd.read_parquet(R6/'predictions_uk_all.parquet')
            need(len(uk)==5664 and not uk.duplicated(['station','utc_epoch_seconds']).any(), 'Wrong UK scope')
            scope = read(GBS/'scope_manifest.json')
            stations = {s['id']:s for s in scope['links']}
            months = sorted(set(pd.to_datetime(uk.utc_epoch_seconds,unit='s',utc=True).dt.strftime('%Y%m')))
            need(months == ['200308','200309','200310','200311','200402','200403'], 'Unexpected scope months')
            fetch = Fetcher()
            metadata = {}
            for ym in months:
                year,month = int(ym[:4]),int(ym[4:])
                filename = f'e5.oper.an.sfc.128_164_tcc.ll025sc.{ym}0100_{ym}{calendar.monthrange(year,month)[1]:02d}23.nc'
                def catalog_validator(body):
                    tree = ET.fromstring(body)
                    hits=[x.attrib for x in tree.iter() if x.tag.endswith('dataset') and x.attrib.get('name')==filename and x.attrib.get('urlPath')]
                    need(len(hits)==1,'TCC catalog identity not unique')
                    p=hits[0]['urlPath']
                    need(p==f'files/d633000/e5.oper.an.sfc/{ym}/{filename}', 'Unexpected archive path')
                    return p
                _,path = fetch.get(HOST+f'catalog/files/g/d633000/e5.oper.an.sfc/{ym}/catalog.xml',f'catalogs/{ym}.xml',catalog_validator)
                das,info = fetch.get(HOST+'dodsC/'+path+'.das',f'metadata/{ym}.das',contract)
                metadata[ym] = dict(url_path=path,das=das,contract=info)
            nodes=[]
            for station,g in uk.groupby('station'):
                need(len(g)=={'chilbolton':3432,'sparsholt':2232}[station], 'Station scope changed')
                s=stations[station]
                lat=math.floor(s['latitude_deg_n']*4+.5)/4
                lon=math.floor(s['longitude_deg_e']*4+.5)/4
                for ym in months:
                    epochs=sorted(int(t) for t in g.utc_epoch_seconds if datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y%m')==ym)
                    for _,group in itertools.groupby(enumerate(epochs),lambda x:x[1]-x[0]*3600):
                        hours=[v for _,v in group]
                        start=int(datetime(int(ym[:4]),int(ym[4:]),1,tzinfo=timezone.utc).timestamp())
                        i0,i1=(hours[0]-start)//3600,(hours[-1]-start)//3600
                        node=dict(id=f'{station}_{ym}_{i0:04d}_{i1:04d}',station=station,month=ym,epochs=hours,latitude=lat,longitude_0_360=lon%360,latitude_index=int((90-lat)*4),longitude_index=int((lon%360)*4),url_path=metadata[ym]['url_path'])
                        node['url']=HOST+'dodsC/'+node['url_path']+f'.ascii?TCC[{i0}:1:{i1}][{node["latitude_index"]}][{node["longitude_index"]}]'
                        nodes.append(node)
            keys=[(n['station'],t) for n in nodes for t in n['epochs']]
            need(len(keys)==5664 and len(set(keys))==5664 and set(keys)==set(zip(uk.station,uk.utc_epoch_seconds)), 'Request plan not exact')
            need(len(nodes)==12, 'Unexpected number of bounded subsets')
            emit('plan.json',dict(created_utc=utc(),nodes=nodes,station_hours=len(keys),new_field='TCC',max_concurrency=1,old_download_resume=False))
            rows=[]
            for node in nodes:
                body,values=fetch.get(node['url'],'payloads/'+node['id']+'.ascii',lambda b:decode(b,metadata[node['month']]['das'],node))
                p=OUT/'payloads'/ (node['id']+'.ascii')
                rows += [dict(station=node['station'],utc_epoch_seconds=t,tcc=v,tcc_valid_utc_epoch_seconds=t,latitude=node['latitude'],longitude_0_360=node['longitude_0_360'],source=key(p),source_sha256=sha(p)) for t,v in zip(node['epochs'],values)]
            cloud=pd.DataFrame(rows).sort_values(['station','utc_epoch_seconds']).reset_index(drop=True)
            need(len(cloud)==5664 and not cloud.duplicated(['station','utc_epoch_seconds']).any(),'Incomplete cloud coverage')
            cloud.to_parquet(OUT/'cloud_all.parquet',index=False)
            after={p:sha(ROOT/p) for p in before}
            need(after==before,'Protected input changed during acquisition')
            emit('sources_after_acquisition.json',after)
            emit('acquisition_complete.json',dict(status='PASS',started_utc=started,completed_utc=utc(),cloud_station_hours=len(cloud),cloud_values=5664,subset_requests=len(nodes),metadata_requests=len(metadata),catalog_requests=len(metadata),successful_requests=len(fetch.success),actual_bytes=sum(r['actual_bytes'] for r in fetch.success),cloud_sha256=sha(OUT/'cloud_all.parquet'),protected_sources_unchanged=len(before),execution_lock='RELEASED',training=0,model_inference=0,old_downloader_resumed=False))
        except Exception as exc:
            emit('acquisition_failure.json',dict(utc=utc(),error=repr(exc),execution_lock='RELEASED_AFTER_FAILURE',partial_files_preserved=True))
            raise

if __name__=='__main__':
    need(sys.argv[1:]==['acquire'],'Only explicit acquire mode allowed')
    acquire()
