"""Bounded five-field acquisition; old downloaders/models are never executed."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
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
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_010'
OLD8 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_008'
OLD9 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_009'
PRED = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_006/predictions_uk_all.parquet'
DATA = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003/expanded_dataset.parquet'
GEOMETRY = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010/scope_manifest.json'
HOST = 'https://tds.gdex.ucar.edu/thredds/dodsC/'
EPOCH1900 = int(datetime(1900, 1, 1, tzinfo=timezone.utc).timestamp())
MAX_BYTES = 16*1024*1024
FIELDS = {
    '2t': dict(parameter=167, long_name='2 metre temperature', units=['K'], column='surface__2m_temperature', lower=180, upper=340),
    '2d': dict(parameter=168, long_name='2 metre dewpoint temperature', units=['K'], column='surface__2m_dewpoint_temperature', lower=130, upper=340),
    'sp': dict(parameter=134, long_name='Surface pressure', units=['Pa'], column='surface__surface_pressure', lower=10000, upper=110000),
    'tcwv': dict(parameter=137, long_name='Total column water vapour', units=['kg m**-2', 'kg m-2'], column='surface__total_column_water_vapour', lower=0, upper=200),
    'tclw': dict(parameter=78, long_name='Total column cloud liquid water', units=['kg m**-2', 'kg m-2'], column='surface__total_column_cloud_liquid_water', lower=0, upper=50),
}
spec = importlib.util.spec_from_file_location('read_only_rev008_ascii', ROOT / 'Code/Combined_Channel_Experiments/Revision_008/cloud_acquisition.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
parse_ascii = legacy.parse_ascii


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def key(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def emit(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write('\n')


@contextlib.contextmanager
def lock():
    with (OUT/'execution.lock').open('a+b') as fh:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def check_sources(manifest):
    for path, digest in manifest.items():
        need(sha(ROOT / path) == digest, 'Protected input SHA changed: ' + path)


def snapshot():
    old = read(OLD9/'sources_after.json')
    check_sources(old)
    receipt = read(OLD9/'build_receipt.json')
    for name, digest in receipt['artifacts'].items():
        need(sha(OLD9/name) == digest, 'Revision009 output changed')
        old[key(OLD9/name)] = digest
    for path in [OLD9/'build_receipt.json', OLD9/'completion.json', DATA, GEOMETRY, PRED,
                 *sorted(p for p in CODE.iterdir() if p.is_file())]:
        old[key(path)] = sha(path)
    need(read(OLD9/'completion.json')['execution_lock'] == 'RELEASED', 'Previous record not released')
    return old


def contract(text, field):
    f = FIELDS[field]
    blocks = dict(re.findall(r'\b([A-Za-z_]\w*)\s*\{([^{}]*)\}', text, re.S))
    matches = [name for name, body in blocks.items() if re.search(r'Int32 ecmwf_parameter\s+'+str(f['parameter'])+r'\s*;', body)]
    need(len(matches) == 1, 'Parameter absent/ambiguous ' + field)
    var = matches[0]
    block = blocks[var]
    need(re.search(r'Int32 ecmwf_local_table\s+128\s*;', block), 'Wrong parameter table')
    def attr(body, name):
        match = re.search(r'String '+name+r'\s+"([^"]+)"\s*;', body)
        need(match is not None, 'Missing attribute ' + name)
        return match[1]
    need(attr(block, 'short_name') == field, 'Wrong short_name')
    need(attr(block, 'long_name').lower() == f['long_name'].lower(), 'Wrong long_name')
    units = attr(block, 'units')
    need(units in f['units'], 'Wrong units for ' + field + ': ' + units)
    grid = attr(block, 'grid_specification')
    need('0.25 degree x 0.25 degree' in grid and '721 x 1440' in grid, 'Unexpected published grid')
    for name, unit in [('latitude','degrees_north'), ('longitude','degrees_east'), ('time','hours since 1900-01-01 00:00:00')]:
        need(name in blocks and attr(blocks[name], 'units') == unit, 'Wrong coordinate units: ' + name)
    need(attr(blocks['time'], 'calendar') == 'gregorian', 'Wrong time calendar')
    need(attr(block, 'rda_dataset_group') == 'ERA5 atmospheric surface analysis [netCDF4]', 'Not instantaneous surface analysis')
    return dict(variable=var, table=128, parameter=f['parameter'], short_name=field,
                long_name=attr(block,'long_name'), units=units, grid=grid, valid_time='UTC t instantaneous analysis')


def decode(text, das, node):
    metadata = contract(das, node['field'])
    var = metadata['variable']
    need(node['url_path'] in text.split('--------------------------------')[0], 'DAP source identity mismatch')
    arrays = parse_ascii(text)
    need(set(arrays) == {var+'.'+x for x in (var,'time','latitude','longitude')}, 'Unexpected/missing arrays')
    for axis, values in [('time',[(t-EPOCH1900)//3600 for t in node['epochs']]),
                         ('latitude',[node['latitude']]), ('longitude',[node['longitude_0_360']])]:
        need(np.array_equal(arrays[var+'.'+axis], values), 'Wrong coordinate ' + axis)
    values = arrays[var+'.'+var]
    need(values.shape == (len(node['epochs']),1,1), 'Wrong payload shape')
    f = FIELDS[node['field']]
    need(np.isfinite(values).all() and ((values >= f['lower']) & (values <= f['upper'])).all(), 'Nonfinite/fill/outside broad physical bounds')
    return values.ravel().tolist()


class Fetcher:
    def __init__(self, allowlist):
        self.allowed = set(allowlist)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({'User-Agent':'T012-bounded-ITU-inputs/010','Accept-Encoding':'identity'})
        self.run_id = stamp()

    def get(self, url, relative, validator):
        need(url in self.allowed and url.startswith(HOST+'files/d633000/e5.oper.an.sfc/'), 'URL outside exact request allowlist')
        path = OUT / relative
        receipt_path = OUT / 'http' / (path.name + '.json')
        if path.exists():
            receipt = read(receipt_path)
            need(receipt['url'] == url and receipt['status'] == 200, 'Cached URL/response identity mismatch')
            need(sha(path) == receipt['sha256'] and path.stat().st_size == receipt['actual_bytes'], 'Cached SHA/size mismatch')
            need(receipt['content_length'] is None or receipt['content_length'] == path.stat().st_size, 'Cached HTTP length mismatch')
            text = path.read_text(encoding='utf-8')
            return text, validator(text)
        need(not receipt_path.exists(), 'Orphan success receipt')
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1,4):
            need(not (OUT/'STOP_REQUESTED.json').exists(), 'New bounded acquisition stopped by user')
            part = OUT/'partial_history'/(path.name+f'.{self.run_id}.attempt{attempt}.part')
            part.parent.mkdir(parents=True, exist_ok=True)
            response = None
            started, tic = utc(), time.monotonic()
            try:
                self.session.cookies.clear()
                response = self.session.get(url, stream=True, allow_redirects=False, timeout=(10,180))
                headers = dict(response.headers)
                if response.status_code != 200:
                    emit('http_failures/'+path.name+f'.{self.run_id}.attempt{attempt}.json', dict(url=url,started_utc=started,finished_utc=utc(),status=response.status_code,headers=headers))
                    retryable = response.status_code in (408,429) or 500 <= response.status_code < 600
                    need(retryable and attempt < 3, 'Blocked HTTP '+str(response.status_code))
                    delay = min(60, 5*attempt)
                    ra = headers.get('Retry-After')
                    if ra:
                        try:
                            ra_seconds = float(ra)
                        except ValueError:
                            ra_seconds = (parsedate_to_datetime(ra)-datetime.now(timezone.utc)).total_seconds()
                        delay = min(60,max(delay,ra_seconds))
                    response.close()
                    time.sleep(delay)
                    continue
                length = headers.get('Content-Length')
                need(length is None or 0 < int(length) <= MAX_BYTES, 'Invalid/excessive Content-Length')
                count = 0
                with part.open('xb') as fh:
                    for chunk in response.iter_content(65536):
                        count += len(chunk)
                        need(count <= MAX_BYTES and time.monotonic()-tic <= 190, 'Bounded response limit exceeded')
                        fh.write(chunk)
                need(count > 0 and (length is None or int(length) == count), 'Incomplete Content-Length')
                text = part.read_text(encoding='utf-8')
                values = validator(text)
                digest = sha(part)
                part.rename(path)
                emit('http/'+path.name+'.json', dict(url=url,path=key(path),request_method='GET',started_utc=started,
                     finished_utc=utc(),status=200,headers=headers,attempt=attempt,actual_bytes=count,
                     content_length=None if length is None else int(length),sha256=digest,validation='PASS'))
                print(json.dumps(dict(acquired=relative,actual_bytes=count,utc=utc())), flush=True)
                return text, values
            except requests.RequestException as exc:
                emit('failures/'+path.name+f'.{self.run_id}.attempt{attempt}.json', dict(utc=utc(),url=url,error=repr(exc),category='network',
                     partial_bytes=part.stat().st_size if part.exists() else None,partial_sha256=sha(part) if part.exists() else None))
                if attempt == 3:
                    raise
                time.sleep(5*attempt)
            except Exception as exc:
                emit('failures/'+path.name+f'.{self.run_id}.attempt{attempt}.json', dict(utc=utc(),url=url,error=repr(exc),category='BLOCKED_validation_or_access',
                     partial_bytes=part.stat().st_size if part.exists() else None,partial_sha256=sha(part) if part.exists() else None))
                raise
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError('Request did not complete')


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    with lock():
        before = snapshot()
        emit('sources_before.json', before)
        prediction = pd.read_parquet(PRED)
        data = pd.read_parquet(DATA)
        cloud = pd.read_parquet(OLD8/'cloud_all.parquet')
        keys = set(zip(prediction.station, prediction.utc_epoch_seconds))
        need(len(keys) == len(prediction) == 5664, 'Bad fixed scope')
        need(keys == set(zip(data.station,data.utc_epoch_seconds)) == set(zip(cloud.station,cloud.utc_epoch_seconds)), 'Frozen table key mismatch')
        blocks = read(OLD8/'plan.json')['nodes']
        need(len(blocks) == 12 and {(b['station'],t) for b in blocks for t in b['epochs']} == keys, 'Source block scope mismatch')
        files = {}
        for month in sorted({b['month'] for b in blocks}):
            catalog = OLD8/'catalogs'/(month+'.xml')
            tree = ET.fromstring(catalog.read_text(encoding='utf-8'))
            for field, f in FIELDS.items():
                prefix = f'e5.oper.an.sfc.128_{f["parameter"]:03d}_{field}.ll025sc.{month}'
                matches = [x.attrib for x in tree.iter() if x.tag.endswith('dataset') and x.attrib.get('name','').startswith(prefix) and x.attrib.get('urlPath')]
                need(len(matches) == 1, 'Nonunique/missing catalog identity: '+month+' '+field)
                entry = matches[0]
                expected_prefix = f'files/d633000/e5.oper.an.sfc/{month}/'
                need(entry['urlPath'] == expected_prefix+entry['name'], 'Unexpected official file path')
                files[month+'_'+field] = dict(month=month,field=field,url_path=entry['urlPath'],
                                             das_url=HOST+entry['urlPath']+'.das',catalog_source=key(catalog),catalog_sha256=sha(catalog))
        emit('scope_plan.json', dict(created_utc=utc(),station_hours=5664,station_dates=236,field_values=28320,
             files=files,blocks=blocks,fields=FIELDS,max_concurrency=1,expected_data_subsets=60,
             old_worker_resume=False,model_execution=False))
        check_sources(before)
        print(json.dumps(dict(status='PREPARED',station_hours=5664,fields=5,metadata_files=30,subsets=60)), flush=True)


def acquire():
    need(OUT.is_dir() and not (OUT/'acquisition_complete.json').exists(), 'No plan or already complete')
    with lock():
        started = utc()
        try:
            before = read(OUT/'sources_before.json')
            check_sources(before)
            plan = read(OUT/'scope_plan.json')
            fetch = Fetcher([f['das_url'] for f in plan['files'].values()])
            meta = {}
            for ident, f in plan['files'].items():
                text, info = fetch.get(f['das_url'], 'metadata/'+ident+'.das', lambda b,field=f['field']: contract(b,field))
                meta[ident] = dict(text=text,info=info)
            nodes = []
            for block in plan['blocks']:
                for field in FIELDS:
                    ident = block['month']+'_'+field
                    node = {k:v for k,v in block.items() if k not in ('url','url_path','id')}
                    node.update(id=block['id']+'_'+field,field=field,url_path=plan['files'][ident]['url_path'],metadata_id=ident,metadata=meta[ident]['info'])
                    month0 = int(datetime(int(block['month'][:4]),int(block['month'][4:]),1,tzinfo=timezone.utc).timestamp())
                    i0,i1 = (block['epochs'][0]-month0)//3600,(block['epochs'][-1]-month0)//3600
                    node['url'] = HOST+node['url_path']+f'.ascii?{meta[ident]["info"]["variable"]}[{i0}:1:{i1}][{node["latitude_index"]}][{node["longitude_index"]}]'
                    nodes.append(node)
            if (OUT/'request_plan.json').exists():
                need(read(OUT/'request_plan.json')['nodes'] == nodes, 'Request plan changed on resume')
            else:
                emit('request_plan.json', dict(nodes=nodes,created_utc=utc(),expected_values=28320,metadata_count=30))
            fetch.allowed.update(n['url'] for n in nodes)
            rows = []
            for node in nodes:
                _, values = fetch.get(node['url'], 'payloads/'+node['id']+'.ascii', lambda b,n=node: decode(b,meta[n['metadata_id']]['text'],n))
                payload = OUT/'payloads'/(node['id']+'.ascii')
                digest = sha(payload)
                rows.extend(dict(station=node['station'],utc_epoch_seconds=t,valid_utc_epoch_seconds=t,
                                 field=node['field'],column=FIELDS[node['field']]['column'],value=v,units=node['metadata']['units'],
                                 latitude=node['latitude'],longitude_0_360=node['longitude_0_360'],source=key(payload),
                                 source_sha256=digest,metadata_source=key(OUT/'metadata'/(node['metadata_id']+'.das')),
                                 metadata_sha256=sha(OUT/'metadata'/(node['metadata_id']+'.das')))
                            for t,v in zip(node['epochs'],values))
            provenance = pd.DataFrame(rows).sort_values(['station','utc_epoch_seconds','field']).reset_index(drop=True)
            need(len(provenance) == 28320 and not provenance.duplicated(['station','utc_epoch_seconds','field']).any(), 'Incomplete field coverage')
            wide = provenance.pivot(index=['station','utc_epoch_seconds'],columns='column',values='value').reset_index()
            wide.columns.name = None
            need(len(wide) == 5664 and wide[list(f['column'] for f in FIELDS.values())].notna().all().all(), 'Missing field combinations')
            source = pd.read_parquet(DATA)
            need(set(zip(wide.station,wide.utc_epoch_seconds)) == set(zip(source.station,source.utc_epoch_seconds)), 'Output key mismatch')
            merged = source.merge(wide,on=['station','utc_epoch_seconds'],how='left',validate='one_to_one',sort=False)
            pd.testing.assert_frame_equal(merged[source.columns],source,check_dtype=True)
            for filename, frame in [('field_provenance.parquet',provenance),('raw_5fields.parquet',wide),('expanded_dataset_with_itu_inputs.parquet',merged)]:
                need(not (OUT/filename).exists(), 'Refuse overwrite '+filename)
                frame.to_parquet(OUT/filename,index=False)
            check_sources(before)
            emit('sources_after_acquisition.json',before)
            receipts = [read(p) for p in (OUT/'http').glob('*.json')]
            emit('acquisition_complete.json',dict(status='ACQUIRED_PENDING_INDEPENDENT_QA',started_utc=started,completed_utc=utc(),
                 station_hours=5664,station_days=236,field_values=28320,data_subsets=60,metadata_files=30,
                 successful_responses=len(receipts),actual_bytes=sum(r['actual_bytes'] for r in receipts),
                 protected_entries=len(before),new_training=0,new_prediction=0,old_worker_resumed=False,execution_lock='RELEASED'))
            print(json.dumps(dict(status='ACQUIRED_PENDING_INDEPENDENT_QA',field_values=28320,station_hours=5664)),flush=True)
        except Exception as exc:
            emit('acquisition_failure_'+stamp()+'.json',dict(utc=utc(),error=repr(exc),partial_files_preserved=True,execution_lock='RELEASED_AFTER_FAILURE'))
            raise


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['acquire']:
        acquire()
    else:
        raise RuntimeError('Use explicit prepare or acquire for this new bounded task only')
