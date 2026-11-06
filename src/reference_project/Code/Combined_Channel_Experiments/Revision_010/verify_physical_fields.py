"""Independent offline raw-series verification; synthetic fixtures are tests only."""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone

import pandas as pd

from acquire_physical_fields import ROOT, OUT, CODE, OLD8, PRED, DATA, GEOMETRY, read, sha, key, need, emit, lock, utc, decode

# Independent explicit field identities (not imported from production FIELDS).
IDENTITIES = {
    '2t': (167, '2 metre temperature', {'K'}, 'surface__2m_temperature', 180, 340),
    '2d': (168, '2 metre dewpoint temperature', {'K'}, 'surface__2m_dewpoint_temperature', 130, 340),
    'sp': (134, 'Surface pressure', {'Pa'}, 'surface__surface_pressure', 10000, 110000),
    'tcwv': (137, 'Total column water vapour', {'kg m**-2','kg m-2'}, 'surface__total_column_water_vapour', 0, 200),
    'tclw': (78, 'Total column cloud liquid water', {'kg m**-2','kg m-2'}, 'surface__total_column_cloud_liquid_water', 0, 50),
}


def independent_metadata(text, field):
    parameter, long_name, units, _, _, _ = IDENTITIES[field]
    leaf_blocks = {}
    name = None
    content = []
    for line in text.splitlines():
        stripped = line.strip()
        start = re.fullmatch(r'([A-Za-z_]\w*)\s*\{',stripped)
        if start:
            name, content = start[1], []
        elif stripped == '}':
            if name is not None:
                leaf_blocks[name] = '\n'.join(content)
                name = None
        elif name is not None:
            content.append(stripped)
    hits = []
    for var, block in leaf_blocks.items():
        attributes = {}
        for line in block.splitlines():
            m = re.fullmatch(r'(?:String|Int32)\s+(\w+)\s+(.+);',line)
            if m:
                attributes[m[1]] = m[2].strip().strip('"')
        if attributes.get('ecmwf_parameter') == str(parameter):
            hits.append((var,attributes))
    need(len(hits)==1, 'Independent parameter uniqueness')
    var, attrs = hits[0]
    need(attrs.get('ecmwf_local_table')=='128' and attrs.get('short_name')==field, 'Independent parameter/table identity')
    need(attrs.get('long_name','').casefold()==long_name.casefold() and attrs.get('units') in units, 'Independent field unit/name')
    need(attrs.get('rda_dataset_group')=='ERA5 atmospheric surface analysis [netCDF4]', 'Independent analysis type')
    grid = attrs.get('grid_specification','')
    need('0.25 degree x 0.25 degree' in grid and '721 x 1440' in grid, 'Independent grid contract')
    for axis, unit in [('time','hours since 1900-01-01 00:00:00'),('latitude','degrees_north'),('longitude','degrees_east')]:
        need(f'String units "{unit}";' in leaf_blocks[axis], 'Independent coordinate units')
    need('String calendar "gregorian";' in leaf_blocks['time'], 'Independent calendar')
    return var, attrs['units']


def raw_series(node):
    text = (OUT/'payloads'/(node['id']+'.ascii')).read_text(encoding='utf-8')
    das = (OUT/'metadata'/(node['metadata_id']+'.das')).read_text(encoding='utf-8')
    var, units = independent_metadata(das,node['field'])
    need(node['url_path'] in text.split('--------------------------------')[0], 'Independent source header')
    heads = list(re.finditer(r'(?m)^([A-Za-z_]\w*\.[A-Za-z_]\w*)((?:\[\d+\])+)[ \t\r]*$',text))
    need(len(heads)==4, 'Independent section count')
    arrays = {}
    for i, heading in enumerate(heads):
        name = heading[1]
        shape = tuple(map(int,re.findall(r'\d+',heading[2])))
        block = text[heading.end():heads[i+1].start() if i+1<len(heads) else len(text)].strip()
        need(name not in arrays, 'Independent duplicate array')
        if name == var+'.'+var:
            need(shape==(len(node['epochs']),1,1), 'Independent value shape')
            values = []
            for row_number,line in enumerate(line.strip() for line in block.splitlines() if line.strip()):
                match = re.fullmatch(r'\[(\d+)\]\[0\],\s*([-+\d.eE]+)',line)
                need(match is not None and int(match[1])==row_number, 'Independent point row identity')
                value = float(match[2])
                need(math.isfinite(value) and IDENTITIES[node['field']][4] <= value <= IDENTITIES[node['field']][5], 'Independent finite/physical value')
                values.append(value)
            need(len(values)==len(node['epochs']), 'Independent point count')
        else:
            need(len(shape)==1, 'Independent coordinate dimensions')
            values = [float(x) for x in re.split(r'[,\s]+',block) if x]
            need(len(values)==shape[0] and all(math.isfinite(x) for x in values), 'Independent coordinates complete')
        arrays[name] = values
    need(set(arrays)=={var+'.'+x for x in (var,'time','latitude','longitude')}, 'Independent unexpected variable')
    origin = datetime(1900,1,1,tzinfo=timezone.utc)
    expected_hours = [(datetime.fromtimestamp(t,timezone.utc)-origin).total_seconds()/3600 for t in node['epochs']]
    need(arrays[var+'.time']==expected_hours, 'Independent UTC identity')
    need(arrays[var+'.latitude']==[node['latitude']] and arrays[var+'.longitude']==[node['longitude_0_360']], 'Independent coordinate identity')
    return arrays[var+'.'+var], units, var


def synthetic_controls():
    passed = []
    for field, (parameter,long_name,units,column,lower,upper) in IDENTITIES.items():
        var = 'TEST_'+field
        unit = sorted(units)[0]
        valid_value = (lower+upper)/2
        origin = int(datetime(1900,1,1,tzinfo=timezone.utc).timestamp())
        node = dict(field=field,url_path='files/d633000/e5.oper.an.sfc/190001/synthetic_test.nc',epochs=[origin,origin+3600],latitude=51.25,longitude_0_360=358.5)
        das = f'''Attributes {{
{var} {{
String long_name "{long_name}";
String short_name "{field}";
String units "{unit}";
Int32 ecmwf_parameter {parameter};
Int32 ecmwf_local_table 128;
String grid_specification "0.25 degree x 0.25 degree from 90N to 90S and 0E to 359.75E (721 x 1440 Latitude/Longitude)";
String rda_dataset_group "ERA5 atmospheric surface analysis [netCDF4]";
}}
time {{ String units "hours since 1900-01-01 00:00:00"; String calendar "gregorian"; }}
latitude {{ String units "degrees_north"; }}
longitude {{ String units "degrees_east"; }}
}}'''
        body = f'''Dataset {{}} {node['url_path']};
---------------------------------------------
{var}.{var}[2][1][1]
[0][0], {valid_value}
[1][0], {valid_value}

{var}.time[2]
0, 1

{var}.latitude[1]
51.25

{var}.longitude[1]
358.5
'''
        need(decode(body,das,node)==[valid_value,valid_value], 'Synthetic positive fixture '+field)
        cases = {
            'parameter': (body,das.replace(f'parameter {parameter};','parameter 999;')),
            'table': (body,das.replace('table 128;','table 235;')),
            'units': (body,das.replace(f'units "{unit}";','units "wrong";')),
            'long_name': (body,das.replace(long_name,'wrong field')),
            'time_units': (body,das.replace('hours since','days since')),
            'calendar': (body,das.replace('gregorian','360_day')),
            'UTC': (body.replace('0, 1','0, 2'),das),
            'latitude': (body.replace('51.25','51.0'),das),
            'longitude': (body.replace('358.5','358.75'),das),
            'shape': (body.replace(f'{var}.{var}[2]',f'{var}.{var}[3]'),das),
            'truncated': (body.replace(f'[1][0], {valid_value}',''),das),
            'duplicate': (body.replace('[1][0],','[0][0],'),das),
            'nonfinite': (body.replace(f'[1][0], {valid_value}','[1][0], nan'),das),
            'fill': (body.replace(f'[1][0], {valid_value}','[1][0], 9.999e20'),das),
            'negative_or_outside': (body.replace(f'[1][0], {valid_value}',f'[1][0], {lower-1}'),das),
            'source': (body.replace('synthetic_test.nc','other_test.nc'),das),
        }
        for label,(bad_body,bad_das) in cases.items():
            try:
                decode(bad_body,bad_das,node)
            except (RuntimeError,ValueError):
                passed.append(field+':'+label)
            else:
                raise RuntimeError('Negative fixture accepted '+field+':'+label)
    return dict(synthetic_only=True,positive=5,negative_rejected=len(passed),cases=passed)


def verify():
    with lock():
        try:
            need(read(OUT/'acquisition_complete.json')['status']=='ACQUIRED_PENDING_INDEPENDENT_QA', 'Acquisition not complete')
            before = read(OUT/'sources_before.json')
            for p,h in before.items():
                need(sha(ROOT/p)==h, 'Protected SHA before verification')
            controls = synthetic_controls()
            plan = read(OUT/'request_plan.json')
            fixed = pd.read_parquet(PRED)
            original = pd.read_parquet(DATA)
            wide = pd.read_parquet(OUT/'raw_5fields.parquet')
            merged = pd.read_parquet(OUT/'expanded_dataset_with_itu_inputs.parquet')
            provenance = pd.read_parquet(OUT/'field_provenance.parquet')
            keys = set(zip(fixed.station,fixed.utc_epoch_seconds))
            need(len(keys)==5664 and len(wide)==len(merged)==5664, 'Independent row coverage')
            need(set(zip(wide.station,wide.utc_epoch_seconds))==set(zip(merged.station,merged.utc_epoch_seconds))==keys, 'Independent exact key coverage')
            pd.testing.assert_frame_equal(original,merged[original.columns],check_dtype=True)
            need(set(wide.columns)=={'station','utc_epoch_seconds'}|{v[3] for v in IDENTITIES.values()}, 'Independent five-field schema')
            stations = {s['id']:s for s in read(GEOMETRY)['links']}
            values = {}
            payload_receipts = set()
            for node in plan['nodes']:
                st = stations[node['station']]
                lat = min([90-i*.25 for i in range(721)],key=lambda a:(abs(a-st['latitude_deg_n']),-a))
                lon = min([-180+i*.25 for i in range(1440)],key=lambda a:(abs(a-st['longitude_deg_e']),-a))%360
                need(node['latitude']==lat and node['longitude_0_360']==lon, 'Independent nearest grid')
                need(node['latitude_index']==round((90-lat)*4) and node['longitude_index']==round(lon*4), 'Independent grid indices')
                raw,units,var = raw_series(node)
                month0 = datetime(int(node['month'][:4]),int(node['month'][4:]),1,tzinfo=timezone.utc)
                h0,h1 = [(datetime.fromtimestamp(t,timezone.utc)-month0).total_seconds()/3600 for t in (node['epochs'][0],node['epochs'][-1])]
                expected_url = 'https://tds.gdex.ucar.edu/thredds/dodsC/'+node['url_path']+f'.ascii?{var}[{int(h0)}:1:{int(h1)}][{int((90-lat)*4)}][{int(lon*4)}]'
                need(node['url']==expected_url, 'Independent exact request URL')
                payload = OUT/'payloads'/(node['id']+'.ascii')
                payload_receipts.add(payload.name+'.json')
                receipt = read(OUT/'http'/(payload.name+'.json'))
                need(receipt['url']==expected_url, 'Independent HTTP query identity')
                for t,v in zip(node['epochs'],raw):
                    identity = (node['station'],t,node['field'])
                    need(identity not in values, 'Independent duplicate source value')
                    values[identity] = (v,units,key(payload),sha(payload),key(OUT/'metadata'/(node['metadata_id']+'.das')),sha(OUT/'metadata'/(node['metadata_id']+'.das')),lat,lon)
            need(len(values)==28320 and len(provenance)==28320 and not provenance.duplicated(['station','utc_epoch_seconds','field']).any(), 'Independent exact field count')
            wide_index = wide.set_index(['station','utc_epoch_seconds'])
            merged_index = merged.set_index(['station','utc_epoch_seconds'])
            for row in provenance.itertuples(index=False):
                identity = (row.station,row.utc_epoch_seconds,row.field)
                raw,units,path,digest,meta_path,meta_digest,lat,lon = values[identity]
                column = IDENTITIES[row.field][3]
                need(row.column==column and row.value==raw and row.units==units, 'Independent field/value/unit mismatch')
                need(row.valid_utc_epoch_seconds==row.utc_epoch_seconds, 'Independent valid UTC')
                need(row.source==path and row.source_sha256==digest and row.metadata_source==meta_path and row.metadata_sha256==meta_digest, 'Independent provenance SHA')
                need(row.latitude==lat and row.longitude_0_360==lon, 'Independent provenance grid')
                need(wide_index.at[(row.station,row.utc_epoch_seconds),column]==raw and merged_index.at[(row.station,row.utc_epoch_seconds),column]==raw, 'Independent merged value')
            receipts = [read(p) for p in (OUT/'http').glob('*.json')]
            need(len(receipts)==90 and len({r['url'] for r in receipts})==90, 'Independent success/request count')
            need(len(payload_receipts)==60 and len(list((OUT/'metadata').glob('*.das')))==30, 'Independent subsets/metadata count')
            for receipt in receipts:
                path = ROOT/receipt['path']
                need(receipt['request_method']=='GET' and receipt['status']==200 and receipt['validation']=='PASS', 'Independent request method/status')
                need(path.stat().st_size==receipt['actual_bytes'] and sha(path)==receipt['sha256'], 'Independent HTTP bytes/SHA')
                need(receipt['content_length'] is None or receipt['content_length']==receipt['actual_bytes'], 'Independent Content-Length')
                if path.suffix=='.das':
                    need(receipt['url']=='https://tds.gdex.ucar.edu/thredds/dodsC/'+next(n['url_path'] for n in plan['nodes'] if n['metadata_id']+'.das'==path.name)+'.das', 'Independent metadata URL')
            dates = wide[['station','utc_epoch_seconds']].copy()
            dates['date_utc'] = pd.to_datetime(dates.utc_epoch_seconds,unit='s',utc=True).dt.strftime('%Y-%m-%d')
            coverage = dates.groupby(['station','date_utc']).size().reset_index(name='complete_hours')
            need(len(coverage)==236 and coverage.complete_hours.eq(24).all(), 'Independent complete-day coverage')
            need(coverage.groupby('station').size().to_dict()=={'chilbolton':143,'sparsholt':93}, 'Independent station-date counts')
            emit('coverage_by_date.json',coverage.to_dict('records'))
            summaries = []
            for field,(_,label,_,column,_,_) in IDENTITIES.items():
                summaries.append(dict(field=field,name=label,column=column,finite_values=len(wide),
                                      min=float(wide[column].min()),max=float(wide[column].max()),missing=0))
            dew_excess = wide['surface__2m_dewpoint_temperature']-wide['surface__2m_temperature']
            # Preserve all originals; flag possible physical inconsistencies without clipping or silently removing.
            crossfield = dict(dewpoint_exceeds_temperature_hours=int((dew_excess>0).sum()),
                              maximum_dewpoint_minus_temperature_K=float(dew_excess.max()),
                              modified_or_clipped_values=0)
            emit('field_summary.json',summaries)
            emit('crossfield_diagnostic.json',crossfield)
            for p,h in before.items():
                need(sha(ROOT/p)==h, 'Protected SHA after verification')
            emit('sources_after_verification.json',before)
            qa = dict(status='PASS',verified_utc=utc(),station_days=236,station_hours=5664,fields=5,
                      independently_verified_raw_values=28320,missing_field_values=0,subsets=60,metadata_files=30,
                      protected_manifest_entries=len(before),old_columns_exactly_preserved=True,original_dates_and_masks_unchanged=True,
                      synthetic_controls=controls,crossfield_diagnostic=crossfield,new_training=0,new_prediction=0,
                      attenuation_formula_executed=False,old_download_worker_resumed=False,execution_lock='RELEASED')
            emit('verification.json',qa)
            report = ['# 英国已完成日期的五项 ITU 输入补齐 — Revision 010','',
                      '状态：采集完成，独立字段/UTC/网格/来源核验 PASS。仅此数据补齐完成，不代表组合 ITU 衰减与对比实验已完成。','',
                      '| 站点 | 原有已完成站点日 | 五项齐全小时 | 新字段值 |','|---|---:|---:|---:|',
                      '| Chilbolton | 143 | 3432 | 17160 |','| Sparsholt | 93 | 2232 | 11160 |','| 合计 | 236 | 5664 | 28320 |','',
                      '原日期缺口保留；不是连续236天，不增加12月未完整数据，也没有扩展到原1127站点日计划。LOCAL不重新下载。','',
                      '## 新增字段','', '| 字段 | 来源参数（table128） | 原始单位 | 完整值数 |','|---|---:|---|---:|',
                      '| 2米温度 | 167 | K | 5664 |','| 2米露点 | 168 | K | 5664 |','| 地表气压 | 134 | Pa | 5664 |',
                      '| 整层水汽 | 137 | kg/m² | 5664 |','| 整层云液态水 | 078 | kg/m² | 5664 |','',
                      '均为 ERA5 瞬时表面分析、有效UTC=t；与平均降雨率的valid=t+1区分。使用各站最近0.25°发布格，未作时空插值、插补、裁负或拟合。','',
                      '## 交付与追溯','',
                      '- raw_5fields.parquet：5664行五字段宽表。',
                      '- expanded_dataset_with_itu_inputs.parquet：原29列逐值保留，新增五列；原数据、标签、预测及评分掩码未改。',
                      '- field_provenance.parquet：28320条逐字段原载荷/元数据SHA、UTC、网格和单位。',
                      '- coverage_by_date.json：236站点日逐日24小时完整性。',
                      '- metadata/、payloads/、http/：30份DAS、60个点子集及90份成功收据。',
                      '- verification.json：独立原始点序列重读和5正例/80负例测试；crossfield_diagnostic保留温度/露点关系诊断，不改原值。','',
                      '来源由 Revision008 留存的官方 NCAR/GDEX 月目录定位，直接匿名GET精确OPeNDAP子集；实际大小与Content-Length（如存在）/SHA核对。缺少Content-Length的响应明确记null，不推测文件大小。','',
                      '## 保持的边界与下一步','',
                      '零训练、零模型推理、零衰减公式计算；旧v011停止标志、tvt PAUSED不变。只进行了已批准的固定日期补齐，完成后退出。',
                      '后续组合ITU仍需固定版本数值实现、相对晴空参考映射和闪烁适用定义；没有用英国标签调偏置或改变参考，也没有发布新的对比指标。',
                      '协议及授权见 ../../../Code/Combined_Channel_Experiments/Revision_010/PROTOCOL.md。历史Revision009的“缺输入”结论由本版数据交付解除，旧文件原样保留。','']
            with (OUT/'REPORT.md').open('x',encoding='utf-8',newline='\n') as fh:
                fh.write('\n'.join(report))
            artifacts = {key(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='execution.lock'}
            emit('completion.json',dict(status='COMPLETE_BOUNDED_FIVE_FIELD_ACQUISITION',verification='PASS',completed_utc=utc(),
                 execution_lock='RELEASED',station_days=236,station_hours=5664,field_values=28320,missing_values=0,
                 artifacts=artifacts,new_training=0,new_prediction=0,combined_ITU_computed=False,t012_complete=False))
            print(json.dumps(qa,ensure_ascii=False),flush=True)
        except Exception as exc:
            emit('verification_failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'.json',
                 dict(utc=utc(),error=repr(exc),execution_lock='RELEASED_AFTER_FAILURE'))
            raise


if __name__=='__main__':
    if sys.argv[1:]==['selftest']:
        print(json.dumps(synthetic_controls()))
    elif not sys.argv[1:]:
        verify()
    else:
        raise RuntimeError('Only selftest or offline verification')
