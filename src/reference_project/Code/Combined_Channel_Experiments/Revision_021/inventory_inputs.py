"""Bounded, read-only inventory of existing LOCAL assets. No network or models."""
import hashlib
import json
import msvcrt
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_020'
RAW = Path('E:/0_Workflow_System/01_PhD/Data/era5_campaign_20250930_20260820/raw')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def clean(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [clean(x) for x in v]
    return v


def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    guards = read(PREV / 'sources_before.json')
    for rel, record in read(PREV / 'delivery_manifest.json')['files'].items():
        p = str((ROOT / rel).resolve())
        assert sha(p) == record['sha256'], p
        guards[p] = record['sha256']
    for directory in (PREV, ROOT / 'Code/Combined_Channel_Experiments/Revision_020', CODE):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                guards.setdefault(str(p.resolve()), sha(p))
    extra_sources = [RAW / 'download_manifest.json',
                     ROOT / 'Code/Dataset_Build/Revision_002/build_matched_dataset.py',
                     ROOT / 'Code/Dataset_Build/ClearSky_Alignment_Revision_004/build_attenuation.py',
                     ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012/selected_local.parquet',
                     Path('E:/0_Workflow_System/README.md')]
    raw_files = sorted(RAW.glob('*.nc'))
    assert len(raw_files) == 340, 'Bounded raw scope changed; review first'
    for p in extra_sources + raw_files:
        guards.setdefault(str(p.resolve()), sha(p))
    for p, digest in guards.items():
        assert sha(p) == digest, f'Protected source changed: {p}'
    with (OUT / 'inventory.lock').open('x+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        emit('execution.json', {'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                               'python': sys.version, 'executable': sys.executable,
                               'platform': platform.platform(), 'numpy': np.__version__,
                               'xarray': xr.__version__, 'argv': sys.argv,
                               'scope': 'offline bounded existing raw header inventory'})
        emit('sources_before.json', guards)
        manifest = read(RAW / 'download_manifest.json')
        jobs = {Path(j['target'].replace('\\', '/')).name: j for j in manifest['jobs']}
        assert len(jobs) == len(manifest['jobs']) == 336
        records = []
        for p in raw_files:
            with xr.open_dataset(p, engine='netcdf4') as ds:
                variables = []
                for key, v in ds.data_vars.items():
                    attrs = {k: clean(x) for k, x in v.attrs.items()
                             if k in ['GRIB_paramId', 'GRIB_typeOfLevel', 'GRIB_stepType',
                                      'GRIB_dataType', 'GRIB_stepUnits', 'long_name',
                                      'standard_name', 'units']}
                    title = (key + ' ' + str(attrs.get('long_name', '')) + ' '
                             + str(attrs.get('standard_name', ''))).lower()
                    terrain = ('pressure_level' not in v.dims and
                               (attrs.get('GRIB_paramId') == 129 or
                                any(w in title for w in ['orograph', 'surface_geopotential',
                                                         'surface_altitude', 'surface elevation'])))
                    variables.append({'key': key, 'dims': list(v.dims), 'attrs': attrs,
                                      'surface_terrain_candidate': terrain})
                r = {'file': p.name, 'sha256': guards[str(p.resolve())], 'bytes': p.stat().st_size,
                     'manifest_job_id': jobs.get(p.name, {}).get('job_id'),
                     'sizes': dict(ds.sizes), 'variables': variables}
                for coord in ('latitude', 'longitude', 'pressure_level'):
                    if coord in ds.coords:
                        r[coord] = ds[coord].values.tolist()
                if 'valid_time' in ds.coords:
                    times = ds.valid_time.values.astype('datetime64[s]').astype(np.int64)
                    r['time'] = {'n': int(len(times)), 'start_epoch_s': int(times.min()),
                                 'end_epoch_s': int(times.max()),
                                 'strict_continuous_hourly': bool(np.all(np.diff(times) == 3600)),
                                 'units': ds.valid_time.encoding.get('units'),
                                 'calendar': ds.valid_time.encoding.get('calendar')}
                records.append(r)
        emit('header_inventory.json', records)
        names = {p.name for p in raw_files}
        summary = {'status': 'PASS', 'scope_directory': str(RAW), 'raw_files': len(records),
                   'raw_bytes': sum(r['bytes'] for r in records), 'manifest_jobs': len(jobs),
                   'missing_manifest_files': sorted(set(jobs) - names),
                   'extra_files': sorted(names - set(jobs)),
                   'terrain_candidates': [{'file': r['file'], 'key': v['key'], 'attrs': v['attrs']}
                                          for r in records for v in r['variables']
                                          if v['surface_terrain_candidate']],
                   'station_declaration': manifest['campaign']['station'],
                   'station_declaration_is_independent_survey': False,
                   'no_absence_claim_outside_bounded_directory': True}
        emit('inventory_summary.json', summary)
        changed = [p for p, digest in guards.items() if sha(p) != digest]
        assert not changed, changed
        emit('inventory_qa.json', {'status': 'PASS', 'protected_source_count': len(guards),
                                  'changed_sources': changed, 'new_downloads': 0,
                                  'new_predictions': 0, 'labels_read': False,
                                  'utc': datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
