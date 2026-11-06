"""Authorized, bounded UK E-SST inputs. No old downloader entry points."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import msvcrt
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_031'
PLAN = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_030'
CHANGE = ROOT / 'Outputs/Automation_Changes/TVT_Comparison_20261018_001'
TVT = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
OLD_TVT_SHA = 'bc5a33cdf2b49fe183de87cec7ec9cb1ac9a1a351c4f1691557c7587bf10891e'
STOP = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
STOP_SHA = '0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a'
BODY_CAP = 255 * 1024 * 1024  # 1 MiB reserved below the authorized total cap.
NAMES = {'temperature': 't', 'geopotential': 'z', 'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}
UNITS = {'t': {'K'}, 'z': {'m**2 s**-2', 'm2 s-2', 'm^2 s^-2'},
         'u': {'m s**-1', 'm s-1', 'm/s'}, 'v': {'m s**-1', 'm s-1', 'm/s'}}

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def put(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def say(**kw):
    print(json.dumps({'utc': utc(), **kw}, ensure_ascii=False), flush=True)

def expected_times(batch):
    return {int(dt.datetime(batch['year'], batch['month'], day, hour, tzinfo=dt.timezone.utc).timestamp())
            for day in batch['days'] for hour in batch['hours_utc']}

def make_request(batch):
    assert batch['dataset'] == 'reanalysis-era5-pressure-levels'
    assert batch['product_type'] == 'reanalysis'
    assert (batch['station'], batch['grid_latitude'], batch['grid_longitude']) in {
        ('chilbolton', 51.25, -1.5), ('sparsholt', 51.0, -1.5)}
    assert set(batch['variables']) <= set(NAMES)
    assert batch['hours_utc'] == list(range(24))
    assert len(expected_times(batch)) * len(batch['variables']) * len(batch['pressure_levels_hpa']) == batch['scalar_values']
    lat, lon = batch['grid_latitude'], batch['grid_longitude']
    return {'product_type': ['reanalysis'], 'variable': batch['variables'],
            'year': [str(batch['year'])], 'month': [f"{batch['month']:02d}"],
            'day': [f'{d:02d}' for d in batch['days']],
            'time': [f'{h:02d}:00' for h in batch['hours_utc']],
            'pressure_level': [str(p) for p in batch['pressure_levels_hpa']],
            'area': [lat, lon, lat, lon], 'data_format': 'netcdf', 'download_format': 'unarchived'}

def plan_keys(batches):
    keys = set()
    for b in batches:
        make_request(b)
        current = {(b['station'], t, NAMES[v], p) for t in expected_times(b)
                   for v in b['variables'] for p in b['pressure_levels_hpa']}
        assert not keys.intersection(current), 'duplicate request values'
        keys.update(current)
    assert len(batches) == 30 and len(keys) == 421104
    return keys

def initialize():
    import tomllib
    assert not OUT.exists(), 'existing output is not overwritten'
    before = read(PLAN / 'sources_before.json')
    assert sha(CHANGE / 'automation_before.toml') == OLD_TVT_SHA
    config = tomllib.loads(TVT.read_text(encoding='utf-8'))
    assert config['id'] == 'tvt' and config['kind'] == 'heartbeat' and config['status'] == 'ACTIVE'
    assert config['rrule'] == 'RRULE:FREQ=MINUTELY;INTERVAL=15'
    assert config['target_thread_id'] == '01a0bda1-92d1-7832-9508-60f4e0ec1634'
    assert '五类' in config['name'] and 'EXCLUDED_USER_DECISION' in config['prompt']
    assert sha(STOP) == STOP_SHA
    protected = {}
    intentional = []
    for name, val in before.items():
        expected = val if isinstance(val, str) else val['sha256']
        actual = sha(name)
        if Path(name).resolve() == TVT.resolve():
            assert expected == OLD_TVT_SHA
            intentional.append({'path': name, 'before_sha256': expected, 'after_sha256': actual,
                                'reason': '2026-10-18 human five-model and bounded acquisition decision'})
        else:
            assert actual == expected, name
        protected[str(Path(name).resolve())] = actual
    assert len(intentional) == 1
    manifest = read(PLAN / 'delivery_manifest.json')
    for name, val in manifest['files'].items():
        p = ROOT / name
        assert sha(p) == val['sha256'], name
        protected[str(p.resolve())] = sha(p)
    assert sha(PLAN / 'delivery_manifest.json') == read(PLAN / 'final_verification.json')['manifest_sha256']
    for p in PLAN.iterdir():
        if p.is_file():
            protected[str(p.resolve())] = sha(p)
    batches = read(PLAN / 'input_batches.json')
    plan_keys(batches)
    OUT.mkdir()
    for sub in ['batches', 'runs']:
        (OUT / sub).mkdir()
    shutil.copyfile(TVT, CHANGE / 'automation_after.toml')
    put(CHANGE / 'automation_change.json', {'utc': utc(), 'verified': True, 'changes': intentional,
        'id': config['id'], 'status': config['status'], 'name': config['name'],
        'same_thread': True, 'same_15_minute_interval': True})
    for p in CHANGE.iterdir():
        if p.is_file():
            protected[str(p.resolve())] = sha(p)
    put(OUT / 'sources_before.json', protected)
    put(OUT / 'code_before.json', {str(p): sha(p) for p in CODE.glob('*') if p.is_file()})
    put(OUT / 'authorization.json', {'utc_recorded': utc(), 'authorized': True,
        'user_text': '那就开始补齐E SST所需数据，NWP PRM数据比较难获取，就不考虑这个模型了。PMOP感觉缺失的数据也很难获取，也不考虑了',
        'scope': 'Revision030 exact 30 batches and input QA only', 'new_values': 421104,
        'total_transfer_cap_bytes': 256 * 1024 * 1024, 'http_body_cap_bytes': BODY_CAP,
        'rmop': 'EXCLUDED_USER_DECISION', 'nwp_prm': 'EXCLUDED_USER_DECISION',
        'mts': 'NOT_COMPUTED_USER_DECISION', 'uk_physical_adaptation_approved': False,
        'plan_sha256': sha(PLAN / 'input_batches.json'), 'stop_sha256': STOP_SHA})
    put(OUT / 'requests.json', [{'batch': b, 'request': make_request(b)} for b in batches])
    put(OUT / 'environment.json', {'utc': utc(), 'python': sys.version, 'executable': sys.executable,
        'versions': {p: importlib.metadata.version(p) for p in ['cdsapi', 'ecmwf-datastores-client', 'requests', 'netCDF4', 'h5py']}})
    say(status='INITIALIZED_AUTHORIZED', batches=30, values=421104, protected_sources=len(protected))

def verify_sources():
    for filename in ['sources_before.json', 'code_before.json']:
        for name, expected in read(OUT / filename).items():
            assert sha(name) == expected, f'protected source changed: {name}'
    assert sha(STOP) == STOP_SHA

class Budget:
    def __init__(self):
        self.path = OUT / 'network_ledger.jsonl'
        self.used = 0
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                self.used += json.loads(line)['bytes']

    def require(self, n):
        if n < 0 or self.used + n > BODY_CAP:
            raise RuntimeError('TRANSFER_BUDGET_EXCEEDED')

    def record(self, n, host, kind):
        self.require(n)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'utc': utc(), 'bytes': n, 'host': host, 'kind': kind}) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.used += n

def permitted_url(url, api=False):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    allowed = host == 'cds.climate.copernicus.eu' if api else (
        host == 'cds.climate.copernicus.eu' or host.endswith('.ecmwf.int') or host.endswith('.climate.copernicus.eu'))
    if parsed.scheme != 'https' or not allowed or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise RuntimeError('UNAPPROVED_SERVICE_HOST')
    return host

def api_session(budget):
    import requests
    class LimitedSession(requests.Session):
        def send(self, request, **kwargs):
            host = permitted_url(request.url, api=True)
            kwargs.update(stream=True, allow_redirects=False, verify=True)
            response = super().send(request, **kwargs)
            if 300 <= response.status_code < 400:
                response.close()
                raise RuntimeError('API_REDIRECT_NOT_ALLOWED')
            length = response.headers.get('Content-Length')
            if length:
                budget.require(int(length))
            chunks = []
            for block in response.iter_content(16384):
                budget.record(len(block), host, 'api')
                chunks.append(block)
            response._content = b''.join(chunks)
            response._content_consumed = True
            return response
    return LimitedSession()

def client_for(budget):
    from cdsapi.api import get_url_key_verify
    from ecmwf.datastores import Client
    url, key, _ = get_url_key_verify(None, None, True)
    if url.rstrip('/') != 'https://cds.climate.copernicus.eu/api':
        raise RuntimeError('UNAPPROVED_API_CONFIGURATION')
    logging.disable(logging.CRITICAL)
    return Client(url=url, key=key, verify=True, timeout=(15, 45), maximum_tries=1,
                  progress=False, cleanup=False, retry_after=1, sleep_max=30,
                  session=api_session(budget), log_callback=lambda *a, **k: None)

def download(results, target, budget):
    import requests
    location = results.location
    host = permitted_url(location)
    expected = int(results.content_length)
    budget.require(expected)
    with requests.Session() as session:
        session.trust_env = False  # no ambient .netrc or credentials sent to object storage
        with session.get(location, stream=True, timeout=(15, 45), verify=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise RuntimeError(f'DOWNLOAD_HTTP_{response.status_code}')
            if response.headers.get('Content-Length'):
                budget.require(int(response.headers['Content-Length']))
            count = 0
            with target.open('xb') as f:
                for block in response.iter_content(16384):
                    budget.record(len(block), host, 'data')
                    f.write(block)
                    count += len(block)
                f.flush()
                os.fsync(f.fileno())
    assert count == expected, 'download length mismatch'
    return {'bytes': count, 'host': host, 'sha256': sha(target)}

def validate_netcdf(path, batch):
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as ds:
        assert ds.data_model.startswith('NETCDF4'), 'expected NetCDF4 for independent HDF5 verification'
        for name, expected, wrap in [('latitude', batch['grid_latitude'], False), ('longitude', batch['grid_longitude'], True)]:
            value = np.asarray(ds[name][:]).reshape(-1)
            if wrap:
                value = (value + 180) % 360 - 180
            assert value.size == 1 and abs(float(value[0]) - expected) < 1e-7, name
        p_name = 'pressure_level' if 'pressure_level' in ds.variables else 'level'
        levels = np.asarray(ds[p_name][:]).astype(float)
        assert ds[p_name].units in ('hPa', 'millibars', 'mbar')
        assert set(levels) == set(batch['pressure_levels_hpa']) and len(levels) == len(set(levels))
        t_name = 'valid_time' if 'valid_time' in ds.variables else 'time'
        tv = ds[t_name]
        dates = netCDF4.num2date(tv[:], tv.units, calendar=getattr(tv, 'calendar', 'standard'))
        times = [int(dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, tzinfo=dt.timezone.utc).timestamp()) for d in dates]
        assert set(times) == expected_times(batch) and len(times) == len(set(times))
        if 'expver' in ds.variables:
            assert set(np.asarray(ds['expver'][:]).astype(str)) <= {'1', '0001'}, 'unexpected reanalysis version'
        stats = {}
        wanted = {NAMES[v] for v in batch['variables']}
        assert set(NAMES.values()).intersection(ds.variables) == wanted
        for name in wanted:
            var = ds[name]
            assert var.units in UNITS[name], (name, var.units)
            assert set(var.dimensions) == {t_name, p_name, 'latitude', 'longitude'}, var.dimensions
            values = np.ma.asarray(var[:])
            assert not np.ma.getmaskarray(values).any() and np.isfinite(values).all()
            assert values.size == len(times) * len(levels)
            stats[name] = {'units': var.units, 'count': int(values.size), 'min': float(values.min()), 'max': float(values.max())}
        assert sum(x['count'] for x in stats.values()) == batch['scalar_values']
    return {'status': 'PASS_INPUT_METADATA', 'variables': stats, 'hours': len(times), 'levels': len(levels),
            'values': batch['scalar_values'], 'sha256': sha(path)}

def safe_error(error):
    result = {'type': type(error).__name__}
    response = getattr(error, 'response', None)
    if response is not None:
        result['http_status'] = response.status_code
        try:
            body = response.json()
            if isinstance(body, dict):
                for key in ['title', 'reason', 'message', 'detail']:
                    if isinstance(body.get(key), str):
                        result[key] = re.sub(r'https?://\S+', '[service URL]', body[key])[:900]
        except Exception:
            pass
    elif isinstance(error, (AssertionError, RuntimeError)):
        result['detail'] = str(error)[:600]
    return result

def run(max_wait_seconds):
    assert read(OUT / 'authorization.json')['authorized'] is True
    lock = (OUT / 'worker_os.lock').open('a+b')
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'1')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    run_dir = OUT / 'runs' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    run_dir.mkdir()
    put(run_dir / 'started.json', {'utc': utc(), 'pid': os.getpid(), 'argv': sys.argv,
                                 'source_sha256': sha(__file__), 'max_wait_seconds': max_wait_seconds})
    batch_id = None
    try:
        verify_sources()
        budget = Budget()
        client = client_for(budget)
        client.check_authentication()  # result deliberately not logged
        say(status='AUTHENTICATED', pid=os.getpid())
        deadline = time.monotonic() + max_wait_seconds
        for item in read(OUT / 'requests.json'):
            batch = item['batch']
            batch_id = batch['id']
            folder = OUT / 'batches' / f'{batch_id:03d}'
            folder.mkdir(exist_ok=True)
            if (folder / 'complete.json').exists():
                done = read(folder / 'complete.json')
                assert sha(folder / 'raw.nc') == done['sha256']
                continue
            if (folder / 'failure.json').exists():
                raise RuntimeError(f'BATCH_{batch_id}_PRIOR_FAILURE_REQUIRES_REVIEW')
            if (folder / 'submitted.json').exists():
                request_id = read(folder / 'submitted.json')['request_id']
                remote = client.get_remote(request_id)
            else:
                if (folder / 'submit_intent.json').exists():
                    raise RuntimeError(f'BATCH_{batch_id}_SUBMISSION_UNCERTAIN_DO_NOT_RESUBMIT')
                put(folder / 'submit_intent.json', {'utc': utc(), 'dataset': batch['dataset'], 'request': item['request']})
                remote = client.submit(batch['dataset'], item['request'])
                request_id = remote.request_id
                put(folder / 'submitted.json', {'utc': utc(), 'request_id': request_id})
                say(status='SUBMITTED', batch=batch_id, request_id=request_id, values=batch['scalar_values'])
            while True:
                state = remote.status
                put(run_dir / f'batch_{batch_id:03d}_poll_{time.time_ns()}.json', {'utc': utc(), 'batch': batch_id, 'status': state, 'request_id': request_id})
                if state == 'successful':
                    break
                if state not in ('accepted', 'running', 'queued'):
                    raise RuntimeError(f'REMOTE_STATUS_{state}')
                if time.monotonic() >= deadline:
                    put(run_dir / 'finished.json', {'utc': utc(), 'status': 'WAITING_REMOTE', 'batch': batch_id, 'request_id': request_id, 'bytes': budget.used})
                    say(status='WAITING_REMOTE', batch=batch_id, request_id=request_id)
                    return
                time.sleep(30)
            assert not (folder / 'raw.nc').exists(), 'preserve previous or partial raw file'
            metadata = download(remote.get_results(), folder / 'raw.nc', budget)
            checked = validate_netcdf(folder / 'raw.nc', batch)
            put(folder / 'complete.json', {'utc': utc(), 'batch': batch_id, 'request_id': request_id, **metadata, **checked})
            say(status='BATCH_INPUT_VERIFIED', batch=batch_id, values=batch['scalar_values'], bytes_used=budget.used)
        verify_sources()
        complete = [read(p / 'complete.json') for p in sorted((OUT / 'batches').iterdir())]
        assert len(complete) == 30 and sum(r['values'] for r in complete) == 421104
        put(OUT / 'acquisition_complete.json', {'utc': utc(), 'status': 'ACQUIRED_PENDING_INDEPENDENT_QA',
            'batches': 30, 'new_values': 421104, 'http_body_bytes': budget.used,
            'protected_sources_unchanged': True, 'new_predictions': 0, 'new_scores': 0})
        put(run_dir / 'finished.json', {'utc': utc(), 'status': 'ACQUIRED_PENDING_INDEPENDENT_QA'})
        say(status='ACQUIRED_PENDING_INDEPENDENT_QA', batches=30, values=421104)
    except Exception as error:
        detail = safe_error(error)
        put(run_dir / 'failure.json', {'utc': utc(), 'batch': batch_id, **detail})
        if batch_id is not None:
            folder = OUT / 'batches' / f'{batch_id:03d}'
            if not (folder / 'failure.json').exists():
                put(folder / 'failure.json', {'utc': utc(), **detail})
        say(status='STOPPED_REVIEW_REQUIRED', batch=batch_id, **detail)
        raise SystemExit(2)
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['init', 'run'])
    parser.add_argument('--max-wait-seconds', type=int, default=3600)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize()
    else:
        assert 60 <= args.max_wait_seconds <= 3600
        run(args.max_wait_seconds)
