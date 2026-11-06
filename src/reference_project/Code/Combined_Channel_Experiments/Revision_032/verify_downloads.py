"""Independent h5py validation of completed downloads, no acquisition imports."""
import argparse
import calendar
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_032'
PLAN = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_030'
MAP = {'temperature': 't', 'geopotential': 'z', 'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}
UNITS = {'t': {'K'}, 'z': {'m**2 s**-2', 'm2 s-2', 'm^2 s^-2'},
         'u': {'m s**-1', 'm s-1', 'm/s'}, 'v': {'m s**-1', 'm s-1', 'm/s'}}

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def txt(x): return x.decode() if isinstance(x, bytes) else str(x)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def keys(b):
    return {(b['station'], calendar.timegm((b['year'], b['month'], d, h, 0, 0)), MAP[v], p)
            for d in b['days'] for h in b['hours_utc'] for v in b['variables'] for p in b['pressure_levels_hpa']}

def main(partial):
    requests = read(OUT / 'requests.json')
    all_expected = set().union(*(keys(b) for b in read(PLAN / 'input_batches.json')))
    expanded = set()
    for r in requests:
        current = keys(r['batch']); assert not current.intersection(expanded); expanded.update(current)
    assert expanded == all_expected and len(expanded) == 421104 and len(requests) == 708
    seen = set(); results = []
    for r in requests:
        b = r['batch']; folder = OUT / 'batches' / f"{b['id']:03d}"
        if not (folder / 'complete.json').exists():
            assert partial, f"subrequest {b['id']} incomplete"
            continue
        prod = read(folder / 'complete.json'); raw = folder / 'raw.nc'
        assert sha(raw) == prod['sha256']
        expected = keys(b); actual = set()
        with h5py.File(raw, 'r') as f:
            assert f['latitude'].size == f['longitude'].size == 1
            assert np.allclose(f['latitude'][()], b['grid_latitude'], rtol=0, atol=1e-7)
            assert np.allclose((f['longitude'][()] + 180) % 360 - 180, b['grid_longitude'], rtol=0, atol=1e-7)
            tn = 'valid_time' if 'valid_time' in f else 'time'
            pn = 'pressure_level' if 'pressure_level' in f else 'level'
            match = re.fullmatch(r'(seconds|hours) since (.+)', txt(f[tn].attrs['units']))
            assert match
            origin = dt.datetime.fromisoformat(match[2].replace('Z', '+00:00'))
            if origin.tzinfo is None: origin = origin.replace(tzinfo=dt.timezone.utc)
            epoch = calendar.timegm(origin.utctimetuple())
            times = [epoch + float(v) * (1 if match[1] == 'seconds' else 3600) for v in f[tn][()]]
            assert len(times) == len(set(times)) and all(t == int(t) for t in times)
            assert txt(f[tn].attrs.get('calendar', b'standard')) in ('standard', 'gregorian', 'proleptic_gregorian')
            levels = f[pn][()].astype(float).tolist()
            assert len(levels) == len(set(levels)) and set(levels) == set(b['pressure_levels_hpa'])
            assert txt(f[pn].attrs['units']) in ('hPa', 'millibars', 'mbar')
            assert set(MAP.values()).intersection(f.keys()) == {MAP[v] for v in b['variables']}
            for var in b['variables']:
                n = MAP[var]; d = f[n]
                dims = [f[refs[0]].name.lstrip('/') for refs in d.attrs['DIMENSION_LIST']]
                assert set(dims) == {tn, pn, 'latitude', 'longitude'}
                data = np.transpose(d[()], [dims.index(x) for x in [tn, pn, 'latitude', 'longitude']])[:, :, 0, 0]
                assert data.shape == (len(times), len(levels)) and np.isfinite(data).all()
                assert txt(d.attrs['units']) in UNITS[n]
                if '_FillValue' in d.attrs: assert not np.any(data == d.attrs['_FillValue'])
                assert 'scale_factor' not in d.attrs and 'add_offset' not in d.attrs, 'packed fields need an explicit QA adapter'
                if 'GRIB_stepType' in d.attrs: assert txt(d.attrs['GRIB_stepType']) == 'instant'
                if 'GRIB_dataType' in d.attrs: assert txt(d.attrs['GRIB_dataType']) == 'an'
                assert data.size == prod['variables'][n]['count']
                assert float(data.min()) == prod['variables'][n]['min'] and float(data.max()) == prod['variables'][n]['max']
                actual.update((b['station'], int(t), n, int(p)) for t in times for p in levels)
        assert actual == expected and not actual.intersection(seen)
        seen.update(actual)
        results.append({'subrequest': b['id'], 'sha256': sha(raw), 'values': len(actual)})
    assert results, 'no completed inputs to verify'
    if not partial: assert seen == all_expected
    prior = read(OUT / 'authorization.json')['prior_http_body_bytes']
    body = prior + sum(json.loads(line)['bytes'] for line in (OUT / 'network_ledger.jsonl').read_text().splitlines())
    assert body <= 255 * 1024 * 1024
    report = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'status': 'PASS_PARTIAL_INPUT_QA' if partial else 'PASS_INPUT_QA',
              'verified_subrequests': len(results), 'verified_values': len(seen),
              'target_values': 421104, 'missing_values': 421104 - len(seen),
              'cumulative_http_body_bytes_at_check': body, 'files': results,
              'script_sha256': sha(__file__), 'independence': 'h5py plus calendar, no production imports; same agent',
              'new_predictions': 0, 'new_scores': 0}
    name = 'qa_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%f') + '.json'
    with (OUT / name).open('x', encoding='utf-8') as f: json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--partial', action='store_true')
    main(p.parse_args().partial)
