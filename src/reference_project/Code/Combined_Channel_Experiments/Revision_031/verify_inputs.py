"""Independent h5py/calendar verification. Does not import acquisition code."""
import calendar
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_031'
PLAN = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_030'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def text(v):
    return v.decode() if isinstance(v, bytes) else str(v)

def main():
    assert not (OUT / 'independent_qa.json').exists(), 'preserve earlier QA'
    acquired = read(OUT / 'acquisition_complete.json')
    assert acquired['batches'] == 30 and acquired['new_values'] == 421104
    seen = set()
    name_map = {'temperature': 't', 'geopotential': 'z', 'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}
    allowed_units = {'t': {'K'}, 'z': {'m**2 s**-2', 'm2 s-2', 'm^2 s^-2'},
                     'u': {'m s**-1', 'm s-1', 'm/s'}, 'v': {'m s**-1', 'm s-1', 'm/s'}}
    rows = []
    for b in read(PLAN / 'input_batches.json'):
        folder = OUT / 'batches' / f"{b['id']:03d}"
        raw = folder / 'raw.nc'
        prod = read(folder / 'complete.json')
        assert digest(raw) == prod['sha256']
        planned_time = {calendar.timegm((b['year'], b['month'], day, hour, 0, 0))
                        for day in b['days'] for hour in b['hours_utc']}
        planned = {(b['station'], t, name_map[v], p) for t in planned_time
                   for v in b['variables'] for p in b['pressure_levels_hpa']}
        actual = set()
        with h5py.File(raw, 'r') as f:
            assert f['latitude'].size == 1 and f['longitude'].size == 1
            assert np.allclose(f['latitude'][()], b['grid_latitude'], rtol=0, atol=1e-7)
            assert np.allclose((f['longitude'][()] + 180) % 360 - 180, b['grid_longitude'], rtol=0, atol=1e-7)
            tn = 'valid_time' if 'valid_time' in f else 'time'
            pn = 'pressure_level' if 'pressure_level' in f else 'level'
            time_unit = text(f[tn].attrs['units'])
            matched = re.fullmatch(r'(seconds|hours) since (.+)', time_unit)
            assert matched, time_unit
            origin = dt.datetime.fromisoformat(matched[2].replace('Z', '+00:00'))
            if origin.tzinfo is None: origin = origin.replace(tzinfo=dt.timezone.utc)
            base = calendar.timegm(origin.utctimetuple())
            scale = 1 if matched[1] == 'seconds' else 3600
            times = [base + float(v) * scale for v in f[tn][()]]
            assert set(times) == planned_time and len(times) == len(set(times))
            levels = f[pn][()].astype(float).tolist()
            assert set(levels) == set(b['pressure_levels_hpa']) and len(levels) == len(set(levels))
            assert text(f[pn].attrs['units']) in ('hPa', 'millibars', 'mbar')
            names = set(name_map.values()).intersection(f.keys())
            assert names == {name_map[v] for v in b['variables']}
            for n in names:
                d = f[n]
                axes = [f[refs[0]].name.lstrip('/') for refs in d.attrs['DIMENSION_LIST']]
                assert set(axes) == {tn, pn, 'latitude', 'longitude'}
                vals = np.transpose(d[()], [axes.index(x) for x in [tn, pn, 'latitude', 'longitude']])[:, :, 0, 0]
                assert text(d.attrs['units']) in allowed_units[n]
                assert vals.shape == (len(times), len(levels)) and np.isfinite(vals).all()
                if '_FillValue' in d.attrs: assert not np.any(vals == d.attrs['_FillValue'])
                if 'scale_factor' in d.attrs or 'add_offset' in d.attrs: raise AssertionError('packed encoding requires explicit independent adaptation')
                assert int(vals.size) == prod['variables'][n]['count']
                assert float(vals.min()) == prod['variables'][n]['min']
                assert float(vals.max()) == prod['variables'][n]['max']
                actual.update((b['station'], int(t), n, int(p)) for t in times for p in levels)
        assert actual == planned and len(actual) == b['scalar_values']
        assert not actual.intersection(seen)
        seen.update(actual)
        rows.append({'batch': b['id'], 'sha256': digest(raw), 'values': len(actual)})
    scope = {(r['station'], r['utc_epoch_seconds']) for r in read(PLAN / 'scope_hours.json')}
    assert len(seen) == 421104 and {(s, t) for s, t, v, p in seen} == scope and len(scope) == 5664
    for filename in ['sources_before.json', 'code_before.json']:
        for name, expected in read(OUT / filename).items(): assert digest(name) == expected, name
    body_bytes = sum(json.loads(line)['bytes'] for line in (OUT / 'network_ledger.jsonl').read_text().splitlines())
    assert body_bytes == acquired['http_body_bytes'] and body_bytes <= 255 * 1024 * 1024
    report = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'PASS_INPUT_QA_ONLY',
              'new_values': len(seen), 'hours': len(scope), 'batches': rows, 'http_body_bytes': body_bytes,
              'all_sources_unchanged': True, 'new_predictions': 0, 'new_scores': 0,
              'independence': 'h5py plus calendar and set algebra, same agent, not external review',
              'script_sha256': digest(__file__)}
    with (OUT / 'independent_qa.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'batches'}))

if __name__ == '__main__': main()
