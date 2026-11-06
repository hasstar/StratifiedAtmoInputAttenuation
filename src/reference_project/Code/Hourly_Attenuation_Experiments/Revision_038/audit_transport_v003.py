"""Independent raw-hour to derived-parent audit; never calls producer decoder."""
import itertools
import re
import numpy as np
from common import *
from audit_inputs import independent_ascii


def verified_array(text, node):
    need(node['epochs'] == [node['start'] + 3600 * i for i in node['axes'][0]], 'Independent exact UTC hour/index mapping')
    need(node['url_path'] in text.split('--------------------------------')[0], 'Independent source identity')
    arrays = independent_ascii(text)
    var = node['variable']
    coords = {'time': [(t + 2208988800) // 3600 for t in node['epochs']],
              'latitude': [90 - i * .25 for i in node['latitude_indices']],
              'longitude': [i * .25 for i in node['longitude_indices']]}
    if node['pressure']:
        coords['level'] = node['levels']
    need(set(arrays) == {var + '.' + var} | {var + '.' + c for c in coords}, 'Independent variable set')
    for c, values in coords.items():
        np.testing.assert_array_equal(arrays[var + '.' + c], values)
    a = arrays[var + '.' + var]
    need(a.shape == tuple(map(len, node['axes'])), 'Independent array shape')
    need(np.isfinite(a).all() and (np.abs(a) < 1e19).all(), 'Independent nonfinite/fill')
    return a


def checked_payload(path, expected_url, raw_required=True):
    receipt = read(path.with_name(path.name + '.receipt.json'))
    need(receipt['url'] == expected_url and receipt['validation'] == 'PASS', 'Independent receipt URL/QA')
    need(path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'], 'Independent payload SHA/bytes')
    if raw_required:
        need(receipt['status'] == 200 and receipt['method'] == 'GET', 'Actual raw HTTP response required')
    return path.read_text(encoding='utf-8'), receipt


def audit_parent(node):
    path = ACQ / 'ncar_values' / (node['id'] + '.ascii')
    text, receipt = checked_payload(path, node['url'], raw_required=False)
    full = verified_array(text, node)
    metadata = ACQ / 'ncar_metadata' / (original.fp(node['url_path'])[:24] + '.das')
    das, _ = checked_payload(metadata, HOST + 'dodsC/' + node['url_path'] + '.das')
    block = re.search(r'\b' + node['variable'] + r'\s*\{([^{}]*)\}', das, re.S)
    need(block is not None, 'Independent metadata variable')
    table, param, short = node['spec'].split('_')
    for key, value in [('ecmwf_local_table', int(table)), ('ecmwf_parameter', int(param))]:
        m = re.search(r'Int32 ' + key + r'\s+(\d+)\s*;', block[1])
        need(m is not None and int(m[1]) == value, 'Independent metadata identity')
    need(f'"{original.EXPECTED_UNITS[node["spec"]]}"' in block[1], 'Independent metadata units')
    for x in re.findall(r'(?:_FillValue|missing_value)\s+([-+\d.eE]+)\s*;', block[1]):
        need(not (full == float(x)).any(), 'Independent metadata fill')
    if receipt.get('method') != 'DERIVED_TIME_CONCAT':
        need(receipt['status'] == 200 and receipt['method'] == 'GET', 'Original response receipt')
        return dict(id=node['id'], kind='RAW_HTTP', sha256=receipt['sha256'], values=int(full.size), hourly_fragments=0)
    need(receipt.get('status') is None and receipt.get('source_kind') == 'DERIVED_NCAR_ASCII', 'Do not misrepresent composite as HTTP response')
    need(receipt['plan_sha256'] == PLAN_SHA and receipt['parent_url_logical_only'] is True, 'Composite frozen plan')
    refs = receipt['fragments']
    need(len(refs) == len(node['epochs']), 'Missing/duplicate hours')
    seen = set()
    for pos, item in enumerate(refs):
        # Reconstruct child request independently, without the producer helper.
        axes = [[node['axes'][0][pos]]] + node['axes'][1:]
        def axis(xs):
            return f'[{xs[0]}]' if len(xs) == 1 else f'[{xs[0]}:{xs[1]-xs[0]}:{xs[-1]}]'
        url = HOST + 'dodsC/' + node['url_path'] + '.ascii?' + node['variable'] + ''.join(axis(a) for a in axes)
        child = dict(node, axes=axes, epochs=[node['epochs'][pos]], url=url)
        expected = ACQ / 'ncar_hourly_v003' / node['id'] / f't_{node["axes"][0][pos]:06d}.ascii'
        need(item['source'] == rel(expected) and item['time_position'] == pos, 'Fragment path/order')
        need(item['source'] not in seen, 'Duplicate fragment')
        seen.add(item['source'])
        raw, rr = checked_payload(expected, url)
        need(rr['sha256'] == item['sha256'] and rr['bytes'] == item['bytes'], 'Fragment reference SHA/bytes')
        a = verified_array(raw, child)
        np.testing.assert_array_equal(a[0], full[pos])
    return dict(id=node['id'], kind='DERIVED_TIME_CONCAT', sha256=receipt['sha256'], values=int(full.size), hourly_fragments=len(refs))


def audit_all(require_complete=True):
    plan = read(ACQ / 'ncar_plan.json')
    need(sha(ACQ / 'ncar_plan.json') == PLAN_SHA, 'Audit plan changed')
    nodes = {n['id']: n for n in plan['nodes']}
    paths = list((ACQ / 'ncar_values').glob('*.ascii'))
    need({p.stem for p in paths} <= set(nodes), 'Unknown parent cache')
    for p in (ACQ / 'ncar_values').glob('*.receipt.json'):
        need(p.with_name(p.name.removesuffix('.receipt.json')).exists(), 'Orphan parent receipt')
    results = [audit_parent(nodes[p.stem]) for p in paths]
    ids = {r['id'] for r in results}
    need(ids == {n['id'] for n in plan['nodes'][:len(ids)]}, 'Noncontiguous verified parent checkpoint')
    if require_complete:
        need(len(results) == 2012 and sum(r['values'] for r in results) == 100232, 'Incomplete original scope')
    return dict(status='PASS' if require_complete else 'PASS_AVAILABLE_PARENTS_ONLY', utc=now(),
                nodes=len(results), plan_sha256=PLAN_SHA, source_manifest_sha256=sha(MANIFEST),
                grid_values=sum(r['values'] for r in results),
                derived_parents=sum(r['kind'] == 'DERIVED_TIME_CONCAT' for r in results),
                raw_hourly_fragments=sum(r['hourly_fragments'] for r in results), parents=results,
                full_input_qa=False, training_calls=0)


if __name__ == '__main__':
    with stage_lock('transport_provenance_v003'):
        check_guards()
        result = audit_all()
        emit(REPAIR_OUT / 'transport_provenance_qa.json', result)
        print(result['status'], result['nodes'], flush=True)
