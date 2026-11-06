"""One bounded stream, resumable hourly payloads, honest derived-parent receipts."""
import argparse
import json
import os
import time
import uuid
import itertools
import requests
import psutil
import numpy as np
from common import *
from ncar_transport_v002 import FetcherV2
from audit_transport_v003 import audit_parent, audit_all, verified_array
from cds_three import step


class TransientReadFailure(RuntimeError):
    pass


def composite_ascii(node, arrays):
    need(len(arrays) == len(node['epochs']), 'Complete hour set required')
    for a in arrays:
        need(a.shape == (1, *map(len, node['axes'][1:])), 'Hour layout')
    full = np.concatenate(arrays, axis=0)
    var = node['variable']
    lines = ['// DERIVED_TIME_CONCAT v003; not a remote daily HTTP response.',
             '// Exact raw hourly responses and SHA identities are in the adjacent receipt.',
             'Dataset { } ' + node['url_path'] + ';', '---------------------------------------------',
             var + '.' + var + ''.join(f'[{x}]' for x in full.shape)]
    for index in itertools.product(*(range(x) for x in full.shape[:-1])):
        lines.append(''.join(f'[{x}]' for x in index) + ', ' + ', '.join(repr(float(x)) for x in full[index]))
    coords = {'time': [(t - original.E1900) // 3600 for t in node['epochs']]}
    if node['pressure']:
        coords['level'] = node['levels']
    coords.update(latitude=[90 - i * .25 for i in node['latitude_indices']], longitude=[i * .25 for i in node['longitude_indices']])
    for key, values in coords.items():
        lines.extend(['', var + '.' + key + f'[{len(values)}]', ', '.join(str(v) for v in values)])
    return '\n'.join(lines) + '\n'


class FetcherV3(FetcherV2):
    def __init__(self, callback=lambda: None, report=lambda **kw: None):
        super().__init__()
        self.callback = callback
        self.report = report

    def fetch_raw(self, url, path, validate, read_timeout, attempts):
        need(url.startswith(HOST + 'dodsC/files/d633000/') and (url.endswith('.das') or '.nc.ascii?' in url), 'Exact public archive only')
        need(not (OUT / 'STOP_REQUESTED.json').exists(), 'T015 stopped')
        receipt_path = path.with_name(path.name + '.receipt.json')
        if path.exists():
            rr = read(receipt_path)
            need(rr['status'] == 200 and rr['method'] == 'GET' and rr['validation'] == 'PASS', 'Raw cache receipt')
            need(rr['url'] == url and rr['sha256'] == sha(path) and rr['bytes'] == path.stat().st_size, 'Raw cache integrity')
            text = path.read_text(encoding='utf-8')
            return text, validate(text), rr
        need(not receipt_path.exists(), 'Orphan raw receipt')
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, attempts + 1):
            self.callback()
            need(not (OUT / 'STOP_REQUESTED.json').exists(), 'T015 stopped')
            ident = uuid.uuid4().hex
            part = ACQ / 'partial_history' / f'{path.name}.{ident}.part'
            part.parent.mkdir(parents=True, exist_ok=True)
            tic = time.monotonic()
            started = now()
            try:
                self.s.cookies.clear()
                with self.s.get(url, stream=True, allow_redirects=False, timeout=(10, read_timeout)) as response:
                    if response.status_code != 200:
                        emit(ACQ / 'http_failures' / f'{ident}.json', dict(utc=now(), url=url, status=response.status_code, attempt=attempt, transport='v003'))
                        if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                            raise TransientReadFailure('Bounded transient HTTP ' + str(response.status_code))
                        raise RuntimeError('Nontransient HTTP ' + str(response.status_code))
                    count = 0
                    length = response.headers.get('Content-Length')
                    with part.open('xb') as handle:
                        for chunk in response.iter_content(65536):
                            count += len(chunk)
                            need(count <= 20 * 1024**2 and time.monotonic() - tic < 240, 'Response byte/total-time cap')
                            handle.write(chunk)
                    need(count > 0 and (length is None or count == int(length)), 'Response completeness')
                    text = part.read_text(encoding='utf-8')
                    value = validate(text)
                    digest = sha(part)
                    need(not path.exists(), 'Never overwrite payload')
                    part.rename(path)
                    rr = dict(url=url, status=200, method='GET', validation='PASS', started_utc=started, completed_utc=now(),
                              bytes=count, sha256=digest, content_length=None if length is None else int(length),
                              source=rel(path), transport='v003', read_timeout_seconds=read_timeout, attempt=attempt,
                              elapsed_seconds=time.monotonic() - tic, source_kind='RAW_NCAR_HTTP')
                    emit(receipt_path, rr)
                    print(json.dumps(dict(acquired=rel(path), bytes=count, seconds=rr['elapsed_seconds'], utc=now())), flush=True)
                    return text, value, rr
            except (requests.Timeout, requests.ConnectionError) as exc:
                emit(ACQ / 'http_failures' / f'{ident}.json', dict(utc=now(), url=url, error_type=type(exc).__name__,
                     attempt=attempt, transport='v003', read_timeout_seconds=read_timeout))
                if attempt == attempts:
                    raise TransientReadFailure('Bounded network failure for exact NCAR input') from None
            except TransientReadFailure:
                if attempt == attempts:
                    raise
            if attempt < attempts:
                time.sleep(15)
        raise RuntimeError('Bounded read exhausted')

    def parent(self, node, das):
        path = ACQ / 'ncar_values' / (node['id'] + '.ascii')
        need(not path.exists() and not path.with_name(path.name + '.receipt.json').exists(), 'Existing parent must be reused, not written')
        validate = lambda s: original.decode(s, das, node)
        existing_hours = any((ACQ / 'ncar_hourly_v003' / node['id']).glob('*.ascii'))
        known_slow = node['spec'] == '128_130_t' and node['pressure'] and 5 in node['levels']
        if not existing_hours and not known_slow:
            try:
                self.fetch_raw(node['url'], path, validate, 60, 1)
                return audit_parent(node)
            except TransientReadFailure:
                need(len(node['epochs']) > 1, 'Single-hour original request failed; inspection required')
                emit(REPAIR_OUT / 'fallbacks' / (uuid.uuid4().hex + '.json'), dict(utc=now(), parent=node['id'],
                     reason='Transient original time-window read failure', action='exact_existing_hours_only'))
        arrays, refs = [], []
        for pos in range(len(node['epochs'])):
            child = child_node(node, pos)
            self.report(current_stage='DOWNLOADING_NCAR_HOUR', current_parent_id=node['id'],
                        current_epoch=child['epochs'][0], verified_hours_in_parent=pos, parent_hours=len(node['epochs']))
            raw, values, rr = self.fetch_raw(child['url'], hourly_path(node, pos), lambda s: original.decode(s, das, child), 180, 2)
            independently = verified_array(raw, child)
            np.testing.assert_array_equal(independently.ravel(), values)
            arrays.append(independently)
            refs.append(dict(source=rr['source'], sha256=rr['sha256'], bytes=rr['bytes'], time_position=pos))
        derived = composite_ascii(node, arrays)
        np.testing.assert_array_equal(validate(derived), np.concatenate(arrays, axis=0).ravel())
        verified_array(derived, node)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(derived)
        emit(path.with_name(path.name + '.receipt.json'), dict(url=node['url'], status=None,
             method='DERIVED_TIME_CONCAT', validation='PASS', source_kind='DERIVED_NCAR_ASCII',
             parent_url_logical_only=True, plan_sha256=PLAN_SHA, source=rel(path), sha256=sha(path),
             bytes=path.stat().st_size, completed_utc=now(), transport='v003', fragments=refs))
        # A derived parent is not counted until this independent read-back succeeds.
        result = audit_parent(node)
        emit(REPAIR_OUT / 'parent_qa' / (node['id'] + '.json'), dict(utc=now(), status='PASS', **result))
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-new-parents', type=int, default=0)
    args = ap.parse_args()
    with stage_lock('ncar_acquire_v003'):
        check_guards()
        need(not (ACQ / 'ncar_download_complete.json').exists(), 'Acquisition already complete')
        ident = uuid.uuid4().hex
        emit(REPAIR_OUT / 'runs' / (ident + '.json'), dict(utc=now(), pid=os.getpid(), create_time=psutil.Process().create_time(),
             command=sys.argv, source_manifest_sha256=sha(MANIFEST)))
        cached = audit_all(require_complete=False)
        emit(REPAIR_OUT / 'prechecks' / (ident + '.json'), cached)
        count = cached['nodes']
        initial = count
        tic = time.monotonic()
        cds_clock = -float('inf')

        def report(**kw):
            obj = dict(utc=now(), pid=os.getpid(), create_time=psutil.Process().create_time(), stage='ACQUIRING',
                       transport='v003', validated_ncar_nodes=count, total_ncar_nodes=2012, new_parent_nodes_this_run=count-initial,
                       elapsed_seconds=time.monotonic()-tic, **kw)
            temp = OUT / 'acquisition_progress.tmp'
            temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
            os.replace(temp, OUT / 'acquisition_progress.json')

        def cds_tick():
            nonlocal cds_clock
            if time.monotonic() - cds_clock >= 60:
                state = step()
                cds_clock = time.monotonic()
                print('CDS', json.dumps(state), flush=True)

        fetcher = FetcherV3(cds_tick, report)
        plan = read(ACQ / 'ncar_plan.json')
        report(current_stage='VERIFIED_CACHE_REUSED')
        for node in plan['nodes'][count:]:
            if args.max_new_parents and count - initial >= args.max_new_parents:
                break
            report(current_stage='DOWNLOADING_NCAR_PARENT', current_parent_id=node['id'], current_file=node['url_path'])
            fileid = original.fp(node['url_path'])[:24]
            das, _, _ = fetcher.fetch_raw(HOST + 'dodsC/' + node['url_path'] + '.das', ACQ / 'ncar_metadata' / (fileid + '.das'),
                lambda s: original.contract(s, node['spec'], node['pressure']), 180, 2)
            result = fetcher.parent(node, das)
            count += 1
            report(current_stage='PARENT_VERIFIED', current_parent_id=node['id'], current_file=node['url_path'])
            print('PARENT_VERIFIED', count, 'OF 2012', node['id'], result['kind'], flush=True)
            cds_tick()
        if count == 2012:
            check_guards()
            qa = audit_all()
            emit(REPAIR_OUT / 'transport_provenance_qa.json', qa)
            emit(ACQ / 'ncar_download_complete.json', dict(utc=now(), nodes=2012, plan_sha256=PLAN_SHA,
                 status='DOWNLOADED_NOT_YET_INDEPENDENT_INPUT_QA', transport='v003',
                 transport_qa_sha256=sha(REPAIR_OUT / 'transport_provenance_qa.json')))
        print('NCAR_VERIFIED_PARENTS', count, 'OF 2012; full input QA separate', flush=True)


if __name__ == '__main__':
    main()
