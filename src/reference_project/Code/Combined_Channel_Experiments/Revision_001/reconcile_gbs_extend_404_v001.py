"""One-shot, evidence-gated recovery of a single diagnosed v011 NCAR 404.

Not a worker or general retry policy. Never advances the plan or runs a model.
All frozen implementations are read-only; existing failure evidence is retained.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

JOB_ID = 'ncar008_50c977302a442e5b6baeb64e'
PLAN_FP = 'b58d2b9f1f83970cca410195504cd0aa21cb0f7e57cf004ddb1e43100c5f56db'
PLAN_SHA = '1f9be56c4351940342a135b487d44aa1d6914c4a1a3dfc0ef4b040f43b53fdac'
REQUEST_FP = '9a1115a440308496910dda14937d5191ff9abb6953dcf8f088dd330578e685b3'
META_SHA = '5fd0af25557ef6782289fd819c88daa39785e277a28909c9839a14b3192e87ea'
FILE = 'e5.oper.an.pl/200308/e5.oper.an.pl.128_131_u.ll025uv.2003082800_2003082823.nc'
BASE = 'https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/'
URL = BASE + FILE + '.ascii?U[0:1:23][24][156][1434]'
RECOVERY = 'recovery_001_20261016_0232'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def guard(saved, plan_fp, request_fp, stopped, locked):
    require(not stopped and not locked, 'Stop flag or another writer; no recovery')
    require(plan_fp == PLAN_FP and request_fp == REQUEST_FP, 'Fingerprint mismatch')
    require(saved.get('request_fingerprint') == REQUEST_FP, 'State request mismatch')
    require(saved.get('plan_fingerprint') == PLAN_FP, 'State plan mismatch')
    require(saved.get('status') == 'BLOCKED_DOWNLOAD_INTEGRITY_OR_ACCESS', 'Not the diagnosed blocked state')
    require(saved.get('attempts') == 1, 'This one-shot recovery was already attempted')
    require(saved.get('error') == {'error_type': 'HttpFailure', 'message': 'NCAR HTTP 404'}, 'Not the diagnosed 404')
    n = saved['job']
    require(n['id'] == JOB_ID and n['file'] == FILE and n['url'] == URL, 'Different request')
    require(n['metadata_url'] == BASE + FILE + '.das', 'Different metadata source')
    require(n['variable'] == 'u_component_of_wind' and n['level_hpa'] == 650, 'Parameter or layer changed')
    require(n['expected_values'] == 24 and n['axes'] == [list(range(24)), [24], [156], [1434]], 'Subset expanded')
    require(n['target'] == 'payloads/' + JOB_ID + '.ascii', 'Unsafe payload target')


def check_dds(text):
    declaration = re.search(r'Float32 U\[time = (\d+)\]\[level = (\d+)\]\[latitude = (\d+)\]\[longitude = (\d+)\];', text)
    require(declaration and tuple(map(int, declaration.groups())) == (24, 1, 1, 1), 'Unexpected constrained DAP structure')
    require('files/d633000/' + FILE in text, 'Different dataset in DDS')


def main():
    # Import redirects only this process to the current v011 output. No entry point.
    import weather_download_gbs_extend_v011 as current
    app, io, dest = current.app, current.io, current.DEST
    plan = app.load_plan()  # Full frozen source and old-protection SHA gate.
    require(app.SHA(current.PLAN) == PLAN_SHA, 'Plan SHA changed')
    node = next(n for n in app.nodes(plan) if n['id'] == JOB_ID)
    before = app.state(node)
    guard(before, plan['fingerprint'], app.FP(node), current.stopped(), io.lock_held())
    recovery = dest / RECOVERY
    require(recovery.is_dir() and not (recovery/'execution_started.json').exists(), 'One-shot output absent or already used')
    require(not (dest/'acquisition_complete.json').exists(), 'Acquisition already complete')
    require(not (dest/node['target']).exists(), 'Unexpected formal payload; inspect separately')

    # Snapshot all existing v011 file identities before any live-state mutation.
    files = [p for p in dest.rglob('*') if p.is_file() and not p.is_relative_to(recovery)]
    old_hashes = {str(p.relative_to(dest)): app.SHA(p) for p in files}
    io.immutable(recovery/'before_manifest.json', old_hashes)
    for p in files:
        rel = p.relative_to(dest)
        if rel.parts[0] in ('failures', 'logs', 'workers') or str(rel) in ('active_worker.json', 'progress.json') or p == dest/'jobs'/(JOB_ID+'.json'):
            target = recovery/'before'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            require(app.SHA(target) == app.SHA(p), 'Snapshot copy differs')
    io.immutable(recovery/'execution_started.json', {'utc':io.now(), 'script_sha256':app.SHA(Path(__file__)),
        'job_id':JOB_ID, 'plan_fingerprint':PLAN_FP, 'request_fingerprint':REQUEST_FP,
        'scope':'One original exact GET after official metadata/structure confirmation; no other jobs, advance, model or retry policy change'})

    try:
        with io.writer('gbs_extend_404_reconcile_001'):
            require(app.state(node) == before and not current.stopped(), 'State changed before lock')
            proofs = {}
            for name, url in [('metadata', node['metadata_url']), ('structure', URL.replace('.nc.ascii?', '.nc.dds?'))]:
                target = recovery/(name+'.response')
                receipt = app.get_body(url, target, node)
                io.immutable(recovery/(name+'.receipt.json'), receipt)
                proofs[name] = receipt
            das = (recovery/'metadata.response').read_text(encoding='utf-8')
            current.transport.metadata_contract(das, node)
            require(proofs['metadata']['sha256'] == META_SHA, 'Official metadata changed since diagnosis')
            check_dds((recovery/'structure.response').read_text(encoding='utf-8'))
            cached = dest/'metadata'/(app.FP(node['file'])+'.das')
            require(app.SHA(cached) == META_SHA, 'Cached metadata differs from official response')
            io.immutable(recovery/'access_gate.json', {'status':'PASS', 'utc':io.now(), 'proofs':proofs,
                'reason':'Same file previously served 1/5 hPa data; current official DAS and same-constraint DDS are HTTP 200. Exact data recovery still must pass decoding.'})

            # Exactly one attempt. The frozen routine retains another failure and
            # keeps BLOCKED/RETRY if it fails; it cannot advance any other request.
            app.download(node, plan)
            after = app.state(node)
            require(after.get('status') == 'VERIFIED_SUBSET', 'Exact subset recovery did not verify; do not start worker')
            require(after.get('attempts') == 2 and after['receipt'].get('http_status') == 200, 'No actual successful second GET')
            rows = app.read_rows(node, after)
            require(len(rows) == 24, 'Wrong recovered sample count')
            io.immutable(recovery/'recovered_job.json', after)

        # Validate frozen sources again, all prior payloads, all prior failures and
        # every old current-directory file except the two intentional live states.
        app.load_plan()
        allowed = {'active_worker.json', str(Path('jobs')/(JOB_ID+'.json'))}
        changed = [name for name, digest in old_hashes.items() if app.SHA(dest/name) != digest]
        require(set(changed) <= allowed, 'Unexpected existing-file mutation: '+str(changed))
        result = {'status':'RECOVERED_EXACT_SUBSET', 'utc':io.now(), 'job_id':JOB_ID,
            'verified_values':24, 'changed_existing_files':changed, 'old_files_checked':len(old_hashes),
            'failure_evidence_retained':True, 'frozen_source_and_protected_hashes_match':True,
            'root_cause':'Original 404 no longer reproduces; precise server-side cause unknown. No general reclassification of 404.',
            'progress':current.progress(plan, True), 'fit_calls':0, 'predict_calls':0}
        io.immutable(recovery/'result.json', result)
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        io.immutable(recovery/'recovery_failure.json', {'utc':io.now(), 'error':type(exc).__name__+': '+str(exc),
            'policy':'Retain evidence, do not start worker without further diagnosis'})
        raise


if __name__ == '__main__':
    main()
