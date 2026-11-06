"""Separately sealed scheduler; immutable transport/scientific code is imported."""
import json
import os
import sys
import uuid
from pathlib import Path

CODE_V4 = Path(__file__).resolve().parent
CODE_V3 = CODE_V4.parent / 'Revision_038'
sys.path.insert(0, str(CODE_V3))
import common as upstream
from common import ROOT, OUT, ACQ, HOST, PLAN_SHA, ORIGINAL_CODE, original
from common import read, sha, now, emit, need, rel, stage_lock

OUT_V4 = OUT / 'repair_v004'
MANIFEST_V4 = OUT_V4 / 'source_manifest.json'
UPSTREAM_SHA = '57b51cbdfcb2a1d1743520655fb6f943c6e48216aa83d11967ccf96920c751f9'


def stopped():
    need(not (OUT / 'STOP_REQUESTED.json').exists(), 'T015 stop requested')


def check_guards(require_tests=True):
    stopped()
    need(sha(upstream.MANIFEST) == UPSTREAM_SHA, 'Upstream manifest changed')
    upstream.check_guards()
    manifest = read(MANIFEST_V4)
    need(manifest['explicit_user_repair_authorization'] is True, 'v004 authority missing')
    need(manifest['upstream_manifest_sha256'] == UPSTREAM_SHA, 'Upstream pin mismatch')
    need(manifest['plan_sha256'] == PLAN_SHA, 'Immutable plan mismatch')
    need({rel(p) for p in CODE_V4.iterdir() if p.is_file()} == set(manifest['scheduler_sources']), 'v004 source set changed')
    for p, digest in manifest['scheduler_sources'].items():
        need(sha(ROOT / p) == digest, 'v004 source changed: ' + p)
    need(sha(ACQ / 'cds_three/plan.json') == manifest['cds_plan_sha256'], 'CDS plan changed')
    if require_tests:
        qa = read(OUT_V4 / 'offline_qa.json')
        need(qa['status'] == 'PASS' and qa['source_manifest_sha256'] == sha(MANIFEST_V4), 'v004 tests missing or stale')
    return manifest


def current_status(path, record):
    """Replace only a declared current-state file; all important events are separate."""
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    emit(temp, record)
    os.replace(temp, path)


def cds_cache_audit():
    """Local identities and bytes only; never contacts a remote service."""
    from cds_three import DEST, COLLECTION, VARIABLES, fingerprint
    from assemble_inputs import read_cds
    plan = read(DEST / 'plan.json')
    need(len(plan['jobs']) == 12, 'Exact 12 CDS batches required')
    complete, active, identities = [], [], 0
    for job in plan['jobs']:
        need(job['collection_id'] == COLLECTION and job['request']['variable'] == VARIABLES, 'CDS parameter scope')
        need(fingerprint(job['request']) == job['request_sha256'], 'CDS request changed')
        folder = DEST / 'jobs' / job['id']
        intent, submitted, downloaded = [folder / f'{name}.json' for name in ('intent', 'submitted', 'download')]
        need(not (folder / 'failure.json').exists(), 'Recorded CDS failure needs inspection')
        if intent.exists():
            item = read(intent)
            need(item['request_sha256'] == job['request_sha256'] and item['plan_sha256'] == sha(DEST / 'plan.json'), 'CDS intent identity')
            need(item['request'] == job['request'] and item['collection_id'] == COLLECTION, 'CDS intent scope')
            need(submitted.exists(), 'Unknown CDS submission; no POST retry')
            saved = read(submitted)
            need(saved['request_sha256'] == job['request_sha256'], 'CDS submitted identity')
            identities += 1
        else:
            need(not submitted.exists() and not downloaded.exists(), 'Orphan CDS identity')
        if downloaded.exists():
            receipt = read(downloaded)
            need(receipt['request_id'] == saved['request_id'] and receipt['request_sha256'] == job['request_sha256'], 'CDS download identity')
            path = (DEST / receipt['file']).resolve()
            need(path.is_relative_to(folder.resolve()), 'CDS payload outside batch')
            need(path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'], 'CDS download integrity')
            values, _ = read_cds(job, receipt)
            complete.append(dict(batch=job['id'], bytes=receipt['bytes'], values=len(values), sha256=receipt['sha256']))
        elif intent.exists():
            active.append(job['id'])
    need(len(active) <= 1, 'Only one unresolved CDS job allowed')
    return dict(downloaded=len(complete), total=12, identities=identities, active=active, payloads=complete)
