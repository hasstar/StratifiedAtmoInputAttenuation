"""Pinned, additive transport repair. Original scientific modules stay read-only."""
import sys
from pathlib import Path

REPAIR_CODE = Path(__file__).resolve().parent
ORIGINAL_CODE = REPAIR_CODE.parent / 'Revision_037'
sys.path.insert(0, str(ORIGINAL_CODE))
from freeze import ROOT, OUT, read, sha, now, emit, need, rel
from transport import ACQ, HOST, stage_lock
import ncar_inputs as original

REPAIR_OUT = OUT / 'repair_v003'
MANIFEST = REPAIR_OUT / 'source_manifest.json'
PLAN_SHA = '92c2f404c4b2e4c9840857b829f10cd807417be412493824fb5285562eddf1b2'


def check_guards(require_tests=True):
    need(not (OUT / 'STOP_REQUESTED.json').exists(), 'T015 stop requested')
    manifest = read(MANIFEST)
    need(manifest['explicit_user_repair_authorization'] is True, 'Repair authority missing')
    need(sha(ACQ / 'ncar_plan.json') == PLAN_SHA == manifest['plan_sha256'], 'Immutable plan changed')
    need({rel(p) for p in REPAIR_CODE.iterdir() if p.is_file()} == set(manifest['repair_sources']), 'Repair source set changed')
    for group in ['original_sources', 'repair_sources', 'plan_inputs']:
        for p, digest in manifest[group].items():
            need(sha(ROOT / p) == digest, 'Pinned source/input changed: ' + p)
    if require_tests:
        qa = read(REPAIR_OUT / 'offline_qa.json')
        need(qa['status'] == 'PASS' and qa['source_manifest_sha256'] == sha(MANIFEST), 'Offline repair QA missing/stale')
    return manifest


def child_node(parent, position):
    """Select exactly one original time; do not add/reorder any other axis."""
    need(0 <= position < len(parent['epochs']), 'Child time out of bounds')
    node = dict(parent)
    node['axes'] = [[parent['axes'][0][position]]] + [list(a) for a in parent['axes'][1:]]
    node['epochs'] = [parent['epochs'][position]]
    node['url'] = HOST + 'dodsC/' + parent['url_path'] + '.ascii?' + parent['variable'] + ''.join(original.expr(a) for a in node['axes'])
    return node


def hourly_path(parent, position):
    return ACQ / 'ncar_hourly_v003' / parent['id'] / f't_{parent["axes"][0][position]:06d}.ascii'
