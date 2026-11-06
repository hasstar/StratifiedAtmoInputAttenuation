"""New repair lineage; all executed upstream code and journals remain read-only."""
import sys
from pathlib import Path

CODE_V5 = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_V5.parent / 'Revision_039'))
import scheduler_common as v4
from scheduler_common import ROOT, OUT, ACQ, HOST, PLAN_SHA, ORIGINAL_CODE, original
from scheduler_common import read, sha, now, emit, need, rel, stage_lock, stopped, current_status, cds_cache_audit
from retry_scheduler import Journal

OUT_V5 = OUT / 'repair_v005'
MANIFEST_V5 = OUT_V5 / 'source_manifest.json'
V4_MANIFEST_SHA = 'd0cd2772fe478a4551eb1db93600526c735579db680f3a2b4379153cd08f1329'
V4_JOURNAL = v4.OUT_V4 / 'scheduler_journal.jsonl'
V4_JOURNAL_SHA = '0f3cc8263e2793571154d008d52658406a9da047cbc2174e7f3c43905462c1a1'
V4_RESULT = v4.OUT_V4 / 'worker_results/606fe7b5f235d20c0bf6ed7c1fa32695.json'
V4_RESULT_SHA = '285ecf4c499392e20f326df1e689ea88d8163fba1d5a98fe9489eb3fc3633b20'


def check_guards(require_tests=True):
    v4.check_guards()
    need(sha(v4.MANIFEST_V4) == V4_MANIFEST_SHA, 'v004 source manifest changed')
    need(sha(V4_JOURNAL) == V4_JOURNAL_SHA, 'Inherited scheduler checkpoint changed')
    need(sha(V4_RESULT) == V4_RESULT_SHA, 'Inspected old worker result changed')
    manifest = read(MANIFEST_V5)
    need(manifest['explicit_user_repair_authorization'] is True, 'v005 authorization missing')
    need(manifest['upstream_manifest_sha256'] == V4_MANIFEST_SHA, 'v005 upstream pin mismatch')
    need(manifest['baseline_journal_sha256'] == V4_JOURNAL_SHA, 'v005 journal pin mismatch')
    need(manifest['plan_sha256'] == PLAN_SHA, 'v005 frozen NCAR plan mismatch')
    need(sha(ACQ / 'cds_three/plan.json') == manifest['cds_plan_sha256'], 'Frozen CDS plan changed')
    need({rel(p) for p in CODE_V5.iterdir() if p.is_file()} == set(manifest['repair_sources']), 'v005 source set changed')
    for path, digest in manifest['repair_sources'].items():
        need(sha(ROOT / path) == digest, 'v005 source changed: ' + path)
    if require_tests:
        qa = read(OUT_V5 / 'offline_qa.json')
        need(qa['status'] == 'PASS' and qa['source_manifest_sha256'] == sha(MANIFEST_V5), 'v005 QA missing/stale')
    return manifest


class InheritedJournal:
    """Immutable baseline + hash-chained delta, preserving per-URL and total budgets."""
    def __init__(self, base_path, base_scope, base_sha, delta_path, scope):
        need(sha(base_path) == base_sha, 'Inherited journal SHA mismatch')
        self.base = Journal(base_path, base_scope)
        self.delta = Journal(delta_path, scope)
        self.entries = {k: dict(v) for k, v in self.base.entries.items()}
        for key, state in self.delta.entries.items():
            self._check(key, state)
            self.entries[key] = dict(state)
        self.transients = self.base.transients + self.delta.transients

    def _check(self, key, state):
        prior = self.entries.get(key)
        if prior is not None:
            need(state['attempts'] >= prior['attempts'], 'Inherited attempt budget reset')
            if prior['status'] in ('SUCCEEDED', 'FALLBACK', 'FATAL', 'EXHAUSTED'):
                need(state['status'] == prior['status'], 'Inherited terminal state changed')

    def record(self, key, kind, state):
        self._check(key, state)
        self.delta.record(key, kind, state)
        self.entries[key] = dict(state)
        self.transients = self.base.transients + self.delta.transients


def ncar_journal():
    return InheritedJournal(V4_JOURNAL, V4_MANIFEST_SHA, V4_JOURNAL_SHA,
                            OUT_V5 / 'scheduler_delta.jsonl', sha(MANIFEST_V5))
