"""One-shot evidence seal. No numerical model or scoring implementation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import msvcrt
import os
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_016'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_015'
PRIOR_HASH = 'a569d73404fd537dd01317fdcdc5573f37140f0422f3ce04923570baf61eb092'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check(guard):
    bad = [p for p, expected in guard.items()
           if not Path(p).is_file() or sha(p) != expected]
    need(not bad, 'Protected file mismatch: ' + repr(bad))


def main():
    need(not (OUT / 'launch.json').exists(), 'Already started; do not rerun')
    need(not (OUT / 'delivery_manifest.json').exists(), 'Already sealed')
    need(sha(PREV / 'delivery_manifest.json') == PRIOR_HASH, 'Prior manifest changed')
    need(read(PREV / 'final_verification.json')['status'] == 'PASS', 'Prior QA missing')
    guard = read(PREV / 'sources_before.json')
    need(len(guard) == 816, 'Unexpected prior protection scope')
    check(guard)
    delivery = read(PREV / 'delivery_manifest.json')
    need(len(delivery) == 22, 'Unexpected prior delivery scope')
    for relative, expected in delivery.items():
        path = ROOT / relative
        need(sha(path) == expected, 'Prior delivery changed: ' + relative)
        guard[str(path)] = expected
    for path in [PREV / 'delivery_manifest.json', PREV / 'final_verification.json']:
        guard[str(path)] = sha(path)

    lock = (OUT / 'execution.lock').open('x+b')
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        emit('launch.json', {'utc': utc(), 'pid': os.getpid(),
             'command': sys.orig_argv, 'cwd': str(Path.cwd()),
             'executable': sys.executable, 'python': sys.version,
             'platform': platform.platform(), 'scope': 'evidence seal only'})
        emit('sources_before.json', guard)
        state = read(OUT / 'readiness.json')
        lookup = read(OUT / 'search_log.json')
        need(state['status'] == 'BLOCKED_NOT_COMPUTED', 'Readiness mislabelled')
        need(state['real_dynamic_outputs'] is None and state['real_dynamic_metrics'] is None,
             'Unexpected dynamic result')
        need(state['new_training'] == state['new_inference'] == state['simulation_runs_executed'] == 0,
             'Unexpected execution')
        need(not state['seven_family_complete'] and not state['weather_acquisition_authorized'],
             'Authority/completion mismatch')
        need(len(state['gates']) == 6 and len({g['id'] for g in state['gates']}) == 6,
             'Missing or duplicated dependency gates')
        need(set(state['remaining_uncomputed_families']) == {'e_sst','rmop','nwp_prm_mts_chain'},
             'Unexpected family scope')
        need(len(state['scored_families_reused']) == 4, 'Scored family scope')
        source_ids = {s['id'] for s in lookup['sources']}
        need(source_ids == {'S1','S2','S3'}, 'Missing source records')
        for gate in state['gates']:
            need(set(gate['evidence']) <= source_ids | {'Revision014','project_contract'},
                 'Unresolved evidence reference')
        need(not lookup['asset_obtained'] and lookup['external_messages_sent'] == 0,
             'Unexpected acquisition/contact')
        previous_scores = read(ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012/verification.json')
        need(previous_scores['status'] == 'PASS' and not previous_scores['all_seven_complete'],
             'Previous score QA changed')
        need(previous_scores['common_uk_hours'] == 5538 and previous_scores['local_test_hours'] == 687,
             'Frozen score scope changed')
        check(guard)
        emit('provenance_qa.json', {'status': 'PASS', 'utc': utc(),
             'scope': 'source registry, dependency-state and immutable-file integrity only',
             'source_records': 3, 'dependency_gates': 6, 'protected_sources': len(guard),
             'source_text_review': 'by root agent; not an independent external review',
             'prior_metric_qa_reused': 'Revision012/verification.json',
             'new_metrics_computed': False, 'mts_implementation_validated': False,
             'seven_family_complete': False})
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
    emit('lock_release.json', {'utc': utc(), 'pid': os.getpid(), 'os_lock': 'RELEASED'})
    check(guard)
    paths = sorted(p for base in [CODE, OUT] for p in base.rglob('*')
                   if p.is_file() and p.name != 'execution.lock')
    need(not any(p.name == 'failure.json' for p in paths), 'Unreviewed failure')
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    emit('delivery_manifest.json', manifest)
    print(json.dumps({'status': 'SEALED_AWAITING_INDEPENDENT_INTEGRITY_CHECK',
                      'protected': len(guard), 'delivery_files': len(manifest),
                      'manifest_sha256': sha(OUT / 'delivery_manifest.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.is_dir() and not (OUT / 'failure.json').exists():
            emit('failure.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise
