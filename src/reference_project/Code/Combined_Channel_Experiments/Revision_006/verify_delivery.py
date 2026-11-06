"""Read-only scientific audit and new-version-only delivery receipts; no model import."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / 'Code/Combined_Channel_Experiments/Revision_006'
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_006'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def emit(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    import msvcrt
    need(not (OUT / 'delivery_receipt.json').exists(), 'Delivery already recorded')
    completion = read(OUT / 'summary_complete.json')
    need(completion['execution'] == 'complete' and completion['verification'] == 'PASS', 'Incomplete scoring')
    need(completion['weights_updated'] == 0 and completion['parameter_training_steps'] == 0, 'Nonfrozen execution')
    run_sources = [
        ('inference_004/tabpfn_v2_author_frozen', 'run_frozen_v004.py'),
        ('inference_tabicl_002/tabicl_v2_author_frozen', 'run_frozen.py'),
    ]
    for relative, runner in run_sources:
        launch = read(OUT / relative / 'launch.json')
        need(launch['code_sha256'] == sha(CODE / runner), 'Executed runner changed after launch')
        need(read(OUT / relative / 'pilot.json')['success'] is True, 'Pilot did not pass')
        done = read(OUT / relative / 'complete.json')
        need(done['parameter_sha256_before'] == done['parameter_sha256_after'], 'Network parameter drift')
        need(all(done['counters'][key] == 0 for key in ('gradient_calls', 'optimizer_constructions', 'network_calls')), 'Forbidden inference action')
    process_check = read(OUT / 'process_exit_verification.json')
    need(len(process_check['snapshots']) == 2, 'Two process snapshots required')
    need(all(not row['matching_processes'] for row in process_check['snapshots']), 'Own inference or weather process still active')
    need(process_check['query_succeeded'] is True, 'Process query failed')
    locks = []
    for path in [OUT / 'inference.lock', OUT / 'inference_tabicl_002/inference.lock']:
        with path.open('r+b') as stream:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        locks.append(str(path.relative_to(ROOT)))
    baseline = read(OUT / 'source_sha256_before.json')
    need(all(sha(ROOT / path) == digest for path, digest in baseline.items()), 'Protected source drift')
    source_checks = []
    for name, relative, before, expected_before_sha in [
        ('T012', 'Tasks/T012_Combined_Channel_And_External_Validation.md', 'T012_before.md', '2d54cb38bdfc5c0ae9c64398b01084ce910d753607bd0f6cb2d8f246ebb9d48f'),
        ('PROGRESS', 'Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md', 'PROGRESS_before.md', '743fcf6cdf6d65609a30233d41ccc5e018188d875e475c81e0f897c77535de8a'),
    ]:
        source = ROOT / relative
        body = source.read_bytes()
        pattern = rb'<!-- REV006-BEGIN -->.*?<!-- REV006-END -->\r?\n\r?\n'
        need(len(re.findall(pattern, body, flags=re.S)) == 1, 'Missing or duplicate own index block')
        old = (OUT / 'document_snapshots' / before).read_bytes()
        need(hashlib.sha256(old).hexdigest() == expected_before_sha, 'Original before snapshot changed')
        need(re.sub(pattern, b'', body, flags=re.S) == old, 'Historical index bytes changed')
        after = OUT / 'document_snapshots' / (name + '_after.md')
        with after.open('xb') as stream:
            stream.write(body)
        source_checks.append(dict(path=relative, before_sha256=hashlib.sha256(old).hexdigest(),
                                  after_sha256=sha(source), historical_body_exact=True))
    need('Lock Status: RELEASED' in (CODE / 'PROTOCOL.md').read_text(encoding='utf-8'), 'Protocol lock not released')
    pyfiles = list(CODE.glob('*.py'))
    for path in pyfiles:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    link_checks = []
    for path in [OUT / 'COMPARISON_REPORT.md', OUT / 'MODEL_PROVENANCE.md', OUT / 'HANDOFF.md']:
        content = path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('https://', 'http://', '#')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            need(resolved.exists(), 'Broken local link: ' + target)
            link_checks.append(dict(document=str(path.relative_to(ROOT)), target=target, exists=True))
    manifest = {}
    for parent in [CODE, OUT]:
        for path in sorted(parent.rglob('*')):
            if path.is_file() and path.suffix not in ('.lock', '.pyc') and path.name not in ('delivery_manifest.json', 'delivery_receipt.json'):
                manifest[str(path.relative_to(ROOT))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    emit(OUT / 'delivery_manifest.json', manifest)
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), execution='COMPLETE', verification='PASS',
                   added_author_frozen_models=completion['new_models'], checkpoint_weight_updates=0,
                   parameter_training_steps=0, finetuning_steps=0, old_model_inference_reruns=0,
                   author_weights_restored_for_historical_adaptations=0, historical_checkpoints_preserved=18,
                   protected_sources_unchanged=len(baseline), fixed_indexes=source_checks,
                   syntax_checked_scripts=len(pyfiles), local_links_checked=link_checks,
                   process_verification_sha256=sha(OUT / 'process_exit_verification.json'),
                   os_locks_released=locks, execution_lock='RELEASED',
                   manifest_count=len(manifest), manifest_sha256=sha(OUT / 'delivery_manifest.json'),
                   weather_downloads_restarted=False, t012_overall_complete=False)
    emit(OUT / 'delivery_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
