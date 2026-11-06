"""One-shot, offline seal of the source audit. No capture/model modules imported."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_026'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def put(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def main():
    assert not (OUT / 'final_verification.json').exists()
    assert not (OUT / 'delivery_manifest.json').exists()
    guard = json.loads((OUT / 'sources_before.json').read_text(encoding='utf-8'))
    changes = [name for name, value in guard.items() if sha(Path(name)) != value]
    assert not changes, changes
    for name, expected in json.loads((OUT / 'code_before.json').read_text(encoding='utf-8')).items():
        assert sha(ROOT / name) == expected, name
    qa = json.loads((OUT / 'independent_qa.json').read_text(encoding='utf-8'))
    assert qa['status'] == 'PASS_SOURCE_EVIDENCE_ONLY'
    assert qa['protected_sources'] == len(guard) and not qa['changed_sources']
    scripts = sorted(CODE.glob('*.py'))
    for script in scripts:
        ast.parse(script.read_text(encoding='utf-8'))
    json_count = 0
    for path in OUT.rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
        json_count += 1
    links_count = 0
    for directory in (CODE, OUT):
        for path in directory.rglob('*.md'):
            for raw in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                target = raw.strip('<>').split('#')[0]
                if not target or re.match(r'^https?://', target):
                    continue
                assert (path.parent / target).exists(), (str(path), target)
                links_count += 1
    state = json.loads((OUT / 'model_status.json').read_text(encoding='utf-8'))
    assert not state['current_required_work_complete']
    assert not state['all_seven_original_models_complete']
    assert not state['automation_modified'] and not state['automation_paused']
    checks = {'utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
              'status': 'PASS', 'python_ast_files': len(scripts), 'json_files_checked': json_count,
              'local_markdown_links_checked': links_count, 'protected_sources': len(guard),
              'source_changes': changes, 'code_before_unchanged': True,
              'no_new_model_results': True, 'scope': 'DOCUMENTARY_SOURCE_AUDIT'}
    put('seal_checks.json', checks)
    files = {}
    for directory in (CODE, OUT):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    put('delivery_manifest.json', {'utc': datetime.now(timezone.utc).isoformat(), 'files': files,
                                 'self_and_final_verification_excluded': True})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
              'phase': 'PUBLIC_IMPLEMENTATION_ROUTE_EVIDENCE_VERIFIED', 'sealed_files': len(files),
              'manifest_sha256': sha(OUT / 'delivery_manifest.json'), 'protected_sources': len(guard),
              'source_changes': 0, 'lock': 'RELEASED', 'new_predictions': 0, 'new_scores': 0,
              'weather_downloads': 0, 'training_runs': 0, 'public_html_captures': 3,
              'public_release_entry_dns_failures': 1, 'all_seven_complete': False,
              'pending_local_adaptation_decision': True, 'mts_status': 'NOT_COMPUTED_USER_DECISION',
              'automation_modified': False, 'automation_paused': False}
    put('final_verification.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
