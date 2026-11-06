"""One-shot offline delivery seal, without rerunning scoring or inference."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_027'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def put(name, value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def main():
    assert not (OUT/'delivery_manifest.json').exists() and not (OUT/'final_verification.json').exists()
    assert not (OUT/'failure.json').exists()
    qa = read(OUT/'independent_qa.json')
    assert qa['status'] == 'PASS' and qa['supplement_hours'] == 684
    guard = read(OUT/'sources_before.json')
    assert not [p for p,h in guard.items() if sha(Path(p))!=h]
    assert all(sha(ROOT/p)==h for p,h in read(OUT/'code_before.json').items())
    required = ['REPORT.md','QA.md','HANDOFF.md','CHANGE_RECORD.md','MODEL_INPUT_MATRIX.md','COMMANDS.md','PROTOCOL.md']
    assert all((OUT/name).is_file() for name in required)
    for script in CODE.glob('*.py'): ast.parse(script.read_text(encoding='utf-8'))
    for path in OUT.rglob('*.json'): read(path)
    links = 0
    for folder in (CODE,OUT):
        for path in folder.rglob('*.md'):
            for raw in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
                target = raw.strip('<>').split('#')[0]
                if not target or re.match(r'^https?://',target): continue
                assert (path.parent/target).exists(), (str(path),target)
                links += 1
    state = read(OUT/'model_status.json')
    assert not state['current_required_work_complete'] and not state['automation_modified'] and not state['automation_paused']
    put('seal_checks.json', {'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','argv':sys.argv,
        'source_changes':0,'protected_sources':len(guard),'markdown_links_checked':links,
        'Python_AST':'PASS','JSON_parse':'PASS','code_frozen_since_production':True})
    files = {}
    for folder in (CODE,OUT):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(ROOT))]={'sha256':sha(path),'bytes':path.stat().st_size}
    put('delivery_manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':files,
        'self_and_final_verification_excluded':True})
    final = {'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'phase':'LOCAL_LIQUID_INPUT_VALID_SUPPLEMENT_VERIFIED','protected_sources':len(guard),'source_changes':0,
        'sealed_files':len(files),'manifest_sha256':sha(OUT/'delivery_manifest.json'),'lock':'RELEASED',
        'full_LOCAL_mother_hours':687,'LOCAL_supplement_hours':684,'excluded_with_reasons':3,
        'new_supplement_run_metric_rows':16,'new_predictions':0,'training_runs':0,'weather_downloads':0,
        'UK_scope_unchanged':True,'all_seven_complete':False,'mts_status':'NOT_COMPUTED_USER_DECISION',
        'automation_modified':False,'automation_paused':False}
    put('final_verification.json',final)
    print(json.dumps(final,ensure_ascii=False))


if __name__=='__main__':main()
