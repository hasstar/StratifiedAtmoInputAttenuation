"""Append-only offline publication check after independent QA and report review."""
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_028'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def emit(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def main():
    (OUT/'seal.claim').open('x',encoding='utf-8').close()
    assert read(OUT/'independent_qa.json')['status']=='PASS'
    for phase in ('calculate','qa'):
        assert read(OUT/(phase+'.exit.json'))['exit_code']==0
    source=read(OUT/'sources_before.json')
    assert all(sha(p)==h for p,h in source.items())
    code=read(OUT/'code_before.json')
    assert all(sha(ROOT/p)==h for p,h in code.items())
    for p in CODE.glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    required=['REPORT.md','QA.md','HANDOFF.md','MODEL_INPUT_MATRIX.md','COMMANDS.md','CHANGE_RECORD.md','PROTOCOL.md']
    for name in required: assert (OUT/name).is_file(),name
    assert read(OUT/'model_status.json')['nwp_mts']=='NOT_COMPUTED_USER_DECISION'
    assert read(OUT/'model_status.json')['all_seven_complete'] is False
    for p in OUT.glob('*.json'): json.loads(p.read_text(encoding='utf-8'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    links=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if not target.startswith(('http:','https:','#')):
                assert (p.parent/target.split('#')[0]).resolve().exists(),(str(p),target)
                links+=1
    emit('publication_checks.json',dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        protected_sources=len(source),changed_sources=0,frozen_code_files=len(code),required_documents=required,
        local_links_checked=links,independent_qa='PASS',all_seven_complete=False))
    files={str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
        for directory in (CODE,OUT) for p in sorted(directory.rglob('*')) if p.is_file()}
    emit('delivery_manifest.json',dict(files=files,utc=datetime.now(timezone.utc).isoformat()))
    for rel,meta in files.items(): assert sha(ROOT/rel)==meta['sha256']
    emit('final_verification.json',dict(status='PASS',phase='LOCAL_ESST_APPROVED_ENGINEERING_ADAPTATION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(OUT/'delivery_manifest.json'),
        sealed_files=len(files),execution_lock='RELEASED',all_seven_complete=False,
        nwp_mts='NOT_COMPUTED_USER_DECISION',local_esst='COMPUTED_PARTIAL_COVERAGE_ENGINEERING_ADAPTATION',
        overwrite_history=False))
    print(json.dumps(read(OUT/'final_verification.json')))


if __name__=='__main__': main()
