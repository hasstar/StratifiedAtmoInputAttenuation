"""Pre-execution seal correction: verify three forward links after generation.

The original, unexecuted seal is retained. Numerical code/output is unchanged.
"""
import ast
import hashlib
import json
import re
import sys
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
    (OUT/'seal_v002.claim').open('x',encoding='utf-8').close()
    emit('seal_command.json',dict(utc=datetime.now(timezone.utc).isoformat(),executable=sys.executable,
        argv=sys.argv,cwd=str(Path.cwd()),source_sha256=sha(__file__),
        initial_seal_executed=False,reason='pre-execution correction of forward-link verification order'))
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
    forward={(OUT/n).resolve() for n in ('final_verification.json','delivery_manifest.json','publication_checks.json')}
    links=[]
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if not target.startswith(('http:','https:','#')):
                resolved=(p.parent/target.split('#')[0]).resolve()
                assert resolved.exists() or resolved in forward,(str(p),target)
                links.append(resolved)
    # Independently check the human-facing tables against the verified JSON.
    table_rows=[line.split('|')[1:-1] for line in (OUT/'REPORT.md').read_text(encoding='utf-8').splitlines()
        if line.startswith('| ') and len(line.split('|')[1:-1])==5]
    computed=[r for r in read(OUT/'seven_family_rankings.json') if r['status']=='COMPUTED']
    display={'ours_frozen_top10':'本文冻结Top10','bai2022_lstm_t0_adapted':'Bai LSTM LOCAL适配',
        'tabm2025_official_budgeted':'TabM LOCAL适配','itu_combined_geometric_approx':'组合ITU近似',
        'esst_hourly_adaptation':'E-SST小时工程适配组合'}
    for r in computed:
        cells=[c.strip() for c in next(x for x in table_rows if x[0].strip()==display[r['family']]
            and x[1].strip().startswith(format(r['metrics_mean']['rmse_db'],'.6f')))]
        for j,key in [(2,'mae_db'),(3,'bias_db')]: assert cells[j]==format(r['metrics_mean'][key],'.6f')
        assert cells[4]==str(r['rank'])
        if r['metrics_sd']['rmse_db'] is not None:
            assert cells[1].split('±')[1].strip()==format(r['metrics_sd']['rmse_db'],'.6f')
    emit('publication_checks.json',dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        protected_sources=len(source),changed_sources=0,frozen_code_files=len(code),required_documents=required,
        local_links_checked=len(links),human_report_numeric_rows_checked=len(computed),
        independent_qa='PASS',all_seven_complete=False,seal_version='v002'))
    files={str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
        for directory in (CODE,OUT) for p in sorted(directory.rglob('*')) if p.is_file()}
    emit('delivery_manifest.json',dict(files=files,utc=datetime.now(timezone.utc).isoformat()))
    for rel,meta in files.items(): assert sha(ROOT/rel)==meta['sha256']
    emit('final_verification.json',dict(status='PASS',phase='LOCAL_ESST_APPROVED_ENGINEERING_ADAPTATION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(OUT/'delivery_manifest.json'),
        sealed_files=len(files),execution_lock='RELEASED',all_seven_complete=False,
        nwp_mts='NOT_COMPUTED_USER_DECISION',local_esst='COMPUTED_PARTIAL_COVERAGE_ENGINEERING_ADAPTATION',
        overwrite_history=False))
    assert all(p.exists() for p in links),'post-generation links'
    print(json.dumps(read(OUT/'final_verification.json')))


if __name__=='__main__': main()
