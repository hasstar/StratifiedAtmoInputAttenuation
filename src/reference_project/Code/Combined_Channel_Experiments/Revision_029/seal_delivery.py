"""One-shot post-QA publication checks and immutable delivery manifest."""
import ast
import hashlib
import json
import msvcrt
import re
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];CODE=Path(__file__).parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_029'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    with (OUT/'seal.claim').open('x',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat())
    emit('seal_command.json',dict(utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),source_sha256=sha(__file__)))
    qa=read(OUT/'independent_qa.json');assert qa['status']=='PASS'
    for stage in ('calculate','qa'):
        assert read(OUT/(stage+'.exit.json'))['exit_code']==0
        assert (OUT/(stage+'.stderr.log')).stat().st_size==0
        with (OUT/(stage+'.lock')).open('r+b') as lock:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    sources=read(OUT/'sources_before.json');code=read(OUT/'code_before.json')
    assert all(sha(p)==h for p,h in sources.items())
    assert all(sha(ROOT/p)==h for p,h in code.items())
    assert {str(p.relative_to(ROOT)).replace('\\','/') for p in CODE.iterdir() if p.is_file()}==set(code)
    for p in CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    for p,h in read(OUT/'calculation_complete.json')['output_sha256'].items():assert sha(OUT/p)==h
    assert qa['calculation_complete_sha256']==sha(OUT/'calculation_complete.json')
    required=['REPORT.md','QA.md','HANDOFF.md','MODEL_INPUT_MATRIX.md','COMMANDS.md','CHANGE_RECORD.md','PROTOCOL.md']
    for name in required:assert (OUT/name).is_file(),name
    forward={(OUT/n).resolve() for n in ('publication_checks.json','delivery_manifest.json','final_verification.json')}
    links=[]
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if not target.startswith(('http:','https:','#')):
                resolved=(p.parent/target.split('#')[0]).resolve()
                assert resolved.exists() or resolved in forward,(p,target)
                links.append(resolved)
    text=(OUT/'REPORT.md').read_text(encoding='utf-8')
    tab=[[c.strip() for c in line.split('|')[1:-1]] for line in text.splitlines() if line.startswith('| ')]
    labels={'ours':'本文冻结Top10','bai_lstm':'Bai LSTM LOCAL适配','tabm':'TabM LOCAL适配',
            'itu':'组合ITU近似','esst':'E-SST小时工程适配组合'}
    weather={'rain':'雨','dry_overcast':'干燥阴'}
    summaries=read(OUT/'family_metrics.json');checked=0
    for r in summaries:
        want=7 if r['weighting']=='date_equal' else 5
        candidates=[x for x in tab if len(x)==want and x[0]==weather[r['event']] and x[1]==labels[r['family']]]
        assert len(candidates)==1,(r['event'],r['weighting'],r['family'])
        cells=candidates[0]
        expected=format(r['crps201_db'],'.6f')
        if want==7:
            if r['crps201_db_sd'] is not None:expected+=' ± '+format(r['crps201_db_sd'],'.6f')
            assert cells[2]==expected
            assert cells[3]==format(r['slow_only_crps_db'],'.6f')
            assert cells[4]==format(100*r['central90_coverage'],'.3f')+'%'
            assert cells[5]==format(100*r['tail99_exceedance'],'.3f')+'%'
            assert cells[6]==str(r['rank_crps201'])
        else:
            assert cells[2]==expected and cells[3]==format(r['crps401_db'],'.6f')
            assert cells[4]==str(r['rank_crps201'])
        checked+=1
    assert read(OUT/'model_status.json')['nwp_mts']=='NOT_COMPUTED_USER_DECISION'
    assert read(OUT/'model_status.json')['all_seven_complete'] is False
    emit('publication_checks.json',dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        protected_sources=len(sources),unchanged_code_files=len(code),human_table_rows_checked=checked,
        local_links_checked=len(links),stages_exited_zero=True,os_locks='RELEASED',required_documents=required,
        all_seven_complete=False))
    files={str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
        for directory in (CODE,OUT) for p in sorted(directory.rglob('*')) if p.is_file()}
    emit('delivery_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=files))
    assert all(sha(ROOT/p)==m['sha256'] for p,m in files.items())
    emit('final_verification.json',dict(status='PASS',phase='LOCAL_ESST_COMMON_FROZEN_FAST_MARGINAL_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(OUT/'delivery_manifest.json'),
        sealed_files=len(files),execution_lock='RELEASED',computed_families=5,score_hours=200,
        new_esst_marginal_rows=200,reused_old_score_rows=1600,new_hourly_predictions=0,
        new_realizations=0,nwp_mts='NOT_COMPUTED_USER_DECISION',all_seven_complete=False))
    assert all(p.exists() for p in links)
    print(json.dumps(read(OUT/'final_verification.json')))


if __name__=='__main__':main()
