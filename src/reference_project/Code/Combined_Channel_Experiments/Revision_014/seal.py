"""Seal Revision014, not a model-comparison completion declaration."""
import ast
import hashlib
import json
import msvcrt
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_014'

def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    need(not (OUT/'final_verification.json').exists(),'Already sealed')
    command='''$ErrorActionPreference='Stop'
$samples=foreach($i in 1..2){
 $rows=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction Stop | Select-Object ProcessId,ExecutablePath,CommandLine,CreationDate,KernelModeTime,UserModeTime)
 [pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');query_success=$true;processes=$rows}
 if($i -eq 1){Start-Sleep -Seconds 3}
}
$samples | ConvertTo-Json -Depth 6 -Compress
'''
    p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,timeout=30)
    need(p.returncode==0,'CIM failed; do not infer absence')
    samples=json.loads(p.stdout)
    save('final_preflight.json',{'current_pid':os.getpid(),'samples':samples})
    for sample in samples:
        need(sample['query_success'],'CIM query unsuccessful')
        for proc in sample['processes']:
            need(proc['ProcessId']==os.getpid(),'Another Python process; review')
            need(proc['CommandLine'] and proc['ExecutablePath'],'Identity unavailable')
    guards=read(OUT/'sources_before.json')
    for path,digest in guards.items():need(sha(path)==digest,'Protected source changed')
    for name,digest in read(OUT/'code_before.json').items():need(sha(CODE/name)==digest,'Tested code changed')
    qa=read(OUT/'independent_qa.json');tests=read(OUT/'test_results.json')
    need(qa['status']==tests['status']=='PASS','QA not passed')
    need(tests['check_count']==50 and qa['independent_scalar_checks']==61,'QA coverage mismatch')
    need(sha(CODE/'prm_kernel.py')==qa['kernel_sha256'],'Kernel changed')
    need(sha(CODE/'verify_independent.py')==qa['verifier_sha256'],'Verifier changed')
    need(sha(OUT/'test_results.json')==qa['test_results_sha256'],'Tests changed')
    need(sha(OUT/'synthetic_cases.json')==qa['synthetic_cases_sha256'],'Cases changed')
    need(not read(OUT/'readiness.json')['coefficient_numeric_bundle'],'No coefficient source obtained')
    for name in ('REPORT.md','HANDOFF.md','CHANGE_RECORD.md','SOURCE_AUDIT.md','FORMULA_AUDIT.md'):
        need((OUT/name).stat().st_size>400,'Missing delivery: '+name)
    for path in CODE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            files=sorted(p for folder in [CODE,OUT] for p in folder.rglob('*') if p.is_file() and p.name!='execution.lock')
            manifest={str(path.relative_to(ROOT)):sha(path) for path in files}
            save('delivery_manifest.json',manifest)
            for path,digest in manifest.items():need(sha(ROOT/path)==digest,'Delivery changed')
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    final={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
           'stage':'PRM equation kernel / bounded coefficient-source audit',
           'synthetic_tests':50,'independent_scalar_checks':61,
           'max_absolute_difference':qa['max_absolute_difference'],
           'source_sha_unchanged':True,'protected_sources':len(guards),
           'delivery_files':len(manifest),'delivery_manifest_sha256':sha(OUT/'delivery_manifest.json'),
           'os_lock':'RELEASED','logical_write_scope':'RELEASED',
           'excluded_from_hash_manifest':['execution.lock (OS mutual exclusion separately checked)'],
           'new_training':0,'new_real_predictions':0,'new_scores':0,'weather_requests':0,
           'automation_modified':False,'seven_family_comparison_complete':False,
           'scores_reference':'Outputs/Experiments/Combined_Channel_Revision_012/REPORT.md'}
    save('final_verification.json',final)
    print(json.dumps(final,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
