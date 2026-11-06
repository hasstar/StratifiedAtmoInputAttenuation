"""Seal the new input-QA delivery, retaining failures and all old artifacts."""
import hashlib
import json
import msvcrt
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_013'
RUN=OUT/'attempt_002'

def need(value,message):
    if not value: raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    need(not (OUT/'final_verification.json').exists(),'Already sealed')
    ps="""$ErrorActionPreference='Stop'
$snapshots=foreach ($n in 1..2) {
  $rows=@(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine,CreationDate,KernelModeTime,UserModeTime)
  [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');query_success=$true;processes=$rows}
  if ($n -eq 1) { Start-Sleep -Seconds 3 }
}
$snapshots | ConvertTo-Json -Depth 6 -Compress
"""
    p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',ps],capture_output=True,text=True,timeout=30)
    need(p.returncode==0,'CIM failed; not evidence of absence: '+p.stderr)
    snapshots=json.loads(p.stdout)
    save('final_preflight.json',{'utc':datetime.now(timezone.utc).isoformat(),'current_pid':os.getpid(),'snapshots':snapshots})
    for s in snapshots:
        need(s['query_success'],'No process evidence')
        for proc in s['processes']:
            need(proc['CommandLine'] and proc['ExecutablePath'],'Process identity unavailable')
            need(proc['ProcessId']==os.getpid(),'Another Python process exists; review before seal')
    guards={**read(RUN/'protected_sources_before.json'),**read(RUN/'raw_sources_before.json')}
    for path,expected in guards.items():need(sha(path)==expected,'Protected source changed: '+path)
    v=read(RUN/'verification.json');pv=read(RUN/'physics_verification.json');a=read(RUN/'audit_complete.json')
    need(v['status']==pv['status']=='PASS','Independent QA not passed')
    need(v['files']==144 and v['variable_file_checks']==264 and v['hourly_ledger_rows']==7800,'Coverage QA')
    need(a['test_ledger_counts']['esst_input_coavailability_only']==574,'Coavailability')
    need(a['test_ledger_counts']['h0_all_corners_finite']==579,'Zero-degree input availability')
    for name,digest in v['checked_outputs_sha256'].items():need(sha(RUN/name)==digest,'Checked output changed: '+name)
    need(sha(RUN/'geometry_diagnostics.json')==pv['geometry_diagnostics_sha256'],'Physics output changed')
    need(sha(CODE/'verify.py')==v['verification_script_sha256'],'Verification script changed')
    need(sha(CODE/'verify_physics.py')==pv['script_sha256'],'Physics verifier changed')
    need(read(OUT/'failure.json')['message']=='Missing variable: deg0l','Failure must remain')
    report=(OUT/'REPORT.md').read_text(encoding='utf-8')
    for phrase in ['144','22','264','7,800','579','574','763','不是可评分','天气下载保持停止','−0.000965595','6,922']:
        need(phrase in report,'Report misses verified finding: '+phrase)
    for name in ['REPORT.md','HANDOFF.md','FORMULA_READINESS.md','CHANGE_RECORD.md']:
        need((OUT/name).stat().st_size>500,'Missing deliverable: '+name)
    with (RUN/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            files=[p for folder in [CODE,OUT] for p in folder.rglob('*') if p.is_file()]
            manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
            save('delivery_manifest.json',manifest)
            for name,digest in manifest.items():need(sha(ROOT/name)==digest,'Delivery changed during seal')
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    result={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
            'stage':'LOCAL existing raw physical-input audit; no new model results',
            'independent_input_qa':'PASS','independent_physics_diagnostics_qa':'PASS',
            'protected_sources':len(guards),'source_sha_unchanged':True,
            'delivery_files':len(manifest),'delivery_manifest_sha256':sha(OUT/'delivery_manifest.json'),
            'raw_files':144,'variables':22,'campaign_hours':7800,'test_hours':687,
            'candidate_test_input_coavailability':574,'candidate_count_is_not_model_eligibility':True,
            'seven_model_comparison_complete':False,'new_training':0,'new_model_inference':0,
            'weather_requests':0,'new_scores':0,'automation_modified':False,
            'process_evidence':'two successful elevated CIM samples; only this seal PID',
            'os_lock':'RELEASED','first_failure_preserved':True}
    save('final_verification.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
