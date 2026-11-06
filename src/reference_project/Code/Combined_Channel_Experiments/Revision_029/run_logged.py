"""One-shot foreground runner, adapted from Revision028 without history writes."""
import json
import msvcrt
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_029'
stage=sys.argv[1]
assert stage in ('calculate','qa')
if stage=='calculate': OUT.mkdir(exist_ok=False)
else: assert OUT.is_dir()
with (OUT/(stage+'.claim')).open('x',encoding='utf-8') as f:
    json.dump({'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()},f)
with (OUT/(stage+'.lock')).open('x+b') as lock:
    lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    filename='calculate.py' if stage=='calculate' else 'verify_independent.py'
    command=[sys.executable,'-B',str(CODE/filename)]
    with (OUT/(stage+'.command.json')).open('x',encoding='utf-8') as f:
        json.dump({'command':command,'cwd':str(ROOT),'utc':datetime.now(timezone.utc).isoformat()},f,indent=2)
    try:
        with (OUT/(stage+'.stdout.log')).open('xb') as out,(OUT/(stage+'.stderr.log')).open('xb') as err:
            result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
with (OUT/(stage+'.exit.json')).open('x',encoding='utf-8') as f:
    json.dump({'exit_code':result.returncode,'utc':datetime.now(timezone.utc).isoformat(),'os_lock':'RELEASED'},f,indent=2)
print(json.dumps({'stage':stage,'exit_code':result.returncode}))
if result.returncode:
    print((OUT/(stage+'.stderr.log')).read_text(encoding='utf-8',errors='replace'))
sys.exit(result.returncode)
