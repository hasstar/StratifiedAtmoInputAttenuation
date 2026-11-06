"""Foreground one-shot log wrapper for production or independent QA only."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_027'


def main():
    stage = sys.argv[1]
    script = {'score':'supplement.py','qa':'verify_independent.py'}[stage]
    OUT.mkdir(exist_ok=True)
    command = [sys.executable,'-B','-X','utf8',str(CODE/script)]
    record = {'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,'cwd':str(ROOT)}
    with (OUT/(stage+'.started.json')).open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2)
    with (OUT/(stage+'.stdout.log')).open('xb') as stdout, (OUT/(stage+'.stderr.log')).open('xb') as stderr:
        completed = subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,check=False)
    record.update(finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=completed.returncode)
    with (OUT/(stage+'.exit.json')).open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2)
    print(json.dumps(record))
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()
