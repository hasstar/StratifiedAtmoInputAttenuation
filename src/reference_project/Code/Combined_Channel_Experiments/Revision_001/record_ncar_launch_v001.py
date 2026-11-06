"""Read live process/state and save a new immutable launch audit only."""
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import psutil
import weather_download_top10_ncar_v008 as d

def main():
    plan=d.load_plan()
    workers=[]
    for proc in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            args=proc.info['cmdline'] or []
            if (proc.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any('weather_download' in a and a.endswith('.py') for a in args):
                workers.append(proc)
        except (psutil.AccessDenied,psutil.NoSuchProcess):continue
    d.require(len(workers)==1,'Need exactly one live weather downloader')
    proc=workers[0];args=proc.cmdline()
    d.require(args[-1]=='run' and any(a.endswith('weather_download_top10_ncar_v008.py') for a in args),'Wrong live entry')
    before=proc.cpu_times();time.sleep(2);after=proc.cpu_times()
    active=d.R(d.DEST/'active_worker.json');d.require(active['pid']==proc.pid and active['status']=='RUNNING' and d.io.lock_held(),'Worker/OS lock mismatch')
    actual=d.progress(plan,True)
    copies=[]
    base=d.DEST/'handoff_state_before'
    for path in sorted(base.rglob('*')):
        if path.is_file():
            source=d.PREVIOUS/path.relative_to(base)
            d.require(d.SHA(path)==d.SHA(source),'Old handoff copy changed')
            copies.append({'source':str(source.relative_to(d.ROOT)),'copy':str(path.relative_to(d.ROOT)),'sha256':d.SHA(path)})
    result={'utc':d.io.now(),'pid':proc.pid,'created_local':datetime.fromtimestamp(proc.create_time()).astimezone().isoformat(),
            'cmdline':args,'cpu_seconds_1':before.user+before.system,'cpu_seconds_2':after.user+after.system,
            'os_lock_held':True,'active_worker':active,'progress':actual,'old_state_copies':copies,
            'plan_sha256':d.SHA(d.PLAN),'source_sha256':plan['source_sha256'],'new_fit_calls':0,'predictions_executed':0}
    d.io.immutable(d.DEST/'launch_verification.json',result)
    d.io.immutable(d.DEST/'handoff.json',{'utc':d.io.now(),'status':'NCAR_ROUTE_LIVE','plan_fingerprint':plan['fingerprint'],
        'old_local_worker_stopped':{'pid':25488,'created_local':'2026-10-15T16:45:45.321752+08:00','stop_verified_before_local':'2026-10-15T17:26:08.9608804+08:00'},
        'old_remote_jobs':plan['retired_cds_jobs'],'old_remote_cancelled':False,'old_auxiliary_downloaded':False,
        'new_worker_pid_snapshot':proc.pid,'new_runbook':'CONTINUATION_RUNBOOK_009.md','launch_verification':'launch_verification.json',
        'new_fit_calls':0,'predictions_executed':0,'full_T012_complete':False})
    print(json.dumps({k:v for k,v in result.items() if k not in ('old_state_copies','source_sha256')},indent=2))

if __name__=='__main__':main()
