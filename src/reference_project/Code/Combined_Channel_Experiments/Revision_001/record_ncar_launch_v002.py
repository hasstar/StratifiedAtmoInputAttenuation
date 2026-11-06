"""Read live NCAR v009 process/state and persist immutable handoff evidence."""
import json
import time
from datetime import datetime
import psutil
import weather_download_top10_ncar_v009 as fix
d=fix.app

def main():
    plan=d.load_plan();workers=[]
    for proc in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            args=proc.info['cmdline'] or []
            if (proc.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any('weather_download' in a and a.endswith('.py') for a in args):workers.append(proc)
        except (psutil.AccessDenied,psutil.NoSuchProcess):continue
    d.require(len(workers)==1,'Need exactly one live downloader')
    proc=workers[0];args=proc.cmdline();d.require(args[-1]=='run' and any(a.endswith('weather_download_top10_ncar_v009.py') for a in args),'Wrong live entry')
    a=proc.cpu_times();time.sleep(2);b=proc.cpu_times()
    active=d.R(d.DEST/'active_worker.json');d.require(active['pid']==proc.pid and active['status']=='RUNNING' and d.io.lock_held(),'Worker lock/status mismatch')
    result={'utc':d.io.now(),'pid':proc.pid,'created_local':datetime.fromtimestamp(proc.create_time()).astimezone().isoformat(),'cmdline':args,
            'cpu_seconds_1':a.user+a.system,'cpu_seconds_2':b.user+b.system,'os_lock_held':True,'active_worker':active,
            'progress':d.progress(plan,True),'plan_sha256':d.SHA(d.PLAN),'source_sha256':plan['source_sha256']}
    d.io.immutable(d.DEST/'launch_verification.json',result)
    d.io.immutable(d.DEST/'handoff.json',{'utc':d.io.now(),'status':'NCAR_V009_ROUTE_LIVE','plan_fingerprint':plan['fingerprint'],
        'old_cds_pid':25488,'old_cds_stop_verified_before_local':'2026-10-15T17:26:08.9608804+08:00','old_cds_remote_cancelled':False,
        'old_v008_pid':29188,'old_v008_failure':'Windows concurrent JSON sharing violation; secondary error mapping mismatch; full stderr remains in v008',
        'old_v008_exit_and_free_lock_verified_local':'2026-10-15T17:43:09+08:00','reused_ncar_subsets':len(plan['ncar_reusable']),
        'reused_ncar_grid_field_hours':sum(x['values'] for x in plan['ncar_reusable']),
        'new_pid_snapshot':proc.pid,'runbook':'CONTINUATION_RUNBOOK_010.md','new_fit_calls':0,'predictions_executed':0,'T012_complete':False})
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
