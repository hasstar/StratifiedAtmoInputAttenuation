"""Passive same-pipeline handoff, then strictly serial completion stages.

Does not restart a failed downloader. It performs no acquisition or inference
until the exact original process has exited with a complete download manifest.
"""
import argparse, os, time, subprocess, sys, uuid
from pathlib import Path
import psutil, msvcrt
from freeze import ROOT, CODE, OUT, read, sha, now, emit, need

def snapshot(status,**kwargs):
    obj=dict(utc=now(),pid=os.getpid(),status=status,**kwargs)
    tmp=OUT/'continuation_status.tmp';tmp.write_text(__import__('json').dumps(obj,indent=2),encoding='utf-8')
    os.replace(tmp,OUT/'continuation_status.json');print(status,kwargs,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);ap.add_argument('--create-time',type=float,required=True);args=ap.parse_args()
    with (OUT/'continuation.lock').open('a+b') as f:
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        run=uuid.uuid4().hex;guard={str(p):sha(p) for p in CODE.iterdir() if p.is_file()}
        emit(OUT/'runs'/f'continuation_{run}.json',dict(utc=now(),pid=os.getpid(),create_time=psutil.Process().create_time(),
            wait_pid=args.wait_pid,wait_create_time=args.create_time,source_sha256=guard,passive_wait_only_until_handoff=True))
        try:
            snapshot('PASSIVE_WAIT_FOR_EXISTING_ACQUISITION',wait_pid=args.wait_pid)
            while psutil.pid_exists(args.wait_pid):
                p=psutil.Process(args.wait_pid)
                if abs(p.create_time()-args.create_time)>.01:break
                need(any('ncar_transport_v002.py' in s for s in p.cmdline()),'Original worker identity')
                need(not (OUT/'STOP_REQUESTED.json').exists(),'New task stopped by user')
                time.sleep(30)
            need((OUT/'acquisition/ncar_download_complete.json').exists(),'Acquisition exited without completion; inspection required, not restarted')
            done=read(OUT/'acquisition/ncar_download_complete.json')
            need(done['plan_sha256']==sha(OUT/'acquisition/ncar_plan.json') and done['nodes']==2012,'Complete original plan')
            for p,digest in guard.items():need(sha(Path(p))==digest,'Continuation source changed')
            from transport import stage_lock
            from cds_three import step
            snapshot('COMPLETING_NEW_CDS_JOBS')
            while True:
                with stage_lock('cds_final_drain'):state=step()
                if state['status']=='ALL_DOWNLOADED_NOT_YET_INPUT_QA':break
                snapshot('COMPLETING_NEW_CDS_JOBS',cds=state);time.sleep(30)
            for script in ['assemble_inputs.py','audit_inputs.py','infer_frozen.py','audit_metrics.py','plot_six.py','write_results.py']:
                need(not (OUT/'STOP_REQUESTED.json').exists(),'New task stopped by user')
                snapshot('RUNNING_SERIAL_STAGE',stage=script)
                result=subprocess.run([sys.executable,'-B',str(CODE/script)],cwd=str(ROOT))
                need(result.returncode==0,'Stage failed; requires inspection, not retried: '+script)
            snapshot('RESULTS_READY_PENDING_VISUAL_REVIEW',states=100,figures=6)
        except Exception as exc:
            emit(OUT/'failures'/f'continuation_{run}.json',dict(utc=now(),error_type=type(exc).__name__,message=str(exc)))
            snapshot('FAILED_REQUIRES_INSPECTION',error=str(exc));raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':main()
