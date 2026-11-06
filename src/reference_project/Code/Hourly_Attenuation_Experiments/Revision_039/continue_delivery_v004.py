"""Passive exact-instance handoff; no collection while the worker is active."""
import argparse
import os
import time
import subprocess
import msvcrt
import psutil
from scheduler_common import *


def snapshot(status, **kw):
    record = dict(utc=now(), pid=os.getpid(), create_time=psutil.Process().create_time(), status=status,
                  transport='v004', **kw)
    current_status(OUT / 'continuation_status.json', record)
    print(status, kw, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--create-time', type=float, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = args.run_id
    with (OUT / 'continuation.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            check_guards()
            emit(OUT / 'runs' / f'continuation_v004_{run}.json', dict(utc=now(), pid=os.getpid(),
                 create_time=psutil.Process().create_time(), wait_pid=args.wait_pid, wait_create_time=args.create_time,
                 run_id=run, source_manifest_sha256=sha(MANIFEST_V4), passive_wait_only_until_handoff=True))
            snapshot('PASSIVE_WAIT_FOR_EXISTING_ACQUISITION', wait_pid=args.wait_pid, run_id=run)
            while True:
                stopped()
                try:
                    process = psutil.Process(args.wait_pid)
                    if abs(process.create_time() - args.create_time) > .001:
                        break
                    command = process.cmdline()
                    need(any(Path(x).resolve() == (CODE_V4 / 'acquire_v004.py').resolve() for x in command if x.endswith('.py')), 'Exact v004 worker command mismatch')
                    need('--run-id' in command and command[command.index('--run-id') + 1] == run, 'Worker run identity mismatch')
                except psutil.NoSuchProcess:
                    break
                time.sleep(30)
            check_guards()
            result = read(OUT_V4 / 'worker_results' / (run + '.json'))
            need(result['status'] == 'SUCCESS' and result['pid'] == args.wait_pid and
                 abs(result['create_time'] - args.create_time) < .001 and result['run_id'] == run,
                 'Exact worker did not succeed; inspection required, not restarted')
            done_path = ACQ / 'ncar_download_complete.json'
            need(result['completion_manifest_sha256'] == sha(done_path) and result['source_manifest_sha256'] == sha(MANIFEST_V4), 'Worker result SHA mismatch')
            done = read(done_path)
            qa_path = OUT_V4 / 'transport_provenance_qa.json'
            qa = read(qa_path)
            need(done['nodes'] == 2012 and done['plan_sha256'] == PLAN_SHA and done['transport'] == 'v004', 'Exact complete plan required')
            need(done['run_id'] == run and done['scheduler_manifest_sha256'] == sha(MANIFEST_V4), 'Completion lineage mismatch')
            need(qa['status'] == 'PASS' and qa['nodes'] == 2012 and qa['scheduler_manifest_sha256'] == sha(MANIFEST_V4)
                 and qa['source_manifest_sha256'] == UPSTREAM_SHA, 'Independent transport QA missing/stale')
            need(done['transport_qa_sha256'] == sha(qa_path), 'Transport QA SHA mismatch')
            from audit_transport_v003 import audit_all
            with stage_lock('transport_handoff_recheck_v004'):
                result = audit_all()
                result['scheduler_manifest_sha256'] = sha(MANIFEST_V4)
                emit(OUT_V4 / 'handoff_transport_qa.json', result)
            from cds_three import step
            snapshot('COMPLETING_NEW_CDS_JOBS')
            while True:
                check_guards()
                with stage_lock('cds_final_drain_v004'):
                    state = step()
                if state['status'] == 'ALL_DOWNLOADED_NOT_YET_INPUT_QA':
                    break
                snapshot('COMPLETING_NEW_CDS_JOBS', cds=state)
                time.sleep(60)
            for script in ['assemble_inputs.py', 'audit_inputs.py', 'infer_frozen.py', 'audit_metrics.py', 'plot_six.py', 'write_results.py']:
                check_guards()
                snapshot('RUNNING_SERIAL_STAGE', stage=script)
                result = subprocess.run([sys.executable, '-B', str(ORIGINAL_CODE / script)], cwd=str(ROOT))
                need(result.returncode == 0, 'Stage failed; requires inspection, not retried: ' + script)
            snapshot('RESULTS_READY_PENDING_VISUAL_REVIEW', states=100, figures=6)
        except Exception as exc:
            emit(OUT / 'failures' / f'continuation_v004_{run}.json', dict(utc=now(), error_type=type(exc).__name__, message=str(exc)))
            snapshot('FAILED_REQUIRES_INSPECTION', error=str(exc))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
