"""Passive exact-instance relay. Reuses CDS retry ledger; never restarts a worker."""
import argparse
import os
import re
import sys
import time
import subprocess
import msvcrt
import psutil
from repair_common_v005 import *


def exact_command(command, run):
    need(any(Path(x).resolve() == (CODE_V5 / 'acquire_v005.py').resolve()
             for x in command if x.endswith('.py')), 'Exact v005 worker command mismatch')
    need('--run-id' in command and command[command.index('--run-id') + 1] == run, 'Worker run identity mismatch')


def validate_success(result, done, qa, pid, created, run, manifest_sha, done_sha, qa_sha):
    need(result['status'] == 'SUCCESS' and result['pid'] == pid and
         abs(result['create_time'] - created) < .001 and result['run_id'] == run, 'Exact worker did not succeed')
    need(result['completion_manifest_sha256'] == done_sha and result['source_manifest_sha256'] == manifest_sha, 'Worker result SHA mismatch')
    need(done['nodes'] == 2012 and done['plan_sha256'] == PLAN_SHA and done['transport'] == 'v005', 'Exact complete plan required')
    need(done['run_id'] == run and done['pid'] == pid and abs(done['create_time'] - created) < .001
         and done['scheduler_manifest_sha256'] == manifest_sha, 'Completion lineage mismatch')
    need(qa['status'] == 'PASS' and qa['nodes'] == 2012 and qa['scheduler_manifest_sha256'] == manifest_sha
         and qa['source_manifest_sha256'] == v4.UPSTREAM_SHA, 'Independent transport QA missing/stale')
    need(done['transport_qa_sha256'] == qa_sha, 'Transport QA SHA mismatch')


def snapshot(status, **kw):
    current_status(OUT / 'continuation_status.json', dict(utc=now(), pid=os.getpid(),
        create_time=psutil.Process().create_time(), status=status, transport='v005', **kw))
    print(status, kw, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--create-time', type=float, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = args.run_id
    need(re.fullmatch('[a-f0-9]{32}', run) is not None, 'Invalid run ID')
    with (OUT / 'continuation.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            check_guards()
            emit(OUT / 'runs' / ('continuation_v005_' + run + '.json'), dict(utc=now(), pid=os.getpid(),
                create_time=psutil.Process().create_time(), wait_pid=args.wait_pid, wait_create_time=args.create_time,
                run_id=run, source_manifest_sha256=sha(MANIFEST_V5), passive_wait_only_until_handoff=True))
            snapshot('PASSIVE_WAIT_FOR_EXISTING_ACQUISITION', wait_pid=args.wait_pid, run_id=run)
            while True:
                stopped()
                try:
                    process = psutil.Process(args.wait_pid)
                    if abs(process.create_time() - args.create_time) > .001:
                        break
                    exact_command(process.cmdline(), run)
                except psutil.NoSuchProcess:
                    break
                time.sleep(30)
            check_guards()
            result = read(OUT_V5 / 'worker_results' / (run + '.json'))
            done_path, qa_path = ACQ / 'ncar_download_complete.json', OUT_V5 / 'transport_provenance_qa.json'
            validate_success(result, read(done_path), read(qa_path), args.wait_pid, args.create_time, run,
                             sha(MANIFEST_V5), sha(done_path), sha(qa_path))
            from audit_transport_v003 import audit_all
            with stage_lock('transport_handoff_recheck_v005'):
                result = audit_all()
                result['scheduler_manifest_sha256'] = sha(MANIFEST_V5)
                emit(OUT_V5 / 'handoff_transport_qa.json', result)
            from cds_scheduler_v005 import ticker, DONE, BLOCKED
            cds_tick = ticker(lambda **kw: snapshot('COMPLETING_NEW_CDS_JOBS', **kw))
            snapshot('COMPLETING_NEW_CDS_JOBS')
            while True:
                check_guards()
                with stage_lock('cds_final_drain_v005'):
                    state = cds_tick()
                need(state['status'] != BLOCKED, 'CDS bounded retry ceiling reached; full input QA blocked')
                if state['status'] == DONE:
                    break
                time.sleep(5)
            cds_cache_audit()
            for script in ['assemble_inputs.py', 'audit_inputs.py', 'infer_frozen.py', 'audit_metrics.py', 'plot_six.py', 'write_results.py']:
                check_guards()
                snapshot('RUNNING_SERIAL_STAGE', stage=script)
                result = subprocess.run([sys.executable, '-B', str(ORIGINAL_CODE / script)], cwd=str(ROOT))
                need(result.returncode == 0, 'Stage failed; requires inspection, not retried: ' + script)
            snapshot('RESULTS_READY_PENDING_VISUAL_REVIEW', states=100, figures=6)
        except Exception as exc:
            emit(OUT / 'failures' / ('continuation_v005_' + run + '.json'),
                 dict(utc=now(), error_type=type(exc).__name__, message=str(exc)))
            snapshot('FAILED_REQUIRES_INSPECTION', error=str(exc))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
