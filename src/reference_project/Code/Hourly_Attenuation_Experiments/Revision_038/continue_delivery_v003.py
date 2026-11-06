"""Exact-instance passive relay for transport v003; frozen delivery stays serial."""
import argparse
import json
import os
import sys
import time
import uuid
import subprocess
import psutil
import msvcrt
from common import *


def snapshot(status, **kw):
    record = dict(utc=now(), pid=os.getpid(), create_time=psutil.Process().create_time(), status=status, transport='v003', **kw)
    temp = OUT / 'continuation_status.tmp'
    temp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    os.replace(temp, OUT / 'continuation_status.json')
    print(status, kw, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--create-time', type=float, required=True)
    args = parser.parse_args()
    with (OUT / 'continuation.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        run = uuid.uuid4().hex
        try:
            check_guards()
            emit(OUT / 'runs' / f'continuation_v003_{run}.json', dict(utc=now(), pid=os.getpid(),
                 create_time=psutil.Process().create_time(), wait_pid=args.wait_pid, wait_create_time=args.create_time,
                 source_manifest_sha256=sha(MANIFEST), passive_wait_only_until_handoff=True))
            snapshot('PASSIVE_WAIT_FOR_EXISTING_ACQUISITION', wait_pid=args.wait_pid)
            while True:
                need(not (OUT / 'STOP_REQUESTED.json').exists(), 'T015 stopped')
                try:
                    process = psutil.Process(args.wait_pid)
                    if abs(process.create_time() - args.create_time) > .001:
                        break
                    command = process.cmdline()
                    need(any(Path(x).resolve() == (REPAIR_CODE / 'ncar_transport_v003.py').resolve() for x in command if x.endswith('.py')), 'Exact v003 worker command mismatch')
                except psutil.NoSuchProcess:
                    break
                time.sleep(30)
            check_guards()
            need((ACQ / 'ncar_download_complete.json').exists(), 'Acquisition exited without completion; inspection required, not restarted')
            done = read(ACQ / 'ncar_download_complete.json')
            qa_path = REPAIR_OUT / 'transport_provenance_qa.json'
            need(done['nodes'] == 2012 and done['plan_sha256'] == PLAN_SHA and done['transport'] == 'v003', 'Exact plan completion required')
            qa = read(qa_path)
            need(qa['status'] == 'PASS' and qa['nodes'] == 2012 and qa['source_manifest_sha256'] == sha(MANIFEST), 'Independent transport QA missing/stale')
            need(done['transport_qa_sha256'] == sha(qa_path), 'Transport QA SHA mismatch')
            from audit_transport_v003 import audit_all
            with stage_lock('transport_handoff_recheck_v003'):
                result = audit_all()
                emit(REPAIR_OUT / 'handoff_transport_qa.json', result)
            from cds_three import step
            snapshot('COMPLETING_NEW_CDS_JOBS')
            while True:
                with stage_lock('cds_final_drain_v003'):
                    state = step()
                if state['status'] == 'ALL_DOWNLOADED_NOT_YET_INPUT_QA':
                    break
                snapshot('COMPLETING_NEW_CDS_JOBS', cds=state)
                time.sleep(30)
            for script in ['assemble_inputs.py', 'audit_inputs.py', 'infer_frozen.py', 'audit_metrics.py', 'plot_six.py', 'write_results.py']:
                check_guards()
                snapshot('RUNNING_SERIAL_STAGE', stage=script)
                result = subprocess.run([sys.executable, '-B', str(ORIGINAL_CODE / script)], cwd=str(ROOT))
                need(result.returncode == 0, 'Stage failed; requires inspection, not retried: ' + script)
            snapshot('RESULTS_READY_PENDING_VISUAL_REVIEW', states=100, figures=6)
        except Exception as exc:
            emit(OUT / 'failures' / f'continuation_v003_{run}.json', dict(utc=now(), error_type=type(exc).__name__, message=str(exc)))
            snapshot('FAILED_REQUIRES_INSPECTION', error=str(exc))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
