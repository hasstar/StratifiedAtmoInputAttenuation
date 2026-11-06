"""Bounded, explicit local gates and one registered CDS diagnostic transition."""
import argparse
import os
import json
import msvcrt
import psutil
import uuid
from repair_common_v005 import *
from acquire_v004 import partial_audit
from audit_transport_v003 import audit_all


def host_gate():
    scripts = {'ncar_transport_v002.py', 'ncar_transport_v003.py', 'acquire_v004.py', 'acquire_v005.py',
        'continue_delivery.py', 'continue_delivery_v003.py', 'continue_delivery_v004.py', 'continue_delivery_v005.py'}
    active = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        if p.pid == os.getpid() or 'python' not in (p.info['name'] or '').lower():
            continue
        command = p.info['cmdline'] or []
        if any(Path(x).name in scripts for x in command if x.endswith('.py')):
            active.append(dict(pid=p.pid, create_time=p.info['create_time'], command=command))
    need(not active, 'An acquisition or relay instance is still active')
    return dict(utc=now(), active_pipeline_processes=active,
                disk_free_gib=psutil.disk_usage(str(ROOT)).free / 1024**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic-cds', action='store_true')
    args = parser.parse_args()
    check_guards()
    host = host_gate()
    with (OUT / 'continuation.lock').open('r+b') as continuation:
        continuation.seek(0)
        msvcrt.locking(continuation.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            with stage_lock('preflight_v005'):
                check_guards()
                cached = audit_all(require_complete=False)
                partial = partial_audit(read(ACQ / 'ncar_plan.json'), cached['nodes'])
                cds = cds_cache_audit()
                need(cached['nodes'] == 517 and partial['parent'] == 'df76bb74bb8e7b464dd4212a'
                    and partial['verified_hours'] == list(range(20)), 'Inspected NCAR checkpoint changed')
                need(cds['downloaded'] >= 5 and cds['identities'] >= 6, 'Inspected CDS checkpoint lost')
                journal = ncar_journal()
                need(journal.base.transients == 5 and journal.base.sequence == 442, 'Inherited NCAR budget mismatch')
                record = dict(utc=now(), status='PASS_AVAILABLE_INPUTS_ONLY', host=host, transport=cached,
                    partial=partial, cds=cds, ncar_inherited_transients=journal.base.transients,
                    source_manifest_sha256=sha(MANIFEST_V5), full_input_qa=False, both_os_locks_held=True)
                emit(OUT_V5 / 'preflights' / (uuid.uuid4().hex + '.json'), record)
                print(json.dumps(dict(status=record['status'], ncar=cached['nodes'], partial_hours=len(partial['verified_hours']),
                    cds=cds['downloaded'], inherited_transients=journal.base.transients, full_input_qa=False)), flush=True)
                if args.diagnostic_cds:
                    from cds_scheduler_v005 import ticker
                    tick = ticker()
                    observed = tick()  # at most one known-job operation, same persistent budget
                    diag = dict(utc=now(), source_manifest_sha256=sha(MANIFEST_V5), state=observed,
                        only_locally_registered_new_job=True, full_input_qa=False)
                    emit(OUT_V5 / 'diagnostics' / (uuid.uuid4().hex + '.json'), diag)
                    print(json.dumps(diag), flush=True)
        finally:
            continuation.seek(0)
            msvcrt.locking(continuation.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
