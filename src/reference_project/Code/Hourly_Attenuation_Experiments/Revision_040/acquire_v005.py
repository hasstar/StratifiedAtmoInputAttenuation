"""One serial collector: inherited NCAR budgets plus fault-isolated CDS ticks."""
import argparse
import os
import re
import sys
import time
import psutil
from repair_common_v005 import *
from acquire_v004 import FetcherV4, partial_audit
from retry_scheduler import RetryScheduler
from ncar_transport_v003 import TransientReadFailure
from audit_transport_v003 import audit_all
from cds_scheduler_v005 import ticker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    need(re.fullmatch('[a-f0-9]{32}', args.run_id) is not None, 'Invalid run ID')
    run = args.run_id
    identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), run_id=run)
    report_state = {}
    count = initial = 0
    tic = time.monotonic()

    def report(**kw):
        report_state.update(kw)
        current_status(OUT / 'acquisition_progress.json', dict(utc=now(), **identity, stage='ACQUIRING',
            transport='v005', validated_ncar_nodes=count, total_ncar_nodes=2012,
            new_parent_nodes_this_run=count - initial, elapsed_seconds=time.monotonic() - tic, **report_state))

    try:
        with stage_lock('ncar_acquire_v005'):
            check_guards()
            need(not (ACQ / 'ncar_download_complete.json').exists(), 'NCAR acquisition already complete')
            emit(OUT_V5 / 'runs' / (run + '.json'), dict(utc=now(), **identity, command=sys.argv,
                 source_manifest_sha256=sha(MANIFEST_V5), upstream_manifest_sha256=V4_MANIFEST_SHA))
            plan = read(ACQ / 'ncar_plan.json')
            cached = audit_all(require_complete=False)
            count = initial = cached['nodes']
            partial = partial_audit(plan, count)
            cds = cds_cache_audit()
            journal = ncar_journal()
            cds_tick = ticker(report)
            emit(OUT_V5 / 'prechecks' / (run + '.json'), dict(utc=now(), status='PASS_AVAILABLE_INPUTS_ONLY',
                transport=cached, partial=partial, cds=cds, inherited_ncar_transients=journal.base.transients,
                total_ncar_transients=journal.transients, cds_retry_state=cds_tick.state, full_input_qa=False))
            report(current_stage='VERIFIED_CACHE_REUSED', cds_downloaded=cds['downloaded'], cds=dict(cds_tick.state))
            scheduler = RetryScheduler(journal, TransientReadFailure, tick=cds_tick, report=report)
            fetcher = FetcherV4(scheduler, report)
            for node in plan['nodes'][count:]:
                report(current_stage='DOWNLOADING_NCAR_PARENT', current_parent_id=node['id'],
                    current_file=node['url_path'], scheduler_state=None, current_epoch=None,
                    verified_hours_in_parent=None, parent_hours=len(node['epochs']))
                fileid = original.fp(node['url_path'])[:24]
                das, _, _ = fetcher.fetch_raw(HOST + 'dodsC/' + node['url_path'] + '.das',
                    ACQ / 'ncar_metadata' / (fileid + '.das'),
                    lambda s: original.contract(s, node['spec'], node['pressure']), 180, 2)
                result = fetcher.parent(node, das)
                count += 1
                emit(OUT_V5 / 'parent_acceptance' / (node['id'] + '.json'), dict(utc=now(), **identity,
                    source_manifest_sha256=sha(MANIFEST_V5), independent_audit=result))
                report(current_stage='PARENT_VERIFIED', current_parent_id=node['id'], scheduler_state=None)
                print('PARENT_VERIFIED', count, 'OF 2012', node['id'], result['kind'], flush=True)
                cds_tick()
            check_guards()
            qa = audit_all()
            qa['scheduler_manifest_sha256'] = sha(MANIFEST_V5)
            emit(OUT_V5 / 'transport_provenance_qa.json', qa)
            emit(ACQ / 'ncar_download_complete.json', dict(utc=now(), **identity, nodes=2012,
                plan_sha256=PLAN_SHA, status='DOWNLOADED_NOT_YET_INDEPENDENT_INPUT_QA', transport='v005',
                scheduler_manifest_sha256=sha(MANIFEST_V5), transport_qa_sha256=sha(OUT_V5 / 'transport_provenance_qa.json')))
            report(current_stage='NCAR_COMPLETE_INPUT_QA_PENDING', scheduler_state=None)
        emit(OUT_V5 / 'worker_results' / (run + '.json'), dict(utc=now(), **identity, status='SUCCESS', nodes=2012,
            completion_manifest_sha256=sha(ACQ / 'ncar_download_complete.json'), source_manifest_sha256=sha(MANIFEST_V5)))
    except Exception as exc:
        emit(OUT_V5 / 'worker_results' / (run + '.json'), dict(utc=now(), **identity,
            status='FAILED_REQUIRES_INSPECTION', error_type=type(exc).__name__, message=str(exc), validated_ncar_nodes=count))
        report(current_stage='FAILED_REQUIRES_INSPECTION', scheduler_state='STOPPED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
