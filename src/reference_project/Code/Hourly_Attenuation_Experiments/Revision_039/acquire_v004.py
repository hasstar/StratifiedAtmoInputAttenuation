"""Single-worker continuation with persistent bounded retries and serial CDS ticks."""
import argparse
import json
import os
import re
import time
import psutil
import numpy as np
from scheduler_common import *
from retry_scheduler import Journal, RetryScheduler, CDSTicker
from ncar_transport_v003 import FetcherV3, TransientReadFailure
from audit_transport_v003 import audit_all, checked_payload, verified_array
from common import child_node, hourly_path
from cds_three import step


class FetcherV4(FetcherV3):
    def __init__(self, scheduler, report):
        super().__init__(report=report)
        self.scheduler = scheduler

    def fetch_raw(self, url, path, validate, read_timeout, attempts):
        # No nested retries: the sealed physical transport always gets one attempt.
        physical = lambda: super(FetcherV4, self).fetch_raw(url, path, validate, read_timeout, 1)
        return self.scheduler.call(url, physical, cache_exists=path.exists(), daily=(read_timeout == 60 and attempts == 1))


def partial_audit(plan, count):
    if count == len(plan['nodes']):
        return dict(parent=None, verified_hours=[])
    node = plan['nodes'][count]
    permitted = {n['id'] for n in plan['nodes'][:count + 1]}
    need({p.name for p in (ACQ / 'ncar_hourly_v003').iterdir() if p.is_dir()} <= permitted, 'Unexpected future hourly cache')
    folder = ACQ / 'ncar_hourly_v003' / node['id']
    positions = []
    if folder.exists():
        for receipt in folder.glob('*.receipt.json'):
            need(receipt.with_name(receipt.name.removesuffix('.receipt.json')).exists(), 'Orphan partial-hour receipt')
        metadata = ACQ / 'ncar_metadata' / (original.fp(node['url_path'])[:24] + '.das')
        das, _ = checked_payload(metadata, HOST + 'dodsC/' + node['url_path'] + '.das')
        expected_paths = {hourly_path(node, pos) for pos in range(len(node['epochs']))}
        need(set(folder.glob('*.ascii')) <= expected_paths, 'Unknown partial-hour payload')
        for pos in range(len(node['epochs'])):
            path = hourly_path(node, pos)
            if path.exists():
                child = child_node(node, pos)
                raw, _ = checked_payload(path, child['url'])
                np.testing.assert_array_equal(verified_array(raw, child).ravel(), original.decode(raw, das, child))
                positions.append(pos)
        need(positions == list(range(len(positions))), 'Noncontiguous partial-hour checkpoint')
    return dict(parent=node['id'], verified_hours=positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    need(re.fullmatch('[a-f0-9]{32}', args.run_id) is not None, 'Invalid run ID')
    run = args.run_id
    process = psutil.Process()
    identity = dict(pid=os.getpid(), create_time=process.create_time(), run_id=run)
    report_state = {}
    count, initial = 0, 0
    tic = time.monotonic()

    def report(**kw):
        report_state.update(kw)
        obj = dict(utc=now(), **identity, stage='ACQUIRING', transport='v004', validated_ncar_nodes=count,
                   total_ncar_nodes=2012, new_parent_nodes_this_run=count - initial,
                   elapsed_seconds=time.monotonic() - tic, **report_state)
        current_status(OUT / 'acquisition_progress.json', obj)

    try:
        with stage_lock('ncar_acquire_v004'):
            check_guards()
            need(not (ACQ / 'ncar_download_complete.json').exists(), 'Acquisition already complete')
            emit(OUT_V4 / 'runs' / (run + '.json'), dict(utc=now(), **identity, command=sys.argv,
                 source_manifest_sha256=sha(MANIFEST_V4), upstream_manifest_sha256=UPSTREAM_SHA))
            plan = read(ACQ / 'ncar_plan.json')
            cached = audit_all(require_complete=False)
            count = initial = cached['nodes']
            partial = partial_audit(plan, count)
            cds = cds_cache_audit()
            emit(OUT_V4 / 'prechecks' / (run + '.json'), dict(utc=now(), status='PASS_AVAILABLE_INPUTS_ONLY',
                 transport=cached, partial=partial, cds=cds, full_input_qa=False))
            report(current_stage='VERIFIED_CACHE_REUSED', cds_downloaded=cds['downloaded'])
            ticker = CDSTicker(step, report)
            journal = Journal(OUT_V4 / 'scheduler_journal.jsonl', sha(MANIFEST_V4))
            scheduler = RetryScheduler(journal, TransientReadFailure, tick=ticker, report=report)
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
                emit(OUT_V4 / 'parent_acceptance' / (node['id'] + '.json'), dict(utc=now(), **identity,
                     source_manifest_sha256=sha(MANIFEST_V4), independent_audit=result))
                report(current_stage='PARENT_VERIFIED', current_parent_id=node['id'], scheduler_state=None)
                print('PARENT_VERIFIED', count, 'OF 2012', node['id'], result['kind'], flush=True)
                ticker()
            check_guards()
            qa = audit_all()
            qa['scheduler_manifest_sha256'] = sha(MANIFEST_V4)
            emit(OUT_V4 / 'transport_provenance_qa.json', qa)
            emit(ACQ / 'ncar_download_complete.json', dict(utc=now(), **identity, nodes=2012,
                 plan_sha256=PLAN_SHA, status='DOWNLOADED_NOT_YET_INDEPENDENT_INPUT_QA', transport='v004',
                 scheduler_manifest_sha256=sha(MANIFEST_V4), transport_qa_sha256=sha(OUT_V4 / 'transport_provenance_qa.json')))
            report(current_stage='NCAR_COMPLETE_INPUT_QA_PENDING', scheduler_state=None)
        emit(OUT_V4 / 'worker_results' / (run + '.json'), dict(utc=now(), **identity, status='SUCCESS',
             nodes=2012, completion_manifest_sha256=sha(ACQ / 'ncar_download_complete.json'),
             source_manifest_sha256=sha(MANIFEST_V4)))
    except Exception as exc:
        # CDS implementation already suppresses secrets; retain only its safe message.
        record = dict(utc=now(), **identity, status='FAILED_REQUIRES_INSPECTION',
                      error_type=type(exc).__name__, message=str(exc), validated_ncar_nodes=count)
        emit(OUT_V4 / 'worker_results' / (run + '.json'), record)
        report(current_stage='FAILED_REQUIRES_INSPECTION', scheduler_state='STOPPED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
