"""Single-worker surface50→1 RFE with the original T014 fitting functions."""
from __future__ import annotations

import argparse
import os
import time
import traceback
import runtime as r


def reuse_start(name, k, fold, names, train, hold):
    r.require(name == 'tabm', 'Only TabM is authorized in T017')
    if k != 50:
        return None
    state = r.read(r.P31 / 'layer/states/0000.json')
    qa = r.read(r.P31 / 'layer/qa/0000.json')
    r.require(qa['status'] == 'PASS', 'Historical mask0000 QA')
    for path, digest in qa['verified_files'].items():
        r.require(r.sha(r.P31 / path) == digest, 'Mask0000 audited file changed: ' + path)
    r.require(state['feature_names'] == names == r.surface_names(), 'Exact initial surface50 pool')
    role = state['full_fit'] if fold == 0 else state['folds'][fold - 1]
    origin = r.P31 / role['result_file']
    old = r.read(origin)
    r.require(r.sha(origin) == role['result_sha256'], 'Initial role receipt SHA')
    r.require(old['config'] == r.config('tabm')['config'] and old['seed'] == 17, 'Initial config/seed')
    r.require(old['selected_epoch'] == old['trained_epochs'] == 54 and old['search'] is False, 'Fixed epochs54 only')
    r.require(old['feature_names'] == names, 'Initial ordered feature names')
    r.require(old['test_used_for_fit_or_selection'] is False, 'Initial test exclusion')
    r.require(old['training']['hour_bjt'] == train.hour_bjt.astype(str).tolist(), 'Initial exact fit hours')
    if hold is not None:
        r.require(old['holdout']['hour_bjt'] == hold.hour_bjt.astype(str).tolist(), 'Initial exact holdout hours')
    else:
        r.require(old['holdout'] is None, 'Initial full fit must not use an official holdout')
    path = r.P31 / old['model_file']
    r.require(r.sha(path) == old['model_sha256'] == role['model_sha256'], 'Initial model digest')
    ref = dict(model_file=r.rel(path), model_sha256=old['model_sha256'], format='tabm31',
               reused=True, new_fit=False, source_file=r.rel(origin), source_sha256=r.sha(origin),
               round_history=old['round_history'],
               fit_metadata=dict(selected_epoch=54, trained_epochs=54, seed=17, weight_policy='uniform',
                                 search=False, historical_fit_seconds=old['fit_seconds'],
                                 reuse_reason='Exact frozen local surface50 starting role; not a new fit'))
    bundle = r.load_fit(ref)
    r.check_prep(bundle, train, names, name)
    for df, rec in [(train, old['training'])] + ([] if hold is None else [(hold, old['holdout'])]):
        r.require(r.np.allclose(r.predict(bundle, df), rec['prediction_db'], rtol=0, atol=2e-6), 'Initial saved prediction replay')
    return ref


def run(resume_authorization=None):
    import psutil
    with r.writer_lock():
        resources = r.resource_check()
        manifest = r.manifest()
        r.require(r.read(r.OUT / 'preflight_qa.json')['status'] == 'PASS', 'Preflight gate')
        prior_status = r.read(r.OUT / 'status.json')['status']
        if prior_status != 'PREPARED_NOT_STARTED':
            r.require(resume_authorization is not None, 'Existing run requires inspected resume authorization')
            auth = r.read(resume_authorization)
            r.require(auth.get('allow_resume') is True and auth.get('manifest') == manifest['fingerprint'], 'Resume authorization identity')
            r.require(bool(auth.get('diagnosis')) and auth.get('prior_process_exit_verified') is True
                      and auth.get('no_effective_user_stop') is True, 'Resume requires an explicit inspected diagnosis')
        identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                        command=psutil.Process().cmdline(), utc=r.now())
        run_id = str(time.time_ns())
        r.emit(r.OUT / 'runs' / (run_id + '.json'), dict(identity, manifest=manifest['fingerprint'], resources=resources,
               resume_authorization=None if resume_authorization is None else str(resume_authorization)))
        r.update(r.OUT / 'active_worker.json', dict(identity, status='RUNNING'))
        r.update(r.OUT / 'status.json', dict(status='RUNNING', manifest=manifest['fingerprint'], run_id=run_id, utc=r.now()))
        failed = False
        try:
            runner, auditor = r.load_reference_functions()
            runner.reuse_ref = reuse_start
            data = r.frames()
            names = list(manifest['feature_names'])
            for k in range(50, 0, -1):
                r.stop_check()
                state = runner.process_state('tabm', k, names, data, manifest)
                r.update(r.OUT / 'state_count.json', dict(qa_pass_states=51-k, total_states=50, latest_k=k,
                         removed_next=state['removed_feature_next'], utc=r.now()))
                print(f'{r.now()} STATE_QA_PASS k={k} completed={51-k}/50', flush=True)
                if k > 1:
                    names = [feature for feature in names if feature != state['removed_feature_next']]
            # Rehash sources before composing the comparison; no historical mutation is accepted.
            r.manifest()
            import report_surface
            report_surface.build(data, manifest)
            r.update(r.OUT / 'status.json', dict(status='NUMERICS_COMPLETE_VISUAL_REVIEW_PENDING',
                     qa_pass_states=50, total_states=50, utc=r.now(),
                     note='Numerical CSV QA passed; actual PNG review and final acceptance remain.'))
            print('NUMERICS_COMPLETE_VISUAL_REVIEW_PENDING', flush=True)
        except BaseException as error:
            failed = True
            status = 'STOPPED_USER_REQUEST' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED_REQUIRES_INSPECTION'
            r.emit(r.OUT / 'failures' / (str(time.time_ns()) + '.json'),
                   dict(identity, status=status, error=repr(error), traceback=traceback.format_exc(), utc=r.now()))
            r.update(r.OUT / 'status.json', dict(status=status, error=repr(error), utc=r.now()))
            raise
        finally:
            r.update(r.OUT / 'active_worker.json', dict(identity, status='FAILED_EXITED' if failed else 'EXITED', exit_utc=r.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run'])
    parser.add_argument('--resume-authorization')
    args = parser.parse_args()
    run(args.resume_authorization)
