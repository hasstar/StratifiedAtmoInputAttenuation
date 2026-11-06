"""Inspected T017 continuation; no change to scientific code or manifest."""
from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'Revision_041'))
import runtime as r
import run_surface as original
sys.path.insert(0, str(HERE.parent / 'Revision_042'))
from atomic_state import atomic_state, sharing

AUDIT = r.ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_043_Surface_Execution_Repair'
BASE_FP = '8d05760b3be12b26ef54ad3329da9ed2996c708bd5a920402504489bae988320'
PRIOR_PID = 41856
PRIOR_CREATE = 1792987763.7064028
ALLOWED = {r.OUT / x for x in ('progress.json', 'state_count.json', 'status.json', 'active_worker.json')}


def prior_exit_and_resources():
    import psutil
    try:
        prior = psutil.Process(PRIOR_PID)
        r.require(abs(prior.create_time() - PRIOR_CREATE) > .01, 'Original worker still exists')
    except psutil.NoSuchProcess:
        pass
    lock = r.probe_lock(r.OUT / 'writer.lock')
    r.require(lock['status'] == 'FREE', 'Scientific writer lock not free')
    resources = r.resource_check()
    return dict(prior_process_exit_verified=True, writer_lock=lock, resources=resources, utc=r.now())


def tests():
    folder = AUDIT / 'synthetic_tests'
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / 'state.json'
    events = []
    atomic_state(target, dict(version=1))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int

    def handle(share):
        value = kernel.CreateFileW(str(target), 0x80000000, share, None, 3, 0x80, None)
        r.require(value not in (None, ctypes.c_void_p(-1).value), 'Synthetic Windows read handle')
        return value

    locked = handle(1 | 2)
    closer = None
    try:
        try:
            r.c.update(target, dict(version=2))
        except OSError as error:
            r.require(sharing(error), 'Expected old Windows denial')
        else:
            raise RuntimeError('Could not reproduce old denial')
        r.require(r.read(target) == dict(version=1), 'Legacy denial preserves target')
        closer = threading.Thread(target=lambda: (time.sleep(.2), kernel.CloseHandle(locked)))
        closer.start()
        atomic_state(target, dict(version=3), event=events.append)
        closer.join()
    finally:
        if closer is None:
            kernel.CloseHandle(locked)
        else:
            closer.join()
    r.require(r.read(target) == dict(version=3) and events[-1]['outcome'] == 'RECOVERED', 'Real handle retry')
    shared = handle(1 | 2 | 4)
    shared_closer = threading.Thread(target=lambda: (time.sleep(.2), kernel.CloseHandle(shared)))
    shared_closer.start()
    try:
        atomic_state(target, dict(version=4), event=events.append)
    finally:
        shared_closer.join()
    r.require(r.read(target) == dict(version=4), 'Bounded retry succeeds with promptly closed delete-sharing reader')

    def denied(*args):
        error = PermissionError(13, 'synthetic persistent denial')
        error.winerror = 5
        raise error

    try:
        atomic_state(target, dict(version=5), replace_fn=denied, sleep_fn=lambda _: None,
                     attempts=2, event=events.append)
    except PermissionError:
        pass
    else:
        raise RuntimeError('Permanent denial swallowed')
    r.require(r.read(target) == dict(version=4), 'Permanent denial preserves previous target')

    def other_error(*args):
        raise OSError(22, 'synthetic non-sharing failure')

    try:
        atomic_state(target, dict(version=6), replace_fn=other_error,
                     sleep_fn=lambda _: (_ for _ in ()).throw(RuntimeError('Unrelated error retried')),
                     event=events.append)
    except OSError as error:
        r.require(error.errno == 22, 'Unrelated error fidelity')
    else:
        raise RuntimeError('Unrelated error swallowed')
    r.require(r.read(target) == dict(version=4), 'Unrelated error preserves target')
    r.emit(folder / 'events.json', events)
    return dict(status='PASS', synthetic_only=True, scientific_fit_calls=0,
                legacy_windows_denial_reproduced=True, real_handle_release_retry_passed=True,
                delete_sharing_reader_release_retry_passed=True, persistent_denial_propagated=True,
                unrelated_error_propagated=True, previous_target_preserved=True)


def committed_boundary(manifest):
    names = list(manifest['feature_names'])
    committed = {}
    qas = list((r.OUT / 'models/tabm').glob('k_*/qa.json'))
    r.require(len(qas) == 40, 'Expected exactly40 accepted states at inspected boundary')
    for k in range(50, 10, -1):
        folder = r.state_dir('tabm', k)
        qa, state = r.read(folder / 'qa.json'), r.read(folder / 'state.json')
        r.require(qa['status'] == 'PASS' and qa['manifest'] == BASE_FP, 'Prior QA identity')
        r.require(state['manifest'] == BASE_FP and state['feature_names'] == names, 'Prior path identity')
        for name, key in (('state.json', 'state_sha256'), ('evaluation.json', 'evaluation_sha256')):
            file = folder / name
            r.require(r.sha(file) == qa[key], 'Prior committed state/evaluation digest')
        for role in [*state['folds'], state['full_fit']]:
            r.require(r.sha(r.ROOT / role['model_file']) == role['model_sha256'], 'Prior model digest')
            committed[role['model_file']] = role['model_sha256']
        names = [name for name in names if name != state['removed_feature_next']]
    r.require(len(names) == 10, 'Independent path ends at k10')
    for folder in (r.OUT / 'models/tabm', r.OUT / 'fits/tabm'):
        for file in folder.rglob('*'):
            if file.is_file() and (file.suffix in ('.joblib', '.npz') or file.name in ('qa.json', 'state.json', 'evaluation.json', 'result.json') or file.name.startswith('permutation_')):
                committed[r.rel(file)] = r.sha(file)
    return names, committed


def checkpoint_identity(names, manifest):
    data = r.frames()
    train = data['train']
    expected = r.fp(dict(manifest=manifest['fingerprint'], model='tabm', k=10, fold=0, names=names,
                         train=train.hour_bjt.astype(str).tolist(), hold=None, config=r.config('tabm'), search=False))
    path = r.OUT / 'fits/tabm/k_0010_fold0/recovery.pt'
    digest = r.sha(path)
    cp = r.c.load_tensor(path)
    r.require(cp['identity'] == expected, 'Interrupted checkpoint exact job identity')
    r.require(cp['epoch'] == 50 and [h['epoch'] for h in cp['history']] == list(range(1, 51)), 'Epoch50 complete checkpoint')
    r.require(all(key in cp for key in ('weights', 'optimizer', 'generator', 'rng')), 'Optimizer/RNG completeness')
    net = r.make_nn('tabm', names)
    net.load_state_dict(cp['weights'], strict=True)
    cfg = r.config('tabm')['config']
    optimizer = r.torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    optimizer.load_state_dict(cp['optimizer'])
    r.require(len(optimizer.state) > 0, 'Saved optimizer has state')
    r.torch.Generator().set_state(cp['generator'])
    r.torch.Generator().set_state(cp['rng'])
    _, bundle = r.prep_fit('tabm', train, names)
    bundle['net'] = net
    actual = r.scores(train, r.predict(bundle, train))
    maximum = 0.
    for weather, metrics in actual.items():
        expected_metrics = cp['history'][-1]['train'][weather]
        for key in ('rmse_db', 'r2', 'mae_db', 'bias_db', 'pearson_r'):
            a, b = metrics[key], expected_metrics[key]
            if a is None or b is None:
                r.require(a is b, 'Saved checkpoint null metric identity')
            else:
                maximum = max(maximum, abs(a - b))
                r.require(math.isclose(a, b, rel_tol=0, abs_tol=2e-6), 'Saved epoch metric replay')
    r.require(r.sha(path) == digest, 'Checkpoint changed during inspection')
    return dict(path=r.rel(path), sha256=digest, identity=expected, epoch=50, resume_epoch=51,
                optimizer_rng_verified=True, training_metric_replay_max_delta=maximum, scientific_fit_calls=0)


def prepare():
    r.require(not AUDIT.exists(), 'Repair directory already exists; inspect instead of retrying prepare')
    resources = prior_exit_and_resources()
    manifest = r.manifest()
    r.require(manifest['fingerprint'] == BASE_FP, 'Frozen scientific manifest identity')
    names, committed = committed_boundary(manifest)
    checkpoint = checkpoint_identity(names, manifest)
    AUDIT.mkdir(parents=True, exist_ok=False)
    synthetic = tests()
    targets = [r.OUT / x for x in ('active_worker.json', 'status.json', 'state_count.json', 'progress.json')]
    targets += [r.ROOT / checkpoint['path']]
    targets += sorted((r.OUT / 'failures').glob('*')) + sorted((r.OUT / 'logs').glob('*'))
    targets += sorted((r.OUT / 'runs').glob('*')) + sorted(r.OUT.glob('.progress.json.*.tmp'))
    targets += sorted((r.OUT / 'fits/tabm/k_0010_fold0').glob('attempt_*.json'))
    preserved = {}
    for source in targets:
        if not source.is_file():
            continue
        dest = AUDIT / 'preserved_before_resume' / source.relative_to(r.OUT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r.require(not dest.exists(), 'Preservation destination already exists')
        digest = r.sha(source)
        shutil.copy2(source, dest)
        r.require(r.sha(dest) == digest == r.sha(source), 'Preserved evidence SHA')
        preserved[r.rel(source)] = dict(copy=r.rel(dest), sha256=digest)
    sources = {r.rel(p): r.sha(p) for p in HERE.iterdir() if p.is_file()}
    for dependency in (HERE.parent / 'Revision_042/atomic_state.py', HERE.parent / 'Revision_042/repair_surface.py'):
        sources[r.rel(dependency)] = r.sha(dependency)
    reused = HERE.parent / 'Revision_033/execution.py'
    repair = dict(version='T017-execution-only-043', source_sha256=sources,
                  adaptation_source=dict(path=r.rel(reused), sha256=r.sha(reused)),
                  scientific_manifest_sha256=r.sha(r.OUT / 'manifest.json'), scientific_fingerprint=BASE_FP,
                  original_sources_checked=len(manifest['source_sha256']), scientific_contract_changed=False,
                  checkpoint=checkpoint, prior_committed_sha256=committed, preserved=preserved,
                  preflight=resources, utc=r.now())
    r.emit(AUDIT / 'repair_manifest.json', repair)
    r.emit(AUDIT / 'QA.json', dict(status='PASS', synthetic=synthetic, checkpoint=checkpoint,
           prior_states_verified=40, prior_committed_files=len(committed),
           original_pinned_sources_verified=len(manifest['source_sha256']),
           repair_manifest_sha256=r.sha(AUDIT / 'repair_manifest.json'), utc=r.now()))
    r.emit(AUDIT / 'resume_authorization.json', dict(allow_resume=True, manifest=BASE_FP,
           diagnosis='Windows denied atomic replacement of mutable progress.json; bounded sharing retry and delete-sharing observer tested. Historical handle owner not proven.',
           prior_process_exit_verified=True, no_effective_user_stop=True,
           user_authorization='2026-10-26: 快点修复', checkpoint=checkpoint,
           repair_manifest_sha256=r.sha(AUDIT / 'repair_manifest.json'), utc=r.now()))
    print('REPAIR_PREPARED', dict(states=40, checkpoint_epoch=50, resume_epoch=51,
          source_count=len(manifest['source_sha256']), preserved_files=len(preserved)), flush=True)


def retry_event(value):
    r.emit(AUDIT / 'retry_events' / f'{time.time_ns()}.json', dict(value, utc=r.now(), pid=os.getpid()))


def mutable_update(path, value):
    r.require(Path(path).resolve() in ALLOWED, 'Repair attempted non-execution-state update')
    atomic_state(path, value, event=retry_event)


def run():
    repair = r.read(AUDIT / 'repair_manifest.json')
    qa = r.read(AUDIT / 'QA.json')
    r.require(qa['status'] == 'PASS' and qa['repair_manifest_sha256'] == r.sha(AUDIT / 'repair_manifest.json'), 'Repair QA gate')
    for file, digest in repair['source_sha256'].items():
        r.require(r.sha(r.ROOT / file) == digest, 'Executed repair source changed')
    r.require(r.sha(r.OUT / 'manifest.json') == repair['scientific_manifest_sha256'], 'Scientific manifest changed')
    r.manifest()
    for file, digest in repair['prior_committed_sha256'].items():
        r.require(r.sha(r.ROOT / file) == digest, 'Previously committed artifact changed')
    cp = repair['checkpoint']
    r.require(r.sha(r.ROOT / cp['path']) == cp['sha256'], 'Inspected checkpoint changed')
    resources = prior_exit_and_resources()
    import psutil
    r.emit(AUDIT / 'launch_receipt.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
           command=psutil.Process().cmdline(), resources=resources, checkpoint=cp, utc=r.now()))
    r.update = mutable_update
    r._base.update = mutable_update
    r.require(r.progress.__globals__['update'] is mutable_update, 'Progress adapter binding')
    print('RESUME_INSPECTED_CHECKPOINT epoch50 -> epoch51; accepted_states40 preserved', flush=True)
    original.run(str(AUDIT / 'resume_authorization.json'))
    for file, digest in repair['prior_committed_sha256'].items():
        r.require(r.sha(r.ROOT / file) == digest, 'Prior artifact changed during continuation')
    r.emit(AUDIT / 'POST_RESUME_QA.json', dict(status='PASS', prior_committed_files_unchanged=len(repair['prior_committed_sha256']),
           actual_visual_review=False, utc=r.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else run()
