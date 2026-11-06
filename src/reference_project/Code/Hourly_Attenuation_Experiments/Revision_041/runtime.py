"""Isolated T017 adapter. Reuses pinned functions, never a historical CLI."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_P32 = _ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE'
_manifest = json.loads((_P32 / 'manifest.json').read_text(encoding='utf-8-sig'))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Verify executable historical dependencies BEFORE importing any of them.
for _name, _expected in _manifest['source_sha256'].items():
    if _name.replace('\\', '/').startswith('Code/'):
        if _digest(_ROOT / _name) != _expected:
            raise RuntimeError('Frozen historical code changed: ' + _name)

_spec = importlib.util.spec_from_file_location('t017_primitives32', _HERE.parent / 'Revision_032/runtime.py')
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
_base.OUT = _ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_041_TabM_Surface_RFE'
for _name in dir(_base):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_base, _name)

HERE, ROOT, OUT, P32 = _HERE, _ROOT, _base.OUT, _P32
MODELS = ('tabm',)
REFERENCE_CODE = HERE.parent / 'Revision_032'
DELIVERY34 = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002'
TASK_PATH = ROOT / 'Tasks/T017_TabM_Surface_Only_RFE.md'


def surface_names():
    return [name for name in key_names() if name.startswith('surface__')]


def stop_check():
    if (OUT / 'STOP_REQUESTED').exists():
        raise InterruptedError('T017 STOP_REQUESTED is effective')
    require(__import__('shutil').disk_usage(OUT).free > 3 * 1024**3, 'Disk safety threshold 3GiB')


def probe_lock(path):
    """Observe an existing OS lock; no file content or metadata replacement."""
    import msvcrt
    path = Path(path)
    if not path.exists():
        return dict(path=rel(path), status='ABSENT')
    with path.open('r+b') as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return dict(path=rel(path), status='OCCUPIED_OR_INACCESSIBLE')
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    return dict(path=rel(path), status='FREE')


def resource_check():
    import psutil
    import shutil
    workers = []
    for process in psutil.process_iter(['pid', 'name', 'create_time', 'cmdline']):
        if process.pid == os.getpid() or 'python' not in (process.info['name'] or '').lower():
            continue
        command = process.info['cmdline'] or []
        try:
            cwd = process.cwd()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            cwd = ''
        if str(ROOT).lower() in (' '.join(command) + ' ' + cwd).lower():
            workers.append(dict(pid=process.pid, create_time=process.info['create_time'], command=command))
    locks = [probe_lock(parent / 'writer.lock') for parent in (P31, P32)]
    other = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE'
    for name in ('execution.lock', 'continuation.lock', 'writer.lock'):
        if (other / name).exists():
            locks.append(probe_lock(other / name))
    result = dict(utc=now(), project_python_workers=workers, old_locks=locks,
                  disk_free_bytes=shutil.disk_usage(OUT).free,
                  available_memory_bytes=psutil.virtual_memory().available)
    require(not workers, 'Another project Python process exists; inspect rather than start concurrently')
    require(all(x['status'] != 'OCCUPIED_OR_INACCESSIBLE' for x in locks), 'Historical OS lock occupied/inaccessible')
    require(result['available_memory_bytes'] > 2 * 1024**3, 'Available RAM below 2GiB')
    stop_check()
    return result


@contextlib.contextmanager
def writer_lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'writer.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('Another T017 worker owns writer.lock') from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def load_reference_functions():
    auditor = module('verify', REFERENCE_CODE / 'verify.py')
    runner = module('t017_reference_run32', REFERENCE_CODE / 'run.py')
    return runner, auditor
