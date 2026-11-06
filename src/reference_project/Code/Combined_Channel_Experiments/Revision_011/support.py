"""Append-only Revision011 support. No machine-learning imports or network at import."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '2'
CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_011'
REV = lambda n: ROOT / f'Outputs/Experiments/Combined_Channel_Revision_{n:03d}'
AUTOMATION = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
STOP = REV(1) / 'external/weather_gbs_extend_v011/STOP_REQUESTED.json'

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def utc(): return datetime.now(timezone.utc).isoformat()
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def new(p):
    p = Path(p)
    need(p.resolve().is_relative_to(OUT.resolve()), 'Outside new output domain')
    need(not p.exists(), 'Refusing overwrite: '+str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def emit(name, value):
    with new(OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

@contextlib.contextmanager
def lock():
    import msvcrt
    with (OUT/'execution.lock').open('a+b') as f:
        if f.tell() == 0: f.write(b'0'); f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

def check(guard):
    for p, h in guard.items(): need(sha(ROOT/p) == h, 'Protected source changed: '+p)

def initialize():
    OUT.mkdir(exist_ok=False)
    with lock():
        guard = read(REV(10)/'sources_before.json')
        check(guard)
        for n in (4,6,8,9,10):
            for p in REV(n).rglob('*'):
                if p.is_file() and p.suffix != '.lock':
                    key = str(p.relative_to(ROOT)); h = sha(p)
                    need(key not in guard or guard[key] == h, 'Guard conflict: '+key)
                    guard[key] = h
        paths = [AUTOMATION, STOP, ROOT/'Tasks/T012_Combined_Channel_And_External_Validation.md',
                 ROOT/'Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md',
                 ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet',
                 ROOT/'Code/Combined_Channel_Experiments/Revision_004/physics.py']
        for p in paths: guard[str(p)] = sha(p)
        need('status = "PAUSED"' in AUTOMATION.read_text(encoding='utf-8'), 'Keep tvt paused')
        emit('sources_before.json', guard)
        emit('launch.json', dict(utc=utc(), pid=os.getpid(), guard_count=len(guard),
             protocol_sha256=sha(CODE/'PROTOCOL.md'), training=0, weather_downloads=0,
             stopped_acquisition_preserved=True))

def offline():
    def deny(*args, **kwargs): raise RuntimeError('Network forbidden during arithmetic/QA')
    socket.socket.connect = deny
    socket.create_connection = deny
