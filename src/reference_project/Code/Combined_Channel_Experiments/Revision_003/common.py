"""Offline-only utilities for the new, immutable expanded-UK delivery."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_003'
REV1 = HERE.parent / 'Revision_001'
REV2 = HERE.parent / 'Revision_002'
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_002'
EXT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external'
ACQ = EXT / 'weather_gbs_extend_v011'
ALIGN = ROOT / 'Outputs/Datasets/ClearSky_Alignment_Revision_004'
GBS = ROOT / '外部样例数据CEDA/GBS_20p7GHz'
R29 = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
INDEXES = [ROOT/'Tasks/T012_Combined_Channel_And_External_Validation.md', REV1/'PROGRESS.md']
FEATURES = ['surface__convective_available_potential_energy',
            'surface__maximum_2m_temperature_since_previous_post_processing',
            'p850__ozone_mass_mixing_ratio', 'p850__relative_humidity',
            'p650__u_component_of_wind', 'p500__ozone_mass_mixing_ratio',
            'p5__ozone_mass_mixing_ratio', 'p5__u_component_of_wind',
            'p1__geopotential', 'p1__u_component_of_wind']
RAIN = ['mean_total_precipitation_rate', 'mean_snowfall_rate']
PLAN_SHA = '1f9be56c4351940342a135b487d44aa1d6914c4a1a3dfc0ef4b040f43b53fdac'
PLAN_FP = 'b58d2b9f1f83970cca410195504cd0aa21cb0f7e57cf004ddb1e43100c5f56db'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def emit(path, obj):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), 'Write outside new output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def signed_check(obj):
    require(obj['fingerprint'] == fp({k:v for k,v in obj.items() if k != 'fingerprint'}), 'Signed record mismatch')

def deny(*args, **kwargs):
    raise RuntimeError('FORBIDDEN: network/training/optimization in frozen offline evaluation')

def offline():
    socket.socket.connect = deny
    socket.create_connection = deny

@contextlib.contextmanager
def lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'evaluation.lock'
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

def protected_verify(manifest):
    for name, digest in manifest.items():
        require(sha(ROOT/name) == digest, 'Protected source changed: ' + name)

def source_snapshot(plan):
    guard = {}
    def add(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected:
            require(digest == expected, 'Frozen SHA mismatch: ' + str(path))
        guard[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = digest
        return path
    for p,h in plan['protected_files'].items(): add(ROOT/p, h)
    for p,h in plan['source_sha256'].items(): add(ROOT/p, h)
    # Guard all stopped downloader and first comparator artifacts, including failures.
    for directory in (ACQ, OLD, REV2):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.lock': add(p)
    for p in (R29/'rfe/models/k_0010.joblib', R29/'rfe/states/k_0010.json', R29/'run_manifest.json',
              ROOT/'Code/Hourly_Attenuation_Experiments/Revision_014/models.py',
              ALIGN/'build_manifest.json', ALIGN/'ceda_gbs_hourly_attenuation.parquet',
              REV1/'audit_gbs_48h_actual_fields_v001.py', REV1/'evaluate_gbs_frozen_top10_v002.py'):
        add(p)
    for p in INDEXES:
        dest = OUT/'document_snapshots/before'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f: f.write(p.read_bytes())
    emit(OUT/'source_sha256_before.json', guard)
    emit(OUT/'index_sha256_before.json', {str(p.relative_to(ROOT)):sha(p) for p in INDEXES})
    return guard, add
