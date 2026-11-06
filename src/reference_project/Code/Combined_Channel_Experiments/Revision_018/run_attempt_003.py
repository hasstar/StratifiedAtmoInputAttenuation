"""Isolated attempt with corrected frozen power-domain target assertion."""
from pathlib import Path
import importlib.util, sys, traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('temporal018_v002_readonly', HERE / 'temporal_v002.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
PARENT = m.OUT
m.need(not list((PARENT / 'attempt_002/days').glob('*.npz')), 'Prior attempt has generated shards: inspect before restart')
m.need(m.read(PARENT / 'attempt_002/calculation_failure.json')['error'] == 'Frozen native hourly target', 'Expected diagnosed failure')
for p, h in m.read(PARENT / 'attempt_002/sources_before.json').items():
    m.need(m.sha(p) == h, 'Prior source changed')
m.OUT = PARENT / 'attempt_003'
original_protected = m.protected_sources


def protected():
    result = original_protected()
    for name in ('PROTOCOL.md', 'sources_before.json', 'started.json', 'attempt_001_interruption.json',
                 'calculation.stdout.log', 'calculation.stderr.log'):
        p = PARENT / name; result[str(p)] = m.sha(p)
    for p in (PARENT / 'attempt_002').rglob('*'):
        if p.is_file() and p.name != 'execution.lock':
            result[str(p)] = m.sha(p)
    return result


m.protected_sources = protected
original_stdout, original_stderr = sys.stdout, sys.stderr
with (m.OUT / 'calculation.stdout.log').open('x', encoding='utf-8') as out, (m.OUT / 'calculation.stderr.log').open('x', encoding='utf-8') as err:
    sys.stdout, sys.stderr = out, err
    try:
        m.main()
    except Exception as exc:
        m.emit('calculation_failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
