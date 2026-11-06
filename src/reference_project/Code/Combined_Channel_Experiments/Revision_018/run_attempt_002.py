"""Fresh isolated output attempt; immutable scientific computation imported read-only."""
from pathlib import Path
import importlib.util, sys, traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('temporal018_readonly', HERE / 'temporal.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OLD_OUT = m.OUT
m.need(not (OLD_OUT / 'tests.json').exists() and not (OLD_OUT / 'days').exists(), 'Attempt001 did generate results: inspect instead')
for p, h in m.read(OLD_OUT / 'sources_before.json').items():
    m.need(m.sha(p) == h, 'Attempt001 source changed')
m.OUT = OLD_OUT / 'attempt_002'
original_protected = m.protected_sources


def protected():
    result = original_protected()
    for name in ('PROTOCOL.md', 'sources_before.json', 'started.json', 'attempt_001_interruption.json',
                 'calculation.stdout.log', 'calculation.stderr.log'):
        p = OLD_OUT / name
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
