"""Route immutable independent verification to the isolated attempt002 output."""
from pathlib import Path
import importlib.util, sys, traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent018_readonly', HERE / 'verify.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.OUT = m.OUT / 'attempt_002'
original_stdout, original_stderr = sys.stdout, sys.stderr
with (m.OUT / 'verification.stdout.log').open('x', encoding='utf-8') as out, (m.OUT / 'verification.stderr.log').open('x', encoding='utf-8') as err:
    sys.stdout, sys.stderr = out, err
    try:
        m.main()
    except Exception as exc:
        m.emit('verification_failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
