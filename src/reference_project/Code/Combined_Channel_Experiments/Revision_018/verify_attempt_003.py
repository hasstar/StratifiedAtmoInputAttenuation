"""Independent corrected power-domain check; no change to stored labels or scores."""
from pathlib import Path
import importlib.util, sys, traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent018_v002_readonly', HERE / 'verify_v002.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.OUT = m.OUT / 'attempt_003'
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
