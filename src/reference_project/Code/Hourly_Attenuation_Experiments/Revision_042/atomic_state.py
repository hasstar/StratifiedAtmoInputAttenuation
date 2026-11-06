"""Execution JSON only; adapted from the immutable Revision033 atomic_state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


def sharing(error):
    return isinstance(error, PermissionError) and getattr(error, 'winerror', None) in (5, 32, 33)


def atomic_state(path, value, *, event=None, attempts=14, replace_fn=None, sleep_fn=None):
    if attempts < 1:
        raise ValueError('At least one attempt required')
    replace_fn = os.replace if replace_fn is None else replace_fn
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.r042.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    errors = []
    for attempt in range(attempts):
        try:
            replace_fn(temp, path)
        except OSError as error:
            errors.append(dict(attempt=attempt + 1, error=repr(error),
                               winerror=getattr(error, 'winerror', None)))
            if not sharing(error) or attempt == attempts - 1:
                if event is not None:
                    event(dict(path=str(path), temporary=str(temp), outcome='FAILED', errors=errors))
                raise
            sleep_fn(min(.05 * 2**attempt, .5))
        else:
            if errors and event is not None:
                event(dict(path=str(path), outcome='RECOVERED', errors=errors))
            return
