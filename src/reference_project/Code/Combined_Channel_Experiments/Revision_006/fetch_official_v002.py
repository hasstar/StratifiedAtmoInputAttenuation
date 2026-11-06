"""Transport correction: identity encoding makes wire length comparable to bytes."""
import sys
import fetch_official as base
base.SESSION.headers['Accept-Encoding'] = 'identity'
base.OUT = base.OUT / 'acquisition_002'
if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != 'sources':
        raise SystemExit('Only sources stage')
    base.sources()
