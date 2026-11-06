"""Independent QA wrapper for the verified NCAR parameter78 source long_name."""
from __future__ import annotations

import json
import sys
import acquire_physical_fields_v002 as fixed
import verify_physical_fields as verifier

old = verifier.IDENTITIES['tclw']
verifier.IDENTITIES['tclw'] = (old[0], 'Total column liquid water', *old[2:])

if __name__=='__main__':
    fixed.validate_fix()
    if sys.argv[1:]==['selftest']:
        print(json.dumps(verifier.synthetic_controls()))
    elif not sys.argv[1:]:
        verifier.verify()
    else:
        raise RuntimeError('Only selftest or offline verification')
    fixed.validate_fix()
