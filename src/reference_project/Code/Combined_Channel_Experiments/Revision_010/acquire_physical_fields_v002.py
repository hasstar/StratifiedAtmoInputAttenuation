"""Versioned exact NCAR TCLW label correction; preserves all v001 files."""
from __future__ import annotations

import copy
import sys
import acquire_physical_fields as base

base.need(base.sha(base.CODE/'acquire_physical_fields.py')=='1628b09b081d94d0b180028b5d623380da8d290fe7b97f4e2138591c2e9c7478', 'Original downloader changed')
base.need(base.sha(base.CODE/'verify_physical_fields.py')=='12bc98c6a1f68ceb52bf604c65c6e0a38a8ab0e601da2f51c47300aaca9aabd1', 'Original verifier changed')
base.FIELDS = copy.deepcopy(base.FIELDS)
base.FIELDS['tclw']['long_name'] = 'Total column liquid water'


def validate_fix():
    receipt = base.read(base.OUT/'implementation_v002.json')
    for name,digest in receipt['implementation_sha256'].items():
        base.need(base.sha(base.CODE/name)==digest, 'Version002 source changed')
    base.need(base.sha(base.OUT/'scope_plan.json')==receipt['scope_plan_sha256'], 'Scope changed')
    base.check_sources(base.read(base.OUT/'sources_before.json'))


def record_fix():
    with base.lock():
        base.check_sources(base.read(base.OUT/'sources_before.json'))
        original = base.OUT/'partial_history/200308_tclw.das.20261016T151800_272797Z.attempt1.part'
        base.need(base.sha(original)=='1113fe0cdebab0547edba57146b21147f9ebe94a3329ce3a5882e4c8d5bf34f6', 'Failed raw metadata changed')
        identity = base.contract(original.read_text(encoding='utf-8'), 'tclw')
        files = ['acquire_physical_fields_v002.py','verify_physical_fields_v002.py','CHANGE_RECORD_v002.md']
        base.emit('implementation_v002.json',dict(version=2,created_utc=base.utc(),source_long_name='Total column liquid water',
                  canonical_field='total_column_cloud_liquid_water',metadata_identity=identity,
                  failure_part=base.key(original),failure_part_sha256=base.sha(original),
                  official_identity_url='https://codes.ecmwf.int/grib/param-db/78',
                  scope_plan_sha256=base.sha(base.OUT/'scope_plan.json'),
                  implementation_sha256={name:base.sha(base.CODE/name) for name in files},
                  scope_changed=False,thresholds_changed=False,old_files_modified=False))
        print('Version002 exact-name correction recorded; same URL and scope retained.',flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['record-fix']:
        record_fix()
    elif sys.argv[1:]==['acquire']:
        validate_fix()
        base.acquire()
        validate_fix()
    else:
        raise RuntimeError('Only record-fix or acquire; never re-prepare')
