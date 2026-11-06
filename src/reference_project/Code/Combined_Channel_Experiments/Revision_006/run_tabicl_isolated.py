"""Independent output/lock for the second frozen model; at most 8 CPU threads total."""
import sys
import run_frozen as base
if __name__=='__main__':
    if sys.argv[1:]!=['run','--model','tabicl_v2_author_frozen']:
        raise SystemExit('Only the declared TabICLv2 frozen run')
    original=base.OUT
    base.OUT=original/'inference_tabicl_002'
    base.OUT.mkdir(exist_ok=False)
    base.emit(base.OUT/'preflight.json',base.read(original/'preflight.json'))
    base.emit(base.OUT/'source_sha256_before.json',base.read(original/'source_sha256_before.json'))
    base.run('tabicl_v2_author_frozen')
