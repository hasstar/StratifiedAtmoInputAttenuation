"""Keep author bytes intact; disable optional memory splitting via public API.

In 2.0.9, memory.py inspects requires_grad of optional None inputs when every
weight is frozen. This fails before inference. No scores informed this setting.
The alternative is the author's normal unsplit computation, not a weight change.
"""
import sys
import run_frozen as base

if __name__=='__main__':
    if sys.argv[1:] != ['run', '--model', 'tabpfn_v2_author_frozen']:
        raise SystemExit('Only the failed TabPFN run is eligible for this correction')
    old_out=base.OUT
    base.OUT=old_out/'inference_002'
    base.OUT.mkdir(exist_ok=False)
    pre=base.read(old_out/'preflight.json')
    pre['overrides']['tabpfn'] += '; memory_saving_mode=False: optional memory optimization disabled to avoid None.requires_grad bug; weights and math unchanged'
    base.emit(base.OUT/'preflight.json',pre)
    base.emit(base.OUT/'source_sha256_before.json',base.read(old_out/'source_sha256_before.json'))
    cls=base.TabPFNRegressor
    def frozen_regressor(**kwargs):
        return cls(**kwargs,memory_saving_mode=False)
    base.TabPFNRegressor=frozen_regressor
    base.run('tabpfn_v2_author_frozen')
