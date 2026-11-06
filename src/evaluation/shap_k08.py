"""Recompute the published marginal SHAP analysis for both frozen k=8 models."""
from pathlib import Path
import argparse, shutil, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--check-kernel',action='store_true',help='Verify attribution algebra only; no model explanations')
    a=ap.parse_args()
    source=s.path('figure_sources')/'TabM_K08_SHAP_20261104_001'
    m=s.load_module('supp_shap_k08',source/'code/calculate_shap_v001.py')
    if a.check_kernel:
        print(m.method_checks());return
    out=a.out or s.path('new_runs')/s.timestamp()/'shap_k08'
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'PROTOCOL.md',out/'PROTOCOL.md')
    m.ROOT=s.ProjectRoot();m.OUT=out
    original=m.module
    m.module=lambda name,path:s.hourly() if Path(path).name=='runtime.py' else original(name,path)
    with s.run_lock('frozen k8 SHAP','shap.lock'):
        m.main()
    s.write(out/'package_provenance.json',dict(source=s.rel(source),source_sha256=s.sha(source/'code/calculate_shap_v001.py'),
        weights='frozen full-set and base-set k8',training_calls=0,main_seed=17,
        background_seed=20261103,permutation_seeds=[20261105,20261106]))

if __name__=='__main__':main()
