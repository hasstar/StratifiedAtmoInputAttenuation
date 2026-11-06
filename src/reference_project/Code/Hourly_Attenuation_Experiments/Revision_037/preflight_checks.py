"""Small offline implementation checks; not training, inference, or timing trials."""
from pathlib import Path
import numpy as np
import pandas as pd
from freeze import *
from assemble_inputs import read_cds, CDS
from ncar_inputs import decode, contract, fp, ACQ
from audit_inputs import independent_ascii, audit_geometry
from infer_frozen import metric
from audit_metrics import reference

def main():
    checks=[]
    for p in CODE.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    checks.append('All new source files compile without executing their main functions')
    n=read(ACQ/'ncar_plan.json')['nodes'][0]
    raw=ACQ/'ncar_values'/f'{n["id"]}.ascii';das=ACQ/'ncar_metadata'/f'{fp(n["url_path"])[:24]}.das'
    values=decode(raw.read_text(),das.read_text(),n)
    second=independent_ascii(raw.read_text())[n['variable']+'.'+n['variable']].ravel()
    np.testing.assert_array_equal(values,second);checks.append('Independent DAP decoder matches first actual 24-hour pressure payload')
    rejected=False
    try:contract(das.read_text().replace('ecmwf_parameter 60;','ecmwf_parameter 61;'),n['spec'],True)
    except RuntimeError:rejected=True
    need(rejected,'Wrong parameter must fail');checks.append('Wrong source parameter ID rejected')
    audit_geometry(pd.read_parquet(ACQ/'geometry.parquet'));checks.append('All 56640 pressure geometry records independently replayed')
    j=read(CDS/'plan.json')['jobs'][0];receipt=read(CDS/'jobs'/j['id']/'download.json');rows,meta=read_cds(j,receipt)
    need(len(rows)==936,'312 hours x three CDS fields');checks.append('First actual CDS batch: 936 values, correct IDs/units/time/grid')
    for y,p in [(np.array([-1.,0,1]),np.array([10.,11,12])),(np.array([1.,1]),np.array([2.,3.])),(np.array([1.]),np.array([1.]))]:
        a=metric(y,p);b=reference(y,p)
        for k,v in b.items():
            need(a[k]==v or (v is not None and abs(a[k]-v)<1e-12),'Independent metric fixtures')
    need(metric(np.array([-1.,0,1]),np.array([10.,11,12]))['r2']<0,'Negative R2 retained')
    checks.append('Independent scalar/numpy metrics agree; negative R2 retained and degenerate R2 null')
    emit(OUT/'implementation_preflight_qa.json',dict(status='PASS_IMPLEMENTATION_ONLY_FULL_DATA_PENDING',utc=now(),checks=checks,
        source_sha256={rel(p):sha(p) for p in CODE.glob('*.py')},actual_sources={rel(p):sha(p) for p in [raw,das,CDS/receipt['file']]},
        training_calls=0,inference_calls=0,full_uk_input_qa_complete=False))
    print('IMPLEMENTATION_PREFLIGHT_PASS_FULL_INPUTS_PENDING',flush=True)
if __name__=='__main__':main()
