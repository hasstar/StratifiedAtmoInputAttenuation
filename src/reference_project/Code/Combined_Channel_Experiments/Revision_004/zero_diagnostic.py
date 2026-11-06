"""Transparent post hoc fixed-zero control, independent of all model fitting."""
from common import *
import math
import numpy as np
import pandas as pd

def main():
    offline()
    with lock():
        require(read(OUT/'independent_qa.json')['status']=='PASS','Primary independent QA first')
        guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
        uk=pd.read_parquet(OUT/'predictions_physical_and_reused.parquet')
        local=pd.read_parquet(REV2/'predictions/local_test.parquet')
        rows=[]
        for label,d in [('UK_common',uk.loc[uk.common_score_eligible]),('LOCAL_test',local)]:
            y=d.truth_db.to_numpy(float);n=len(y);m=float(np.mean(y));ss=float(np.mean(y*y));variance=float(np.mean((y-m)**2))
            record={'cohort':label,'n':n,'prediction_db':0.,'rmse_db':math.sqrt(ss),'mae_db':float(np.mean(np.abs(y))),'bias_db':-m,'r2':1-ss/variance}
            # Independent scalar accumulation verifies the vectorized diagnostics.
            require(abs(record['rmse_db']-math.sqrt(math.fsum(float(v)**2 for v in y)/n))<1e-12,'Independent zero RMSE')
            require(abs(record['mae_db']-math.fsum(abs(float(v)) for v in y)/n)<1e-12,'Independent zero MAE')
            require(abs(record['bias_db']+math.fsum(map(float,y))/n)<1e-12,'Independent zero bias')
            rows.append(record)
        protected_verify(guard)
        emit(OUT/'posthoc_zero_diagnostic.json',{'status':'PASS','post_hoc':True,'not_selected_or_fitted':True,'point_estimates_only':True,'records':rows,'code_sha256':sha(Path(__file__))})
        print(rows,flush=True)

if __name__=='__main__':main()
