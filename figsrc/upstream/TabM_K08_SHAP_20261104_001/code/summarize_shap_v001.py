"""Describe stored k8 attributions and compare archived test errors; no fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
P=Path(__file__).resolve().parents[1]
assert json.loads((P/'qa/independent_qa.json').read_text('utf-8'))['all_pass']
assert not (P/'analysis_evidence.json').exists()
summary={}
for name in ('full','base'):
 ident=json.loads((P/'results'/f'{name}_identity.json').read_text('utf-8'))
 raw=pd.read_csv(P/'results'/f'{name}_test_inputs.csv')
 arr=np.load(P/'results'/f'{name}_shap.npz');phi=arr['phi']
 scores=pd.read_csv(P/'results'/f'{name}_shap_summary.csv')
 imp=np.mean(abs(phi),axis=0);rows=[]
 for j in np.argsort(-imp):
  x=raw[ident['features'][j]].to_numpy(float);ok=np.isfinite(x)
  q25,q75=np.quantile(x[ok],[.25,.75]);lo=ok&(x<=q25);hi=ok&(x>=q75)
  rows.append({'feature':ident['features'][j],'mean_abs_shap_db':float(imp[j]),'mean_signed_shap_db':float(phi[:,j].mean()),'min_shap_db':float(phi[:,j].min()),'max_shap_db':float(phi[:,j].max()),'n_missing_raw':int((~ok).sum()),'raw_value_vs_shap_spearman':float(spearmanr(x[ok],phi[ok,j]).statistic),'raw_lower_quartile_mean_shap_db':float(phi[lo,j].mean()),'raw_upper_quartile_mean_shap_db':float(phi[hi,j].mean())})
 summary[name]={'baseline_db':float(arr['baseline_db']),'test_metrics':ident['test_metrics'],'ordered_features':rows,'weather_summary':scores.to_dict('records')}
summary['interpretation']='All numbers summarize fitted prediction functions relative to 64 training references. Raw-value/SHAP association is descriptive, not causal or conditional effect. Importance is not a physical attenuation budget.'
summary['source_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((P/'results').glob('*_shap.npz')),*sorted((P/'results').glob('*_test_inputs.csv'))]}
(P/'analysis_evidence.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
for name in ('full','base'):
 print(name,'baseline',summary[name]['baseline_db'])
 for row in summary[name]['ordered_features']: print(row['feature'],round(row['mean_abs_shap_db'],6),round(row['raw_value_vs_shap_spearman'],3))
