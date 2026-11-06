"""Inspect existing scientific exports without changing them."""
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'最终提交的修改材料/论文图表_20261030_001'
OUT.mkdir(exist_ok=True)
(OUT/'05_QA').mkdir(exist_ok=True)
paths=[
'Outputs/Datasets/Signal_Detectability_Revision_003/train.parquet',
'Outputs/Datasets/Signal_Detectability_Revision_003/validation.parquet',
'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet',
'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s_hourly_summary.csv',
'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002/tables/all_states_1560.csv',
'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002/tables/all_test_metrics_15605_rows.csv',
'最终提交的修改材料/ITU组合模型_20261027_001/01_结果/ITU_共同684小时_逐小时预测.csv',
'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/same_k_comparison.csv',
'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/all_states_metrics.csv',
'最终提交的修改材料/TabM模型使用50个参数进行RFE结果/retained_features.csv',
'Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE/evaluation/metrics_all_weather_station.csv',
'最终提交的修改材料/补充材料_20261024_001/快变与TabM组合_分天气汇总.csv',
'最终提交的修改材料/补充材料_20261024_001/快变_ACF分天气汇总.csv',
'Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble/hourly_results.csv',
'Outputs/Experiments/Fast_Fading_Revision_002/dynamics/r16/candidate_ranking.csv',
]
report=[]
for rel in paths:
 p=ROOT/rel
 if not p.exists(): report.append({'path':rel,'missing':True});continue
 d=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
 report.append({'path':rel,'rows':len(d),'columns':d.columns.tolist(),'sample':d.iloc[:1].to_dict('records')})
 print(rel, '\n rows=',len(d),'columns=',','.join(d.columns[:80]))
 if 'cohort' in d: print(d[['cohort','station','event']].to_string(index=False))
 if 'protocol' in d: print(d.protocol.unique())
(OUT/'05_QA/source_inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
