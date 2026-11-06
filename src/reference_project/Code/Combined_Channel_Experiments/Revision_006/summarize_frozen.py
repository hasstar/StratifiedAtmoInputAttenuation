"""Score frozen predictions, reuse historical outputs, and independently verify."""
from __future__ import annotations
import hashlib,json,math,re,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_006'
OLD=ROOT/'Outputs/Experiments/Combined_Channel_Revision_002'
EXP=ROOT/'Outputs/Experiments/Combined_Channel_Revision_003'
PHY=ROOT/'Outputs/Experiments/Combined_Channel_Revision_004'
RUNS={
 'tabpfn_v2_author_frozen':OUT/'inference_004/tabpfn_v2_author_frozen',
 'tabicl_v2_author_frozen':OUT/'inference_tabicl_002/tabicl_v2_author_frozen',
}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def emit(p,o):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def need(v,m):
 if not v:raise RuntimeError(m)
def score(y,p):
 e=p-y;den=np.sum((y-np.mean(y))**2)
 return dict(n=len(y),rmse_db=float(np.sqrt(np.mean(e**2))),mae_db=float(np.mean(abs(e))),bias_db=float(np.mean(e)),r2=float(1-np.sum(e**2)/den) if den>0 else None)
def independent_score(y,p):
 n=len(y);err=[float(a)-float(b) for a,b in zip(p,y)];center=math.fsum(float(x) for x in y)/n
 ss=math.fsum(x*x for x in err);den=math.fsum((float(x)-center)**2 for x in y)
 return dict(n=n,rmse_db=math.sqrt(ss/n),mae_db=math.fsum(abs(x) for x in err)/n,bias_db=math.fsum(err)/n,r2=1-ss/den if den>0 else None)
def json_clean(v):
 if isinstance(v,dict):return {k:json_clean(x) for k,x in v.items()}
 if isinstance(v,list):return [json_clean(x) for x in v]
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,np.integer):return int(v)
 return v

def main():
 need(not (OUT/'summary_complete.json').exists(),'Already summarized')
 old_models=[r['id'] for r in read(OLD/'frozen_selection.json')['models']]+['ours_frozen_top10']
 need(len(old_models)==20,'Historical model count')
 local_old=pd.read_parquet(OLD/'predictions/local_test.parquet')
 uk_old=pd.read_parquet(PHY/'predictions_physical_and_reused.parquet')
 local=local_old.copy();uk=uk_old.copy();done={}
 for model,directory in RUNS.items():
  receipt=read(directory/'complete.json');need(receipt['training_steps']==0 and receipt['finetune_steps']==0 and receipt['optimizer_steps']==0,'Training occurred')
  need(receipt['parameter_sha256_before']==receipt['parameter_sha256_after'] and receipt['original_parameters_exact_match'],'Weights not frozen')
  need(all(receipt['counters'][k]==0 for k in ('gradient_calls','optimizer_constructions','network_calls')),'Forbidden operation')
  need(read(directory/'fixed_settings.json')['parameters']['n_estimators']==8,'Nondefault ensemble')
  for dataset,base,keys in [('local_test',local,['hour_bjt']),('uk_expanded',uk,['station','utc_epoch_seconds'])]:
   fresh=pd.read_parquet(directory/(dataset+'.parquet'))
   chunks=pd.concat([pd.read_parquet(p) for p in sorted(directory.glob(dataset+'_chunk_*.parquet'))],ignore_index=True)
   pd.testing.assert_frame_equal(fresh,chunks)
   need(len(fresh)==len(base) and not fresh.duplicated(keys).any(),'Prediction count/keys')
   joined=base[keys].merge(fresh,on=keys,how='left',validate='one_to_one')
   need(np.isfinite(joined[model]).all(),'Prediction coverage')
   base[model]=joined[model].to_numpy()
  done[model]=receipt
 old_chunks=sorted((OUT/'inference_003/tabpfn_v2_author_frozen').glob('local_test_chunk_*.parquet'))
 replay_check=dict(prior_completed_rows=0,max_abs_difference_db=None)
 if old_chunks:
  prior=pd.concat([pd.read_parquet(path) for path in old_chunks],ignore_index=True)
  pair=prior.merge(local[['hour_bjt','tabpfn_v2_author_frozen']],on='hour_bjt',validate='one_to_one',suffixes=('_old','_new'))
  a=pair.tabpfn_v2_author_frozen_old.to_numpy();b=pair.tabpfn_v2_author_frozen_new.to_numpy()
  need(np.allclose(a,b,rtol=1e-5,atol=1e-5),'Larger-query-chunk replay mismatch')
  replay_check=dict(prior_completed_rows=len(pair),max_abs_difference_db=float(np.max(abs(a-b))))
 pd.testing.assert_frame_equal(local[local_old.columns],local_old)
 pd.testing.assert_frame_equal(uk[uk_old.columns],uk_old)
 raw=pd.read_parquet(ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet',columns=['hour_bjt','clear_sky_attenuation_db'])
 labels=local[['hour_bjt','truth_db']].merge(raw,on='hour_bjt',validate='one_to_one')
 need(np.array_equal(labels.truth_db,labels.clear_sky_attenuation_db),'LOCAL labels changed')
 expanded=pd.read_parquet(EXP/'expanded_dataset.parquet')
 pd.testing.assert_frame_equal(uk[expanded.columns],expanded)
 local['itu838_geometric_proxy']=local['physics_geom_circular']
 local['itu618_hourly_proxy']=local['physics_p618_hourly_circular']
 uk['itu838_geometric_proxy']=uk['itu838_geom_circular_nominal']
 uk['itu618_hourly_proxy']=uk['itu618_proxy_circular_nominal']
 models=old_models+list(RUNS)
 physical=['itu838_geometric_proxy','itu618_hourly_proxy']
 cohorts=[('LOCAL_test',local,np.ones(len(local),dtype=bool),models+physical),
          ('UK_all_eligible',uk,uk.primary_score_eligible.to_numpy(),models),
          ('UK_physical_common',uk,uk.common_score_eligible.to_numpy(),models+physical)]
 need([int(mask.sum()) for _,_,mask,_ in cohorts]==[687,5593,5538],'Main cohort counts')
 for station in ['chilbolton','sparsholt']:
  cohorts.append(('UK_'+station,uk,(uk.primary_score_eligible&uk.station.eq(station)).to_numpy(),models))
 for period in sorted(uk.period.unique()):
  mask=(uk.primary_score_eligible&uk.period.eq(period)).to_numpy()
  if mask.any():cohorts.append(('UK_'+period,uk,mask,models))
 for flag in ['rain','heavy_rain','clear_dry','dry_overcast']:
  if local[flag].any():cohorts.append(('LOCAL_'+flag,local,local[flag].to_numpy(dtype=bool),models+physical))
 for band in sorted(uk.rain_band.dropna().unique()):
  mask=(uk.primary_score_eligible&uk.rain_band.eq(band)).to_numpy()
  if mask.any():cohorts.append(('UK_rain_'+str(band),uk,mask,models))
 rows=[];checks=[]
 for cohort,frame,mask,methods in cohorts:
  y=frame.loc[mask,'truth_db'].to_numpy(dtype=float)
  for model in methods:
   p=frame.loc[mask,model].to_numpy(dtype=float)
   need(np.isfinite(y).all() and np.isfinite(p).all(),'Missing finite value in common cohort')
   a=score(y,p);b=independent_score(y,p)
   for key in a:
    need((a[key] is None and b[key] is None) or math.isclose(a[key],b[key],rel_tol=1e-10,abs_tol=1e-10),'Independent metric mismatch')
   family=re.sub(r'_seed\d+$','',model)
   provenance='author_pretrained_frozen_LOCAL_context' if model in RUNS else ('fixed_physical_proxy_reused' if model in physical else 'historical_frozen_reused')
   rows.append(dict(cohort=cohort,model=model,family=family,provenance=provenance,**a))
   checks.append(dict(cohort=cohort,model=model,independent_metric_match=True))
 metrics=pd.DataFrame(rows)
 family=[]
 for (cohort,name),g in metrics.groupby(['cohort','family'],sort=True):
  row=dict(cohort=cohort,family=name,n=int(g.n.iloc[0]),n_runs=len(g),provenance=g.provenance.iloc[0])
  for key in ('rmse_db','mae_db','bias_db','r2'):
   row[key+'_mean']=float(g[key].mean())
   row[key+'_sd']=float(g[key].std(ddof=1)) if len(g)>1 else None
  family.append(row)
 families=pd.DataFrame(family)
 local.to_parquet(OUT/'predictions_local_all.parquet',index=False)
 uk.to_parquet(OUT/'predictions_uk_all.parquet',index=False)
 metrics.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
 families.to_csv(OUT/'family_summary.csv',index=False,encoding='utf-8-sig')
 emit(OUT/'metrics.json',json_clean(rows));emit(OUT/'family_summary.json',json_clean(family))
 before=read(OUT/'source_sha256_before.json')
 mismatches=[p for p,v in before.items() if sha(ROOT/p)!=v]
 need(not mismatches,'Protected sources changed')
 emit(OUT/'independent_qa.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_count=len(before),source_mismatches=mismatches,
       local_historical_predictions_exact=True,uk_historical_predictions_exact=True,all_labels_exact=True,
       chunk_files_equal_final=True,metric_checks=checks,new_runs=done,
       author_model_parameters_unchanged=True,new_parameter_training_steps=0,uk_context_rows=0,test_context_rows=0,
       tabpfn_prior_chunk_replay=replay_check))
 completion=dict(utc=datetime.now(timezone.utc).isoformat(),new_models=list(RUNS),
       original_author_checkpoints_restored_for_old_models=0,historical_checkpoints_preserved=18,
       metrics=len(metrics),family_summaries=len(families),new_model_predictions=2*(687+5664),
       inference_query_chunk_sizes={'tabpfn_v2_author_frozen':2048,'tabicl_v2_author_frozen':512},
       sources_unchanged=len(before),weights_updated=0,parameter_training_steps=0,finetuning_steps=0,
       local_hours=687,uk_ml_hours=5593,uk_physical_common_hours=5538,
       execution='complete',verification='PASS',t012_overall_complete=False)
 report=['# 新增冻结作者模型：LOCAL与扩展英国小时比较','',
   '本轮新增TabPFN-v2与TabICLv2两项作者预训练回归对照。没有网络参数训练、微调、优化器步骤、种子选择或根据英国结果改参数；所有旧预测逐值复用。作者原始权重和推理前后网络参数完全一致。LOCAL上下文和作者默认推理变换的准备与网络训练分开记录。',
   '', 'Built with PriorLabs-TabPFN。TabPFN-v2见[论文](https://doi.org/10.1038/s41586-024-08328-6)与[权重](https://huggingface.co/Prior-Labs/TabPFN-v2-reg)；TabICLv2见[论文](https://arxiv.org/abs/2602.11139)与[权重](https://huggingface.co/jingang/TabICL)。',
   '', '## 同样本主要结果','',
   'RMSE单位dB。UK5593为所有气象/标签合格小时；UK5538为另加物理输入合格门槛的共同小时。各列只在列内比较，不把5538与5593混排。带±的数字是旧三种子均值±样本SD，不是置信区间；新两项使用作者默认单次设置、内部8成员推理集成。','',
   '| 模型 | LOCAL687 | UK5593 | UK5538 |','|---|---:|---:|---:|']
 names={'ours_frozen_top10':'本文冻结Top10','tabpfn_v2_author_frozen':'TabPFN-v2（作者原权重）','tabicl_v2_author_frozen':'TabICLv2（作者原权重）',
        'tabm2025_official_budgeted':'TabM（此前LOCAL训练适配版）','bai2022_mlp_adapted':'Bai2022 MLP适配版','bai2022_lstm_t0_adapted':'Bai2022零历史LSTM适配版',
        'itu838_geometric_proxy':'P.838几何雨衰近似','itu618_hourly_proxy':'P.618小时代理','train_mean':'LOCAL训练均值常数'}
 ordered=['ours_frozen_top10',*RUNS,'tabm2025_official_budgeted','bai2022_mlp_adapted','bai2022_lstm_t0_adapted','ridge','elastic_net','svr_rbf','random_forest','xgboost','train_mean',*physical]
 for name in ordered:
  values=[]
  for cohort in ['LOCAL_test','UK_all_eligible','UK_physical_common']:
   found=families[(families.cohort==cohort)&(families.family==name)]
   if len(found)==0:values.append('—');continue
   row=found.iloc[0];v=f"{row.rmse_db_mean:.4f}"
   if row.n_runs>1:v+=f" ± {row.rmse_db_sd:.4f}"
   values.append(v)
  report.append('| '+names.get(name,name)+' | '+' | '.join(values)+' |')
 report.extend(['','## 输入、上下文与冻结边界','',
   '同一小时十列气象输入，不使用历史信道测量。LOCAL原2404训练小时为两项新基础模型的上下文；343验证小时只取前16小时做形状/分块一致性检查，不用于选参。LOCAL687测试标签与英国标签仅用于评分。英国完整气象5664小时均产生预测，其中5593小时符合既有标签质量门槛；71个不合格标签小时保留，不评分。',
   '作者接口中的fit只准备上下文和默认推理变换，没有参数训练；LOCAL训练标签仍作为上下文提供，不能把它称为“完全不使用本地标签”。网络的外部预训练资源与历史LOCAL训练方法不同，来源分组披露，不宣称训练预算完全相同。',
   'TabPFN使用官方2.0.9及v2默认回归权重，TabICL使用官方2.2.0及v2-20260212回归权重；网络结构/权重、默认8集成和各自默认随机状态不变。依赖隔离安装在新目录。CPU线程/文件路径/禁联网为执行设置；TabPFN保留作者默认内存管理，通过inference_mode与硬性梯度/优化器拦截保持权重冻结。首次不兼容试运行和高内存试运行保留在CHANGE_RECORD。',
   'TabPFN小批16与8+8预测差异仅1.9521e-6dB；因CPU重复上下文计算较慢，v004只将外部查询分块512增至2048，并原样重放验证检查。模型所有get_params设置、上下文和网络参数都与v003相同；此执行改动发生在查看新模型测试指标之前。TabICL仍用512分块。',
   '', '## 旧模型参数是否恢复','',
   '18个旧学习模型检查点已核对，均保持原样；没有改参数、重新初始化或重新训练。Bai和TabM此前是我们训练的适配版，不是作者提供的原权重。未取得兼容的作者原检查点，因此本轮旧模型“恢复作者权重”数量为0。尤其TabM的256宽度不能在旧权重不变时改成512；库默认超参数不是作者已训练参数。旧结果不删除、不重新命名为作者原版。',
   'FT-Transformer、RealMLP、TabR、TabNet、NODE、CatBoost在本轮有界官方来源检查中未确认可直接用于本任务的通用预训练回归权重，因此没有为凑数训练或伪造评分。详见[来源与恢复审计](MODEL_PROVENANCE.md)。',
   '', '## 核验与解释','',
   f'独立指标复算{len(metrics)}行、家族汇总{len(families)}行；{len(before)}项保护源SHA不变。新预测键、完整分块拼接、标签、旧预测逐值一致核验通过。每项原作者学习参数逐值匹配，网络参数前后SHA相同，反向传播/优化器构造/推理网络请求计数均0。',
   '这些结果是既有已暴露测试集上的新增比较，不是新的盲测。英国两站共用气象且原生日期有缺口，不把站点小时当独立站点/气候复制。基础模型是通用气象表格回归对照，不冒充原生卫星信道论文的完整复现；ITU仍是单列近似物理分量，不宣称等同完整小时衰减机制。',
   '本轮未恢复气象下载；原停止标志保留。模型训练0、微调0、原模型重新推理0；新作者模型冻结推理已完成，不等于整个T012历史任务自动完成。',
   '', '## 文件','',
   '[全部指标](metrics.csv) · [家族汇总](family_summary.csv) · [本地逐小时预测](predictions_local_all.parquet) · [英国逐小时预测](predictions_uk_all.parquet) · [独立核验](independent_qa.json) · [完成收据](summary_complete.json)'])
 with (OUT/'COMPARISON_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
 emit(OUT/'summary_complete.json',completion)
 print(families[families.cohort.isin(['LOCAL_test','UK_all_eligible','UK_physical_common'])][['cohort','family','n','n_runs','rmse_db_mean','rmse_db_sd']].to_string(index=False),flush=True)

if __name__=='__main__':main()
