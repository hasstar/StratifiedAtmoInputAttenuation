"""Independent, read-only checks of the new manuscript package.
Uses standard-library CSV/math/ZIP, not scientific training/evaluation imports.
Writes one report in the new package only. Visual assessment is a separate record.
"""
from pathlib import Path
from collections import Counter
import csv,math,json,hashlib,argparse,zipfile,re,datetime,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(v):return math.fsum(v)/len(v)
def close(a,b,atol=1e-10):return math.isclose(float(a),float(b),rel_tol=0,abs_tol=atol)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=ROOT/'最终提交的修改材料/论文图表_20261030_001');ap.add_argument('--skip-originals',action='store_true');ap.add_argument('--render-replay',action='store_true');a=ap.parse_args();p=a.package;q=p/'05_QA';d=p/'04_Source_Data';s=d/'frozen_exports';checks={}
 csv.field_size_limit(20*1024*1024)
 source=read(q/'SOURCE_MANIFEST.json');upstream=read(q/'UPSTREAM_CODE_MANIFEST.json');copied=0
 for record in source+upstream:
  if not a.skip_originals:assert sha(record['original_path'])==record['sha256'],record['original_path']
  if record.get('snapshot'):assert sha(p/record['snapshot'])==record['sha256'],record['snapshot'];copied+=1
 checks['source_integrity']={'source_records':len(source),'upstream_code_records':len(upstream),'matching_snapshots':copied,'originals_checked':not a.skip_originals}
 models={'tabm','bilstm','mlp','gcn','svr_rbf','bai_lstm','transformer','catboost','random_forest','elastic_net'}
 local=[r for r in rows(s/'local_metrics.csv') if r['population']=='ML_687' and r['model'] in models]
 assert len(local)==7800
 assert len({(r['model'],int(r['k']),r['weather']) for r in local})==7800
 counts=Counter(r['model'] for r in local);assert all(v==780 for v in counts.values())
 for r in local:assert int(r['n'])=={'overall':687,'rain':51,'dry_overcast':158,'clear_dry':322,'other':156}[r['weather']]
 lk10=[r for r in local if int(r['k'])==10 and r['weather']=='overall'];best=min(lk10,key=lambda r:float(r['rmse_db']));assert best['model']=='tabm'
 states=rows(s/'local_states.csv');assert len(states)==1560
 checks['local_rfe']={'states':len(states),'weather_metric_rows':len(local),'local_k10_lowest_rmse_model':best['model'],'tabm_rmse_db':float(best['rmse_db'])}
 uk=rows(s/'uk_metrics.csv');assert len(uk)==1200 and len({(r['model'],r['k'],r['station'],r['weather']) for r in uk})==1200
 assert all(int(r['training_calls'])==0 for r in uk)
 uoverall=[r for r in uk if r['station']=='ALL' and r['weather']=='overall'];assert len(uoverall)==100 and all(float(r['r2'])<0 for r in uoverall)
 for r in uk:
  if r['weather']=='overall':assert int(r['n'])=={'ALL':5593,'chilbolton':3382,'sparsholt':2211}[r['station']]
 checks['uk_frozen']={'states':100,'weather_station_rows':1200,'all_pooled_overall_r2_negative':True,'training_calls':0}
 itu=rows(s/'itu_predictions.csv');truth=[float(r['truth_db']) for r in itu];error=[float(r['prediction_db'])-float(r['truth_db']) for r in itu];rmse=math.sqrt(mean([v*v for v in error]));center=mean(truth);r2=1-math.fsum(v*v for v in error)/math.fsum((v-center)**2 for v in truth)
 assert len(itu)==684 and abs(rmse-1.008963)<1e-6 and abs(r2-.028100)<1e-6
 checks['itu']={'n':len(itu),'rmse_db':rmse,'r2':r2,'definition':'1-SSE/SST'}
 comp=rows(d/'table_data/All_RFE_Comparison_Values.csv');assert len(comp)==8388
 for r in comp:
  t,b=float(r['tabm_rmse']),float(r['baseline_rmse']);assert close(r['rmse_reduction_percent'],100*(b-t)/b)
  assert close(r['delta_r2'],float(r['tabm_r2'])-float(r['baseline_r2']))
 checks['comparison_formula_rows']=len(comp)
 feature=rows(s/'tabm_retained_features.csv');f10=[r for r in feature if int(r['k'])==10];assert len(f10)==20
 assert sum(r['pool']=='all156' and not r['feature_name'].startswith('surface__') for r in f10)==5
 checks['independent_k10_feature_sets']={'all156':10,'base50':10,'pressure_features_in_all156_k10':5}
 hourly=rows(s/'fast_hourly.csv');summary=rows(s/'fast_summary.csv');assert len(hourly)==6623 and len(summary)==25
 scorecols=['fast_crps_db','fast_zero_crps_db','combined_crps_db','slow_only_crps_db','combined_minus_slow_db','observed_sd_db','generated_sd_mean_db','sd_absolute_error_db','generated_mean_q','generated_drift_db_mean','combined_coverage90']
 for r in summary:
  g=[h for h in hourly if (('LOCAL_'+h['split']==r['cohort'] and h['dataset']=='LOCAL') or (r['cohort']=='UK' and h['dataset']=='UK')) and (r['station']=='pooled' or h['station']==r['station']) and (r['event']=='overall' or h['event']==r['event'])]
  assert len(g)==int(r['hours']) and sum(int(h['n_seconds']) for h in g)==int(r['valid_seconds'])
  for c in scorecols:assert close(mean([float(h[c]) for h in g]),r[c]),(r['cohort'],r['event'],c)
  assert close(float(r['combined_crps_db'])-float(r['slow_only_crps_db']),r['combined_minus_slow_db'])
  assert close(100*(1-float(r['combined_crps_db'])/float(r['slow_only_crps_db'])),r['combined_improvement_percent'])
  assert float(r['delta_cluster_ci95_low'])<=float(r['delta_cluster_ci95_high'])
 checks['fast003']={'hour_files':len(hourly),'summary_rows':len(summary),'valid_seconds':sum(int(h['n_seconds']) for h in hourly),'checked_hour_equal_mean_columns':scorecols,'bootstrap_intervals':'Reused hash-pinned original intervals; not newly resampled.'}
 dense=read(q/'dense_acf_qa.json');assert dense['status']=='PASS' and dense['hours']==6623 and dense['aggregate_275_old_rows_match']
 acf=rows(d/'Fig10_dense_acf_1_to_1800.csv');assert len(acf)==45000
 assert len({(r['cohort'],r['station'],r['event'],r['lag_seconds']) for r in acf})==45000
 old=rows(s/'fast_acf.csv');index={(r['cohort'],r['station'],r['event'],r['lag_seconds']):r for r in acf}
 for r in old:
  t=index[(r['cohort'],r['station'],r['event'],r['lag_seconds'])]
  for c in ['observed_acf','generated_acf_mean','acf_mae']:assert close(r[c],t[c],1e-9)
 plotacf=[r for r in acf if (r['cohort'],r['station']) in [('LOCAL_test','local'),('UK','chilbolton'),('UK','sparsholt')] and r['event']!='overall']
 for r in plotacf:
  for c in ['observed_acf','generated_acf_mean']:assert -.12<=float(r[c])<=.62
 checks['dense_acf']=dense
 duration=read(q/'duration_distribution_qa.json');assert duration.get('status')=='PASS'
 checks['duration_distributions']=duration
 receipts=read(q/'Fast003_hour_receipts.json');assert len(receipts)==6623
 figures=sorted((p/'01_Figures_PDF').glob('Fig*.pdf'));assert len(figures)==14
 inventory=read(q/'pdf_inventory.json');assert len(inventory)==14
 for r in inventory:assert r['pages']==1 and sha(p/r['file'])==r['sha256'] and r['text_characters']>20
 tins=read(q/'table_inventory.json');assert len(tins)==14 and len(list((p/'02_Tables_LaTeX').glob('Table*.tex')))==14
 log=(q/'latex_build/main.log').read_text(encoding='utf8',errors='replace');assert '15 pages' in log
 assert not any(x in log for x in ['Overfull','Underfull','LaTeX Warning','! LaTeX Error','Missing character']), 'LaTeX diagnostic requires inspection'
 ppt=p/'03_Editable_PPTX/Fig02_Fig05_Fig09_Editable.pptx';validation=read(q/'ppt_build/validation_final.json');assert sha(ppt)==validation['finalSha256']
 assert validation['packageIntegrity']['status']=='pass' and validation['presentationLayout']['finding_count']==0
 with zipfile.ZipFile(ppt) as z:
  slides=[x for x in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',x)];assert len(slides)==3
  native=[];ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
  for sn in slides:
   root=ET.fromstring(z.read(sn));native.append({'slide':sn,'shapes':len(root.findall('.//p:sp',ns)),'text_runs':len(root.findall('.//a:t',ns)),'pictures':len(root.findall('.//p:pic',ns))})
  assert all(r['shapes']>=10 and r['text_runs']>10 and r['pictures']==0 for r in native)
 checks['artifacts']={'individual_figure_pdfs':14,'main_figures':10,'supplementary_figures':4,'main_table_numbers':9,'latex_fragments':14,'table_preview_pages':15,'editable_pptx':native,'native_powerpoint_gui_rendering':False}
 if not a.render_replay:
  cross=read(q/'ASSET_CROSSWALK.json');assert len(cross)==28
  for r in cross:
   for f in r['outputs']:assert (p/f).is_file(),f
   assert (p/'06_Generation_Code'/r['generator']).is_file()
   for f in r['inputs']:assert (d/f).exists(),f
  for fn in ['00_从这里开始.md','01_图表代码与数据溯源.md','02_双语图注.md','03_检查与口径说明.md','图表说明与双语图注.docx','00_表格预览.pdf']:assert (p/fn).is_file(),fn
  visual=read(q/'VISUAL_REVIEW.json')
  for r in visual['artifacts']:assert sha(p/r['file'])==r['sha256'] and r['visually_reviewed']
  checks['visual_review']={'reviewed_artifacts':len(visual['artifacts']),'record':'VISUAL_REVIEW.json'}
 report={'status':'PASS','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Frozen source integrity, independent numerical aggregation and rendering-package checks; no fitting or new ensemble generation','checks':checks,'visual_scope':'Original PDF/LaTeX views reviewed separately' if not a.render_replay else 'Reproduced output requires a new visual review','verifier_sha256':sha(Path(__file__))}
 (q/'FINAL_QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'status':'PASS','figures':14,'table_fragments':14,'source_snapshots':copied,'fast_hours':len(hourly),'acf_rows':len(acf)},ensure_ascii=False))
if __name__=='__main__':main()
