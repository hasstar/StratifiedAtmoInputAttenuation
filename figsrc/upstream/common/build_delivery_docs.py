"""Write the manuscript asset guide, bilingual captions and source crosswalk.
This script documents existing evidence; it never imports scientific workers.
"""
from pathlib import Path
import argparse,json,hashlib,shutil,sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
FIGS=[
('Fig01_Measurement_Campaign','1','现场与采集装置','Measurement campaign', 'Section II.A',
 'Measurement campaign: (a) outdoor receiving installation and (b) indoor acquisition hardware. The analysis uses 100-ms power records aggregated to 1-s and hourly resolutions.',
 '测量现场与采集装置：（a）室外接收设备；（b）室内采集硬件。当前分析将100毫秒功率记录汇总为秒级和小时级数据。'),
('Fig02_Stratified_Atmospheric_Information','2','垂直分辨气象信息','Vertically resolved meteorological information','Section II.A–C',
 'Meteorological input representation. The 156-feature pool contains 50 base features and 106 variable–pressure-level features. The base pool includes surface variables, single-level diagnostics and column-integrated quantities. Pressure layers and the propagation path are schematic; layer spacing does not represent geometric altitude.',
 '气象输入表示：156维候选池包含50项基础特征和106项变量—压力层特征。基础池包含地表变量、单层诊断量及柱积分量。压力层和传播路径为示意，层间距离不表示实际几何高度。'),
('Fig03_Hourly_And_Second_Power','3','小时功率与秒级细节','Hourly and second-scale measured power','Section II.A–B',
 'Measured power at two temporal scales: (a) all quality-screened local hours, (b) the hourly context of a randomly selected complete hour and (c) its 1-s power series. Weather is marked by colored bands. Selection uses seed 20261030 and gives 15 July 2026, 23:00 Beijing time. Missing recording intervals remain gaps; the dashed mean is computed in linear power.',
 '双时间尺度实测功率：（a）质量筛选后的全部本地小时；（b）随机选中完整小时的相邻小时背景；（c）该小时的秒级功率。色带表示天气。固定随机种子20261030选中北京时间2026年7月15日23时；缺测不连线，均值在功率线性域计算。'),
('Fig04_ITU_Hourly_Attenuation','4','ITU小时衰减对比','ITU hourly attenuation comparison','Section III.B',
 'Hourly attenuation estimated by the ITU-based gas–cloud–rain combination on the 684 common valid local test hours: (a) estimate versus measurement and (b) signed error versus measurement. The dashed lines indicate perfect agreement and zero error, respectively. RMSE is 1.0090 dB and R² is 0.0281.',
 'ITU气体—云—雨组合模型在共同有效的684个本地测试小时上的小时衰减结果：（a）估计值与实测值；（b）有符号误差与实测值。虚线分别表示完全一致和零误差；RMSE为1.0090 dB，R²为0.0281。'),
('Fig05_TabM_Architecture_And_Protocol','5','TabM结构和任务适配','TabM architecture and task-specific protocol','Section III.A',
 'TabM hourly attenuation model and task-specific protocol. Training-fitted preprocessing feeds the inherited parameter-efficient ensemble backbone. The 32 member outputs are averaged and transformed back to dB. Independent feature-elimination paths, weather-aware validation and frozen external evaluation specify its use in this study.',
 'TabM小时衰减模型及任务适配流程。训练集拟合的预处理连接原有参数高效集成骨干，32个成员的输出取平均并还原至dB。独立特征消除路径、天气感知验证准则和外站冻结评价规定其在本文中的使用方式。'),
('Fig06_Ten_Model_RFE','6','十模型完整RFE路径','Ten-model recursive feature elimination','Section III.B.1',
 'Local test performance of ten models along their independent 156-to-1 feature-elimination paths. Upper panels show all retained-feature counts; lower panels enlarge k = 1–10. All states use the same 687 test hours. TabM is highlighted. Feature elimination proceeds from right to left.',
 '十个模型沿各自156→1特征消除路径的本地测试性能。上排显示全部保留特征数，下排放大k＝1–10。各状态均使用同一687小时测试集，TabM用加粗绿色曲线突出，消除方向由右向左。'),
('Fig07_TabM_Independent_RFE_Paths','7','TabM两条候选路径比较','TabM comparison at equal input dimensions','Section III.B.2',
 'TabM performance at equal retained-feature counts for two independently eliminated candidate pools: all 156 features and 50 non-vertically-resolved base features. The upper panels compare k = 1–50; the lower panels enlarge k = 1–10. Both paths are evaluated on the same 687 local test hours.',
 'TabM两条独立候选路径在相同保留特征数下的比较：156维全量候选池与50维非垂直分辨基础池。上排为k＝1–50，下排放大k＝1–10；均在同一687小时本地测试集上评价。'),
('Fig08_UK_Frozen_Model_Comparison','8','英国外站冻结评价','Frozen evaluation at the UK sites','Section III.C',
 'UK evaluation of 100 frozen local model states, comprising ten models at k = 1–10. Rows show the pooled sites, Chilbolton and Sparsholt. No UK retraining or preprocessing refitting is performed. R² uses an explicitly labeled symmetric-log axis to display its complete negative range; RMSE uses a linear axis.',
 '100组本地冻结模型状态在英国的直接评价，包括十模型各k＝1–10。三行依次为两站合并、Chilbolton和Sparsholt。英国数据不参与训练或预处理重拟合；R²采用明确标注的对称对数坐标完整保留负值，RMSE采用线性坐标。'),
('Fig09_Second_Scale_Generation_Procedure','9','快变生成流程','Second-scale generation procedure','Section IV.A',
 'Weather-conditioned second-scale generation. Independent draws from a frozen empirical power-ratio distribution supply amplitudes, while concatenated measured 256-s blocks supply a rank template. Rank reordering, a random circular shift and conversion F = −10 log10(Q) produce one realization. The process is repeated independently 32 times per evaluated hour.',
 '天气条件化秒级生成流程。冻结功率比经验分布的独立抽样提供幅度，实测256秒连续块拼接提供秩模板。经过秩重排、随机循环平移和F＝−10log10(Q)转换得到一次实现；每个评价小时独立生成32条曲线。'),
('Fig10_Fast_Temporal_Statistics','10','实测与生成时间结构','Observed and generated temporal statistics','Section IV.B',
 'Observed and generated temporal statistics for local test, Chilbolton and Sparsholt. Left: mean autocorrelation at each integer lag from 1 to 1800 s. Right: complete-fade duration CDFs above weather-specific F95 thresholds fixed from local training data. Colors denote weather and line styles denote observations or generated realizations. ACF pairs never bridge missing intervals; duration distributions equally weight paired hours, with boundary-censored events excluded.',
 '本地测试、Chilbolton和Sparsholt的实测与生成时间结构。左列为1–1800秒每个整数滞后的平均ACF；右列为超过本地训练集固定天气阈值F95的完整衰落持续时间CDF。颜色表示天气，线型表示实测或生成。ACF不跨缺测区间取对；持续时间按配对小时等权，排除小时边界和缺测边界截断事件。'),
('FigS01_Combined_Second_Scale_Examples','S1','慢快组合秒级实例','Combined second-scale examples','Supplement / Section IV.C',
 'Illustrative hourly–second-scale combination for the earliest rainy hour in each evaluated cohort. Observed attenuation, the frozen TabM hourly estimate, ensemble median and 5–95% generated quantiles are shown. Examples are selected by timestamp, not agreement. The shaded interval describes generated ensemble spread, not a calibrated confidence interval.',
 '每个评价范围中时间最早雨天小时的慢快组合示例，展示实测衰减、冻结TabM小时估计、生成中位数及5%–95%分位区间。示例按时间选择，不按拟合好坏筛选。阴影表示生成集合范围，不表示已校准的置信区间。'),
('FigS02_Local_RFE_By_Weather','S2','本地分天气RFE','Local RFE by weather','Supplement / Section III.B',
 'Weather-resolved local test performance along all ten independent feature-elimination paths. Rain, overcast and overall results are displayed separately. All retained-feature counts and negative R² values remain visible.',
 '十模型各自完整消除路径的本地分天气测试结果，分别展示雨天、阴天和总体，保留全部状态及负R²。'),
('FigS03_UK_RFE_By_Weather','S3','英国分天气冻结评价','UK frozen evaluation by weather','Supplement / Section III.C',
 'Weather-resolved pooled UK performance of the 100 frozen model states. Rain, overcast and overall results are shown separately, without UK refitting. Symmetric-log R² axes are labeled where used.',
 '英国两站合并的100组冻结状态分天气结果，分别展示雨天、阴天和总体，不在英国重新拟合；使用对称对数的R²坐标均已标注。'),
('FigS04_F99_Fade_Duration_Distributions','S4','较深衰落持续时间','F99 fade-duration distributions','Supplement / Section IV.B',
 'Complete-fade duration distributions above weather-specific F99 thresholds fixed from local training data. Observations and generated realizations use the same censoring and equal-paired-hour aggregation as Fig. 10.',
 '超过本地训练集固定天气阈值F99的完整衰落持续时间分布。实测和生成采用与图10相同的截断事件处理及配对小时等权汇总。')]

TABLE_ZH={
'Table01_Measurement_Configuration':('测量配置与分析分辨率','区分原始IQ采样设置、100毫秒功率记录、秒级分析及小时汇总；列出当前训练集确定的晴空参考。'),
'Table02_Dataset_Overview':('数据集划分、日期与天气构成','列出本地训练/验证/测试、英国两站及合并范围的小时数、有效日期数、日期覆盖和四类互斥天气数量。'),
'Table03_Training_And_RFE_Protocol':('模型配置和训练评价协议','列出冻结配置、正式训练轮数/树数、权重策略及共同RFE/预处理/验证规则；完整配置见配套JSON。'),
'Table04_Local_Ten_Model_Comparison':('本地测试中TabM与其他九模型的比较','按天气及代表性k列出TabM的绝对指标、RMSE相对降低率与R²绝对差；正值有利于TabM，负值有利于对照。全部156状态数值另附CSV。'),
'Table05_TabM_Independent_Pool_Comparison':('TabM两条独立路径的等维比较','在相同k和相同本地测试小时上比较两候选池的绝对RMSE、R²及差值；下标156和50指起始候选池，非当前输入维数。'),
'Table06_UK_Frozen_Model_Comparison':('英国两站合并的冻结模型比较','按天气及k＝10、5、1列出TabM绝对指标和对各模型的有符号差值；其他天气沿用小时级原定义，包含晴朗干燥。'),
'Table07_Block_Length_Development':('训练内块长度开发比较','展示14个候选方案，包括不同块长、IID和高斯AR(1)，指标为开发期16条实现VS-1及8条稳定性检查。256秒为所存结果的最低分；这不替代最新32条曲线正式评价。'),
'Table08a_Fast_Amplitude_And_SD':('快变幅度评分与波动强度','各数据范围及天气分别给出32条曲线逐秒CRPS、零快变基线CRPS、实测与生成标准差及逐实现标准差绝对误差；先小时内计算，再小时等权。'),
'Table08b_Fast_ACF_And_Duration':('快变自相关与衰落持续时间','给出实测/生成60秒ACF，F95及F99完整事件平均持续时间和相应有效/配对小时数。完整曲线见图10和补图S4。'),
'Table09_Hourly_Second_Scale_CRPS':('小时—秒级组合的CRPS改善','相同有效秒上比较冻结TabM小时点估计和慢快组合的CRPS，报告相对降低率、配对差值及按日期聚类的95%区间。'),
'TableS01_TabM_K10_Retained_Features':('两条TabM路径的k＝10特征集合','给出每项气象变量名称及压力层/基础类型。顺序是模型输入顺序，不是重要性排名。'),
'TableS02_UK_K10_Absolute_Metrics':('英国k＝10绝对指标','保留两站分别及合并、每模型、每天气的绝对RMSE、MAE、Bias及R²，共120行，避免仅以相对变化判断性能。'),
'TableS03_Combination_By_Weather':('慢快组合分天气评分','补充各数据范围/天气的绝对CRPS、降低率、配对区间及生成区间覆盖率；覆盖率是经验诊断，不宣称分布已校准。'),
'TableS04_Valid_Seconds_And_Generator_Means':('有效秒数与生成均值诊断','列出有效秒数、功率比均值及小时均值漂移。生成实现未强制归一化成平均功率比1或平均dB快变0。')}

UPSTREAM={
'小时级十模型RFE': ['Code/Hourly_Attenuation_Experiments/Revision_032/run.py','Code/Hourly_Attenuation_Experiments/Revision_032/runtime.py','Code/Hourly_Attenuation_Experiments/Revision_034/export_audit_v2.py'],
'TabM基础50维独立RFE': ['Code/Hourly_Attenuation_Experiments/Revision_041/prepare_surface.py','Code/Hourly_Attenuation_Experiments/Revision_041/run_surface.py','Code/Hourly_Attenuation_Experiments/Revision_041/runtime.py','Code/Hourly_Attenuation_Experiments/Revision_041/report_surface.py'],
'英国冻结推理及指标': ['Code/Hourly_Attenuation_Experiments/Revision_037/infer_frozen.py','Code/Hourly_Attenuation_Experiments/Revision_037/audit_metrics.py','Code/Hourly_Attenuation_Experiments/Revision_037/write_results.py'],
'快变冻结资产与开发选择': ['Code/Fast_Fading_Experiments/Revision_002/freeze_marginal.py','Code/Fast_Fading_Experiments/Revision_002/freeze_generator.py','Code/Fast_Fading_Experiments/Revision_002/run_dynamics.py'],
'最新32曲线快变和慢快组合评价': ['Code/Fast_Fading_Experiments/Revision_003/evaluate.py','Code/Fast_Fading_Experiments/Revision_003/common.py','Code/Fast_Fading_Experiments/Revision_003/summarize_and_plot.py','Code/Fast_Fading_Experiments/Revision_003/independent_qa.py'],
'ITU物理分量及合法样本': ['Code/Combined_Channel_Experiments/Revision_011/physical.py','Code/Combined_Channel_Experiments/Revision_011/calculate.py','Code/Combined_Channel_Experiments/Revision_027/supplement.py']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def link(p,label=None):return '['+(label or p.name)+'](<'+p.as_posix()+'>)'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=ROOT/'最终提交的修改材料/论文图表_20261030_001');ap.add_argument('--project',type=Path,default=ROOT);a=ap.parse_args();out=a.package.resolve();project=a.project.resolve();qa=out/'05_QA'
 manifests=json.loads((qa/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
 plot=json.loads((qa/'plot_inventory.json').read_text(encoding='utf8'));tables=json.loads((qa/'table_inventory.json').read_text(encoding='utf8'))
 upstream=[]
 for group,files in UPSTREAM.items():
  for rel in files:
   p=project/rel;assert p.is_file(),str(p);dest=out/'04_Source_Data/upstream_code'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert sha(dest)==sha(p)
   else:shutil.copy2(p,dest)
   upstream.append(dict(group=group,original_path=str(p),snapshot=dest.relative_to(out).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
 dump(qa/'UPSTREAM_CODE_MANIFEST.json',upstream)
 # Skill-owned document helpers are copied as named provenance-bearing utilities.
 helper=Path('C:/Users/SDR_HH/.codex/skills/paper-figures/scripts');vendor=HERE/'report_helpers';vendor.mkdir(exist_ok=True)
 for fn in ['report_docx.py','docx_tables.py']:
  f=helper/fn;d=vendor/fn
  if not d.exists():shutil.copy2(f,d)
  assert sha(d)==sha(f)
 sys.path.insert(0,str(vendor));from report_docx import build_report
 items=[];assets=[];captions=['# 图表双语图注','', '图内统一英文。以下文字为可编辑图注草稿，按最终正文首次引用顺序统一编号。','']
 byid={r['id']:r for r in plot}
 diagram_inputs={'2':['methods/01_PROTOCOL.md','methods/02_PROTOCOL.md','Table03_model_configs.json'],'5':['Table03_model_configs.json','methods/02_PROTOCOL.md','methods/05_PROTOCOL.md'],'9':['methods/06_PROTOCOL.md','methods/07_evaluate.py','methods/08_common.py','methods/11_DYNAMICS_PLAN.md']}
 for fid,num,zh,en,section,capen,capzh in FIGS:
  generator='build_diagrams.mjs' if num in ['2','5','9'] else 'build_plots.py'
  inputs=diagram_inputs.get(num,byid.get(fid,{}).get('inputs',[]));method=byid.get(fid,{}).get('method',capen)
  assets.append(dict(kind='figure',id=fid,number=num,title_zh=zh,title_en=en,section=section,outputs=['01_Figures_PDF/'+fid+'.pdf'],generator=generator,inputs=inputs,method=method))
  captions+=['## Figure '+num+' — '+zh,'',capen,'',capzh,'']
  items.append(dict(kind='figure',number=num,title={'en':en,'zh':zh},image='05_QA/previews/'+fid+'.png',width_in=6,claim=zh,caption={'en':capen,'zh':capzh},citation=section,repro='06_Generation_Code/'+generator+' → 01_Figures_PDF/'+fid+'.pdf; data: '+', '.join(inputs)))
 for r in tables:
  zh,note=TABLE_ZH[r['id']];num=r['number'];en=r['caption']
  assets.append(dict(kind='table',id=r['id'],number=num,title_zh=zh,title_en=en,outputs=r['files'],generator='build_tables.py',inputs=r['inputs'],method=r.get('note',note)))
  captions+=['## Table '+num+' — '+zh,'',en,'',note,'']
  items.append(dict(kind='table',number=num,title={'en':en,'zh':zh},claim=note,caption={'en':en,'zh':note},repro='06_Generation_Code/build_tables.py → '+r['files'][0]+'; data: '+', '.join(r['inputs'])))
 dump(qa/'ASSET_CROSSWALK.json',assets)
 (out/'02_双语图注.md').write_text('\n'.join(captions),encoding='utf8')
 build_report(out/'图表说明与双语图注.docx',title='Manuscript figures and tables / 论文图表与双语图注',subtitle='10 main figures + 4 supplementary figures; 9 main tables + 4 supplementary tables',meta=[('Date','2026-10-30'),('Evidence','Frozen local/UK hourly models and latest Fast003 32-realization evaluation'),('Editing','Diagrams: native editable PPTX. Tables: LaTeX. Numeric plots: Python.'),('Scope','Source/result crosswalk and SHA256 receipts supplied. Captions are editable author-review drafts.')],items=items,base_dir=out,lang='bilingual',body_font='Arial')
 from docx import Document
 document=Document(out/'图表说明与双语图注.docx')
 for row,asset in zip(document.tables[0].rows[1:],assets):row.cells[1].text=Path(asset['outputs'][0]).name
 document.save(out/'图表说明与双语图注.docx')
 intro='''# 从这里开始：最终提纲图表

本目录包含 **10幅主图＋4幅补图、9张主表＋4张补表**。所有图片均提供PDF；图2、图5、图9另提供同一个3页可编辑PPTX。表8为可读性拆成8(a)、8(b)，仍属于一张主表。

## 最快查阅

- [全部图片预览](00_全部图片预览.pdf)：按1—10、S1—S4排列，含PDF书签。
- [全部表格预览](00_表格预览.pdf)：已编译的15页检查版；投稿使用下面的LaTeX源码。
- [逐张图表的代码和数据来源](01_图表代码与数据溯源.md)。
- [中英文图注](02_双语图注.md)；另有[Word版说明与图注](图表说明与双语图注.docx)。
- [三个示意图的可编辑PPTX](03_Editable_PPTX/Fig02_Fig05_Fig09_Editable.pptx)：第1/2/3页分别对应图2/5/9；文字、框、箭头均可编辑。
- [LaTeX总入口](02_Tables_LaTeX/main.tex)：用于独立检查；正文引用各Table*.tex片段。需booktabs、graphicx、longtable、array等包，清单见main.tex。
- [生成代码和一键重绘](06_Generation_Code/reproduce.ps1)：要求新建输出目录，复用本包冻结数据，避免覆盖本次交付。
- [检查说明](03_检查与口径说明.md)及[机器可读最终核验](05_QA/FINAL_QA.json)。

## 主图

| 图号 | 内容 | PDF | 直接生成代码 |
|---|---|---|---|
'''
 for r in assets:
  if r['kind']=='figure':intro+=f"| {r['number']} | {r['title_zh']} | [{r['id']}]({r['outputs'][0]}) | [{r['generator']}](06_Generation_Code/{r['generator']}) |\n"
 intro+='\n## 主表与补表\n\n| 表号 | 内容 | LaTeX源码 |\n|---|---|---|\n'
 for r in assets:
  if r['kind']=='table':intro+=f"| {r['number']} | {r['title_zh']} | [{r['id']}]({r['outputs'][0]}) |\n"
 intro+='''
## 补充图表的用途

- S1图展示慢快组合的实际曲线和生成区间；S2/S3图把总体RFE拆到雨天、阴天；S4图补充F99深衰落持续时间。
- S1表列两条TabM路径k＝10的确切气象特征；S2表给英国各模型绝对指标；S3表拆分慢快组合天气结果；S4表记录有效秒数和生成均值。
- 正文表4、表5、表6选取代表性k，防止整页挤入几百轮。全迭代指标保留在`04_Source_Data/frozen_exports`与`table_data/All_RFE_Comparison_Values.csv`，没有只保留有利状态。

## 文件组织与使用

`01_Figures_PDF`放最终矢量图（照片面板保留原始栅格）；`02_Tables_LaTeX`放表格片段；`03_Editable_PPTX`放可编辑示意图；`04_Source_Data`放冻结结果副本及可审计派生数据；`05_QA`放SHA、统计和渲染检查；`06_Generation_Code`放本次完整生成脚本。

本次从已保存结果制作图表，未重新训练、重跑英国推理或重估快变分布。图10新计算完整整数滞后ACF，但输入仍是原来6623小时的实测与32条冻结生成曲线；每条曲线的11个旧滞后统计点均回读核对。生成器、模型权重和原科研结果未修改。

保留用户指定图号：图4是ITU，图5是TabM。按当前章节顺序，图5可能先于图4被引用，排入正式论文时需统一首次引用编号。图注为投稿前可编辑草稿，需由作者结合正文确认。
'''
 (out/'00_从这里开始.md').write_text(intro,encoding='utf8')
 lines=['# 图表—生成代码—已有结果—原科研实现','', '本页把“科学实验产生结果”和“本次脚本把结果画成图/表”分开登记。直接重绘只调用本包数据和新绘图脚本。原训练/评价代码副本用于倒查方法，不是本次执行入口。','', '## 一、逐张索引','']
 for r in assets:
  lines+=['### '+('Figure ' if r['kind']=='figure' else 'Table ')+r['number']+' — '+r['title_zh'],'', '- 输出：'+', '.join(link(out/x) for x in r['outputs']),'- 直接生成代码：'+link(out/'06_Generation_Code'/r['generator']),'- 直接输入：']
  lines+=['  - '+link(out/'04_Source_Data'/x) for x in r['inputs']]
  if r['number'] in ['2','5','9'] and r['kind']=='figure':lines+=['- 可编辑源：'+link(out/'03_Editable_PPTX/Fig02_Fig05_Fig09_Editable.pptx')+'；组装PDF的脚本为assemble_pdfs.py。']
  lines+=['- 数据选择与处理：'+r['method'],'']
 lines+=['## 二、派生数据怎样得到','',
 '| 派生文件 | 来源与处理 | 代码 |','|---|---|---|',
 '| Fig03_all_local_hours.csv / Fig03_selected_seconds.csv | Signal_Detectability_Revision_003的小时状态及划分；Fast001清洗秒功率。固定随机种子选完整3600秒小时，记录候选数量/选中键。 | prepare_plot_data.py |',
 '| Table02_datasets.csv | 本地固定划分与英国冻结评价小时，按小时键计数；日期按本地BJT、英国UTC。 | prepare_plot_data.py |',
 '| Table03_model_configs.json | Hourly032 manifest及Bai全156状态中的实际配置；未将旧搜索上限当正式轮数。 | prepare_plot_data.py |',
 '| Fig10_duration_distributions.csv | 原Fast003每小时diagnostics中的完整事件，实测与生成按有事件小时配对；各小时内非空实现等权，再小时等权。原F95/F99平均持续时间逐组核对。 | prepare_plot_data.py |',
 '| Fig10_dense_acf_1_to_1800.csv | 原Fast003的6623份NPZ，逐连续有效段FFT求交叉积并用配对样本均值方差算Pearson ACF；不跨缺测。每个整数滞后1–1800独立计算；每小时每实现原11滞后及275汇总点回读一致。 | derive_dense_acf.py |',
 '| FigS01_examples.csv及examples/*.npz | 各范围时间最早的雨天小时；原NPZ按SHA逐字节复制，叠加原冻结TabM小时预测。 | prepare_plot_data.py |',
 '| table_data/*.csv | LaTeX使用的表值，及全部迭代的高精度比较矩阵。仅格式化时四舍五入。 | build_tables.py |','',
 '详细源路径/输入哈希见下面清单。Fast003全部6623小时回执位于'+link(qa/'Fast003_hour_receipts.json')+'；没有复制整个大体积曲线库。使用本包派生CSV可直接重绘；从原曲线重新派生时，需原项目目录保持可读。','', '## 三、每个冻结输入对应原位置','', '| 本包副本 / 用途 | 原文件 | SHA256 |','|---|---|---|']
 for r in manifests:lines.append('| '+(link(out/r['snapshot']) if r.get('snapshot') else r['role']+'（由该源派生）')+' | '+link(Path(r['original_path']))+' | `'+r['sha256']+'` |')
 lines+=['','## 四、原科研代码按大功能追溯','', '以下文件逐字节复制并登记UPSTREAM_CODE_MANIFEST.json。科研运行入口依赖原项目的模型、数据及历史实现；完整科学复现工程仍在“实验归档_20261027_002”和相应ITU归档中。本次reproduce.ps1只重绘，不调用这些训练或推理入口。','']
 for group,files in UPSTREAM.items():
  lines+=['### '+group,'']
  for rel in files:lines+=['- 原位置：'+link(project/rel)+'；审计副本：'+link(out/'04_Source_Data/upstream_code'/rel)]
 lines+=['','## 五、本次生成脚本功能','', '| 脚本 | 功能 |','|---|---|',
 '| inspect_sources.py | 有界读取已有结果字段和样本口径。 |',
 '| extract_original.py | 从原投稿PDF提取照片与文字，不合成或修改实景。 |',
 '| prepare_plot_data.py | 冻结结果复制、源SHA登记、小时/秒级示例选择、持续时间分布重建。 |',
 '| derive_dense_acf.py | 用保存曲线计算完整1–1800秒ACF，与旧统计交叉核验。 |',
 '| build_plots.py | 11幅数据图（含照片图）和审阅PNG；图内英文、TabM强调、负值保留。 |',
 '| build_diagrams.mjs | 用原生可编辑文本/形状/连线制作图2/5/9，导出PPTX和PDF。 |',
 '| artifact_runtime_adapter.mjs | 适配本机已有Artifact运行库字体模块；不安装依赖，不修改系统包。 |',
 '| assemble_pdfs.py | 统一示意图PDF尺寸，合并带书签的14图预览。 |',
 '| build_tables.py | 全部LaTeX表和数值CSV，百分比/绝对差公式固定。 |',
 '| build_delivery_docs.py | 导航、双语图注、Word图表说明、原科学代码副本及对应清单。 |',
 '| verify_delivery.py | 核验数值、文件和SHA，保存FINAL_QA.json。 |',
 '| seal_package.py | 比较隔离重绘输出，保存代码副本、运行库版本和全目录SHA清单。 |',
 '| reproduce.ps1 | 在新目录从冻结输入重绘PDF/PPTX/LaTeX，编译表格预览。 |',
 '| report_helpers/* | paper-figures技能的Word汇报辅助脚本副本。 |','',
 '文件清单用SHA256检测输入变化；复现应比较数据值和图面内容，不要求PDF/PPTX文件的生成时间等元数据逐字节相同。所有本机依赖路径和复现命令均在reproduce.ps1及下方检查说明中。']
 (out/'01_图表代码与数据溯源.md').write_text('\n'.join(lines),encoding='utf8')
 notes='''# 检查与指标口径

## 数值与样本

1. 本地模型对比为687测试小时；ITU仅用共同合法684小时，另外3小时因输入合法性被排除。两者不能混用样本总数。
2. 本地2404/343/687仍是原随机小时划分，训练内RFE才按日期分组。没有将它改写为按日期留出测试。
3. 156为特征维数：50基础＋106变量—压力层组合。基础50含单层诊断及柱积分，不能称作仅用地表传感器。
4. TabM在统一k＝10下取得本地最低总体RMSE；不写成全部维数/天气/英国均最优。图表保留所有负向比较。
5. 英国5593小时：Chilbolton 3382，Sparsholt 2211。小时模型原天气other包含晴朗干燥；快变表采用四种互斥天气。英国R²负值完整保留。
6. RMSE改善率=100×(对照−TabM)/对照；R²使用绝对差，不报不稳定的R²相对百分比。
7. 表7仅是训练内开发选择证据：16条实现VS-1和8条稳定性检查，包含IID/AR(1)。正式快变与组合结果均使用最新32条逐秒CRPS。256秒比128秒仅约0.12%的开发评分差，不据此声称显著优越。
8. 快变CRPS评价相对小时波动F；组合CRPS评价绝对秒级衰减A。这两者目标不同。表9的仅慢变是冻结小时点估计在该小时重复，其CRPS等于绝对误差。
9. 图10展示全部1–1800秒整数滞后，直接从保存曲线计算；原11滞后的每小时/每实现结果都经过核对。ACF配对要求完整滞后区间有效；缺测不补0。
10. 持续时间使用训练集固定天气F95/F99阈值，排除小时边界或缺测截断事件，并报告有完整事件的配对小时数。图中CDF按小时等权，不是把全部事件堆积后的事件等权CDF。
11. 为一条秒级曲线加上恒定小时预测，不改变小时内SD、中心化ACF或上述相对F阈值持续时间；因此不重复列同一组组合时间结构数值。这里没有评价固定绝对衰减阈值下的跨小时持续时间。
12. 表9沿用已保存的按日期聚类、配对2000次bootstrap区间（seed 20261024）；英国同UTC日两站在同一日期簇。区间对应组合−慢变差值，不是某个均值或一条生成曲线的置信带。

## 图面与编辑

- 全部14张最终PDF已实际渲染检查；表格独立编译后检查分页、字形和截断。检查对象是导出PDF，不只看代码。
- PPTX含3页原生文本、形状和连线，已通过结构、版式、字体和重新导入检查。该检查没有调用本机Microsoft PowerPoint GUI；编辑后建议在实际投稿软件再次检查字体和导出。
- 现场照片来自原投稿PDF嵌入图片，只有原始像素分辨率。未生成假照片、未超分辨率补细节。若作者有原始高清照片，可直接替换图1图片输入。
- 所有统计图PDF保留矢量曲线/文字；照片本身为栅格。图内使用英文，双语图注另存。
- 表格LaTeX按宽表准备，使用代表性k；全迭代原数值不删减。表8拆成两部分，S2为跨页长表；正式IEEE版面可将部分内容移入补充材料。

## 复现

在现有电脑上执行`06_Generation_Code/reproduce.ps1 -OutputDir <一个尚不存在的新目录>`。脚本复用已有Python、Node与TeX，先复制本包冻结绘图数据和回执，再重绘并编译；没有fit、下载或英国推理步骤。

从科研原始曲线重新推导图10，可另用`derive_dense_acf.py --project <原项目根目录> --package <含冻结数据及回执的新绘图包>`。从原结果重新准备所有数据需要先按prepare_plot_data.py构建隔离输出，其源路径固定记录在清单中。

最终数值和文件检查见FINAL_QA.json；原输入SHA、上游代码SHA、6623曲线SHA各自独立保存。已在新的隔离目录完整执行一遍reproduce.ps1，核验记录为REPRODUCIBILITY_QA.json。绘图重现与重新训练整个实验属于不同任务。
'''
 (out/'03_检查与口径说明.md').write_text(notes,encoding='utf8')
 print(f'Documented {len(FIGS)} figures and {len(tables)} table parts; upstream code snapshots {len(upstream)}')
if __name__=='__main__':main()
