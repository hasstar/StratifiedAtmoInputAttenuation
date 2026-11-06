import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile, FileBlob } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../../..');
const output = path.join(root, 'Outputs/Experiments/Hourly_Attenuation_Revision_026_Ten_Model_Test');
const report = JSON.parse(await fs.readFile(path.join(output, 'final_report.json'), 'utf8'));
if (report.status !== 'COMPLETE' || report.independent_qa.status !== 'PASS') throw new Error('QA gate failed');
const modelRows = report.model_rows;
const groups = ['overall', 'clear_dry', 'rain', 'dry_overcast', 'other'];
const weatherLabels = {overall:'总体', clear_dry:'晴天', rain:'雨天', dry_overcast:'干燥阴天', other:'其他天气'};

const wb = Workbook.create();
const ranking = wb.worksheets.add('测试排名');
const weather = wb.worksheets.add('分天气指标');
const settings = wb.worksheets.add('训练参数');
const seeds = wb.worksheets.add('三种子指标');
const fontName = 'Arial';

function style(sheet, columns, rows, title, headers, tableRows) {
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, rows, columns).format.font = {name: fontName, size:10, color:'#17202A'};
  sheet.getRangeByIndexes(0, 0, rows, columns).format.verticalAlignment = 'center';
  sheet.getRangeByIndexes(0, 0, rows, columns).format.rowHeight = 22;
  sheet.getRange('A2').values = [[title]];
  sheet.getRange('A2').format.font = {name:fontName, size:14, bold:true};
  sheet.getRangeByIndexes(2, 0, 1, columns).format.borders = {bottom:{style:'thin',color:'#1F4E78'}};
  sheet.getRangeByIndexes(5, 0, 1, columns).values = [headers];
  const head = sheet.getRangeByIndexes(5, 0, 1, columns);
  head.format = {fill:'#1F4E78', font:{name:fontName,size:10,bold:true,color:'#FFFFFF'},
                 horizontalAlignment:'center',verticalAlignment:'center', rowHeight:30};
  if (tableRows > 0) sheet.getRangeByIndexes(6,0,tableRows,columns).format.numberFormat = '0.0000';
  sheet.getRangeByIndexes(5,0,tableRows+1,columns).format.borders = {
    bottom:{style:'thin',color:'#D9E2F3'}};
}

const rawRows = [];
const seedReferences = new Map();
for (const family of modelRows) {
  const refs = [];
  for (const seed of report.seeds) {
    const record = JSON.parse(await fs.readFile(path.join(output,'test',`${family.model}_seed${seed}.json`),'utf8'));
    for (const group of groups) {
      const metric = record.metrics[group];
      const excelRow = 7 + rawRows.length;
      if (group === 'overall') refs.push(excelRow);
      rawRows.push([family.model_label,seed,weatherLabels[group],metric.n_hours,metric.n_dates,
        metric.rmse_db,metric.mae_db,metric.bias_db,
        group === 'overall' ? record.latency.median_ms : null]);
    }
  }
  seedReferences.set(family.model,refs);
}

style(seeds,9,158,'三种子测试指标',[ '模型','种子','天气','小时数','日期数','RMSE / dB','MAE / dB','Bias / dB','单条耗时 / ms'],150);
seeds.getRange('A4').values = [['每个种子独立评分；不是三个模型预测平均后的集成分数。']];
seeds.getRange('A7:I156').values = rawRows;
seeds.getRange('B7:E156').setNumberFormat('0');
seeds.getRange('A1:A158').format.columnWidth=20;
seeds.getRange('B1:B158').format.columnWidth=8;
seeds.getRange('C1:C158').format.columnWidth=16;
seeds.getRange('D1:E158').format.columnWidth=11;
seeds.getRange('F1:I158').format.columnWidth=17;
seeds.freezePanes.freezeRows(6);

style(ranking,11,30,'十模型测试集排名',[
  'RMSE排名','模型','总体RMSE / dB','总体MAE / dB','Bias / dB','单条耗时 / ms','50/50排名','50/50分数',
  'RMSE归一化','耗时归一化','对数耗时'],10);
ranking.tabColor='#1F4E78';
ranking.getRange('A4').values = [['同一687小时测试集；全部156列输入；指标为三个种子平均。']];
for (let i=0;i<modelRows.length;i++) {
  const model = modelRows[i];
  const r = i+7;
  const refs = seedReferences.get(model.model);
  ranking.getRange(`B${r}`).values = [[model.model_label]];
  ranking.getRange(`C${r}:E${r}`).formulas = [[
    `=AVERAGEIFS('三种子指标'!$F$7:$F$156,'三种子指标'!$A$7:$A$156,B${r},'三种子指标'!$C$7:$C$156,"总体")`,
    `=AVERAGEIFS('三种子指标'!$G$7:$G$156,'三种子指标'!$A$7:$A$156,B${r},'三种子指标'!$C$7:$C$156,"总体")`,
    `=AVERAGEIFS('三种子指标'!$H$7:$H$156,'三种子指标'!$A$7:$A$156,B${r},'三种子指标'!$C$7:$C$156,"总体")`]];
  ranking.getRange(`F${r}`).formulas = [[`=MEDIAN(${refs.map(n=>`'三种子指标'!I${n}`).join(',')})`]];
  ranking.getRange(`K${r}`).formulas = [[`=LN(F${r})`]];
  ranking.getRange(`I${r}`).formulas = [[`=(C${r}-MIN($C$7:$C$16))/(MAX($C$7:$C$16)-MIN($C$7:$C$16))`]];
  ranking.getRange(`J${r}`).formulas = [[`=(K${r}-MIN($K$7:$K$16))/(MAX($K$7:$K$16)-MIN($K$7:$K$16))`]];
  ranking.getRange(`H${r}`).formulas = [[`=0.5*I${r}+0.5*J${r}`]];
  ranking.getRange(`A${r}`).formulas = [[`=1+COUNTIFS($C$7:$C$16,"<"&C${r})`]];
  ranking.getRange(`G${r}`).formulas = [[`=1+COUNTIFS($H$7:$H$16,"<"&H${r})`]];
}
ranking.getRange('A7:A16').setNumberFormat('0');
ranking.getRange('G7:G16').setNumberFormat('0');
ranking.getRange('B1:B30').format.columnWidth=19;
ranking.getRange('A1:A30').format.columnWidth=12;
ranking.getRange('C1:F30').format.columnWidth=20;
ranking.getRange('G1:K30').format.columnWidth=16;
const notes = [
  'RMSE与MAE越小越好；Bias = 预测 − 实测，越接近0越好。',
  '50/50分数：总体测试RMSE归一化与对数推理耗时归一化各占50%，越小越好。',
  '耗时为模型训练好后，一条输入到输出的时间，包含特征切片、填补、标准化和推理。',
  '晴天322小时；雨天51小时；干燥阴天158小时；其他156小时。',
  '旧表BiLSTM 0.2537 dB是seed17单模型；本表是三个种子指标平均。',
  '旧表SVR 0.4482 dB使用分层阶段另调的C=10；本次十模型冻结SVR使用C=1。',
  '历史测试集上按用户要求排名，不是全新盲测；所有156个测试日期与训练日期重叠。'
];
ranking.getRange('A19:A25').values = notes.map(n=>[n]);
ranking.getRange('A27').values = [['结果来源：Revision 026 冻结测试报告；原模型配置与轮数继承Revision 023。']];

style(weather,9,61,'不同天气测试指标',[
  '模型','天气','小时数','日期数','RMSE / dB','MAE / dB','Bias / dB','RMSE种子SD','seed17 RMSE'],50);
weather.getRange('A4').values = [['雨天合并全部雨天；其他天气为晴天、雨天、干燥阴天之外的小时。']];
let idx=0;
for (const model of modelRows) {
  for (const group of groups) {
    const r=7+idx++;
    const s=model.metrics[group];
    weather.getRange(`A${r}:D${r}`).values=[[model.model_label,weatherLabels[group],s.n_hours,s.n_dates]];
    weather.getRange(`E${r}:G${r}`).formulas = [['F','G','H'].map(col=>
      `=AVERAGEIFS('三种子指标'!$${col}$7:$${col}$156,'三种子指标'!$A$7:$A$156,A${r},'三种子指标'!$C$7:$C$156,B${r})`)];
    const source = seedReferences.get(model.model).map(n=>n+groups.indexOf(group));
    weather.getRange(`H${r}`).formulas=[[`=STDEV(${source.map(n=>`'三种子指标'!F${n}`).join(',')})`]];
    weather.getRange(`I${r}`).formulas=[[`='三种子指标'!F${source[0]}`]];
  }
}
weather.getRange('C7:D56').setNumberFormat('0');
weather.getRange('A1:A61').format.columnWidth=20;
weather.getRange('B1:B61').format.columnWidth=16;
weather.getRange('C1:D61').format.columnWidth=10;
weather.getRange('E1:I61').format.columnWidth=18;
weather.getRange('A59').values=[['种子SD描述重复训练的变化，不是独立观测日期的不确定性区间。']];
weather.freezePanes.freezeRows(6);

style(settings,5,31,'冻结训练设置',['模型','停止轮数/树数','种子','训练参数','停止原因'],10);
settings.getRange('A4').values=[['仅重建已冻结的模型，没有新调参；测试阶段拟合次数为0。']];
settings.getRange('A7:E16').values=modelRows.map(m=>[
  m.model_label,m.formal_rounds===null?'不适用':m.formal_rounds,report.seeds.join('/'),JSON.stringify(m.effective_config),
  m.formal_rounds===null?'求解器收敛；没有逐轮早停':'继承Rev023验证集选定轮数，固定轮数重建']);
settings.getRange('B7:B16').setNumberFormat('0');
settings.getRange('A1:A31').format.columnWidth=20;
settings.getRange('B1:C31').format.columnWidth=17;
settings.getRange('D1:D31').format.columnWidth=82;
settings.getRange('E1:E31').format.columnWidth=54;
settings.getRange('D7:E16').format.wrapText=true;
settings.getRange('D7:E16').format.rowHeight=48;
settings.getRange('A19').values=[['缺失填补、缩放和神经网络目标标准化均仅在2404训练小时拟合。']];
settings.getRange('A20').values=[['验证集用于原参数与停止轮数选择；本次仅用于复现核验，不反向调整权重。']];
settings.getRange('A21').values=[['当前按用户要求用测试指标排名，没有按测试改参数、特征或轮数。']];
settings.getRange('A23').values=[['既有实验与数据未修改；全部30重建模型通过旧验证预测及保存/重载核验。']];
settings.getRange('A25').values=[['来源：Revision 023模型设置；Revision 026逐种子测试记录；独立QA PASS。']];

wb.recalculate();
const mainValues = ranking.getRange('A7:K16').values;
const mainErrors=[];
for(let i=0;i<10;i++) {
  const got=mainValues[i], expected=modelRows[i];
  const references=[expected.metrics.overall.rmse_db_mean,expected.metrics.overall.mae_db_mean,
                    expected.metrics.overall.bias_db_mean,expected.latency_median_ms];
  for(let j=0;j<4;j++) if(typeof got[j+2]!=='number'||Math.abs(got[j+2]-references[j])>1e-10)
    mainErrors.push({model:expected.model,column:j+2,actual:got[j+2],expected:references[j]});
  if(got[0]!==expected.rank_test_rmse || got[6]!==expected.rank_test_50_50 ||
     Math.abs(got[7]-expected.score_50_50)>1e-10) mainErrors.push({model:expected.model,ranks:got.slice(0,8)});
}
if(mainErrors.length) throw new Error(JSON.stringify(mainErrors));
const weatherValues=weather.getRange('E7:I56').values;
let weatherIndex=0;
for(const family of modelRows) for(const group of groups) {
  const values=weatherValues[weatherIndex++], metric=family.metrics[group];
  const expected=[metric.rmse_db_mean,metric.mae_db_mean,metric.bias_db_mean,
                  metric.rmse_db_sd_seed,metric.seed17.rmse_db];
  for(let j=0;j<5;j++) if(typeof values[j]!=='number'||Math.abs(values[j]-expected[j])>1e-10)
    throw new Error(`Weather formula differs: ${family.model}/${group}/${j}`);
}
console.log((await wb.inspect({kind:'table',range:'测试排名!A6:H16',include:'values,formulas',
  tableMaxRows:11,tableMaxCols:8,maxChars:3500})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:40},summary:'formula error scan',maxChars:2500});
console.log(errors.ndjson);
await fs.mkdir(path.join(output,'workbook_previews'),{recursive:true});
for(const [sheetName,range,file] of [
 ['测试排名','A1:H28','ranking.png'],
 ['分天气指标','A1:I20','weather.png'],
 ['训练参数','A1:E27','settings.png'],
 ['三种子指标','A1:I22','seeds.png']]) {
 const blob=await wb.render({sheetName,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(output,'workbook_previews',file),new Uint8Array(await blob.arrayBuffer()));
}
const artifact=await SpreadsheetFile.exportXlsx(wb);
const xlsxPath=path.join(output,'Ten_Model_Test_Results.xlsx');
await artifact.save(xlsxPath);
const restored=await SpreadsheetFile.importXlsx(await FileBlob.load(xlsxPath));
const exportedValues=restored.worksheets.getItem('测试排名').getRange('A7:K16').values;
for(let i=0;i<10;i++) for(const j of [0,2,3,4,5,6,7]) {
  if(typeof exportedValues[i][j]!=='number'||Math.abs(exportedValues[i][j]-mainValues[i][j])>1e-10)
    throw new Error(`Saved workbook cache differs: ${i}/${j}`);
}
await fs.writeFile(path.join(output,'workbook_qa.json'),JSON.stringify({status:'PASS',
  report_fingerprint:report.fingerprint,headline_numeric_crosschecks:50,
  rank_crosschecks:20,weather_numeric_crosschecks:250,exported_value_checks:70,
  sheets:4,weather_rows:50,seed_weather_rows:150,
  workbook:xlsxPath,formula_error_scan:errors.ndjson,visual_review:'pending'},null,2));
console.log(JSON.stringify({status:'EXPORTED',path:xlsxPath}));
