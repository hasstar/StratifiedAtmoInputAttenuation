import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require=createRequire(import.meta.url);
const resolved=require.resolve('@oai/artifact-tool',{paths:['C:/Users/SDR_HH/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules']});
const { Workbook, SpreadsheetFile }=await import(pathToFileURL(resolved));

const root=process.cwd();
const dir=path.join(root,'最终提交的修改材料','补充材料_20261024_001');
const data=JSON.parse(await fs.readFile(path.join(dir,'rfe_workbook_data.json'),'utf8'));
const zh=JSON.parse(await fs.readFile(path.join(root,'Code/Fast_Fading_Experiments/Revision_003/feature_labels_zh.json'),'utf8'));
const chinese=new Map(data.dictionary.map(r=>{if(!zh[r[2]])throw new Error('Missing feature label '+r[2]);return [r[1],(r[3]==='surface/column'?'':r[3]+' hPa ')+zh[r[2]]];}));
const shownRows=data.rows.map(r=>r.map((v,j)=>(j===4||j===8)&&v?chinese.get(v)+'\n'+v:v));
const shownHeaders=[...data.headers];shownHeaders[4]='本轮删除参数（中文／原名）';shownHeaders[8]='下一轮删除参数（中文／原名）';
const wb=Workbook.create();
const trace=wb.worksheets.add('逐轮删除与保留');
const dict=wb.worksheets.add('参数字典');

function setup(sh,title,note,headers,rows,widths){
  const end=String.fromCharCode(64+headers.length);
  sh.showGridLines=false;
  sh.getRange(`A1:${end}${rows.length+5}`).format={font:{name:'Microsoft YaHei',size:10,color:'#202B36'},verticalAlignment:'center',wrapText:true};
  sh.getRange('A1').values=[[title]];
  sh.getRange(`A1:${end}1`).merge();
  sh.getRange(`A1:${end}1`).format.font={size:14,bold:true,color:'#183B56'};
  sh.getRange(`A1:${end}1`).format.rowHeight=29;
  sh.getRange('A2').values=[[note]];
  sh.getRange(`A2:${end}3`).merge();
  sh.getRange(`A2:${end}3`).format.font={size:10,color:'#52616B'};
  sh.getRange(`A2:${end}3`).format.rowHeight=22;
  sh.getRange(`A5:${end}5`).values=[headers];
  sh.getRange(`A6:${end}${rows.length+5}`).values=rows;
  sh.getRange(`A5:${end}5`).format={fill:'#183B56',font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:36};
  widths.forEach((width,i)=>{const c=String.fromCharCode(65+i);sh.getRange(`${c}:${c}`).format.columnWidthPx=width;});
  sh.freezePanes.freezeRows(5);sh.freezePanes.freezeColumns(1);
  const table=sh.tables.add(`A5:${end}${rows.length+5}`,true,sh===trace?'RFETrace':'FeatureDictionary');
  table.showFilterButton=true;
}

setup(trace,'RFE逐轮参数记录｜10模型 × 156状态',
  '第0轮是156参数基线；第1–155轮各删除1项。“本轮删除”用于进入当前状态，“下一轮删除”用于离开当前状态。保留参数ID按原输入顺序排列；完整名称见参数字典及随附原名CSV。来源：T014 delivery_002，1550次相邻状态差集已校验。',
  shownHeaders,shownRows,[110,90,95,85,330,95,680,95,330]);
trace.getRange(`B6:D${data.rows.length+5}`).setNumberFormat('0');
trace.getRange(`F6:F${data.rows.length+5}`).setNumberFormat('0');
for(const c of ['B','C','D','F','H'])trace.getRange(`${c}6:${c}${data.rows.length+5}`).format.horizontalAlignment='center';
data.rows.forEach((r,i)=>{
  trace.getRange(`A${i+6}:I${i+6}`).format.rowHeight=Math.max(64,14*Math.ceil(r[5]/19)+12);
  if(r[1]===0)trace.getRange(`A${i+6}:I${i+6}`).format.fill='#EAF1F5';
  if(r[5]===10)trace.getRange(`A${i+6}:I${i+6}`).format.fill='#E3F4F0';
});
setup(dict,'参数字典｜156个冻结大气输入',
  'F001–F156只作为排版编号，不改变原始字段或物理含义。压力层字段以hPa列明；surface/column保留原数据分组，不凭字段名推断单位。原字段名称是可复核标识。',
  ['参数ID','参数中文名称','原始字段名','参数英文名称','压力层hPa／类别','来源分组'],data.dictionary.map(r=>[r[0],chinese.get(r[1]),r[1],r[2],r[3],r[4]]),[85,320,480,430,160,160]);
dict.getRange('A6:F161').format.rowHeight=39;
wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:trace.id,range:'A5:F8',maxChars:2000,tableMaxCellChars:90})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
await fs.mkdir(path.join(dir,'预览'),{recursive:true});
for(const [sheet,range,name] of [[trace,'A1:I8','RFE_表头与基线'],[trace,'A150:I161','RFE_k10至k1示例'],[dict,'A1:F12','参数字典']]){
  const img=await wb.render({sheetName:sheet.name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(dir,'预览',name+'.png'),new Uint8Array(await img.arrayBuffer()));
}
const out=path.join(dir,'RFE_每模型每轮删除与保留参数.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(out);
console.log(JSON.stringify({output:out,states:data.rows.length,features:data.dictionary.length}));
