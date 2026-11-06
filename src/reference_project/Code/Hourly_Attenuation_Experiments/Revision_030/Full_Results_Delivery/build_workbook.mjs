import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const codeDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(codeDir, '../../../..');
const out = path.join(root, 'Outputs/Experiments/Hourly_Attenuation_Revision_030_Original_Training_Aligned/Full_Results');
const payload = JSON.parse(await fs.readFile(path.join(out, 'workbook_data.json'), 'utf8'));
try {await fs.access(payload.xlsx_path); throw new Error('Workbook exists; do not overwrite.');}
catch (error) {if (error.code !== 'ENOENT') throw error;}
const workbook = Workbook.create();
const qa = {status: 'BUILDING', sheets: [], errors: null};
const font = 'Arial';

function colName(index) {
  let result = '';
  for (let n=index+1; n>0; n=Math.floor((n-1)/26)) result = String.fromCharCode(65+(n-1)%26)+result;
  return result;
}

for (let number=0; number<payload.sheets.length; number++) {
  const spec = payload.sheets[number];
  const sheet = workbook.worksheets.add(spec.name);
  const end = colName(spec.headers.length-1);
  sheet.showGridLines = false;
  sheet.getRange('A2').values = [[spec.name]];
  sheet.getRange('A2').format.font = {name:font, size:14, bold:true, color:'#172B4D'};
  sheet.getRange('A4').values = [[`来源：${spec.source}；唯一seed17，固定53轮。完整数值，无筛选。`]];
  sheet.getRange('A4').format.font = {name:font, size:10, italic:true, color:'#536171'};
  sheet.getRange(`A6:${end}6`).values = [spec.headers];
  // One rectangular block avoids repeatedly rebuilding a very large table.
  sheet.getRangeByIndexes(6, 0, spec.rows.length, spec.headers.length).values = spec.rows;
  const last = spec.rows.length+6;
  const used = sheet.getRange(`A6:${end}${last}`);
  used.format.font = {name:font, size:10, color:'#243746'};
  used.format.verticalAlignment = 'center';
  used.format.wrapText = false;
  const header = sheet.getRange(`A6:${end}6`);
  header.format.fill = '#27486A';
  header.format.font = {name:font, size:10, bold:true, color:'#FFFFFF'};
  header.format.horizontalAlignment = 'center';
  header.format.rowHeight = 42;
  header.format.wrapText = true;
  header.format.columnWidth = 17;
  header.format.borders = {preset:'inside', style:'thin', color:'#FFFFFF'};
  for (let c=0; c<spec.headers.length; c++) {
    const label = spec.headers[c];
    const range = sheet.getRangeByIndexes(6,c,spec.rows.length,1);
    const sample = spec.rows.find(row=>row[c]!==null && row[c]!==undefined && row[c]!=='')?.[c];
    if (typeof sample === 'number') {
      range.format.horizontalAlignment = 'right';
      range.setNumberFormat(/rmse|mae|bias|objective|_se|SE|CV|seconds|improvement|delta|lower|upper/i.test(label)?'0.000000':'#,##0');
      if (/fraction_helpful/.test(label)) range.setNumberFormat('0.0%');
    } else if (typeof sample === 'boolean') {
      range.format.horizontalAlignment = 'center';
      sheet.getRangeByIndexes(5,c,1,1).format.columnWidth = 15;
    } else {
      range.format.horizontalAlignment = 'left';
      if (/feature_name|deleted_feature|下一删除列/.test(label)) sheet.getRangeByIndexes(5,c,1,1).format.columnWidth = 105;
      if (/choice_json|dates_json|unit/.test(label)) sheet.getRangeByIndexes(5,c,1,1).format.columnWidth = 65;
      if (/sha256/.test(label)) sheet.getRangeByIndexes(5,c,1,1).format.columnWidth = 69;
      if (/气压面|levels_hpa|comparison_weather/.test(label)) sheet.getRangeByIndexes(5,c,1,1).format.columnWidth = 35;
    }
  }
  if (spec.name === '说明') {
    sheet.getRange('A6').format.columnWidth = 29;
    sheet.getRange('B6').format.columnWidth = 115;
    sheet.getRange(`B7:B${last}`).format.wrapText = true;
    sheet.getRange(`A7:B${last}`).format.rowHeight = 34;
  }
  const table = sheet.tables.add(`A6:${end}${last}`, true, `ResultsTable${number+1}`);
  table.showFilterButton = true;
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(spec.name === '说明' ? 1 : 2);
  if (number<2) sheet.tabColor='#27486A';
  // Capture an exact content check for first, middle and last records without truncating the dataset.
  for (const rowIndex of [...new Set([0, Math.floor(spec.rows.length/2), spec.rows.length-1])]) {
    const actual = sheet.getRangeByIndexes(6+rowIndex,0,1,spec.headers.length).values[0];
    const want = spec.rows[rowIndex];
    for (let c=0; c<want.length; c++) {
      if (want[c] === null || want[c] === '') {
        if (actual[c] !== null && actual[c] !== '') throw new Error(`${spec.name} blank mismatch`);
      } else if (actual[c] !== want[c]) throw new Error(`${spec.name} value mismatch at ${rowIndex}:${c}`);
    }
  }
  qa.sheets.push({name:spec.name, rows:spec.rows.length, columns:spec.headers.length, table_range:`A6:${end}${last}`, first_middle_last_checked:true});
  console.log(`populated ${spec.name}: ${spec.rows.length} complete records`);
}

workbook.recalculate();
const inspection = await workbook.inspect({kind:'table', range:'分层组合1024!A6:K10', include:'values,formulas', tableMaxRows:5, tableMaxCols:11, maxChars:3000});
console.log(inspection.ndjson);
const errors = await workbook.inspect({kind:'match', searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:100},summary:'final formula error scan',maxChars:3000});
qa.errors = errors.ndjson;
if (/"matched"\s*:\s*true|"cell"\s*:/i.test(errors.ndjson) && /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson)) throw new Error('Unexpected formula error');

for (const spec of payload.sheets) {
  const lastCol = colName(Math.min(spec.headers.length, 11)-1);
  const preview = await workbook.render({sheetName:spec.name,range:`A1:${lastCol}14`,scale:1.5,format:'png'});
  await fs.writeFile(path.join(payload.preview_dir, `${spec.name}.png`), new Uint8Array(await preview.arrayBuffer()));
  console.log(`rendered ${spec.name}`);
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(payload.xlsx_path);
qa.status='PASS';
qa.new_training_fits=0;
qa.new_model_inference_calls=0;
qa.recalculated_before_export=true;
await fs.writeFile(path.join(out,'workbook_build_qa.json'),JSON.stringify(qa,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:'PASS',sheets:qa.sheets.length,xlsx:payload.xlsx_path}));
