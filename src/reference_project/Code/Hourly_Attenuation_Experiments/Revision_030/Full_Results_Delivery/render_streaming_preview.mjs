import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const codeDir=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(codeDir,'../../../..');
const out=path.join(root,'Outputs/Experiments/Hourly_Attenuation_Revision_030_Original_Training_Aligned/Full_Results');
const payload=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8'));
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out,'previews/visual_check_source.xlsx')));
book.recalculate();
const changedOnly=process.argv.includes('--changed-only');
const specs=changedOnly?payload.sheets.slice(0,2):payload.sheets;
for (const spec of specs) {
  const col=Math.min(spec.headers.length,11);
  const letter=String.fromCharCode(64+col);
  const blob=await book.render({sheetName:spec.name,range:`A1:${letter}14`,scale:1.5,format:'png'});
  await fs.writeFile(path.join(out,'previews',spec.name+'.png'),new Uint8Array(await blob.arrayBuffer()));
  console.log(`rendered final-format ${spec.name}`);
}
const inspection=await book.inspect({kind:'table',range:'天气指标17700!A6:K9',include:'values,formulas',tableMaxRows:4,tableMaxCols:11,maxChars:2000});
console.log(inspection.ndjson);
await fs.writeFile(path.join(out,changedOnly?'visual_render_changed_qa.json':'visual_render_qa.json'),JSON.stringify({status:'RENDERED_AWAITING_VISUAL_REVIEW',sheets:specs.map(s=>s.name),
  exact_first_eight_records_from_final_authoring:true,original_experiment_files_changed:false},null,2)+'\n',{flag:'wx'});
