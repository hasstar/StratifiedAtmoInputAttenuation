/** Figure 5 v003: native editable reconstruction of the image2 composition.
 * Reads the frozen model config; does not run training or alter experiment data.
 * node build_figure.mjs [--finalize]
 */
import fs from 'node:fs/promises';
import {constants} from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import './artifact_runtime_adapter.mjs';
const here=path.dirname(fileURLToPath(import.meta.url));
const project=path.resolve(here,'../..');
const runtime='C:/Users/SDR_HH/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const {Presentation,PresentationFile,registerPdfFontFace}=await import(pathToFileURL(path.join(process.env.RUNTIME_NODE_MODULES,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
for(const [weight,style,file] of [[400,'normal','times.ttf'],[700,'normal','timesbd.ttf'],[400,'italic','timesi.ttf']])registerPdfFontFace({family:'Times New Roman',weight,style,source:new Uint8Array(await fs.readFile('C:/Windows/Fonts/'+file))});
registerPdfFontFace({family:'Segoe UI Symbol',weight:400,style:'normal',source:new Uint8Array(await fs.readFile('C:/Windows/Fonts/seguisym.ttf'))});
const configPath=path.join(project,'最终提交的修改材料/论文图表_20261030_001/04_Source_Data/Table03_model_configs.json');
const tabm=JSON.parse(await fs.readFile(configPath,'utf8')).tabm;
if(tabm.config.width!==256||tabm.config.blocks!==3||tabm.config.k!==32||tabm.config.dropout!==0.1||tabm.config.embedding!==0||tabm.config.gate!==false||tabm.epochs!==54||tabm.weight_policy!=='uniform')throw new Error('Frozen configuration changed; check labels before rebuilding.');
await fs.mkdir(path.join(here,'qa'),{recursive:true});
const W=1760,H=790;
const p=Presentation.create({slideSize:{width:W,height:H}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#10243D',blue:'#2275AF',blueFill:'#E3F1FB',blueSoft:'#F5FAFE',salmon:'#C76144',salmonFill:'#FCE9E2',green:'#25876F',greenFill:'#E8F5EF',grey:'#596A7C',grayFill:'#EAF0F4',rule:'#C6D6E3'};
const inventory=[];
function box(name,x,y,w,h,t='',o={}){
 const q=s.shapes.add({name,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:o.fill??'none',line:{fill:o.stroke??'none',width:o.stroke?1.5:0,style:'solid'}});
 if(t){q.text=t;q.text.style={typeface:o.typeface??'Times New Roman',fontSize:o.size??26,color:o.color??C.ink,bold:o.bold??false,italic:o.italic??false,alignment:o.align??'center',verticalAlignment:'middle',autoFit:'none',insets:o.pad??4,wrap:'square'};}
 inventory.push({name,x,y,w,h,text:t});return q;
}
function text(n,x,y,w,h,t,size=26,o={}){return box(n,x,y,w,h,t,{size,pad:0,align:'left',...o});}
function line(n,x1,y1,x2,y2,color=C.blue,width=1.8,head=false){
 const q=s.shapes.add({name:n,geometry:'connector',kind:'straight',position:{left:x1,top:y1,width:x2-x1,height:y2-y1},line:{fill:color,width},...(head?{tail:{type:'triangle',width:'med',length:'med'}}:{})});inventory.push({name:n,x1,y1,x2,y2});return q;
}
function arrow(n,x1,y1,x2,y2,color=C.blue){return line(n,x1,y1,x2,y2,color,1.8,true);}
function matrix(n,x,y,w,h,rows,color=C.grayFill){
 const rh=27,gap=5;
 for(let i=0;i<rows.length;i++){const yy=y+i*(rh+gap);box(n+'-row-'+i,x+8,yy,w-16,rh,rows[i],{size:27,fill:rows[i]==='⋮'?'none':color,italic:rows[i]!=='⋮',pad:0,typeface:rows[i]==='⋮'?'Segoe UI Symbol':'Times New Roman'});}
 line(n+'-left',x,y-3,x,y+h,C.ink,1.5);line(n+'-left-top',x,y-3,x+6,y-3,C.ink,1.5);line(n+'-left-bottom',x,y+h,x+6,y+h,C.ink,1.5);
 line(n+'-right',x+w,y-3,x+w,y+h,C.ink,1.5);line(n+'-right-top',x+w-6,y-3,x+w,y-3,C.ink,1.5);line(n+'-right-bottom',x+w-6,y+h,x+w,y+h,C.ink,1.5);
}

// (a) One observation hour, represented in all 32 members.
text('panel-a-title',30,13,1680,44,'(a)  Hourly attenuation estimator',34,{bold:true});
text('input-heading',30,74,245,32,'Hourly inputs',27,{bold:true,align:'center'});
box('full-feature-pool',30,115,245,112,'',{fill:C.greenFill,stroke:C.green});
text('full-pool-label',45,120,210,32,'Full pool: 156',25,{bold:true});
text('full-pool-counts',45,158,215,59,'50 base + 106\npressure-level features',21);
text('alternative-or',30,234,245,30,'or',25,{italic:true,align:'center'});
box('base-feature-pool',30,273,245,78,'Base pool: 50\nNo vertical resolution',{fill:C.greenFill,stroke:C.green,size:22});
line('input-pool-join-top',284,171,284,312,C.green,1.5);
line('input-pool-short-top',275,171,284,171,C.green,1.5);line('input-pool-short-bottom',275,312,284,312,C.green,1.5);
arrow('selected-to-preprocessing',284,248,310,248);
text('retained-feature-count',280,184,34,58,'k',29,{italic:true,align:'center'});
text('retained-feature-note',30,368,245,31,'k retained features',25,{align:'center',italic:true});
box('preprocessing',312,117,181,230,'',{fill:C.blueSoft,stroke:C.blue});
text('preprocessing-title',322,126,161,33,'Preprocessing',22,{bold:true,align:'center'});
text('preprocessing-scope',322,166,161,59,'Fit on local\ntraining data',22,{align:'center'});
box('median-imputation',325,238,155,42,'Median imputation',{fill:'#FFFFFF',stroke:C.rule,size:19});
box('standardization',325,292,155,42,'Standardization',{fill:'#FFFFFF',stroke:C.rule,size:20});
arrow('preprocessing-to-replication',497,248,531,248);
text('same-hour-title',527,89,164,57,'Same input\nfor M members',23,{align:'center'});
text('replication-matrix-name',536,152,150,33,'X: M × k',27,{italic:true,align:'center'});
matrix('replicated-input',548,195,124,123,['xᵀ','xᵀ','⋮','xᵀ'],C.greenFill);
text('member-count',520,339,180,36,'M = 32 members',26,{align:'center'});
arrow('replicated-input-to-backbone',682,248,710,248);
box('tabm-backbone-container',710,96,378,254,'',{fill:C.blueSoft,stroke:C.rule});
text('backbone-heading',720,103,358,37,'TabM backbone',31,{bold:true,align:'center'});
text('backbone-subheading',720,146,358,52,'Shared hidden weights\nMember-specific adapters',24,{align:'center'});
for(let i=0;i<3;i++){
 const x=724+i*121;
 box('block-'+(i+1),x,215,108,114,'',{fill:'#FFFFFF',stroke:C.blue});
 text('block-heading-'+i,x+4,221,100,31,'Block '+(i+1),24,{bold:true,align:'center'});
 text('block-spec-'+i,x+4,251,100,76,'256 units\nReLU\nDropout 0.1',20,{align:'center'});
 if(i<2)arrow('between-blocks-'+i,x+108,269,x+121,269);
}
text('inside-block-crossref',717,365,370,31,'See (b) for the block operation',25,{italic:true,align:'center'});
text('heads-title',1106,81,180,59,'Member-specific\nheads',22,{bold:true,align:'center'});
line('backbone-to-heads-bus',1088,269,1100,269);line('heads-input-bus',1100,165,1100,354);
const headYs=[145,205,265,334],headLabels=['1','2','3','32'],zhats=['ẑ₁,h','ẑ₂,h','ẑ₃,h','ẑ₃₂,h'];
for(let i=0;i<4;i++){
 const yy=headYs[i],cy=yy+20;
 arrow('to-head-'+i,1100,cy,1113,cy);
 box('member-head-'+headLabels[i],1113,yy,72,40,'h'+headLabels[i]+'(·)',{fill:C.salmonFill,stroke:C.salmon,size:25,italic:true});
 arrow('head-to-output-'+i,1187,cy,1203,cy);
 text('member-output-'+i,1206,yy,74,40,zhats[i],26,{italic:true});
 line('output-collect-'+i,1279,cy,1294,cy,C.ink,1.7);
}
text('heads-ellipsis',1114,306,68,24,'⋮',27,{align:'center',typeface:'Segoe UI Symbol'});text('output-ellipsis',1208,306,69,24,'⋮',27,{align:'center',typeface:'Segoe UI Symbol'});
line('all-member-output-bus',1294,165,1294,354,C.ink,1.8);
arrow('mean-branch',1294,252,1320,252);
box('member-mean',1320,204,124,97,'',{fill:C.greenFill,stroke:C.green});
text('mean-title',1324,208,116,29,'Mean',25,{bold:true,align:'center'});
text('mean-standardized-output',1324,241,116,29,'z̄h',26,{italic:true,align:'center'});
text('mean-count',1324,270,116,24,'32 members',18,{align:'center'});
arrow('mean-to-unscale',1444,252,1468,252);
box('inverse-target-scaling',1468,204,107,97,'Inverse\ntarget\nscaling',{fill:C.greenFill,stroke:C.green,size:22});
arrow('unscale-to-attenuation',1575,252,1598,252);
box('hourly-attenuation',1598,197,132,111,'Hourly\nattenuation\nÂh (dB)',{fill:C.greenFill,stroke:C.green,size:22,bold:true});
line('all-outputs-to-loss-vertical',1294,354,1294,389,C.ink,1.8);
arrow('all-outputs-to-loss',1294,389,1320,389,C.ink);
box('memberwise-loss',1320,355,255,64,'Member-wise MSE\nTraining only',{fill:C.salmonFill,stroke:C.salmon,size:25});
line('panel-a-rule',30,435,1730,435,C.rule,1.3);

// (b) The same operation is applied in every hidden block; W differs by block.
text('panel-b-title',30,448,1680,43,'(b)  Inside one TabM block',34,{bold:true});
const y=541;
const mats=[
 ['hidden-input',50,124,['h₁ᵀ','h₂ᵀ','⋮','h₃₂ᵀ'],C.grayFill,'Input H','M × input width'],
 ['input-adapter',260,124,['r₁ᵀ','r₂ᵀ','⋮','r₃₂ᵀ'],C.salmonFill,'Input adapter R','M × input width'],
 ['output-adapter',730,124,['s₁ᵀ','s₂ᵀ','⋮','s₃₂ᵀ'],C.salmonFill,'Output adapter S','M × output width'],
 ['member-bias',960,124,['b₁ᵀ','b₂ᵀ','⋮','b₃₂ᵀ'],C.salmonFill,'Bias B','M × output width'],
 ['block-output',1530,140,['h′₁ᵀ','h′₂ᵀ','⋮','h′₃₂ᵀ'],C.grayFill,'Output H′','M × output width']
];
for(const [n,x,w,rows,fill,title,dim]of mats){
 text(n+'-title',x-38,501,w+76,34,title,26,{bold:true,align:'center'});
 matrix(n,x,y,w,123,rows,fill);
 text(n+'-shape',x-35,678,w+70,32,dim,23,{align:'center'});
}
text('shared-weight-title',441,501,222,34,'Shared weight W',26,{bold:true,align:'center'});
box('shared-weight-matrix',480,538,144,126,'',{fill:C.blueFill,stroke:C.blue});
for(let i=1;i<6;i++){line('W-grid-v-'+i,480+i*24,538,480+i*24,664,'#A5CDE9',1);line('W-grid-h-'+i,480,538+i*21,624,538+i*21,'#A5CDE9',1);}
text('W-symbol',500,574,104,56,'W',42,{italic:true,align:'center'});
text('W-shape',430,678,244,32,'Input width × output width',23,{align:'center'});
for(const [n,x,op]of [['elementwise-input',191,'⊙'],['matrix-multiply',411,'×'],['elementwise-output',658,'⊙'],['bias-add',884,'+']])text(n,x,573,49,55,op,43,{align:'center',typeface:'Segoe UI Symbol'});
arrow('bias-to-relu',1095,603,1132,603);
box('relu',1132,577,103,52,'ReLU',{fill:'#FFFFFF',stroke:C.grey,size:27});
arrow('relu-to-dropout',1235,603,1270,603);
box('dropout',1270,577,118,52,'Dropout',{fill:'#FFFFFF',stroke:C.grey,size:27});
arrow('dropout-to-output',1388,603,1518,603);
text('dropout-training',1248,638,162,33,'0.1, training only',23,{align:'center'});
box('shared-legend-swatch',127,730,32,23,'',{fill:C.blueFill,stroke:C.blue});
text('shared-legend',169,723,546,38,'Shared across members: one W per block',25);
box('member-legend-swatch',727,730,32,23,'',{fill:C.salmonFill,stroke:C.salmon});
text('member-legend',770,723,850,38,'Member-specific: input/output scales, biases and regression heads',25);


s.speakerNotes.text='Two-panel revision of Figure5. Panel(a) shows concurrent meteorological inputs, training-fitted preprocessing, 32 members, 3 blocks of width256, separate regression heads, standardized prediction averaging, inverse target scaling and memberwise MSE. Panel(b) shows shared W and member-specific R/S/B followed by ReLU and training-only dropout0.1. The network is standard TabM from the supplied original paper Fig1 and local Revision031 implementation; matrices W differ across depth. The former protocol panel and displayed equation are relocated to the aligned manuscript text in this revision. All visible elements are mapped to explanatory prose in the accompanying guide. All native shapes/text/connectors remain editable; no experiments were rerun.';
await fs.writeFile(path.join(here,'qa/native_objects.json'),JSON.stringify(inventory,null,2));
const png=await p.export({slide:s,format:'png',scale:1.4});
await fs.writeFile(path.join(here,'qa/Fig05_native_preview.png'),new Uint8Array(await png.arrayBuffer()));
const pdf=await p.export({slide:s,format:'pdf'});
await fs.writeFile(path.join(here,'qa/Fig05_native_raw.pdf'),new Uint8Array(await pdf.arrayBuffer()));
const candidate=path.join(here,'qa/Fig05_editable_candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
console.log(JSON.stringify({candidate,objects:inventory.length,rendered:true}));
if(process.argv.includes('--finalize')){
 const SKILL='C:/Users/SDR_HH/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
 const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
 await fs.mkdir(path.join(here,'editable'),{recursive:true});
 const finalPath=path.join(here,'editable/Fig05_TabM_Architecture_And_Protocol_Editable.pptx');
 const originalLink=fs.link;
 fs.link=async(source,target)=>{try{return await originalLink(source,target);}catch(e){if(process.platform==='win32'&&e.code==='EPERM'&&String(source).endsWith('.validated')&&path.resolve(target)===path.resolve(finalPath)){await fs.copyFile(source,target,constants.COPYFILE_EXCL);return;}throw e;}};
 const result=await finalizePresentation({explicitTotalSlideCount:1,workspaceDir:project,candidatePath:candidate,finalPath,pythonExecutable:path.join(runtime,'python/python.exe'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],requiredNativeTableOwnerSlides:[],fontPolicy:{basis:'design',families:['Times New Roman','Segoe UI Symbol']},verifyArtifactToolImport:true,receiptPath:path.join(here,'qa/final_validation.json')});
 console.log(JSON.stringify(result));
}
