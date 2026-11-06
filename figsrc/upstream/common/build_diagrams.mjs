/** Editable native-shape scientific diagrams. No generated/rasterized slide artwork.
 * Uses the existing bundled @oai/artifact-tool; no dependency installation.
 * Run node build_diagrams.mjs [--package ABS_DIRECTORY] [--finalize].
 */
import fs from 'node:fs/promises';
import {constants as fsConstants} from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import './artifact_runtime_adapter.mjs';
process.on('uncaughtException',err=>{console.error(err.name+': '+err.message);process.exit(1);});

const RUNTIME=process.env.CODEX_ARTIFACT_NODE_MODULES || 'C:/Users/SDR_HH/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES=RUNTIME;
const {Presentation,PresentationFile,registerPdfFontFace}=await import(pathToFileURL(path.join(RUNTIME,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
for(const [weight,file]of [[400,'arial.ttf'],[700,'arialbd.ttf']])registerPdfFontFace({family:'Arial',weight,style:'normal',source:new Uint8Array(await fs.readFile('C:/Windows/Fonts/'+file))});
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../../..');
const pi=process.argv.indexOf('--package');
const out=pi>=0?path.resolve(process.argv[pi+1]):path.join(root,'最终提交的修改材料/论文图表_20261030_001');
// The manually laid-out labels below must match the frozen configuration.
const sourceConfig=JSON.parse(await fs.readFile(path.join(out,'04_Source_Data/Table03_model_configs.json'),'utf8')).tabm;
if(sourceConfig.epochs!==54 || sourceConfig.weight_policy!=='uniform' || sourceConfig.config.width!==256 || sourceConfig.config.blocks!==3 || sourceConfig.config.k!==32 || sourceConfig.config.dropout!==0.1 || sourceConfig.config.embedding!==0 || sourceConfig.config.gate!==false)throw new Error('Diagram labels do not match the frozen TabM configuration');
const build=path.join(out,'05_QA/ppt_build');
await fs.mkdir(build,{recursive:true});
await fs.mkdir(path.join(out,'03_Editable_PPTX'),{recursive:true});
const pres=Presentation.create({slideSize:{width:1280,height:800}});
const C={ink:'#263D51',blue:'#166A92',pale:'#EDF5FA',green:'#167348',lightgreen:'#EAF5EE',orange:'#A65C20',grey:'#66717B',line:'#B2C5CF'};
function box(s,n,x,y,w,h,t,{fill='none',stroke='none',size=25,color=C.ink,bold=false,align='center',pad=10}={}){
 const q=s.shapes.add({name:n,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,
 line:{fill:stroke,width:stroke==='none'?0:1.5,style:'solid'}});
 q.text=t;q.text.style={typeface:'Arial',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',insets:pad,wrap:'square'};return q;
}
function text(s,n,x,y,w,h,t,size=25,opts={}){return box(s,n,x,y,w,h,t,{size,align:'left',pad:0,...opts});}
function arrow(s,a,b,from='right',to='left',kind='straight',color=C.blue){return s.shapes.connect(a,b,{fromSide:from,toSide:to,kind,line:{fill:color,width:2.4},tail:{type:'triangle',width:'med',length:'med'}});}
function line(s,n,x,y,dx,dy,color=C.line,width=1.4,head=false){return s.shapes.add({name:n,geometry:'connector',kind:'straight',position:{left:x,top:y,width:dx,height:dy},line:{fill:color,width},...(head?{tail:{type:'triangle',width:'med',length:'med'}}:{})});}
function slide(title){const s=pres.slides.add();s.background.fill='#FFFFFF';text(s,'figure-title',40,23,1200,53,title,32,{bold:true});line(s,'title-rule',40,87,1200,0,C.line,1);return s;}

// Figure 2: coordinate spacing is deliberately schematic, not altitude.
{
 const s=slide('Vertically resolved and base meteorological information');
 const levels=[1,5,30,200,300,500,650,700,850,925];
 text(s,'level-heading',45,103,160,62,'Pressure\n(hPa)',23,{bold:true});
 for(let i=0;i<levels.length;i++){
  const y=160+i*47;
  box(s,`pressure-band-${levels[i]}`,205,y-18,505,36,'',{fill:'none',pad:0});
  line(s,`pressure-line-${levels[i]}`,205,y,505,0,'#C7DCE8',1.2);
  text(s,`pressure-label-${levels[i]}`,70,y-17,110,33,String(levels[i]),25,{align:'right'});
 }
 line(s,'surface',170,655,550,0,C.green,3);
 const rx=box(s,'ground-receiver',160,676,230,58,'Ground receiver',{fill:C.lightgreen,stroke:C.green,size:25});
 const satellite=box(s,'geo-satellite',585,103,175,48,'GEO satellite',{fill:'#FFFFFF',stroke:C.ink,size:24});
 arrow(s,rx,satellite,'top','bottom','straight',C.ink);
 text(s,'path-label',360,622,320,35,'Slant propagation path',23,{color:C.ink});
 const b=box(s,'base-feature-pool',805,156,410,178,'50 base features\nSurface and single-level diagnostics\nColumn-integrated quantities',{fill:C.lightgreen,stroke:C.green,size:25});
 const p=box(s,'pressure-feature-pool',805,394,410,184,'106 pressure-level features\nVariable–level combinations\nRetained across 10 pressure levels',{fill:C.pale,stroke:C.blue,size:25});
 text(s,'base-explanation',805,336,410,47,'No explicit vertical resolution',23,{color:C.green,align:'center'});
 text(s,'pool-total',805,622,410,100,'Full candidate pool: 156 features\nIndependent base pool: 50 features',24,{bold:true,align:'center'});
 text(s,'schematic-disclosure',40,749,1200,35,'Schematic spacing; pressure levels are not equally spaced in physical height. Candidate counts follow training-only screening.',21,{color:C.grey});
 s.speakerNotes.text='Fig02. Sources: prepared methods copies of Dataset README and Hourly Revision 032/041 protocols; original paths and SHA-256 in 05_QA/SOURCE_MANIFEST.json. Current pressure levels are 925,850,700,650,500,300,200,30,5,1 hPa. 156 retained features = 50 base + 106 variable-level features. Geometry is a conceptual schematic, not a ray-tracing or physical-altitude result. Native text/shapes/connectors remain editable.';
}
// Figure 5: inherited TabM architecture plus the actual task-specific protocol.
{
 const s=slide('TabM hourly attenuation model and task-specific protocol');
 const inp=box(s,'selected-inputs',35,175,237,190,'Meteorological inputs\nk retained features\n156- or 50-feature\ncandidate pool',{fill:C.lightgreen,stroke:C.green,size:26});
 const prep=box(s,'frozen-preprocessing',325,175,232,190,'Preprocessing\nMedian imputation\nFeature scaling\nFit on LOCAL train',{fill:C.pale,stroke:C.blue,size:25});
 const container=box(s,'tabm-backbone-container',610,125,365,327,'',{fill:'#FAFBFC',stroke:C.ink});
 text(s,'tabm-backbone-title',628,142,330,43,'Inherited TabM backbone',27,{bold:true,align:'center'});
 const backbone=box(s,'tabm-shared-ensemble',632,205,321,214,'32 parallel members\n3 shared linear blocks\nWidth 256 · ReLU\nMember-specific scales / biases\nDropout 0.1 during training',{fill:'#FFFFFF',stroke:C.line,size:24});
 const pred=box(s,'ensemble-readout',1030,175,214,190,'Mean readout\nInverse target scaling\nHourly attenuation\nin dB',{fill:C.lightgreen,stroke:C.green,size:25});
 arrow(s,inp,prep);arrow(s,prep,container);arrow(s,container,pred);
 text(s,'protocol-label',38,481,1200,40,'Task adaptation and model selection',28,{bold:true});
 box(s,'rfe-protocol',36,540,377,156,'Independent RFE paths\n156 → 1 and 50 → 1\nTraining-only, 3 date-group folds\nOne feature removed per round',{fill:C.pale,stroke:C.line,size:24});
 box(s,'validation-protocol',450,540,377,156,'Validation criterion\nOverall, rain and overcast MSE\nEqually weighted in the criterion\nFixed 54 training epochs',{fill:C.pale,stroke:C.line,size:24});
 box(s,'frozen-transfer-protocol',864,540,377,156,'Frozen external evaluation\nLocal weights and preprocessing\nLocal retained feature sets\nUK inference without refitting',{fill:C.pale,stroke:C.line,size:24});
 text(s,'tabm-scope-note',40,741,1200,37,'The 32 TabM members form the hourly point estimate; they are distinct from the 32 second-scale stochastic realizations.',21,{color:C.grey});
 s.speakerNotes.text='Fig05. Frozen TabM config: width=256, blocks=3, k=32, dropout=0.1, embedding=0, gate=false, lr=0.002, weight_decay=0.01, epochs=54, uniform training weights. Candidate construction, preprocessing, weather-aware validation objective and RFE are task adaptations; no new TabM gate/backbone is claimed. Direct input: 04_Source_Data/Table03_model_configs.json. Upstream: Code/Hourly_Attenuation_Experiments/Revision_032/run.py and runtime.py, Hourly032 manifest/protocol, Hourly041 protocol; see SOURCE_MANIFEST and UPSTREAM_CODE_MANIFEST. MSE criterion sqrt((MSEoverall+MSErain+MSEovercast)/3), distinct from loss weighting. Shared blocks depicted conceptually, not individual tensor operations.';
}
// Figure 9: frozen local assets, two streams, ranks, shift and dB.
{
 const s=slide('Weather-conditioned generation of second-scale fluctuations');
 const cdf=box(s,'frozen-weather-cdf',40,136,330,144,'Frozen LOCAL training assets\nWeather-conditioned\nempirical power-ratio CDF',{fill:C.lightgreen,stroke:C.green,size:25});
 const library=box(s,'frozen-block-library',40,354,330,144,'Continuous-block library\nWeather-matched training hours\nQ = Psecond / Phour',{fill:C.lightgreen,stroke:C.green,size:23});
 const draws=box(s,'draw-amplitudes',440,136,330,144,'Sample amplitudes\n3600 independent draws\nfrom the weather CDF',{fill:C.pale,stroke:C.blue,size:25});
 const template=box(s,'make-rank-template',440,354,330,144,'Time template: 3600 s\nDraw continuous 256-s blocks\n14 full blocks + final 16 s',{fill:C.pale,stroke:C.blue,size:23});
 const rank=box(s,'rank-reordering',880,230,355,176,'Rank reordering\nSort the 3600 new values\nPlace them in the positions\nof the template ranks',{fill:'#FCF3E9',stroke:C.orange,size:26});
 arrow(s,cdf,draws);arrow(s,library,template);arrow(s,draws,rank,'right','top','elbow');arrow(s,template,rank,'right','left','elbow');
 const shift=box(s,'random-circular-shift',880,554,355,92,'Random circular shift',{fill:C.pale,stroke:C.blue,size:26});
 const db=box(s,'power-ratio-to-db',440,554,330,92,'Convert to dB\nF = -10 log10(Q)',{fill:C.pale,stroke:C.blue,size:27});
 const repeat=box(s,'32-independent-realizations',40,554,330,92,'Repeat independently\n32 realizations per hour',{fill:C.lightgreen,stroke:C.green,size:26});
 arrow(s,rank,shift,'bottom','top');arrow(s,shift,db,'left','right');arrow(s,db,repeat,'left','right');
 box(s,'evaluation-and-combination',40,693,1195,78,'Evaluate amplitude and temporal statistics separately.\nFor the combined model, add each generated F to the frozen TabM k = 10 hourly estimate.',{fill:'#F6F8F9',stroke:C.line,size:23});
 s.speakerNotes.text='Fig09. Sources: Code/Fast_Fading_Experiments/Revision_003/evaluate.py and common.py, and Code/Fast_Fading_Experiments/Revision_002/run_dynamics.py plus block-selection evidence; frozen generator asset from LOCAL train. Equal mass per training hour and within-hour weighting by valid record count. Select an eligible training hour and a contiguous 256-s block; repeat to3600, last block16. Independent amplitude draws reordered by rank template; circular shift; F=-10log10(Q). Repeat32 separately. No UK CDF/block fitting. Do not force each generated curve to mean Q=1 or mean F=0. Combined A=TabM_k10_hour+F; hourly point estimate has no second-to-second information. Sources and SHAs in SOURCE_MANIFEST and UPSTREAM_CODE_MANIFEST.';
}
const names=['Fig02_Stratified_Atmospheric_Information','Fig05_TabM_Architecture_And_Protocol','Fig09_Second_Scale_Generation_Procedure'];
const slides=pres.slides.items;
for(let i=0;i<3;i++){
 const s=slides[i];
 const png=await pres.export({slide:s,format:'png',scale:1.2});
 await fs.writeFile(path.join(out,'05_QA/previews',names[i]+'.png'),new Uint8Array(await png.arrayBuffer()));
 const pdf=await pres.export({slide:s,format:'pdf'});
 await fs.writeFile(path.join(build,names[i]+'_raw.pdf'),new Uint8Array(await pdf.arrayBuffer()));
 console.log('Rendered diagram '+names[i]);
}
const candidate=path.join(build,'editable_diagrams_candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
await fs.writeFile(path.join(out,'05_QA/diagram_inventory.json'),JSON.stringify(names.map((n,i)=>({id:n,slide:i+1,generator:'build_diagrams.mjs',files:['01_Figures_PDF/'+n+'.pdf','03_Editable_PPTX/Fig02_Fig05_Fig09_Editable.pptx'],inputs:['04_Source_Data/Table03_model_configs.json','04_Source_Data/methods/','05_QA/SOURCE_MANIFEST.json'],method:'Native editable shapes/text; sources in slide speaker notes. Vector PDF exported by Artifact Tool and resized without rasterization.'})),null,2));
if(process.argv.includes('--finalize')){
 const SKILL='C:/Users/SDR_HH/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
 const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
 // This Windows workspace does not support the finalizer's hard-link publication.
 // Preserve its exclusive-create and subsequent byte-hash verification by using
 // COPYFILE_EXCL only for the named, already-validated staged presentation.
 const originalLink=fs.link;
 fs.link=async(source,target)=>{try{return await originalLink(source,target);}catch(e){
  if(process.platform==='win32' && e.code==='EPERM' && String(source).endsWith('.validated') && path.resolve(target)===path.resolve(out,'03_Editable_PPTX/Fig02_Fig05_Fig09_Editable.pptx')){
   await fs.copyFile(source,target,fsConstants.COPYFILE_EXCL);return;
  }throw e;
 }};
 const result=await finalizePresentation({explicitTotalSlideCount:3,workspaceDir:root,candidatePath:candidate,
 finalPath:path.join(out,'03_Editable_PPTX/Fig02_Fig05_Fig09_Editable.pptx'),
 pythonExecutable:'C:/Users/SDR_HH/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu',`${1280*9525},${800*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],
 requiredNativeTableOwnerSlides:[],fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation_final.json')});
 console.log(JSON.stringify(result));
}
