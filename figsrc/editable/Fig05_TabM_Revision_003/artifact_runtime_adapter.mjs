/** Compatibility bridge for this bundled runtime's missing #standard-fonts aliases.
 * Uses existing Adobe AFM metrics distributed with matplotlib, without modifying
 * installed packages, installing dependencies, or altering scientific data.
 * Only the current Node process is affected. Native Arial remains the slide font.
 */
import fs from 'node:fs';
import path from 'node:path';
import Module from 'node:module';
const dir=process.env.CODEX_PDF_AFM_DIR || 'C:/Users/SDR_HH/AppData/Roaming/Python/Python312/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts';
const load=Module._load;const cache=new Map();
Module._load=function(request,parent,isMain){
 if(!request.startsWith('#standard-fonts/'))return load.apply(this,arguments);
 if(cache.has(request))return cache.get(request);
 const id=request.split('/')[1];const file=fs.readdirSync(dir).find(n=>n.replace('.afm','').replaceAll('-','')===id);
 if(!file)throw new Error('Missing existing AFM resource '+id);
 const str=fs.readFileSync(path.join(dir,file),'utf8');const lines=str.split(/\r?\n/);
 const prop=k=>lines.find(l=>l.startsWith(k+' '))?.slice(k.length+1);
 const glyph=[],width=[];const pairs=[];
 for(const l of lines){
  if(l.startsWith('C ')){const n=l.match(/; N (\S+) ;/),w=l.match(/; WX ([-\d.]+) ;/);if(n&&w){glyph.push(n[1]);width.push(Number(w[1]));}}
  if(l.startsWith('KPX ')){const [,a,b,v]=l.split(/\s+/);pairs.push([a,b,Number(v)]);}
 }
 const idx=new Map(glyph.map((g,i)=>[g,i])),groups=new Map();
 for(const [a,b,v]of pairs){if(idx.has(a)&&idx.has(b)){if(!groups.has(v))groups.set(v,[]);groups.get(v).push(idx.get(a)*glyph.length+idx.get(b));}}
 const kern=[];for(const [v,nums]of groups){nums.sort((a,b)=>a-b);let prior=0;kern.push(v,nums.map(n=>{const delta=n-prior;prior=n;return delta;}));}
 const metrics={name:prop('FontName'),bbox:prop('FontBBox').split(/\s+/).map(Number),ascender:Number(prop('Ascender')||0),descender:Number(prop('Descender')||0),xHeight:Number(prop('XHeight')||0),capHeight:Number(prop('CapHeight')||0),glyphNames:glyph.join(' '),glyphWidths:width,kernPairs:kern};
 cache.set(request,metrics);return metrics;
};
