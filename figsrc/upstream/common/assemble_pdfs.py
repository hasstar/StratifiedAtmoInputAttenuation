"""Assemble vector diagram PDFs and an ordered overview; pypdf only."""
from pathlib import Path
import argparse,json,hashlib
from pypdf import PdfReader,PdfWriter
ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=ROOT/'最终提交的修改材料/论文图表_20261030_001');a=ap.parse_args();out=a.package
for f in (out/'05_QA/ppt_build').glob('*_raw.pdf'):
 r=PdfReader(f);assert len(r.pages)==1
 page=r.pages[0];page.scale_by(7.15*72/float(page.mediabox.width));w=PdfWriter();w.add_page(page)
 w.add_metadata({'/Title':f.stem.removesuffix('_raw'),'/Subject':'Native editable diagram exported to vector PDF; source PPTX supplied.'})
 with (out/'01_Figures_PDF'/(f.stem.removesuffix('_raw')+'.pdf')).open('wb') as h:w.write(h)
figs=sorted((out/'01_Figures_PDF').glob('Fig*.pdf'));assert len(figs)==14,len(figs)
writer=PdfWriter();meta=[]
for f in figs:
 reader=PdfReader(f);assert len(reader.pages)==1
 title=f.stem.replace('_',' ');writer.add_page(reader.pages[0]);writer.add_outline_item(title,len(writer.pages)-1)
 p=reader.pages[0];txt=p.extract_text() or ''
 meta.append({'file':str(f.relative_to(out)).replace('\\','/'),'pages':1,'width_pt':float(p.mediabox.width),'height_pt':float(p.mediabox.height),'text_characters':len(txt),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
writer.add_metadata({'/Title':'Manuscript figures: 10 main + 4 supplementary','/Subject':'Frozen experimental results; English figure labels; bilingual captions and provenance supplied separately.'})
with (out/'00_全部图片预览.pdf').open('wb') as h:writer.write(h)
(out/'05_QA/pdf_inventory.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
print('Assembled 14 individual PDFs and one ordered overview.')
