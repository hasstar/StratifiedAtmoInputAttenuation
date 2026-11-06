"""Extract original manuscript text and embedded photography, preserving provenance."""
from pathlib import Path
import json
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'最终提交的修改材料/论文图表_20261030_001'
dest=OUT/'04_Source_Data/original_manuscript'
dest.mkdir(parents=True,exist_ok=True)
r=PdfReader(ROOT/'Materials/Original_Submission/English_2026_TVT.pdf')
text=[];rows=[]
for i,p in enumerate(r.pages):
 text.append(f'\nPAGE {i+1}\n'+p.extract_text())
 if i<5:
  for j,im in enumerate(p.images):
   out=dest/f'page_{i+1:02d}_image_{j:02d}_{im.name}'
   out.write_bytes(im.data)
   rows.append({'page':i+1,'file':out.name,'size':im.image.size})
   print(rows[-1])
(dest/'original_text.txt').write_text('\n'.join(text),encoding='utf-8')
(dest/'image_inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('\n'.join(text[:4]))
