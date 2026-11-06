"""Export the edited native figure and align the existing Section 3 source.
Reuses all comparison/result paragraphs and other figures/tables unchanged.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = HERE.parent / 'Section03_Hourly_Modeling_20261030_001'
MANUSCRIPT = HERE / 'manuscript'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

pdf = HERE / 'Fig05_TabM_Architecture_And_Protocol.pdf'
page = PdfReader(HERE / 'qa/Fig05_native_raw.pdf').pages[0]
page.scale_to(7.16*72, 7.16*72*790/1760)
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({'/Title':'TabM architecture for hourly attenuation estimation',
                     '/Subject':'Two-panel revision; protocol prose and block equation are in the manuscript'})
with pdf.open('xb') as stream:
    writer.write(stream)
doc = pdfium.PdfDocument(str(pdf))
pg = doc[0]
pg.render(scale=3.2).to_pil().save(HERE / 'qa/Fig05_final_pdf_preview.png')
pg.close()
doc.close()

prior = (OLD / 'section03.tex').read_text(encoding='utf-8')
start = prior.index('\\subsubsection{Task-adapted TabM}')
stop = prior.index('\\subsubsection{Comparison models}')
replacement = (HERE / 'tabm_architecture_text.tex').read_text(encoding='utf-8')
updated = prior[:start] + replacement.rstrip() + '\n\n' + prior[stop:]
with (MANUSCRIPT / 'section03.tex').open('x', encoding='utf-8') as stream:
    stream.write(updated)
for name in ['references.bib','supplementary_tables.tex']:
    shutil.copy2(OLD / name, MANUSCRIPT / name)
for folder in ['figures','tables']:
    shutil.copytree(OLD / folder, MANUSCRIPT / folder)
shutil.copy2(pdf, MANUSCRIPT / 'figures' / pdf.name)

ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
ppt=HERE / 'editable/Fig05_TabM_Architecture_And_Protocol_Editable.pptx'
with ZipFile(ppt) as archive:
    root=ET.fromstring(archive.read('ppt/slides/slide1.xml'))
    slide_text='\n'.join(t.text or '' for t in root.findall('.//a:t',ns))
    counts={'native_shapes':len(root.findall('.//p:sp',ns)),
            'native_connectors':len(root.findall('.//p:cxnSp',ns)),
            'images':len(root.findall('.//p:pic',ns))}
assert 'Task-specific protocol' not in slide_text
assert 'H′ = Dropout' not in slide_text
assert '\\label{eq:tabm_hidden_block}' in updated
assert updated[updated.index('\\subsubsection{Comparison models}'):] == prior[stop:]
assert len(PdfReader(pdf).pages[0].images)==0
checks={'panel_c_removed':True,'displayed_equation_removed_from_figure':True,
        'equation_added_to_body':True,'comparison_protocol_and_results_tail_unchanged':True,
        'ppt_objects':counts,'pdf_pages':1,'pdf_embedded_images':0,
        'figure_height_reduction_percent':round((1020-790)/1020*100,2),
        'visual_review':'pending','latex_compilation':'pending'}
(HERE / 'qa/content_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
sources=[OLD / 'section03.tex', HERE.parent / 'Fig05_TabM_Revision_002/build_figure.mjs',
         PROJECT / 'Code/Hourly_Attenuation_Experiments/Revision_031/core.py',
         PROJECT / 'Code/Hourly_Attenuation_Experiments/Revision_032/run.py',
         PROJECT / 'Code/Hourly_Attenuation_Experiments/Revision_041/PROTOCOL.md',
         PROJECT / '最终提交的修改材料/论文图表_20261030_001/04_Source_Data/Table03_model_configs.json']
outputs=[pdf,ppt,HERE/'tabm_architecture_text.tex',MANUSCRIPT/'section03.tex',
         HERE/'build_figure.mjs',HERE/'export_and_align.py',HERE/'prepare_revision.py']
record={'task':'T026 Revision003','created_date_bjt':'2026-10-30',
        'sources':[{'path':str(f.relative_to(PROJECT)),'sha256':sha(f)} for f in sources],
        'outputs':[{'path':str(f.relative_to(HERE)),'sha256':sha(f)} for f in outputs],
        'unchanged_section_tail_sha256':hashlib.sha256(prior[stop:].encode('utf-8')).hexdigest(),
        'change':'Remove non-structural protocol panel and displayed equation; explain all remaining nodes in the manuscript.'}
(HERE/'provenance.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
