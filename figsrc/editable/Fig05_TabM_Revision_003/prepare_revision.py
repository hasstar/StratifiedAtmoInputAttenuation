"""Create the new figure source from the delivered editable Revision 002.
This changes layout only. No scientific implementation or result is modified.
"""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'Fig05_TabM_Revision_002'
source = (OLD / 'build_figure.mjs').read_text(encoding='utf-8')
source = source.replace('Figure 5 v002', 'Figure 5 v003')
source = source.replace('const W=1760,H=1020;', 'const W=1760,H=790;')
start = source.index('// (b)')
end = source.index('// (c)')
panel_b = source[start:end]
panel_b = '\n'.join(line for line in panel_b.splitlines()
                    if not line.startswith("text('block-equation'")
                    and not line.startswith("line('panel-b-rule'")) + '\n'
# Keep the panel title; remove the equation row and close the space beneath it.
coordinates = {541, 581, 718, 578, 704, 614, 613, 643, 617, 678, 770, 763}
panel_b = re.sub(r'\b(?:' + '|'.join(map(str, sorted(coordinates))) + r')\b',
                 lambda match: str(int(match[0])-40), panel_b)
panel_b = panel_b.replace("text('member-legend',730,", "text('member-legend',770,")
tail_start = source.index('s.speakerNotes.text=', end)
source = source[:start] + panel_b + '\n' + source[tail_start:]
note_start = source.index('s.speakerNotes.text=')
note_end = source.index('\nawait fs.writeFile', note_start)
source = source[:note_start] + "s.speakerNotes.text='Two-panel revision of Figure5. Panel(a) shows concurrent meteorological inputs, training-fitted preprocessing, 32 members, 3 blocks of width256, separate regression heads, standardized prediction averaging, inverse target scaling and memberwise MSE. Panel(b) shows shared W and member-specific R/S/B followed by ReLU and training-only dropout0.1. The network is standard TabM from the supplied original paper Fig1 and local Revision031 implementation; matrices W differ across depth. The former protocol panel and displayed equation are relocated to the aligned manuscript text in this revision. All visible elements are mapped to explanatory prose in the accompanying guide. All native shapes/text/connectors remain editable; no experiments were rerun.';" + source[note_end:]
with (HERE / 'build_figure.mjs').open('x', encoding='utf-8') as stream:
    stream.write(source)
shutil.copy2(OLD / 'artifact_runtime_adapter.mjs', HERE / 'artifact_runtime_adapter.mjs')
(HERE / 'qa').mkdir(exist_ok=True)
print('Prepared two-panel source, canvas 1760 x 790; equation and panel(c) removed.')
