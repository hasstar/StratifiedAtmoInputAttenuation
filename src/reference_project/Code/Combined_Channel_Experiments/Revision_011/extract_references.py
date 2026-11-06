"""Offline PDF reference extraction, tables, and visual audit pages."""
from support import *
import re
from pypdf import PdfReader
import pypdfium2 as pdfium

def main():
    offline()
    with lock():
        manifest=read(OUT/'references_complete.json')
        need(manifest['status']=='PASS','References incomplete')
        for r in manifest['records']: need(sha(OUT/'references'/r['name'])==r['sha256'],'Reference SHA')
        pages={'P676_13.pdf':[4,5,8,9,10,11,27,29,30], 'P840_9.pdf':[4,5],
               'P453_14.pdf':[3,4], 'P835_6.pdf':[2,3,5]}
        texts={}
        for name,indices in pages.items():
            doc=PdfReader(OUT/'references'/name)
            texts[name]={str(i+1):doc.pages[i].extract_text(extraction_mode='layout') for i in indices}
        oxygen=[]
        body=texts['P676_13.pdf']['10']+'\n'+texts['P676_13.pdf']['11'].split('TABLE 2')[0]
        for line in body.replace('–','-').replace('−','-').splitlines():
            tokens=line.split()
            if len(tokens)==7 and re.fullmatch(r'\d+\.\d{6}',tokens[0]):
                oxygen.append([float(x) for x in tokens])
        need(len(oxygen)==44 and oxygen[0][0]==50.474214 and oxygen[-1][0]==834.145546,'Oxygen table parse identity')
        emit('references/oxygen_table_from_P676_13.json',dict(pdf_sha256=sha(OUT/'references/P676_13.pdf'),pdf_pages=[10,11],rows=oxygen))
        emit('references/formula_excerpts.json',texts)
        for name,indices in {'P676_13.pdf':[8,9,11,27,29], 'P840_9.pdf':[5], 'P453_14.pdf':[3]}.items():
            doc=pdfium.PdfDocument(str(OUT/'references'/name))
            for i in indices:
                doc[i].render(scale=1.5).to_pil().save(new(OUT/'references'/f'{Path(name).stem}_page{i+1}.png'))
        print('Reference extraction PASS: 44 oxygen lines and seven audit images')

if __name__=='__main__': main()
