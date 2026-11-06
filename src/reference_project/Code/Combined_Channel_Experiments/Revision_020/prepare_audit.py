"""Read-only source guard and PDF-page extraction for Revision020; no model imports."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_020'
LIT = ROOT / 'Outputs/Literature_Retrieval/IEEE_Retry_20261017_001'
AUTO = 'C:\\Users\\SDR_HH\\.codex\\automations\\tvt\\automation.toml'
AUTO_CURRENT = 'bc5a33cdf2b49fe183de87cec7ec9cb1ac9a1a351c4f1691557c7587bf10891e'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def main():
    OUT.mkdir(exist_ok=False)
    old = read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_011/sources_before.json')
    baseline, exceptions = {}, []
    for name, expected in old.items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        actual = sha(path)
        if actual != expected:
            if str(path).lower() != AUTO.lower() or actual != AUTO_CURRENT:
                raise ValueError('Unexpected old source change: '+str(path))
            exceptions.append({'path':str(path),'historical_sha256':expected,'current_sha256':actual,
                               'authorization':'Outputs/Automation_Changes/TVT_Comparison_20261017_001 and 002'})
        baseline[str(path)] = actual
    rev11 = ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'
    contract = read(rev11/'calculation_contract.json')
    for name, expected in contract['code_sha256'].items():
        path=ROOT/'Code/Combined_Channel_Experiments/Revision_011'/name
        if sha(path)!=expected: raise ValueError('Revision011 code changed: '+name)
        baseline[str(path)]=expected
    for name, expected in read(rev11/'final_verification.json')['prediction_sha256'].items():
        path=rev11/name
        if sha(path)!=expected: raise ValueError('Revision011 prediction changed: '+name)
        baseline[str(path)]=expected
    extra = [ROOT/'Tasks/T012_Combined_Channel_And_External_Validation.md',
             ROOT/'Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md',
             Path('E:/0_Workflow_System/README.md')]
    for revision in ('011','012','013','014','015','017','018','019'):
        folder=ROOT/f'Outputs/Experiments/Combined_Channel_Revision_{revision}'
        extra.extend(p for p in folder.glob('*') if p.is_file() and p.suffix in ('.md','.json'))
    # Stable LOCAL availability ledger/QA, not raw-data processing or new eligibility.
    extra.extend((ROOT/'Outputs/Experiments/Combined_Channel_Revision_013/attempt_002').glob('*.json'))
    extra.extend(p for p in LIT.rglob('*') if p.is_file())
    for path in extra: baseline[str(path)]=sha(path)
    for item in read(LIT/'manifest.json')['papers']:
        if sha(LIT/item['file'])!=item['sha256_expected']: raise ValueError('PDF source SHA changed')
    save(OUT/'sources_before.json', baseline)
    save(OUT/'reuse_check.json', {'utc':datetime.now(timezone.utc).isoformat(),
       'rev011_historical_sources':len(old),'unchanged_historical_sources':len(old)-len(exceptions),
       'authorized_exceptions':exceptions,'rev011_code_and_predictions_match':True,
       'old_numerical_results_recomputed':False,'protected_sources':len(baseline)})
    pages=OUT/'paper_pages'; pages.mkdir()
    selection={'E_SST_2020_IEEE.pdf':[5,6], 'NWP_2024_Model_Chain.pdf':[4],
               'EuCAP_2024_NWP_PRM.pdf':[2,3]}
    page_records=[]
    for name, numbers in selection.items():
        source=LIT/name
        text=PdfReader(source); doc=pdfium.PdfDocument(str(source))
        for number in numbers:
            stem=source.stem+f'_p{number:02d}'
            with (pages/(stem+'.txt')).open('x',encoding='utf-8') as f:
                f.write(text.pages[number-1].extract_text() or '')
            page=doc[number-1]; bitmap=page.render(scale=1.7)
            im=bitmap.to_pil(); im.save(pages/(stem+'.png')); im.close(); bitmap.close(); page.close()
            page_records.append({'source':str(source),'page_1_based':number,
                                 'png':str((pages/(stem+'.png')).relative_to(OUT)),
                                 'png_sha256':sha(pages/(stem+'.png'))})
        doc.close()
    save(OUT/'paper_pages.json',page_records)
    print(json.dumps({'status':'PREPARED_READ_ONLY_FORMULA_AUDIT', 'sources':len(baseline),
                      'rendered_pages':len(page_records),'output':str(OUT)}))

if __name__=='__main__': main()
