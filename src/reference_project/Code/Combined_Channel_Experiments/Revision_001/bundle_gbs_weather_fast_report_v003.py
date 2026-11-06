"""Finalize schema-corrected report after v002 receipt-path typo; no evaluation rerun."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OLD=BASE/'report_003'
OUT=BASE/'report_004'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(line):
    return ' '.join(re.sub(r'<!--\s*-->','',re.sub(r'\[(?:claim|evidence):[^]]+\]','',line)).strip().split())
def main():
    a=[clean(x) for x in (OLD/'RESULTS_ZH.md').read_text(encoding='utf-8').splitlines() if clean(x)]
    b=[clean(x).replace('../report_003/figures/','figures/') for x in (OUT/'RESULTS_ZH.md').read_text(encoding='utf-8').split('\n## 登记格式修订说明')[0].splitlines() if clean(x)]
    assert a==b
    assert (OUT/'TABLES_ZH.md').read_text(encoding='utf-8').rstrip()==(OLD/'TABLES_ZH.md').read_text(encoding='utf-8').rstrip()
    assert read(OUT/'source_schema_check.json')['exit_code']==0
    assert read(OUT/'consistency_check.json')['exit_code']==0
    claim_check=read(OUT/'claim_publication_gate.json')
    allowed={'CLAIM_NOT_VERIFIED','UNVERIFIED_CLAIM_EVIDENCE','UNVERIFIED_CITATION_MARKER'}
    assert claim_check['result']['issues'] and all(x['code'] in allowed for x in claim_check['result']['issues'])
    files=read(OUT/'evidence_files.json')['files']
    assert all(sha(ROOT / p['path'])==p['sha256'] for p in files)
    value={'status':'PASS_INTERNAL_REPORT_STRUCTURE_AND_NUMERIC_CONSISTENCY','claims':51,'numeric_facts':171,
      'source_schema_pass':True,'numeric_consistency_pass':True,'claim_publication_gate':'NOT_PASSED_UNVERIFIED_HUMAN_REVIEW_ONLY',
      'claim_format_errors_remaining':0,'numerical_content_identical_to_report_003':True,'source_hashes_unchanged':True,
      'T012_complete':False,'submission_ready':False,'named_human_review':'pending','report_sha256':sha(OUT/'RESULTS_ZH.md'),
      'supplement_sha256':sha(OUT/'TABLES_ZH.md'),'bundler_sha256':sha(__file__),
      'preserved_receipt_failure':{'path':'bundle_failure_v002.json','sha256':sha(OUT/'bundle_failure_v002.json')},
      'original_numeric_checks':{'path':str((OLD/'report_delivery_checks.json').relative_to(ROOT)),'sha256':sha(OLD/'report_delivery_checks.json')}}
    with (OUT/'report_delivery_checks.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':value['status'],'publication_gate':value['claim_publication_gate'],'T012_complete':False}))
if __name__=='__main__':main()
