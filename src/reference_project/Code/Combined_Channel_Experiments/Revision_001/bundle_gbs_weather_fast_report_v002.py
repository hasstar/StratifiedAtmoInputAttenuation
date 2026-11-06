"""Schema-only correction of the internal report registry; frozen results stay read-only."""
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OLD=BASE/'report_003'
OUT=BASE/'report_004'
SKILL=Path('C:/Users/SDR_HH/.codex/skills/scientific-writing/scripts')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def emit(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def clean(line):
    line=re.sub(r'\[(?:claim|evidence):[^]]+\]','',line)
    line=re.sub(r'<!--\s*-->','',line)
    return ' '.join(line.strip().split())
def check_script(script,args):
    run=subprocess.run([sys.executable,'-B',str(SKILL/script),*map(str,args)],capture_output=True,text=True,encoding='utf-8')
    return {'exit_code':run.returncode,'result':json.loads(run.stdout),'stderr':run.stderr,'script_sha256':sha(SKILL/script)}


def main():
    old_checks=read(OLD/'report_delivery_checks.json')
    assert sha(OLD/'RESULTS_ZH.md')==old_checks['report_sha256']
    old_text=(OLD/'RESULTS_ZH.md').read_text(encoding='utf-8')
    new_text=(OUT/'RESULTS_ZH.md').read_text(encoding='utf-8')
    # Numerical/prose content before the added metadata note must match the original.
    new_original=new_text.split('\n## 登记格式修订说明')[0]
    a=[clean(x) for x in old_text.splitlines() if clean(x)]
    b=[clean(x).replace('../report_003/figures/','figures/') for x in new_original.splitlines() if clean(x)]
    assert a==b, 'Only inline markers/figure relative link may change'
    assert (OUT/'TABLES_ZH.md').read_text(encoding='utf-8').rstrip()==(OLD/'TABLES_ZH.md').read_text(encoding='utf-8').rstrip()
    legacy_sources=read(OLD/'source_manifest.json')
    files=[];entries=[]
    for source in legacy_sources['sources']:
        for item in source['files']:
            assert sha(ROOT/item['path'])==item['sha256']
            files.append({'evidence_id':source['evidence_id'],**item})
        entry={key:source[key] for key in ('evidence_id','title','source_type','authors','year','locator')}
        entry['confidentiality']='restricted'
        entry['identifiers']={'doi':'','isbn':'','pmcid':'','pmid':'','url':(ROOT/source['files'][0]['path']).as_uri()}
        entry['locator']='Exact local artifact fields; full paths/SHA256 in evidence_files.json for '+entry['evidence_id']
        entry['verification']={'source_opened':True,'status':'unverified','verified_by':'','verified_on':''}
        entries.append(entry)
    emit('source_manifest.json',{'schema_version':'1.0','sources':entries})
    emit('evidence_files.json',{'files':files,'source_opened_by':'Codex local automated read','named_human_review':'pending',
                              'old_schema_preserved':str(OLD.relative_to(ROOT)),'new_training_or_numeric_recalculation':False})
    claims=[];section='声明'
    for line in new_text.splitlines():
        if line.startswith('## '):section=line[3:]
        match=re.search(r'\[claim:(C\d+)\]',line)
        if not match:continue
        evidence=re.search(r'\[evidence:([^]]+)\]',line)
        ids=evidence[1].split(',')
        assert set(ids).issubset({e['evidence_id'] for e in entries})
        claims.append({'claim_id':match[1],'section':section,'claim_kind':'numeric' if line.startswith('|') else 'factual',
                       'claim_text_sha256':hashlib.sha256(clean(line).encode()).hexdigest(),'evidence_ids':';'.join(ids),
                       'verification_status':'unverified','uncertainty':'not_estimated','analysis_intent':'descriptive'})
    assert len(claims)==51 and len({r['claim_id'] for r in claims})==51
    with (OUT/'claims.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(claims[0]));writer.writeheader();writer.writerows(claims)
    emit('consistency_manifest.json',read(OLD/'consistency_manifest.json'))
    emit('figure_visual_review.json',read(OLD/'figure_visual_review.json'))
    emit('draft_scope_and_approvals.json',read(OLD/'draft_scope_and_approvals.json'))
    old_claim_check=check_script('audit_claims.py',[OLD/'RESULTS_ZH.md',OLD/'claims.csv',OLD/'source_manifest.json'])
    old_source_check=check_script('validate_manifest.py',[OLD/'source_manifest.json','--kind','source'])
    emit('legacy_registry_checks.json',{'claims':old_claim_check,'sources':old_source_check,'numeric_results_affected':False})
    source_check=check_script('validate_manifest.py',[OUT/'source_manifest.json','--kind','source'])
    consistency=check_script('check_consistency.py',[OUT/'consistency_manifest.json'])
    claim_check=check_script('audit_claims.py',[OUT/'RESULTS_ZH.md',OUT/'claims.csv',OUT/'source_manifest.json'])
    emit('source_schema_check.json',source_check)
    emit('consistency_check.json',consistency)
    emit('claim_publication_gate.json',claim_check)
    assert source_check['exit_code']==0 and consistency['exit_code']==0
    allowed={'CLAIM_NOT_VERIFIED','UNVERIFIED_CLAIM_EVIDENCE','UNVERIFIED_CITATION_MARKER'}
    assert claim_check['result']['issues'] and all(x['code'] in allowed for x in claim_check['result']['issues'])
    emit('report_delivery_checks.json',{'status':'PASS_INTERNAL_REPORT_STRUCTURE_AND_NUMERIC_CONSISTENCY',
      'claims':len(claims),'numeric_facts':171,'source_schema_pass':True,'numeric_consistency_pass':True,
      'claim_publication_gate':'NOT_PASSED_UNVERIFIED_HUMAN_REVIEW_ONLY','claim_format_errors_remaining':0,
      'numerical_content_identical_to_report_003':True,'source_hashes_unchanged':all(sha(ROOT[p['path']])==p['sha256'] for p in files),
      'T012_complete':False,'submission_ready':False,'named_human_review':'pending','report_sha256':sha(OUT/'RESULTS_ZH.md'),
      'supplement_sha256':sha(OUT/'TABLES_ZH.md'),'bundler_sha256':sha(__file__),
      'original_numeric_checks':{'path':str((OLD/'report_delivery_checks.json').relative_to(ROOT)),'sha256':sha(OLD/'report_delivery_checks.json')}})
    print(json.dumps({'status':'PASS_INTERNAL_REPORT_STRUCTURE_AND_NUMERIC_CONSISTENCY','claims':51,'numeric_facts':171,
      'publication_gate':'NOT_PASSED_UNVERIFIED_HUMAN_REVIEW_ONLY','T012_complete':False}))


if __name__=='__main__':main()
