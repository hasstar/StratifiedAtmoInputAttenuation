"""Local draft evidence registries; never assert human verification/submission readiness."""
import csv
import hashlib
import json
import re
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[3]
    base=root/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
    out=base/'report_002'
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    def emit(name,obj):
        with (out/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    sources={
      'E001':['scope_manifest.json'],
      'E002':['field_audit_001/actual_field_qa.json','field_audit_001/negative_controls.json'],
      'E003':['evaluation_002/native_label_qa.json','evaluation_002/native_label_source_evidence.json'],
      'E004':['evaluation_002/hourly_metrics.json','evaluation_002/inference_receipt.json'],
      'E005':['evaluation_independent_qa_001/qa.json'],
      'E006':['evaluation_002/PROTOCOL.md','evaluation_002/compatibility_gate.json'],
      'E007':['report_002/figures/figure_provenance.json','report_002/figures/plotted_primary_hours.csv','report_002/figures/gbs_hourly_validation.png','report_002/figures/gbs_hourly_validation.pdf'],
      'E008':['stop_verification.json','handoff.json'],
      'E009':['evaluation_001/failure.json'],
      'E010':['handoff_012.json'],
    }
    entries=[]
    for identity,paths in sources.items():
        entries.append({'evidence_id':identity,'title':paths[0],'source_type':'other','authors':[],
                        'year':2026,'confidentiality':'internal','identifiers':{},'locator':'Local named artifact and exact JSON fields/report sections',
                        'verification':{'source_opened':False,'status':'unverified','verified_by':'','verified_on':''},
                        'automated_checks':'Opened and checked locally by Codex; not named human verification',
                        'files':[{'path':str((base/p).relative_to(root)),'sha256':sha(base/p)} for p in paths]})
    emit('source_manifest.json',{'schema_version':'1.0','sources':entries,'submission_ready':False,'human_scientific_review':'pending'})
    text=(out/'RESULTS_ZH.md').read_text(encoding='utf-8')
    claims=[]
    section=''
    for line in text.splitlines():
        if line.startswith('## '):section=line[3:]
        claim=re.search(r'\[claim:(C\d+)\]',line)
        if not claim:continue
        evidence=re.search(r'\[evidence:([^]]+)\]',line)
        assert evidence and all(e in sources for e in evidence[1].split(','))
        clean=re.sub(r'\[(?:claim|evidence):[^]]+\]','',line).strip()
        normalized=' '.join(clean.split())
        claims.append({'claim_id':claim[1],'section':section,'claim_kind':'result' if section=='小时评分' else 'factual',
                       'claim_text_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'evidence_ids':evidence[1],
                       'verification_status':'unverified','uncertainty':'Human scientific review pending; automated checks are not human approval',
                       'analysis_intent':'descriptive'})
    assert len(claims)==18 and len({c['claim_id'] for c in claims})==18
    with (out/'claims.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(claims[0]));writer.writeheader();writer.writerows(claims)
    metrics=read(base/'evaluation_002/hourly_metrics.json')
    facts=[]
    for site,row in metrics['primary_coverage_ge_95'].items():
        for metric in ['n_station_hours','rmse_db','mae_db','bias_db']:
            value=row[metric]
            facts.append({'fact_id':f'N{len(facts)+1:03}','concept':site+'_'+metric,
                          'value':value,'unit':'station-hours' if metric=='n_station_hours' else 'dB',
                          'analysis_set':'finite native target and >=95% native second coverage',
                          'section':'小时评分','sample_size':row['n_station_hours'],'denominator':None,'numerator':None,
                          'evidence_ids':['E004','E005']})
    facts += [{'fact_id':'N013','concept':'all_weather_complete_station_hours','value':1872,'unit':'station-hours',
               'analysis_set':'frozen 2-station/39-date scope','section':'范围与分析样本','sample_size':1872,'denominator':None,'numerator':None,'evidence_ids':['E001','E002']}]
    emit('consistency_manifest.json',{'schema_version':'1.0','methods':[{'method_id':'M001','name':'Frozen hourly zero-training prediction and descriptive error calculation',
               'analysis_intent':'descriptive','protocol_status':'prespecified','outcome_ids':['O001']}],
               'numeric_facts':facts,'results':[{'result_id':'R001','method_id':'M001','outcome_id':'O001','analysis_intent':'descriptive',
               'sample_size':1847,'reported_sections':['小时评分'],'evidence_ids':['E004','E005']}]})
    # Reconcile the displayed table with immutable independently checked metrics.
    for row in metrics['primary_coverage_ge_95'].values():
        assert f"{row['rmse_db']:.4f}" in text and f"{row['mae_db']:.4f}" in text and f"{row['bias_db']:+.4f}" in text
    figure=pd_data=read(out/'figures/figure_provenance.json')
    with (out/'figures/plotted_primary_hours.csv').open(encoding='utf-8',newline='') as f:plot_rows=list(csv.DictReader(f))
    assert len(plot_rows)==1847 and figure['all_eligible_points_plotted']
    emit('figure_visual_review.json',{
       'rounds':2,'reviewer':'Codex visual inspection, not human approval','status':'PASS_LOCAL_RENDER_REVIEW',
       'enumerated_elements':['GBS 20.7 GHz frozen Top10 BiLSTM title','Chilbolton/Sparsholt panels','native attenuation x-axis','prediction dB y-axis',
                              'all 914/933 primary points','dashed 1:1 references','n/RMSE/Bias annotations','1847-hour and approximation footer'],
       'round1_issue':'Metric annotation overlapped some high-prediction points. Original render/code retained.',
       'round2_resolution':'Moved metric annotation to blank right-hand area; no data or axis changes.',
       'data_integrity':'Same full axis range in both panels; all eligible data retained; no fitted line or significance claim.',
       'accessibility':'Separate station panels and dashed reference encode meaning without color dependence; labels readable.',
       'optional_improvement':'A separate near-zero inset could reveal dense overlap, but the current full-range display intentionally retains extremes.'})
    emit('draft_scope_and_approvals.json',{'document_type':'internal engineering external-validation stage report',
               'submission_ready':False,'human_approval':'pending','authorship_and_journal_declarations':'not applicable to this internal handoff; not generated',
               'clinical_reporting_guideline':'not applicable; not a human/animal clinical study or systematic review',
               'external_disclosure':'none; source material processed locally','skill_effect':'scientific-writing evidence/uncertainty binding; matplotlib full-range accessible rendering',
               'missing_scientific_gates':['weather stratification applicability','frozen fast-component external applicability','other model-family compatibility'],
               'scope_reduction_question':'User explicitly chose continued weather/fast compatibility audit; T012 remains open'})
    emit('report_delivery_checks.json',{'status':'PASS_INTERNAL_STAGE_REPORT_MACHINE_CHECKS','claims':18,
               'all_claims_bound_to_existing_local_evidence':True,'displayed_metrics_match':True,'figure_points':1847,
               'human_verification':'pending','T012_complete':False,'report_sha256':sha(out/'RESULTS_ZH.md'),
               'bundler_sha256':sha(__file__),'sources_unchanged_after_read':all(sha(root/p['path'])==p['sha256'] for e in entries for p in e['files'])})
    print(json.dumps({'status':'PASS_INTERNAL_STAGE_REPORT_MACHINE_CHECKS','claims':len(claims),'human_review':'pending'}))


if __name__=='__main__':main()
