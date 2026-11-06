"""Bind frozen GBS weather/fast results to a descriptive internal report; no fitting."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT = BASE / 'report_003'
EVENTS = {'overall':'总体','clear_dry':'晴/干燥','rain':'雨（含强雨子集）',
          'heavy_rain_subset':'强雨子集','dry_overcast':'干燥阴天','other':'other','unknown':'unknown'}
STATIONS = {'chilbolton':'Chilbolton','sparsholt':'Sparsholt','pooled':'两站总体'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def emit(name, obj):
    with (OUT/name).open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def metric(value, digits=6, percent=False):
    if value is None:
        return '不可评估'
    return f'{value*100:.2f}%' if percent else f'{value:.{digits}f}'


def data():
    weather = read(BASE/'conditional_evaluation_001/weather_hourly_metrics.json')
    probability = read(BASE/'conditional_evaluation_001/probability_summary.json')
    dynamics = read(BASE/'dynamic_evaluation_002/summary.json')
    assert read(BASE/'conditional_evaluation_independent_qa_001/qa.json')['status'] == 'PASS_WEATHER_AND_FROZEN_MARGINAL_EVALUATION'
    assert read(BASE/'dynamic_independent_qa_001/qa.json')['status'] == 'PASS_EXTERNAL_FROZEN_TEMPORAL_EVALUATION'
    return weather, probability, dynamics


def tables():
    weather, probability, dynamics = data()
    lines = ['# 分站与同源同频总体完整主指标', '',
             '内部草稿；具名人工科学审阅待完成。各表由已独立 QA 的冻结 JSON 展开，数值未重新拟合或选样。', '',
             '小时评分采用小时等权、原生有效秒覆盖率≥95%；概率和时间依赖采用同一覆盖门槛并要求 known 天气，主表日期等权。强雨是雨的重叠子集；unknown 不指定 CDF。总体按同日站点小时聚合后日期等权，不是两站结果简单平均。', '',
             '以下每项对应 RESULTS_ZH.md 的 E004/E005（小时和概率）、E007/E008（时间依赖）；全部精度、小时等权及有限低覆盖诊断仍保存在原 JSON。', '']
    for station, label in STATIONS.items():
        lines += [f'## {label}：小时误差', '', '| 天气 | 站点小时 | 日期 | 唯一UTC小时 | RMSE/dB | MAE/dB | Bias/dB |', '|---|---:|---:|---:|---:|---:|---:|']
        for row in weather['primary']:
            if row['station'] != station:
                continue
            lines.append(f"| {EVENTS[row['event']]} | {row['n_station_hours']} | {row['n_dates']} | {row['n_unique_utc_hours']} | {row['rmse_db']:.4f} | {row['mae_db']:.4f} | {row['bias_db']:+.4f} |")
        lines += ['', f'## {label}：边缘概率', '', '| 天气 | 站点小时 | 日期 | 基线CRPS/dB | 组合CRPS/dB | 90%覆盖 | 98%覆盖 |', '|---|---:|---:|---:|---:|---:|---:|']
        for row in probability['primary']:
            if row['station'] != station or row['weighting'] != 'date_equal':
                continue
            vals = [metric(row['deterministic_slow_instant_mae_db']),metric(row['combined_crps201_db']),
                    metric(row['combined_central90_coverage'],percent=True),metric(row['combined_central98_coverage'],percent=True)]
            lines.append(f"| {EVENTS[row['event']]} | {row['n_station_hours']} | {row['n_dates']} | " + ' | '.join(vals) + ' |')
        lines += ['', f'## {label}：时间依赖', '', '| 天气 | 站点小时 | 日期 | IID32/dB² | block32/dB² | IID16/dB² | block16/dB² |', '|---|---:|---:|---:|---:|---:|---:|']
        for event, label_event in EVENTS.items():
            selected = {r['candidate']:r for r in dynamics['rows'] if r['subset']=='primary' and r['station']==station and r['event']==event and r['weighting']=='date_equal'}
            assert set(selected)=={'iid','block_0256'}
            a,b=selected['iid'],selected['block_0256']
            assert a['n_station_hours']==b['n_station_hours'] and a['n_dates']==b['n_dates']
            vals=[metric(a['vs32']),metric(b['vs32']),metric(a['vs16']),metric(b['vs16'])]
            lines.append(f"| {label_event} | {a['n_station_hours']} | {a['n_dates']} | " + ' | '.join(vals) + ' |')
        lines += ['']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--print-tables',action='store_true')
    args=parser.parse_args()
    if args.print_tables:
        print(json.dumps({'tables_markdown':tables()},ensure_ascii=False))
        return
    weather, probability, dynamics=data()
    assert (OUT/'TABLES_ZH.md').read_text(encoding='utf-8').rstrip()==tables().rstrip()
    sources={
       'E001':['scope_manifest.json','field_audit_001/actual_field_qa.json'],
       'E002':['evaluation_002/hourly_metrics.json','evaluation_002/hourly_predictions.parquet','evaluation_002/compatibility_gate.json','evaluation_independent_qa_001/qa.json'],
       'E003':['conditional_fast_001/weather_availability_qa.json','conditional_fast_001/weather_labels.parquet','conditional_fast_001/fast_compatibility_gate.json','conditional_fast_independent_qa_001/qa.json'],
       'E004':['conditional_evaluation_001/weather_hourly_metrics.json','conditional_evaluation_001/probability_summary.json','conditional_evaluation_001/probability_hourly.parquet','conditional_evaluation_001/completion.json'],
       'E005':['conditional_evaluation_independent_qa_001/qa.json'],
       'E006':['conditional_fast_001/PROTOCOL.md','dynamic_evaluation_002/PROTOCOL.md'],
       'E007':['dynamic_evaluation_002/completion.json','dynamic_evaluation_002/summary.json','dynamic_evaluation_002/hourly.parquet','dynamic_evaluation_002/shards.json'],
       'E008':['dynamic_independent_qa_001/qa.json'],
       'E009':['report_003/figures/figure_provenance.json','report_003/figures/plotted_probability_summary.csv','report_003/figures/gbs_probability_validation.png','report_003/figures/gbs_probability_validation.pdf'],
       'E010':['dynamic_evaluation_001/failure.json'],
       'E011':['../weather_top10_los_v004/model_inventory.json'],
       'E012':['handoff_012.json'],
    }
    entries=[]
    for identity, paths in sources.items():
        entries.append({'evidence_id':identity,'title':paths[0],'source_type':'other','authors':[],'year':2026,
          'confidentiality':'internal','identifiers':{},'locator':'Exact local artifacts, fields and immutable hashes',
          'verification':{'source_opened':False,'status':'unverified','verified_by':'','verified_on':''},
          'automated_checks':'Opened locally and independently machine checked; not named human verification',
          'files':[{'path':str((BASE/p).resolve().relative_to(ROOT)),'sha256':sha(BASE/p)} for p in paths]})
    emit('source_manifest.json',{'schema_version':'1.0','sources':entries,'submission_ready':False,'human_scientific_review':'pending'})
    text=(OUT/'RESULTS_ZH.md').read_text(encoding='utf-8')
    claims=[]; section=''
    for line in text.splitlines():
        if line.startswith('## '): section=line[3:]
        match=re.search(r'\[claim:(C\d+)\]',line)
        if not match: continue
        evidence=re.search(r'\[evidence:([^]]+)\]',line)
        assert evidence and all(x in sources for x in evidence[1].split(','))
        clean=' '.join(re.sub(r'\[(?:claim|evidence):[^]]+\]','',line).strip().split())
        claims.append({'claim_id':match[1],'section':section,'claim_kind':'result' if section in ['本轮结论','分天气小时误差','冻结快变边缘概率评价','时间依赖与增量评分'] else 'factual',
                       'claim_text_sha256':hashlib.sha256(clean.encode()).hexdigest(),'evidence_ids':evidence[1],
                       'verification_status':'unverified','uncertainty':'Named human scientific review pending; descriptive external transfer only',
                       'analysis_intent':'descriptive'})
    assert len(claims)==20 and len(set(c['claim_id'] for c in claims))==20
    with (OUT/'claims.csv').open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(claims[0]));writer.writeheader();writer.writerows(claims)
    facts=[]
    def fact(concept,value,unit,n,analysis,evidence,section):
        facts.append({'fact_id':f'N{len(facts)+1:03}','concept':concept,'value':value,'unit':unit,
                      'analysis_set':analysis,'section':section,'sample_size':n,'denominator':None,'numerator':None,'evidence_ids':evidence})
    for row in weather['primary']:
        for field in ('rmse_db','mae_db','bias_db'):
            fact(f"hourly_{row['station']}_{row['event']}_{field}",row[field],'dB',row['n_station_hours'],'>=95% valid native seconds, finite truth',['E004','E005'],'分天气小时误差')
    for row in probability['primary']:
        if row['weighting']!='date_equal':continue
        for field,unit in [('combined_crps201_db','dB'),('deterministic_slow_instant_mae_db','dB'),('combined_central90_coverage','fraction'),('combined_central98_coverage','fraction')]:
            if row[field] is not None:
                fact(f"probability_{row['station']}_{row['event']}_{field}",row[field],unit,row['n_station_hours'],'known-weather primary, date equal',['E004','E005'],'冻结快变边缘概率评价')
    for row in dynamics['rows']:
        if row['subset']=='primary' and row['weighting']=='date_equal' and row['vs32'] is not None:
            fact(f"temporal_{row['station']}_{row['event']}_{row['candidate']}",row['vs32'],'dB2',row['n_station_hours'],'known-weather primary, 32 realizations, date equal',['E007','E008'],'时间依赖与增量评分')
    methods=[{'method_id':'M001','name':'Frozen descriptive weather/hourly, probability and temporal transfer audit','analysis_intent':'descriptive','protocol_status':'prespecified','outcome_ids':['O001','O002','O003']}]
    results=[{'result_id':f'R{i:03}','method_id':'M001','outcome_id':f'O{i:03}','analysis_intent':'descriptive','sample_size':n,'reported_sections':[s],'evidence_ids':e} for i,n,s,e in [(1,1847,'分天气小时误差',['E004','E005']),(2,1844,'冻结快变边缘概率评价',['E004','E005']),(3,1844,'时间依赖与增量评分',['E007','E008'])]]
    emit('consistency_manifest.json',{'schema_version':'1.0','methods':methods,'numeric_facts':facts,'results':results})
    # All displayed main-table numeric values must occur exactly at their stored precision.
    for row in weather['primary']:
        if row['station']=='pooled':
            assert f"{row['rmse_db']:.4f}" in text and f"{row['mae_db']:.4f}" in text and f"{row['bias_db']:+.4f}" in text
    for row in probability['primary']:
        if row['weighting']=='date_equal' and row['event']!='unknown' and (row['station']=='pooled' or row['event']=='overall'):
            for field in ('combined_crps201_db','deterministic_slow_instant_mae_db'): assert metric(row[field]) in text
            for field in ('combined_central90_coverage','combined_central98_coverage'): assert metric(row[field],percent=True) in text
    for row in dynamics['rows']:
        if row['subset']=='primary' and row['weighting']=='date_equal' and row['event']!='unknown' and (row['station']=='pooled' or row['event']=='overall'):
            assert metric(row['vs32']) in text
    total=next(r for r in probability['primary'] if r['station']=='pooled' and r['event']=='overall' and r['weighting']=='date_equal')
    crps_reduction=100*(1-total['combined_crps201_db']/total['deterministic_slow_instant_mae_db'])
    selected={r['candidate']:r for r in dynamics['rows'] if r['subset']=='primary' and r['station']=='pooled' and r['event']=='overall' and r['weighting']=='date_equal'}
    vs_reduction=100*(1-selected['block_0256']['vs32']/selected['iid']['vs32'])
    assert f'{crps_reduction:.2f}%' in text and f'{vs_reduction:.2f}%' in text
    plot=read(OUT/'figures/figure_provenance.json')
    assert all(sha(OUT/'figures'/name)==value for name,value in plot['outputs_sha256'].items())
    emit('figure_visual_review.json',{'status':'PASS_LOCAL_RENDER_REVIEW','rounds':1,'reviewer':'Codex visual inspection; not named human approval',
        'enumerated_elements':['two titled panels','CRPS/dB axis from zero','coverage axis and 90/98 nominal references','six finite weather categories with sample sizes','heavy rain subset label','different square/circle markers','deterministic/frozen-CDF legends','9.37% reduction and44.06% coverage footer'],
        'result':'Labels, legends and footer readable, no clipping/overlap observed; all finite prespecified groups retained.',
        'data_integrity':'Unknown is unscorable, not zero; heavy subset overlaps rain; no aggregate/model selection from performance.',
        'accessibility':'Shape and dashed/dotted reference encoding supplement color; internal PNG and vector PDF both retained.',
        'optional_improvement':'Full exact per-station values are supplied in TABLES_ZH.md rather than overcrowding this pooled figure.'})
    emit('draft_scope_and_approvals.json',{'document_type':'internal scientific external-transfer supplement','submission_ready':False,
        'named_human_review':'pending','analysis_intent':'descriptive','clinical_reporting_guideline':'not applicable',
        'external_disclosure':'none; all data processed locally','T012_complete':False,
        'remaining':['Read-only compatibility audit of other existing model families','Named human scientific review','Final task-level scope acceptance has not occurred'],
        'skill_effect':'Scientific-writing evidence and denominator binding; matplotlib scores plus calibration display; no modeling changes'})
    emit('report_delivery_checks.json',{'status':'PASS_INTERNAL_WEATHER_FAST_REPORT_MACHINE_CHECKS','claims':len(claims),'numeric_facts':len(facts),
        'main_tables_numeric_checks':True,'supplement_tables_exact_json_expansion':True,'source_sha_unchanged':all(sha(ROOT/p['path'])==p['sha256'] for e in entries for p in e['files']),
        'probability_reduction_percent':crps_reduction,'temporal_reduction_percent':vs_reduction,
        'T012_complete':False,'human_verification':'pending','report_sha256':sha(OUT/'RESULTS_ZH.md'),'supplement_sha256':sha(OUT/'TABLES_ZH.md'),'bundler_sha256':sha(__file__)})
    print(json.dumps({'status':'PASS_INTERNAL_WEATHER_FAST_REPORT_MACHINE_CHECKS','claims':len(claims),'facts':len(facts),'T012_complete':False}))


if __name__=='__main__':
    main()
