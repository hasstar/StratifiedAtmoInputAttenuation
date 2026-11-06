"""Seal verified tables/reports, fixed-index before/after and unchanged sources."""
from common import *
import re
import traceback
import numpy as np
import pandas as pd

def main():
    offline();require(not (OUT/'delivery_verification.json').exists(),'Already sealed')
    with lock():
        qa=read(OUT/'independent_qa.json');require(qa['status']=='PASS','Independent numeric QA')
        guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
        require(sha(ROOT.parents[2]/'README.md')=='39741aa8c69136661b4b02948b29714a3654f3f989ab7b488a54f0c02f93f7c7','Previously fully read root rules unchanged')
        require(re.search(r'^status\s*=\s*"PAUSED"\s*$',AUTOMATION.read_text(encoding='utf-8'),re.M),'Automation must stay paused')
        require(read(ACQ/'STOP_REQUESTED.json')['reason']=='Direct human stop instruction','Stop marker')
        for s in read(OUT/'runtime_observation.json')['samples']:
            require(s['all_python_count']==s['missing_command_line_count']==0,'Process evidence')
        for r in read(OUT/'references_complete.json')['records']:
            p=OUT/'references'/r['file'];require(sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'Official reference unchanged')
            require(r['content_length'] is None or r['content_length']==r['bytes'],'Official reference recorded size')
        for filename,digest in read(OUT/'calculation_complete.json')['code_sha256'].items():require(sha(HERE/filename)==digest,'Executed code unchanged')
        require(sha(HERE/'verify.py')==qa['verification_code_sha256'],'Verifier unchanged')
        report=(OUT/'COMPARISON_REPORT.md').read_text(encoding='utf-8')
        require(sha(OUT/'COMPARISON_REPORT.md')==read(OUT/'report_generation.json')['report_sha256'],'Generated report unchanged')
        family=pd.read_csv(OUT/'family_summary.csv');metrics=pd.read_csv(OUT/'metrics.csv')
        require(len(metrics)==1196 and len(family)==856,'Numeric row counts')
        labels=[('ours_frozen_top10','本文 Top10 BiLSTM（冻结）'),('train_mean','LOCAL训练均值常数'),('ridge','Ridge'),('elastic_net','ElasticNet'),('svr_rbf','RBF-SVR'),
                ('random_forest','随机森林'),('xgboost','XGBoost'),('bai2022_mlp_adapted','Bai2022 MLP适配'),('bai2022_lstm_t0_adapted','Bai2022 LSTM-T0适配'),('tabm2025_official_budgeted','TabM2025官方架构/固定预算')]
        for name,label in labels:
            pair=[]
            for cohort in ('LOCAL_test','UK_all'):
                r=family.loc[family.cohort.eq(cohort)&family.group.eq('overall')&family.family.eq(name)].iloc[0]
                value=f'{r.rmse_mean_db:.4f}'
                if r.runs>1:value+=f' ± {r.rmse_seed_sd_db:.4f}'
                pair.append(value)
            require(f'| {label} | {pair[0]} | {pair[1]} | {r.mae_mean_db:.4f} | {r.bias_mean_db:+.4f} |' in report,'Main table report alignment')
        def metric(cohort,model,group='overall'):
            a=metrics.loc[metrics.cohort.eq(cohort)&metrics.model.eq(model)&metrics.group.eq(group)]
            require(len(a)==1,'Unique reported metric');return a.iloc[0]
        for local,uk,label in [('physics_geom_circular','itu838_geom_circular_nominal','P.838＋P.839几何雨衰近似'),('physics_p618_hourly_circular','itu618_proxy_circular_nominal','P.618路径缩减小时代理')]:
            l,u=metric('LOCAL_test',local),metric('UK_all',uk)
            require(f'| {label} | {l.rmse_db:.4f} | {u.rmse_db:.4f} | {u.mae_db:.4f} | {u.bias_db:+.4f} |' in report,'Physical report row')
        for suffix,label in [('minus050','P.839雨层高度−0.5 km'),('nominal','P.839原值（主情景）'),('plus050','P.839雨层高度＋0.5 km')]:
            a,b=metric('UK_all','itu838_geom_circular_'+suffix),metric('UK_all','itu618_proxy_circular_'+suffix)
            require(f'| {label} | {a.rmse_db:.4f} | {b.rmse_db:.4f} |' in report,'Sensitivity report row')
        for group,label in [('rain_below_0.1','0≤R<0.1'),('rain_0.1_to_1','0.1≤R<1'),('rain_at_least_1','R≥1')]:
            a,b,c=[metric('UK_all',name,group) for name in ('ours_frozen_top10','itu838_geom_circular_nominal','itu618_proxy_circular_nominal')]
            require(f'| {label} mm/h | {int(a.n)} | {a.rmse_db:.4f} | {b.rmse_db:.4f} | {c.rmse_db:.4f} |' in report,'Rain-band report row')
        for name,label in [('ours_frozen_top10','本文Top10'),('train_mean','LOCAL训练均值'),('elastic_net_seed17','ElasticNet'),('itu838_geom_circular_nominal','P.838几何近似'),('itu618_proxy_circular_nominal','P.618小时代理')]:
            a,b=metric('UK_original39',name),metric('UK_extension2003',name)
            require(f'| {label} | {a.rmse_db:.4f} | {b.rmse_db:.4f} |' in report,'Period report row')
        for r in read(OUT/'date_cluster_bootstrap.json')['results']:
            if r['cohort']=='UK_all':
                require(f"{r['delta_rmse_db']:+.4f}" in report and f"[{r['unadjusted_ci95_db'][0]:.4f}, {r['unadjusted_ci95_db'][1]:.4f}]" in report,'Report paired interval')
        frame=pd.read_parquet(OUT/'predictions_physical_and_reused.parquet')
        require(len(frame)==5664 and int(frame.common_score_eligible.sum())==5538 and int(frame.primary_score_eligible.sum())==5593,'Report sample sizes')
        require(frame.loc[frame.primary_score_eligible&~frame.physical_input_valid].period.eq('original39').all(),'55 negative differences in original39')
        require(frame.loc[frame.common_score_eligible].hour_utc.dt.date.nunique()==143,'Paired UTC-date count')
        sensitivity=pd.read_csv(OUT/'sensitivity_all_scenarios.csv')
        expected=metrics.loc[metrics.cohort.eq('UK_all')&metrics.group.eq('overall')&metrics.model.str.startswith('itu')].reset_index(drop=True)
        pd.testing.assert_frame_equal(sensitivity,expected,check_exact=False,rtol=1e-12,atol=1e-12)
        require(len(sensitivity)==18,'All scenarios retained')
        zero=next(r for r in read(OUT/'posthoc_zero_diagnostic.json')['records'] if r['cohort']=='UK_common')
        require(abs(np.sqrt(np.mean(frame.loc[frame.common_score_eligible].truth_db**2))-zero['rmse_db'])<1e-12,'Posthoc zero independent confirmation')
        index_evidence={};blocks=[]
        pattern=rb'<!-- REV004-BEGIN -->\n.*?<!-- REV004-END -->\n\n'
        for p in INDEXES:
            raw=p.read_bytes();matches=re.findall(pattern,raw,re.S)
            require(len(matches)==1,'One new fixed-index block')
            before=OUT/'document_snapshots/before'/p.relative_to(ROOT)
            require(re.sub(pattern,b'',raw,flags=re.S)==before.read_bytes(),'Old fixed-index bytes preserved')
            require(sha(before)==read(OUT/'index_sha256_before.json')[str(p.relative_to(ROOT))],'Before snapshot authentic')
            index_evidence[str(p.relative_to(ROOT))]={'before_sha256':sha(before),'after_sha256':sha(p),'historical_body_byte_identical':True}
            blocks.append((p,matches[0].decode('utf-8')))
        docs=[(p,p.read_text(encoding='utf-8')) for p in OUT.glob('*.md')]+blocks;links=[]
        for p,content in docs:
            require('TODO' not in content and '{{' not in content,'No unfilled placeholders')
            for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
                target=raw.strip().strip('<>')
                if target.startswith(('http://','https://','#')):continue
                resolved=(p.parent/target.split('#')[0]).resolve()
                require(resolved.is_relative_to(ROOT),'Local link outside project')
                require(resolved.exists() or resolved==(OUT/'delivery_verification.json').resolve(),'Broken local link '+target)
                links.append({'source':str(p.relative_to(ROOT)),'target':str(resolved.relative_to(ROOT))})
        for p in INDEXES:
            with newpath(OUT/'document_snapshots/after'/p.relative_to(ROOT)).open('xb') as f:f.write(p.read_bytes())
        emit(OUT/'index_sha256_after.json',index_evidence)
        emit(OUT/'delivery_link_check.json',{'status':'PASS','local_links':links})
        protected_verify(guard)
        files={}
        for directory in (HERE,OUT):
            for p in sorted(directory.rglob('*')):
                if p.is_file() and p.suffix!='.lock' and '__pycache__' not in p.parts:
                    files[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
        emit(OUT/'delivery_manifest.json',{'created':now(),'files':files,'excludes':['evaluation.lock','delivery_manifest.json','delivery_verification.json']})
        receipt={'status':'PASS','scope_status':'COMPLETED','scope':'Authorized approximate UK ITU rain baselines, existing LOCAL baselines, same-sample ML comparison and sensitivity',
                 'time':now(),'source_sha256_records':len(guard),'protected_source_sha_unchanged':True,'main_report_numeric_rows_checked':29,
                 'physical_scenarios':18,'physical_finite_values_recomputed':qa['physical_finite_values_independently_recomputed'],
                 'independent_qa_sha256':sha(OUT/'independent_qa.json'),'manifest_sha256':sha(OUT/'delivery_manifest.json'),'report_sha256':sha(OUT/'COMPARISON_REPORT.md'),
                 'artifacts':len(files),'local_links':len(links),'fixed_index_old_bytes_preserved':True,'zero_control_is_post_hoc':True,
                 'new_fit_calls':0,'new_ml_inference_calls':0,'weather_downloads':0,'official_method_reference_files_retrieved':4,
                 'stop_marker_sha256':sha(ACQ/'STOP_REQUESTED.json'),'tvt_status':'PAUSED','automation_sha256':sha(AUTOMATION),
                 'native_polarization':'LHCP confirmed236 files','rain_height':'P839 annual climate map +0.36km, not hourly observation',
                 'historical_T012_all_goals_complete':False,'scientific_acceptance':'awaiting human review','execution_lock':'RELEASED',
                 'finalizer_sha256':sha(Path(__file__))}
    emit(OUT/'delivery_verification.json',receipt)
    print({k:receipt[k] for k in ('status','scope_status','artifacts','local_links','execution_lock')},flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        emit(OUT/('failure_finalize_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{'time':now(),'traceback':traceback.format_exc()});raise
