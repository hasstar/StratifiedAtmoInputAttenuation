"""Post-hoc rain/overcast RMSE of frozen saved predictions, full completed UK."""
from __future__ import annotations
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from cloud_acquisition import ROOT,CODE,OUT,R2,R3,R6,GBS,RAIN,EPOCH1900,read,sha,key,need,emit,lock,utc,parse_ascii

PHYSICAL=['itu838_geometric_proxy','itu618_hourly_proxy']
EVENTS=['rain','dry_overcast']
NAMES={
 'ours_frozen_top10':'本文冻结Top10',
 'tabpfn_v2_author_frozen':'TabPFN-v2（作者原权重）',
 'tabicl_v2_author_frozen':'TabICLv2（作者原权重）',
 'tabm2025_official_budgeted':'TabM（本地适配版）',
 'bai2022_mlp_adapted':'Bai2022 MLP（适配版）',
 'bai2022_lstm_t0_adapted':'Bai2022 零历史LSTM（适配版）',
 'ridge':'Ridge','elastic_net':'Elastic Net','svr_rbf':'RBF-SVR',
 'random_forest':'Random Forest','xgboost':'XGBoost','train_mean':'LOCAL训练均值常数',
 'itu838_geometric_proxy':'P.838几何雨衰近似','itu618_hourly_proxy':'P.618小时代理'}

def audit_rain(uk):
    p=pd.read_parquet(R3/'field_provenance.parquet')
    p=p[p.name.isin(RAIN)].copy()
    need(len(p)==11328 and not p.duplicated(['station','utc_epoch_seconds','name']).any(),'Rain provenance count/uniqueness')
    evidence={r['source']:r for r in read(R3/'raw_source_evidence.json')}
    index=uk.set_index(['station','utc_epoch_seconds'])
    scope={s['id']:s for s in read(GBS/'scope_manifest.json')['links']}
    for source,g in p.groupby('source'):
        ev=evidence[source]
        var=g.name.iloc[0]
        symbol,pid={'mean_total_precipitation_rate':('MTPR',55),'mean_snowfall_rate':('MSR',31)}[var]
        need(g.name.nunique()==1 and ev['variable']==var and ev['parameter_table']==235 and ev['parameter']==pid,'Rain identity mismatch')
        need(ev['units'] in ['kg m**-2 s**-1','kg m-2 s-1'] and sha(ROOT/source)==ev['sha256'],'Rain unit/SHA')
        body=(ROOT/source).read_text(encoding='utf-8')
        need('e5.oper.fc.sfc.meanflux' in body.split('--------------------------------')[0] and f'.235_{pid:03d}_' in body.split('--------------------------------')[0],'Rain source header identity')
        data=parse_ascii(body)
        dimensions=['forecast_initial_time','forecast_hour','latitude','longitude']
        need(set(data)=={symbol+'.'+x for x in [symbol]+dimensions},'Rain arrays/maps mismatch')
        coords=[data[symbol+'.'+d] for d in dimensions]
        values=data[symbol+'.'+symbol]
        need(list(values.shape)==ev['shape'] and values.shape==tuple(len(a) for a in coords),'Rain shape')
        need(all(1<=x<=12 for x in coords[1]),'Rain forecast steps')
        lookup={}
        for coord,value in zip(itertools.product(*coords),values.ravel()):
            init,lead,lat,lon=coord
            need(init%1==0 and lead%1==0 and lat*4%1==0 and lon*4%1==0,'Rain coordinate lattice')
            valid=EPOCH1900+int(init+lead)*3600
            k=(valid,int(round(lat*4)),int(round((((lon+180)%360)-180)*4)))
            need(k not in lookup,'Duplicate rain key')
            lookup[k]=float(value)
        for r in g.itertuples(index=False):
            need(r.valid_utc_epoch_seconds==r.utc_epoch_seconds+3600,'Rain t+1 alignment')
            s=scope[r.station]
            lat=math.floor(s['latitude_deg_n']*4+.5)
            lon=math.floor(s['longitude_deg_e']*4+.5)
            need((r.latitude_quarter_index,r.longitude_quarter_index)==(lat,lon),'Rain nearest station grid')
            k=(r.valid_utc_epoch_seconds,lat,lon)
            need(k in lookup and lookup[k]==r.value==float(index.loc[(r.station,r.utc_epoch_seconds),var]),'Raw rain value/sample mismatch')
    for field,water in zip(RAIN,['total_water_mm','snow_water_mm']):
        need(np.array_equal(3600*uk[field].to_numpy(),uk[water].to_numpy()),'Saved rate conversion changed')
    return dict(verified_raw_values=len(p),raw_payloads=p.source.nunique(),alignment='valid=t+1h',parameter_table=235,parameters=[55,31],units='kg m-2 s-1')

def classify(p,s,c):
    need(np.isfinite(p).all() and np.isfinite(s).all() and np.isfinite(c).all(),'Incomplete weather input')
    return np.select([(p>=.1)&(s<.1),(p<.1)&(s<.1)&(c>=.8)],['rain','dry_overcast'],default='other')

def describe(frame,station='LOCAL'):
    return dict(hours=len(frame),weather_counts={str(k):int(v) for k,v in frame.event.value_counts().items()},station=station)

def score():
    need(read(OUT/'acquisition_complete.json')['status']=='PASS','Cloud acquisition not passed')
    need(not (OUT/'scoring_complete.json').exists(),'No overwrite/redundant scoring')
    started=utc()
    with lock():
        try:
            before=read(OUT/'sources_before.json')
            need(all(sha(ROOT/p)==h for p,h in before.items()),'Protected input changed before scoring')
            uk=pd.read_parquet(R6/'predictions_uk_all.parquet')
            local=pd.read_parquet(R6/'predictions_local_all.parquet')
            cloud=pd.read_parquet(OUT/'cloud_all.parquet')
            need(sha(OUT/'cloud_all.parquet')==read(OUT/'acquisition_complete.json')['cloud_sha256'],'Cloud table SHA')
            need(len(local)==687 and local.hour_bjt.is_unique and len(uk)==5664,'Input scope')
            expanded=pd.read_parquet(R3/'expanded_dataset.parquet')
            need(sha(R3/'expanded_dataset.parquet')==read(R3/'dataset_complete.json')['dataset_sha256'],'Expanded source SHA')
            columns=['station','utc_epoch_seconds','primary_score_eligible','truth_db']+RAIN
            pd.testing.assert_frame_equal(uk[columns].sort_values(columns[:2]).reset_index(drop=True),expanded[columns].sort_values(columns[:2]).reset_index(drop=True))
            rain_qa=audit_rain(uk)
            uk=uk.merge(cloud,on=['station','utc_epoch_seconds'],how='left',validate='one_to_one')
            need(uk.tcc.notna().all() and uk.tcc.between(0,1).all(),'Cloud missing/outside range')
            need((uk.tcc_valid_utc_epoch_seconds==uk.utc_epoch_seconds).all(),'Cloud t alignment')
            uk['event']=classify(uk.total_water_mm,uk.snow_water_mm,uk.tcc)
            uk['date_utc']=pd.to_datetime(uk.utc_epoch_seconds,unit='s',utc=True).dt.strftime('%Y-%m-%d')
            need(int(uk.primary_score_eligible.sum())==5593 and int(uk.common_score_eligible.sum())==5538,'Original quality masks changed')
            need((~uk.common_score_eligible | uk.primary_score_eligible).all(),'Common mask not subset of primary')
            need(((uk.primary_score_eligible & uk.physical_input_valid)==uk.common_score_eligible).all(),'Mask semantics')
            local['event']='other'
            need((local[EVENTS].astype(int).sum(axis=1)<=1).all(),'Overlapping LOCAL events')
            for event in EVENTS:
                local.loc[local[event].astype(bool),'event']=event
            labels=uk[['station','utc_epoch_seconds','date_utc','period','tcc','tcc_valid_utc_epoch_seconds','latitude','longitude_0_360','source','source_sha256']+RAIN+['total_water_mm','snow_water_mm','liquid_negative','event','primary_score_eligible','physical_input_valid','common_score_eligible','exclusion_reason']].copy()
            labels['primary_excluded']=~labels.primary_score_eligible
            labels['physical_excluded_after_primary']=labels.primary_score_eligible&~labels.physical_input_valid
            labels['rain_scored_common']=labels.common_score_eligible&labels.event.eq('rain')
            labels['overcast_scored_common']=labels.common_score_eligible&labels.event.eq('dry_overcast')
            labels.to_parquet(OUT/'weather_labels_all.parquet',index=False)
            dates=[]
            for (station,date),g in labels.groupby(['station','date_utc']):
                need(len(g)==24,'Incomplete target day')
                dates.append(dict(station=station,date_utc=date,acquired_hours=24,weather_known_hours=int(g.tcc.notna().sum()),primary_hours=int(g.primary_score_eligible.sum()),common_hours=int(g.common_score_eligible.sum()),rain_all=int(g.event.eq('rain').sum()),overcast_all=int(g.event.eq('dry_overcast').sum()),rain_common=int(g.rain_scored_common.sum()),overcast_common=int(g.overcast_scored_common.sum())))
            need(len(dates)==236,'Wrong day coverage')
            emit('coverage_by_date.json',dates)
            bystation=[]
            for station,g in labels.groupby('station'):
                dd=[r for r in dates if r['station']==station]
                bystation.append(dict(station=station,dates=len(dd),station_hours=len(g),weather_known_hours=len(g),primary_hours=int(g.primary_score_eligible.sum()),common_hours=int(g.common_score_eligible.sum()),dates_with_common_hours=sum(r['common_hours']>0 for r in dd),weather_counts_all=g.event.value_counts().to_dict(),weather_counts_primary=g[g.primary_score_eligible].event.value_counts().to_dict(),weather_counts_common=g[g.common_score_eligible].event.value_counts().to_dict(),rain_dates_common=sum(r['rain_common']>0 for r in dd),overcast_dates_common=sum(r['overcast_common']>0 for r in dd),date_min=min(g.date_utc),date_max=max(g.date_utc)))
            emit('coverage_by_station.json',bystation)
            old=pd.read_parquet(GBS/'conditional_fast_001/weather_labels.parquet')
            old['utc_epoch_seconds']=pd.to_datetime(old.hour_utc).map(lambda t:int(t.timestamp()))
            overlap=uk[uk.period.eq('original39')].merge(old[['station','utc_epoch_seconds','tcc','tp','sf','event']],on=['station','utc_epoch_seconds'],validate='one_to_one',suffixes=('_ncar','_cds'))
            need(len(overlap)==1872,'Old39 overlap mismatch')
            overlap['old_comparable_event']=overlap.event_cds.where(overlap.event_cds.isin(EVENTS+['unknown']),'other')
            overlap['classification_changed']=overlap.event_ncar!=overlap.old_comparable_event
            overlap['p_ncar_minus_cds_mm']=overlap.total_water_mm-overlap.tp*1000
            overlap['s_ncar_minus_cds_mm']=overlap.snow_water_mm-overlap.sf*1000
            overlap['tcc_ncar_minus_cds']=overlap.tcc_ncar-overlap.tcc_cds
            overlap[['station','utc_epoch_seconds','tcc_ncar','tcc_cds','total_water_mm','tp','snow_water_mm','sf','event_ncar','event_cds','classification_changed','p_ncar_minus_cds_mm','s_ncar_minus_cds_mm','tcc_ncar_minus_cds']].to_parquet(OUT/'old39_source_comparison.parquet',index=False)
            source_comparison=dict(overlap_hours=len(overlap),old_unknown=int(overlap.event_cds.eq('unknown').sum()),new_unknown=int(overlap.event_ncar.eq('unknown').sum()),changed_categories_including_recovered_unknown=int(overlap.classification_changed.sum()),max_abs_tcc_difference=float(overlap.tcc_ncar_minus_cds.abs().max()),max_abs_total_precip_mm_difference=float(overlap.p_ncar_minus_cds_mm.abs().max()),max_abs_snow_mm_difference=float(overlap.s_ncar_minus_cds_mm.abs().max()),transitions=[dict(old=a,new=b,hours=int(n)) for (a,b),n in overlap.groupby(['old_comparable_event','event_ncar']).size().items()])
            emit('old39_source_comparison.json',source_comparison)
            models=[m['id'] for m in read(R2/'frozen_selection.json')['models']]+['ours_frozen_top10','tabpfn_v2_author_frozen','tabicl_v2_author_frozen']+PHYSICAL
            need(len(models)==24 and len({re.sub(r'_seed\d+$','',m) for m in models})==14,'Model coverage')
            cohorts=[('LOCAL_test',local,models),('UK_full_common',uk[uk.common_score_eligible],models)]
            cohorts += [(f'UK_{s}_full_common',uk[uk.common_score_eligible&uk.station.eq(s)],models) for s in ['chilbolton','sparsholt']]
            cohorts += [('UK_full_ML_primary_sensitivity',uk[uk.primary_score_eligible],[m for m in models if m not in PHYSICAL])]
            cohorts += [(f'UK_{s}_ML_primary_sensitivity',uk[uk.primary_score_eligible&uk.station.eq(s)],[m for m in models if m not in PHYSICAL]) for s in ['chilbolton','sparsholt']]
            prior={(r['cohort'],r['model']):r for r in read(R6/'metrics.json')}
            runs,cohortcounts,members,zero=[],[],[],[]
            for cohort,g,methods in cohorts:
                need(np.isfinite(g.truth_db).all(),'Nonfinite eligible target')
                need(np.isfinite(g[methods].to_numpy()).all(),'Missing model prediction: cannot drop per model')
                cohortcounts.append(dict(cohort=cohort,eligible_hours=len(g),weather_counts=g.event.value_counts().to_dict()))
                for event in EVENTS:
                    part=g[g.event.eq(event)]
                    need(len(part)>0,'Missing group')
                    members += [dict(cohort=cohort,event=event,station=getattr(r,'station','LOCAL'),hour=str(getattr(r,'hour_bjt',getattr(r,'hour_utc',None)))) for r in part.itertuples(index=False)]
                    y=part.truth_db.to_numpy(dtype=float)
                    for model in methods:
                        rmse=float(np.sqrt(np.mean((part[model].to_numpy(dtype=float)-y)**2)))
                        if cohort=='LOCAL_test':
                            oldm=prior[('LOCAL_'+event,model)]
                            need(oldm['n']==len(part) and math.isclose(oldm['rmse_db'],rmse,rel_tol=1e-12,abs_tol=1e-12),'LOCAL prior metric changed')
                        runs.append(dict(cohort=cohort,event=event,model=model,family=re.sub(r'_seed\d+$','',model),n=len(part),rmse_db=rmse))
                    zero.append(dict(cohort=cohort,event=event,n=len(part),diagnostic='fixed_zero_db_not_ranked',rmse_db=float(np.sqrt(np.mean(y*y)))))
            families=[]
            rdf=pd.DataFrame(runs)
            for (cohort,event,family),g in rdf.groupby(['cohort','event','family']):
                need(g.n.nunique()==1,'Family sample mismatch')
                families.append(dict(cohort=cohort,event=event,family=family,name=NAMES[family],n=int(g.n.iloc[0]),n_runs=len(g),rmse_db_mean=float(g.rmse_db.mean()),rmse_db_sd=float(g.rmse_db.std(ddof=1)) if len(g)>1 else None))
            for r in families:
                r['rank']=1+sum(q['rmse_db_mean']<r['rmse_db_mean'] for q in families if (q['cohort'],q['event'])==(r['cohort'],r['event']))
            emit('run_metrics.json',runs)
            emit('family_rankings.json',families)
            emit('cohort_coverage.json',cohortcounts)
            emit('sample_membership.json',members)
            emit('zero_constant_diagnostic.json',zero)
            emit('rain_provenance_qa.json',rain_qa)
            after={p:sha(ROOT/p) for p in before}
            need(after==before,'Protected input changed during scoring')
            emit('sources_after_scoring.json',after)
            report=['# 全部已完成日期：雨天与干燥阴天的模型比较','','## 结论和覆盖范围','','本次结果替代“仅原39日”的天气分组表作为全量主表。所有已完成扩展日期均进入天气分类；没有重新训练或运行任何模型。','',
              'Chilbolton：143个日期、3432小时；Sparsholt：93个日期、2232小时；合计236个站点日、5664个站点小时。TCC已逐小时补齐，MTPR/MSR复用完整载荷。原1127站点日计划不属于已完成数据，不自动恢复下载。','',
              '日期保留原生缺口：Chilbolton的2003-08-19—2003-11-30；Sparsholt的2003-10-08—2003-11-30；另有两站各自原2004-02-18—21、02-25—28、03-01—31的39日。不是从2003年起到2004年3月连续无缺口。','',
              '“全量雨天/阴天”指在全部上述日期中筛选相应天气的小时，不是把这些日期的所有小时都标为雨或阴。没有按模型成绩选日期，也不取两站日期交集。LOCAL仍是原687小时测试集。','',
              '全部5664小时有天气字段；沿用原质量规则，标签合格5593小时。加入ITU的14家族共同表使用5538小时（71小时标签不合格，另55小时液态估计为负、ITU输入无效），再按天气分组。逐日覆盖和排除记录见coverage_by_date.json与weather_labels_all.parquet。','',
              '| 范围 | 日期数 | 有天气的小时 | 标签合格小时 | 共同小时 | 共同雨小时 | 共同阴小时 | 有共同小时的日期数 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
            for r in bystation:
                report.append(f"| {r['station']} | {r['dates']} | {r['station_hours']} | {r['primary_hours']} | {r['common_hours']} | {r['weather_counts_common'].get('rain',0)} | {r['weather_counts_common'].get('dry_overcast',0)} | {r['dates_with_common_hours']} |")
            lookup={(r['cohort'],r['event'],r['family']):r for r in families}
            main=[x[0] for x in cohorts[:4]]
            for event,title in [('rain','雨天（雨小时）'),('dry_overcast','干燥阴天（阴小时）')]:
                report += ['', '## '+title+'：全部14家族共同样本', '', '单元格为RMSE dB（排名）；三种子模型为逐运行RMSE的均值±样本SD。排名用未舍入均值，SD不是置信区间。','', '| 模型 | LOCAL测试集 | 英国两站全量合并 | Chilbolton全143日 | Sparsholt全93日 |','|---|---:|---:|---:|---:|']
                for r in sorted([r for r in families if r['cohort']=='LOCAL_test' and r['event']==event],key=lambda x:x['rank']):
                    cells=[]
                    for cohort in main:
                        q=lookup[(cohort,event,r['family'])]
                        cells.append(f"{q['rmse_db_mean']:.4f}"+(f" ± {q['rmse_db_sd']:.4f}" if q['rmse_db_sd'] is not None else '')+f"（#{q['rank']}）")
                    report.append('| '+r['name']+' | '+' | '.join(cells)+' |')
                report += ['', '样本数：'+ '；'.join(f"{c} {next(r['n'] for r in families if r['cohort']==c and r['event']==event)}小时" for c in main)+'。']
            report += ['', '## 不含ITU时的12家族敏感性结果', '', '以下使用全部5593个标签合格小时，不额外施加ITU物理门槛；不与14家族表混排。','', '| 模型 | 英国全量雨 | 英国全量阴 | Chilbolton雨 | Chilbolton阴 | Sparsholt雨 | Sparsholt阴 |','|---|---:|---:|---:|---:|---:|---:|']
            for family in NAMES:
                if family in PHYSICAL:
                    continue
                cells=[]
                for c,_,_ in cohorts[4:]:
                    for event in EVENTS:
                        q=lookup[(c,event,family)]
                        cells.append(f"{q['rmse_db_mean']:.4f}（#{q['rank']}）")
                report.append('| '+NAMES[family]+' | '+' | '.join(cells)+' |')
            report += ['', '## 分类、来源和解释', '', '雨：总水当量p≥0.1mm且雪水当量s<0.1mm；干燥阴：p<0.1mm、s<0.1mm且总云量TCC≥0.8。p=3600×MTPR，s=3600×MSR；率的valid UTC=t+1h，云量为t。全量使用NCAR ERA5站点最近0.25度格，不插值，不使用衰减值推定天气。其它天气不列入雨/阴表，本次不展示晴天结果。', '', '旧39日表使用CDS tp/sf/tcc；此次所有日期统一采用NCAR MTPR/MSR/TCC，阈值不变，来源变化逐行记录在old39_source_comparison.parquet。', '', f"原39日重叠{source_comparison['overlap_hours']}小时，旧天气未知{source_comparison['old_unknown']}小时，新天气未知{source_comparison['new_unknown']}小时；包括未知恢复在内，三类（雨/阴/其它）变化{source_comparison['changed_categories_including_recovered_unknown']}小时。总云量最大绝对差{source_comparison['max_abs_tcc_difference']:.8g}，总水当量最大差{source_comparison['max_abs_total_precip_mm_difference']:.8g}mm，雪水当量最大差{source_comparison['max_abs_snow_mm_difference']:.8g}mm。", '', '天气是再分析代理而非现场天气实测。P.838/P.618保持既有近似、圆极化与名义雨层高度；没有为英国调参。TabM/Bai适配版及其它历史LOCAL训练检查点均未更动；TabPFN/TabICL作者权重亦未更动。', '', '每列用同样小时、同一真实标签比较全部模型。英国合并RMSE按站点小时合并，不是两个站点RMSE简单平均；多种子列先分别算RMSE再取均值，不挑最好种子、不创造新集成。相邻小时和两站天气相关，排名为描述性结果，没有显著性或独立事件数主张。', '', '原有低衰减样本较多，单列保留0dB常数诊断zero_constant_diagnostic.json，不把它加入既定14家族排名。新字段仅用于天气分类，未加入Top10输入。', '', '## 可核查文件', '', '- cloud_all.parquet、payloads/、metadata/、catalogs/、http/：5664个云量值及精确匿名GET凭证。', '- weather_labels_all.parquet、coverage_by_date.json、coverage_by_station.json：全部日期/小时覆盖、天气及评分门槛。', '- run_metrics.json、family_rankings.json、sample_membership.json：每运行指标、家族均值/SD/排名和共同样本。', '- rain_provenance_qa.json：11328条原始雨率值的来源重核。', '- verification.json、completion.json：独立离线核验和最终交付状态；未出现PASS前不得称最终通过。', '', '历史输出只读保留；本报告不恢复已停止的v011持续下载或tvt自动化。不据本项统计声明整个T012完成。']
            with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:
                f.write('\n'.join(report)+'\n')
            emit('scoring_complete.json',dict(status='COMPUTED_PENDING_INDEPENDENT_QA',started_utc=started,completed_utc=utc(),runs=24,families=14,weather_known_uk_hours=5664,uk_station_dates=236,common_hours=5538,primary_hours=5593,local_hours=687,protected_sources_unchanged=len(before),new_training=0,new_inference=0,execution_lock='RELEASED'))
            print(json.dumps(dict(status='COMPUTED_PENDING_INDEPENDENT_QA',coverage=bystation,winners=[r for r in families if r['rank']==1]),ensure_ascii=False))
        except Exception as exc:
            emit('scoring_failure.json',dict(utc=utc(),error=repr(exc),execution_lock='RELEASED_AFTER_FAILURE'))
            raise

if __name__=='__main__':
    score()
