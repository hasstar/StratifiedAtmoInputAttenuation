"""Publish audited numbers, honest material status and append-only handoff."""
from support import *
import pandas as pd

ORDER=['ours_frozen_top10','bai2022_mlp_adapted','bai2022_lstm_t0_adapted','tabm2025_official_budgeted','itu_combined_geometric_approx']
COHORTS=[('LOCAL_test','LOCAL 测试集'),('UK_full_common','英国两站全量合并'),
         ('UK_chilbolton_full_common','Chilbolton 全部 143 日'),('UK_sparsholt_full_common','Sparsholt 全部 93 日')]

def write_md(name,lines):
    with new(OUT/name).open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')

def main():
    offline()
    with lock():
        guard=read(OUT/'sources_before.json'); check(guard)
        qa=read(OUT/'verification.json')
        need(qa['status']=='PASS' and qa['all_physical_rows_checked']==6351,'QA not complete')
        calc=read(OUT/'calculation_complete.json')
        for name,h in calc['prediction_sha256'].items(): need(sha(OUT/name)==h,'Predictions changed')
        contract=read(OUT/'calculation_contract.json')
        for name,h in contract['code_sha256'].items(): need(sha(CODE/name)==h,'Calculation code changed')
        need(sha(CODE/'PROTOCOL.md')==contract['protocol_sha256'],'Protocol changed')
        need(sha(CODE/'independent_qa.py')==qa['verification_script_sha256'],'QA code changed')
        ranking=read(OUT/'family_rankings.json')
        idx={(r['cohort'],r['event'],r['family']):r for r in ranking}
        for cohort,_ in COHORTS:
            for event in ('rain','dry_overcast'):
                ordered=sorted((idx[(cohort,event,k)] for k in ORDER),key=lambda r:r['rmse_db_mean'])
                for rank,row in enumerate(ordered,1): need(row['rank']==rank,'Ranking mismatch')
        refs=read(OUT/'fixed_reference_parameters.json')
        local=pd.read_parquet(OUT/'predictions_local.parquet')
        uk=pd.read_parquet(OUT/'predictions_uk.parquet')
        dates=[]
        for station,g in uk.groupby('station'):
            dates.append(dict(station=station,dates=[d.isoformat() for d in sorted(set(g.hour_utc.dt.date))],
                         hours=len(g),primary_hours=int(g.primary_score_eligible.sum()),common_hours=int(g.common_score_eligible.sum())))
        need([(r['station'],len(r['dates']),r['hours']) for r in dates]==[('chilbolton',143,3432),('sparsholt',93,2232)],'Date scope')
        emit('final_date_coverage.json',dates)
        lines=['# 组合 ITU 与冻结论文模型对比 — Revision 011','',
            '2026-10-17。组合 ITU 计算及独立算术 QA 已完成；8 个新增论文候选的筛选已记录，但尚无新增非 ITU 论文模型具备可报告的复现结果。此处不宣称所有候选论文、所有 SI 或整个 T012 已完成。','',
            '本轮零训练、零新学习模型推理、零气象下载；本文、Bai、TabM 只复用已有预测。旧数据、模型和排名文件不改写。','',
            '## 结果概览','',
            '- LOCAL 雨天：TabM 既有适配版最优；LOCAL 干燥阴天：本文冻结 Top10 最优。',
            '- 英国两站合并雨天：TabM 最优；英国两站合并干燥阴天：本次新增的组合 ITU 最优。',
            '- 分站看，Chilbolton 雨天与阴天均为组合 ITU 最优；Sparsholt 雨天为 TabM、阴天为组合 ITU。',
            '- 因此不能声称本文模型在本地与英国、雨天与阴天均排名第一。名次是当前主名单内的描述性结果，不是统计显著性结论。','',
            '## 数据与共同评价口径','',
            '英国使用全部已完成的 236 站点日/5664 站点小时，不缩回原 39 日，也不把未下载的 1127 站点日计划当成已完成。所有现有日期都进入天气分类；评分沿用冻结的质量与物理有效性规则。','',
            '| 数据集 | 已完成日期/小时 | 共同评分母集 | 雨小时 | 干燥阴小时 |','|---|---:|---:|---:|---:|',
            '| LOCAL 测试集 | 原测试集 687 小时 | 687 | 51 | 158 |',
            '| Chilbolton | 143 日 / 3432 小时 | 3351 | 408 | 1424 |',
            '| Sparsholt | 93 日 / 2232 小时 | 2187 | 327 | 868 |',
            '| 英国合并 | 236 站点日 / 5664 小时 | 5538 | 735 | 2292 |','',
            '日期：Chilbolton 的 2003-08-19—11-30，Sparsholt 的 2003-10-08—11-30；另有两站各自的 2004-02-18—21、02-25—28、03-01—31。保留原缺口，不是两站连续 236 天。逐日期列表见 final_date_coverage.json。','',
            '英国 71 小时标签质量不合格，另 55 小时液态雨量估计为负、物理输入无效，按原掩码保留但不评分；没有按模型删除不同的小时。仅学习模型的原 5593 小时敏感性表仍在 Revision008，不与此共同母集混排。“雨天/阴天”均指这些日期内相应天气的小时，不是将整日全部小时都归为同一天气。','',
            '指标为小时衰减 RMSE，单位 dB。Bai/TabM 为种子 17、29、43 各自 RMSE 的均值 ± 样本 SD，不挑最优种子，也不是新集成；SD 不是置信区间。排名使用未舍入值。主名单共 5 个模型条目；Bai 的两个结构属于同一篇论文，ITU 敏感性版本不另计独立论文。','']
        for cohort,title in COHORTS:
            lines += ['## '+title,'','| 模型 | 雨天 RMSE（名次） | 干燥阴天 RMSE（名次） |','|---|---:|---:|']
            for family in ORDER:
                rr=[]
                for event in ('rain','dry_overcast'):
                    r=idx[(cohort,event,family)]
                    value=f"{r['rmse_db_mean']:.4f}"
                    if r['rmse_db_sd'] is not None: value+=f" ± {r['rmse_db_sd']:.4f}"
                    rr.append(value+f"（#{r['rank']}）")
                lines.append('| '+idx[(cohort,'rain',family)]['name']+' | '+' | '.join(rr)+' |')
            sensitivity=[idx[(cohort,e,'itu_combined_p618_hourly_approx')]['rmse_db_mean'] for e in ('rain','dry_overcast')]
            lines += ['',f'同一 ITU 家族的 P.618 小时雨路径敏感性：雨 {sensitivity[0]:.4f} dB，阴 {sensitivity[1]:.4f} dB。不据结果替换预先冻结的几何主版本。','']
        lines += ['## 组合 ITU 的可核验定义','',
            'A(t) = [A_gas(t) − A_gas,ref] + A_cloud(t) + A_rain(t)。','',
            '气体使用 P.676-13 的氧气谱线/等效高度与瞬时水汽柱方法；云雾使用 P.840-9 柱液态水方法；雨复用冻结的 P.838 几何雨路径。P.618 小时雨路径只作为同家族敏感性。','',
            '固定参考使用 P.835-6 的低空标准状态和 2 km 指数水汽尺度高，Vref=15 exp(−h/2) kg/m²。该积分略去完整标准大气高空混合比下限，故明确叫“固定参考近似”，不是实测晴空绝对基准。参考未用任何实测衰减拟合。','',
            '| 数据集 | 频率 GHz | 仰角 ° | 地面海拔 km | 固定气体参考 dB |','|---|---:|---:|---:|---:|']
        for name in ('LOCAL','chilbolton','sparsholt'):
            r=refs[name]
            lines.append(f"| {name} | {r['frequency_GHz']:.2f} | {r['elevation_deg']:.6f} | {r['height_km']:.3f} | {r['gas_absolute_db']:.6f} |")
        lines += ['',
            'LOCAL 海拔沿用原物理基线的海平面近似，不按成绩选高度；英国采用原站海拔与固定雨层高度。LOCAL 雨分量沿用此前逐小时零度层近似及既有冻结回退值，本轮不重估。原 LOCAL 实现中的液态雨量截零策略如实保留，不能称为本轮重新校准。','',
            '云与雾使用同一 TCLW 柱只计一次；TCC 只作天气分类，不当作液态水。未加入闪烁，因为未取得与现有小时标签一致的定义和输入；这不表示真实闪烁为零。也不加自由空间或馈线损耗。参考差和最终预测允许为负，不裁成零。','',
            '英国雨率仍为 valid UTC 前一小时的 MTPR/MSR：对 [t,t+1h) 取 valid=t+1h；其余五项气象分析值在 t。小时平均率与瞬时路径状态的代入是近似，不把小时雨率叫 R0.01，不把本基线叫完整 P.618 超越概率模型。ERA5 是再分析，不能据此声称这是实时未来预报。','',
            '## 每个模型的实际输入','',
            '| 模型 | 输入 | 推理时使用 LOCAL 标签例题？ | 本轮训练？ |','|---|---|---|---|',
            '| 本文冻结 Top10 | 原固定十个小时天气特征，既有几何定义及训练期冻结预处理 | 否 | 否；复用预测 |',
            '| Bai2022 MLP / 零历史 LSTM 既有适配版 | 同一 Top10；LSTM 不使用过去信道序列，原适配状态不变 | 否 | 否；复用预测 |',
            '| TabM 既有本地适配版 | 同一 Top10、既有 LOCAL 冻结参数 | 否 | 否；复用预测 |',
            '| 组合 ITU | 2 m 温度、2 m 露点、地表气压、TCWV、TCLW、既有液态雨量及雨高；频率、极化、仰角、站高和固定物理参考 | 否 | 无训练参数 |','',
            'Top10：z/u@1 hPa、o3/u@5 hPa、o3@500 hPa、u@650 hPa、o3/r@850 hPa、地面 CAPE 与 maximum_2m_temperature_since_previous_post_processing。物理基线使用不同的、物理必需的输入；这里统一的是小时标签、日期与评分样本，不宣称所有方法的信息输入相同。Bai/TabM 是此前获得授权后形成的适配版，不是作者原始权重。','',
            '## 验证和材料状态','',
            f"独立标量实现复核 {qa['all_physical_rows_checked']} 条物理记录，复算 {qa['run_metrics_checked']} 个指标，18 个无效输入控制全部拒绝；气体/云分量最大绝对差低于 5×10⁻¹⁶ dB。该验证是同一代理编写的第二种算术实现，不冒称独立人工审稿或新数据验证。",
            '七张官方公式页已渲染检查；完整 44 条氧气谱线与 Part1/Part2 系数表有源 SHA。已核验零液水、零水汽柱、1/sin(仰角)比例、固定参考、系数结点与负相对衰减保留。冻结预测、五字段来源及共同样本键均与来源完全一致。',
            f"{len(guard)} 项历史来源 SHA 在开始和结束一致；tvt PAUSED 和旧 STOP 保留。没有天气请求、训练或学习模型重新推理。",'',
            '文献与 SI 状态见 [LITERATURE_SCREENING.md](LITERATURE_SCREENING.md)：四份官方标准和两份配套系数已保存；三次新论文/手册的联网保存未成功，错误原件已保留。没有宣称所有全文/SI 已补齐。当前没有新增非 ITU 论文模型的数值；相关输入、权重或可访问原文缺口已逐篇列明。','',
            '## 交付文件','',
            '- predictions_local.parquet / predictions_uk.parquet：逐小时预测及物理分量，旧列完整保留用于审计；退出主名单的列不参与本报告排名。',
            '- run_metrics.json / family_rankings.json：逐种子和家族级指标；sample_membership.json：共同评分成员。',
            '- fixed_reference_parameters.json / physical_input_audit.json / component_diagnostics.json：物理参数与组件审计。',
            '- verification.json / final_verification.json / completion.json：双实现 QA、最终完整性和明确的阶段完成状态。',
            '- LITERATURE_SCREENING.md / OPERATION_NOTES.md / HANDOFF.md：候选、失败状态与续接边界。','',
            '## 官方方法来源','',
            '- [P.676-13](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-13-202208-I!!PDF-E.pdf) 与[配套系数](https://www.itu.int/oth/R1101000002/en)。',
            '- [P.840-9](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-9-202308-I!!PDF-E.pdf)。',
            '- [P.835-6 固定版本](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.835-6-201712-S!!PDF-E.pdf)。',
            '- [P.453-14](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-14-201908-I!!PDF-E.pdf)。','']
        write_md('REPORT.md',lines)
        write_md('HANDOFF.md',[
            '# Revision 011 交接','',
            '组合 ITU 本阶段已完成且 QA PASS。主报告为 REPORT.md，主排名为 family_rankings.json。',
            '新增非 ITU 候选尚未执行；缺口详见 LITERATURE_SCREENING.md。不将本阶段完成标记当作所有论文已复现或 T012 全完成。','',
            '保持 tvt PAUSED 和旧 STOP，不运行任何旧天气 run/smoke/prepare/initialize。',
            '不训练、不微调、不变更作者/既有模型参数，不提供 LOCAL 标签作为推理上下文。',
            '所有英国主表继续使用 143/93 日全已完成日期及 5538 小时共同母集；不能退回39日。',
            'calculate.py、physical.py、PROTOCOL.md 已在计算前冻结；不能覆盖本版输出或重跑初始化。后续修复另存新版本。',
            '没有独立非 ITU 新论文复现结果；下一步必须先获得该论文的充分原始资料/代码/权重与适用输入，不通过训练或扩范围采集静默补齐。',
            '官方标准 PDF/系数已保存。三份论文/手册 PDF 保存失败（403/403/303），已记录而未绕过。',
            'CPU 短任务均已退出；阶段锁在 finalize_report 结束时释放。未知外部写者不声称不存在。'])
        check(guard)
        emit('sources_after_final.json',{p:sha(ROOT/p) for p in guard})
        emit('final_verification.json',dict(status='PASS',utc=utc(),
             historical_sources_verified=len(guard),prediction_sha256=calc['prediction_sha256'],
             ranking_groups_verified=8,code_contract_unchanged=True,
             independent_qa_sha256=sha(OUT/'verification.json'),
             report_sha256=sha(OUT/'REPORT.md'),literature_screening_sha256=sha(OUT/'LITERATURE_SCREENING.md'),
             automation_paused='status = "PAUSED"' in AUTOMATION.read_text(encoding='utf-8'),
             stopped_acquisition_preserved=STOP.exists()))
        artifacts={str(p.relative_to(ROOT)):sha(p) for parent in (OUT,CODE) for p in parent.rglob('*') if p.is_file() and p.suffix!='.lock'}
        emit('completion.json',dict(status='COMBINED_ITU_COMPLETE_LITERATURE_SCREENING_WITH_GAPS',completed_utc=utc(),
            verification='PASS',execution_lock='RELEASE_ON_SCRIPT_EXIT',
            local_test_hours=687,uk_station_days=236,uk_station_hours=5664,uk_common_hours=5538,
            new_combined_itu_variants=2,new_independent_non_itu_paper_models=0,
            additional_paper_candidates_screened=8,all_candidate_models_reproduced=False,
            all_paper_fulltexts_and_SI_saved=False,new_training=0,new_ml_inference=0,new_weather_requests=0,
            fixed_physical_reference_authorized=True,supplementary_material_saving_authorized=True,
            old_weather_worker_remains_stopped=True,t012_complete=False,artifacts=artifacts))
        print('REPORT AND COMPLETION SAVED; historical SHA guard PASS; combined ITU complete; literature gaps explicit.')

if __name__=='__main__': main()
