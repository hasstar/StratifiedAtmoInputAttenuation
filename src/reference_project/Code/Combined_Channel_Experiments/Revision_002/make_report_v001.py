"""Generate a source-linked report only from independently verified metrics."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[3]
    out=root/'Outputs/Experiments/Combined_Channel_Revision_002'
    qa=json.loads((out/'independent_qa.json').read_text())
    assert qa['status']=='PASS'
    df=pd.read_csv(out/'family_summary.csv')
    registry=json.loads((out/'frozen_selection.json').read_text())['models']
    names={
        'ours_frozen_top10':'本文 Top10 BiLSTM（冻结）',
        'train_mean':'训练均值常数', 'ridge':'Ridge','elastic_net':'ElasticNet',
        'svr_rbf':'RBF-SVR','random_forest':'随机森林','xgboost':'XGBoost',
        'bai2022_mlp_adapted':'Bai2022 MLP（输入/实现适配）',
        'bai2022_lstm_t0_adapted':'Bai2022 LSTM-T0（输入/实现适配）',
        'tabm2025_official_budgeted':'TabM2025（官方架构，固定预算）',
        'physics_geom_circular':'P.838 + 几何湿路径（圆极化）',
        'physics_p618_hourly_circular':'P.618 小时代理（非年度标准验证）'}
    def metric(cohort,group,family):
        rows=df[(df.cohort==cohort)&(df['group']==group)&(df.family==family)]
        assert len(rows)==1
        return rows.iloc[0]
    def text(row):
        return f'{row.rmse_mean_db:.4f}' if row.runs==1 else f'{row.rmse_mean_db:.4f} ± {row.rmse_sd_db:.4f}'
    order=['ours_frozen_top10','bai2022_mlp_adapted','bai2022_lstm_t0_adapted',
           'tabm2025_official_budgeted','svr_rbf','random_forest','xgboost','ridge','elastic_net','train_mean']
    lines=['# T012：统一输入对比实验，第一批结果','',
           '状态：18个新模型检查点已完成；独立数值审计PASS。另有训练均值常数、冻结本文模型，以及两条只读复用的物理代理。全部新结果单独保存，旧代码、数据、模型和结果未覆盖。','',
           '本报告回答“同样十个气象输入，哪些模型在当前数据上表现更好”。它不把年度统计模型、未来信道预测和小时同步估计混成一种任务。','',
           '## 1. 同输入主比较','',
           '指标为RMSE/dB，越低越好。±为三个固定种子17/29/43之间的样本标准差，不是置信区间；单次值不虚构种子方差。没有按测试成绩挑种子。','',
           '| 方法 | LOCAL测试687小时 | GBS两站39天，1847合格站点小时 | 保存运行数 |',
           '|---|---:|---:|---:|']
    for family in order:
        a=metric('LOCAL_test','overall',family);b=metric('GBS39_pooled','eligible_all',family)
        lines.append(f'| {names[family]} | {text(a)} | {text(b)} | {int(a.runs)} |')
    lines+=['','GBS1872个天气完整站点小时全部保留预测；1847小时进入主评价，另25小时按原缺测/覆盖规则保留但不评分。两站分别为Chilbolton914小时、Sparsholt933小时。两站相邻且共享气象，不是两个独立气候重复。','',
            '### 外站分站结果','',
            '| 方法 | Chilbolton RMSE/dB | Sparsholt RMSE/dB |','|---|---:|---:|']
    # Match actual station spelling rather than assuming case.
    station_cohorts=sorted(c for c in df.cohort.unique() if c.startswith('GBS39_') and c!='GBS39_pooled')
    assert len(station_cohorts)==2 and 'chilbolton' in station_cohorts[0].lower()
    for family in order:
        a=metric(station_cohorts[0],'eligible_all',family);b=metric(station_cohorts[1],'eligible_all',family)
        lines.append(f'| {names[family]} | {text(a)} | {text(b)} |')
    lines+=['','### LOCAL分天气结果','',
            '| 方法 | 雨天 RMSE/dB | 干燥阴天 RMSE/dB | 晴干 RMSE/dB |','|---|---:|---:|---:|']
    for family in order:
        rows=[metric('LOCAL_test',g,family) for g in ['rain','dry_overcast','clear_dry']]
        lines.append('| '+names[family]+' | '+' | '.join(text(r) for r in rows)+' |')
    lines+=['','## 2. 物理代理：不同输入、不同衰减分量，单列','',
            '| 方法 | LOCAL整体 RMSE/dB | 雨天 RMSE/dB |','|---|---:|---:|']
    for family in ['physics_geom_circular','physics_p618_hourly_circular']:
        lines.append(f"| {names[family]} | {text(metric('LOCAL_test','overall',family))} | {text(metric('LOCAL_test','rain',family))} |")
    lines+=['','物理项逐小时预测直接复用Rev016，不运行旧入口。它们额外使用雨量、雨层高度、仰角与极化，而且只描述雨衰，不等同于总相对晴空衰减。P.618小时代理不能写作完整ITU年度标准模型；也不能从小时平均雨率伪造1分钟R0.01。','',
            '## 3. 这一批结果能说明什么','']
    local=df[(df.cohort=='LOCAL_test')&(df['group']=='overall')&df.comparison_group.eq('common_top10')]
    external=df[(df.cohort=='GBS39_pooled')&(df['group']=='eligible_all')]
    for label,subset in [('LOCAL',local),('GBS39迁移',external)]:
        best=subset.loc[subset.rmse_mean_db.idxmin()]
        ours=subset[subset.family=='ours_frozen_top10'].iloc[0]
        lines.append(f"- {label}：按本轮家族平均RMSE，最低为{names[best.family]}（{best.rmse_mean_db:.4f}dB）；冻结本文模型为{ours.rmse_mean_db:.4f}dB。这个排序不用于反过来改变参数。")
    if any(subset.loc[subset.rmse_mean_db.idxmin(),'family']!='ours_frozen_top10' for subset in [local,external]):
        lines.append('- 本轮不支持“本文模型在所有比较条件下均最优”的表述；较强对手和跨站退化结果均保留。')
    lines+=['- 按LOCAL自然日成簇的2000次配对重抽结果另存day_cluster_bootstrap.json。差值为对比方法RMSE减本文RMSE，负值有利于对比方法。它仅反映本测试集重抽不确定性，不消除随机小时划分的时序相关和既往测试可见性。',
            '- 本文冻结模型曾经历既往搜索和RFE；新传统方法是每家族三组验证候选，Bai网络按论文预算早停，TabM是固定配置。这里统一了数据、输入、目标与计分，不宣称总调参预算完全相同。','',
            '## 4. 复现层级和输入差异','',
            '- Bai2022：沿论文层宽与零历史设定实现，改用当前Top10、小时数据和PyTorch；缺失的批大小/激活等细节已预先固定并记录。不是原七地面传感器/LASSO完整复现，也不是作者代码。',
            '- TabM2025：固定提交的作者官方实现，许可证和源码SHA保留；使用基础无数值嵌入版本、固定CPU预算，不冒称论文全量超参搜索复现。',
            '- 需要历史信道的Al Homssi/Kaselimi等不能直接放进天气-only同输入表；SST需要分钟雨量，Biscarini/Queyrel等还需要额外物理变量/校准统计。未满足输入或未建立完整方法证据的模型列为未执行，不伪造结果。',
            '- 标准与近年论文的完整候选/输入/代码矩阵见[模型登记表](../../../Code/Combined_Channel_Experiments/Revision_002/MODEL_REGISTRY.md)。','',
            '## 5. 数据范围和停止状态','',
            'LOCAL为原2404/343/687随机小时划分，旧测试成绩早已查看，不能称新盲测或未来预测。GBS为原审核通过39天；新下载收尾时已采齐的Chilbolton143天/Sparsholt93天尚未通过同等级全字段独立评价门槛，本批未混入它们。部分December pressure不完整日期也不插补。',
            'GBS频率20.7GHz、LOCAL19.13GHz；外站原生晴空参考与本地冻结参考、固定高度最近LOS格点与LOCAL逐时等压面双线性采样不同。全部外站结果是无重标定的近似跨域迁移，不做外站训练、偏差校正或归一化重估。',
            '气象STOP_REQUESTED保留，tvt仍PAUSED；公共论文/模型源码读取不等于恢复气象下载。','',
            '## 6. 审计与可复现交付','',
            f"- {qa['protected_sources']}个保护源SHA前后及独立审计一致；冻结本文预测与既有文件逐值一致。",
            f"- {qa['metric_rows']}行分组指标独立复算通过，最大差{qa['maximum_metric_delta']:.3g}dB；新检查点在LOCAL/GBS分别重放校验。",
            '- 新插补器/标准化器仅拟合LOCAL train2404小时；外站无fit，标签不插补，预测不裁零。',
            '- 5个传统家族×3候选 + RF/XGBoost各两个追加种子 + 3个神经家族×3种子 = 28次真实训练，保存18个最终检查点；常数控制另列。',
            '- 逐小时预测：predictions/local_test.parquet、local_validation.parquet、gbs39.parquet；指标：metrics.csv、family_summary.csv；所有选择、曲线、模型及源指纹在同目录。',
            '- 独立审计v002在执行前修正了v001潜在的布尔掩码视图原位修改风险，并增加外站检查点重放；v001未执行且保持原样。此修正没有改变训练或指标定义。',
            '- 原下载器/原模型/原实验入口未重跑。本批完成不表示全部T012动态信道/年度统计模型比较已经完成。','',
            '## 来源','',
            '[Bai等，IEEE TAP2022](https://doi.org/10.1109/TAP.2021.3137285)；[TabM官方代码](https://github.com/yandex-research/tabm)；[TabM论文](https://arxiv.org/abs/2410.24210)；[ITU-R P.838-3](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf)；[ITU-R P.618-14](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-14-202308-I!!PDF-E.pdf)。','']
    # Three levels up from Outputs/Experiments/revision is the project root.
    with (out/'COMPARISON_REPORT.md').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    source={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__),out/'family_summary.csv',out/'independent_qa.json',out/'COMPARISON_REPORT.md']}
    with (out/'report_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(source,f,indent=2)
    print('REPORT_CREATED')


if __name__=='__main__':
    main()
