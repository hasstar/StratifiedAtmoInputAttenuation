"""Package verified UK tables and figures; no model selection or retraining."""
from pathlib import Path
import json
import pandas as pd
from freeze import *
from transport import stage_lock

def main():
    with stage_lock('uk_results_package'):
        evidence=read(OUT/'evaluation/independent_numeric_qa.json');fig=read(OUT/'figures/FIGURE_QA.json')
        need(evidence['status']=='PASS' and fig['all_600_artist_values_exact'],'Numeric/figure gate')
        table=pd.read_csv(OUT/'evaluation/UK_100_states_six_metrics.csv',float_precision='round_trip')
        labels=dict(tabm='TabM（本文模型）',bai_lstm='Bai LSTM',bilstm='BiLSTM',transformer='Transformer',mlp='MLP',
            catboost='CatBoost',gcn='GCN',random_forest='Random Forest',svr_rbf='SVR-RBF',elastic_net='Elastic Net')
        lines=['# 英国数据集：10模型冻结RFE 10→1输入测试','',
            '已完成100个冻结状态的数值测试，独立输入QA、LOCAL预测重放和独立指标QA通过。六张图已导出；最终人工视觉检查尚待完成，不能将本文件等同于全部交付完成。','',
            '## 测试口径','',
            '- 10模型均沿用各自LOCAL独立RFE的10→1输入和原训练权重、预处理器、配置及种子17。没有英国重训、预处理重拟合、英国删列、选种子或测试调参。',
            '- 英国Chilbolton和Sparsholt共236站点日、5664小时；沿用既有71小时质量排除，所有模型总体均为5593小时，雨天735小时，干燥阴天2317小时。总体包含other，不按模型成绩或天气额外删样本。',
            '- R²=1−SSE/SST，为决定系数，不是相关系数平方；负值完整保留。RMSE单位dB。',
            '- 每条曲线按各模型自己的输入路径排列，k相同不代表气象参数相同。横坐标1→10仅为显示顺序，原RFE方向156→1。',
            '- 英国标签沿用GBS 20.7 GHz原生标签−10log10(小时平均线性透射率)；没有重估CDF、拟合偏移或裁零。该英国集合过去已查看，不声称首次盲测。',
            '- 压力层保持既有视线取样约定：1 hPa复用已核验逐小时几何，其他层使用原有标准大气固定高度近似。新增层是同一近似的延伸，不是观测逐小时高度。',
            '- 补采只覆盖现有站点/日期：24项NCAR精确子集，K指数、Total totals指数和降水类型使用已授权官方CDS；旧下载队列和停止标记未改。','',
            '## 文件','',
            '- 完整六指标表：[UK_100_states_six_metrics.csv](evaluation/UK_100_states_six_metrics.csv)',
            '- 天气、站点、样本数和实际日期：[metrics_all_weather_station.csv](evaluation/metrics_all_weather_station.csv)',
            '- 所有状态输入顺序：[retained_inputs_all_100_states.csv](evaluation/retained_inputs_all_100_states.csv)',
            '- 逐小时预测：[all_hourly_predictions.parquet](evaluation/all_hourly_predictions.parquet)',
            '- 输入来源与SHA：[new_field_provenance.parquet](inputs/new_field_provenance.parquet)',
            '- 冻结权重及路径：[frozen_states_100.json](frozen_states_100.json)','',
            '## 六张图','',
            '每图10条原始折线，TabM加粗并带青色荧光；荧光不表示误差带。没有平滑或隐藏负R²。不同面板采用覆盖全部数据的纵轴范围。','']
        for f in fig['figures_detail']:
            lines += [f'![{f["id"]}](figures/{f["id"]}.png)','']
        for model in MODELS:
            rows=table[table.model.eq(model)].sort_values('k',ascending=False)
            lines += ['## '+labels[model],'','| 参数数 | 雨天R² | 阴天R² | 总体R² | 雨天RMSE | 阴天RMSE | 总体RMSE |',
                '|---:|---:|---:|---:|---:|---:|---:|']
            columns=['rain_r2','dry_overcast_r2','overall_r2','rain_rmse_db','dry_overcast_rmse_db','overall_rmse_db']
            for _,r in rows.iterrows():lines.append('| '+str(int(r.k))+' | '+' | '.join(f'{r[c]:.6f}' for c in columns)+' |')
            lines.append('')
            rows.to_csv(OUT/'evaluation'/f'{model}_UK_six_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.17g',mode='x')
        lines+=['## 核验与异常','',
            f'- 独立指标实现使用math.fsum；校验{evidence["metrics_checked"]}个指标，最大绝对差{evidence["maximum_metric_abs_difference"]:.3g}。',
            f'- 重验{evidence["protected_files_checked"]}项历史保护文件SHA，全部一致；6240角色槽权重保持原样。',
            '- 原始下载失败、超时和中间文件全部保留；没有以修改旧结果的方式通过QA。',
            '- 已有质量排除理由见原数据exclusion_reason；新增气象负值与范围见inputs/new_field_summary.csv，未经裁零。','',
            '图表与表格是描述性跨域比较，不据此重新选择k或宣称等预算调参。完整精度请使用CSV，而非本页六位小数。','']
        with (OUT/'UK_RESULTS_CN.md').open('x',encoding='utf-8') as h:h.write('\n'.join(lines))
        failures=[]
        for sub in [OUT/'failures',OUT/'acquisition/http_failures']:
            for p in sorted(sub.glob('*.json')):failures.append(dict(path=rel(p),sha256=sha(p),record=read(p)))
        emit(OUT/'failure_ledger.json',failures)
        emit(OUT/'delivery_status.json',dict(status='NUMERICS_TABLES_SIX_FIGURES_REPORT_READY_PENDING_VISUAL_REVIEW',utc=now(),
            states=100,figures=6,training_calls=0,all_required_numerics_verified=True,visual_review_complete=False,
            report=rel(OUT/'UK_RESULTS_CN.md'),report_sha256=sha(OUT/'UK_RESULTS_CN.md'),
            tables_sha256={p.name:sha(p) for p in (OUT/'evaluation').glob('*.csv')},
            all_task_complete=False,old_automations_changed=False))
        print('UK_RESULTS_PACKAGE_READY_PENDING_VISUAL_REVIEW',flush=True)
if __name__=='__main__':main()
