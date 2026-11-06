"""Post-fit descriptive interpretation; no model fitting or inference."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / 'Outputs/Experiments/Combined_Channel_Revision_002'
    dest = out / 'interpretation_001'
    if dest.exists():
        raise FileExistsError(dest)
    assert json.loads((out / 'independent_qa.json').read_text())['status'] == 'PASS'
    local = pd.read_parquet(out / 'predictions/local_test.parquet')
    external = pd.read_parquet(out / 'predictions/gbs39.parquet')
    selection = json.loads((out / 'frozen_selection.json').read_text())
    mapping = {'ours_frozen_top10': 'ours_frozen_top10',
               **{r['id']: r['family'] for r in selection['models']}}
    mapping.update({k: k for k in ['physics_geom_circular', 'physics_p618_hourly_circular']})
    masks = {g: local[g].to_numpy(dtype=bool, copy=True)
             for g in ['rain', 'dry_overcast', 'clear_dry']}
    flags = np.column_stack(list(masks.values()))
    assert not (flags.sum(axis=1) > 1).any()
    masks['other_or_unclassified'] = ~flags.any(axis=1)
    assert sum(int(m.sum()) for m in masks.values()) == len(local) == 687
    y = local.truth_db.to_numpy(float)
    records = []
    for model, family in mapping.items():
        p = local[model].to_numpy(float)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        for group, mask in masks.items():
            err = p[mask] - y[mask]
            records.append(dict(model=model, family=family, group=group, n=len(err),
                                rmse_db=float(np.sqrt(np.mean(err**2))),
                                mae_db=float(np.mean(abs(err))), bias_db=float(np.mean(err))))
    scores = pd.DataFrame(records)
    # Verify the already-reported three strata using independently expressed norms.
    old = pd.read_csv(out / 'metrics.csv')
    for row in records:
        if row['group'] == 'other_or_unclassified':
            continue
        match = old[(old.cohort == 'LOCAL_test') & (old.model == row['model']) &
                    (old['group'] == row['group'])]
        if len(match):
            assert len(match) == 1
            assert int(match.iloc[0]['n']) == row['n']
            np.testing.assert_allclose(match.iloc[0].rmse_db, row['rmse_db'], rtol=0, atol=1e-12)
    # Recombine ALL mutually exclusive strata, retaining the omitted 156 hours.
    max_recombination_delta = 0.0
    for model in mapping:
        rows = scores[scores.model == model]
        joined = np.sqrt(np.sum(rows.n * rows.rmse_db**2) / 687)
        direct = float(np.linalg.norm(local[model].to_numpy(float) - y) / np.sqrt(687))
        np.testing.assert_allclose(joined, direct, rtol=0, atol=1e-12)
        max_recombination_delta = max(max_recombination_delta, abs(joined - direct))
    eligible = external.primary_score_eligible.to_numpy(dtype=bool, copy=True)
    truth = external.truth_db.to_numpy(float)[eligible]
    assert len(truth) == 1847 and np.isfinite(truth).all()
    constant = external.train_mean.to_numpy(float)[eligible]
    assert np.ptp(constant) == 0
    distribution = dict(n=len(truth), mean_db=float(truth.mean()),
                        sd_sample_db=float(truth.std(ddof=1)), min_db=float(truth.min()),
                        max_db=float(truth.max()), above_0p5_db=int((truth > .5).sum()),
                        above_1_db=int((truth > 1).sum()),
                        train_mean_prediction_db=float(constant[0]),
                        note='Attenuation thresholds are descriptive, not rain labels or new evaluation strata.')
    bootstrap = json.loads((out / 'day_cluster_bootstrap.json').read_text())
    selected_intervals = [b for b in bootstrap
                          if b['model'].startswith(('tabm2025_', 'xgboost_'))]
    assert len(selected_intervals) == 6
    assert all(b['ci95_db'][0] < 0 < b['ci95_db'][1] for b in selected_intervals)
    dest.mkdir()
    scores.to_csv(dest / 'all_local_weather_groups.csv', index=False)
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    nature='POST_HOC_DESCRIPTIVE_NO_SELECTION_OR_MODEL_CHANGE',
                    local_group_counts={g: int(m.sum()) for g, m in masks.items()},
                    gbs39_target_distribution=distribution,
                    local_tabm_xgboost_paired_intervals=selected_intervals,
                    all_strata_recombine_overall=True,
                    maximum_recombined_rmse_delta_db=float(max_recombination_delta),
                    input_sha256={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [out / 'predictions/local_test.parquet',
                                            out / 'predictions/gbs39.parquet', out / 'metrics.csv',
                                            out / 'day_cluster_bootstrap.json']},
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (dest / 'evidence.json').open('x', encoding='utf-8') as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    other = scores[scores['group'] == 'other_or_unclassified'].groupby('family').rmse_db.agg(['count', 'mean', 'std'])
    lines = [
        '# 第一批对比结果：解释补充与完整分组', '',
        '本补充只读取已完成、独立QA通过的预测，不重新训练、选参数、选种子或推理。主报告、原指标及全部不利结果保留。以下补充在查看结果后形成，属于描述性分析，不作为预注册检验。', '',
        '## 主要结论', '',
        '- 本轮比较的是自然RFE的冻结Top10模型，不是本文完整156特征模型；不能据此评价完整模型的性能排名。',
        '- TabM和XGBoost的LOCAL平均RMSE点估计低于冻结Top10，但六个固定种子的自然日成簇配对差值95%区间全部跨零。因此可报告数值差异，不能据此宣称已建立统计显著优势，也不能把跨零解释为模型等效。',
        '- GBS39的训练均值常数控制优于本轮所有学习方法。现有结果不足以支持复杂模型在该外站群体上优于简单常数；这一结果不因本地排名较好而被隐藏。',
        '- GBS目标集中于弱衰减，是常数控制表现较好的可见背景；频率、晴空参考和气象采样差异也存在，但本轮没有分离各因素的因果贡献。不得将观测到的误差直接归因为某一个因素。', '',
        '## 外站目标分布（原1847个合格小时）', '',
        f'- 均值{distribution["mean_db"]:.6f} dB；样本标准差{distribution["sd_sample_db"]:.6f} dB；范围[{distribution["min_db"]:.6f}, {distribution["max_db"]:.6f}] dB。',
        f'- 高于0.5 dB：{distribution["above_0p5_db"]}小时；高于1 dB：{distribution["above_1_db"]}小时。阈值仅描述信号衰减，不当作降雨分类，不改变原计分集合。',
        f'- 训练均值常数实际输出{distribution["train_mean_prediction_db"]:.6f} dB，来自LOCAL训练集，未用外站标签调整。', '',
        '## LOCAL配对重抽区间', '',
        '156个自然日成簇，2000次重抽，固定种子20261016；差值=对比方法RMSE−冻结Top10 RMSE。保持历史随机小时划分，不构成独立时序盲测。', '',
        '| 运行 | RMSE差值/dB | 95%重抽区间/dB |',
        '|---|---:|---:|',
    ]
    for b in selected_intervals:
        lines.append(f'| {b["model"]} | {b["delta_rmse_model_minus_ours_db"]:.4f} | [{b["ci95_db"][0]:.4f}, {b["ci95_db"][1]:.4f}] |')
    lines += ['', '## LOCAL分天气表的完整性补充', '',
              '主报告列出了rain51、dry_overcast158、clear_dry322小时，合计531小时，并非全部687小时。另有156小时不属于上述三组；它们从未被整体评分删除。补充按原三个布尔标记的补集定义other_or_unclassified，不猜测天气类型、不重划分样本。四组样本互斥且完整覆盖687小时，平方误差加权重组与原整体RMSE一致。', '',
              '下表列出补集的全部方法，±为固定种子之间的样本标准差；物理代理仍属于不同输入和分量的辅助参考。完整逐运行四组值见[CSV](all_local_weather_groups.csv)。', '',
              '| 方法族 | other_or_unclassified（156小时）RMSE/dB |', '|---|---:|']
    for family, row in other.iterrows():
        value = f'{row["mean"]:.4f}'
        if row['count'] > 1:
            value += f' ± {row["std"]:.4f}'
        lines.append(f'| {family} | {value} |')
    lines += ['', '## 未完成边界', '',
              '本轮新训练、推理和结果审计已完成，不等于全部T012完成。需要分钟雨量、历史信道或完整物理场的论文/标准尚未执行；未借此授权恢复下载。新增143/93日采集范围尚未按同等字段QA并入评价；外站本批仍是原39天。下载STOP标志保留，tvt保持暂停。', '',
              '证据：[独立QA](../independent_qa.json)、[主报告](../COMPARISON_REPORT.md)、[描述性证据](evidence.json)、[模型/输入矩阵](../../../../Code/Combined_Channel_Experiments/Revision_002/MODEL_REGISTRY.md)。', '']
    with (dest / 'INTERPRETATION_ADDENDUM.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(json.dumps({'status': 'DESCRIPTIVE_CHECKS_PASS', **evidence['local_group_counts'],
                      'gbs_distribution': distribution,
                      'max_delta': max_recombination_delta}, ensure_ascii=False))


if __name__ == '__main__':
    main()
