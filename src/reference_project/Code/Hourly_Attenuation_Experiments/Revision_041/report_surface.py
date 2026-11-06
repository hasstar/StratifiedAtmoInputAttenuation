"""Full-path tables and fair same-k comparison, never selects by test results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import runtime as r


def write_csv(path, records):
    r.require(bool(records), 'Cannot write an empty required table')
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plots(directory, chosen):
    """Save three PDF/PNG comparisons with 6 panels each, including true negatives."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style='whitegrid', font='DejaVu Sans')
    table = r.pd.read_csv(directory / 'all_states_metrics.csv')
    data = table.loc[table['split'] == 'test']
    colors = ['#4575b4', '#d73027']
    groups = [('overall', 'Overall', 687), ('rain', 'Rain', 51), ('dry_overcast', 'Dry overcast', 158)]
    for limit, tag in ((50, 'same_k_1_to_50'), (10, 'same_k_1_to_10'), (156, 'full_path_context')):
        # --- Plot ---
        fig = plt.figure(figsize=(13, 9), dpi=150)
        grid = gridspec.GridSpec(3, 2, figure=fig, wspace=0.12, hspace=0.35,
                                left=0.08, right=0.93, top=0.87, bottom=0.09)
        axes = []
        for i, (group, label, count) in enumerate(groups):
            for j, (metric, ylabel) in enumerate((('rmse_db', 'RMSE (dB)'), ('r2', r'$R^2$'))):
                ax = fig.add_subplot(grid[i, j])
                axes.append(ax)
                for pool, name, color, style in zip(('all156', 'surface50'),
                        ('All-atmosphere RFE (156 candidates)', 'Surface/base-only RFE (50 candidates)'), colors, ('-', '--')):
                    subset = data.loc[(data['pool'] == pool) & (data['weather'] == group) & (data['k'] <= limit), ['k', metric]].dropna().sort_values('k')
                    ax.plot(subset['k'], subset[metric], color=color, linestyle=style, linewidth=2.1,
                            marker='o' if limit <= 10 else None, markersize=4, label=name)
                if metric == 'r2':
                    ax.axhline(0, linewidth=0.8, color='lightgrey', zorder=0)
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                ax.set_title(r'$\bf{(' + chr(97 + i*2 + j) + r')}$ ' + label + f' (n={count})', loc='left', fontsize=11)
                ax.set_ylabel(ylabel, color='dimgrey', fontsize=10)
                ax.set_xlim(1, limit)
                if limit <= 10:
                    ax.set_xticks(range(1, 11))
                if i == 2:
                    ax.set_xlabel('Retained features, k', color='dimgrey')
                ax.grid(False)
                ax.tick_params(axis='both', which='both', length=0, labelcolor='dimgrey')
        sns.despine(left=True, bottom=True)
        for ax in axes:
            ax.patch.set_edgecolor('lightgrey')
            ax.patch.set_linewidth(0.8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2,
                   frameon=True, facecolor='white', framealpha=0.8, edgecolor='lightgrey', labelcolor='dimgrey')
        title = 'TabM: independent RFE paths at matched feature counts' if limit <= 50 else 'TabM: complete candidate-pool RFE paths'
        fig.suptitle(title, y=0.99, color='dimgrey', fontsize=14)
        note = f'Training-CV-selected k: all-atmosphere={chosen["all156"]}, surface/base={chosen["surface50"]}. Both use the same 687 test hours.'
        fig.text(0.5, 0.025, note, ha='center', color='dimgrey', fontsize=9)
        # --- Save ---
        for suffix in ('pdf', 'png'):
            fig.savefig(directory / (tag + '.' + suffix), dpi=150, bbox_inches='tight')
        plt.close(fig)


def build(data, manifest):
    directory = r.OUT / 'delivery_001'
    r.require(not directory.exists(), 'Preserve existing delivery, inspect any partial report; no overwrite')
    directory.mkdir()
    metric_rows, hourly_rows, path_rows, feature_rows, role_rows, importance_rows, source_index = [], [], [], [], [], [], []
    chosen = {}
    chosen_rows = []
    for pool, count, parent in (('all156', 156, r.P32), ('surface50', 50, r.OUT)):
        removed = []
        candidates = []
        for k in range(count, 0, -1):
            folder = parent / 'models/tabm' / f'k_{k:04d}'
            state_path, eval_path, qa_path = folder / 'state.json', folder / 'evaluation.json', folder / 'qa.json'
            state, evaluation, qa = r.read(state_path), r.read(eval_path), r.read(qa_path)
            state_sha = r.sha(state_path)
            r.require(qa['status'] == 'PASS' and qa['state_sha256'] == state_sha
                      and qa['evaluation_sha256'] == r.sha(eval_path), 'Committed source state QA')
            r.require(len(state['feature_names']) == k and state['model'] == 'tabm', 'State identity')
            if pool == 'surface50':
                r.require(state['manifest'] == manifest['fingerprint'], 'New state manifest')
            candidates.append((state['cv_mean_rmse_db'], k))
            source_index.append(dict(pool=pool, k=k, state_path=r.rel(state_path), state_sha256=state_sha,
                        evaluation_path=r.rel(eval_path), evaluation_sha256=r.sha(eval_path),
                        qa_path=r.rel(qa_path), qa_sha256=r.sha(qa_path)))
            path_rows.append(dict(pool=pool, k=k, cv_mean_rmse_db=state['cv_mean_rmse_db'], cv_se_rmse_db=state['cv_se_rmse_db'],
                     removed_next=state['removed_feature_next'], retained_names_json=json.dumps(state['feature_names']),
                     removed_so_far_json=json.dumps(removed), state_sha256=state_sha))
            for position, name in enumerate(state['feature_names'], 1):
                feature_rows.append(dict(pool=pool, k=k, position=position, feature_name=name,
                     feature_type='surface_base' if name.startswith('surface__') else 'pressure_level',
                     removed_next=int(name == state['removed_feature_next']), state_sha256=state_sha))
            if k > 1:
                removed.append(state['removed_feature_next'])
            if pool == 'surface50':
                for role in [*state['folds'], state['full_fit']]:
                    role_rows.append(dict(pool=pool, k=k, fold=role['fold'], reused=int(role['reused']), new_fit=int(role['new_fit']),
                             model_file=role['model_file'], model_sha256=role['model_sha256'], train_hour_sha256=role['train_hour_sha256'],
                             seed=role['seed'], epochs=role['fit_metadata']['selected_epoch']))
                for item in state['permutation_importance']:
                    importance_rows.append(dict(k=k, feature_name=item['feature_name'], mean_delta_rmse_db=item['mean_delta_rmse_db'],
                            fold1_delta_rmse_db=item['fold_delta_rmse_db'][0], fold2_delta_rmse_db=item['fold_delta_rmse_db'][1],
                            fold3_delta_rmse_db=item['fold_delta_rmse_db'][2], removed_next=int(item['feature_name'] == state['removed_feature_next'])))
            for split in ('validation', 'test'):
                rec = evaluation['results'][split]
                r.require(rec['hour_bjt'] == data[split].hour_bjt.astype(str).tolist(), 'Comparison exact hour alignment')
                r.require(rec['truth_db'] == data[split][r.TARGET].to_numpy(float).tolist(), 'Comparison exact targets')
                for i, hour in enumerate(rec['hour_bjt']):
                    weather = [group for group in r.WEATHER[1:] if rec['weather_masks'][group][i]]
                    r.require(len(weather) == 1, 'Exclusive weather masks')
                    hourly_rows.append(dict(pool=pool, k=k, split=split, hour_bjt=hour, weather=weather[0],
                                truth_db=rec['truth_db'][i], prediction_db=rec['prediction_db'][i]))
                independently_produced = r.scores(data[split], rec['prediction_db'])
                for group in r.WEATHER:
                    mask = r.np.ones(len(data[split]), bool) if group == 'overall' else data[split][group].to_numpy(bool)
                    truth = r.np.asarray(rec['truth_db'])[mask]
                    pred = r.np.asarray(rec['prediction_db'])[mask]
                    metrics = independently_produced[group]
                    metric_rows.append(dict(pool=pool, k=k, split=split, weather=group,
                        **{key: metrics[key] for key in ('n', 'n_dates', 'rmse_db', 'mae_db', 'bias_db', 'r2', 'pearson_r', 'pearson_r_squared')},
                        truth_std_db=float(truth.std()), prediction_std_db=float(pred.std()),
                        prediction_unique_values=len(set(pred.tolist())), state_sha256=state_sha))
        chosen[pool] = min(candidates)[1]
    # All representative choices depend solely on committed training-CV scores.
    for row in metric_rows:
        if row['k'] == chosen[row['pool']]:
            chosen_rows.append(dict(selection='minimum_training_CV_RMSE', **row))
    lookup = {(row['pool'], row['k'], row['split'], row['weather']): row for row in metric_rows}
    comparison_rows = []
    for k in range(1, 51):
        for split in ('validation', 'test'):
            for group in r.WEATHER:
                a, b = lookup[('surface50', k, split, group)], lookup[('all156', k, split, group)]
                r.require(a['n'] == b['n'], 'Matched n')
                row = dict(k=k, split=split, weather=group, n=a['n'])
                for metric in ('rmse_db', 'r2', 'mae_db', 'bias_db'):
                    row['surface_' + metric], row['all_' + metric] = a[metric], b[metric]
                    row['surface_minus_all_' + metric] = None if a[metric] is None or b[metric] is None else a[metric] - b[metric]
                comparison_rows.append(row)
    for name, records in (('all_states_metrics.csv', metric_rows), ('hourly_predictions.csv', hourly_rows),
             ('rfe_paths.csv', path_rows), ('retained_features.csv', feature_rows), ('surface_model_roles.csv', role_rows),
             ('surface_permutation_importance.csv', importance_rows), ('same_k_comparison.csv', comparison_rows),
             ('training_cv_selected_states.csv', chosen_rows)):
        write_csv(directory / name, records)
    r.emit(directory / 'representative_selection.json', dict(selected_k=chosen, rule=manifest['representative_selection'], test_used=False))
    r.progress(stage='REPORTING_FIGURES', completed_states=50)
    plots(directory, chosen)
    win_count = sum(row['surface_minus_all_rmse_db'] > 0 for row in comparison_rows if row['split'] == 'test' and row['weather'] == 'overall')
    lines = ['# TabM 地面/非分层基础变量 RFE 对照', '',
             '新增 surface50 路径从 50 个候选变量独立删至 1，共 50 个状态；全参数参考为既有 all156 独立 156→1 路径。', '',
             '## 方法与范围', '',
             'Surface/base-only 指冻结 156 列中的 50 个 surface__ 基础变量，包含单层、柱积分量和诊断量，并非严格近地面站观测。两组候选池不同，各自独立计算训练内三折置换重要性，不能从全参数路径截取替代。', '',
             '保持相同的 2404/343/687 小时划分、目标、天气、TabM 配置、seed17、54 epochs、预处理及评分方法。k50 的四个角色复用已逐一重新核验的相同权重；k49–1 新拟合 196 个角色。验证与测试不决定删除顺序，也没有新增调参、英国拟合或天气补采。', '',
             '## 比较结果', '',
             f'训练内 CV 最小 RMSE 对应：全参数路径 k={chosen["all156"]}，surface/base 路径 k={chosen["surface50"]}。这是各自候选池范围内的选择，并非按测试最优筛选。', '',
             f'同为 k=1–50 时，现有全参数路径在总体测试 RMSE 上较低的状态为 {win_count}/50；其余状态和并列情况完整保留于 same_k_comparison.csv。此计数仅是描述，不是独立重复试验或统计显著性证据。', '',
             '| 候选池 | CV 选择 k | 总体测试 RMSE (dB) | 总体测试 R² |', '|---|---:|---:|---:|']
    for pool in ('all156', 'surface50'):
        row = lookup[(pool, chosen[pool], 'test', 'overall')]
        lines.append(f'| {pool} | {chosen[pool]} | {row["rmse_db"]:.6f} | {row["r2"]:.6f} |')
    lines += ['', '## 阅读文件', '',
              '- rfe_paths.csv：每轮完整保留集合、此前删除顺序与本轮下一步删除变量。retained_features.csv 展开为逐变量记录。',
              '- surface_permutation_importance.csv：每个候选变量的三折置换增量与均值。surface_model_roles.csv：200 个角色的复用/新拟合身份和权重 SHA。',
              '- all_states_metrics.csv 与 hourly_predictions.csv：验证、测试各天气完整评分和逐小时预测；另存预测标准差与唯一值数，便于识别常数或近常数输出。',
              '- same_k_comparison.csv：相同 k 下 surface 减 all 的指标差。RMSE 差为正表示 all 更低；R² 差为正表示 surface 更高。',
              '- same_k_1_to_50、same_k_1_to_10、full_path_context：三组 PDF/PNG，均包含总体、雨、无雨阴天的 RMSE/R²。', '',
              '测试集此前已有访问，这些是同一冻结划分上的探索性对照。单一固定种子不支持不确定性或普遍优势的断言。若某个状态更差，仍完整保留。', '',
              '数值及 CSV 独立检查与实际图片视觉检查分开。自动生成本报告时视觉验收尚未进行，不能据此宣称最终提交交付完成。', '']
    with (directory / 'REPORT_中文.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    file_hashes = {p.name: r.sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    r.emit(directory / 'source_index.json', dict(manifest=manifest['fingerprint'], states=source_index, files_sha256=file_hashes, utc=r.now()))
    import verify_delivery
    qa = verify_delivery.verify(directory, r.ROOT)
    r.emit(directory / 'NUMERICAL_QA.json', dict(qa, utc=r.now()))
    r.emit(r.OUT / 'completion_numerical.json', dict(status='PASS', new_states=50, roles=200, reused=4, new_fits=196,
           delivery=r.rel(directory), numerical_qa_sha256=r.sha(directory / 'NUMERICAL_QA.json'), actual_visual_review=False, utc=r.now()))
    r.progress(stage='NUMERICS_COMPLETE_VISUAL_REVIEW_PENDING', completed_states=50, delivery=r.rel(directory))
