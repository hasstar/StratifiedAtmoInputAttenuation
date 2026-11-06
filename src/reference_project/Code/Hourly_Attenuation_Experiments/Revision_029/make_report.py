"""Complete single-seed weather tables, 512-context layer effects and RFE ranks."""
from __future__ import annotations

import hashlib
import json

import runtime as r


def metric_rows(m):
    for kind in ('layer', 'rfe'):
        for key in r.keys(kind):
            state = r.read(r.state_path(kind, key))
            evaluation = r.read(r.evaluation_path(kind, key))
            records = {'train': state['full_fit']['training'], **evaluation['results']}
            for split, record in records.items():
                for weather in r.WEATHER:
                    yield {'kind': kind, 'key': key, 'n_features': state['n_features'], 'n_base': state['n_base'],
                           'n_pressure': state['n_pressure'], 'levels_hpa': ';'.join(map(str, state['levels_hpa'])),
                           'seed': 17, 'epochs': 53, 'split': split, 'weather': weather, **record['metrics'][weather]}


def feature_rows(m):
    for kind in ('layer', 'rfe'):
        for key in r.keys(kind):
            state = r.read(r.state_path(kind, key))
            yield {'kind': kind, 'key': key, 'n_features': len(state['feature_names']),
                   'feature_names_json': json.dumps(state['feature_names'], ensure_ascii=False)}


def curve_rows(m):
    for kind in ('layer', 'rfe'):
        for key in r.keys(kind):
            state = r.read(r.state_path(kind, key))
            for rec in [state['full_fit'], *state['folds']]:
                for history in rec['round_history']:
                    row = {'kind': kind, 'key': key, 'seed': 17, 'fit_role': 'formal' if rec['fold'] == 0 else 'internal_fold',
                           'fold': rec['fold'], 'round': history['round'], 'train_rmse_db': history['train_rmse_db'],
                           'train_objective_db': history['train_objective_db']}
                    for group in r.WEATHER:
                        row[f'train_{group}_rmse_db'] = history['train_weather_metrics'][group]['rmse_db']
                        row[f'internal_holdout_{group}_rmse_db'] = None if history['internal_holdout_metrics'] is None else history['internal_holdout_metrics'][group]['rmse_db']
                    yield row


def importance_rows(m):
    for key in r.keys('rfe'):
        state = r.read(r.state_path('rfe', key))
        for importance in state['permutation_importance']:
            yield {'k': key, 'feature_name': importance['feature_name'], 'mean_delta_rmse_db': importance['mean_delta_rmse_db'],
                   'fold_delta_rmse_db_json': json.dumps(importance['fold_delta_rmse_db']),
                   'deleted_next': importance['feature_name'] == state['removed_feature_next']}


def survival_rows(m):
    ranking = {}
    for key in r.keys('rfe'):
        state = r.read(r.state_path('rfe', key))
        if key == 1:
            ranking[state['feature_names'][0]] = 1
        else:
            ranking[state['removed_feature_next']] = key
    for name, rank in sorted(ranking.items(), key=lambda i: i[1]):
        yield {'survival_rank': rank, 'feature_name': name, 'type': 'base' if name.startswith('surface__') else 'pressure',
               'level_hpa': '' if name.startswith('surface__') else int(name.split('__')[0][1:])}


def layer_index():
    return {key: r.read(r.evaluation_path('layer', key))['results']['test']['metrics'] for key in r.keys('layer')}


def layer_pair_rows(m):
    table = layer_index()
    for i, level in enumerate(r.LEVELS):
        for without in range(1024):
            if without & (1 << i):
                continue
            with_level = without | (1 << i)
            for weather in r.WEATHER:
                a, b = table[without][weather], table[with_level][weather]
                yield {'level_hpa': level, 'mask_without': without, 'mask_with': with_level, 'weather': weather,
                       'n_hours': a['n_hours'], 'rmse_without_db': a['rmse_db'], 'rmse_with_db': b['rmse_db'],
                       'rmse_improvement_db': a['rmse_db'] - b['rmse_db'],
                       'mae_without_db': a['mae_db'], 'mae_with_db': b['mae_db'],
                       'bias_without_db': a['bias_db'], 'bias_with_db': b['bias_db']}


def layer_effect_rows(m):
    table = layer_index()
    for i, level in enumerate(r.LEVELS):
        for weather in r.WEATHER:
            values = r.np.asarray([table[key][weather]['rmse_db'] - table[key | (1 << i)][weather]['rmse_db'] for key in range(1024) if not key & (1 << i)])
            yield {'level_hpa': level, 'weather': weather, 'contexts': 512,
                   'mean_rmse_improvement_db': float(values.mean()), 'median_rmse_improvement_db': float(r.np.median(values)),
                   'fraction_helpful': float(r.np.mean(values > 0)),
                   'full_context_rmse_improvement_db': table[1023 ^ (1 << i)][weather]['rmse_db'] - table[1023][weather]['rmse_db']}


def bootstrap(a, b, group, name, m):
    r.require(a['hour_bjt'] == b['hour_bjt'] and a['truth_db'] == b['truth_db'], 'Paired date alignment')
    keep = r.np.ones(len(a['truth_db']), bool) if group == 'overall' else r.np.asarray(a['weather_masks'][group], bool)
    dates = r.np.asarray([hour[:10] for hour in a['hour_bjt']])[keep]
    unique, ids = r.np.unique(dates, return_inverse=True)
    y = r.np.asarray(a['truth_db'])[keep]
    sums = [r.np.bincount(ids, weights=(r.np.asarray(item['prediction_db'])[keep] - y) ** 2) for item in (a, b)]
    counts = r.np.bincount(ids)
    seed = int.from_bytes(hashlib.sha256(f'{m["bootstrap_seed"]}|{name}|{group}'.encode()).digest()[:8], 'big')
    draw = r.np.random.default_rng(seed).integers(0, len(unique), size=(m['bootstrap_replicates'], len(unique)))
    denominator = counts[draw].sum(axis=1)
    difference = r.np.sqrt(sums[0][draw].sum(axis=1) / denominator) - r.np.sqrt(sums[1][draw].sum(axis=1) / denominator)
    return {'rmse_improvement_db': a['metrics'][group]['rmse_db'] - b['metrics'][group]['rmse_db'],
            'percentile95_db': r.np.quantile(difference, [.025, .975]).tolist(), 'n_dates': len(unique),
            'replicates': m['bootstrap_replicates'], 'bootstrap_seed': seed, 'unit': 'weather-eligible date with all its hours'}


def table_sources(m):
    return {'all_state_weather_metrics.csv': metric_rows, 'all_feature_lists.csv': feature_rows,
            'all_rounds.csv': curve_rows, 'rfe_importance.csv': importance_rows,
            'rfe_survival_rank.csv': survival_rows, 'layer_pairs_all_weather.csv': layer_pair_rows,
            'layer_contributions_all_weather.csv': layer_effect_rows}


def generate(m):
    import verify_independent as q
    r.require(r.manifest()['fingerprint'] == m['fingerprint'], 'Frozen sources changed before reporting')
    qa = r.read(r.OUT / 'evaluation_qa.json')
    r.require(qa['status'] == 'PASS' and qa['fingerprint'] == m['fingerprint'], 'Final prediction QA gate')
    selection = r.read(r.OUT / 'selection.json')['selections']
    sources = table_sources(m)
    written = {}
    for filename, function in sources.items():
        first = next(function(m))
        written[filename] = r.csv_file(r.OUT / filename, list(first), function(m))
    expected = {'all_state_weather_metrics.csv': 17700, 'all_feature_lists.csv': 1180, 'all_rounds.csv': 250160,
                'rfe_importance.csv': 12245, 'rfe_survival_rank.csv': 156, 'layer_pairs_all_weather.csv': 25600,
                'layer_contributions_all_weather.csv': 50}
    r.require({filename: item['rows'] for filename, item in written.items()} == expected, 'Report table row counts')
    def test(kind, key):
        return r.read(r.evaluation_path(kind, key))['results']['test']
    base, full = test('layer', 0), test('layer', 1023)
    comparisons = [('layer_full_vs_base', base, full), ('layer_train_selected_vs_base', base, test('layer', selection['layer']['selected_key']))]
    for i, level in enumerate(r.LEVELS):
        comparisons.append((f'layer_full_add_p{level}', test('layer', 1023 ^ (1 << i)), full))
    comparisons.append(('rfe_train_selected_vs_full', test('rfe', 156), test('rfe', selection['rfe']['selected_key'])))
    intervals = {f'{name}/{weather}': bootstrap(a, b, weather, name, m) for name, a, b in comparisons for weather in r.WEATHER}
    r.atomic(r.OUT / 'date_intervals.json', {'fingerprint': m['fingerprint'], 'comparisons': intervals,
                                         'not_used_to_change_model_or_feature_selection': True, 'utc': r.now()})
    effects = list(layer_effect_rows(m))
    rank = list(survival_rows(m))
    highlights = {'layer_base53': base['metrics'], 'layer_full53': full['metrics'],
                  'layer_train_selected': test('layer', selection['layer']['selected_key'])['metrics'],
                  'rfe_full156': test('rfe', 156)['metrics'], 'rfe_train_selected': test('rfe', selection['rfe']['selected_key'])['metrics'],
                  'rfe_k10': test('rfe', 10)['metrics'], 'rfe_k1': test('rfe', 1)['metrics']}
    result = {'fingerprint': m['fingerprint'], 'seed': 17, 'epochs': 53, 'stage_status': 'AWAITING_DELIVERY_QA',
              'layer_states': 1024, 'rfe_states': 156, 'main_committed_fits': 4720, 'selection': selection,
              'highlights': highlights, 'layer_effects': effects, 'top10_survival': rank[:10],
              'full_features_names': m['retained_names'], 'test_not_used_to_select_k_or_layers': True,
              'historical_test_access': True, 'table_inventory': written, 'utc': r.now()}
    r.atomic(r.OUT / 'final_report.json', result)
    verified = q.verify_delivery(m, {name: fun(m) for name, fun in sources.items()}, comparisons)
    result['report_qa_sha256'] = r.sha(r.OUT / 'report_qa.json')
    lines = ['# Revision 029 当前53轮BiLSTM分层与RFE结果', '',
             '本轮所有正式指标都来自seed=17的单个模型。学习器/53轮/lr0.001/dropout0.2/weight_decay0.001/统一权重冻结。',
             '1024分层状态和156 RFE状态均逐状态重新训练，共4720次主路径拟合；三个训练内日期折不是三种子平均。',
             '每个正式模型用2404训练小时，测试评价仅加载模型；每状态完整保留53轮、输入名单、正式及折模型、逐小时预测。', '',
             '## 最终687小时测试指标', '', '|状态|天气|小时数|RMSE dB|MAE dB|Bias dB|', '|---|---|---:|---:|---:|---:|']
    for label, group_metrics in highlights.items():
        for weather in r.WEATHER:
            metric = group_metrics[weather]
            lines.append(f'|{label}|{r.LABELS[weather]}|{metric["n_hours"]}|{metric["rmse_db"]:.6f}|{metric["mae_db"]:.6f}|{metric["bias_db"]:+.6f}|')
    lines += ['', '基础53轮是本轮重新训练的同预算基线，不是Rev028基础46轮加权模型。BiLSTM沿同小时气压面而非跨小时序列。',
              f'训练内one-SE冻结分层mask={selection["layer"]["selected_key"]}，RFE k={selection["rfe"]["selected_key"]}；不按测试曲线重选。', '',
              '## 每层贡献：512个相同其他层上下文的平均ΔRMSE', '',
              'Δ=不含层RMSE−含层RMSE，正值表示加入层后误差较小；不预设正贡献，不解释为物理因果。', '',
              '|hPa|天气|均值 dB|中位数 dB|误差下降比例|全层上下文Δ dB|', '|---:|---|---:|---:|---:|---:|']
    for item in effects:
        lines.append(f'|{item["level_hpa"]}|{r.LABELS[item["weather"]]}|{item["mean_rmse_improvement_db"]:+.6f}|{item["median_rmse_improvement_db"]:+.6f}|{item["fraction_helpful"]:.1%}|{item["full_context_rmse_improvement_db"]:+.6f}|')
    lines += ['', '全部5120层配对、五天气指标、65个预定日期区间和完整156列生存排序保存在CSV/JSON，不将上下文数512当独立样本量。', '',
              '## RFE自然保留的前10列：按生存顺序', '', '|顺序|完整列名|类型|hPa|', '|---:|---|---|---:|']
    for item in rank[:10]:
        lines.append(f'|{item["survival_rank"]}|{item["feature_name"]}|{item["type"]}|{item["level_hpa"]}|')
    lines += ['', 'RFE每次删1列：打乱某列后的折外总体RMSE减原RMSE，三个训练内日期折平均后删除最小列。无分层配额。',
              '53轮来自Rev028全层验证阶段，本版固定同预算；各状态到第53轮结束，不声称第53轮是每个子集自己的最优早停。',
              '官方验证数据仅作最终描述性复核；历史测试已被查看，训练测试存在同日重叠，本轮不称新的盲测。',
              '训练/评价/交付独立QA均PASS；新实验没有修改原数据、Rev028或旧结果。', '']
    r.atomic_text(r.OUT / 'RESULTS.md', '\n'.join(lines))
    result['stage_status'] = 'COMPLETE'
    r.atomic(r.OUT / 'final_report.json', result)
    r.atomic(r.OUT / 'completion.json', {'status': 'COMPLETE', 'fingerprint': m['fingerprint'],
             'train_qa_sha256': r.sha(r.OUT / 'train_qa.json'), 'evaluation_qa_sha256': r.sha(r.OUT / 'evaluation_qa.json'),
             'report_qa_sha256': r.sha(r.OUT / 'report_qa.json'), 'final_report_sha256': r.sha(r.OUT / 'final_report.json'),
             'results_sha256': r.sha(r.OUT / 'RESULTS.md'), 'utc': r.now()})
    r.progress('全部实验、独立QA、报告完成', {'layer_states': 1024, 'rfe_states': 156, 'seed': 17, 'epochs': 53})
    return {'status': 'COMPLETE', 'layer_states': 1024, 'rfe_states': 156, 'report_qa': verified['status']}
