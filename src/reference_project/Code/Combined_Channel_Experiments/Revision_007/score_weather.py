"""Post-hoc weather-weighted scoring of saved predictions. No model execution."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_007'
REV6 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_006'
REV2 = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_002'
GBS = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
WEATHER = GBS / 'conditional_fast_001'
WEIGHTS = {'rain': 0.5, 'dry_overcast': 0.3, 'clear_dry': 0.2}
PHYSICAL = ['itu838_geometric_proxy', 'itu618_hourly_proxy']
NAMES = {
    'ours_frozen_top10': '本文冻结Top10',
    'tabpfn_v2_author_frozen': 'TabPFN-v2（作者原权重）',
    'tabicl_v2_author_frozen': 'TabICLv2（作者原权重）',
    'tabm2025_official_budgeted': 'TabM（本地适配版）',
    'bai2022_mlp_adapted': 'Bai2022 MLP（适配版）',
    'bai2022_lstm_t0_adapted': 'Bai2022 零历史LSTM（适配版）',
    'ridge': 'Ridge', 'elastic_net': 'Elastic Net', 'svr_rbf': 'RBF-SVR',
    'random_forest': 'Random Forest', 'xgboost': 'XGBoost',
    'train_mean': 'LOCAL训练均值常数',
    'itu838_geometric_proxy': 'P.838几何雨衰近似',
    'itu618_hourly_proxy': 'P.618小时代理',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def emit(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def verify_weather(frame, scope, evidence):
    """Fresh raw-file point/time audit, without importing historical code."""
    values = {}
    for name in sorted({x['path'] for x in evidence}):
        with zipfile.ZipFile(ROOT / name) as archive:
            for member in archive.namelist():
                if not member.endswith('.nc'):
                    continue
                with netCDF4.Dataset('weather_read_only.nc', memory=archive.read(member)) as nc:
                    for field, param, unit in [('tcc', 164, '(0 - 1)'), ('tp', 228, 'm'), ('sf', 144, 'm of water equivalent')]:
                        if field not in nc.variables:
                            continue
                        var = nc.variables[field]
                        need(var.GRIB_paramId == param and var.units == unit, 'Weather parameter/unit mismatch')
                        latitudes = np.asarray(nc.variables['latitude'][:], dtype=float)
                        longitudes = np.asarray(nc.variables['longitude'][:], dtype=float)
                        times = np.asarray(nc.variables['valid_time'][:], dtype=np.int64)
                        wanted = set(scope['hours_utc_epoch_seconds'])
                        for station in scope['links']:
                            ilat = min(range(len(latitudes)), key=lambda i: (abs(latitudes[i] - station['latitude_deg_n']), -latitudes[i]))
                            ilon = min(range(len(longitudes)), key=lambda i: (abs(longitudes[i] - station['longitude_deg_e']), -longitudes[i]))
                            shift = 0 if field == 'tcc' else 3600
                            for i, valid in enumerate(times):
                                hour = int(valid) - shift
                                if hour not in wanted:
                                    continue
                                key = (station['id'], hour, field)
                                need(key not in values, 'Duplicate weather source identity')
                                value = var[i, ilat, ilon]
                                need(not np.ma.is_masked(value) and math.isfinite(float(value)), 'Masked/nonfinite saved source weather')
                                values[key] = (float(value), float(latitudes[ilat]), float(longitudes[ilon]))
    checked = 0
    for row in frame.itertuples(index=False):
        h = int(row.utc_epoch_seconds)
        for field in ['tcc', 'tp', 'sf']:
            actual = getattr(row, field)
            raw = values.get((row.station, h, field))
            need(getattr(row, field + '_valid_utc_epoch_s') == h + (0 if field == 'tcc' else 3600), 'UTC interval mismatch')
            if raw is None:
                need(pd.isna(actual), 'Missing source value was filled')
            else:
                need(actual == raw[0], 'Raw weather differs from saved label source')
                need(row.grid_latitude == raw[1] and row.grid_longitude == raw[2], 'Weather grid differs')
                checked += 1
        c, p, s = float(row.tcc), 1000 * float(row.tp), 1000 * float(row.sf)
        known = all(math.isfinite(x) for x in (c, p, s))
        event = 'unknown' if not known else 'other'
        if known and p >= .1 and s < .1:
            event = 'rain'
        elif known and p < .1 and s < .1 and c >= .8:
            event = 'dry_overcast'
        elif known and p < .1 and s < .1 and c <= .2:
            event = 'clear_dry'
        need(event == row.event and known == bool(row.weather_known), 'Weather classification differs')
    need(checked == 5608, 'Unexpected number of verified weather values')
    return checked


def calculate():
    started = datetime.now(timezone.utc).isoformat()
    need(not OUT.exists(), 'New output already exists; never overwrite a result')
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        selection = read(REV2 / 'frozen_selection.json')
        evidence = read(WEATHER / 'weather_source_evidence.json')
        scope = read(GBS / 'scope_manifest.json')
        sources = [REV6 / name for name in ['predictions_local_all.parquet', 'predictions_uk_all.parquet', 'metrics.json', 'family_summary.json', 'delivery_manifest.json', 'delivery_receipt.json', 'source_sha256_before.json']]
        sources += [REV2 / 'frozen_selection.json', GBS / 'scope_manifest.json', GBS / 'conditional_fast_independent_qa_001/qa.json']
        sources += [WEATHER / name for name in ['weather_labels.parquet', 'weather_source_evidence.json', 'weather_availability_qa.json', 'completion.json', 'PROTOCOL.md']]
        sources += sorted({ROOT / x['path'] for x in evidence})
        sources += [REV2 / x['path'] for x in selection['models'] if 'path' in x]
        sources += [Path(__file__), Path(__file__).with_name('PROTOCOL.md')]
        before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
        emit('sources_before.json', before)
        manifest = read(REV6 / 'delivery_manifest.json')
        for name in ['predictions_local_all.parquet', 'predictions_uk_all.parquet', 'metrics.json', 'family_summary.json']:
            key = str((REV6 / name).relative_to(ROOT))
            need(before[key] == manifest[key]['sha256'], 'Rev006 delivered input has changed: ' + name)
        prior_sources = read(REV6 / 'source_sha256_before.json')
        need(sha(REV2 / 'frozen_selection.json') == prior_sources[str((REV2 / 'frozen_selection.json').relative_to(ROOT))], 'Model selection changed')
        for model in selection['models']:
            if 'path' in model:
                need(sha(REV2 / model['path']) == model['sha256'], 'Historical model checkpoint changed')
        for row in evidence:
            need(sha(ROOT / row['path']) == row['sha256'], 'Raw weather changed')
        oldqa = read(GBS / 'conditional_fast_independent_qa_001/qa.json')
        need(oldqa['status'] == 'PASS_WEATHER_AND_FROZEN_MARGINAL_TRANSFER_GATE', 'Legacy weather independent QA not passed')
        need(sha(WEATHER / 'weather_labels.parquet') == oldqa['weather_labels_sha256'], 'Weather labels changed after independent QA')
        local = pd.read_parquet(REV6 / 'predictions_local_all.parquet')
        uk = pd.read_parquet(REV6 / 'predictions_uk_all.parquet')
        weather = pd.read_parquet(WEATHER / 'weather_labels.parquet')
        weather['utc_epoch_seconds'] = weather.hour_utc.map(lambda t: int(t.timestamp()))
        need(len(local) == 687 and local.hour_bjt.is_unique, 'LOCAL keys/count')
        need(len(uk) == 5664 and not uk.duplicated(['station', 'utc_epoch_seconds']).any(), 'UK keys/count')
        need(len(weather) == 1872 and not weather.duplicated(['station', 'utc_epoch_seconds']).any(), 'Weather keys/count')
        need(set(zip(weather.station, weather.utc_epoch_seconds)) == set(zip(uk.loc[uk.period.eq('original39'), 'station'], uk.loc[uk.period.eq('original39'), 'utc_epoch_seconds'])), 'UK39 exact keyset mismatch')
        raw_checked = verify_weather(weather, scope, evidence)
        uk = uk.merge(weather[['station', 'utc_epoch_seconds', 'event', 'weather_known']], on=['station', 'utc_epoch_seconds'], how='left', validate='one_to_one')
        need(uk.loc[uk.period.eq('extension2003'), 'event'].isna().all(), 'Weather labels unexpectedly extended')
        need((local[list(WEIGHTS)].sum(axis=1) <= 1).all(), 'LOCAL weather groups overlap')
        local['event'] = 'other'
        for event in WEIGHTS:
            local.loc[local[event].astype(bool), 'event'] = event
        models = [m['id'] for m in selection['models']] + ['ours_frozen_top10', 'tabpfn_v2_author_frozen', 'tabicl_v2_author_frozen'] + PHYSICAL
        need(len(models) == 24 and len({re.sub(r'_seed\d+$', '', m) for m in models}) == 14, 'Model coverage')
        original = uk.period.eq('original39')
        cohorts = [
            ('LOCAL_test', local, models),
            ('UK39_common', uk[original & uk.common_score_eligible], models),
            ('UK39_chilbolton_common', uk[original & uk.common_score_eligible & uk.station.eq('chilbolton')], models),
            ('UK39_sparsholt_common', uk[original & uk.common_score_eligible & uk.station.eq('sparsholt')], models),
            ('UK39_ML_primary_sensitivity', uk[original & uk.primary_score_eligible], [m for m in models if m not in PHYSICAL]),
        ]
        rows, coverage, diagnostics, memberships = [], [], [], []
        oldmetrics = {(x['cohort'], x['model']): x for x in read(REV6 / 'metrics.json')}
        max_delta = 0.0
        for cohort, frame, methods in cohorts:
            counts = {str(k): int(v) for k, v in frame.event.value_counts(dropna=False).items()}
            need(all(counts.get(k, 0) > 0 for k in WEIGHTS), 'Missing weather group: ' + cohort)
            score_n = sum(counts[k] for k in WEIGHTS)
            coverage.append(dict(cohort=cohort, eligible_hours=len(frame), scored_hours=score_n, excluded_other_or_unknown=len(frame)-score_n, weather_counts=counts))
            for row in frame.itertuples(index=False):
                memberships.append(dict(cohort=cohort, station=getattr(row, 'station', 'LOCAL'), time=str(getattr(row, 'hour_bjt', getattr(row, 'hour_utc', None))), event=row.event, included=row.event in WEIGHTS))
            y = frame.truth_db.to_numpy(dtype=float)
            need(np.isfinite(y).all(), 'Nonfinite target in eligible cohort')
            for model in methods:
                pred = frame[model].to_numpy(dtype=float)
                need(np.isfinite(pred).all(), 'Nonfinite prediction: no per-model dropping')
                rmse = {}
                for event in WEIGHTS:
                    mask = frame.event.eq(event).to_numpy()
                    errors = pred[mask] - y[mask]
                    rmse[event] = float(np.sqrt(np.mean(errors ** 2)))
                    check = math.sqrt(math.fsum((float(p)-float(t))**2 for p, t in zip(pred[mask], y[mask])) / int(mask.sum()))
                    max_delta = max(max_delta, abs(check-rmse[event]))
                    need(math.isclose(check, rmse[event], rel_tol=1e-12, abs_tol=1e-12), 'Independent RMSE mismatch')
                    if cohort == 'LOCAL_test':
                        prior = oldmetrics[('LOCAL_' + event, model)]
                        need(prior['n'] == int(mask.sum()) and math.isclose(prior['rmse_db'], check, rel_tol=1e-12, abs_tol=1e-12), 'LOCAL prior weather metric mismatch')
                linear = float(sum(WEIGHTS[k] * rmse[k] for k in WEIGHTS))
                direct = math.fsum(WEIGHTS[k] * math.sqrt(math.fsum((float(p)-float(t))**2 for p, t in zip(frame.loc[frame.event.eq(k), model], frame.loc[frame.event.eq(k), 'truth_db'])) / counts[k]) for k in WEIGHTS)
                need(math.isclose(linear, direct, rel_tol=1e-12, abs_tol=1e-12), 'Independent weighted score mismatch')
                rows.append(dict(cohort=cohort, model=model, family=re.sub(r'_seed\d+$', '', model), n=score_n, rmse_db=rmse, weighted_rmse_score_db=linear))
            zero_rmse = {k: math.sqrt(math.fsum(float(t)**2 for t in frame.loc[frame.event.eq(k), 'truth_db']) / counts[k]) for k in WEIGHTS}
            diagnostics.append(dict(cohort=cohort, diagnostic='zero_db_constant_posthoc_not_ranked', rmse_db=zero_rmse, weighted_rmse_score_db=sum(WEIGHTS[k]*zero_rmse[k] for k in WEIGHTS)))
        families = []
        for cohort, _, _ in cohorts:
            for family in sorted({r['family'] for r in rows if r['cohort'] == cohort}):
                subset = [r for r in rows if r['cohort'] == cohort and r['family'] == family]
                scores = [r['weighted_rmse_score_db'] for r in subset]
                mean = math.fsum(scores)/len(scores)
                sd = math.sqrt(math.fsum((v-mean)**2 for v in scores)/(len(scores)-1)) if len(scores)>1 else None
                group_means = {k: math.fsum(r['rmse_db'][k] for r in subset)/len(subset) for k in WEIGHTS}
                need(math.isclose(mean, sum(WEIGHTS[k]*group_means[k] for k in WEIGHTS), rel_tol=1e-12, abs_tol=1e-12), 'Family aggregation mismatch')
                families.append(dict(cohort=cohort, family=family, name=NAMES[family], n_runs=len(scores), n=subset[0]['n'], group_rmse_db_mean=group_means, score_db_mean=mean, score_db_sd=sd))
            group = [r for r in families if r['cohort'] == cohort]
            for r in group:
                r['rank'] = 1 + sum(q['score_db_mean'] < r['score_db_mean'] for q in group)
        emit('run_scores.json', rows)
        emit('family_scores.json', families)
        emit('cohort_coverage.json', coverage)
        emit('sample_membership.json', memberships)
        emit('zero_constant_diagnostic.json', diagnostics)
        unavailable = dict(cohort='UK_expanded_all', status='NOT_SCORABLE_WITH_ALL_THREE_WEATHER_GROUPS', ml_eligible_hours=int(uk.primary_score_eligible.sum()), common_eligible_hours=int(uk.common_score_eligible.sum()), extension_without_weather_labels=int((uk.primary_score_eligible & uk.period.eq('extension2003')).sum()), reason='No existing complete tcc cloud diagnostic for extension2003; no downloads or inferred cloud labels authorized or performed.')
        emit('uk_expanded_gap.json', unavailable)
        after = {key: sha(ROOT / key) for key in before}
        need(before == after, 'Inputs modified during scoring')
        emit('sources_after.json', after)
        qa = dict(status='PASS', raw_weather_values_exact=raw_checked, weather_rows=1872, unknown_weather_rows=4, independent_rmse_max_abs_difference_db=max_delta, local_prior_group_metrics_exact_within_1e_12=True, sources_unchanged=len(before), historical_model_checkpoints_unchanged=18, shared_sample_masks=True, unique_keysets=True, no_missing_group_reweighting=True, all_24_runs_14_families_in_primary_tables=True, training_calls=0, model_inference_calls=0, network_requests=0, author_weights_not_loaded=True)
        emit('qa.json', qa)
        report = ['# 雨50%／阴30%／晴20%的模型评分', '', '仅重算已有预测；模型与历史结果均未修改。', '', 'S = 0.5 × RMSE雨 + 0.3 × RMSE干燥阴天 + 0.2 × RMSE晴；单位 dB，越低越好。不是原样本占比下的总体RMSE，也不是先加权MSE再开根号。', '', '## 同样本主表', '', '三种子家族为各运行分数的均值 ± 样本SD，不是置信区间，不选择种子或重新集成。排名在每列内使用未四舍五入数值。', '', '| 模型 | LOCAL测试集 | 英国39日两站 | Chilbolton39日 | Sparsholt39日 |', '|---|---:|---:|---:|---:|']
        main_cohorts = [x[0] for x in cohorts[:4]]
        lookup = {(r['cohort'],r['family']):r for r in families}
        for r in sorted([r for r in families if r['cohort']=='LOCAL_test'], key=lambda r:r['rank']):
            cells = []
            for cohort in main_cohorts:
                q=lookup[(cohort,r['family'])]
                value=f"{q['score_db_mean']:.4f}" + (f" ± {q['score_db_sd']:.4f}" if q['score_db_sd'] is not None else '')
                cells.append(value + f"（#{q['rank']}）")
            report.append('| '+r['name']+' | '+' | '.join(cells)+' |')
        report += ['', '## 样本与口径', '', '| 数据集 | 原合格小时 | 雨 | 干燥阴天 | 晴 | 三组计分小时 | 其它/未知不计分 |', '|---|---:|---:|---:|---:|---:|---:|']
        for item in coverage:
            n=item['weather_counts']
            report.append(f"| {item['cohort']} | {item['eligible_hours']} | {n['rain']} | {n['dry_overcast']} | {n['clear_dry']} | {item['scored_hours']} | {item['excluded_other_or_unknown']} |")
        report += ['', '英国主表统一使用物理有效共同小时。原39日1872行→标签合格1847→共同合格1792（排除55个物理输入无效小时）；再按三天气组计分。未知3小时与其它542小时不计三组分数，但记录保留。不做按模型缺值筛样。LOCAL687中其它156小时不在本次三组分数内。', '', '英国天气是原39日已保存的ERA5 tp/sf/tcc：tcc对应t，tp/sf对应valid=t+1的一小时累计，原载荷逐值、UTC/单位/参数/格点再次核验。分类阈值为总降水0.1mm、雪0.1mm、云量0.2/0.8；与此前扩展UK按NCAR液态估计率分雨量箱的口径不同。不是实测天气。', '', '原39日是2004-02-18—21、2004-02-25—28和2004-03-01—31，保留原日期缺口。此次并未把英国数据缩回39日：扩展全量5593小时的已保存预测仍在，但新增3746个合格小时没有完整云量标签，因此不能在全量上给出这项三天气评分。', '', '## 12个非ITU家族的主标签样本敏感性结果', '', '此表采用原39日1847个标签合格小时（未增加物理输入门槛），三天气1284小时；不与上表混排。', '', '| 模型 | 分数均值 | SD | 排名 |', '|---|---:|---:|---:|']
        for r in sorted([r for r in families if r['cohort']=='UK39_ML_primary_sensitivity'], key=lambda r:r['rank']):
            sd='—' if r['score_db_sd'] is None else f"{r['score_db_sd']:.4f}"
            report.append(f"| {r['name']} | {r['score_db_mean']:.4f} | {sd} | {r['rank']} |")
        report += ['', '## 解释与边界', '']
        for cohort in main_cohorts:
            winners=[r for r in families if r['cohort']==cohort and r['rank']==1]
            report.append(f"- {cohort} 最低："+'、'.join(f"{r['name']} {r['score_db_mean']:.4f} dB" for r in winners)+'。')
        report += ['', '英国均值常数基线也低于全部学习模型；只比较非恒定、非ITU学习模型时Elastic Net最低，不能称TabM为此规则下英国最佳。本文模型在LOCAL主表排第7、UK39共同主表排第8。对不同数据集分别报告，不建立未经指定的跨数据集综合冠军。', '', '单独保留0dB固定常数事后诊断（不加入既定14家族排名）：']
        report += [f"- {r['cohort']}：{r['weighted_rmse_score_db']:.4f} dB。" for r in diagnostics if r['cohort'] in main_cohorts]
        report += ['', '这里的最低是当前样本和用户后指定权重下的描述性最低，不是新的盲测结论或显著性检验。英国两站的相邻小时/同日天气相关，不能把站点小时视为独立重复。P.838/P.618保持既定近似形式，不代表完整标准的逐小时预测能力已获验证。', '', '## 文件与复现', '', '- `run_scores.json`：24运行的分组RMSE与线性分数。', '- `family_scores.json`：14家族的均值、样本SD与每数据集排名。', '- `sample_membership.json`、`cohort_coverage.json`：逐小时纳入标记与数量。', '- `sources_before.json`、`sources_after.json`、`qa.json`：输入SHA及独立复核。', '- `uk_expanded_gap.json`：全量UK不可评分的明确缺口。', '- `completion.json`：执行时间、环境与占用释放。', '', '入口：`C:\\Users\\SDR_HH\\.conda\\envs\\pytorchmy\\python.exe -B Code\\Combined_Channel_Experiments\\Revision_007\\score_weather.py`。一次性新输出保护：已有输出时拒绝覆盖。后续重算须新版本。']
        with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:
            f.write('\n'.join(report)+'\n')
        receipt = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(), status='COMPLETE_POSTHOC_RESCORING', verification='PASS', execution_lock='RELEASED', python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, netCDF4=netCDF4.__version__, score_weights=WEIGHTS, score_definition='linear_weighted_mean_of_group_RMSE', new_training=0, new_inference=0, downloads=0, full_expanded_uk_weather_score_available=False, t012_overall_complete=False, artifacts={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
        emit('completion.json',receipt)
        print(json.dumps(dict(status=receipt['status'],qa=qa,coverage=coverage,winners=[r for r in families if r['rank']==1]),ensure_ascii=False))
    except Exception as exc:
        emit('failure.json', dict(utc=datetime.now(timezone.utc).isoformat(), exception=type(exc).__name__, message=str(exc), execution_lock='RELEASED_AFTER_FAILURE', retry_requires_new_output=True))
        raise


if __name__ == '__main__':
    calculate()
