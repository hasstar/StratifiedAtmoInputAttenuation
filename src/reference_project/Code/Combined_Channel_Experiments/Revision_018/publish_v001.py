"""Publish only after all-hour independent QA. No model or source mutation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_018'
DATA = OUT / 'attempt_003'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def emit(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def write(name, txt):
    with (OUT / name).open('x', encoding='utf-8') as f:
        f.write(txt)


def main():
    qa = read(DATA / 'independent_qa.json')
    assert qa['status'] == 'PASS_EXTENDED_FROZEN_TEMPORAL_DIAGNOSTIC'
    assert qa['verified_hours'] == 3027 and qa['reused_hours'] == 932
    protected = read(DATA / 'sources_before.json')
    assert all(sha(p) == h for p, h in protected.items())
    complete = read(DATA / 'calculation_complete.json')
    assert sha(DATA / 'calculation_complete.json') == qa['source_completion_sha256']
    for p, h in complete['outputs_sha256'].items():
        assert sha(ROOT / p) == h, p
    assert (DATA / 'calculation.stderr.log').stat().st_size == 0
    assert (DATA / 'verification.stderr.log').stat().st_size == 0
    rows = read(DATA / 'summary.json')['rows']
    lags = read(DATA / 'lag_summary.json')
    ledger = pd.read_parquet(DATA / 'all_uk_hour_ledger.parquet')
    scores = pd.read_parquet(DATA / 'hourly_scores.parquet')
    lag_frame = pd.read_parquet(DATA / 'lag_diagnostics.parquet')
    assert len(rows) == 24 and len(lags) == 264 and len(ledger) == 5664 and len(scores) == 6054
    names = {'pooled': '英国两站合并', 'chilbolton': 'Chilbolton', 'sparsholt': 'Sparsholt',
             'rain': '雨天', 'dry_overcast': '干燥阴天'}
    def score_table(weighting):
        result = ['|范围|天气|小时/日期数|iid 变差评分|block_0256 变差评分|较低者|',
                  '|---|---|---:|---:|---:|---|']
        for site in ('pooled', 'chilbolton', 'sparsholt'):
            for event in ('rain', 'dry_overcast'):
                cell = {r['candidate']: r for r in rows if r['station'] == site and r['event'] == event and r['weighting'] == weighting}
                a, b = cell['iid'], cell['block_0256']
                assert a['n_station_hours'] == b['n_station_hours'] and a['n_dates'] == b['n_dates']
                winner = 'block_0256' if b['vs32'] < a['vs32'] else 'iid' if b['vs32'] > a['vs32'] else '相同'
                result.append(f"|{names[site]}|{names[event]}|{a['n_station_hours']}/{a['n_dates']}|{a['vs32']:.6f}|{b['vs32']:.6f}|{winner}|")
        return '\n'.join(result)
    date_table = score_table('date_equal'); hour_table = score_table('hour_equal')
    acf_table = ['|天气|滞后(s)|观测 ACF|iid ACF|block_0256 ACF|', '|---|---:|---:|---:|---:|']
    for event in ('rain', 'dry_overcast'):
        for lag_s in (1, 60, 300, 900):
            c = {r['candidate']: r for r in lags if r['station'] == 'pooled' and r['event'] == event
                 and r['weighting'] == 'date_equal' and r['lag_s'] == lag_s}
            def fmt(x):
                return '缺失' if x is None else f'{x:.6f}'
            acf_table.append(f"|{names[event]}|{lag_s}|{fmt(c['iid']['observed_acf'])}|{fmt(c['iid']['generated_acf'])}|{fmt(c['block_0256']['generated_acf'])}|")
    delta = max(abs(r['vs32'] - r['vs16']) for r in rows)
    missing_obs = int(lag_frame.observed_acf.isna().sum())
    missing_gen = int(lag_frame.generated_acf.isna().sum())
    neg = int(scores.vs32.lt(0).sum())
    content = f"""# 英国扩展日期：共同冻结快变的时间结构核验（Revision018）

## 本轮完成项

已将共享快变的时间结构诊断扩展到当前雨天/干燥阴天共同母集的 **3027 个英国站点小时**（雨 735、干燥阴天 2292）。包含全部已完成日期中符合既有质量和物理共同口径的小时，未退回原39天。

全范围台账仍是 Chilbolton 143 日、Sparsholt 93 日，共5664小时；非目标天气及不符合既有共同口径的小时逐项保留。本轮复用932个完全相同的旧动态小时；新范围2094小时，另有1小时随既有天气源更新而改变条件，共新生成2095小时、{complete['new_realizations']}条冻结随机实现。没有重算未改变的旧小时，也没有改动小时衰减标签或七类模型的小时排名。

这是 **同一冻结快变生成器的时间结构诊断**，不是新增论文模型。iid 是既定无时间依赖控制，block_0256 是此前已选定并冻结的256秒块方案；本轮没有再次选择块长或种子。所有小时慢变模型加上同一快变后，恒定慢变在增量和中心化相关中抵消，因此这里不能给七类慢变模型排名。

## 变差评分（dB²，越低越好；描述性比较）

32次固定重复，沿用Monte Carlo校正；负估计保留。先在每小时内按完整有效间隔形成滞后配对，再在预设11个滞后中有有效配对的时滞上计算既有评分；无有效配对的时滞不计入、不补零，每小时实际 n_lags 另存。日期等权为本次主汇总：两站合并时，先按同一UTC日期内的站点小时平均，再平均日期。

{date_table}

按小时等权的并列结果如下；两种权重不能混成一个排名。

{hour_table}

全部24个分层/权重/候选汇总中，16次与32次重复评分最大绝对差为 {delta:.6g} dB²。这只是既定重复数敏感性，不是误差条、显著性检验或选择最佳随机种子。

## 自相关诊断

以下为英国两站日期等权的预选代表滞后。所有11个滞后、增量、有效小时数和原始每小时记录均另存；下表不是按结果选出的最佳时滞。ACF 是对每个滞后同一有效配对的 Pearson 相关，不是跨缺失/跨小时拼接的序列相关。

{chr(10).join(acf_table)}

完整记录共{len(lag_frame)}行（两个候选分别记行），其中观测ACF缺失{missing_obs}行，生成平均ACF缺失{missing_gen}行；零方差或少于3个有效配对保持缺失。32次校正评分为负的候选小时数为{neg}，未裁零。

## 独立核验

独立实现逐一复算3027小时的原生观测、连续有效段、统计量与评分；核验5664小时全台账和236个站点日期。另按计算前固定规则重放{qa['independent_replay_hours']}个新增/条件变化小时，共{qa['independent_replay_realizations']}条实现。核验{qa['gap_safe_pairs']}个完整间隔配对，覆盖{qa['valid_native_seconds']}个有效原生秒；配对和秒相互依赖，不视为独立统计重复。

评分最大复算差 {qa['max_score_difference_db2']:.6g} dB²，重放增量最大差 {qa['max_replay_increment_difference_db']:.6g} dB。8项计算控制和3项独立控制通过，{qa['protected_sources_unchanged']}个保护文件校验未变。最终文件清单与进程/锁释放状态见 final_verification.json；打开报告本身不替代该终检。

统计核验技能 nature-statistics 要求明确嵌套单位、完整日期与缺失、两种汇总权重和描述性边界，已落实。这里的“独立”是另一个算术/生成实现，不是外部同行评审。没有将秒或配对数量用作独立样本量计算显著性。

## 原有结果和剩余项

- 小时慢变四类可计算结果及七类输入/来源状态仍以 [Revision012](../Combined_Channel_Revision_012/) 为准；共有快变边缘概率评分见 [Revision017](../Combined_Channel_Revision_017/REPORT.md)。LOCAL原有动态结果不重跑。
- 本版没有E-SST原公式、RMOP原辐射传输包或NWP–MTS原始事件库的新输入，不改变这三类的BLOCKED状态。没有把简单公式替代它们。
- 本版不包含完整MTS的事件分布、衰落持续时间或完整斜率分布对照，也不把随机合成序列视为真实UTC逐秒预测；因此**七类完整比较尚未全部完成**。
- 天气采集保持停止，STOP未改；无训练、重估归一化/晴空参考/CDF、重新选择种子或新的小时预测。时域统计不改变“本文是否最优”的现有小时评价结论。

## 文件索引

[当前协议](PROTOCOL.md)、[执行修正说明](attempt_003/PROTOCOL.md)、[全小时台账](attempt_003/all_uk_hour_ledger.parquet)、[逐日期覆盖](attempt_003/coverage_by_date.parquet)、[逐小时评分](attempt_003/hourly_scores.parquet)、[全部滞后诊断](attempt_003/lag_diagnostics.parquet)、[汇总](attempt_003/summary.json)、[独立QA](attempt_003/independent_qa.json)、[新旧分片索引](attempt_003/combined_shard_index.json)。

最初的沙盒分离进程启动中断和第二次校验错误已原样保留。第二次错误为新校验代码把功率域小时均值误写为dB均值，在任何新轨迹生成前被阻止；v002仅修正该断言并加强单位/UTC核验，没有改动原有数据、目标或模型。不得复用失败尝试作为评分成果。
"""
    write('REPORT.md', content)
    write('HANDOFF.md', """# Revision018 交付与续接

本版为英国全部已完成日期内雨/干燥阴天3027小时的共享冻结快变时间结构诊断。唯一有效计算目录 attempt_003；attempt_001/002仅保留启动/校验失败证据。报告及独立QA已生成，最终是否交付通过以 final_verification.json 为准。

禁止重复运行 temporal.py、run_attempt_002.py、run_attempt_003.py、verify_attempt_002.py、verify_attempt_003.py 或 publish_v001.py。结果已经独占创建；复用现有分片及SHA。小时慢变模型结果继续指向Revision012，边缘概率比较指向Revision017；未退回原39天。932旧小时直接复用，2095小时新生成，1个历史天气条件变化单独记录。

E-SST、RMOP、NWP–PRM–MTS仍待原文/实现/输入/合法事件库。既有用户询问尚待回复；无新材料不要反复检索同一受限入口、重新计算同一结果或扩充近似冒充原版。需要新气象字段时先列具体范围与必要性，等待人工授权。停止下载的授权和STOP继续有效。

这不是七类慢变模型动态排名，亦非完整MTS比较。尚缺完整MTS的事件统计、衰落持续时间及斜率分布，不得宣称七类全部完成。共同小时慢变的常量在本指标抵消；不为模型排名重新训练或调参。

同一 tvt 保持既有调度，不新建任务。所有约定交付仍未齐，不按“完成”暂停自动化。无安全独立工作且缺必需材料时，保留成果并等待人工材料/授权，而非反复计算。根README/T012/Revision001 PROGRESS未改，最新续接入口为本文件。

逻辑占用范围为本版输出及代码；计算与QA结束后不再占用其他实验目录，OS锁释放须由终检确认。历史PID不能作为今后存活证据。
""")
    emit('statistical_review.json', {'status': 'PASS_BOUNDED_DESCRIPTIVE_TEMPORAL_DIAGNOSTIC',
         'skill': 'nature-statistics', 'independent_units_not_seconds': True,
         'paired_candidates_same_hours_marginals': True, 'no_significance_test_or_CI': True,
         'all_dates_retained': True, 'date_equal_and_hour_equal_separate': True,
         'common_slow_constant_cancels': True, 'full_MTS_complete': False,
         'three_families_blocked': ['E-SST', 'RMOP', 'NWP-PRM-MTS']})
    emit('publication_qa.json', {'status': 'PASS_REPORT_FROM_INDEPENDENTLY_VERIFIED_SCORES',
         'score_rows': len(rows), 'lag_summary_rows': len(lags), 'ledger_hours': len(ledger),
         'selected_hours': 3027, 'hourly_candidate_rows': len(scores),
         'report_sha256': sha(OUT / 'REPORT.md'), 'qa_sha256': sha(DATA / 'independent_qa.json'),
         'publisher_sha256': sha(Path(__file__)), 'source_paths_preserved': True,
         'seven_families_complete': False, 'remaining_blockers_unchanged': True})
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'execution.lock']
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += list(Path(__file__).parent.glob('*.ps1'))
    emit('delivery_manifest.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'status': 'PUBLISHED_PENDING_PROCESS_AND_HASH_FINAL_CHECK',
         'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print(json.dumps({'status': 'PUBLISHED_PENDING_FINAL_CHECK', 'files': len(paths), 'hours': 3027}))


if __name__ == '__main__':
    main()
