"""Publish verified, existing scores; no ML, network, or metric fitting."""
import contextlib
import hashlib
import json
import msvcrt
import socket
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'
COHORTS = {'LOCAL_test': 'LOCAL 原测试集', 'UK_full_common': '英国两站合并',
           'UK_chilbolton_full_common': 'Chilbolton',
           'UK_sparsholt_full_common': 'Sparsholt'}
EVENTS = {'rain': '雨天小时', 'dry_overcast': '干燥阴天小时'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def write(name, text):
    path = OUT / name
    need(path.resolve().parent == OUT.resolve(), 'Output escapes version')
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)


def emit(name, value):
    write(name, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@contextlib.contextmanager
def locked():
    with (OUT / 'execution.lock').open('r+b') as f:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def ranges(values):
    days = sorted(set(values))
    need(bool(days), 'Empty dates')
    spans = []
    first = last = days[0]
    for d in days[1:]:
        if (date.fromisoformat(d) - date.fromisoformat(last)).days == 1:
            last = d
        else:
            spans.append(first if first == last else first + '—' + last)
            first = last = d
    spans.append(first if first == last else first + '—' + last)
    return '；'.join(spans)


def cell(row, metric):
    mean = row[metric + '_mean']
    if mean is None:
        return '未计算'
    formatted = f'{mean:+.6f}' if metric == 'bias_db' else f'{mean:.6f}'
    sd = row.get(metric + '_sd')
    return formatted if sd is None else formatted + f' ± {sd:.6f}'


def main():
    def offline(*args, **kwargs):
        raise RuntimeError('Publication is offline')
    socket.socket.connect = offline
    socket.create_connection = offline
    guard = read('sources_before.json')
    for name, expected in guard.items():
        need(sha(ROOT / name) == expected, 'Protected SHA changed: ' + name)
    qa = read('verification.json')
    need(qa['status'] == 'PASS' and not qa['all_seven_complete'], 'QA/status gate')
    need(qa['new_training'] == qa['new_inference'] == qa['weather_downloads'] == 0, 'Scope gate')
    calculation = read('calculation_complete.json')
    for name, expected in calculation['output_sha256'].items():
        need(sha(OUT / name) == expected, 'Calculation output changed: ' + name)
    for p in list(OUT.iterdir()) + list(CODE.iterdir()):
        if p.is_file() and p.suffix not in ('.lock', '.pyc'):
            guard[str(p)] = sha(p)
    with locked():
        emit('publication_sources.json', guard)
        rows = read('seven_family_rankings.json')
        ledger = read('all_hour_ledger.json')
        coverage = read('coverage_by_date.json')
        need(len(rows) == 56 and len(ledger) == 6351, 'Unexpected rows')
        report = [
            '# 七类模型对比：已完成部分与待解条件', '',
            '2026-10-17 · Revision012 · 冻结结果复用与独立实现重评分', '',
            '本轮完成四类已有模型的范围、掩码、数值与排名复核；E-SST、RMOP、NWP–传播模型链尚未产生合格预测，不能称七类比较完成。没有新增训练、重新推理或天气下载。', '',
            '## 结论', '',
            '在目前可评分的四类中：LOCAL 雨天 TabM 最低、LOCAL 干燥阴天本文模型最低；英国两站合并的雨天 TabM 最低、干燥阴天组合 ITU 近似最低。本文模型不是跨全部数据集和天气条件的最优模型。这里是描述性排名，不等于显著性检验结论。', '',
            'Bai LSTM、TabM 均为既有 LOCAL 适配版，不是作者原始预训练权重。全名单只保留本文、Bai LSTM、TabM、组合 ITU、E-SST、RMOP、NWP 链；Bai MLP 和此前筛选的通用模型不进入本轮主表。', '',
            '## 实际日期与样本范围', '',
            '| 数据集 | 气象/源小时 | 原标签合格小时 | 四类共同可评分小时 | 完整来源站点日/测试日期 |',
            '|---|---:|---:|---:|---:|'
        ]
        for station, label in [('LOCAL', 'LOCAL 原测试集'), ('chilbolton', 'Chilbolton'), ('sparsholt', 'Sparsholt')]:
            items = [x for x in ledger if x['station'] == station]
            ds = sorted({x['date'] for x in items})
            report.append(f'| {label} | {len(items)} | {sum(x["primary_score_eligible"] for x in items)} | {sum(x["common_score_eligible"] for x in items)} | {len(ds)} |')
        report += ['', 'LOCAL 是原划分的 687 个测试小时，分布在 2025-10-10—2026-08-20 的 156 个日期，并非连续 156 整天。英国是全部已完成的 5664 个气象站点小时，143/93 个完整站点日；不是退回原 39 天，也不是已下载整个原定 2003—2005 日历。', '']
        for station in ('chilbolton', 'sparsholt'):
            ds = [x['date'] for x in coverage if x['station'] == station]
            report.append(f'- {station} 完整日期：{ranges(ds)}（{len(ds)} 日）。')
        report += ['', '各站日期分别保留，不取交集。12 月已有的部分压力层记录不满足整天 Top10/必要气象齐全条件，不算完成日期。逐日全部天气/可评分计数见 [coverage_by_date.json](coverage_by_date.json)，逐小时剔除原因见 [all_hour_ledger.json](all_hour_ledger.json)。', '',
                   '| 站点 | 标签缺失 | 原生覆盖不足 95% | 标签合格但液态雨估计负差 |',
                   '|---|---:|---:|---:|']
        for station in ('chilbolton', 'sparsholt'):
            count = Counter(x['exclusion_reason'] for x in ledger if x['station'] == station)
            report.append(f'| {station} | {count["missing_native_target"]} | {count["native_coverage_below_95"]} | {count["negative_liquid_rain_estimate"]} |')
        report += ['', '因此英国原标签合格 5593 小时，共同物理掩码再排除 55 小时后为 5538。缺失不补零；全部 5664 小时都保留在台账。主表四类在每个数据集/天气组逐键使用同一掩码，不把不同样本的结果混排。', '',
                   '## 天气、指标与多次运行口径', '',
                   '“雨天/阴天”在此均指对应的小时类别，不是把整日统一贴标签。英国逐小时以总降水水当量 >=0.1 mm 且雪水当量 <0.1 mm 为 rain；二者均 <0.1 mm 且总云量 >=0.8 为 dry_overcast。全部完成日期先分类，再检查标签/物理输入；未挑选有利日期。LOCAL 沿用并核对冻结分类。英国小时 [t,t+1h) 的平均降水率使用 valid=t+1h。', '',
                   '目标为小时相对晴空衰减，单位 dB；Bias=预测−实测。RMSE、MAE 均越低越好。Bai/TabM 的 ± 为既有 17/29/43 三次运行指标的样本标准差（ddof=1），不是置信区间、最佳种子或新预测集成。单次模型不伪造标准差。N 是观测小时/站点小时，不声称相互独立；合并英国按小时加权，不是两站等权。名次按未舍入的 RMSE 均值，仅限已算四类。', '']
        for cohort, title in COHORTS.items():
            report += [f'## {title}', '']
            for event, event_title in EVENTS.items():
                group = [r for r in rows if r['cohort'] == cohort and r['event'] == event]
                scored = sorted((r for r in group if r['status'] == 'SCORED_FROZEN'), key=lambda r: r['rank'])
                blocked = [r for r in group if r['status'] == 'BLOCKED_NOT_COMPUTED']
                need(len(scored) == 4 and len(blocked) == 3, 'Roster changed')
                n = scored[0]['n']
                counts = '；'.join(f'{s} {len(ds)} 个实际有合格样本的日期' for s, ds in scored[0]['dates_by_station'].items())
                report += [f'### {event_title}', '', f'共同目标 N={n}；{counts}。这不是来源总天数；没有该类天气的来源日期仍在完整台账中。', '',
                           '| 名次 | 模型 | N（已评分/目标） | RMSE (dB) | MAE (dB) | Bias (dB) |',
                           '|---:|---|---:|---:|---:|---:|']
                for r in scored + blocked:
                    rank = str(r['rank']) if r['rank'] is not None else '—'
                    report.append(f'| {rank} | {r["name"]} | {r["n"]}/{n} | {cell(r,"rmse_db")} | {cell(r,"mae_db")} | {cell(r,"bias_db")} |')
                report += ['']
        report += ['## 三类待算模型与动态比较', '',
                   'E-SST 需完整公式/雨型规则核验；英国尚缺零度层及 700 hPa 风等输入。RMOP 需可用的 GSDSU/论文配置和兼容三维场；NASA 的软件入口涉及使用协议。NWP 的 PRM 缺论文专用散射系数及英国廓线，MTS 另缺原始秒级事件库。详见 [七类来源与输入矩阵](MODEL_INPUT_MATRIX.md) 和 [来源检索记录](SOURCE_LOOKUP.md)。缺项均为未计算，不将 0 当性能、不静默缩小母集。', '',
                   '动态比较目前 BLOCKED_NOT_COMPUTED，见 [dynamic_comparison_status.json](dynamic_comparison_status.json)。MTS 不与实测轨迹作对应时刻的逐秒 RMSE，后续需单列分布/ACF/衰落持续时间/斜率；本轮没有创建随机替代结果或叠加新快变。', '',
                   '## 复核与文件', '',
                   f'独立 scalar math.fsum 实现复算 64 组运行指标、32 行已评分汇总及 24 行待算记录；最大指标误差 {qa["max_metric_absolute_error"]:.3g}。14 个离线测试通过。逐键预测身份、天气/UTC/掩码、日期、599 个保护来源 SHA 均通过。独立实现由同一代理编写，不冒称外部同行审查。', '',
                   '[verification.json](verification.json) 是计算 QA；最终发布文件与锁检查见 [final_verification.json](final_verification.json)。历史 calculation_complete.json 的 COMPUTED_PENDING_QA 是计算结束时的不可覆盖快照，当前 QA 状态以 verification/final_verification 为准。', '',
                   '完整数值：[seven_family_rankings.json](seven_family_rankings.json)；逐次运行：[run_metrics.json](run_metrics.json)；同样本键：[sample_membership.json](sample_membership.json)；来源保护：[sources_before.json](sources_before.json)；下一步：[HANDOFF.md](HANDOFF.md)。', '',
                   '本轮复用完整结果，统计报告技能用于区分小时样本数与独立重复、明确种子标准差和同样本排名；文献检索技能用于把公式、输入与许可缺口逐项登记。没有通过改权重、重估晴空参考或换评价日期改善排名。', '']
        write('REPORT.md', '\n'.join(report))
        for name, expected in guard.items():
            need(sha(ROOT / name) == expected, 'Publication changed protected source: ' + name)
    released = datetime.now(timezone.utc).isoformat()
    write('HANDOFF.md', f'''# Revision012 续接记录

阶段：四类既有结果已评分并经独立实现 QA；七类整体未完成。公开文献核验和输入准备仍有可推进部分，不修改同一 tvt 的周期/状态，不新增任务。天气 STOP 持续有效。

## 本轮真正完成

- 复核 Revision011 组合 ITU 和冻结学习模型，按最新七类名单交付；Bai MLP 等已从本版主表删除，历史结果未删除。
- LOCAL 687 小时；英国 Chilbolton/Sparsholt 143/93 个完整站点日、5664 气象小时，5593 标签合格/5538 共同可评分。全日期与逐小时缺失台账保留。
- 64 组运行指标、32 行四类汇总和 24 行三类 BLOCKED 行；14 测试，scalar math.fsum 独立实现误差上限 {qa['max_metric_absolute_error']:.17g}。
- 核验已有 LOCAL 原始 ERA5 小网格线索、UK E-SST 缺列、RMOP 与 PRM 不同水凝物需求、GSDSU 软件使用条件。代码/输入可得性不冒充完整实现。
- 新训练=0，新模型推理=0，天气下载=0。未载入模型权重执行；已有 checkpoint 仅校验 SHA。

## 命令与环境

项目根 cwd；解释器 `C:/Users/SDR_HH/.conda/envs/pytorchmy/python.exe`；所有 Python 命令带 `-B`。已成功运行 `Code/Combined_Channel_Experiments/Revision_012/tests.py`、`audit.py`、`verify.py`；发布执行 `publish.py`，随后只运行 `verify_delivery.py`。不要重跑已有一次性输出入口。具体版本/启动记录在 launch.json、test_results.json、verification.json；每次运行都是前台有限任务，不是长驻后台下载器。

## 锁、指纹与并发

Execution Lock: RELEASED。publish.py 的 OS 字节锁已在 {released} 释放，文件 execution.lock 保留。最终交付验证会再次尝试获取/释放同一锁，结果写入 final_verification.json；不是看锁文件存在就声称进程存活。12:24:49/52 (+08) 的两次 elevated CIM 成功且 Python 列表为空，见 publication_preflight.json；早先 sandbox CIM 失败不作为进程不存在证据。

初始 sources_before.json 保护 599 项；publication_sources.json 另覆盖发布脚本/文献矩阵/既有输出，delivery_manifest.json 记录最终交付 SHA。旧保护集中的 tvt 配置变化是本次用户授权的调度更新，已有 Automation_Changes/TVT_Comparison_20261017_001 记录；本轮未改调度。STOP_REQUESTED SHA 必须仍为 `0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a`。

## 后续安全工作顺序

1. 从合法公开或已登录入口核对 E-SST 完整式(8)、事件边界及雨型/雨高定义。小时步长必须 3600 s；不猜 10 mm/h 阈值，不用插值声称分钟观测。公式确定后可对已有连续 LOCAL 气象做无标签适配 QA；不能串接随机测试小时。
2. 对已有 LOCAL 原始 ERA5 温湿压/云雨水文件执行逐场单位、UTC、层、网格/有限值 QA。当前只确认清单与一个温度头；processed LOS 列不是共同垂直廓线。新实现或输出使用下一个未占用版本，不修补本版冻结文件。
3. 找 PRM 论文散射数据库/系数的公开出处，核对本工程频率极化；缺系数不可将 ITU 公式冠名 PRM。RMOP 与 PRM 输入要求不同，见 MODEL_INPUT_MATRIX.md。
4. GSDSU 软件协议/非公开仓库、作者联系、原始 Italsat 事件库需要用户方向，不擅自申请。完整 MTS 只做独立动态统计，不做逐秒同刻 RMSE、不叠加本工程快变。
5. 英国若确需新气象采集，先写清字段身份、全部日期、逻辑量/精确请求量、时刻及必要性，再等待新授权。候选 28320 站点—字段—小时不是授权、不是请求数。保持全部旧下载器停止，绝不删除 STOP。

当前三类待算状态不等于整个任务完成。四类结果见 REPORT.md；不调整权重/种子/偏移让本文排名第一。最终交付 QA 通过后保留本版只读，继续同一 tvt 的未完成核验工作。
''')
    manifest = {}
    for p in list(OUT.iterdir()) + list(CODE.iterdir()):
        if p.is_file() and p.suffix not in ('.lock', '.pyc'):
            manifest[str(p.relative_to(ROOT))] = sha(p)
    emit('delivery_manifest.json', manifest)
    print(json.dumps({'published': 'REPORT.md', 'rank_rows': len(rows),
                      'all_seven_complete': False, 'os_lock_released_utc': released}))


if __name__ == '__main__':
    main()
