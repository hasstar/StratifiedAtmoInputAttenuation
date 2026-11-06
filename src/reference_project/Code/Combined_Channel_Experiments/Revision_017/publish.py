"""Seal already independently verified scores; no recalculation or model loading."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import json,hashlib,msvcrt
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def md(n,s):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def now():return datetime.now(timezone.utc).isoformat()
def main():
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            q=read(OUT/'independent_qa.json');assert q['status']=='PASS'
            assert all(sha(OUT/p)==h for p,h in q['verified_outputs'].items())
            protected=read(OUT/'sources_before.json');assert all(sha(p)==h for p,h in protected.items())
            ledger=read(OUT/'all_hour_ledger.json');scores=read(OUT/'family_metrics.json');native=read(OUT/'native_hour_audit.json')
            rows=[];hours=defaultdict(list)
            for r in ledger:hours[(r['station'],r['date'])].append(r)
            seconds={(r['station'],r['time']):r['n_seconds'] for r in native}
            for (site,day),rs in sorted(hours.items()):
                rows.append({'station':site,'date':day,'all_completed_hours':len(rs),'label_valid_hours':sum(r['primary_score_eligible'] for r in rs),
                 'physical_common_hours':sum(r['common_score_eligible'] for r in rs),'hybrid_hours':sum(r['hybrid_score_eligible'] for r in rs),
                 'rain_hours':sum(r['hybrid_score_eligible'] and r['event']=='rain' for r in rs),'overcast_hours':sum(r['hybrid_score_eligible'] and r['event']=='dry_overcast' for r in rs),
                 'hybrid_valid_seconds':sum(seconds.get((r['station'],r['time']),0) for r in rs),'exclusions':dict(Counter(r['hybrid_exclusion_reason'] for r in rs if not r['hybrid_score_eligible']))})
            emit('coverage_by_date.json',rows)
            emit('statistical_review.json',{'status':'PASS_BOUNDED_DESCRIPTIVE_COMPARISON','skill':'nature-statistics',
             'influence':'Explicit native-second weights, nested hour/date/site units, fixed date/hour aggregation, multi-run SD not CI, no independent-seconds significance claim.',
             'independent_climate_replication':'Not established; times/sites dependent. Descriptive ranks only.',
             'scope':'Current four families, rain and dry-overcast only, entire completed-date ledger; three author-chain implementations remain blocked.',
             'prediction_intervals_not_confidence_intervals':True,'test_or_UK_label_fitting':False,'new_significance_tests':False,
             'issues':['Fast marginal scores do not evaluate temporal ordering; no MTS/ACF/full-chain completion.',
                       'LOCAL record-count-weighted seconds and UK native equal-weight seconds are distinct instruments.',
                       'Coverage below nominal must not be called calibrated or universally reliable.',
                       'Date-equal and hour-equal results are separately labeled; hourly RMSE ranking is a different endpoint.']})
            names={'ours':'本文Top10＋冻结快变','bai_lstm':'Bai LSTM本地适配＋冻结快变','tabm':'TabM本地适配＋冻结快变','itu':'组合ITU近似＋冻结快变'}
            events={'rain':'雨','dry_overcast':'干燥阴天'}
            def fmt(r,key):
                s=f"{r[key]:.6f}"
                if r.get(key+'_sd') is not None:s+=f" ± {r[key+'_sd']:.6f}"
                return s
            text=['# 四类模型共同冻结快变：雨/阴秒级概率补充比较\n',
             '本轮新增评价已通过独立实现复核。它复用四类已验证小时预测，叠加同一既有天气条件快变CDF；不是重训，也不是作者原版秒级模型。七类主名单仍有三类受材料/输入阻塞，不能说全部完成。\n',
             '小时级 RMSE、MAE、Bias 及七类输入/来源矩阵继续见 [Revision012报告](../Combined_Channel_Revision_012/REPORT.md) 和 [输入矩阵](../Combined_Channel_Revision_012/MODEL_INPUT_MATRIX.md)，本版不改其任何数值。\n',
             '## 范围与方法\n',
             '完整台账保留 LOCAL 687 小时（156日期）、英国 Chilbolton 143 日/3432小时和 Sparsholt 93日/2232小时。英国合格标签5593、物理共同5538分别保留；本版雨/阴比较在共同掩码内，不退回原39日。原始缺日不补，所有完成日期即使当日无雨/阴可评分小时仍在 coverage_by_date.json。\n',
             f"实际雨/阴评价为 LOCAL 209小时＋英国3027站点小时，共{q['seconds_replayed']:,}个有效原生秒、{q['run_hours_replayed']:,}条运行—小时评分。有效秒不是独立气候重复。LOCAL保留>=9条合格100ms记录的秒并按记录数加权；UK原生1秒等权。\n",
             '主指标是原冻结201分位数中点网格CRPS（dB，越小越好），401网格为固定数值敏感性；同时报预测区间覆盖率。每小时先评分，再按日期等权；小时等权完整并列。Bai LSTM/TabM均为原先已训练的LOCAL适配版，不是作者预训练权重；沿用17/29/43全部运行，表中±是运行间样本SD，不是置信区间。\n',
             '四类各自只改变冻结小时中心位置，使用相同快变CDF。IID/block的边缘CDF相同；本结果不能表明时间依赖改善、秒级确定性预测成功或完整MTS已实现。未生成随机轨迹，未重新选择种子/块长或拟合参考/CDF。\n',
             '## 日期等权主补充结果\n',
             '| 数据集 | 天气 | 站点小时/日期 | 模型 | CRPS201(dB) | 排名 |\n|---|---|---:|---|---:|---:|']
            for cohort in ['LOCAL','UK','chilbolton','sparsholt']:
                for e in ['rain','dry_overcast']:
                    for r in sorted([r for r in scores if r['cohort']==cohort and r['event']==e and r['weighting']=='date_equal'],key=lambda r:r['rank_crps201']):
                        text.append(f"| {cohort} | {events[e]} | {r['n_station_hours']}/{r['n_dates']} | {names[r['family']]} | {fmt(r,'crps201_db')} | {r['rank_crps201']} |")
            text.extend(['\n日期等权下：LOCAL雨/阴都是TabM最低；UK合并雨/阴均为组合ITU近似最低；Chilbolton同样两组ITU最低，Sparsholt雨为TabM、阴为ITU最低。本文不是本版全部组别最优。该结论只对应此概率指标和固定聚合方式，不替换小时RMSE结论。\n',
             '## 小时等权并列结果\n','| 数据集 | 天气 | 模型 | CRPS201(dB) | 排名 |\n|---|---|---|---:|---:|'])
            for cohort in ['LOCAL','UK','chilbolton','sparsholt']:
                for e in ['rain','dry_overcast']:
                    for r in sorted([r for r in scores if r['cohort']==cohort and r['event']==e and r['weighting']=='hour_equal'],key=lambda r:r['rank_crps201']):
                        text.append(f"| {cohort} | {events[e]} | {names[r['family']]} | {fmt(r,'crps201_db')} | {r['rank_crps201']} |")
            text.extend(['\n## 预测区间覆盖：日期等权\n','| 数据集 | 天气 | 模型 | 中央90%覆盖 | 中央98%覆盖 |\n|---|---|---|---:|---:|'])
            for r in scores:
                if r['weighting']=='date_equal':text.append(f"| {r['cohort']} | {events[r['event']]} | {names[r['family']]} | {100*r['central90_coverage']:.2f}% | {100*r['central98_coverage']:.2f}% |")
            text.extend(['\n多数组别低于名义覆盖率，不能声称这些区间已校准。英国气候/仪器/晴空定义迁移均保留，未用英国标签调宽分布。误差小或CRPS下降本身不等于概率覆盖可靠。\n',
             '## QA与可续接产物\n',
             f"生产端加权pinball与独立能量形式逐秒复核全部25,888行，最大评分差{q['max_score_delta']:.4e}；原生小时标签最大差{q['max_native_target_delta_db']:.4e} dB；逐键台账6351行、运行汇总128行、家族汇总64行均复核。7项生产合成测试和3项独立合成测试通过。独立指不同实现、同一代理，不是外部科学审稿。\n",
             '统计审查技能促使本版显式区分秒/小时/日期单位、样本SD与置信区间，并将两种预定聚合完整并列；没有增加显著性检验或改样本。201/401、尾超越、慢变退化CRPS完整保存在 family_metrics.json / run_metrics.json，逐小时在 hourly_scores.parquet。最终封存以 final_verification.json 为准。\n',
             'E-SST仍缺已核全文公式/配置与部分气象输入；RMOP仍缺合法软件、作者传播配置及兼容三维水凝物路径；NWP–MTS仍缺原版系数/实现与合法事件库。此前已发出的材料问题等待答复，本轮没有新增天气请求、作者联系、训练或推理。三者没有以通用近似冒充原版、也没有填零进排名。\n'])
            md('REPORT.md','\n'.join(text))
            md('CHANGE_RECORD.md',f'''# Revision017 修改记录

{now()}。新增四类已验证小时预测的共同冻结快变概率评价，雨/阴全部已完成日期；不是新增第五类论文模型。新代码 common_fast.py / verify.py / publish.py；只在017范围写入。旧39日动态结果和旧LOCAL十一模型结果不重跑，旧输出、权重、输入、STOP、调度均未改。

执行环境、命令和PID见 launch.json/started.json；生产与独立QA stdout/stderr各自保存。新计算3236站点小时、25888运行—小时，独立重放11631746秒。源码/源SHA见 sources_before.json、delivery_manifest.json。最终以 final_verification.json 为封存验收。

未成功的只读元数据探查曾把列表台账当字典，得到AttributeError；未执行评分或写数据，随后按实际列表结构处理。无下载/训练/公式拟合失败被隐藏。
''')
            md('HANDOFF.md',f'''# Revision017 续接与释放

{now()}；本版共同冻结快变概率评价和独立实现QA已完成；最终文件/源一致性由 final_verification.json 确认。LOCK scope=/root 原线程的017新目录，计算和复核OS锁均已释放，封存后逻辑占用RELEASED；execution.lock保留，不以存在判断进程。

禁止重复执行 common_fast.py / verify.py / publish.py：一轮式排他创建，已完成结果应复用。新增结果仅为四类慢变＋同一冻结CDF的边缘概率比较；不包含新的随机序列/ACF/持续时间/斜率，不将其称为完整动态MTS。

当前主入口：本目录REPORT及独立QA；小时排名和七类来源矩阵仍在Revision012；原始输入核验013、PRM条件核014、RMOP015、MTS许可/来源016继续有效。英国完整143/93日期没有缩回39日，所有6351行可查本版ledger与coverage_by_date。

E-SST、RMOP、NWP链及完整动态对照仍有外部材料/输入门槛。待用户已询问材料答复后，再做版本、许可、SHA、单位、频率/极化、几何和UTC兼容检查；不能无材料重复审计、制造占位计算或再跑本版充当进展。无新材料且无安全独立工作时，清楚说明阻塞并等待方向。不得为了排名调参数、训练、重估快变/晴空参考，也不得恢复天气采集或自动删除STOP。

七类整体未完成，同一tvt保持原设置，不另建任务或自行暂停。新授权/新材料出现前，不外联作者、不购置、不签许可。
''')
            assert all(sha(p)==h for p,h in protected.items())
            emit('publication_qa.json',{'status':'PASS','utc':now(),'coverage_dates_by_station':dict(Counter(r['station'] for r in rows)),
             'coverage_hours':sum(r['all_completed_hours'] for r in rows),'score_hours':sum(r['hybrid_hours'] for r in rows),
             'score_seconds':sum(r['hybrid_valid_seconds'] for r in rows),'independent_qa_sha256':sha(OUT/'independent_qa.json'),
             'source_sha_unchanged':True,'full_seven_complete':False})
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    emit('publication_lock_release.json',{'utc':now(),'status':'RELEASED'})
    delivery={str(p.relative_to(ROOT)):sha(p) for base in [HERE,OUT] for p in base.iterdir() if p.is_file() and p.name not in ['execution.lock'] and p.suffix!='.log'}
    emit('delivery_manifest.json',{'utc':now(),'status':'READY_FOR_FINAL_FILE_VERIFICATION','files':delivery})
    print('PUBLICATION_READY',flush=True)
if __name__=='__main__':main()
