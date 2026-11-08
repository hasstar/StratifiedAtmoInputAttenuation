# Revision 002 续跑手册

## 最新状态：全部技术实验与交付已完成，禁止重跑

2026-10-07北京时间05:54:50，`final_evaluation/delivery/delivery_qa.json` **PASS**，`whole_fast_fading_stage_complete=true`。A/B/C1、C2 R8/R16、最终生成器、C3、D1、D2、独立17,647条报告审计、10组PNG/PDF图表及中文报告均完整。1,349指纹和967数值复核通过，零次新拟合、零次最终数据回调；所有旧阶段和记录只读。

- 主交付：`Outputs/Experiments/Fast_Fading_Revision_002/final_evaluation/delivery/FINAL_RESULTS_ZH.md`。
- 图表选定 `delivery/figures_v3`；前两次排版和22份完全一致的源CSV保留。视觉检查为AI检查，尚待我审阅并确认是否用于投稿，不影响本次技术实验交付完成。
- 当前没有本版存活worker。原`tvt`已于2026-10-07北京时间05:59:48通过应用接口置为 **PAUSED**，实际配置复核通过，原提示词、周期、会话及创建时间均未变。完成回执为 `delivery/completion_receipt.json`。收到过期心跳时先读最终QA与完成回执，不恢复旧PID、不运行旧run/report、不新建定时任务。
- 将来的新科学问题需要独立授权和版本；不能依据最终测试改善/不足重新选CDF、块长、窗口或慢变模型。历史测试访问及同日训练重叠、合成区间覆盖不足、雨天连续块不足与高频谱空覆盖均保留。

下文为**历史运行记录与恢复规则**，不是当前待执行命令。只有新故障证据与明确范围授权才检查恢复；整体完成后默认只读核验。

## 历史代码与阶段（均已完成）

工作目录：`E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`。

1. `python Code\Fast_Fading_Experiments\Revision_002\day_block_uncertainty.py run`：A阶段。
2. `python Code\Fast_Fading_Experiments\Revision_002\day_block_uncertainty.py verify`：A阶段独立复核。
3. `python Code\Fast_Fading_Experiments\Revision_002\window_diagnostics.py run`：B阶段，按月断点续跑。
4. `python Code\Fast_Fading_Experiments\Revision_002\window_diagnostics.py verify`：B阶段复核。
5. `python Code\Fast_Fading_Experiments\Revision_002\window_diagnostics.py report`：B阶段小时/日期汇总。

6. `python Code\Fast_Fading_Experiments\Revision_002\fit_marginals.py run`：C1，八候选×五日期折，共40条候选-折记录、19,232条小时评分。
7. `python Code\Fast_Fading_Experiments\Revision_002\fit_marginals.py verify`：C1复核；包括每候选-折一个源小时的独立评分复算。
8. `python Code\Fast_Fading_Experiments\Revision_002\fit_marginals.py report`：只以训练日期CV冻结边缘族。201/401积分网格获胜者不同则先诊断数值精度，不能直接进入最终评分。
9. `python Code\Fast_Fading_Experiments\Revision_002\freeze_marginal.py`：全训练小时拟合冻结族，保存解析参数或完整经验支持；从保存的支持/CDF独立复算单位功率矩。重复调用只复核，不重拟合。
10. `python Code\Fast_Fading_Experiments\Revision_002\document_marginals.py`：保存完整边缘结果与同日期配对不确定性文档。

A/B已完成，`MODEL_PLAN.md`已在C1拟合前冻结。相关生成器阶段还需实现并冻结`DYNAMICS_PLAN.md`。不得将边缘族选择完成称为完整快变模型完成。

## 历史运行期间每次唤醒先检查

- 完整读取`PROTOCOL.md`、`PARAMETERS.md`、`PROGRESS.md`和本手册，以及输出`run_manifest.json`、阶段QA。
- 运行A、B、C1三个脚本的`status`，读取stdout/stderr、每月结果和候选-折数量。C1日志为`marginal.stdout.log/marginal.stderr.log`。
- 查精确后台命令行、PID和CPU；需要时使用只读`Get-CimInstance Win32_Process`与`Get-Process`。后台启动必须`Start-Process -WindowStyle Hidden`并保存stdout/stderr。
- 同命令worker存活，或锁内PID存活时，不再启动。
- PID已结束先区分正常完成/失败。正常完成就复核QA；失败且无匹配进程时，查日志、指纹与最末月，按原命令恢复。脚本不重写已有已核验月。
- 原始数据、Rev001和慢变冻结版本只读。

## A门槛

同一批839训练小时、65日期、7长度；每天气2,000次日期块重采样；输出28条长度汇总、12条配对对照和56,000条重采样数值。源表和脚本指纹一致；`verify`从保存的重采样数值重算区间、从源表重算点估计。全部PASS才能推进。

## B门槛

九个月输入集合和SHA256与Rev001一致；每月四质量门槛×七长度均保留状态，空状态明确记录。每个结果均属于官方train小时，窗口不跨小时，气象类别与源小时一致。输出原始窗诊断、小时汇总和同小时/日期区间。零插值。`verify`PASS后才拟合分布。

## C/D必须落实的规则

- 先冻结按日期隔离的训练折；事件条件可以用于分组，不能使用验证/测试残差选择参数。
- 分布与动态生成器的每个参数补到登记表，保存实际拟合值、样本小时/日期数、配置和数值失败。
- 对功率矩没有保证的dB重尾分布不能直接用于功率守恒生成。
- 每个模型同时给边缘与动态评分；先以真实小时均值研究快变，再与冻结慢变预测合成。
- 15分钟能成为最终参数必须有本数据证据；不能因为先前论文写过就固定。
- 完成所有QA、图表和详细结果报告后更新进度并停用本定时任务。状态未变时保持安静，仅阶段完成、失败或需由我作出决定时通知。

## 2026-10-07 历史接续：D2原始100ms诊断

A/B/C1、C2 R8/R16、全训练生成器、C3及D1最终评价/报告均已完成且QA PASS，全部只读，不重跑。当前细化运行规则以 `FINAL_EVALUATION_PLAN.md` 和 `FINAL_EVALUATION_RUNBOOK.md` 为准；先完整读取这两份文件以及PROTOCOL、PARAMETERS、PROGRESS和 `final_evaluation` 阶段QA。

- 当前D2 worker为PID22056，2026-10-07北京时间05:06:23隐藏启动，精确命令 `D:\Anaconda202410\python.exe -u Code\Fast_Fading_Experiments\Revision_002\run_resolution_diagnostics.py run`，日志 `resolution.stdout.log/resolution.stderr.log`。PID只作定位，必须核对命令行、CPU、OS锁和active_worker；不把旧PID或元数据文件当存活证据。
- 先运行D1及D2 `status`；同命令worker存活只读健康检查，不修改冻结规则、不重复启动。确认失败且无同进程才按相同run/指纹从已核验月份恢复，恢复另名日志、保留failure审计。
- D2门槛：9个月、3,434小时、120,936,525质量零100ms记录、12,087,671主质量秒；原始W逐秒复现clean_1s，Jensen差与单位功率矩通过，原时间戳严格规则段、全部128/32/64秒PSD及Nyquist/频带积分独立复算、所有hash通过。空覆盖不能放宽规则或伪造数据。run自动verify，QA PASS且进程正常结束后才执行D2 report。
- D1已形成133检查点、1,030小时、2,060动态/合成记录、65,920实现、279,441窗口状态，独立CRPS/VS/窗口概率QA与报告QA PASS。D manifest固定 `97af97e9ef46f571d6b279a88aa2a6589839549380060a06d193f78fc6c3f96a`。没有新增拟合或测试回调。
- D2 QA/报告PASS后，另存独立汇总与连续1/3/7日区间审计、图表脚本、完整中文报告；图表只用冻结计划最早示例小时。报告慢变、快变、合成各自结果及总体/晴天/雨天/干燥阴天/other、质量数、设备/谱配置、100ms与同日训练重叠。测试历史访问保持审计，不称新盲测。
- 256秒是生成块长，主90%工程窗晴天/雨天15分钟、干燥阴天/other10分钟，不由最终数据重新选择。IID/block单时刻CDF相同，随机实现误差不能用作边缘改善；12:38是随机场景而非确定性逐点预测。全部QA、独立审计、图表与报告完成才停用原`tvt`，D1或D2完成均不等于整个快变实验完成。

### 历史更新：D2 run已完成，report在运行

D2/PID22056已于北京时间05:09:46正常结束且QA PASS，9个月/3,434小时/120,936,525原始记录/12,087,671主质量秒齐全，不得重跑。当前唯一worker为 **PID2176**，05:13:49启动 `run_resolution_diagnostics.py report`；日志 `resolution_report.stdout.log/resolution_report.stderr.log`。先核对精确命令行、CPU、OS锁与active_worker，存活时只做只读健康检查。stderr只有空/全NA拼接FutureWarning，不为此修改冻结脚本。正常结束且 `final_evaluation/resolution/report_qa.json` PASS后，才另存独立D1/D2汇总与连续日区间审计、最早小时图表及完整中文报告；不得把诊断完成称为全实验完成。若确证report失败且无同进程，诊断后仅用相同report命令恢复、日志另名，不重跑D1/D2 run。原`tvt`仍ACTIVE/每15分钟，全部报告和独立核验完成后才停用。
