# T012 continuation 008 — 仅自然Top10，四层固定层高＋视线最近网格，零训练

2026-10-15我明确撤销4个辅助位势下载，并确认“接受固定层高近似，只新下载Top10”。本文件与T012最新段落优先于007的逐小时四层辅助z定位及全部旧下载方案。003零训练、原数据/模型/结果/QA只读不变。固定高度近似已经授权；不把它说成逐小时真实等压面。

## 唯一正式入口

cwd项目根；输出`Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_fixedheight_v007/`。

```powershell
& 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B 'Code\Combined_Channel_Experiments\Revision_001\weather_download_top10_fixedheight_v007.py' run
```

只读入口末尾改`status`。initialize已于16:40一次性完成，绝不重跑。所有旧nearest v006/v005、LOS v004、Top1、v002/v001均不恢复run/smoke/prepare/initialize。不要把新下载器v007与旧runbook007混淆：当前指引为008。运行/冻结代码和计划不修改，修复另存新版本。

- 下载器SHA256：`50398dc98db8c623e003f2d8dccdf65ef19dab9d1a5392b934dbb3470c516d68`。
- fixedheight_los_core_v001.py：`565e480b78fff3ff5a5b21eee40e4ebb9d46e7d70e3dfb41acbd47aee665feb5`。
- 测试SHA256：`a6d74409611012b75c65a9dbbe03c590240b8e3615de4630757170130872cd6d`。
- 独立审计SHA256：`81e4e1f2a8152723850319a4ad080a5f691cd56a31e92119f3673756763b6518`。
- 预检入口SHA256：`d1b87dee3f6767e41aed7a3c1bfc3f5c56ef283a39c3dc696fc68e854dbc6ea0`。
- plan fingerprint：`adec64ebf2cfefc45fe3e1a153daeaf784a31680af64783a3a86e56cdfa06a73`。
- plan SHA256：`369d29d0fce76aaddfc9b8efc1e02710077e74fa094b167b0a0e6d4d77aa51f9`。

16:41冻结预检PASS：61新测试＋20传输恢复测试，共81项；独立原日历/218个1hPa初始射线/871个有效固定层交点/1204几何来源SHA/299旧保护文件SHA/精确样本打包审计PASS。另有1个固定层交点明确无效并保留。仅计划/迁移/静态几何和离线字段门控测试，不是完整实际字段或模型兼容QA。未执行任何模型训练或预测。

## 只允许新下载的十列

8个压力特征对：1hPa geopotential/u_component_of_wind；5hPa ozone_mass_mixing_ratio/u_component_of_wind；500hPa ozone_mass_mixing_ratio；650hPa u_component_of_wind；850hPa ozone_mass_mixing_ratio/relative_humidity。地面两列：convective_available_potential_energy、maximum_2m_temperature_since_previous_post_processing。

不新增z@5/500/650/850，不下载云雨雪或其他诊断变量。`auxiliary_pairs=[]`，自然Top10不重选。新请求生成、读取waves/分裂子节点及POST前均检查变量—层白名单，每请求仅一个压力层；不做4×5笛卡尔扩展。原77父项中的旧request只提供日历来源，绝不是新POST模板。目标字段直接下载，不从其他气象变量推导。

1hPa位势本身属于Top10，不是辅助字段；继续使用该层已有/所需z逐小时定位最近格，最多8轮，初始48000m只作猜测。缺z仅请求z@1，稳定后下载该点u@1。循环、不收敛、缺值或低于天线保留无效。

其余四层用固定标准大气几何高度：5hPa=35979.046m、500hPa=5579.330m、650hPa=3592.717m、850hPa=1457.635m；按原站星前向WGS84射线求交，取最近0.25度发布网格。标准高度来自[NASA/TM-2005-213659 Appendix A](https://ntrs.nasa.gov/api/citations/20050207438/downloads/20050207438.pdf)的标准大气解析压力—高度关系；独立审计用其舍入表基值与压力二分核验，最大差0.010731m。常数和高度非数据拟合；几何方程数值残差不代表真实大气高度误差。

FRA-00289天线1622m高于固定850hPa层1457.635m，该层`INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA`，不抬高层面、不取反向交点、不补零。链路/日期身份全部保留，其缺失特征和预测兼容性需明确列缺口，不能说该站完整Top10可用。FRA-00609原方向差0.5424°仍待复核，不放宽0.5°阈值、不改指向，其预测待几何核验；其他214不宣称精密星历验证。

地面字段取站点最近格；所有高空字段沿站星视线，不改为地面垂直柱。最近格按经纬逐轴最近，中点取北/东；不传`grid`再网格化、不做空间或时间插值。必须标注“5/500/650/850hPa固定标准层高＋视线最近发布网格近似；1hPa用Top10 z逐小时定位；与原模型双线性采样不同”。不声称CDS发布格就是ERA5底层未插值谱/高斯格。

## 日历、复用和计数

原77父项/45663日历请求小时、33压力地区月/22824小时及原日期缺口、surface补时、站点分区保持。此计数不是去重后新增下载数据量。23份旧载荷只读候选复用；必须实际字段身份/UTC/层/坐标/单位/有限值及SHA相容才计所需样本，不能整份计新LOS覆盖。

每波仅补缺少的精确字段—层—格点—小时；相同时间掩码和无空洞矩形才合并，逐项检查不增日期/时刻/格点/字段/重复值。按原父项顺序推进，不按预测成绩择日。`units/`只是采集/几何记录，包含无效行，不代表字段独立QA或模型预测完成。

独立静态未扣缓存估计：四层直接目标＋1hPa初始z共505请求/1580832格点字段小时，surface128请求/461528格点字段小时；不包含后续1hPa纠正/u请求，不是最终新增总量。严格点请求仍可能排队较多，不保证更快。若报告“小时”，必须区分日历请求小时、站点小时、唯一格点字段小时及十列齐全的站点小时，不混用。

## 旧进程和遗留辅助请求

旧v006 PID25784/创建12:00:26.505335及精确命令核验后，于16:23:42.5712088停止本地调度；旧active_worker的RUNNING仅历史。8份状态/日志副本保存在`handoff_state_before/`，完整SHA核对；未取消远端，旧计划/代码/日志/载荷不改。

唯一非终态继承ID：`dbac99a1-b428-417e-9d26-5442ca99679d`，旧February z@5。新版只GET其状态，绝不get_results/下载其载荷、不重提、不分裂。accepted/running仍占一槽；官方确认successful后标记`RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD`释放槽，failed/rejected/dismissed/deleted明确终态记录为`RETIRED_REMOTE_TERMINAL_NO_RETRY`，不算新数据覆盖。401/403/404/410访问问题必须BLOCKED_REMOTE_ACCESS，不可推断任务已终止而释放槽；网络错误同ID退避。

已完成的旧1hPa February z及February surface、December z等留在原处按所需值复用。所有完成历史请求不重新提交。新启动PID、创建时间、日志和实际请求见`launch_verification.json`/`handoff.json`；这些均为历史快照，每轮必须实查。

## 每轮检查与恢复

完整读T012、003、本008、PROGRESS以及新目录plan/handoff/launch_verification/preflight_qa/plan_independent_qa/waves/units/jobs/progress/active_worker/logs/failures/partial_history；plan引用的旧geometry/model_inventory只读。

核对全部weather_download*.py实际命令、PID/创建时间、两次CPU、共享OS字节锁、新旧SHA、日志/载荷。存活只读健康检查，不重复启动、不改冻结代码；低CPU/accepted/running不是失败。最多两条实际在途请求，遗留poll-only辅助ID在未明确终态时也占槽。

只有实际下载进程均无、OS锁释放、下载未齐、冻结指纹一致、失败已诊断且没有提交或访问歧义时，隐藏恢复上述唯一v007 run并使用新日志名；绝不重跑initialize/prepare，不重启旧版本。SUBMITTING/AMBIGUOUS先官方只读精确对账，不盲POST；BLOCKED_REMOTE_ACCESS不释放槽。新的Top10请求只有明确大小失败才按原日期二分，最低单日；非大小/账户/条款问题先诊断，不换代理。part核对实际与官方大小、SHA、可读性，完整才转正式，不完整原样保留partial_history后同ID重下。任何修复另存新版本，旧失败不覆盖。

## 下游和完成边界

实际下载字段还须独立UTC/层/单位/参数身份/网格/数量/有限值/SHA/射线残差QA，再核对目标与既有模型兼容，才可先做明确覆盖范围的局部推理。固定四层不声称真实等压面残差验证。

现成k_0010 SHA=`b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5`；BiLSTM seed17/53轮，真实LOCAL2404小时拟合，十列/原生268槽。只加载冻结模型和预处理做transform/predict/组合/指标，不给156列网络补零。禁止fit/partial_fit/fit_transform/训练/微调/选种子/调参/优化器更新/重估归一化、插补、晴空参考和CDF。LOCAL全部QA/模型/结果只读不重跑。

GBS原生秒级relative-clearsky可复用旧ClearSky Alignment004小时标签，OpenSat原生分钟RSL平均域未知，不插值造秒真值、不重拟合分钟CDF。其他模型族仅已有且兼容才比较；缺模型/特征/目标尺度/天气分组诊断要列不可评估或需决策，不伪造、不扩下载。分别按来源频段及总体/晴/雨含强雨/干燥阴天/other报告RMSE、MAE、Bias及适用CRPS/校准和样本/日期/站点，不跨频段合并RMSE。

历史测试访问/同日重叠不称新盲测；随机场景不是确定性秒预测；IID/block共享CDF常数增量相消，不把共同VS当慢变增益。LOCAL CRPS0.130487较自身0.162335降19.62%、90覆盖73.98%不是已校准。保留负结果、空数据和数值失效。

原tvt、同一线程、15分钟保持，不另建任务。正常无变化安静，阶段完成/失败/需决策才通知。下载、字段QA、局部试跑均非T012整项完成；只有既定外部评价/独立QA/图表/中文报告/进度齐备，或我明确缩减最终范围且满足，才总completion并暂停同一tvt。
