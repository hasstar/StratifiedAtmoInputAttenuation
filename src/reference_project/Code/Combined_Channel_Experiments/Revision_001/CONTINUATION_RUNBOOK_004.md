# 安全继续：仅调整日期分批，恢复并完成外部气象下载

2026-10-14 08:42 +08:00。我明确授权：“仅调整下载分批方式后继续，外部天气必须下载完成，下载进程退出了你要重新下载啊”。本补充只替代旧runbook的下载分批/入口/恢复条款；003的零训练约束继续具有优先级。不是重新训练、扩展日期/站点/变量范围或重跑LOCAL的授权。

## 原因、保留与新版本

原worker38056于2026-10-14 00:06:45 +08:00失败退出，服务端HTTP403：cost limits exceeded / Your request is too large, please reduce your selection。原failure在external/weather_failures/1791907605254129200.json。原请求未留下该失败请求ID，不能虚构ID；不盲目重跑相同月级大请求。原weather_download.py、external_prepare.py、77项weather_plan.json、六个已完成载荷（正式4、smoke2）、全部job/log/failure/smoke QA保持只读。

新程序weather_download_v002.py不导入任何模型或训练模块；新输出只在external/weather_v002。原共享OS字节锁external/weather_writer.lock继续用于排他，不改其内容；新活动记录在weather_v002/active_worker.json，旧active_worker保持FAILED历史记录。

原77项科学覆盖不变：GBS England/Dundee及France原日期/时刻/变量/压力层/坐标/格式原样保留，含缺口和末日之后00UTC降水/降雪补时。复用4个原正式载荷；未完成压力请求先每批最多7个已有日期，地面初始沿用原月/补时批次，共196个新初始批次。7日只是保守运行选择，不声称是官方上限。若官方明确拒绝“请求过大”，新程序保留失败和父子来源，确定性二分该批day列表，最低1个日期；不删变量、压力层、小时、地区或日期。运行期叶批次数可能因继续拆日增加，但原77项覆盖不得变化。

新计划：external/weather_v002/plan.json。

- fingerprint：248cd58b1dc1fc7ae0baadcb1da39adb827daeb81b220335a5b9f363113158a0。
- 文件SHA256：8bdd4a48620dcccfa8f01fb182d79cf6ee62166dd20256bcf81db10158ec6149。
- weather_download_v002.py SHA256：26e2ac7205a4de17ac1d81af50c52912a30e8fbed3d9ea57583111ece7a0f2ec。
- test_weather_download_v002.py SHA256：340e3b3a27380ebdbc02541e93766f2aba3d8ef4e17f0b36568cf506dc5af860。
- preflight_qa.json SHA256：9e2d111ea42acc80f22400d92c8cab98f762cc3bd1d6288da4aa6896268b6b41；20项离线恢复测试PASS，77父请求日期多重集/全部非日期字段/补时/28原始文件指纹检查PASS，无网络请求或训练。

原plan fingerprint仍为c7fa6ff824fb436477f534fcec022ef7d60aed6afd4ae507f9dd1575e0282e16。四个正式复用项为GBS England 2004-02 pressure/surface、2004-03 surface、2004-04 surface。两个smoke不计入正式覆盖。初始覆盖1848/45663请求小时（不同地区/字段类型各自计数，不是独立观测数），完整父项4/77；196个新批次不能与77父项混算百分比。

## 当前worker及健康证据

2026-10-14 08:40:09.189529 +08:00隐藏启动PID39924。精确命令：

`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download_v002.py run`

工作目录：E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction。

日志：external/weather_v002/weather_run_002_20261014_084009.stdout.log及同名stderr.log。两次CPU0.328125/0.40625秒、创建时间/命令与RUNNING活动记录匹配，共享字节锁持有；stderr为空。08:41检查两批均accepted：

- GBS_England_2004-03_pressure__b001：761638d0-8222-4d54-86bb-cfe2889ebe71。
- GBS_England_2004-03_pressure__b002：e1a9d038-f1c0-4865-ab1a-f82d7038b625。

此为历史快照，后续必须重读status、jobs、progress、logs与实际进程，不以旧PID或accepted误报完成。

## 每次心跳的实际恢复动作

1. 完整读取T012、003、本文件、PROGRESS、历史协议和已有QA。只读执行新入口`weather_download_v002.py status`，核对新/旧计划与代码SHA、28个保护文件、已完成载荷SHA、jobs及失败审计。不要再执行原run/smoke/prepare或原verify_weather.py。不要重跑prepare或20项离线QA来覆盖冻结产物。
2. 枚举实际Python命令，识别所有weather_download*.py的run/smoke/prepare；核对精确命令、PID、创建时间和两次CPU，查新active_worker及共享OS字节锁。存在匹配worker或锁被占用时不另启worker，不修改冻结代码/计划。accepted/running以及低CPU网络等待是正常状态。
3. 若天气尚未完成、确证无匹配worker且共享锁释放，检查退出/失败记录和.partial。正常退出但coverage未齐、系统结束留下RUNNING活动记录、或已知可恢复网络失败，都不能只写“继续观察”：在指纹一致且没有未解决的提交/访问歧义后，立即以本文件的新run命令隐藏恢复，使用带新时间戳的stdout/stderr文件。命令必须加-B，不重建计划；原完成文件校验跳过，原request_id续接，不重新提交排队任务。恢复后检查真实PID/创建时间、两次CPU、锁和远端ID。
4. 新程序对已有ID的网络/传输失败保留ID与审计，按60秒到900秒退避重试；不因此新增远端请求。退出后按第3条恢复即可。程序同一时刻最多2个活跃远端请求。
5. `.part`不静默删除：先检查SHA、实际大小、NetCDF可读性和同request_id的官方预期大小。完整的part可转正式；不完整的part移到新partial_history保留审计，再从同远端ID重新下载。不能绕过诊断或删除失败证据。
6. `SUBMITTING/AMBIGUOUS_SUBMISSION`表示提交结果未知；先只读对账官方近期作业、精确请求与创建时间，找到唯一对应远端ID后另存恢复证据并以新状态恢复；未对清之前不得盲目POST。`BLOCKED_REMOTE_ACCESS`（401/403/404/410）不能当成远端已终止来释放槽位；先诊断。`BLOCKED_HTTP`若是账户/条款须由我处理，不换代理。`BLOCKED_REMOTE`保留其terminal ID及原因，先诊断；只有明确是请求大小的失败才自动日期二分。最小单日仍超限时报告，不擅自减变量/改时间。任何新恢复代码必须另存版本，不能修改正在运行/冻结的v002。

## 完成门槛与零训练

全部196初始批次及必要的日期子批成功、加4复用载荷精确覆盖原77父项，才生成weather_v002/download_qa.json，状态仍仅PASS_DOWNLOAD_READABILITY_ONLY。不能以此或smoke替代全量ERA5字段/坐标/UTC/单位/数量QA；全量独立天气QA继续另存，数据空值/数量异常必须保留。后续只能清点已有模型及预处理兼容性，再做兼容模型的推理/组合/指标评价。不得新训练、微调、fit/partial_fit、调参、重拟合归一化/插补/晴空参考/CDF。

原tvt维持15分钟、原线程、ACTIVE，不另建任务。我持续推进外部气象完整下载；健康不变安静，失败/阶段完成/需由我决定时通知。下载结束或LOCAL完成都不是整项完成；LOCAL/GBS/OpenSat全部既定独立QA、图表和中文报告齐备才写总completion并暂停同一tvt。

## 独立审计与自动化回查（08:45 +08:00）

audit_weather_batching_v002.py不导入下载器，独立逐父项比较日期多重集及全部非日期字段，复核4个复用文件/28个旧保护文件、45663请求小时、新代码无模型导入与训练调用，PASS。审计器SHA256为934cf15ca6c8203218b83cdd333098dd0ce7b4abd9a66bdf5a606f4527e2f141；external/weather_v002/batching_independent_qa.json SHA256为37bbcaa0382ee99639dd6650821c923aa917fa8b4ad4f3c0281bf20b0b3c602b。该QA仅审计日期分批和溯源，不冒充全量ERA5载荷QA。

原tvt已通过自动化工具更新为新版入口及主动恢复，保存提示逐字回查一致；原id、名称、线程、15分钟间隔及ACTIVE状态均保留，未新建任务。08:45:35实际仅新worker39924存活、锁持有、CPU0.75秒、stderr为空；b001已remote running，b002仍accepted，正式父项完成仍4/77，新批verified=0。最新每次检查应读取实际状态而非此快照。
