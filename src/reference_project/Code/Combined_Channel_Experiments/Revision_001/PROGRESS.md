# Combined Channel Revision 001 进度

<!-- REV006-BEGIN -->
## 当前：新增两项冻结作者预训练对照完成，零重训（2026-10-16）

最新人工指令优先：作者权重不改、不重训。本轮新增TabPFN-v2、TabICLv2官方预训练回归权重，全部学习参数原值/前后SHA一致，梯度/优化器/推理联网计数0；LOCAL2404小时只作固定上下文，UK/测试标签只评分。同一Top10，LOCAL687/扩展UK5593小时：TabPFN RMSE0.4299/1.1190，TabICL0.3546/1.3366dB；本文冻结Top10仍0.5041/1.8286。

18个旧检查点原样保留、不重训，旧预测全部复用；Bai/TabM是此前LOCAL适配版，未取得兼容作者检查点，恢复作者权重0，不把默认配置当作训练好模型。12702新预测、342指标/192家族汇总、1513保护源SHA及分块/键/标签独立核验PASS。TabPFN外部分块调整及原失败记录保留，模型参数不变。

采用[报告](../../../Outputs/Experiments/Combined_Channel_Revision_006/COMPARISON_REPORT.md)、[来源审计](../../../Outputs/Experiments/Combined_Channel_Revision_006/MODEL_PROVENANCE.md)、[交接](../../../Outputs/Experiments/Combined_Channel_Revision_006/HANDOFF.md)；[执行协议](../Revision_006/PROTOCOL.md)。两项进程均正常退出，19:02具权限CIM两次Python进程0，最终封存/释放以新delivery_receipt.json为准。完整固定索引before/after另存，历史正文保持。下载停止、tvt PAUSED，不关闭全部T012，不根据下方旧授权恢复训练或天气入口。
<!-- REV006-END -->

<!-- REV004-BEGIN -->
## 当前：近似ITU雨衰缺项已补算并核验（2026-10-16）

最新人工授权“雨层高度可以近似”“近似ITU基线、没完成的请补算”。WRITE仅新Code/Combined_Channel_Experiments/Revision_004与Outputs/Experiments/Combined_Channel_Revision_004及本固定索引交接；原/root线程，C1—C8/OS锁/READ边界详见Revision004/PROTOCOL.md，完整索引before已保存。

英国已补齐P838几何雨衰、P618小时代理及预定18高度/极化情景，未训练或重新调用学习模型；全部20运行复用旧预测并按同样本评分。两站236原生文件确认20.7GHz发射/接收匹配左旋圆极化，纠正下方历史“极化缺失”；P839气候雨层海拔约2.415/2.420km，按接收机84/118m海拔扣除，±0.5km仅敏感性，不按分数选高度。

UK共同5538小时（原5593中55个负液态差值不用于物理公式）：本文Top10 RMSE1.8375、P8380.1772、P618小时代理0.2011dB；另存固定0dB事后诊断0.1732dB。LOCAL687本文0.5041、TabM均值0.4009，原P8381.0378/P618代理1.0405只读复用。本文并非当前统一比较最优；小绝对误差不等于UK时变机制已验证。

7074保护源SHA、236原生元数据、100944有限物理值、1196指标、856家族汇总及28配对区间独立QA PASS。当前采用[Revision004中文报告](../../../Outputs/Experiments/Combined_Channel_Revision_004/COMPARISON_REPORT.md)；旧39日/扩展集/模型/标签/历史结果不改。完整年度P618及随机动态模型未伪装为小时基线。训练0、ML推理0、气象下载0；仅取4份官方ITU方法参考。STOP_REQUESTED保留、tvt PAUSED；固定索引after、最终封存及锁释放以Revision004交付收据为准。历史T012总目标未自动标完成，以下旧段原样保留。
<!-- REV004-END -->

<!-- REV003-BEGIN -->
## 当前交接：扩展英国比较已完成，下载仍停止（2026-10-16 16:20 +08:00）

用户要求完成两组比较及扩展集评价。本轮另存[Revision003交接](../../../Outputs/Experiments/Combined_Channel_Revision_003/HANDOFF.md)，不运行Revision001旧入口。Chilbolton143日/Sparsholt93日的全部完整采集日期已做实际字段/UTC/单位/网格/有限值/源SHA/几何及原生标签QA；5664个气象完整站点小时，主评分5593。新增2003年104/54日至11月30日，另含原39日；12月不完整高空不混入。

20固定运行/10家族：LOCAL687与旧39日1872行预测逐值复用，新增3792行每模型冻结推理。新增训练/参考重估/气象下载0。6981保护源、1112行指标、562家族汇总、20运行18小时重放及266日期簇对比独立QA PASS。修复均另存，旧失败/原图/数据完整保留，固定索引before/after随本次交付。

LOCAL本文Top10 RMSE0.5041，TabM0.4009；扩展UK本文1.8286、训练均值常数0.2558、ElasticNet0.4072、TabM0.6374dB。采用[中文报告](../../../Outputs/Experiments/Combined_Channel_Revision_003/COMPARISON_REPORT.md)与figures_v002；SD不是CI，不据英国分数选种子。英国ITU缺雨层高度/极化输入未执行，LOCAL既有物理代理单列；历史信道预测任务排除。

本轮可执行小时比较包完成并验证，论文采用待审阅，历史T012总验收不自动完成。16:18具权限CIM两次Python进程0；tvt回读PAUSED、STOP标志不变。最终占用释放与完整性由Revision003交付收据记录，不得依据下方历史ACTIVE段恢复下载。
<!-- REV003-END -->

## 当前交接：新比较另存Revision002，下载仍停止（2026-10-16 15:02 +08:00）

用户在叫停天气下载后明确授权直接开展模型对比。第一批新训练/报告位于[Revision002输出](../../../Outputs/Experiments/Combined_Channel_Revision_002/HANDOFF.md)，不是重启Revision001：28次新训练、18检查点，传统方法/Bai2022结构适配/TabM2025官方架构统一Top10；旧本文预测及物理代理只读复用。548保护源、228指标及新检查点重放独立QA PASS。LOCAL数值最低为TabM，外站GBS39的常数控制优于全部学习方法；不改参数追求预期排名。

采用报告及限制由[T012当前记录](../../../Tasks/T012_Combined_Channel_And_External_Validation.md)与新输出HANDOFF索引。原模型/代码/结果/STOP标志不变，tvt保持PAUSED。143/93日扩充仅采集完成，未混入本批原39日评价。此次第一批完成不等于全部动态/年度统计模型或T012验收完成；下面历史记录均保留，不能作为自动恢复下载的授权。

## 最新：按人工要求收尾停止下载（2026-10-16 13:46 +08:00）

用户已明确叫停。13:44:28执行v011 stop写STOP_REQUESTED.json，禁止新GET；当时两项在途请求于13:46:04/05读取超时，原失败保留、不再重试。worker35736于13:46:05正常退出；13:46:32/34具权限CIM两次无下载进程，13:47 status确认OS锁释放且冻结/旧保护/载荷SHA通过。progress=STOPPED_BY_USER，active_worker=EXITED_NORMAL。同一tvt已PAUSED并回读确认；停止标志保留，未经新的明确人工授权不得恢复。

完整Top10＋MTPR/MSR采集日：Chilbolton143日（新增104日，2003-08-19—11-30，加旧39日）；Sparsholt93日（新增54日，2003-10-08—11-30，加旧39日）。旧39日为2004-02-18—21、02-25—28、03-01—31；不是整个连续时段。12月地面/降水已齐，高空未齐，不计入完整日期。最终1072子集/48408新唯一格点字段小时/1759559字节；23采集单元，不代表全字段独立QA。所有已得/未完成文件保留，未改代码/计划、未重跑LOCAL或39日结果，fit/predict=0。T012未完成；后台锁已释放，停止交接与固定索引before/after位于external/weather_gbs_extend_v011/user_stop_001_20261016_134428。

## 最新：单请求404已诊断恢复，继续唯一v011下载（2026-10-16 02:38 +08:00）

02:17旧worker因2003-08-28 U@650hPa精确子集404保护性退出；02:26/02:34具权限CIM、OS锁确认无实际下载器。核对同一文件已有1/5hPa成功载荷，再检查官方原DAS及同约束DDS均200，按一次性新版本恢复工具对唯一原请求重试。02:36实际GET200，1203字节/24值通过身份、UTC、单位、坐标、数量、有限值、SHA门控；原失败完整保留，具体服务器原因未知，未放宽后续404策略。

15项恢复正负控制PASS，541个既存v011文件及原保护SHA核验，冻结下载代码/计划/旧模型结果均不改。02:37隐藏恢复唯一v011 run，PID35736、创建02:37:20.394549+08:00、两次CPU2.5625秒、实际命令/创建/RUNNING/共享OS锁匹配，stderr空。02:37:47快照126子集/7392新唯一格点字段小时，2个GET在途；rain/surface采集记录各2184站点小时，pressure1872，非完整Top10总进度，不能相加。

本轮人工占用RELEASED，后台继续下载；012、同一tvt/原线程/15分钟不变，不重跑一次性reconcile工具。恢复记录位于`external/weather_gbs_extend_v011/recovery_001_20261016_0232`，包含原状态/失败/日志快照、真实响应和收据、15测试结果、恢复与启动证据及完整文档before/after。只有下载与校验，fit/predict=0。

## 最新：两站扩时段与小时雨量下载已启动（2026-10-16 01:20 +08:00）

用户明确扩充现有两站原10参数日期及雨量，直到叫停；本轮仅采集，不进行新实验。012和T012最新段落覆盖011的39日限制，保留原39日全部冻结数据/评价及LOCAL，不恢复旧218链路计划。唯一新入口`weather_download_gbs_extend_v011.py run/status`，prepare已一次性完成，输出`external/weather_gbs_extend_v011`。

Chilbolton原生589日期/14136小时（2003-08-19—2005-03-31）；Sparsholt 538日期/12912小时（2003-10-08—2005-03-31），合计27048站点小时，不填缺日。以现有两站原生档案决定范围，先补39日降水，再时间顺序扩充。新增NCAR MTPR与MSR均为小时平均水当量率，保留原始率；valid=t+1h对齐信标[t,t+1h)，不当累计量差分/日最高/分钟实测。总降水不等于纯液态雨，CEDA10秒雨量计实测需登录，当前未取得，不阻塞公开NCAR下载。

新58项离线测试及独立两站日历/672代表性精确请求值身份/2317旧保护SHA PASS；1127原生GBS文件长度/MD5/几何头/源SHA核验。仅启动门控，不是全量实际字段QA。PID38832创建01:18:02.259803+08:00已隐藏启动，01:18:53/56真实命令/创建/两次CPU10.59375→12.6875、active_worker与OS锁一致。01:19:22快照21新子集4068唯一格点字段小时，原39日1872站点小时雨量已采集；扩充仍在进行。完整证据launch_verification.json，冻结SHA/FP见012。fit/predict=0。

同一tvt原线程15分钟维持ACTIVE，仅健康检查/必要时安全续传本版，不重复启动。用户叫停则写STOP_REQUESTED、停止新GET、已发有界请求收尾、保留文件并暂停tvt；有限归档全部采齐则安静待命，不擅自扩站或做实验。旧模型比较调研不构成运行实验授权。人工文档交接状态见handoff，后台持共享OS锁。

## 历史：GBS分天气与冻结快变评价独立QA完成，任务继续（2026-10-15 22:20 +08:00）

用户已答复“继续核验分天气和快变评价，暂不结束 T012”，不是仅小时验收。固定011范围两站39日/1872站点小时不变，下载已齐，旧全量下载和全部训练/LOCAL重跑继续禁止。

已有ERA5诊断按冻结原规则核验，1868小时known/4小时unknown，缺8个边界降水雪值保留；无需新增诊断下载。小时主集1847不变，known快变主集1844（Chilbolton913/Sparsholt931），6637720有效原生秒；1853known有限小时另含9个低覆盖诊断小时。分天气小时误差和概率指标已独立逐秒复算，42天气/84概率汇总PASS，最大CRPS差1.78e-15 dB以下。

主日期等权组合CRPS0.4745621 dB，对应确定性小时基线0.5236461降9.37%；90%/98%区间实际覆盖44.06%/69.71%，明确不是已校准。IID/block同一边缘CDF，不把相同CRPS当动态提升。冻结block0256的日期等权时间VS0.0210320 dB²，IID0.0226282，降7.05%，但晴天/干燥阴天/other组升高。全部1853小时按32次和前16敏感性、11固定滞后计算；独立66064687合法原生样本对、168汇总和8小时/512次生成重放PASS。增量中常数慢变相消，不能称BiLSTM慢变增益或确定性秒预测。

动态v001把time=-1填充值当有效重复时间而失败，保留原版；v002只修有效时间筛选，核身份/原生观测后复用48已完成小时，余小时同规则继续，未补造秒/改原值。所有新fit/参考/CDF拟合/下载/慢变重新推理=0；原始与冻结源SHA前后保持。

权威产物：conditional_fast_001及其独立QA；conditional_evaluation_001与conditional_evaluation_independent_qa_001；dynamic_evaluation_002与dynamic_independent_qa_001；report_004/RESULTS_ZH.md及TABLES_ZH.md，图复用report_003/figures PNG/PDF。report003登记schema失败及004收据路径表达式错误均留存，修复不改数值；最新登记51陈述/171数值事实结构与一致性通过，具名人工审阅仍待完成，不称投稿已批准。报告同时展示评分降低和覆盖不足。

22:16:19具权限实查weather_download进程0，先前沙盒CIM失败空列表未采用；本轮OS锁未复查，无恢复下载。下一步仅只读盘点其他已有模型族Top10输入/目标/尺度/预处理兼容性，不把未核验说成不存在，不补训练或扩量。用户未授权结束T012；同一tvt原线程15分钟ACTIVE保持，本轮未改automation配置。稳定索引完整before/after见continuation_013，handoff_013释放人工占用，后续禁止重跑以上完成计算。

## 历史：GBS两站小时外部评价及独立QA完成（2026-10-15 21:23 +08:00）

011固定范围不变，所有旧下载入口仍禁止恢复。新结果在external/validation_gbs_48h_v010：field_audit_001独立解码161源文件/11240唯一格点字段小时，值差0、缺值0；独立WGS84复核9360压力交点、1872地面点，24项控制PASS。78原生GBS文件与冻结小时标签复核PASS，未重估参考；原经度环绕差异失败保存evaluation_001，另存evaluation_002修复表示后通过，未改坐标。

现成Top10 BiLSTM k_0010已零训练预测1872行，主评分1847小时（914/933），16小时缺标签、9小时覆盖不足95%留作不可评分/次要诊断。主汇总RMSE0.9894442、MAE0.4951477、Bias+0.4251498 dB；全部有限标签1856小时另存诊断。权重/插补/标准化/目标参数SHA保持，新fit=0、下载=0。原LOCAL经验参考−76.19362933508154 dBm与GBS原生参考不同，未调偏移；采样仍为已授权固定四层LOS最近格近似，不声称原双线性完全同定义。

evaluation_independent_qa_001独立指标复算与直接网络1872行重放PASS，最大差5.6503e-8 dB，10项控制PASS。report_002/RESULTS_ZH.md及PNG/PDF完成小时阶段展示；首版遮挡文字另存v002，未删点/改轴/变更分数，内部草稿待人工审阅。frozen hourly_metrics文件的PENDING是先前快照，最新独立QA优先。

尚未完成分天气/快变概率及其他模型族兼容核验；已询问用户是否仅以该小时验证作为最低外部验收，未获答复不关闭T012或暂停tvt。下一轮先读handoff_012与本轮新产物，禁止重跑已完成审计/推理或任何旧全量下载。人工写占用交接RELEASED、完整before/after在continuation_012；同一tvt/原线程/15分钟ACTIVE保持。

## 最新：两天期限缩量，GBS两站所需气象已齐（2026-10-15 20:42 +08:00）

按用户允许减少站点/外部数据集、两天内完成下载的最新要求，当前外部验收仅GBS England两站chilbolton/sparsholt、39原日期（2004-02八日＋03整月）、936小时/站，共1872站点小时。全部最早已完整采集单元在预测前冻结，未按成绩/天气筛选。新权威为011和external/validation_gbs_48h_v010/scope_manifest.json；Dundee/OpenSat/其他月份不再扩量，旧完整计划不计为完成，全部文件保留。

20:39:39旧唯一v009 PID31720精确命令/创建/两次CPU核验后停止；真实weather_download进程0、共享OS锁释放。旧v009状态及所有文件完整快照逐项SHA核验，不改旧RUNNING历史。禁止恢复任何旧全量下载run，包括v009；不重跑initialize。两天下载目标已满足，不再等待原全量计划。

20:42:06本地可用性PASS：1872/1872十列齐全站点小时；18720站点字段小时；去重11240网格字段小时＝旧CDS7680＋NCAR3560，161实际来源SHA核验、缺值0、新下载需求0。表与清单保存新目录。此检查复用原解码/几何实现，独立实际字段QA、外部标签/冻结模型兼容、推理/指标/图表/中文报告仍未完成；fit=0/predict=0。Top10/无辅助位势/固定四层LOS最近格/零训练及LOCAL只读不变。同一tvt每15分钟转入小范围验证，旧全量恢复授权撤销。

20:47交接：8项独立范围/计数/身份绑定/实际来源SHA/旧数据保持测试PASS；完整实际字段独立解码和模型兼容仍待执行。原tvt最新011提示逐字核验PASS，ACTIVE/原线程/15分钟不变；20:47:25实际天气下载进程0、共享锁释放、旧冻结计划及保护SHA不变。人工范围冻结/交接锁RELEASED，下载不再是后续前置等待；进入小范围独立QA及零训练评价。

## 历史：全部后续下载切换NCAR，v009已启动（2026-10-15 17:50 +08:00）

依据用户“全部改为NCAR下载”，唯一正式入口更新为010/weather_download_top10_ncar_v009.py run/status，新输出external/weather_top10_ncar_v009。停止旧CDS本地调度，保留所有远端任务及旧文件，不再按旧runbook恢复任何CDS版；initialize不重跑。Top10、站星视线最近0.25度发布格、四固定高度与1hPa使用Top10 z、原日历/链路/零训练不变，禁止辅助z或其他诊断和空间/时间插值。

旧CDS PID25488精确核验后于17:26:08前停止。首次NCAR v008实际取回5个子集/120唯一格点字段小时/6459响应字节后，Windows状态文件并发共享冲突导致退出；二次错误mapping问题及完整日志/状态/载荷均原样保留。另存v009修复本地状态同步及错误记录，未修改冻结v008。41离线/并发/恢复测试、独立原日历/2840代表性精确子请求/9431唯一格点字段小时展开、390旧保护SHA和5份迁移响应解码PASS。仅计划/迁移/完整性门控，不是完整字段或外部预测QA。

v009 plan FP=9e799f019b833745edd8f103ea6e183c154590604eeb7c60ba0f3137970c7b61，代码SHA=5b13a126108a92c3a064ad1b07ef1eaa67afbe3ebbe88dfbe3e3138b08fa3563。新PID31720/创建17:47:52.558971已隐藏启动；当前存活/CPU/锁及实际完成数见launch_verification、handoff和实时status。首波剩19个NCAR日文件子集/456格点字段小时，旧120值只读复用不重下。最多2个公开GET，仅所需字段/网格/时刻，不拉全球文件；不再受旧CDS排队槽阻塞，不保证全量ETA。

全部23旧CDS载荷、5旧NCAR子集、失败及旧QA/模型/LOCAL保持。FRA-00289和FRA-00609原缺口不变，无fit/predict。同一tvt原线程15分钟继续，仅更新路线和恢复规则，不另建任务。完整before/after和交接核验保留新目录；下载切换不是T012完成。

17:53交接核验：唯一31720的真实命令/创建/RUNNING/OS锁及两次CPU3.71875→3.71875匹配。17:53:27累计NCAR15子集360唯一格点字段小时，其中v009新10子集240值、v008复用120值；2个GET活动，无新失败、stderr空，非360小时完整Top10。原tvt最终010提示逐字核验PASS，ACTIVE/原线程/15分钟不变。人工代码/文档/审计占用RELEASED，后台OS锁仍持有，不重复启动。

## 历史：仅Top10，四层固定高度近似版已启动（2026-10-15 16:50 +08:00）

用户拒绝新增4列辅助位势，并确认固定标准大气层高近似。已另存fixedheight v007与008：5/500/650/850hPa固定层高沿原站星射线取最近发布格，1hPa z自身为Top10仍保留。新请求不含辅助z，不插值、不改成地面垂直柱；固定高度不是逐小时实际等压面。旧唯一25784已于16:23:42精确核验后停止，本地进程0/OS锁释放，远端ID保留未取消。旧v006与全部历史记录不改；同一tvt原线程15分钟保持ACTIVE，禁止恢复任何旧run。零训练/LOCAL只读保持。完整before与交接占用见external/weather_top10_fixedheight_v007/。

16:40一次性initialize完成；16:41冻结预检81项测试及独立计划审计PASS，核验218个1hPa初始射线、871个有效固定层交点、1204几何来源SHA、299旧保护文件SHA和精确样本打包。FRA-00289的固定850hPa层低于天线，保留1个无效层位置；不补零、不抬高。FRA-00609方向待复核不变。23份旧载荷只读候选复用。旧z@5 ID dbac99a1-b428-417e-9d26-5442ca99679d改为仅查状态，不下载载荷、不重提；未明确终态仍占槽。

唯一正式入口008/weather_download_top10_fixedheight_v007.py run/status；计划FP=adec64ebf2cfefc45fe3e1a153daeaf784a31680af64783a3a86e56cdfa06a73，代码SHA=50398dc98db8c623e003f2d8dccdf65ef19dab9d1a5392b934dbb3470c516d68。这是新空间近似的计划/迁移/离线QA，不是完整字段QA或外部预测。

16:45:45.3217522已隐藏启动PID25488，真实命令/创建时间/两次CPU0.59375→1.484375/RUNNING/共享OS锁一致，stderr空。16:48旧辅助poll-only ID与新u@1请求bbcb2544-3c0a-46a2-b8de-1714bf9e3926均accepted，占两槽；首波仅补u@1及o3/u@5各192小时，共576格点字段小时，后一请求未提交。首个surface单元复用旧值；新完整Top10与字段独立QA仍未完成。无4个辅助z新提交/下载。启动快照见launch_verification/handoff，未来必须重查真实存活。同一tvt按008更新并逐字核验ACTIVE/原线程/15分钟；人工代码/文档/审计锁释放，后台下载继续持有OS锁。before/after与交接核验留存，旧文件SHA保持、训练/预测=0，T012整体未完成。

## 历史：视线最近发布网格减量版已启动（2026-10-15 12:04 +08:00）

用户已明确接受最近网格近似、优先减量。正式版本为weather_download_top10_nearest_v006.py，唯一指引CONTINUATION_RUNBOOK_007.md，替代旧006宽包络/双线性；v005预检失败留存，不能运行。自然Top10、原77父项/45663小时、33压力月/22824小时、站点和零训练不变。原模型8压力特征对+必要几何z4对=12对，取代旧4×5=20；surface仍仅两列。按实际z逐小时沿站星射线定位最近0.25度发布格，目标值直接下载，既不空间/时间插值也不退回垂直柱；高度猜测只作起点，不作为实际等压面。

64离线测试和独立计划审计PASS（1090初始射线点、1204几何来源SHA、266旧保护文件SHA、精确变量/层/网格/日期/时刻打包），不是完整实际字段/交点/模型QA。首轮v005发现部分高站点超过850hPa初始猜测，原失败/代码/计划保留；v006初始值限定在天线以上，实际低于天线/不收敛/循环保留无效。模型输入采样变化已由用户接受，但后续结果仍标注最近发布格近似，不宣称与原双线性输入完全相同。

旧31160精确核验后已于11:44:31前停止本地调度，远端未取消，7份交接副本与原件SHA一致。新PID25784于12:00:26.505335隐藏启动，12:01真实命令/创建时间/两次CPU3.515625→3.59375/RUNNING/OS锁一致，stderr空。两原ID2b658003-b255-4eee-817e-3de5f05bb899、5473a85d-0cd4-4886-ad0c-702c5ed29e0b均accepted、占两槽，尚无新POST。首个England February surface单元复用历史值；wave0001仅需z@1[50.5,-2]和z@5[50.5,-1.75]各192原UTC小时，2单点请求/384字段格点小时待槽位。20历史文件只按所需值候选复用，不整份计新覆盖。units=1只是采集记录，不是Top10完成。

plan FP=a3de9e41ce10dec6db7499423dcc1b030dd3f02ace19b613ce841986d404e5bd；代码SHA=221215512e7e7497dda4fe79199b3eb03491a64a610ec853b1e08f0058525f72。全域静态未扣缓存估计初始z505、surface128请求，后续纠正和目标字段另加；严格点级减量可能增加排队次数，不保证更快。FRA-00609仍待方向核验。旧全部结果/QA/代码/LOCAL只读，fit=0、predict=0。原tvt已按007更新并逐字核验，ACTIVE/原线程/15分钟；人工代码/文档/审计锁释放，后台worker继续独占共享OS锁。after快照与交接核验存v006；整项T012未完成，不暂停。

## 历史：恢复Top10并修正LOS空间支持（2026-10-15 11:04 +08:00）

按用户最新人工要求，当前优先006，旧Top1/005成为历史；零训练保持。旧英国单列/单点网格无法支持原LOS双线性插值，不计新方案覆盖。新weather_top10_los_v004使用各站—卫星射线的包络补下载Top10，压力面实际取值仍需逐小时位势面求交，非站点垂直剖面。地面两列仍在站点取值；z@各选定压力层是必要几何辅助，不新增模型输入、不扩云雨雪诊断。

新计划保留原33压力月/22824小时、地面日期/时刻/补时，总77日历父项/45663请求小时，新201批+2继承ID。54离线测试及独立计划/迁移/包络审计PASS：897243密集射线点、1204几何来源/203历史文件SHA核对。GBS1197原文件几何一致；OpenSat215条用发布指向，FRA-00609标称方向偏差0.5424°仍保留待复核，未放宽门槛或改坐标，其预测不视为兼容；其他214通过该方向核验。上述PASS不代表完整字段、真实交点、目标或外部预测QA。

已有Top10 k_0010检查点SHA=b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5，seed17/53轮、原生268结构槽/10输入与冻结预处理核验PASS，fit=0、predict=0；没有补零改网络。首次几何标签枚举、原生布局断言和一条指向检查失败均另存preflight_history，新版修复不覆盖旧源。

旧Top1 PID24648于10:50:19按用户新范围停止本地调度，全部旧文件与23份交接副本保持；远端未取消。新PID31160（11:02:39.819062）隐藏运行唯一v004 run，创建时间/命令/两次CPU/OS锁/RUNNING匹配、stderr空。November继承请求已下载81811字节，不计LOS覆盖；December原ID仍accepted；一槽已自动提交新England February地面两列ID2b658003-b255-4eee-817e-3de5f05bb899，accepted。新LOS完成0/77，201新批中1已提交、200未提交。低CPU/排队不判失败。

计划FP=373cd26b98e2250487c1ab24ce3cafa92024f783ddf81dfe73c4ff8c53ece05b；代码SHA=b2ddb189f7a3b9e406e7f4ce487a73f04977e523b56cf726b5a79329dfd1fdd1。详细入口、hash、恢复规则见CONTINUATION_RUNBOOK_006.md；旧v001/v002/Top1不再恢复，新prepare不重跑。此前Top1的5976小时/14文件仍原样历史保留。LOCAL和旧结果不重跑，完整外部评价未完成，同一tvt原线程15分钟继续。

## 主机重启恢复及August—October增量QA（2026-10-15 09:22 +08:00）

凌晨主机02:58:58.5重启，本地worker3164随之中断；09:12核实实际下载进程均无、OS锁释放，旧RUNNING仅历史。原请求August/September经官方只读核对均successful；冻结指纹一致，无failure/partial/提交或访问歧义后，09:16:49.677323隐藏恢复唯一Top1 run，真实PID24648。沿用两个原ID、不重下原11文件、不运行旧Top10；旧状态/日志与三份固定入口before/after完整保留在recovery_002_20261015_0916/。

原August/September分别09:16:57、09:17:02完成，新增696/720小时，实际/官方大小81086/81802字节一致；随后仅提交单字段请求，October原ID abffde50-137c-487e-b815-c69d0697d828于09:18:52完成744小时、82468字节。新field_audit_008/009与各自独立readback均PASS：UTC、完整坐标、1hPa层、位势身份/单位、数量、有限值、官方大小及SHA通过。008有12字段测试/22记录负控制/13独立提取/98来源SHA；009有12字段测试/11记录负控制/14独立提取/103来源SHA。原审计行不变，August缺8月3/4日按冻结计划保留，不补造。

累计14文件、5976/22824请求小时（26.18%）、11952网格值、9/33计划地区/月完整；余16848小时未字段QA。v009代码SHA256=1679ce3b13c01cecb9f39cdc0fa48460ccf04ee3d5118239824be98bef95a565；field_audit_009/qa.json SHA256=0f880ca4c561ab900c3b02547f258d3defb2e8af637e064e4a3bd4b92476edaa；delivery_readback_qa.json SHA256=550039620dde4269a7c6326d7d37939b7f97e96314bf8ba9937a9735477d3003。008/009不可覆盖，后续另存版本；只读readback可重放。

09:21:23唯一实际worker24648的命令/创建时间/两次CPU0.78125→0.78125秒/RUNNING/OS锁匹配；stderr空，无failure/partial。November ID cbe3263c-c3ad-4573-95eb-00774a965430和December ID 5473a85d-0cd4-4886-ad0c-702c5ed29e0b均accepted，占两槽，新单字段verified=7，余22项未提交。日志为weather_top1_run_005_20261015_091649.stdout.log/stderr.log；旧003/004日志未变。完整健康与恢复证据见recovery_002_20261015_0916/。本轮文档/离线审计锁释放，下载worker锁继续持有；未来须重新核查真实存活，不能依赖这段历史PID。

模型输入空间定义仍BLOCKED_SPATIAL_INPUT_DEFINITION，未收到最近网格近似/补视线范围的明确选择。“继续”只推进已授权下载及QA，不视作替换采样授权；不预测、不扩下载、不补天气诊断。新fit=0、模型预测=0、LOCAL与旧模型/参数/CDF不变；完整Top1字段QA、旧Top10及整项完成均false。同一tvt继续15分钟，不暂停、不另建任务。

## July单字段下载与增量QA完成（2026-10-15 02:40 +08:00）

GBS England 2004年7月Top1原ID cc915b69-3eec-4920-b0d7-97a95bbe4d5d于02:29:42.949660完成，81108字节与官方一致。按原计划29日期新增696请求小时，7月18日、21日不在冻结范围，不补造。累计11文件/3816 of 22824小时（16.72%）、7632个z@1hPa网格值、6/33计划地区/月精确覆盖。

field_audit_007/qa.json为PASS_INCREMENTAL_TOP1_FIELDS_ONLY；完整UTC/坐标/层/身份/单位/数量/有限值/大小及SHA通过。12字段正反测试、11记录负控制、11独立提取和91来源前后指纹通过；独立PowerShell回读枚举全部696小时并保留日期缺口，原10行不变。qa SHA256=015ac6e9eefcc169315460b9905ccf7dbf39effbcbcca2398543c7b343b4b19a；回读SHA256=b8417edc51c08b28697b025ff1c25217e1b46febbe915ca027f431a002e878fb。新代码v007与前后文档快照另存，不覆盖旧QA。

02:39:26实际worker3164，命令/创建时间/共享锁/RUNNING一致，CPU18.296875→18.296875秒、stderr空，无failure/partial。August d9fc779a-7f4a-4e51-89a9-2e3a81d9ba18与September 5505f067-13da-4708-a3e4-c700523864fe均accepted；新单字段verified=4、另25未提交。继续健康只读，不重复启动；下轮重读实际状态。

余19008小时待字段QA，空间输入兼容性仍BLOCKED_SPATIAL_INPUT_DEFINITION，原待决选择无人工答复；不重复问、不扩网格、不擅自近邻替代。外部预测=0、新fit=0，LOCAL及旧模型/预处理/CDF/结果不变；完整Top1 QA与总completion均false，同一tvt保持15分钟。

## June单字段下载与增量QA完成（2026-10-15 00:41 +08:00）

GBS England 2004年6月Top1原ID 3a270adb-0a66-4b5e-bbc9-000b3dc259a9于00:26:59.543474下载完成，81798字节与官方大小一致。新增720请求小时，累计10文件/3120 of 22824小时（13.67%）、6240个z@1hPa网格值、5/33完整地区/月；不能据此称原Top10或完整外部实验完成。

field_audit_006/qa.json为PASS_INCREMENTAL_TOP1_FIELDS_ONLY，UTC/坐标/1hPa/位势身份/单位/数量/有限值及SHA通过。12字段正反测试、11记录负控制、10文件独立提取与86来源前后指纹通过；独立PowerShell回读枚举全部新增720个UTC，原9行审计保持不变。qa SHA256=f47ba0aa95f9858033f3ab9d9114a78cea70acc376171c5f343338915ce04b7e；readback SHA256=f35c2405fa239d93a00e85e23dceebc6a2f48ae4dbea3df1a1fc759e5374ba86。新审计v006及交接快照另存，未覆盖旧QA。

00:41:02真实下载worker3164、Top1命令/创建时间/OS锁/RUNNING一致，两次CPU12.015625→12.046875秒、stderr空、无failure/partial。July cc915b69-3eec-4920-b0d7-97a95bbe4d5d和August d9fc779a-7f4a-4e51-89a9-2e3a81d9ba18均accepted；新单字段verified=3、另26未提交。存活只读、不重复启动；下次须重读实际状态。

余19704小时未做字段QA。空间定义兼容性仍BLOCKED_SPATIAL_INPUT_DEFINITION，原待决选择未答复；不擅自近邻替代/扩网格，不重复询问。外部预测=0、新fit=0，LOCAL与旧模型/预处理/CDF/结果不变；完整Top1 QA和总completion均false，同一tvt继续15分钟。

## May单字段下载与增量QA完成（2026-10-14 22:44 +08:00）

May Top1原ID 97e01216-7578-42a7-a72a-2a3e6a8bf899于22:30:53下载完成，新增744小时。field_audit_005/qa.json与独立回读PASS：累计9文件、2400/22824请求小时（10.52%）、4800个z@1hPa网格值、4/33完整地区/月。UTC/坐标/层/单位/身份/数量/有限值/SHA、12字段测试、11记录负控制、9文件独立提取及81来源前后指纹均通过，原8行审计不变；旧QA不覆盖。代码SHA=b3520520870783ae700c31d5806423f734e21bfcaec1dd929bf1a4ebdf5bca9b，qa SHA=894494a1c1fa12a0a92f7c209b7314255463d202f546bbb96b31889f1757a243，回读SHA=d8892bd3222b9d78cd68262c45c6a4b44d875f37d4a910f68e787701f75c3a41。

22:43真实worker3164及共享OS锁健康，两次CPU4.84375→4.84375秒、命令/创建时间匹配；June/July两项accepted，新单字段已完成2，余27项未提交，stderr空且无failure/partial。新请求仍只geopotential/1hPa，不重启健康worker，不恢复旧Top10。完整健康证据、文档before/after快照及交接QA另存field_audit_005，T012维护本轮占用与释放。

这是局部字段QA，不是外部验证完成：余20424请求小时未核验，空间输入不兼容及用户待决选择不变。外部预测/新fit均0，不扩网格、不补诊断、不改模型、预处理、CDF或LOCAL。完整Top1字段QA与整项completion仍false，同一tvt继续15分钟。

## 重启恢复与首批单字段QA（2026-10-14 21:46 +08:00）

19:00—19:02系统重启导致原Top1 worker中断，21:33真实下载进程已无且OS锁释放；旧active_worker的RUNNING只作历史记录。只读核对官方原ID及9冻结/57保护/71已有字段审计来源后，21:40:10.256955隐藏恢复同一Top1入口，PID3164，另名日志weather_top1_run_004_20261014_214010.stdout.log/stderr.log。实际命令/创建时间、两次CPU0.484375→0.53125秒及共享锁匹配，不重复POST、不重跑prepare/旧Top10。完整诊断/恢复前状态/真实启动证据见external/weather_top1_v003/recovery_001_20261014_2138/；系统异常关机和计划升级重启事件均保留，不虚构精确退出时刻。

原April top1 ID4552a271-a060-41b5-a6cc-a76f11b2cdb6于21:40:18落盘，新增552小时（2004-04-08至30），77022字节与官方预期相同；SHA256=da77b866564af46f17de26ce69a7365e6e4437dda09b2c48d9977fa13c04ad07。释放一个槽后冻结下载器自动提交June top1 ID3a270adb-0a66-4b5e-bbc9-000b3dc259a9，与原May ID97e01216-7578-42a7-a72a-2a3e6a8bf899两项accepted；21:45复查worker与锁健康，stderr空、无failure/partial。新单字段完成1，另28请求未提交。

另存field_audit_004/qa.json，PASS_INCREMENTAL_TOP1_FIELDS_ONLY：累计8文件、1656/22824请求小时、3312个z@1hPa网格值、3/33完整地区/月。新增552小时UTC/坐标/单层/位势身份/单位/数量/有限值及SHA通过，12字段测试、11记录负控制、8文件独立数值/Unix秒UTC提取和76来源前后SHA通过；独立PowerShell回读PASS，原7行审计保持完全相同。新审计代码SHA256=723f2c02cd360c7508dee9f65df2315966613cdff968c3e2d95639836afa3030；qa SHA256=8ca54461c82f8bead2a4a32f2c0609aa12f6b66417f2bca21720f880cb1e69cc；delivery_readback_qa SHA256=1da4ee70fcfe5d1ce0e246afabdf1e93d3b92322ca72c318f181470c7ff345cd。旧审计只读，完整字段QA仍需余下21168小时。

空间输入兼容性仍BLOCKED_SPATIAL_INPUT_DEFINITION，未收到近邻近似/扩视线网格的人工选择；不推理、不训练、不补诊断、不扩空间。模型/预处理/CDF和LOCAL不变；新fit=0、预测=0、整项completion=false。T012记录恢复与审计占用、验证及释放；三份固定入口的完整before/after快照同轮保存于恢复证据目录，后台下载占用仍由worker3164持有。同一tvt继续15分钟，待决问题不重复通知。

## April继承请求完成与字段QA（2026-10-14 18:28 +08:00）

18:14:47，April b001原ID ba565558-5cf8-4c6e-b298-45c0a235eb3f完成2004-04-01至07的168小时下载，84676字节/官方预期/SHA一致。两个继承ID现均完成；健康worker32908自动提交May top1（97e01216-7578-42a7-a72a-2a3e6a8bf899），与April top1（4552a271-a060-41b5-a6cc-a76f11b2cdb6）两项单字段请求accepted；未提交29，新单字段完成0，最多两槽。未重启或修改冻结下载器/计划。

新audit_top1_incremental_fields_v003.py与field_audit_003/qa.json实际运行PASS；原两轮字段QA只读保留。新增168小时/336网格值，累计7文件/1104请求小时/2208个z@1hPa网格值、2/33完整地区/月；全部UTC/坐标/层/位势身份/单位/有限值/数量/SHA、7文件独立提取和Unix秒解码、12项字段正反测试、10项记录负控制及71来源指纹均PASS。qa SHA256=d8dfdc9d0b69e2ad457ace730972ffc2c62aa152edcb6de30e6060c5bfec33c1；新审计器SHA256=3a73444b92f8a7ab9d62cfaa7dea6b4deaa1c2a737eda4926f4712aac161bf17；独立交接回读证据在同目录delivery_readback_qa.json。

首次独立PowerShell回读因自动DateTime解析与字符串比较失败，诊断另存readback_attempt_001.json；显式-DateKind String后168个UTC逐项核对通过，字段QA/载荷未改，只读重放脚本readback_check.ps1已实际通过。固定入口before/after快照另存同轮document_snapshots，T012记录范围、占用及释放。

剩余21720请求小时未核验，完整Top1字段QA仍false；空间输入兼容性门槛未解除，不扩网格、不替换采样、不生成外部预测/指标，新fit=0。LOCAL和全部旧结果不变；既有待决问题不重复通知，同一tvt继续ACTIVE，未写总completion。

## 增量Top1字段QA完成（2026-10-14 16:14 +08:00）

继承March b005原ID于15:55完成（2004-03-29至31，72请求小时）；下载器随后自动提交首个单字段April top1请求4552a271-a060-41b5-a6cc-a76f11b2cdb6，与继承April b001继续占两槽等待。实际worker32908正常，未重启、未改冻结下载器/计划。累计下载覆盖936/22824请求小时、2/33个计划地区/月；5复用文件+1继承完成，新单字段完成0，其余新请求尚有30未提交。

另存field_audit_002/qa.json（PASS_INCREMENTAL_TOP1_FIELDS_ONLY）：原864小时字段QA只读保留，加上新72小时全部UTC/坐标/1hPa/位势身份/单位/数量/有限值及SHA通过；6文件共1872网格值。独立重新提取数值/Unix秒解码UTC、12项字段正反检查、6项记录负控制、66来源SHA和readback核验均PASS。qa SHA256=e3654b8184c7c6c20b4123641a51b2f382fd503a38272895026b636f5d1ae87f；代码audit_top1_incremental_fields_v002.py SHA256=b827855dcfd625c247a22afaafee5adf21be550d5f6a620e94c53a918698250f。固定进度入口的前后副本保留在field_audit_002/document_snapshots/；T012记录本轮最小占用及交接。

仅部分字段QA完成；完整Top1字段QA仍false，不能与原Top10完成混算。原空间兼容性缺口和用户待决选择不变，外部预测/评分/新fit均0。未补天气诊断、未改变空间采样、未拟合或修改模型/预处理/CDF；LOCAL不变。安全下载继续，同一待决问题不重复通知，tvt不暂停。

## 最新检查：已有Top1字段QA通过，空间输入需用户决定（2026-10-14 15:45 +08:00）

另存 audit_top1_available_fields_v001.py、audit_top1_input_compatibility_v001.py 及独立复核器/12项正反测试。新 field_audit_001/qa.json 仅覆盖 plan.reused 的5个正式压力载荷：864请求小时、1728个z@1hPa网格值，UTC/完整坐标/压力层/单位/身份/数量/有限值/SHA全部PASS。没有转换位势为高度，没有插值，没有smoke混算；不是完整22824小时字段QA。独立重新提取数值及37个来源指纹核验PASS。

兼容性结果 input_compatibility_001.json 为 BLOCKED_SPATIAL_INPUT_DEFINITION：原模型实际train2404/validation343/test687小时的p1__geopotential与本地动态卫星视线交点的双线性采样CSV逐行完全相同；不是站点最近网格特征。现有英国载荷只有latitude=[51.25,51.0]、longitude=[-1.5]，不足以按原规则构造视线特征。独立抽取原纯插值函数，2×2控制样例通过，单经度样例返回NaN；没有运行旧构建入口或修改旧代码。

已有英国目标可与这864天气小时对齐：Chilbolton原生有效848/高覆盖842小时，Sparsholt864/861小时，合计1703高覆盖目标小时；这不是1703个合法模型输入或预测。Dundee/OpenSat本轮可用Top1天气仍为0。GBS经纬度按原units处理：英国两站为degrees_east，Dundee及satellite_position为degrees_west；没有擅自更改下载地区。OpenSat分钟平均域、原经验晴空目标和快变/天气分组兼容性仍待核验。

已向用户请求选择：显式允许另标注“最近网格近似”的零训练试跑，或仍只补1个字段、另行批准视线所需空间网格。尚无新决定，不能自行替换特征、扩下载或用已有插补器掩盖整列缺失。模型加载/新fit/外部预测均为0，已有模型及LOCAL结果未变。

15:43—15:45：唯一实际下载进程仍PID32908（15:25:32.272867创建），Top1精确命令/两次CPU1.0625→1.0625秒、共享OS锁和RUNNING匹配，stderr空；两个继承ID仍accepted，不重启/不新增第三个请求。安全Top1下载继续；空间缺口只阻止未获授权的试跑，不暂停tvt或标记整项完成。后续优先读005新兼容性节，不反复重跑/覆盖上述不可变审计。

## 最新：RFE最终单特征先行（2026-10-14 15:26 +08:00）

用户明确因下载慢改为只下载RFE最后一列先试；本节及CONTINUATION_RUNBOOK_005.md覆盖下方历史Top10下载优先和旧恢复入口，零训练保持。末列为p1__geopotential（1hPa位势）。已有Rev029 rfe/models/k_0001.joblib及其已有中位数/标准化、目标变换SHA核验通过，没有fit。

已停止旧调度PID39924，旧文件/日志/远端请求全部保留。新weather_top1_v003计划：原33个地区/月、22824请求小时，复用5压力载荷（864小时），继承两个已accepted远端ID，剩余31个新请求均仅geopotential/1hPa。新增天气诊断字段为0，不能用本次单列下载冒充完整分天气/快变外部验证。

37离线检查及独立迁移/日期/字段/60来源指纹审计PASS；首轮smoke枚举失败另存审计，未改旧产物。新plan fingerprint=b80606aa0b5619ce8a447f4dabcc6cfabe6ca5aeb17b95c0a40451545fe70500。15:25:32.272867隐藏启动PID32908，真实命令、创建时间、CPU0.25→0.28125秒、共享OS锁、新活动记录和两个官方请求ID已匹配。日志external/weather_top1_v003/weather_top1_run_003_20261014_152532.stdout.log/stderr.log，stderr空。Top1后续健康监控/主动恢复仅使用新入口，禁止恢复旧v001/v002。LOCAL与所有旧研究结果不变，尚未产生外部预测或评价，整项T012未完成。

2026-10-13：已授权并完成冻结来源/版本/现有worker检查；新目录隔离建立。LOCAL实现与纯合成QA准备中，尚未产生本轮组合结果。GBS/OpenSat原始副本及既有小时标签已找到，不重复下载原始测量。

- 本地：Rev028 full156 seed17/53轮为主，原base50 seed17/46轮保留；11组旧预测，只做新的组合评价。
- 外部：Top10排名已只读核对；完整对应气象未齐备。新域少列模型/分钟生成器另存，不修改原156列模型和秒级快变包。
- 原始测量、旧Revision和全部冻结结果只读。
- 首次纯合成17项PASS，但prepare因新入口漏写clean_1s子目录而在创建manifest/研究评分前失败。完整初版代码/计划另存Preflight_001，数值QA原样归档local/preflight_history/001，failure保留。仅纠正新入口的源路径，旧代码/数据未变；重新合成检查后再冻结。
- 当前尚未启动外部拟合、尚无外部测试成绩；整项任务未完成。

## LOCAL数值评价与报告审计完成（2026-10-13 14:57 +08:00）

- 冻结指纹：2f7dbdda002880e885baf7e459ab042c989703154818f4f0ca6df8c59fb42cc9；855来源文件SHA256核对通过。
- worker PID9992，2026-10-13 14:51:36隐藏启动，14:57:33正常退出；OS锁释放，local_001.stdout.log/stderr保留，stderr为空。
- 17纯合成检查PASS；11个冻结seed17模型对3集合共37774预测重放PASS；本地新增fit=0。
- 37检查点、1030正式小时、11330组合记录；79310标量及11330CDF曲线独立核验PASS；63预定源小时×11模型逐点CRPS见证PASS，最大误差1.07e-13 dB。
- 既有65920随机实现复用，全部VS复算最大误差4.44e-16 dB²；旧动态/窗口/100ms不重跑、不更改。
- 报告：165慢变行、2310概率/区间行、630配对行、22110CDF校准行；2940个日期块区间独立复算PASS。RESULTS_ZH.md及report_qa.json保存。
- 主BiLSTM测试总体：确定性瞬时CRPS0.162335→组合0.130487 dB，差异19.62%；中央90/98覆盖73.98%/94.10%，没有改分布来补足覆盖。
- 该完成仅指LOCAL数值/报告阶段；调用示例和新图表继续另存，GBS/OpenSat未完成，不写总completion。
- 用户已明确恢复原tvt每15分钟检查；原id保留，状态ACTIVE，新任务范围已更新，不另建重复自动化。

## LOCAL完整交付与外部气象预检（2026-10-13 15:49 +08:00）

- 调用入口callable_local.py：8个固定最早示例原慢变/快变流完全复现；120个训练来源块独立核查PASS。首次单样本float32批量舍入差另存审计；仅改新调用的数值封装，不改原权重、训练或旧指标。
- 4组PNG/PDF已生成并视觉QA PASS；audit_local_delivery.py复核855源文件与全部新图表/示例/报告，delivery_qa.json PASS，最大组合恒等式误差4.44e-16 dB。本地阶段完整交付，HANDOFF_ZH.md另存；总任务未完成。
- external_prepare.py已仅依元数据固定Top10、GBS399日期/3站和OpenSat172拟合站/43未见站及月分区。metadata/weather_plan另存，77正式请求，计划指纹c7fa6ff824fb436477f534fcec022ef7d60aed6afd4ae507f9dd1575e0282e16；中文路径NetCDF兼容性预检失败已保留，不更改源文件。
- 气象smoke worker PID40032，于15:24:07隐藏启动。精确命令python -B Code\Combined_Channel_Experiments\Revision_001\weather_download.py smoke；日志external/weather_smoke_001.stdout.log/stderr，OS锁存活。
- 截至15:49：压力预检VERIFIED，地面请求58dd28f0-1788-464a-8237-292f17dff64f服务端accepted，正在等待。1/2下载已完成，完整字段/单位QA尚未通过，因此不启动77项正式下载或外部训练。任何同worker存活只读检查，不重新提交。
- 下一步：正常退出及smoke下载完成→verify_weather.py独立字段/UTC/单位QA→隐藏weather_download.py run（已核验smoke将复用）→全部源气象QA→外部训练协议/参数/合成数值QA冻结→GBS/OpenSat有界模型比较。天气积算最后小时所需次日00 UTC含日期缺口已预先登记，分钟域不造秒级真值。

## 气象预检完成、正式下载启动（2026-10-13 16:58 +08:00）

- 原smoke PID40032于16:35:48正常结束，两项载荷VERIFIED，锁已释放，原日志保留。先前1/2等待状态已结束。
- 首次独立单位核验失败：verify_weather.py仅接受sf单位m，而官方载荷为m of water equivalent，standard_name为lwe_thickness_of_snowfall_amount。ECMWF官方字段表核对后，另存verify_weather_v002.py，不修改原核验器、冻结下载脚本、请求计划或任何载荷。失败记录保留external/preflight_history/002_smoke_unit_failure.json。
- v002的2项正确单位/身份接受与3项错误单位/身份拒绝检查PASS；两个载荷共7字段、28网格值的字段/UTC/坐标/压力层/单位/非缺失/指纹检查PASS，weather_smoke_unit_qa.json已保存。没有数值单位转换，没有读取信道实测，没有新拟合。
- v002 SHA256：4878bca90615e004ff083e9715ef34ccd597e7d09b8c6a16f1432182be041c53；单位QA SHA256：74972e0e631f420e8963a0208789210c31ed0b3ac3e8d870710d7c7ef2a0360e。
- 冻结run入口会复核并跳过两个smoke载荷，同时刷新smoke QA的completed_utc。原被审计的QA已另存不可变输入快照external/preflight_history/002_unit_audit/smoke_download_input.json；单位QA引用该快照指纹。更新时间变化不表示重新下载，载荷及其SHA仍必须完全一致。
- 确认不存在任何weather_download.py进程、OS锁释放及上述QA/代码/计划/载荷SHA一致后，16:56:54隐藏启动正式worker PID38056。精确命令：C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download.py run。新日志external/weather_run_001.stdout.log/stderr，未覆盖旧日志。
- 16:57:49核查：实际命令/PID/创建时间匹配，active_worker为RUNNING且OS锁持有；3秒CPU两次均1.53125秒，为远端等待，stderr为空。77正式请求中2项已提交、0项下载完成，其余排队；status总submitted=4/verified=2包括两个已完成的smoke，不能误报为正式完成2项。
- 已提交GBS_England_2004-02_pressure，request_id=1fc5722d-0abe-4080-852f-77f3c9a4c56b；surface，request_id=9cc31575-8e77-4a40-8142-225c4547eb94。保持最多2并行，不重复提交。
- 后续使用CONTINUATION_RUNBOOK_002.md。下载完整后仍需全部字段/坐标/UTC/单位与数量QA，再冻结外部训练计划并完成数值QA；GBS/OpenSat拟合和测试均未开始。LOCAL与所有旧结果不变，T012未完成。

## 用户收紧后续比较为零训练（2026-10-13 20:32 +08:00）

- 用户最新明确：“之后的对比实验不需要训练任何模型，就是拿已有的模型进行对比。”该指令覆盖上述历史外部训练计划；不再新增训练、重训、微调、拟合或调参。只加载已有模型、权重、预处理及冻结快变模型推理/评价，不重估CDF、晴空参考或缩放参数。
- 已另存CONTINUATION_RUNBOOK_003.md并在T012登记最新执行边界；PROTOCOL、PARAMETERS、EXTERNAL_PREPARATION与两个旧runbook原样留存作历史依据，不再执行其中外部新拟合步骤。
- 20:30—20:31只读检查：本项目仅PID38056的weather_download.py run在运行；status的model_training_started=false，external未见训练模型或训练入口，只有天气下载/元数据/预检审计。下载仍持有OS锁；正式已提交4项，2/77完成，另有2/2 smoke完成。此为当时快照，后续实际进度须读取状态。
- 已有LOCAL、原基础50列46轮和全156列53轮的模型/结果不动。现有天气计划/下载源码指纹未变；气象下载可按原授权继续，下载完成不触发训练。
- 下一步改为已有模型清点与输入兼容性核对，再仅对兼容模型做推理/组合和指标评价。Top10不能直接喂给156列网络，模型/特征/目标/秒分钟尺度不兼容时如实报告，未经用户新决定不补训、不擅自扩大下载或修改实验定义。T012仍未完成。
- 原tvt已通过自动化工具同步零训练约束，并只读回查保存提示逐字一致；id、名称、原线程、每15分钟间隔和ACTIVE状态全部保留，未新建任务。新提示明确优先读取003，并删除原“下载后训练外部模型”的执行授权；冻结下载脚本、external_prepare、PROTOCOL和PARAMETERS指纹未变。

## 用户授权仅调整日期分批并恢复下载（2026-10-14 08:42 +08:00）

- 原PID38056于00:06:45因官方HTTP403请求过大退出；已确认实际进程不存在、OS锁释放、无未完成旧request_id或partial。旧failure、旧代码/计划/六个载荷及所有LOCAL/外部QA原样保留。
- 用户直接要求仅改下载分批、外部天气必须下载完成、进程退出需主动恢复。另存weather_download_v002.py、test_weather_download_v002.py与CONTINUATION_RUNBOOK_004.md，不修改任何冻结旧代码或科学范围。
- 复用4个正式文件；余下压力请求最多7个日期一批，地面初始维持原批，共196个新初始批。明确大小拒绝才继续日期二分；原77父项的所有非日期参数、日期多重集、45663请求小时和00UTC补时均精确保留。20项离线恢复测试PASS，28个原文件指纹PASS，无网络/模型训练。
- 新计划fingerprint：248cd58b1dc1fc7ae0baadcb1da39adb827daeb81b220335a5b9f363113158a0；新下载器SHA：26e2ac7205a4de17ac1d81af50c52912a30e8fbed3d9ea57583111ece7a0f2ec；所有新天气/state/failed/日志仅写external/weather_v002，共享旧字节锁但不改旧active_worker记录。
- 08:40:09.189529隐藏启动PID39924，精确命令C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download_v002.py run。实际PID/创建时间/命令、CPU0.328125→0.40625秒和RUNNING/字节锁匹配。日志weather_v002/weather_run_002_20261014_084009.stdout.log/stderr.log；stderr为空。
- 08:41：GBS_England_2004-03_pressure__b001（761638d0-8222-4d54-86bb-cfe2889ebe71）与__b002（e1a9d038-f1c0-4865-ab1a-f82d7038b625）均accepted。新批次已提交2、下载完成0；原父项仍4/77，两个smoke单列，不把请求已接收当完成。
- 新程序保存提交意图与request_id，网络失败保留ID退避重试，partial先诊断并保留后续下，未明确终止的远端状态不释放并发槽位；未知提交/账户条款/非大小远端失败保留证据再诊断。进程退出且恢复门槛满足时原tvt必须主动续启新run入口，不只报告停滞。
- LOCAL、原模型/参数/预处理/旧结果全部未变，后续零训练。完整天气下载仍需另存全量字段/UTC/坐标/单位/数量QA及外部兼容模型评价，T012未完成。
- 08:44:46独立分批审计PASS：不导入下载器，独立检查77父请求、196初始日期批、4复用载荷、28保护文件、全部非日期参数及45663请求小时。batching_independent_qa.json SHA256为37bbcaa0382ee99639dd6650821c923aa917fa8b4ad4f3c0281bf20b0b3c602b；不代表全量载荷字段QA。
- 08:45已更新同一tvt并逐字回查提示一致，原id/线程/15分钟/ACTIVE保留，新规则明确退出后诊断并主动恢复。08:45:35真实worker39924仍RUNNING、字节锁持有、CPU0.75秒、stderr空；b001远端running，b002 accepted，尚无新批verified。未写总completion，未暂停自动化。
