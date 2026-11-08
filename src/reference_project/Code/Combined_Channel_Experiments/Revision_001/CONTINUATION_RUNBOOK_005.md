# RFE 最终单特征先行，零训练不变

2026-10-14 我明确要求：“外部天气下载太慢了，可以不需要下载10个，下载RFE留下的最后一个天气试试先”。本文件及 T012 最新节优先于 004 的 Top10 全量下载/恢复规则；003 零训练约束全部保持。历史文件和旧结果不修改。

## 范围与切换

RFE 末列为 `p1__geopotential`，即 1 hPa 位势，不是天气类别。Rev029 的 `rfe/models/k_0001.joblib` 是已保存的 seed17 / 53轮 BiLSTM；只能加载其已有权重和预处理。不得用一列填补旧156列网络，也不得补训其他模型。

新代码 `weather_download_top1_v003.py`，新输出 `external/weather_top1_v003/`。新请求仅 ERA5 pressure geopotential、pressure_level=1；保持原 GBS England/Dundee 和 OpenSat France 的日期、小时和区域。按月请求剩余日期，只有明确请求过大才继续日期二分。云/雨/雪等诊断不在本次新提交中，不能因此伪造天气标签或声称完整组合/分天气验证已完成。既有诊断载荷只读可核验复用。

旧 v001/v002 不得再启动。保留并只读复用已下载压力文件，所有已提交远端 ID 纳入新计划，继续同 ID 等待/下载；不取消远端作业、不重复 POST。迁移旧请求需要的文件只写新目录；旧 plan/jobs/active_worker/logs/failures 保持切换时状态，旧 RUNNING 记录不是进程存活证据。新下载器仍共用原 OS 字节锁、最多两活跃远端 ID。

## 继续规则

本文件初始为切换声明，启动完成及指纹将追加。实际进程与新 plan/preflight/QA 未齐前不擅自启动。每次读取本文件/T012、003及新版状态、真实命令/PID/创建时间、两次CPU、锁、日志及已持久化远端ID。健康存活不重启；退出且锁释放、指纹一致、失败已诊断、无提交/访问歧义时隐藏恢复同一 Top1 新入口，另名日志。任何 SUBMITTING/AMBIGUOUS_SUBMISSION 先精确对账，访问失败不视为远端已结束，不释放未知槽位。失败/partial 保留，凭同ID续接，不能重新准备冻结计划。

## 后续评价

下载可读性通过后仍需单字段全量 UTC/坐标/层/单位/数量/有限值独立 QA。先核对现成单特征模型的SHA、输入和已有变换、目标/小时尺度及训练来源；可兼容部分先做纯推理试验，不调参、不新 fit、不重估晴空参考/CDF。GBS/OpenSat 缺兼容目标/快变模型/天气标签时如实报告；只做单特征下载不等于全部外部实验完成。已有 LOCAL 不重跑。

原 tvt 保持每15分钟，只有阶段完成/失败/需决策才通知；不能自动恢复被我本次撤销的 Top10 下载优先级。

## 最新恢复及August—October字段QA快照（2026-10-15 09:22 +08:00）

主机02:58:58.5重启中断worker3164，09:12实际下载进程均无且锁释放，旧RUNNING仅历史。官方只读核对原August ID d9fc779a-7f4a-4e51-89a9-2e3a81d9ba18和September ID 5505f067-13da-4708-a3e4-c700523864fe均successful，冻结指纹通过且无failure/partial或提交/访问歧义，09:16:49.677323隐藏恢复指定Top1 run。当前历史启动快照PID24648，active_worker.started_utc=2026-10-15T01:16:49.795954+00:00；日志weather_top1_run_005_20261015_091649.stdout.log/stderr.log。没有重下旧11文件、没有重提两原ID、没有恢复Top10；启动和诊断证据在recovery_002_20261015_0916/，旧状态/日志及固定入口完整before/after已保留。

August/September原ID分别09:16:57和09:17:02完成696/720小时，October单字段ID abffde50-137c-487e-b815-c69d0697d828随后09:18:52完成744小时。field_audit_008/009与独立readback均PASS，累计14文件/5976请求小时/11952个z@1hPa值、9/33地区/月完整。008新增1416小时、12字段测试/22记录负控制/13独立提取/98来源SHA；009新增744小时、12字段测试/11记录负控制/14独立提取/103来源SHA，UTC完整枚举、单位/坐标/层/有限值/官方大小/SHA全部通过。August缺8月3/4日保持原计划，原QA行不变，不填缺口。

v008代码SHA256=5eb94e1ca319b0ba8971f0dc4bde65c3870b4ade8f79a3fe512031e5a1b2eb4a；008 qa=af9f1624dbf8427b517290276513381a18c6e571939cd3d81e61732a4b48c61a；readback=d68b954eba205ff10e45e624eebab106285961a9ab9881bebece3502fb0f07a0。v009代码SHA256=1679ce3b13c01cecb9f39cdc0fa48460ccf04ee3d5118239824be98bef95a565；009 qa=0f880ca4c561ab900c3b02547f258d3defb2e8af637e064e4a3bd4b92476edaa；readback=550039620dde4269a7c6326d7d37939b7f97e96314bf8ba9937a9735477d3003。两版是不可覆盖的固定审计快照，后续新载荷另存新版，不运行旧main覆盖；readback_check.ps1只读可重放。

09:21:23唯一真实worker24648，命令/09:16:49.677323创建时间/RUNNING/OS锁匹配，两次CPU0.78125→0.78125秒，stderr空、无failure/partial，stdout450字节且旧日志SHA不变。当前两槽为November cbe3263c-c3ad-4573-95eb-00774a965430和December 5473a85d-0cd4-4886-ad0c-702c5ed29e0b，均accepted；新单字段verified=7、余22未提交。启动初期SUBMITTING是正常请求中的短暂快照，现已保存ID，无未解决歧义。每轮仍须重新核查全部实际进程/创建时间/两次CPU/锁/日志，不能凭本段历史PID认定存活或重启。

余16848小时未字段QA，完整Top1字段QA=false、旧Top10完成=false、整项完成=false。BLOCKED_SPATIAL_INPUT_DEFINITION仍未解决，我说“继续”并不表示最近网格近似/扩视线空间范围的选择；不重复问、不改变采样、不推理、不训练、不补诊断。LOCAL/模型/预处理/CDF只读。同一tvt维持15分钟，未暂停、未另建任务；本轮恢复/文档/离线审计占用释放，后台worker24648下载锁未释放。

## 历史July字段QA快照（2026-10-15 02:40 +08:00）

只读field_audit_007/qa.json及delivery_readback_qa.json，不重跑固定审计main。新增July 696小时/1392网格值，按冻结29日期（不含18日/21日）；累计11文件、3816/22824小时、7632个z@1hPa值、6/33计划地区/月精确覆盖。PASS_INCREMENTAL_TOP1_FIELDS_ONLY；12字段正反测试、11记录负控制、11独立提取、91来源SHA及独立回读PASS，原10行不变，余19008小时待字段QA。

qa SHA256=015ac6e9eefcc169315460b9905ccf7dbf39effbcbcca2398543c7b343b4b19a；readback SHA256=b8417edc51c08b28697b025ff1c25217e1b46febbe915ca027f431a002e878fb；audit_top1_incremental_fields_v007.py SHA256=c7ac7d088d1193c95d5a2c6feb6420f2c07af9212ed70f926214a9089ecb0d10。July原ID cc915b69-3eec-4920-b0d7-97a95bbe4d5d于02:29:42.949660完成81108字节，SHA256=5f74bda9392678e6100f086db79509026f91191e64243bd7ce7eb5cb8f5bb235。

02:39:26唯一真实PID3164（2026-10-14 21:40:10.256955创建），Top1命令/RUNNING/OS锁匹配，两次CPU18.296875→18.296875秒，stderr空、无failure/partial。August d9fc779a-7f4a-4e51-89a9-2e3a81d9ba18与September 5505f067-13da-4708-a3e4-c700523864fe均accepted；新单字段verified=4、余25未提交。此段仅历史快照；下一轮重读真实进程/锁/jobs/logs，不据本段重启或宣称持续存活。

下载器与冻结计划不改，健康只读、退出仅按既定无歧义门槛恢复同一Top1入口，最多两ID；新完成载荷QA另存版本，不运行旧Top10或prepare。空间输入仍BLOCKED_SPATIAL_INPUT_DEFINITION，未收到原待决近邻/视线网格选择；不重复询问、不推理、不扩网格/诊断，新fit=0。LOCAL和旧结果不变，局部字段QA不等于全下载或整项完成，同一tvt继续15分钟。

## 历史June字段QA快照（2026-10-15 00:41 +08:00）

只读field_audit_006/qa.json及delivery_readback_qa.json，不重跑固定审计main。新增June 720小时/1440网格值，累计10文件、3120/22824小时、6240个z@1hPa网格值、5/33完整父月；PASS_INCREMENTAL_TOP1_FIELDS_ONLY。12字段正反测试、11记录负控制、10独立提取、86来源SHA及独立回读PASS；原9行审计不变，余19704小时待全量字段QA。

qa SHA256=f47ba0aa95f9858033f3ab9d9114a78cea70acc376171c5f343338915ce04b7e；readback SHA256=f35c2405fa239d93a00e85e23dceebc6a2f48ae4dbea3df1a1fc759e5374ba86；audit_top1_incremental_fields_v006.py SHA256=2201d9d7f6149968653213e071b1baf2e27899c8480df2ed52591e0fee395927。June原ID 3a270adb-0a66-4b5e-bbc9-000b3dc259a9于00:26:59.543474完成81798字节，SHA256=a41c74ccf7947605e78bb4ec9d11a61cffe2d3c12bc17a07773be13219e514b3。

00:41:02检查唯一实际PID3164（2026-10-14 21:40:10.256955创建），Top1 run命令/RUNNING/OS锁一致，CPU12.015625→12.046875秒、stderr空，无failure/partial。July cc915b69-3eec-4920-b0d7-97a95bbe4d5d与August d9fc779a-7f4a-4e51-89a9-2e3a81d9ba18均accepted；新单字段verified=3、余26未提交。此为历史快照，下次重读真实进程/锁/jobs/logs；健康不重启，退出仅按既定无歧义恢复门槛恢复同一Top1入口，最多两活跃ID。

冻结计划/下载器不改，新载荷QA另存版本，不运行旧Top10入口或prepare。空间输入仍BLOCKED_SPATIAL_INPUT_DEFINITION，近邻近似/补视线范围选择未收到人工答复；不重复询问、不推理、不扩网格/诊断，新fit=0。LOCAL/旧模型/结果不变；局部字段QA不等于完整下载或整项完成，同一tvt继续15分钟。

## 历史May字段QA快照（2026-10-14 22:44 +08:00）

只读field_audit_005/qa.json及delivery_readback_qa.json，不重跑不可变审计main。新增May 744小时/1488网格值，累计9文件、2400/22824小时、4800个z@1hPa网格值、4/33父月，PASS_INCREMENTAL_TOP1_FIELDS_ONLY；12字段正反测试、11记录负控制、9独立提取、81来源SHA和独立回读PASS。qa SHA256=894494a1c1fa12a0a92f7c209b7314255463d202f546bbb96b31889f1757a243；回读SHA256=d8892bd3222b9d78cd68262c45c6a4b44d875f37d4a910f68e787701f75c3a41；新审计器v005 SHA256=b3520520870783ae700c31d5806423f734e21bfcaec1dd929bf1a4ebdf5bca9b。旧四轮QA和原8行记录只读不变，余20424小时仍需后续字段QA。

22:43:21实际PID3164（21:40:10.256955创建）继续唯一Top1 run，命令/共享OS锁/RUNNING一致，两次CPU4.84375→4.84375秒、stderr空，无failure/partial。June ID3a270adb-0a66-4b5e-bbc9-000b3dc259a9与July IDcc915b69-3eec-4920-b0d7-97a95bbe4d5d均accepted；新单字段完成2、余27请求未提交。May原ID已在22:30:53完成82449字节，载荷SHA512ad2b700ee23f10f5668a791cf9599e788b75af0323417afe02ebf160e9352。此为历史检查快照，下次必须重读真实进程/锁/日志/jobs，不以本段证明存活。

同一v003下载入口和冻结计划不变；健康只读、退出按原门槛主动恢复，最多两ID。新完成载荷的字段QA另存新版本，不重跑prepare/旧Top10。空间输入仍BLOCKED_SPATIAL_INPUT_DEFINITION，无人工近邻近似/扩视线网格选择；继续禁止推理与训练，不重复询问。LOCAL、旧模型和结果不变；局部字段QA不是完整下载或整项完成。

## 历史恢复与字段QA快照（2026-10-14 21:46 +08:00）

原worker32908因19:00—19:02主机重启而中断，21:33核对真实下载进程均无、OS锁释放，旧RUNNING不是存活证据。系统Kernel-General启动/关机、TrustedInstaller计划升级重启及异常关机事件一并保留在recovery_001_20261014_2138/diagnosis.json，精确worker退出时刻未确认。两个既有Top1 ID均经官方只读查询可访问：April已successful、May accepted；状态与日志先保存完整快照，9冻结/57保护/71已有字段审计来源SHA通过，无提交/访问歧义、无partial。

21:40:10.256955 +08:00已隐藏恢复同一唯一入口，当前历史启动快照PID3164；精确命令仍为python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download_top1_v003.py run，cwd项目根。日志另名weather_top1_run_004_20261014_214010.stdout.log/stderr.log；两次CPU0.484375→0.53125秒、真实创建时间/命令、RUNNING及共享锁一致，证据在restart_verification.json。不改冻结代码/计划、不重跑prepare/旧版本、不重复POST已有ID。仍须每轮重读实时状态，不能拿本段PID当持续存活证据。

原April Top1 ID4552a271-a060-41b5-a6cc-a76f11b2cdb6于21:40:18完成552小时，77022字节与官方预期一致，SHA256=da77b866564af46f17de26ce69a7365e6e4437dda09b2c48d9977fa13c04ad07。释放一个槽后自动仅提交June Top1 ID3a270adb-0a66-4b5e-bbc9-000b3dc259a9；当前两槽为原May ID97e01216-7578-42a7-a72a-2a3e6a8bf899与June，21:45均accepted，新单字段verified=1、未提交28。两个继承旧ID已完成；下方初始继承槽/32908快照均是历史，不再作为当前状态。

新增audit_top1_incremental_fields_v004.py与field_audit_004/qa.json实际PASS_INCREMENTAL_TOP1_FIELDS_ONLY；累计8文件/1656请求小时/3312个位势网格值、3/33完整地区/月，新增552小时/1104值。12字段正反测试、11记录负控制、8载荷独立数值提取/Unix秒UTC解码、76来源前后SHA及独立PowerShell回读均PASS，原7行审计记录完全不变。新代码SHA256=723f2c02cd360c7508dee9f65df2315966613cdff968c3e2d95639836afa3030；qa SHA256=8ca54461c82f8bead2a4a32f2c0609aa12f6b66417f2bca21720f880cb1e69cc；delivery_readback_qa SHA256=1da4ee70fcfe5d1ce0e246afabdf1e93d3b92322ca72c318f181470c7ff345cd。此004也是固定快照，不是可重复覆盖的通用审计入口；后续新载荷另存下一版，旧main不运行。field_audit_004/readback_check.ps1仅只读复核，可重放。

余21168小时尚未字段QA，完整Top1字段QA=false、原Top10完成=false、整项完成=false。空间输入仍BLOCKED_SPATIAL_INPUT_DEFINITION，我尚未作近邻近似/补视线范围选择，不重复问、不擅自替换输入/扩下载，不推理或训练。LOCAL/模型/预处理/CDF不变；tvt保持同一线程15分钟。三份固定入口before/after及document_handoff_qa.json在本轮recovery目录，文档/离线审计占用已释放，后台worker3164的下载占用未释放。

## 历史字段QA快照（2026-10-14 18:28 +08:00）

April b001原ID ba565558-5cf8-4c6e-b298-45c0a235eb3f于18:14:47完成，新增2004-04-01至07的168请求小时。field_audit_003/qa.json独立核验PASS_INCREMENTAL_TOP1_FIELDS_ONLY：累计7文件/1104请求小时/2208个z@1hPa网格值、2/33计划地区/月完整；12字段正反测试、10记录负控制、全部7文件独立数值提取/Unix秒UTC解码及71来源SHA通过。新qa SHA256=d8dfdc9d0b69e2ad457ace730972ffc2c62aa152edcb6de30e6060c5bfec33c1；audit_top1_incremental_fields_v003.py SHA256=3a73444b92f8a7ab9d62cfaa7dea6b4deaa1c2a737eda4926f4712aac161bf17；delivery_readback_qa SHA256=da6866c40472a78e1d609ee9b4a137f392036a065dabba8e224759e293dc77f2。原001/002字段QA和input_compatibility_001全部不变，后续增量仍另存，不运行旧main覆盖。

独立PowerShell回读的首次日期类型比较失败保留在readback_attempt_001.json，诊断为ConvertFrom-Json自动生成DateTime，显式-DateKind String后原始168个UTC字符串逐项通过，未修改源数据或字段QA。可重放只读检查为field_audit_003/readback_check.ps1。三份固定执行入口的完整before/after快照同轮保留。

两个继承ID均已完成，当前两槽为April top1（4552a271-a060-41b5-a6cc-a76f11b2cdb6）和May top1（97e01216-7578-42a7-a72a-2a3e6a8bf899），18:28均accepted；新单字段已完成0，另29未提交。实际worker32908持续运行且共享锁持有，日志无错误；仍须每轮重新核对，不能把此历史快照当存活/完成证据。不要重启旧入口，不另起worker。

剩余21720请求小时尚无字段QA，完整Top1字段QA=false、旧Top10完成=false、整项完成=false。空间输入仍BLOCKED_SPATIAL_INPUT_DEFINITION，我尚未选择近邻近似或扩视线空间范围；不推理、不拟合、不扩网格/诊断，不重复提问。下方936/864小时均是历史快照，不代表新目标评分。tvt继续15分钟，LOCAL与模型/预处理/CDF只读。

## 增量字段QA快照（2026-10-14 16:14 +08:00）

March b005继承原ID于15:55完成，新增2004-03-29至31的72小时已独立核验，field_audit_002/qa.json为PASS_INCREMENTAL_TOP1_FIELDS_ONLY。累计6文件/936请求小时/1872网格值，2/33计划地区/月完整；12字段正反测试、6记录负控制、全部6文件独立数值提取与Unix秒UTC解码、66来源SHA及readback均PASS。qa SHA256=e3654b8184c7c6c20b4123641a51b2f382fd503a38272895026b636f5d1ae87f；audit_top1_incremental_fields_v002.py SHA256=b827855dcfd625c247a22afaafee5adf21be550d5f6a620e94c53a918698250f。原field_audit_001与input_compatibility_001.json不变，不能覆盖两轮审计。以后新完成载荷的QA另存版本，完整字段QA仍需覆盖全部22824小时。

首次单字段April top1已由健康下载器自动提交：4552a271-a060-41b5-a6cc-a76f11b2cdb6；另一槽仍为继承April b001 ba565558-5cf8-4c6e-b298-45c0a235eb3f。16:08两项均accepted，worker32908精确命令/创建时间/CPU/锁一致；新单字段已完成0，不把排队计入完成。当前时点须重读，不据此快照启动重复worker。

空间输入仍BLOCKED_SPATIAL_INPUT_DEFINITION，我尚未选择近邻近似或补视线空间范围；不替换输入、不扩下载、不启动模型试跑。外部预测=0、新fit=0；健康Top1下载继续，不暂停tvt。执行记录前后副本及交接QA在field_audit_002/内，限T012新范围。下面的864小时与1703目标小时为15:45旧快照，不能误作本次新增目标评分或完整外部结果。

## 已有字段QA与待决空间输入门槛（2026-10-14 15:45 +08:00）

只读已有5个正式压力文件的独立核验完成，结果在本新版输出下：

- field_audit_001/qa.json：PASS_REUSED_TOP1_FIELDS_ONLY，864请求小时/1728个z@1hPa网格值；完整UTC、坐标、压力层、m**2 s**-2单位、geopotential身份、数量、有限值及SHA通过。不含smoke，不代表全22824小时字段QA；SHA256=325b9d787e2f12bf09640d48ede3bec4f5db7830a196868bce619d816a581c09。
- input_compatibility_001.json：BLOCKED_SPATIAL_INPUT_DEFINITION；SHA256=497b6993ee19c8537a5ea65f1008889ddf6a8833e8cf30d8c4f836f8d73261a8。
- field_audit_001/delivery_qa.json：12个正反字段测试、独立数值提取与37来源指纹、原纯插值函数两项控制检查PASS；只是审计证据通过，外部模型兼容性仍未通过。SHA256=df4914f93d1964db03958d0cd5c591242fb18d8b1558204a9713d78d8eeabe0c。

原模型实际2404/343/687小时输入的p1__geopotential，与本地动态卫星视线交点的双线性采样CSV逐行完全相等。新下载英国网格仅latitude=[51.25,51.0]、longitude=[-1.5]，与原输入构造不兼容；原插值函数在单经度时会返回NaN，不能拿整列缺失交给已有插补器变成常数推理，也不能擅自改成站点最近网格。1hPa压力层/位势单位相同不足以说明空间语义相同。

GBS现有目标与这864天气小时对齐：Chilbolton有效原生848、高覆盖842；Sparsholt有效864、高覆盖861。1703个高覆盖目标小时不是合法同定义输入/预测数；当前模型预测数0、新fit0。Dundee/OpenSat尚无本轮可用Top1载荷；秒/分钟快变、旧OpenSat经验晴空参考和天气分组仍需独立兼容性核验。GBS元数据的西经单位已经核实：Dundee longitude_receiver与全部satellite_position为degrees_west，英国两站longitude_receiver为degrees_east；不能按未经单位转换的正数误读地理位置。

当时需要我在以下方案中选择：1）批准另标注“最近网格近似”的单特征零训练试跑，不称同定义外部验证；或2）仍只补同一位势字段，批准另行规划视线采样所需空间范围。未得到新人工选择前两者都不执行，不能擅自扩下载/改输入/改变目标/训练。已有审计不可重跑覆盖，后续全量字段QA或修复另存版本。

空间门槛只阻止不兼容推理，不停止已授权Top1下载或tvt。15:45唯一实际worker32908仍健康，精确命令/创建时间/两次CPU1.0625→1.0625秒、共享OS锁/RUNNING匹配，stderr空；两个继承ID仍accepted。后续按实时状态继续下载，不重复启动、不改冻结代码/计划、不把这次阶段QA当整体完成。健康不变保持安静；同一待决问题不要每次心跳反复通知。

## 实际切换与冻结入口（2026-10-14 15:26 +08:00）

- 15:17:29 已核对旧 PID39924 精确命令和创建时间后停止本地 Top10 调度；无剩余匹配进程。远端请求未取消，旧文件未删除/覆盖，handoff.json 保留证据。旧 active_worker 的 RUNNING 是历史记录，不得据此重启旧入口。
- 新计划原33个地区/月压力覆盖，22824请求小时；新 POST 只含 geopotential/1hPa。只读复用5个压力载荷，已覆盖864请求小时/1个完整月；续接原两个远端ID，再新增31个月内剩余日期的单字段请求。不能与原77父项/196批混算完成率，亦不声称下载齐或外部评价完成。
- 37项离线恢复/迁移测试PASS；不导入下载器的独立计划审计PASS，复核60个旧/模型/交接来源文件。首次预检把SMOKE当正式父项造成KeyError，证据保留 preflight_history/001_smoke_enumeration_failure.json；排除smoke后通过，smoke不计正式覆盖。无网络提交测试、无训练。
- 新plan fingerprint：b80606aa0b5619ce8a447f4dabcc6cfabe6ca5aeb17b95c0a40451545fe70500；plan SHA256：265fc2dbddd0f7b8215aad9f4d99518fd1fc1dc6727000ee14536172aebf06d3。
- weather_download_top1_v003.py SHA256：76d285c80399e4ef9dbd59772f5a404b6bdaa05426930a2aa3f0bba8613d760a；test SHA256：c9e1c9fec76cc690891a9b46b18cb83888b116408f2418d3f0a690e753e08a71。
- 独立审计器 audit_weather_top1_plan_v003.py SHA256：05e4d06936880eea270d3287197a9a65680a3084ac9e3a16fbdb4675cc292e0e。
- 已有单特征检查点 SHA256：1df33b76440d9187af25ee452e5fa8a454d4dee8fa1038e6f150ff5affa79733；仅加载已有SimpleImputer/StandardScaler核验参数和SHA，new_fit_calls=0。k=1 自身原生204布局槽只有1个实际预测特征，这不是给156特征模型补零。完整外部特征构造/目标/时间尺度兼容性尚待核验，inventory PASS不等于外部科学验证PASS。

唯一运行入口：`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download_top1_v003.py run`，工作目录仍为项目根。状态入口将末尾run换为status。禁止重新prepare、重跑冻结QA覆盖产物，禁止旧v001/v002 run/smoke/prepare。

15:25:32.272867 +08:00已隐藏启动PID32908；真实命令/创建时间、两次CPU0.25→0.28125秒、OS共享字节锁及新RUNNING活动记录匹配，stderr为空。日志为 weather_top1_v003/weather_top1_run_003_20261014_152532.stdout.log/stderr.log。

继承的两个请求仍为accepted，且已由新worker实际按原ID查询：March b005=09473232-0465-4d92-8d9e-c6028ae3036a；April b001=ba565558-5cf8-4c6e-b298-45c0a235eb3f。两者占满远端槽位时不发送第三个请求；任一完成后自动提交单字段请求。继承请求若failed/不可访问须诊断，不能二分后重发其旧多字段请求。新请求大小拒绝才可日期二分。

后续只读监控新版plan/jobs/progress/active_worker/logs/partial_history及所有SHA；旧 transport v002 模块仅作为经SHA核对的恢复函数库，在新进程中把输出目录限定到新目录，未改写旧模块。运行中/冻结代码不得修改；新修复另存版本。

15:29已将原tvt正式提示替换为Top1范围，保存后逐字回查一致；id/原线程/15分钟/ACTIVE保持，无重复任务。preflight_qa SHA256=d246a9d105b69e79101c64de6c718ff66e888bb7b4270226fbadd9ce368e8c09；plan_independent_qa SHA256=1b53944d17417bb37b605d2cda04c13df8d9b4fb18045fa0b5d467cf441c6ed7；model_inventory SHA256=2ae8a999b41538d661c299fb04757d5aae41f5733d8eddb74326d91bfd341f45。下一步优先对已有5个压力载荷的Top1字段做独立QA和外部特征/目标兼容性核对，可兼容时先按明确可用范围纯推理试跑，不等待被撤销的Top10全量；不得按试跑成绩改变模型、挑日期或重拟合。目前没有外部预测或评价分数。
