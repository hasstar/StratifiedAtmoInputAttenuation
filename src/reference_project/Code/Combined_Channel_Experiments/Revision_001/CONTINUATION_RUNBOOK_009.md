# T012 continuation 009 — 全部后续 Top10 下载改用 NCAR/GDEX

2026-10-15 我最新的决定：“全部改为NCAR下载”。本文件优先于008及所有旧CDS恢复授权。仅替换获取路线，保留008已批准的自然Top10、四层固定高度、站星视线最近发布网格、原日期/时刻/站点，以及003零训练边界。

## 唯一正式入口

项目根为cwd；新输出 `Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_ncar_v008/`。

```powershell
& 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B 'Code\Combined_Channel_Experiments\Revision_001\weather_download_top10_ncar_v008.py' run
```

只读入口末尾改 `status`。initialize已一次性完成，绝不重跑。所有旧v007/v006/v005/v004/Top1/v002/v001下载器的run/smoke/prepare/initialize均禁止恢复。旧源码/计划/载荷/日志/QA/模型/LOCAL全只读；修复另存新版本，不改冻结运行代码。

冻结身份：

- 下载器SHA256：`d828e4c22ee5853103920c23c68391f283aaf1c780faeb53ab614915e4e29510`。
- NCAR传输模块SHA256：`b548e0c21375e35baf23ab5443b46049bc2236314317542ecc289df75bdaf8b3`。
- 测试SHA256：`2d03cff3cdbf9c6597b9956c1f71171b61909bb6b4272b4ec072faf0af1e9b5f`。
- 独立计划审计SHA256：`44b2c02bad4a11c625faaf041e439274eb8c35a0fedf0ba16ee4720815f45377`。
- plan fingerprint：`265af5f1fe9cadb43bc514e40f832c4fee1d696fa61bdff199ad663364d97807`。
- plan SHA256：`7ad663b779f04c6fb87e08feb30c6d7fc3e16063d3b658d659589371baf77fa1`。

30项新离线正反/恢复测试PASS。独立审计保持原77父项/45663日历请求小时、33压力月/22824小时、218链路、871有效固定交点及1无效交点；代表性2840个NCAR精确子请求/9431唯一格点字段小时的独立坐标时间展开通过，335旧保护文件SHA保持。不是全量实际字段、跨平台数值一致性或外部预测QA。首次测试因沙箱临时目录权限失败原样存preflight_history；指定本版test_scratch作为临时目录后通过，未放宽测试。

## NCAR 路由与下载范围

仅公开GET：`https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/`，使用NCAR的ERA5已发布0.25度NetCDF文件的OPeNDAP ASCII精确子集。无需CDS排队POST，不读取CDS密钥，不改代理，不下载整份全球文件；不把ASCII输出说成未经格式转换的原始二进制。

模型仍只有8个压力特征对：z/u@1hPa、o3/u@5hPa、o3@500hPa、u@650hPa、o3/r@850hPa；地面仅CAPE与maximum_2m_temperature_since_previous_post_processing。每次GET一个字段和一个压力层，禁止新增z@5/500/650/850、云雨雪或其他诊断。目标参数直接读取，不从其他变量推导。

压力文件按日、CAPE按月、MX2T按半月预报档案组织；仅取需要的日期/小时/网格，不补原日期缺口。MX2T必须用原档案的06/18UTC起报与1—12小时预报时效相加形成valid UTC，含跨日/月边界，不当日最高温，也不插值。它是同一201号参数的档案定位，不是重算气象值。返回参数ID/单位、坐标、层、UTC、数量、有限值及SHA通过才标VERIFIED_SUBSET。下载完整性检查不代替独立字段QA，更不声称跨平台值逐项等同。

最近格仍逐轴最近、中点北/东；不添加grid再网格化，不做空间或时间插值。5/500/650/850hPa固定标准几何高度35979.046/5579.330/3592.717/1457.635m，沿原站星前向WGS84射线求交再取最近格，不是地面垂直柱。四层不读/补辅助z；1hPa的z本身为Top10，仍用于该层逐小时定位（最多8轮）。固定四层不声称是真实逐小时等压面。缺值、循环、不收敛、低于天线保留无效，不补零。

FRA-00289天线高于固定850hPa层，保留INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA；FRA-00609方向差0.5424度仍待复核，不放宽0.5度门槛、不改指向。全部链路身份与历史分区保留，不将这些缺口伪装为完整可预测。

23旧载荷只读候选复用，按实际需要的变量/层/坐标/UTC/单位/有限值及SHA选值，不能整份计新LOS覆盖。新NCAR原始ASCII响应、DAS元数据、逐值canonical records和job状态分开保存。waves按实际缺值动态产生，原父项顺序不变；精确打包不能新增字段、层、网格、日期、时刻或重复样本。

## CDS 停止与新后台交接

旧v007唯一实际PID25488、创建2026-10-15 16:45:45.321752及精确命令和两次CPU核对后，于17:26:08.9608804前停止本地调度；实际旧下载进程为空，共享OS锁释放。旧active_worker的RUNNING仅历史，禁止据此重启。完整状态/日志副本保存在handoff_state_before。

旧两个Top10远端ID `bbcb2544-3c0a-46a2-b8de-1714bf9e3926`、`d01c9599-a74f-433d-958a-f9fddcc54164`最后本地记录accepted，远端没有取消。旧辅助ID `dbac99a1-b428-417e-9d26-5442ca99679d`已记录RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD。新路线对这些ID不再轮询、不取载荷、不重提、不分裂；这些远端历史状态不算NCAR在途任务，也不计新完成量。

新worker启动历史快照：PID29188，2026-10-15 17:38:58.272437 +08:00，唯一v008 run；日志weather_top10_ncar_run_001_20261015_173858.stdout.log/stderr.log。17:39实际命令/创建时间及两次CPU约3.08→3.09秒核验；精确值/共享OS锁/进度/初始下载收据另见launch_verification.json。历史PID不可替代下轮实查。

首个surface采集单元复用旧值。首波所需576格点字段小时拆为24个NCAR日文件精确子集（原8日期各24小时，u@1、o3/u@5），不是全项目总请求数或完成量。

## 检查与恢复

先读T012、003、009、PROGRESS，以及新目录plan、handoff、launch_verification、preflight_qa、plan_independent_qa、waves/units/jobs/progress/active_worker/logs/failures/partial_history。plan引用的模型和几何只读。

检查全部weather_download*.py的真实命令、PID/创建时间、两次CPU、共享OS字节锁、冻结SHA和日志载荷。存活仅只读检查，不重复启动或改代码。NCAR最多两个GET，连接超时10秒、读取超时180秒，单响应上限16MiB；按同URL退避30—300秒，429尊重有界Retry-After，不创建远端任务。服务端慢/低CPU不等于本地进程死亡。

只有实际进程均无、锁释放、未完成且指纹一致、失败已诊断时，隐藏恢复上述唯一v008 run并另名日志。GET幂等，已完成并核SHA的子集跳过；中断GET同URL重试，不退回CDS。HTTP401/403/404/410、参数单位/UTC/坐标/数量/有限值失败保留BLOCKED状态，先诊断；不换源、不换代理、不伪造完成。

part先检查是否为完整DAP响应，变量/所有坐标/层/UTC/形状/全部行数与精确需求相同且无缺值方可转正式，记录SHA。否则原样移至本版partial_history留存再同URL重下。HTTP Content-Length存在时核实际字节数；无此字段时不杜撰官方大小，以完整响应解析检查并注明。冻结运行源码不改，修复另存新版本。

VERIFIED_SUBSET和units只表示采集记录/下载完整性，不是完整字段独立QA或预测。报告日历小时、站点小时、唯一格点字段小时、十列齐全站点小时要区分；不会把单字段24值称为24小时完整Top10，也不能用小样本速度保证全量ETA。

## 后续与完成条件

下载后做独立实际字段/UTC/单位/坐标/数量/有限值/SHA及几何残差QA，核对外部目标和已有模型兼容性，再只做明确覆盖范围的推理。模型k_0010 SHA=b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5，BiLSTM seed17/53轮，实际LOCAL2404小时拟合，10列/原生268槽；不是给156列网络补零。

只冻结transform/predict/组合/指标，禁止训练、fit/partial_fit/fit_transform、微调、调参、选种子、优化器更新、重估插补/归一化/晴空参考/CDF。LOCAL及旧模型结果QA只读不重跑。GBS秒级relative-clearsky可复用旧Alignment004；OpenSat分钟RSL平均域未知，不插值造秒真值、不重新拟合分钟CDF。缺模型、特征、目标尺度或天气分组明确列缺口由我决定，禁止擅自扩下载。

按来源/频段与总体、晴、雨含强雨、干燥阴天、other分别报告RMSE/MAE/Bias及适用CRPS/覆盖校准和样本日期站点；缺诊断保留不可评估，不跨频段合并RMSE。历史访问/同日重叠不称新盲测，随机场景不是确定性秒预测；IID/block共享CDF、常数增量相消，不将共同VS当慢变增益。LOCAL CRPS0.130487较自身0.162335降19.62%，90覆盖73.98%，不是已校准。负结果/空数据/数值失效保留。

同一tvt、原线程、15分钟保持；正常无变化安静，阶段完成/失败/需决策才通知。路线切换、下载、字段QA或局部试跑均非T012整体完成；仅既定外部评价、独立QA、图表、中文报告和进度齐备，或我明确缩减最终范围后满足，才总completion并暂停。
