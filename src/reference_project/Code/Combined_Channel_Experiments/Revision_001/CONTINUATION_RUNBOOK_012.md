# T012 continuation 012 — 两站扩时段与小时雨量下载

## 人工授权与边界

2026-10-16我明确要求扩充现有两个站的原10个气象参数时间范围、补充雨量，修改原定时任务开始下载，并在我要求停止时停止。012及T012最新段落优先于011的39日采集限制和全部旧下载恢复规则；原39日验证、LOCAL、旧代码/模型/计划/数据/结果/QA均只读。此轮仅数据采集与字段QA，不训练、不推理、不重算评价或启动对比实验。

同一`tvt`、原线程`01a0bda1-92d1-7832-9508-60f4e0ec1634`、15分钟保持ACTIVE。只新增`external/weather_gbs_extend_v011`和本Revision的新版本代码。只读复用旧函数和已完成载荷，不运行其入口，不恢复Dundee/OpenSat或218链路全量。停止旧全量是人工范围变更，不是异常待恢复。

## 固定扩充范围

计划：`Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/plan.json`，配套`CHANGE_RECORD.md`、`preflight_qa.json`、`plan_independent_qa.json`、`launch_verification.json`、`handoff.json`、waves/units/jobs/progress/active_worker/logs/failures/partial_history。

| 站点 | 原生GBS归档日期范围 | 实际日期数 | UTC站点小时 |
|---|---|---:|---:|
| chilbolton | 2003-08-19—2005-03-31 | 589 | 14136 |
| sparsholt | 2003-10-08—2005-03-31 | 538 | 12912 |
| 合计 | 保留各站原有日期缺口 | 1127站点日 | 27048 |

只以已有两站原生GBS文件清单作为日期来源，不按天气或预测成绩选择。1127份文件的长度/MD5/坐标单位与位置/天线高度/20.7GHz载频/卫星经度约定/粗指向已核，SHA入清单。先补旧39日的降水，再按时间扩充各站自己的日期；相同日期掩码才合并，不使用旧三站日历交集，不填不存在日期。27048是最终目标站点小时，不是新增下载小时；缓存复用和唯一格点字段小时另列。

自然Top10仍仅：z/u@1hPa、o3/u@5hPa、o3@500hPa、u@650hPa、o3/r@850hPa；surface CAPE和maximum_2m_temperature_since_previous_post_processing。无辅助z。四固定层5/500/650/850hPa几何高度35979.046/5579.330/3592.717/1457.635m，沿原前向WGS84站星射线取最近0.25度发布格；1hPa只用Top10自身z逐小时定位、最多8轮。地面取站点最近格；中点取北东。固定层非逐小时真实等压面，与原模型双线性空间采样不同，不空间/时间插值，不增加grid再网格化，不称地面垂直柱。缺值/循环/低于天线/不收敛保留无效。

## 新增雨量数据及时间定义

NCAR公开ERA5 meanflux源，额外下载两列原始率，独立于Top10输入：

- MTPR / mean_total_precipitation_rate，ECMWF local table 235 / parameter 55（235055）：总降水率，含液态与固态。
- MSR / mean_snowfall_rate，table 235 / parameter 31（235031）：降雪水当量率。
- 单位均kg m-2 s-1，平均期为valid UTC结束的前1小时。原始率完整保留，乘3600得到该小时水当量mm；不对均值率再做累计差分。
- 对信标小时[t,t+1h)请求valid UTC=t+1h，保留跨月/跨年补时。NCAR文件按半月、06/18UTC起报、1—12小时预报时效定位。
- 后续如需液态雨量可明确另列3600×(MTPR−MSR)的再分析估计，负差保留并标记，不裁零；此轮未计算模型指标。不能把总降水当纯雨量，也不能把小时ERA5当分钟雨强或雨量计实测。
- 已检查该NCAR目录不提供独立tp积累文件；旧猜测路径404仅为路线探索失败，不重试或退回CDS。现用实际目录和DAS确认的MTPR/MSR文件。
- CEDA原生10秒雨量计文件访问实际重定向登录（302），未取得载荷，不读取剪贴板/凭据、不绕过权限。该缺口不阻塞已授权NCAR两站下载；若我需要实测雨量，须另行提供合法的登录方式或文件。不得声称实测雨量已齐。

## 唯一入口与冻结校验

项目根cwd，唯一正式下载命令：

`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download_gbs_extend_v011.py run`

只读状态末尾改`status`。`prepare`已一次性完成，绝不重跑。禁止旧weather_download的run/smoke/prepare/initialize，包括NCAR v009/v008及所有CDS版；新v011只读导入旧实现，所有写入定向新目录。

- 下载器SHA：72ada17b4ff24c0aaf1eed8b89f28136bb812fea4557765780c1e461d3bb2e32
- 降水传输v002 SHA：e984e6eb5086652f2468df0e0722249331c3d7b4c4d2f4ac5fd74c8cd39489f2
- 测试SHA：23a01896a7c250acbfc70242c9f73f0a7e7effac0f2843e0df3778cbc113b28b
- 独立计划审计SHA：fa70e33803358f59c687b546a7c737946281f56428749a90cbbbd5acf012f50f
- 预检SHA：40990e176eef50dc33f768c421ad7de31d1906ec51a18cf6dab286596c5d53c8
- plan FP：b58d2b9f1f83970cca410195504cd0aa21cb0f7e57cf004ddb1e43100c5f56db
- plan SHA：1f9be56c4351940342a135b487d44aa1d6914c4a1a3dfc0ef4b040f43b53fdac

58项离线测试PASS；独立两站日历/27048站点小时/672代表性精确请求值身份及2317旧保护文件SHA检查PASS。新雨字段负控制包括错误表号/名称/单位/层、跨月时效、精确时间掩码和停止阻断GET。只是离线、计划与来源QA，不是全量实际字段或模型QA。开发期间fixture和临时目录权限失败及修复说明留在CHANGE_RECORD，不抹去失败。

## 下载、恢复与停止

1. 每轮完整读当前T012、003零训练、012、PROGRESS、新计划/交接/预检/审计和状态；只读plan引用旧数据。先检查最新人类消息，我叫停优先于历史自动指令。
2. 实查全部weather_download Python进程的真实命令/PID创建时间/两次CPU、共享OS锁、active_worker、代码/计划SHA、日志及载荷。01:18启动PID38832/创建2026-10-16T01:18:02.259803+08:00仅历史，不代替每轮实查。沙盒CIM失败不能当作进程为空。
3. 存活且身份一致只读检查，不重复启动/改代码。只有实际下载进程均无、共享锁释放、未采集完成、没有STOP_REQUESTED、冻结SHA一致、失败已诊断且无访问/身份歧义时，隐藏恢复上述唯一v011 run，另名日志；继续原精确GET，已完成SHA核验复用。
4. 全部新缺值仅匿名GET `https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/` 精确OPeNDAP子集，最多2并发；不拉全球文件、不换代理、不CDS POST、不轮询/取消/取旧CDS远端载荷。日期/字段/层/网格不得多取；共享无空洞格点矩形仅在相同时间掩码下合并。
5. 10秒连接/180秒读超时、16MiB响应上限；网络/408/429/5xx同URL有界退避，429尊重有界Retry-After。401/403/404/410或参数/单位/坐标/UTC/数量/有限值异常BLOCKED，保留失败先诊断，不伪造成功。正式载荷需SHA、实际字节、Content-Length如有、完整形状与精确值身份通过；无Content-Length不编造官方大小。不完整part原样入partial_history再同URL重下。
6. 我要求停止：执行本版`stop`写入STOP_REQUESTED.json，立即禁止新GET，已开始的有界GET允许收尾后释放锁；保留全部已得/未完成文件，然后暂停同一tvt。停止标志存在禁止恢复，不能自动删除标志。需要强制终止时先精确核PID/命令/创建、保留失败状态，不盲杀所有Python。
7. 若两站有限原生日期全部采齐，记录acquisition_complete并安静待命，不擅自扩大到其他站/无标签年份，不自动启动实验；本次“直到说停”不意味着无限重下已完成文件，也不取消错误保护。
8. 下载记录中的VERIFIED_SUBSET是传输/解码完整性门控，units只是采集记录。实际字段独立QA另存版本，可在固定新范围核对参数ID/UTC/层/单位/真实网格/数量/有限值/SHA和LOS几何；不做新模型实验。相邻站共享网格时重复使用不算独立气象数据。

旧LOCAL/39日评价结果均保持历史口径，不因本轮数据扩充重跑、重拟合、修改报告指标或称盲测。旧模型比较调研不等于实施授权。本次下载启动不标T012完成。正常无变化安静；阶段完成、错误、需要决策才通知。
