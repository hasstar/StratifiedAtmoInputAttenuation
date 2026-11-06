# 安全继续

先完整读取本目录PROTOCOL/PARAMETERS/PROGRESS和T012，核对manifest与QA。只使用本目录新入口；禁止启动旧Rev028/Rev029/FastRev002的训练/run/report入口。

LOCAL顺序：纯合成检查 → prepare冻结 → run（按32小时检查点）→独立verify→report→独立report-audit。run末尾可自动推进核验/报告，但存在同命令worker时只读status，绝不重复启动。

所有worker必须隐藏启动、独立stdout/stderr、OS字节锁与active_worker。旧PID不是存活证据；核对实际Python命令行/创建时间/两次CPU、锁、日志推进、检查点和指纹。失败只在同worker已退出且锁释放时恢复相同命令、同指纹、缺失检查点，使用另名日志，保留failure审计。

外部字段下载与处理入口在本地QA PASS后登记。不得用外部测试选择Top10、块长、种子或比较模型。气象账户/授权若缺失，保存明确缺口；不要输出API密钥或通过改用未声明来源绕过。

总任务completion必须包含LOCAL、GBS、OpenSat三个完整QA和报告；任何单阶段完成不称全部完成。2026-10-13用户已明确恢复原tvt每15分钟检查，原id已ACTIVE；不另建重复任务，全部交付齐全后才暂停。

LOCAL已于2026-10-13 14:57正常退出且数值/报告QA PASS，不重新运行已完成评分。冻结指纹2f7dbdda002880e885baf7e459ab042c989703154818f4f0ca6df8c59fb42cc9，37检查点，11组/1030小时/11330记录。旧worker9992不是当前存活证据。调用/图表也已交付，下一阶段核对外部元数据、完整ERA5请求，另行冻结外部协议与数值QA，不回调本地结果。

截至15:49，LOCAL调用/图表及delivery_qa也已PASS，不重跑本地。先完整读取EXTERNAL_PREPARATION.md和CALLABLE_GUIDE.md，再读external/metadata.json、weather_plan.json、weather_jobs、smoke QA和失败历史。后续正式外部模型尚未实现/冻结，不能跳过外部字段与训练规则QA直接测试。

当前天气预检worker为PID40032，15:24:07隐藏启动，精确入口weather_download.py smoke，日志weather_smoke_001.stdout.log/stderr。先运行weather_download.py status；实际命令行/创建时间、两次CPU、OS锁/active_worker、日志、已存request_id与SHA256都检查。1/2小请求完成，地面服务端accepted不代表失败。存活时不另起smoke或run；排队状态正常保持安静。不可因一个旧PID存在就认为命令匹配，也不可因CPU低就判断网络等待失败。

只有smoke进程正常结束且weather_smoke_qa已保存，才能运行verify_weather.py；它只读现有下载，字段/有效UTC/单位全部PASS后才隐藏启动weather_download.py run。正式77请求最多2项同时，已有VERIFIED校验后跳过，已有request_id恢复服务端原任务。新stdout/stderr另名，禁止覆盖旧日志。失败后先查weather_failures并确认同入口无进程、OS锁释放和计划/代码指纹一致；服务端failed/rejected或部分载荷必须诊断，不盲目重复提交。

正式下载complete也只是下载阶段，需全部月份源数量/字段/坐标/UTC/单位QA，随后固定训练可用小时、训练天气与合法块、参数预算及数值QA才启动新外部训练。完整LOCAL/GBS/OpenSat交付前，原tvt继续ACTIVE。
