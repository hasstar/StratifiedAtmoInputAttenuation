# T012 / Revision013：已有 LOCAL 物理输入独立核验

2026-10-17；承接已交付 Revision012。七类名单、四类既有评分与英国 143/93 日范围不变。本版只推进 LOCAL 既有原始气象资产 QA 与 E-SST 可实现性审计；不是模型训练、推理、排名更新或气象采集。

## Execution Lock 与 Conflict Check

holder=/root，thread=01a0bda1-92d1-7832-9508-60f4e0ec1634；本版准备时声明 LOCKED，实际 OS 锁由一次性审计程序获取/释放，最终 HANDOFF 记录 RELEASED。2026-10-17T04:34:18.8378363Z 与 04:34:21.9178819Z elevated CIM 查询成功，全部 Python/pythonw 进程为空。04:39 左右只读重算 Revision012 的 599 个保护 SHA，无变化；本版代码及输出目录不存在。当前线程列表只有本线程处于相关 active 工作；系统 BLOCKERS 尚未盘点，不作为无冲突证明，未登记会话和非 Python 外部占用仍未知。未知占用通过只读源数据、前后 SHA 和独立新输出隔离，不宣称全系统无冲突。

READ ONLY：ERA5 campaign/raw/download_manifest.json；其中明确选定的九种气压层字段和三类地面文件；Revision012 的 time 键、保护清单、全部既有交付；根 README、T012、PROGRESS、STOP。WRITE ONLY：本 Code/Revision_013 与 Outputs/Experiments/Combined_Channel_Revision_013。不得改原始数据、旧代码、模型权重、预测、标签、指标、STOP 或 tvt 配置。语义相同的历史成果只引用，不复制重算。

- C1/C2 LOW：新目录独占写入、源只读，SHA 前后复核；若源变化即停止发布。
- C3 LOW：不导入旧下载入口或模型执行函数，不修改冻结代码。
- C4 MEDIUM：高度基准、预报平均率时标、离地压力层和水凝物定义仍是模型适配门槛；本版只登记，不自行确定缺失假设。
- C5 LOW：不以排名为目标，不触碰英国/测试衰减标签拟合。
- C6 LOW：有限单进程 CPU/netCDF 只读；无 GPU、外部天气请求、后台常驻任务。
- C7 MEDIUM：E-SST 全文/事件边界未完整核验，PRM 专用散射系数、RMOP 软件及英国廓线仍待落实。输入 QA 通过不解除这些阻塞。
- C8 LOW：四类评分直接复用 Revision012；新增工作是此前尚未完成的全场检查。

## 预定数据契约

九种气压层字段：t(130,K)、q(133,kg/kg)、z(129,m²/s²)、u/v(131/132,m/s)、clwc/ciwc/crwc/cswc(246/247/75/76,kg/kg)。完整检查各月、UTC、11 层、4×2 发布网格、参数身份、单位、有限/填充值、范围与跨场时间/空间一致性。只对已有数据计算描述性输入诊断，不拟合、插补、裁零或生成衰减。

地面源：surface_instant 的 deg0l/ptype/t2m/d2m/sp/tclw/tcc；surface_mean_rate 的 avg_tprate/avg_cpr/avg_lsprate；surface_supp_mean_rate_01 的 avg_tsrwe/avg_csfr/avg_lssfr。新 shortName 与旧 MTPR/MSR 名字按 GRIB paramId 显式对应，不根据名字猜测。检查地面 2×2 网格，不将它误称与气压 4×2 完全同域。

原始 valid_time 为 UTC。manifest 的 start_utc/end_utc 字符串实际携带 +08 偏移，以显式时区解析；连续 campaign 为 7800 小时。降水平均率支持信标 [t,t+1h) 时使用 valid=t+1h 的输入核验；不在本版改变历史 LOCAL 评分的对齐。边界原始整日文件应有这一个补时。LOCAL 687 测试键只读取 time 列，不读取 truth/prediction 进行分析或选择。

deg0l 文件 long_name 明确 above ground；AMSL 转换需要相应网格地形，不能用站点海拔替代整个网格的地形。ptype 是降水相态编码，不自动等于层状/对流。z/g0 是位势高度，仅作排序/低于天线诊断，不冒称几何高度或完成斜射线追踪。气压与地表压力比较仅在共同的 2×2 网格上做，不推断缺少地表场的另外两行纬度。

## 输出及验收

一次性 raw_sources_before、file_inventory、field_summary、hourly_input_ledger、geometry_diagnostics、QA、中文报告和 HANDOFF。数值、缺失、负值全部原样计数；有问题时发布“审计成功发现限制”，不宣称“模型输入完全合格”。独立 verify 用 netCDF4 重新读取而不调用 production 的统计函数；其独立性是同一 agent 的另一实现，不是外部同行评审。

每个输出用独占创建，不覆盖；失败输出保留，修复另存新脚本/新结果目录。源 SHA 通过且独立复核通过后才发布输入结论。全部七类尚未完成，不暂停对比自动化；天气停止继续有效。
