# Revision023 — LOCAL 原测试小时的精确采样支持核验

T012；原线程 01a0bda1-92d1-7832-9508-60f4e0ec1634；2026-10-18 +08:00。仅补齐 Revision022 F08 的时间/质量元数据证据，不计算真实 E-SST 雨衰或新排名。communications-formula-auditor 用于区分已验证记录支持与尚需作者确认的适配约定。

## Preflight / Reuse / Conflict

已读根 README 适用原子规则、T012/001进度及022协议、交接、QA和最新人工决定。16:58:54Z 核对1124旧保护来源、022的38件封存产物无SHA变化；其中含011五份计算源码和两份既有预测，继续复用。17:04:03.3726019Z 具权限CIM核验Python/pythonw/MATLAB/Octave/Rscript进程0，023两个目录均不存在。原系统锁与未加载外部写者信息的局限沿用022，不宣称全机排他。磁盘约153.74GiB空闲。

复用/检索结论：022完整小时均匀功率积分缺少实际记录采样支持；017质量0秒级快变掩码与687主测试小时的物理有效口径不同，不能替代。原Dataset002构建源码和Run011源码已有确切去重/时钟合同；直接提取冻结quality_code的time/value，无需新增库或寻找通用实现。Run011原生100ms记录保留源时间戳，只有原生10ms文件clock-bin；严禁将全部时间取整到100ms网格。

READ：022完整封存、原1124来源；Dataset002的measurement_hourly仅下列元数据列及原manifest/source_snapshot/validation/QA；Run011的quality_code两列、处理代码及验证记录。peak_power_dbm只核SHA，不解析其数据列。WRITE：仅Code/Revision_023与Outputs/Combined_Channel_Revision_023。CPU短任务，无GPU、无下载、无训练；原模型/数据/代码/参考/结果/失败/STOP/tvt均只读。

C1/C2/C3=POTENTIAL/LOW（新版本独占创建与SHA）；C4=CONFIRMED/MEDIUM（记录支持≠连续100ms积分窗口，实际物理适配未放行）；C5=NONE（没有可直接复用的相同掩码）；C6=POTENTIAL/LOW（顺序读取约221MB质量文件，独立二次扫描）；C7=CONFIRMED（E-SST其他约定、UK输入、RMOP/PRM仍受阻）；C8=CONFIRMED/LOW（仅新增元数据计算）。Decision=PARALLEL_ALLOWED仅指隔离写域，不授权新agent或重复worker。

## Execution Lock

LOCKED为启动声明；生产与复核/封存分阶段使用同一本版字节锁，记录实际PID、命令、环境及释放时间。最终状态见final_verification.json。协议不表示跨进程的永久全机锁。失败原样保留，修复另存新文件/版本。

## 冻结合同比较前声明

1. 687测试成员完全沿用022 derived_inputs.npz的campaign_epoch_s/test_member，按UTC精确键匹配，不按雨、成绩或质量再次选小时。读hour_utc、原始/唯一/重复记录数、物理有效/无效数、质量位计数、coverage及首末时间；不读取实测功率、衰减、标签值。
2. raw quality_code.time为无时区BJT毫秒时间，按Asia/Shanghai的UTC+08转换UTC。精确同毫秒时间戳去重，重复quality按位OR；与Dataset002原规则一致。物理屏蔽为bits1/2/4/8，bits16/32保留。原目标还检查功率有限性：本轮不解析功率，只继承已封存构建/验证记录，并以全部687小时physical_invalid=0、唯一数=有效数、SHA和逐键质量计数一致作为有界证据。不是重新独立校验原功率值。
3. 每小时保存排序后的uint32毫秒偏移、UTC键和indptr，附OR后的uint16质量码供审计。不得把质量码当气象模型输入；将来仅允许在模型输出后按既有评价支持聚合。无插值、无时间取整、无补缺、无改标签。
4. 每条保留的唯一物理有效记录在既有小时目标中等权；原100ms窗口内部真实积分时段/原10ms样本个数不能从本轮元数据恢复。名义36000条的计数差不等于未观测时长，采样间隔不一定恰为100ms。
5. 生产用Parquet行组min/max筛选后分小时精确去重；独立复核用Arrow Dataset无行组/时间预筛完整扫描，逐批membership过滤，并用Python字典逐时间戳OR归并。不导入生产函数。逐点时间/质量、全部每小时计数与首末边界、总步长直方图严格相等；coverage容差2e-15预先冻结。
6. 合成单测覆盖非100ms网格、跨小时/BJT至UTC、乱序、重复与质量OR、物理屏蔽、空输入等。合成值不计入实测成绩，不拟合任何参数。
7. 七类、687/5664/5593/5538、英国143/93站点日、既有种子/排名保持。MTS=NOT_COMPUTED_USER_DECISION，空值不排名；天气STOP及同一tvt/原线程15分钟不变。E-SST真实H/v/事件/几何和总目标适配仍需显式合同；本版只部分关闭采样支持缺口，不能声称七类完成。
