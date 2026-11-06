# Revision 010 — 已完成英国日期的五项 ITU 输入补齐

2026-10-16，/root，原线程 T012 追加执行记录。直接授权：用户确认“补齐”，承接 Revision009 请求，只补英国已完成 236 站点日/5664 站点小时的 2 m 温度、2 m 露点、地表气压、整层水汽、整层云液态水。

## 范围与验收

逐小时键只取冻结 Revision006 predictions_uk_all.parquet，并与 Revision003 expanded_dataset、Revision008 cloud_all 核对。Chilbolton 143 日/3432 小时，Sparsholt 93 日/2232 小时，保持原缺日，不取交集、不扩站或日期。目标是 5664 x 5 = 28320 个气象值，另存宽表和逐字段来源表。所有参数属于 ERA5 surface analysis，valid UTC=t；原 meanflux 降雨 valid=t+1 规则不用于这五项。

字段：ECMWF table128 parameter167 (2t)、168 (2d)、134 (sp)、137 (tcwv)、078 (tclw)。必须由真实目录与 DAS 确认变量、short_name、long_name、单位、坐标、日历、发布格；不由文件名独自认定。温度 K、气压 Pa、柱水量 kg/m²，原值原单位保留，不插补、不裁负、不计算衰减。

站点最近 0.25° 发布格（中点北/东），同 Revision008 TCC，非高空射线交点或空间插值。每站每月只按连续实际目标小时构成精确 OPeNDAP 点子集；6个月、12个站点时间块，每块5字段，共60个小子集。复用 Revision008 六份已核 SHA 的真实月目录，不重新下载全球数据或无关日期。DAS 为30份，无气象样本扩展。

## 当前上下文、复用与冲突检查

根 README 为现行规则；沿用已读未改变的全局约束。已读 T012 最新段、Revision008/009 协议、交付和数据口径。受限 CIM 失败未当作进程为空；2026-10-16 23:10:26 +08:00 具权限 CIM 查全部 Python 为0。Revision009已释放占用；旧v011 EXITED_NORMAL、STOP_REQUESTED存在，tvt PAUSED。未登记的其它会话仍UNKNOWN，不宣称全系统无冲突。

READ：Revision009保护清单及所指历史输入、Rev008交付及目录/TCC、Rev006冻结预测、Rev003扩展表、原站点几何、Rev008只读ASCII解析器。WRITE：仅全新 Code/Combined_Channel_Experiments/Revision_010 与 Outputs/Experiments/Combined_Channel_Revision_010，两路径建立前不存在。T012/PROGRESS固定索引及旧代码/权重/预测/停止文件/自动化配置均不改。

Execution Lock：/root占用上述两个新写域；输出目录 exclusive mkdir；各执行阶段新目录内 OS 字节锁防重复。失败保留带时间戳收据和partial_history，不覆盖失败或已交付数据；最终completion.json释放。此协议与新报告是正式T012追加修改记录。代码修复若发生运行失败后另存版本，不能隐藏失败。

C1/C2 文件/数据：独立新写域、只读来源、执行前后SHA，LOW；C3 代码：复用已核验纯解析函数，不调用任何旧下载入口，LOW；C4 逻辑：CONFIRMED/MEDIUM，以DAS确认瞬时字段与UTC，不混同降雨平均率、TCC或TCW；C5 目标：最新授权仅定向补齐，不恢复持续下载/训练；C6 资源：匿名GET单并发、短时CPU，无GPU；C7 依赖：现有完整小时键已具备，物理参考映射及组合指标仍另待核验；C8 重复：复用旧月目录/解析器/网格规则，不重取已有同URL完整载荷。未登记写者UNKNOWN，通过只读与SHA守卫隔离。

Reuse Check：完整读取 Revision008 cloud_acquisition.py 与 verify_full_weather.py；前者 parse_ascii 纯函数可只读导入，原模块所有写函数/入口不调用。对其字段身份/收据/独立逐点QA模式做新版本适配。新 transport 明确 allowlist、缓存身份、失败分型和停止检查；不安装包，不读取凭据/浏览器登录状态。

## 网络与失败规则

仅匿名 GET https://tds.gdex.ucar.edu/thredds/dodsC/ 下、来自已核验目录的指定文件 DAS/精确ASCII点子集。单并发；禁全文件fileServer、禁CDS、禁POST、禁代理、禁认证、禁跳转。10秒连接/180秒读，16MiB单响应上限；保存实际字节、响应头、Content-Length若存在、SHA和UTC。无Content-Length写null，不推测官方大小。

网络/408/429/5xx同URL最多3次有界重试，Retry-After最多60秒；401/403/404/410或其它非200、参数/单位/UTC/格点/形状/有限值异常立即BLOCKED，原样保留part与诊断，不换端点或猜路径。相同成功URL只有在SHA/大小/完整解码/样本身份一致时才复用。part不当成正式数据，保留后同URL重下。

新目录STOP_REQUESTED若出现则不发新GET，已开始有界请求收尾。旧v011的STOP标志保持；本次获准的有限补齐不是恢复v011，tvt继续PAUSED。完成后退出，不建立新自动化。

## 验证与交付界限

独立离线程序从全部原始ASCII逐点重读，不复用生产解码器；重核DAS字段/单位、UTC、最近格、28320值、5664键、236日、全日期无缺、来源SHA/大小/HTTP200；对生产解码器做参数/单位/UTC/网格/源/形状/截断/重复/非有限/填充值等负例。前后180项既有保护条目及新增来源一致。所有已有数据列、标签、预测、日期/评分掩码保持不变。

输出 raw_5fields.parquet、field_provenance.parquet、expanded_dataset_with_itu_inputs.parquet、coverage_by_date.json、验证收据和报告。整合表新增五字段，不修改原表。不得把下载完成当成组合ITU实验完成；零训练、零模型推理、零偏置校准、零晴空参考重估，仍不输出新的衰减排名。
