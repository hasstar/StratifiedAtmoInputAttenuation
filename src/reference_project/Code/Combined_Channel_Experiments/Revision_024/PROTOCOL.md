# Revision024 — 英国 E-SST 现存输入与连续雨量上下文审计

T012，原线程 01a0bda1-92d1-7832-9508-60f4e0ec1634。仅做本地资产清点、逐键覆盖与量纲核验，不运行物理衰减、模型推理、评分、训练或网络请求。communications-formula-auditor 区分“变量存在”“时空匹配”“模型可用”三种结论。

## Preflight / Reuse / Conflict

承接已读根 README、T012、Revision001/PROGRESS 及最新023 PROTOCOL/HANDOFF/QA。根规则SHA未变，沿用已读适用原子规则。17:22:03Z复核023的1175保护源和34封存产物不变，含011五份源码和两份预测。17:28:49.5671489Z具权限CIM显示Python/pythonw/MATLAB/Octave/Rscript进程0，新代码/输出024均不存在，E盘空闲153.70GiB。仅新版本独占写；全机未登记写者仍UNKNOWN，不宣称绝对排他。

READ：023保护链及封存产物；003原始完整日期、输入表、字段来源；011表仅键和冻结评价掩码；001/external的23个旧CDS payload和008/009/011 NCAR records、原payload、DAS；008/010已有字段表；236原生GBS文件的头与几何标量（不读衰减值）；3份OpenSat4Weather2022文件仅头与时间坐标。WRITE：仅Code/Revision_024和Outputs/Combined_Channel_Revision_024。Python短CPU任务，不用GPU。原下载入口一律不运行、不导入。

复用：已有003/008/010/011输入和评价母集完整；不是重做数据集。旧CDS用实际文件magic、ZIP成员内存打开核对，NCAR用records与原ASCII头/DAS互证，不按文件名推断字段。OpenSat4Weather里的deg0l不能跨2022与2003/2004日期复用。LOCAL的2025/2026资产不跨时空借给英国。

C1/C2/C3=POTENTIAL/LOW（独占新目录、输入/旧结果SHA）；C4=CONFIRMED/MEDIUM（缺700hPa双分量和合格零度层，已有650hPa风不是替代）；C5=NONE（复用原掩码，缺项不缩范围）；C6=POTENTIAL/LOW（顺序读取小payload与头文件）；C7=CONFIRMED（LOCAL假设待确认、UK输入、RMOP/PRM实现门槛）；C8=CONFIRMED/LOW（新增可续接输入审计）。Decision=PARALLEL_ALLOWED仅表示隔离写域，不授权其他agent或重复worker。

## 冻结审计合同

1. 236份native_days逐日期展开24小时，严格等于003和011的5664个(station, UTC interval start)键。只读011的primary_score_eligible/common_score_eligible，逐键确认共同掩码是合格标签与有限非负液态雨残差的交集；不读标签/预测数值，不生成新排名。
2. rain = 3600*MTPR - 3600*MSR，源kg m^-2 s^-1换算为小时平均mm/h（1 kg/m²水当量=1 mm）。源forecast valid time为区间结束，beacon start=valid-3600；不是累积差分，不当R0.01。总量和固态均非负且残差非负才视为有效；负值原样保留，不容差裁零。
3. 使用全部已存在、通过源SHA/DAS核验的NCAR小时雨量记录，按原计划的地面最近网格匹配两站。评价外的额外雨量只作为连续性证据，不新增评价日期。缺源/非有限/负残差均切断连续上下文；精确零值作为完整一小时无雨。连续湿小时事件的前后均须有相邻有效零雨小时才能称双侧完整。此处只诊断事件边界，不分对流/层状、不计算H/v或E-SST。
4. 风速需同一站位和有效时刻的u700、v700；v=sqrt(u²+v²)，不是u650或仅u。零度层需原生h0的明确垂直基准，或T/z剖面与地面压力、唯一交叉及地形/高度基准合同。现存1hPa z不构成可用零度层。频率/极化/名义几何只核已有证据，不外推新模型。
5. 每个评价小时保留缺项状态。最低逻辑输入清单为u700/v700/合格h0各5664站点小时，共16992字段-站点小时；不等于请求网格数/字节数。若改用T/z剖面，压力层集合尚须按物理包络明确，不能预称有限层足够。前后额外上下文量依赖未知H/v以及事件封闭，不能任意填一天并声称充分。
6. 独立QA不导入生产脚本：用h5py核CDS变量/坐标和ZIP头；用原ASCII数字行及独立解析的forecast坐标重建雨量键值，与生产canonical records逐值核对；独立重建完整日期、雨量事件/逐小时台账并复核掩码。数值雨量容差1e-12 mm/h预先固定，UTC/成员/状态要求精确一致。所有指标仍沿用既有结果。
7. 输入/源码/权重/STOP/tvt SHA前后不变，新输出独占创建，进程PID/命令/环境和锁释放记录入盘。失败保留，修复另存。LOCAL适配问题已在023询问，本轮不催问、不默认为批准；MTS=NOT_COMPUTED_USER_DECISION，null数值/排名。天气保持停止。七类整体仍未完成，自动化不修改或暂停。

## 执行锁

生产持有本版Windows字节锁并独占新目录；独立复核只新增本版QA文件。最终封存再次持锁并先哈希锁文件，避免自锁读取失败。最终状态仅以final_verification.json为准。
