# Revision020：正式 PDF 公式核验与 E-SST 小时数值核心

2026-10-17；Task=T012；/root；原线程 01a0bda1-92d1-7832-9508-60f4e0ec1634。本轮新增依据为刚完成获取 QA 的 IEEE 正式全文，不重跑已交付的组合 ITU 或学习模型。

## 预检、复用与占用

已读取根 README 适用规则及补充说明、项目/WF01、T012、Revision001 PROGRESS、Revision019 PROTOCOL/HANDOFF/QA、Revision011 协议/完成/QA、Revision013 输入审计、Revision014 PRM、Revision015 RMOP 和新文献交接。旧 T012 顶部十模型及 PAUSED 是历史，最新人工七类名单及 MTS 决定优先。

2026-10-17T15:33:36.8295522Z 具权限 CIM：Python/pythonw/Matlab/Octave/Rscript 为 0。T006/T010 相关锁为历史 RELEASED；最近10个线程及全部置顶范围的任务检查，仅当前“数据集重新构建”线程匹配本项目；其他未加载线程和外部写者 UNKNOWN。E盘空闲约165GB。Code/Revision_020 与 Output/Combined_Channel_Revision_020 创建前均不存在。

Revision011 五份计算源码 SHA 和两份预测表 SHA 与原冻结记录一致；524 项旧来源中523项原值一致，唯一差异为 tvt 配置，原值 10f9dcb0e27d52c6c8494502760027655634e1d008a29440eef174b85b8b0f45，经已批准001/002调度记录变为 bc5a33cdf2b49fe183de87cec7ec9cb1ac9a1a351c4f1691557c7587bf10891e。该差异记录为授权变更，不改旧清单。全部组合 ITU 数值复用，不重新计算或评分。

READ：上述已冻结代码、模型/输入/预测的旧SHA清单、报告、已取得PDF和只读输入台账。WRITE：仅全新 Code/Combined_Channel_Experiments/Revision_020 与 Outputs/Experiments/Combined_Channel_Revision_020，含PDF页渲染、公式/输入契约、新合成单元测试、独立标量复算和交接。原始气象、模型、结果、固定索引、STOP及自动化不写。

C1/C2/C3=POTENTIAL/LOW：新写域、create-new 输出、旧源前后SHA。C4=CONFIRMED/MEDIUM：雨衰与总相对衰减的目标不同；数学核心不冒称完整模型。C5=NONE（本范围）：推进新全文可解决的公式门槛。C6=POTENTIAL/LOW：CPU短任务、少量磁盘，无GPU/网络/采集。C7=CONFIRMED：E-SST站点输入/高度/极化及真实适配门槛仍未全齐；PRM系数、RMOP实现/输入仍缺。C8=CONFIRMED/LOW：组合ITU、学习预测、已有PRM核心和输入审计复用，仅新增未有E-SST核心。Decision=PARALLEL_ALLOWED，只限上述隔离写域；Reuse=COMBINE/ADAPT，不复制下载器或训练入口。

Execution Lock=LOCKED；Holder=/root、上述线程；Started=2026-10-17T15:36:00Z；Path Base=项目根；本版两个新目录含子目录 WRITE，旧源READ，CPU/磁盘SHARED。声明为逻辑占用，不声称OS排他；最终 final_verification.json 记录释放。出现同worker即只查进度，不重复启动；实际执行为短时单进程。

## 数值核心的限定契约（无真实预测）

- 依据 E-SST 正式PDF第5–6页 §III、式(8)–(10)、Fig.7；将原始1分钟雨强改为真实3600秒的分段常数小时雨强，只实现**显式命名的小时雨量数值核心**，不恢复分钟观测。
- 事件按连续正雨强小时组成，整小时零雨强提供至少1小时干段；峰值严格大于10 mm/h为对流，否则层状。缺失/负值拒绝，不作为干段；记录序列首末未被干段包围的删失事件。整事件分类用完整气象上下文，不能宣称因果预报。
- 层状等效高度 h0+0.36 km，对流 h0；必须先将h0及天线转入同一绝对基准，本核心显式接受 m_AMSL。AGL、未核实地形和零度层低于天线不自动换算/裁零。
- 单层数学式 A=(1/cos(theta))*integral(k*R(l)^alpha dl)，水平积分长 H/tan(theta)，d=v*t。核心显式使用每个锚点冻结的风速/高度及前向雨时间窗，小时内分段常数；这是数值适配约定，不自动代替真实模型的适配验收。90度采用解析极限 k*R^alpha*H。
- k/alpha无默认值，单位、频率/极化和P.838系数来源由真实接入另验。本轮测试系数全部为标明的人工合成夹具，不进入任何排名，不挑种子。
- 时间窗超出连续输入即拒绝，无外推、补零、跨缺日拼接或边缘复制。高度0为零路径数学极限；实际站点雨层合法性另验。风速必须正、单位m/s；雨强mm/h、几何高度km、仰角degree。
- 直接E-SST只给雨衰，不把论文的气体/云标签分离或带测得信道的频率缩放分支接入。本工程相对晴空总衰减仍有目标对齐门槛；若未来加入固定气体/云分量，须明确命名组合适配并只计一次，不能拟合标签偏移。

## 验收与不可执行分支

对真实输入不运行模型。先通过公式视觉核查、量纲/零输入/均匀雨/时间窗/事件阈值/仰角极限测试，再由不导入生产核心的独立时间积分实现复算合成病例。测试通过只证明这部分数值算子，不证明原模型复现或小时适配有效。所有已评分结果、掩码和雨/阴排名继续引用Revision012；动态继续引用Revision017/018。

新PDF对PRM系数和输入要求重新核对；没有数值系数包就不算真实PRM。RMOP不以简化公式替代。MTS=NOT_COMPUTED_USER_DECISION，无数值/无排名、非当前待要事件库；不重启。天气采集/训练/重新归一化/参考和CDF重估/外部发送本地数据均为0。仅所有本轮局部QA通过后交付本版，七类整体仍未完成，tvt不因此暂停。
