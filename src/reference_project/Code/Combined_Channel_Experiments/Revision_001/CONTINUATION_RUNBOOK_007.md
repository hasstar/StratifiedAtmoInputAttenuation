# T012 continuation 007 — Top10 视线最近发布网格、优先减量、零训练

2026-10-15人工确认“接受最近网格近似，优先减量”。本文件及T012最新段落优先于006双线性/宽包络下载、旧Top1/005；003零训练和全部原始/历史结果只读保持。Top10不变，气压面位置仍沿卫星—地面站视线，不改为站点垂直柱。

## 唯一正式入口

cwd项目根；新输出`Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_nearest_v006/`。唯一正式下载run：

```powershell
& 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B 'Code\Combined_Channel_Experiments\Revision_001\weather_download_top10_nearest_v006.py' run
```

只读入口末尾改`status`。一次性initialize已经完成，绝不重跑。旧LOS v004、失败预检的nearest v005、Top1、v002、v001均不恢复run/smoke/prepare/initialize。任何修复另存新版本，不编辑冻结运行代码、计划或旧QA。new v006不是旧CONTINUATION_RUNBOOK_006；当前指引为007。

- 下载器SHA256：`221215512e7e7497dda4fe79199b3eb03491a64a610ec853b1e08f0058525f72`。
- 最近网格核心nearest_los_core_v002.py：`d678625c880ad00f1ad1425231002ad775b8c69d647fd9c745b3417fd22e7fee`。
- 测试SHA256：`36cef1b53e8b68c7d5e6d8936d77dc9016b45b2c315cd06c23d22d10b4b62212`。
- 独立审计audit_top10_nearest_plan_v002.py：`9c6ad7cd687434f3f4043a2c18702e20bde27e860fddc4bb7c40675ca90a3b07`。
- plan fingerprint：`a3de9e41ce10dec6db7499423dcc1b030dd3f02ace19b613ce841986d404e5bd`。
- plan SHA256：`fecc5fe9119fa4354a0617def1e039f690578bbba01495390b7e1710857c44f2`。

44个新离线测试+20个原传输恢复测试共64项PASS；独立计划/原日历/1090个初始射线交点/精确网格请求打包/1204几何来源SHA/266受保护旧文件SHA审计PASS。只是离线和计划QA，不能称完整字段、真实逐小时交点或模型兼容QA。nearest v005首次独立审计因高海拔站点高于850hPa初始猜测而失败，代码/计划/失败原样保留，不启动也不覆盖；新v006只修正初始猜测不低于天线+1m，实际气压面低于天线仍无效。

## 实际减量规则

模型8个压力特征对固定为：1hPa z/u；5hPa o3/u；500hPa o3；650hPa u；850hPa o3/r。必要几何辅助仅z@5/500/650/850，合计12个压力变量—层对（原笛卡尔20对）；surface仅CAPE和maximum_2m_temperature_since_previous_post_processing两列。目标变量直接下载，不用其他气象变量推导，不下载雨/雪/云诊断。

使用ERA5 CDS发布的0.25度规则网格，不请求再网格化`grid`参数，不声称这些是ERA5底层未插值的原始谱/高斯网格。最近邻明确定义为经纬坐标逐轴四舍五入、正好中点取北/东，日界线归一化；结果必须标注“视线最近发布网格近似，与原模型双线性采样不同”。没有空间/时间插值，没有站点合并；同格下载值可以共享，但218条链路身份保留。

按原77父项固定顺序（33压力地区月/22824小时，含地面补时共45663日历请求小时）逐父项推进，不按模型结果选日期。每层用高度猜测作为起点，先查询已有载荷中该点当小时z；若缺，仅下载该点/层/小时z。由该z按原单位换算求与前向WGS84射线的交点，再选最近发布网格；若格子变化，仅补查所需新点z，最多8轮。只有z求交后仍落回同一最近格，才请求该小时/点的实际Top10变量。固定猜测绝不当作最终逐小时等压面。循环、不收敛、缺值、实际等压面低于天线保留无效，不插值补造。

同变量/层/日期/时刻的所需格点可打包为无空洞矩形；打包逐项核验没有增加任何网格、变量、时刻、日期或重复样本。不再下载宽区域加0.5度缓冲。每请求一个气压层，避免4×5无关组合。原日期缺口和地面补时保留；仅明确新请求过大才按日期二分，最低单日。

新点块按同一原月份内相同小时掩码合并，不机械沿用旧宽包络7日初始分批；原日期/UTC并集仍完全相同。新请求是必要样本的精确矩形，不通过重复日期或额外格点换取较少请求。

20份历史已完成载荷只读候选复用；必须SHA、CF UTC、参数ID、单位、层、实际坐标和所需样本一致才用，不能把整份旧文件直接计作新LOS覆盖。新请求还核对实际返回的全部变量/层/网格/时刻和有限值。`units/`只记录采集/最近格几何单元，不是独立字段QA或预测成功；无效行/几何待复核继续标明。`waves/`为按已下载实际z确定的不可变增量请求，含依赖SHA/所需样本指纹；必须读取，不用原201批当本版总数。

严格点级减量会增加请求数量，不能保证队列时间更短。静态未扣缓存估计：地面128请求、初始z505请求；后续z纠正/直接目标变量还需追加，不是最终批数。已有缓存可减免其中一部分。此取舍依据我“优先减量”的决定，不得宣称必定加速。

## 交接与恢复

旧LOS v004实际PID31160/创建11:02:39.819062/精确命令核验后已于11:44:31前确认停止本地调度，旧RUNNING是历史。7份完整状态/日志副本在nearest_v005/handoff_state_before，前后逐份SHA核验。没有远端取消。继承两个原ID不变：

- England February surface：`2b658003-b255-4eee-817e-3de5f05bb899`。
- 旧Top1 December pressure：`5473a85d-0cd4-4886-ad0c-702c5ed29e0b`。

两条迁移请求的原payload/request完全保留，计入最多两并行槽；不因新方案已能复用部分值而偷偷取消或重复提交。完成后按实际所需网格复用，不把下载完成等同全字段/研究完成。新worker实际信息另存launch_verification.json，不能把历史PID当当前存活。

12:00:26.505335已隐藏启动v006，历史PID25784；12:01实际唯一命令/创建时间/两次CPU3.515625→3.59375/RUNNING/OS锁匹配，stderr空。日志weather_top10_nearest_run_001_20261015_120026.stdout.log/stderr.log；两继承ID均accepted，新wave0001两个点请求未提交。England February surface首单元复用旧值，units=1仅采集记录。未来必须重查真实状态。

每轮先读T012、003、本007、PROGRESS、plan、旧geometry/model_inventory（plan中路径）、preflight_qa、plan_independent_qa、waves/units/jobs/progress/active_worker/logs/failures/partial_history/handoff。先实查全部weather_download*.py实际命令/PID/创建时间、两次CPU、共享OS字节锁、新旧SHA及载荷。存活只读健康检查，不重复启动/不改冻结代码。低CPU/accepted/running不是失败。

仅当实际下载进程均无、OS锁释放、下载未齐、指纹一致、失败已诊断且没有提交/访问歧义时，隐藏恢复上述唯一v006 run，另名日志；已完成载荷跳过、已有ID续接。SUBMITTING/AMBIGUOUS先官方只读精确对账，不盲POST；BLOCKED_REMOTE_ACCESS不释放槽。所有非大小/账户/条款问题先诊断，不换代理；继承ID失败不得分裂重提。part先比对SHA/官方大小/可读性，完整才转正式；不完整保留partial_history后同ID重下。

## 下游仍须验证

已有k_0010检查点SHA=`b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5`，BiLSTM seed17/53轮、真实LOCAL2404小时训练、10列/原生268槽；只读其已冻结预处理及目标变换，不改156网络、不补零。不得fit/partial_fit/fit_transform/微调/训练/优化器/选种子/调参，亦不重估插补归一化晴空参考CDF。nearest变化已被我接受，但完整特征/目标/单位/时间尺度兼容仍需核验。

FRA-00609发布方向与标称卫星相差0.5424度仍待复核，不放宽0.5阈值、不改原指向；下载以发布方向定位，预测继续标记待几何复核，其他214也不声称精密星历。地面字段在站点最近发布格取值，气压面绝不站点垂直取值。

GBS原秒级relative-clearsky可复用旧ClearSky_Alignment_004小时标签；OpenSat原1分钟RSL平均口径未知，不插值造秒真值或重拟合分钟CDF。按来源/频段分别评价，不合并跨频段RMSE；天气分组缺诊断则不可评估。其余模型族仅已有且兼容才比较，缺口明确列出。历史测试访问不称新盲测，随机场景不称确定性秒预测；LOCAL全冻结不重跑，CRPS0.130487较自身0.162335降19.62%、90覆盖73.98%不是已校准。

保持原tvt、原线程、每15分钟，不另建任务；正常无变化安静，阶段完成/失败/需决策才通知。下载/局部试跑/字段QA均非整项完成，原范围外部评价、独立QA、图表和中文报告及进度齐备，或我明确缩减最终范围且满足后，才写总completion并暂停同一tvt。
