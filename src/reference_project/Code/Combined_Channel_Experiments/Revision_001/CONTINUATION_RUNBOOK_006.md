# T012 continuation 006 — Top10 LOS，零训练

2026-10-15最新人工要求恢复自然RFE Top10，并强调气压面参数必须在卫星—地面站视线路径上，而非站点垂直柱。本指引优先于旧Top1/005；003零训练及原结果只读仍生效。

## 当前唯一正式入口（2026-10-15 11:03 +08:00）

cwd 为项目根；唯一下载run：

```powershell
& 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B 'Code\Combined_Channel_Experiments\Revision_001\weather_download_top10_los_v004.py' run
```

只读状态将末尾改为 `status`。不重新运行prepare。旧v001/v002/Top1的run/smoke/prepare均停止授权。所有旧代码、计划、状态、载荷、模型、结果、日志、QA原样只读；运行中的冻结v004不修改，任何修复另存版本。

新输出 `Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_los_v004/`。plan fingerprint=`373cd26b98e2250487c1ab24ce3cafa92024f783ddf81dfe73c4ff8c53ece05b`；plan SHA256=`6ffc0a344d986bf4dcafa4959c71a528fd1cbe2ae8c54f166254d74609394800`；下载器SHA256=`b2ddb189f7a3b9e406e7f4ce487a73f04977e523b56cf726b5a79329dfd1fdd1`；测试SHA256=`e3e8d60c1aa1874fd32c123036ed5e3977e390a3dc181e34a7238f113382766e`；独立审计代码SHA256=`473869e4b60ad75b0fd9f7f3b05a7577712032e7fda077df2f93b2303f8c75a9`。

54项离线迁移/恢复/范围/几何正反测试PASS；独立日历/请求/迁移/几何包络审计PASS，897243密集射线点、1204几何来源SHA和203旧保护文件SHA通过。均为计划与几何包络QA，绝非完整字段或外部预测QA。前两次卫星标签枚举/检查点原生布局断言、单条指向差异失败保留preflight_history；对应v001/v002源不覆盖。

原33地区/月气压面日期、24时刻、22824压力请求小时保留；包括原地面补时共77个日历父项/45663请求小时，但这是新空间/字段方案，不代表旧77父项完成。新201初始批（压力每批最多7个原日期、地面按原月/补时），另2个继承请求。只有新请求被明确判定过大才日期二分到最低单日，不因低CPU/accepted/running分批或判失败。新压力请求采用4变量×5层的CDS笛卡尔组合；仅原8个气压面特征对进入模型，z@5/500/650/850用于交点几何，其余伴随字段不作为新预测输入。地面只保留Top10两列，不补云雨雪诊断。原地面补时仅溯源，不新增评估日期。

## 串行交接与实际worker

旧Top1 PID24648命令及创建时间09:16:49.677323已精确核验，于10:50:19停止本地调度，OS锁释放；未取消远端。旧active_worker RUNNING仅历史，不可据此重启。23份完整交接状态/日志副本位于handoff_state_before，已逐份SHA比对；旧14文件/5976小时/9月仅旧网格历史，不算新LOS覆盖。

新worker历史启动快照PID31160，创建时间2026-10-15 11:02:39.819062 +08:00；精确命令为上述v004 run，隐藏启动。日志 `weather_top10_los_run_001_20261015_110239.stdout.log/stderr.log`。11:02:45两次CPU0.515625→0.53125秒、RUNNING与共享OS字节锁匹配；以后必须重读实际进程，不以本段PID代替存活核查。

两个继承ID：November=`cbe3263c-c3ad-4573-95eb-00774a965430`、December=`5473a85d-0cd4-4886-ad0c-702c5ed29e0b`。前者11:02已续接下载81811字节，后者11:03仍accepted；继承下载不计新LOS覆盖。释放的一槽已自动提交新England 2004-02地面两列请求，ID=`2b658003-b255-4eee-817e-3de5f05bb899`，11:03 accepted。仍最多两槽，201新批中1已提交、200未提交，新LOS完成0。状态会变化，以上仅交接快照。

## 健康检查与恢复

先核对全部weather_download*.py真实命令、PID/创建时间、两次CPU、共享OS字节锁、冻结SHA、active_worker、jobs、progress、logs、failures、partial_history。真实worker存活仅只读，不重复启动/改代码/重写旧状态。实际下载进程均无且锁释放、未完成、指纹一致、失败已诊断且无提交/访问歧义时，主动隐藏恢复唯一v004 run并使用另名日志。已完成文件跳过，已有ID续接。

SUBMITTING/AMBIGUOUS_SUBMISSION先精确官方只读对账，不能盲POST；BLOCKED_REMOTE_ACCESS不等于任务终止、不释放槽。账户/条款/非大小失败先诊断，不换代理。继承请求失败不得二分重提。part核对SHA、实际与官方大小、NetCDF可读性；完整才转正式，不完整保留partial_history后同ID重下。完整下载后仍需全字段UTC/层/单位/数量/有限值/坐标/几何交点/SHA独立QA，readability/smoke/局部试跑都不能替代。

## 几何与模型固定信息

GBS 1197文件=原399日期×3站，来源字节/MD5/SHA和逐文件几何一致性已核对：经度单位degrees_west正确转负，卫星23°W；按原WGS84射线算法、标称GEO高度35786km，使用原接收机AMSL高度约定，未伪称精密星历。OpenSat逐条使用发布的azimuth/elevation（215条），不是按最近网格合并；地面字段按各站位置双线性取值。几何包络按前向射线至80km高度加0.5°插值支撑，ERA5 0.25°网格；包络仅下载用途。

压力包络NWSE：England=[51.75,-3,49.25,-0.75]；Dundee=[57,-4.75,54.25,-2.25]；France=[45,2.25,41.5,8.5]。地面包络分别=[51.75,-2,50.5,-0.75]、[57,-3.5,55.75,-2.25]、[45,2.25,42.5,8.25]。不得回退成站点垂直柱或最近邻。每小时等压面交点应前向、残差<=5m，低于天线/缺位势保留无效与NaN；越界不能静默外推/补零/扩范围。

FRA-00609（Astra 23.5）发布指向与标称GEO几何夹角0.5424155774°，超过初始0.5°诊断阈值；未放宽阈值、未改原指向。下载包络覆盖两种候选射线，该链路预测仍需几何复核/用户决定，不能宣称全部215条几何兼容；其余214条通过该标称方向核对，仍非精密星历验证。

已有模型 `Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/rfe/models/k_0010.joblib`，SHA256=`b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5`，BiLSTM seed17/53轮，真实训练来源LOCAL2404小时，不是GBS/OpenSat旧计划。模型输入顺序与冻结SimpleImputer/StandardScaler/目标参数已核验；原生268结构槽=188+5×16，是k=10自身保存布局，不给156列模型补零。model_inventory仅检查点及预处理PASS，外部完整输入/目标/尺度兼容待核验。实际Top10见plan.feature_names，模型顺序见model_inventory.feature_names，不按成绩重排/选列。

## 不变的科学边界

自然Top10固定；只使用已有k_0010及其原预处理参数，禁止任何训练/fit/partial_fit/fit_transform、归一化/插补/晴空参考/CDF重估。气压面特征沿站星射线与逐小时等压位势面求交、双线性取值；所有选定气压面的位势是必需几何辅助，不增加模型输入列。地面两列在站点取值。原日期、UTC小时和站点分区不变。旧数据/LOCAL/代码/QA/模型/结果只读，不补零改网络、不引入最近网格近似、不补云雨雪诊断。

几何包络仅决定下载网格，并非最终取值坐标；后续须按逐小时实际位势面求交，检查残差、前向射线、网格包含和单位。完整字段QA、目标与时间尺度兼容性核验之后才可做明确覆盖范围的零训练推理。GBS秒级relative-clearsky与OpenSat分钟RSL口径分别保留，不插值造秒真值、不跨频段合并RMSE、不将历史访问称盲测。缺天气诊断/其他兼容模型继续列缺口。

T012未完成；同一tvt原线程15分钟，不另建任务、不写总completion。
