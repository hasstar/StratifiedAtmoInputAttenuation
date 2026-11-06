# 安全继续：正式外部气象下载阶段

2026-10-13 16:58 +08:00另存。原CONTINUATION_RUNBOOK.md保留为历史记录，本文件更新其天气worker及单位预检入口；不改变任何模型、请求或科学评价规则。

先完整读取T012、PROTOCOL.md、PARAMETERS.md、PROGRESS.md、原runbook、本文件、EXTERNAL_PREPARATION.md和CALLABLE_GUIDE.md。核对LOCAL全部manifest/QA、external元数据、weather_plan、weather_jobs、smoke下载/单位QA及preflight_history/weather_failures。新文件仅写T012范围；旧来源、模型、预测、图表和结果只读。

## 已完成门槛

LOCAL完整交付，零新拟合，不重跑。主Rev028全156列seed17/53轮，原基础50列seed17/46轮不替换。11330组合记录、855来源指纹、全部独立QA及调用/图表/报告PASS。

smoke PID40032已于16:35:48正常结束，两个载荷VERIFIED。原verify_weather.py因sf单位字符串过窄失败，代码原样保留；失败审计在external/preflight_history/002_smoke_unit_failure.json。官方ECMWF字段表确认sf为m of water equivalent，另存verify_weather_v002.py，仅修正单位识别并增加坐标/尺寸/重复字段与负向接口检查，不修改载荷或训练规则。

v002独立单位预检PASS：7字段、28网格值；5项正反单位/身份检查。v002 SHA256为4878bca90615e004ff083e9715ef34ccd597e7d09b8c6a16f1432182be041c53；weather_smoke_unit_qa.json SHA256为74972e0e631f420e8963a0208789210c31ed0b3ac3e8d870710d7c7ef2a0360e。不要再次运行旧verify_weather.py或把旧失败抹掉。

冻结run入口复用相同smoke载荷时会刷新weather_smoke_qa.json的completed_utc。原被审计输入在external/preflight_history/002_unit_audit/smoke_download_input.json不可变保存，单位QA引用其SHA。仅更新时间刷新不是重新下载；两个载荷的SHA、字段和请求记录始终必须一致。

## 当前worker与健康检查

正式worker PID38056，于2026-10-13 16:56:54 +08:00隐藏启动：

`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe -B Code\Combined_Channel_Experiments\Revision_001\weather_download.py run`

工作目录为项目根；日志external/weather_run_001.stdout.log和weather_run_001.stderr.log。先执行weather_download.py status，核对所有实际Python命令行/PID/创建时间、两次CPU、OS字节锁/weather_active_worker、stdout/stderr、request_id、文件数量和SHA。旧PID不是证据。

计划指纹c7fa6ff824fb436477f534fcec022ef7d60aed6afd4ae507f9dd1575e0282e16，77正式请求，最多2并行。16:57的正式进度为2项已提交/0项完成；status的submitted=4/verified=2还包括两个smoke，不得混算正式进度。已提交GBS_England_2004-02_pressure ID1fc5722d-0abe-4080-852f-77f3c9a4c56b和surface ID9cc31575-8e77-4a40-8142-225c4547eb94。

同入口存活只做只读健康检查，不改冻结脚本/请求计划，不重复启动。低CPU、远端accepted/排队均不是失败。确证退出失败、没有同命令进程、锁释放、计划和代码指纹一致时，才按相同run命令从原request_id恢复；另名日志并保留weather_failures。remote failed/rejected或部分载荷先诊断，不能盲目重复提交。

下载脚本SHA256为8a6b4b7c28e2b61e6104baa86bee0528e9e125cce2f7836bf05c5799cb410c52；external_prepare.py为63fb3025aba2f7fc6e0c62f7eeb242aa9786f3a91b3c1d0cfd40bafb475a8a7a。检查新下载数量/字段、文件SHA、源元数据与这些代码指纹。

## 后续门槛

正式worker正常完成且weather_download_qa.json存在仍只是可读性通过。必须另存全部77请求的字段、坐标、UTC、单位和数量QA，核验源日期最后23时的降水/降雪对应次日00UTC、日期缺口补时齐全。不得拿smoke单位QA替代全量QA。

之后先冻结外部训练协议、参数why/how/source、质量门槛及纯合成数值QA，再允许GBS/OpenSat新拟合。Top10、站点/月份分区、五个模型族、seed17和公平有界预算遵循EXTERNAL_PREPARATION，不按测试成绩回调。GBS秒级256秒块；OpenSat原生分钟级4分钟主块、2/8分钟敏感性，不造秒级真值。旧156列网络不能直接接收10列。

三个来源分别报告总体和全部天气、质量样本数、慢变/快变/组合指标、日期/站点不确定性及所有失效。全部LOCAL/GBS/OpenSat独立QA、图表、中文报告及进度文档齐全后才写总completion并暂停原tvt；下载结束不等于整项实验完成。
