# Revision 018 断点续跑手册

工作目录：`E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`。Python：`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`。代码在 `Code\Hourly_Attenuation_Experiments\Revision_018`，输出仅在 `Outputs\Experiments\Hourly_Attenuation_Revision_018_Correlation_090`。

每次先读 `PROTOCOL.md`、`PROGRESS.md`、相关 `run_manifest.json`，执行 `run_experiment.py status` 与 `layer_pipeline.py status`，核查后台进程命令行和 CPU、stdout/stderr、记录数量和指纹。进程仍运行只做只读检查，绝不重复启动；确认原进程失败或退出后才诊断并用同一命令断点续跑。不要修改已冻结的协议或写入旧版本。

1. `run_experiment.py prepare` 和 `verify_features.py`：独立核查筛选指纹、348 列完整审计、保留列成对 `|rho|<0.90`、十层非空。不通过则停止拟合。
2. `run_experiment.py models`：按冻结四配置×十模型，保存 40 试验与 30 三种子验证拟合，运行中不可重复启动。退出后 `run_experiment.py verify-models`，只有 `model_selection_qa.json` PASS 才进分层。
3. `layer_pipeline.py preflight`、`smoke`，然后 `subsets`：自动先建基础折外残差缓存，断点保存 1024 个掩码。进程仍运行只做只读检查。
4. 1024/1024 齐全后运行 `layer_pipeline.py report`、`verify`；核对 5120 对有/无层比较、逐组特征名单和验证指标复算。层 QA PASS 后才执行一次 `layer_pipeline.py test`，再 `verify-all`。
5. 最终 QA PASS 后运行 `document_results.py`，更新进度，记录总体、雨天（含强雨）、干燥阴天指标与测试访问限制；停用本聊天每 15 分钟续跑任务。

原始 H5、ERA5、Run_011、现有数据集和全部旧实验只读。用户允许删除旧实验，但新版本无需先删旧结果；若未来清理，只能精确核对目标路径、确认不再被进程或文档依赖后进行。
