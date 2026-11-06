# Revision 017 断点续跑手册

工作目录：`E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`。十模型脚本：`Code\Hourly_Attenuation_Experiments\Revision_017\run_experiment.py`；分层脚本：同目录的 `layer_pipeline.py`；输出：`Outputs\Experiments\Hourly_Attenuation_Revision_017_Correlation_Pruned`。Python 优先使用 `C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`。

每次先读 `PROTOCOL.md`、`PROGRESS.md` 和已有 `run_manifest.json`，再执行 `run_experiment.py status`。检查同一 worker 的进程命令行、CPU 变化、stderr、trial/validation/mask 文件数与指纹；运行中只做只读检查，**绝不重复启动**。确认进程不存在或失败，诊断后用同命令断点续跑。

阶段顺序与门槛：

1. `prepare`：只读训练特征，输出并核验保留名单、去除原因、最高剩余成对相关、十层覆盖和输入指纹。失败就停止，不得开始拟合。
2. `models`：40 个配置试验、30 次种子重训及选模结果。每个记录必须保存完整参数、逐轮曲线和预测。进程中不重启；确认退出后同命令可断点续跑。
3. `verify-models`：逐条核对计数、指纹、保留列、验证指标与选模规则。只有 `model_selection_qa.json` 为 `PASS` 才进入分层。
4. `layer_pipeline.py preflight`、`smoke`：核对获胜模型、筛后基础与十层特征数，短样本拟合通过后开始分层正式训练。
5. `layer_pipeline.py baseline`、`subsets`：先训练基础折外残差，再断点续跑全部 1024 个 mask。`subsets` 可传 `--limit` 小批量运行。不可把部分掩码结果称为完成。
6. `layer_pipeline.py report`、`verify`：核对 1024 个掩码、5120 组成对比较、训练内部掩码选择、完整特征名单、指标复算与指纹。只有 `layers/validation_qa.json` 为 `PASS` 才执行一次 `test`。
7. `layer_pipeline.py test`、`verify-all`：固定所有规则后评价 687 测试小时，输出总体、雨天（含强雨）、干燥阴天，明确测试已在历史实验中查看的边界。再生成结果解读、更新本进度文档并停用自动续跑任务。

原始 H5、原 ERA5、Run_011、数据集及 Revision 014–016 均只读；不能删除或覆盖。所有新文件仅限本版路径。
