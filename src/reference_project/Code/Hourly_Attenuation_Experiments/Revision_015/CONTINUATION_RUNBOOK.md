# Revision 015 续跑手册

工作区：`E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`

解释器：`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`

代码：`Code\Hourly_Attenuation_Experiments\Revision_015\run_feature_experiment.py`

输出：`Outputs\Experiments\Hourly_Attenuation_Revision_015_Event_Features`

1. 先读 `PROTOCOL.md`、`PROGRESS.md`、输出目录的 `run_manifest.json`。执行 `python -B ...\run_feature_experiment.py status`。再以 PowerShell 查看进程命令行、CPU 时间、stderr 及每个 `trial_*.json` 的指纹。
2. `prepare` 只需一次。输入或代码指纹不匹配时**停止**，不得覆盖旧结果、手工改指纹或混合版本。
3. `run` 共 72 个候选文件、216 次分组拟合。仍有同一 worker 运行时只读检查，**绝不重复启动**。确认失败且无同一进程后，用相同 `run` 命令断点续跑；已完成且指纹正确的 trial 会跳过。
4. 72/72 后运行 `report`：重新核对每个 trial 的 3 折、预测、气象组 n、指纹、选择规则，并冻结四个正式模型及验证结果。执行 `verify`，要求 PASS。正式验证只作诊断，不据此改候选。
5. 只有 `verify` 为 PASS 且报告已冻结，才运行 `test` 一次，随后再次 `verify` 和 `document`。测试集曾在历史实验查看，不能描述为新的完全盲测。
6. 原始 H5、原 ERA5、Run_011、现有数据集和 Revision 014 保持只读。新文件只放 Revision 015；不把外部异站或 100 ms 快变混入本阶段。

当前 Windows 环境对该工作区的普通受限进程拒绝新文件写入；已使用经审核的提权命令创建 Revision 015 目录并启动原 worker。后续 `run/report/verify/test/document` 若因同一原因拒绝写入，应对**上述确定的命令和 Revision 015 输出路径**申请相同的执行权限；不要修改 ACL，也不要在其他目录写结果。检查 `status`、日志、进程和结果数量属于只读操作。

若阶段不变，定时检查保持安静；仅在失败、阶段完成、需要我作出决定或全部完成时通知。
