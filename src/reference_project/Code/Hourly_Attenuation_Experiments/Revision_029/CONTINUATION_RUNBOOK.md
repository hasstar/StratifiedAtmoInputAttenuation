# Revision 029 续跑

Python：C:/Users/SDR_HH/.conda/envs/pytorchmy/python.exe；所有命令加-B；工作目录为论文根目录。

先完整读PROTOCOL.md、PARAMETERS.md、PROGRESS.md、此文件及新输出run_manifest/阶段QA。执行run_analysis.py status --deep，同时用Win32_Process核对精确Revision_029命令/PID/创建时间、CPU两次采样、stdout/stderr和OS锁；active_worker或旧PID不单独证明存活。任何匹配worker存活只读检查，不重复启动、不改冻结代码。

主命令：python -B Code/Hourly_Attenuation_Experiments/Revision_029/run_analysis.py all。它依次完成层和RFE训练，再自动调用独立QA，冻结训练内方案，然后仅加载模型评价及出报告。出现任何错误停止并记failures，不跳过QA。恢复前确定无相同命令进程、锁释放，冻结源码/输入哈希一致；相同all命令从完成state检查点恢复，日志另名，旧日志保留。主状态检查点内包含3折和正式模型哈希；不得把只有模型文件而没有state的片段称为已完成。

独立子命令：layer-run、rfe-run、verify-train、evaluate、verify-evaluation、report。需要全部两条训练路径QA PASS和选择冻结才能evaluate；不能提前读测试去改变删除路径或状态数。report只有最终独立评价QA PASS后运行；报告独立核验PASS后方可宣告全部完成。

预期：layer1024 states/1024 models/3072 fold_models；rfe156 states/156 models/468 fold_models；155次删除、36735次置换；各训练53轮；所有模型种子17。评价layer1024+rfe156记录，各含343验证与687测试预测、5天气RMSE/MAE/Bias。对mask1023和k156进行Rev028锚点逐预测核验。

日志新建时用隐藏Start-Process。主worker正常健康且未阶段完成保持安静；仅阶段完成、失败、需由我作出决定、全部完成通知。完成全部QA/报告/进度后暂停原tvt任务。测试历史访问、同日训练重叠不能抹除，不能据测试曲线选k、强制分层列入前10或反向修改参数。
