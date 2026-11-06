# Revision033 — T014执行层文件占用修复

基线：完全冻结的Hourly032科研管线与其manifest，另存本版本，不修改032代码/协议/模型/QA/失败。2026-10-19 02:50:21+08原PID9228在更新progress.json时os.replace抛WinError5，03:前已确认真实进程退出、writer.lock释放。20状态QA已通过（每模型156和155两个状态），无已提交待QA状态。TabM k154 fold1恢复检查点epoch35、优化器/随机生成器/RNG完整，identity=ef1c82430c29602f09d3d6aeef16f89c8d25dc1f41bd05ca9a01f022694f2c15。全部1028个已冻结输入SHA未变化。

修复仅限执行状态I/O：给可变JSON原子替换增加有限的Windows占用重试，使用唯一临时文件名避免覆盖失败残留；progress写入最终仍受占用时把同一进度及错误另存deferred_progress事件，不让非科研显示文件中止训练。已提交结果、模型、指标、删除决策的写入和错误处理不变。active/status更新不能降级，永久拒绝仍报错。新增运行状态纠正初始化PREPARED_NOT_STARTED可能残留的问题，只影响显示。

复用032的全部训练/优化器/数据/折/特征/种子/轮数/复用/评价/QA/报告函数，运行时仅替换runtime.update与runtime.progress执行函数，不编辑任何原文件，也不全局替换os.replace。不新增模型、不重新选种子/轮数、不改变科学manifest和job identity。恢复epoch36起，未保存的中断epoch按旧随机状态确定性重放，不用结果挑选检查点。

Conflict Check：真实进程与锁已释放；新Code/Hourly033和Outputs/Hourly033_Execution_Repair原先不存在。C1/C2/C3中等风险仅在旧032允许的可变状态/恢复文件与尚未占用结果槽接续，由同一writer.lock排他保护；旧已提交结果、失败和原始输入只读。C4/C5科研合同不变；C6单CPU两线程单worker，不安装服务；C7恢复35轮检查点先验SHA/identity检查；C8不重做20已验收状态。Decision=SEQUENTIAL_RECOVERY，修复验证前不启动训练。

写入范围：本新033代码/修复证据、032既有授权的可变状态与未完成结果。失败json、stderr、原progress临时文件和中断checkpoint先复制到033证据目录并记录SHA；复核副本一致后才允许旧可变检查点接续更新。系统/天气停止边界不变。

验证：合成JSON无占用、实际Windows共享句柄拒绝旧replace而新重试成功、永久拒绝保持旧目标及保存临时证据、非共享错误不吞掉、进度降级保留详细事件；恢复检查点身份与优化器/RNG检查、已完成20状态QA摘要SHA复核、032输入SHA全核验；不做计时试跑或额外科研拟合。封存repair_manifest后入口execution.py run；原032/run.py不再直接用于接续。
