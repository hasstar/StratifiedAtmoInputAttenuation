# C2动态阶段续跑补充

旧`CONTINUATION_RUNBOOK.md`不改动；本文件定义新增阶段命令。工作目录为论文根目录。Python为`D:\Anaconda202410\python.exe`。本机受限环境无法访问SciPy临时目录时，可用已授权的正常本机执行权限；不能把环境导入失败称为实验数值失败。

1. 读`DYNAMICS_PLAN.md`及原PROTOCOL/PARAMETERS/MODEL_PLAN/PROGRESS，检查现有后台命令行、CPU、`active_worker.json`和OS锁。
2. `python Code\Fast_Fading_Experiments\Revision_002\test_dynamics_numerics.py`：必须通过再启动正式作业。
3. `python Code\Fast_Fading_Experiments\Revision_002\run_dynamics.py run --replicates 8`：70状态，含无资格的长块状态；折内边缘支持只读复用本阶段缓存，训练库不得跨日期。启动隐藏窗口并保存`dynamics_r08.stdout.log`/stderr。
4. `python Code\Fast_Fading_Experiments\Revision_002\run_dynamics.py status --replicates 8`：只读状态。worker存活仅检查；无进程后判断正常完成/失败，确认失败且指纹未变才原命令恢复。
5. `python Code\Fast_Fading_Experiments\Revision_002\run_dynamics.py verify --replicates 8`：必须PASS。run正常结束会自动执行一次verify；不要重新run已完成部分。
6. `python Code\Fast_Fading_Experiments\Revision_002\run_dynamics.py report --replicates 8`：保存候选/天气/日期/小时评分。若8与前4第一名不一致，选择状态为`requires_monte_carlo_refinement`，按同一脚本另存运行16实现、verify、report；16与前8不一致再32。原层级不覆盖。32仍不同则保留选择不确定性，不能跳过门槛。
7. 主/半预算第一名一致、所有QA PASS后冻结生成器类型/块长；最终全训练拟合、平稳诊断、343验证/687测试、100ms与设备/配置诊断仍是后续阶段，当前脚本不读取最终验证/测试。

每次核对源、脚本、计划、CDF支持、试验记录和小时结果指纹。每折允许片段缺乏导致候选无资格，但真正数值异常必须保留并诊断。每个完整候选每个训练小时恰好一次折外评分。源实测F不改、零次原始插值。训练折的秩模板接缝是合成边界，生成评分不能删掉接缝。

精度层级/阶段更新记入PROGRESS。无变化安静，阶段完成/失败/需用户决定通知。完整快变实验全部QA/图表/报告完成后才停用`tvt`。
