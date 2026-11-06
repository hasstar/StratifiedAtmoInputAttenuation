# C3运行与交接手册

只在论文工作区用`D:\Anaconda202410\python.exe`。原A/B/C1/C2和慢变包只读，禁止修改冻结源码或重跑旧阶段。

1. `python Code\Fast_Fading_Experiments\Revision_002\test_stationarity_numerics.py`：仅合成案例，8项数值检查，PASS才可执行下游。
2. `python Code\Fast_Fading_Experiments\Revision_002\freeze_generator.py run`：打包全训练CDF参照和片段库，保存`final_generator/model.json`、库和`qa.json`；已有包只复核，不重写。
3. `python Code\Fast_Fading_Experiments\Revision_002\freeze_generator.py verify`：独立复算CDF功率矩、合法片段起点数与秩重排功率集合。
4. `python Code\Fast_Fading_Experiments\Revision_002\run_stationarity.py run`：冻结C3 manifest，五日期折，每小时128实现、8小时检查点；每窗同时计算主/半双银行参考尾概率。结束自动verify。
5. `python Code\Fast_Fading_Experiments\Revision_002\run_stationarity.py status`：只读计数、QA/选择及worker元数据；精确Python命令行、OS字节锁、CPU和日志仍需从操作系统独立检查。
6. 全2404小时、653430状态齐全且QA PASS，进程结束后执行`run_stationarity.py report`，保存逐小时/逐日期/各天气/质量/窗口的相容比例及连续日块区间；训练日期冻结工程建议。无任何满足窗也必须保存，不改阈值追分。

新增长时worker日志为`stationarity.stdout.log`、`stationarity.stderr.log`。必须隐藏窗口启动；同命令进程存活时只读健康检查，绝不重复启动。只有确认失败、无相同进程，且源码/计划/缓存指纹不变，才以同一run命令从8小时检查点恢复；保留failure审计与原日志（恢复用另名日志）。OS锁复用本版worker.lock，旧PID不作为存活证据。

每检查点保存原B窗口状态、六个真实端点、全128个模拟参考端点（二进制NPZ）、主/半参考尾概率与相容标记、折号/天气/每实现种子和接缝数。按日汇总，秒/窗口/模拟实现不能当真实独立日期样本。

QA复核源B端点、全部唯一参照窗的双银行概率、全部源/缓存/输出SHA256、完整小时和窗口键。全部QA后才report；report并不自动读取正式验证/测试，也不代表快变实验全部完成。下一阶段须另存最终评价与100ms/设备/谱配置/合成计划及实现，冻结所有规则后才评分343验证和687测试小时。后续的窗口参照不能从这些结果回调。
