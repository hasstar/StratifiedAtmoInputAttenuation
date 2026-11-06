# 执行与核验

本版本是一次性报告调整，不是后台训练。先读 PROTOCOL.md、PARAMETERS.md 与 PROGRESS.md。所有原版本只读。

Python：`C:/Users/SDR_HH/.conda/envs/pytorchmy/python.exe`，工作目录为论文根目录，使用 `-B`。

1. `python -B Code/Hourly_Attenuation_Experiments/Revision_030/run_alignment.py build`
2. `python -B Code/Hourly_Attenuation_Experiments/Revision_030/verify_alignment.py`
3. `python -B Code/Hourly_Attenuation_Experiments/Revision_030/run_alignment.py status`

build 只允许新建一次，不覆盖任何旧输出。若失败，保留失败信息，先查来源/实现，不启动训练或为了有利结果改数据。独立核验 PASS 与 completion.json 齐全后才宣告报告对齐完成。没有新的长时worker；原tvt维持暂停，不把Rev029旧PID当作活动任务。

独立QA不重新执行训练或模型推理；它核对旧源哈希并从旧预测复算。原Rev028/029模型重放QA继续作为各自原设计的证据。任何原源指纹不一致都必须停止。
