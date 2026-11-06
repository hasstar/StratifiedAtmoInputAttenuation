# Revision 014 断点续跑与验收手册

工作区：`E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`。Python：`C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`。所有运行均从工作区执行。此项目原 H5、ERA5、Run_011、现有数据集与旧实验只读；不要在它们的目录写文件。

## 当前阶段顺序（绝不越门槛）

1. `python -B Code\Hourly_Attenuation_Experiments\Revision_014\run_selection.py status`，如果十模型作业仍在运行，仅检查进程/结果数，不重复启动。10 模型×4 配置＝40 个正式验证调参记录；每条包含可定义的全部逐轮曲线。选好配置及轮数后，10 模型×3 种子＝30 个定长重训验证记录。若进程确实失败，再用完全相同的 `run_selection.py run` 断点续跑；脚本按指纹跳过完整结果。
2. 40/40、30/30 后运行 `validate_selection.py`，要求 `model_selection_qa.json` 为 PASS。再运行 `document_results.py model`，核对 `MODEL_REPORT.md`、`training_parameters.json` 和 `ten_model_rounds.parquet`。十模型正式选择只基于验证集 50/50 规则；测试不参与。
3. 运行 `run_layers.py preflight`，确认胜出模型及参数。再运行 `run_layers.py baseline`，要求 `layers/baseline_cache.json` 存在。基线按天 5 折折外残差，只用正式训练集。
4. 运行 `run_layers.py subsets`。如果仍在运行，只做只读健康检查，绝不同时启动两个写入者。每个 `layers/subsets/mask_XXXX.json` 是独立重训：完整特征名、3 次按自然日分组留出×最多 120 轮和正式训练逐轮曲线、α 及验证预测。必须到 1024/1024，无缺号且指纹一致。
5. 运行 `run_layers.py report`，要求 1024 组合、5120 层配对；运行 `validate_layers.py validation`，要求 `layers/validation_qa.json` 为 PASS。此门槛之前禁止运行测试。
6. 唯一一次运行 `run_layers.py test`。它会再次核对锁定验证预测后才读取测试文件，生成 `layers/final_test.json`。随后运行 `validate_layers.py all`，要求 `layers/final_qa.json` 为 PASS。最后运行 `document_results.py final`，核对 `RESULTS.md`、`layer_rounds.parquet`、汇总 CSV。

运行命令中的 `python` 需替换为上述完整解释器路径。退出码非 0 或 QA 不是 PASS，先诊断并停止推进，不要只根据部分 JSON 数量宣称完成。`run_selection.py` 与 `run_layers.py` 均有状态命令；`run_layers.py` 使用同一写入锁。

## 指纹和数据边界

十模型 manifest 在输出根的 `run_manifest.json`；分层 manifest 在 `layers/run_manifest.json`。若代码、协议或训练/验证文件改变导致指纹不一致，不得手工修改旧结果指纹或混合文件，应新建修订号重跑。测试数据在最终 `test` 命令前不得用于决策；历史曾查看/评分该测试集的事实仍须在论文中披露。训练折目标、天气标签与时间键不作为输入。雪和少数深衰落仅作描述，不用测试集挑最佳天气模型。
