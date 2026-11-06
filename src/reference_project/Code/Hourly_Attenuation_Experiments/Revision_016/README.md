# Revision 016 运行与复盘

正式物理、数据、划分、评价边界先读 [PROTOCOL.md](PROTOCOL.md)。主代码为 `run_itu_baseline.py`；独立复算代码为 `validate_independently.py`。两者均只读取现有数据集、ERA5 派生小时表和 Revision_015 保存结果，写入独立的 `Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline`。

使用包含 `numpy/pandas/scikit-learn/xgboost/pyarrow` 的 Python 环境，按顺序执行：

```powershell
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py selftest
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py prepare
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py fit
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py verify
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py test
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py verify_final
python Code\Hourly_Attenuation_Experiments\Revision_016\run_itu_baseline.py finalize
python Code\Hourly_Attenuation_Experiments\Revision_016\validate_independently.py
```

`fit` 的每个训练内候选按模型名断点保存于 `cv/`；正式模型、所有列名、中位数、逐轮曲线、验证小时预测在 `formal_models.json` 与 `models/`。测试前 `verify` 写 `qa_pretest.json`，然后 `test` 只允许一次，写 687 小时的完整预测与指标到 `final_test.json`。`verify_final` 和独立脚本分别生成 `qa_final.json`、`independent_validation.json`。主结果见 [RESULTS.md](../../../Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline/RESULTS.md)。

断点重试规则：已经生成的 `run_manifest.json` 固定输入和代码指纹；如果代码/协议或输入变化，停止并创建新 Revision，不删除或覆盖本版测试结果。`fit` 允许跳过同指纹的已完成训练折，但一旦 `formal_models.json` 存在则禁止重拟合。`test` 若 `final_test.json` 存在则禁止重测。任何分析后的参数调整必须成为独立后续实验，不能宣称本版测试仍未被查看。
