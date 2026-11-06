# Revision 014 进度

- 2026-09-30：为满足“只保留最新实验”的整理要求，制作并运行自包含的 Revision 014。`models.py` 与运行所需第三方库在本目录；原来跨实验目录调用的数据加载、排名和写入函数整合到 `support.py`。本版按相同数据及冻结规则重新拟合，没有复制旧结果冒充复跑。
- 十模型：40 套候选配置和 30 次正式验证拟合完成；选择 XGBoost；`model_selection_qa.json` 为 `PASS`。模型指纹为 `b48ada86d5317e75d1c7e8959e97a34c495341acbfe0c70ec2b64c4b0a103364`。
- 分层：基线缓存和 1024/1024 个组合完成；验证质量核验为 `PASS`，所选掩码 33（925 与 300 hPa）。正式测试 687 小时及指标复算均为 `PASS`。分层指纹为 `5e595ff1313d191deb519ec7beaa8e029a95320275a1070541fc395464786470`。
- 清理前逐条核对：40 个候选预测、30 次正式拟合预测、1024 组分层预测及四种测试预测，与旧版一致；最大差异分别为 0、约 `5.6e-17`、0 和 0 dB。完整记录见输出目录 `REPRODUCTION_CHECK.json`，其状态为 `PASS`。
- 仅在上述门槛通过后，精确删除 12 个旧小时实验代码目录、12 个旧小时实验输出目录和 2 个旧 XGBoost 实验图目录，共 26 个目录、约 605.89 MiB。永久删除，没有创建可恢复归档。清单见输出目录 `CLEANUP_LOG.json`。
- 删除后复查：`Code/Hourly_Attenuation_Experiments` 只含 `Revision_014`；`Outputs/Experiments` 只含 `Hourly_Attenuation_Revision_014_SelfContained`。模型与分层状态命令都能从保留版本独立读取；分层全量质量核验再次 `PASS`。数据构建代码、全部数据集、原始 H5、ERA5、Run_011 及数据质量检查图没有删除。

`validate_selection.py` 带有“正式测试写出后不得重新执行模型选择 QA”的阶段门槛。因此最终测试完成后的重跑会按设计拒绝，不能把这一拒绝解释为旧代码依赖或原先 QA 失败；测试前留下的 `model_selection_qa.json` 已通过。`compare_cleanup_replay.py` 是删除旧版前的一次性数值审计脚本；旧版已删除，事后请查看保存的 `REPRODUCTION_CHECK.json`，不要重跑该对比脚本。

## 同目录补充：1024 组训练后统一测试

- 根据新的逐层测试指标要求，先在 `PER_LAYER_TEST_PROTOCOL.md` 冻结阶段和规则；随后由 `run_test_by_mask.py` 生成训练计划、只用 2404 个训练小时重新拟合 1023 个非空残差模型及 Base188，并在 `layers/heldout_all_masks/TRAIN_LOCK.json` 锁定全部 1024 种方案。
- 锁定后才依次计算 343 小时验证诊断与 687 小时测试结果；两阶段 QA 都为 `PASS`，指标复算差异均为 0。逐层 5120 组成对测试结果和完整 1024 组天气指标已经保存。旧四方案中 Base188、所选两层和全十层的测试预测与本轮相应掩码逐小时完全一致。
- 审计、阶段门槛和所有输出入口见 `Outputs/Experiments/Hourly_Attenuation_Revision_014_SelfContained/layers/heldout_all_masks/AUDIT.md`。原始数据、现有数据集、原实验代码与结果均未覆盖或删除。
