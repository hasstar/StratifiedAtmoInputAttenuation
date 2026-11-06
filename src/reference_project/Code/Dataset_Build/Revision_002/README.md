# Dataset Build — Revision_002

本版本使用已验证的 `Run_011` 与 ERA5 北京时间逐小时参数，重建 2025-09-30 至 2026-08-20 的论文数据集。源 H5、Run_011 和 ERA5 均只读；生成结果只写入论文工作区。

## 与 Revision_001 的关系

- `Revision_001` 依赖存在问题的旧 H5 派生结果，任务和数值结论已失效，输出已删除。
- 本版只复用其 ERA5 时间解析、11个气压层宽表和五层等权平均逻辑。
- 实测目标改为直接流式读取 `Run_011/metrics/peak_power_dbm`，同时读取 `quality_code` 和 `source_bin_count` 完成质量分层。

## 主处理口径

1. `peak_power_dbm` 按精确时间戳去重，dBm 转为 W 后按小时等时间平均，再转换回 dBm。
2. 主目标保留质量 bit 16/32；另给出 `quality_code=0` 的敏感性目标。物理无效 bit 1/2/4/8 才从主目标中排除。
3. 只保留 `[2025-09-30 00:00, 2026-08-21 00:00)` 北京时间内的实测；无实测小时不插值。
4. ERA5 与实测按时区明确的北京时间整点一对一精确连接。
5. 保留11个原始气压层表示和论文原有五层等权平均表示。1000 hPa无有效视线交点仍为NaN，不外推。
6. 主模型表只含时间、冻结的时间划分、接收功率目标和ERA5预测变量；同一时刻的频谱形态、质量码和源频谱点数不进入模型输入。
7. 同期ERA5到同期功率属于 estimation/reconstruction 数据集；本版本不生成未经证实的未来预测标签。
8. 参考功率校准冲突解决前，不生成正式衰减标签。

## 冻结划分

- blocked：训练集截止2026-04-30，验证集为2026-05-01至2026-06-30，测试集为2026-07-01至2026-08-20。
- strict_24h：在训练—验证和验证—测试边界之前各保留24小时隔离区。
- 下游缺失值处理、归一化、特征筛选和调参仍必须只拟合训练／验证区间。

## 构建

```powershell
D:\Anaconda202410\python.exe build_matched_dataset.py
```

脚本拒绝覆盖已有 `Outputs/Datasets/Revision_002` 或同名临时目录。

## 独立验证

```powershell
D:\Anaconda202410\python.exe validate_matched_dataset.py
```

验证包括：构建产物校验和、输入文件身份、源目录未变化、输出时间键、冻结划分、模型列无同刻信号泄漏、抽样小时功率独立复算、全部诊断指标抽样复算、ERA5地表／11层／五层来源一致性和月度覆盖复算。

## 主要输出

- `measurement_hourly.*`：主目标、质量分层、采集分辨率和覆盖率。
- `era5_hourly_pressure_levels.*`、`era5_hourly_five_layers.*`：完整7,800小时ERA5表。
- `matched_hourly_*`：审计用的实测—ERA5完整连接表。
- `model_table_*`：不含同刻频谱派生信息的无泄漏模型表。
- `run011_hourly_diagnostics.*`：频谱指标逐小时描述性汇总，不是主模型输入。
- `feature_sets.csv`：S、S+B、S+B+L及11气压层敏感性特征清单。
- `time_splits.csv`、`split_summary.csv`：冻结时间划分。
- `monthly_coverage.csv`：逐月覆盖和质量统计。
- `data_dictionary.csv`：字段来源、角色、单位和是否允许作为主模型输入。
- `run_manifest.json`、`QA.md`、`checksums.sha256`：构建证据。
- `VALIDATION.md`、`validation_report.json`、`validation_checksums.sha256`：独立验证证据。
