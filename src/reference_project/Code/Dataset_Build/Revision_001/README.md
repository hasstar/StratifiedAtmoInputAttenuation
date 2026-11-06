# Dataset Build — Revision_001

本版本把 Run_009 的实测接收功率与 2025-09-30 至 2026-08-20 的 ERA5 北京时间逐小时数据精确匹配。

## 核心口径

1. 源 H5 和 ERA5 目录只读。
2. 复用 Run_009 已验证的 10 分钟功率，不重新扫描 111.5 GB H5。
3. 10 分钟 dBm 先还原为 W，按源实测行数加权求小时平均，再转回 dBm。
4. 无测量小时不进入数据集，不插值。
5. ERA5 和实测按北京时间整点精确连接。
6. 保留 11 个压力层宽表，并按固定高度边界生成五层等权压力层平均特征。
7. 只输出实测接收功率，不在参考功率未核清前生成衰减标签。

## 运行

```powershell
D:\Anaconda202410\python.exe build_matched_dataset.py
```

脚本拒绝覆盖已存在的 `Outputs/Datasets/Revision_001`。如需修改构建逻辑，应建立下一版本目录，不覆盖本版本。

## 独立验证

运行命令：

    D:\Anaconda202410\python.exe validate_matched_dataset.py

验证脚本从 Run_009 和 ERA5 源 CSV 独立复算小时功率、地面特征、11 个压力层和 5 个高度层，并核对输出与输入校验和。验证结果见 VALIDATION.md、validation_report.json 和 validation_checksums.sha256。

## 输出

- `measurement_hourly.*`：实测小时功率及覆盖质量字段。
- `matched_hourly_pressure_levels_all_measurements.*`：地面 ERA5、11 个压力层和实测功率。
- `matched_hourly_five_layers_all_measurements.*`：地面 ERA5、五层平均特征和全部有测量小时。
- `matched_hourly_five_layers_full_measurement_hours.*`：仅保留六个 10 分钟箱均存在的小时。
- `data_dictionary.csv`：字段来源、单位和含义。
- `monthly_coverage.csv`：按月份的实测覆盖。
- `run_manifest.json`：输入、处理合同、校验值、输出和验证结果。
- `QA.md`：人工可读质量报告。
