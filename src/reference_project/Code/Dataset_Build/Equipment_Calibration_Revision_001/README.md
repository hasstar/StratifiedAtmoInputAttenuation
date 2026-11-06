# 本地设备更换增益校准

运行 `python build_equipment_calibration.py`，仅从已验证的小时与快变版本读取，
新建 `Outputs/Datasets/Revision_006/` 和
`Outputs/Datasets/Fast_Variation_Revision_002/`。目标存在即拒绝覆盖。

校准口径：北京时间 2026-05-14 为分界；两段全部 ERA5 `clear_dry`
候选小时的已汇总 W 功率分别等权求均值 `a1`、`a2`。后期所有实测
功率在 W 域乘 `a1/a2`，等价于 dBm 统一加 `10log10(a1/a2)`；
绝非给 W 加一个固定数。该操作不按模型划分重新估计增益。

小时表只改变功率目标及其派生的绝对低功率代理。快变 100 ms/1 s
数据中，记录功率和对应小时慢变功率同乘增益，归一化残差保持不变；
窗口表的绝对功率同步缩放。新版本保持旧版表的列、类型和行数。

构建完成后运行 `python validate_equipment_calibration.py` 做小时与快变
全量独立验证。然后运行 `python refresh_three_source_stage.py`，从原
三源指标整理包派生 `ClearSky_Alignment_Revision_003`（只更新本地功率，
两张外部数据表逐字节复制）；用 `python validate_three_source_stage.py`
独立核验。源数据、原 H5、Run_011、ERA5 和已有
版本完全只读。失败的版本不会覆盖重试，应保留诊断并另建版本。
