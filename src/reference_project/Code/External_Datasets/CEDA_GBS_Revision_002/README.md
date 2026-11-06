# CEDA GBS 下载与核验代码 · Revision_002

基于 [Revision_001](../CEDA_GBS_Revision_001/) 另存。2026-09-24 修订原因：首轮内容核验发现 Dundee 的缺测行常以 `time=0` 表示，虽然其时间变量另声明 `_FillValue=-1`；不能单看原始时间向量判定有效信号的时间顺序。本版增加同一行衰减非填充值筛选后的时间单调性与 0 秒交叉计数，并在独立的 `QA_Revision_002/` 输出，不覆盖首轮报告。下载脚本与 Revision_001 内容相同。

- [download_gbs.ps1](download_gbs.ps1)：PowerShell 7.6.5，读取 CEDA 官方 JSON 目录和用户剪贴板中的临时令牌；生成 `source_manifest.csv`，只下载未存在文件，已有文件先核对大小与 MD5，不匹配则不覆盖。每次验证另存一个报告。`-InventoryOnly` 只生成/读取清单；`-Limit 1` 可做单文件试验；`-VerifyOnly` 只复核本地文件，不读取令牌。
- [audit_gbs.py](audit_gbs.py)：Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1；独立复核清单中全部 NetCDF 的大小、MD5、关键字段、填充值、声明范围、时间顺序与经度方向。只读源文件，仅在不存在 `QA_Revision_002/` 时创建报告；不能覆盖已交付 QA。

本轮执行：1982/1982 文件的 CEDA 大小与 MD5 匹配；第二版独立核验 1982/1982 可读，所有按非缺测衰减配对的日时间序列严格递增。具体结果、来源、限制与文件布局见 [数据说明](../../../外部样例数据CEDA/GBS_20p7GHz/README.md) 和 [任务 T006](../../../Tasks/T006_Download_CEDA_GBS_External_Validation.md)。

SHA-256：`download_gbs.ps1` 为 `3CFF54770172D4EED1D69E3C50B317DF6A894C77A575DE8AA2AA39650C6BB334`；`audit_gbs.py` 为 `58D22C2F80EE8FD83E80E063539A6CAA5B839B4FB066213CD2BBF17B2928370D`。令牌不在代码、清单或日志中。此代码与数据核验不等于已完成跨站模型实验。
