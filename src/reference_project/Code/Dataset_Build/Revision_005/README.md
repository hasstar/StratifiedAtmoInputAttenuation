# Revision_005：十气压面小时数据派生

基线是已验证的 Outputs/Datasets/Revision_004/。构建脚本只读取它，从 11 层主表删除 1000 hPa 的 16 个气象列，并把剩余十层按压力从大到小排列；目标、小时键、事件划分和事件标签均不重算。

在论文工作区执行：

    D:\Anaconda202410\python.exe Code\Dataset_Build\Revision_005\build_ten_level_dataset.py
    D:\Anaconda202410\python.exe Code\Dataset_Build\Revision_005\build_ten_level_dataset.py --write
    D:\Anaconda202410\python.exe Code\Dataset_Build\Revision_005\validate_ten_level_dataset.py

第一个命令只读预检；第二个命令新建 Outputs/Datasets/Revision_005/，如果目录已存在则拒绝覆盖。构建失败时临时目录保留以供诊断；独立验证通过后写入 validation_report.json 和 VALIDATION.md。所有旧版数据集、原始 ERA5、H5 和 Run_011 保持只读。

subset_catalog.csv 是 1,024 个列组合的索引，layer_pair_catalog.csv 是 5,120 个单层增量对照索引。它们都没有训练模型。后续模型脚本必须依据 feature_sets.csv 和 feature_groups.csv 取预测列，不能将目标、时间划分或事件标签当输入。10 层和 348 列是与 Revision_004 的 11 层和 364 列不兼容的新输入合同；旧结果不能直接继承。
