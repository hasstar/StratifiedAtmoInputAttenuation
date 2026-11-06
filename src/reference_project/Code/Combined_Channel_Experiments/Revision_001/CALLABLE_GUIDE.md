# 本地慢变与快变组合调用

该接口加载原Rev028全156列BiLSTM、seed17、53轮与原FastRev002四天气CDF/256秒块库。一次调用返回3600个秒级随机场景值，不重新训练，也不改变既有指标。原始46轮基础模型仍保留为对照。

## 输入与输出

调用`FrozenLocalChannel()(features, weather, hour_key, repetition=0)`。`features`必须含冻结名单中全部156个键；原始缺测可显式给`None`，由原训练期插补器处理。不能只给RFE前10个变量就调用原156列网络。`weather`为`clear_dry`、`rain`、`dry_overcast`或`other`，是提供的小时条件；`hour_key`用于可复现随机流。

输出包括小时慢变衰减、3600个快变功率比、快变dB衰减、二者相加的组合衰减，以及接收功率dBm/W。还返回组合分布的1/5/50/95/99百分位和15次抽块的来源小时/起点/使用长度。14次使用256点，第15次从一段合法256秒片段中使用前16点。每块均取自对应天气的训练小时，不跨真实缺口。

组合关系为`A_scenario(t) = A_hour_prediction + F_random(t)`。CDF理论总体功率矩为1；每条有限曲线的平均功率可能偏离1，接口保留偏离量，不再逐条归一化。小时预测不是随机曲线的平均值，随机场景也不要求逐秒与实测重合。

## 数值复现

旧神经网络以float32、最多512条一批进行推理。单条矩阵核可产生约10^-8 dB的舍入差。新接口将同一输入复制到512行，仅取首个输出，保持批量数值路径；这不是512条新实现，也没有取预测平均。8个固定示例的原慢变值与快变随机流均按严格检查完全复现，见`local/callable_qa.json`。

论文指标始终来自已保存的批量预测和正式评价，不由接口新调用值替换。调用预检首次舍入差记录保存在本轮`local/preflight_history/002_callable_batch_failure.json`，旧模型/预测未变。

## 程序入口

`callable_local.py verify-examples`验证8个预定最早验证/测试天气小时。检查通过后重用已有新示例，不覆盖旧FastRev002示例。部署场景命令为`callable_local.py scenario --features <156键JSON> --weather rain --hour-key <小时标识> --repetition 0 --output <新local/scenarios文件.npz>`；只允许本轮场景目录写入，不覆盖已存在文件。

该输入若采用再分析气象，结果属于给定气象条件的重建/随机场景。不能据此称实际未来气象可得，或声称12:38确定性逐点预测。
