# T014 六指标十模型曲线图：TabM加粗与荧光强调

创建：2026-10-20 09:42 +08:00。执行者：Codex / root / T014_Six_Metrics_TabM_Glow_20261020_001。

## 本次范围与输入

按最新人工要求重画六张独立图，依次为雨天R²、阴天R²、总体R²、雨天RMSE、阴天RMSE、总体RMSE；每张十条曲线，TabM（本文模型）加粗、青色荧光晕光。横轴是保留输入特征数k=1..156，不是网络权重数；阴天仍为dry_overcast。晕光仅视觉强调，不表示误差范围，不推断TabM全面最优。

READ ONLY：Revision_034/delivery_002已验收指标及最终QA，Revision_035既有绘图脚本/协议，现有样式辅助器和pytorchmy环境。
WRITE：本Revision_036及Outputs/Figures/T014_Six_Metrics_TabM_Glow_20261020_001两个新目录。
不修改032/033/034/035源码、原数据、模型、旧结果、旧图、状态、自动化、天气停止标记。无训练、无网络、无安装、无新任务。

源CSV：Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002/tables/all_test_metrics_15605_rows.csv。
SHA256=e787db26030865517dd839dfa22bc91a7f5732a925748584395af748691b08b1。
FINAL_QA SHA256=33bc8db491146d40278b585fd3bda9508176207daf58767f51630e7f1625da51，PASS，1560状态。

仅ML_687的overall/rain/dry_overcast；各组n=687/51/158，十模型每个k完整，6×10×156=9360绘图值。R²=1-SSE/SST，非Pearson r²；保留负值。RMSE单位dB。各模型独立筛选特征，k相同不代表相同特征集。无平滑、无插值、无抽样、无新选择/调参；全部点直接连线，仅标记间隔显示。单种子17，无多种子置信带。

## Conflict Check / Reuse Check

根README SHA仍为39741aa8c69136661b4b02948b29714a3654f3f989ab7b488a54f0c02f93f7c7，与前轮完整读取一致；沿用适用规则。已读最新T014完成索引、034交接与035脚本/协议。09:41检查新代码和输出目标不存在。
C1/C2/C3=NONE：只读冻结输入，新目录写入；C4/C5=NONE：用户新绘图样式要求，不改科研口径；C6=UNKNOWN：不声称全机无其他进程，本次短时CPU绘图、无GPU/训练独占；C7=CONFIRMED：已验收数据齐备；C8=CONFIRMED：复用035绘图及034数据，新增两个R²面板及TabM视觉强调。Risk=LOW（隔离输出），Decision=PARALLEL_ALLOWED。
Reuse Decision=ADAPT。使用现有pytorchmy及Matplotlib/Seaborn，不新建环境。保留旧图。仅调整呈现，不重算/改写科研指标。

## Execution Lock

Lock Status：RELEASED。Holder=本记录。范围为本代码目录及上述新输出目录递归WRITE；冻结指标READ；本机CPU/RAM短时SHARED。Released=2026-10-20 09:45:09 +08:00。前台绘图和回读核验均正常退出，未启动后台进程或训练，不释放他人锁。

## 绘图与验收

使用matplotlib和scientific-visualization技能：颜色、线型、标记跨六图一致；10类别全部保留，TabM蓝色粗线叠加青色半透明路径描边。P3有序横轴设计参考不覆盖用户原始折线要求，不使用移动平均或柱图。三张R²图共用[-0.9,1]范围，三张RMSE共用[0,1.45]，不裁数据。x边界[0,157]留端点余量，实际数据/刻度为1..156。
导出六PNG（150dpi）和六矢量PDF、1560状态六指标CSV、QA与来源记录。图元与源表逐点核验、CSV精确回读、SHA前后不变、导出元数据检查；所有实际PNG逐图视觉检查，修订最多三轮。当前为准备阶段，未经实际检查不标PASS。正式投稿期刊尺寸和作者批准不在本次范围。

## 实际交付与验收结果

2026-10-20 09:44:06生成六组PNG/PDF及1560行六指标图源CSV。入口build_six_metric_figures.py；输出T014_Six_Metrics_TabM_Glow_20261020_001。旧四图及全部科研源产物保留。

09:45核验：六图均10条模型曲线×156点，9360图元值与原始指标严格相同；另独立回读导出CSV，雨天/干燥阴天/总体的R²及RMSE六列最大绝对差均为0。13个导出文件SHA全部匹配，源CSV和原最终QA的SHA未改变。六PNG均白色不透明背景，DPI元数据150.0124；R²三图1886×1129，RMSE三图1864×1129；六PDF均含嵌入TrueType字体，无Type3字体。

第一轮已逐一实际查看F01至F06六张PNG：标题、单位、样本量、k刻度、十项图例完整，图例不遮数据，TabM粗蓝线及青色荧光可辨；十模型颜色/线型/标记一致，负R²完整显示。共用纵轴产生的空白保留以便跨天气比较，不通过缩轴裁掉负值。可选改进是另加曲线密集区域的局部放大框，便于比较接近数值；本次原始全范围图没有需修复的版面问题，不额外添加用户未要求的面板。荧光只作视觉强调，非误差带或优越性结论。

Verification Status=VERIFIED（仅本次图表、图源及导出/数值/目视检查）；FIGURE_QA_AND_PROVENANCE.json=PASS。技能帮助统一跨图风格、冗余编码、完整数值范围和实际导出检查。无新训练、无天气/远端查询、无自动化变更、无数据外传。
