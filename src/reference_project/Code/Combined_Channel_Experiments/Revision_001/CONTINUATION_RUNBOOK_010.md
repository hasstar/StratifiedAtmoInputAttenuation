# T012 continuation 010 — NCAR-only v009 正式续传

2026-10-15最新直接人工指令“全部改为NCAR下载”。本文件优先于009/v008（首次本地并发失败）及008和所有旧CDS恢复授权。保留已批准的Top10、四层固定标准高度、站星视线最近发布格、原日期/时刻/站点及003零训练。

## 唯一入口与冻结身份

cwd项目根；新输出 `Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_ncar_v009/`。

```powershell
& 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B 'Code\Combined_Channel_Experiments\Revision_001\weather_download_top10_ncar_v009.py' run
```

只读入口末尾改status。initialize已一次性完成，绝不重跑。v008及所有CDS旧版v007/v006/v005/v004/Top1/v002/v001不得恢复run/smoke/prepare/initialize。v009显式只读复用冻结v008的传输/采集函数，加载的是模块，不启动旧run；写入只到v009目录。

- v009下载器SHA256：`5b13a126108a92c3a064ad1b07ef1eaa67afbe3ebbe88dfbe3e3138b08fa3563`。
- 冻结v008基础编排SHA256：`d828e4c22ee5853103920c23c68391f283aaf1c780faeb53ab614915e4e29510`。
- NCAR传输v001 SHA256：`b548e0c21375e35baf23ab5443b46049bc2236314317542ecc289df75bdaf8b3`。
- 回归测试v002 SHA256：`667252f6e3879866babb33e462113d2b92ef059ae6a5d2ae595980d9e5e1c579`。
- 独立审计v002 SHA256：`0f0a9cf467e487e17c4468be5ab4910f9ea67fbab03333a75e86d55a962104dd`。
- plan FP：`9e799f019b833745edd8f103ea6e183c154590604eeb7c60ba0f3137970c7b61`。
- plan SHA256：`d857e59a91b672c4364d2f45a7f4441d0c06ab49861ae06d1c3d1e2a986911f5`。

41项离线测试PASS（30原传输/正反/恢复控制、6项修复版恢复复测及5项新增并发/错误回归）。独立计划审计保持77父项/45663日历请求小时、33压力月/22824小时、218链路、871有效固定交点和1无效交点；代表性2840子请求/9431唯一格点字段小时独立展开一致、390旧保护文件SHA通过。5份已完成NCAR响应重新解码与canonical records逐项一致，120唯一格点字段小时只读复用。只是计划/迁移/下载完整性控制，不是全量实际字段、跨平台数值等同或模型兼容QA。

## 失败与交接保留

旧CDS PID25488在精确命令/创建时间核验后，于17:26:08前停止本地调度；远端未取消。两个Top10原ID bbcb2544-3c0a-46a2-b8de-1714bf9e3926、d01c9599-a74f-433d-958a-f9fddcc54164最后记录accepted；旧辅助dbac99a1-b428-417e-9d26-5442ca99679d已RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD。NCAR版对这些ID不再轮询、不取载荷、不取消、不重提、不分裂；不算NCAR在途量。

首次NCAR v008 PID29188（17:38:58.272437）取回5个O3@5子集、120格点字段小时、6459响应字节后，Windows并发状态JSON读写共享冲突使本地退出；错误上报又暴露safe_error返回字符串而原writer需要mapping。完整日志/状态/载荷/代码/009均原样保留；17:43实际下载进程均无、共享OS锁释放。v008旧RUNNING/滞后progress不代表存活。

修复另存v009：状态读写用可重入锁串行化，临时PermissionError最多2秒有界重试；safe_error保持mapping，不读CDS凭据。永久权限错误仍报失败，不掩盖。新plan按所需值只读复用23份CDS历史载荷及5份NCAR响应，不重复下载、不覆盖旧文件。

新worker历史快照：PID31720，创建2026-10-15 17:47:52.558971 +08:00，唯一v009 run，隐藏启动；日志weather_top10_ncar_run_002_20261015_174752.stdout.log/stderr.log。精确CPU/创建时间/OS锁/已完成响应和源SHA见launch_verification.json、handoff.json。每轮必须实查，不能用此历史PID冒充存活。新首波19个精确日文件子集/456剩余格点字段小时，来自原576需求扣除已完成120；不是全项目总量。

## 只下载Top10，路线变化不改科学范围

全部后续缺值仅用NCAR/GDEX公开GET：`https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/`。通过OPeNDAP取所需发布格点/时刻/字段，不下载整份全球文件、不新增CDS请求、不读取CDS密钥、不换代理。服务端延迟或超时不等于排队完成，也不保证全量ETA。

仅z/u@1hPa，o3/u@5hPa，o3@500hPa，u@650hPa，o3/r@850hPa；surface仅CAPE与maximum_2m_temperature_since_previous_post_processing。auxiliary_pairs=[]，不增z@5/500/650/850、云雨雪诊断；单请求单字段/单压力层，绝不4×5笛卡尔扩展，不推导目标参数。

压力日文件、CAPE月文件、MX2T半月文件仅按精确时间索引裁取。MX2T valid UTC=06/18UTC起报+1—12小时预报时效，保留跨日/月原补时，不改日最高温、不插值。返回DAS参数ID/单位及时间/坐标单位、实际坐标、层、UTC、数量、有限值与SHA核对后才标VERIFIED_SUBSET；完整性不等同独立字段QA。ASCII与NCAR压缩格式不宣称同CDS逐位相同或底层未经插值ERA5原始谱/高斯网格。

5/500/650/850hPa固定标准几何高度35979.046/5579.330/3592.717/1457.635m，沿原站星前向WGS84射线求交取最近0.25度发布格，非站点垂直柱。四层不读/补辅助z，不能称真实逐小时等压面；1hPa z属于Top10仍用于该层逐小时定位（最多8轮）。最近格逐轴最近、中点北东；不做空间/时间插值或添加grid再网格化。循环、不收敛、缺值、低于天线均保留无效。

FRA-00289天线1622m高于固定850层，保留INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA，不抬高/补零；FRA-00609方向差0.5424度仍待复核，不放宽0.5阈值/不改指向。所有链路身份、原日期缺口/补时/分区保留。单位/相同参数名不等于模型兼容。

## 状态检查、恢复、计数

先完整读T012、003、010、PROGRESS与v009 plan/handoff/launch_verification/preflight_qa/plan_independent_qa、waves/units/jobs/progress/active_worker/logs/failures/partial_history。plan引用的v008旧失败/记录和原几何/模型只读。

核对全部weather_download*.py真实命令、PID/创建时间、两次CPU、共享OS字节锁、全部冻结SHA与日志。存活worker只读检查，不重复启动，不改冻结代码。最多两个NCAR GET（不受旧CDS远端历史槽阻塞）；连接10秒/读取180秒、单响应16MiB上限，网络/408/429/5xx按同URL退避30—300秒（尊重有界Retry-After）。无POST或远端提交歧义。

全部实际下载进程均无、锁释放、未齐、指纹一致且失败已诊断时，隐藏恢复唯一v009 run并另名日志。已完成SHA正确的子集跳过，幂等GET按同URL恢复，不退回CDS。401/403/404/410、解析/参数/单位/网格/UTC/数量/有限值失败保留BLOCKED，先诊断；修复另存新版本，不放宽检查或换代理。

part先核SHA、HTTP Content-Length（若有）/实际大小和完整DAP数组及精确样本身份，完整且无缺值才转正式；缺Content-Length不编造官方大小。损坏/不完整part原样保存在本版partial_history后同URL重下。断点恢复允许完整final/part重解码后登记，不能把半截响应视为成功。

progress中verified_unique_grid_field_hours只算v009新响应；reused_completed_ncar_grid_field_hours为旧v008的120；total_verified_ncar_grid_field_hours_including_migrated为两者合计。旧CDS复用量另列，不能整份算覆盖。单字段24值不是24小时完整Top10。日历小时、站点小时、唯一格点字段小时、十列齐全站点小时分别报告；units只是采集记录，尚非完整字段/模型QA。

## 下游与完成条件

实际全字段独立UTC/层/单位/身份/网格/数量/有限值/SHA及几何残差QA后，核对目标和已有模型兼容，再做标明覆盖范围的局部纯推理。现成k_0010 SHA=b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5，BiLSTM seed17/53轮，LOCAL2404小时拟合，10列/原生268槽；不是给156网络补零。

零训练：只冻结transform/predict/组合/指标；禁止fit/partial_fit/fit_transform/训练/微调/调参/选种子/优化器更新/重估插补归一化晴空参考CDF。LOCAL/所有旧模型/结果/QA只读不重跑。GBS秒级relative-clearsky可复用旧Alignment004；OpenSat分钟RSL平均域未知，不插值造秒真值、不重拟合分钟CDF。其他模型族仅已有兼容检查点；缺模型/特征/目标尺度/天气分组要列缺口由我决定，不扩下载。

按来源频段、总体/晴/雨含强雨/干燥阴天/other分别报告RMSE/MAE/Bias及适用CRPS/校准与样本日期站点；缺诊断保留不可评估，不跨频段合并RMSE。历史访问/同日重叠不称新盲测，随机场景不称确定性秒预测；IID/block共享CDF常数增量相消，不把共同VS当慢变增益；LOCAL CRPS0.130487较自身0.162335降19.62%、90覆盖73.98%不是已校准。负结果/空数据/数值失效保留。

同一tvt、原线程、15分钟保持；正常无变化安静，仅阶段完成/失败/需决策通知。NCAR切换、下载、字段QA或局部试跑均非T012完成。只有既定外部评价/独立QA/图表/中文报告/进度齐备，或人工明确缩减最终范围且满足，才总completion并暂停。
