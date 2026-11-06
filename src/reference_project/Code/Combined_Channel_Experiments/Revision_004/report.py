"""Generate evidence-linked result tables and byte-preserving index handoff."""
from common import *
import pandas as pd

LABELS=[('ours_frozen_top10','本文 Top10 BiLSTM（冻结）'),('train_mean','LOCAL训练均值常数'),
        ('ridge','Ridge'),('elastic_net','ElasticNet'),('svr_rbf','RBF-SVR'),('random_forest','随机森林'),
        ('xgboost','XGBoost'),('bai2022_mlp_adapted','Bai2022 MLP适配'),('bai2022_lstm_t0_adapted','Bai2022 LSTM-T0适配'),
        ('tabm2025_official_budgeted','TabM2025官方架构/固定预算')]

def write(name,text):
    with newpath(OUT/name).open('x',encoding='utf-8') as f:f.write(text)

def main():
    offline()
    with lock():
        require(read(OUT/'independent_qa.json')['status']=='PASS','Independent QA')
        require(read(OUT/'posthoc_zero_diagnostic.json')['status']=='PASS','Zero diagnostic QA')
        guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
        metrics=pd.read_csv(OUT/'metrics.csv');family=pd.read_csv(OUT/'family_summary.csv')
        stations=read(OUT/'station_parameters.json');qa=read(OUT/'independent_qa.json')
        coverage=read(OUT/'coverage.json');zeros=read(OUT/'posthoc_zero_diagnostic.json')['records']
        def row(cohort,model,group='overall'):
            part=metrics.loc[metrics.cohort.eq(cohort)&metrics.model.eq(model)&metrics.group.eq(group)]
            require(len(part)==1,'Unique report numeric row');return part.iloc[0]
        def frow(cohort,name):
            part=family.loc[family.cohort.eq(cohort)&family.family.eq(name)&family.group.eq('overall')]
            require(len(part)==1,'Unique family row');return part.iloc[0]
        def fmt(r):
            return f'{r.rmse_mean_db:.4f} ± {r.rmse_seed_sd_db:.4f}' if r.runs>1 else f'{r.rmse_mean_db:.4f}'
        mainrows=[]
        for name,label in LABELS:
            a,b=frow('LOCAL_test',name),frow('UK_all',name)
            mainrows.append(f'| {label} | {fmt(a)} | {fmt(b)} | {b.mae_mean_db:.4f} | {b.bias_mean_db:+.4f} |')
        physicalrows=[]
        for a,b,label in [('physics_geom_circular','itu838_geom_circular_nominal','P.838＋P.839几何雨衰近似'),
                          ('physics_p618_hourly_circular','itu618_proxy_circular_nominal','P.618路径缩减小时代理')]:
            x,y=row('LOCAL_test',a),row('UK_all',b)
            physicalrows.append(f'| {label} | {x.rmse_db:.4f} | {y.rmse_db:.4f} | {y.mae_db:.4f} | {y.bias_db:+.4f} |')
        stationrows=[]
        for s in coverage['station_counts']:
            stationrows.append(f"| {s['station']} | {s['candidate_dates']} | {s['candidate_hours']} | {s['old_primary_hours']} | {s['rain_invalid_primary']} | {s['common_hours']} |")
        parameterrows=[]
        for name,p in stations.items():
            parameterrows.append(f"| {name} | {p['h0_km_amsl']:.6f} | {p['hR_km_amsl']:.6f} | {p['receiver_km_amsl']:.3f} | {p['hR_km_amsl']-p['receiver_km_amsl']:.6f} | {p['elevation_deg']:.6f} |")
        sensitivity=[]
        for suffix,label in [('minus050','P.839雨层高度−0.5 km'),('nominal','P.839原值（主情景）'),('plus050','P.839雨层高度＋0.5 km')]:
            a,b=row('UK_all','itu838_geom_circular_'+suffix),row('UK_all','itu618_proxy_circular_'+suffix)
            sensitivity.append(f'| {label} | {a.rmse_db:.4f} | {b.rmse_db:.4f} |')
        group_rows=[]
        for group,label in [('rain_below_0.1','0≤R<0.1'),('rain_0.1_to_1','0.1≤R<1'),('rain_at_least_1','R≥1')]:
            a,b,c=row('UK_all','ours_frozen_top10',group),row('UK_all','itu838_geom_circular_nominal',group),row('UK_all','itu618_proxy_circular_nominal',group)
            group_rows.append(f'| {label} mm/h | {int(a.n)} | {a.rmse_db:.4f} | {b.rmse_db:.4f} | {c.rmse_db:.4f} |')
        periodrows=[]
        for name,label in [('ours_frozen_top10','本文Top10'),('train_mean','LOCAL训练均值'),('elastic_net_seed17','ElasticNet'),('itu838_geom_circular_nominal','P.838几何近似'),('itu618_proxy_circular_nominal','P.618小时代理')]:
            a,b=row('UK_original39',name),row('UK_extension2003',name)
            periodrows.append(f'| {label} | {a.rmse_db:.4f} | {b.rmse_db:.4f} |')
        cirows=[]
        for r in read(OUT/'date_cluster_bootstrap.json')['results']:
            if r['cohort']!='UK_all':continue
            name='P.838几何近似' if r['model'].startswith('itu838') else 'P.618小时代理'
            ref='本文Top10' if r['reference']=='ours_frozen_top10' else 'LOCAL训练均值'
            cirows.append(f"| {name} − {ref} | {r['delta_rmse_db']:+.4f} | [{r['unadjusted_ci95_db'][0]:.4f}, {r['unadjusted_ci95_db'][1]:.4f}] |")
        z=next(x for x in zeros if x['cohort']=='UK_common')
        report=f'''# 近似 ITU 基线补算：LOCAL 与扩展英国

状态：补算及独立数值核验完成。2026-10-16。代码/输出均另存Revision004；历史结果不改。

## 直接结论

本文冻结Top10并不是这次统一比较的最优模型。LOCAL测试仍以TabM三个固定种子的均值RMSE0.4009 dB最低，XGBoost0.4219 dB，本文0.5041 dB；本文优于两条既有LOCAL雨衰代理（约1.04 dB）。英国同5538小时上，本文RMSE1.8375 dB，P.838几何近似0.1772 dB、P.618小时代理0.2011 dB。英国的学习方法中ElasticNet0.4077 dB最低；训练均值控制0.2561 dB。

补充的固定零衰减诊断为{z['rmse_db']:.4f} dB，略低于主P.838近似；不能把低绝对误差直接写成ITU已重建了英国衰减变化。这个零控制是看到结果后增加的透明诊断，不是预登记主对手、不拟合常数，也未用于选高度。主P.838和P.618代理的R²分别{row('UK_all','itu838_geom_circular_nominal').r2:.4f}和{row('UK_all','itu618_proxy_circular_nominal').r2:.4f}，均为负。

本次指标针对自然Top10检查点，不代表本文全部156特征变体。物理基线另外使用降水等输入，不能把它们称为同Top10输入模型；所有英国方法的评分小时完全相同。

## 1. 主结果（dB；误差越低越好）

LOCAL为687个冻结测试小时；UK为5538个共同有效站点小时。三种子方法显示均值±种子样本SD，SD不是置信区间；所有种子均保留，无外站选种子。

| 同Top10学习方法/控制 | LOCAL RMSE | UK RMSE | UK MAE | UK Bias |
|---|---:|---:|---:|---:|
{chr(10).join(mainrows)}

下表单列额外雨量输入的雨衰物理代理：UK主情景使用源文件确认的圆极化及P.839气候平均雨层高度；LOCAL是之前冻结的小时零度层输入方案，直接复用预测，未换成UK气候高度。

| 雨衰物理代理 | LOCAL RMSE | UK RMSE | UK MAE | UK Bias |
|---|---:|---:|---:|---:|
{chr(10).join(physicalrows)}

固定零衰减事后诊断的UK MAE为{z['mae_db']:.4f} dB、Bias为{z['bias_db']:+.4f} dB；它未添加进冻结的主metrics.csv。详见[诊断记录](posthoc_zero_diagnostic.json)。原5593小时本文RMSE1.8286 dB仍是原报告结果；本表1.8375是去掉55个物理无效雨量小时后的同样本分数，不是模型变化。

## 2. 完整日期与共同样本

| 站点 | 完整日期数 | 气象完整小时 | 原合格评分小时 | 负液态差值排除 | 共同评分小时 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(stationrows)}

Chilbolton新增2003-08-19—11-30共104日、Sparsholt新增2003-10-08—11-30共54日，再加各站原2004年39日。原39日是02-18—21、02-25—28、03-01—31，并非连续39天。两站合计236站点日，但只有143个不同UTC日期。12月高空气象未完整，未并入本次比较。没有重新下载或填补原生缺日。

5664个候选小时中56个液态差值为负；其中55个本来符合标签评分条件，全部排除于物理共同评分，剩余5538。37缺标签/34低覆盖小时仍保留；不修改标签、不补零、不删除原始行。负值全部发生在原39日合格小时；新增2003合格3746小时未受此筛选影响。

| 方法 | 原39日共同1792小时 RMSE | 新增2003共同3746小时 RMSE |
|---|---:|---:|
{chr(10).join(periodrows)}

## 3. 有来源的雨层高度近似及极化纠正

采用[ITU-R P.839-4](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.839-4-201309-I!!PDF-E.pdf)公开数字网格：对站点的四个邻近气候格点按标准做双线性插值，得到年平均零度层高度h0，hR=h0+0.36 km。网格文件明确是海拔基准，不是离地高度；湿路径必须再减接收机海拔hs。这里仅插值标准静态气候图，不插值/重网格化已冻结的气象数据。

| 站点 | h0海拔 km | hR海拔 km | 接收机海拔 km | 雨层垂直厚度 km | 仰角 ° |
|---|---:|---:|---:|---:|---:|
{chr(10).join(parameterrows)}

四角数值、权重、源ZIP SHA和每个情景的k/α/斜路径长度均留存在[站点参数](station_parameters.json)、[物理参数](physical_parameters.json)及[参考来源收据](references_complete.json)。该高度是年平均近似，不冒充每小时真实雨层。此为历史数据回顾性比较，不声称在2003年实时获得了后续版本的标准图。

需要纠正前版的一点：236个原生文件的title、polarisation_transmit、polarisation_receive均明确给出“Left Hand Circularly Polarized”，20.7GHz发射与接收匹配。此前“英国极化缺失”是检查未深入，不是文件没有元数据。本次已逐文件确认，主情景用圆极化；水平/垂直极化只是反事实敏感性控制，不再作为未知极化候选。P.838的圆极化混合取τ=45°，此公式不区分左右旋。

仰角沿用已有名义WGS84站星几何约28.05°/28.09°；原文件的30°粗值同时保留。没有轨道星历，不称精准实测指向。频率、坐标、海拔、卫星23°W和几何均由独立代码复核。

## 4. 公式及任务边界

R=3600×(MTPR−MSR)，其中原始率为kg m⁻² s⁻¹；一小时水当量mm的数值对应该小时平均mm/h。MTPR包含雨雪、MSR为雪水当量；两者valid UTC为信标小时结束t+1h，均值率不再次差分。负差保留标记、不裁零，不能称实测雨强。

[P.838-3](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf)：γR=kR^α，单位dB/km；k、α由频率、仰角和极化按标准表1–4/式4–5计算。主几何代理A=γR(hR−hs)/sinθ。它假设站点小时平均雨率沿湿斜路径均匀，仅估雨衰；没有冒充完整气体/云/闪烁衰减。

[P.618-14](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-14-202308-I!!PDF-E.pdf)的路径缩减/垂直调整公式另算小时代理。标准本来使用一年0.01%超越的**一分钟**雨率R0.01；用小时均值代入只是一种项目代理，不是完整P.618年度标准预测。没有伪造R0.01或年超越概率，不将这条代理包装成标准年度验证。

## 5. 高度和极化敏感性（不选最低分）

圆极化固定，预先规定hR±0.5km；这一范围是人为敏感性区间，不是置信区间。虽然降低高度的误差更低，主结果仍使用未调节的P.839原值。

| 圆极化高度情景 | P.838几何 RMSE | P.618小时代理 RMSE |
|---|---:|---:|
{chr(10).join(sensitivity)}

三高度×三极化×两公式共18条预测全部保留。[全情景UK汇总](sensitivity_all_scenarios.csv)包含水平/垂直的反事实结果；它们不是18个独立训练模型，也不用于宣称18次独立验证。

小时液态再分析率分层如下，不把这些阈值称为通用气象强雨定义：

| 再分析液态率 | 小时数 | 本文 RMSE | P.838几何 RMSE | P.618代理 RMSE |
|---|---:|---:|---:|---:|
{chr(10).join(group_rows)}

低雨率小时占大多数，必须结合分雨量结果解读总体分数。主圆极化P.838在R≥1mm/h组Bias约+0.504dB，P.618代理约+0.792dB；不能把总体低误差理解为各种降水条件都准确。

## 6. 配对误差差值（探索性）

同UTC日期内两站一起重采样，143个日期簇，2000次，固定种子20261016。区间未做多重比较校正，也未建模相邻日期依赖，不等于两个独立区域的普遍泛化证据。负差值表示前者RMSE较小。

| 比较 | ΔRMSE dB | 日期簇95%区间 dB |
|---|---:|---|
{chr(10).join(cirows)}

## 7. 已补齐与仍不属于本次任务的项目

| 项目 | 本次状态 |
|---|---|
| UK的P.838几何雨衰 | 已补算、独立核验，约2.42km海拔气候雨层，已核圆极化 |
| UK的P.618小时代理 | 已补算、独立核验，明确不是年度标准 |
| LOCAL两条既有代理 | 原687行预测只读复用，指标再次核验 |
| 所有20个已完成学习/常数运行 | 原预测不变，UK仅按5538共同样本重新汇总 |
| 高度/极化敏感性、逐站/原39日/新增日期/分雨量结果 | 18情景完整保存与核验 |
| 完整年度P.618、P.1853、Maseng–Bakken、SST/S-SST | 不属于同期小时点预测，且分钟雨/分量统计等输入不足；不伪造补算 |
| 用过去实测信道预测未来的模型 | 按用户明确要求排除 |
| 其他缺完整水凝物场/校准参数/可核架构的论文 | 仍按原[方法可行性表](../Combined_Channel_Revision_003/METHOD_FEASIBILITY.md)登记，不把任意替代结构称作者复现 |

这完成了本次“近似ITU基线”的缺项，不代表所有历史T012目标或所有文献模型均已完成。下载仍STOPPED_BY_USER，同一tvt仍PAUSED。

## 8. 核验与可追溯产物

独立核验PASS：{qa['protected_sources']}个保护源SHA保持；236个原生文件极化/频率/坐标/海拔确认；100944个有限物理预测值逐值独立标量复算，最大差{qa['max_physical_abs_difference_db']:.3g}dB；1196行指标、856行家族汇总、28个日期簇配对区间均复算。全部旧ML预测及标签逐值不变。标准表数值控制、零雨/负雨/缺值/雨层低于站点/不支持低仰角等{len(qa['controls'])}项控制通过。

官方四个方法参考文件实际取得200响应并核长度/ZIP CRC/SHA。首次请求被沙盒套接字权限拒绝，失败记录保留；后续经批准按同URL取回，没换代理。气象下载0、训练0、ML预测调用0。三个标准的关键公式页已渲染目检，检查分式/指数/角度约定；数值核验不是对所有传播物理假设的证明。

- [逐小时全部预测与原输入/标签](predictions_physical_and_reused.parquet)
- [完整指标](metrics.csv)、[家族均值与SD](family_summary.csv)、[样本数](cohort_counts.csv)
- [完整独立QA](independent_qa.json)、[公式审计](FORMULA_AUDIT.md)
- [完整源码与冻结协议](../../../Code/Combined_Channel_Experiments/Revision_004/PROTOCOL.md)、[极化证据修订](../../../Code/Combined_Channel_Experiments/Revision_004/PROTOCOL_ADDENDUM_001.md)
- [交接](HANDOFF.md)、[最终交付收据](delivery_verification.json)
'''
        write('COMPARISON_REPORT.md',report)
        metrics.loc[metrics.cohort.eq('UK_all')&metrics.group.eq('overall')&metrics.model.str.startswith('itu')].to_csv(newpath(OUT/'sensitivity_all_scenarios.csv'),index=False)
        write('HANDOFF.md','''# Revision004 交接

最新授权：近似ITU基线、缺项补算。本轮可执行的两条UK雨衰近似及18情景已完成并独立核验，采用[中文报告](COMPARISON_REPORT.md)。它替代Revision003“UK雨衰未执行”的当前状态，不修改该历史报告。

原Top10模型/检查点/20运行预测/原始标签及全部历史结果只读。新训练0、ML推理0、气象下载0。官方标准方法参考获取4份，原停止标志及tvt PAUSED保持。不得根据旧heartbeat重启下载。

源文件已确认两站20.7GHz匹配左旋圆极化；此前“极化缺失”是检查不充分，已更正。雨层高度按P839气候图约2.42km海拔近似，不是小时实测。负液态率不裁零，物理共同评分5538小时；原5593小时ML指标不作废。主参数固定，不因敏感性最低分改变。

主对照：LOCAL本文0.5041、TabM0.4009、P8381.0378、P618代理1.0405dB；UK同样本本文1.8375、P8380.1772、P618代理0.2011dB，零衰减事后诊断0.1732dB。不能宣称本文最优，也不能只用总体低RMSE宣称ITU准确恢复时变衰减。

原始结果、失败、所有情景和索引before/after保留。核验详见[independent_qa.json](independent_qa.json)；交付封存、索引字节保持及锁释放以[delivery_verification.json](delivery_verification.json)为准。科学采用待用户审阅，历史T012总体未自动关闭。年度/随机动态/缺输入/信道历史预测任务不在本次完成范围。
''')
        write('CHANGE_RECORD.md','''# 变更记录

- 新增Revision004物理计算、源信息核验、独立标量公式审计与报告；历史代码/数据/结果保持只读。
- 先保存T012/PROGRESS完整before快照并冻结7074项源SHA，新增执行锁及协议。
- 官方参考首次GET遇沙盒WinError10013；原失败/源码保留，v002仅核验复用before快照后按原4URL继续，不更改代理或科学定义。
- 原生NetCDF非ASCII路径直接打开的只读探测失败；用原文件字节内存接口成功，未改名/复制/改动源数据。
- 原生236文件明确LHCP；在输出物理结果前新增协议证据修订，圆极化从未确认假设改为源确认事实，原协议保留。
- 18情景/100944有限值、1196指标、856汇总、28配对区间独立核验通过；所有学习预测不变。
- 看到物理结果后增设固定0dB事后诊断，另存且不并入预登记主表，不使用UK均值拟合常数或选择高度。
- 仅机械插入固定索引的新版本交接段，旧字节与before快照须在最终封存时一致；生成after快照。
- 通过[公式审计](FORMULA_AUDIT.md)记录单位/高度基准/极化修订及代理与原标准任务差异；没有新训练、ML推理或气象下载。
''')
        # Mechanical insertion of an already-authored handoff block, preserving every
        # old byte including CRLF. Fixed-index before copies were made at launch.
        for p in INDEXES:
            old_bytes=(OUT/'document_snapshots/before'/p.relative_to(ROOT)).read_bytes()
            require(p.read_bytes()==old_bytes,'Index changed since before snapshot')
            link='../Outputs/Experiments/Combined_Channel_Revision_004/COMPARISON_REPORT.md' if p.parent.name=='Tasks' else '../../../Outputs/Experiments/Combined_Channel_Revision_004/COMPARISON_REPORT.md'
            block=f'''<!-- REV004-BEGIN -->
## 当前：近似ITU雨衰缺项已补算并核验（2026-10-16）

最新人工授权“雨层高度可以近似”“近似ITU基线、没完成的请补算”。WRITE仅新Code/Combined_Channel_Experiments/Revision_004与Outputs/Experiments/Combined_Channel_Revision_004及本固定索引交接；原/root线程，C1—C8/OS锁/READ边界详见Revision004/PROTOCOL.md，完整索引before已保存。

英国已补齐P838几何雨衰、P618小时代理及预定18高度/极化情景，未训练或重新调用学习模型；全部20运行复用旧预测并按同样本评分。两站236原生文件确认20.7GHz发射/接收匹配左旋圆极化，纠正下方历史“极化缺失”；P839气候雨层海拔约2.415/2.420km，按接收机84/118m海拔扣除，±0.5km仅敏感性，不按分数选高度。

UK共同5538小时（原5593中55个负液态差值不用于物理公式）：本文Top10 RMSE1.8375、P8380.1772、P618小时代理0.2011dB；另存固定0dB事后诊断0.1732dB。LOCAL687本文0.5041、TabM均值0.4009，原P8381.0378/P618代理1.0405只读复用。本文并非当前统一比较最优；小绝对误差不等于UK时变机制已验证。

7074保护源SHA、236原生元数据、100944有限物理值、1196指标、856家族汇总及28配对区间独立QA PASS。当前采用[Revision004中文报告]({link})；旧39日/扩展集/模型/标签/历史结果不改。完整年度P618及随机动态模型未伪装为小时基线。训练0、ML推理0、气象下载0；仅取4份官方ITU方法参考。STOP_REQUESTED保留、tvt PAUSED；固定索引after、最终封存及锁释放以Revision004交付收据为准。历史T012总目标未自动标完成，以下旧段原样保留。
<!-- REV004-END -->

'''.encode('utf-8')
            marker=b'<!-- REV003-BEGIN -->';require(old_bytes.count(marker)==1,'Known fixed-index insertion anchor')
            new_bytes=old_bytes.replace(marker,block+marker,1)
            with p.open('wb') as f:f.write(new_bytes)
            require(new_bytes.replace(block,b'',1)==old_bytes,'Mechanical insertion preserves historical bytes')
        protected_verify(guard)
        emit(OUT/'report_generation.json',{'status':'GENERATED_AWAITING_FINAL_SEAL','time':now(),'report_sha256':sha(OUT/'COMPARISON_REPORT.md'),'report_code_sha256':sha(Path(__file__)),'mechanical_index_insertion':'REV004; before bytes preserved','numeric_main_table_rows':12})
        print('Reports and scoped index additions generated; final seal pending.',flush=True)

if __name__=='__main__':main()
