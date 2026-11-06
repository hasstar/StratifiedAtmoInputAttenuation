"""Chinese technical delivery report, incident ledger and claim/evidence map."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import csv
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002"
BASE=ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE"
REPAIR=ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_033_Execution_Repair"
LABELS=dict(tabm="TabM",bai_lstm="Bai LSTM",bilstm="BiLSTM",transformer="Transformer",mlp="MLP",
            catboost="CatBoost",gcn="GCN",random_forest="Random Forest",svr_rbf="SVR-RBF",elastic_net="Elastic Net",
            combined_ITU_approx="组合ITU小时近似")
MODELS=list(LABELS)[:-1]
WEATHERS={"overall":"总体","clear_dry":"晴天","rain":"雨天","dry_overcast":"干燥阴天","other":"其他"}
def jread(p):
    return json.loads(p.read_text(encoding="utf-8"))
def jwrite(p,obj):
    with p.open("x",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def save(p,text):
    with p.open("x",encoding="utf-8") as f: f.write(text+"\n")
def link(name,rel):
    p=(OUT/rel).resolve()
    return f"[{name}]({p.as_posix()})"
def tab(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(str(x) for x in row)+" |" for row in rows])
def n(v):
    return "null" if pd.isna(v) else f"{v:.6f}"
def main():
    q=jread(OUT/"evidence/INDEPENDENT_AGGREGATE_QA.json")
    assert q["status"]=="PASS"
    d=pd.read_csv(OUT/"tables/all_test_metrics_15605_rows.csv")
    pts=pd.read_csv(OUT/"tables/representative_points_300_rows.csv")
    s=pd.read_csv(OUT/"tables/all_states_1560.csv")
    cov=pd.read_csv(OUT/"tables/weather_sample_coverage.csv")
    qa=pd.read_csv(OUT/"evidence/state_qa_index_1560.csv")
    roles=pd.read_csv(OUT/"tables/model_roles_6240_with_sha.csv")
    oldbad=jread(OUT/"evidence/itu_negative_liquid_hours.json")
    hourly=pd.read_csv(OUT/"hourly/tabm_test_687_all_steps.csv")
    h=hourly[hourly.k.eq(156)].copy()
    h["key"]=pd.to_datetime(h.hour_bjt)
    anomalies=[]
    incidents=[]
    def incident(i,phase,description,resolution,evidence):
        incidents.append(dict(id=i,phase=phase,description=description,resolution=resolution,evidence=evidence))
    incident("I01","startup","PID5784首次沙箱启动在正式训练前退出；日志为空。",
             "未计入有效训练；随后正式隐藏启动9228。",
             "Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE/logs/20261019_023301_stderr.log")
    for i,f in enumerate(sorted((BASE/"failures").glob("*.json")),2):
        v=jread(f)
        incident(f"I{i:02d}","training_runtime",f"PID{v['pid']}：{v['error']}；progress.json原子替换WinError5。",
                 "冻结032不改；另存033执行层有限重试修复，保存失败与epoch35副本后恢复。",
                 f.relative_to(ROOT).as_posix())
    incident("I03","recoverable_interruption","PID29116于12:06:59后至13:21检查前退出；具体退出原因未确定。",
             "主机进程与锁释放、1028来源SHA及Transformer101/fold2/epoch5检查点验证后，同管线恢复为PID27652；记录间隔4834.84秒不是精确停机时长。",
             "Outputs/Experiments/Hourly_Attenuation_Revision_033_Execution_Repair/recovery_events/20261019_1324_Interrupted_Process/OUTCOME.json")
    synthetic=list((REPAIR/"deferred_progress").glob("*.json"))
    for i,f in enumerate(synthetic,1):
        incident(f"SYN{i:02d}","synthetic_execution_test","故意构造的进度文件写入拒绝。",
                 "仅SYNTHETIC_TEST_ONLY执行层测试；不记作科研训练新失败。",f.relative_to(ROOT).as_posix())
    for i,f in enumerate(sorted((OUT.parent/"evidence").glob("audit_failure_*.json")),1):
        incident(f"R{i:02d}","report_audit","首次报告审计误把零基特征order当作一基，首状态检查失败。",
                 "保留原审计代码、失败JSON及空表头中间文件；export_audit_v2修正模式后在delivery_002从首状态重审，无科研文件改动。",
                 f.relative_to(ROOT).as_posix())
    incident("R02","report_environment","随附绘图库导入时缺少pyparsing。",
             "改用已存在的pytorchmy环境（matplotlib3.10.8/seaborn0.13.2），未安装环境或依赖。",
             "Code/Hourly_Attenuation_Experiments/Revision_034/PROTOCOL.md")
    for i,a in enumerate(oldbad,1):
        current=h.loc[h.key.eq(pd.Timestamp(a["hour_bjt"]))].iloc[0]
        row={**a,"frozen_dataset_weather":current.weather,
             "legacy_event_differs_from_frozen_weather":a["event"]!=current.weather,
             "main_687_included":True,"common_684_included":False}
        anomalies.append(row)
        incident(f"A{i:02d}","physical_input",
                 f"{a['hour_bjt']}液态差={a['signed_liquid_proxy_mmph']:.12g} mm/h，小于0；旧event={a['event']}，冻结天气={current.weather}。",
                 "原值留账、不裁零；仅ITU同样本补充排除，机器学习主687小时保留。",
                 "Outputs/Experiments/Combined_Channel_Revision_027/excluded_hours.json")
    incident("A04","weather_metadata","2026-03-08 01时旧异常账本event=other，但冻结数据集天气为clear_dry。",
             "两个字段并存；所有指标按Signal_Detectability_Revision_003冻结天气，旧event只作历史溯源，不覆盖标签。",
             "Outputs/Datasets/Signal_Detectability_Revision_003/hourly_sample_status.parquet")
    pd.DataFrame(incidents).to_csv(OUT/"evidence/failure_anomaly_ledger.csv",index=False,encoding="utf-8-sig",mode="x")
    pd.DataFrame(anomalies).to_csv(OUT/"tables/ITU_three_excluded_hours_with_canonical_weather.csv",index=False,encoding="utf-8-sig",mode="x")
    jwrite(OUT/"evidence/itu_anomaly_reconciliation.json",anomalies)
    full=pts.query("basis=='full_156' and population=='ML_687' and weather=='overall'").set_index("model")
    cv=pts.query("basis=='training_cv_min_descriptive' and population=='ML_687' and weather=='overall'").set_index("model")
    best=pts.query("basis=='test_min_exploratory' and population=='ML_687' and weather=='overall'").set_index("model")
    common=d.query("population=='COMMON_684' and weather=='overall' and (k==156 or model=='combined_ITU_approx')").set_index("model")
    t=full.loc["tabm"]; tc=cv.loc["tabm"]
    reduction=(1-tc.rmse_db/t.rmse_db)*100
    count_negative=int(d[d.population.eq("ML_687")].r2.lt(0).sum())
    nullr=int(d.r2.isna().sum())
    nullcorr=int(d.pearson_r.isna().sum())
    retained=jread(OUT/"paths/tabm_156_to_1.json")[156-int(tc.k)]["retained"]
    report=[
        "# T014 十学习模型独立156→1 RFE与组合ITU：中文技术报告",
        f"生成：{datetime.now().astimezone().isoformat(timespec='seconds')}。数值来源：冻结Hourly032；执行修复：Hourly033；交付审计：Hourly034。本文为本地实验技术交付，不代替作者投稿批准。",
        "## 1. 结论",
        f"十个模型的156条特征数状态均已完成并通过逐状态QA，共1560状态。独立汇总复核确认10条路径互不相同、1550次逐列删除、6240角色槽及367350次置换评价证据链完整。最终交付状态以{link('FINAL_QA','evidence/FINAL_QA.json')}为准，旧032的NUMERICS_COMPLETE_REPORT_PENDING为冻结数值阶段记录，不重写为交付完成。[E01/E02]",
        f"共同全156输入下，TabM测试总体RMSE={t.rmse_db:.6f} dB、R²={t.r2:.6f}，总体RMSE排第3；BiLSTM为{full.loc['bilstm'].rmse_db:.6f} dB，Transformer为{full.loc['transformer'].rmse_db:.6f} dB。因而本轮不支持“TabM在全输入总体误差上最优”的断言。[C01/E03]",
        f"沿训练内CV总体RMSE最小值提取的描述性状态为TabM k={int(tc.k)}，测试RMSE={tc.rmse_db:.6f} dB、R²={tc.r2:.6f}；相对其全156状态，测试RMSE降低{reduction:.2f}%，特征数减少{(1-tc.k/156)*100:.2f}%。这一紧凑候选的结果有报告价值，但不是预先锁定的最终部署模型，也不是未经选择的独立泛化估计。[C02/E04]",
        "训练内CV最小点与测试最小点并不一致。CV点是在同一自适应RFE路径上事后提取，未增加嵌套验证；测试已在历史工作中查看，因此测试最优k仅列为探索性结果，不反向更改删列、轮数、种子或超参数。[E02/E04]",
        "## 2. 数据、模型和RFE协议",
        "固定Signal_Detectability_Revision_003的2404训练/343验证/687测试小时；所有学习模型从Rev023同一有序156列（50基础、106压力层）起步。三折GroupKFold以北京时间日期分组，仅在训练集内；每状态保存三折角色与全训练角色。内部折日期互斥不等于原始训练/验证/测试日期完全隔离，本文不作后者声明。预处理只拟合相应角色的训练部分。[E01/E05]",
        "每个模型在各自当前子集上实际拟合，或逐项登记完全相同历史权重的复用；各模型分别对每个保留特征计算三折holdout总体RMSE增量，删去均值最小者，并列按特征名称字典序。固定置换随机机制相同不等于共用排序。结构化网络的序列/图表示大气压力结构，不是新增历史信道序列；没有用置零推理冒充子集重训。[E01/E02]",
        "TabM固定official_base宽256、3块、k32、dropout0.1、54轮。Bai LSTM为单时刻天气输入的LOCAL适配，非作者预训练；原验证总体RMSE、耐心5、上限1000选择32轮后冻结，未使用测试选轮。其余模型沿用各家冻结LOCAL配置；完整配置与角色训练元数据逐项附存，历史搜索预算不同，不宣称公平等预算重新调参。单种子17，不是多种子统计。[E06]",
        "报告R²=1−SSE/SST，Pearson r和r²另列；Bias=预测−真值。保留负R²；不足样本/常量向量导致的未定义指标用null及原因保存。没有按天气或成绩缩减687小时主样本。",
        "### 样本与日期覆盖",
        tab(["群体","天气","小时数n","实际北京时间日期数"],[
            ["主687" if r.population=="test" else "共同684",WEATHERS[r.weather],int(r.n),int(r.n_dates)]
            for _,r in cov[cov.population.isin(["test","test_itu_common684"])].iterrows()]),
        f"完整逐组日期清单见{link('样本覆盖表','tables/weather_sample_coverage.csv')}和每状态指标CSV；日期数不能作为独立重复试验数。所有雨天状态仅51测试小时，本文不从单次结果声称统计显著性。[E05]",
        "## 3. 共同全156输入：687小时主表",
        tab(["模型","RMSE/dB","R²","Pearson r","Pearson r²","MAE/dB","Bias/dB"],[
            [LABELS[m],*[n(full.loc[m,c]) for c in ["rmse_db","r2","pearson_r","pearson_r_squared","mae_db","bias_db"]]]
            for m in full.sort_values("rmse_db").index]),
        "所有行均为687小时、seed17，按总体RMSE排列；这是冻结配置比较，不是相同搜索预算下的新调参竞赛。[E03]",
        "## 4. 训练内CV最小点：描述性摘要",
        "从每条已冻结路径的三折总体RMSE均值选最小k；若完全并列，摘要取较小k。该摘要规则不改变任何RFE删列，不使用测试挑k，也不触发额外拟合。CV标准误仅来自三折，不能解释为多种子不确定性或独立测试置信区间。",
        tab(["模型","k","训练CV均值/dB","折间SE/dB","测试RMSE/dB","测试R²"],[
            [LABELS[m],int(cv.loc[m,"k"]),*[n(cv.loc[m,c]) for c in ["cv_mean_rmse_db","cv_se_rmse_db","rmse_db","r2"]]]
            for m in MODELS]),
        f"TabM k={int(tc.k)}保留：",
        "\n".join(f"- {feature}" for feature in retained),
        "部分模型CV摘要点的测试误差较全156更大，完整保留，不再按测试结果换回某个子集来制造统一改善。完整逐状态特征、已删列表及下一删除特征见各模型path JSON和总表。[E04/E07]",
        "## 5. 天气分组",
        "以下每格为RMSE(dB) / R²。晴天clear_dry、雨天rain、干燥阴天dry_overcast、其他other互斥覆盖主687小时；天气分组不改变总体计分范围。",
        "### 全156状态",
        tab(["模型","晴天n322","雨天n51","干燥阴天n158","其他n156"],[
            [LABELS[m]]+[f"{n(r.rmse_db)} / {n(r.r2)}" for w in list(WEATHERS)[1:]
                         for _,r in pts.query("basis=='full_156' and population=='ML_687'").query("model==@m and weather==@w").iterrows()]
            for m in MODELS]),
        "### 训练CV摘要状态",
        tab(["模型(k)","晴天n322","雨天n51","干燥阴天n158","其他n156"],[
            [f"{LABELS[m]} ({int(cv.loc[m,'k'])})"]+[f"{n(r.rmse_db)} / {n(r.r2)}" for w in list(WEATHERS)[1:]
                         for _,r in pts.query("basis=='training_cv_min_descriptive' and population=='ML_687'").query("model==@m and weather==@w").iterrows()]
            for m in MODELS]),
        f"上述两组摘要及全部1560状态的Pearson r、r²、MAE、Bias、n、日期和null原因均见{link('完整天气指标','tables/all_test_metrics_15605_rows.csv')}。全路径主687天气指标中有{count_negative}条负R²，全部保留；全部15605条主/补充指标的R²缺失数={nullr}，Pearson r缺失数={nullcorr}。[E03/E04]",
        "## 6. 组合ITU固定物理基线：共同684小时补充",
        "采用用户原稿II.D三分量结构对应的已核验小时几何近似：固定物理气体参考+云一次+液态雨。来源为Combined011预测与Combined027合法性审计；不重新拟合偏移，不把小时雨率当R0.01，不把总降水当纯雨，不把负液态差裁零。这不是完整ITU概率模型，也不是逐字复现原稿额外RH因子及标签拟合0.5 dB偏移。ITU不做RFE。[E08]",
        tab(["模型（学习模型均全156）","n","RMSE/dB","R²","Pearson r"],[
            [LABELS[m],int(common.loc[m,"n"]),*[n(common.loc[m,c]) for c in ["rmse_db","r2","pearson_r"]]]
            for m in common.sort_values("rmse_db").index]),
        "本表仅共同684小时，不能与上一主表687小时直接混排行。全部学习模型所有k的684小时补充均已保存，预测为原687小时预测按同一合法掩码提取，没有重新拟合。",
        tab(["北京时间","液态差mm/h","旧账本event","冻结天气","处理"],[
            [a["hour_bjt"],f"{a['signed_liquid_proxy_mmph']:.12g}",a["event"],a["frozen_dataset_weather"],"主687保留；补充684排除"]
            for a in anomalies]),
        "旧异常账本的2026-03-08 01时event=other与冻结数据集clear_dry不同，已双字段留账；计分以冻结数据集天气为准。故684补充比主表少1个晴天和2个干燥阴天小时，rain及other计数不变。此差异不更改冻结标签或结果。[E08/E09]",
        "## 7. 必要图表",
    ]
    figs=[
        ("F01_all_RFE_test_RMSE","图1：十模型完整RFE测试RMSE。每面板156状态，共享纵轴；橙色菱形为全156，绿色星号为训练CV最小描述点。横轴从156递减至1，非按测试性能排序。"),
        ("F02_all_RFE_test_R2","图2：相同路径的决定系数R²。所有负值保留，虚线为零；并非Pearson r²。"),
        ("F03_weather_main687","图3：主687小时四类天气的全156与训练CV摘要点。按模型固定次序排列，样本数标在面板标题；没有测试最优点。"),
        ("F04_common684_full156_ITU","图4：共同684小时全156学习模型与ITU固定小时近似，按RMSE排列。RMSE、R²、Pearson r分别成轴；不混入687小时主表。")
    ]
    for name,caption in figs:
        report.extend([f"![{caption}]({(OUT/'figures'/f'{name}.png').as_posix()})",caption+" [E10]"])
    report.extend([
        "## 8. 拟合、复用与QA证据",
        "6240正式角色槽=5604新角色拟合+635历史权重复用+1次Bai本轮选轮权重复用；另有1次Bai全156原验证选轮拟合。因此实际新拟合调用5605，fit_ledger共6241条；不能把636次复用都称为历史复用，也不能把选轮重复计作额外正式角色。[C03/E11]",
        tab(["模型","正式角色槽","历史复用","本轮Bai复用","新角色拟合"],[
            [LABELS[m],len(roles[roles.model.eq(m)]),
             int(((roles.model==m)&roles.category.eq("HISTORICAL_WEIGHT_REUSE")).sum()),
             int(((roles.model==m)&roles.category.eq("IN_RUN_BAI_SELECTED_WEIGHT_REUSE")).sum()),
             int(((roles.model==m)&roles.category.eq("NEW_ROLE_FIT")).sum())] for m in MODELS]),
        f"逐状态独立QA重放全部四角色模型；独立标量计算全部367350置换指标，并每折对首/中/末列共13920次置换预测做权重重放，不宣称全部367350置换预测均做权重重放。最大已有预测重放差为{qa.maximum_prior_prediction_replay_delta_db.max():.12g} dB，小于2e-6 dB。[E02]",
        f"新增最终汇总审计不调用训练代码：检查1028来源、全部状态/评价/QA摘要链、6240模型文件及置换文件SHA；重算34325组指标、核对全部1550次删除与687→684预测对应。标量重算与存量指标最大绝对差{q['maximum_numeric_differences']['metric']:.12g}，汇总CSV数值差为0。[E01/E02]",
        "最后一层交付QA从导出后的1071720行逐小时CSV重新分组计算指标，并核对表格、路径、图源、报告事实及文件摘要；结果另存FINAL_QA.json，不用数值阶段PASS替代最终交付验收。",
        "## 9. 全部失败与异常",
        tab(["编号","阶段","事件","处置/边界"],[[i["id"],i["phase"],i["description"],i["resolution"]] for i in incidents]),
        f"逐条证据路径见{link('完整失败/异常台账','evidence/failure_anomaly_ledger.csv')}。旧失败、空日志、合成测试、恢复副本与报告首轮失败全部保留；没有跳失败状态、重写旧结果或为了PASS降低科研检验标准。[E09]",
        "## 10. 解释边界与可用性",
        "结论仅覆盖冻结LOCAL数据、既有目标定义、当前单种子与历史配置。RFE曲线提供特征子集敏感性证据，不能从单种子相关误差曲线推断统计显著胜出；同一训练内CV参与自适应路径和摘要，不是嵌套验证。历史测试已查看，不可将探索性最优k包装为预注册独立验证。TabM仍是用户拟采用候选，但本交付不改变用户主模型决定或启动任何补充调参。",
        "技术交付由本地文件、可重算指标及摘要链组成；权重与原始逐状态JSON继续位于冻结032/历史来源目录，角色清单提供全部路径和SHA，避免复制数千模型导致来源混淆。作者对最终学术解释、署名和投稿的批准未由自动化代行；未作投稿。",
        "天气采集、远端状态查询与旧STOP_REQUESTED保持停止/原样。E-SST仍EXCLUDED_USER_DECISION_DEADLINE；RMOP/NWP–PRM仍EXCLUDED_USER_DECISION；MTS仍NOT_COMPUTED_USER_DECISION。未恢复T013/1024、UK重训或快变CDF重估，未安装环境、外联作者、付费或外传数据/凭据。",
        "## 附录A：测试最优k（仅探索性）",
        "以下直接使用测试总体RMSE找最小k，存在测试选择偏差；只满足完整报告用途，不用于决定删列/轮数/种子/再训练，不作为独立优越性证据。",
        tab(["模型","探索性测试最优k","测试RMSE/dB","测试R²"],[
            [LABELS[m],int(best.loc[m,"k"]),n(best.loc[m,"rmse_db"]),n(best.loc[m,"r2"])] for m in MODELS]),
        "## 附录B：证据索引",
    ])
    evidence=[
        ("E01","独立汇总审计","evidence/INDEPENDENT_AGGREGATE_QA.json"),
        ("E02","1560状态QA和摘要链","evidence/state_qa_index_1560.csv"),
        ("E03","15605条完整测试天气指标","tables/all_test_metrics_15605_rows.csv"),
        ("E04","全156/CV描述/测试探索三套摘要","tables/representative_points_300_rows.csv"),
        ("E05","样本/天气/实际日期清单","tables/weather_sample_coverage.csv"),
        ("E06","6240角色完整配置/训练元数据/模型SHA","tables/model_roles_6240_with_sha.csv"),
        ("E07","独立RFE路径及保留删除名单","tables/independent_rfe_paths_1560.csv"),
        ("E08","ITU三异常小时与天气归属对照","tables/ITU_three_excluded_hours_with_canonical_weather.csv"),
        ("E09","失败/异常台账","evidence/failure_anomaly_ledger.csv"),
        ("E10","图源与说明","figures/figure_manifest.json"),
        ("E11","逐模型拟合计账","evidence/fit_accounting_by_model.csv"),
        ("E12","最终交付QA","evidence/FINAL_QA.json")
    ]
    report.append(tab(["ID","证据"],[[i,link(label,path)] for i,label,path in evidence]))
    save(OUT/"REPORT_中文.md","\n\n".join(report))
    claim_map=[
        dict(id="C01",claim="Full156 TabM third on main687 overall RMSE",source="tables/all_test_metrics_15605_rows.csv",
             filter=dict(population="ML_687",weather="overall",model="tabm",k=156),
             values={v:float(t[v]) for v in ["rmse_db","r2"]},rank=3),
        dict(id="C02",claim="TabM training-CV descriptive minimum k9, not test minimum k11",source="tables/representative_points_300_rows.csv",
             filter=dict(basis="training_cv_min_descriptive",population="ML_687",weather="overall",model="tabm"),
             values={v:float(tc[v]) for v in ["k","rmse_db","r2"]},
             relative_rmse_reduction_percent=float(reduction)),
        dict(id="C03",claim="5604 new roles + 1 Bai selection; 635 historical + 1 in-run reuse",
             source="evidence/INDEPENDENT_AGGREGATE_QA.json",values=dict(role_slots=6240,new_fit_calls_total=5605,historical_weight_reuse=635,in_run_bai_selected_weight_reuse=1)),
        dict(id="C04",claim="Main687 has 322 clear,51 rain,158 overcast,156 other; common684 has321,51,156,156",
             source="tables/weather_sample_coverage.csv",values=dict(test=[322,51,158,156],test_itu_common684=[321,51,156,156])),
    ]
    jwrite(OUT/"evidence/CLAIM_EVIDENCE_MAP.json",claim_map)
    readme=[
        "# T014 最终交付索引",
        "本交付为Frozen032科研结果的只读审计和整理；最终通过状态见FINAL_QA.json。先读中文报告，再按需求取CSV、路径或逐小时预测。",
        link("中文报告","REPORT_中文.md"),
        tab(["交付项","文件/目录","规模"],[
            ["每状态详细主表",link("all_states_1560.csv","tables/all_states_1560.csv"),"1560行"],
            ["全部主/共同样本天气指标",link("完整指标CSV","tables/all_test_metrics_15605_rows.csv"),"15605行；687与684以population分开"],
            ["保留特征",link("保留特征CSV","tables/retained_features_122460_rows.csv"),"122460行，order从0开始"],
            ["独立路径",link("路径总表","tables/independent_rfe_paths_1560.csv")+"；paths/内10个JSON","每模型156→1，含累计已删除名单"],
            ["逐小时预测","hourly/内10个模型CSV及ITU CSV","每模型107172行；合计1071720学习模型测试行"],
            ["角色/模型/配置",link("6240角色清单","tables/model_roles_6240_with_sha.csv"),"完整模型路径、SHA、config_json、fit_metadata_json"],
            ["输入SHA",link("1028来源","evidence/frozen_inputs_1028_sha.csv"),"冻结数据/代码/历史来源"],
            ["样本和日期",link("覆盖表","tables/weather_sample_coverage.csv"),"每组n/日期数/实际日期列表"],
            ["ITU异常",link("3小时原值及归属","tables/ITU_three_excluded_hours_with_canonical_weather.csv"),"负液态差不裁零；主687保留"],
            ["失败/异常",link("台账","evidence/failure_anomaly_ledger.csv"),"包含训练、恢复、合成、报告阶段"],
            ["图表","figures/内4组PNG与PDF","完整RMSE/R²路径、天气、684比较"],
            ["最终QA",link("交付QA","evidence/FINAL_QA.json"),"导出CSV反算、表/图/报告/来源一致性"]
        ]),
        "## 数据字典与使用",
        "model为冻结模型ID；k为保留输入特征数。population=ML_687是十模型完整官方测试集，COMMON_684是ITU合法小时与学习模型同样本补充；二者不得混排。weather=overall/clear_dry/rain/dry_overcast/other。RMSE/MAE/Bias单位dB，Bias=预测−真值；r2为决定系数，pearson_r_squared为相关系数平方。",
        "CSV空值表示未定义数值而非0；r2_reason/pearson_reason保留原因。日期列表为北京时间实际覆盖日期；预测表hour_bjt含+08:00。每模型逐小时CSV里的itu_common_684=False正是三个异常小时；可筛True得到同684小时补充，无需再推理。",
        "retained_names_json/removed_so_far_json/config_json/fit_metadata_json为JSON；模型权重与完整每轮round_history仍保存在冻结来源，按model_file及state_path定位。路径JSON的retained保留原有顺序，removed_so_far为累计已删除顺序；k1无removed_next。",
        "代表点三种basis分别为full_156、training_cv_min_descriptive、test_min_exploratory。后两者不是新的训练或预注册最优模型；特别是测试最优不能当作无选择偏差的泛化性能。",
        "本包没有独立复制6240模型权重；模型和输入SHA清单指向本工作区冻结文件。若搬移项目，应保留这些相对路径所指文件。PACKAGE_SHA256.csv记录本交付文件（不包含清单自身）的校验摘要。",
        "## 复用与范围",
        "635次历史复用、1次Bai本轮选轮权重复用、5604新角色拟合；加Bai选轮1次共5605新拟合调用。只读交付程序不训练。旧032/033输出未改，首次报告审计失败证据在上一级evidence，后续完整重审在本delivery_002。",
        "独立QA指不同算式/实现与摘要链复核，不声称额外第三方专家或人工作者已验证。作者解释/投稿批准另行，不属于自动化代办。"
    ]
    save(OUT/"README.md","\n\n".join(readme))
    print("Chinese report, evidence map and complete incident ledger generated")


if __name__=="__main__":
    main()
