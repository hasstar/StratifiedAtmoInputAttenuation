"""Final tables only after all fixed RFE paths have passed independent state QA."""
import runtime as r

def build(data,m):
    directory=r.OUT/'final'; directory.mkdir(parents=True,exist_ok=True)
    rows=[]; features=[]; inventory={}; files={}
    for name in r.MODELS:
        expected=list(m['feature_names']); inventory[name]=[]
        for k in range(156,0,-1):
            d=r.state_dir(name,k); q=r.read(d/'qa.json'); s=r.read(d/'state.json'); e=r.read(d/'evaluation.json')
            r.require(q['status']=='PASS' and q['state_sha256']==r.sha(d/'state.json') and q['evaluation_sha256']==r.sha(d/'evaluation.json'),'Final state digest')
            r.require(s['feature_names']==expected and len(expected)==k,'Final independent path')
            for group,score in e['results']['test']['metrics'].items(): rows.append(dict(model=name,k=k,population='ML_687',weather=group,**score))
            for group,score in e['results']['test_itu_common684']['metrics'].items(): rows.append(dict(model=name,k=k,population='COMMON_684',weather=group,**score))
            for order,feature in enumerate(expected):
                prefix,variable=feature.split('__',1)
                features.append(dict(model=name,k=k,order=order,feature=feature,variable=variable,level_hpa=None if prefix=='surface' else int(prefix[1:]),group=prefix,removed_next=s['removed_feature_next']==feature))
            inventory[name].append(dict(k=k,cv_mean_rmse_db=s['cv_mean_rmse_db'],cv_se_rmse_db=s['cv_se_rmse_db'],removed_next=s['removed_feature_next']))
            files[r.rel(d/'qa.json')]=r.sha(d/'qa.json')
            if k>1: expected=[x for x in expected if x!=s['removed_feature_next']]
    itu=r.read(r.OUT/'itu_baseline.json')
    for group,score in itu['results']['metrics'].items(): rows.append(dict(model='combined_ITU_approx',k=None,population='COMMON_684',weather=group,**score))
    for filename,table in (('all_test_metrics.csv',rows),('retained_features_all_steps.csv',features)):
        p=directory/filename
        if not p.exists(): r.pd.DataFrame(table).to_csv(p,index=False,encoding='utf-8-sig',mode='x')
    if not (directory/'path_inventory.json').exists(): r.emit(directory/'path_inventory.json',inventory)
    else: r.require(r.read(directory/'path_inventory.json')==inventory,'Saved final path inventory changed')
    if not (directory/'numeric_qa.json').exists(): r.emit(directory/'numeric_qa.json',dict(status='PASS',models=10,states=1560,expected_test_n=687,common_n=684,state_qa_sha256=files,utc=r.now()))
    report='''# T014 数值交付（最终人工复核待完成）

10个学习模型各自156→1的RFE路径与ITU固定物理基线已生成。每个状态均保存训练内选择、权重、逐小时预测、完整测试指标和独立QA。详细指标见all_test_metrics.csv；每一步保留/删除参数见retained_features_all_steps.csv。

R²为决定系数，Pearson r与r²另列；阴天指干燥阴天。模型主表687小时；ITU及全部模型684小时共同样本另列，不混排。没有保证TabM排名第一，也没有以测试成绩改变RFE路径。历史测试已查看，测试最优k仅能作探索性描述。

学习模型使用各自已有LOCAL适配的固定配置，历史优化预算并非相同。单种子17不能称多种子结果。Bai为单时刻天气输入适配，结构化网络的序列/图是大气压力层而非过去信道。ITU为论文三分量的已核验小时近似，未使用测试拟合的偏移。

当前是数值封装完成，不自动代表最终报告已交付。下一步需复查汇总表与逐状态文件一致性、绘制RFE曲线、给出各模型全156及训练内选定状态和完整负结果、中文结论及最终QA，然后通过原线程交付并暂停tvt。
'''
    if not (directory/'REPORT_DRAFT.md').exists():
        with (directory/'REPORT_DRAFT.md').open('x',encoding='utf-8') as stream: stream.write(report)
