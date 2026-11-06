"""Export immutable RFE transitions; no model loading or training."""
import json
import pandas as pd
from common import *

def main():
    DELIVERY.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(RFE/'tables/independent_rfe_paths_1560.csv',keep_default_na=False)
    detail=pd.read_csv(RFE/'tables/retained_features_122460_rows.csv',keep_default_na=False)
    metrics=pd.read_csv(ROOT/'最终提交的修改材料/plot_data_1560_states.csv')
    need(len(a)==1560 and a.model.nunique()==10,'RFE state count')
    need(len(detail)==122460,'RFE feature detail count')
    first=json.loads(a[a.k.eq(156)].iloc[0].retained_names_json)
    ids={f:f'F{i+1:03d}' for i,f in enumerate(first)}
    names={'tabm':'TabM','bai_lstm':'BaiLSTM','bilstm':'BiLSTM','transformer':'Transformer','mlp':'MLP','catboost':'CatBoost','gcn':'GCN','random_forest':'Random Forest','svr_rbf':'SVR-RBF','elastic_net':'Elastic Net'}
    rows=[]; full=[];checks=0
    for model,g in a.groupby('model',sort=False):
        prev=None;history=[]
        for r in g.sort_values('k',ascending=False).itertuples(index=False):
            kept=json.loads(r.retained_names_json);gone=json.loads(r.removed_so_far_json)
            need(len(kept)==r.k and len(set(kept))==r.k,'Retained count')
            entered=''
            if prev is None:need(kept==first and gone==[],'Full-set order')
            else:
                entered=prev['next'];history.append(entered)
                need(kept==[x for x in prev['kept'] if x!=entered],'Adjacent state/order mismatch')
                checks+=1
            need(gone==history,'Removed history mismatch')
            next_removed=r.removed_next
            need((r.k==1 and not next_removed) or (r.k>1 and next_removed in kept),'Next-delete identity')
            observed=detail[detail.model.eq(model)&detail.k.eq(r.k)].sort_values('order').feature.tolist()
            need(observed==kept,'Long table disagree')
            rows.append([names[model],156-r.k,r.k+1 if r.k<156 else None,ids.get(entered,''),entered,r.k,' '.join(ids[f] for f in kept),ids.get(next_removed,''),next_removed])
            full.append(dict(model=model,completed_deletions=156-r.k,features_before_deletion=r.k+1 if r.k<156 else None,removed_to_enter_state=entered,k=r.k,retained_names_json=json.dumps(kept),removed_so_far_json=json.dumps(gone),removed_next=next_removed))
            prev={'kept':kept,'next':next_removed}
    need(checks==1550,'Deletion count')
    joined=pd.DataFrame(full).merge(metrics,on=['model','k'],validate='one_to_one')
    need(len(joined)==1560,'Metric join count')
    dictionary=[]
    for f in first:
        r=detail[detail.feature.eq(f)].iloc[0]
        dictionary.append([ids[f],f,str(r.variable).replace('_',' '),str(r.level_hpa) if r.level_hpa!='' else 'surface/column',r.group])
    headers=['模型','已完成删除轮数','本轮删除前参数数','本轮删除参数ID','本轮删除参数原名','当前保留参数数k','当前保留参数ID（原输入顺序）','下一轮删除ID','下一轮删除原名']
    pd.DataFrame(rows,columns=headers).to_csv(DELIVERY/'RFE_逐轮删除与保留.csv',index=False,encoding='utf-8-sig')
    joined.to_csv(DELIVERY/'RFE_完整参数原名与指标.csv',index=False,encoding='utf-8-sig')
    detail.to_csv(DELIVERY/'RFE_保留参数长表_122460行.csv',index=False,encoding='utf-8-sig')
    write(DELIVERY/'rfe_workbook_data.json',{'headers':headers,'rows':rows,'dictionary_headers':['参数ID','原始字段名','参数英文名称','压力层hPa／类别','来源分组'],'dictionary':dictionary})
    sources={relative(p):sha(p) for p in [RFE/'tables/independent_rfe_paths_1560.csv',RFE/'tables/retained_features_122460_rows.csv',RFE/'evidence/FINAL_QA.json',ROOT/'最终提交的修改材料/plot_data_1560_states.csv']}
    write(DELIVERY/'RFE_QA.json',{'status':'PASS','utc':now(),'states':1560,'models':10,'verified_deletions':checks,'retained_feature_rows':122460,'metrics_join_rows':len(joined),'baseline_round':0,'terminal_k1_no_next_deletion':True,'sources_sha256':sources})
    print(json.dumps({'status':'PASS','states':1560,'deletions':checks},ensure_ascii=False))
if __name__=='__main__':main()
