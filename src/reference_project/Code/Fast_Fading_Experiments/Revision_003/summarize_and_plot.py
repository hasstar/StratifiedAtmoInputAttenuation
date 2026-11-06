# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Descriptive T016 delivery from independently audited frozen results."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,time,json,hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from common import *

def subsets(frame):
    for split in ('validation','test'):
        part=frame[(frame.dataset=='LOCAL')&(frame.split==split)]
        for event in ('overall','clear_dry','rain','dry_overcast','other'):
            yield 'LOCAL_'+split,'local',event,part if event=='overall' else part[part.event==event]
    for station in ('pooled','chilbolton','sparsholt'):
        part=frame[frame.dataset=='UK']
        if station!='pooled':part=part[part.station==station]
        for event in ('overall','clear_dry','rain','dry_overcast','other'):
            yield 'UK',station,event,part if event=='overall' else part[part.event==event]

def aggregate(frame):
    rng=np.random.default_rng(20261024);rows=[]
    metrics=['fast_crps_db','fast_fair_crps_db','fast_zero_crps_db','combined_crps_db','combined_fair_crps_db','slow_only_crps_db','combined_minus_slow_db','observed_sd_db','generated_sd_mean_db','sd_absolute_error_db','acf_absolute_error','vs_db2','fast_coverage90','combined_coverage90','generated_mean_q','generated_drift_db_mean']
    for cohort,station,event,g in subsets(frame):
        row={'cohort':cohort,'station':station,'event':event,'hours':len(g),'dates':g.date.nunique(),'valid_seconds':int(g.n_seconds.sum()),'complete3600_hours':int(g.n_seconds.eq(3600).sum()),'min_valid_seconds':int(g.n_seconds.min()) if len(g) else None}
        row.update({m:float(g[m].mean()) for m in metrics})
        row['combined_improvement_percent']=100*(row['slow_only_crps_db']-row['combined_crps_db'])/row['slow_only_crps_db'] if len(g) and row['slow_only_crps_db'] else np.nan
        days=g.groupby('date').combined_minus_slow_db.agg(['sum','count','mean'])
        row['delta_date_equal_db']=float(days['mean'].mean())
        if len(days)>1:
            ix=rng.integers(0,len(days),size=(2000,len(days)))
            boot=days['sum'].to_numpy()[ix].sum(1)/days['count'].to_numpy()[ix].sum(1)
            row['delta_cluster_ci95_low'],row['delta_cluster_ci95_high']=map(float,np.quantile(boot,[.025,.975]))
        else:row['delta_cluster_ci95_low']=row['delta_cluster_ci95_high']=np.nan
        for tag in ('95','99'):
            cols=[f'observed_duration{tag}_mean_s',f'generated_duration{tag}_mean_s']
            matched=g.dropna(subset=cols)
            row[f'duration{tag}_matched_hours']=len(matched)
            row[f'observed_duration{tag}_matched_mean_s']=float(matched[cols[0]].mean())
            row[f'generated_duration{tag}_matched_mean_s']=float(matched[cols[1]].mean())
            row[f'duration{tag}_absolute_error_s']=float((matched[cols[0]]-matched[cols[1]]).abs().mean())
            row[f'observed_duration{tag}_complete_count']=int(g[f'observed_duration{tag}_complete_count'].sum())
            row[f'observed_duration{tag}_censored_count']=int(g[f'observed_duration{tag}_censored_count'].sum())
        rows.append(row)
    return pd.DataFrame(rows)

def lag_summary(frame):
    rows=[]
    for p in (OUT/'hours').glob('*.npz'):
        with np.load(p) as z:
            meta=json.loads(str(z['metadata']));d=json.loads(str(z['diagnostics']))
            obs=np.array([np.nan if x is None else x for x in d['observed_acf']])
            gen=np.array(d['generated_acf'],dtype=float)
            for i,lag in enumerate(LAGS):
                rows.append({**{k:meta[k] for k in ('dataset','split','station','hour','event')},'lag_seconds':lag,'observed_acf':obs[i],'generated_acf_mean':np.nanmean(gen[:,i]),'pair_count':d['pair_counts'][i]})
    all_lags=pd.DataFrame(rows);aggregated=[]
    for cohort,station,event,g in subsets(all_lags):
        for lag,a in g.groupby('lag_seconds'):
            valid=a.dropna(subset=['observed_acf','generated_acf_mean'])
            aggregated.append(dict(cohort=cohort,station=station,event=event,lag_seconds=lag,hours_with_defined_acf=len(valid),observed_acf=float(valid.observed_acf.mean()),generated_acf_mean=float(valid.generated_acf_mean.mean()),acf_mae=float((valid.observed_acf-valid.generated_acf_mean).abs().mean()),valid_pair_count=int(valid.pair_count.sum())))
    return pd.DataFrame(aggregated)

def figures(summary,lags,frame):
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0,style='whitegrid',font='DejaVu Sans')
    dest=DELIVERY/'图表';dest.mkdir(exist_ok=True)
    cohorts=[('LOCAL_validation','local','LOCAL validation'),('LOCAL_test','local','LOCAL test'),('UK','chilbolton','UK Chilbolton'),('UK','sparsholt','UK Sparsholt')]
    weather=['overall','clear_dry','rain','dry_overcast','other'];labels=['All','Clear','Rain','Overcast','Other']
    gray='#6B7280';blue='#4575b4';legend=dict(frameon=True,facecolor='white',framealpha=.8,edgecolor='lightgrey',labelcolor='dimgrey',fontsize=8)
    def layout(title,ylabel):
        fig=plt.figure(figsize=(14,8),dpi=150);gs=gridspec.GridSpec(2,2,figure=fig,wspace=.12,hspace=.30,left=.08,right=.99,top=.90,bottom=.12)
        axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
        fig.suptitle(title,color='dimgrey',fontsize=14,y=.98)
        for i,(ax,(_,_,name)) in enumerate(zip(axes,cohorts)):
            ax.set_title(r'$\bf{('+chr(97+i)+r')}$ '+name,loc='left',fontsize=11)
            ax.grid(False);ax.tick_params(axis='both',which='both',length=0,labelcolor='dimgrey',labelsize=9)
            if i%2==0:ax.set_ylabel(ylabel,color='dimgrey',fontsize=10)
            else:ax.tick_params(labelleft=False)
        return fig,axes
    def save(fig,axes,name,note):
        fig.text(.08,.025,note,color='dimgrey',fontsize=9)
        sns.despine(left=True,bottom=True)
        for ax in axes:ax.patch.set_edgecolor('lightgrey');ax.patch.set_linewidth(.8)
        for ext in ('png','pdf'):fig.savefig(dest/(name+'.'+ext),dpi=150,bbox_inches='tight')
        plt.close(fig)
    def bars(name,title,ylabel,c1,c2,l1,l2,note):
        fig,axes=layout(title,ylabel);maxv=0
        selected=[]
        for cohort,station,_ in cohorts:
            g=summary[(summary.cohort==cohort)&(summary.station==station)].set_index('event').reindex(weather)
            selected.append(g);vals=g[[c1,c2]].to_numpy();maxv=max(maxv,np.nanmax(vals))
        for ax,g in zip(axes,selected):
            x=np.arange(5)
            ax.bar(x-.18,g[c1],width=.34,color=gray,label=l1,hatch='//',alpha=.85)
            ax.bar(x+.18,g[c2],width=.34,color=blue,label=l2)
            for j,row in enumerate(g.itertuples()):
                a=getattr(row,c1);b=getattr(row,c2)
                for xx,v in ((j-.18,a),(j+.18,b)):
                    if np.isfinite(v):ax.text(xx,v+maxv*.016,f'{v:.3f}' if maxv<2 else f'{v:.1f}',ha='center',va='bottom',fontsize=7,color='dimgrey',rotation=45)
            ax.set_xticks(x,labels);ax.set_ylim(0,maxv*1.29);ax.legend(**legend)
        save(fig,axes,name,note)
    # --- Data / Plot ---
    bars('01_快变_幅度CRPS','Fast fading: 32-member empirical CRPS','CRPS (dB; lower is better)','fast_zero_crps_db','fast_crps_db','Zero-fast baseline','Frozen generator','Each hour: mean over valid native seconds; then equal-hour mean. Weather-specific training CDFs remain frozen.')
    bars('02_快变_标准差','Fast fading: within-hour standard deviation','Within-hour SD (dB)','observed_sd_db','generated_sd_mean_db','Observed','Generated (32 mean)','Generated curves use the same valid-second mask as observations. These are descriptive means, not 32 independent observed samples.')
    bars('04_快变_衰落持续时间','Fast fading: complete fade durations above training F95','Mean complete fade duration (s)','observed_duration95_matched_mean_s','generated_duration95_matched_mean_s','Observed','Generated (32 mean)','Matched hours with complete episodes in both summaries. Gap/hour-edge episodes are censored; support counts are in the table.')
    bars('05_TabM_慢快组合CRPS','TabM k10: slow-only versus slow + fast','Second-wise CRPS (dB; lower is better)','slow_only_crps_db','combined_crps_db','Slow only (point)','Slow + fast (32 members)','Same observed seconds and fixed TabM predictions. No pointwise RMSE criterion is applied to randomly generated curves.')
    fig,axes=layout('Fast fading: temporal autocorrelation','Mean ACF')
    for ax,(cohort,station,_) in zip(axes,cohorts):
        g=lags[(lags.cohort==cohort)&(lags.station==station)&(lags.event=='overall')].dropna(subset=['observed_acf','generated_acf_mean']).sort_values('lag_seconds')
        ax.semilogx(g.lag_seconds,g.observed_acf,'o-',color=gray,lw=1.7,ms=4,label='Observed')
        ax.semilogx(g.lag_seconds,g.generated_acf_mean,'s--',color=blue,lw=1.8,ms=4,label='Generated (32 mean)')
        ax.set_ylim(-1,1);ax.axhline(0,color='lightgrey',lw=.8);ax.set_xlabel('Lag (s; log scale)',color='dimgrey',fontsize=10);ax.legend(**legend)
    save(fig,axes,'03_快变_时间相关性','Overall equal-hour curves; an interval crossing any real missing second is excluded. Synthetic block and phase seams are retained.')
    fig,axes=layout('Slow + fast: preselected first rain hour in each cohort','Attenuation (dB)')
    minima=[];maxima=[];csvs=[]
    for ax,(cohort,station,title) in zip(axes,cohorts):
        if cohort.startswith('LOCAL'):g=frame[(frame.dataset=='LOCAL')&(frame.split==cohort.split('_')[1])]
        else:g=frame[(frame.dataset=='UK')&(frame.station==station)]
        r=g[g.event=='rain'].sort_values('hour').iloc[0]
        ident=hashlib.sha256((r.dataset+'|'+r.split+'|'+r.station+'|'+r.hour).encode()).hexdigest()[:20]
        with np.load(OUT/'hours'/(ident+'.npz')) as z:
            actual=z['observed_fast_db']+r.truth_db;gen=z['generated_fast_db']+r.prediction_db
            low,high=np.quantile(gen,[.05,.95],axis=0);x=np.arange(3600)/60
            ax.fill_between(x,low,high,color=blue,alpha=.20,label='32-member 5–95% band')
            ax.plot(x,gen[0],color=blue,lw=.7,ls='--',label='Realization 1')
            ax.plot(x,actual,color='#252525',lw=.7,label='Observed')
            ax.set_xlabel('Minute within hour',color='dimgrey',fontsize=10);ax.legend(**legend)
            ax.text(.02,.97,r.hour[:16],transform=ax.transAxes,va='top',color='dimgrey',fontsize=8)
            minima.append(min(np.nanmin(actual),np.min(gen[0]),np.min(low)));maxima.append(max(np.nanmax(actual),np.max(gen[0]),np.max(high)))
            csvs.append(pd.DataFrame({'cohort':title,'hour':r.hour,'second':np.arange(3600),'observed_db':actual,'realization0_db':gen[0],'p05_db':low,'p95_db':high}))
    pad=(max(maxima)-min(minima))*.08
    for ax in axes:ax.set_ylim(min(minima)-pad,max(maxima)+pad)
    pd.concat(csvs).to_csv(dest/'06_示例曲线数据.csv',index=False,encoding='utf-8-sig')
    save(fig,axes,'06_慢快组合_预定小时示例','The chronologically first rain hour is fixed, not selected for visual agreement. Random paths are evaluated statistically, not second-by-second matching.')

def report(summary,frame):
    rows=['# RFE与TabM冻结慢快组合补充结果','','## 数据与评价口径','','- RFE：10模型、1560状态、1550次删除；第0轮为156参数基线，k1不再删除。参数ID对应字典，完整原名及指标有独立CSV。','- 快变生成器：本地2404训练小时冻结的天气经验CDF与256秒片段库；每小时32条3600秒序列。英国不拟合CDF、不抽取英国模板。','- 慢变：本轮RFE的TabM k=10、seed17既有预测；验证343小时、测试687小时、英国5593主标签合格小时。','- CRPS：每个有效秒由32条序列形成预测分布，先逐秒计算、再小时内等权平均，最后小时等权汇总。缺测秒不填补，不伪称所有小时都评价了3600秒。','- 快变CRPS对F评价；组合CRPS对Ahat_TabM+F评价。慢变单独的逐秒CRPS等于绝对误差，三者含义不能混用。','- SD、ACF和相对快变阈值的持续时间检验时间结构；加上常数慢变不改变这些量。它们不能替代组合幅度分布的CRPS。','','## 总体主结果','','| 范围 | 小时数 | 快变CRPS/dB | 慢变单独CRPS/dB | 慢快组合CRPS/dB | 组合相对改善/% |','|---|---:|---:|---:|---:|---:|']
    zh={('LOCAL_validation','local'):'本地验证',('LOCAL_test','local'):'本地测试',('UK','pooled'):'英国两站合并',('UK','chilbolton'):'英国Chilbolton',('UK','sparsholt'):'英国Sparsholt'}
    for r in summary[summary.event=='overall'].itertuples(index=False):
        rows.append(f'| {zh[(r.cohort,r.station)]} | {r.hours} | {r.fast_crps_db:.6f} | {r.slow_only_crps_db:.6f} | {r.combined_crps_db:.6f} | {r.combined_improvement_percent:.2f} |')
    rows += ['', '“相对改善”正值表示组合CRPS较低，负值表示较高；原值保留。分天气、晴雨阴其他、各站点和日期聚类区间见汇总表，不能从总体行代替分组结论。','','## 覆盖与不确定性','',f'- 实际评价{int(frame.n_seconds.sum()):,}个有效秒，其中{int(frame.n_seconds.eq(3600).sum()):,}个小时具有完整3600秒；英国原5664候选小时中的71个标签质量排除保留。','- 英国的雨/阴掩码与T015一致；原“other”另用既有p/s/TCC阈值分出clear_dry，未改历史事件标签。天气是ERA5代理，不是现场天气观测。','- 持续时间只把两侧边界都观测到的衰落计为完整；缺口及小时端点删失次数、非空支持小时数另存。不把无完整衰落记为零秒。','- 组合减慢变CRPS的95%区间采用2000次日期聚类重采样；英国两站同UTC日联合保留。日期等权结果另列；这不是相邻日期独立性的证明，也不把秒或32条随机曲线当观测重复。','- 本地验证/测试日期与训练日期有历史重叠，英国及现有测试集已在以前工作中查看；本补充是冻结方案描述性评价，不声明新盲测或重新选择最优k。','','## 文件','','- RFE_每模型每轮删除与保留参数.xlsx：中文参数说明、完整逐轮轨迹、参数字典。','- RFE_完整参数原名与指标.csv：可追溯原字段及已交付指标。','- RFE_保留参数长表_122460行.csv：逐模型、逐k、逐输入顺序记录。','- 快变与TabM组合_分天气汇总.csv：25组总体/天气/站点汇总及有效样本量。','- 快变与TabM组合_逐小时结果.csv：6623小时的CRPS、质量覆盖和时间诊断。','- 快变_ACF分天气汇总.csv：11个滞后的配对有效样本量和观测/生成ACF。','- 图表/：6组PDF＋PNG及示例曲线数值。','- 完整32条曲线、逐秒CRPS、种子、相位及完整/删失衰落记录保存在Outputs/Experiments/Fast_Fading_Revision_003_TabM_Ensemble/hours，未重复拷贝至投稿材料。','','## QA与来源','','独立数值核验见随附independent_numeric_qa.json；图像是否完成视觉检查以VISUAL_QA.json为准，不能仅凭生成图片宣布完成。旧代码、模型、CDF和原提交四文件SHA检查均保持；0训练、0预处理拟合、0CDF拟合、0新模型推理。CRPS计算采用[官方scoringrules文档的有限集合定义](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html)，旧201/401分位数积分CRPS并未冒充本次32成员逐秒CRPS。']
    (DELIVERY/'结果说明.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')

def main():
    """Save audited summary tables, explanatory notes and six PDF/PNG figures."""
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        while not (OUT/'independent_numeric_qa.json').exists():
            need(not any((OUT/'failures').glob('*.json')),'A recorded failure requires inspection before delivery')
            time.sleep(5)
    need(read(OUT/'independent_numeric_qa.json')['status']=='PASS','Independent QA required')
    frame=pd.read_csv(OUT/'hourly_results.csv');summary=aggregate(frame);lags=lag_summary(frame)
    need(len(summary)==25 and len(lags)==275,'Summary dimensions')
    summary.to_csv(DELIVERY/'快变与TabM组合_分天气汇总.csv',index=False,encoding='utf-8-sig')
    frame.to_csv(DELIVERY/'快变与TabM组合_逐小时结果.csv',index=False,encoding='utf-8-sig')
    lags.to_csv(DELIVERY/'快变_ACF分天气汇总.csv',index=False,encoding='utf-8-sig')
    write(DELIVERY/'independent_numeric_qa.json',read(OUT/'independent_numeric_qa.json'))
    figures(summary,lags,frame);report(summary,frame)
    write(OUT/'summary_complete.json',{'state':'TABLES_AND_FIGURES_PENDING_VISUAL_QA','utc':now(),'summary_groups':25,'lag_rows':275,'figure_pairs':6,'script_sha256':sha(CODE/'summarize_and_plot.py')})
    print(summary[summary.event.eq('overall')][['cohort','station','hours','fast_crps_db','slow_only_crps_db','combined_crps_db','combined_improvement_percent']].to_string(index=False),flush=True)
if __name__=='__main__':main()
