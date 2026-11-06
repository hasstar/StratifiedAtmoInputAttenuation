# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Requested 10-curve UK plots: frozen own-RFE k=1..10, with TabM glow."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import seaborn as sns
from freeze import ROOT, OUT, sha, read, emit, now, need
from transport import stage_lock

def main():
    """Render original UK metrics without smoothing or removal of negative R².

    Parameters
    ----------
    None
        Reads the independently verified UK 100-state CSV.

    Saves
    -----
    Six separate PNG/PDF figures and exact artist-value/export QA.
    """
    with stage_lock('six_uk_figures'):
        # --- Data ---
        qa=read(OUT/'evaluation/independent_numeric_qa.json');need(qa['status']=='PASS','Numerical gate')
        source=OUT/'evaluation/UK_100_states_six_metrics.csv'
        need(sha(source)==qa['numeric_files'][source.name],'Verified plotting source SHA')
        data=pd.read_csv(source,float_precision='round_trip')
        need(not data.isna().any().any() and len(data)==100,'Complete plotting data')
        data=data.dropna();ks=np.arange(1,11)
        styles=[('tabm','TabM (Ours)','#0072B2','-','o'),('bai_lstm','Bai LSTM','#D55E00','--','s'),
            ('bilstm','BiLSTM','#009E73','-.','^'),('transformer','Transformer','#7A5195',':','D'),
            ('mlp','MLP','#008EAA',(0,(6,1)),'v'),('catboost','CatBoost','#B3446C',(0,(3,1,1,1)),'P'),
            ('gcn','GCN','#303030',(0,(5,1,1,1,1,1)),'X'),('random_forest','Random Forest','#967100',(0,(1,1)),'<'),
            ('svr_rbf','SVR-RBF','#777777',(0,(3,2)),'>'),('elastic_net','Elastic Net','#8F4E2F',(0,(7,2,1,2)),'h')]
        need(set(data.model)=={s[0] for s in styles},'10 model curves')
        for model,group in data.groupby('model'):np.testing.assert_array_equal(group.sort_values('k').k,ks)
        specs=[('F01_rain_R2','rain','r2','雨天R²',735),('F02_overcast_R2','dry_overcast','r2','阴天R²（干燥阴天）',2317),
            ('F03_overall_R2','overall','r2','总体R²',5593),('F04_rain_RMSE','rain','rmse_db','雨天RMSE',735),
            ('F05_overcast_RMSE','dry_overcast','rmse_db','阴天RMSE（干燥阴天）',2317),('F06_overall_RMSE','overall','rmse_db','总体RMSE',5593)]
        # --- Style Setup ---
        sns.set_theme(font_scale=1.0,style='whitegrid',font='DejaVu Sans')
        plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'path.simplify':False,'savefig.facecolor':'white','text.color':'dimgrey'})
        chinese=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        folder=OUT/'figures';folder.mkdir(exist_ok=False);records=[]
        for fid,weather,metric,title,count in specs:
            # --- Plot ---
            fig,ax=plt.subplots(figsize=(12.8,7.6),dpi=150)
            fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.77)
            curves=[]
            for model,label,color,style,marker in styles:
                part=data[data.model.eq(model)].sort_values('k');values=part[f'{weather}_{metric}'].to_numpy();tabm=model=='tabm'
                need(part[f'{weather}_n'].eq(count).all(),'Same scored cohort per curve')
                line,=ax.plot(ks,values,color=color,linestyle=style,marker=marker,linewidth=2.8 if tabm else 1.35,
                    markersize=5.2 if tabm else 4.5,markerfacecolor='white',markeredgewidth=1,label=label,zorder=5 if tabm else 3)
                if tabm:line.set_path_effects([pe.Stroke(linewidth=10.5,foreground='#00E5FF',alpha=.07),
                    pe.Stroke(linewidth=7,foreground='#00E5FF',alpha=.13),pe.Stroke(linewidth=4.8,foreground='#00E5FF',alpha=.26),pe.Normal()])
                np.testing.assert_array_equal(line.get_xdata(),ks);np.testing.assert_array_equal(line.get_ydata(),values)
                curves.append(dict(model=model,points=10,minimum=float(values.min()),maximum=float(values.max()),negative=int((values<0).sum())))
            need(len(ax.lines)==10,'Exactly 10 data curves')
            ax.set_xlim(.7,10.3);ax.set_xticks(ks)
            ax.set_xlabel('大气参数个数（各模型独立RFE保留输入数）',fontproperties=chinese,fontsize=13,labelpad=12)
            ax.set_ylabel(title+(' (dB)' if metric=='rmse_db' else ''),fontproperties=chinese,fontsize=13,labelpad=12)
            vals=data[f'{weather}_{metric}'].to_numpy();low=float(vals.min());high=float(vals.max());padding=max((high-low)*.08,.02)
            if metric=='r2':
                ax.set_ylim(min(low-padding,-.03),max(1.02,high+padding));ax.axhline(0,color='#aaaaaa',lw=.8,ls='--',zorder=1)
            else:ax.set_ylim(0,high+padding)
            need(np.all((vals>=ax.get_ylim()[0])&(vals<=ax.get_ylim()[1])),'No clipped observations')
            sns.despine(ax=ax,left=True,bottom=True);ax.grid(False);ax.grid(axis='y',color='#e4e4e4',lw=.65)
            ax.tick_params(axis='both',which='both',length=0,labelcolor='dimgrey',labelsize=11,pad=7)
            legend=fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.54,.835),ncol=5,fontsize=10.5,
                handlelength=3.5,columnspacing=1.6,labelspacing=.9,frameon=True,facecolor='white',framealpha=.8,edgecolor='lightgrey',labelcolor='dimgrey')
            legend.get_texts()[0].set_weight('bold');legend.get_texts()[0].set_color('#0072B2')
            fig.text(.10,.966,'英国数据集 · 冻结模型测试：'+title,fontproperties=chinese,fontsize=19,color='dimgrey',va='top')
            fig.text(.10,.912,f'Chilbolton + Sparsholt  ·  n = {count} 小时  ·  10个模型 × 10个参数状态',fontproperties=chinese,fontsize=11.5,color='dimgrey',va='top')
            neg=int((vals<0).sum())
            insight=f'当前面板 {neg}/100 个状态的R²为负；全部保留。' if metric=='r2' else 'RMSE越低越好；全部模型在相同小时集合评分。'
            fig.text(.10,.06,insight,fontproperties=chinese,fontsize=10,color='dimgrey')
            fig.text(.10,.028,'不在英国数据上重训或重选参数；各模型遵循原RFE路径。TabM晕光仅作视觉强调，不表示误差范围。',fontproperties=chinese,fontsize=9.2,color='dimgrey')
            fig.canvas.draw();bounds=legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            need(bounds.y0>ax.get_position().y1 and bounds.x0>=0 and bounds.x1<=1,'Legend layout')
            # --- Save ---
            for extension in ['png','pdf']:fig.savefig(folder/f'{fid}.{extension}',dpi=150,bbox_inches='tight',pad_inches=.12)
            records.append(dict(id=fid,metric=metric,weather=weather,scoring_hours=count,curves=curves,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
            plt.close(fig)
        emit(folder/'FIGURE_QA.json',dict(status='ARTIST_AND_EXPORT_QA_PASS_PENDING_VISUAL_REVIEW',utc=now(),figures=6,
            curves_per_figure=10,points_per_curve=10,all_600_artist_values_exact=True,clipping=False,smoothing=False,
            source_sha256=sha(source),figures_detail=records,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
            script_sha256=sha(Path(__file__)),style='Consistent colours/markers/linetypes; TabM thick blue with cyan halo'))
        print('SIX_UK_FIGURES_SAVED_PENDING_VISUAL_REVIEW',flush=True)
if __name__=='__main__':main()
