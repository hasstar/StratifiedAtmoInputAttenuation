# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Visualize verified frozen results, with all fixed seeds visible."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import seaborn as sns
import numpy as np
import pandas as pd

def main():
    """Plot two complementary comparison figures.

    Parameters
    ----------
    None; reads the verified Revision003 CSV outputs.

    Saves
    -----
    figures/local_and_expanded_uk.pdf/.png and original_vs_new_uk.pdf/.png,
    with their exact plotted source data and descriptive legend notes.
    """
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style='whitegrid', font='DejaVu Sans')
    root=Path(__file__).resolve().parents[3]
    out=root/'Outputs/Experiments/Combined_Channel_Revision_003'
    figures=out/'figures_v002';figures.mkdir(exist_ok=False)
    assert json.loads((out/'independent_qa.json').read_text(encoding='utf-8'))['status']=='PASS'
    # --- Data ---
    metrics=pd.read_csv(out/'metrics.csv').dropna(subset=['rmse_db','family','model','cohort'])
    summary=pd.read_csv(out/'family_summary.csv').dropna(subset=['rmse_mean_db','rmse_seed_sd_db'])
    metrics=metrics[(metrics.group=='overall')&metrics.comparison.str.startswith('common_top10')]
    summary=summary[(summary.group=='overall')&summary.comparison.str.startswith('common_top10')]
    order=['ours_frozen_top10','train_mean','ridge','elastic_net','svr_rbf','random_forest','xgboost',
           'bai2022_mlp_adapted','bai2022_lstm_t0_adapted','tabm2025_official_budgeted']
    labels=['Proposed Top10','Train mean','Ridge','ElasticNet','RBF-SVR','Random forest','XGBoost',
            'Bai2022 MLP*','Bai2022 LSTM t=0*','TabM2025*']
    groups=[('local_and_expanded_uk',[('LOCAL_test','LOCAL test'),('UK_all','UK expanded: both stations'),
                                    ('UK_chilbolton','UK Chilbolton'),('UK_sparsholt','UK Sparsholt')],2),
            ('original_vs_new_uk',[('UK_original39','UK original 39 dates / station'),
                                   ('UK_extension2003','UK newly added 2003 dates')],1)]
    # --- Plot ---
    for filename,panels,nrows in groups:
        fig=plt.figure(figsize=(13,9.2 if nrows==2 else 5.6),dpi=150)
        gs=gridspec.GridSpec(nrows,2,figure=fig,left=.20,right=.98,top=.87 if nrows==2 else .80,
                            bottom=.15 if nrows==2 else .24,wspace=.10,hspace=.32)
        axes=[]
        for i,(cohort,title) in enumerate(panels):
            ax=fig.add_subplot(gs[i//2,i%2]);axes.append(ax)
            raw=metrics[metrics.cohort.eq(cohort)];stats=summary[summary.cohort.eq(cohort)].set_index('family')
            assert len(stats)==10 and len(raw)==20
            for j,family in enumerate(order):
                values=raw.loc[raw.family.eq(family),'rmse_db'].to_numpy()
                mu=stats.loc[family,'rmse_mean_db'];sd=stats.loc[family,'rmse_seed_sd_db']
                color='#bd0c0c' if family=='ours_frozen_top10' else '#324465'
                marker='D' if family=='ours_frozen_top10' else 's' if family=='train_mean' else 'o'
                offsets=np.linspace(-.13,.13,len(values)) if len(values)>1 else np.array([0.])
                ax.scatter(values,j+offsets,color='silver',s=27,marker='o',edgecolors='dimgrey',linewidths=.4,zorder=4)
                ax.errorbar(mu,j,xerr=sd if len(values)>1 else None,fmt=marker,color=color,
                            markersize=5.5,capsize=3,elinewidth=1.5,zorder=5)
                value=f'{mu:.3f}' if len(values)==1 else f'{mu:.3f} ± {sd:.3f}'
                ax.text(.985,j,value,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color='dimgrey')
            baseline=stats.loc['ours_frozen_top10','rmse_mean_db']
            ax.axvline(baseline,color='#bd0c0c',linestyle=':',linewidth=.9,alpha=.6,zorder=0)
            ax.set_xscale('log');ax.set_xlim(.20,10.0);ax.set_ylim(9.65,-.65)
            ax.set_yticks(range(10),labels if i%2==0 else ['']*10)
            ax.set_xticks([.25,.5,1,2,4]);ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.tick_params(axis='both',which='both',length=0,labelcolor='dimgrey',labelsize=9)
            ax.set_title(r'$\bf{('+chr(97+i)+r')}$ '+title+f"\n{int(stats.n.iloc[0]):,} scored hours",loc='left',fontsize=11,pad=8)
            ax.grid(False)
            if i//2==nrows-1:ax.set_xlabel('RMSE (dB; logarithmic scale; lower is better)',fontsize=9,color='dimgrey')
            if i%2==0:ax.set_ylabel('Frozen model / comparator',fontsize=10,color='dimgrey')
        sns.despine(left=True,bottom=True)
        for ax in axes:ax.patch.set_edgecolor('lightgrey');ax.patch.set_linewidth(.8)
        handles=[plt.Line2D([],[],color='#bd0c0c',marker='D',linestyle=':',label='Proposed frozen Top10'),
                 plt.Line2D([],[],color='#324465',marker='o',linestyle='-',label='Mean ± seed SD (not a CI)'),
                 plt.Line2D([],[],color='silver',marker='o',linestyle='',markeredgecolor='dimgrey',label='Individual fixed runs')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.96 if nrows==2 else .94),ncol=3,fontsize=8,
                   frameon=True,facecolor='white',framealpha=.8,edgecolor='lightgrey',labelcolor='dimgrey')
        fig.suptitle('Frozen common-Top10 model comparison',x=.59,y=.99,fontsize=15,color='dimgrey')
        if nrows==2:
            local=summary[summary.cohort.eq('LOCAL_test')].sort_values('rmse_mean_db').iloc[0]
            uk=summary[summary.cohort.eq('UK_all')].sort_values('rmse_mean_db').iloc[0]
            insight=f"Lowest descriptive family RMSE: LOCAL TabM {local.rmse_mean_db:.3f}; expanded UK train mean {uk.rmse_mean_db:.3f} dB."
        else:
            points=[summary[(summary.cohort==c)&(summary.family=='ours_frozen_top10')].rmse_mean_db.iloc[0] for c,t in panels]
            insight=f'Proposed Top10 RMSE: original dates {points[0]:.3f} → added dates {points[1]:.3f} dB; parameters unchanged.'
        fig.text(.20,.065 if nrows==2 else .10,insight,color='dimgrey',fontsize=9)
        fig.text(.20,.025 if nrows==2 else .04,'* Declared input/budget adaptations. Seeds 17/29/43 shown; no seed selection. UK: ≥95% native coverage.\n'
                 'Separate cohorts, not pooled LOCAL–UK errors. P.838 / P.618 physical proxies use additional inputs and are reported separately.',color='dimgrey',fontsize=8)
        # --- Save ---
        for suffix in ('pdf','png'):fig.savefig(figures/f'{filename}.{suffix}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    summary.to_csv(figures/'plotted_summary.csv',index=False)
    metrics.to_csv(figures/'plotted_runs.csv',index=False)

if __name__=='__main__':main()
