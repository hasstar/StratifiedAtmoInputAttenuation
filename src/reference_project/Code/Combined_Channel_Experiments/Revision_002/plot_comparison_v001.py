# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Render real, QA-passed common-input scores; no synthetic figure data."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    """Plot seed-level LOCAL and GBS39 RMSE without pooling the two domains.

    Parameters
    ----------
    None. Reads the immutable metrics and model registry of this revision.

    Saves
    -----
    PDF, PNG, exact plot values and figure manifest in the new figures directory.
    """
    root = Path(__file__).resolve().parents[3]
    out = root/'Outputs/Experiments/Combined_Channel_Revision_002'
    assert json.loads((out/'independent_qa.json').read_text())['status']=='PASS'
    dest = out/'figures'
    assert not dest.exists(), 'Never replace figure artifacts'
    dest.mkdir()
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style='whitegrid', font='DejaVu Sans')
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    # --- Data ---
    scores = pd.read_csv(out/'metrics.csv').dropna(subset=['model','cohort','group','rmse_db'])
    models = json.loads((out/'frozen_selection.json').read_text())['models']
    mapping = {'ours_frozen_top10':'ours_frozen_top10',**{r['id']:r['family'] for r in models}}
    scores['family'] = scores.model.map(mapping)
    scores = scores.dropna(subset=['family'])
    order = ['ours_frozen_top10','bai2022_mlp_adapted','bai2022_lstm_t0_adapted',
             'tabm2025_official_budgeted','svr_rbf','random_forest','xgboost','ridge','elastic_net','train_mean']
    labels = ['Proposed Top10 (frozen, 1 run)','Bai2022 MLP adapted (3 seeds)',
              'Bai2022 LSTM-T0 adapted (3 seeds)','TabM2025 official / bounded (3 seeds)',
              'RBF-SVR (deterministic)','Random forest (3 seeds)','XGBoost (3 seeds)',
              'Ridge (deterministic)','ElasticNet (deterministic)','Train-mean control']
    selected = scores[((scores.cohort=='LOCAL_test') & (scores['group']=='overall')) |
                      ((scores.cohort=='GBS39_pooled') & (scores['group']=='eligible_all'))].copy()
    assert selected.rmse_db.gt(0).all()
    selected.to_csv(dest/'plot_values.csv',index=False)
    # --- Plot ---
    fig, axes = plt.subplots(1,2,figsize=(16,9),dpi=150,sharey=True)
    titles = ['(a) LOCAL test: 687 hours','(b) GBS39 transfer: 1,847 station-hours']
    observations = []
    for ax,cohort,title in zip(axes,['LOCAL_test','GBS39_pooled'],titles):
        data = selected[selected.cohort==cohort]
        stats = data.groupby('family').rmse_db.agg(['min','max','mean']).reindex(order)
        y = np.arange(len(order))
        ax.hlines(y,stats['min'],stats['max'],color='grey',alpha=.5,lw=3,zorder=1)
        ax.scatter(stats['min'],y,s=48,color=pal[5],label='Seed min/max',zorder=3)
        ax.scatter(stats['max'],y,s=48,color=pal[5],zorder=3)
        ax.scatter(stats['mean'],y,s=60,marker='D',color=pal[2],edgecolors='white',label='Mean / single run',zorder=4)
        for i,family in enumerate(order):
            values=data.loc[data.family==family,'rmse_db'].to_numpy()
            ax.scatter(values,np.full(len(values),i),s=22,facecolors='none',edgecolors=pal[5],zorder=5)
            s=stats.loc[family]
            ax.annotate(f"{s['mean']:.3f}",(s['mean'],i),xytext=(0,11),textcoords='offset points',
                        ha='center',fontsize=9,color='dimgrey',weight='medium')
            if len(values)>1:
                ax.annotate(f"[{s['min']:.3f}, {s['max']:.3f}]",(s['mean'],i),xytext=(0,-15),
                            textcoords='offset points',ha='center',fontsize=8,color='dimgrey')
        # Distinct positive domains can differ by orders of magnitude; explicit log axes.
        ax.set_xscale('log')
        low,high=data.rmse_db.min(),data.rmse_db.max()
        ax.set_xlim(low/1.45,high*1.5)
        ax.set_yticks(y,labels)
        ax.set_ylim(len(order)-.45,-.65)
        ax.set_xlabel('RMSE (dB; logarithmic axis; lower is better)',fontsize=11,color='dimgrey')
        ax.set_title(title,loc='left',fontsize=13,color='dimgrey',pad=12)
        ax.axvline(stats.loc['ours_frozen_top10','mean'],color='grey',ls=':',lw=1,zorder=0)
        ax.grid(False)
        ax.tick_params(axis='both',which='both',length=0,labelcolor='dimgrey')
        best=stats['mean'].idxmin()
        observations.append(f"{cohort}: lowest family-mean RMSE = {labels[order.index(best)].split(' (')[0]}")
    axes[0].legend(frameon=True,facecolor='white',framealpha=.8,edgecolor='lightgrey',
                   labelcolor='dimgrey',loc='upper left',bbox_to_anchor=(0,-.10),ncol=2,fontsize=9)
    sns.despine(left=True,bottom=True)
    for ax in axes:
        ax.patch.set_edgecolor('lightgrey')
        ax.patch.set_linewidth(.8)
    fig.suptitle('Same ten weather inputs, two separate evaluation domains',x=.32,y=.98,
                 ha='left',fontsize=15,color='dimgrey')
    fig.text(.32,.935,'Points show every seed; diamonds show the mean. Brackets are seed ranges, not confidence intervals.',
             fontsize=10,color='dimgrey')
    fig.subplots_adjust(left=.32,right=.98,top=.88,bottom=.17,wspace=.18)
    fig.text(.32,.055,'\n'.join(observations),fontsize=9,color='dimgrey')
    fig.text(.32,.015,'Historical LOCAL random-hour split; approximate zero-shot GBS transfer. Rain-only ITU proxies are reported separately.',
             fontsize=9,color='dimgrey')
    # --- Save ---
    for ext in ['pdf','png']:
        fig.savefig(dest/f'common_top10_comparison.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    manifest={'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'metric_source_sha256':hashlib.sha256((out/'metrics.csv').read_bytes()).hexdigest(),
              'exact_plot_values_sha256':hashlib.sha256((dest/'plot_values.csv').read_bytes()).hexdigest(),
              'rows':len(selected),'truthful_scales':'logarithmic; independently labeled domains',
              'intervals':'seed range, not confidence interval','visual_qa':'pending manual image inspection'}
    with (dest/'manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2)
    print('FIGURES_RENDERED',dest)


if __name__=='__main__':
    main()
