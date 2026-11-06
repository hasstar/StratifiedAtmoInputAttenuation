# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas", "pillow"]
# ///
"""Render all data-based manuscript figures from packaged, frozen plot inputs.

No training, model inference, generator fitting, or stochastic curve generation.
Use --package to render a copied delivery bundle. PDF is the publication output;
PNG files are verification previews. Diagrams are authored by build_diagrams.mjs.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
import seaborn as sns

ROOT=Path(__file__).resolve().parents[3]
MODELS=['tabm','bilstm','mlp','gcn','svr_rbf','bai_lstm','transformer','catboost','random_forest','elastic_net']
LABELS=['TabM','BiLSTM','MLP','GCN','SVR-RBF','Bai LSTM','Transformer','CatBoost','Random Forest','Elastic Net']
COLORS=['#008C4A','#0072B2','#D55E00','#CC79A7','#6C4A85','#A87800','#009EAC','#6F6F6F','#9C4A35','#647B36']
DASH=['-','--','-.',':',(0,(4,1,1,1)),(0,(6,2)),(0,(2,1)),(0,(8,1,1,1)),(0,(4,3)),(0,(1,3))]
WEATHER=['rain','dry_overcast','clear_dry','other']
WCOL={'rain':'#0072B2','dry_overcast':'#D55E00','clear_dry':'#009E73','other':'#8A6AA6'}
WLAB={'rain':'Rain','dry_overcast':'Overcast','clear_dry':'Clear / dry','other':'Other','overall':'Overall'}


import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import package_support as support
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 data=support.path('figure_inputs')/'common';src=data/'frozen_exports';dest=args.out;preview=dest
 dest.mkdir(parents=True,exist_ok=True)
 # --- Style Setup ---
 sns.set_theme(font_scale=1.0,style='whitegrid',font='DejaVu Sans')
 plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.2,'axes.titleweight':'normal','pdf.fonttype':42,'ps.fonttype':42,'grid.color':'#E3E6E8','grid.linewidth':.45,'axes.labelcolor':'#333333','text.color':'#333333','lines.linewidth':1.15,'figure.dpi':150,'savefig.dpi':300})
 legend=dict(frameon=True,facecolor='white',framealpha=.8,edgecolor='lightgrey',labelcolor='dimgrey')
 inventory=[]
 def save(fig,name,inputs,note):
  for ax in fig.axes:
   ax.tick_params(axis='both',which='both',length=0,labelcolor='dimgrey')
   sns.despine(ax=ax,left=True,bottom=True)
  fig.savefig(dest/(name+'.pdf'),bbox_inches='tight',pad_inches=.05)
  fig.savefig(preview/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.05)
  plt.close(fig);inventory.append(dict(id=name,files=[f'01_Figures_PDF/{name}.pdf'],inputs=inputs,method=note,generator='build_plots.py'))
  print('Rendered',name,flush=True)
 def modelplot(ax,d,metric,zoom=False):
  for i,model in enumerate(MODELS):
   g=d[d.model.eq(model)].sort_values('k')
   if zoom:g=g[g.k.le(10)]
   ln,=ax.plot(g.k,g[metric],color=COLORS[i],ls=DASH[i],lw=1.9 if i==0 else 1.05,zorder=10 if i==0 else 3,alpha=1 if i==0 else .9)
   if i==0:ln.set_path_effects([pe.Stroke(linewidth=4,foreground='#C8EF66',alpha=.8),pe.Normal()])
  if metric=='r2':ax.axhline(0,color='#9C9C9C',lw=.65,ls=':')
  ax.set_ylabel('RMSE (dB)' if metric=='rmse_db' else r'$R^2$')
  ax.set_xlabel('Retained features, k');ax.set_xlim(1,10 if zoom else 156)
  ax.set_xticks([1,2,4,6,8,10] if zoom else [1,25,50,75,100,125,156])
 def modellegend(fig,y=1.01):
  handles=[Line2D([0],[0],color=c,lw=2 if i==0 else 1.2,ls=DASH[i],label=LABELS[i]) for i,c in enumerate(COLORS)]
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,y),ncol=5,**legend,columnspacing=1.3,handlelength=2.5)
 # --- Figure 10: ACF and paired-hour duration ECDFs ---
 acf=pd.read_csv(data/'Fig10_dense_acf_1_to_1800.csv');duration=pd.read_csv(data/'Fig10_duration_distributions.csv')
 groups=[('LOCAL_test','local','Local test'),('UK','chilbolton','Chilbolton'),('UK','sparsholt','Sparsholt')]
 fig,axs=plt.subplots(3,2,figsize=(7.15,7));fig.subplots_adjust(top=.87,bottom=.07,left=.09,right=.98,hspace=.43,wspace=.27)
 for row,(cohort,station,name) in enumerate(groups):
  for e in WEATHER:
   g=acf[(acf.cohort==cohort)&(acf.station==station)&(acf.event==e)].sort_values('lag_seconds')
   axs[row,0].semilogx(g.lag_seconds,g.observed_acf,color=WCOL[e],ls='-',lw=1.1)
   axs[row,0].semilogx(g.lag_seconds,g.generated_acf_mean,color=WCOL[e],ls='--',lw=1.1)
   for kind,style in [('obs','-'),('gen','--')]:
    g=duration[(duration.cohort==cohort)&(duration.station==station)&(duration.event==e)&(duration.threshold=='F95')&(duration.kind==kind)].sort_values('duration_s')
    axs[row,1].step(np.r_[.8,g.duration_s,3600],np.r_[0,g.cdf,1],where='post',color=WCOL[e],ls=style,lw=1.15)
  axs[row,0].axhline(0,color='#999999',lw=.6);axs[row,0].set(xlim=(1,1800),ylim=(-.12,.62),xlabel='Lag (s; log scale)',ylabel='Mean ACF');axs[row,0].set_title(f'({chr(97+row*2)}) {name}: temporal correlation',loc='left')
  axs[row,1].set(xscale='log',xlim=(1,3600),ylim=(0,1.02),xlabel='Complete fade duration (s; log scale)',ylabel='Equal-hour CDF');axs[row,1].set_title(f'({chr(98+row*2)}) {name}: F95 events',loc='left')
 handles=[Line2D([0],[0],color=WCOL[e],label=WLAB[e]) for e in WEATHER]+[Line2D([0],[0],color='black',ls='-',label='Observed'),Line2D([0],[0],color='black',ls='--',label='Generated')]
 fig.legend(handles=handles,loc='upper center',ncol=3,**legend)
 save(fig,'Fig10_Fast_Temporal_Statistics',['Fig10_dense_acf_1_to_1800.csv','Fig10_duration_distributions.csv'],'ACF at all 1800 integer lags computed directly from saved curves, with no interpolation or bridging missing intervals; checked against all 11 archived lags for every realization/hour. F95 thresholds from frozen LOCAL weather CDFs. Duration CDFs equally weight paired hours and nonempty generated realizations; censored events excluded.')

if __name__=='__main__':main()
