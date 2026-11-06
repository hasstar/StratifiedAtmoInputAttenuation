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
 examples=pd.read_csv(data/'FigS01_examples.csv');fig,axs=plt.subplots(3,1,figsize=(7.15,6.3));fig.subplots_adjust(top=.85,bottom=.08,left=.1,right=.98,hspace=.44)
 for i,(ax,r) in enumerate(zip(axs,examples.to_dict('records'))):
  p=data/'examples'/f"{r['dataset']}_{r['station']}.npz"
  with np.load(p,allow_pickle=False) as z:obs=z['observed_fast_db']+r['truth_db'];gen=z['generated_fast_db']+r['prediction_db'];valid=z['valid'].astype(bool)
  obs[~valid]=np.nan;lo,med,hi=np.quantile(gen,[.05,.5,.95],axis=0);x=np.arange(3600)
  ax.fill_between(x,lo,hi,color='#94B4CD',alpha=.55,label='Generated 5-95% quantiles');ax.plot(x,med,color='#0072B2',lw=.6,label='Generated median');ax.plot(x,obs,color='#333333',lw=.55,label='Observed');ax.axhline(r['prediction_db'],color='#D55E00',ls='--',lw=.9,label='TabM hourly estimate')
  hourlabel=pd.Timestamp(r['hour']).strftime('%Y-%m-%d %H:%M')+(' BJT' if r['dataset']=='LOCAL' else ' UTC')
  site='Local test' if r['dataset']=='LOCAL' else r['station'].title()
  ax.set(xlim=(0,3600),ylabel='Attenuation (dB)',xlabel='Seconds within hour');ax.set_title(f"({chr(97+i)}) {site} | {hourlabel}",loc='left',fontsize=8.5)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,**legend)
 save(fig,'FigS01_Combined_Second_Scale_Examples',['FigS01_examples.csv','examples/LOCAL_local.npz','examples/UK_chilbolton.npz','examples/UK_sparsholt.npz'],'Earliest rainy hour in each fixed cohort, selected by time not fit quality. Quantile band is an ensemble spread, not a calibrated confidence interval; all 3600 generated positions shown and observed gaps retained.')

if __name__=='__main__':main()
