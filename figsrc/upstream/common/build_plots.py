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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=ROOT/'最终提交的修改材料/论文图表_20261030_001');a=ap.parse_args()
 out=a.package;data=out/'04_Source_Data';src=data/'frozen_exports';dest=out/'01_Figures_PDF';preview=out/'05_QA/previews'
 dest.mkdir(exist_ok=True);preview.mkdir(parents=True,exist_ok=True)
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
 # --- Figure 1: exact embedded photographs, orientation corrected at rendering ---
 photos=data/'original_manuscript'
 # Embedded PDF images use bottom-up scanlines; undo this vertical inversion.
 # The orientation is verified against the original rendered PDF page 3.
 outdoor=np.flipud(np.asarray(Image.open(photos/'page_03_image_00_Image5.jpg')))
 indoor=np.flipud(np.asarray(Image.open(photos/'page_03_image_01_Image6.jpg')))
 fig=plt.figure(figsize=(7.1,4.6));gs=fig.add_gridspec(2,2,width_ratios=[.88,1.3],height_ratios=[1,1])
 axs=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1])]
 for ax,im,title in zip(axs,[outdoor,indoor],['(a) Outdoor receiving installation','(b) Indoor acquisition hardware']):
  ax.imshow(im);ax.axis('off');ax.set_title(title,loc='left',pad=7)
 info=fig.add_subplot(gs[1,1]);info.axis('off')
 info.text(.02,.85,'Measurement and analysis chain',fontsize=9,weight='bold')
 info.text(.02,.62,'Ka-band beacon: 19.13 GHz\nReceiving antenna and acquisition hardware',fontsize=8.5,linespacing=1.6)
 info.text(.02,.3,'Power records: 100 ms\nSecond-scale analysis: 1 s\nMeteorological alignment: 1 h',fontsize=8.5,linespacing=1.6)
 fig.subplots_adjust(wspace=.13,hspace=.15,top=.9,bottom=.02,left=.015,right=.99)
 save(fig,'Fig01_Measurement_Campaign',['original_manuscript/page_03_image_00_Image5.jpg','original_manuscript/page_03_image_01_Image6.jpg'],'Lossless pixel orientation at rendering only; original embedded images from manuscript page 3. No generated or retouched scene.')
 # --- Figure 3: all actual hours, fixed random hour and native seconds ---
 local=pd.read_csv(data/'Fig03_all_local_hours.csv');local['time']=pd.to_datetime(local.hour_bjt).dt.tz_localize(None)
 sec=pd.read_csv(data/'Fig03_selected_seconds.csv');selection=json.loads((data/'Fig03_selection.json').read_text());chosen=pd.Timestamp(selection['selected_hour']).tz_localize(None)
 fig,axs=plt.subplots(3,1,figsize=(7.15,6.6),gridspec_kw={'height_ratios':[1,1,1.05]});fig.subplots_adjust(hspace=.53,top=.91,bottom=.08,left=.11,right=.98)
 grid=local.set_index('time').power_dbm.reindex(pd.date_range(local.time.min(),local.time.max(),freq='h'))
 axs[0].plot(grid.index,grid.values,color='#263D51',lw=.55)
 for r in local.itertuples():axs[0].axvspan(r.time,r.time+pd.Timedelta(hours=1),ymin=.95,ymax=1,facecolor=WCOL[r.event],lw=0)
 axs[0].axvline(chosen,color='#D55E00',ls='--',lw=.9)
 axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2));axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
 axs[0].set_title('(a) Quality-screened local hourly power',loc='left');axs[0].set_xlabel('Calendar time (Beijing time)');axs[0].set_ylabel('Power (dBm)')
 context=local[local.time.between(chosen-pd.Timedelta(hours=24),chosen+pd.Timedelta(hours=24))].copy();context['relative_h']=(context.time-chosen).dt.total_seconds()/3600
 ix=pd.Index(range(-24,25));line=context.set_index('relative_h').power_dbm.reindex(ix)
 for r in context.itertuples():axs[1].axvspan(r.relative_h,r.relative_h+1,color=WCOL[r.event],alpha=.15,lw=0)
 axs[1].plot(ix,line.values,'o-',ms=2.4,color='#263D51',lw=.95);axs[1].axvspan(0,1,facecolor='none',edgecolor='#D55E00',lw=1.4)
 axs[1].set(xlim=(-24,25),xlabel='Hours relative to the selected hour',ylabel='Power (dBm)');axs[1].set_title('(b) Hourly context; unobserved intervals remain gaps',loc='left')
 axs[2].plot(sec.second_in_hour,sec.power_dbm,lw=.55,color='#0072B2');m=local[local.time.eq(chosen)].power_dbm.iloc[0];axs[2].axhline(m,color='#D55E00',ls='--',lw=1,label='Hourly power mean (linear-domain)')
 axs[2].set(xlim=(0,3600),xticks=[0,600,1200,1800,2400,3000,3600],xlabel='Seconds within the selected hour',ylabel='Power (dBm)')
 axs[2].set_title(f'(c) {chosen:%Y-%m-%d %H}:00 BJT; fixed random seed 20261030',loc='left');axs[2].legend(loc='lower right',**legend)
 fig.legend(handles=[Patch(facecolor=WCOL[e],label=WLAB[e]) for e in WEATHER],loc='upper center',ncol=4,**legend)
 save(fig,'Fig03_Hourly_And_Second_Power',['Fig03_all_local_hours.csv','Fig03_selected_seconds.csv','Fig03_selection.json'],'All 3434 detected hours; gaps are not connected. Seeded selection among complete 3600-second LOCAL hours. Colored bands are hour-level weather labels.')
 # --- Figure 4: no received-power/attenuation mixing ---
 itu=pd.read_csv(src/'itu_predictions.csv');obs=itu.truth_db.to_numpy();pred=itu.prediction_db.to_numpy();rmse=np.sqrt(np.mean((pred-obs)**2));r2=1-np.sum((pred-obs)**2)/np.sum((obs-obs.mean())**2)
 assert len(itu)==684 and abs(rmse-1.008963)<1e-6 and abs(r2-.028100)<1e-6
 fig,axs=plt.subplots(1,2,figsize=(7.15,3.25));fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.87,wspace=.34)
 for e in WEATHER:
  g=itu[itu.weather.eq(e)];axs[0].scatter(g.truth_db,g.prediction_db,s=10,color=WCOL[e],alpha=.55,edgecolors='none',label=WLAB[e])
 lim=[min(obs.min(),pred.min())-.2,max(obs.max(),pred.max())+.3];axs[0].plot(lim,lim,'--',color='#444444',lw=.9);axs[0].set(xlim=lim,ylim=lim,xlabel='Measured hourly attenuation (dB)',ylabel='ITU estimate (dB)');axs[0].set_aspect('equal',adjustable='box');axs[0].set_title('(a) Common valid test hours',loc='left')
 axs[0].text(.04,.96,f'n = 684\nRMSE = {rmse:.4f} dB\n$R^2$ = {r2:.4f}',transform=axs[0].transAxes,va='top',fontsize=8,color='dimgrey',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
 for e in WEATHER:
  g=itu[itu.weather.eq(e)];axs[1].scatter(g.truth_db,g.prediction_db-g.truth_db,s=10,color=WCOL[e],alpha=.55,edgecolors='none')
 axs[1].axhline(0,color='#444444',ls='--',lw=.9);axs[1].set(xlabel='Measured hourly attenuation (dB)',ylabel='Estimate minus measurement (dB)');axs[1].set_title('(b) Signed estimation error',loc='left')
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=4,**legend)
 save(fig,'Fig04_ITU_Hourly_Attenuation',['frozen_exports/itu_predictions.csv'],'Fixed-reference gas/cloud/rain hourly ITU approximation; common 684 hours; R2 = 1-SSE/SST, not Pearson r squared.')
 # --- Figure 6: full ten-model paths plus compact region ---
 metrics=pd.read_csv(src/'local_metrics.csv');ml=metrics[metrics.population.eq('ML_687')]
 overall=ml[ml.weather.eq('overall')]
 fig,axs=plt.subplots(2,2,figsize=(7.15,5.35));fig.subplots_adjust(top=.83,bottom=.10,left=.09,right=.98,hspace=.46,wspace=.3)
 for i,(metric,zoom) in enumerate([('r2',False),('rmse_db',False),('r2',True),('rmse_db',True)]):
  ax=axs.flat[i];modelplot(ax,overall,metric,zoom);ax.set_title(f'({chr(97+i)}) '+('Compact inputs: k = 1-10' if zoom else 'Complete paths: k = 1-156'),loc='left')
 modellegend(fig,.99)
 save(fig,'Fig06_Ten_Model_RFE',['frozen_exports/local_metrics.csv'],'All ten independent RFE paths on the same 687 test hours. Top panels show all states; bottom panels are declared compact-input zooms. Deletion proceeds right to left.')
 # --- Figure 7: equal dimensions, independent candidate pools ---
 comparison=pd.read_csv(src/'tabm_same_k.csv');comp=comparison[(comparison.split=='test')&(comparison.weather=='overall')].sort_values('k')
 fig,axs=plt.subplots(2,2,figsize=(7.15,5));fig.subplots_adjust(top=.87,bottom=.10,left=.09,right=.98,hspace=.45,wspace=.3)
 for i,(metric,zoom) in enumerate([('r2',False),('rmse_db',False),('r2',True),('rmse_db',True)]):
  ax=axs.flat[i];g=comp[comp.k.le(10)] if zoom else comp
  ax.plot(g.k,g['all_'+metric],'-',color='#008C4A',lw=1.8,label='156-feature candidate pool')
  ax.plot(g.k,g['surface_'+metric],'--',color='#0072B2',lw=1.4,label='50-feature base candidate pool')
  ax.set(xlim=(1,10 if zoom else 50),xlabel='Retained features, k',ylabel='RMSE (dB)' if metric=='rmse_db' else r'$R^2$');ax.set_title(f'({chr(97+i)}) '+('k = 1-10' if zoom else 'k = 1-50'),loc='left')
  ax.set_xticks([1,2,4,6,8,10] if zoom else [1,10,20,30,40,50])
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,**legend)
 save(fig,'Fig07_TabM_Independent_RFE_Paths',['frozen_exports/tabm_same_k.csv'],'Independent 156-to-1 and base50-to-1 paths compared at equal k, on identical 687 test hours; base50 includes column-integrated variables.')
 # --- Figure 8: UK site-specific frozen states ---
 uk=pd.read_csv(src/'uk_metrics.csv');fig,axs=plt.subplots(3,2,figsize=(7.15,7.1));fig.subplots_adjust(top=.88,bottom=.07,left=.1,right=.98,hspace=.42,wspace=.32)
 for row,(station,name) in enumerate([('ALL','UK pooled'),('chilbolton','Chilbolton'),('sparsholt','Sparsholt')]):
  d=uk[(uk.station==station)&(uk.weather=='overall')]
  for col,metric in enumerate(['r2','rmse_db']):
   ax=axs[row,col];modelplot(ax,d,metric,True);ax.set_title(f'({chr(97+row*2+col)}) {name}; n = {int(d.n.iloc[0])}',loc='left')
   if metric=='r2':
    ax.set_yscale('symlog',linthresh=1);ax.set_ylim(-520,.2);ax.set_yticks([-500,-100,-20,-5,-1,0],labels=['-500','-100','-20','-5','-1','0']);ax.set_ylabel(r'$R^2$ (symmetric-log axis)')
   else:ax.set_ylim(0,4.1)
 modellegend(fig,1.0)
 save(fig,'Fig08_UK_Frozen_Model_Comparison',['frozen_exports/uk_metrics.csv'],'100 frozen states, no UK refitting. Same scale across sites for each metric. Explicit symmetric-log R2 axis retains all negative values; linear RMSE.')
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
 # --- Supplements: combined realizations, weather-resolved RFE, F99 tails ---
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
 for suffix,frame,station_filter,zoom,title in [('FigS02_Local_RFE_By_Weather',ml,None,False,'Local test'),('FigS03_UK_RFE_By_Weather',uk,'ALL',True,'UK pooled')]:
  fig,axs=plt.subplots(3,2,figsize=(7.15,7.1));fig.subplots_adjust(top=.88,bottom=.07,left=.1,right=.98,hspace=.42,wspace=.33)
  for row,e in enumerate(['rain','dry_overcast','overall']):
   g=frame[frame.weather==e]
   if station_filter:g=g[g.station==station_filter]
   for col,metric in enumerate(['r2','rmse_db']):
    ax=axs[row,col];modelplot(ax,g,metric,zoom);ax.set_title(f'({chr(97+row*2+col)}) {title}: {WLAB[e]}',loc='left')
    if metric=='r2' and g[metric].min()<-3:
     ax.set_yscale('symlog',linthresh=1);ax.set_ylim(float(g[metric].min())*1.15,.2);ax.set_ylabel(r'$R^2$ (symmetric-log axis)')
  modellegend(fig,1.0);save(fig,suffix,['frozen_exports/'+('local_metrics.csv' if station_filter is None else 'uk_metrics.csv')],'All model states retained; rain/overcast/overall panels. Any symmetric-log R2 scale is explicitly labelled; no clipping of negative values.')
 fig,axs=plt.subplots(1,3,figsize=(7.15,2.9));fig.subplots_adjust(top=.78,bottom=.2,left=.075,right=.99,wspace=.28)
 for ax,(cohort,station,name) in zip(axs,groups):
  for e in WEATHER:
   for kind,style in [('obs','-'),('gen','--')]:
    g=duration[(duration.cohort==cohort)&(duration.station==station)&(duration.event==e)&(duration.threshold=='F99')&(duration.kind==kind)].sort_values('duration_s')
    ax.step(np.r_[.8,g.duration_s,3600],np.r_[0,g.cdf,1],where='post',color=WCOL[e],ls=style,lw=1)
  ax.set(xscale='log',xlim=(1,3600),ylim=(0,1.02),xlabel='Duration (s; log scale)');ax.set_title(name,loc='left')
 axs[0].set_ylabel('Equal-hour CDF');fig.legend(handles=handles,loc='upper center',ncol=3,**legend)
 save(fig,'FigS04_F99_Fade_Duration_Distributions',['Fig10_duration_distributions.csv'],'Separate F99 tail analysis; same complete-event censoring and equal-hour weighting as Fig10.')
 (out/'05_QA/plot_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
