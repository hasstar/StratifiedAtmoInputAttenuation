# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Immutable, evidence-bound external probability plots, no data selection by scores."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

def main():
    """Plot frozen probability scores and coverage.

    Saves
    -----
    PNG, PDF and machine-readable plotting values in report_003/figures.
    """
    root=Path(__file__).resolve().parents[3]
    base=root/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
    out=base/'report_003/figures';out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0,style='whitegrid',font='DejaVu Sans')
    # --- Data ---
    qa=json.loads((base/'conditional_evaluation_independent_qa_001/qa.json').read_text())
    assert qa['status']=='PASS_WEATHER_AND_FROZEN_MARGINAL_EVALUATION'
    source=base/'conditional_evaluation_001/probability_summary.json'
    assert sha(source)==qa['input_outputs_sha256']['probability_summary.json']
    frame=pd.DataFrame(json.loads(source.read_text())['primary'])
    frame=frame[(frame.station=='pooled')&(frame.weighting=='date_equal')]
    # Only unknown is unscorable; no finite group or adverse outcome excluded.
    unavailable=frame[frame.combined_crps201_db.isna()].copy()
    frame=frame.dropna(subset=['combined_crps201_db','combined_central90_coverage']).copy()
    frame.to_csv(out/'plotted_probability_summary.csv',index=False)
    unavailable.to_csv(out/'unscorable_groups.csv',index=False)
    labels={'overall':'Overall','clear_dry':'Clear / dry','rain':'Rain','heavy_rain_subset':'Heavy subset','dry_overcast':'Dry overcast','other':'Other'}
    # --- Plot ---
    fig=plt.figure(figsize=(12,5.5),dpi=150);gs=gridspec.GridSpec(1,2,figure=fig,wspace=.12,left=.16,right=.99,top=.80,bottom=.19)
    axes=[fig.add_subplot(gs[0,i]) for i in range(2)];y=np.arange(len(frame))
    c1,c2='#1f253f','#90c1c6'
    axes[0].hlines(y,frame.combined_crps201_db,frame.deterministic_slow_instant_mae_db,color='lightgrey',lw=3)
    axes[0].plot(frame.deterministic_slow_instant_mae_db,y,'s',color='grey',label='Deterministic hourly baseline')
    axes[0].plot(frame.combined_crps201_db,y,'o',color=c1,label='Hourly + frozen fast CDF')
    axes[0].set_xlim(0,max(frame.deterministic_slow_instant_mae_db)*1.16)
    axes[0].set_xlabel('CRPS (dB; lower is better)')
    axes[0].set_yticks(y,[f"{labels[r.event]}  (n={r.n_station_hours})" for r in frame.itertuples()])
    axes[1].plot(100*frame.combined_central90_coverage,y,'o',color=c1,label='90% interval')
    axes[1].plot(100*frame.combined_central98_coverage,y,'s',color=c2,label='98% interval')
    axes[1].axvline(90,color=c1,ls='--',lw=1);axes[1].axvline(98,color=c2,ls=':',lw=1.5)
    axes[1].set_xlim(0,102);axes[1].set_xlabel('Actual coverage (%)');axes[1].set_yticks(y,[])
    for i,ax in enumerate(axes):
        ax.set_ylim(len(frame)-.5,-.5);ax.grid(False);ax.tick_params(axis='both',which='both',length=0,labelcolor='dimgrey')
        ax.set_title(r'$\bf{('+chr(97+i)+r')}$ '+['Probability error','Nominal vs actual coverage'][i],loc='left',fontsize=12)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.32),frameon=True,facecolor='white',framealpha=.8,edgecolor='lightgrey',labelcolor='dimgrey',fontsize=8)
    sns.despine(left=True,bottom=True)
    for ax in axes:ax.patch.set_edgecolor('lightgrey');ax.patch.set_linewidth(.8)
    overall=frame[frame.event=='overall'].iloc[0]
    drop=(1-overall.combined_crps201_db/overall.deterministic_slow_instant_mae_db)*100
    fig.suptitle('GBS 20.7 GHz | Frozen-model transfer, two stations / 39 dates',y=1.03,fontsize=13,color='dimgrey')
    fig.text(.16,.055,f"Overall CRPS reduction: {drop:.2f}%; nominal 90% coverage: {overall.combined_central90_coverage*100:.2f}% actual. Not calibrated.",color='dimgrey',fontsize=10)
    fig.text(.16,.005,'Date-equal means; primary n=1844 station-hours. Rain includes the project-defined heavy subset; unknown excluded only from CDF scoring.',color='dimgrey',fontsize=8)
    # --- Save ---
    for ext in ('png','pdf'):fig.savefig(out/f'gbs_probability_validation.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    record={'source_sha256':sha(source),'code_sha256':sha(__file__),'qa_sha256':sha(base/'conditional_evaluation_independent_qa_001/qa.json'),
        'analysis_set':'1844 primary known-weather station-hours; date equal','all_finite_weather_groups_shown':True,'unscorable_groups':['unknown'],
        'same_source_same_frequency':True,'new_fit_calls':0,'alt_text':'Six weather categories compare frozen-combination CRPS against a degenerate hourly baseline. The right panel shows every 90% interval coverage below nominal; the 98% coverage is also below nominal. Heavy rain is a subset, not a seventh independent group.',
        'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    with (out/'figure_provenance.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps({'status':'GENERATED_PENDING_VISUAL_REVIEW','overall_crps_reduction_percent':drop}))
if __name__=='__main__':main()
