# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Render all eligible external hourly predictions, not selected examples."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    """Plot actual prediction/truth pairs for each frozen GBS station.

    Parameters
    ----------
    None. Reads frozen evaluation_002 and independent QA outputs.

    Saves
    -----
    report_002/figures: PDF, PNG, exact plotted data and provenance JSON.
    """
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    root = Path(__file__).resolve().parents[3]
    base = root/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
    target = base/'report_002/figures'
    target.mkdir(parents=True, exist_ok=False)
    # --- Data ---
    qa = json.loads((base/'evaluation_independent_qa_001/qa.json').read_text(encoding='utf-8'))
    assert qa['status'] == 'PASS_HOURLY_EVALUATION_ONLY'
    source = base/'evaluation_002/hourly_predictions.parquet'
    frame = pd.read_parquet(source)
    frame = frame[frame.primary_score_eligible].copy()
    frame = frame.dropna(subset=['truth_db', 'prediction_db'])
    assert len(frame) == 1847
    data_path = target/'plotted_primary_hours.csv'
    frame[['station', 'hour_utc', 'truth_db', 'prediction_db', 'valid_seconds']].to_csv(data_path, index=False)
    # Read the exact displayed source back; never infer points from an image.
    data = pd.read_csv(data_path).dropna(subset=['truth_db', 'prediction_db'])
    bounds = np.concatenate([data.truth_db.to_numpy(), data.prediction_db.to_numpy(), [0.]])
    lower, upper = float(bounds.min()), float(bounds.max())
    pad = max(.15, .05*(upper-lower))
    limits = [lower-pad, upper+pad]
    # --- Plot ---
    fig = plt.figure(figsize=(10, 5), dpi=150)
    grid = gridspec.GridSpec(1, 2, left=.08, right=.99, bottom=.16, top=.86, wspace=.09)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    color = sns.cubehelix_palette(6, rot=-.25, light=.7)[5]
    for i, (ax, station) in enumerate(zip(axes, ['chilbolton', 'sparsholt'])):
        d = data[data.station == station]
        m = qa['recomputed_metrics']['primary_coverage_ge_95'][station]
        ax.scatter(d.truth_db, d.prediction_db, s=10, color=color, alpha=.45, edgecolors='none')
        ax.plot(limits, limits, '--', color='grey', linewidth=1, label='1:1 reference')
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(r'$\bf{(' + chr(97+i) + r')}$ ' + station.capitalize(), loc='left', fontsize=12)
        ax.set_xlabel('Native hourly attenuation (dB)', fontsize=10, color='dimgrey')
        if i == 0:
            ax.set_ylabel('Frozen-model prediction (dB)', fontsize=10, color='dimgrey')
        else:
            ax.tick_params(labelleft=False)
        ax.text(.97, .48, f"n = {len(d)} hours\nRMSE = {m['rmse_db']:.3f} dB\nBias = {m['bias_db']:+.3f} dB",
                transform=ax.transAxes, va='top', ha='right', fontsize=10, color='dimgrey')
        ax.legend(loc='lower right', fontsize=8, frameon=True, facecolor='white', framealpha=.8,
                  edgecolor='lightgrey', labelcolor='dimgrey')
        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=0, labelcolor='dimgrey')
    fig.suptitle('GBS 20.7 GHz: frozen Top10 BiLSTM external hourly trial', y=.98, color='dimgrey', fontsize=13)
    pooled = qa['recomputed_metrics']['primary_coverage_ge_95']['pooled_same_source_same_frequency']
    fig.text(.5, .055, f"{len(data):,} high-coverage station hours; pooled prediction-minus-truth bias = {pooled['bias_db']:+.3f} dB",
             ha='center', color='dimgrey', fontsize=10)
    fig.text(.5, .015, 'Authorized fixed-height / nearest-grid approximation; no retraining or bias correction',
             ha='center', color='dimgrey', fontsize=9)
    sns.despine(left=True, bottom=True)
    for ax in axes:
        ax.patch.set_edgecolor('lightgrey')
        ax.patch.set_linewidth(.8)
    # --- Save ---
    for ext in ('pdf', 'png'):
        fig.savefig(target/f'gbs_hourly_validation.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    paths = [source, data_path, Path(__file__), base/'evaluation_independent_qa_001/qa.json']
    provenance = {'figure_id':'F001', 'version':1, 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  'n_primary_station_hours':1847, 'station_counts':data.groupby('station').size().to_dict(),
                  'all_eligible_points_plotted':True, 'axis_limits_db':limits, 'same_axis_limits':True,
                  'aggregation':'none, each point is an eligible station-hour', 'uncertainty_intervals':'not estimated',
                  'creator':'Codex automated local rendering', 'human_review':'pending',
                  'alt_text':'Two station-specific scatter plots show frozen-model predictions against native hourly attenuation on identical axes, with 1:1 reference lines and sample/error annotations. Positive average bias is retained.',
                  'source_material':'new plot from local project data; no externally reused image',
                  'layout_improvement_considered':'Dense low-attenuation points could use a separate zoom inset; no points or outliers were removed.'}
    with (target/'figure_provenance.json').open('x',encoding='utf-8') as f:
        json.dump(provenance,f,ensure_ascii=False,indent=2)
    print(json.dumps({'figure':str(target/'gbs_hourly_validation.png'),'rows':len(data)}))


if __name__ == '__main__':
    main()
