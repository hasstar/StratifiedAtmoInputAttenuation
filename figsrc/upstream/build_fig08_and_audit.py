"""Fig08 RMSE-only revision and diagnostic audit of saved frozen predictions.

No estimator is loaded, fitted, selected, or run. Standard deviations use ddof=0.
The plotted metric values come from the existing frozen metric export and are
independently checked against the saved station-hour predictions here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

MODELS = ['tabm', 'bilstm', 'mlp', 'gcn', 'svr_rbf', 'bai_lstm',
          'transformer', 'catboost', 'random_forest', 'elastic_net']
LABELS = ['TabM', 'BiLSTM', 'MLP', 'GCN', 'SVR-RBF', 'Bai LSTM',
          'Transformer', 'CatBoost', 'Random Forest', 'Elastic Net']
COLORS = ['#008C4A', '#0072B2', '#D55E00', '#CC79A7', '#6C4A85',
          '#A87800', '#009EAC', '#6F6F6F', '#9C4A35', '#647B36']
DASHES = ['-', '--', '-.', ':', (0, (4, 1, 1, 1)), (0, (6, 2)),
          (0, (2, 1)), (0, (8, 1, 1, 1)), (0, (4, 3)), (0, (1, 3))]
STATIONS = [('ALL', 'Pooled UK sites', 5593), ('chilbolton', 'Chilbolton', 3382),
            ('sparsholt', 'Sparsholt', 2211)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metrics', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layout', type=Path)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ['Fig08_UK_Frozen_Model_Comparison.pdf', 'Fig08_UK_Frozen_Model_Comparison.png',
             'uk_prediction_diagnostics.csv', 'uk_k10_diagnostics.csv',
             'diagnostic_findings.json', 'render_manifest.json']
    if any((out / f).exists() for f in names):
        raise FileExistsError('Use a new output directory; existing versions are preserved.')
    metrics = pd.read_csv(a.metrics)
    pred = pd.read_parquet(a.predictions)
    assert len(pred) == 566400
    assert set(pred.model) == set(MODELS)
    eligible = pred.loc[pred.primary_score_eligible.astype(bool)].copy()
    assert len(eligible) == 559300
    assert np.isfinite(eligible[['truth_db', 'prediction_db']].to_numpy()).all()
    reference = eligible.loc[(eligible.model == 'tabm') & (eligible.k == 10)]
    keys = ['station', 'utc_epoch_seconds']
    reference = reference.set_index(keys).sort_index()
    assert len(reference) == 5593 and not reference.index.has_duplicates
    rows = []
    errors = []
    for model in MODELS:
        for k in range(1, 11):
            cohort = eligible.loc[(eligible.model == model) & (eligible.k == k)].set_index(keys).sort_index()
            assert cohort.index.equals(reference.index)
            assert np.array_equal(cohort.truth_db.to_numpy(), reference.truth_db.to_numpy())
            for station, _, expected_n in STATIONS:
                group = cohort if station == 'ALL' else cohort.xs(station, level='station')
                y = group.truth_db.to_numpy(dtype=float)
                yp = group.prediction_db.to_numpy(dtype=float)
                assert len(y) == expected_n
                residual = yp - y
                mse = np.mean(residual ** 2)
                rmse = float(np.sqrt(mse))
                bias = float(np.mean(residual))
                sy, sp = float(np.std(y)), float(np.std(yp))
                covariance = float(np.mean((y - y.mean()) * (yp - yp.mean())))
                corr = covariance / (sy * sp) if sy > 0 and sp > 0 else None
                unique, counts = np.unique(yp, return_counts=True)
                record = {'model': model, 'k': k, 'station': station, 'n': len(y),
                          'rmse_db': rmse, 'bias_db': bias, 'absolute_bias_db': abs(bias),
                          'truth_mean_db': float(np.mean(y)), 'truth_sd_db': sy,
                          'prediction_mean_db': float(np.mean(yp)), 'prediction_sd_db': sp,
                          'prediction_min_db': float(np.min(yp)), 'prediction_max_db': float(np.max(yp)),
                          'prediction_range_db': float(np.ptp(yp)), 'unique_predictions': len(unique),
                          'modal_fraction': float(counts.max() / len(yp)),
                          'prediction_observation_correlation': corr,
                          'sd_ratio_prediction_to_observation': sp / sy,
                          'exactly_constant': len(unique) == 1,
                          'r2_archival_check_only': float(1 - mse / np.var(y))}
                existing = metrics.loc[(metrics.model == model) & (metrics.k == k) &
                                       (metrics.station == station) & (metrics.weather == 'overall')]
                assert len(existing) == 1 and int(existing.n.iloc[0]) == expected_n
                for field in ['rmse_db', 'bias_db']:
                    delta = abs(record[field] - float(existing[field].iloc[0]))
                    errors.append(delta)
                    assert delta < 1e-10, (model, k, station, field, delta)
                assert abs(record['r2_archival_check_only'] - float(existing.r2.iloc[0])) < 1e-9
                rows.append(record)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(out / names[2], index=False, float_format='%.17g')
    k10 = diagnostics.loc[diagnostics.k == 10].copy()
    k10['rmse_rank_within_station'] = k10.groupby('station').rmse_db.rank(method='min').astype(int)
    k10['absolute_bias_rank_within_station'] = k10.groupby('station').absolute_bias_db.rank(method='min').astype(int)
    k10.to_csv(out / names[3], index=False, float_format='%.17g')
    pooled = k10.loc[k10.station == 'ALL'].set_index('model')
    sparsholt = k10.loc[k10.station == 'sparsholt'].set_index('model')
    assert pooled.loc['tabm', 'rmse_rank_within_station'] == 3
    assert pooled.loc['tabm', 'absolute_bias_rank_within_station'] == 1
    assert sparsholt.loc['tabm', 'rmse_rank_within_station'] == 1
    lowk = diagnostics.loc[(diagnostics.model == 'elastic_net') &
                           diagnostics.k.isin([1, 2]) & (diagnostics.station == 'ALL')]
    assert (lowk.prediction_sd_db < 3e-5).all() and (lowk.modal_fraction > .9985).all()
    assert (pooled.loc[['svr_rbf', 'elastic_net'], 'prediction_sd_db'] > .2).all()
    assert (pooled.loc[['svr_rbf', 'elastic_net'], 'unique_predictions'] == 5593).all()
    findings = {'valid_states': 100, 'station_scope_metric_checks': len(rows),
                'all_states_use_identical_eligible_hours_and_truth': True,
                'maximum_absolute_rmse_or_bias_difference': max(errors),
                'exact_constant_pooled_states': diagnostics.loc[(diagnostics.station == 'ALL') &
                                                                 diagnostics.exactly_constant].to_dict('records'),
                'near_constant_elastic_net_states': lowk.to_dict('records'),
                'k10_pooled': pooled.reset_index().to_dict('records'),
                'k10_sparsholt': sparsholt.reset_index().to_dict('records'),
                'training_calls': 0, 'preprocessing_fit_calls': 0, 'new_inference_calls': 0,
                'standard_deviation_ddof': 0,
                'interpretation': 'Near-constant output is established for Elastic Net k=1,2. Neither SVR-RBF nor Elastic Net is constant at k=10. RMSE-only display does not alter archived R2 values.'}
    (out / names[4]).write_text(json.dumps(findings, indent=2) + '\n', encoding='utf-8')

    layout = {'figsize': [7.15, 3.05], 'left': .065, 'right': .985, 'bottom': .18,
              'top': .73, 'wspace': .16, 'font_size': 9, 'tick_size': 8,
              'legend_size': 8, 'line_width': 1.0, 'tabm_width': 2.0, 'glow_width': 3.8}
    if a.layout:
        layout.update(json.loads(a.layout.read_text(encoding='utf-8')))
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'font.size': layout['font_size'], 'axes.labelsize': layout['font_size'],
                         'xtick.labelsize': layout['tick_size'], 'ytick.labelsize': layout['tick_size'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .6})
    fig, axes = plt.subplots(1, 3, figsize=layout['figsize'], sharey=True)
    fig.subplots_adjust(**{key: layout[key] for key in ['left', 'right', 'bottom', 'top', 'wspace']})
    point_checks = []
    for index, (station, label, n) in enumerate(STATIONS):
        ax = axes[index]
        scope = metrics.loc[(metrics.station == station) & (metrics.weather == 'overall')]
        lines = {}
        for model, display, color, style in zip(MODELS, LABELS, COLORS, DASHES):
            group = scope.loc[scope.model == model].sort_values('k')
            assert group.k.tolist() == list(range(1, 11))
            assert group.n.eq(n).all() and group.rmse_db.lt(4.1).all()
            line, = ax.plot(group.k, group.rmse_db, label=display, color=color, linestyle=style,
                            linewidth=layout['tabm_width'] if model == 'tabm' else layout['line_width'],
                            marker='o' if model == 'tabm' else None, markersize=2.8,
                            zorder=5 if model == 'tabm' else 2)
            if model == 'tabm':
                line.set_path_effects([pe.Stroke(linewidth=layout['glow_width'], foreground='#B6F2C8', alpha=.9), pe.Normal()])
            assert np.array_equal(line.get_ydata(), group.rmse_db.to_numpy())
            lines[model] = line
            point_checks.append({'station': station, 'model': model, 'points': len(line.get_xdata())})
        ax.set(xlim=(1, 10), ylim=(0, 4.1), xlabel=r'Retained features, $k$')
        ax.set_xticks(range(1, 11))
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_title(f'({chr(97+index)}) {label}; n = {n}', loc='left', fontsize=9, pad=6)
        ax.grid(color='#E1E5E9', linewidth=.5)
        for spine in ax.spines.values():
            spine.set_color('#A6B0B9')
        ax.tick_params(length=2.8, width=.6, pad=3)
        if index == 0:
            ax.set_ylabel('RMSE (dB)')
            handles = [lines[m] for m in MODELS]
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.51, .995),
               ncol=5, frameon=False, fontsize=layout['legend_size'],
               handlelength=2.6, handletextpad=.5, columnspacing=1.2, labelspacing=.65)
    fig.savefig(out / names[0], metadata={'Title': 'Frozen UK evaluation: RMSE only',
                                         'Creator': 'T028 build_fig08_and_audit.py',
                                         'CreationDate': None, 'ModDate': None})
    fig.savefig(out / names[1], dpi=450)
    plt.close(fig)
    manifest = {'inputs': {str(a.metrics.resolve()): sha(a.metrics),
                            str(a.predictions.resolve()): sha(a.predictions)},
                'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                'versions': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                             'pandas': pd.__version__, 'numpy': np.__version__},
                'layout': layout, 'figure_point_checks': point_checks,
                'outputs': {name: sha(out / name) for name in names[:-1]}}
    (out / names[5]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'states': 100, 'station_metric_checks': len(rows), 'plot_curves': len(point_checks),
                      'max_metric_difference': max(errors), 'output': str(out)}))


if __name__ == '__main__':
    main()
