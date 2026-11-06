"""All valid ITU test-hour estimates and ten randomly sampled rainy 1-s hours.

--prepare extracts frozen records once; later drawing only needs bundled CSVs.
No training, new physical predictions, smoothing, or replacement of old outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PROJECT = Path(r'E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction')
PACKAGE = '最终提交的修改材料/ITU与雨天观测图_20261102_001'
ITU_STEM = 'Fig04_ITU_And_Observed_All_Test_Hours'
RAIN_STEM = 'Fig03_Ten_Random_Rainy_Hours'
TZ = 'Asia/Shanghai'
SEED = 20261102
DETAIL_LIMITS = (-0.72, 3.10)
BLUE, ORANGE = '#0072B2', '#D55E00'
WEATHER = {'rain': ('Rain', '#7AAED1'), 'dry_overcast': ('Overcast', '#DEA683'),
           'clear_dry': ('Clear/dry', '#8CC4AF'), 'other': ('Other', '#B9ACC9')}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str), encoding='utf-8')


def times(values):
    return pd.to_datetime(values, utc=True).dt.tz_convert(TZ)


def csv(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')


def prepare(project, package):
    data = package / 'Data'
    data.mkdir(parents=True, exist_ok=False)
    (package / 'QA').mkdir(exist_ok=True)
    original = project / '最终提交的修改材料/ITU组合模型_20261027_001'
    sources = {
        'itu_valid_predictions': original / '01_结果/ITU_共同684小时_逐小时预测.csv',
        'itu_original_components': original / '01_结果/ITU_本地687小时_原始分量及预测.parquet',
        'itu_exclusions': original / '01_结果/ITU_排除3小时及原因.csv',
        'itu_fixed_reference': original / '01_结果/ITU_固定物理参考参数.json',
        'local_hours': project / '最终提交的修改材料/论文图表_20261030_001/04_Source_Data/Fig03_all_local_hours.csv',
        'second_coverage': project / 'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s_hourly_summary.csv',
        'gas_cloud_code': project / 'Code/Combined_Channel_Experiments/Revision_011/physical.py',
        'combination_code': project / 'Code/Combined_Channel_Experiments/Revision_011/calculate.py',
        'rain_code': project / 'Code/Hourly_Attenuation_Experiments/Revision_016/run_itu_baseline.py',
        'itu_protocol': project / 'Code/Combined_Channel_Experiments/Revision_011/PROTOCOL.md',
    }
    itu = pd.read_csv(sources['itu_valid_predictions'], float_precision='round_trip')
    itu['time'] = times(itu.hour_bjt)
    itu = itu.sort_values('time').reset_index(drop=True)
    itu.insert(0, 'ordered_test_hour', np.arange(1, len(itu) + 1))
    assert len(itu) == 684 and itu.time.is_unique
    raw = pd.read_parquet(sources['itu_original_components'])
    raw['time'] = times(raw.hour_bjt)
    raw = raw.set_index('time')
    assert len(raw) == 687 and raw.index.is_unique
    for source, target in [('gas_relative_db', 'gas_relative_db'),
                           ('cloud_db', 'cloud_db'), ('itu838_geometric_proxy', 'rain_db')]:
        itu[target] = raw.loc[itu.time, source].to_numpy()
    assert np.allclose(itu.truth_db, raw.loc[itu.time, 'truth_db'], atol=1e-12, rtol=0)
    assert np.allclose(itu.prediction_db, itu.gas_relative_db + itu.cloud_db + itu.rain_db,
                       atol=1e-12, rtol=0)
    itu['observed_above_detail_axis'] = itu.truth_db > DETAIL_LIMITS[1]
    csv(itu.drop(columns='time'), data / 'itu_all_684_hours.csv')
    raw_keep = ['hour_bjt', 'truth_db', 'itu_combined_geometric_approx', 'gas_relative_db',
                'cloud_db', 'itu838_geometric_proxy']
    raw_export = raw[raw_keep].copy()
    raw_export['included_in_684_scoring'] = raw.index.isin(itu.time)
    csv(raw_export.sort_index(), data / 'itu_687_original_records.csv')
    shutil.copy2(sources['itu_exclusions'], data / 'itu_excluded_3_hours.csv')
    shutil.copy2(sources['itu_fixed_reference'], data / 'itu_fixed_reference.json')

    hours = pd.read_csv(sources['local_hours'], float_precision='round_trip')
    hours['time'] = times(hours.hour_bjt)
    coverage = pd.read_csv(sources['second_coverage'])
    coverage['time'] = times(coverage.hour_bjt)
    eligible = hours.merge(coverage[['time', 'primary_seconds']], on='time', validate='one_to_one')
    eligible = eligible[(eligible.event == 'rain') & (eligible.primary_seconds == 3600)]
    eligible = eligible.sort_values('time').reset_index(drop=True)
    assert len(eligible) == 60
    rng = np.random.Generator(np.random.PCG64(SEED))
    indices = np.sort(rng.choice(len(eligible), size=10, replace=False))
    eligible['selected'] = eligible.index.isin(indices)
    selected = eligible.iloc[indices].copy()
    selected['panel'] = list('abcdefghij')
    groups = []
    for month, g in selected.groupby(selected.time.dt.strftime('%Y-%m')):
        path = project / f'Outputs/Experiments/Fast_Fading_Revision_001/clean_1s/clean_1s_{month}.parquet'
        sources['second_records_' + month] = path
        for hour in g.time:
            records = pd.read_parquet(path, filters=[('hour_bjt', '>=', hour),
                                         ('hour_bjt', '<', hour + pd.Timedelta(hours=1))])
            records = records.sort_values('second_bjt').reset_index(drop=True)
            assert len(records) == 3600 and records.primary_usable_1s.all()
            assert records.second_bjt.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all()
            records['power_dbm'] = 10 * np.log10(records.received_power_w) + 30
            records['recorded_hour_mean_dbm'] = 10 * np.log10(records.hour_mean_power_w) + 30
            records['elapsed_second'] = (records.second_bjt - hour).dt.total_seconds().astype(int)
            expected_mean = selected.loc[selected.time == hour, 'power_dbm'].iloc[0]
            assert np.max(abs(records.recorded_hour_mean_dbm - expected_mean)) < 1e-10
            records['official_weather'] = 'rain'
            keep = ['hour_bjt', 'second_bjt', 'elapsed_second', 'received_power_w',
                    'hour_mean_power_w', 'power_dbm', 'recorded_hour_mean_dbm',
                    'primary_usable_1s', 'quality_zero_records', 'official_weather', 'split']
            groups.append(records[keep])
    seconds = pd.concat(groups).sort_values('second_bjt')
    csv(seconds, data / 'rain_10h_seconds.csv')
    csv(selected.drop(columns='time'), data / 'selected_rain_hours.csv')
    csv(eligible.drop(columns='time'), data / 'all_eligible_rain_hours.csv')
    shutil.copy2(sources['local_hours'], data / 'all_local_hours_original.csv')
    js(data / 'selection.json', {
        'seed': SEED, 'generator': 'numpy.random.Generator(numpy.random.PCG64)',
        'algorithm': 'Sort eligible hours chronologically, choice(60, 10, replace=False), sort selected indices.',
        'eligible_hours': len(eligible), 'selected_indices_zero_based': indices.tolist(),
        'eligibility': 'Official local weather rain; exactly 3600 valid primary 1-s samples.',
        'pool': 'All local train/validation/test hours, for measurement illustration only.',
        'selection_does_not_use': 'Fluctuation magnitude, model errors, visual appearance, or ITU performance.',
        'selected_hours_bjt': selected.hour_bjt.tolist(), 'seconds': len(seconds),
        'hour_mean_definition': 'Frozen mean of valid native linear-power samples, converted to dBm; not mean dBm.',
        'nonconsecutive_hours_shown_in_separate_panels': True, 'smoothing': False,
        'dropped_or_clipped_second_values': 0,
    })
    manifest = [{'role': role, 'path': str(p.resolve()), 'sha256': sha(p),
                 'bytes': p.stat().st_size} for role, p in sources.items()]
    js(package / 'QA/source_manifest.json', manifest)
    print('Prepared all 684 eligible ITU pairs and 36000 original rainy seconds.')


def scores(frame):
    # math.fsum independently recalculates the frozen CSV scores.
    y = frame.truth_db.to_list(); p = frame.prediction_db.to_list(); n = len(y)
    mean_y = math.fsum(y) / n
    sse = math.fsum((a - b) ** 2 for a, b in zip(y, p))
    sst = math.fsum((v - mean_y) ** 2 for v in y)
    return {'hours': n, 'rmse_db': math.sqrt(sse / n), 'r2': 1 - sse / sst,
            'mae_db': math.fsum(abs(a - b) for a, b in zip(y, p)) / n,
            'bias_db_prediction_minus_observed': math.fsum(b - a for a, b in zip(y, p)) / n,
            'observed_sd_population_db': math.sqrt(sst / n),
            'predicted_sd_population_db': float(np.std(p)),
            'observed_min_db': min(y), 'observed_max_db': max(y),
            'predicted_min_db': min(p), 'predicted_max_db': max(p)}


def verify(package, originals=False):
    data = package / 'Data'
    itu = pd.read_csv(data / 'itu_all_684_hours.csv', float_precision='round_trip')
    selected = pd.read_csv(data / 'selected_rain_hours.csv', float_precision='round_trip')
    candidates = pd.read_csv(data / 'all_eligible_rain_hours.csv', float_precision='round_trip')
    seconds = pd.read_csv(data / 'rain_10h_seconds.csv', float_precision='round_trip')
    report = scores(itu)
    assert abs(report['rmse_db'] - 1.008963) < 1e-6 and abs(report['r2'] - .028100) < 1e-6
    assert len(itu) == 684 and len(selected) == 10 and len(seconds) == 36000
    assert len(candidates) == 60 and selected.event.eq('rain').all()
    rng = np.random.Generator(np.random.PCG64(SEED))
    indices = np.sort(rng.choice(len(candidates), 10, replace=False))
    assert selected.hour_bjt.to_list() == candidates.iloc[indices].hour_bjt.to_list()
    means = selected.set_index('hour_bjt').power_dbm
    max_difference = float(np.max(abs(seconds.recorded_hour_mean_dbm - seconds.hour_bjt.map(means))))
    assert max_difference < 1e-10 and seconds.primary_usable_1s.all()
    assert np.max(abs(seconds.power_dbm - (10 * np.log10(seconds.received_power_w) + 30))) < 1e-10
    for _, group in seconds.groupby('hour_bjt'):
        assert np.array_equal(group.elapsed_second.to_numpy(), np.arange(3600))
    component_error = float(np.max(abs(itu.prediction_db - itu[['gas_relative_db', 'cloud_db', 'rain_db']].sum(axis=1))))
    assert component_error < 1e-12
    source_ok = None
    if originals:
        source_ok = all(sha(item['path']) == item['sha256'] for item in
                        json.loads((package / 'QA/source_manifest.json').read_text(encoding='utf-8')))
        assert source_ok
    return {'status': 'PASS', 'itu_overall': report,
            'itu_by_weather': {key: scores(group) for key, group in itu.groupby('weather')},
            'component_sum_max_difference_db': component_error,
            'selected_rain_hours': len(selected), 'seconds': len(seconds), 'random_selection_reproduced': True,
            'all_second_values_kept': True, 'hour_mean_source_max_difference_db': max_difference,
            'above_detail_axis': int((itu.truth_db > DETAIL_LIMITS[1]).sum()),
            'all_684_pairs_visible_on_full_axis': True, 'original_hashes_unchanged': source_ok,
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 8.4, 'axes.labelsize': 8, 'xtick.labelsize': 7.3,
        'ytick.labelsize': 7.3, 'legend.fontsize': 7.5, 'axes.linewidth': .5,
        'axes.edgecolor': '#BDC5CB', 'axes.labelcolor': '#303B44',
        'text.color': '#303B44', 'xtick.color': '#54616B', 'ytick.color': '#54616B',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'path.simplify': False,
        'figure.dpi': 150, 'savefig.dpi': 300})


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.2, width=.5, pad=2)
    ax.grid(axis='y', color='#E3E7EB', linewidth=.45, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out, stem, title):
    metadata = {'Title': title, 'Author': 'TVT manuscript figure preparation',
                'CreationDate': datetime(2026, 11, 2, tzinfo=timezone.utc),
                'ModDate': datetime(2026, 11, 2, tzinfo=timezone.utc)}
    fig.savefig(out / (stem + '.pdf'), bbox_inches='tight', pad_inches=.04, metadata=metadata)
    fig.savefig(out / (stem + '.png'), bbox_inches='tight', pad_inches=.04, dpi=300,
                metadata={'Software': 'Matplotlib Manuscript_Figures Revision_005'})
    plt.close(fig)


def draw_itu(package, output):
    a = pd.read_csv(package / 'Data/itu_all_684_hours.csv', float_precision='round_trip')
    x = a.ordered_test_hour.to_numpy(); y = a.truth_db.to_numpy(); p = a.prediction_db.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.45), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.15]})
    fig.subplots_adjust(left=.072, right=.987, bottom=.15, top=.87, hspace=.30)
    for ax in axes:
        tidy(ax)
        ax.plot(x, y, color=BLUE, linewidth=.65, marker='.', markersize=1.4, label='Measured attenuation', zorder=3)
        ax.plot(x, p, color=ORANGE, linewidth=.95, linestyle='--', label='ITU-based combined estimate', zorder=4)
        ax.set(xlim=(0, 685), ylabel='Attenuation (dB)')
    axes[0].set(ylim=(-1.0, 14.1), yticks=[0, 4, 8, 12])
    axes[0].set_title('(a) Complete range: all 684 valid test hours', loc='left', pad=6)
    for row in a.itertuples():
        axes[0].axvspan(row.ordered_test_hour - .5, row.ordered_test_hour + .5,
                       color=WEATHER[row.weather][1], alpha=.14, linewidth=0, zorder=0)
    axes[0].text(.018, .94, r'$R^2=0.0281$' + '\nRMSE = 1.0090 dB',
                 ha='left', va='top', transform=axes[0].transAxes,
                 bbox={'facecolor': 'white', 'edgecolor': '#D7DDE1', 'alpha': .94, 'pad': 3}, fontsize=8.2)
    axes[1].set_ylim(DETAIL_LIMITS)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_xticks([1, 100, 200, 300, 400, 500, 600, 684])
    above = y > DETAIL_LIMITS[1]
    axes[1].plot(x[above], np.full(above.sum(), DETAIL_LIMITS[1] - .04), '^',
                 color=BLUE, markersize=3.5, markeredgewidth=0, zorder=5)
    axes[1].set_title('(b) Vertical enlargement of the same sequence', loc='left', pad=6)
    axes[1].set_xlabel('Test-hour index in chronological order (calendar gaps omitted)', labelpad=5)
    lines = [Line2D([], [], color=BLUE, lw=.9, label='Measured attenuation'),
             Line2D([], [], color=ORANGE, lw=1.0, ls='--', label='ITU-based combined estimate')]
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(.53, .987), ncol=2,
               frameon=False, handlelength=2.6, columnspacing=2.5)
    fig.legend(handles=[Patch(facecolor=color, alpha=.5, edgecolor='none', label=label)
                        for label, color in WEATHER.values()], loc='lower center',
               bbox_to_anchor=(.53, .006), ncol=4, frameon=False, columnspacing=2)
    save(fig, output, ITU_STEM, 'All 684 paired hourly attenuation observations and ITU estimates')


def draw_rain(package, output):
    selected = pd.read_csv(package / 'Data/selected_rain_hours.csv', float_precision='round_trip')
    sec = pd.read_csv(package / 'Data/rain_10h_seconds.csv', float_precision='round_trip')
    fig, axes = plt.subplots(5, 2, figsize=(7.1, 7.05), sharex=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.07, top=.925, hspace=.55, wspace=.23)
    for ax, row in zip(axes.flat, selected.itertuples()):
        g = sec[sec.hour_bjt == row.hour_bjt]
        hour = pd.Timestamp(row.hour_bjt)
        ax.plot(g.elapsed_second / 60, g.power_dbm, color=BLUE, linewidth=.30, alpha=.93, zorder=3)
        ax.axhline(row.power_dbm, color=ORANGE, linestyle='--', linewidth=1.0, zorder=4)
        span = float(g.power_dbm.max() - g.power_dbm.min())
        padding = max(.03, .08 * span)
        ax.set(ylim=(g.power_dbm.min() - padding, g.power_dbm.max() + padding),
               xlim=(0, 60), xticks=[0, 15, 30, 45, 60])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.set_title(f'({row.panel}) {hour:%Y-%m-%d}  {hour:%H}:00\N{EN DASH}{(hour + pd.Timedelta(hours=1)):%H}:00',
                     loc='left', fontsize=8, pad=5)
        tidy(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel('Minutes within the selected hour')
    fig.supylabel('Received power (dBm)', x=.008, fontsize=8.5)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=.7, label='Measured 1-s power'),
                        Line2D([], [], color=ORANGE, ls='--', lw=1, label='Hourly mean (linear-power average)')],
               loc='upper center', bbox_to_anchor=(.54, .996), ncol=2, frameon=False,
               handlelength=2.4, columnspacing=2)
    fig.text(.54, .956, 'Ten randomly selected complete rainy hours | Beijing time', ha='center', fontsize=8)
    save(fig, output, RAIN_STEM, 'Ten randomly selected rainy hours, full 1-s measurements and native hourly means')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-originals', action='store_true')
    args = parser.parse_args()
    package = args.package or args.project / PACKAGE
    if args.prepare:
        prepare(args.project, package)
    output = args.output_dir or package
    output.mkdir(parents=True, exist_ok=True)
    (output / 'QA').mkdir(exist_ok=True)
    assert not (output / (ITU_STEM + '.pdf')).exists(), 'Use a fresh output directory.'
    report = verify(package, originals=args.prepare or args.verify_originals)
    style(); draw_itu(package, output); draw_rain(package, output)
    js(output / 'QA/numeric_check.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
