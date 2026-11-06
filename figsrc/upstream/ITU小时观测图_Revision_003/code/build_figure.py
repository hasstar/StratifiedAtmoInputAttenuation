"""Fig04 revision: all local measurements above, unchanged ITU test data below.

Only changes presentation. All source records are frozen; there is no fitting.
Use --prepare once; then --package PACKAGE --output-dir NEW_DIR to reproduce.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = Path(r'E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction')
OLD = '最终提交的修改材料/ITU与雨天观测图_20261102_001'
NEW = '最终提交的修改材料/ITU小时观测图_Revision_003'
STEM = 'Fig04_All_Hourly_Observations_And_ITU_Test'
BLUE, ORANGE = '#0072B2', '#D55E00'
WEATHER = {'rain': ('Rain', '#7AAED1'), 'dry_overcast': ('Overcast', '#DEA683'),
           'clear_dry': ('Clear/dry', '#8CC4AF'), 'other': ('Other', '#B9ACC9')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str), encoding='utf-8')


def prepare(project, package):
    old = project / OLD
    data = package / 'Data'
    data.mkdir(parents=True, exist_ok=False)
    (package / 'QA').mkdir(exist_ok=True)
    records = []
    for source, dest in [(old / 'Data/all_local_hours_original.csv', data / 'all_local_hours_original.csv'),
                         (old / 'Data/itu_all_684_hours.csv', data / 'itu_test_684_hours.csv')]:
        before = sha(source)
        shutil.copy2(source, dest)
        assert before == sha(dest) == sha(source)
        records.append({'original': str(source.resolve()), 'copy': str(dest.relative_to(package)), 'sha256': before})
    hourly = pd.read_csv(data / 'all_local_hours_original.csv', float_precision='round_trip')
    hourly['time'] = pd.to_datetime(hourly.hour_bjt, utc=True)
    hourly = hourly.sort_values('time').reset_index(drop=True)
    hourly.insert(0, 'all_hour_index', np.arange(1, len(hourly) + 1))
    hourly.drop(columns='time').to_csv(data / 'all_3434_hours_in_plot_order.csv', index=False, float_format='%.17g')
    records.extend([{'original': str((project / 'Code/Manuscript_Figures/Revision_005/build_figures_v002.py').resolve()),
                     'sha256': sha(project / 'Code/Manuscript_Figures/Revision_005/build_figures_v002.py')},
                    {'original': str((old / 'Figures/Fig03_Ten_Random_Rainy_Hours.pdf').resolve()),
                     'sha256': sha(old / 'Figures/Fig03_Ten_Random_Rainy_Hours.pdf'),
                     'role': 'Unchanged second-scale figure; not regenerated'}])
    js(package / 'QA/source_manifest.json', records)


def verify(package, originals=False):
    data = package / 'Data'
    all_h = pd.read_csv(data / 'all_3434_hours_in_plot_order.csv', float_precision='round_trip')
    original = pd.read_csv(data / 'all_local_hours_original.csv', float_precision='round_trip')
    original['sort_time'] = pd.to_datetime(original.hour_bjt, utc=True)
    original = original.sort_values('sort_time').reset_index(drop=True)
    b = pd.read_csv(data / 'itu_test_684_hours.csv', float_precision='round_trip')
    assert len(all_h) == 3434 and all_h.hour_bjt.is_unique
    assert np.array_equal(all_h.all_hour_index, np.arange(1, 3435))
    assert all_h.event.tolist() == original.event.tolist()
    assert np.array_equal(all_h.clear_sky_attenuation_db, original.clear_sky_attenuation_db)
    assert len(b) == 684 and np.array_equal(b.ordered_test_hour, np.arange(1, 685))
    observed = all_h.set_index('hour_bjt')
    assert observed.loc[b.hour_bjt, 'split'].eq('test').all()
    assert np.array_equal(b.truth_db, observed.loc[b.hour_bjt, 'clear_sky_attenuation_db'])
    sse = np.sum((b.truth_db - b.prediction_db) ** 2)
    rmse = float(np.sqrt(sse / len(b)))
    r2 = float(1 - sse / np.sum((b.truth_db - b.truth_db.mean()) ** 2))
    assert abs(rmse - 1.008963) < 1e-6 and abs(r2 - .028100) < 1e-6
    assert all_h.clear_sky_attenuation_db.between(-1.0, 14.1).all()
    hashes = json.loads((package / 'QA/source_manifest.json').read_text(encoding='utf-8'))
    for item in hashes:
        if 'copy' in item:
            assert sha(package / item['copy']) == item['sha256']
        if originals:
            assert sha(item['original']) == item['sha256']
    return {'status': 'PASS', 'top_panel_hours': len(all_h),
            'top_split_counts': all_h.split.value_counts().to_dict(),
            'top_weather_counts': all_h.event.value_counts().to_dict(),
            'top_range_db': [all_h.clear_sky_attenuation_db.min(), all_h.clear_sky_attenuation_db.max()],
            'top_original_values_exact': True, 'top_prediction_curve': False,
            'top_all_observations_within_axis': True,
            'bottom_test_hours': len(b), 'bottom_csv_byte_identical_to_previous': True,
            'bottom_ylim_db': [-.72, 3.10], 'bottom_above_axis': int((b.truth_db > 3.10).sum()),
            'rmse_db': rmse, 'r2': r2, 'both_axes_omit_calendar_gaps': True,
            'all_times_retained_in_csv': True, 'source_hashes_checked': originals,
            'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}


def draw(package, output):
    a = pd.read_csv(package / 'Data/all_3434_hours_in_plot_order.csv', float_precision='round_trip')
    b = pd.read_csv(package / 'Data/itu_test_684_hours.csv', float_precision='round_trip')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 8.4, 'axes.labelsize': 8, 'xtick.labelsize': 7.3,
        'ytick.labelsize': 7.3, 'legend.fontsize': 7.5, 'axes.linewidth': .5,
        'axes.edgecolor': '#BDC5CB', 'axes.labelcolor': '#303B44',
        'text.color': '#303B44', 'xtick.color': '#54616B', 'ytick.color': '#54616B',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'path.simplify': False,
        'figure.dpi': 150, 'savefig.dpi': 300})
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.85), sharex=False,
                             gridspec_kw={'height_ratios': [1, 1.15]})
    fig.subplots_adjust(left=.072, right=.987, bottom=.14, top=.92, hspace=.64)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.2, width=.5, pad=2)
        ax.grid(axis='y', color='#E3E7EB', linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel('Attenuation (dB)')
    # Merge adjacent equal-weather colours only on the plotted observation index.
    events = a.event.to_numpy()
    boundaries = np.r_[0, np.flatnonzero(events[1:] != events[:-1]) + 1, len(a)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        axes[0].axvspan(start + .5, stop + .5, color=WEATHER[events[start]][1],
                       alpha=.20, linewidth=0, zorder=0)
    axes[0].plot(a.all_hour_index, a.clear_sky_attenuation_db, color=BLUE,
                 linewidth=.65, zorder=3)
    axes[0].set(xlim=(0, 3435), ylim=(-1.0, 14.1), yticks=[0, 4, 8, 12],
                xticks=[1, 500, 1000, 1500, 2000, 2500, 3000, 3434],
                xlabel='All-hour index in chronological order (calendar gaps omitted)')
    axes[0].set_title('(a) All local hourly measurements (3,434 h)', loc='left', pad=8)
    axes[0].legend(handles=[Patch(facecolor=c, alpha=.55, edgecolor='none', label=lab)
                           for lab, c in WEATHER.values()], loc='upper left', ncol=4,
                   frameon=True, facecolor='white', framealpha=.90, edgecolor='#DDE2E6',
                   fontsize=7, handlelength=1.4, columnspacing=1.3, borderpad=.4)

    # Keep the old bottom panel's data, order, lines, limits, and overflow marks.
    x, y, p = b.ordered_test_hour.to_numpy(), b.truth_db.to_numpy(), b.prediction_db.to_numpy()
    axes[1].plot(x, y, color=BLUE, linewidth=.65, marker='.', markersize=1.4, zorder=3)
    axes[1].plot(x, p, color=ORANGE, linewidth=.95, linestyle='--', zorder=4)
    axes[1].set(xlim=(0, 685), ylim=(-.72, 3.10), yticks=[0, 1, 2, 3],
                xticks=[1, 100, 200, 300, 400, 500, 600, 684],
                xlabel='Test-hour index in chronological order (calendar gaps omitted)')
    above = y > 3.10
    axes[1].plot(x[above], np.full(above.sum(), 3.10 - .04), '^', color=BLUE,
                 markersize=3.5, markeredgewidth=0, zorder=5)
    axes[1].set_title('(b) Test subset: measured attenuation and ITU predictions (684 h)', loc='left', pad=8)
    axes[1].text(.018, .94, r'$R^2=0.0281$' + '\nRMSE = 1.0090 dB',
                 ha='left', va='top', transform=axes[1].transAxes,
                 bbox={'facecolor': 'white', 'edgecolor': '#D7DDE1', 'alpha': .94, 'pad': 3}, fontsize=7.6)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=.9, label='Measured attenuation'),
                        Line2D([], [], color=ORANGE, lw=1.0, ls='--', label='ITU-based combined prediction (test only)')],
               loc='lower center', bbox_to_anchor=(.53, .005), ncol=2, frameon=False,
               handlelength=2.6, columnspacing=2.0)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / (STEM + '.pdf')).exists(), 'Use a fresh output directory.'
    metadata = {'Title': 'All local hourly attenuation and the frozen ITU test comparison',
                'Author': 'TVT manuscript figure preparation',
                'CreationDate': datetime(2026, 11, 2, tzinfo=timezone.utc),
                'ModDate': datetime(2026, 11, 2, tzinfo=timezone.utc)}
    fig.savefig(output / (STEM + '.pdf'), bbox_inches='tight', pad_inches=.04, metadata=metadata)
    fig.savefig(output / (STEM + '.png'), bbox_inches='tight', pad_inches=.04, dpi=300,
                metadata={'Software': 'Matplotlib Manuscript_Figures Revision_007'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-originals', action='store_true')
    args = parser.parse_args()
    package = args.package or args.project / NEW
    if args.prepare:
        prepare(args.project, package)
    out = args.output_dir or package
    checks = verify(package, originals=args.prepare or args.verify_originals)
    draw(package, out)
    (out / 'QA').mkdir(exist_ok=True)
    js(out / 'QA/numeric_check.json', checks)
    print(json.dumps(checks, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
