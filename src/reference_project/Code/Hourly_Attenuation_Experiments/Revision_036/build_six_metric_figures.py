# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas"]
# ///
"""Render six requested overlays with a highlighted TabM from accepted T014 metrics."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import seaborn as sns


def main():
    """Render six original-value, ten-model curves without any model calls.

    Parameters
    ----------
    None
        All input/output paths are pinned to this project and this revision.

    Saves
    -----
    Six PNG/PDF figures, a 1560-row six-metric CSV and a provenance/QA JSON.
    """
    root = Path(__file__).resolve().parents[3]
    source_dir = root / "Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002"
    source = source_dir / "tables/all_test_metrics_15605_rows.csv"
    source_qa = source_dir / "evidence/FINAL_QA.json"
    out = root / "Outputs/Figures/T014_Six_Metrics_TabM_Glow_20261020_001"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    qa_hash = hashlib.sha256(source_qa.read_bytes()).hexdigest()
    assert source_hash == "e787db26030865517dd839dfa22bc91a7f5732a925748584395af748691b08b1"
    assert qa_hash == "33bc8db491146d40278b585fd3bda9508176207daf58767f51630e7f1625da51"
    accepted = json.loads(source_qa.read_text(encoding="utf-8"))
    assert accepted["status"] == "PASS" and accepted["states"] == 1560
    assert not out.exists(), f"Refusing to overwrite {out}"

    # --- Data ---
    raw = pd.read_csv(source, float_precision="round_trip")
    data = raw.loc[raw.population.eq("ML_687") & raw.weather.isin(["overall", "rain", "dry_overcast"])].copy()
    assert len(data) == 4680
    required = ["model", "k", "n", "rmse_db", "r2"]
    assert not data[required].isna().any().any(), "Missing data must be disclosed, never hidden."
    # No rows removed: the assertion above enforces complete requested data.
    data = data.dropna(subset=required)
    assert np.isfinite(data[["k", "n", "rmse_db", "r2"]].to_numpy()).all()
    data["k"] = data.k.astype(int)
    styles = [
        ("tabm", "TabM (Ours)", "#0072B2", "-", "o"),
        ("bai_lstm", "Bai LSTM", "#D55E00", "--", "s"),
        ("bilstm", "BiLSTM", "#009E73", "-.", "^"),
        ("transformer", "Transformer", "#7A5195", ":", "D"),
        ("mlp", "MLP", "#008EAA", (0, (6, 1)), "v"),
        ("catboost", "CatBoost", "#B3446C", (0, (3, 1, 1, 1)), "P"),
        ("gcn", "GCN", "#303030", (0, (5, 1, 1, 1, 1, 1)), "X"),
        ("random_forest", "Random Forest", "#967100", (0, (1, 1)), "<"),
        ("svr_rbf", "SVR-RBF", "#777777", (0, (3, 2)), ">"),
        ("elastic_net", "Elastic Net", "#8F4E2F", (0, (7, 2, 1, 2)), "h"),
    ]
    models = [s[0] for s in styles]
    assert set(data.model) == set(models)
    expected_k = np.arange(1, 157)
    for (model, weather), group in data.groupby(["model", "weather"]):
        assert np.array_equal(group.sort_values("k").k, expected_k), (model, weather)
        assert group.n.eq({"overall": 687, "rain": 51, "dry_overcast": 158}[weather]).all()

    # --- Style Setup ---
    helper_dir = Path("C:/Users/SDR_HH/.codex/skills/paper-figures/scripts")
    sys.path.insert(0, str(helper_dir))
    from figstyle import apply_preset
    apply_preset("generic", column="double")
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelcolor": "dimgrey", "text.color": "dimgrey",
        "axes.titleweight": "normal", "path.simplify": False,
        "savefig.facecolor": "white", "savefig.transparent": False,
    })
    chinese = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    out.mkdir(parents=True, exist_ok=False)
    specs = [
        ("F01_rain_R2", "rain", "r2", "雨天R²", r"雨天 $R^2$", 51),
        ("F02_overcast_R2", "dry_overcast", "r2", "阴天R²（干燥阴天）", r"干燥阴天 $R^2$", 158),
        ("F03_overall_R2", "overall", "r2", "总体R²", r"总体 $R^2$", 687),
        ("F04_rain_RMSE", "rain", "rmse_db", "雨天RMSE", "雨天 RMSE (dB)", 51),
        ("F05_overcast_RMSE", "dry_overcast", "rmse_db", "阴天RMSE（干燥阴天）", "干燥阴天 RMSE (dB)", 158),
        ("F06_overall_RMSE", "overall", "rmse_db", "总体RMSE", "总体 RMSE (dB)", 687),
    ]
    plot_data = data[data.weather.eq("overall")][["model", "k", "r2", "rmse_db"]].rename(
        columns={"r2": "overall_r2", "rmse_db": "overall_rmse_db"})
    for weather in ["rain", "dry_overcast"]:
        plot_data = plot_data.merge(data[data.weather.eq(weather)][["model", "k", "r2", "rmse_db"]].rename(
            columns={"r2": f"{weather}_r2", "rmse_db": f"{weather}_rmse_db"}),
            on=["model", "k"], validate="one_to_one")
    plot_data["population"] = "ML_687"
    plot_data["overall_n"] = 687
    plot_data["rain_n"] = 51
    plot_data["dry_overcast_n"] = 158
    plot_data["seed"] = 17
    plot_data = plot_data.sort_values(["model", "k"])
    plot_data.to_csv(out / "plot_data_1560_states.csv", index=False, encoding="utf-8-sig", float_format="%.17g", mode="x")
    round_trip = pd.read_csv(out / "plot_data_1560_states.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(plot_data.reset_index(drop=True), round_trip, check_exact=True)
    figures = []
    all_line_points = 0

    # --- Plot ---
    for name, weather, metric, title, ylabel, sample_n in specs:
        fig, ax = plt.subplots(figsize=(12.8, 7.6), dpi=150)
        fig.subplots_adjust(left=0.088, right=0.98, bottom=0.14, top=0.77)
        selected = data[data.weather.eq(weather)]
        curves = []
        for index, (model, label, color, linestyle, marker) in enumerate(styles):
            series = selected[selected.model.eq(model)].sort_values("k")
            is_tabm = model == "tabm"
            line, = ax.plot(series.k, series[metric], color=color, linestyle=linestyle,
                            linewidth=2.8 if is_tabm else 1.35, marker=marker,
                            markersize=5.2 if is_tabm else 4.2,
                            markevery=(index, 15), markerfacecolor="white", markeredgewidth=1.0,
                            label=label, alpha=1.0, zorder=5 if is_tabm else 3)
            if is_tabm:
                # The fluorescent halo follows the identical line; it is not uncertainty.
                line.set_path_effects([
                    pe.Stroke(linewidth=10.5, foreground="#00E5FF", alpha=0.07),
                    pe.Stroke(linewidth=7.0, foreground="#00E5FF", alpha=0.13),
                    pe.Stroke(linewidth=4.8, foreground="#00E5FF", alpha=0.26),
                    pe.Normal(),
                ])
            np.testing.assert_array_equal(line.get_xdata(), expected_k)
            np.testing.assert_array_equal(line.get_ydata(), series[metric].to_numpy())
            curves.append({"model": model, "points": len(series), "minimum": float(series[metric].min()),
                           "maximum": float(series[metric].max()), "negative_points": int(series[metric].lt(0).sum())})
            all_line_points += len(series)
        assert len(ax.lines) == 10
        ax.set_xlim(0, 157)
        ax.set_xticks([1, 20, 40, 60, 80, 100, 120, 140, 156])
        ax.set_xlabel("参数数目（保留输入特征数 k）", fontproperties=chinese, fontsize=13, labelpad=12)
        ax.set_ylabel(ylabel, fontproperties=chinese, fontsize=13, labelpad=12)
        if metric == "r2":
            ax.set_ylim(-0.90, 1.00)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.axhline(0, color="#999999", linewidth=0.85, linestyle="--", zorder=1)
        else:
            ax.set_ylim(0, 1.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
        assert selected[metric].between(*ax.get_ylim()).all()
        sns.despine(ax=ax, left=True, bottom=True)
        ax.grid(False)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.65)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey", labelsize=11, pad=7)
        legend = fig.legend(*ax.get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.535, 0.835),
                            ncol=5, fontsize=10.5, handlelength=3.5, columnspacing=1.6, labelspacing=0.9,
                            frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
                            labelcolor="dimgrey")
        legend.get_texts()[0].set_weight("bold")
        legend.get_texts()[0].set_color("#0072B2")
        fig.text(0.088, 0.966, f"十模型独立RFE：{title}", fontproperties=chinese,
                 fontsize=19, color="dimgrey", va="top")
        scope = {"overall": "完整测试集", "rain": "测试集雨天子集", "dry_overcast": "测试集干燥阴天子集"}[weather]
        fig.text(0.088, 0.912, f"{scope}  ·  n = {sample_n} 小时  ·  每条曲线包含156个状态",
                 fontproperties=chinese, fontsize=11.5, color="dimgrey", va="top")
        note = ("R²为决定系数，越高越好；三张R²图使用相同纵轴，负值完整保留。"
                if metric == "r2" else "RMSE越低越好；三张RMSE图使用相同纵轴范围。")
        fig.text(0.088, 0.041, note, fontproperties=chinese, fontsize=10, color="dimgrey")
        fig.text(0.088, 0.011, "TabM（本文模型）以粗线及青色荧光强调，晕光不表示误差范围；各模型全部156点直接连线，未平滑。",
                 fontproperties=chinese, fontsize=9.2, color="dimgrey")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lb = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert lb.y0 > ax.get_position().y1, "Legend overlaps the data panel."
        assert lb.x0 >= 0 and lb.x1 <= 1 and lb.y1 <= 1

        # --- Save ---
        for suffix in ["png", "pdf"]:
            target = out / f"{name}.{suffix}"
            assert not target.exists()
            fig.savefig(target, dpi=150, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        dimensions = list(plt.imread(out / f"{name}.png").shape[:2][::-1])
        assert (out / f"{name}.pdf").read_bytes().startswith(b"%PDF-")
        figures.append({"id": name, "title": title, "weather": weather, "metric": metric,
                        "n_hours": sample_n, "model_curves": 10, "points_per_curve": 156,
                        "png_pixels": dimensions, "export_dpi": 150, "curves": curves,
                        "x_limits": list(ax.get_xlim()), "y_limits": list(ax.get_ylim()),
                        "tabm_linewidth_pt": 2.8, "other_linewidth_pt": 1.35,
                        "tabm_halo": "Three cyan #00E5FF path strokes, widths 10.5/7.0/4.8 pt, alpha 0.07/0.13/0.26; visual emphasis only",
                        "caption": f"{title}随保留输入特征数k的变化。十模型各156状态，单种子17；{scope}{sample_n}小时。"
                                   "各模型独立RFE，未按测试性能重新选择参数。横轴升序展示，实际删列方向为156至1。",
                        "alt": f"十条折线比较十个模型在k等于1至156时的{title}；颜色、线型和标记在六图中一致；TabM加粗并附青色荧光晕光，非误差带。"})
        plt.close(fig)

    assert all_line_points == 9360
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(source_qa.read_bytes()).hexdigest() == qa_hash
    color_audit = []
    for model, label, color, linestyle, marker in styles:
        rgb = np.array(matplotlib.colors.to_rgb(color))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
        color_audit.append({"model": model, "label": label, "color": color, "marker": marker,
                            "linestyle": linestyle, "contrast_on_white": 1.05 / (luminance + 0.05)})
    manifest = {
        "status": "NUMERIC_AND_EXPORT_QA_PASS_PENDING_VISUAL_REVIEW",
        "created_bjt": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "source": str(source), "source_sha256": source_hash, "accepted_qa_sha256": qa_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_style_helper_sha256": hashlib.sha256((helper_dir / "figstyle.py").read_bytes()).hexdigest(),
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                        "matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
        "filter": "population == ML_687; overall/rain/dry_overcast; ten learning models",
        "metric_definition": "r2 is 1-SSE/SST, not Pearson r squared; RMSE in dB",
        "x_definition": "Retained input feature count, not trainable network weight count",
        "all_plot_artist_values_exactly_equal_source": True, "plotted_values": all_line_points,
        "plot_data_csv_round_trip_exact": True, "source_unchanged": True,
        "missing_requested_values": 0, "smoothing": False, "training_calls": 0,
        "uncertainty": "Single frozen seed 17; no multi-seed error bars or significance inference",
        "review_context": "General on-screen review figures; no journal-format compliance claim",
        "color_audit": color_audit, "figures": figures,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(out.iterdir()) if p.is_file()],
    }
    with (out / "FIGURE_QA_AND_PROVENANCE.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(out), "figures": 6, "model_curves_per_figure": 10,
                      "plotted_values": all_line_points, "source_unchanged": True,
                      "color_contrast_min": min(c["contrast_on_white"] for c in color_audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
