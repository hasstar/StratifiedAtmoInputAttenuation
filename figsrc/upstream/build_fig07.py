"""Redraw frozen TabM RFE metrics: two panels, each with a k=1--10 inset.

No fitting, feature selection, smoothing, or recomputation of predictions.
Run with the existing scientific Python; see the delivered reproduce.ps1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", type=Path, help="Optional JSON layout override")
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ["Fig07_TabM_Independent_RFE_Paths.pdf",
             "Fig07_TabM_Independent_RFE_Paths.png",
             "comparison_audit.csv", "numerical_qa.json", "render_manifest.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Use a new output directory; existing artifacts are preserved.")

    data = pd.read_csv(source)
    frame = data.loc[(data["split"] == "test") &
                     (data["weather"] == "overall")].sort_values("k").copy()
    assert frame["k"].tolist() == list(range(1, 51))
    assert frame["n"].eq(687).all()
    cols = ["all_r2", "surface_r2", "all_rmse_db", "surface_rmse_db"]
    assert np.isfinite(frame[cols].to_numpy()).all()
    dr = frame["surface_rmse_db"] - frame["all_rmse_db"]
    d2 = frame["all_r2"] - frame["surface_r2"]
    full = frame.loc[dr > 1e-12, "k"].astype(int).tolist()
    base = frame.loc[dr < -1e-12, "k"].astype(int).tolist()
    tied = frame.loc[np.abs(dr) <= 1e-12, "k"].astype(int).tolist()
    assert full == list(range(8, 48)) + [49]
    assert base == list(range(3, 8)) + [48, 50]
    assert tied == [1, 2]
    assert np.array_equal(np.sign(dr.to_numpy()), np.sign(d2.to_numpy()))

    layout = {
        "figsize": [7.15, 3.55], "font": "Times New Roman",
        "font_size": 13, "tick_size": 12, "inset_tick_size": 10.5,
        "legend_size": 12.2, "line_width": 1.85,
        "left": .085, "right": .982, "bottom": .19, "top": .845,
        "wspace": .34,
        "r2_inset": [.31, .10, .65, .57],
        "rmse_inset": [.31, .43, .65, .53],
    }
    if args.layout:
        layout.update(json.loads(args.layout.read_text(encoding="utf-8")))
    plt.rcParams.update({
        "font.family": layout["font"], "font.size": layout["font_size"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelsize": layout["font_size"],
        "xtick.labelsize": layout["tick_size"],
        "ytick.labelsize": layout["tick_size"],
        "axes.linewidth": .65, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=layout["figsize"])
    fig.subplots_adjust(**{k: layout[k] for k in
                           ("left", "right", "bottom", "top", "wspace")})
    palette = [("all_", "#008C4A", "-", "156-candidate pool"),
               ("surface_", "#0072B2", "--", "50-candidate base pool")]
    evidence = []
    for ax, metric, panel, ylim, yticks, box in [
        (axes[0], "r2", "(a)", (0, 1.01), [0, .2, .4, .6, .8, 1.0], layout["r2_inset"]),
        (axes[1], "rmse_db", "(b)", (.2, 1.03), [.2, .4, .6, .8, 1.0], layout["rmse_inset"]),
    ]:
        ax.set(xlim=(1, 50), ylim=ylim, xlabel=r"Retained features, $k$",
               ylabel=r"$R^2$" if metric == "r2" else "RMSE (dB)")
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_yticks(yticks)
        ax.set_title(panel + r"  $k=1$–$50$", loc="left", fontsize=13, pad=7)
        ax.grid(color="#DEE3E7", linewidth=.55, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#A9B2BA")
        ax.tick_params(length=3, width=.6, pad=3)
        inset = ax.inset_axes(box, facecolor="white", zorder=5)
        inset.set(xlim=(1, 10), ylim=(0, 1.01) if metric == "r2" else (.25, 1.03))
        inset.set_xticks([1, 5, 10])
        inset.set_yticks([0, .4, .8] if metric == "r2" else [.3, .6, .9])
        inset.set_title(r"$k=1$–$10$", fontsize=layout["inset_tick_size"] + .5, pad=4)
        inset.grid(color="#E9ECEF", linewidth=.45, zorder=0)
        inset.tick_params(labelsize=layout["inset_tick_size"], length=2.2, width=.5, pad=2)
        for spine in inset.spines.values():
            spine.set_color("#8E99A3")
            spine.set_linewidth(.65)
        zoom = frame.loc[frame["k"] <= 10]
        for prefix, color, style, label in palette:
            line, = ax.plot(frame["k"], frame[prefix + metric], linestyle=style,
                            color=color, linewidth=layout["line_width"], label=label,
                            solid_capstyle="round", zorder=3)
            small, = inset.plot(zoom["k"], zoom[prefix + metric], linestyle=style,
                               color=color, linewidth=layout["line_width"] * .85,
                               solid_capstyle="round", zorder=3)
            assert np.array_equal(line.get_ydata(), frame[prefix + metric].to_numpy())
            assert np.array_equal(small.get_ydata(), zoom[prefix + metric].to_numpy())
            evidence.append({"metric": metric, "pool": prefix,
                             "main_points": len(line.get_xdata()),
                             "inset_points": len(small.get_xdata())})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=2, frameon=False, fontsize=layout["legend_size"],
               handlelength=2.5, columnspacing=1.7)
    fig.savefig(out / names[0], metadata={"Title": "Independent TabM RFE paths with low-dimensional insets",
                                        "Creator": "build_fig07.py / T027", "CreationDate": None,
                                        "ModDate": None})
    fig.savefig(out / names[1], dpi=450)
    plt.close(fig)

    audit = frame[["k", "n"] + cols].copy()
    audit["rmse_50_minus_156_db"] = dr
    audit["r2_156_minus_50"] = d2
    audit["better_pool"] = np.where(dr > 1e-12, "156", np.where(dr < -1e-12, "50", "tie"))
    audit.to_csv(out / names[2], index=False, float_format="%.17g")
    k10 = frame.loc[frame["k"] == 10].iloc[0]
    qa = {"scope": "overall/local-test, same 687 hours, saved metrics only",
          "source_rows": len(data), "plotted_rows": len(frame),
          "full_pool_better_k": full, "base_pool_better_k": base, "tied_k": tied,
          "k10_rmse_reduction_percent": float(100 * (1 - k10.all_rmse_db / k10.surface_rmse_db)),
          "max_base_advantage_k48_k50_db": float((-dr.loc[frame.k.isin([48, 50])]).max()),
          "main_and_inset_values_exactly_equal_saved_metrics": True,
          "curve_point_counts": evidence,
          "training_or_feature_selection_performed": False}
    (out / names[3]).write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    manifest = {"input": {"path": str(source), "sha256": sha256(source)},
                "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
                "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__},
                "font_path": font_manager.findfont(layout["font"]), "layout": layout,
                "outputs": {name: sha256(out / name) for name in names[:-1]}}
    (out / names[4]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "full_wins": len(full), "base_wins": len(base),
                      "ties": len(tied), "k10_improvement_percent": qa["k10_rmse_reduction_percent"]}))


if __name__ == "__main__":
    main()
