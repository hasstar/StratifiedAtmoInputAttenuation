"""Plot saved Revision 016 test-hour ITU rain attenuation against observations.

No fitting, no model inference, no source-data edits. Exports IEEE-sized vector PDFs,
600-dpi PNG previews, paired plotting data, bilingual captions and a figure report.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
RUN = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline"
OUT = RUN / "Figures/ITU_vs_Observed_Revision_001"
SKILL_SCRIPTS = Path("C:/Users/SDR_HH/.codex/skills/paper-figures/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from figstyle import apply_preset, save_figure  # noqa: E402

OBS_COLOR = "#0072B2"
ITU_COLOR = "#D55E00"


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data() -> tuple[pd.DataFrame, dict]:
    qa = json.loads((RUN / "qa_final.json").read_text(encoding="utf-8"))
    independent = json.loads((RUN / "independent_validation.json").read_text(encoding="utf-8"))
    check(qa["status"] == independent["status"] == "PASS", "Source QA is not PASS")
    source = json.loads((RUN / "final_test.json").read_text(encoding="utf-8"))
    obs = np.asarray(source["truth_db"], dtype=float)
    itu = np.asarray(source["results"]["physics_geom_circular"]["prediction_db"], dtype=float)
    rain = np.asarray(source["rain"], dtype=bool)
    time = pd.to_datetime(source["hour_bjt"])
    check(len(obs) == len(itu) == len(rain) == len(time) == 687 and rain.sum() == 51,
          "Saved test-hour schema/count changed")
    check(np.isfinite(obs).all() and np.isfinite(itu).all(), "Nonfinite plotted values")
    check(time.is_monotonic_increasing and time.is_unique, "Hours not time ordered")
    frame = pd.DataFrame({"hour_bjt": time.astype(str),
                          "observed_clear_sky_attenuation_db": obs,
                          "itu_p838_geometry_circular_db": itu,
                          "rain_hour": rain})
    return frame, source


def date_ticks(ax: plt.Axes, frame: pd.DataFrame, count: int) -> None:
    positions = np.unique(np.linspace(0, len(frame) - 1, count, dtype=int))
    dates = pd.to_datetime(frame.hour_bjt.iloc[positions])
    labels = [d.strftime("%Y-%m") if len(frame) > 100 else d.strftime("%Y-%m-%d")
              for d in dates]
    ax.set_xticks(positions, labels)


def decorate(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.axhline(0, color="#777777", lw=.5, alpha=.65, zorder=0)
    if grid:
        ax.grid(axis="y", color="#E4E7EA", lw=.45)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    frame, source = data()
    expected = [OUT / name for name in ("FigS1.png", "FigS1.pdf", "FigS2.png",
                                         "FigS2.pdf", "plot_values.csv",
                                         "FIGURE_REPORT.md", "figure_items.json",
                                         "figure_manifest.json")]
    check(not any(path.exists() for path in expected),
          "Figure Revision 001 already exists; do not overwrite it")
    OUT.mkdir(parents=True, exist_ok=True)
    apply_preset("ieee", column="double")
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42})

    x = np.arange(len(frame))
    observed = frame.observed_clear_sky_attenuation_db.to_numpy()
    itu = frame.itu_p838_geometry_circular_db.to_numpy()
    rain = frame.rain_hour.to_numpy()
    all_rmse = float(np.sqrt(np.mean((observed - itu) ** 2)))
    rain_rmse = float(np.sqrt(np.mean((observed[rain] - itu[rain]) ** 2)))

    # Full range is kept in panel a; panel b is explicitly a zoom of identical data.
    fig, axes = plt.subplots(2, 1, figsize=(7.13, 4.7), sharex=True,
                             gridspec_kw={"height_ratios": [1.05, 1]})
    for ax in axes:
        ax.plot(x, observed, color=OBS_COLOR, lw=.7, marker=".", ms=2.0,
                label="Measured relative clear-sky attenuation", zorder=2)
        ax.plot(x, itu, color=ITU_COLOR, lw=.9, ls="--", marker=".", ms=1.6,
                label="ITU P.838 + geometric rain path", zorder=3)
        decorate(ax)
        ax.set_ylabel("Attenuation (dB)")
        ax.set_xlim(-3, len(frame) + 2)
    axes[0].set_ylim(-.9, 14)
    axes[0].set_title(f"(a) All test hours (n = 687; RMSE = {all_rmse:.3f} dB)", loc="left")
    axes[0].legend(loc="upper left", frameon=False, ncol=2,
                   handlelength=2.5, columnspacing=.9)
    axes[1].set_ylim(-.8, 2.5)
    axes[1].set_title("(b) Same data, expanded typical range; extremes remain in (a)", loc="left")
    date_ticks(axes[1], frame, 7)
    axes[1].set_xlabel("Time-ordered test hour (gaps omitted; tick labels are calendar dates)")
    save_figure(fig, number="S1", preset="ieee", outdir=str(OUT), kind="line",
                formats=["pdf", "png"])
    plt.close(fig)

    rainy = frame.loc[frame.rain_hour].reset_index(drop=True)
    rx = np.arange(len(rainy))
    ry = rainy.observed_clear_sky_attenuation_db.to_numpy()
    rp = rainy.itu_p838_geometry_circular_db.to_numpy()
    fig, ax = plt.subplots(figsize=(7.13, 3.35))
    ax.plot(rx, ry, color=OBS_COLOR, lw=1.0, marker="o", ms=2.8,
            label="Measured relative clear-sky attenuation", zorder=3)
    ax.plot(rx, rp, color=ITU_COLOR, lw=1.0, ls="--", marker="s", ms=2.5,
            label="ITU P.838 + geometric rain path", zorder=4)
    decorate(ax)
    ax.set_xlim(-.7, len(rainy) - .3)
    ax.set_ylim(-.6, 8.7)
    ax.set_ylabel("Attenuation (dB)")
    ax.set_xlabel("Time-ordered rainy test hour (gaps omitted; tick labels are calendar dates)")
    ax.set_title(f"Rain-hour comparison (n = 51; RMSE = {rain_rmse:.3f} dB)", loc="left")
    date_ticks(ax, rainy, 6)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.5,
              columnspacing=.9)
    save_figure(fig, number="S2", preset="ieee", outdir=str(OUT), kind="line",
                formats=["pdf", "png"])
    plt.close(fig)

    frame.to_csv(OUT / "plot_values.csv", index=False, encoding="utf-8-sig")
    caption_en_1 = (
        "Measured hourly relative clear-sky attenuation and the ITU-R P.838-3 "
        "specific-rain-attenuation plus geometric rain-path estimate at 19.13 GHz, "
        "circular polarization, for 687 quality-qualified test hours. Panel (a) "
        "retains the full data range; panel (b) expands −0.8 to 2.5 dB. Samples "
        "are ordered by time, but unobserved/test-excluded intervals are omitted "
        "from the horizontal axis. The ITU quantity is a rain-only hourly proxy, "
        "not the standard P.618 long-term exceedance prediction."
    )
    caption_zh_1 = (
        "687 个质量合格测试小时的实测相对晴空衰减与 ITU-R P.838-3 比雨衰减"
        "加几何雨区路径估计，频率 19.13 GHz、圆极化。上图保留全部数值范围，"
        "下图只放大 −0.8 至 2.5 dB；横轴按时间排序，但省略了未纳入测试的时段。"
        "ITU 曲线仅是小时雨衰代理，不是 P.618 的长期超越概率预测。"
    )
    caption_en_2 = (
        "Same paired comparison restricted to all 51 quality-qualified rainy test "
        "hours. Each point is one hour; lines only guide visual pairing and do "
        "not imply continuity between rain events or missing hours. The observed "
        "target includes all components of relative clear-sky attenuation, whereas "
        "the ITU estimate represents rain alone. Rain-hour RMSE is 1.330 dB."
    )
    caption_zh_2 = (
        "全部 51 个质量合格雨天测试小时的逐点对照。每点是一个小时；连线仅辅助"
        "比较，不代表不同降雨过程或缺测小时之间连续观测。实测目标包含相对晴空衰减"
        "的全部成分，ITU 只估计雨衰，雨天 RMSE 为 1.330 dB。"
    )
    report_lines = [
        "# ITU 雨衰估计与实测衰减：图像说明 / Figure report", "",
        "- 来源：`final_test.json` 中 `truth_db` 与 `physics_geom_circular.prediction_db`；"
        "未重新训练或重算预测。",
        f"- 源结果 SHA-256：`{sha256(RUN / 'final_test.json')}`。",
        f"- 样本：测试小时 n=687；雨天测试小时 n=51；总体 RMSE={all_rmse:.4f} dB，"
        f"雨天 RMSE={rain_rmse:.4f} dB。", "",
        "## Fig. S1：全部测试小时及典型数值放大", "",
        f"English caption: {caption_en_1}", "",
        f"中文图注：{caption_zh_1}", "",
        "建议引用位置：修订稿实验结果中物理基线与实测对比段落；若仅作讨论图，作为补充图 S1。", "",
        "## Fig. S2：雨天小时", "",
        f"English caption: {caption_en_2}", "",
        f"中文图注：{caption_zh_2}", "",
        "建议引用位置：修订稿雨天分事件误差分析段落；若仅作讨论图，作为补充图 S2。", "",
        "两图均为 IEEE 双栏宽度的草稿；作者在投稿前需最终核查数值、图注和排版。", "",
    ]
    (OUT / "FIGURE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    items = [
        {"kind": "figure", "number": "S1", "title": {"en": "All test hours", "zh": "全部测试小时"},
         "image": "FigS1.png", "width_in": 6.3,
         "claim": {"en": "The rain-only ITU proxy differs substantially from total measured attenuation.",
                   "zh": "只估计雨衰的 ITU 代理与实测总衰减存在明显差别。"},
         "caption": {"en": caption_en_1, "zh": caption_zh_1},
         "annotations": {"en": ["Panel (a): full range; panel (b): −0.8 to 2.5 dB zoom.",
                                "Time gaps omitted; n = 687; no statistical significance test."],
                         "zh": ["(a) 完整范围；(b) −0.8 至 2.5 dB 放大。",
                                "省略时间间隔；n=687；未做显著性检验。"]},
         "citation": {"en": "Suggested: Results, physics-baseline comparison.",
                      "zh": "建议：结果部分物理基线比较段落。"},
         "repro": "plot_itu_vs_observed.py + final_test.json"},
        {"kind": "figure", "number": "S2", "title": {"en": "Rainy test hours", "zh": "雨天测试小时"},
         "image": "FigS2.png", "width_in": 6.3,
         "claim": {"en": "The rain-only proxy often misses measured rain-hour fade depth.",
                   "zh": "纯雨衰代理经常未能捕捉实测雨小时衰减深度。"},
         "caption": {"en": caption_en_2, "zh": caption_zh_2},
         "annotations": {"en": ["n = 51 rainy test hours across 26 calendar days.",
                                "Connected lines do not represent continuous time."],
                         "zh": ["26 个日历日中的 51 个雨天测试小时。",
                                "连线不代表连续时间。"]},
         "citation": {"en": "Suggested: Results, rain-event error analysis.",
                      "zh": "建议：结果部分雨天误差分析段落。"},
         "repro": "plot_itu_vs_observed.py + final_test.json"},
    ]
    (OUT / "figure_items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS", "source_final_test_sha256": sha256(RUN / "final_test.json"),
        "source_qa_final_sha256": sha256(RUN / "qa_final.json"),
        "source_independent_validation_sha256": sha256(RUN / "independent_validation.json"),
        "plot_script_sha256": sha256(HERE / "plot_itu_vs_observed.py"),
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "n_test": len(frame), "n_rain": int(rain.sum()),
        "all_rmse_db": all_rmse, "rain_rmse_db": rain_rmse,
        "artifacts_sha256": {p.name: sha256(p) for p in expected if p.exists()},
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT),
                      "n_test": len(frame), "n_rain": int(rain.sum()),
                      "all_rmse_db": all_rmse, "rain_rmse_db": rain_rmse},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
