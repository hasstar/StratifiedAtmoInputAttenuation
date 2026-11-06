"""Plot immutable hourly UK attenuation labels; no model execution or network."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_005"
BASE = ROOT / "Outputs/Experiments/Combined_Channel_Revision_003"
SKILL = Path(r"C:\Users\SDR_HH\.codex\skills\scientific-visualization")
OUT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl_config")
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(SKILL / "scripts"))
from figure_export import export_figure

SOURCE = BASE / "expanded_dataset.parquet"
PRED = BASE / "predictions_expanded.parquet"
STOP = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json"
EXPECTED = {
    SOURCE: "7175ec16e6c6ccb70ef2034f3461c86dbdae819193adc85d68c9c2513abc1de7",
    PRED: "aefadd3c96bd270b9569e58f83f20e90c95f07c8f0f9d6259804bb035d426b89",
    STOP: "0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a",
}
NAMES = {"chilbolton": "Chilbolton", "sparsholt": "Sparsholt"}
COLORS = {"chilbolton": "#0072B2", "sparsholt": "#D55E00"}
START = pd.Timestamp("2003-08-19", tz="UTC")
END = pd.Timestamp("2004-04-01", tz="UTC")
PLOTS = []


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def emit_json(p, data):
    with p.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def absent_date_spans(g, start, end):
    dates = set(g.hour_utc.dt.normalize())
    cursor = None
    for d in pd.date_range(start.normalize(), end.normalize(), freq="D", inclusive="left"):
        if d not in dates and cursor is None:
            cursor = d
        if d in dates and cursor is not None:
            yield cursor, d
            cursor = None
    if cursor is not None:
        yield cursor, end


def series(g, start, end):
    # An explicit regular grid inserts NaNs for gaps; it never fills values.
    grid = pd.date_range(start, end, freq="h", inclusive="left")
    x = g.set_index("hour_utc")
    y = x.truth_db.where(x.primary_score_eligible).reindex(grid)
    expected = int((g.primary_score_eligible & g.hour_utc.ge(start) & g.hour_utc.lt(end)).sum())
    assert int(y.notna().sum()) == expected
    np.testing.assert_array_equal(y.dropna().to_numpy(), x.loc[x.index.isin(grid) & x.primary_score_eligible, "truth_db"].to_numpy())
    return grid, y


def draw(ax, g, sid, start, end, title, ylim, overview=False):
    for a, b in absent_date_spans(g, start, end):
        ax.axvspan(a, b, facecolor="#ECEFF2", zorder=0)
    grid, y = series(g, start, end)
    line, = ax.plot(grid, y.to_numpy(), color=COLORS[sid], lw=0.85, marker=".", ms=1.7, zorder=3)
    low = g[g.truth_db.notna() & ~g.primary_score_eligible & g.hour_utc.ge(start) & g.hour_utc.lt(end)]
    ax.scatter(low.hour_utc, low.truth_db, c="#666666", marker="x", s=19, linewidths=0.9, zorder=4)
    ax.axhline(0, color="#555555", ls="--", lw=0.6, zorder=1)
    ax.set(xlim=(start, end), ylim=ylim, ylabel="小时衰减（dB）")
    ax.set_title(title, loc="left", fontweight="bold", pad=11)
    ax.grid(axis="y", color="#D8DDE3", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    if overview:
        ax.xaxis.set_major_locator(mdates.MonthLocator(tz=timezone.utc))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m", tz=timezone.utc))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2 if (end-start).days > 60 else 1, tz=timezone.utc))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))
    eligible = g[g.primary_score_eligible & g.hour_utc.ge(start) & g.hour_utc.lt(end)]
    if overview and len(eligible):
        peak = eligible.loc[eligible.truth_db.idxmax()]
        ax.annotate(f"{peak.truth_db:.3f} dB\n{peak.hour_utc:%Y-%m-%d %H}:00 UTC",
                    (peak.hour_utc, peak.truth_db), xytext=(10 if sid == "chilbolton" else -12, -15),
                    textcoords="offset points", ha="left" if sid == "chilbolton" else "right", va="top",
                    fontsize=9, color="#202630", bbox={"facecolor":"white", "edgecolor":"none", "pad":2},
                    arrowprops={"arrowstyle":"-", "color":"#596673", "lw":0.7})
    PLOTS.append({"panel":title,"station":sid,"start":start.isoformat(),"end_exclusive":end.isoformat(),
                  "valid_line_points":int(y.notna().sum()),"low_coverage_points":len(low),
                  "regular_grid_hours":len(grid),"nan_line_breaks":int(y.isna().sum())})
    np.testing.assert_array_equal(np.asarray(line.get_ydata()), y.to_numpy())


def save(fig, name, provenance):
    result = export_figure(fig, OUT/name, formats=["png","pdf"], dpi=220,
        transparent=False, bbox_inches=None, font_mode="truetype", write_manifest=True,
        metadata={"Title":"UK hourly attenuation labels"},
        provenance=provenance)
    fig.canvas.draw()
    # Check labels do not leave the figure canvas (not a substitute for visual review).
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    outside = []
    for ax in fig.axes:
        for text in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
            if not text.get_visible() or not text.get_text(): continue
            b=text.get_window_extent(renderer)
            if b.x0 < -2 or b.y0 < -2 or b.x1 > w+2 or b.y1 > h+2: outside.append(text.get_text())
    assert not outside, outside
    plt.close(fig)
    return result


def main():
    if (OUT/"validation.json").exists():
        raise RuntimeError("Output already exists: preserve it and use a new version.")
    for p, sha in EXPECTED.items(): assert digest(p)==sha, str(p)
    d=pd.read_parquet(SOURCE).sort_values(["station","hour_utc"]).reset_index(drop=True)
    p=pd.read_parquet(PRED).sort_values(["station","hour_utc"]).reset_index(drop=True)
    assert len(d)==5664 and not d.duplicated(["station","hour_utc"]).any()
    for col in ["station","hour_utc","truth_db","valid_seconds","coverage_fraction","primary_score_eligible"]:
        pd.testing.assert_series_equal(d[col],p[col])
    assert (d.primary_score_eligible == (np.isfinite(d.truth_db)&d.valid_seconds.ge(3420))).all()
    assert (d.hour_utc.astype("int64")//10**9 == d.utc_epoch_seconds).all()
    assert int(d.primary_score_eligible.sum())==5593
    assert int(d.truth_db.isna().sum())==37
    assert int((d.truth_db.notna()&~d.primary_score_eligible).sum())==34
    assert int((d.primary_score_eligible&d.liquid_negative).sum())==55
    font_path=Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name=font_manager.FontProperties(fname=str(font_path)).get_name()
    source_hashes={str(x.relative_to(ROOT)):sha for x,sha in EXPECTED.items()}
    provenance={"sources_sha256":source_hashes,"code_sha256":digest(Path(__file__)),
        "helper_sha256":digest(SKILL/"scripts/figure_export.py"),
        "scope":"Current completed-weather comparison cohort, not all archived beacon days",
        "quantity":"Native hourly -10 log10(mean linear transmittance), dB",
        "timestamp":"UTC hour start; native integration-end seconds assigned to (h,h+1h]",
        "transformation":"Read saved labels unchanged; insert NaN calendar gaps only",
        "missing":"37 missing labels are gaps; 34 finite low-coverage hours are separate gray x markers",
        "rain_mask":"55 valid-label negative-liquid hours retained; no rain-based filter",
        "publisher":"Not a journal submission figure; no compliance claim"}
    chosen=["station","hour_utc","period","truth_db","valid_seconds","coverage_fraction","primary_score_eligible","liquid_negative","source_file"]
    with (OUT/"plotted_hourly_labels.csv").open("x",encoding="utf-8-sig",newline="") as f:
        d[chosen].to_csv(f,index=False,float_format="%.17g",date_format="%Y-%m-%dT%H:%M:%SZ")
    ymin=float(d.truth_db.min()); ymax=float(d.truth_db.max())
    ylim=(np.floor((ymin-.08)*10)/10,np.ceil((ymax+.12)*10)/10)
    handles=[Line2D([0],[0],color=COLORS["chilbolton"],lw=1.2,label="Chilbolton：有效秒覆盖 ≥95%"),
             Line2D([0],[0],color=COLORS["sparsholt"],lw=1.2,label="Sparsholt：有效秒覆盖 ≥95%"),
             Line2D([0],[0],color="#666666",ls="none",marker="x",label="覆盖 <95%：仅展示，不计主评分"),
             Patch(facecolor="#ECEFF2",label="未纳入当前完整气象对比集的日期")]
    style={"font.family":font_name,"font.size":10,"axes.titlesize":12,"axes.labelsize":11,
           "axes.unicode_minus":False,"figure.facecolor":"white","axes.facecolor":"white",
           "xtick.labelsize":9,"ytick.labelsize":9,"path.simplify":False}
    exports=[]
    with plt.style.context(str(SKILL/"assets/publication.mplstyle")),plt.rc_context(style):
        fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,sharey=True)
        fig.subplots_adjust(left=.075,right=.98,bottom=.16,top=.80,hspace=.30)
        fig.suptitle("英国两站：小时级衰减随时间变化",fontsize=18,x=.075,ha="left",y=.975,fontweight="bold")
        fig.text(.075,.927,"实测信标聚合标签 · 真实日期间隔 · 不平滑、不插值、不裁剪负值",fontsize=11,color="#455463")
        fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.069,.902),ncol=2,frameon=False,fontsize=9)
        for ax,(sid,g) in zip(axes,d.groupby("station",sort=True)):
            valid=int(g.primary_score_eligible.sum()); days=g.hour_utc.dt.normalize().nunique()
            draw(ax,g,sid,START,END,f"{NAMES[sid]}  |  {days} 个纳入日期 · {valid:,} 个合格小时",ylim,True)
        axes[-1].set_xlabel("日期（UTC；每小时一个样本）",labelpad=8)
        fig.text(.075,.065,"彩色曲线 5593 小时；灰叉 34 个低覆盖小时；37 个缺标签小时断线。灰区不等于零衰减，也不表示原始档案不存在。",fontsize=9,color="#455463")
        fig.text(.075,.035,"范围：2003 年扩展日期 + 2004 年原 39 天。55 个因雨量输入而未进入 ITU 共同评分的有效信标小时仍保留在图中。",fontsize=9,color="#455463")
        exports.append(save(fig,"uk_hourly_attenuation_overview",provenance))
        fig,axes=plt.subplots(2,2,figsize=(15,8),sharey=True)
        fig.subplots_adjust(left=.065,right=.985,bottom=.15,top=.79,hspace=.35,wspace=.12)
        fig.suptitle("分时期展开：英国两站小时衰减",fontsize=18,x=.065,ha="left",y=.975,fontweight="bold")
        fig.text(.065,.925,"与全期图完全相同的数据和纵轴；只展开时间轴，没有额外滤波或聚合。",fontsize=11,color="#455463")
        fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.059,.9),ncol=2,frameon=False,fontsize=9)
        periods=[(START,pd.Timestamp("2003-12-01",tz="UTC"),"2003 年扩展日期"),
                 (pd.Timestamp("2004-02-18",tz="UTC"),END,"2004 年原 39 天")]
        for i,(sid,g) in enumerate(d.groupby("station",sort=True)):
            for j,(a,b,label) in enumerate(periods):
                draw(axes[i,j],g,sid,a,b,f"{NAMES[sid]}  |  {label}",ylim)
                if i==1: axes[i,j].set_xlabel("日期（月-日，UTC）")
                if j==1: axes[i,j].set_ylabel("")
        fig.text(.065,.06,"彩色线：有效秒覆盖 ≥95%；灰叉：低覆盖；灰区：当前对比集未纳入的日期。所有负衰减标签原样保留。",fontsize=9,color="#455463")
        fig.text(.065,.03,"这是小时聚合后的信标衰减，不是秒级衰落曲线，也不是模型预测曲线。",fontsize=9,color="#455463")
        exports.append(save(fig,"uk_hourly_attenuation_period_detail",provenance))
    for path,sha in EXPECTED.items(): assert digest(path)==sha
    metadata=[]
    for png in sorted(OUT.glob("*.png")):
        with Image.open(png) as im:
            extrema=im.getchannel("A").getextrema() if im.mode=="RGBA" else None
            assert extrema is None or extrema==(255,255)
            metadata.append({"path":png.name,"pixels":list(im.size),"mode":im.mode,"dpi":im.info.get("dpi"),"opaque":True,"sha256":digest(png)})
    stats=[]
    for sid,g in d.groupby("station"):
        q=g[g.primary_score_eligible]; peak=q.loc[q.truth_db.idxmax()]
        stats.append({"station":sid,"candidate_hours":len(g),"eligible_hours":len(q),"low_coverage_hours":int((g.truth_db.notna()&~g.primary_score_eligible).sum()),
            "missing_label_hours":int(g.truth_db.isna().sum()),"max_eligible_db":float(peak.truth_db),"max_utc":peak.hour_utc.isoformat()})
    emit_json(OUT/"validation.json",{"status":"PASS_NUMERIC_AND_EXPORT_CHECKS","visual_review":"PENDING",
        "created":datetime.now(timezone(timedelta(hours=8))).isoformat(),"source_sha256_unchanged":True,
        "source_labels_equal_frozen_prediction_labels":True,"station_summaries":stats,
        "panels":PLOTS,"png_metadata":metadata,"ylim":list(ylim),"provenance":provenance,
        "versions":{"python":sys.version,"matplotlib":matplotlib.__version__,"pandas":pd.__version__,"numpy":np.__version__},
        "new_training":0,"new_model_inference":0,"downloads":0})
    print(json.dumps({"status":"PASS","outputs":[x["path"] for e in exports for x in e["outputs"]],"station_summaries":stats},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
