# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "seaborn", "numpy", "pandas", "pyarrow"]
# ///
"""Plot saved, independently verified local results; no refitting or rescoring."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import local_engine as e


def main():
    """Render four views of frozen local channel results.

    Saves
    -----
    PNG and vector PDF figures, plus source and figure hashes, under the new
    local/figures directory. Old tables and original example files are read-only.
    """
    report=e.read(e.LOCAL/"report_manifest.json")
    e.require(e.read(e.LOCAL/"report_qa.json")["status"]=="PASS","Independent report QA required")
    callable_qa=e.read(e.LOCAL/"callable_qa.json")
    e.require(callable_qa["status"]=="PASS","Callable examples QA required")
    source_paths=[e.LOCAL/n for n in report["output_sha256"]]
    source_paths += [e.LOCAL/"report_qa.json",e.LOCAL/"callable_qa.json",Path(__file__)]
    source_paths += [e.LOCAL/x["path"] for x in callable_qa["examples"]]
    for name,digest in report["output_sha256"].items():
        e.require(e.sha(e.LOCAL/name)==digest,"Saved report table changed")
    for row in callable_qa["examples"]:
        e.require(e.sha(e.LOCAL/row["path"])==row["new_example_sha256"],"Saved fixed example changed")
    source_hash={str(p.relative_to(e.ROOT)):e.sha(p) for p in source_paths}
    destination=e.LOCAL/"figures"
    destination.mkdir(exist_ok=True)
    marker=destination/"manifest.json"
    if marker.exists():
        saved=e.read(marker)
        e.require(saved["source_sha256"]==source_hash,"Figure sources/script changed")
        for name,digest in saved["output_sha256"].items():
            e.require(e.sha(destination/name)==digest,"Existing figure changed")
        print("SKIP unchanged figure render",flush=True)
        return

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0,style="whitegrid",font="DejaVu Sans")
    legend={"frameon":True,"facecolor":"white","framealpha":.8,
            "edgecolor":"lightgrey","labelcolor":"dimgrey","fontsize":8}
    palette=sns.cubehelix_palette(6,rot=-.25,light=.7)
    names={"bilstm":"BiLSTM","transformer":"Transformer","mlp":"MLP","svr_rbf":"SVR-RBF",
           "xgboost":"XGBoost","catboost":"CatBoost","gcn":"GCN","lightgbm":"LightGBM",
           "random_forest":"Random forest","elastic_net":"Elastic Net"}
    events=("overall","clear_dry","rain","dry_overcast","other")
    labels=("Overall","Clear","All rain","Dry overcast","Other")

    # --- Data ---
    slow=pd.read_parquet(e.LOCAL/"slow_hour_metrics.parquet").dropna(subset=["rmse_db","mae_db","bias_db"])
    summary=pd.read_parquet(e.LOCAL/"combined_summary.parquet").dropna(subset=["date_mean"])
    calibration=pd.read_parquet(e.LOCAL/"cdf_calibration.parquet").dropna(subset=["probability","date_actual_cdf"])
    holdout=summary.loc[summary.split.eq("test")&summary.block_days.eq(1)]
    main=holdout.loc[holdout.variant.eq("full:bilstm")]
    example_rows=[x for x in callable_qa["examples"] if x["split"]=="test"]
    figures=[]

    # --- Plot: hourly point forecast vs instantaneous probability score ---
    full=slow.loc[slow.split.eq("test")&slow.event.eq("overall")&slow.variant.str.startswith("full:")].sort_values("rmse_db")
    variants=full.variant.tolist()
    model_labels=[names[v.split(":")[1]] for v in variants]
    values=holdout.loc[holdout.event.eq("overall")&holdout.quantity.eq("crps201_db")].set_index("variant")
    fig=plt.figure(figsize=(10,6),dpi=150)
    gs=gridspec.GridSpec(1,2,left=.18,right=.98,top=.90,bottom=.13,wspace=.12)
    axes=[fig.add_subplot(gs[0,i]) for i in range(2)]
    for i,(ax,x,title) in enumerate(zip(axes,[full.rmse_db.to_numpy(),values.loc[variants,"date_mean"].to_numpy()],
                ["Hourly slow-model RMSE","Instantaneous combined CRPS"])):
        ax.barh(np.arange(10),x,color=[palette[5] if v=="full:bilstm" else "lightgrey" for v in variants])
        ax.set_yticks(np.arange(10),model_labels if i==0 else [""]*10)
        ax.invert_yaxis();ax.set_xlim(0,max(x)*1.22)
        ax.set_xlabel("dB; lower is better",color="dimgrey")
        ax.set_title(r"$\bf{("+chr(ord('a')+i)+r")}$ "+title,loc="left",fontsize=11)
        for j,n in enumerate(x):
            ax.text(n+max(x)*.02,j,f"{n:.4f}",va="center",fontsize=8,color="dimgrey")
        ax.axvline(0,color="lightgrey",lw=.8);ax.grid(False)
        ax.tick_params(length=0,labelcolor="dimgrey")
        ax.patch.set_edgecolor("lightgrey");ax.patch.set_linewidth(.8)
    sns.despine(left=True,bottom=True)
    fig.suptitle("Local test: ten unchanged seed-17 full-input models",color="dimgrey",fontsize=13)
    fig.text(.18,.025,"Same 687 hours; RMSE is hour-weighted, CRPS is date-weighted. These are different targets.",
             color="dimgrey",fontsize=8)
    figures.append(("01_model_comparison",fig))

    # --- Plot: main-model weather-specific score and probability calibration ---
    fig=plt.figure(figsize=(11,8),dpi=150)
    gs=gridspec.GridSpec(2,2,left=.08,right=.98,top=.92,bottom=.13,wspace=.12,hspace=.35)
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    positions=np.arange(5)
    get=lambda name:main.loc[main.quantity.eq(name)].set_index("event").loc[list(events),"date_mean"].to_numpy()
    deterministic=get("slow_only_instant_mae_crps_db");combined=get("crps201_db")
    ax=axes[0]
    ax.bar(positions-.18,deterministic,.36,color="lightgrey",label="Slow only: deterministic")
    ax.bar(positions+.18,combined,.36,color=palette[5],label="Slow + frozen fast CDF")
    ax.legend(**legend);ax.set_ylabel("Instantaneous CRPS (dB)",color="dimgrey")
    ax.set_ylim(0,max(deterministic)*1.35)
    for j,n in enumerate(combined):ax.text(j+.18,n+.004,f"{n:.3f}",ha="center",fontsize=8,color="dimgrey")
    for ax,quantity,target in zip(axes[1:3],["central90_coverage","central98_coverage"],[.90,.98]):
        y=get(quantity)
        ax.bar(positions,y,color=palette[5])
        ax.axhline(target,ls="--",color="dimgrey",lw=1,label=f"Nominal {target:.0%}")
        ax.set_ylim(0,1.13);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        for j,n in enumerate(y):ax.text(j,n+.015,f"{n:.1%}",ha="center",fontsize=8,color="dimgrey")
        ax.legend(**legend)
    axes[2].set_ylabel("Observed interval coverage",color="dimgrey")
    ax=axes[3]
    for quantity,target,marker_style,color in (("tail95_exceedance",.05,"o",palette[3]),("tail99_exceedance",.01,"s",palette[5])):
        ax.plot(positions,get(quantity),marker=marker_style,color=color,label=f"Above {1-target:.0%} quantile")
        ax.axhline(target,ls="--" if target==.05 else ":",color=color,lw=.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.set_ylim(bottom=0);ax.legend(**legend)
    titles=("CRPS compares probabilities with measured seconds","Central 90% interval","Central 98% interval","Upper-tail calibration; target 5% / 1%")
    for i,ax in enumerate(axes):
        ax.set_xticks(positions,labels);ax.tick_params(length=0,labelcolor="dimgrey",labelsize=8)
        ax.set_title(r"$\bf{("+chr(ord('a')+i)+r")}$ "+titles[i],loc="left",fontsize=10)
        ax.grid(False);ax.patch.set_edgecolor("lightgrey");ax.patch.set_linewidth(.8)
    sns.despine(left=True,bottom=True)
    improvement=100*(deterministic[0]-combined[0])/deterministic[0]
    fig.suptitle("Main BiLSTM combination: test results by weather",color="dimgrey",fontsize=13)
    fig.text(.08,.035,f"Overall CRPS improves {improvement:.2f}%; 90% coverage is {get('central90_coverage')[0]:.2%}, below nominal.",color="dimgrey",fontsize=9)
    figures.append(("02_weather_probability",fig))

    # --- Plot: all weather calibration curves, keeping other-weather results ---
    fig=plt.figure(figsize=(11,10),dpi=150)
    gs=gridspec.GridSpec(3,2,left=.08,right=.98,top=.93,bottom=.09,wspace=.12,hspace=.35)
    axes=[fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
    for i,(event,label) in enumerate(zip(events,labels)):
        ax=axes[i]
        ax.plot([0,1],[0,1],ls="--",color="grey",lw=1,label="Perfect calibration")
        for variant,style,color,display in (("full:bilstm","-",palette[5],"Full BiLSTM"),
                     ("full:transformer","--",palette[3],"Full Transformer"),("base:bilstm",":","dimgrey","Original base BiLSTM")):
            part=calibration.loc[calibration.split.eq("test")&calibration.event.eq(event)&calibration.variant.eq(variant)].sort_values("probability")
            e.require(len(part)==201,"Missing saved calibration points")
            ax.plot(part.probability,part.date_actual_cdf,style,color=color,label=display,lw=1.4)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(False);ax.tick_params(length=0,labelcolor="dimgrey")
        ax.set_title(r"$\bf{("+chr(ord('a')+i)+r")}$ "+label,loc="left",fontsize=11)
        if i%2==0:ax.set_ylabel("Observed probability",color="dimgrey")
        else:ax.tick_params(labelleft=False)
        if i>=3:ax.set_xlabel("Forecast quantile probability",color="dimgrey")
        ax.patch.set_edgecolor("lightgrey");ax.patch.set_linewidth(.8)
        if i==0:ax.legend(**legend)
    axes[5].set_visible(False)
    sns.despine(left=True,bottom=True)
    fig.suptitle("Test probability calibration: saved 201-point diagnostics",color="dimgrey",fontsize=13)
    fig.text(.08,.025,"Curves on the diagonal are calibrated; a lower CRPS alone does not establish calibration.",color="dimgrey",fontsize=9)
    figures.append(("03_cdf_calibration",fig))

    # --- Plot: fixed earliest test hour in each weather, realization zero ---
    arrays=[]
    for weather in events[1:]:
        row=next(r for r in example_rows if r["weather"]==weather)
        with np.load(e.LOCAL/row["path"]) as data:arrays.append((row,{k:data[k].copy() for k in data.files}))
    y=np.concatenate([np.r_[d["observed_attenuation_db"],d["combined_attenuation_db"],d["combined_quantiles_db"]] for _,d in arrays])
    y=y[np.isfinite(y)];padding=(y.max()-y.min())*.06
    fig=plt.figure(figsize=(12,8),dpi=150)
    gs=gridspec.GridSpec(2,2,left=.08,right=.98,top=.92,bottom=.13,wspace=.10,hspace=.35)
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    for i,(ax,(row,data)) in enumerate(zip(axes,arrays)):
        t=np.arange(3600)/60
        ax.fill_between(t,data["combined_quantiles_db"][1],data["combined_quantiles_db"][3],
                        color=palette[2],alpha=.18,label="Central 90% distribution interval")
        ax.plot(t,data["combined_attenuation_db"],color=palette[3],lw=.55,alpha=.75,label="One random combined scenario")
        ax.plot(t,data["observed_attenuation_db"],color="black",lw=.55,label="Measured attenuation")
        ax.axhline(row["slow_attenuation_db"],color=palette[5],ls="--",lw=1,label="Frozen slow prediction")
        ax.set_ylim(y.min()-padding,y.max()+padding);ax.set_xlim(0,60)
        ax.set_title(r"$\bf{("+chr(ord('a')+i)+r")}$ "+labels[i+1]+"  "+row["hour"][:16],loc="left",fontsize=10)
        if i%2==0:ax.set_ylabel("Attenuation (dB)",color="dimgrey")
        else:ax.tick_params(labelleft=False)
        if i>=2:ax.set_xlabel("Minutes within the hour",color="dimgrey")
        ax.grid(False);ax.tick_params(length=0,labelcolor="dimgrey")
        ax.patch.set_edgecolor("lightgrey");ax.patch.set_linewidth(.8)
    axes[0].legend(**legend,ncol=1)
    sns.despine(left=True,bottom=True)
    fig.suptitle("Earliest test hour per weather; no selection of favorable random paths",color="dimgrey",fontsize=13)
    fig.text(.08,.035,"One scenario is not a pointwise forecast. Actual missing seconds remain gaps; no finite-hour renormalization.",color="dimgrey",fontsize=9)
    figures.append(("04_fixed_random_scenarios",fig))

    # --- Save ---
    outputs={}
    for name,fig in figures:
        for extension in ("pdf","png"):
            target=destination/(name+"."+extension)
            e.require(not target.exists(),"Never overwrite an existing figure")
            fig.savefig(target,dpi=150,bbox_inches="tight")
            outputs[target.name]=e.sha(target)
        plt.close(fig)
    e.atomic(marker,{"status":"AWAITING_VISUAL_QA","fingerprint":report["fingerprint"],
        "source_sha256":source_hash,"output_sha256":outputs,"figure_count":4,
        "new_fits":0,"new_scores":0,"fixed_hours":[{k:r[k] for k in ("split","weather","hour")} for r in example_rows]})
    print("Rendered four new PNG/PDF figures; visual QA remains",flush=True)


if __name__=="__main__":
    main()
