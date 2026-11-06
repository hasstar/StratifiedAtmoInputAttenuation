"""Build descriptive summary tables and truthful plots from audited exports only."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/"Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002"
LABELS = dict(tabm="TabM",bai_lstm="Bai LSTM",bilstm="BiLSTM",transformer="Transformer",mlp="MLP",
              catboost="CatBoost",gcn="GCN",random_forest="Random Forest",svr_rbf="SVR-RBF",elastic_net="Elastic Net",
              combined_ITU_approx="Combined ITU approx.")
MODELS = list(LABELS)[:-1]
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"


def write(frame, name):
    path = OUT/"tables"/name
    assert not path.exists(), str(path)
    frame.to_csv(path,index=False,encoding="utf-8-sig",mode="x")


def format_axis(ax):
    sns.despine(ax=ax,left=True,bottom=True)
    ax.tick_params(axis="both",length=0,colors="dimgrey")
    ax.xaxis.label.set_color("dimgrey")
    ax.yaxis.label.set_color("dimgrey")


def save(fig, name):
    fig.savefig(OUT/"figures"/(name+".png"),dpi=150,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/"figures"/(name+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    qa=json.loads((OUT/"evidence/INDEPENDENT_AGGREGATE_QA.json").read_text(encoding="utf-8"))
    assert qa["status"]=="PASS" and qa["states"]==1560
    (OUT/"figures").mkdir(exist_ok=False)
    d=pd.read_csv(OUT/"tables/all_test_metrics_15605_rows.csv")
    s=pd.read_csv(OUT/"tables/all_states_1560.csv")
    coverage=pd.read_csv(OUT/"tables/weather_sample_coverage.csv")
    cv=s.sort_values(["model","cv_mean_rmse_db","k"]).drop_duplicates("model")
    chosen=dict(zip(cv.model,cv.k.astype(int)))
    exploratory=d.query("population=='ML_687' and weather=='overall'").sort_values(["model","rmse_db","k"]).drop_duplicates("model")
    test_chosen=dict(zip(exploratory.model,exploratory.k.astype(int)))
    points=[]
    for model in MODELS:
        for basis,k in (("full_156",156),("training_cv_min_descriptive",chosen[model]),("test_min_exploratory",test_chosen[model])):
            sr=s.query("model==@model and k==@k").iloc[0]
            for pop in ("ML_687","COMMON_684"):
                rows=d[(d.model==model)&d.k.eq(k)&d.population.eq(pop)]
                for _,row in rows.iterrows():
                    points.append(dict(basis=basis,cv_mean_rmse_db=float(sr.cv_mean_rmse_db),cv_se_rmse_db=float(sr.cv_se_rmse_db),**row.to_dict()))
    points=pd.DataFrame(points)
    write(points,"representative_points_300_rows.csv")
    write(points.query("basis=='full_156' and population=='ML_687'").sort_values(["weather","rmse_db"]), "full156_main687_weather.csv")
    write(points.query("basis=='training_cv_min_descriptive' and population=='ML_687'").sort_values(["weather","rmse_db"]), "cv_min_descriptive_main687_weather.csv")
    write(points.query("basis=='test_min_exploratory' and population=='ML_687'").sort_values(["weather","rmse_db"]), "test_min_EXPLORATORY_main687_weather.csv")
    fullcommon=points.query("basis=='full_156' and population=='COMMON_684'")
    itu=d.query("model=='combined_ITU_approx'")
    write(pd.concat([fullcommon,itu],ignore_index=True),"full156_common684_with_ITU_weather.csv")
    sns.set_theme(style="whitegrid",font="DejaVu Sans",font_scale=1.0)
    plt.rcParams.update({"grid.color":"#e0e0e0","grid.linewidth":0.6,"axes.labelcolor":"dimgrey",
                         "text.color":"dimgrey","legend.facecolor":"white","legend.edgecolor":"lightgrey",
                         "legend.framealpha":0.8,"pdf.fonttype":42})
    overall=d.query("population=='ML_687' and weather=='overall'")
    for metric,title,ylabel,name in (
        ("rmse_db","Independent 156-to-1 RFE: test RMSE","RMSE (dB)","F01_all_RFE_test_RMSE"),
        ("r2","Independent 156-to-1 RFE: test coefficient of determination","R² = 1 - SSE/SST","F02_all_RFE_test_R2")):
        fig,axs=plt.subplots(5,2,figsize=(12.4,14.5),sharex=True,sharey=True,
                             gridspec_kw={"hspace":0.40,"wspace":0.12})
        for i,(ax,model) in enumerate(zip(axs.flat,MODELS)):
            ds=overall[overall.model.eq(model)].sort_values("k",ascending=False)
            ax.plot(ds.k,ds[metric],color=BLUE,lw=1.35)
            for kval,marker,color in ((156,"D",ORANGE),(chosen[model],"*",GREEN)):
                pr=ds[ds.k.eq(kval)].iloc[0]
                ax.scatter([kval],[pr[metric]],marker=marker,s=80 if marker=="*" else 28,
                           color=color,edgecolor="white",linewidth=.6,zorder=5)
            ax.set_title(f"{chr(97+i)}  {LABELS[model]}  |  CV-min k={chosen[model]}",loc="left",fontsize=10.5)
            ax.set_xlim(159,-2)
            ax.set_xticks([156,125,100,75,50,25,1])
            if metric=="r2":
                ax.axhline(0,color="#999999",lw=.7,ls="--")
            format_axis(ax)
        if metric=="rmse_db":
            axs[0,0].set_ylim(0,max(overall.rmse_db)*1.05)
        else:
            axs[0,0].set_ylim(min(0,overall.r2.min())-.05,1.02)
        fig.suptitle(title+"\nAll 156 states per model; identical official 687 test hours",fontsize=15,y=.99)
        fig.supxlabel("Retained feature count k (deletion proceeds left to right)",y=.055,fontsize=11)
        fig.supylabel(ylabel,x=.035,fontsize=11)
        handles=[Line2D([0],[0],color=BLUE,label="Complete test curve"),
                 Line2D([0],[0],marker="D",color="none",markerfacecolor=ORANGE,label="Full 156"),
                 Line2D([0],[0],marker="*",color="none",markerfacecolor=GREEN,markersize=11,label="Train-CV-min descriptive point")]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,frameon=True)
        fig.subplots_adjust(top=.93,bottom=.105,left=.095,right=.985)
        save(fig,name)
    # No test-based k selection in this figure.
    weather_list=["clear_dry","rain","dry_overcast","other"]
    weather_names={"clear_dry":"Clear / dry","rain":"Rain","dry_overcast":"Dry overcast","other":"Other"}
    fig,axs=plt.subplots(2,2,figsize=(13.5,10.5),sharex=True,sharey=True,
                         gridspec_kw={"wspace":.12,"hspace":.20})
    y=np.arange(10)
    selected=points[points.basis.isin(["full_156","training_cv_min_descriptive"])&points.population.eq("ML_687")]
    xmax=selected.rmse_db.max()*1.08
    for i,(ax,w) in enumerate(zip(axs.flat,weather_list)):
        a=selected[(selected.basis=="full_156")&(selected.weather==w)].set_index("model").loc[MODELS]
        b=selected[(selected.basis=="training_cv_min_descriptive")&(selected.weather==w)].set_index("model").loc[MODELS]
        for j in y:
            ax.plot([a.iloc[j].rmse_db,b.iloc[j].rmse_db],[j,j],color="#b5b5b5",lw=1.1,zorder=1)
        ax.scatter(a.rmse_db,y,color=BLUE,marker="o",s=40,label="Full 156",zorder=3)
        ax.scatter(b.rmse_db,y,color=GREEN,marker="D",s=32,label="Train-CV-min",zorder=4)
        ax.set_yticks(y,[LABELS[m] for m in MODELS]); ax.set_xlim(0,xmax)
        ax.set_title(f"{chr(97+i)}  {weather_names[w]} | n={int(a.iloc[0]['n'])}",loc="left")
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
        format_axis(ax)
    fig.suptitle("Weather-specific test RMSE: full input and train-CV-min descriptive points\nOfficial 687-hour population; every model shown",fontsize=15,y=.98)
    fig.supxlabel("RMSE (dB); lower is better",y=.065)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.015),ncol=2)
    fig.subplots_adjust(top=.88,bottom=.13,left=.13,right=.98)
    save(fig,"F03_weather_main687")
    # Fixed full156 common sample comparison, no mixing with 687.
    common=pd.concat([fullcommon.query("weather=='overall'"),itu.query("weather=='overall'")],ignore_index=True)
    order=common.sort_values("rmse_db").model.tolist()
    write(common.sort_values("rmse_db"),"F04_plot_data_common684_full156.csv")
    fig,axs=plt.subplots(1,3,figsize=(14,6.5),sharey=True,gridspec_kw={"wspace":.15})
    for ax,metric,label in zip(axs,["rmse_db","r2","pearson_r"],["RMSE (dB) ↓","R² ↑","Pearson r ↑"]):
        ds=common.set_index("model").loc[order]
        for i,m in enumerate(order):
            ax.scatter(ds.loc[m,metric],i,s=70 if m=="combined_ITU_approx" else 45,
                       marker="X" if m=="combined_ITU_approx" else "o",
                       color="#333333" if m=="combined_ITU_approx" else BLUE)
        ax.set_yticks(range(11),[LABELS[m] for m in order])
        ax.set_xlabel(label)
        if metric!="rmse_db": ax.set_xlim(min(-.02,ds[metric].min()-.05),1.02)
        else: ax.set_xlim(0,ds[metric].max()*1.1)
        format_axis(ax)
    axs[0].invert_yaxis()
    fig.suptitle("Common 684-hour supplement: all full-156 models and fixed ITU approximation\nExcluded negative-liquid-input hours remain in the 687-hour main ML results",fontsize=14,y=.98)
    fig.subplots_adjust(top=.82,bottom=.14,left=.17,right=.98)
    save(fig,"F04_common684_full156_ITU")
    manifest=dict(status="GENERATED_PENDING_VISUAL_REVIEW",figures=[
        dict(id="F01",name="F01_all_RFE_test_RMSE",source="all_test_metrics_15605_rows.csv + all_states_1560.csv",
             alt="Ten small multiples show every test RMSE along independently determined RFE paths; full156 and training-CV-min points identified."),
        dict(id="F02",name="F02_all_RFE_test_R2",source="all_test_metrics_15605_rows.csv + all_states_1560.csv",
             alt="All ten test R-squared curves use a shared axis including negative values; R-squared is not squared correlation."),
        dict(id="F03",name="F03_weather_main687",source="representative_points_300_rows.csv",
             alt="Four weather panels compare full156 with train-CV-min descriptive points for all ten models, with weather counts."),
        dict(id="F04",name="F04_common684_full156_ITU",source="F04_plot_data_common684_full156.csv",
             alt="Common684 full156 model and ITU comparison, sorted by RMSE, with separate RMSE, R-squared and correlation panels.")
    ],test_best_points_not_plotted=True,training_calls=0,negative_r2_not_clipped=True)
    (OUT/"figures/figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(points.query("weather=='overall' and population=='ML_687'")[["basis","model","k","rmse_db","r2"]].to_string(index=False))
    print("FIGURES_GENERATED_PENDING_VISUAL_REVIEW")


if __name__=="__main__":
    main()
