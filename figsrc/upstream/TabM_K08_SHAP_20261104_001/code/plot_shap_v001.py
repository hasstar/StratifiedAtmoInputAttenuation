"""Plot frozen k8 attributions; no fitting or inference. Refuse overwrite."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output-root",type=Path,default=None,help="Empty output directory for a new rendering; data always come from this package.")
args=parser.parse_args()
SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE if args.output_root is None else args.output_root.resolve()
ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/"Picture";OUT.mkdir(exist_ok=True)
assert not (OUT/"Fig_SHAP_TabM_K08.pdf").exists()
qa=json.loads((SOURCE/"qa/independent_qa.json").read_text("utf-8"))
assert qa["all_pass"],"Independent numerical QA must pass before publication plotting."
labels={
"surface__convective_available_potential_energy":"CAPE",
"surface__maximum_2m_temperature_since_previous_post_processing":"Maximum 2-m temperature",
"surface__k_index":"K index",
"surface__vertical_integral_of_eastward_water_vapour_flux":"Column water-vapour flux (E)",
"surface__total_column_ozone":"Total-column ozone",
"p850__ozone_mass_mixing_ratio":"Ozone mixing ratio, 850 hPa *",
"p650__specific_humidity":"Specific humidity, 650 hPa *",
"p300__temperature":"Temperature, 300 hPa *",
"p5__u_component_of_wind":"Zonal wind, 5 hPa *",
"p1__u_component_of_wind":"Zonal wind, 1 hPa *",
"surface__surface_pressure":"Surface pressure",
"surface__mean_large_scale_precipitation_rate":"Large-scale precipitation rate",
"surface__vertical_integral_of_eastward_mass_flux":"Column mass flux (E)",
"surface__vertical_integral_of_eastward_ozone_flux":"Column ozone flux (E)",
"surface__vertical_integral_of_northward_heat_flux":"Column heat flux (N)"
}
datasets={}
for model in ("full","base"):
    identity=json.loads((SOURCE/"results"/f"{model}_identity.json").read_text("utf-8"))
    a=np.load(SOURCE/"results"/f"{model}_shap.npz")
    inputs=pd.read_csv(SOURCE/"results"/f"{model}_test_inputs.csv")
    phi=a["phi"];features=identity["features"];importance=np.mean(abs(phi),axis=0)
    order=np.argsort(-importance,kind="stable")
    datasets[model]=(identity,phi,inputs,importance,order,a["replicates"])
maxi=max(max(v[3]) for v in datasets.values())
limit=max(np.max(abs(v[1])) for v in datasets.values())*1.055
style={"font.family":"Times New Roman","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
       "xtick.labelsize":7,"ytick.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":.6}
rng=np.random.default_rng(20261103)
with mpl.rc_context(style):
    fig=plt.figure(figsize=(7.16,5.5),facecolor="white")
    for row,model in enumerate(("full","base")):
        identity,phi,raw,imp,order,reps=datasets[model]
        y0=.575 if row==0 else .14
        ax=fig.add_axes([.305,y0,.235,.32])
        bx=fig.add_axes([.605,y0,.345,.32])
        names=[identity["features"][j] for j in order]
        colours=["#D55E00" if f.startswith("p") else "#0072B2" for f in names]
        y=np.arange(len(order))
        ax.barh(y,imp[order],height=.62,color=colours,alpha=.9,zorder=3)
        # Thin ranges are permutation-replicate spread, explicitly not statistical CIs.
        replicate_imp=np.mean(abs(reps),axis=1)[:,order]
        ax.hlines(y,replicate_imp.min(0),replicate_imp.max(0),color="#222222",lw=.8,zorder=4)
        ax.set(yticks=y,yticklabels=[labels[f] for f in names],xlim=(0,maxi*1.27),ylim=(7.6,-.6))
        ax.set_xlabel("Mean |SHAP| (dB)")
        for pos,j in enumerate(order):
            ax.text(imp[j]+maxi*.025,pos,f"{imp[j]:.3f}",va="center",fontsize=6.5,color="#333333")
        ax.set_title(f"({'a' if row==0 else 'c'}) {'Full' if row==0 else 'Base'} pool: importance",loc="left",pad=9)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.grid(axis="x",color="#dddddd",lw=.5,zorder=0)
        bx.axvline(0,color="#666666",lw=.7,zorder=1)
        for rank,j in enumerate(order):
            values=raw[identity["features"][j]].to_numpy(float)
            finite=np.isfinite(values);lo,hi=np.nanmin(values),np.nanmax(values)
            c=np.full(len(values),.5) if hi<=lo else np.clip((values-lo)/(hi-lo),0,1)
            yy=rank+rng.uniform(-.26,.26,len(values))
            bx.scatter(phi[finite,j],yy[finite],c=c[finite],cmap="coolwarm",vmin=0,vmax=1,s=4.2,
                       alpha=.60,linewidths=0,rasterized=False,zorder=2)
            if not finite.all(): bx.scatter(phi[~finite,j],yy[~finite],color="#777777",s=4.2,linewidths=0)
        bx.set(xlim=(-limit,limit),ylim=(7.6,-.6),yticks=y,yticklabels=[])
        bx.set_xlabel("SHAP value (dB)")
        bx.set_title(f"({'b' if row==0 else 'd'}) {'Full' if row==0 else 'Base'} pool: signed effects",loc="left",pad=9)
        bx.grid(axis="x",color="#dddddd",lw=.5)
        for a in (ax,bx):
            a.spines["top"].set_visible(False);a.spines["right"].set_visible(False)
            a.tick_params(axis="y",length=0)
    handles=[Line2D([],[],color="#0072B2",lw=5,label="Base input"),
             Line2D([],[],color="#D55E00",lw=5,label="Pressure-level input (*)")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.64,.975),ncol=2,frameon=False,fontsize=8)
    cax=fig.add_axes([.655,.045,.24,.015])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap="coolwarm"),cax=cax,orientation="horizontal")
    cb.set_ticks([0,1]);cb.set_ticklabels(["Low","High"]);cb.ax.tick_params(length=0,labelsize=7)
    cb.set_label("Feature value (scaled separately per feature)",fontsize=7,labelpad=1)
    fig.savefig(OUT/"Fig_SHAP_TabM_K08.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(OUT/"Fig_SHAP_TabM_K08.png",dpi=450,facecolor="white")
    plt.close(fig)

rows=[]
for model,(ident,phi,raw,imp,order,reps) in datasets.items():
    for rank,j in enumerate(order,1):
        rows.append({"model":model,"rank":rank,"feature":ident["features"][j],"label":labels[ident["features"][j]],
                     "mean_abs_shap_db":float(imp[j]),"mean_shap_db":float(np.mean(phi[:,j])),
                     "minimum_shap_db":float(phi[:,j].min()),"maximum_shap_db":float(phi[:,j].max()),
                     "mean_abs_replicate0_db":float(np.mean(abs(reps[0,:,j]))),
                     "mean_abs_replicate1_db":float(np.mean(abs(reps[1,:,j])))})
pd.DataFrame(rows).to_csv(ROOT/"SHAP_Ranking_And_Values.csv",index=False)
manifest={"script":str(Path(__file__).resolve()),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "source":"results/*_shap.npz","n_test_hours":687,"point_jitter_seed":20261103,
          "axes":"Shared linear limits across models; no attribution clipped.",
          "colour":"Within-feature min/max scaling; missing raw values gray.",
          "bars":"Mean absolute SHAP over all 687 test hours, no filtering.",
          "thin_ranges":"Range of two Monte Carlo permutation-replicate global estimates, not confidence intervals.",
          "dimensions_inches":[7.16,5.5],"source_package":str(SOURCE),
          "source_sha256":{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/"manifest.json",SOURCE/"qa/independent_qa.json",*sorted((SOURCE/"results").glob("*_shap.npz")),*sorted((SOURCE/"results").glob("*_test_inputs.csv"))]},"figure_sha256":hashlib.sha256((OUT/"Fig_SHAP_TabM_K08.pdf").read_bytes()).hexdigest()}
(ROOT/"figure_provenance.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(pd.DataFrame(rows)[["model","rank","feature","mean_abs_shap_db"]].to_string(index=False))
