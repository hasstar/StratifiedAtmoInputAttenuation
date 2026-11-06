"""Final local artifact integrity and algebra audit; no model fit/old writes."""
from pathlib import Path
import json
import numpy as np
import local_engine as e


def audit():
    m=e.manifest(check_sources=True)
    report=e.read(e.LOCAL/"report_qa.json")
    e.require(report["status"]=="PASS" and e.sha(e.LOCAL/"RESULTS_ZH.md")==report["report_sha256"],"Report gate/hash differs")
    manifest=e.read(e.LOCAL/"report_manifest.json")
    for name,digest in manifest["output_sha256"].items():
        e.require(e.sha(e.LOCAL/name)==digest,"Saved independent report table changed")
    callable_qa=e.read(e.LOCAL/"callable_qa.json")
    e.require(callable_qa["status"]=="PASS" and callable_qa["local_new_fit_calls"]==0,"Callable QA gate")
    e.require(e.sha(e.HERE/"callable_local.py")==callable_qa["script_sha256"],"Callable source changed")
    package=e.read(e.FAST/"final_generator/model.json")
    moment={}
    for event,cdf in e.load_cdfs().items():
        moment[event]=float(np.diff(np.r_[0,cdf["support_cdf"]])@cdf["support_q"])
        e.require(abs(moment[event]-1)<1e-10,"Population unit power differs")
    pieces_checked=0;largest=0.;libraries={}
    try:
        for row in callable_qa["examples"]:
            path=e.LOCAL/row["path"]
            e.require(e.sha(path)==row["new_example_sha256"],"Callable example changed")
            with np.load(path) as data:
                q=data["fast_power_ratio"];f=-10*np.log10(q)
                tests=[np.max(np.abs(data["fast_attenuation_db"]-f)),
                       np.max(np.abs(data["combined_attenuation_db"]-row["slow_attenuation_db"]-f)),
                       np.max(np.abs(data["received_power_dbm"]+data["combined_attenuation_db"]-e.original_d1.CLEAR_DBM)),
                       abs(float(q.mean())-row["finite_realization_mean_power_ratio"])]
                largest=max(largest,*map(float,tests));e.require(max(tests)<1e-12,"Combined physical identity failed")
                e.require(np.allclose(data["received_power_w"],10**(data["received_power_dbm"]/10)/1000,rtol=1e-13,atol=0),"dBm to W differs")
                e.require(row["source_valid_seconds"]==int(np.isfinite(data["observed_attenuation_db"]).sum()),"Observed gaps changed")
            weather=row["weather"]
            if weather not in libraries:
                libraries[weather]=np.load(e.FAST/"final_generator"/package["components"][weather]["library_file"])
            bank=libraries[weather]
            lookup={str(h):i for i,h in enumerate(bank["hours"])}
            pieces=row["block_provenance"]
            e.require([x["length_used_s"] for x in pieces]==[256]*14+[16],"Wrong block assembly")
            for piece in pieces:
                e.require(piece["hour"] in lookup,"Selected block not in matching training weather")
                i=lookup[piece["hour"]];start=piece["start_s"]
                q=bank["q"][i,start:start+256];count=bank["records"][i,start:start+256]
                e.require(len(q)==256 and np.isfinite(q).all() and (q>0).all() and (count>=9).all(),"Source block crossed hour/gap/quality gate")
                pieces_checked+=1
    finally:
        for bank in libraries.values():bank.close()
    figure_manifest=e.read(e.LOCAL/"figures/manifest.json")
    visual=e.read(e.LOCAL/"figures/visual_qa.json")
    e.require(visual["status"]=="PASS" and len(visual["reviewed_files"])==4,"Visual review required")
    for name,digest in figure_manifest["source_sha256"].items():
        e.require(e.sha(e.ROOT/name)==digest,"Figure source changed")
    for name,digest in figure_manifest["output_sha256"].items():
        e.require(e.sha(e.LOCAL/"figures"/name)==digest,"Rendered figure changed")
    obj={"status":"PASS","scope":"LOCAL delivery only; no external completion","fingerprint":m["fingerprint"],
        "script_sha256":e.sha(Path(__file__)),"report_qa_sha256":e.sha(e.LOCAL/"report_qa.json"),
        "callable_qa_sha256":e.sha(e.LOCAL/"callable_qa.json"),"figure_manifest_sha256":e.sha(e.LOCAL/"figures/manifest.json"),
        "visual_qa_sha256":e.sha(e.LOCAL/"figures/visual_qa.json"),"source_files_verified":len(m["source_sha256"]),
        "callable_examples":8,"training_source_blocks_independently_checked":pieces_checked,
        "max_combined_identity_error_db":largest,"population_power_moments":moment,
        "png_pdf_pairs":4,"new_model_fits":0,"local_complete":True,"GBS_complete":False,"OpenSat_complete":False}
    target=e.LOCAL/"delivery_qa.json"
    if target.exists():e.require(e.read(target)==obj,"Local delivery QA changed")
    else:e.atomic(target,obj)
    return obj


if __name__=="__main__":
    print(json.dumps(audit(),indent=2),flush=True)
