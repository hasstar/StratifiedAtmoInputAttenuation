"""Read-only frozen local model API; new example outputs only, zero fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import local_engine as e
import freeze_generator as fg


class FrozenLocalChannel:
    """Original full156 BiLSTM seed17/53 + frozen four-weather 256-second model.

    Weather is a supplied hourly condition, not inferred from the target signal.
    Each call returns a random scenario, not a pointwise deterministic forecast.
    All 156 predictor keys are required; explicit missing values use the original
    training-fitted imputer. No omitted column is substituted by a new RFE model.
    """

    def __init__(self):
        self.manifest=e.manifest(check_sources=True)
        self.slow=e.slow_module()
        self.source=e.read(e.SLOW/"run_manifest.json")
        self.features=tuple(self.source["retained_names"])
        self.bundle=self.slow.v.joblib.load(e.SLOW/"models/full_bilstm_seed17.joblib")
        e.require(tuple(self.bundle["retained_names"])==self.features,"Original input list changed")
        self.package,self.cdfs,pools=fg.load_package()
        e.require(self.package["candidate"]=="block_0256","Frozen dependency changed")
        self.libraries={event:e.dyn.block_library(pool,256) for event,pool in pools.items()}
        e.require(all(self.libraries.values()),"Missing legal training-weather blocks")

    def __call__(self,features,weather,hour_key,repetition=0,candidate="block_0256"):
        e.require(set(features)==set(self.features),"Supply exactly the frozen 156 predictor keys")
        e.require(weather in self.cdfs,"Weather must be clear_dry/rain/dry_overcast/other")
        e.require(candidate in ("iid","block_0256"),"Only frozen IID and block_0256 are permitted")
        e.require(isinstance(repetition,int) and repetition>=0,"Repetition must be a nonnegative integer")
        e.require(isinstance(hour_key,str) and bool(hour_key),"An explicit stable scenario key is required")
        raw=np.full((1,len(self.source["original_names"])),np.nan)
        for name in self.features:
            v=features[name]
            if v is not None:
                v=float(v)
                e.require(np.isnan(v) or np.isfinite(v),"Infinite predictor is not a missing value")
                raw[0,self.source["original_names"].index(name)]=v
        # The inherited float32 network evaluates chunks of at most 512 rows.
        # Repeat one identical input to keep its GEMM kernel in batch mode.
        # This is numerical packaging only, not 512 realizations or new fitting.
        slow_db=float(self.slow.v.predict(self.bundle,np.repeat(raw,512,axis=0))[0])
        e.require(np.isfinite(slow_db),"Frozen slow prediction is not finite")
        q,meta=e.dyn.generate(candidate,self.cdfs[weather],self.libraries[weather],
                              -1,"D-final-"+hour_key,repetition)
        f=-10*np.log10(q)
        attenuation=slow_db+f
        received_dbm=e.original_d1.CLEAR_DBM-attenuation
        probabilities=np.array([.01,.05,.5,.95,.99])
        quantiles=slow_db+e.fm.quantile_f(self.cdfs[weather],probabilities)
        provenance=[]
        if candidate=="block_0256":
            _,provenance=e.dyn.block_template(self.libraries[weather],256,
                         np.random.default_rng(meta["dynamics_seed"]))
            e.require(len(provenance)==15 and sum(x["length_used_s"] for x in provenance)==3600,
                      "Expected 14 full blocks and one 16-second prefix")
        return {"hour_key":hour_key,"weather":weather,"candidate":candidate,"repetition":repetition,
                "slow_attenuation_db":slow_db,"fast_power_ratio":q,"fast_attenuation_db":f,
                "combined_attenuation_db":attenuation,"received_power_dbm":received_dbm,
                "received_power_w":10**(received_dbm/10)/1000,
                "quantile_probabilities":probabilities,"combined_quantiles_db":quantiles,
                "metadata":{**meta,"block_provenance":provenance,
                   "finite_realization_mean_power_ratio":float(q.mean()),
                   "finite_hour_level_drift_db":float(-10*np.log10(q.mean())),
                   "finite_realization_renormalized":False,"model_fit_calls":0,
                   "inference_padding_identical_rows":512,"padding_predictions_not_averaged":True,
                   "fingerprint":self.manifest["fingerprint"],"sample_interval_s":1,
                   "random_scenario_not_pointwise_forecast":True}}


def verify_examples():
    e.require(e.read(e.LOCAL/"report_qa.json")["status"]=="PASS","Completed local QA required")
    predictor=FrozenLocalChannel()
    predictions=e.read(e.SLOW/"run_manifest.json")
    slow=e.slow_module()
    prior=e.old_frames("marginal")
    output=e.LOCAL/"callable_examples"
    output.mkdir(exist_ok=True)
    audit=[]
    for split in ("validation","test"):
        frame=slow.v.data_frame(split,predictions,events=True).set_index("hour_bjt")
        saved=e.read(e.SLOW/"training/full_bilstm_seed17.json" if split=="validation"
                     else e.SLOW/"test/full_bilstm_seed17.json")
        keys=("validation_hour_bjt","validation_prediction_db") if split=="validation" else ("hour_bjt","prediction_db")
        known=dict(zip(saved[keys[0]],saved[keys[1]]))
        for weather in ("clear_dry","rain","dry_overcast","other"):
            hour=prior.loc[prior.split.eq(split)&prior.event.eq(weather)].sort_values("hour").iloc[0]
            features={name:float(frame.loc[hour.hour,name]) for name in predictor.features}
            result=predictor(features,weather,hour.hour)
            e.require(abs(result["slow_attenuation_db"]-known[hour.hour])<1e-10,"Original slow prediction not reproduced")
            old_path=e.D1/"chunks"/f"example_{split}_{weather}.npz"
            with np.load(old_path) as old:
                e.require(np.array_equal(result["fast_power_ratio"],old["block_q"]),"Old example random stream differs")
                iid=predictor(features,weather,hour.hour,candidate="iid")
                e.require(np.array_equal(iid["fast_power_ratio"],old["iid_q"]),"Old IID example stream differs")
                e.require(np.array_equal(np.sort(iid["fast_power_ratio"]),np.sort(result["fast_power_ratio"])),"IID/block margins differ")
                observed_q=old["observed_q"].copy();records=old["records"].copy()
            observed=np.full(3600,np.nan)
            good=(records>=9)&np.isfinite(observed_q)&(observed_q>0)
            observed[good]=hour.actual_hour_attenuation_db-10*np.log10(observed_q[good])
            payload={k:result[k] for k in ("fast_power_ratio","fast_attenuation_db","combined_attenuation_db",
                     "received_power_dbm","received_power_w","quantile_probabilities","combined_quantiles_db")}
            payload.update(observed_attenuation_db=observed,records=records,
                           iid_combined_attenuation_db=iid["combined_attenuation_db"])
            path=output/f"{split}_{weather}.npz"
            if path.exists():
                with np.load(path) as previous:
                    e.require(set(previous.files)==set(payload) and all(np.array_equal(previous[k],v,equal_nan=True)
                              for k,v in payload.items()),"Existing callable example differs")
            else:
                np.savez_compressed(path,**payload)
            audit.append({"split":split,"weather":weather,"hour":hour.hour,"old_example_sha256":e.sha(old_path),
                "new_example_sha256":e.sha(path),"path":str(path.relative_to(e.LOCAL)),
                "slow_attenuation_db":result["slow_attenuation_db"],"actual_hour_attenuation_db":float(hour.actual_hour_attenuation_db),
                "independent_slow_replay_error_db":abs(result["slow_attenuation_db"]-known[hour.hour]),
                "old_fast_stream_exactly_reproduced":True,"source_valid_seconds":int(good.sum()),
                "source_records":int(records[good].sum()),**result["metadata"]})
    qa={"status":"PASS","version":"frozen-local-callable-v1","fingerprint":predictor.manifest["fingerprint"],
        "script_sha256":e.sha(Path(__file__)),"examples":audit,"sample_count_each":3600,
        "local_new_fit_calls":0,"old_files_modified":False,"old_example_outputs_overwritten":False}
    dest=e.LOCAL/"callable_qa.json"
    if dest.exists():
        e.require(e.read(dest)==qa,"Callable QA changed")
    else:
        e.atomic(dest,qa)
    return {"status":"PASS","examples":len(audit),"new_fits":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("verify-examples","scenario"))
    parser.add_argument("--features",type=Path)
    parser.add_argument("--weather",choices=("clear_dry","rain","dry_overcast","other"))
    parser.add_argument("--hour-key")
    parser.add_argument("--repetition",type=int,default=0)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.command=="verify-examples":
        print(json.dumps(verify_examples()),flush=True)
        return
    e.require(args.features and args.weather and args.hour_key and args.output,"Features/weather/hour-key/output are all required")
    dest=args.output.resolve()
    e.require(dest.is_relative_to((e.LOCAL/"scenarios").resolve()) and dest.suffix==".npz","Only new local/scenarios/*.npz outputs permitted")
    e.require(not dest.exists() and not dest.with_suffix(".json").exists(),"Existing scenario will not be overwritten")
    result=FrozenLocalChannel()(e.read(args.features),args.weather,args.hour_key,args.repetition)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    e.atomic(dest.with_suffix(".json"),{k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
    print(json.dumps({"status":"SAVED_RANDOM_SCENARIO","samples":3600,"output":str(dest),"fit_calls":0}),flush=True)


if __name__=="__main__":
    main()
