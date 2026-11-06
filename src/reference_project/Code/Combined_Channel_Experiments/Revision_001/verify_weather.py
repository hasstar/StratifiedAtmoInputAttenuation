"""Separate read-only ERA5 smoke field/time/unit audit; never submits requests."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile
import numpy as np
import netCDF4
import local_engine as e
import external_prepare as p


def members(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                e.require(not Path(name).is_absolute() and ".." not in Path(name).parts,"Unsafe archive path")
                if name.endswith(".nc"):
                    with netCDF4.Dataset("weather_unit_audit.nc",memory=archive.read(name)) as dataset:
                        yield name,dataset
    else:
        with netCDF4.Dataset(path) as dataset:
            yield path.name,dataset


def smoke():
    download=p.DEST/"weather_smoke_qa.json"
    e.require(download.exists(),"Smoke worker has not completed; do not start full downloads yet")
    e.require(e.read(download)["status"]=="PASS_DOWNLOAD_READABILITY_ONLY","Smoke download gate failed")
    required={"pressure":{"z":{"m**2 s**-2","m2 s-2"},"u":{"m s**-1","m s-1"}},
              "surface":{"cape":{"J kg**-1","J kg-1"},"mx2t":{"K"},
                         "tcc":{"(0 - 1)","1"},"tp":{"m"},"sf":{"m"}}}
    expected=["2004-02-18T00:00:00Z","2004-02-18T01:00:00Z"]
    rows=[]
    for kind,variables in required.items():
        name="SMOKE_GBS_England_2004-02-18_"+kind
        record=e.read(p.DEST/"weather_jobs"/(name+".json"))
        path=p.DEST/record["job"]["target"]
        e.require(record["status"]=="VERIFIED" and e.sha(path)==record["sha256"],"Smoke payload changed")
        present=set();times=set()
        for member,dataset in members(path):
            tv=dataset.variables.get("valid_time",dataset.variables.get("time"))
            e.require(tv is not None and tv.size==2,"Wrong smoke time axis")
            dates=netCDF4.num2date(tv[:],tv.units,calendar=getattr(tv,"calendar","standard"))
            utc=[d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in np.asarray(dates).reshape(-1)]
            e.require(sorted(utc)==expected,"Requested UTC times not returned")
            times.update(utc)
            if kind=="pressure":
                level=dataset.variables["pressure_level"]
                e.require(np.array_equal(np.asarray(level[:]).reshape(-1),[1]),"Wrong pressure level")
                e.require(str(level.units) in ("hPa","millibars"),"Pressure level unit not declared")
            for key,allowed in variables.items():
                if key not in dataset.variables:
                    continue
                variable=dataset.variables[key]
                e.require(str(variable.units) in allowed,"Unknown ERA5 field unit: "+key+" "+str(variable.units))
                data=np.ma.asarray(variable[:])
                e.require(not np.ma.getmaskarray(data).any() and np.isfinite(np.asarray(data)).all(),"Missing/nonfinite smoke field: "+key)
                present.add(key)
                rows.append({"job":name,"member":member,"variable":key,"units":str(variable.units),
                    "shape":list(data.shape),"values":int(data.size),"minimum":float(data.min()),
                    "maximum":float(data.max()),"valid_times":utc,"payload_sha256":record["sha256"]})
        e.require(present==set(variables) and sorted(times)==expected,"Requested weather fields are incomplete")
    qa={"status":"PASS","scope":"two official ERA5 smoke requests only; not the full weather/source model QA",
        "script_sha256":e.sha(Path(__file__)),"download_qa_sha256":e.sha(download),
        "weather_plan_sha256":e.sha(p.DEST/"weather_plan.json"),"checks":rows,
        "official_accumulation_alignment":"tp/sf at t apply to the preceding hour [t-1h,t); no daily differencing for ERA5 hourly",
        "model_fits":0,"full_weather_available":False}
    target=p.DEST/"weather_smoke_unit_qa.json"
    if target.exists():
        e.require(e.read(target)==qa,"Saved unit preflight changed")
    else:
        e.atomic(target,qa)
    return {"status":"PASS","field_checks":len(rows),"full_weather_available":False,"model_fits":0}


if __name__=="__main__":
    print(json.dumps(smoke(),indent=2),flush=True)
