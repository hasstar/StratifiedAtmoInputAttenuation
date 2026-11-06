"""Metadata-only external planning. Does not fit or inspect test errors."""
from __future__ import annotations
import argparse
import calendar
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4
import local_engine as e

DEST = e.OUT / "external"
GBS = e.ROOT / "外部样例数据CEDA/GBS_20p7GHz"
OPEN = e.ROOT / "Outputs/Datasets/External_OpenSat4Weather_001"
RFE = e.ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/rfe_survival_rank.csv"
PRESSURE = ["1", "5", "500", "650", "850"]
PVARS = ["geopotential", "u_component_of_wind", "ozone_mass_mixing_ratio", "relative_humidity"]
SVARS = ["convective_available_potential_energy", "maximum_2m_temperature_since_previous_post_processing",
         "total_cloud_cover", "total_precipitation", "snowfall"]


def value(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8",errors="replace")
    if isinstance(obj,np.generic):
        return value(obj.item())
    if isinstance(obj,np.ndarray):
        return [value(x) for x in obj.tolist()]
    if isinstance(obj,float) and not np.isfinite(obj):
        return "NaN" if np.isnan(obj) else ("Infinity" if obj>0 else "-Infinity")
    return obj


def attrs(variable):
    return {name:value(variable.getncattr(name)) for name in variable.ncattrs()
            if name in ("units","long_name","standard_name","scale_factor","add_offset","_FillValue","valid_min","valid_max","cell_methods")}


def immutable_json(path,obj):
    if path.exists():
        e.require(e.read(path)==obj,"Existing external preparation changed: "+path.name)
    else:
        e.atomic(path,obj)


def metadata():
    e.require(e.read(e.LOCAL/"report_qa.json")["status"]=="PASS","LOCAL report QA gate required")
    with RFE.open(encoding="utf-8-sig",newline="") as handle:
        rank=list(csv.DictReader(handle))
    e.require(len(rank)==156 and [int(r["survival_rank"]) for r in rank]==list(range(1,157)),"Frozen natural RFE ranking invalid")
    top=rank[:10]
    expected=["p1__geopotential","p1__u_component_of_wind","p5__ozone_mass_mixing_ratio",
        "p650__u_component_of_wind","surface__convective_available_potential_energy",
        "surface__maximum_2m_temperature_since_previous_post_processing","p850__relative_humidity",
        "p850__ozone_mass_mixing_ratio","p5__u_component_of_wind","p500__ozone_mass_mixing_ratio"]
    e.require([r["feature_name"] for r in top]==expected,"RFE Top10 changed")
    with (GBS/"source_manifest.csv").open(encoding="utf-8-sig",newline="") as handle:
        files=list(csv.DictReader(handle))
    by_site={site:sorted(r["date_utc"] for r in files if r["site"]==site)
             for site in ("chilbolton","sparsholt","dundee")}
    common=sorted(set.intersection(*(set(dates) for dates in by_site.values())))
    e.require(len(files)==1982 and len(common)==399,"GBS inventory/common calendar changed")
    gbs_meta=[]
    for site in by_site:
        record=next(r for r in files if r["site"]==site)
        path=GBS/record["relative_path"]
        # The installed Windows netCDF C library does not accept this Unicode
        # directory name. Read immutable file bytes with Python, then open RAM.
        with netCDF4.Dataset("gbs_metadata.nc",memory=path.read_bytes()) as dataset:
            item={"site":site,"file":str(path.relative_to(e.ROOT)),"file_sha256":e.sha(path),
                  "variables":{n:{"shape":list(v.shape),"attributes":attrs(v)} for n,v in dataset.variables.items()},
                  "scalar_coordinates":{}}
            for n,v in dataset.variables.items():
                if v.size==1 and v.dtype.kind not in "SU":
                    item["scalar_coordinates"][n]=float(np.asarray(v[:]).reshape(-1)[0])
            gbs_meta.append(item)
    with netCDF4.Dataset(OPEN/"extracted/sml_data_2022.nc") as dataset:
        latitude=np.asarray(dataset.variables["site_0_lat"][:],float)
        longitude=np.asarray(dataset.variables["site_0_lon"][:],float)
        e.require(len(latitude)==len(longitude)==215,"OpenSat station count changed")
        cells=sorted(set(zip((np.rint(latitude*4)/4).tolist(),(np.rint(longitude*4)/4).tolist())))
        identifiers=np.asarray(dataset.variables["sml_id"][:])
        if identifiers.ndim==2 and identifiers.dtype.kind=="S":
            identifiers=netCDF4.chartostring(identifiers)
        identifiers=[str(value(x)).strip() for x in identifiers.reshape(-1)]
        e.require(len(identifiers)==len(set(identifiers))==215 and len(cells)==55,"OpenSat IDs/grid cells invalid")
        held=sorted(identifiers,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:43]
        open_meta={"stations":215,"variables":{n:{"shape":list(v.shape),"attributes":attrs(v)} for n,v in dataset.variables.items()},
                   "era5_nearest_cells":[list(x) for x in cells],"era5_unique_cells":len(cells),
                   "held_out_stations":sorted(held),"fit_stations":sorted(set(identifiers)-set(held)),
                   "minute_averaging_domain":"author readme states 1-min averages, not whether linear W or dB averaging",
                   "minute_model_scope":"native published RSL statistics; no recovery/interpolation of second-scale truth"}
    source_paths=[RFE,GBS/"source_manifest.csv",GBS/"QA_Revision_002/qa_summary.json",
                  OPEN/"extracted/sml_data_2022.nc",OPEN/"extracted/readme.txt",e.HERE/"external_prepare.py"]
    dates={"train":common[:239],"validation":common[239:319],"test":common[319:]}
    e.require(all(set(dates[a]).isdisjoint(dates[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))),"Calendar leakage")
    prepared={"version":"external-metadata-rfe-top10-v1","source_sha256":{str(p.relative_to(e.ROOT)):e.sha(p) for p in source_paths},
        "RFE_top10":top,"predictor_names":expected,"download_pressure_levels":PRESSURE,
        "download_pressure_variables":PVARS,"download_surface_variables":SVARS,
        "download_cartesian_note":"CDS retrieves 4 variable types on 5 levels; only the 8 selected pressure columns enter X; cloud/rain/snow are diagnostics",
        "GBS":{"sites":gbs_meta,"common_dates":common,"date_split":dates,
               "train_validation_sites":["chilbolton","sparsholt"],"test_sites":["chilbolton","sparsholt","dundee"],
               "never_fit_site":"dundee","primary_scope":"common 399-file-date window only, not all archived 2003-2006 records",
               "split_basis":"first239/next80/last80 common dates in source manifest, no attenuation values inspected",
               "sampling_seconds":1,"main_block_seconds":256},
        "OpenSat":{**open_meta,"train_dates":["2022-08-01","2022-10-31"],
                   "validation_dates":["2022-11-01","2022-11-30"],"test_dates":["2022-12-01","2022-12-31"],
                   "station_split_rule":"SHA256 of original sml_id; lowest 43 hashes held out, remaining172 training/validation; report held-out and seen stations separately",
                   "sampling_seconds":60,"main_block_minutes":4,"sensitivity_block_minutes":[2,8]},
        "models":["bilstm","transformer","mlp","svr_rbf","xgboost"],"seed":17,
        "new_domain_models_are_not_original156_weights":True,"preflight_only":True,
        "external_training_started":False,"test_errors_inspected":False,
        "weather_accumulation_note":"ERA5 precipitation/snowfall must be aligned by documented accumulation interval, not silently treated as instantaneous",
        "references":{"api":"https://cds.climate.copernicus.eu/en/how-to-api",
                      "pressure":"https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels",
                      "surface":"https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"}}
    prepared["fingerprint"]=e.fingerprint(prepared)
    immutable_json(DEST/"metadata.json",prepared)
    return prepared


def requests_plan():
    meta=metadata()
    areas={"GBS_England":[51.25,-1.5,51.,-1.5],"GBS_Dundee":[56.5,-3.,56.5,-3.],
           "OpenSat_France":[44.5,2.5,42.75,7.75]}
    calendar_dates=meta["GBS"]["common_dates"]
    months_gbs=sorted(set(x[:7] for x in calendar_dates))
    windows={"GBS_England":months_gbs,"GBS_Dundee":months_gbs,
             "OpenSat_France":["2022-08","2022-09","2022-10","2022-11","2022-12"]}
    jobs=[]
    for area_name,months in windows.items():
        for month in months:
            year,number=map(int,month.split("-"))
            if area_name.startswith("GBS"):
                days=sorted({x[-2:] for x in calendar_dates if x.startswith(month)})
            else:
                days=[f"{i:02d}" for i in range(1,calendar.monthrange(year,number)[1]+1)]
            for kind in ("surface","pressure"):
                request={"product_type":["reanalysis"],"variable":SVARS if kind=="surface" else PVARS,
                         "year":[str(year)],"month":[f"{number:02d}"],"day":days,
                         "time":[f"{i:02d}:00" for i in range(24)],"area":areas[area_name],
                         "data_format":"netcdf","download_format":"unarchived"}
                if kind=="pressure":
                    request["pressure_level"]=PRESSURE
                jobs.append({"id":f"{area_name}_{month}_{kind}","collection_id":
                    "reanalysis-era5-single-levels" if kind=="surface" else "reanalysis-era5-pressure-levels",
                    "request":request,"expected_hours":len(days)*24,"target":f"weather/{area_name}_{month}_{kind}.download"})
    # Rain/snow at t describe the preceding hour. Include the next midnight
    # after every study date even when that following date has no GBS file.
    next_days=sorted({(dt.date.fromisoformat(x)+dt.timedelta(days=1)).isoformat()
                      for x in calendar_dates}-set(calendar_dates))
    for area_name,padding_days in (("GBS_England",next_days),("GBS_Dundee",next_days),
                                  ("OpenSat_France",["2023-01-01"])):
        for month in sorted({x[:7] for x in padding_days}):
            request={"product_type":["reanalysis"],"variable":SVARS,
                     "year":[month[:4]],"month":[month[5:]],
                     "day":[x[-2:] for x in padding_days if x.startswith(month)],
                     "time":["00:00"],"area":areas[area_name],
                     "data_format":"netcdf","download_format":"unarchived"}
            jobs.append({"id":f"{area_name}_{month}_surface_padding","collection_id":"reanalysis-era5-single-levels",
                         "request":request,"expected_hours":len(request["day"]),
                         "target":f"weather/{area_name}_{month}_surface_padding.download"})
    plan={"version":"external-top10-cds-request-plan-v1","metadata_sha256":e.sha(DEST/"metadata.json"),
          "downloader_sha256":e.sha(e.HERE/"weather_download.py"),
          "jobs":jobs,"max_in_flight_requests":2,"first_step":"one-day pressure1 and surface smoke checks before all jobs",
          "token_serialized":False,"existing_raw_ERA5_modified":False,"model_input_columns":10,
          "accumulation_alignment":"For receiver hour [h,h+1), rain/snow use ERA5 validity time h+1; next-day 00 UTC padding includes gaps in GBS file dates",
          "license_action":"reuse previously accepted account; manual acceptance required if server reports missing terms"}
    plan["fingerprint"]=e.fingerprint(plan)
    immutable_json(DEST/"weather_plan.json",plan)
    return plan


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("metadata","plan"))
    args=parser.parse_args()
    result=metadata() if args.command=="metadata" else requests_plan()
    print(json.dumps({"status":"PREPARED","fingerprint":result["fingerprint"],
          "predictors":10,"request_count":len(result.get("jobs",[])),"training_started":False},indent=2),flush=True)
