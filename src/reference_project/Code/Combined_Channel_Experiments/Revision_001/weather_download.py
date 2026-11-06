"""Bounded official CDS requests, persistent remote IDs, no credential logging."""
from __future__ import annotations
import argparse
import contextlib
import json
import logging
import os
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import netCDF4
import yaml
import local_engine as e
import external_prepare as p

DEST=p.DEST


def config():
    path=Path.home()/".cdsapirc"
    e.require(path.is_file(),"Existing CDS credentials are absent; user action required")
    obj=yaml.safe_load(path.read_text(encoding="utf-8"))
    e.require(isinstance(obj,dict) and bool(obj.get("key")),"CDS key is not configured")
    url=str(obj.get("url","https://cds.climate.copernicus.eu/api")).rstrip("/")
    e.require(urlparse(url).hostname=="cds.climate.copernicus.eu" and urlparse(url).scheme=="https",
              "Unexpected credential endpoint; do not transmit credential")
    if url.endswith("/api/v2"):
        url=url[:-3]
    return url,str(obj["key"])


def client():
    sys.path.insert(0,str(e.ROOT/"Code/Dataset_Build/_vendor_cdsapi"))
    from ecmwf.datastores import Client
    url,key=config()
    logging.getLogger("ecmwf.datastores").setLevel(logging.ERROR)
    logging.getLogger("multiurl").setLevel(logging.ERROR)
    return Client(url=url,key=key,verify=True,timeout=60,progress=False,sleep_max=30,
                  retry_after=30,maximum_tries=3)


def safe_error(exc):
    text=str(exc)
    try:
        _,key=config()
        text=text.replace(key,"[REDACTED]")
    except Exception:
        pass
    return {"error_type":type(exc).__name__,"message":text[:2500]}


@contextlib.contextmanager
def lock(command):
    import msvcrt
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/"weather_writer.lock").open("a+b") as f:
        if f.tell()==0:
            f.write(b"0");f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        active={"pid":os.getpid(),"python":sys.executable,"argv":sys.argv,"command":command,
                "status":"RUNNING","started_utc":e.now()}
        e.atomic(DEST/"weather_active_worker.json",active)
        state="EXITED_NORMAL"
        try:
            yield
        except BaseException as exc:
            state="FAILED"
            e.atomic(DEST/"weather_failures"/(str(time.time_ns())+".json"),
                     {**active,"failed_utc":e.now(),**safe_error(exc)})
            raise RuntimeError(json.dumps(safe_error(exc),ensure_ascii=False)) from None
        finally:
            e.atomic(DEST/"weather_active_worker.json",{**active,"status":state,"finished_utc":e.now()})
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def inspect_file(path):
    members=[]
    files=[]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                e.require(not Path(name).is_absolute() and ".." not in Path(name).parts,"Unsafe archive member")
                if name.endswith(".nc"):
                    raw=archive.read(name)
                    with netCDF4.Dataset("in_memory.nc",memory=raw) as ds:
                        files.append({"member":name,"dimensions":{n:len(d) for n,d in ds.dimensions.items()},
                            "variables":{n:{"shape":list(v.shape),"attributes":p.attrs(v)} for n,v in ds.variables.items()}})
                    members.append(name)
    else:
        with netCDF4.Dataset(path) as ds:
            files.append({"member":None,"dimensions":{n:len(d) for n,d in ds.dimensions.items()},
                "variables":{n:{"shape":list(v.shape),"attributes":p.attrs(v)} for n,v in ds.variables.items()}})
    e.require(bool(files),"No readable NetCDF in downloaded result")
    return {"netcdf_members":len(files),"files":files}


def one(job,plan_fp):
    state_path=DEST/"weather_jobs"/(job["id"]+".json")
    target=DEST/job["target"]
    require_prefix=target.resolve().is_relative_to((DEST/"weather").resolve())
    e.require(require_prefix,"Weather target outside new weather directory")
    saved=e.read(state_path) if state_path.exists() else {}
    e.require(not saved or saved["request_fingerprint"]==e.fingerprint(job),"Existing request changed")
    if saved.get("status")=="VERIFIED":
        e.require(target.is_file() and e.sha(target)==saved["sha256"],"Verified download changed")
        print("WEATHER SKIP",job["id"],flush=True)
        return saved
    c=client()
    if saved.get("request_id"):
        remote=c.get_remote(saved["request_id"])
        e.require(remote.collection_id==job["collection_id"],"Recovered remote collection differs")
        print("WEATHER RESUME REMOTE",job["id"],remote.request_id,flush=True)
    else:
        remote=c.submit(job["collection_id"],job["request"])
        saved={"job":job,"request_fingerprint":e.fingerprint(job),"plan_fingerprint":plan_fp,
               "request_id":remote.request_id,"status":"SUBMITTED","submitted_utc":e.now(),"key_saved":False}
        e.atomic(state_path,saved)
        print("WEATHER SUBMITTED",job["id"],remote.request_id,flush=True)
    e.require(remote.status not in ("failed","rejected"),"Remote job failed; preserve ID and diagnose, do not blindly resubmit")
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_name(target.name+".part")
        e.require(not temporary.exists(),"Partial payload exists; preserve and diagnose before download retry")
        remote.download(str(temporary))
        inspection=inspect_file(temporary)
        temporary.rename(target)
    else:
        inspection=inspect_file(target)
    result={**saved,"status":"VERIFIED","downloaded_utc":e.now(),"sha256":e.sha(target),
            "bytes":target.stat().st_size,"inspection":inspection}
    e.atomic(state_path,result)
    print("WEATHER VERIFIED",job["id"],result["bytes"],flush=True)
    return result


def smoke():
    plan=p.requests_plan()
    jobs=[]
    for kind in ("pressure","surface"):
        template=next(x for x in plan["jobs"] if x["id"].startswith("GBS_England_") and x["id"].endswith(kind))
        job=json.loads(json.dumps(template))
        job["id"]="SMOKE_GBS_England_2004-02-18_"+kind
        job["request"]["day"]=["18"]
        job["request"]["time"]=["00:00","01:00"]
        job["expected_hours"]=2
        if kind=="pressure":
            job["request"]["pressure_level"]=["1"]
            job["request"]["variable"]=["geopotential","u_component_of_wind"]
        job["target"]="weather/"+job["id"]+".download"
        jobs.append(job)
    results=[one(job,plan["fingerprint"]) for job in jobs]
    qa={"status":"PASS_DOWNLOAD_READABILITY_ONLY","plan_fingerprint":plan["fingerprint"],
        "smoke_requests":2,"key_saved":False,"completed_utc":e.now(),
        "files":{r["job"]["id"]:r["sha256"] for r in results},
        "source_and_unit_mapping_QA_complete":False}
    e.atomic(DEST/"weather_smoke_qa.json",qa)
    return qa


def run():
    plan=p.requests_plan()
    smoke()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda job:one(job,plan["fingerprint"]),plan["jobs"]))
    qa={"status":"PASS_DOWNLOAD_READABILITY_ONLY","plan_fingerprint":plan["fingerprint"],
        "completed_utc":e.now(),"requests":len(results),"key_saved":False,
        "files":{r["job"]["id"]:r["sha256"] for r in results},"source_and_unit_mapping_QA_complete":False,
        "model_training_started":False}
    e.atomic(DEST/"weather_download_qa.json",qa)
    return qa


def status():
    jobs=[e.read(x) for x in sorted((DEST/"weather_jobs").glob("*.json"))]
    result={"active_worker":e.read(DEST/"weather_active_worker.json") if (DEST/"weather_active_worker.json").exists() else None,
            "submitted":len(jobs),"verified":sum(x.get("status")=="VERIFIED" for x in jobs),
            "download_qa":e.read(DEST/"weather_download_qa.json").get("status") if (DEST/"weather_download_qa.json").exists() else None,
            "pending_remote_ids":[{"job":x["job"]["id"],"request_id":x.get("request_id"),"status":x.get("status")}
                                  for x in jobs if x.get("status")!="VERIFIED"],"model_training_started":False}
    lock_path=DEST/"weather_writer.lock"
    result["os_lock_held"]=None
    if lock_path.exists():
        import msvcrt
        with lock_path.open("r+b") as handle:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                result["os_lock_held"]=True
            else:
                result["os_lock_held"]=False
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("smoke","run","status"))
    args=parser.parse_args()
    if args.command=="status":
        print(json.dumps(status(),indent=2),flush=True)
    else:
        with lock(args.command):
            print(json.dumps(smoke() if args.command=="smoke" else run(),indent=2),flush=True)
