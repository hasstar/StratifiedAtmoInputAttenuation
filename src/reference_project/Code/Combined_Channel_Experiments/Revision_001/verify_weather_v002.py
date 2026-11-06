"""Separate smoke audit v002; preserve v001 and all downloaded bytes unchanged.

Recognize ECMWF's explicit snowfall water-equivalent unit. This does not change
the weather plan, downloader, model inputs, or any numerical weather values.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import zipfile

import netCDF4
import numpy as np

import local_engine as e
import external_prepare as p

PLAN_FP = "c7fa6ff824fb436477f534fcec022ef7d60aed6afd4ae507f9dd1575e0282e16"
BASELINE_SHA = "37c13a0265ecf6b852f9766be696c46f480a0d7da63c0b653bf28dd40da7dffd"
DOWNLOADER_SHA = "8a6b4b7c28e2b61e6104baa86bee0528e9e125cce2f7836bf05c5799cb410c52"
PREPARATION_SHA = "63fb3025aba2f7fc6e0c62f7eeb242aa9786f3a91b3c1d0cfd40bafb475a8a7a"
SOURCE_URL = "https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation"
EXPECTED_UTC = ["2004-02-18T00:00:00Z", "2004-02-18T01:00:00Z"]


def immutable(path, obj):
    if path.exists():
        e.require(e.read(path) == obj, "Existing audit differs: " + path.name)
    else:
        e.atomic(path, obj)


def snowfall_unit(units, standard_name):
    e.require(units in {"m", "m of water equivalent"}, "Unknown snowfall unit: " + units)
    e.require(standard_name == "lwe_thickness_of_snowfall_amount", "Not snowfall water equivalent")
    return "m of water equivalent"


def synthetic_unit_checks():
    for units in ("m", "m of water equivalent"):
        e.require(snowfall_unit(units, "lwe_thickness_of_snowfall_amount") == "m of water equivalent", "Snowfall unit test failed")
    for units, name in (("mm", "lwe_thickness_of_snowfall_amount"),
                        ("kg m**-2 s**-1", "lwe_thickness_of_snowfall_amount"),
                        ("m of water equivalent", "snow_depth")):
        try:
            snowfall_unit(units, name)
        except RuntimeError:
            continue
        raise RuntimeError("Invalid snowfall unit/identity was accepted")
    return {"status": "PASS", "accepted_cases": 2, "rejected_cases": 3}


def members(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                member_path = PurePosixPath(name.replace("\\", "/"))
                e.require(not member_path.is_absolute() and ".." not in member_path.parts
                          and ":" not in name, "Unsafe archive member")
                if name.endswith(".nc"):
                    with netCDF4.Dataset("weather_unit_audit_v002.nc", memory=archive.read(name)) as dataset:
                        yield name, dataset
    else:
        with netCDF4.Dataset(path) as dataset:
            yield path.name, dataset


def smoke():
    tests = synthetic_unit_checks()
    here = Path(__file__).resolve().parent
    e.require(e.sha(here / "verify_weather.py") == BASELINE_SHA, "Failed baseline changed")
    e.require(e.sha(here / "weather_download.py") == DOWNLOADER_SHA, "Frozen downloader changed")
    e.require(e.sha(here / "external_prepare.py") == PREPARATION_SHA, "Frozen preparation changed")
    plan_path = p.DEST / "weather_plan.json"
    e.require(e.read(plan_path)["fingerprint"] == PLAN_FP, "Weather plan changed")
    download = p.DEST / "weather_smoke_qa.json"
    live = e.read(download)
    e.require(live["status"] == "PASS_DOWNLOAD_READABILITY_ONLY" and
              live["plan_fingerprint"] == PLAN_FP and live["smoke_requests"] == 2,
              "Smoke download gate failed")
    required = {
        "pressure": {"z": {"m**2 s**-2", "m2 s-2"}, "u": {"m s**-1", "m s-1"}},
        "surface": {"cape": {"J kg**-1", "J kg-1"}, "mx2t": {"K"},
                    "tcc": {"(0 - 1)", "1"}, "tp": {"m"}, "sf": {"m", "m of water equivalent"}},
    }
    rows = []
    for kind, variables in required.items():
        name = "SMOKE_GBS_England_2004-02-18_" + kind
        record = e.read(p.DEST / "weather_jobs" / (name + ".json"))
        path = p.DEST / record["job"]["target"]
        e.require(path.resolve().is_relative_to((p.DEST / "weather").resolve()), "Unexpected payload path")
        e.require(record["status"] == "VERIFIED" and record["plan_fingerprint"] == PLAN_FP
                  and e.fingerprint(record["job"]) == record["request_fingerprint"], "Request record changed")
        e.require(e.sha(path) == record["sha256"] == live["files"][name], "Smoke payload changed")
        e.require(record["job"]["request"]["area"] == [51.25, -1.5, 51.0, -1.5], "Smoke area changed")
        present = set()
        for member, dataset in members(path):
            tv = dataset.variables.get("valid_time", dataset.variables.get("time"))
            e.require(tv is not None and tv.size == 2, "Wrong smoke time axis")
            calendar = getattr(tv, "calendar", "standard")
            e.require(calendar in {"standard", "gregorian", "proleptic_gregorian"}, "Unknown calendar")
            dates = netCDF4.num2date(tv[:], tv.units, calendar=calendar)
            utc = [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in np.asarray(dates).reshape(-1)]
            e.require(utc == EXPECTED_UTC, "Requested UTC order/values not returned")
            coords = {}
            for key, values, units in (("latitude", [51.25, 51.0], "degrees_north"),
                                       ("longitude", [-1.5], "degrees_east")):
                variable = dataset.variables[key]
                e.require(str(variable.units) == units and
                          np.array_equal(np.asarray(variable[:]).reshape(-1), values), "Wrong coordinates: " + key)
                coords[key] = values
            dimensions = (tv.name, "latitude", "longitude")
            shape = (2, 2, 1)
            if kind == "pressure":
                level = dataset.variables["pressure_level"]
                e.require(np.array_equal(np.asarray(level[:]).reshape(-1), [1]) and
                          str(level.units) in {"hPa", "millibars"}, "Wrong pressure level/unit")
                dimensions = (tv.name, "pressure_level", "latitude", "longitude")
                shape = (2, 1, 2, 1)
            for key, allowed in variables.items():
                if key not in dataset.variables:
                    continue
                e.require(key not in present, "Duplicate smoke field: " + key)
                variable = dataset.variables[key]
                units = str(variable.units)
                e.require(units in allowed, "Unknown ERA5 field unit: " + key + " " + units)
                identity = str(getattr(variable, "standard_name", ""))
                canonical = snowfall_unit(units, identity) if key == "sf" else units
                e.require(variable.dimensions == dimensions and variable.shape == shape, "Wrong field axes: " + key)
                data = np.ma.asarray(variable[:])
                e.require(not np.ma.getmaskarray(data).any() and np.isfinite(np.asarray(data)).all(), "Missing/nonfinite field: " + key)
                present.add(key)
                rows.append({"job": name, "member": member, "variable": key, "units": units,
                             "canonical_units": canonical, "standard_name": identity,
                             "numeric_conversion_applied": False, "shape": list(data.shape),
                             "values": int(data.size), "minimum": float(data.min()), "maximum": float(data.max()),
                             "valid_times": utc, "coordinates": coords, "payload_sha256": record["sha256"]})
        e.require(present == set(variables), "Requested weather fields incomplete")
    e.require(len(rows) == 7 and sum(r["values"] for r in rows) == 28, "Wrong smoke field/value counts")
    history = p.DEST / "preflight_history" / "002_unit_audit"
    snapshot_path = history / "smoke_download_input.json"
    if snapshot_path.exists():
        snapshot = e.read(snapshot_path)
        # The frozen run entry refreshes only completed_utc when reusing smoke.
        semantic = lambda obj: {k: v for k, v in obj.items() if k != "completed_utc"}
        e.require(semantic(snapshot["contents"]) == semantic(live), "Smoke semantics changed after audit")
    else:
        snapshot = {"original_path": "external/weather_smoke_qa.json",
                    "original_sha256": e.sha(download), "contents": live}
    immutable(snapshot_path, snapshot)
    failure_path = p.DEST / "preflight_history" / "002_smoke_unit_failure.json"
    qa = {"status": "PASS", "scope": "two official smoke requests only; not full weather/model QA",
          "script_sha256": e.sha(Path(__file__)), "preserved_v001_sha256": BASELINE_SHA,
          "downloader_sha256": DOWNLOADER_SHA, "preparation_sha256": PREPARATION_SHA,
          "weather_plan_sha256": e.sha(plan_path), "plan_fingerprint": PLAN_FP,
          "download_qa_snapshot": str(snapshot_path.relative_to(e.OUT)).replace("\\", "/"),
          "download_qa_snapshot_sha256": e.sha(snapshot_path),
          "download_bookkeeping_note": "run rechecks identical smoke payloads and refreshes completed_utc; audited original is immutable",
          "failure_audit_sha256": e.sha(failure_path), "synthetic_unit_checks": tests,
          "checks": rows, "official_unit_source": SOURCE_URL,
          "official_accumulation_alignment": "tp/sf at t apply to [t-1h,t); no daily differencing for ERA5 hourly",
          "model_fits": 0, "channel_measurements_read": 0, "full_weather_available": False}
    immutable(p.DEST / "weather_smoke_unit_qa.json", qa)
    return {"status": "PASS", "field_checks": len(rows), "values": 28,
            "synthetic_unit_checks": tests, "full_weather_available": False, "model_fits": 0}


if __name__ == "__main__":
    print(json.dumps(smoke(), indent=2), flush=True)
