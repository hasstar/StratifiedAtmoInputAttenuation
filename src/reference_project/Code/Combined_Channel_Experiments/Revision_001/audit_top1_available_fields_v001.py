"""Independent, offline audit of the five frozen reused Top1 weather payloads.

No downloader/model imports, network access, source mutation, or fitting.
This is a partial-field audit, not full Top1 coverage or model compatibility QA.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import netCDF4
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top1_v003"
PLAN_SHA = "265fc2dbddd0f7b8215aad9f4d99518fd1fc1dc6727000ee14536172aebf06d3"
PLAN_FP = "b80606aa0b5619ce8a447f4dabcc6cfabe6ca5aeb17b95c0a40451545fe70500"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def request_times(request):
    times = [f"{y}-{m}-{d}T{t}:00Z" for y, m, d, t in itertools.product(
        request["year"], request["month"], request["day"], request["time"])]
    require(len(set(times)) == len(times), "Duplicate requested UTC")
    for value in times:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    return sorted(times)


def expected_grid(request):
    north, west, south, east = request["area"]
    require(north >= south and east >= west, "Unsupported reversed area")
    ticks = np.asarray([north, west, south, east], dtype=float) * 4
    require(np.array_equal(ticks, np.rint(ticks)), "Area not on frozen 0.25 degree grid")
    return (np.arange(round(north * 4), round(south * 4) - 1, -1) / 4,
            np.arange(round(west * 4), round(east * 4) + 1) / 4)


def safe_member(name):
    path = PurePosixPath(name.replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts and ":" not in name,
            "Unsafe archive member")


def datasets(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                safe_member(name)
                if name.endswith(".nc"):
                    with netCDF4.Dataset("top1_field_audit.nc", memory=archive.read(name)) as ds:
                        yield name, ds
    else:
        with netCDF4.Dataset("top1_field_audit.nc", memory=Path(path).read_bytes()) as ds:
            yield Path(path).name, ds


def inspect_dataset(ds, request):
    require("geopotential" in request["variable"] and "1" in request["pressure_level"],
            "Request is not Top1-compatible")
    tv = ds.variables.get("valid_time", ds.variables.get("time"))
    require(tv is not None and len(tv.dimensions) == 1, "Missing/invalid time axis")
    require(str(getattr(tv, "standard_name", "")) == "time", "Time identity differs")
    calendar = str(getattr(tv, "calendar", "standard"))
    require(calendar in {"standard", "gregorian", "proleptic_gregorian"}, "Unknown calendar")
    time_data = np.ma.asarray(tv[:])
    require(not np.ma.getmaskarray(time_data).any(), "Masked UTC coordinate")
    decoded = netCDF4.num2date(time_data, tv.units, calendar=calendar)
    utc = [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in np.asarray(decoded).reshape(-1)]
    require(utc == request_times(request), "Returned UTC differs from exact request")
    lat, lon = expected_grid(request)
    for name, expected, unit in (("latitude", lat, "degrees_north"),
                                 ("longitude", lon, "degrees_east")):
        var = ds.variables[name]
        require(var.dimensions == (name,), "Coordinate axis differs: " + name)
        require(str(var.units) == unit, "Coordinate unit differs: " + name)
        actual = np.ma.asarray(var[:])
        require(not np.ma.getmaskarray(actual).any() and np.array_equal(actual, expected),
                "Coordinate values differ: " + name)
    level = ds.variables["pressure_level"]
    require(level.dimensions == ("pressure_level",), "Pressure axis differs")
    require(str(level.units) in {"hPa", "millibars"}, "Pressure unit differs")
    levels = np.ma.asarray(level[:])
    require(not np.ma.getmaskarray(levels).any(), "Masked pressure coordinate")
    require(len(levels) == len(set(levels.tolist())) and
            set(levels.tolist()) == set(float(x) for x in request["pressure_level"]),
            "Returned pressure levels differ from request")
    positions = np.flatnonzero(np.asarray(levels) == 1.0)
    require(len(positions) == 1, "Exactly one 1 hPa level required")
    var = ds.variables["z"]
    require(str(var.units) in {"m**2 s**-2", "m2 s-2"}, "Geopotential unit differs")
    require(str(getattr(var, "standard_name", "")) == "geopotential", "Not geopotential")
    require(int(getattr(var, "GRIB_paramId", -1)) == 129, "Wrong GRIB parameter")
    require(str(getattr(var, "GRIB_stepType", "")) == "instant", "Not instantaneous geopotential")
    dims = (tv.dimensions[0], "pressure_level", "latitude", "longitude")
    shape = (len(utc), len(levels), len(lat), len(lon))
    require(var.dimensions == dims and var.shape == shape, "Geopotential shape/axes differ")
    field = np.ma.asarray(var[:, int(positions[0]), :, :])
    require(not np.ma.getmaskarray(field).any() and np.isfinite(np.asarray(field)).all(),
            "Missing/nonfinite Top1 geopotential")
    numeric = np.asarray(field, dtype="<f8")
    return {
        "field": "p1__geopotential", "variable": "z", "pressure_hpa": 1,
        "pressure_index_in_payload": int(positions[0]),
        "all_payload_pressure_levels_hpa": levels.tolist(),
        "units": str(var.units), "standard_name": str(var.standard_name),
        "grib_param_id": 129, "step_type": "instant", "calendar": calendar,
        "utc": utc, "request_hours": len(utc), "shape": list(field.shape),
        "latitude": lat.tolist(), "longitude": lon.tolist(),
        "grid_values": int(field.size), "finite_grid_values": int(np.isfinite(numeric).sum()),
        "minimum": float(numeric.min()), "maximum": float(numeric.max()),
        "float64_c_order_sha256": hashlib.sha256(numeric.tobytes(order="C")).hexdigest(),
        "numeric_conversion_applied": False, "spatial_interpolation_applied": False,
    }


def build_audit():
    plan_path = DEST / "plan.json"
    require(sha(plan_path) == PLAN_SHA, "Frozen Top1 plan SHA differs")
    plan = read(plan_path)
    require(plan["fingerprint"] == PLAN_FP and plan["feature_names"] == ["p1__geopotential"],
            "Frozen Top1 scope differs")
    require(len(plan["reused"]) == 5, "This snapshot audit expects exactly five reused files")
    sources = {str(plan_path): sha(plan_path), str(Path(__file__)): sha(Path(__file__))}
    coverage, checks = set(), []
    for item in plan["reused"]:
        path = (ROOT / item["target"]).resolve()
        state_path = (ROOT / item["source_state"]).resolve()
        require(path.is_relative_to(ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external"),
                "Payload outside declared T012 outputs")
        record = read(state_path)
        require(record["status"] == "VERIFIED" and record["sha256"] == item["sha256"],
                "Reused verified record differs")
        require(record["job"]["request"] == item["job"]["request"] and
                record["request_id"] == item["request_id"], "Original request/ID differs")
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "Reused payload SHA/size differs")
        sources[str(path)], sources[str(state_path)] = sha(path), sha(state_path)
        found = []
        for member, ds in datasets(path):
            if "z" in ds.variables:
                row = inspect_dataset(ds, item["job"]["request"])
                require(row["request_hours"] == item["job"]["expected_hours"], "Hour count differs")
                row.update({"parent_id": item["parent_id"], "job_id": item["job"]["id"],
                            "payload": item["target"], "member": member,
                            "payload_sha256": item["sha256"], "payload_bytes": item["bytes"],
                            "request_id": item["request_id"]})
                found.append(row)
        require(len(found) == 1, "Missing or duplicate geopotential member")
        row = found[0]
        keys = {(row["parent_id"], t) for t in row["utc"]}
        require(not coverage.intersection(keys), "Overlapping reused request hours")
        coverage.update(keys)
        checks.append(row)
    expected = {(p["id"], t) for p in plan["parents"] for t in request_times(p["request"])}
    require(coverage.issubset(expected), "Coverage outside frozen Top1 parent scope")
    require(len(coverage) == 864 and sum(x["grid_values"] for x in checks) == 1728,
            "Five-file frozen coverage count differs")
    require(len(expected) == 22824, "Full Top1 expected hours differ")
    for path, digest in sources.items():
        require(sha(path) == digest, "Audit source changed during read: " + path)
    return {
        "status": "PASS_REUSED_TOP1_FIELDS_ONLY", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exactly the five immutable plan.reused pressure payloads; z at 1 hPa only",
        "plan_fingerprint": PLAN_FP, "plan_sha256": PLAN_SHA,
        "audited_payloads": len(checks), "audited_request_hours": len(coverage),
        "audited_grid_values": sum(x["grid_values"] for x in checks),
        "full_top1_request_hours": len(expected), "not_audited_request_hours": len(expected - coverage),
        "complete_top1_parent_months_in_snapshot": sum(
            {(p["id"], t) for t in request_times(p["request"])}.issubset(coverage) for p in plan["parents"]),
        "full_top1_field_qa_pass": False, "old_top10_complete": False,
        "model_input_compatibility": "NOT_ESTABLISHED_BY_FIELD_QA",
        "smoke_included": False, "model_fits": 0, "model_predictions": 0,
        "other_fields_and_levels_audited": False,
        "checks": checks, "source_sha256": sources, "source_files_unchanged_after_read": True,
    }


def main():
    report = build_audit()
    out = DEST / "field_audit_001" / "qa.json"
    require(not out.exists(), "Immutable field audit already exists; use a new version")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "audited_payloads", "audited_request_hours",
                                            "audited_grid_values", "full_top1_field_qa_pass")}, indent=2))


if __name__ == "__main__":
    main()
