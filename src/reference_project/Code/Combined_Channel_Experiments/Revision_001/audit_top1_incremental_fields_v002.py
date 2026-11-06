"""Offline incremental field audit: original five files plus March b005 only.

Reuses the frozen field checker, never a downloader or model. Prior audits and
all inputs stay read-only; only field_audit_002 receives immutable new outputs.
This snapshot is partial, and does not lift the pending spatial-input gate.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

sys.dont_write_bytecode = True
import audit_top1_available_fields_v001 as field

ROOT, DEST = field.ROOT, field.DEST
OUT = DEST / "field_audit_002"
JOB_ID = "GBS_England_2004-03_pressure__b005"
REQUEST_ID = "09473232-0465-4d92-8d9e-c6028ae3036a"
PAYLOAD_SHA = "c01148ee9c1331a0e59787eb1b48536bd5173eeb5cd5e727666bb31ccbdf1fe6"
FROZEN = {
    DEST / "plan.json": field.PLAN_SHA,
    DEST / "field_audit_001/qa.json": "325b9d787e2f12bf09640d48ede3bec4f5db7830a196868bce619d816a581c09",
    DEST / "field_audit_001/delivery_qa.json": "df4914f93d1964db03958d0cd5c591242fb18d8b1558204a9713d78d8eeabe0c",
    DEST / "input_compatibility_001.json": "497b6993ee19c8537a5ea65f1008889ddf6a8833e8cf30d8c4f836f8d73261a8",
    Path(field.__file__): "35682b28775e9c8a32e6c5c4da6986647a62463407fd7ced005f42528c989a1b",
    Path(__file__).with_name("test_audit_top1_available_fields_v001.py"):
        "26e721c931c02c1ec9bda000e509832fbcc608b0a4df9e68c30353fe48e1bbbc",
}


def validate_record(record, job, adoption):
    field.require(record["status"] == "VERIFIED" and record["remote_status"] == "successful",
                  "Record is not a completed verified download")
    field.require(record["plan_fingerprint"] == field.PLAN_FP, "Plan fingerprint differs")
    field.require(record["job"] == job == adoption["original_job"], "Job differs from frozen adoption")
    field.require(record["request_id"] == adoption["request_id"] == REQUEST_ID,
                  "Inherited request ID differs")
    field.require(record["migrated_from"] == adoption["source_state"] and
                  record["migrated_from_sha256"] == adoption["source_state_sha256"],
                  "Migration provenance differs")
    field.require(record["bytes"] == record["remote_expected_bytes"] and record["bytes"] > 0,
                  "Official expected size differs")
    field.require(record["sha256"] == PAYLOAD_SHA, "Completed payload SHA differs")


def independently_check(row):
    """Second extraction; UTC decoded using Unix seconds, not num2date."""
    path = (ROOT / row["payload"]).resolve()
    field.require(path.stat().st_size == row["payload_bytes"] and
                  field.sha(path) == row["payload_sha256"], "Independent payload SHA/size differs")
    with netCDF4.Dataset("independent_increment.nc", memory=path.read_bytes()) as ds:
        levels = np.asarray(ds["pressure_level"][:]).tolist()
        field.require(levels.count(1.0) == 1, "Independent level selection failed")
        array = np.ma.asarray(ds["z"][:, levels.index(1.0), :, :])
        field.require(not np.ma.getmaskarray(array).any(), "Independent extraction is masked")
        values = np.asarray(array, dtype="<f8")
        field.require(np.isfinite(values).all(), "Independent extraction is nonfinite")
        field.require(list(values.shape) == row["shape"] and values.size == row["grid_values"],
                      "Independent dimensions/count differ")
        field.require(hashlib.sha256(values.tobytes(order="C")).hexdigest() ==
                      row["float64_c_order_sha256"], "Independent numeric SHA differs")
        for name in ("latitude", "longitude"):
            field.require(np.asarray(ds[name][:]).tolist() == row[name], "Independent coordinates differ")
        tv = ds["valid_time"]
        field.require(tv.units == "seconds since 1970-01-01", "Independent time decoder unsupported")
        seconds = np.asarray(tv[:], dtype=float)
        field.require(np.isfinite(seconds).all() and np.array_equal(seconds, np.rint(seconds)),
                      "Nonintegral/nonfinite epoch seconds")
        times = [datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                 for t in seconds]
        field.require(times == row["utc"], "Independent UTC differs")
    return {"job_id": row["job_id"], "request_hours": len(times), "grid_values": int(values.size),
            "payload_sha256": row["payload_sha256"], "numeric_sha_match": True,
            "utc_coordinates_dimensions_finite_match": True}


def build_audit():
    for path, digest in FROZEN.items():
        field.require(field.sha(path) == digest, "Frozen audit/plan differs: " + str(path))
    plan = field.read(DEST / "plan.json")
    field.require(plan["fingerprint"] == field.PLAN_FP, "Frozen plan fingerprint differs")
    previous = field.read(DEST / "field_audit_001/qa.json")
    field.require(previous["status"] == "PASS_REUSED_TOP1_FIELDS_ONLY" and
                  previous["audited_request_hours"] == 864, "Previous field QA differs")
    job = next(j for j in plan["jobs"] if j["id"] == JOB_ID)
    adoption = next(j for j in plan["adopted"] if j["id"] == JOB_ID)
    state_path = DEST / "jobs" / (JOB_ID + ".json")
    state_digest = field.sha(state_path)
    record = field.read(state_path)
    validate_record(record, job, adoption)
    payload = (DEST / job["target"]).resolve()
    field.require(payload.is_relative_to((DEST / "payloads").resolve()), "Payload escaped Top1 outputs")
    field.require(payload.stat().st_size == record["bytes"] and field.sha(payload) == PAYLOAD_SHA,
                  "Actual payload SHA/size differs")
    sources = dict(previous["source_sha256"])
    sources.update({str(p.resolve()): h for p, h in FROZEN.items()})
    sources.update({str((ROOT / p).resolve()): h for p, h in plan["protected_files"].items()})
    sources.update({str(state_path): state_digest, str(payload): PAYLOAD_SHA,
                    str(Path(__file__)): field.sha(__file__)})
    for path, digest in sources.items():
        field.require(field.sha(path) == digest, "Source SHA differs: " + path)
    rows = []
    for member, ds in field.datasets(payload):
        if "z" in ds.variables:
            row = field.inspect_dataset(ds, job["request"])
            field.require(row["request_hours"] == job["expected_hours"] == 72, "Increment hours differ")
            row.update({"parent_id": job["parent_id"], "job_id": JOB_ID,
                        "payload": str(payload.relative_to(ROOT)), "member": member,
                        "payload_sha256": PAYLOAD_SHA, "payload_bytes": record["bytes"],
                        "request_id": REQUEST_ID})
            rows.append(row)
    field.require(len(rows) == 1, "Missing/duplicate new geopotential member")
    combined = copy.deepcopy(previous["checks"]) + rows
    coverage = set()
    for row in combined:
        keys = {(row["parent_id"], t) for t in row["utc"]}
        field.require(not coverage.intersection(keys), "Overlapping parent/hour coverage")
        coverage.update(keys)
    expected = {(p["id"], t) for p in plan["parents"] for t in field.request_times(p["request"])}
    field.require(coverage.issubset(expected) and len(expected) == 22824, "Parent scope differs")
    field.require(len(coverage) == 936 and sum(r["grid_values"] for r in combined) == 1872,
                  "Combined snapshot count differs")
    independent = [independently_check(row) for row in combined]
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("test_audit_top1_available_fields_v001")
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    field.require(result.wasSuccessful() and result.testsRun == 12, "Field tests failed")
    # The six negative controls start from the real, validated immutable record.
    controls = (("status", "WAITING_REMOTE"), ("remote_status", "accepted"),
                ("request_id", "different-id"), ("plan_fingerprint", "wrong-plan"),
                ("remote_expected_bytes", record["bytes"] + 1), ("sha256", "0" * 64))
    rejected = []
    for key, value in controls:
        bad = copy.deepcopy(record)
        bad[key] = value
        try:
            validate_record(bad, job, adoption)
        except ValueError as error:
            rejected.append({"changed_field": key, "expected_rejection": str(error)})
        else:
            raise ValueError("Negative record control accepted: " + key)
    for path, digest in sources.items():
        field.require(field.sha(path) == digest, "Source changed during audit: " + path)
    return {
        "status": "PASS_INCREMENTAL_TOP1_FIELDS_ONLY", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Original five formally audited payloads plus completed March b005; z at 1 hPa only",
        "plan_fingerprint": field.PLAN_FP, "plan_sha256": field.PLAN_SHA,
        "prior_field_qa": "field_audit_001/qa.json", "added_job_ids": [JOB_ID],
        "added_request_hours": 72, "added_grid_values": 144,
        "audited_payloads": 6, "audited_request_hours": len(coverage), "audited_grid_values": 1872,
        "full_top1_request_hours": len(expected), "not_audited_request_hours": len(expected - coverage),
        "complete_top1_parent_months_in_snapshot": sum(
            {(p["id"], t) for t in field.request_times(p["request"])}.issubset(coverage) for p in plan["parents"]),
        "full_top1_field_qa_pass": False, "old_top10_complete": False, "overall_completion": False,
        "model_input_compatibility": "BLOCKED_SPATIAL_INPUT_DEFINITION",
        "new_fit_calls": 0, "model_predictions": 0, "network_requests": 0, "smoke_included": False,
        "other_fields_and_levels_audited": False, "checks": combined,
        "independent_extraction": independent, "positive_negative_field_tests": result.testsRun,
        "field_test_log": log.getvalue(), "negative_record_controls": rejected,
        "source_sha256": sources, "source_files_unchanged_after_read": True,
        "completed_job_state_snapshot": record,
    }


def main():
    output = OUT / "qa.json"
    failure = OUT / "failure.json"
    field.require(not output.exists() and not failure.exists(), "Immutable audit already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        report = build_audit()
    except Exception as error:
        with failure.open("x", encoding="utf-8") as stream:
            json.dump({"status": "FAILED_INCREMENTAL_FIELD_AUDIT", "error_type": type(error).__name__,
                       "message": str(error), "auditor_sha256": field.sha(__file__),
                       "checked_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        raise
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "audited_payloads", "audited_request_hours",
                                            "audited_grid_values", "model_input_compatibility")}, indent=2))


if __name__ == "__main__":
    main()
