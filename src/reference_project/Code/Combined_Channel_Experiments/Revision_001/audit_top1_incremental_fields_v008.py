"""Offline immutable snapshot: eleven prior payloads plus August/September.

Adapt v007 to two completed single-field records. Reuse the frozen v001 field
checker and v002 independent extraction; never their main functions. No model,
downloader, network or fitting calls. Only field_audit_008 receives new output.
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top1_v003"
OUT = DEST / "field_audit_008"
PRIOR = DEST / "field_audit_007/qa.json"
ADDITIONS = (
    {"job_id": "GBS_England_2004-08_pressure__top1",
     "request_id": "d9fc779a-7f4a-4e51-89a9-2e3a81d9ba18",
     "payload_sha": "7fea1005680c677015839c65cd75cfe0a05504bec98249be3c7bc743e61a4a89",
     "state_sha": "663b940afc184b97c1c79bef8340e37c435272baadbf501660b4b041615f0819",
     "bytes": 81086, "hours": 696, "month": "08",
     "days": [f"{d:02d}" for d in range(1, 32) if d not in (3, 4)]},
    {"job_id": "GBS_England_2004-09_pressure__top1",
     "request_id": "5505f067-13da-4708-a3e4-c700523864fe",
     "payload_sha": "171ea38bb2076b229919b03c5266cdeb3acf7b7eebe6f8743e6afa7077e64d9b",
     "state_sha": "e0a48f374142f8eee6d327a359dcfe87920a0a3ba9d111bec1d1f4d8ce4ef76b",
     "bytes": 81802, "hours": 720, "month": "09",
     "days": [f"{d:02d}" for d in range(1, 31)]},
)
FROZEN = {
    HERE / "audit_top1_available_fields_v001.py":
        "35682b28775e9c8a32e6c5c4da6986647a62463407fd7ced005f42528c989a1b",
    HERE / "audit_top1_incremental_fields_v002.py":
        "b827855dcfd625c247a22afaafee5adf21be550d5f6a620e94c53a918698250f",
    HERE / "audit_top1_incremental_fields_v007.py":
        "c7ac7d088d1193c95d5a2c6feb6420f2c07af9212ed70f926214a9089ecb0d10",
    HERE / "test_audit_top1_available_fields_v001.py":
        "26e721c931c02c1ec9bda000e509832fbcc608b0a4df9e68c30353fe48e1bbbc",
    DEST / "plan.json": "265fc2dbddd0f7b8215aad9f4d99518fd1fc1dc6727000ee14536172aebf06d3",
    PRIOR: "015ac6e9eefcc169315460b9905ccf7dbf39effbcbcca2398543c7b343b4b19a",
    DEST / "field_audit_007/delivery_readback_qa.json":
        "b8417edc51c08b28697b025ff1c25217e1b46febbe915ca027f431a002e878fb",
    DEST / "input_compatibility_001.json":
        "497b6993ee19c8537a5ea65f1008889ddf6a8833e8cf30d8c4f836f8d73261a8",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_record(record, job, plan_fp, spec):
    require(record["status"] == "VERIFIED" and record["remote_status"] == "successful",
            "Record is not a completed verified download")
    require(record["plan_fingerprint"] == plan_fp, "Plan fingerprint differs")
    require(record["job"] == job and job["id"] == spec["job_id"], "Frozen Top1 job differs")
    require(job["request"]["variable"] == ["geopotential"] and
            job["request"]["pressure_level"] == ["1"], "Not single-field 1 hPa request")
    require(record["request_fingerprint"] == fingerprint(job), "Request fingerprint differs")
    require(record["request_id"] == spec["request_id"], "Original request ID differs")
    require("migrated_from" not in record and "migrated_from_sha256" not in record,
            "Unexpected legacy migration provenance")
    require(record["bytes"] == record["remote_expected_bytes"] == spec["bytes"],
            "Official expected size differs")
    require(record["sha256"] == spec["payload_sha"], "Completed payload SHA differs")


def build_audit():
    for path, digest in FROZEN.items():
        require(sha(path) == digest, "Frozen source differs: " + str(path))
    import audit_top1_available_fields_v001 as field
    from audit_top1_incremental_fields_v002 import independently_check

    plan = field.read(DEST / "plan.json")
    require(plan["fingerprint"] == field.PLAN_FP and
            plan["feature_names"] == ["p1__geopotential"], "Plan scope differs")
    previous = field.read(PRIOR)
    require(previous["status"] == "PASS_INCREMENTAL_TOP1_FIELDS_ONLY" and
            previous["audited_payloads"] == 11 and previous["audited_request_hours"] == 3816 and
            previous["audited_grid_values"] == 7632, "Previous snapshot differs")
    compatibility = field.read(DEST / "input_compatibility_001.json")
    require(compatibility["status"] == "BLOCKED_SPATIAL_INPUT_DEFINITION" and
            compatibility["new_fit_calls"] == compatibility["model_predictions"] == 0,
            "Pending compatibility gate changed; reassess snapshot")
    sources = dict(previous["source_sha256"])
    sources.update({str(p.resolve()): h for p, h in FROZEN.items()})
    for path, digest in plan["protected_files"].items():
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        sources[str(resolved.resolve())] = digest
    sources[str(Path(__file__).resolve())] = sha(__file__)
    inputs = []
    for spec in ADDITIONS:
        job = next(j for j in plan["jobs"] if j["id"] == spec["job_id"])
        require(job["request"]["year"] == ["2004"] and
                job["request"]["month"] == [spec["month"]] and
                job["request"]["day"] == spec["days"], "Frozen dates/gaps differ")
        require(job["id"] not in {a["id"] for a in plan["adopted"]}, "Unexpected inherited job")
        state_path = DEST / "jobs" / (job["id"] + ".json")
        require(sha(state_path) == spec["state_sha"], "Completed state snapshot changed")
        record = field.read(state_path)
        validate_record(record, job, field.PLAN_FP, spec)
        payload = (DEST / job["target"]).resolve()
        require(payload.is_relative_to((DEST / "payloads").resolve()), "Payload escaped new outputs")
        require(payload.stat().st_size == spec["bytes"] and sha(payload) == spec["payload_sha"],
                "Actual payload SHA/size differs")
        sources.update({str(state_path): spec["state_sha"], str(payload): spec["payload_sha"]})
        inputs.append((spec, job, record, payload))
    for path, digest in sources.items():
        require(sha(path) == digest, "Source SHA differs: " + path)

    rows, rejected = [], []
    for spec, job, record, payload in inputs:
        found = []
        for member, ds in field.datasets(payload):
            if "z" in ds.variables:
                row = field.inspect_dataset(ds, job["request"])
                require(row["request_hours"] == job["expected_hours"] == spec["hours"],
                        "New hour count differs")
                require(row["grid_values"] == spec["hours"] * 2 and
                        row["all_payload_pressure_levels_hpa"] == [1.0] and
                        row["pressure_index_in_payload"] == 0, "Single-level count differs")
                require({name for name, var in ds.variables.items() if len(var.dimensions) == 4} == {"z"},
                        "Unexpected extra four-dimensional field")
                row.update({"parent_id": job["parent_id"], "job_id": job["id"],
                            "payload": str(payload.relative_to(ROOT)), "member": member,
                            "payload_sha256": spec["payload_sha"], "payload_bytes": spec["bytes"],
                            "request_id": spec["request_id"]})
                found.append(row)
        require(len(found) == 1, "Missing/duplicate new geopotential member")
        rows.extend(found)
        controls = (("status", "WAITING_REMOTE"), ("remote_status", "accepted"),
                    ("request_id", "different-id"), ("plan_fingerprint", "wrong-plan"),
                    ("request_fingerprint", "0" * 64), ("remote_expected_bytes", spec["bytes"] + 1),
                    ("bytes", spec["bytes"] - 1), ("sha256", "0" * 64),
                    ("migrated_from", "unexpected"), ("migrated_from_sha256", "0" * 64),
                    ("job", {}))
        for key, value in controls:
            bad = copy.deepcopy(record)
            bad[key] = value
            try:
                validate_record(bad, job, field.PLAN_FP, spec)
            except ValueError as error:
                rejected.append({"job_id": job["id"], "changed_field": key,
                                 "expected_rejection": str(error)})
            else:
                raise ValueError("Negative record control accepted: " + key)
    require(len(rows) == 2 and len(rejected) == 22, "Added rows or negative control count differs")
    combined = copy.deepcopy(previous["checks"]) + rows
    coverage = set()
    for row in combined:
        keys = {(row["parent_id"], t) for t in row["utc"]}
        require(len(keys) == row["request_hours"] and not coverage.intersection(keys),
                "Duplicate parent/hour coverage")
        coverage.update(keys)
    expected = {(p["id"], t) for p in plan["parents"] for t in field.request_times(p["request"])}
    require(coverage.issubset(expected) and len(expected) == 22824, "Parent scope differs")
    require(len(combined) == 13 and len(coverage) == 5232 and
            sum(r["grid_values"] for r in combined) == 10464, "Combined count differs")
    independent = [independently_check(row) for row in combined]
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("test_audit_top1_available_fields_v001")
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == 12, "Field tests failed")
    for path, digest in sources.items():
        require(sha(path) == digest, "Source changed during audit: " + path)
    return {
        "status": "PASS_INCREMENTAL_TOP1_FIELDS_ONLY", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Eleven prior audited payloads plus August/September single-field Top1; z at 1 hPa only",
        "plan_fingerprint": field.PLAN_FP, "plan_sha256": field.PLAN_SHA,
        "prior_field_qa": "field_audit_007/qa.json", "added_job_ids": [s["job_id"] for s in ADDITIONS],
        "added_request_hours": 1416, "added_grid_values": 2832,
        "audited_payloads": 13, "audited_request_hours": 5232, "audited_grid_values": 10464,
        "full_top1_request_hours": 22824, "not_audited_request_hours": len(expected - coverage),
        "complete_top1_parent_months_in_snapshot": sum(
            {(p["id"], t) for t in field.request_times(p["request"])}.issubset(coverage) for p in plan["parents"]),
        "full_top1_field_qa_pass": False, "old_top10_complete": False, "overall_completion": False,
        "model_input_compatibility": compatibility["status"], "new_fit_calls": 0,
        "model_predictions": 0, "network_requests": 0, "smoke_included": False,
        "other_fields_and_levels_audited": False, "checks": combined,
        "independent_extraction": independent, "positive_negative_field_tests": result.testsRun,
        "field_test_log": log.getvalue(), "negative_record_controls": rejected,
        "source_sha256": sources, "source_files_unchanged_after_read": True,
        "completed_job_state_snapshots": [record for _, _, record, _ in inputs],
    }


def main():
    output, failure = OUT / "qa.json", OUT / "failure.json"
    require(not output.exists() and not failure.exists(), "Immutable audit already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        report = build_audit()
    except Exception as error:
        with failure.open("x", encoding="utf-8") as stream:
            json.dump({"status": "FAILED_INCREMENTAL_FIELD_AUDIT", "error_type": type(error).__name__,
                       "message": str(error), "auditor_sha256": sha(__file__),
                       "checked_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        raise
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "status", "audited_payloads", "audited_request_hours", "audited_grid_values",
        "positive_negative_field_tests", "model_input_compatibility")}, indent=2))


if __name__ == "__main__":
    main()
