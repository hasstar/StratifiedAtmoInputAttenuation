"""Offline snapshot audit: prior six payloads plus inherited April b001.

Adapted from v002; reuses frozen field checks and independent extraction without
calling either old main. Writes only immutable field_audit_003 evidence. No
downloader/model imports, network, interpolation, prediction or fitting.
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
OUT = DEST / "field_audit_003"
JOB_ID = "GBS_England_2004-04_pressure__b001"
REQUEST_ID = "ba565558-5cf8-4c6e-b298-45c0a235eb3f"
PAYLOAD_SHA = "3795d797be79491c2d5bf8a378155f50497cd4e6eaba384ce4b5ceb9b08304ff"
STATE_SHA = "1b81db3a52f02a99c547485691c035671e6f1430033913a9690a33ab637556d8"
PRIOR = DEST / "field_audit_002/qa.json"
FROZEN = {
    HERE / "audit_top1_available_fields_v001.py":
        "35682b28775e9c8a32e6c5c4da6986647a62463407fd7ced005f42528c989a1b",
    HERE / "audit_top1_incremental_fields_v002.py":
        "b827855dcfd625c247a22afaafee5adf21be550d5f6a620e94c53a918698250f",
    HERE / "test_audit_top1_available_fields_v001.py":
        "26e721c931c02c1ec9bda000e509832fbcc608b0a4df9e68c30353fe48e1bbbc",
    DEST / "plan.json": "265fc2dbddd0f7b8215aad9f4d99518fd1fc1dc6727000ee14536172aebf06d3",
    PRIOR: "e3654b8184c7c6c20b4123641a51b2f382fd503a38272895026b636f5d1ae87f",
    DEST / "field_audit_002/delivery_readback_qa.json":
        "23ac4fb59e6afc6640532353615a47fd923f098c2e3c7653e0db9b61a7854d64",
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
    encoded = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_record(record, job, adoption, plan_fp):
    require(record["status"] == "VERIFIED" and record["remote_status"] == "successful",
            "Record is not a completed verified download")
    require(record["plan_fingerprint"] == plan_fp, "Plan fingerprint differs")
    require(record["job"] == job == adoption["original_job"], "Frozen inherited job differs")
    require(record["request_fingerprint"] == fingerprint(job), "Request fingerprint differs")
    require(record["request_id"] == adoption["request_id"] == REQUEST_ID,
            "Inherited request ID differs")
    require(record["migrated_from"] == adoption["source_state"] and
            record["migrated_from_sha256"] == adoption["source_state_sha256"],
            "Migration provenance differs")
    require(record["bytes"] == record["remote_expected_bytes"] == 84676,
            "Official expected size differs")
    require(record["sha256"] == PAYLOAD_SHA, "Completed payload SHA differs")


def build_audit():
    # Verify imported read-only dependencies before loading their definitions.
    for path, digest in FROZEN.items():
        require(sha(path) == digest, "Frozen source differs: " + str(path))
    import audit_top1_available_fields_v001 as field
    from audit_top1_incremental_fields_v002 import independently_check

    plan = field.read(DEST / "plan.json")
    require(plan["fingerprint"] == field.PLAN_FP and
            plan["feature_names"] == ["p1__geopotential"], "Plan scope differs")
    previous = field.read(PRIOR)
    require(previous["status"] == "PASS_INCREMENTAL_TOP1_FIELDS_ONLY" and
            previous["audited_payloads"] == 6 and previous["audited_request_hours"] == 936 and
            previous["audited_grid_values"] == 1872, "Previous snapshot differs")
    compatibility = field.read(DEST / "input_compatibility_001.json")
    require(compatibility["status"] == "BLOCKED_SPATIAL_INPUT_DEFINITION" and
            compatibility["new_fit_calls"] == compatibility["model_predictions"] == 0,
            "Pending compatibility gate changed; reassess before using this snapshot")
    job = next(j for j in plan["jobs"] if j["id"] == JOB_ID)
    adoption = next(j for j in plan["adopted"] if j["id"] == JOB_ID)
    state_path = DEST / "jobs" / (JOB_ID + ".json")
    require(sha(state_path) == STATE_SHA, "Completed state snapshot changed")
    record = field.read(state_path)
    validate_record(record, job, adoption, field.PLAN_FP)
    payload = (DEST / job["target"]).resolve()
    require(payload.is_relative_to((DEST / "payloads").resolve()), "Payload escaped new outputs")
    require(payload.stat().st_size == record["bytes"] and sha(payload) == PAYLOAD_SHA,
            "Actual payload SHA/size differs")
    sources = dict(previous["source_sha256"])
    sources.update({str(p.resolve()): h for p, h in FROZEN.items()})
    for path, digest in plan["protected_files"].items():
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        sources[str(resolved.resolve())] = digest
    sources.update({str(state_path): STATE_SHA, str(payload): PAYLOAD_SHA,
                    str(Path(__file__).resolve()): sha(__file__)})
    for path, digest in sources.items():
        require(sha(path) == digest, "Source SHA differs: " + path)

    rows = []
    for member, ds in field.datasets(payload):
        if "z" in ds.variables:
            row = field.inspect_dataset(ds, job["request"])
            require(row["request_hours"] == job["expected_hours"] == 168, "New hour count differs")
            require(row["grid_values"] == 336, "New field count differs")
            row.update({"parent_id": job["parent_id"], "job_id": JOB_ID,
                        "payload": str(payload.relative_to(ROOT)), "member": member,
                        "payload_sha256": PAYLOAD_SHA, "payload_bytes": record["bytes"],
                        "request_id": REQUEST_ID})
            rows.append(row)
    require(len(rows) == 1, "Missing/duplicate new geopotential member")
    combined = copy.deepcopy(previous["checks"]) + rows
    coverage = set()
    for row in combined:
        keys = {(row["parent_id"], t) for t in row["utc"]}
        require(len(keys) == row["request_hours"] and not coverage.intersection(keys),
                "Duplicate parent/hour coverage")
        coverage.update(keys)
    expected = {(p["id"], t) for p in plan["parents"] for t in field.request_times(p["request"])}
    require(coverage.issubset(expected) and len(expected) == 22824, "Parent scope differs")
    require(len(combined) == 7 and len(coverage) == 1104 and
            sum(r["grid_values"] for r in combined) == 2208, "Combined count differs")
    independent = [independently_check(row) for row in combined]
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("test_audit_top1_available_fields_v001")
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == 12, "Field tests failed")
    controls = (("status", "WAITING_REMOTE"), ("remote_status", "accepted"),
                ("request_id", "different-id"), ("plan_fingerprint", "wrong-plan"),
                ("request_fingerprint", "0" * 64), ("remote_expected_bytes", 84677),
                ("sha256", "0" * 64), ("migrated_from", "different-source"),
                ("migrated_from_sha256", "0" * 64), ("job", {}))
    rejected = []
    for key, value in controls:
        bad = copy.deepcopy(record)
        bad[key] = value
        try:
            validate_record(bad, job, adoption, field.PLAN_FP)
        except ValueError as error:
            rejected.append({"changed_field": key, "expected_rejection": str(error)})
        else:
            raise ValueError("Negative record control accepted: " + key)
    for path, digest in sources.items():
        require(sha(path) == digest, "Source changed during audit: " + path)
    return {
        "status": "PASS_INCREMENTAL_TOP1_FIELDS_ONLY", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Prior six audited payloads plus inherited April b001; z at 1 hPa only",
        "plan_fingerprint": field.PLAN_FP, "plan_sha256": field.PLAN_SHA,
        "prior_field_qa": "field_audit_002/qa.json", "added_job_ids": [JOB_ID],
        "added_request_hours": 168, "added_grid_values": 336,
        "audited_payloads": 7, "audited_request_hours": 1104, "audited_grid_values": 2208,
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
        "completed_job_state_snapshot": record,
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
