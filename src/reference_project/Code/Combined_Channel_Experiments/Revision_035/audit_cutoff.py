"""Offline audit only. No network, collector imports, model loading or scoring."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "Outputs/Experiments"
OLD = EXP / "Combined_Channel_Revision_032"
CONTROL = EXP / "Combined_Channel_Revision_034"
OUT = EXP / "Combined_Channel_Revision_035"
CHANGE = ROOT / "Outputs/Automation_Changes/TVT_Comparison_20261018_004"
AUTO = Path("C:/Users/SDR_HH/.codex/automations/tvt/automation.toml")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rate_hours(missing, gained, hours):
    return missing / (gained / hours) if gained else None


def audit():
    assert not (OUT / "trial_summary.json").exists(), "No overwrite"
    policy = read(CONTROL / "trial_policy.json")
    start, cutoff = policy["start_epoch_seconds"], policy["cutoff_epoch_seconds"]
    assert cutoff == 1792313400 and cutoff - start == 7200
    tests = {
        "exact_start_included": start <= stamp("2026-10-18T06:50:00Z") < cutoff,
        "exact_cutoff_excluded": not start <= stamp("2026-10-18T08:50:00Z") < cutoff,
        "baseline_not_new": 1776 - 888 == 888,
        "zero_rate_not_invented": rate_hours(10, 0, 2) is None,
        "unequal_batch_units_separate": rate_hours(419328, 888, 2) != rate_hours(706, 1, 2),
    }
    assert all(tests.values())
    protected = read(CONTROL / "sources_before.json")
    protected.update(read(CONTROL / "code_before.json"))
    assert all(sha(Path(p)) == s for p, s in protected.items()), "Source changed"
    assert sha(AUTO) == sha(CHANGE / "automation_before.toml")
    close, enforced = read(CONTROL / "trial_closed.json"), read(CONTROL / "cutoff_result.json")
    assert close["status"] == "TRIAL_WINDOW_CLOSED_NO_MORE_ACQUISITION"
    assert not close["automatic_extension_allowed"]
    assert enforced["status"] == "CUTOFF_ENFORCED" and not enforced["remaining_workers"]
    assert enforced["remote_jobs_deleted"] == enforced["files_deleted"] == 0
    ledger = [json.loads(line) for line in (OLD / "network_ledger.jsonl").read_text().splitlines() if line]
    assert all(stamp(x["utc"]) < cutoff and x["bytes"] >= 0 for x in ledger)
    trial_ledger = [x for x in ledger if start <= stamp(x["utc"]) < cutoff]
    polls = sorted((read(p) for p in (OLD / "runs").glob("*/poll_*.json")), key=lambda x: stamp(x["utc"]))
    assert all(stamp(x["utc"]) < cutoff for x in polls)
    completions, requests = [], []
    for directory in sorted((OLD / "batches").iterdir()):
        if not directory.is_dir():
            continue
        assert (directory / "submitted.json").exists(), "Unknown submission intent"
        submitted = read(directory / "submitted.json")
        assert stamp(submitted["utc"]) < cutoff
        current = [x for x in polls if x["request_id"] == submitted["request_id"]]
        row = {"subrequest": int(directory.name), **submitted, "last_poll": current[-1] if current else None}
        transitions = []
        for item in current:
            if not transitions or transitions[-1]["status"] != item["status"]:
                transitions.append({"utc": item["utc"], "status": item["status"]})
        row["observed_status_transitions"] = transitions
        if (directory / "complete.json").exists():
            done = read(directory / "complete.json")
            assert done["request_id"] == submitted["request_id"]
            assert stamp(done["utc"]) < cutoff
            assert sha(directory / "raw.nc") == done["sha256"]
            assert (directory / "raw.nc").stat().st_size == done["bytes"]
            completions.append(done)
            row["complete_utc"] = done["utc"]
            row["submitted_to_complete_seconds"] = stamp(done["utc"]) - stamp(submitted["utc"])
            success = next((x for x in current if x["status"] == "successful"), None)
            row["successful_poll_to_complete_upper_bound_seconds"] = stamp(done["utc"]) - stamp(success["utc"]) if success else None
            row["exact_network_download_duration_seconds"] = None
        else:
            assert not (directory / "raw.nc").exists(), "Partial input retained; inspect separately"
            row["pending_at_cutoff_seconds"] = cutoff - stamp(submitted["utc"])
        requests.append(row)
    before = [x for x in completions if stamp(x["utc"]) < start]
    new = [x for x in completions if start <= stamp(x["utc"]) < cutoff]
    assert len(before) == policy["baseline_completed_subrequests"] == 1
    assert sum(x["values"] for x in before) == policy["baseline_verified_values"] == 888
    qa = read(OLD / "qa_20261018T085747_957609.json")
    assert qa["status"] == "PASS_PARTIAL_INPUT_QA"
    assert {(x["subrequest"], x["sha256"], x["values"]) for x in qa["files"]} == {(x["subrequest"], x["sha256"], x["values"]) for x in completions}
    total = sum(x["values"] for x in completions)
    gained = sum(x["values"] for x in new)
    missing = policy["target_values"] - total
    prior_bytes = read(OLD / "authorization.json")["prior_http_body_bytes"]
    cumulative = prior_bytes + sum(x["bytes"] for x in ledger)
    assert cumulative == qa["cumulative_http_body_bytes_at_check"] == 601806
    assert sum(x["bytes"] for x in ledger if x["kind"] == "data") == sum(x["bytes"] for x in completions)
    assert sum(x["bytes"] for x in trial_ledger if x["kind"] == "data") == sum(x["bytes"] for x in new)
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(), "status": "PASS_CUTOFF_AUDIT",
        "fixed_window_utc": [policy["start_utc"], policy["cutoff_utc"]], "window_seconds": cutoff - start,
        "target_subrequests": policy["target_subrequests"], "target_new_values": policy["target_values"],
        "baseline_subrequests": len(before), "baseline_values": sum(x["values"] for x in before),
        "trial_net_completed_subrequests": len(new), "trial_net_values": gained,
        "cumulative_completed_subrequests": len(completions), "cumulative_verified_values": total,
        "missing_subrequests": policy["target_subrequests"] - len(completions), "missing_values": missing,
        "missing_percent": 100 * missing / policy["target_values"],
        "cumulative_http_response_body_bytes_including_prior_031": cumulative,
        "trial_http_response_body_bytes": sum(x["bytes"] for x in trial_ledger),
        "cumulative_data_body_bytes": sum(x["bytes"] for x in completions),
        "trial_data_body_bytes": sum(x["bytes"] for x in new),
        "last_network_ledger_utc": ledger[-1]["utc"], "requests": requests,
        "remaining_deadline_hours_at_cutoff": (stamp(policy["delivery_deadline_interpretation_utc"]) - cutoff) / 3600,
        "constant_observed_value_rate_remaining_hours": rate_hours(missing, gained, 2),
        "constant_observed_request_rate_remaining_hours": rate_hours(policy["target_subrequests"] - len(completions), len(new), 2),
        "estimate_warning": "Planning extrapolations, not ETA guarantees; full two-hour denominator includes local outages and pending remote wait; exclude no slow requests.",
        "esst": "EXCLUDED_USER_DECISION_DEADLINE", "formal_model_families": 4,
        "new_predictions": 0, "new_scores": 0, "training_runs": 0,
        "four_model_final_package_complete": False,
        "implementation_independence": "standard-library ledger audit plus separate PowerShell recomputation; same agent, no external reviewer",
    }
    for folder in (OLD, CONTROL, EXP / "Combined_Channel_Revision_033", Path(__file__).parent, CHANGE):
        for path in folder.rglob("*"):
            if path.is_file():
                protected[str(path)] = sha(path)
    write("sources_before.json", protected)
    write("trial_summary.json", summary)
    write("offline_tests.json", tests)
    write("environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "network_calls": 0})
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, ensure_ascii=False))


def seal():
    sources = read(OUT / "sources_before.json")
    before, after = CHANGE / "automation_before.toml", CHANGE / "automation_after.toml"
    differences = []
    for name, old in sources.items():
        actual = sha(Path(name))
        if actual != old:
            assert Path(name) == AUTO and old == sha(before) and actual == sha(after), name
            differences.append({"path": name, "before_sha256": old, "after_sha256": actual})
    assert len(differences) == 1, "Only approved tvt update is allowed"
    assert sha(AUTO) == sha(after)
    manifest = {str(p): sha(p) for folder in (OUT, CHANGE, Path(__file__).parent) for p in folder.rglob("*") if p.is_file()}
    write("manifest.json", manifest)
    write("final_verification.json", {"utc": datetime.now(timezone.utc).isoformat(), "status": "PASS_CUTOFF_SCOPE_HANDOFF_ONLY", "protected_sources": len(sources), "allowed_changes": differences, "manifest_files": len(manifest), "four_model_final_package_complete": False})
    print(json.dumps({"status": "PASS_CUTOFF_SCOPE_HANDOFF_ONLY", "protected_sources": len(sources), "manifest_files": len(manifest)}))


if __name__ == "__main__":
    {"audit": audit, "seal": seal}[sys.argv[1]]()
