"""Date-only re-batching of frozen ERA5 requests. No model imports or training.

User authorization: 2026-10-14, change download batching only and finish weather.
Original plan, scripts, payloads, job records, and failure audits remain read-only.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXTERNAL = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external"
DEST = EXTERNAL / "weather_v002"
PLAN = DEST / "plan.json"
LOCK = EXTERNAL / "weather_writer.lock"
ORIGINAL_FP = "c7fa6ff824fb436477f534fcec022ef7d60aed6afd4ae507f9dd1575e0282e16"
ORIGINAL_SHA = "850cff67d9d04048bdfb09a02f805ba7aafabc47e697e22adbe57e9626a8d917"
FROZEN = {
    HERE / "weather_download.py": "8a6b4b7c28e2b61e6104baa86bee0528e9e125cce2f7836bf05c5799cb410c52",
    HERE / "external_prepare.py": "63fb3025aba2f7fc6e0c62f7eeb242aa9786f3a91b3c1d0cfd40bafb475a8a7a",
    HERE / "verify_weather_v002.py": "4878bca90615e004ff083e9715ef34ccd597e7d09b8c6a16f1432182be041c53",
    EXTERNAL / "weather_smoke_unit_qa.json": "74972e0e631f420e8963a0208789210c31ed0b3ac3e8d870710d7c7ef2a0360e",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic(path, obj):
    path = Path(path).resolve()
    require(path.is_relative_to(DEST.resolve()), "Write outside v002 denied")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def immutable(path, obj):
    if path.exists():
        require(read(path) == obj, "Immutable v002 artifact differs: " + path.name)
    else:
        atomic(path, obj)


def config():
    import yaml
    obj = yaml.safe_load((Path.home() / ".cdsapirc").read_text(encoding="utf-8"))
    require(isinstance(obj, dict) and obj.get("key"), "Existing CDS key absent; user action required")
    url = str(obj.get("url", "https://cds.climate.copernicus.eu/api")).rstrip("/")
    require(urlparse(url).scheme == "https" and urlparse(url).hostname == "cds.climate.copernicus.eu",
            "Unexpected credential endpoint")
    return (url[:-3] if url.endswith("/api/v2") else url), str(obj["key"])


def client():
    sys.path.insert(0, str(ROOT / "Code/Dataset_Build/_vendor_cdsapi"))
    from ecmwf.datastores import Client
    url, key = config()
    logging.getLogger("ecmwf.datastores").setLevel(logging.ERROR)
    logging.getLogger("multiurl").setLevel(logging.ERROR)
    # No implicit POST retries: a lost submit response must not create duplicates.
    return Client(url=url, key=key, verify=True, timeout=60, progress=False,
                  sleep_max=30, retry_after=30, maximum_tries=1, cleanup=False)


def safe_error(exc):
    message = str(exc)
    try:
        message = message.replace(config()[1], "[REDACTED]")
    except Exception:
        pass
    # Signed asset URLs must not become durable logs either.
    import re
    message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", message)
    return {"error_type": type(exc).__name__, "message": message[:2500]}


def too_large(message):
    message = str(message).lower()
    return "cost limits exceeded" in message or "request is too large" in message


def failure(kind, node=None, exc=None, **details):
    record = {"at_utc": now(), "kind": kind, "job_id": node["id"] if node else None,
              **details, **(safe_error(exc) if exc else {})}
    path = DEST / "failures" / (str(time.time_ns()) + ".json")
    atomic(path, record)
    return str(path.relative_to(DEST))


def child(node, days, suffix):
    result = copy.deepcopy(node)
    result["id"] = node["id"] + suffix
    result["request"]["day"] = list(days)
    result["expected_hours"] = len(days) * len(result["request"]["time"])
    result["target"] = "payloads/" + result["id"] + ".download"
    return result


def partition_check(parent, children):
    require(bool(children), "Empty date partition")
    days = []
    for node in children:
        require(node["parent_id"] == parent["parent_id"], "Parent lineage differs")
        require(node["collection_id"] == parent["collection_id"], "Collection changed")
        require({k: v for k, v in node["request"].items() if k != "day"} ==
                {k: v for k, v in parent["request"].items() if k != "day"}, "Non-date request changed")
        require(node["request"]["day"], "Empty day list")
        require(node["expected_hours"] == len(node["request"]["day"]) * len(node["request"]["time"]),
                "Wrong hour count")
        days.extend(node["request"]["day"])
    require(Counter(days) == Counter(parent["request"]["day"]), "Missing or duplicated date")


def build_plan():
    require(sha(EXTERNAL / "weather_plan.json") == ORIGINAL_SHA, "Original plan SHA changed")
    old = read(EXTERNAL / "weather_plan.json")
    require(old["fingerprint"] == ORIGINAL_FP and len(old["jobs"]) == 77, "Original plan differs")
    for path, expected in FROZEN.items():
        require(sha(path) == expected, "Frozen dependency changed: " + path.name)
    reused, nodes = [], []
    for original in old["jobs"]:
        state_path = EXTERNAL / "weather_jobs" / (original["id"] + ".json")
        if state_path.exists():
            saved = read(state_path)
            require(saved["request_fingerprint"] == fingerprint(original), "Old request changed")
            require(saved["status"] == "VERIFIED", "Unresolved original remote ID; reconcile before batching")
            require(sha(EXTERNAL / original["target"]) == saved["sha256"], "Old payload changed")
            reused.append({"parent_id": original["id"], "state": str(state_path.relative_to(ROOT)),
                           "state_sha256": sha(state_path), "target": original["target"],
                           "sha256": saved["sha256"], "request_id": saved["request_id"]})
            continue
        node = {**copy.deepcopy(original), "parent_id": original["id"]}
        days = original["request"]["day"]
        # Seven is a conservative operational batch, NOT an asserted CDS limit.
        cap = 7 if "pressure_level" in original["request"] else len(days)
        chunks = [child(node, days[i:i + cap], "__b%03d" % (i // cap + 1))
                  for i in range(0, len(days), cap)]
        partition_check(node, chunks)
        nodes.extend(chunks)
    # Snapshot all original external artifacts, including every failure and payload.
    protected = {str(p.relative_to(ROOT)): sha(p) for p in EXTERNAL.rglob("*")
                 if p.is_file() and not p.is_relative_to(DEST) and p != LOCK}
    protected.update({str(p.relative_to(ROOT)): value for p, value in FROZEN.items()})
    plan = {"version": 2, "original_plan_fingerprint": ORIGINAL_FP, "original_plan_sha256": ORIGINAL_SHA,
            "authorization": "2026-10-14: only adjust download batching; finish external weather; zero training",
            "code_sha256": sha(Path(__file__)), "test_sha256": sha(HERE / "test_weather_download_v002.py"),
            "max_in_flight_requests": 2, "initial_pressure_date_cap": 7,
            "fallback": "Only explicit request-too-large rejection permits balanced date bisection, minimum one date",
            "reused": reused, "jobs": nodes, "original_parent_jobs": 77,
            "protected_files": protected, "model_training_started": False}
    plan["fingerprint"] = fingerprint(plan)
    return plan


def load_plan(check_protected=True):
    plan = read(PLAN)
    require(plan["fingerprint"] == fingerprint({k: v for k, v in plan.items() if k != "fingerprint"}),
            "v002 plan fingerprint changed")
    require(plan["code_sha256"] == sha(Path(__file__)), "Frozen v002 code changed")
    require(plan["test_sha256"] == sha(HERE / "test_weather_download_v002.py"), "Frozen v002 tests changed")
    if check_protected:
        for path, expected in plan["protected_files"].items():
            require(sha(ROOT / path) == expected, "Protected original changed: " + path)
    return plan


def state_path(node):
    return DEST / "jobs" / (node["id"] + ".json")


def state(node):
    path = state_path(node)
    saved = read(path) if path.exists() else {}
    require(not saved or saved["request_fingerprint"] == fingerprint(node), "v002 request changed")
    return saved


def save(node, saved, **updates):
    saved = {**saved, "job": node, "request_fingerprint": fingerprint(node), **updates}
    atomic(state_path(node), saved)
    return saved


def split_rejected(node, saved, evidence):
    days = node["request"]["day"]
    if len(days) == 1:
        save(node, saved, status="BLOCKED_SIZE_ONE_DATE", failure=evidence)
        return
    half = len(days) // 2
    children = [child(node, days[:half], "__a"), child(node, days[half:], "__b")]
    partition_check(node, children)
    save(node, saved, status="SPLIT_SIZE_REJECTION", children=children, failure=evidence)
    print("WEATHER DATE SPLIT", node["id"], [x["request"]["day"] for x in children], flush=True)


def leaves(plan):
    result = []
    def visit(node):
        saved = state(node)
        if saved.get("status") == "SPLIT_SIZE_REJECTION":
            partition_check(node, saved["children"])
            for item in saved["children"]:
                visit(item)
        else:
            result.append(node)
    for node in plan["jobs"]:
        visit(node)
    require(len({n["id"] for n in result}) == len(result), "Duplicate runtime leaf ID")
    return result


def attrs(variable):
    values = {}
    for name in variable.ncattrs():
        value = variable.getncattr(name)
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            value = str(value)
        values[name] = value
    return values


def inspect_file(path):
    import netCDF4
    files = []
    def inspect(ds, member):
        files.append({"member": member, "dimensions": {n: len(d) for n, d in ds.dimensions.items()},
                      "variables": {n: {"shape": list(v.shape), "attributes": attrs(v)}
                                    for n, v in ds.variables.items()}})
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe archive member")
                if name.endswith(".nc"):
                    with netCDF4.Dataset("memory.nc", memory=archive.read(name)) as ds:
                        inspect(ds, name)
    else:
        with netCDF4.Dataset(path) as ds:
            inspect(ds, None)
    require(files, "No readable NetCDF members")
    return {"netcdf_members": len(files), "files": files}


def download_result(node, saved, result):
    target = (DEST / node["target"]).resolve()
    require(target.is_relative_to((DEST / "payloads").resolve()), "Payload path outside v002")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = result.content_length
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        # Diagnose and retain an interrupted payload before any retry; never delete it.
        diagnosis = {"request_id": saved["request_id"], "bytes": temporary.stat().st_size,
                     "sha256": sha(temporary), "remote_expected_bytes": expected}
        try:
            inspect_file(temporary)
            diagnosis["readable"] = True
        except Exception as exc:
            diagnosis.update(readable=False, inspection_error=safe_error(exc))
        if diagnosis["readable"] and diagnosis["bytes"] == expected and not target.exists():
            failure("PARTIAL_DIAGNOSED_COMPLETE", node, **diagnosis)
            temporary.rename(target)
        else:
            retained = DEST / "partial_history" / (node["id"] + "__" + str(time.time_ns()) + ".part")
            require(temporary.is_relative_to(DEST.resolve()) and retained.resolve().is_relative_to(DEST.resolve()),
                    "Unsafe partial move")
            retained.parent.mkdir(parents=True, exist_ok=True)
            failure("PARTIAL_RETAINED_REDOWNLOAD_SAME_ID", node,
                    retained_path=str(retained.relative_to(DEST)), **diagnosis)
            temporary.rename(retained)
    if not target.exists():
        result.download(str(temporary))
        require(temporary.stat().st_size == expected, "Downloaded size differs from remote")
        inspection = inspect_file(temporary)
        temporary.rename(target)
    else:
        require(target.stat().st_size == expected, "Existing final payload size differs")
        inspection = inspect_file(target)
    saved = save(node, saved, status="VERIFIED", downloaded_utc=now(), sha256=sha(target),
                 bytes=target.stat().st_size, remote_expected_bytes=expected, inspection=inspection,
                 remote_status="successful", retry_count=0)
    print("WEATHER VERIFIED", node["id"], saved["bytes"], flush=True)


def submit(node, plan, c):
    # Durable intent closes the crash window between remote creation and ID persistence.
    saved = save(node, {}, status="SUBMITTING", plan_fingerprint=plan["fingerprint"], started_utc=now())
    try:
        remote = c.submit(node["collection_id"], node["request"])
    except Exception as exc:
        evidence = failure("SUBMIT_FAILED", node, exc)
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
        if code in (400, 403, 413) and too_large(str(exc)):
            split_rejected(node, saved, evidence)
        elif code is not None and 400 <= code < 500:
            save(node, saved, status="BLOCKED_HTTP", http_status=code, failure=evidence)
        else:
            save(node, saved, status="AMBIGUOUS_SUBMISSION", failure=evidence)
        print("WEATHER SUBMIT FAILURE", node["id"], state(node)["status"], flush=True)
        return
    # Accessing request_id is local; persist before any subsequent remote query/download.
    save(node, saved, status="SUBMITTED", request_id=remote.request_id, submitted_utc=now(), key_saved=False)
    print("WEATHER SUBMITTED", node["id"], remote.request_id, flush=True)


def poll(node, c):
    saved = state(node)
    if time.time() < saved.get("retry_after_epoch", 0):
        return
    try:
        remote = c.get_remote(saved["request_id"])
        require(remote.collection_id == node["collection_id"], "Recovered collection differs")
        remote_status = remote.status
        saved = save(node, saved, last_checked_utc=now(), remote_status=remote_status)
        if remote_status in ("accepted", "running"):
            save(node, saved, status="WAITING_REMOTE", retry_count=0, retry_after_epoch=0)
        elif remote_status == "successful":
            download_result(node, saved, remote.get_results())
        elif remote_status in ("failed", "rejected", "dismissed", "deleted"):
            message = "Remote state " + remote_status
            if remote_status in ("failed", "rejected"):
                try:
                    remote.get_results()
                except Exception as exc:
                    message = safe_error(exc)["message"]
            evidence = failure("REMOTE_TERMINAL", node, request_id=saved["request_id"],
                               remote_status=remote_status, message=message)
            if remote_status in ("failed", "rejected") and too_large(message):
                split_rejected(node, saved, evidence)
            else:
                save(node, saved, status="BLOCKED_REMOTE", failure=evidence)
        else:
            raise RuntimeError("Unknown remote status: " + remote_status)
    except Exception as exc:
        evidence = failure("POLL_OR_DOWNLOAD_FAILED_SAME_ID_RETAINED", node, exc,
                           request_id=saved["request_id"])
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in (401, 403, 404, 410):
            save(node, saved, status="BLOCKED_REMOTE_ACCESS", failure=evidence, http_status=code)
        else:
            retries = saved.get("retry_count", 0) + 1
            delay = min(900, 60 * (2 ** min(retries - 1, 4)))
            save(node, saved, status="RETRY_SAME_ID", failure=evidence, retry_count=retries,
                 retry_after_epoch=time.time() + delay)
        print("WEATHER RETAIN ID", node["id"], state(node)["status"], flush=True)


def lock_held():
    import msvcrt
    with LOCK.open("r+b") as f:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    return False


@contextlib.contextmanager
def writer(command):
    import msvcrt
    # Use the SAME OS byte lock as v001, but never rewrite the old worker record.
    with LOCK.open("r+b") as f:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        active = {"pid": os.getpid(), "python": sys.executable, "argv": sys.argv,
                  "command": command, "status": "RUNNING", "started_utc": now()}
        worker_path = DEST / "workers" / (str(time.time_ns()) + ".json")
        atomic(worker_path, active)
        atomic(DEST / "active_worker.json", active)
        try:
            yield
        except BaseException as exc:
            evidence = failure("WORKER_FAILED", exc=exc)
            active.update(status="FAILED", failure=evidence)
            raise RuntimeError(json.dumps(safe_error(exc), ensure_ascii=False)) from None
        else:
            active["status"] = "EXITED_NORMAL"
        finally:
            active["finished_utc"] = now()
            atomic(worker_path, active)
            atomic(DEST / "active_worker.json", active)
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def progress(plan, verify=False):
    nodes = leaves(plan)
    states = [(node, state(node)) for node in nodes]
    done = [(n, s) for n, s in states if s.get("status") == "VERIFIED"]
    if verify:
        for node, saved in done:
            require(sha(DEST / node["target"]) == saved["sha256"], "Verified v002 payload changed")
    original = read(EXTERNAL / "weather_plan.json")
    covered = Counter({item["parent_id"]: next(j["expected_hours"] for j in original["jobs"]
                                            if j["id"] == item["parent_id"]) for item in plan["reused"]})
    for node, _ in done:
        covered[node["parent_id"]] += node["expected_hours"]
    complete = sum(covered[j["id"]] == j["expected_hours"] for j in original["jobs"])
    pending = [{"id": n["id"], "request_id": s.get("request_id"), "status": s.get("status"),
                "remote_status": s.get("remote_status"), "last_checked_utc": s.get("last_checked_utc")}
               for n, s in states if s and s.get("status") != "VERIFIED"]
    return {"plan_fingerprint": plan["fingerprint"], "original_parent_total": 77,
            "original_parent_complete": complete, "reused_original_payloads": len(plan["reused"]),
            "initial_new_batches": len(plan["jobs"]), "runtime_leaf_batches": len(nodes),
            "new_batches_verified": len(done), "remaining_unsubmitted": sum(not s for _, s in states),
            "original_expected_request_hours": sum(j["expected_hours"] for j in original["jobs"]),
            "covered_request_hours": sum(covered.values()), "pending_or_blocked": pending,
            "smoke_reused_not_counted_as_formal": 2, "model_training_started": False}


def run():
    plan = load_plan()
    qa = read(DEST / "preflight_qa.json")
    require(qa["status"] == "PASS" and qa["code_sha256"] == plan["code_sha256"] and
            qa["test_sha256"] == plan["test_sha256"], "Matching offline preflight absent")
    progress(plan, verify=True)
    c = client()
    while True:
        nodes = leaves(plan)
        # Resume every saved ID before considering a new submission.
        for node in nodes:
            saved = state(node)
            if saved.get("request_id") and saved.get("status") != "VERIFIED" and not saved.get("status", "").startswith("BLOCKED"):
                poll(node, c)
        nodes = leaves(plan)
        states = [(n, state(n)) for n in nodes]
        ambiguous = [n for n, s in states if s.get("status") in ("SUBMITTING", "AMBIGUOUS_SUBMISSION")]
        require(not ambiguous, "Submission outcome unknown; reconcile remote jobs before any new submission")
        require(not any(s.get("status") == "BLOCKED_REMOTE_ACCESS" for _, s in states),
                "Remote accessibility unknown; do not release an unconfirmed active slot")
        require(not any(s.get("status") == "BLOCKED_HTTP" for _, s in states),
                "Definitive non-size HTTP failure requires diagnosis before new submissions")
        active = [n for n, s in states if s.get("request_id") and s.get("status") != "VERIFIED"
                  and not s.get("status", "").startswith("BLOCKED")]
        require(len(active) <= 2, "More than two active remote IDs")
        for node, saved in states:
            if len(active) >= 2:
                break
            if not saved:
                submit(node, plan, c)
                current = state(node)
                require(current["status"] != "AMBIGUOUS_SUBMISSION", "Submit response ambiguous; reconcile before retry")
                require(current["status"] != "BLOCKED_HTTP", "Non-size HTTP rejection; diagnose before continuing")
                if current.get("request_id"):
                    active.append(node)
        report = progress(plan)
        atomic(DEST / "progress.json", {"checked_utc": now(), **report})
        if report["original_parent_complete"] == 77:
            load_plan()  # Recheck all original fingerprints at the completion boundary.
            progress(plan, verify=True)
            atomic(DEST / "download_qa.json", {"status": "PASS_DOWNLOAD_READABILITY_ONLY", **report,
                   "completed_utc": now(), "full_field_coordinate_UTC_unit_count_QA_complete": False})
            print("WEATHER ALL ORIGINAL COVERAGE DOWNLOADED", flush=True)
            return
        if not active and report["remaining_unsubmitted"] == 0:
            raise RuntimeError("Unresolved blocked weather jobs; see preserved failure audits")
        time.sleep(30)


def status():
    plan = load_plan()
    return {**progress(plan, verify=True), "os_lock_held": lock_held(),
            "active_worker": read(DEST / "active_worker.json") if (DEST / "active_worker.json").exists() else None,
            "download_qa": read(DEST / "download_qa.json") if (DEST / "download_qa.json").exists() else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "status"))
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2), flush=True)
    else:
        with writer(args.command):
            if args.command == "prepare":
                plan = build_plan()
                immutable(PLAN, plan)
                print(json.dumps(progress(plan), indent=2), flush=True)
            else:
                run()
