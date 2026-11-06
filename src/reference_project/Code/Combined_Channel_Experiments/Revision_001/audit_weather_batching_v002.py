"""Independent, offline date-partition audit. Does not import the downloader."""
import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external"
DEST = EXT / "weather_v002"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    old = load(EXT / "weather_plan.json")
    new = load(DEST / "plan.json")
    assert len(old["jobs"]) == 77
    assert digest(EXT / "weather_plan.json") == new["original_plan_sha256"]
    canonical = json.dumps({k: v for k, v in new.items() if k != "fingerprint"},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == new["fingerprint"]
    assert digest(HERE / "weather_download_v002.py") == new["code_sha256"]
    assert digest(HERE / "test_weather_download_v002.py") == new["test_sha256"]
    assert new["max_in_flight_requests"] == 2
    reused = {n["parent_id"]: n for n in new["reused"]}
    parents = {n["id"]: n for n in old["jobs"]}
    assert len(reused) == 4 and len(new["jobs"]) == 196
    assert set(reused).isdisjoint(n["parent_id"] for n in new["jobs"])
    assert set(reused) | {n["parent_id"] for n in new["jobs"]} == set(parents)
    assert len({n["id"] for n in new["jobs"]}) == len(new["jobs"])
    hours = 0
    for parent_id, parent in parents.items():
        if parent_id in reused:
            item = reused[parent_id]
            assert digest(ROOT / item["state"]) == item["state_sha256"]
            saved = load(ROOT / item["state"])
            assert saved["job"] == parent and saved["status"] == "VERIFIED"
            assert digest(EXT / item["target"]) == item["sha256"] == saved["sha256"]
            hours += parent["expected_hours"]
            continue
        batch_days = []
        for node in (n for n in new["jobs"] if n["parent_id"] == parent_id):
            assert node["collection_id"] == parent["collection_id"]
            assert node["request"].keys() == parent["request"].keys()
            for key in parent["request"]:
                if key != "day":
                    assert node["request"][key] == parent["request"][key], (parent_id, key)
            assert node["request"]["day"]
            assert node["expected_hours"] == len(node["request"]["day"]) * len(node["request"]["time"])
            batch_days.extend(node["request"]["day"])
            hours += node["expected_hours"]
            assert (DEST / node["target"]).resolve().is_relative_to((DEST / "payloads").resolve())
        assert Counter(batch_days) == Counter(parent["request"]["day"]), parent_id
    assert hours == sum(n["expected_hours"] for n in old["jobs"]) == 45663
    for name, expected in new["protected_files"].items():
        assert digest(ROOT / name) == expected, name
    tree = ast.parse((HERE / "weather_download_v002.py").read_text(encoding="utf-8"))
    prohibited = {"torch", "sklearn", "local_engine", "external_prepare", "fit_marginals", "run_dynamics"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not prohibited.intersection(n.name.split(".")[0] for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in prohibited
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"fit", "partial_fit", "fit_transform", "backward", "train"}
    result = {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "auditor_sha256": digest(Path(__file__)), "plan_fingerprint": new["fingerprint"],
              "plan_sha256": digest(DEST / "plan.json"), "original_parent_jobs": 77,
              "initial_new_batches": 196, "reused_original_payloads": 4,
              "request_hours_preserved": hours, "date_multisets_equal": True,
              "all_non_date_request_fields_equal": True, "padding_00UTC_preserved": True,
              "protected_original_artifacts_unchanged": len(new["protected_files"]),
              "model_imports_and_training_calls": 0, "network_requests": 0,
              "scope": "batch partition/provenance only; NOT full downloaded ERA5 field QA"}
    path = DEST / "batching_independent_qa.json"
    assert not path.exists(), "Existing independent QA is immutable; use a new audit version"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
