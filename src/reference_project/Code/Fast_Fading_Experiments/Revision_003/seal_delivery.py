"""Seal the completed isolated T016 delivery after all acceptance gates."""
from common import *


def main():
    gates = {
        "numeric": OUT / "independent_numeric_qa.json",
        "rfe": DELIVERY / "RFE_QA.json",
        "workbook": DELIVERY / "RFE_WORKBOOK_QA.json",
        "summary": DELIVERY / "SUMMARY_QA.json",
        "visual": DELIVERY / "VISUAL_QA.json",
    }
    for label, path in gates.items():
        need(read(path)["status"] == "PASS", label + " gate not passed")
    need(not any((OUT / "failures").glob("*.json")), "Inspect failures first")
    need(read(OUT / "computation_complete.json")["hours"] == 6623, "Hours incomplete")
    need(sha(DELIVERY / "RFE_每模型每轮删除与保留参数.xlsx") == read(gates["workbook"])["workbook_sha256"], "Workbook changed after QA")
    for path, expected in read(OUT / "manifest.json")["sources"].items():
        need(sha(ROOT / path) == expected, "Protected source changed: " + path)
    # Nonblocking lock probe only: no data-generation or model calls.
    with lock():
        pass
    artifact_hashes = {relative(p): sha(p) for p in sorted(DELIVERY.rglob("*")) if p.is_file()}
    code_hashes = {relative(p): sha(p) for p in sorted(CODE.iterdir()) if p.is_file()}
    evidence = [OUT / n for n in ("manifest.json", "computation_complete.json", "independent_numeric_qa.json", "hourly_results.csv", "hourly_results.parquet", "summary_complete.json", "closeout_observations_20261024T0803Z.json")]
    sealed = {"utc": now(), "artifact_sha256": artifact_hashes, "code_sha256": code_hashes, "evidence_sha256": {relative(p): sha(p) for p in evidence}, "sources_unchanged": True}
    write(OUT / "delivery_manifest.json", sealed)
    result = {
        "state": "COMPLETE",
        "utc": now(),
        "task": "T016",
        "scope": "RFE all ten models; fast and TabM k10 only, LOCAL validation/test and UK both stations",
        "hours": 6623,
        "realizations": 211936,
        "local_validation_hours": 343,
        "local_test_hours": 687,
        "uk_primary_hours": 5593,
        "rfe_states": 1560,
        "rfe_deletions": 1550,
        "summary_groups": 25,
        "figure_pairs": 6,
        "qa_gates": {label: "PASS" for label in gates},
        "sources_unchanged": True,
        "new_fits": 0,
        "new_slow_inference": 0,
        "os_lock": "RELEASED",
        "delivery_manifest_sha256": sha(OUT / "delivery_manifest.json"),
        "delivery_path": relative(DELIVERY),
        "other_slow_models_combined": "Not executed; explicitly deferred by user",
    }
    write(OUT / "delivery_complete.json", result)
    write(OUT / "status.json", {"state": "COMPLETE", "updated_utc": result["utc"], "completed_hours": 6623, "total_hours": 6623, "delivery_receipt": "delivery_complete.json"}, replace=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
