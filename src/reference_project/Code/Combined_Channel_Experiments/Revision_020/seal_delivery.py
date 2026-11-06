"""Seal this new, locally verified phase without changing any historical file."""
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_020"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main():
    if (OUT / "delivery_qa.json").exists() or (OUT / "final_verification.json").exists():
        raise FileExistsError("Delivery already attempted; do not overwrite or reseal")
    checks = []

    def check(name, condition, detail=None):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    qa = read(OUT / "independent_qa.json")
    state = read(OUT / "model_status.json")
    ready = read(OUT / "input_readiness.json")
    execution = read(OUT / "synthetic_tests/execution.json")
    before = read(OUT / "sources_before.json")
    after = {name: sha(name) if Path(name).is_file() else None for name in before}
    differences = [name for name in before if before[name] != after[name]]
    check("protected_sources_unchanged", not differences, {"count": len(before), "differences": differences})
    check("independent_synthetic_qa_pass", qa["status"] == "PASS" and not qa["failures"])
    check("all_synthetic_cases_accounted", qa["deterministic_cases"] == 6480
          and qa["value_cases"] == 6264 and qa["refused_context_cases"] == 216)
    check("unit_checks_complete", qa["unit_checks"] == 56)
    for name, digest in execution["code_sha256"].items():
        check("executed_code_unchanged_" + name, sha(CODE / name) == digest)
    for p in CODE.glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        check("syntax_" + p.name, True)
    kernel_tree = ast.parse((CODE / "esst_hourly_kernel.py").read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(kernel_tree):
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    check("kernel_imports_only_pure_math_and_types", set(imports) == {"math", "collections.abc"}, imports)
    oracle_tree = ast.parse((CODE / "verify_independent.py").read_text(encoding="utf-8"))
    oracle_imports = [item.name for n in ast.walk(oracle_tree) if isinstance(n, ast.Import) for item in n.names]
    oracle_imports += [n.module for n in ast.walk(oracle_tree) if isinstance(n, ast.ImportFrom)]
    check("oracle_does_not_import_kernel", not any("esst_hourly_kernel" in x for x in oracle_imports))
    models = {m["id"]: m for m in state["models"]}
    check("exactly_seven_families", set(models) == {"proposed", "bai_lstm", "tabm", "itu_combined", "e_sst", "rmop", "nwp_chain"}
          and len(state["models"]) == 7 and state["formal_family_count"] == 7)
    check("four_reused_computed_families", sum(m["hourly_status"] == "COMPUTED_REUSED" for m in models.values()) == 4
          and state["computed_hourly_family_count"] == 4)
    for name in ("proposed", "bai_lstm", "tabm", "itu_combined"):
        score_source = (OUT / models[name]["scores_source"]).resolve()
        check("frozen_score_reference_" + name, score_source.is_file()
              and str(score_source) in before and sha(score_source) == before[str(score_source)])
    for name in ("e_sst", "rmop"):
        item = models[name]
        check("no_false_result_" + name, item["hourly_status"] == "BLOCKED_NOT_COMPUTED"
              and item["numeric_results"] is None and item["rank"] is None and item["required_for_current_delivery"])
    prm = models["nwp_chain"]["branches"]["prm_hourly"]
    check("prm_remains_required_and_uncomputed", prm["status"] == "BLOCKED_NOT_COMPUTED"
          and prm["numeric_results"] is None and prm["rank"] is None and prm["required_for_current_delivery"])
    mts = models["nwp_chain"]["branches"]["mts_seconds"]
    check("mts_latest_user_decision", mts["status"] == "NOT_COMPUTED_USER_DECISION"
          and mts["numeric_results"] is None and mts["rank"] is None
          and not mts["required_for_current_delivery"] and not mts["awaiting_user_materials"]
          and mts["resume_requires_new_explicit_user_decision"])
    check("no_false_overall_completion", not state["all_seven_original_models_complete"]
          and not state["current_required_work_complete"] and not state["submission_ready"])
    check("esst_formula_obstacle_resolved_not_model_gate", models["e_sst"]["original_formula_and_rain_type_rules_verified"]
          and ready["formal_pdf_equations_and_event_rules_verified"]
          and not ready["real_scoring_allowed"] and not any(ready["evidence_gates"].values()))
    for key in ("new_training", "new_real_predictions", "new_real_physics_runs", "new_real_scoring", "new_weather_requests"):
        check("no_new_" + key, state[key] == 0)
    check("full_scope_not_shrunk", state["data_scope"] == {
        "local_original_test_hours": 687, "uk_complete_station_hours": 5664,
        "uk_source_station_days": {"chilbolton": 143, "sparsholt": 93},
        "uk_qualified_target_hours": 5593, "uk_common_physics_hours": 5538,
        "source": "../Combined_Channel_Revision_012/REPORT.md",
        "new_keywise_reconstruction_this_revision": False, "changed": False})
    check("coavailability_not_score_count", not ready["existing_local_asset_audit"]["coavailability_is_scoring_eligibility"])
    for token, field in (("STOP_REQUESTED.json", "stop_sha256"), ("automation.toml", "automation_sha256")):
        selected = [(p, h) for p, h in after.items() if p.endswith(token)]
        check("frozen_" + token, len(selected) == 1 and selected[0][1] == state[field], selected)
    check("no_automation_change", not state["automation_modified"])
    for page in read(OUT / "paper_pages.json"):
        check("paper_render_" + page["png"], sha(OUT / page["png"]) == page["png_sha256"])
    for name in ("PROTOCOL.md", "FORMULA_AUDIT.md", "MODEL_INPUT_MATRIX.md", "REPORT.md", "QA.md",
                 "HANDOFF.md", "COMMANDS.md", "CHANGE_RECORD.md"):
        path = OUT / name
        check("document_present_" + name, path.is_file() and path.stat().st_size > 100)
        broken = []
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                linked = (path.parent / target.split("#")[0]).resolve()
                if not linked.is_file():
                    broken.append(target)
        check("local_links_" + name, not broken, broken)
    failures = [x for x in checks if not x["passed"]]
    save("source_guard_final.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                    "before_count": len(before), "differences": differences,
                                    "sources_after": after})
    save("delivery_qa.json", {"status": "PASS" if not failures else "FAIL", "checks": checks,
                              "failures": failures, "checks_count": len(checks),
                              "scope": "LOCAL_PHASE_ONLY_NO_NEW_REAL_MODEL_SCORES"})
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, ensure_ascii=False))
        raise AssertionError("Delivery QA failed; preserve failed output")
    inventory = {}
    for folder in (CODE, OUT):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                inventory[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    save("delivery_manifest.json", {"files": inventory,
                                   "excludes": ["delivery_manifest.json", "final_verification.json"],
                                   "reason": "avoid self-referential hashes; final record seals this manifest"})
    # Read back every artifact; this is an integrity check, not model evaluation.
    assert all(sha(ROOT / name) == item["sha256"] for name, item in inventory.items())
    final = {"status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
             "local_phase": "ESST_HOURLY_NUMERICAL_KERNEL_AND_FORMULA_AUDIT_VERIFIED",
             "manifest_sha256": sha(OUT / "delivery_manifest.json"), "sealed_artifacts": len(inventory),
             "delivery_checks": len(checks), "protected_sources": len(before),
             "protected_source_changes": 0, "execution_lock": "RELEASED",
             "lock_kind": "logical_declaration_no_OS_exclusivity_claim",
             "new_real_predictions": 0, "new_real_scores": 0,
             "seven_family_work_complete": False, "all_seven_original_models_complete": False,
             "automation_modified": False, "automation_pause_requested": False,
             "mts_status": "NOT_COMPUTED_USER_DECISION",
             "human_publication_acceptance": "NOT_RECORDED"}
    save("final_verification.json", final)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
