"""Package the C2-selected generator using TRAIN data and the frozen C1 CDF.

Does not refit or rewrite C1/C2. Existing final packages are verified, not
overwritten. No official validation/test residuals are read by this module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import run_dynamics as dyn
import fit_marginals as fm
from window_diagnostics import OUT, now, sha256, worker_lock, write_json

HERE = Path(__file__).resolve().parent
DEST = OUT / "final_generator"
R16 = OUT / "dynamics/r16"


def upstream() -> dict:
    selection = dyn.read_json(R16 / "selection.json")
    report = dyn.read_json(R16 / "report_qa.json")
    qa = dyn.read_json(R16 / "qa.json")
    if (selection["status"] != "dependency_family_frozen"
            or selection["winner"] != selection["half_budget_winner"]
            or selection["winner"] != "block_0256" or report["status"] != "PASS"
            or qa["status"] != "PASS" or qa["trial_states"] != 70
            or report["selection_sha256"] != sha256(R16 / "selection.json")
            or selection["qa_sha256"] != sha256(R16 / "qa.json")):
        raise RuntimeError("C2 selection/QA gate is not frozen and valid")
    for name, digest in selection["output_sha256"].items():
        if sha256(R16 / name) != digest:
            raise RuntimeError("C2 report changed: " + name)
    numerical = dyn.read_json(OUT / "stationarity/numerical_tests_qa.json")
    if numerical["status"] != "PASS" or numerical["builder_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("Current final builder lacks numerical QA")
    dyn.freeze_manifest(16)  # Existing immutable manifests only get checked.
    return selection


def load_package() -> tuple[dict, dict, dict]:
    saved = dyn.read_json(DEST / "model.json")
    models, pools = {}, {}
    for event, component in saved["components"].items():
        support_path = OUT / component["cdf_file_relative_to_revision_output"]
        library_path = DEST / component["library_file"]
        if (sha256(support_path) != component["cdf_sha256"]
                or sha256(library_path) != component["library_sha256"]):
            raise RuntimeError("Final generator component changed")
        with np.load(support_path) as a:
            models[event] = {"family": "empirical", "support_q": a["q"], "support_cdf": a["cdf"]}
        with np.load(library_path) as a:
            pools[event] = [dyn.Series(str(h), str(h)[:10], event, -1, q, n)
                            for h, q, n in zip(a["hours"], a["q"], a["records"])]
    return saved, models, pools


def verify() -> dict:
    selection = upstream()
    saved, models, pools = load_package()
    if (saved["builder_sha256"] != sha256(Path(__file__))
            or saved["plan_sha256"] != sha256(HERE / "STATIONARITY_PLAN.md")
            or saved["selection_sha256"] != sha256(R16 / "selection.json")
            or saved["marginal_model_sha256"] != sha256(OUT / "marginal_model/model.json")
            or saved["candidate"] != selection["winner"]):
        raise RuntimeError("Final generator fingerprint changed")
    means, libraries, hours = {}, {}, []
    for event, pool in pools.items():
        model = models[event]
        weights = np.diff(np.r_[0., model["support_cdf"]])
        if ((weights <= 0).any() or model["support_cdf"][-1] != 1
                or (model["support_q"] <= 0).any() or not np.isfinite(model["support_q"]).all()):
            raise RuntimeError("CDF invalid")
        means[event] = float(np.sum(weights * model["support_q"]))
        if abs(means[event] - 1) > 1e-10:
            raise RuntimeError("Final CDF unit power failed")
        library = dyn.block_library(pool, saved["block_length_s"])
        total_starts = 0
        for entry in library:
            h = entry["hour"]
            valid = h.records >= 9
            # Independent literal sliding-mask count, not library run arithmetic.
            bad = np.r_[0, np.cumsum(~valid)]
            n = int(((bad[256:] - bad[:-256]) == 0).sum())
            if n != entry["n_starts"]:
                raise RuntimeError("Final legal-start count mismatch")
            total_starts += n
        component = saved["components"][event]
        if (len(pool) != component["training_hours"]
                or len(library) != component["eligible_hours"]
                or total_starts != component["legal_start_count"]):
            raise RuntimeError("Final library count mismatch")
        q, _ = dyn.generate("iid", model, None, -1, "final-qa-" + event, 0)
        g, _ = dyn.generate(saved["candidate"], model, library, -1, "final-qa-" + event, 0)
        if not np.array_equal(np.sort(q), np.sort(g)):
            raise RuntimeError("Rank generator altered marginal samples")
        libraries[event] = {"training_hours": len(pool), "eligible_hours": len(library),
                            "legal_start_count": total_starts, "rank_sample_multiset_equal": True}
        hours.extend(h.hour for h in pool)
    if len(hours) != 2404 or len(set(hours)) != 2404 or sorted(hours) != saved["training_hours"]:
        raise RuntimeError("Final generator training-hour coverage mismatch")
    qa = {"status": "PASS", "candidate": saved["candidate"], "block_length_s": saved["block_length_s"],
          "training_hours": 2404, "training_dates": len({h[:10] for h in hours}),
          "unit_power_moment": means, "libraries": libraries,
          "model_sha256": sha256(DEST / "model.json"), "final_validation_test_reads": 0,
          "whole_fast_fading_stage_complete": False}
    path = DEST / "qa.json"
    if path.exists():
        old = dyn.read_json(path)
        if any(old.get(k) != v for k, v in qa.items()):
            raise RuntimeError("Existing final generator QA differs")
        return old
    qa["completed_utc"] = now()
    write_json(path, qa)
    return qa


def run() -> dict:
    upstream()
    if (DEST / "model.json").exists():
        return verify()
    DEST.mkdir(exist_ok=True)
    with worker_lock("freeze_generator.py run"):
        series = dyn.load_series(dyn.core_design())
        marginal = dyn.read_json(OUT / "marginal_model/model.json")
        components = {}
        for event in fm.CLASSES:
            pool = [h for h in series if h.event == event]
            library = dyn.block_library(pool, 256)
            if not library:
                raise RuntimeError("No legal final blocks for " + event)
            path = DEST / (event + "_train_series.npz")
            runs = np.array([(i, a, b) for i, h in enumerate(pool) for a, b in dyn.valid_runs(h.valid)], dtype=np.int32)
            dyn.atomic_npz(path, hours=np.array([h.hour for h in pool]),
                           q=np.stack([h.q for h in pool]), records=np.stack([h.records for h in pool]), runs=runs)
            m = marginal["components"][event]
            eligible = {e["hour"].hour for e in library}
            components[event] = {
                "cdf_file_relative_to_revision_output": "marginal_model/" + m["support_file"],
                "cdf_sha256": m["support_sha256"], "library_file": path.name, "library_sha256": sha256(path),
                "training_hours": len(pool), "training_dates": len({h.day for h in pool}),
                "eligible_hours": len(library), "eligible_dates": len({e["hour"].day for e in library}),
                "legal_start_count": sum(e["n_starts"] for e in library),
                "excluded_hours_no_256s_run": [h.hour for h in pool if h.hour not in eligible],
            }
            print("Saved final generator/" + event, components[event]["training_hours"], flush=True)
        write_json(DEST / "model.json", {
            "created_utc": now(), "candidate": "block_0256", "block_length_s": 256,
            "builder_sha256": sha256(Path(__file__)), "plan_sha256": sha256(HERE / "STATIONARITY_PLAN.md"),
            "selection_sha256": sha256(R16 / "selection.json"),
            "marginal_model_sha256": sha256(OUT / "marginal_model/model.json"),
            "training_hours": sorted(h.hour for h in series), "training_dates": sorted({h.day for h in series}),
            "definition": "Q=P_1s/Pbar_hour; F=-10log10(Q); E(Q)=1, not per-realization normalization",
            "seed_rule": "C2 stable SHA256 seeds; use fold=-1 for final-only generation",
            "components": components, "final_validation_test_reads": 0,
            "stationary_window_selected": False, "whole_fast_fading_stage_complete": False,
        })
        return verify()


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    print(json.dumps(run() if args.command == "run" else verify(), indent=2), flush=True)
