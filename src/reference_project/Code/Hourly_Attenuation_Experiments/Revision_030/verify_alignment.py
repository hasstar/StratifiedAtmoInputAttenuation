"""Independent arithmetic and provenance audit; no training or model inference."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_030_Original_Training_Aligned"
R28 = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced"
R29 = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE"
WEATHER = ["overall", "clear_dry", "rain", "dry_overcast", "other"]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_new(name: str, data: dict) -> None:
    path = (OUT / name).resolve()
    assert path.parent == OUT.resolve()
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def independent_scores(source: dict) -> dict:
    hours = source["hour_bjt"]
    errors = np.asarray(source["prediction_db"], dtype=np.float64) - np.asarray(source["truth_db"], dtype=np.float64)
    assert len(errors) == len(hours) and np.isfinite(errors).all()
    result = {}
    for weather in WEATHER:
        mask = np.ones(len(hours), dtype=bool) if weather == "overall" else np.asarray(source["weather_masks"][weather], dtype=bool)
        part = errors[mask]
        assert len(part) > 0
        result[weather] = {"n_hours": int(mask.sum()), "n_dates": len({h[:10] for h, keep in zip(hours, mask) if keep}),
                           "rmse_db": float(np.sqrt(np.dot(part, part) / len(part))),
                           "mae_db": float(np.abs(part).mean()), "bias_db": float(part.mean())}
    return result


def compare(scores: dict, output: dict) -> float:
    error = 0.0
    for weather in WEATHER:
        for field in ["n_hours", "n_dates"]:
            assert scores[weather][field] == output[weather][field]
        for field in ["rmse_db", "mae_db", "bias_db"]:
            delta = abs(scores[weather][field] - output[weather][field])
            assert delta <= 1e-12, (weather, field, delta)
            error = max(error, delta)
    return error


def main() -> None:
    assert not (OUT / "alignment_qa.json").exists() and not (OUT / "completion.json").exists(), "Existing audit must not be overwritten."
    manifest = load(OUT / "run_manifest.json")
    assert manifest["type"] == "report_only" and manifest["seed"] == 17
    assert manifest["new_fit_calls"] == manifest["new_model_inference_calls"] == 0
    for name, expected in manifest["code_sha256"].items():
        assert digest(CODE / name) == expected, f"New audit code changed: {name}"
    assert digest(OUT / "source_snapshot.json") == manifest["source_snapshot_sha256"]
    snapshot = load(OUT / "source_snapshot.json")
    assert snapshot["count"] == manifest["source_files"] == len(snapshot["source_sha256"])
    for name, expected in snapshot["source_sha256"].items():
        assert digest(ROOT / name) == expected, f"Old source changed: {name}"

    originals = load(OUT / "original_seed17_test_metrics.json")
    fixed = load(OUT / "fixed53_all_state_metrics.json")
    summary = load(OUT / "analysis_summary.json")
    lineage = load(OUT / "training_lineage.json")
    assert len(originals) == len(lineage) == 20 and len(fixed) == 1180
    assert summary["scope"] == "report_alignment_no_new_training" and summary["seed_average"] is False
    assert summary["new_fit_calls"] == summary["new_model_inference_calls"] == summary["new_inferential_tests"] == 0
    maximum, scalar_checks, original_scores, fixed_scores = 0.0, 0, {}, {}
    reference = load(R28 / "test/full_bilstm_seed17.json")
    for row in originals:
        source = load(ROOT / row["source_record"])
        assert source["seed"] == 17 and row["feature_names"] == source["feature_names"]
        assert all(source[name] == reference[name] for name in ["hour_bjt", "truth_db", "weather_masks"])
        scores = independent_scores(source)
        maximum = max(maximum, compare(scores, row["metrics"]), compare(scores, source["metrics"]))
        original_scores[(row["case"], row["model"])] = scores
        scalar_checks += 15
    assert len(original_scores) == 20
    for row in lineage:
        train = load(ROOT / row["training_record"])
        assert row["original_frozen_plan"] == train["plan"]
        assert row["fit_metadata"] == train["fit_metadata"] and row["round_history"] == train["round_history"]
        assert digest(R28 / "models" / f'{row["case"]}_{row["model"]}_seed17.joblib') == row["model_sha256"] == train["model_sha256"]
    for row in fixed:
        source = load(ROOT / row["source_record"])
        assert row["feature_names"] == source["feature_names"] and row["seed"] == 17 and row["epochs"] == 53
        assert all(source["results"]["test"][name] == reference[name] for name in ["hour_bjt", "truth_db", "weather_masks"])
        scores = {}
        for split in ["validation", "test"]:
            scores[split] = independent_scores(source["results"][split])
            maximum = max(maximum, compare(scores[split], row["metrics"][split]), compare(scores[split], source["results"][split]["metrics"]))
            scalar_checks += 15
        fixed_scores[(row["kind"], row["key"])] = scores
    assert len(fixed_scores) == 1180 and scalar_checks == 35700
    assert {key for kind, key in fixed_scores if kind == "layer"} == set(range(1024))
    assert {key for kind, key in fixed_scores if kind == "rfe"} == set(range(1, 157))

    ordered = sorted([model for case, model in original_scores if case == "full"],
                     key=lambda model: (original_scores[("full", model)]["overall"]["rmse_db"], model))
    assert ordered == [r["model"] for r in summary["original_full_model_ranking"]] and ordered[0] == "bilstm"
    for index, row in enumerate(summary["original_full_model_ranking"]):
        assert row["rank"] == index + 1
        maximum = max(maximum, compare(original_scores[("full", row["model"])], row["metrics"]))
    base, full = original_scores[("base", "bilstm")], original_scores[("full", "bilstm")]
    for row in summary["original_primary_comparison"]:
        weather = row["weather"]
        assert row["base_original_46"]["n_hours"] == base[weather]["n_hours"]
        assert row["full_original_53"]["n_hours"] == full[weather]["n_hours"]
        for metric in ["rmse_db", "mae_db", "bias_db"]:
            assert abs(row["base_original_46"][metric] - base[weather][metric]) <= 1e-12
            assert abs(row["full_original_53"][metric] - full[weather][metric]) <= 1e-12
        difference = base[weather]["rmse_db"] - full[weather]["rmse_db"]
        assert difference > 0 and abs(difference - row["rmse_difference_base_minus_full_db"]) <= 1e-12
        assert abs(100 * difference / base[weather]["rmse_db"] - row["rmse_reduction_percent"]) <= 1e-9
    trace = {r["id"]: r for r in summary["baseline_trace"]}
    maximum = max(maximum, compare(base, trace["rev028_original_base46"]["metrics"]),
                  compare(full, trace["rev028_original_full53"]["metrics"]),
                  compare(fixed_scores[("layer", 0)]["test"], trace["rev029_uniform_base53"]["metrics"]))
    assert trace["rev028_original_base46"]["choice"]["estimator"]["epochs"] == 46
    assert trace["rev028_original_base46"]["choice"]["weight_policy"] == "balanced"
    assert trace["rev029_uniform_base53"]["choice"] == trace["rev028_original_full53"]["choice"]
    assert trace["rev029_uniform_base53"]["metrics"]["rain"]["rmse_db"] < full["rain"]["rmse_db"]
    for path in [R29 / "layer/evaluation/mask_1023.json", R29 / "rfe/evaluation/k_0156.json"]:
        anchor = load(path)
        assert anchor["results"]["test"]["prediction_db"] == reference["prediction_db"]
        assert anchor["feature_names"] == reference["feature_names"]
    for row in summary["fixed53_highlights"]:
        maximum = max(maximum, compare(fixed_scores[(row["kind"], row["key"])]["test"], row["metrics"]["test"]))

    levels = load(R29 / "run_manifest.json")["levels_hpa"]
    assert len(summary["layer_effects"]) == 50
    for row in summary["layer_effects"]:
        bit = levels.index(row["level_hpa"])
        deltas = np.asarray([fixed_scores[("layer", key)]["test"][row["weather"]]["rmse_db"] -
                             fixed_scores[("layer", key + 2 ** bit)]["test"][row["weather"]]["rmse_db"]
                             for key in range(1024) if (key // 2 ** bit) % 2 == 0])
        assert len(deltas) == row["contexts"] == 512
        for name, value in [("mean_rmse_improvement_db", deltas.mean()), ("median_rmse_improvement_db", np.median(deltas)),
                            ("fraction_helpful", (deltas > 0).mean()),
                            ("full_context_rmse_improvement_db", fixed_scores[("layer", 1023 - 2 ** bit)]["test"][row["weather"]]["rmse_db"] - full[row["weather"]]["rmse_db"])]:
            assert abs(float(value) - row[name]) <= 1e-12
    fns = {(r["kind"], r["key"]): r["feature_names"] for r in fixed}
    assert len(summary["rfe_survival_rank"]) == 156
    for k, row in enumerate(summary["rfe_survival_rank"], 1):
        remaining = set(fns[("rfe", k)])
        before = set() if k == 1 else set(fns[("rfe", k - 1)])
        assert len(remaining) == k and before < remaining and remaining - before == {row["feature_name"]}
        assert row["survival_rank"] == k
    assert summary["fixed53_layer_rfe_selection_unchanged"] == load(R29 / "selection.json")
    claims = load(OUT / "claim_audit.json")
    assert len(claims) == 8 and [r["status"] for r in claims] == ["SUPPORTED"] * 5 + ["NOT_SUPPORTED"] * 3
    text = (OUT / "RESULTS.md").read_text(encoding="utf-8")
    for row in summary["original_primary_comparison"]:
        assert f'{row["base_original_46"]["rmse_db"]:.6f}' in text
        assert f'{row["full_original_53"]["rmse_db"]:.6f}' in text
        assert f'{row["rmse_reduction_percent"]:.2f}%' in text
    for row in summary["original_full_model_ranking"]:
        assert all(f'{row["metrics"][w]["rmse_db"]:.6f}' in text for w in WEATHER)
    for row in summary["fixed53_highlights"]:
        assert all(f'{row["metrics"]["test"][w]["rmse_db"]:.6f}' in text for w in WEATHER)
    assert all(phrase in text for phrase in ["0.283988", "负结果有效且保留", "不及全层锚点", "不是新模型", "不称全新盲测"])
    output_hashes = {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}
    for name, expected in snapshot["source_sha256"].items():
        assert digest(ROOT / name) == expected
    qa = {"status": "PASS", "scope": "report_alignment_only", "source_files_unchanged": snapshot["count"],
          "original_models_checked_by_sha256": 20, "original_test_weather_rows_independently_recomputed": 100,
          "fixed53_states_independently_recomputed": 1180, "metric_scalar_checks": scalar_checks,
          "max_metric_difference_db": maximum, "layer_effect_rows_independently_recomputed": 50,
          "layer_context_pairs_preserved": 5120, "rfe_survival_rows_verified": 156, "primary_rounds": [46, 53],
          "new_fit_calls": 0, "new_model_inference_calls": 0, "new_inferential_tests": 0,
          "original_negative_results_preserved": True, "old_model_replay_qa_preserved_not_rerun": True,
          "output_sha256": output_hashes, "utc": datetime.now(timezone.utc).isoformat()}
    save_new("alignment_qa.json", qa)
    save_new("completion.json", {"status": "COMPLETE", "completed_scope": "original_training_aligned_report_and_independent_audit",
                                 "new_training_experiment_completed": False, "new_fit_calls": 0, "new_model_inference_calls": 0,
                                 "alignment_qa_sha256": digest(OUT / "alignment_qa.json"),
                                 "results_sha256": digest(OUT / "RESULTS.md"), "utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({key: qa[key] for key in ["status", "source_files_unchanged", "metric_scalar_checks", "max_metric_difference_db", "new_fit_calls", "new_model_inference_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
