"""Independent scalar time-domain oracle; never imports the E-SST kernel.

Checks synthetic numerical results only, not model validity or real input
availability. The production implementation partitions horizontal distance;
this implementation overlaps each hourly time interval with the ray's
travel-time window. All historical sources must remain byte-identical.
"""
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_020"
TESTS = OUT / "synthetic_tests"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_oracle(rates, anchor, p):
    height, angle = p["height_above_antenna"], p["elevation"]
    if height == 0:
        return "VALUE", 0.0
    if angle == 90:
        return "VALUE", p["k"] * rates[anchor] ** p["alpha"] * height
    theta = angle * math.pi / 180
    duration = (1000 * height / p["wind_speed"]) / math.tan(theta)
    if duration > (len(rates) - anchor) * 3600:
        return "INSUFFICIENT_CONTEXT", None
    terms = []
    for index, rate in enumerate(rates):
        # Shift the time origin to the anchor to avoid subtracting two large
        # timestamps when a near-zenith ray has a very short travel time.
        start = (index - anchor) * 3600.0
        stop = (index + 1 - anchor) * 3600.0
        overlap = max(0.0, min(duration, stop) - max(0.0, start))
        if overlap:
            # d l [km] = v [m/s] * d t [s] / 1000.
            terms.append(overlap * p["wind_speed"] / 1000 * p["k"] * rate ** p["alpha"])
    return "VALUE", math.fsum(terms) / math.cos(theta)


def main():
    destination = OUT / "independent_qa.json"
    if destination.exists():
        raise FileExistsError(destination)
    data = read(TESTS / "kernel_cases.json")
    unit = read(TESTS / "unit_checks.json")
    summary = read(TESTS / "test_summary.json")
    execution = read(TESTS / "execution.json")
    before = read(OUT / "sources_before.json")
    failures, changed = [], []
    max_abs, max_scaled, value_count, refusal_count = 0.0, 0.0, 0, 0
    worst = None
    # Declared before execution; no tolerance selection from observed errors.
    rtol, atol = 2e-12, 2e-12
    for case in data["cases"]:
        rates = data["rain_fixtures_mm_per_h"][case["fixture_id"]]
        state, expected = temporal_oracle(rates, case["anchor_bin"], case["parameters"])
        if state != case["status"]:
            failures.append({"case": case["case_id"], "kind": "coverage_status", "expected": state})
            continue
        if state == "INSUFFICIENT_CONTEXT":
            refusal_count += 1
            if "attenuation_db" in case:
                failures.append({"case": case["case_id"], "kind": "invalid_missing_filled"})
            continue
        value_count += 1
        observed = case["attenuation_db"]
        error = abs(observed - expected)
        scaled = error / max(1.0, abs(expected))
        if error > max_abs:
            max_abs, worst = error, case["case_id"]
        max_scaled = max(max_scaled, scaled)
        if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=rtol, abs_tol=atol):
            failures.append({"case": case["case_id"], "kind": "integral", "expected": expected,
                             "observed": observed, "absolute_error": error})

    uniform_checks, uniform_max = 0, 0.0
    for case in data["cases"]:
        if case["fixture_id"] != "uniform_2p5" or case["status"] != "VALUE":
            continue
        p = case["parameters"]
        expected = p["k"] * 2.5 ** p["alpha"] * p["height_above_antenna"] / math.sin(math.radians(p["elevation"]))
        error = abs(case["attenuation_db"] - expected)
        uniform_max = max(uniform_max, error)
        uniform_checks += 1
        if not math.isclose(case["attenuation_db"], expected, rel_tol=2e-12, abs_tol=2e-12):
            failures.append({"case": case["case_id"], "kind": "uniform_analytic_identity"})

    event_check = next(x for x in unit if x["id"] == "event_threshold_dry_separator_and_censor_flags")
    event_projection = [(x["start_bin"], x["stop_bin_exclusive"], x["peak_mm_per_h"],
                         x["regime"], x["left_censored"], x["right_censored"])
                        for x in event_check["observed"]]
    if event_projection != [(0, 2, 10.0, "stratiform", True, False),
                            (3, 4, 10.0001, "convective", False, False),
                            (5, 6, 2.0, "stratiform", False, True)]:
        failures.append({"kind": "independent_event_boundary_rule"})
    if not all(x["passed"] for x in unit) or summary["status"] != "PASS":
        failures.append({"kind": "reported_unit_tests_not_pass"})
    if len(data["cases"]) != 6480 or value_count + refusal_count != 6480:
        failures.append({"kind": "synthetic_case_coverage"})
    if data["synthetic_only"] is not True or data["coefficients"] != "ARTIFICIAL_TEST_VALUES_NOT_P838":
        failures.append({"kind": "synthetic_label_missing"})
    for path, digest in before.items():
        actual = sha(path) if Path(path).is_file() else None
        if actual != digest:
            changed.append({"path": path, "expected": digest, "actual": actual})
    if changed:
        failures.append({"kind": "protected_sources_changed", "count": len(changed)})
    changed_code = [name for name, digest in execution["code_sha256"].items()
                    if sha(CODE / name) != digest]
    if changed_code:
        failures.append({"kind": "executed_code_changed", "files": changed_code})
    qa = dict(status="PASS" if not failures else "FAIL", utc=datetime.now(timezone.utc).isoformat(),
              scope="SYNTHETIC_HOURLY_KERNEL_ONLY_NOT_SITE_ADAPTER", python=sys.executable,
              platform=platform.platform(), independent_implementation="time_overlap_scalar_no_kernel_import",
              unit_checks=len(unit), deterministic_cases=len(data["cases"]),
              value_cases=value_count, refused_context_cases=refusal_count,
              max_absolute_difference_db=max_abs, max_scaled_difference=max_scaled, worst_case_id=worst,
              comparison_relative_tolerance=rtol, comparison_absolute_tolerance_db=atol,
              uniform_analytic_checks=uniform_checks, uniform_max_absolute_difference_db=uniform_max,
              protected_source_files=len(before), protected_source_changes=changed,
              code_changes_after_test_launch=changed_code, failures=failures,
              real_predictions=0, real_metrics=0, model_complete=False,
              all_seven_original_models_complete=False,
              mts_status="NOT_COMPUTED_USER_DECISION", weather_downloads=0, retraining_calls=0)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(qa, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in qa.items() if k not in ("platform", "python")}, ensure_ascii=False))
    if failures:
        raise AssertionError("independent QA failed; preserve all artifacts")


if __name__ == "__main__":
    main()
