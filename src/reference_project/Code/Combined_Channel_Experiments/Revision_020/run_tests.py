"""Deterministic synthetic tests only; exclusive new output, no site inference."""
import hashlib
import itertools
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import esst_hourly_kernel as kernel

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_020/synthetic_tests"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main():
    OUT.mkdir(exist_ok=False)
    execution = {"started_utc": datetime.now(timezone.utc).isoformat(),
                 "python": sys.executable, "python_version": sys.version,
                 "platform": platform.platform(), "argv": sys.argv,
                 "working_directory": str(Path.cwd()),
                 "code_sha256": {p.name: sha(p) for p in sorted(CODE.glob("*.py"))},
                 "synthetic_only": True, "weather_reads": 0, "label_reads": 0,
                 "training_calls": 0, "random_draws": 0, "network_calls": 0}
    save("execution.json", execution)
    try:
        _run()
    except Exception:
        save("run_failure.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                   "traceback": traceback.format_exc()})
        raise


def _run():
    checks = []
    times = [3600 * i for i in range(12)]
    base = dict(wind_speed=8.0, height_above_antenna=3.0, elevation=35.0,
                k=0.02, alpha=1.2, timestep_seconds=3600,
                rain_unit="mm/h", wind_unit="m/s", height_unit="km_above_antenna",
                angle_unit="degree")

    def close(name, observed, expected, tolerance=2e-12):
        passed = math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance)
        checks.append(dict(id=name, kind="numeric", passed=passed,
                           observed=observed, expected=expected, tolerance=tolerance))

    def equal(name, observed, expected):
        checks.append(dict(id=name, kind="exact", passed=observed == expected,
                           observed=observed, expected=expected))

    def rejects(name, fn, error=kernel.ContractError):
        try:
            fn()
        except error as exc:
            checks.append(dict(id=name, kind="rejection", passed=True,
                               observed=type(exc).__name__, message=str(exc)))
        else:
            checks.append(dict(id=name, kind="rejection", passed=False,
                               observed="NO_EXCEPTION"))

    def evaluate(rates=None, t=None, anchor=0, **changes):
        config = base | changes
        return kernel.attenuation_at_bin_start([2.5] * 12 if rates is None else rates,
                                               times if t is None else t, anchor, **config)

    close("uniform_rain_closed_form", evaluate(),
          0.02 * 2.5 ** 1.2 * 3 / math.sin(math.radians(35)))
    close("zero_rain", evaluate([0.0] * 12), 0.0)
    close("zero_path_height", evaluate(height_above_antenna=0.0), 0.0)
    close("zero_coefficient", evaluate(k=0.0), 0.0)
    close("zenith_analytic_limit", evaluate(elevation=90.0), 0.02 * 2.5 ** 1.2 * 3)
    close("near_zenith_uniform", evaluate(elevation=89.999999),
          0.02 * 2.5 ** 1.2 * 3 / math.sin(math.radians(89.999999)))
    close("shifted_clock_same_integral", evaluate(t=[x + 1_700_000_000 for x in times]), evaluate())
    close("two_bin_exact_overlap", evaluate([1, 4] + [0] * 10,
          wind_speed=1.0, height_above_antenna=5.4, elevation=45.0, k=1.0, alpha=1.0),
          (3.6 + 1.8 * 4) / math.cos(math.radians(45.0)))
    close("stratiform_height", kernel.equivalent_height_above_antenna_km(
          2000.0, 117.5, absolute_height_unit="m_AMSL", regime="stratiform"), 2.2425)
    close("convective_height", kernel.equivalent_height_above_antenna_km(
          2000.0, 117.5, absolute_height_unit="m_AMSL", regime="convective"), 1.8825)
    events = kernel.classify_hourly_events([1, 10, 0, 10.0001, 0, 2],
             [3600 * i for i in range(6)], timestep_seconds=3600, rain_unit="mm/h")
    expected_events = [
        dict(start_bin=0, stop_bin_exclusive=2, peak_mm_per_h=10.0,
             regime="stratiform", left_censored=True, right_censored=False),
        dict(start_bin=3, stop_bin_exclusive=4, peak_mm_per_h=10.0001,
             regime="convective", left_censored=False, right_censored=False),
        dict(start_bin=5, stop_bin_exclusive=6, peak_mm_per_h=2.0,
             regime="stratiform", left_censored=False, right_censored=True)]
    equal("event_threshold_dry_separator_and_censor_flags", events, expected_events)
    equal("dry_sequence_has_no_events", kernel.classify_hourly_events([0, 0], [0, 3600],
          timestep_seconds=3600, rain_unit="mm/h"), [])
    rejects("negative_rain", lambda: evaluate([-0.001] + [0] * 11))
    rejects("missing_rain_not_dry", lambda: evaluate([None] + [0] * 11))
    rejects("nan_rain", lambda: evaluate([float("nan")] + [0] * 11))
    rejects("boolean_rain", lambda: evaluate([True] + [0] * 11))
    rejects("empty_input", lambda: evaluate([], []))
    rejects("length_mismatch", lambda: evaluate([1], times))
    rejects("gap_not_concatenated", lambda: evaluate(t=times[:5] + [x + 3600 for x in times[5:]]))
    rejects("duplicate_timestamp", lambda: evaluate(t=[0, 0] + times[2:]))
    rejects("nonhourly_grid", lambda: evaluate(t=[60 * i for i in range(12)]))
    rejects("minute_step_not_hourly", lambda: evaluate(timestep_seconds=60))
    rejects("amount_unit_not_rate", lambda: evaluate(rain_unit="mm"))
    rejects("wind_unit_wrong", lambda: evaluate(wind_unit="km/h"))
    rejects("height_unit_wrong", lambda: evaluate(height_unit="m_AGL"))
    rejects("angle_unit_wrong", lambda: evaluate(angle_unit="radian"))
    for key, value in [("wind_speed", 0), ("wind_speed", -1), ("height_above_antenna", -1),
                       ("elevation", 0), ("elevation", 91), ("k", -1), ("alpha", 0),
                       ("alpha", float("inf"))]:
        rejects(f"invalid_{key}_{value}", lambda key=key, value=value: evaluate(**{key: value}))
    rejects("float_anchor", lambda: evaluate(anchor=0.5))
    rejects("boolean_anchor", lambda: evaluate(anchor=True))
    rejects("anchor_outside", lambda: evaluate(anchor=12))
    rejects("short_forward_context", lambda: evaluate(anchor=11, wind_speed=0.5,
            height_above_antenna=8, elevation=10), kernel.InsufficientContextError)
    rejects("overflow_not_reported_as_finite", lambda: evaluate([1e300] * 12, alpha=5))
    rejects("agl_not_relabelled_amsl", lambda: kernel.equivalent_height_above_antenna_km(
            2000, 117.5, absolute_height_unit="m_AGL", regime="stratiform"))
    rejects("zero_level_below_antenna", lambda: kernel.equivalent_height_above_antenna_km(
            100, 117.5, absolute_height_unit="m_AMSL", regime="stratiform"))
    rejects("missing_regime", lambda: kernel.equivalent_height_above_antenna_km(
            2000, 117.5, absolute_height_unit="m_AMSL", regime="unknown"))
    rejects("event_missing_is_not_dry_separator", lambda: kernel.classify_hourly_events(
            [1, None, 1], [0, 3600, 7200], timestep_seconds=3600, rain_unit="mm/h"))
    rejects("real_scoring_fail_closed", lambda: kernel.require_real_scoring_contract({}))
    gates = {name: True for name in kernel.REAL_SCORING_GATES}
    for name in kernel.REAL_SCORING_GATES:
        rejects("missing_evidence_" + name,
                lambda name=name: kernel.require_real_scoring_contract(gates | {name: False}))
    rejects("integer_is_not_verified_evidence", lambda: kernel.require_real_scoring_contract(
            gates | {kernel.REAL_SCORING_GATES[0]: 1}))
    rejects("unknown_evidence_gate", lambda: kernel.require_real_scoring_contract(
            gates | {"synthetic_test_passed": True}))
    # A structurally complete synthetic declaration tests the guard, not real evidence.
    equal("guard_structural_positive_control", kernel.require_real_scoring_contract(gates), None)

    fixtures = {
        "all_dry": [0.0] * 12,
        "uniform_2p5": [2.5] * 12,
        "two_pulses": [0, 0, 12, 3, 0, 0, 0, 2, 4, 0, 0, 0],
        "mixed": [1, 0.2, 0, 10, 10.1, 0, 4, 2, 0.5, 0, 0, 1],
    }
    cases = []
    for fid, wind, height, angle, alpha, k, anchor in itertools.product(
            fixtures, (0.75, 7.5, 20.0), (0.0, 0.36, 2.0, 5.0),
            (10.0, 35.0, 67.0, 89.999, 90.0), (0.7, 1.0, 1.4), (0.0, 0.02, 0.3), (0, 5, 11)):
        config = base | dict(wind_speed=wind, height_above_antenna=height,
                             elevation=angle, alpha=alpha, k=k)
        case = dict(case_id=len(cases), fixture_id=fid, anchor_bin=anchor, parameters=config)
        try:
            case.update(status="VALUE", attenuation_db=kernel.attenuation_at_bin_start(
                        fixtures[fid], times, anchor, **config))
        except kernel.InsufficientContextError:
            case.update(status="INSUFFICIENT_CONTEXT")
        cases.append(case)
    save("unit_checks.json", checks)
    save("kernel_cases.json", dict(synthetic_only=True, coefficients="ARTIFICIAL_TEST_VALUES_NOT_P838",
                                  bin_start_times_seconds=times, rain_fixtures_mm_per_h=fixtures, cases=cases))
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    summary = dict(status=status, unit_checks=len(checks),
                   unit_failures=[x["id"] for x in checks if not x["passed"]],
                   deterministic_cases=len(cases),
                   value_cases=sum(x["status"] == "VALUE" for x in cases),
                   incomplete_context_cases=sum(x["status"] == "INSUFFICIENT_CONTEXT" for x in cases),
                   independent_verification="PENDING_SEPARATE_IMPLEMENTATION",
                   real_predictions=0, new_metrics=0, model_complete=False)
    save("test_summary.json", summary)
    print(json.dumps(summary))
    if status != "PASS":
        raise AssertionError("synthetic unit checks failed; preserve this run")


if __name__ == "__main__":
    main()
