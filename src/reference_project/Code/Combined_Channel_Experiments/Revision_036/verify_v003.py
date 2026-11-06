"""Independent scalar QA v003: retain null labels and gap-dependent usable lags.
The first attempt is retained unchanged; no scoring or eligibility threshold change.
Never imports the build or former scoring modules."""
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import math
import statistics
import traceback
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "Outputs/Experiments/Combined_Channel_Revision_036"
def rev(n):
    return BASE / f"Outputs/Experiments/Combined_Channel_Revision_{n:03d}"

FAMILIES = {
    "ours_frozen_top10": ("ours_frozen_top10",),
    "bai2022_lstm_t0_adapted": tuple("bai2022_lstm_t0_adapted_seed" + str(s) for s in (17, 29, 43)),
    "tabm2025_official_budgeted": tuple("tabm2025_official_budgeted_seed" + str(s) for s in (17, 29, 43)),
    "itu_combined_geometric_approx": ("itu_combined_geometric_approx",),
}
RUNS = tuple(x for values in FAMILIES.values() for x in values)
SHORT = dict(zip(FAMILIES, ("ours", "bai_lstm", "tabm", "itu")))
DMET = ("crps201_db", "crps401_db", "central90_coverage", "central98_coverage", "tail95_exceedance", "tail99_exceedance", "slow_only_crps_db")
ERRORS = defaultdict(float)
CHECKS = Counter()


def get(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def put(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise AssertionError(message)
    CHECKS[message] += 1


def epoch(t):
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    require(t.tzinfo is not None, "timezone_present")
    return int(t.timestamp())


def key(r, time="time"):
    return r["station"], epoch(r[time])


def index(rows, fields):
    result = {}
    for r in rows:
        k = tuple(r[f] for f in fields)
        require(k not in result, "unique_" + "_".join(fields))
        result[k] = r
    return result


def close(a, b, category, atol=1e-10):
    if a is None or b is None:
        require(a is None and b is None, category + "_null")
        return
    if math.isnan(a) or math.isnan(b):
        require(math.isnan(a) and math.isnan(b), category + "_nan_identity")
        return
    require(math.isfinite(a) and math.isfinite(b), category + "_finite")
    d = abs(a - b)
    ERRORS[category] = max(ERRORS[category], d)
    require(d <= atol, category + "_within_tolerance")


def avg(values):
    require(len(values) > 0 and all(math.isfinite(x) for x in values), "nonempty_finite_group")
    return math.fsum(values) / len(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def scores(truth, prediction):
    require(len(truth) == len(prediction) and len(truth) > 0, "score_dimensions")
    residual = [p - y for y, p in zip(truth, prediction)]
    return {"rmse_db": math.sqrt(avg([x*x for x in residual])),
            "mae_db": avg([abs(x) for x in residual]), "bias_db": avg(residual)}


def weighted(rows, metric, weighting):
    if weighting == "hour_equal":
        return avg([r[metric] for r in rows])
    require(weighting == "date_equal", "known_weighting")
    dates = defaultdict(list)
    for row in rows:
        dates[row["date"]].append(row[metric])
    return avg([avg(v) for v in dates.values()])


def expect_reject(action):
    try:
        action()
    except (AssertionError, ValueError):
        return
    raise AssertionError("Invalid control was not rejected")


def controls():
    expect_reject(lambda: avg([]))
    expect_reject(lambda: avg([float("nan")]))
    expect_reject(lambda: index([{"x": 1}, {"x": 1}], ["x"]))
    close(scores([1., 2.], [2., 4.])["bias_db"], 1.5, "control_bias")
    close(sd([1., 2., 3.]), 1., "control_sample_sd")
    require(sd([1.]) is None, "control_single_run_sd_null")
    a = [{"date": "a", "v": 0.}, {"date": "b", "v": 2.}, {"date": "b", "v": 2.}]
    close(weighted(a, "v", "date_equal"), 1., "control_date_weight")
    close(weighted(a, "v", "hour_equal"), 4/3, "control_hour_weight")
    require([1 + sum(b < a for b in [1., 1., 2.]) for a in [1., 1., 2.]] == [1, 1, 3], "control_tied_ranks")
    require(not (-1e-12 >= 0) and 0. >= 0., "control_strict_zero_input_mask")


def main():
    controls()
    sources = get(OUT / "sources_before.json")
    for p, h in sources.items():
        require(sha(p) == h, "source_sha_before")
    for name, h in get(OUT / "calculation_complete.json")["output_sha256"].items():
        require(sha(OUT / name) == h, "consolidated_output_sha")
    tables = {s: pq.read_table(rev(12) / ("selected_local.parquet" if s == "LOCAL" else "selected_uk.parquet")).to_pylist() for s in ("LOCAL", "UK")}
    selected = {}
    for row in tables["LOCAL"] + tables["UK"]:
        k = key(row)
        require(k not in selected, "selected_unique_utc_key")
        selected[k] = row
    require(len(selected) == 6351, "full_6351_hour_mother_set")
    old11 = {}
    for dataset in ("local", "uk"):
        for row in pq.read_table(rev(11) / f"predictions_{dataset}.parquet").to_pylist():
            if dataset == "local":
                row["station"] = "LOCAL"
            k = key(row, "hour_bjt" if dataset == "local" else "hour_utc")
            require(k not in old11, "itu011_unique_key")
            old11[k] = row
    require(set(old11) == set(selected), "itu011_selected012_full_key_identity")
    fixed = get(rev(11) / "fixed_reference_parameters.json")
    for k, row in selected.items():
        previous = old11[k]
        for field in RUNS + ("truth_db",):
            close(row[field], previous[field], "prediction_identity", 0.)
        require(row["event"] == previous["event"] and row["common_score_eligible"] == previous["common_score_eligible"], "event_and_mask_011_012_identity")
        if row["common_score_eligible"]:
            require(all(math.isfinite(row[m]) for m in RUNS + ("truth_db",)), "eligible_predictions_finite")
            close(previous["gas_reference_db"], fixed[k[0]]["gas_absolute_db"], "fixed_reference", 0.)
            close(previous["gas_absolute_db"] - previous["gas_reference_db"] + previous["cloud_db"] + previous["itu838_geometric_proxy"], row["itu_combined_geometric_approx"], "itu_component_identity")

    weather = pq.read_table(BASE / "Outputs/Datasets/Revision_002/era5_hourly_pressure_levels.parquet",
        columns=["hour_bjt", "surface__total_precipitation", "surface__snowfall"]).to_pylist()
    weather_index = {epoch(r["hour_bjt"]): r for r in weather}
    require(len(weather_index) == len(weather), "local_weather_unique")
    audit27 = {key(r, "hour_utc"): r for r in get(rev(27) / "all_hour_ledger.json")}
    ledger = get(OUT / "all_hour_ledger.json")
    ledger_map = index(ledger, ["station", "utc_epoch_seconds"])
    require(set(ledger_map) == set(selected), "ledger_full_key_identity")
    liquid = {}
    for k, row in selected.items():
        line = ledger_map[k]
        require(line["event"] == row["event"] and line["legacy012_common_score_eligible"] == row["common_score_eligible"], "ledger_selected_identity")
        require(epoch(line["time_utc"]) == k[1], "ledger_utc_normalized")
        day = datetime.fromtimestamp(k[1], timezone(timedelta(hours=8)) if k[0] == "LOCAL" else timezone.utc).date().isoformat()
        require(day == row["date"] == line["date"], "date_timezone_identity")
        if k[0] == "LOCAL":
            w = weather_index[k[1] + 3600]
            tp, sf = w["surface__total_precipitation"], w["surface__snowfall"]
            diff = 1000*tp - 1000*sf
            valid = math.isfinite(tp) and math.isfinite(sf) and tp >= 0 and sf >= 0 and diff >= 0
            liquid[k] = valid
            a = audit27[k]
            require(valid == a["supplement_eligible"] == line["liquid_input_common_eligible"], "local_mask_from_raw_weather")
            close(diff, a["signed_liquid_proxy_mmph"], "local_signed_liquid_identity", 0.)
            require(epoch(a["source_valid_utc"]) == k[1]+3600, "local_t_plus_one_input_alignment")
            require(line["local_liquid_audit"] == a, "local_full_audit_preserved")
        else:
            p = old11[k]
            total = 3600*p["mean_total_precipitation_rate"]
            snow = 3600*p["mean_snowfall_rate"]
            require(math.isfinite(total) and math.isfinite(snow) and 0 <= p["tcc"] <= 1, "uk_weather_finite")
            event = "rain" if total >= .1 and snow < .1 else "dry_overcast" if total < .1 and snow < .1 and p["tcc"] >= .8 else "other"
            require(event == row["event"], "uk_weather_classification")
            close(total, p["total_water_mm"], "uk_rate_conversion", 0.)
            close(snow, p["snow_water_mm"], "uk_rate_conversion", 0.)
            close(total-snow, p["rain_liquid_mm_per_h"], "uk_signed_liquid", 1e-12)
            primary = p["truth_db"] is not None and math.isfinite(p["truth_db"]) and p["valid_seconds"] >= 3420
            require(primary == p["primary_score_eligible"] == line["primary_score_eligible"], "uk_label_quality_gate")
            physical = math.isfinite(total-snow) and total-snow >= 0
            require(physical == p["physical_input_valid"], "uk_physical_input_gate")
            require(row["common_score_eligible"] == (primary and physical), "uk_common_gate")
            require(line["liquid_input_common_eligible"] == row["common_score_eligible"], "uk_mask_unchanged")
            require(epoch(p["rain_valid_utc_end"]) == k[1] + 3600, "uk_t_plus_one_input_alignment")
            liquid[k] = row["common_score_eligible"]
    require(sum(liquid[k] for k in selected if k[0] == "LOCAL") == 684, "local_684_legal_input_hours")
    require(sum(liquid[k] for k in selected if k[0] != "LOCAL") == 5538, "uk_5538_common_hours")
    require(sum(r["primary_score_eligible"] for r in ledger if r["station"] != "LOCAL") == 5593, "uk_5593_label_hours")
    require({key(r, "hour_utc") for r in get(OUT / "local_excluded_hours.json")} == {k for k in selected if k[0] == "LOCAL" and not liquid[k]}, "three_anomalies_retained")

    coverage = get(OUT / "coverage_by_date.json")
    covmap = index(coverage, ["station", "date"])
    bydate = defaultdict(list)
    for r in ledger:
        bydate[(r["station"], r["date"])].append(r)
    require(set(bydate) == set(covmap), "all_dates_retained")
    require(Counter(k[0] for k in bydate) == {"LOCAL": 156, "chilbolton": 143, "sparsholt": 93}, "all_station_date_counts")
    for k, rows in bydate.items():
        c = covmap[k]
        require(len(rows) == c["source_hours"], "date_source_count")
        if k[0] != "LOCAL":
            require(len(rows) == 24, "uk_completed_days_24h")
        for f, old in (("label_eligible_hours", "primary_score_eligible"), ("legacy_common_hours", "legacy012_common_score_eligible"), ("liquid_common_hours", "liquid_input_common_eligible")):
            require(c[f] == sum(r[old] for r in rows), "date_gate_count")
        for ev in ("rain", "dry_overcast", "other"):
            require(c["all_weather_hours"][ev] == sum(r["event"] == ev for r in rows), "date_all_weather_count")
        for ev in ("rain", "dry_overcast"):
            for field, gate in (("legacy_weather_hours", "legacy012_common_score_eligible"), ("liquid_weather_hours", "liquid_input_common_eligible")):
                require(c[field][ev] == sum(r["event"] == ev and r[gate] for r in rows), "date_scored_weather_count")

    def selected_keys(cohort, event):
        result = []
        for k, r in selected.items():
            if r["event"] != event:
                continue
            if cohort.startswith("LOCAL"):
                ok = k[0] == "LOCAL" and (r["common_score_eligible"] if cohort == "LOCAL_legacy_687_reference" else liquid[k])
            else:
                ok = k[0] != "LOCAL" and r["common_score_eligible"]
                for station in ("chilbolton", "sparsholt"):
                    if station in cohort:
                        ok = ok and k[0] == station
            if ok:
                result.append(k)
        return set(result)

    member = index(get(OUT / "sample_membership.json"), ["cohort", "event"])
    run_hour = get(OUT / "hourly_run_metrics.json")
    index(run_hour, ["cohort", "event", "model"])
    hour_values = {}
    for row in run_hour:
        keys = selected_keys(row["cohort"], row["event"])
        require(len(keys) == row["n"], "hourly_run_sample_count")
        vals = scores([selected[k]["truth_db"] for k in sorted(keys)], [selected[k][row["model"]] for k in sorted(keys)])
        for m, v in vals.items():
            close(v, row[m], "hourly_run_metrics")
        hour_values[(row["cohort"], row["event"], row["model"])] = vals
        recorded = member[(row["cohort"], row["event"])]
        require(keys == {(s, epoch(t)) for s, t in recorded["keys"]} and len(keys) == recorded["n"], "scored_membership_full_identity")
    family_hour = get(OUT / "hourly_family_metrics.json")
    require(len(run_hour) == 80 and len(family_hour) == 40, "hourly_result_counts")
    for r in family_hour:
        family_runs = FAMILIES[r["family"]]
        require(len(family_runs) == r["n_runs"], "frozen_seed_count")
        for m in ("rmse_db", "mae_db", "bias_db"):
            v = [hour_values[(r["cohort"], r["event"], run)][m] for run in family_runs]
            close(avg(v), r[m+"_mean"], "hourly_family_mean")
            close(sd(v), r[m+"_sd"], "hourly_family_sample_sd")
        require(r["rank"] == 1 + sum(x["rmse_db_mean"] < r["rmse_db_mean"] for x in family_hour if x["cohort"] == r["cohort"] and x["event"] == r["event"]), "hourly_competition_rank")
        kset = selected_keys(r["cohort"], r["event"])
        dates = {s: sorted({selected[k]["date"] for k in kset if k[0] == s}) for s in sorted({k[0] for k in kset})}
        require(dates == r["dates_by_station"] == member[(r["cohort"], r["event"])]["dates_by_station"], "actual_date_list_identity")

    dynamic = pq.read_table(rev(17) / "hourly_scores.parquet").to_pylist()
    index(dynamic, ["station", "time", "run"])
    runkeys = defaultdict(set)
    for r in dynamic:
        k = key(r)
        require(k in selected and r["run"] in RUNS, "dynamic_source_key_and_run")
        s = selected[k]
        require(r["event"] == s["event"] and r["date"] == s["date"] and s["common_score_eligible"], "dynamic_event_date_mask")
        close(r["truth_hour_db"], s["truth_db"], "dynamic_hour_truth_identity", 0.)
        close(r["prediction_hour_db"], s[r["run"]], "dynamic_hour_prediction_identity", 0.)
        require(all(math.isfinite(r[m]) for m in DMET), "dynamic_frozen_scores_finite")
        runkeys[r["run"]].add(k)
    mother_dynamic = selected_keys("LOCAL_legacy_687_reference", "rain") | selected_keys("LOCAL_legacy_687_reference", "dry_overcast") | selected_keys("UK", "rain") | selected_keys("UK", "dry_overcast")
    require(len(dynamic) == 25888 and all(v == mother_dynamic for v in runkeys.values()) and len(runkeys) == 8, "dynamic_all_eight_runs_identical_keys")
    dynruns = get(OUT / "dynamic_run_metrics.json")
    dynfamilies = get(OUT / "dynamic_family_metrics.json")
    index(dynruns, ["cohort", "event", "weighting", "run"])
    dynvalue = {}
    for r in dynruns:
        kset = selected_keys(r["cohort"], r["event"])
        rows = [x for x in dynamic if x["run"] == r["run"] and key(x) in kset]
        require(len(rows) == len(kset) == r["n_station_hours"], "dynamic_run_count")
        dates = sorted({x["date"] for x in rows})
        require(len(dates) == r["n_dates"] and dates[0] == r["date_min"] and dates[-1] == r["date_max"], "dynamic_dates")
        require(sum(x["n_seconds"] for x in rows) == r["n_seconds"], "dynamic_native_seconds")
        values = {m: weighted(rows, m, r["weighting"]) for m in DMET}
        for m in DMET:
            close(values[m], r[m], "dynamic_run_aggregation")
        dynvalue[(r["cohort"], r["event"], r["weighting"], r["run"])] = values
    for r in dynfamilies:
        fullfamily = next(f for f in FAMILIES if SHORT[f] == r["family"])
        runs = FAMILIES[fullfamily]
        require(len(runs) == r["n_frozen_runs"], "dynamic_seed_count")
        for m in DMET:
            v = [dynvalue[(r["cohort"], r["event"], r["weighting"], run)][m] for run in runs]
            close(avg(v), r[m], "dynamic_family_aggregation")
            close(sd(v), r[m+"_sd"], "dynamic_family_sample_sd")
        require(r["rank_crps201"] == 1 + sum(x["crps201_db"] < r["crps201_db"] for x in dynfamilies if all(x[f] == r[f] for f in ("cohort", "event", "weighting"))), "dynamic_crps_rank")
    require(len(dynruns) == 160 and len(dynfamilies) == 80, "dynamic_result_counts")
    removed = {k for k in mother_dynamic if k[0] == "LOCAL" and not liquid[k]}
    require(len(removed) == 2 and removed == {key(r) for r in get(OUT / "dynamic_mask_audit.json")["excluded_keys"]}, "dynamic_two_invalid_hours_removed_only_in_supplement")

    temporal = pq.read_table(rev(18) / "attempt_003/hourly_scores.parquet").to_pylist()
    index(temporal, ["station", "hour_utc", "candidate"])
    ukdynamic = {k for k in mother_dynamic if k[0] != "LOCAL"}
    require(len(temporal) == 6054, "temporal_6054_rows")
    for candidate in ("iid", "block_0256"):
        require({key(r, "hour_utc") for r in temporal if r["candidate"] == candidate} == ukdynamic, "temporal_full_uk_mask")
    lag_records = pq.read_table(rev(18) / "attempt_003/lag_diagnostics.parquet").to_pylist()
    index(lag_records, ["station", "hour_utc", "candidate", "lag_s"])
    lag_groups = defaultdict(list)
    expected_lags = {1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1800}
    for lag in lag_records:
        lag_groups[(key(lag, "hour_utc"), lag["candidate"])].append(lag)
        require(lag["n_pairs"] >= 0 and 0 <= lag["generated_acf_finite_reps"] <= 32, "lag_counts_in_bounds")
        if lag["n_pairs"] == 0:
            require(all(lag[m] is None for m in ("observed_increment_db", "observed_acf", "generated_increment_db", "generated_acf")), "zero_pair_lags_not_imputed")
    require(len(lag_records) == 66594 and len(lag_groups) == 6054, "complete_lag_inventory")
    for r in temporal:
        s = selected[key(r, "hour_utc")]
        require(r["date"] == s["date"] and r["event"] == s["event"], "temporal_date_event_identity")
        lg = lag_groups[(key(r, "hour_utc"), r["candidate"])]
        require({x["lag_s"] for x in lg} == expected_lags, "temporal_all_eleven_lags_recorded")
        require(r["n_lags"] == sum(x["observed_increment_db"] is not None and math.isfinite(x["observed_increment_db"]) for x in lg), "temporal_actual_usable_lags")
        require(all(x["date"] == r["date"] and x["event"] == r["event"] for x in lg), "lag_date_event_identity")
    tr = get(OUT / "temporal_reuse.json")
    require(tr["family_rank_applicable"] is False and len(tr["rows"]) == 24, "no_temporal_family_ranking")
    for r in tr["rows"]:
        rows = [x for x in temporal if x["event"] == r["event"] and x["candidate"] == r["candidate"] and (r["station"] == "pooled" or x["station"] == r["station"])]
        require(len(rows) == r["n_station_hours"] and len({x["date"] for x in rows}) == r["n_dates"], "temporal_count_dates")
        require(min(epoch(x["hour_utc"]) for x in rows) == epoch(r["start_utc"]) and max(epoch(x["hour_utc"]) for x in rows) == epoch(r["end_utc"]), "temporal_actual_endpoints")
        for m in ("vs16", "vs32", "plugin16", "plugin32"):
            require(sum(math.isfinite(x[m]) for x in rows) == r[m+"_finite_hours"] == len(rows), "temporal_finite_count")
            close(weighted(rows, m, r["weighting"]), r[m], "temporal_aggregation")
    require(get(rev(17) / "final_verification.json")["status"].startswith("PASS") and get(rev(18) / "attempt_003/independent_qa.json")["status"].startswith("PASS"), "native_second_and_no_gap_qa_reused")
    for p, h in sources.items():
        require(sha(p) == h, "source_sha_after")
    put("verification.json", {
        "status": "PASS_FINAL_NUMERIC_CONSOLIDATION", "utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "Independent PyArrow records + standard library math.fsum/statistics; no production imports",
        "reviewer": "same agent, separate implementation; not an independent human review",
        "checks": dict(CHECKS), "max_absolute_errors": dict(ERRORS), "protected_sources": len(sources),
        "full_hours": 6351, "full_station_dates": 392, "local_legacy_hours": 687, "local_legal_liquid_hours": 684,
        "uk_weather_hours": 5664, "uk_label_hours": 5593, "uk_physical_common_hours": 5538,
        "hourly_run_metrics": 80, "hourly_family_metrics": 40,
        "frozen_dynamic_run_hour_rows": 25888, "local_dynamic_supplement_hours": 207,
        "dynamic_run_metrics": 160, "dynamic_family_metrics": 80, "temporal_rows": 6054, "temporal_summary_rows": 24,
        "new_predictions": 0, "new_trajectories": 0, "new_training": 0,
        "human_manuscript_approval": "NOT_CLAIMED_NOT_PART_OF_EXPERIMENT_PACKAGE_QA",
        "verifier_sha256": sha(__file__), "calculation_complete_sha256": sha(OUT / "calculation_complete.json")})
    print("PASS_FINAL_NUMERIC_CONSOLIDATION", dict(ERRORS))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        put("qa_failure_" + stamp + ".json", {"utc": stamp, "traceback": traceback.format_exc(), "checks": dict(CHECKS), "max_errors": dict(ERRORS)})
        raise
