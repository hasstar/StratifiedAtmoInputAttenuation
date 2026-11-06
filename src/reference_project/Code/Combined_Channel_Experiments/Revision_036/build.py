"""Consolidate accepted results; only the LOCAL dynamic mask adds aggregation."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import json
import os
import platform
import sys
import traceback
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_036"
REV = lambda n: ROOT / f"Outputs/Experiments/Combined_Channel_Revision_{n:03d}"
FAMILIES = {
    "ours_frozen_top10": ["ours_frozen_top10"],
    "bai2022_lstm_t0_adapted": [f"bai2022_lstm_t0_adapted_seed{s}" for s in (17, 29, 43)],
    "tabm2025_official_budgeted": [f"tabm2025_official_budgeted_seed{s}" for s in (17, 29, 43)],
    "itu_combined_geometric_approx": ["itu_combined_geometric_approx"],
}
DYN_FAMILY = dict(zip(FAMILIES, ["ours", "bai_lstm", "tabm", "itu"]))
METRICS = ["crps201_db", "crps401_db", "central90_coverage", "central98_coverage", "tail95_exceedance", "tail99_exceedance", "slow_only_crps_db"]
EVENTS = ("rain", "dry_overcast")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def put(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def key(station, time):
    return station, int(pd.Timestamp(time).timestamp())


def unique(rows, fields):
    result = {tuple(r[f] for f in fields): r for r in rows}
    assert len(result) == len(rows), f"Duplicate {fields}"
    return result


def summarize_local(frame):
    assert not frame.duplicated(["station", "time", "run"]).any()
    assert np.isfinite(frame[METRICS]).all().all()
    runs, families = [], []
    for event in EVENTS:
        part = frame[frame.event.eq(event)]
        for weighting in ("date_equal", "hour_equal"):
            cell = []
            for run, group in part.groupby("run", sort=True):
                assert len(group) and group.time.nunique() == len(group)
                data = group.groupby("date")[METRICS].mean() if weighting == "date_equal" else group[METRICS]
                row = {"cohort": "LOCAL_liquid_input_valid_supplement", "event": event, "weighting": weighting,
                       "run": run, "family": group.family.iloc[0], "n_station_hours": len(group),
                       "n_dates": int(group.date.nunique()), "n_seconds": int(group.n_seconds.sum()),
                       "date_min": group.date.min(), "date_max": group.date.max(),
                       **{m: float(data[m].mean()) for m in METRICS}}
                cell.append(row)
            assert len(cell) == 8
            runs.extend(cell)
            family_cell = []
            for family in DYN_FAMILY.values():
                values = [r for r in cell if r["family"] == family]
                r = {k: values[0][k] for k in ["cohort", "event", "weighting", "n_station_hours", "n_dates", "n_seconds", "date_min", "date_max"]}
                r.update(family=family, n_frozen_runs=len(values))
                for m in METRICS:
                    a = np.array([v[m] for v in values])
                    r[m] = float(a.mean())
                    r[m + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else None
                family_cell.append(r)
            for r in family_cell:
                r["rank_crps201"] = 1 + sum(v["crps201_db"] < r["crps201_db"] for v in family_cell)
            families.extend(family_cell)
    return runs, families


def main():
    guards = read(REV(35) / "sources_before.json")
    for change in read(REV(35) / "final_verification.json")["allowed_changes"]:
        assert guards[change["path"]] == change["before_sha256"]
        guards[change["path"]] = change["after_sha256"]
    guards.update(read(REV(35) / "manifest.json"))
    assert all(sha(p) == h for p, h in guards.items()), "Prior source changed"
    for n in (11, 12, 17, 18, 27):
        assert read(REV(n) / "final_verification.json")["status"].startswith("PASS")
        for p in REV(n).rglob("*"):
            if p.is_file():
                if str(p) in guards:
                    assert guards[str(p)] == sha(p)
                guards[str(p)] = sha(p)
    for p in CODE.iterdir():
        if p.is_file():
            guards[str(p)] = sha(p)
    fast = ROOT / "Outputs/Experiments/Fast_Fading_Revision_002/final_evaluation/delivery"
    for name in ("FINAL_RESULTS_ZH.md", "delivery_qa.json", "report_manifest.json"):
        guards[str(fast / name)] = sha(fast / name)
    put("sources_before.json", guards)
    put("environment.json", {"utc": utc(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
        "command": "python.exe -B Code/Combined_Channel_Experiments/Revision_036/build.py", "network_calls": 0, "model_loads": 0})

    local = pd.read_parquet(REV(12) / "selected_local.parquet")
    uk = pd.read_parquet(REV(12) / "selected_uk.parquet")
    assert len(local) == 687 and len(uk) == 5664
    raw_ledger = read(REV(12) / "all_hour_ledger.json")
    liquid = {key(r["station"], r["hour_utc"]): r for r in read(REV(27) / "all_hour_ledger.json")}
    assert len(liquid) == 687 and sum(x["supplement_eligible"] for x in liquid.values()) == 684
    ledger = []
    for old in raw_ledger:
        k = key(old["station"], old["time"])
        r = {**old, "utc_epoch_seconds": k[1], "time_utc": datetime.fromtimestamp(k[1], timezone.utc).isoformat(),
             "date_timezone": "Asia/Shanghai" if k[0] == "LOCAL" else "UTC"}
        r["legacy012_common_score_eligible"] = r.pop("common_score_eligible")
        r["legacy012_physical_input_valid"] = r.pop("physical_input_valid")
        r["liquid_input_common_eligible"] = liquid[k]["supplement_eligible"] if k[0] == "LOCAL" else r["legacy012_common_score_eligible"]
        r["local_liquid_audit"] = liquid.get(k)
        ledger.append(r)
    assert len(unique(ledger, ["station", "utc_epoch_seconds"])) == 6351
    coverage = []
    for (station, date), group in pd.DataFrame(ledger).groupby(["station", "date"], sort=True):
        coverage.append({"station": station, "date": date, "date_timezone": group.date_timezone.iloc[0],
            "source_hours": len(group), "label_eligible_hours": int(group.primary_score_eligible.sum()),
            "legacy_common_hours": int(group.legacy012_common_score_eligible.sum()),
            "liquid_common_hours": int(group.liquid_input_common_eligible.sum()),
            "all_weather_hours": {e: int(group.event.eq(e).sum()) for e in (*EVENTS, "other")},
            "legacy_weather_hours": {e: int((group.event.eq(e) & group.legacy012_common_score_eligible).sum()) for e in EVENTS},
            "liquid_weather_hours": {e: int((group.event.eq(e) & group.liquid_input_common_eligible).sum()) for e in EVENTS}})
    assert len(coverage) == 392
    put("all_hour_ledger.json", ledger)
    put("coverage_by_date.json", coverage)
    put("local_excluded_hours.json", read(REV(27) / "excluded_hours.json"))
    put("mask_contract.json", read(REV(27) / "mask_contract.json"))

    hourly = [r for r in read(REV(27) / "seven_family_rankings.json") if r["family"] in FAMILIES]
    hourly += [r for r in read(REV(12) / "seven_family_rankings.json") if r["family"] in FAMILIES and r["cohort"] != "LOCAL_test"]
    for r in hourly:
        r.pop("name", None)
        if "dates_bjt" in r:
            r["dates_by_station"] = {"LOCAL": r["dates_bjt"]}
        r["source_revision"] = 27 if r["cohort"].startswith("LOCAL") else 12
    run_hourly = read(REV(27) / "run_metrics.json")
    for r in run_hourly:
        r["model"] = r.pop("run")
    run_hourly += [r for r in read(REV(12) / "run_metrics.json") if r["cohort"] != "LOCAL_test"]
    assert len(hourly) == 40 and len(run_hourly) == 80
    assert len(unique(hourly, ["cohort", "event", "family"])) == 40
    memberships = []
    for cohort in dict.fromkeys(r["cohort"] for r in hourly):
        group = [r for r in ledger if (r["station"] == "LOCAL" if cohort.startswith("LOCAL") else r["station"] != "LOCAL")]
        if cohort.startswith("UK_chilbolton"):
            group = [r for r in group if r["station"] == "chilbolton"]
        if cohort.startswith("UK_sparsholt"):
            group = [r for r in group if r["station"] == "sparsholt"]
        flag = "legacy012_common_score_eligible" if cohort == "LOCAL_legacy_687_reference" else "liquid_input_common_eligible"
        for event in EVENTS:
            selected = [r for r in group if r[flag] and r["event"] == event]
            memberships.append({"cohort": cohort, "event": event, "n": len(selected),
                "keys": [[r["station"], r["time_utc"]] for r in selected],
                "dates_by_station": {s: sorted({r["date"] for r in selected if r["station"] == s}) for s in sorted({r["station"] for r in selected})}})
    put("hourly_family_metrics.json", hourly)
    put("hourly_run_metrics.json", run_hourly)
    put("sample_membership.json", memberships)

    dynamic = pd.read_parquet(REV(17) / "hourly_scores.parquet")
    assert len(dynamic) == 25888 and not dynamic.duplicated(["station", "time", "run"]).any()
    good = dynamic.station.eq("LOCAL") & dynamic.time.map(lambda t: liquid[key("LOCAL", t)]["supplement_eligible"] if key("LOCAL", t) in liquid else False)
    subset = dynamic[good].copy()
    assert len(subset) == 207 * 8
    extra_runs, extra_families = summarize_local(subset)
    dynamic_runs, dynamic_families = read(REV(17) / "run_metrics.json"), read(REV(17) / "family_metrics.json")
    for r in dynamic_runs + dynamic_families:
        if r["cohort"] == "LOCAL":
            r["cohort"] = "LOCAL_legacy_687_reference"
    put("dynamic_run_metrics.json", dynamic_runs + extra_runs)
    put("dynamic_family_metrics.json", dynamic_families + extra_families)
    excluded_dyn = dynamic[dynamic.station.eq("LOCAL") & ~good]
    put("dynamic_mask_audit.json", {"old_local_hours": 209, "new_local_hours": 207,
        "old_local_run_hour_rows": 1672, "new_local_run_hour_rows": len(subset),
        "excluded_keys": excluded_dyn[["station", "time", "date", "event"]].drop_duplicates().to_dict("records"),
        "new_family_aggregate_rows": len(extra_families), "new_run_aggregate_rows": len(extra_runs),
        "new_per_second_scores": 0, "new_trajectories": 0, "source": str(REV(17) / "hourly_scores.parquet")})
    assert read(REV(18) / "attempt_003/independent_qa.json")["status"].startswith("PASS")
    put("temporal_reuse.json", {"status": "REUSED_INDEPENDENT_QA_VERIFIED", "source_revision": 18,
        "hourly_source": str(REV(18) / "attempt_003/hourly_scores.parquet"),
        "all_lags_source": str(REV(18) / "attempt_003/lag_diagnostics.parquet"),
        "rows": read(REV(18) / "attempt_003/summary.json")["rows"], "family_rank_applicable": False})
    put("model_scope.json", {**read(REV(35) / "MODEL_SCOPE.json"), "latest_consolidation_revision": 36,
        "four_family_final_package_complete": False, "status": "CONSOLIDATED_PENDING_INDEPENDENT_QA"})
    assert all(sha(p) == h for p, h in guards.items()), "Source changed during consolidation"
    put("calculation_complete.json", {"utc": utc(), "status": "CONSOLIDATED_PENDING_INDEPENDENT_QA",
        "hourly_family_rows": 40, "hourly_run_rows": 80, "dynamic_family_rows": 80, "dynamic_run_rows": 160,
        "ledger_rows": 6351, "date_rows": 392, "new_predictions": 0, "new_training": 0, "new_trajectories": 0,
        "protected_sources": len(guards), "output_sha256": {p.name: sha(p) for p in OUT.glob("*.json")}})
    print("Consolidated 40 hourly family rows, 80 dynamic family rows; independent QA pending.", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=False)
    with (OUT / "build.stdout.log").open("x", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log):
                main()
        except BaseException:
            put("build_failure.json", {"utc": utc(), "traceback": traceback.format_exc()})
            raise
    print("BUILD_OK; no publication claim before QA")
