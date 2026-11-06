"""New combination analysis. All inherited experiment directories are read-only."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001"
LOCAL = OUT / "local"
FAST_CODE = ROOT / "Code/Fast_Fading_Experiments/Revision_002"
FAST = ROOT / "Outputs/Experiments/Fast_Fading_Revision_002"
D1 = FAST / "final_evaluation"
RAW = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001/clean_1s"
SLOW = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced"
DATA = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
sys.path.insert(0, str(FAST_CODE))
import numpy as np
import pandas as pd
import fit_marginals as fm
import run_dynamics as dyn
import run_final_evaluation as original_d1
import window_diagnostics as wd

FAMILIES = ("bilstm", "transformer", "mlp", "svr_rbf", "xgboost", "catboost",
            "gcn", "lightgbm", "random_forest", "elastic_net")
VARIANTS = tuple("full:" + n for n in FAMILIES) + ("base:bilstm",)
GROUPS = ("overall", "clear_dry", "rain", "dry_overcast", "other")
ZH = dict(zip(GROUPS, ("总体", "晴天", "雨天（含全部强雨）", "干燥阴天", "其他天气")))
METRICS = ("crps201_db", "crps401_db", "central90_coverage", "central98_coverage",
           "tail95_exceedance", "tail99_exceedance", "slow_only_instant_mae_crps_db")
SPLITS = {"train": 2404, "validation": 343, "test": 687}
CHUNK = 32


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic(path, obj):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), "Write outside new output denied")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def parquet(path, frame):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), "Write outside new output denied")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


@contextlib.contextmanager
def lock(command):
    import msvcrt
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / "writer.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        start = now()
        active = {"pid": os.getpid(), "python": sys.executable, "argv": sys.argv,
                  "command": command, "status": "RUNNING", "started_utc": start}
        atomic(LOCAL / "active_worker.json", active)
        status = "EXITED_NORMAL"
        try:
            yield
        except BaseException:
            status = "FAILED"
            atomic(LOCAL / "failures" / (str(time.time_ns()) + ".json"),
                   {**active, "failed_utc": now(), "traceback": traceback.format_exc()})
            raise
        finally:
            atomic(LOCAL / "active_worker.json", {**active, "status": status, "finished_utc": now()})
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def slow_module():
    path = ROOT / "Code/Hourly_Attenuation_Experiments/Revision_028/run_weather.py"
    spec = importlib.util.spec_from_file_location("combined_frozen_weather28", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        raise RuntimeError("Any local model fit is forbidden")
    module.e.fit = forbidden  # RAM-only guard; frozen source files are never edited.
    return module


def source_inventory():
    slow_manifest = read(SLOW / "run_manifest.json")
    old = read(D1 / "manifest.json")
    old_qa = read(D1 / "qa.json")
    require(old_qa["status"] == "PASS" and sha(D1 / "manifest.json") == old_qa["manifest_sha256"],
            "Frozen fast D1 QA/manifest mismatch")
    expected = {}
    def pin(path, digest=None):
        path = Path(path)
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        actual = sha(path)
        require(digest is None or actual == digest, "Source changed: " + key)
        if key in expected:
            require(expected[key] == actual, "Conflicting source digest")
        expected[key] = actual
    for name, digest in slow_manifest["source_sha256"].items():
        pin(ROOT / name, digest)
    for filename in ("run_manifest.json", "selection.json", "selection_qa.json", "train_qa.json",
                     "independent_qa.json", "delivery_qa.json", "Seed_017_Report/seed17_report_qa.json"):
        pin(SLOW / filename)
        if filename.endswith("qa.json"):
            require(read(SLOW / filename)["status"] == "PASS", "Slow QA failed: " + filename)
    plans = read(SLOW / "selection.json")["plans"]
    for variant in VARIANTS:
        case, name = variant.split(":")
        rec = read(SLOW / "training" / f"{case}_{name}_seed17.json")
        require(rec["seed"] == 17 and rec["plan"] == plans[case][name] and
                not rec["test_read_for_fit"] and not rec["validation_changed_formal_fit"],
                "Frozen slow fit/selection changed")
        require(len(rec["feature_names"]) == (156 if case == "full" else 50), "Input count changed")
        pin(SLOW / "training" / f"{case}_{name}_seed17.json")
        pin(SLOW / "models" / f"{case}_{name}_seed17.joblib", rec["model_sha256"])
        test = read(SLOW / "test" / f"{case}_{name}_seed17.json")
        require(test["fit_calls_during_evaluation"] == 0 and test["model_sha256"] == rec["model_sha256"],
                "Stored test prediction provenance changed")
        pin(SLOW / "test" / f"{case}_{name}_seed17.json")
    require(plans["full"]["bilstm"]["formal_rounds"] == 53 and
            plans["base"]["bilstm"]["formal_rounds"] == 46, "Original 53/46 plans must not change")
    for split in SPLITS:
        pin(DATA / (split + ".parquet"))
    pin(DATA / "hourly_sample_status.parquet")
    for filename in ("manifest.json", "qa.json", "report_qa.json", "numerical_tests_qa.json"):
        pin(D1 / filename)
    for name, digest in old["frozen_design"]["source_clean_1s"].items():
        pin(RAW / name, digest)
    for filename in ("marginal_model/qa.json",):
        if (FAST / filename).exists():
            pin(FAST / filename)
    for filename in ("marginal_model_qa.json", "final_generator/model.json", "final_generator/qa.json",
                     "dynamics/r16/selection.json", "stationarity/selection.json"):
        pin(FAST / filename)
    for component in read(FAST / "final_generator/model.json")["components"].values():
        pin(FAST / component["cdf_file_relative_to_revision_output"], component["cdf_sha256"])
        pin(FAST / "final_generator" / component["library_file"], component["library_sha256"])
    for name in ("fit_marginals.py", "run_dynamics.py", "run_final_evaluation.py", "window_diagnostics.py",
                 "freeze_generator.py", "run_stationarity.py", "FINAL_EVALUATION_PLAN.md"):
        pin(FAST_CODE / name, old["frozen_design"]["pinned_files"].get(
            "Code/Fast_Fading_Experiments/Revision_002/" + name))
    for name, digest in old_qa["checkpoint_hashes"].items():
        p = D1 / "chunks" / name
        pin(p, digest)
        marker = read(p)
        require(marker["status"] == "complete", "Old D1 checkpoint incomplete")
        for filename, value in marker["output_sha256"].items():
            if filename.endswith(("_marginal.parquet", "_observed.parquet", "_scores.parquet", "_replicates.parquet")) \
                    or filename.startswith("example_"):
                pin(D1 / "chunks" / filename, value)
    for filename in ("local_engine.py", "run_local.py", "verify_local.py", "PROTOCOL.md", "PARAMETERS.md"):
        pin(HERE / filename)
    return expected


def prepare():
    from verify_local import numerical
    qa = numerical(save=True)
    require(qa["status"] == "PASS" and qa["study_sample_reads"] == 0, "Synthetic gate failed")
    sources = source_inventory()
    design = {"version": "combined-local-rev028-seed17-v1", "variants": list(VARIANTS),
        "main": "full:bilstm", "seed": 17, "main_rounds": 53, "original_base_rounds": 46,
        "official_splits": SPLITS, "combination_splits": ["validation", "test"],
        "combination_hours": 1030, "combination_records": 11330, "new_fits": 0,
        "fast_generator": "Fast_Fading_Revision_002/block_0256", "replicates_reused": 32,
        "prior_dynamic_records_reused": 2060, "prior_realization_records_reused": 65920,
        "min_records_per_second": 9, "crps_grids": [201, 401], "bootstrap_replicates": 2000,
        "block_days": [1, 3, 7], "resume_chunk_hours": CHUNK,
        "historical_test_access": True, "select_from_new_test": False, "source_sha256": sources,
        "numerical_qa_sha256": sha(LOCAL / "numerical_qa.json")}
    design["fingerprint"] = fingerprint(design)
    path = LOCAL / "manifest.json"
    if path.exists():
        saved = read(path)
        require(saved["fingerprint"] == design["fingerprint"], "New frozen design changed")
        return saved
    design["created_utc"] = now()
    atomic(path, design)
    print("LOCAL PREPARED", design["fingerprint"], "sources", len(sources), flush=True)
    return design


def manifest(check_sources=False):
    m = read(LOCAL / "manifest.json")
    require(sha(LOCAL / "numerical_qa.json") == m["numerical_qa_sha256"], "Numerical QA changed")
    paths = m["source_sha256"] if check_sources else {
        k: v for k, v in m["source_sha256"].items() if k.startswith("Code/Combined_Channel_Experiments/")}
    for filename, digest in paths.items():
        require(sha(ROOT / filename) == digest, "Frozen source changed: " + filename)
    return m


def load_cdfs():
    package = read(FAST / "final_generator/model.json")
    models = {}
    for event, component in package["components"].items():
        path = FAST / component["cdf_file_relative_to_revision_output"]
        require(sha(path) == component["cdf_sha256"], "CDF changed")
        with np.load(path) as data:
            model = {"family": "empirical", "support_q": data["q"], "support_cdf": data["cdf"]}
        weights = np.diff(np.r_[0., model["support_cdf"]])
        require(np.all(weights > 0) and abs(np.dot(weights, model["support_q"]) - 1) < 1e-10,
                "Population unit-power moment failed")
        models[event] = model
    return models


def predictions(replay=False):
    m = manifest()
    weather = slow_module()
    source_manifest = read(SLOW / "run_manifest.json")
    records = []
    largest = 0.
    for split, size in SPLITS.items():
        frame = weather.v.data_frame(split, source_manifest, events=True)
        require(len(frame) == size, "Slow split counts differ")
        hours = frame.hour_bjt.astype(str).tolist()
        truth = frame[source_manifest["target"]].to_numpy(float)
        for variant in VARIANTS:
            case, name = variant.split(":")
            training = read(SLOW / "training" / f"{case}_{name}_seed17.json")
            saved = training if split != "test" else read(SLOW / "test" / f"{case}_{name}_seed17.json")
            keys = (split + "_hour_bjt", split + "_prediction_db") if split != "test" else ("hour_bjt", "prediction_db")
            require(saved[keys[0]] == hours, "Stored slow hours not aligned")
            values = np.asarray(saved[keys[1]], float)
            if split == "test":
                require(np.max(np.abs(truth - saved["truth_db"])) < 1e-10, "Stored slow truth changed")
            if replay:
                bundle = weather.v.joblib.load(SLOW / "models" / f"{case}_{name}_seed17.joblib")
                x = weather.v.raw(frame, weather.case_plan(source_manifest, case))
                reproduction = weather.v.predict(bundle, x)
                delta = float(np.max(np.abs(values - reproduction)))
                largest = max(largest, delta)
                require(delta < 1e-10, "Frozen weight prediction replay failed: " + variant)
                del bundle, x
            for j, hour in enumerate(hours):
                event = "rain" if frame.rain.iloc[j] else ("dry_overcast" if frame.dry_overcast.iloc[j]
                    else ("clear_dry" if frame.clear_dry.iloc[j] else "other"))
                records.append({"hour": hour, "day": hour[:10], "split": split, "event": event,
                    "variant": variant, "truth_db": float(truth[j]), "prediction_db": float(values[j]),
                    "error_db": float(values[j] - truth[j]), "seed": 17,
                    "feature_count": len(training["feature_names"]),
                    "formal_rounds": training["plan"]["formal_rounds"], "model_sha256": training["model_sha256"]})
    frame = pd.DataFrame(records)
    require(len(frame) == 37774 and not frame.duplicated(["split", "hour", "variant"]).any(), "Prediction records differ")
    if replay:
        atomic(LOCAL / "prediction_qa.json", {"status": "PASS", "fingerprint": m["fingerprint"],
            "records": len(frame), "models_replayed": len(VARIANTS), "split_model_replays": 33,
            "max_prediction_error_db": largest, "fit_calls": 0, "completed_utc": now()})
    return frame


def old_frames(kind):
    paths = sorted((D1 / "chunks").glob("*_" + kind + ".parquet"))
    require(paths, "Missing old verified records")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def source_frame(path):
    columns = ["second_bjt", "hour_bjt", "received_power_w", "hour_mean_power_w", "fast_residual_db",
               "quality_zero_records", "k11_records", "k101_records", "split", "weather_stratum",
               "rain", "dry_overcast", "clear_dry"]
    return pd.read_parquet(path, columns=columns, filters=[("split", "in", ["validation", "test"])]) \
        .sort_values(["hour_bjt", "second_bjt"]).reset_index(drop=True)


def observed_hour(group):
    a = wd.hour_arrays(group)
    good = (a["count"] >= 9) & np.isfinite(a["power"]) & (a["power"] > 0)
    require(good.sum() > 0, "No eligible source seconds")
    f = -10 * np.log10(a["power"][good] / a["base"])
    weight = a["count"][good].astype(float)
    hour = original_d1.marginal_hour(f, weight, str(group.hour_bjt.iloc[0]))
    return a, good, f, weight, hour


def checked_marker(path):
    marker = read(path)
    require(marker["status"] == "complete" and marker["fingerprint"] == manifest()["fingerprint"],
            "New checkpoint fingerprint changed")
    for filename, digest in marker["output_sha256"].items():
        require(sha(LOCAL / "chunks" / filename) == digest, "New checkpoint content changed")
    return marker


def run():
    m = manifest(check_sources=True)
    pred = predictions(replay=True)
    prediction_path = LOCAL / "slow_predictions.parquet"
    if prediction_path.exists():
        require(pd.read_parquet(prediction_path).equals(pred), "Prediction output changed")
    else:
        parquet(prediction_path, pred)
    indexed = pred.set_index(["variant", "hour"])
    previous = old_frames("marginal").set_index("hour")
    require(len(previous) == 1030 and previous.index.is_unique, "Old formal hour coverage differs")
    models = load_cdfs()
    source_names = read(D1 / "manifest.json")["frozen_design"]["source_clean_1s"]
    for filename in sorted(source_names):
        frame = source_frame(RAW / filename)
        grouped = list(frame.groupby("hour_bjt", sort=True))
        for start in range(0, len(grouped), CHUNK):
            key = filename.replace(".parquet", "") + f"_{start:04d}"
            marker_path = LOCAL / "chunks" / (key + ".json")
            batch = grouped[start:start + CHUNK]
            expected_hours = [str(h) for h, _ in batch]
            if marker_path.exists():
                require(checked_marker(marker_path)["hours"] == expected_hours, "Checkpoint hour set changed")
                continue
            rows = []
            for h, group in batch:
                h = str(h)
                a, good, f, weight, hour = observed_hour(group)
                prior = previous.loc[h]
                event = original_d1.event_name(group)
                require(event == prior.event and str(group.split.iloc[0]) == prior.split and
                        int(good.sum()) == prior.n_seconds and int(weight.sum()) == prior.n_records,
                        "Old source/quality/hour membership not reproduced")
                truth = original_d1.CLEAR_DBM - 10 * math.log10(1000 * a["base"])
                require(abs(truth - prior.actual_hour_attenuation_db) < 1e-10, "Source target baseline changed")
                for variant in VARIANTS:
                    p = indexed.loc[(variant, h)]
                    require(p.split == prior.split and p.event == event and abs(p.truth_db - truth) < 1e-8,
                            "Current slow and frozen fast target mismatch")
                    offset = float(p.prediction_db - truth)
                    score = original_d1.score_marginal(hour, models[event], offset)
                    rows.append({"hour": h, "day": h[:10], "month": h[:7], "split": prior.split,
                        "event": event, "variant": variant, "truth_hour_db": truth,
                        "predicted_hour_db": float(p.prediction_db), "slow_error_db": offset,
                        "n_seconds": int(good.sum()), "n_records": int(weight.sum()),
                        "hour_mean_power_w": a["base"],
                        "equipment_epoch": prior.equipment_epoch,
                        "spectrum_configuration": prior.spectrum_configuration,
                        "slow_only_instant_mae_crps_db": float(np.average(np.abs(f - offset), weights=weight)),
                        **score})
            out_path = LOCAL / "chunks" / (key + ".parquet")
            parquet(out_path, pd.DataFrame(rows))
            atomic(marker_path, {"status": "complete", "fingerprint": m["fingerprint"],
                "hours": expected_hours, "records": len(rows), "completed_utc": now(),
                "output_sha256": {out_path.name: sha(out_path)}})
            print("LOCAL CHUNK", key, "hours", len(batch), "combined", len(rows), flush=True)
    print("LOCAL SCORING COMPLETE; independent verification next", flush=True)


def combined_frame():
    paths = sorted((LOCAL / "chunks").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def summarize():
    m = manifest()
    qa = read(LOCAL / "qa.json")
    require(qa["status"] == "PASS" and qa["fingerprint"] == m["fingerprint"], "Independent QA required")
    if (LOCAL / "report_manifest.json").exists():
        old = read(LOCAL / "report_manifest.json")
        for name, digest in old["output_sha256"].items():
            require(sha(LOCAL / name) == digest, "Published report changed")
        return old
    combined = combined_frame()
    predictions_frame = pd.read_parquet(LOCAL / "slow_predictions.parquet")
    slow_rows, summary, effects, calibration = [], [], [], []
    for split in SPLITS:
        for event in GROUPS:
            slow = predictions_frame.loc[predictions_frame.split.eq(split) &
                (predictions_frame.event.eq(event) if event != "overall" else True)]
            for variant, part in slow.groupby("variant", sort=True):
                error = part.error_db.to_numpy(float)
                slow_rows.append({"split": split, "event": event, "variant": variant,
                    "hours": len(part), "dates": part.day.nunique(), "rmse_db": float(np.sqrt(np.mean(error**2))),
                    "mae_db": float(np.mean(np.abs(error))), "bias_db": float(np.mean(error))})
            if split == "train":
                continue
            frame = combined.loc[combined.split.eq(split) &
                (combined.event.eq(event) if event != "overall" else True)]
            for variant, part in frame.groupby("variant", sort=True):
                for quantity in METRICS:
                    days = part.groupby("day")[quantity].mean()
                    for length in (1, 3, 7):
                        summary.append({"split": split, "event": event, "variant": variant,
                            "quantity": quantity, "date_mean": float(days.mean()), "hour_mean": float(part[quantity].mean()),
                            "hours": len(part), "dates": len(days), "valid_seconds": int(part.n_seconds.sum()),
                            "records": int(part.n_records.sum()), "block_days": length,
                            **original_d1.date_interval(days, length, f"C1|{split}|{event}|{variant}|{quantity}")})
                daily = [[np.stack(day[column].to_numpy()).mean(axis=0)
                          for column in ("actual_cdf201", "model_cdf201_left", "model_cdf201_right")]
                         for _, day in part.groupby("day", sort=True)]
                mean = np.asarray(daily).mean(axis=0)
                for j, p in enumerate(fm.P201):
                    calibration.append({"split": split, "event": event, "variant": variant,
                        "probability": float(p), "date_actual_cdf": float(mean[0,j]),
                        "model_cdf_left": float(mean[1,j]), "model_cdf_right": float(mean[2,j])})
            pivot = frame.pivot(index=["hour", "day"], columns="variant", values="crps201_db")
            comparisons = {"full_BiLSTM_vs_" + v: pivot[v] - pivot["full:bilstm"]
                           for v in VARIANTS if v != "full:bilstm"}
            for variant, part in frame.groupby("variant", sort=True):
                comparisons[variant + "_combination_vs_slow_only"] = pd.Series(
                    (part.slow_only_instant_mae_crps_db - part.crps201_db).to_numpy(),
                    index=pd.MultiIndex.from_arrays([part.hour, part.day], names=["hour", "day"]))
            for label, difference in comparisons.items():
                days = difference.groupby(level="day").mean()
                for length in (1, 3, 7):
                    effects.append({"split": split, "event": event, "contrast": label,
                        "positive_means_first_named_is_better": True, "date_difference_db": float(days.mean()),
                        "hour_difference_db": float(difference.mean()), "hours": len(difference),
                        "dates": len(days), "block_days": length,
                        **original_d1.date_interval(days, length, f"C1-effect|{split}|{event}|{label}")})
    frames = {"slow_hour_metrics.parquet": pd.DataFrame(slow_rows),
              "combined_summary.parquet": pd.DataFrame(summary),
              "paired_effects.parquet": pd.DataFrame(effects), "cdf_calibration.parquet": pd.DataFrame(calibration)}
    for name, frame in frames.items():
        parquet(LOCAL / name, frame)
    report = {"created_utc": now(), "fingerprint": m["fingerprint"], "status": "ANALYSIS_COMPLETE_AWAITING_REPORT_AUDIT",
        "qa_sha256": sha(LOCAL / "qa.json"), "new_model_fits": 0,
        "counts": {n: len(f) for n, f in frames.items()},
        "output_sha256": {n: sha(LOCAL / n) for n in frames},
        "main_model": "Rev028/full156/BiLSTM/seed17/53", "prior_base": "Rev028/base50/BiLSTM/seed17/46",
        "dynamic_reference": str((D1 / "report_qa.json").relative_to(ROOT)),
        "shared_fast_cdf_implies_identical_IID_block_marginal_scores": True,
        "within_hour_constant_cancellation_not_double_counted": True,
        "external_stages_complete": False}
    atomic(LOCAL / "report_manifest.json", report)
    return report


def status():
    result = {"new_output": str(LOCAL), "manifest_exists": (LOCAL / "manifest.json").exists(),
        "checkpoints": len(list((LOCAL / "chunks").glob("*.json"))),
        "qa_status": read(LOCAL / "qa.json").get("status") if (LOCAL / "qa.json").exists() else None,
        "report_qa_status": read(LOCAL / "report_qa.json").get("status") if (LOCAL / "report_qa.json").exists() else None,
        "all_three_stages_complete": False}
    active = LOCAL / "active_worker.json"
    result["active_worker"] = read(active) if active.exists() else None
    lock_path = LOCAL / "writer.lock"
    if lock_path.exists():
        import msvcrt
        with lock_path.open("r+b") as handle:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                result["os_lock_held"] = False
            except OSError:
                result["os_lock_held"] = True
    else:
        result["os_lock_held"] = None
    if (LOCAL / "manifest.json").exists():
        result["fingerprint"] = manifest()["fingerprint"]
    return result
