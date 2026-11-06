"""Read-only independent final audit/export. No training module imports, no fitting."""
from __future__ import annotations
import ast
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[env_name] = "2"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE"
REPAIR = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_033_Execution_Repair"
OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery"
GROUPS = ("overall", "clear_dry", "rain", "dry_overcast", "other")
NUMERIC = ("n", "rmse_db", "mae_db", "bias_db", "r2", "pearson_r", "pearson_r_squared")
HASH_CACHE = {}
COUNTS = collections.Counter()
MAX_DIFF = collections.defaultdict(float)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path):
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in HASH_CACHE:
        with path.open("rb") as handle:
            HASH_CACHE[key] = hashlib.file_digest(handle, "sha256").hexdigest()
    return HASH_CACHE[key]


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv_writer(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("x", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    return handle, writer


def write_csv(path, rows, fields=None):
    require(bool(rows) or fields is not None, "empty table needs schema")
    handle, writer = csv_writer(path, fields or list(rows[0]))
    with handle:
        writer.writerows(rows)


def same_number(a, b, where, tolerance=1e-9):
    if a is None or b is None:
        require(a is None and b is None, "null mismatch: " + where)
    else:
        require(math.isfinite(float(a)) and math.isfinite(float(b)), "nonfinite " + where)
        MAX_DIFF[where.split("/")[0]] = max(MAX_DIFF[where.split("/")[0]], abs(a-b))
        require(math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance),
                f"number mismatch {where}: {a} vs {b}")


def scalar(y, p):
    n = len(y)
    require(n == len(p) and all(math.isfinite(v) for v in y+p), "bad prediction")
    if not n:
        return dict(n=0, rmse_db=None, mae_db=None, bias_db=None, r2=None,
                    pearson_r=None, pearson_r_squared=None,
                    r2_reason="EMPTY", pearson_reason="EMPTY")
    errors = [a-b for a, b in zip(p, y)]
    sse = math.fsum(e*e for e in errors)
    ym, pm = math.fsum(y)/n, math.fsum(p)/n
    sy = math.fsum((v-ym)**2 for v in y)
    sp = math.fsum((v-pm)**2 for v in p)
    valid = n >= 2 and sy > 0
    corr = math.fsum((a-ym)*(b-pm) for a, b in zip(y, p))/math.sqrt(sy*sp) if valid and sp > 0 else None
    return dict(n=n, rmse_db=math.sqrt(sse/n), mae_db=math.fsum(abs(e) for e in errors)/n,
                bias_db=math.fsum(errors)/n, r2=1-sse/sy if valid else None,
                pearson_r=corr, pearson_r_squared=None if corr is None else corr*corr,
                r2_reason=None if valid else "N_LT_2_OR_CONSTANT_TARGET",
                pearson_reason=None if corr is not None else "N_LT_2_OR_CONSTANT_VECTOR")


def frame(split):
    data_dir = ROOT / "Outputs/Datasets/Signal_Detectability_Revision_003"
    data = pd.read_parquet(data_dir / f"{split}.parquet",
                           columns=["hour_bjt", "clear_sky_attenuation_db"])
    status = pd.read_parquet(data_dir / "hourly_sample_status.parquet")
    status = status.loc[status.split.eq(split),
                        ["hour_bjt", "use_for_regression", "clear_dry", "rain", "dry_overcast"]]
    data = data.merge(status, on="hour_bjt", validate="one_to_one", how="left")
    require(data.hour_bjt.is_unique and data.use_for_regression.eq(True).all(), "source join " + split)
    require(not data[list(GROUPS[1:4])].isna().any().any(), "missing weather")
    membership = data[list(GROUPS[1:4])].astype(int).sum(axis=1)
    require(membership.le(1).all(), "overlapping weather")
    data["other"] = membership.eq(0)
    data = data.sort_values("hour_bjt").reset_index(drop=True)
    require(len(data) == {"train": 2404, "validation": 343, "test": 687}[split], "source sample count")
    return data


def expected(df):
    return dict(keys=df.hour_bjt.astype(str).tolist(),
                y=df.clear_sky_attenuation_db.astype(float).tolist(),
                dates=df.hour_bjt.dt.strftime("%Y-%m-%d").tolist(),
                masks={w: df[w].astype(bool).tolist() for w in GROUPS[1:]})


def record_check(record, exp, full_metrics=True):
    require(record["hour_bjt"] == exp["keys"], "hour keys/order")
    require(record["truth_db"] == exp["y"], "truth values")
    require(record["weather_masks"] == exp["masks"], "weather masks")
    require(len(record["prediction_db"]) == len(exp["y"]), "prediction length")
    groups = GROUPS if full_metrics else ("overall",)
    for group in groups:
        inds = list(range(len(exp["y"]))) if group == "overall" else [i for i, v in enumerate(exp["masks"][group]) if v]
        values = scalar([exp["y"][i] for i in inds], [record["prediction_db"][i] for i in inds])
        old = record["metrics"][group]
        for name in NUMERIC:
            same_number(values[name], old[name], "metric/" + group + "/" + name)
        for name in ("r2_reason", "pearson_reason"):
            require(values[name] == old[name], "null reason")
        dates = sorted({exp["dates"][i] for i in inds})
        require(dates == old["dates"] and len(dates) == old["n_dates"], "date coverage")
        COUNTS["recomputed_metric_groups"] += 1


def normpath(value):
    return str(Path(value)).replace("\\", "/")


def export_audit():
    started = time.perf_counter()
    require(read(BASE/"status.json")["status"] == "NUMERICS_COMPLETE_REPORT_PENDING", "numeric stage")
    manifest = read(BASE/"manifest.json")
    require(digest(BASE/"manifest.json") == "2aea3eb40885c33441d179dde0ad32ae028cd3818d29da8fc00d2d9cebe0fcea", "manifest anchor")
    require(read(BASE/"active_worker.json")["status"] == "EXITED", "worker not exited")
    require(not (BASE/"STOP_REQUESTED").exists(), "T014 stop decision must be inspected")
    models = manifest["models"]
    require(len(models) == 10 and len(manifest["feature_names"]) == 156, "contract counts")
    numeric_qa = read(BASE/"final/numeric_qa.json")
    require(numeric_qa["status"] == "PASS" and numeric_qa["states"] == 1560, "numeric QA")
    inputs = []
    for name, sha in manifest["source_sha256"].items():
        actual = digest(ROOT/name)
        require(actual == sha, "source SHA " + name)
        inputs.append(dict(path=normpath(name), expected_sha256=sha, actual_sha256=actual,
                           bytes=(ROOT/name).stat().st_size, status="PASS"))
    repair_manifest = read(REPAIR/"repair_manifest.json")
    for name, sha in repair_manifest["sources"].items():
        require(digest(ROOT/name) == sha, "repair source SHA")
    print("All 1028 pinned sources and repair code verified", flush=True)
    data = {split: frame(split) for split in ("train", "validation", "test")}
    bad = read(ROOT/"Outputs/Experiments/Combined_Channel_Revision_027/excluded_hours.json")
    bad_keys = {pd.Timestamp(v["hour_bjt"]) for v in bad}
    common = data["test"].loc[~data["test"].hour_bjt.isin(bad_keys)]
    require(len(common) == 684 and len(bad_keys) == 3, "ITU exact common hours")
    exp = {split: expected(df) for split, df in data.items()}
    exp["test_itu_common684"] = expected(common)
    folds = []
    for f in manifest["folds"]:
        train_dates = data["train"].hour_bjt.dt.strftime("%Y-%m-%d")
        fit_days, hold_days = set(f["fit_days"]), set(f["hold_days"])
        require(fit_days.isdisjoint(hold_days), "CV day leakage")
        fit = data["train"].loc[train_dates.isin(fit_days)]
        hold = data["train"].loc[train_dates.isin(hold_days)]
        require(len(fit)+len(hold) == 2404, "CV coverage")
        folds.append((expected(fit), expected(hold)))
    require(sum(len(f[1]["keys"]) for f in folds) == 2404, "CV held count")
    require(len(set(k for f in folds for k in f[1]["keys"])) == 2404, "CV unique held hours")

    with (BASE/"final/all_test_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        original_metrics = list(csv.DictReader(handle))
    metric_map = {}
    for row in original_metrics:
        key = (row["model"], None if not row["k"] else int(float(row["k"])), row["population"], row["weather"])
        require(key not in metric_map, "duplicate metrics key")
        metric_map[key] = row
    with (BASE/"final/retained_features_all_steps.csv").open(encoding="utf-8-sig", newline="") as handle:
        original_features = list(csv.DictReader(handle))
    feature_map = collections.defaultdict(list)
    for row in original_features:
        feature_map[(row["model"], int(row["k"]))].append(row)
    inventory = read(BASE/"final/path_inventory.json")
    ledger = [json.loads(line) for line in (BASE/"fit_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    ledger_roles = collections.Counter((v["model"], v["k"], v["fold"], v["reused"],
                                       normpath(v["model_file"]), v["model_sha256"]) for v in ledger)
    summaries, roles, path_rows, qa_rows, permutations, metric_rows, val_metrics = [], [], [], [], [], [], []
    expected_ledger = collections.Counter()
    per_model_roles = collections.defaultdict(collections.Counter)
    source_file_rows = []
    for model in models:
        names = manifest["feature_names"].copy()
        removed = []
        path_detail = []
        hourly_fields = ["model", "k", "hour_bjt", "truth_db", "prediction_db", "error_db",
                         "weather", "itu_common_684", "evaluation_sha256"]
        handle, writer = csv_writer(OUT/f"hourly/{model}_test_687_all_steps.csv", hourly_fields)
        with handle:
            for k in range(156, 0, -1):
                folder = BASE/f"models/{model}/k_{k:04d}"
                state_sha, evaluation_sha, qa_sha = [digest(folder/f"{x}.json") for x in ("state", "evaluation", "qa")]
                s, e, q = [read(folder/f"{x}.json") for x in ("state", "evaluation", "qa")]
                require(q["status"] == "PASS" and q["state_sha256"] == state_sha and q["evaluation_sha256"] == evaluation_sha, "state QA chain")
                require(e["state_sha256"] == state_sha and e["training_calls"] == 0, "evaluation chain")
                require(numeric_qa["state_qa_sha256"][str((folder/"qa.json").relative_to(ROOT))] == qa_sha, "global numeric QA chain")
                require(s["manifest"] == manifest["fingerprint"] and s["seed"] == 17, "state contract")
                require(s["model"] == model and s["k"] == k and s["feature_names"] == names, "independent nested path")
                require(s["official_test_used_for_deletion"] is False, "test deletion leakage flag")
                cfg = dict(manifest["model_configs"][model])
                if model == "bai_lstm":
                    cfg["epochs"] = 32
                require(s["config"] == cfg, "frozen config " + model)
                require(q["all_four_models_replayed"] is True and q["training_calls_during_qa"] == 0, "prior replay")
                require(q["maximum_prediction_replay_delta_db"] < 2e-6, "prior replay tolerance")
                require(q["independent_permutation_metric_count"] == (3*k if k > 1 else 0), "prior permutation count")
                COUNTS["prior_independent_permutation_metrics"] += q["independent_permutation_metric_count"]
                COUNTS["prior_sampled_permutation_prediction_replays"] += q["permutation_prediction_replay_count"]
                fold_rmse = []
                for role, (fit_exp, hold_exp) in zip(s["folds"], folds):
                    record_check(role["training"], fit_exp, False)
                    record_check(role["holdout"], hold_exp, False)
                    fold_rmse.append(scalar(role["holdout"]["truth_db"], role["holdout"]["prediction_db"])["rmse_db"])
                record_check(s["full_fit"]["training"], exp["train"], False)
                same_number(math.fsum(fold_rmse)/3, s["cv_mean_rmse_db"], "cv/mean")
                same_number(float(np.std(fold_rmse, ddof=1)/math.sqrt(3)), s["cv_se_rmse_db"], "cv/se")
                inv = inventory[model][156-k]
                require(inv["k"] == k and inv["removed_next"] == s["removed_feature_next"], "path inventory")
                same_number(inv["cv_mean_rmse_db"], s["cv_mean_rmse_db"], "inventory/mean")
                same_number(inv["cv_se_rmse_db"], s["cv_se_rmse_db"], "inventory/se")
                for role in [*s["folds"], s["full_fit"]]:
                    rp = ROOT/role["model_file"]
                    require(digest(rp) == role["model_sha256"], "model SHA")
                    require(role["k"] == k and role["model"] == model and role["feature_names"] == names and role["seed"] == 17, "role identity")
                    require(role["new_fit"] is not role["reused"], "reuse/new distinction")
                    category = ("IN_RUN_BAI_SELECTED_WEIGHT_REUSE" if role["reused"] and "full156_epoch_selection" in role["model_file"]
                                else "HISTORICAL_WEIGHT_REUSE" if role["reused"] else "NEW_ROLE_FIT")
                    per_model_roles[model][category] += 1
                    key = (model, k, role["fold"], role["reused"], normpath(role["model_file"]), role["model_sha256"])
                    expected_ledger[key] += 1
                    roles.append(dict(model=model, k=k, fold=role["fold"], category=category,
                                      model_file=normpath(role["model_file"]), model_sha256=role["model_sha256"],
                                      bytes=rp.stat().st_size, format=role["format"],
                                      seed=role["seed"], train_hour_sha256=role["train_hour_sha256"],
                                      state_sha256=state_sha, identity=role["identity"],
                                      fit_metadata_json=json.dumps(role["fit_metadata"], ensure_ascii=False, sort_keys=True),
                                      config_json=json.dumps(s["config"], ensure_ascii=False, sort_keys=True)))
                require([r["fold"] for r in s["folds"]] == [1,2,3] and s["full_fit"]["fold"] == 0, "four role slots")
                if k > 1:
                    require(len(s["permutations"]) == 3 and len(s["permutation_importance"]) == k, "permutation structure")
                    for perm in s["permutations"]:
                        require(digest(ROOT/perm["path"]) == perm["sha256"], "permutation SHA")
                        permutations.append(dict(model=model, k=k, path=normpath(perm["path"]),
                                                 sha256=perm["sha256"], shape=json.dumps(perm["shape"]),
                                                 namespace=perm["namespace"]))
                    for j, item in enumerate(s["permutation_importance"]):
                        require(item["feature_name"] == names[j], "importance order")
                        same_number(math.fsum(item["fold_delta_rmse_db"])/3, item["mean_delta_rmse_db"], "importance/mean", 1e-8)
                    chosen = min(s["permutation_importance"], key=lambda x:(x["mean_delta_rmse_db"],x["feature_name"]))
                    require(chosen["feature_name"] == s["removed_feature_next"], "minimum-delta deletion")
                else:
                    require(s["removed_feature_next"] is None and s["permutation_importance"] == [], "terminal")
                    chosen = None
                feature_rows = feature_map[(model,k)]
                require([r["feature"] for r in feature_rows] == names, "feature CSV names")
                require([int(r["order"]) for r in feature_rows] == list(range(1,k+1)), "feature CSV order")
                for r in feature_rows:
                    require(r["removed_next"].lower() == str(r["feature"] == s["removed_feature_next"]).lower(), "feature CSV deletion flag")
                for split in ("validation","test","test_itu_common684"):
                    record_check(e["results"][split], exp[split])
                common_keys = set(exp["test_itu_common684"]["keys"])
                rec, cre = e["results"]["test"], e["results"]["test_itu_common684"]
                inds = [i for i, key in enumerate(rec["hour_bjt"]) if key in common_keys]
                require([rec["prediction_db"][i] for i in inds] == cre["prediction_db"], "common prediction identity")
                for pop, split in (("ML_687","test"),("COMMON_684","test_itu_common684")):
                    for group in GROUPS:
                        val = e["results"][split]["metrics"][group]
                        old = metric_map[(model,k,pop,group)]
                        for name in NUMERIC:
                            old_value = None if old[name] == "" else float(old[name])
                            same_number(val[name], old_value, "CSV/" + name)
                        require(ast.literal_eval(old["dates"]) == val["dates"] and int(old["n_dates"]) == val["n_dates"], "CSV dates")
                        for reason in ("r2_reason","pearson_reason"):
                            require((old[reason] or None) == val[reason], "CSV reason")
                        metric_rows.append(dict(model=model,k=k,population=pop,weather=group,**val))
                for group in GROUPS:
                    val_metrics.append(dict(model=model,k=k,weather=group,**e["results"]["validation"]["metrics"][group]))
                for i, key in enumerate(rec["hour_bjt"]):
                    weather = next(w for w in GROUPS[1:] if rec["weather_masks"][w][i])
                    writer.writerow(dict(model=model,k=k,hour_bjt=key,truth_db=rec["truth_db"][i],
                                         prediction_db=rec["prediction_db"][i],
                                         error_db=rec["prediction_db"][i]-rec["truth_db"][i],weather=weather,
                                         itu_common_684=key in common_keys,evaluation_sha256=evaluation_sha))
                    COUNTS["hourly_test_rows"] += 1
                row = dict(model=model,k=k,cv_mean_rmse_db=s["cv_mean_rmse_db"],cv_se_rmse_db=s["cv_se_rmse_db"],
                           removed_next=s["removed_feature_next"],retained_names_json=json.dumps(names, ensure_ascii=False),
                           removed_so_far_json=json.dumps(removed, ensure_ascii=False),
                           state_path=relative(folder/"state.json"),state_sha256=state_sha,
                           evaluation_sha256=evaluation_sha,qa_sha256=qa_sha)
                for group in GROUPS:
                    for field in NUMERIC:
                        row["test_"+group+"_"+field] = rec["metrics"][group][field]
                summaries.append(row)
                path_rows.append(dict(model=model,k=k,removed_next=s["removed_feature_next"],
                                      retained_names_json=json.dumps(names),removed_so_far_json=json.dumps(removed),
                                      mean_delta_rmse_db=None if chosen is None else chosen["mean_delta_rmse_db"],
                                      fold_delta_rmse_db_json=None if chosen is None else json.dumps(chosen["fold_delta_rmse_db"])))
                path_detail.append(dict(k=k,retained=names.copy(),removed_so_far=removed.copy(),
                                        removed_next=s["removed_feature_next"],deletion_importance=chosen))
                qa_rows.append(dict(model=model,k=k,status="PASS",state_sha256=state_sha,evaluation_sha256=evaluation_sha,
                                    qa_sha256=qa_sha,maximum_prior_prediction_replay_delta_db=q["maximum_prediction_replay_delta_db"],
                                    prior_independent_permutation_metric_count=q["independent_permutation_metric_count"],
                                    prior_permutation_prediction_replay_count=q["permutation_prediction_replay_count"]))
                if chosen:
                    removed.append(chosen["feature_name"])
                    names = [v for v in names if v != chosen["feature_name"]]
                COUNTS["states"] += 1
                if k % 25 == 0:
                    print(f"Audited {model} through k={k}; total={COUNTS['states']}", flush=True)
            write_json(OUT/f"paths/{model}_156_to_1.json", path_detail)
        print(f"PASS {model}: 156 states", flush=True)

    bai = read(BASE/"bai_epoch_selection.json")
    selection_row = ledger[0]
    require(selection_row["model"] == "bai_lstm" and "full156_epoch_selection" in selection_row["model_file"] and selection_row["reused"] is False, "Bai initial selection ledger")
    require(selection_row["model_sha256"] == bai["model_sha256"] and bai["test_used"] is False and bai["selected_epoch"] == 32, "Bai selection")
    selection_key = (selection_row["model"],selection_row["k"],selection_row["fold"],selection_row["reused"],
                     normpath(selection_row["model_file"]),selection_row["model_sha256"])
    expected_ledger[selection_key] += 1
    require(expected_ledger == ledger_roles, "exact ledger role/selection reconciliation")
    roles_count = collections.Counter(r["category"] for r in roles)
    require(roles_count == {"HISTORICAL_WEIGHT_REUSE":635,"IN_RUN_BAI_SELECTED_WEIGHT_REUSE":1,"NEW_ROLE_FIT":5604}, "reuse accounting")
    require(len(ledger) == 6241 and len(roles) == 6240, "role ledger slots")
    require(COUNTS["states"] == 1560 and COUNTS["prior_independent_permutation_metrics"] == 367350, "final counts")
    require(len(original_features) == 122460 and len(original_metrics) == 15605, "aggregate lengths")
    require(COUNTS["hourly_test_rows"] == 1071720, "hourly rows")
    itu, iq = read(BASE/"itu_baseline.json"), read(BASE/"itu_qa.json")
    require(iq["status"] == "PASS" and iq["result_sha256"] == digest(BASE/"itu_baseline.json"), "ITU QA digest")
    record_check(itu["results"], exp["test_itu_common684"])
    require(itu["all_hours"] == 687 and itu["scored_hours"] == 684 and itu["fitted_label_offset"] is False, "ITU contract")
    for group in GROUPS:
        val = itu["results"]["metrics"][group]
        old = metric_map[("combined_ITU",None,"COMMON_684",group)]
        for name in NUMERIC:
            same_number(val[name], None if old[name] == "" else float(old[name]), "ITU CSV/" + name)
        metric_rows.append(dict(model="combined_ITU",k=None,population="COMMON_684",weather=group,**val))
    write_csv(OUT/"tables/all_test_metrics_15605_rows.csv", metric_rows)
    write_csv(OUT/"tables/all_validation_metrics_7800_rows.csv", val_metrics)
    write_csv(OUT/"tables/all_states_1560.csv", summaries)
    write_csv(OUT/"tables/retained_features_122460_rows.csv", original_features)
    write_csv(OUT/"tables/independent_rfe_paths_1560.csv", path_rows)
    write_csv(OUT/"tables/model_roles_6240_with_sha.csv", roles)
    write_csv(OUT/"tables/permutation_files_with_sha.csv", permutations)
    write_csv(OUT/"evidence/state_qa_index_1560.csv", qa_rows)
    write_csv(OUT/"evidence/frozen_inputs_1028_sha.csv", inputs)
    write_csv(OUT/"evidence/fit_accounting_by_model.csv", [
        dict(model=m,historical_weight_reuse=per_model_roles[m]["HISTORICAL_WEIGHT_REUSE"],
             in_run_bai_selected_weight_reuse=per_model_roles[m]["IN_RUN_BAI_SELECTED_WEIGHT_REUSE"],
             new_role_fits=per_model_roles[m]["NEW_ROLE_FIT"],role_slots=sum(per_model_roles[m].values()))
        for m in models])
    write_json(OUT/"evidence/frozen_model_configs.json", manifest["model_configs"])
    write_json(OUT/"evidence/itu_negative_liquid_hours.json", itu["missing"])
    write_json(OUT/"evidence/bai_epoch_selection.json", bai)
    itu_rec = itu["results"]
    write_csv(OUT/"hourly/combined_ITU_common_684.csv",[
        dict(hour_bjt=key,truth_db=itu_rec["truth_db"][i],prediction_db=itu_rec["prediction_db"][i],
             weather=next(w for w in GROUPS[1:] if itu_rec["weather_masks"][w][i]))
        for i,key in enumerate(itu_rec["hour_bjt"])])

    provenance = []
    for path in [BASE/"manifest.json",BASE/"fit_ledger.jsonl",BASE/"progress_events.jsonl",BASE/"status.json",
                 BASE/"active_worker.json",BASE/"itu_baseline.json",BASE/"itu_qa.json",
                 BASE/"bai_epoch_selection.json",BASE/"bai_epoch_selection_qa.json",
                 *sorted((BASE/"final").glob("*")),REPAIR/"repair_manifest.json",REPAIR/"QA.json",
                 *sorted((BASE/"failures").glob("*")),
                 *sorted((BASE/"logs").glob("*")),*sorted((REPAIR/"logs").glob("*")),
                 *sorted((REPAIR/"deferred_progress").glob("*"))]:
        if path.is_file():
            provenance.append(dict(path=relative(path),sha256=digest(path),bytes=path.stat().st_size))
    write_csv(OUT/"evidence/frozen_provenance_sha.csv", provenance)
    weather_rows = []
    for split, ex in exp.items():
        for group in GROUPS:
            ii = list(range(len(ex["y"]))) if group == "overall" else [i for i,v in enumerate(ex["masks"][group]) if v]
            dates = sorted({ex["dates"][i] for i in ii})
            weather_rows.append(dict(population=split,weather=group,n=len(ii),n_dates=len(dates),dates=json.dumps(dates)))
    write_csv(OUT/"tables/weather_sample_coverage.csv", weather_rows)
    all_paths = [tuple(p["removed_next"] for p in path_rows if p["model"]==m) for m in models]
    write_json(OUT/"evidence/INDEPENDENT_AGGREGATE_QA.json",dict(
        status="PASS",utc=stamp(),implementation="standalone read-only export_audit.py; no training imports or fits",
        training_calls=0,source_input_files=len(inputs),states=1560,role_slots=6240,ledger_rows=6241,
        new_role_fits=5604,new_bai_epoch_selection_fit=1,new_fit_calls_total=5605,
        historical_weight_reuse=635,in_run_bai_selected_weight_reuse=1,
        unique_model_files=len({r["model_file"] for r in roles}),
        unique_rfe_paths=len(set(all_paths)),independent_deletion_transitions=1550,
        counts=dict(COUNTS),maximum_numeric_differences=dict(MAX_DIFF),
        prior_QA_scope="All 4 roles/state replayed; all permutation scores independently recomputed; only first/middle/last permutation predictions per fold replayed. Final audit checks digests and recorded deletion arithmetic, not a second weight replay.",
        all_model_and_permutation_file_digests_match=True,
        official_test_n=687,common_itu_n=684,itu_missing_hours=3,
        elapsed_seconds=time.perf_counter()-started,
        final_delivery_complete=False,remaining="plots, Chinese report, independent delivery readback and final QA"))
    print("AGGREGATE_AUDIT_PASS", json.dumps(dict(COUNTS)), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        export_audit()
    except Exception:
        write_json(OUT/f"evidence/audit_failure_{time.time_ns()}.json",
                   dict(utc=stamp(),error=traceback.format_exc(),counts=dict(COUNTS),
                        note="Preserved report-stage failure, does not alter frozen experimental results."))
        raise


if __name__ == "__main__":
    main()
