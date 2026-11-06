"""Date-isolated comparison of unit-mean positive-power marginal laws.

Only the official training hours enter any fit or candidate score. All valid
primary seconds are used with record weights inside an hour and equal total
weight per hour. CRPS is evaluated by quantile integration with an independent
401-point numerical-resolution check.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import digamma, gammaln, i0e, ndtri
from scipy.stats import gamma, ncx2

from window_diagnostics import OUT, SOURCE, now, read_train, sha256, worker_lock, write_json

HERE = Path(__file__).resolve().parent
TRIALS = OUT / "marginal_trials"
FAMILIES = ("lognormal", "gamma", "rice", "empirical")
CONDITIONS = ("global", "hourly_weather")
CLASSES = ("clear_dry", "dry_overcast", "rain", "other")
FOLDS = 5
SEED = 20261006
P201 = (np.arange(201) + .5) / 201
P401 = (np.arange(401) + .5) / 401


@dataclass
class Hour:
    hour: str
    day: str
    event: str
    fold: int
    q: np.ndarray
    weight: np.ndarray
    f_sorted: np.ndarray
    cum_w: np.ndarray
    cum_wf: np.ndarray
    n_records: int

    def actual_cdf(self, values: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.f_sorted, values, side="right")
        return self.cum_w[position]

    def pinball(self, values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.f_sorted, values, side="right")
        cdf, first = self.cum_w[position], self.cum_wf[position]
        loss = probabilities * (self.cum_wf[-1] - values) + values * cdf - first
        return np.maximum(loss, 0.)


def prepare_folds() -> pd.DataFrame:
    qa = json.loads((OUT / "window_report_qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS":
        raise RuntimeError("Window report QA is not PASS")
    summary = pd.read_csv(SOURCE / "clean_1s_hourly_summary.csv")
    train = summary.loc[summary.split.eq("train")].copy()
    train["hour_bjt"] = pd.to_datetime(train.hour_bjt)
    if len(train) != 2404 or not train.hour_bjt.is_unique:
        raise RuntimeError("Official training-hour scope changed")
    train["date_bjt"] = train.hour_bjt.dt.strftime("%Y-%m-%d")
    train["month"] = train.hour_bjt.dt.strftime("%Y-%m")
    train["equipment_epoch"] = np.where(train.date_bjt < "2026-05-14", "before",
                                         np.where(train.date_bjt.eq("2026-05-14"), "transition", "after"))
    days = train.groupby("date_bjt", sort=True).agg(month=("month", "first"), rain_day=("rain", "max"),
                                                   equipment_epoch=("equipment_epoch", "first"), n_train_hours=("hour_bjt", "size")).reset_index()
    days["fold"] = -1
    rng = np.random.default_rng(SEED)
    cursor = 0
    for _, part in days.groupby(["month", "rain_day", "equipment_epoch"], sort=True):
        indices = rng.permutation(part.index.to_numpy())
        for index in indices:
            days.loc[index, "fold"] = cursor % FOLDS
            cursor += 1
    train = train.merge(days[["date_bjt", "fold"]], on="date_bjt", how="left", validate="many_to_one")
    train["event"] = np.select([train.clear_dry, train.dry_overcast, train.rain],
                                ["clear_dry", "dry_overcast", "rain"], default="other")
    if train.groupby("date_bjt").fold.nunique().max() != 1 or set(train.fold) != set(range(FOLDS)):
        raise RuntimeError("Invalid complete-day folds")
    path = OUT / "marginal_fold_days.csv"
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(days):
            raise RuntimeError("Existing date folds differ from the frozen rule")
    else:
        days.to_csv(path, index=False)
        train[["hour_bjt", "date_bjt", "event", "fold"]].to_csv(OUT / "marginal_fold_hours.csv", index=False)
    return train


def frozen_manifest() -> tuple[dict, pd.DataFrame]:
    assignment = prepare_folds()
    source_qa = json.loads((SOURCE / "clean_1s_qa.json").read_text(encoding="utf-8"))
    design = {
        "scope": "2404 official training hours; complete-day 5-fold CV",
        "families": list(FAMILIES), "conditions": list(CONDITIONS), "classes": list(CLASSES),
        "folds": FOLDS, "seed": SEED, "primary_seconds_min_records": 9,
        "quantile_grid_main": 201, "quantile_grid_sensitivity": 401,
        "weights": "record-weighted seconds, equal total weight per hour",
        "source_partitions": source_qa["monthly_parquet_sha256"],
        "fold_days_sha256": sha256(OUT / "marginal_fold_days.csv"),
        "fold_hours_sha256": sha256(OUT / "marginal_fold_hours.csv"),
        "model_plan_sha256": sha256(HERE / "MODEL_PLAN.md"),
        "script_sha256": sha256(Path(__file__)),
        "window_report_qa_sha256": sha256(OUT / "window_report_qa.json"),
    }
    path = OUT / "marginal_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["frozen_design"] != design:
            raise RuntimeError("Frozen marginal design or fingerprint changed")
    else:
        manifest = {"created_utc": now(), "frozen_design": design}
        write_json(path, manifest)
    return manifest, assignment


def load_observations(assignment: pd.DataFrame, manifest: dict) -> list[Hour]:
    mapping = {str(row.hour_bjt): row for row in assignment.itertuples(index=False)}
    observations = []
    for name, digest in manifest["frozen_design"]["source_partitions"].items():
        path = SOURCE / "clean_1s" / name
        if sha256(path) != digest:
            raise RuntimeError("Marginal source changed")
        frame = read_train(path)
        frame = frame.loc[frame.quality_zero_records >= 9]
        for hour, group in frame.groupby("hour_bjt", sort=True):
            key = str(hour)
            info = mapping[key]
            q = (group.received_power_w / group.hour_mean_power_w).to_numpy(dtype=float)
            records = group.quality_zero_records.to_numpy(dtype=float)
            if not np.isfinite(q).all() or (q <= 0).any():
                raise RuntimeError("Invalid positive power ratio")
            weight = records / records.sum()
            f = -10 * np.log10(q)
            order = np.argsort(f, kind="stable")
            fs, ws = f[order], weight[order]
            cw, cwf = np.r_[0., np.cumsum(ws)], np.r_[0., np.cumsum(ws * fs)]
            cw[-1] = 1.
            observations.append(Hour(key, info.date_bjt, info.event, int(info.fold), q, weight, fs, cw, cwf, int(records.sum())))
        print(f"Loaded {name}: {frame.hour_bjt.nunique()} training hours", flush=True)
    if len(observations) != 2404 or len({x.hour for x in observations}) != 2404:
        raise RuntimeError("Missing or duplicate training hour")
    return sorted(observations, key=lambda x: x.hour)


def rice_logpdf(q: np.ndarray, b: float) -> np.ndarray:
    size = b * b + 2.
    z = b * np.sqrt(size * q)
    return np.log(size / 2.) - .5 * (size * q + b * b) + np.log(i0e(z)) + z


def fit(family: str, hours: list[Hour]) -> dict:
    q = np.concatenate([x.q for x in hours])
    weight = np.concatenate([x.weight / len(hours) for x in hours])
    logq = np.log(q)
    mean_q = float(np.dot(weight, q))
    var_q = float(np.dot(weight, (q - mean_q) ** 2))
    model = {"family": family, "n_hours": len(hours), "n_days": len({x.day for x in hours}),
             "n_primary_seconds": len(q), "training_mean_q": mean_q, "training_variance_q": var_q}
    if family == "lognormal":
        moment = float(np.dot(weight, logq ** 2))
        variance_log = 2 * moment / (math.sqrt(1 + moment) + 1)
        if variance_log <= 0:
            raise RuntimeError("Degenerate lognormal scale")
        model.update(sigma=math.sqrt(variance_log), mu=-variance_log / 2., expected_q=1.)
    elif family == "gamma":
        target = float(np.dot(weight, q - logq - 1))
        if target <= 0:
            raise RuntimeError("Degenerate Gamma shape")
        log_shape, solver = brentq(lambda value: value - digamma(math.exp(value)) - target,
                                    -20, 30, full_output=True)
        shape = math.exp(log_shape)
        model.update(shape=shape, scale=1 / shape, expected_q=1.,
                     root_iterations=int(solver.iterations), root_converged=bool(solver.converged),
                     mle_target=target, log_shape_bounds=[-20, 30])
    elif family == "rice":
        if var_q <= 0:
            raise RuntimeError("Degenerate Rician scale")
        initial = 2 / math.sqrt(var_q)
        lower, upper = max(math.exp(-8), initial / 100), min(10000., initial * 100)
        if lower >= upper:
            raise RuntimeError("Rician numerical range collapsed")
        result = minimize_scalar(lambda value: -float(np.dot(weight, rice_logpdf(q, math.exp(value)))),
                                 bounds=(math.log(lower), math.log(upper)), method="bounded",
                                 options={"maxiter": 100, "xatol": 1e-5})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Rician optimizer failed")
        b = math.exp(float(result.x))
        boundary_loss = float(np.dot(weight, q))  # b=0 gives mean-one exponential Q.
        if boundary_loss < result.fun:
            b = 0.
        model.update(b=b, envelope_scale=1 / math.sqrt(b * b + 2), expected_q=1.,
                     optimizer_iterations=int(result.nit), optimizer_evaluations=int(result.nfev),
                     optimizer_success=bool(result.success), search_b_bounds=[lower, upper], initial_b=initial,
                     boundary_rayleigh_nll=boundary_loss,
                     at_search_edge=bool(b != 0 and min(abs(result.x - math.log(lower)), abs(result.x - math.log(upper))) < 1e-4))
    elif family == "empirical":
        order = np.argsort(q, kind="stable")
        model["support_q"] = q[order] / mean_q
        model["support_cdf"] = np.cumsum(weight[order])
        model["support_cdf"][-1] = 1.
        model.update(power_normalizer=mean_q, expected_q=float(np.dot(weight, q / mean_q)))
    else:
        raise ValueError(family)
    if not math.isclose(model["expected_q"], 1., rel_tol=0, abs_tol=1e-10):
        raise RuntimeError("Unit power moment failed")
    return model


def quantile_f(model: dict, probabilities: np.ndarray) -> np.ndarray:
    p = 1 - probabilities
    family = model["family"]
    if family == "lognormal":
        log_q = model["mu"] + model["sigma"] * ndtri(p)
        result = -10 / np.log(10.) * log_q
    elif family == "gamma":
        result = -10 * np.log10(gamma.ppf(p, model["shape"], scale=model["scale"]))
    elif family == "rice":
        b = model["b"]
        result = -10 * np.log10(ncx2.ppf(p, 2, b * b) / (b * b + 2))
    else:
        # The generalized inverse of the weighted empirical CDF preserves
        # its discrete unit power moment. Linear interpolation would silently
        # turn it into a different, not exactly normalized distribution.
        position = np.searchsorted(model["support_cdf"], p, side="left")
        q = model["support_q"][position]
        result = -10 * np.log10(q)
    if not np.isfinite(result).all() or (np.diff(result) < -1e-10).any():
        raise RuntimeError("Invalid distribution quantiles")
    return result


def nll_db(model: dict, hour: Hour) -> float | None:
    q, family = hour.q, model["family"]
    logq = np.log(q)
    if family == "empirical":
        return None
    if family == "lognormal":
        sigma, mu = model["sigma"], model["mu"]
        logpdf = -logq - math.log(sigma) - .5 * math.log(2 * math.pi) - (logq - mu) ** 2 / (2 * sigma ** 2)
    elif family == "gamma":
        a = model["shape"]
        logpdf = a * math.log(a) - gammaln(a) + (a - 1) * logq - a * q
    else:
        logpdf = rice_logpdf(q, model["b"])
    logpdf_f = logpdf + math.log(math.log(10) / 10) + logq
    value = -float(np.dot(hour.weight, logpdf_f))
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite logarithmic score")
    return value


def serializable_model(model: dict) -> dict:
    result = {key: value for key, value in model.items() if not isinstance(value, np.ndarray)}
    result["f_quantile_probabilities_201"] = P201.tolist()
    result["f_quantiles_201_db"] = quantile_f(model, P201).tolist()
    result["f_quantiles_401_db"] = quantile_f(model, P401).tolist()
    return result


def run() -> None:
    manifest, assignment = frozen_manifest()
    TRIALS.mkdir(exist_ok=True)
    fingerprint = sha256(OUT / "marginal_manifest.json")
    with worker_lock("fit_marginals.py run"):
        observations = load_observations(assignment, manifest)
        for family in FAMILIES:
            for condition in CONDITIONS:
                candidate = f"{family}__{condition}"
                for fold in range(FOLDS):
                    stem = f"{candidate}__fold{fold}"
                    jp, pp = TRIALS / f"{stem}.json", TRIALS / f"{stem}.parquet"
                    if jp.exists():
                        old = json.loads(jp.read_text(encoding="utf-8"))
                        if old["manifest_sha256"] != fingerprint or old["status"] != "complete" or sha256(pp) != old["scores_sha256"]:
                            raise RuntimeError("Candidate checkpoint changed")
                        print(f"Skip {stem}", flush=True)
                        continue
                    training = [x for x in observations if x.fold != fold]
                    held = [x for x in observations if x.fold == fold]
                    if {x.day for x in training} & {x.day for x in held}:
                        raise RuntimeError("Date leak across a fold")
                    labels = ("global",) if condition == "global" else CLASSES
                    models = {}
                    for label in labels:
                        selected = training if label == "global" else [x for x in training if x.event == label]
                        if not selected:
                            raise RuntimeError("Absent training weather class")
                        models[label] = fit(family, selected)
                        print(f"Fit {stem}/{label}: {len(selected)} train hours", flush=True)
                    predicted = {label: (quantile_f(model, P201), quantile_f(model, P401)) for label, model in models.items()}
                    rows = []
                    for hour in held:
                        label = "global" if condition == "global" else hour.event
                        q201, q401 = predicted[label]
                        rows.append({
                            "candidate": candidate, "fold": fold, "hour_bjt": hour.hour,
                            "date_bjt": hour.day, "event": hour.event,
                            "n_primary_seconds": len(hour.q), "n_quality_zero_records": hour.n_records,
                            "measured_primary_mean_q": float(np.dot(hour.weight, hour.q)),
                            "crps201_db": 2 * float(np.mean(hour.pinball(q201, P201))),
                            "crps401_db": 2 * float(np.mean(hour.pinball(q401, P401))),
                            "nll_db": nll_db(models[label], hour),
                            "actual_cdf_at_predicted_q201": hour.actual_cdf(q201).tolist(),
                        })
                    result = pd.DataFrame(rows)
                    temp = pp.with_suffix(".tmp.parquet")
                    result.to_parquet(temp, index=False, compression="zstd")
                    temp.replace(pp)
                    write_json(jp, {
                        "status": "complete", "completed_utc": now(), "candidate": candidate, "fold": fold,
                        "manifest_sha256": fingerprint, "scores_sha256": sha256(pp),
                        "n_train_hours": len(training), "n_train_days": len({x.day for x in training}),
                        "n_score_hours": len(held), "n_score_days": len({x.day for x in held}),
                        "train_hour_ids": [x.hour for x in training], "score_hour_ids": [x.hour for x in held],
                        "parameters": {label: serializable_model(model) for label, model in models.items()},
                    })
                    print(f"Saved {stem}: {len(held)} date-isolated hour scores", flush=True)
    verify(observations)


def verify(observations: list[Hour] | None = None) -> None:
    manifest, assignment = frozen_manifest()
    fingerprint = sha256(OUT / "marginal_manifest.json")
    expected_names = {f"{family}__{condition}__fold{fold}" for family in FAMILIES for condition in CONDITIONS for fold in range(FOLDS)}
    actual = {p.stem for p in TRIALS.glob("*__fold*.json")}
    if actual != expected_names:
        raise RuntimeError(f"Incomplete candidate-fold set: {len(actual)}/40")
    if observations is None:
        observations = load_observations(assignment, manifest)
    source_hour = {x.hour: x for x in observations}
    records = []
    independent_errors = []
    for name in sorted(expected_names):
        jp, pp = TRIALS / f"{name}.json", TRIALS / f"{name}.parquet"
        meta = json.loads(jp.read_text(encoding="utf-8"))
        if meta["manifest_sha256"] != fingerprint or meta["status"] != "complete" or sha256(pp) != meta["scores_sha256"]:
            raise RuntimeError("Marginal result fingerprint mismatch")
        data = pd.read_parquet(pp)
        if len(data) != meta["n_score_hours"] or data.hour_bjt.duplicated().any():
            raise RuntimeError("Marginal hourly score count mismatch")
        train_dates = {value[:10] for value in meta["train_hour_ids"]}
        if train_dates & set(data.date_bjt):
            raise RuntimeError("Day leakage detected")
        numeric = data[["crps201_db", "crps401_db", "measured_primary_mean_q"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise RuntimeError("Nonfinite marginal scoring")
        for values in data.actual_cdf_at_predicted_q201:
            if len(values) != 201 or (np.asarray(values) < 0).any() or (np.asarray(values) > 1).any():
                raise RuntimeError("CDF calibration vector invalid")
        for parameters in meta["parameters"].values():
            if not math.isclose(parameters["expected_q"], 1., abs_tol=1e-10):
                raise RuntimeError("Model power moment invalid")
            family = parameters["family"]
            actual_power_moment = (math.exp(parameters["mu"] + parameters["sigma"] ** 2 / 2) if family == "lognormal"
                                   else parameters["shape"] * parameters["scale"] if family == "gamma"
                                   else parameters["envelope_scale"] ** 2 * (parameters["b"] ** 2 + 2) if family == "rice"
                                   else parameters["training_mean_q"] / parameters["power_normalizer"])
            if not math.isclose(actual_power_moment, 1., rel_tol=0, abs_tol=1e-10):
                raise RuntimeError("Saved parameter power moment does not recompute")
            quantiles = np.array(parameters["f_quantiles_201_db"])
            if len(quantiles) != 201 or not np.isfinite(quantiles).all() or (np.diff(quantiles) < -1e-10).any():
                raise RuntimeError("Saved quantiles invalid")
        # A separate literal pointwise pinball implementation checks one actual
        # source hour in every candidate-fold, not the cumulative-sum shortcut.
        row = data.sort_values("hour_bjt").iloc[0]
        source = source_hour[row.hour_bjt]
        label = "global" if row.candidate.endswith("__global") else source.event
        parameters = meta["parameters"][label]
        ordered_weight = np.diff(source.cum_w)
        for count, probabilities in ((201, P201), (401, P401)):
            q = np.array(parameters[f"f_quantiles_{count}_db"])
            difference = source.f_sorted[:, None] - q[None, :]
            loss = np.where(difference >= 0, probabilities * difference, (probabilities - 1) * difference)
            recomputed = 2 * float(np.mean(np.sum(ordered_weight[:, None] * loss, axis=0)))
            independent_errors.append(abs(recomputed - row[f"crps{count}_db"]))
        cdf = np.sum(ordered_weight[:, None] * (source.f_sorted[:, None] <= np.array(parameters["f_quantiles_201_db"])[None, :]), axis=0)
        independent_errors.append(float(np.max(np.abs(cdf - np.array(row.actual_cdf_at_predicted_q201)))))
        records.append(data)
    all_scores = pd.concat(records, ignore_index=True)
    assigned = {str(row.hour_bjt): int(row.fold) for row in assignment.itertuples(index=False)}
    for candidate, group in all_scores.groupby("candidate"):
        if len(group) != 2404 or set(group.hour_bjt) != set(assigned) or group.hour_bjt.duplicated().any():
            raise RuntimeError("Candidate does not have one out-of-fold score per train hour")
        if any(row.fold != assigned[row.hour_bjt] for row in group.itertuples(index=False)):
            raise RuntimeError("Stored fold mismatch")
    if not np.isfinite(independent_errors).all() or max(independent_errors) > 1e-10:
        raise RuntimeError("Independent marginal score recomputation failed")
    write_json(OUT / "marginal_cv_qa.json", {
        "status": "PASS", "completed_utc": now(), "candidate_fold_records": 40,
        "candidate_hour_scores": len(all_scores), "hours_per_candidate": 2404,
        "independent_score_dates": int(assignment.date_bjt.nunique()),
        "max_abs_crps_grid_difference_db": float(np.max(np.abs(all_scores.crps201_db - all_scores.crps401_db))),
        "source_hours_independently_recomputed": 40,
        "max_independent_recompute_error": float(max(independent_errors)),
        "marginal_manifest_sha256": fingerprint,
        "note": "Train-only blocked CV; validation and test not scored.",
    })
    print("Marginal CV QA PASS: 40 fold records, 19,232 hourly scores", flush=True)


def report() -> None:
    qa = json.loads((OUT / "marginal_cv_qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS" or qa["marginal_manifest_sha256"] != sha256(OUT / "marginal_manifest.json"):
        raise RuntimeError("Marginal CV QA not valid")
    scores = pd.concat([pd.read_parquet(p) for p in sorted(TRIALS.glob("*__fold*.parquet"))], ignore_index=True)
    day = scores.groupby(["candidate", "date_bjt"])[["crps201_db", "crps401_db"]].mean().reset_index()
    rows = []
    for candidate, group in scores.groupby("candidate", sort=True):
        for event in ("all_train",) + CLASSES:
            part = group if event == "all_train" else group.loc[group.event.eq(event)]
            date_scores = part.groupby("date_bjt")[["crps201_db", "crps401_db"]].mean()
            rows.append({"candidate": candidate, "event": event, "n_hours": len(part), "n_days": len(date_scores),
                         "date_mean_crps201_db": float(date_scores.crps201_db.mean()),
                         "date_mean_crps401_db": float(date_scores.crps401_db.mean()),
                         "date_se_crps201_db": float(date_scores.crps201_db.std(ddof=1) / math.sqrt(len(date_scores)))})
    table = pd.DataFrame(rows)
    overall = table.loc[table.event.eq("all_train")].sort_values(["date_mean_crps201_db", "candidate"])
    main_winner = overall.iloc[0].candidate
    sensitivity_winner = table.loc[table.event.eq("all_train")].sort_values(["date_mean_crps401_db", "candidate"]).iloc[0].candidate
    table.to_csv(OUT / "marginal_cv_metrics.csv", index=False)
    day.to_csv(OUT / "marginal_cv_day_scores.csv", index=False)
    scores.to_parquet(OUT / "marginal_cv_hour_scores.parquet", index=False, compression="zstd")
    write_json(OUT / "marginal_selection.json", {
        "status": "frozen_marginal_family" if main_winner == sensitivity_winner else "requires_numerical_refinement",
        "selected_by": "minimum date-mean training-CV CRPS201", "candidate": main_winner,
        "candidate_401_grid": sensitivity_winner, "grid_rank_winner_agrees": main_winner == sensitivity_winner,
        "completed_utc": now(), "marginal_cv_qa_sha256": sha256(OUT / "marginal_cv_qa.json"),
        "note": "Marginal family only; temporal generator remains to be selected.",
    })
    print(overall.to_string(index=False), flush=True)


def status() -> None:
    for name in ("active_worker.json", "marginal_cv_qa.json", "marginal_selection.json"):
        path = OUT / name
        print(path.read_text(encoding="utf-8") if path.exists() else f"{name}: pending", flush=True)
    print(json.dumps({"completed_fold_records": len(list(TRIALS.glob("*__fold*.json"))) if TRIALS.exists() else 0,
                      "expected_fold_records": 40}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "report", "status"))
    args = parser.parse_args()
    {"run": run, "verify": verify, "report": report, "status": status}[args.command]()
