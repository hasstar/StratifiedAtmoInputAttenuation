"""Revision 016: frozen ITU-coefficient rain baseline and paired XGBoost test.

Run in order: selftest, prepare, fit, verify, test, finalize. Existing data and
Revision 015 are read-only. The test parquet is not opened before `test`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REV15 = PROJECT / "Code/Hourly_Attenuation_Experiments/Revision_015/run_feature_experiment.py"
SPEC = importlib.util.spec_from_file_location("frozen_revision_015", REV15)
assert SPEC is not None and SPEC.loader is not None
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)

DATA = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_003"
ERA5 = PROJECT / "Outputs/Datasets/Revision_002/era5_hourly_pressure_levels.parquet"
OLD_OUT = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_015_Event_Features"
OUT = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline"
FREQ_GHZ = 19.13
LAT_DEG, LON_DEG, SAT_LON_DEG = 39.9, 116.3, 125.0
EARTH_KM, GEO_ALT_KM = 6371.0, 35786.0
MAIN_POL = "circular"
POLARIZATIONS = ("horizontal", "vertical", MAIN_POL)
MODELS = {
    "full_direct": ("full348", "shallow", "uniform", False),
    "full_residual": ("full348", "shallow", "uniform", True),
    "local_direct": ("local117", "profile", "event_moderate", False),
    "local_residual": ("local117", "profile", "event_moderate", True),
}
TARGET = old.TARGET

COEFFICIENT_TABLES = {
    "k_h": ([-5.33980, -0.35351, -0.23789, -0.94158],
            [-0.10008, 1.26970, 0.86036, 0.64552],
            [1.13098, 0.45400, 0.15354, 0.16817], -0.18961, 0.71147),
    "k_v": ([-3.80595, -3.44965, -0.39902, 0.50167],
            [0.56934, -0.22911, 0.73042, 1.07319],
            [0.81061, 0.51059, 0.11899, 0.27195], -0.16398, 0.63297),
    "alpha_h": ([-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
                [1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
                [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990],
                0.67849, -1.95537),
    "alpha_v": ([-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
                [2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
                [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479],
                -0.053739, 0.83433),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def input_paths() -> list[Path]:
    return [DATA / "train.parquet", DATA / "validation.parquet",
            DATA / "feature_columns.csv", DATA / "hourly_sample_status.parquet",
            PROJECT / "Outputs/Datasets/Revision_002/feature_sets.csv", ERA5,
            REV15, OLD_OUT / "run_manifest.json", HERE / "PROTOCOL.md",
            HERE / "run_itu_baseline.py"]


def fingerprint() -> tuple[str, dict]:
    hashes = {str(p.relative_to(PROJECT)): digest(p) for p in input_paths()}
    token = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return token, hashes


def checked_manifest() -> dict:
    path = OUT / "run_manifest.json"
    require(path.exists(), "Run prepare first")
    manifest = read(path)
    require(manifest["fingerprint"] == fingerprint()[0], "Frozen input/code changed")
    return manifest


def elevation_deg() -> float:
    lat = np.deg2rad(LAT_DEG)
    delta_lon = np.deg2rad(SAT_LON_DEG - LON_DEG)
    cos_psi = np.cos(lat) * np.cos(delta_lon)
    return float(np.rad2deg(np.arctan2(
        cos_psi - EARTH_KM / (EARTH_KM + GEO_ALT_KM),
        np.sqrt(1.0 - cos_psi * cos_psi),
    )))


def itu_coefficients(frequency: float, polarization: str, elevation: float) -> tuple[float, float]:
    x = np.log10(frequency)

    def fitted(kind: str) -> float:
        a, b, c, m, intercept = COEFFICIENT_TABLES[kind]
        total = sum(aj * np.exp(-((x - bj) / cj) ** 2)
                    for aj, bj, cj in zip(a, b, c))
        answer = total + m * x + intercept
        return float(10.0 ** answer if kind.startswith("k_") else answer)

    kh, kv = fitted("k_h"), fitted("k_v")
    ah, av = fitted("alpha_h"), fitted("alpha_v")
    tilt = {"horizontal": 0.0, "vertical": 90.0,
            MAIN_POL: 45.0}[polarization]
    factor = np.cos(np.deg2rad(elevation)) ** 2 * np.cos(np.deg2rad(2 * tilt))
    k = (kh + kv + (kh - kv) * factor) / 2.0
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * factor) / (2.0 * k)
    return float(k), float(alpha)


def aligned_frame(split: str, names: list[str]) -> pd.DataFrame:
    frame = old.load_partition(split, names)
    weather = pd.read_parquet(ERA5, columns=["hour_bjt",
        "surface__total_precipitation", "surface__snowfall"])
    require(weather.hour_bjt.is_unique, "ERA5 duplicate hour")
    weather["hour_bjt"] = weather.hour_bjt - pd.Timedelta(hours=1)
    weather = weather.rename(columns={
        "surface__total_precipitation": "aligned_total_m",
        "surface__snowfall": "aligned_snow_m",
    })
    frame = frame.merge(weather, on="hour_bjt", how="left", validate="one_to_one")
    total = frame.aligned_total_m.to_numpy(dtype=float) * 1000.0
    snow = frame.aligned_snow_m.to_numpy(dtype=float) * 1000.0
    available = np.isfinite(total) & np.isfinite(snow)
    expected_rain = available & (total >= 0.1) & (snow < 0.1)
    require(np.array_equal(expected_rain, frame.rain.to_numpy()),
            f"{split}: aligned rainfall does not reproduce weather event label")
    frame["aligned_liquid_rain_mmph"] = np.where(
        available, np.maximum(total - snow, 0.0), np.nan)
    frame["physical_rain_known"] = available
    require(int((~available).sum()) <= 1, f"{split}: unexpected ERA5 gap")
    return frame


def rain_h0_median(frame: pd.DataFrame) -> float:
    value = frame["surface__zero_degree_level"].to_numpy(dtype=float)
    median = float(np.nanmedian(value))
    require(np.isfinite(median) and median >= 0, "Cannot fit 0C-height median")
    return median


def physical(frame: pd.DataFrame, h0_median_m: float,
             polarization: str) -> dict[str, np.ndarray]:
    elevation = elevation_deg()
    theta = np.deg2rad(elevation)
    k, alpha = itu_coefficients(FREQ_GHZ, polarization, elevation)
    rain_raw = frame.aligned_liquid_rain_mmph.to_numpy(dtype=float)
    rain = np.where(np.isfinite(rain_raw), np.maximum(rain_raw, 0.0), 0.0)
    h0_raw = frame.surface__zero_degree_level.to_numpy(dtype=float)
    h0 = np.where(np.isfinite(h0_raw), np.maximum(h0_raw, 0.0), h0_median_m)
    delta_h = h0 / 1000.0 + 0.36
    slant = delta_h / np.sin(theta)
    gamma = k * np.power(rain, alpha)
    geom = gamma * slant

    # P.618-14 steps 6-9 with hourly R in place of its stipulated R_0.01.
    horizontal = slant * np.cos(theta)
    reduction = 1.0 / (1.0 + 0.78 * np.sqrt(horizontal * gamma / FREQ_GHZ)
                       - 0.38 * (1.0 - np.exp(-2.0 * horizontal)))
    zeta = np.rad2deg(np.arctan2(delta_h, horizontal * reduction))
    lr = np.where(zeta > elevation, horizontal * reduction / np.cos(theta), slant)
    chi = max(36.0 - abs(LAT_DEG), 0.0)
    vertical = 1.0 / (1.0 + np.sqrt(np.sin(theta)) * (
        31.0 * (1.0 - np.exp(-elevation / (1.0 + chi)))
        * np.sqrt(lr * gamma) / FREQ_GHZ ** 2 - 0.45))
    p618_hourly = gamma * lr * vertical
    require(np.isfinite(geom).all() and np.isfinite(p618_hourly).all()
            and (geom >= 0).all() and (p618_hourly >= 0).all(),
            "Invalid physical rain attenuation")
    return {"geom": geom, "p618_hourly": p618_hourly,
            "slant_km": slant, "gamma_db_per_km": gamma}


def model_columns(view: str, names: list[str], local: list[str]) -> list[str]:
    base = names if view == "full348" else local
    return [*base, "aligned_liquid_rain_mmph", "itu838_geom_db"]


def metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    return old.scores(frame, np.asarray(pred, dtype=float))


def model_frame(frame: pd.DataFrame, h0_median: float) -> pd.DataFrame:
    result = frame.copy()
    result["itu838_geom_db"] = physical(frame, h0_median, MAIN_POL)["geom"]
    return result


def fit_one(fit: pd.DataFrame, hold: pd.DataFrame, name: str,
            names: list[str], local: list[str], *, save_model: bool) -> dict:
    view, config, weight, residual = MODELS[name]
    h0 = rain_h0_median(fit)
    fit = model_frame(fit, h0)
    hold = model_frame(hold, h0)
    columns = model_columns(view, names, local)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xfit = imputer.fit_transform(fit[columns].replace([np.inf, -np.inf], np.nan))
    xhold = imputer.transform(hold[columns].replace([np.inf, -np.inf], np.nan))
    base_fit = fit.itu838_geom_db.to_numpy(dtype=float)
    base_hold = hold.itu838_geom_db.to_numpy(dtype=float)
    yfit = fit[TARGET].to_numpy(dtype=float) - (base_fit if residual else 0.0)
    yhold = hold[TARGET].to_numpy(dtype=float) - (base_hold if residual else 0.0)
    model: XGBRegressor = old.predictor(config)
    model.fit(xfit, yfit, sample_weight=old.sample_weights(fit, weight),
              eval_set=[(xfit, yfit), (xhold, yhold)], verbose=False)
    pred_fit = np.asarray(model.predict(xfit), dtype=float) + (base_fit if residual else 0.0)
    pred_hold = np.asarray(model.predict(xhold), dtype=float) + (base_hold if residual else 0.0)
    history = model.evals_result()
    curve_fit = history["validation_0"]["rmse"]
    curve_hold = history["validation_1"]["rmse"]
    require(len(curve_fit) == len(curve_hold) == old.CONFIGS[config]["n_estimators"],
            "Incomplete fixed training schedule")
    record = {
        "model_name": name, "view": view, "config": config, "weight": weight,
        "residual_learning": residual, "feature_names": columns,
        "feature_count": len(columns), "h0_train_median_m": h0,
        "imputer_medians": imputer.statistics_.astype(float).tolist(),
        "parameters": old.CONFIGS[config], "event_weights": old.WEIGHTS[weight],
        "trained_rounds": len(curve_fit),
        "stopping_reason": "fixed tree count from frozen Revision 015 configuration; no validation early stopping",
        "rounds_fit_model_target_rmse_db": [float(x) for x in curve_fit],
        "rounds_hold_model_target_rmse_db": [float(x) for x in curve_hold],
        "fit_metrics_original_target": metrics(fit, pred_fit),
        "hold_metrics_original_target": metrics(hold, pred_hold),
        "hold_hour_bjt": hold.hour_bjt.astype(str).tolist(),
        "hold_truth_db": hold[TARGET].astype(float).tolist(),
        "hold_prediction_db": pred_hold.tolist(),
        "hold_physics_db": base_hold.tolist(),
    }
    if save_model:
        path = OUT / "models" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.json")
        model.save_model(temp)
        os.replace(temp, path)
        record["model_file"] = str(path.relative_to(OUT))
        record["model_sha256"] = digest(path)
    return record


def selftest() -> None:
    elev = elevation_deg()
    require(40 < elev < 50, "Unexpected GEO elevation")
    for f, kh, ah, kv, av in [(19.0, .08084, 1.0691, .08642, .9930),
                               (20.0, .09164, 1.0568, .09611, .9847)]:
        # P.838 tabulates 0-degree horizontal/vertical specific coefficients.
        def fitted(kind: str) -> float:
            x = np.log10(f)
            a, b, c, m, intercept = COEFFICIENT_TABLES[kind]
            y = sum(aj * np.exp(-((x - bj) / cj) ** 2)
                    for aj, bj, cj in zip(a, b, c)) + m * x + intercept
            return float(10 ** y if kind.startswith("k_") else y)
        for kind, expected in [("k_h", kh), ("alpha_h", ah),
                               ("k_v", kv), ("alpha_v", av)]:
            require(abs(fitted(kind) - expected) < 0.0001,
                    f"P.838 official table self-test failed: {f}/{kind}")
    names, local = old.names_and_local()
    train = aligned_frame("train", names)
    validation = aligned_frame("validation", names)
    require(len(local) == 117 and train.rain.sum() == 179
            and validation.rain.sum() == 25, "Frozen data counts changed")
    h0 = rain_h0_median(train)
    all_phys = {p: physical(train, h0, p) for p in POLARIZATIONS}
    require(all(np.allclose(x["geom"][~train.physical_rain_known], 0)
                for x in all_phys.values()), "Missing-rain baseline handling")
    require(np.allclose(all_phys[MAIN_POL]["geom"][train.aligned_liquid_rain_mmph.eq(0)], 0),
            "Zero rain should have zero rain attenuation")
    print(json.dumps({"status": "selftest_pass", "elevation_deg": elev,
                      "rain_counts": [int(train.rain.sum()), int(validation.rain.sum())],
                      "h0_train_median_m": h0,
                      "coefficient_19p13": {p: itu_coefficients(FREQ_GHZ, p, elev)
                                            for p in POLARIZATIONS}}, ensure_ascii=False))


def prepare() -> None:
    require(not (OUT / "run_manifest.json").exists(), "Revision 016 already prepared")
    names, _ = old.names_and_local()
    train = aligned_frame("train", names)
    validation = aligned_frame("validation", names)
    old_manifest = read(OLD_OUT / "run_manifest.json")
    folds = old_manifest["folds"]
    require(len(folds) == 3 and all(len(f["hold_indices"]) > 0 for f in folds),
            "Frozen grouped folds unavailable")
    require(int(train.rain.sum()) == 179 and int(validation.rain.sum()) == 25,
            "Unexpected split event counts")
    token, hashes = fingerprint()
    save(OUT / "run_manifest.json", {
        "fingerprint": token, "source_sha256": hashes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_revision_015_fingerprint": old_manifest["fingerprint"],
        "frequency_ghz": FREQ_GHZ, "lat_deg": LAT_DEG,
        "station_lon_deg": LON_DEG, "satellite_lon_deg": SAT_LON_DEG,
        "elevation_deg": elevation_deg(), "main_polarization": MAIN_POL,
        "frequency_and_circular_polarization_confirmed_by_user_2026_09_30": True,
        "rain_input": "(ERA5 tp(t+1)-snowfall(t+1))*1000 mm/h, floored at zero",
        "same_hour_weather_is_available_only_after_hour_end": True,
        "p618_hourly_is_not_standard_long_term_prediction": True,
        "models": MODELS, "folds": folds,
        "n_train": len(train), "n_validation": len(validation),
        "test_not_in_selection_fingerprint": True,
        "software": {"python": sys.version, "numpy": np.__version__,
                     "pandas": pd.__version__, "xgboost": __import__("xgboost").__version__},
    })
    print(json.dumps({"status": "prepared", "fingerprint": token,
                      "folds": len(folds), "models": len(MODELS)}))


def fit() -> None:
    manifest = checked_manifest()
    require(not (OUT / "formal_models.json").exists(), "Fit already exists; do not overwrite")
    names, local = old.names_and_local()
    train = aligned_frame("train", names)
    validation = aligned_frame("validation", names)
    for name in MODELS:
        path = OUT / "cv" / f"{name}.json"
        if path.exists():
            require(read(path)["fingerprint"] == manifest["fingerprint"],
                    "Existing CV result has different input fingerprint")
            continue
        records = []
        for fold in manifest["folds"]:
            fit_frame = train.iloc[fold["fit_indices"]].reset_index(drop=True)
            hold_frame = train.iloc[fold["hold_indices"]].reset_index(drop=True)
            record = fit_one(fit_frame, hold_frame, name, names, local, save_model=False)
            record["fold"] = fold["fold"]
            records.append(record)
        save(path, {"fingerprint": manifest["fingerprint"], "folds": records})
        print(json.dumps({"cv_complete": name,
                          "rain_rmse": [r["hold_metrics_original_target"]["rain"]["rmse"]
                                        for r in records]}), flush=True)
    formal = {}
    for name in MODELS:
        formal[name] = fit_one(train, validation, name, names, local, save_model=True)
        print(json.dumps({"formal_complete": name,
                          "validation_rain_rmse": formal[name]["hold_metrics_original_target"]["rain"]["rmse"]}),
              flush=True)
    h0 = rain_h0_median(train)
    physical_validation = {}
    for pol in POLARIZATIONS:
        physical_validation[pol] = {
            key: metrics(validation, prediction)
            for key, prediction in physical(validation, h0, pol).items()
            if key in ("geom", "p618_hourly")
        }
    save(OUT / "formal_models.json", {"fingerprint": manifest["fingerprint"],
                                      "n_train": len(train), "n_validation": len(validation),
                                      "models": formal,
                                      "physical_validation_metrics": physical_validation})


def predict_saved(info: dict, frame: pd.DataFrame) -> np.ndarray:
    path = OUT / info["model_file"]
    require(digest(path) == info["model_sha256"], "Frozen fitted model changed")
    source = model_frame(frame, info["h0_train_median_m"])
    columns = info["feature_names"]
    x = source[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    medians = np.asarray(info["imputer_medians"], dtype=float)
    x = np.where(np.isfinite(x), x, medians[None, :])
    model = XGBRegressor()
    model.load_model(path)
    prediction = np.asarray(model.predict(x), dtype=float)
    if info["residual_learning"]:
        prediction += source.itu838_geom_db.to_numpy(dtype=float)
    return prediction


def same_scores(a: dict, b: dict) -> bool:
    return all(int(a[event]["n"]) == int(b[event]["n"])
               and all(np.isclose(a[event][metric], b[event][metric], atol=1e-9)
                       for metric in ("rmse", "mae", "bias"))
               for event in old.EVENTS)


def verify(pretest: bool) -> None:
    manifest = checked_manifest()
    names, _ = old.names_and_local()
    train = aligned_frame("train", names)
    validation = aligned_frame("validation", names)
    require((OUT / "formal_models.json").exists(), "Formal fits incomplete")
    formal = read(OUT / "formal_models.json")
    require(formal["fingerprint"] == manifest["fingerprint"], "Formal fingerprint mismatch")
    for name in MODELS:
        cv = read(OUT / "cv" / f"{name}.json")
        require(cv["fingerprint"] == manifest["fingerprint"] and len(cv["folds"]) == 3,
                f"Incomplete CV {name}")
        for saved, fold in zip(cv["folds"], manifest["folds"]):
            hold = train.iloc[fold["hold_indices"]].reset_index(drop=True)
            require(saved["fold"] == fold["fold"] and
                    saved["hold_hour_bjt"] == hold.hour_bjt.astype(str).tolist(),
                    "Stored CV holdout is inconsistent")
            require(np.allclose(saved["hold_truth_db"], hold[TARGET], atol=1e-12),
                    "CV truth mismatch")
            require(same_scores(metrics(hold, saved["hold_prediction_db"]),
                                saved["hold_metrics_original_target"]), "CV score mismatch")
            require(len(saved["rounds_fit_model_target_rmse_db"]) ==
                    len(saved["rounds_hold_model_target_rmse_db"]) ==
                    old.CONFIGS[saved["config"]]["n_estimators"], "CV curve incomplete")
        info = formal["models"][name]
        require(info["feature_count"] == len(info["feature_names"]) ==
                len(info["imputer_medians"]), "Formal feature mismatch")
        prediction = predict_saved(info, validation)
        require(np.allclose(prediction, info["hold_prediction_db"], atol=1e-5),
                "Formal model does not reproduce validation predictions")
        require(same_scores(metrics(validation, prediction),
                            info["hold_metrics_original_target"]),
                "Formal validation scores inconsistent")
    if pretest:
        require(not (OUT / "final_test.json").exists(), "Test already exists")
        save(OUT / "qa_pretest.json", {"status": "PASS", "fingerprint": manifest["fingerprint"],
                                      "cv_fits": 12, "formal_fits": 4,
                                      "n_validation": len(validation),
                                      "test_parquet_not_opened": True})
    else:
        final = read(OUT / "final_test.json")
        test = aligned_frame("test", names)
        require(final["fingerprint"] == manifest["fingerprint"] and
                final["test_parquet_sha256"] == digest(DATA / "test.parquet") and
                final["hour_bjt"] == test.hour_bjt.astype(str).tolist(),
                "Test provenance mismatch")
        require(np.allclose(final["truth_db"], test[TARGET], atol=1e-12),
                "Test truth mismatch")
        for name, row in final["results"].items():
            require(same_scores(metrics(test, row["prediction_db"]), row["metrics"]),
                    f"Test metric mismatch {name}")
            if name in MODELS:
                require(np.allclose(row["prediction_db"],
                                    predict_saved(formal["models"][name], test), atol=1e-5),
                        f"Test prediction/model mismatch {name}")
        save(OUT / "qa_final.json", {"status": "PASS", "fingerprint": manifest["fingerprint"],
                                    "n_test": len(test), "result_count": len(final["results"]),
                                    "all_metrics_recomputed": True})
    print(json.dumps({"status": "PASS", "pretest": pretest,
                      "models": len(MODELS)}))


def day_bootstrap(frame: pd.DataFrame, first: np.ndarray,
                  second: np.ndarray, seed: int = 20260930) -> dict:
    rainy = frame.rain.to_numpy(dtype=bool)
    dates = frame.hour_bjt.dt.date.astype(str).to_numpy()
    unique = np.unique(dates[rainy])
    rng = np.random.default_rng(seed)
    deltas = []
    truth = frame[TARGET].to_numpy(dtype=float)
    for _ in range(2000):
        draws = rng.choice(unique, size=len(unique), replace=True)
        picked = np.concatenate([np.flatnonzero(rainy & (dates == day)) for day in draws])
        e1 = first[picked] - truth[picked]
        e2 = second[picked] - truth[picked]
        deltas.append(np.sqrt(np.mean(e1 * e1)) - np.sqrt(np.mean(e2 * e2)))
    return {"n_rain_days": len(unique), "n_bootstrap": len(deltas),
            "direct_minus_residual_rmse_db": float(np.sqrt(np.mean((first[rainy]-truth[rainy])**2))
                                                   - np.sqrt(np.mean((second[rainy]-truth[rainy])**2))),
            "ci95_db": [float(x) for x in np.quantile(deltas, [.025, .975])],
            "positive_means_residual_better": True}


def test() -> None:
    manifest = checked_manifest()
    require(read(OUT / "qa_pretest.json")["status"] == "PASS", "Pretest QA missing")
    require(not (OUT / "final_test.json").exists(), "Test already evaluated; do not overwrite")
    names, _ = old.names_and_local()
    train = aligned_frame("train", names)
    test_frame = aligned_frame("test", names)
    formal = read(OUT / "formal_models.json")["models"]
    results = {}
    h0 = rain_h0_median(train)
    for pol in POLARIZATIONS:
        phys = physical(test_frame, h0, pol)
        for key in ("geom", "p618_hourly"):
            name = f"physics_{key}_{pol}"
            pred = phys[key]
            results[name] = {"prediction_db": pred.tolist(),
                             "metrics": metrics(test_frame, pred),
                             "formula_polarization": pol}
    for name in MODELS:
        pred = predict_saved(formal[name], test_frame)
        results[name] = {"prediction_db": pred.tolist(),
                         "metrics": metrics(test_frame, pred),
                         "feature_count": formal[name]["feature_count"],
                         "parameters": formal[name]["parameters"],
                         "trained_rounds": formal[name]["trained_rounds"]}
    legacy_path = OLD_OUT / "final_test.json"
    legacy = read(legacy_path)
    require(legacy["hour_bjt"] == test_frame.hour_bjt.astype(str).tolist() and
            np.allclose(legacy["truth_db"], test_frame[TARGET], atol=1e-12),
            "Frozen Revision 015 test is not the same hours and target")
    for role in ("reference", "winner"):
        pred = np.asarray(legacy["models"][role]["prediction_db"], dtype=float)
        require(same_scores(metrics(test_frame, pred), legacy["models"][role]["metrics"]),
                "Frozen prior result score mismatch")
        results[f"legacy_{role}"] = {"prediction_db": pred.tolist(),
                                     "metrics": metrics(test_frame, pred),
                                     "source_sha256": digest(legacy_path)}
    bootstrap = {
        view: day_bootstrap(test_frame,
            np.asarray(results[f"{view}_direct"]["prediction_db"]),
            np.asarray(results[f"{view}_residual"]["prediction_db"]))
        for view in ("full", "local")
    }
    save(OUT / "final_test.json", {
        "fingerprint": manifest["fingerprint"],
        "test_parquet_sha256": digest(DATA / "test.parquet"),
        "legacy_final_test_sha256": digest(legacy_path),
        "n_test": len(test_frame), "hour_bjt": test_frame.hour_bjt.astype(str).tolist(),
        "truth_db": test_frame[TARGET].astype(float).tolist(),
        "rain": test_frame.rain.astype(bool).tolist(),
        "dry_overcast": test_frame.dry_overcast.astype(bool).tolist(),
        "aligned_liquid_rain_mmph": [None if not np.isfinite(x) else float(x)
                                     for x in test_frame.aligned_liquid_rain_mmph],
        "h0_train_median_m": h0, "results": results, "paired_bootstrap": bootstrap,
        "test_used_for_model_or_polarization_selection": False,
        "not_new_blind_test_due_to_prior_experiments": True,
    })
    print(json.dumps({"status": "test_saved", "rain_n": int(test_frame.rain.sum()),
                      "rain_rmse": {k: round(v["metrics"]["rain"]["rmse"], 5)
                                    for k, v in results.items()}}, ensure_ascii=False))


def finalize() -> None:
    require(read(OUT / "qa_final.json")["status"] == "PASS", "Final QA not PASS")
    manifest = checked_manifest()
    final = read(OUT / "final_test.json")
    formal = read(OUT / "formal_models.json")["models"]
    lines = ["# Revision 016：ITU 雨衰物理基线实验", "",
        f"输入/代码指纹：`{manifest['fingerprint']}`。训练 2404、验证 343、测试 687 小时。",
        "目标：同小时相对晴空衰减（dB）；雨天包含全部强雨。",
        "用户确认整段观测期为 19.13 GHz、圆极化；水平/垂直仅为敏感性情景。",
        "P.618 小时代理不是标准的 1 分钟 R0.01 长期预测。", "",
        "## 测试结果（RMSE / dB）", "",
        "| 方法 | 总体 | 雨天 | 干燥阴天 | 雨天 MAE |", "|---|---:|---:|---:|---:|" ]
    order = ["legacy_reference", "legacy_winner", "physics_geom_circular",
             "physics_p618_hourly_circular", "full_direct", "full_residual",
             "local_direct", "local_residual",
             "physics_geom_horizontal", "physics_geom_vertical",
             "physics_p618_hourly_horizontal", "physics_p618_hourly_vertical"]
    for name in order:
        m = final["results"][name]["metrics"]
        lines.append(f"| {name} | {m['overall']['rmse']:.4f} | {m['rain']['rmse']:.4f} | "
                     f"{m['dry_overcast']['rmse']:.4f} | {m['rain']['mae']:.4f} |")
    lines += ["", "雨天测试仅 51 个合格小时；不能把波动解释为普遍气候增益。", "",
              "## 同输入配对的雨天差异", ""]
    for view, value in final["paired_bootstrap"].items():
        lines.append(f"- {view}: 直接模型 RMSE 减残差模型 RMSE = "
                     f"{value['direct_minus_residual_rmse_db']:+.4f} dB；"
                     f"按 {value['n_rain_days']} 个自然日配对重抽的 95% 区间 "
                     f"[{value['ci95_db'][0]:+.4f}, {value['ci95_db'][1]:+.4f}] dB。"
                     "正值才表示残差模型更好。")
    lines += ["", "## 训练设置", "",
              "两套输入均额外使用对齐的当前小时液态降水与 P.838+几何雨衰。"
              "直接和残差方法在相同输入、相同 XGBoost 超参数/权重下比较。", ""]
    for name, info in formal.items():
        val = info["hold_metrics_original_target"]
        cv = read(OUT / "cv" / f"{name}.json")
        fold_rain = [row["hold_metrics_original_target"]["rain"]["rmse"]
                     for row in cv["folds"]]
        lines.append(f"- `{name}`: {info['feature_count']} 列；配置 `{info['config']}`，"
                     f"权重 `{info['weight']}`；{info['trained_rounds']} 棵树（固定终点，"
                     "验证集不早停）；训练内三折雨天 RMSE "
                     f"{[round(x, 4) for x in fold_rain]}；正式验证雨天 "
                     f"{val['rain']['rmse']:.4f} dB。逐轮曲线、全部参数及逐小时预测"
                     "在 `formal_models.json` 与 `cv/`。")
    lines += ["", "## 物理与解释边界", "",
              f"由原稿场景算得 GEO 仰角 {manifest['elevation_deg']:.3f}°，"
              f"频率 {FREQ_GHZ} GHz、圆极化由用户确认；站点近似坐标和名义轨位仍需测量元数据核准。",
              "降水的 ERA5 `t+1` 累计量对应实测 `[t,t+1)`；所以这是小时结束后的估计，"
              "不是小时开始时的实时预报。0°C 高度缺失由训练中位数填补。",
              "P.838 比衰减配几何湿路径是简化场景；P.618 输入被改为小时雨率只能当代理。",
              "质量不足、深衰落不可检出的小时已由数据集排除，本结果仅针对可检出小时。",
              "旧测试小时在先前实验中已被查看；本版方案固定后复评，但不能称新盲测。",
              "逐小时随机划分会让同一降雨过程跨集合，本结果不证明异时段/异地泛化。", "",
              "官方方法：[P.838-3](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf)、"
              "[P.839-4](https://www.itu.int/dms_pubrec/itu-r/rec/p/r-rec-p.839-4-201309-i!!pdf-e.pdf)、"
              "[P.618-14](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-14-202308-I!!PDF-E.pdf)。", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "documented", "result": str(OUT / "RESULTS.md")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "prepare", "fit", "verify",
                                            "test", "verify_final", "finalize"))
    args = parser.parse_args()
    {
        "selftest": selftest, "prepare": prepare, "fit": fit,
        "verify": lambda: verify(True), "test": test,
        "verify_final": lambda: verify(False), "finalize": finalize,
    }[args.command]()


if __name__ == "__main__":
    main()
