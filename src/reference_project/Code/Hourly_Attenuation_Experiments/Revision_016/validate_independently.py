"""Independent, read-only recomputation of Revision 016 provenance and scores.

Does not import the training module or call its metrics/physics functions.
Writes only a separate validation result in the Revision 016 output directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_003"
ERA5 = PROJECT / "Outputs/Datasets/Revision_002/era5_hourly_pressure_levels.parquet"
OUT = PROJECT / "Outputs/Experiments/Hourly_Attenuation_Revision_016_ITU_Baseline"
TARGET = "clear_sky_attenuation_db"


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recalc(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    error = pred[mask] - truth[mask]
    return {"n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error))}


def main() -> None:
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf8"))
    final = json.loads((OUT / "final_test.json").read_text(encoding="utf8"))
    formal = json.loads((OUT / "formal_models.json").read_text(encoding="utf8"))
    check(final["fingerprint"] == formal["fingerprint"] == manifest["fingerprint"],
          "Experiment fingerprint mismatch")
    for rel, expected in manifest["source_sha256"].items():
        check(sha256(PROJECT / rel) == expected, f"Input changed: {rel}")
    check(sha256(DATA / "test.parquet") == final["test_parquet_sha256"],
          "Test parquet changed")
    weather = pd.read_parquet(ERA5, columns=["hour_bjt",
        "surface__total_precipitation", "surface__snowfall"])
    check(weather.hour_bjt.is_unique, "Duplicate ERA5 hour")
    weather.hour_bjt = weather.hour_bjt - pd.Timedelta(hours=1)
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet",
                             columns=["hour_bjt", "rain", "dry_overcast", "clear_dry", "split"])
    count = {}
    for split, expected in [("train", 2404), ("validation", 343), ("test", 687)]:
        part = pd.read_parquet(DATA / f"{split}.parquet",
                               columns=["hour_bjt", TARGET])
        check(len(part) == expected and part.hour_bjt.is_unique, f"{split} size/key")
        part = part.merge(weather, on="hour_bjt", how="left", validate="one_to_one")
        part = part.merge(status, on="hour_bjt", how="left", validate="one_to_one")
        check(part.split.eq(split).all(), f"{split} membership")
        total = part.surface__total_precipitation.to_numpy(dtype=float) * 1000
        snow = part.surface__snowfall.to_numpy(dtype=float) * 1000
        known = np.isfinite(total) & np.isfinite(snow)
        inferred_rain = known & (total >= .1) & (snow < .1)
        check(np.array_equal(inferred_rain, part.rain.to_numpy(dtype=bool)),
              f"{split} t+1 rainfall alignment")
        count[split] = {"n": len(part), "rain": int(part.rain.sum()),
                        "missing_aligned_weather": int((~known).sum())}
        if split == "test":
            test = part
    check(count["train"]["rain"] == 179 and count["validation"]["rain"] == 25
          and count["test"]["rain"] == 51, "Rain event counts changed")
    check(test.hour_bjt.astype(str).tolist() == final["hour_bjt"], "Test order changed")
    truth = test[TARGET].to_numpy(dtype=float)
    check(np.allclose(truth, final["truth_db"], rtol=0, atol=1e-12), "Test truth changed")
    check(len(final["results"]) == 12, "Missing method result")
    for name, row in final["results"].items():
        pred = np.asarray(row["prediction_db"], dtype=float)
        check(pred.shape == truth.shape and np.isfinite(pred).all(),
              f"Invalid predictions: {name}")
        for event, mask in {
            "overall": np.ones(len(test), dtype=bool),
            "rain": test.rain.to_numpy(dtype=bool),
            "dry_overcast": test.dry_overcast.to_numpy(dtype=bool),
            "clear_dry": test.clear_dry.to_numpy(dtype=bool),
        }.items():
            got = recalc(truth, pred, mask)
            saved = row["metrics"][event]
            for key in ("n", "rmse", "mae", "bias"):
                check(np.isclose(got[key], saved[key], rtol=0, atol=1e-10),
                      f"{name}/{event}/{key} score mismatch")
        if name.startswith("physics_"):
            no_rain = np.asarray(final["aligned_liquid_rain_mmph"], dtype=object)
            dry = np.array([x is not None and x == 0 for x in no_rain], dtype=bool)
            check(np.allclose(pred[dry], 0, atol=1e-12),
                  f"Physical no-rain value is not zero: {name}")
    feature_sets = {"full_direct": 350, "full_residual": 350,
                    "local_direct": 119, "local_residual": 119}
    for name, expected in feature_sets.items():
        info = formal["models"][name]
        columns = info["feature_names"]
        check(len(columns) == len(set(columns)) == expected, f"{name} feature count")
        check(columns[-2:] == ["aligned_liquid_rain_mmph", "itu838_geom_db"],
              f"{name} added physics schema")
        check(not any(c in columns for c in [TARGET, "hour_bjt", "hour_utc",
                                             "rain", "dry_overcast", "clear_dry"]),
              f"{name} target/event/time leakage")
        check(info["trained_rounds"] == info["parameters"]["n_estimators"]
              == len(info["rounds_hold_model_target_rmse_db"]),
              f"{name} training round count")
        check(sha256(OUT / info["model_file"]) == info["model_sha256"],
              f"{name} fitted model changed")
    report = {"status": "PASS", "source_hashes_checked": len(manifest["source_sha256"]),
              "event_alignment": count, "methods_recomputed": len(final["results"]),
              "metric_groups_per_method": 4, "model_feature_lists_checked": len(feature_sets),
              "no_test_target_or_event_as_input": True,
              "limitations": ["independent arithmetic/provenance QA, not physical ground truth",
                              "test hours had been viewed in previous experiments"]}
    path = OUT / "independent_validation.json"
    if path.exists():
        check(json.loads(path.read_text(encoding="utf8")) == report,
              "Existing independent validation differs")
    else:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
