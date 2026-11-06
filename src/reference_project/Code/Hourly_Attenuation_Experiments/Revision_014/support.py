"""Self-contained data loading, ranking, and record helpers for Revision 014.

The implementations reproduce the functions used by Revision 013 without
importing or reading code/results from any older experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models import MODEL_NAMES


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DATA = PROJECT / "Outputs/Datasets/Signal_Detectability_Revision_003"
LEVELS = (925, 850, 700, 650, 500, 300, 200, 30, 5, 1)
TARGET = "clear_sky_attenuation_db"
EVENTS = ("rain", "heavy_rain", "dry_overcast")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_inputs() -> tuple[pd.DataFrame, list[str], list[dict]]:
    names = pd.read_csv(DATA / "feature_columns.csv")["column"].tolist()
    assert len(names) == len(set(names)) == 348
    assert all(not x.startswith("p") for x in names[:188])
    suffixes = [x.split("__", 1)[1] for x in names[188:204]]
    assert len(suffixes) == 16
    for i, pressure in enumerate(LEVELS):
        assert names[188 + 16 * i:188 + 16 * (i + 1)] == [
            f"p{pressure}__{suffix}" for suffix in suffixes
        ]

    partitions = []
    for split, count in (("train", 2404), ("validation", 343)):
        table = pd.read_parquet(DATA / f"{split}.parquet")
        assert len(table) == count and list(table.columns) == [
            "hour_bjt", "hour_utc", *names, TARGET
        ]
        assert table["hour_bjt"].is_unique
        assert np.isfinite(table[TARGET].to_numpy(dtype=float)).all()
        table["split"] = split
        partitions.append(table)
    frame = pd.concat(partitions, ignore_index=True)
    assert frame["hour_bjt"].is_unique
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet", columns=[
        "hour_bjt", "split", "use_for_regression", *EVENTS
    ])
    assert status["hour_bjt"].is_unique
    frame = frame.merge(status, on="hour_bjt", how="left", validate="one_to_one",
                        suffixes=("", "_status"))
    assert frame["split"].eq(frame["split_status"]).all()
    assert frame["use_for_regression"].eq(True).all()
    assert frame[list(EVENTS)].notna().all().all()
    for event in EVENTS:
        frame[event] = frame[event].astype(bool)
    frame = frame.sort_values("hour_bjt").reset_index(drop=True)
    assert frame.loc[frame.split.eq("train"), "rain"].sum() == 179
    assert frame.loc[frame.split.eq("validation"), "rain"].sum() == 25
    assert frame.loc[frame.split.eq("validation"), "heavy_rain"].sum() == 5
    assert frame.loc[frame.split.eq("validation"), "dry_overcast"].sum() == 79
    assert str(frame.hour_bjt.dt.tz) == "Asia/Shanghai"
    assert frame.hour_utc.dt.tz is not None

    train_positions = np.flatnonzero(frame.split.to_numpy() == "train")
    n = len(train_positions)
    size = n // 8
    first = n - 3 * size
    folds = []
    for index in range(3):
        start = first + index * size
        val = train_positions[start:start + size]
        start_time = frame.loc[val[0], "hour_bjt"]
        earlier = train_positions[:start]
        fit = earlier[(frame.loc[earlier, "hour_bjt"] <
                       start_time - pd.Timedelta(hours=24)).to_numpy()]
        assert len(fit) > 1000 and len(val) == size
        assert frame.loc[fit, "hour_bjt"].max() < start_time - pd.Timedelta(hours=24)
        folds.append({
            "id": index + 1,
            "train_index": fit,
            "validation_index": val,
            "n_train": len(fit),
            "n_validation": len(val),
            "train_start_bjt": frame.loc[fit[0], "hour_bjt"].isoformat(),
            "train_end_bjt": frame.loc[fit[-1], "hour_bjt"].isoformat(),
            "validation_start_bjt": start_time.isoformat(),
            "validation_end_bjt": frame.loc[val[-1], "hour_bjt"].isoformat(),
        })
    return frame, names, folds


def load_training() -> tuple[pd.DataFrame, list[str], list[dict]]:
    frame, names, folds = load_inputs()
    extra = pd.read_parquet(DATA / "hourly_sample_status.parquet", columns=[
        "hour_bjt", "clear_dry", "snow", "weather_stratum"
    ])
    frame = frame.merge(extra, on="hour_bjt", how="left", validate="one_to_one")
    assert frame[["clear_dry", "snow", "weather_stratum"]].notna().all().all()
    assert frame["hour_bjt"].is_monotonic_increasing
    return frame, names, folds


def load_test(names: list[str]) -> pd.DataFrame:
    table = pd.read_parquet(DATA / "test.parquet")
    assert len(table) == 687 and list(table.columns) == [
        "hour_bjt", "hour_utc", *names, TARGET
    ]
    status = pd.read_parquet(DATA / "hourly_sample_status.parquet", columns=[
        "hour_bjt", "split", "use_for_regression", "rain", "heavy_rain",
        "dry_overcast", "clear_dry", "snow", "weather_stratum"
    ])
    table = table.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    assert table["split"].eq("test").all() and table["use_for_regression"].eq(True).all()
    assert table[TARGET].notna().all() and np.isfinite(table[TARGET]).all()
    assert table[["rain", "heavy_rain", "dry_overcast", "clear_dry", "snow"]].notna().all().all()
    return table.sort_values("hour_bjt").reset_index(drop=True)


def normalize(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def rank_models(rows: list[dict]) -> tuple[list[dict], dict]:
    assert len(rows) == len(MODEL_NAMES) == 10
    rmse = np.asarray([row["overall_rmse_mean"] for row in rows])
    latency = np.asarray([row["latency_median_ms"] for row in rows])
    assert np.isfinite(rmse).all() and np.isfinite(latency).all() and (latency > 0).all()
    rnorm = normalize(rmse)
    lnorm = normalize(np.log(latency))
    for i, row in enumerate(rows):
        row["normalized_rmse"] = float(rnorm[i])
        row["normalized_log_latency"] = float(lnorm[i])
        row["score_50_50"] = float(0.5 * (rnorm[i] + lnorm[i]))
        row["pareto_nondominated"] = not any(
            rmse[j] <= rmse[i] and latency[j] <= latency[i]
            and (rmse[j] < rmse[i] or latency[j] < latency[i])
            for j in range(10) if j != i
        )

    def winner(weight: float) -> str:
        return min(rows, key=lambda row: (
            weight * row["normalized_rmse"] +
            (1 - weight) * row["normalized_log_latency"],
            row["overall_rmse_mean"], row["event_rmse_mean"], row["name"]
        ))["name"]

    ranked = sorted(rows, key=lambda row: (
        row["score_50_50"], row["overall_rmse_mean"],
        row["event_rmse_mean"], row["name"]
    ))
    for i, row in enumerate(ranked, 1):
        row["rank_50_50"] = i
        row["rank_rmse_only"] = 1 + int(np.sum(rmse < row["overall_rmse_mean"]))
    return ranked, {
        "primary_winner": winner(0.5),
        "rmse_only_winner": min(rows, key=lambda row: (
            row["overall_rmse_mean"], row["name"]))["name"],
        "weight_sensitivity": {str(w): winner(w) for w in (0.25, 0.5, 0.75)},
        "pareto_models": [row["name"] for row in rows if row["pareto_nondominated"]],
        "rmse_range_db": [float(rmse.min()), float(rmse.max())],
        "latency_range_ms": [float(latency.min()), float(latency.max())],
    }
