"""Compare the Open-Meteo ERA5 proxy with one official CDS day and audit coverage."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004"
PROXY = OUT / "openmeteo_era5_proxy"
SAMPLE = OUT / "era5_france/era5_france_20220801.zip"


def official(variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with zipfile.ZipFile(SAMPLE) as archive:
        for name in archive.namelist():
            with h5py.File(io.BytesIO(archive.read(name)), "r") as source:
                if variable in source:
                    return (source["latitude"][:], source["longitude"][:],
                            source[variable][:].astype(float))
    raise ValueError(f"Missing {variable} from CDS sample")


def main() -> None:
    cells = {}
    files = sorted(PROXY.glob("batch_*.json"))
    if len(files) != 6:
        raise ValueError(f"Need 6 complete proxy batches; found {len(files)}")
    expected_time = pd.date_range("2022-08-01", periods=3672, freq="h")
    for path in files:
        batch = json.loads(path.read_text(encoding="utf-8"))
        for pair, record in zip(batch["request"]["coordinates"], batch["response"]):
            key = tuple(pair)
            if key in cells:
                raise ValueError(f"Duplicate grid centre {key}")
            hours = pd.to_datetime(record["hourly"]["time"])
            if not hours.equals(expected_time):
                raise ValueError(f"Proxy hour grid mismatch for {key}")
            cells[key] = record["hourly"]
    if len(cells) != 55:
        raise ValueError(f"Expected 55 grid cells, got {len(cells)}")
    lat, lon, tcc = official("tcc")
    _, _, tp = official("tp")
    cloud_error = []
    precip_error = []
    cloud_threshold_difference = 0
    precip_threshold_difference = 0
    for (site_lat, site_lon), hourly in cells.items():
        i = int(np.argmin(np.abs(lat - site_lat)))
        j = int(np.argmin(np.abs(lon - site_lon)))
        if abs(lat[i] - site_lat) > 1e-8 or abs(lon[j] - site_lon) > 1e-8:
            raise ValueError("Proxy grid centre not in CDS sample")
        cloud_native = tcc[:, i, j] * 100.0
        cloud_proxy = np.asarray(hourly["cloud_cover"][:24], dtype=float)
        precip_native = tp[:, i, j] * 1000.0
        precip_proxy = np.asarray(hourly["precipitation"][:24], dtype=float)
        cloud_error.extend(np.abs(cloud_native - cloud_proxy).tolist())
        precip_error.extend(np.abs(precip_native - precip_proxy).tolist())
        cloud_threshold_difference += int(np.count_nonzero(
            (cloud_native <= 20.0) != (cloud_proxy <= 20.0)))
        precip_threshold_difference += int(np.count_nonzero(
            (precip_native < .1) != (precip_proxy < .1)))
    result = {"validated_utc": datetime.now(timezone.utc).isoformat(),
              "comparison_scope": "2022-08-01, 24 hours x 55 France ERA5 grid centres",
              "compared_cell_hours": len(cloud_error),
              "max_abs_cloud_percentage_points": max(cloud_error),
              "max_abs_precip_mm": max(precip_error),
              "cloud_clear_threshold_disagreements": cloud_threshold_difference,
              "precip_dry_threshold_disagreements": precip_threshold_difference,
              "limitation": "Only one CDS day available; Open-Meteo is rounded and may diverge on other dates."}
    path = OUT / "weather_proxy_comparison.json"
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
