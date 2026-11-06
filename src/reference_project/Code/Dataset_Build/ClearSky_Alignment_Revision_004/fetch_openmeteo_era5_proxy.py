"""Fetch Open-Meteo's explicit ERA5 model at OpenSat grid centres.

This is an auditable fallback if the official CDS requests remain queued.
Open-Meteo rounds cloud cover to integer percent and precipitation to 0.1 mm;
it is not byte-identical to the official CDS NetCDF and must be labelled so.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT = Path(__file__).resolve().parents[3]
STAGE = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_003/opensat_hourly_power.parquet"
OUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004/openmeteo_era5_proxy"
API = "https://archive-api.open-meteo.com/v1/archive"


def main() -> None:
    sites = pd.read_parquet(STAGE, columns=["link_id", "site_lat_deg", "site_lon_deg"])
    sites = sites.drop_duplicates("link_id")
    # Exact ERA5 0.25-degree grid centres, matching official CDS sample cells.
    cells = sorted(set(zip((np.rint(sites.site_lat_deg * 4) / 4).tolist(),
                           (np.rint(sites.site_lon_deg * 4) / 4).tolist())))
    if len(cells) != 55:
        raise ValueError(f"Expected 55 unique OpenSat ERA5 cells, got {len(cells)}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "provider": "Open-Meteo Historical Weather API, models=era5",
                "url": API, "cell_count": len(cells), "batches": []}
    for start in range(0, len(cells), 10):
        batch = cells[start:start + 10]
        path = OUT / f"batch_{start // 10:02d}.json"
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")
        params = {
            "latitude": ",".join(f"{lat:.2f}" for lat, _ in batch),
            "longitude": ",".join(f"{lon:.2f}" for _, lon in batch),
            "elevation": ",".join("nan" for _ in batch),
            "start_date": "2022-08-01", "end_date": "2022-12-31",
            "hourly": "cloud_cover,precipitation,snowfall", "models": "era5",
            "timezone": "UTC", "cell_selection": "nearest",
        }
        response = requests.get(API, params=params, timeout=120)
        response.raise_for_status()
        body = response.json()
        results = body if isinstance(body, list) else [body]
        if len(results) != len(batch):
            raise ValueError(f"Batch {start // 10}: count mismatch {len(results)}")
        for (lat, lon), record in zip(batch, results):
            if record.get("error"):
                raise ValueError(f"Provider error at {lat},{lon}: {record.get('reason')}")
            hourly = record.get("hourly", {})
            if not all(len(hourly.get(name, [])) == 3672 for name in
                       ("time", "cloud_cover", "precipitation", "snowfall")):
                raise ValueError(f"Incomplete ERA5 proxy at {lat},{lon}")
            units = record.get("hourly_units", {})
            if (units.get("cloud_cover"), units.get("precipitation"),
                    units.get("snowfall")) != ("%", "mm", "cm"):
                raise ValueError(f"Unexpected units at {lat},{lon}: {units}")
        payload = {"request": {"coordinates": batch, "params": params},
                   "response": results}
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        manifest["batches"].append({"file": path.name, "coordinates": batch,
                                    "bytes": path.stat().st_size})
        print(f"Saved {path.name}: {len(batch)} grid cells, {path.stat().st_size} bytes",
              flush=True)
    path = OUT / "request_manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
