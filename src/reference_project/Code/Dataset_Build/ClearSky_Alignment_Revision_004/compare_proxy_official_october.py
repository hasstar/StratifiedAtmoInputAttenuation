"""Sensitivity audit against the available official CDS October 2022 month.

Compares weather at the same ERA5 grid cells and valid timestamps. This does
not retroactively change the frozen Revision_004 target or its manifest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_attenuation import OUT, PROXY, read_zipped_era5


def main() -> None:
    source = OUT / "era5_france/era5_france_202210.zip"
    if not source.is_file():
        raise FileNotFoundError(source)
    t_tcc, lat, lon, cloud = read_zipped_era5(source, "tcc")
    t_tp, lat_p, lon_p, rain = read_zipped_era5(source, "tp")
    t_sf, lat_s, lon_s, snow = read_zipped_era5(source, "sf")
    if not (np.array_equal(t_tcc, t_tp) and np.array_equal(t_tcc, t_sf)
            and np.array_equal(lat, lat_p) and np.array_equal(lat, lat_s)
            and np.array_equal(lon, lon_p) and np.array_equal(lon, lon_s)):
        raise ValueError("Official October field grids are not identical")
    official_hours = pd.to_datetime(t_tcc, unit="s", utc=True)
    if len(official_hours) != 744 or official_hours[0].strftime("%Y-%m-%d") != "2022-10-01":
        raise ValueError("Official October time grid mismatch")
    cloud_errors: list[np.ndarray] = []
    rain_errors: list[np.ndarray] = []
    cloud_disagreements = rain_disagreements = snow_disagreements = joint_disagreements = 0
    cells = 0
    for batch in range(6):
        payload = json.loads((PROXY / f"batch_{batch:02d}.json").read_text(encoding="utf-8"))
        for key, response in zip(payload["request"]["coordinates"], payload["response"]):
            i = int(np.argmin(np.abs(lat - key[0])))
            j = int(np.argmin(np.abs(lon - key[1])))
            if abs(lat[i] - key[0]) > 1e-8 or abs(lon[j] - key[1]) > 1e-8:
                raise ValueError(f"Missing official grid cell: {key}")
            proxy_hours = pd.to_datetime(response["hourly"]["time"], utc=True)
            positions = proxy_hours.get_indexer(official_hours)
            if np.any(positions < 0):
                raise ValueError(f"Missing proxy timestamps: {key}")
            p_cloud = np.asarray(response["hourly"]["cloud_cover"], dtype=float)[positions]
            p_rain = np.asarray(response["hourly"]["precipitation"], dtype=float)[positions]
            p_snow = np.asarray(response["hourly"]["snowfall"], dtype=float)[positions]
            o_cloud = cloud[:, i, j] * 100.0
            o_rain = rain[:, i, j] * 1000.0
            o_snow = snow[:, i, j] * 1000.0
            cloud_errors.append(np.abs(p_cloud - o_cloud))
            rain_errors.append(np.abs(p_rain - o_rain))
            c_proxy = p_cloud <= 20.0
            c_official = o_cloud <= 20.0
            r_proxy = p_rain < .1
            r_official = o_rain < .1
            s_proxy = p_snow == 0.0
            s_official = o_snow < .1
            cloud_disagreements += int(np.count_nonzero(c_proxy != c_official))
            rain_disagreements += int(np.count_nonzero(r_proxy != r_official))
            snow_disagreements += int(np.count_nonzero(s_proxy != s_official))
            joint_disagreements += int(np.count_nonzero(
                (c_proxy & r_proxy & s_proxy) != (c_official & r_official & s_official)))
            cells += 1
    a_cloud = np.concatenate(cloud_errors)
    a_rain = np.concatenate(rain_errors)
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": str(source),
        "scope": "2022-10-01 through 2022-10-31; 55 grid cells",
        "compared_cell_hours": int(len(a_cloud)), "grid_cells": cells,
        "max_abs_cloud_percentage_points": float(a_cloud.max()),
        "max_abs_precip_mm": float(a_rain.max()),
        "cloud_threshold_disagreements": cloud_disagreements,
        "precip_threshold_disagreements": rain_disagreements,
        "snow_zero_disagreements": snow_disagreements,
        "joint_clear_weather_disagreements": joint_disagreements,
        "caveat": "Proxy rounding can alter candidate membership; radar, coverage, and receiver state are not assessed in this grid-only comparison.",
    }
    target = OUT / "weather_proxy_comparison_202210.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
