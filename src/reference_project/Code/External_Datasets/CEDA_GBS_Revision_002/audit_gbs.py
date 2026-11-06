"""Independent integrity and NetCDF-content audit for the CEDA GBS download.

The input manifest and NetCDF files are read-only. Outputs are created once in a
new QA_Revision_001 directory; an existing QA directory is never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "外部样例数据CEDA" / "GBS_20p7GHz"
MANIFEST = DATA / "source_manifest.csv"
QA_DIR = DATA / "QA_Revision_002"
REQUIRED = (
    "time",
    "clearsky_attenuation",
    "frequency",
    "latitude_receiver",
    "longitude_receiver",
    "receiver_altitude_amsl",
    "satellite_position",
    "elevation_angle",
)
FIELDS = (
    "site", "date_utc", "relative_path", "status", "expected_bytes", "actual_bytes",
    "md5_match", "record_count", "declared_valid_time_count", "time_min_s", "time_max_s",
    "raw_declared_time_strictly_increasing", "time_zero_count", "time_zero_with_signal_count",
    "signal_time_count", "signal_time_unique_count", "signal_time_min_s", "signal_time_max_s",
    "signal_time_strictly_increasing", "attenuation_fill_count", "attenuation_nonfinite_count",
    "attenuation_nonfill_count", "attenuation_below_valid_min", "attenuation_above_valid_max",
    "attenuation_ge_3db_count", "attenuation_ge_10db_count", "attenuation_p50_db",
    "attenuation_p95_db", "attenuation_p99_db", "frequency_ghz", "latitude_deg_north",
    "longitude_raw", "longitude_units", "longitude_deg_east_normalized",
    "receiver_altitude_m", "satellite_position_raw", "satellite_position_units",
    "satellite_longitude_deg_east_normalized", "elevation_angle_deg", "time_units", "error",
)


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def scalar(variable: object) -> float:
    return float(np.asarray(variable.data).ravel()[0])


def normalized_east(value: float, units: str) -> float:
    if units == "degrees_east":
        east = value
    elif units == "degrees_west":
        east = -value
    else:
        raise ValueError(f"Unexpected longitude direction: {units}")
    return float(((east + 180.0) % 360.0) - 180.0)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_one(manifest_row: dict[str, str]) -> dict[str, object]:
    row: dict[str, object] = {name: "" for name in FIELDS}
    row.update({k: manifest_row[k] for k in ("site", "date_utc", "relative_path")})
    row["status"] = "error"
    row["expected_bytes"] = int(manifest_row["bytes"])
    try:
        path = (DATA / manifest_row["relative_path"]).resolve()
        if not path.is_relative_to(DATA.resolve()):
            raise ValueError("Manifest path escapes the GBS directory")
        if not path.is_file():
            row["status"] = "missing"
            row["error"] = "File absent"
            return row
        row["actual_bytes"] = path.stat().st_size
        row["md5_match"] = file_md5(path) == manifest_row["md5"].lower()
        if row["actual_bytes"] != row["expected_bytes"] or not row["md5_match"]:
            row["status"] = "integrity_mismatch"
            row["error"] = "Size or MD5 differs from CEDA inventory"
            return row

        with netcdf_file(path, mode="r", mmap=False) as dataset:
            absent = [name for name in REQUIRED if name not in dataset.variables]
            if absent:
                raise ValueError("Missing variables: " + ", ".join(absent))
            time_var = dataset.variables["time"]
            time_data = np.asarray(time_var.data, dtype=np.float64).ravel()
            time_fill = float(time_var._attributes.get("_FillValue", -1))
            good_time = np.isfinite(time_data) & (time_data != time_fill)
            observed_time = time_data[good_time]
            row["record_count"] = len(time_data)
            row["declared_valid_time_count"] = int(good_time.sum())
            row["time_min_s"] = float(observed_time.min()) if observed_time.size else ""
            row["time_max_s"] = float(observed_time.max()) if observed_time.size else ""
            row["raw_declared_time_strictly_increasing"] = bool(np.all(np.diff(observed_time) > 0))
            row["time_units"] = decode(time_var._attributes.get("units", ""))

            attenuation_var = dataset.variables["clearsky_attenuation"]
            attenuation = np.asarray(attenuation_var.data, dtype=np.float64).ravel()
            if len(attenuation) != len(time_data):
                raise ValueError("Time and attenuation vector lengths differ")
            attributes = attenuation_var._attributes
            fill = float(attributes["_FillValue"])
            nonfinite = ~np.isfinite(attenuation)
            fill_mask = attenuation == fill
            nonfill = ~(nonfinite | fill_mask)
            signal_time = time_data[nonfill & good_time]
            row["time_zero_count"] = int((time_data == 0).sum())
            row["time_zero_with_signal_count"] = int(((time_data == 0) & nonfill).sum())
            row["signal_time_count"] = len(signal_time)
            row["signal_time_unique_count"] = len(np.unique(signal_time))
            row["signal_time_min_s"] = float(signal_time.min()) if signal_time.size else ""
            row["signal_time_max_s"] = float(signal_time.max()) if signal_time.size else ""
            row["signal_time_strictly_increasing"] = bool(np.all(np.diff(signal_time) > 0))
            valid_min = float(attributes["valid_min"])
            valid_max = float(attributes["valid_max"])
            below = nonfill & (attenuation < valid_min)
            above = nonfill & (attenuation > valid_max)
            within = nonfill & ~below & ~above & good_time
            values = attenuation[within]
            row["attenuation_fill_count"] = int(fill_mask.sum())
            row["attenuation_nonfinite_count"] = int(nonfinite.sum())
            row["attenuation_nonfill_count"] = int(nonfill.sum())
            row["attenuation_below_valid_min"] = int(below.sum())
            row["attenuation_above_valid_max"] = int(above.sum())
            row["attenuation_ge_3db_count"] = int((values >= 3).sum())
            row["attenuation_ge_10db_count"] = int((values >= 10).sum())
            if values.size:
                p50, p95, p99 = np.percentile(values, (50, 95, 99))
                row["attenuation_p50_db"] = float(p50)
                row["attenuation_p95_db"] = float(p95)
                row["attenuation_p99_db"] = float(p99)

            row["frequency_ghz"] = scalar(dataset.variables["frequency"])
            row["latitude_deg_north"] = scalar(dataset.variables["latitude_receiver"])
            longitude = dataset.variables["longitude_receiver"]
            row["longitude_raw"] = scalar(longitude)
            row["longitude_units"] = decode(longitude._attributes.get("units", ""))
            row["longitude_deg_east_normalized"] = normalized_east(
                float(row["longitude_raw"]), str(row["longitude_units"])
            )
            row["receiver_altitude_m"] = scalar(dataset.variables["receiver_altitude_amsl"])
            satellite = dataset.variables["satellite_position"]
            row["satellite_position_raw"] = scalar(satellite)
            row["satellite_position_units"] = decode(satellite._attributes.get("units", ""))
            row["satellite_longitude_deg_east_normalized"] = normalized_east(
                float(row["satellite_position_raw"]), str(row["satellite_position_units"])
            )
            row["elevation_angle_deg"] = scalar(dataset.variables["elevation_angle"])
        row["status"] = "readable_verified"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    if not manifest:
        raise RuntimeError("Source manifest is empty")
    rows = [audit_one(item) for item in manifest]
    status_counts = Counter(str(row["status"]) for row in rows)
    by_site: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        site = str(row["site"])
        by_site[site]["files"] += 1
        if row["status"] == "readable_verified":
            by_site[site]["readable_files"] += 1
            for field in (
                "record_count", "declared_valid_time_count", "time_zero_count",
                "time_zero_with_signal_count", "signal_time_count", "attenuation_fill_count",
                "attenuation_nonfinite_count", "attenuation_nonfill_count",
                "attenuation_below_valid_min", "attenuation_above_valid_max",
                "attenuation_ge_3db_count", "attenuation_ge_10db_count",
            ):
                by_site[site][field] += int(row[field])
            if not row["raw_declared_time_strictly_increasing"]:
                by_site[site]["raw_time_nonmonotone_files"] += 1
            if not row["signal_time_strictly_increasing"]:
                by_site[site]["signal_time_nonmonotone_files"] += 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Integrity and NetCDF-content audit; not a model or climate-transfer validation",
        "manifest_files": len(manifest),
        "status_counts": dict(status_counts),
        "by_site": {site: dict(counts) for site, counts in sorted(by_site.items())},
        "limitations": [
            "clearsky_attenuation is relative-to-clear-sky dB, not absolute received dBm",
            "counts above 3/10 dB are attenuation thresholds, not weather-event labels",
            "longitude direction is normalized from each file's units attribute",
            "Dundee uses zero time values for many missing attenuation records; assess time order on paired non-fill signal rows",
            "out-of-range values are reported, not silently removed from the source files",
        ],
    }
    QA_DIR.mkdir(parents=True, exist_ok=False)
    with (QA_DIR / "qa_by_file.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (QA_DIR / "qa_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status_counts": dict(status_counts), "output": str(QA_DIR)}, ensure_ascii=False))
    if status_counts != {"readable_verified": len(manifest)}:
        raise RuntimeError("Some official files were missing, mismatched, or unreadable; inspect QA output")
    if any(row["status"] == "readable_verified" and not row["signal_time_strictly_increasing"] for row in rows):
        raise RuntimeError("Some signal-bearing time series are nonmonotone; inspect QA output")


if __name__ == "__main__":
    main()
