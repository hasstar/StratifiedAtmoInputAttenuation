"""Retrieve bounded ERA5 fields for OpenSat clear-sky candidate screening.

The CDS credential is read by cdsapi from the user's existing .cdsapirc; this
script never prints, copies, or serializes the token. Original source data is
never modified. Run a one-day smoke test before requesting all months.
"""

from __future__ import annotations

import argparse
import sys
from calendar import monthrange
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
VENDOR = PROJECT / "Code/Dataset_Build/_vendor_cdsapi"
OUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_004/era5_france"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, required=True, choices=range(8, 13))
    parser.add_argument("--day", type=int, help="One-day smoke test; omit for a full month")
    args = parser.parse_args()
    sys.path.insert(0, str(VENDOR))
    import cdsapi  # type: ignore[import-not-found]

    suffix = f"{args.month:02d}{args.day:02d}" if args.day else f"{args.month:02d}"
    target = OUT / f"era5_france_2022{suffix}.zip"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    OUT.mkdir(parents=True, exist_ok=True)
    last_day = monthrange(2022, args.month)[1]
    if args.day and not 1 <= args.day <= last_day:
        parser.error(f"day must be between 1 and {last_day}")
    days = [f"{args.day:02d}"] if args.day else [f"{d:02d}" for d in range(1, last_day + 1)]
    request = {
        "product_type": ["reanalysis"],
        "variable": ["total_cloud_cover", "total_precipitation", "snowfall"],
        "year": ["2022"],
        "month": [f"{args.month:02d}"],
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [44.5, 2.5, 42.75, 7.75],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print(f"Requesting {target.name}; area={request['area']}; days={len(days)}")
    cdsapi.Client(quiet=True, progress=False).retrieve(
        "reanalysis-era5-single-levels", request, str(target)
    )
    print(f"Saved {target.name}, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
