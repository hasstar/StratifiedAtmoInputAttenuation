"""Independent read-only checks for the three-source staging package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import netcdf_file


PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_002"
LOCAL = PROJECT / "Outputs/Datasets/Revision_005"
OPENSAT = PROJECT / "Outputs/Datasets/External_OpenSat4Weather_001/extracted/sml_data_2022.nc"
GBS = PROJECT / "外部样例数据CEDA/GBS_20p7GHz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_local(local: pd.DataFrame) -> dict[str, int]:
    original = pd.read_parquet(
        LOCAL / "model_table_primary_p10_all_measurements.parquet",
        columns=["hour_utc", "received_power_dbm", "split_event_chrono_24h"],
    )
    matched = local.merge(
        original, on="hour_utc", how="outer", validate="one_to_one", indicator=True
    )
    assert len(matched) == len(local) == 3869
    assert matched["_merge"].eq("both").all()
    assert np.array_equal(
        matched["observed_power_dbm"].to_numpy(),
        matched["received_power_dbm"].to_numpy(),
    )
    assert (matched["split_event_chrono_24h_x"] ==
            matched["split_event_chrono_24h_y"]).all()
    assert local["clear_sky_attenuation_db"].isna().all()
    assert local["target_status"].eq("baseline_not_established").all()
    return {"matched_hours": len(matched), "ready_attenuation_hours": 0}


def check_opensat(opensat: pd.DataFrame) -> dict[str, int]:
    assert len(opensat) == 215 * 3672
    assert not opensat.duplicated(["link_id", "hour_utc"]).any()
    assert opensat["clear_sky_attenuation_db"].isna().all()
    assert opensat["target_status"].eq("baseline_not_established").all()
    assert int(opensat["valid_minutes"].sum()) == 46944206
    assert opensat["valid_minutes"].between(0, 60).all()
    assert np.allclose(
        opensat["coverage_fraction"], opensat["valid_minutes"] / 60.0
    )
    assert opensat.loc[opensat["valid_minutes"].eq(0),
                       "observed_power_dbm"].isna().all()
    samples = 0
    with h5py.File(OPENSAT, "r") as native:
        ids = [x.decode("utf-8") for x in native["sml_id"][:]]
        for link_index in [0, 107, 214]:
            link_id = ids[link_index]
            subset = opensat.loc[
                (opensat["link_id"] == link_id) &
                (opensat["valid_minutes"] > 0)
            ]
            assert not subset.empty
            for row in [subset.iloc[0], subset.iloc[len(subset) // 2], subset.iloc[-1]]:
                hour_offset = int((row["hour_utc"] - pd.Timestamp(
                    "2022-08-01", tz="UTC")).total_seconds() // 3600)
                raw = native["rsl"][link_index, hour_offset * 60:(hour_offset + 1) * 60]
                valid = raw != -9999
                assert int(valid.sum()) == row["valid_minutes"]
                expected = 10.0 * np.log10(np.mean(
                    np.power(10.0, raw[valid].astype(np.float64) * 0.001 / 10.0)
                ))
                assert abs(expected - row["observed_power_dbm"]) < 1e-10
                samples += 1
    return {
        "link_hours": len(opensat),
        "valid_minute_total": int(opensat["valid_minutes"].sum()),
        "independent_recomputed_hours": samples,
        "ready_attenuation_hours": 0,
    }


def check_gbs(gbs: pd.DataFrame) -> dict[str, int]:
    assert len(gbs) == 1982 * 24
    assert not gbs.duplicated(["site", "hour_utc"]).any()
    assert 0 < int(gbs["valid_seconds"].sum()) < 168379274
    assert gbs["valid_seconds"].between(0, 3600).all()
    assert gbs["clear_sky_attenuation_db"].isna().eq(
        gbs["valid_seconds"].eq(0)).all()
    assert int(gbs["clear_sky_attenuation_db"].notna().sum()) == 46905
    assert np.allclose(gbs["coverage_fraction"], gbs["valid_seconds"] / 3600)
    samples = 0
    candidates = pd.concat([
        gbs.loc[gbs["clear_sky_attenuation_db"].notna()].groupby("site").head(1),
        gbs.nlargest(2, "clear_sky_attenuation_db"),
        gbs.nsmallest(2, "clear_sky_attenuation_db"),
    ]).drop_duplicates(["site", "hour_utc"])
    for row in candidates.itertuples():
        with netcdf_file(GBS / row.source_file, "r", mmap=False) as native:
            values = np.asarray(native.variables["clearsky_attenuation"].data,
                                dtype=np.float64)
            times = np.asarray(native.variables["time"].data, dtype=np.float64)
        hour = row.hour_utc.hour
        valid = (np.isfinite(values) & (values != -100.0) &
                 (values >= -5.0) & (values <= 100.0))
        if hour == 0:
            in_hour = (times > 0) & (times <= 3600)
        else:
            in_hour = (times > hour * 3600) & (times <= (hour + 1) * 3600)
        chosen = values[valid & in_hour]
        assert len(chosen) == row.valid_seconds
        expected = -10.0 * np.log10(np.mean(np.power(10.0, -chosen / 10.0)))
        assert abs(expected - row.clear_sky_attenuation_db) < 1e-10
        samples += 1
    return {
        "site_hours": len(gbs),
        "valid_seconds": int(gbs["valid_seconds"].sum()),
        "ready_attenuation_hours": int(gbs["clear_sky_attenuation_db"].notna().sum()),
        "negative_valid_attenuation_hours_preserved": int(
            (gbs["clear_sky_attenuation_db"] < 0).sum()),
        "independent_recomputed_hours": samples,
    }


def main() -> None:
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "STAGING_ONLY_NOT_A_THREE_DATASET_ATTENUATION_TARGET"
    assert manifest["counts"]["gbs"]["aggregated_valid_seconds"] == \
        manifest["counts"]["gbs"]["valid_native_seconds"] - \
        manifest["counts"]["gbs"]["excluded_midnight_boundary_seconds"]
    for name, expected in manifest["outputs_sha256"].items():
        assert sha256(OUTPUT / name) == expected, name
    local = pd.read_parquet(OUTPUT / "local_hourly_power_audit.parquet")
    opensat = pd.read_parquet(OUTPUT / "opensat_hourly_power.parquet")
    gbs = pd.read_parquet(OUTPUT / "ceda_gbs_hourly_attenuation.parquet")
    results = {
        "status": "VERIFIED_STAGING_ONLY",
        "output_hashes_verified": len(manifest["outputs_sha256"]),
        "local": check_local(local),
        "opensat": check_opensat(opensat),
        "ceda_gbs": check_gbs(gbs),
    }
    assert results["ceda_gbs"]["valid_seconds"] == \
        manifest["counts"]["gbs"]["aggregated_valid_seconds"]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
