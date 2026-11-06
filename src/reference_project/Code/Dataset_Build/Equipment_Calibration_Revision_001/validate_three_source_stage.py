"""Independent checks for the post-calibration three-source staging revision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "Outputs/Datasets"
OLD = DATA / "ClearSky_Alignment_Revision_002"
NEW = DATA / "ClearSky_Alignment_Revision_003"
CAL = DATA / "Revision_006"
BOUNDARY = pd.Timestamp("2026-05-14", tz="Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((NEW / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["outputs_sha256"].items():
        assert sha256(NEW / name) == expected, name
    for name in ("opensat_hourly_power.parquet", "ceda_gbs_hourly_attenuation.parquet"):
        assert sha256(NEW / name) == sha256(OLD / name), name
    old = pd.read_parquet(OLD / "local_hourly_power_audit.parquet")
    new = pd.read_parquet(NEW / "local_hourly_power_audit.parquet")
    target = pd.read_parquet(
        CAL / "model_table_primary_p10_all_measurements.parquet",
        columns=["hour_utc", "received_power_dbm"],
    )
    assert len(old) == len(new) == len(target) == 3869
    assert list(old.columns) == list(new.columns)
    assert old.drop(columns=["dataset", "observed_power_dbm"]).equals(
        new.drop(columns=["dataset", "observed_power_dbm"]))
    assert new["clear_sky_attenuation_db"].isna().all()
    assert new["dataset"].eq("local_Run011_Revision006").all()
    matched = new.merge(target, on="hour_utc", validate="one_to_one")
    assert np.array_equal(matched["observed_power_dbm"].to_numpy(),
                          matched["received_power_dbm"].to_numpy())
    mask = pd.to_datetime(new["hour_bjt"]) >= BOUNDARY
    assert int(mask.sum()) == 1365
    assert np.array_equal(new.loc[~mask, "observed_power_dbm"].to_numpy(),
                          old.loc[~mask, "observed_power_dbm"].to_numpy())
    shift = float(manifest["local_equipment_db_shift"])
    assert np.allclose(new.loc[mask, "observed_power_dbm"],
                       old.loc[mask, "observed_power_dbm"] + shift,
                       atol=1e-12, rtol=0)
    monthly = pd.read_csv(NEW / "local_clear_candidate_monthly.csv")
    for row in monthly.itertuples():
        group = new.loc[new["month_bjt"] == row.month_bjt]
        clear = group.loc[group["clear_dry"], "observed_power_dbm"]
        assert len(group) == row.measured_hours
        assert len(clear) == row.clear_dry_hours
        if len(clear):
            assert abs(clear.median() - row.clear_power_p50_dbm) < 1e-10
    report = {
        "status": "VERIFIED_STAGING_ONLY",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "local_hours": len(new),
        "local_corrected_hours": int(mask.sum()),
        "local_attenuation_hours": int(new["clear_sky_attenuation_db"].notna().sum()),
        "external_tables_byte_identical": 2,
        "monthly_rows_rechecked": len(monthly),
        "output_hashes_verified": len(manifest["outputs_sha256"]),
    }
    (NEW / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
