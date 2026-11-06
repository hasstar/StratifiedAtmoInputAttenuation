"""Refresh only the local power in the three-source attenuation staging package."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
DATASETS = PROJECT / "Outputs/Datasets"
SOURCE = DATASETS / "ClearSky_Alignment_Revision_002"
CALIBRATED = DATASETS / "Revision_006"
DEST = DATASETS / "ClearSky_Alignment_Revision_003"
BOUNDARY = pd.Timestamp("2026-05-14", tz="Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Never overwrite an existing version: {DEST}")
    report = json.loads((CALIBRATED / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "VERIFIED":
        raise ValueError("Calibrated hourly dataset not independently verified")
    old = pd.read_parquet(SOURCE / "local_hourly_power_audit.parquet")
    new = pd.read_parquet(
        CALIBRATED / "model_table_primary_p10_all_measurements.parquet",
        columns=["hour_utc", "received_power_dbm"],
    )
    if not old["hour_utc"].is_unique or not new["hour_utc"].is_unique:
        raise ValueError("Duplicate local hour")
    if len(old) != len(new) or len(old) != 3869:
        raise ValueError("Local row count changed")
    lookup = new.set_index("hour_utc")["received_power_dbm"]
    corrected = old["hour_utc"].map(lookup)
    if corrected.isna().any():
        raise ValueError("Missing calibrated local target")
    mask = pd.to_datetime(old["hour_bjt"]) >= BOUNDARY
    if not np.array_equal(old.loc[~mask, "observed_power_dbm"].to_numpy(),
                          corrected.loc[~mask].to_numpy()):
        raise ValueError("Pre-change local power unexpectedly differs")
    shift = float(report["db_shift"])
    if not np.allclose(corrected.loc[mask], old.loc[mask, "observed_power_dbm"] + shift,
                       rtol=0, atol=1e-10):
        raise ValueError("Post-change local power shift differs")
    old["observed_power_dbm"] = corrected.to_numpy()
    old["dataset"] = "local_Run011_Revision006"
    if not old["clear_sky_attenuation_db"].isna().all():
        raise ValueError("An unapproved attenuation label exists")

    rows = []
    for month, group in old.groupby("month_bjt", sort=True):
        clear = group.loc[group["clear_dry"], "observed_power_dbm"]
        screened = group.loc[
            group["clear_dry"] & group["measurement_high_coverage_95"]
            & (group["quality_zero_fraction"] >= 0.95), "observed_power_dbm"
        ]
        rows.append({
            "month_bjt": month,
            "measured_hours": len(group),
            "clear_dry_hours": len(clear),
            "screened_clear_dry_hours": len(screened),
            "clear_power_p10_dbm": clear.quantile(0.1) if len(clear) else np.nan,
            "clear_power_p50_dbm": clear.median() if len(clear) else np.nan,
            "clear_power_p90_dbm": clear.quantile(0.9) if len(clear) else np.nan,
            "screened_clear_power_p50_dbm": screened.median() if len(screened) else np.nan,
        })

    DEST.mkdir(parents=True)
    for name in ("opensat_hourly_power.parquet", "ceda_gbs_hourly_attenuation.parquet"):
        shutil.copy2(SOURCE / name, DEST / name)
    old.to_parquet(DEST / "local_hourly_power_audit.parquet", index=False)
    pd.DataFrame(rows).to_csv(DEST / "local_clear_candidate_monthly.csv", index=False)
    registry = pd.read_csv(SOURCE / "source_registry.csv")
    local = registry["dataset"] == "local_Run011_Revision005"
    if int(local.sum()) != 1:
        raise ValueError("Source registry local entry changed")
    registry.loc[local, "dataset"] = "local_Run011_Revision006"
    registry.loc[local, "reference_status"] = (
        "equipment gain calibrated using all clear_dry candidate hours; "
        "clear-sky attenuation baseline still not established"
    )
    registry.to_csv(DEST / "source_registry.csv", index=False)
    (DEST / "README.md").write_text(f"""# 三源相对晴空衰减指标整理（Revision 003）

本版从已独立验证的 `ClearSky_Alignment_Revision_002` 派生，**仅**将本地
小时接收功率更新为设备增益校准后的 `Revision_006` 值；本地 3,869 个小时、
原有质量/天气/时间划分及全部列保持不变。2026-05-14 之后的 1,365 个
本地小时 dBm 统一加 {shift:.12f} dB；更换前不变。

OpenSat4Weather 与 CEDA GBS 两张小时表和原版逐字节相同。
CEDA 仍有 46,905 个小时的原生相对晴空衰减 dB；本地与 OpenSat 的
`clear_sky_attenuation_db` 仍全部为空，因为设备增益校准**不等于**
建立无雨无云晴空衰减基线，不能把 dBm 直接改名为 dB。

`local_clear_candidate_monthly.csv` 已按校准后的本地功率重新计算，
可用于检查单一设备增益校准后剩余的月度变化。源与输出哈希见
`manifest.json`，设备系数见 `../Revision_006/equipment_calibration.json`。
外部法国/英国 ERA5 尚未下载或匹配；原始 H5、Run_011、ERA5、
此前三个数据集版本全部保持不变。
""", encoding="utf-8")
    source_hashes = {
        "previous_alignment_manifest": sha256(SOURCE / "manifest.json"),
        "calibration_report": sha256(CALIBRATED / "validation_report.json"),
        "calibrated_hourly_model": sha256(
            CALIBRATED / "model_table_primary_p10_all_measurements.parquet"
        ),
        "builder": sha256(Path(__file__)),
    }
    output_hashes = {
        path.name: sha256(path) for path in sorted(DEST.iterdir()) if path.is_file()
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_STAGING_ONLY_NOT_THREE_SOURCE_ATTENUATION",
        "local_equipment_db_shift": shift,
        "local_hours": len(old),
        "local_corrected_hours": int(mask.sum()),
        "local_attenuation_hours": 0,
        "opensat_attenuation_hours": 0,
        "ceda_attenuation_hours": 46905,
        "source_sha256": source_hashes,
        "outputs_sha256": output_hashes,
    }
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in (
        "status", "local_hours", "local_corrected_hours",
        "local_attenuation_hours", "opensat_attenuation_hours",
        "ceda_attenuation_hours")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
