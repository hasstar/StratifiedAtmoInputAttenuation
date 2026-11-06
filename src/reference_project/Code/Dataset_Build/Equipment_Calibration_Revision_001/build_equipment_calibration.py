"""Versioned, retrospective receiver-gain correction for all local power data.

The gain is fitted from ALL ERA5 clear_dry candidate hours on either side of
the known 2026-05-14 equipment change, without using ML split membership.
Every post-change measured W value is multiplied by that single gain. This is
equivalent to adding a single dB offset and preserves deep-fade differences.
Original H5, Run_011, ERA5 and earlier dataset versions are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[3]
DATASETS = PROJECT / "Outputs/Datasets"
SOURCE_H = DATASETS / "Revision_005"
SOURCE_W = DATASETS / "Revision_002/measurement_hourly.csv"
SOURCE_F = DATASETS / "Fast_Variation_Revision_001"
DEST_H = DATASETS / "Revision_006"
DEST_F = DATASETS / "Fast_Variation_Revision_002"
BOUNDARY_BJT = pd.Timestamp("2026-05-14 00:00:00", tz="Asia/Shanghai")
BOUNDARY_NAIVE = BOUNDARY_BJT.tz_localize(None)
MODEL_PREFIX = "model_table_primary_p10"
SCOPES = ("all_measurements", "high_coverage_95")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamped() -> str:
    return datetime.now(timezone.utc).isoformat()


def calibration() -> dict[str, object]:
    source_report = json.loads((SOURCE_H / "validation_report.json").read_text(encoding="utf-8"))
    require(source_report.get("status") == "VERIFIED", "Revision_005 not verified")
    labels = pd.read_csv(SOURCE_H / "hourly_event_labels.csv", usecols=["hour_bjt", "clear_dry"])
    power = pd.read_csv(SOURCE_W, usecols=["hour_bjt", "received_power_w"])
    labels["hour_bjt"] = pd.to_datetime(labels["hour_bjt"])
    power["hour_bjt"] = pd.to_datetime(power["hour_bjt"])
    joined = labels.merge(power, on="hour_bjt", how="inner", validate="one_to_one")
    require(len(joined) == len(labels) == 3869, "Calibration hour linkage changed")
    require(joined["received_power_w"].notna().all(), "Calibration has missing W")
    clear = joined.loc[joined["clear_dry"]]
    before = clear.loc[clear["hour_bjt"] < BOUNDARY_BJT, "received_power_w"]
    after = clear.loc[clear["hour_bjt"] >= BOUNDARY_BJT, "received_power_w"]
    require(len(before) == 1432 and len(after) == 269,
            "Clear candidate counts changed; investigate before calibration")
    a1 = float(before.mean())
    a2 = float(after.mean())
    require(np.isfinite(a1) and np.isfinite(a2) and a1 > a2 > 0,
            "Unexpected clear candidate power means")
    gain = a1 / a2
    shift = 10.0 * math.log10(gain)
    return {
        "calibration_kind": "retrospective_equipment_gain",
        "selection": "all existing clear_dry candidate hours, independent of ML split",
        "hour_weighting": "each hourly W-domain mean has equal weight",
        "boundary_bjt": BOUNDARY_BJT.isoformat(),
        "before_clear_hours": len(before),
        "after_clear_hours": len(after),
        "a1_before_mean_w": a1,
        "a2_after_mean_w": a2,
        "linear_gain": gain,
        "db_shift": shift,
        "formula_w": "P_corrected_W=P_original_W*(a1/a2) for time >= boundary",
        "formula_dbm": "P_corrected_dBm=P_original_dBm+10*log10(a1/a2) for time >= boundary",
        "input_sha256": {
            "clear_dry_event_labels": sha256(SOURCE_H / "hourly_event_labels.csv"),
            "hourly_power_w": sha256(SOURCE_W),
            "source_model_all": sha256(SOURCE_H / f"{MODEL_PREFIX}_all_measurements.parquet"),
            "source_fast_manifest": sha256(SOURCE_F / "run_manifest.json"),
            "builder": sha256(Path(__file__)),
        },
    }


def correct_hourly_table(scope: str, shift: float) -> pd.DataFrame:
    path = SOURCE_H / f"{MODEL_PREFIX}_{scope}.parquet"
    source = pd.read_parquet(path)
    require("received_power_dbm" in source.columns and "hour_bjt" in source.columns,
            f"Missing power/time in {path}")
    mask = source["hour_bjt"] >= BOUNDARY_BJT
    source.loc[mask, "received_power_dbm"] = (
        source.loc[mask, "received_power_dbm"].to_numpy(dtype=np.float64) + shift
    )
    require(source["received_power_dbm"].notna().all(), f"Null corrected hourly power: {scope}")
    stem = f"{MODEL_PREFIX}_{scope}"
    source.to_parquet(DEST_H / f"{stem}.parquet", index=False)
    source.to_csv(DEST_H / f"{stem}.csv", index=False)
    return source


def correct_event_sidecars(all_hours: pd.DataFrame) -> dict[str, int]:
    events = pd.read_csv(SOURCE_H / "hourly_event_labels.csv")
    columns = list(events.columns)
    times = pd.to_datetime(events["hour_bjt"])
    require(times.is_unique and times.is_monotonic_increasing, "Event times are not unique/in order")
    lookup = all_hours.set_index("hour_bjt")["received_power_dbm"]
    values = times.map(lookup)
    require(values.notna().all(), "Event hours do not match calibrated model table")
    old_low = events["low_power_le_minus85_dbm"].astype(bool)
    low = values <= -85.0
    starts = low & (~low.shift(fill_value=False) | times.diff().ne(pd.Timedelta(hours=1)))
    events["low_power_le_minus85_dbm"] = low.to_numpy(dtype=bool)
    events["low_power_le_minus85_dbm_run_id"] = starts.cumsum().where(low).astype("Int64")
    require(list(events.columns) == columns, "Event sidecar schema changed")
    events.to_csv(DEST_H / "hourly_event_labels.csv", index=False)

    counts = pd.read_csv(SOURCE_H / "event_counts.csv")
    for idx, row in counts.loc[counts["event"] == "low_power_le_minus85_dbm"].iterrows():
        scoped = events if row["scope"] == "all_measurements" else events.loc[
            events["measurement_high_coverage_95"]]
        split_col = (
            "split_strict_24h" if row["scheme"] == "original_chronological"
            else "split_event_chrono_24h"
        )
        group = scoped.loc[scoped[split_col] == row["split"]]
        hit = group.loc[group["low_power_le_minus85_dbm"]]
        require(len(group) == int(row["total_split_hours"]), "Split hours changed")
        counts.loc[idx, "hours"] = len(hit)
        counts.loc[idx, "calendar_days"] = times.loc[hit.index].dt.normalize().nunique()
        counts.loc[idx, "contiguous_event_runs"] = hit[
            "low_power_le_minus85_dbm_run_id"].nunique()
    counts.to_csv(DEST_H / "event_counts.csv", index=False)
    return {
        "low_power_hours_before": int(old_low.sum()),
        "low_power_hours_after": int(low.sum()),
    }


def build_hourly(info: dict[str, object]) -> dict[str, int]:
    changed = {
        "README.md", "run_manifest.json", "checksums.sha256",
        "validation_report.json", "VALIDATION.md", "hourly_event_labels.csv",
        "event_counts.csv",
        *(f"{MODEL_PREFIX}_{scope}.{extension}" for scope in SCOPES
          for extension in ("csv", "parquet")),
    }
    copied = 0
    for source in SOURCE_H.iterdir():
        if source.is_file() and source.name not in changed:
            shutil.copy2(source, DEST_H / source.name)
            copied += 1
    all_hours = correct_hourly_table("all_measurements", float(info["db_shift"]))
    high_hours = correct_hourly_table("high_coverage_95", float(info["db_shift"]))
    joined = high_hours[["hour_bjt", "received_power_dbm"]].merge(
        all_hours[["hour_bjt", "received_power_dbm"]],
        on="hour_bjt", how="left", validate="one_to_one", suffixes=("_high", "_all")
    )
    require(np.array_equal(joined["received_power_dbm_high"].to_numpy(),
                           joined["received_power_dbm_all"].to_numpy()),
            "High-coverage and all-hour targets disagree")
    event_stats = correct_event_sidecars(all_hours)
    return {
        "all_hours": len(all_hours),
        "high_coverage_hours": len(high_hours),
        "corrected_all_hours": int((all_hours["hour_bjt"] >= BOUNDARY_BJT).sum()),
        "copied_static_sidecars": copied,
        **event_stats,
    }


def write_power_parquet(source_path: Path, dest_path: Path, gain: float,
                        shift: float, watts: tuple[str, ...],
                        dbm: tuple[str, ...], time_field: str) -> int:
    native = pq.ParquetFile(source_path)
    rows = 0
    with pq.ParquetWriter(dest_path, native.schema_arrow, compression="zstd") as writer:
        for row_group in range(native.num_row_groups):
            frame = native.read_row_group(row_group).to_pandas()
            require((frame[time_field] >= BOUNDARY_NAIVE).all(),
                    f"Pre-change fast record unexpectedly in post-change file: {source_path}")
            for column in watts:
                frame[column] = frame[column].to_numpy(dtype=np.float64) * gain
            for column in dbm:
                frame[column] = (
                    frame[column].to_numpy(dtype=np.float64) + shift
                ).astype(np.float32)
            table = pa.Table.from_pandas(frame, schema=native.schema_arrow,
                                         preserve_index=False, safe=True)
            writer.write_table(table, row_group_size=len(frame))
            rows += len(frame)
    require(rows == native.metadata.num_rows, f"Fast row count changed: {source_path}")
    return rows


def correct_dictionary() -> None:
    source_path = SOURCE_F / "data_dictionary.csv"
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    assert columns is not None
    affected = {
        "peak_power_dbm", "peak_power_w", "slow_hourly_power_dbm",
        "slow_hourly_power_w", "received_power_dbm", "received_power_w",
    }
    for row in rows:
        if row["column"] in affected:
            row["description"] += (
                "; post-2026-05-14 values gain-calibrated by common factor "
                "from all clear_dry candidate hours"
            )
    with (DEST_F / "data_dictionary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_fast(info: dict[str, object]) -> dict[str, int]:
    gain = float(info["linear_gain"])
    shift = float(info["db_shift"])
    counts: dict[str, int] = {}
    for layer, watts, dbm, time_field in (
        ("fast_100ms", ("peak_power_w", "slow_hourly_power_w"),
         ("peak_power_dbm", "slow_hourly_power_dbm"), "time_bjt"),
        ("fast_1s", ("received_power_w", "slow_hourly_power_w"),
         ("received_power_dbm", "slow_hourly_power_dbm"), "second_bjt"),
    ):
        dest_dir = DEST_F / layer
        dest_dir.mkdir()
        pre = post = 0
        rows_post = 0
        for source_path in sorted((SOURCE_F / layer).glob("*.parquet")):
            dest_path = dest_dir / source_path.name
            if source_path.stem[-7:] < "2026-05":
                shutil.copy2(source_path, dest_path)
                pre += 1
            else:
                rows_post += write_power_parquet(
                    source_path, dest_path, gain, shift, watts, dbm, time_field
                )
                post += 1
        require(pre == 6 and post == 4, f"Unexpected {layer} month partition count")
        counts[f"{layer}_pre_files_unchanged"] = pre
        counts[f"{layer}_post_files_corrected"] = post
        counts[f"{layer}_post_rows_corrected"] = rows_post

    source_path = SOURCE_F / "fast_window_statistics.parquet"
    windows = pd.read_parquet(source_path)
    mask = windows["window_start_bjt"] >= BOUNDARY_NAIVE
    windows.loc[mask, "received_power_w"] *= gain
    windows.loc[mask, "received_power_dbm"] += shift
    windows.to_parquet(DEST_F / source_path.name, index=False)
    counts["window_rows"] = len(windows)
    counts["window_rows_corrected"] = int(mask.sum())
    shutil.copy2(SOURCE_F / "monthly_summary.csv", DEST_F / "monthly_summary.csv")
    shutil.copy2(SOURCE_F / "time_split_codes.csv", DEST_F / "time_split_codes.csv")
    correct_dictionary()
    return counts


def write_readmes(info: dict[str, object], hourly: dict[str, int],
                  fast: dict[str, int]) -> None:
    shift = float(info["db_shift"])
    gain = float(info["linear_gain"])
    (DEST_H / "README.md").write_text(f"""# Revision_006：设备更换后的本地小时功率校准

用户确认 2026-05-14 更换本地接收设备。本版从已验证的 Revision_005 派生，
保留所有表名、列、10 个气压面、348 个气象特征与原事件时间划分。
原始 H5、ERA5、Run_011 和既有 Revision 均未修改。

晴空候选标志沿用已冻结的 `clear_dry`，不按训练/验证/测试划分。
分界前 {info['before_clear_hours']} 小时、分界后 {info['after_clear_hours']} 小时；
两段小时 W 域均值分别为 {info['a1_before_mean_w']:.17g} W、
{info['a2_after_mean_w']:.17g} W。每个晴空候选小时等权。

设备增益因子 `g=a1/a2={gain:.15g}`；后期每个有效功率在 W 域乘以 `g`，
等价于每个 dBm 加 `{shift:.12f} dB`。**不在 W 域加固定常数。**
2026-05-14 当日无实测，首批修正记录从 2026-05-15 开始。
校准覆盖后期所有 {hourly['corrected_all_hours']} 个有测量的小时，
不只晴空小时，也不区分训练/验证/测试。更换前逐值不变。

`model_table_primary_p10_all_measurements` 与 `...high_coverage_95` 仍有
{hourly['all_hours']} / {hourly['high_coverage_hours']} 行；只改变接收功率目标。
天气事件、质量、时间划分及特征清单不变。原 `low_power_le_minus85_dbm`
是目标派生的绝对门限代理，已随校准重算，不能继续使用旧版计数。

成套快变版本：`../Fast_Variation_Revision_002/`。校准系数、来源哈希见
`equipment_calibration.json`；独立核验见 `validation_report.json`。
本版仍是接收功率 dBm，不是相对晴空衰减 dB；后续的衰减目标需另行构建。
""", encoding="utf-8")
    (DEST_F / "README.md").write_text(f"""# Fast Variation Revision_002：设备更换后的快变功率校准

与小时 `Revision_006` 使用同一增益因子 `g={gain:.15g}`
（dBm 统一加 `{shift:.12f} dB`）。2026-05-14 之前 100 ms / 1 s 月文件
逐字节沿用已验证的 Revision_001；之后四个月文件的**所有**测量功率和
对应小时慢变功率均按同一因子缩放。窗口平均功率同样缩放。

行数、列名、类型、时间、质量标志、划分代码保持不变。由于快变记录与
本小时慢变功率同乘 `g`，`fast_residual_db`、功率比、归一化包络及其
窗口统计不变；绝对功率字段变更。100 ms 后期修正 {fast['fast_100ms_post_rows_corrected']:,} 行，
1 s 后期修正 {fast['fast_1s_post_rows_corrected']:,} 行。

原始 H5、Run_011 和 Fast_Variation_Revision_001 均未改动。
校准说明见小时 `../Revision_006/README.md`；独立 QA 见 `validation_report.json`。
""", encoding="utf-8")


def checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and
                   path.name not in {"checksums.sha256", "validation_report.json", "VALIDATION.md"})
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require(not DEST_H.exists() and not DEST_F.exists(),
            "A destination already exists; never overwrite a dataset version")
    info = calibration()
    DEST_H.mkdir(parents=True)
    DEST_F.mkdir(parents=True)
    (DEST_H / "equipment_calibration.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hourly = build_hourly(info)
    print("Hourly complete", json.dumps(hourly, ensure_ascii=False), flush=True)
    fast = build_fast(info)
    print("Fast complete", json.dumps(fast, ensure_ascii=False), flush=True)
    write_readmes(info, hourly, fast)
    for root, counts in ((DEST_H, hourly), (DEST_F, fast)):
        manifest = {
            "created_utc": timestamped(),
            "status": "BUILT_AWAITING_INDEPENDENT_VALIDATION",
            "source_version": "Revision_005" if root == DEST_H else "Fast_Variation_Revision_001",
            "calibrated_hourly_version": "Revision_006",
            "gain_factor": info["linear_gain"],
            "db_shift": info["db_shift"],
            "counts": counts,
            "environment": {
                "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "pyarrow": pa.__version__,
            },
        }
        (root / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksums(root)
    print("Built", DEST_H, DEST_F, flush=True)


if __name__ == "__main__":
    main()
