"""Independent full-row validation of the versioned receiver-gain correction."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[3]
DATASETS = PROJECT / "Outputs/Datasets"
SOURCE_H = DATASETS / "Revision_005"
SOURCE_W = DATASETS / "Revision_002/measurement_hourly.csv"
SOURCE_F = DATASETS / "Fast_Variation_Revision_001"
DEST_H = DATASETS / "Revision_006"
DEST_F = DATASETS / "Fast_Variation_Revision_002"
BOUNDARY = pd.Timestamp("2026-05-14 00:00:00", tz="Asia/Shanghai")
BOUNDARY_NAIVE = BOUNDARY.tz_localize(None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def verify_checksums(root: Path) -> int:
    rows = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for row in rows:
        expected, rel = row.split("  ", 1)
        require(sha256(root / rel) == expected, f"Checksum mismatch: {root / rel}")
    return len(rows)


def recompute_gain() -> tuple[float, float, int, int]:
    events = pd.read_csv(SOURCE_H / "hourly_event_labels.csv",
                         usecols=["hour_bjt", "clear_dry"])
    power = pd.read_csv(SOURCE_W,
                        usecols=["hour_bjt", "received_power_w"])
    events["hour_bjt"] = pd.to_datetime(events["hour_bjt"])
    power["hour_bjt"] = pd.to_datetime(power["hour_bjt"])
    clear = events.merge(power, on="hour_bjt", validate="one_to_one")
    clear = clear.loc[clear["clear_dry"]]
    before = clear.loc[clear["hour_bjt"] < BOUNDARY, "received_power_w"]
    after = clear.loc[clear["hour_bjt"] >= BOUNDARY, "received_power_w"]
    return float(before.mean()), float(after.mean()), len(before), len(after)


def validate_hourly(shift: float, gain: float) -> dict[str, int]:
    results: dict[str, int] = {}
    for scope, expected_rows in (("all_measurements", 3869),
                                 ("high_coverage_95", 3529)):
        stem = f"model_table_primary_p10_{scope}"
        old = pd.read_parquet(SOURCE_H / f"{stem}.parquet")
        new = pd.read_parquet(DEST_H / f"{stem}.parquet")
        require(list(old.columns) == list(new.columns), f"Hourly columns changed: {scope}")
        require(len(old) == len(new) == expected_rows, f"Hourly rows changed: {scope}")
        require(old.drop(columns="received_power_dbm").equals(
            new.drop(columns="received_power_dbm")),
            f"Non-power hourly field changed: {scope}")
        mask = old["hour_bjt"] >= BOUNDARY
        require(not old.loc[old["hour_bjt"].dt.date == BOUNDARY.date()].shape[0],
                "Unexpected measurement on equipment-change date")
        require(np.array_equal(old.loc[~mask, "received_power_dbm"].to_numpy(),
                               new.loc[~mask, "received_power_dbm"].to_numpy()),
                f"Pre-change hourly power altered: {scope}")
        require(np.allclose(new.loc[mask, "received_power_dbm"],
                            old.loc[mask, "received_power_dbm"] + shift,
                            rtol=0, atol=1e-12),
                f"Post-change hourly shift mismatch: {scope}")
        csv_frame = pd.read_csv(DEST_H / f"{stem}.csv", usecols=["hour_bjt", "received_power_dbm"])
        require(len(csv_frame) == len(new), f"CSV rows changed: {scope}")
        require(np.allclose(csv_frame["received_power_dbm"], new["received_power_dbm"],
                            rtol=0, atol=1e-12), f"CSV/Parquet target mismatch: {scope}")
        results[f"{scope}_hours"] = len(new)
        results[f"{scope}_corrected_hours"] = int(mask.sum())

    original_events = pd.read_csv(SOURCE_H / "hourly_event_labels.csv")
    corrected_events = pd.read_csv(DEST_H / "hourly_event_labels.csv")
    changed = ["low_power_le_minus85_dbm", "low_power_le_minus85_dbm_run_id"]
    require(list(original_events.columns) == list(corrected_events.columns),
            "Event columns changed")
    require(original_events.drop(columns=changed).equals(
        corrected_events.drop(columns=changed)), "Weather or split event changed")
    all_power = pd.read_parquet(
        DEST_H / "model_table_primary_p10_all_measurements.parquet",
        columns=["hour_bjt", "received_power_dbm"]
    )
    targets = pd.Series(all_power["received_power_dbm"].to_numpy(),
                        index=all_power["hour_bjt"])
    expected_low = pd.to_datetime(corrected_events["hour_bjt"]).map(targets) <= -85.0
    require(np.array_equal(expected_low.to_numpy(),
                           corrected_events["low_power_le_minus85_dbm"].to_numpy()),
            "Low-power event labels mismatch corrected target")
    counts_old = pd.read_csv(SOURCE_H / "event_counts.csv")
    counts_new = pd.read_csv(DEST_H / "event_counts.csv")
    require(counts_old.loc[counts_old.event != "low_power_le_minus85_dbm"].equals(
        counts_new.loc[counts_new.event != "low_power_le_minus85_dbm"]),
        "Non-power event counts changed")
    require(int(expected_low.sum()) == 53, "Unexpected corrected low-power count")
    results["low_power_hours_before"] = int(original_events["low_power_le_minus85_dbm"].sum())
    results["low_power_hours_after"] = int(expected_low.sum())

    original_w = pd.read_csv(SOURCE_W, usecols=["hour_bjt", "received_power_w"])
    original_w["hour_bjt"] = pd.to_datetime(original_w["hour_bjt"])
    comparison = all_power.merge(original_w, on="hour_bjt", validate="one_to_one")
    factor = np.where(comparison["hour_bjt"] >= BOUNDARY, gain, 1.0)
    expected_dbm = 10.0 * np.log10(comparison["received_power_w"] * factor / 0.001)
    require(np.allclose(expected_dbm, comparison["received_power_dbm"],
                        rtol=0, atol=1e-10), "Hourly W/dBm conversion mismatch")
    return results


def validate_fast_file(source_path: Path, dest_path: Path, gain: float,
                       shift: float, watts: tuple[str, ...],
                       dbm: tuple[str, ...], time_field: str) -> int:
    old = pq.ParquetFile(source_path)
    new = pq.ParquetFile(dest_path)
    require(old.schema_arrow.equals(new.schema_arrow, check_metadata=True),
            f"Fast schema/metadata changed: {source_path}")
    require(old.metadata.num_rows == new.metadata.num_rows and
            old.num_row_groups == new.num_row_groups,
            f"Fast rows/row groups changed: {source_path}")
    untouched = [name for name in old.schema_arrow.names if name not in watts + dbm]
    for index in range(old.num_row_groups):
        before = old.read_row_group(index)
        after = new.read_row_group(index)
        require(before.select(untouched).equals(after.select(untouched),
                                               check_metadata=False),
                f"Non-power fast values changed: {source_path} group {index}")
        times = after[time_field].to_pandas()
        require((times >= BOUNDARY_NAIVE).all(),
                f"Uncalibrated pre-change time in post file: {source_path}")
        for name in watts:
            before_values = before[name].to_numpy(zero_copy_only=False)
            after_values = after[name].to_numpy(zero_copy_only=False)
            require(np.allclose(after_values, before_values * gain,
                                rtol=2e-15, atol=0, equal_nan=True),
                    f"W factor mismatch: {source_path} {name} group {index}")
        for name in dbm:
            before_values = before[name].to_numpy(zero_copy_only=False)
            after_values = after[name].to_numpy(zero_copy_only=False)
            expected = (before_values.astype(np.float64) + shift).astype(np.float32)
            require(np.array_equal(after_values, expected, equal_nan=True),
                    f"dBm shift mismatch: {source_path} {name} group {index}")
    return old.metadata.num_rows


def validate_fast(gain: float, shift: float) -> dict[str, int]:
    stats: dict[str, int] = {}
    for layer, watts, dbm, time_field in (
        ("fast_100ms", ("peak_power_w", "slow_hourly_power_w"),
         ("peak_power_dbm", "slow_hourly_power_dbm"), "time_bjt"),
        ("fast_1s", ("received_power_w", "slow_hourly_power_w"),
         ("received_power_dbm", "slow_hourly_power_dbm"), "second_bjt"),
    ):
        pre_files = post_files = post_rows = 0
        for source in sorted((SOURCE_F / layer).glob("*.parquet")):
            dest = DEST_F / layer / source.name
            require(dest.is_file(), f"Missing calibrated file: {dest}")
            if source.stem[-7:] < "2026-05":
                require(sha256(source) == sha256(dest),
                        f"Pre-change file not byte-identical: {source}")
                pre_files += 1
            else:
                post_rows += validate_fast_file(
                    source, dest, gain, shift, watts, dbm, time_field
                )
                post_files += 1
        require(pre_files == 6 and post_files == 4, f"Fast month count changed: {layer}")
        stats[f"{layer}_pre_files_identical"] = pre_files
        stats[f"{layer}_post_files_full_row_verified"] = post_files
        stats[f"{layer}_post_rows_full_row_verified"] = post_rows

    old_windows = pd.read_parquet(SOURCE_F / "fast_window_statistics.parquet")
    new_windows = pd.read_parquet(DEST_F / "fast_window_statistics.parquet")
    require(list(old_windows.columns) == list(new_windows.columns) and
            len(old_windows) == len(new_windows), "Window schema/rows changed")
    cols = ["received_power_w", "received_power_dbm"]
    require(old_windows.drop(columns=cols).equals(new_windows.drop(columns=cols)),
            "Window non-power statistics changed")
    mask = old_windows["window_start_bjt"] >= BOUNDARY_NAIVE
    require(np.array_equal(old_windows.loc[~mask, cols].to_numpy(),
                           new_windows.loc[~mask, cols].to_numpy(), equal_nan=True),
            "Pre-change window power changed")
    require(np.allclose(new_windows.loc[mask, "received_power_w"],
                        old_windows.loc[mask, "received_power_w"] * gain,
                        rtol=2e-15, atol=0, equal_nan=True),
            "Post-change window W factor mismatch")
    require(np.allclose(new_windows.loc[mask, "received_power_dbm"],
                        old_windows.loc[mask, "received_power_dbm"] + shift,
                        rtol=0, atol=1e-12, equal_nan=True),
            "Post-change window dBm shift mismatch")
    stats["window_rows"] = len(new_windows)
    stats["window_post_rows_full_row_verified"] = int(mask.sum())
    return stats


def main() -> None:
    info = json.loads((DEST_H / "equipment_calibration.json").read_text(encoding="utf-8"))
    a1, a2, before_n, after_n = recompute_gain()
    gain = a1 / a2
    shift = 10.0 * math.log10(gain)
    require(before_n == 1432 and after_n == 269, "Clear candidate hours changed")
    require(math.isclose(a1, info["a1_before_mean_w"], rel_tol=1e-15) and
            math.isclose(a2, info["a2_after_mean_w"], rel_tol=1e-15) and
            math.isclose(gain, info["linear_gain"], rel_tol=1e-15) and
            math.isclose(shift, info["db_shift"], rel_tol=1e-15),
            "Independently recomputed gain differs")
    for name, path in (
        ("clear_dry_event_labels", SOURCE_H / "hourly_event_labels.csv"),
        ("hourly_power_w", SOURCE_W),
        ("source_model_all", SOURCE_H / "model_table_primary_p10_all_measurements.parquet"),
        ("source_fast_manifest", SOURCE_F / "run_manifest.json"),
        ("builder", Path(__file__).with_name("build_equipment_calibration.py")),
    ):
        require(sha256(path) == info["input_sha256"][name],
                f"Source changed since build: {name}")
    hourly = validate_hourly(shift, gain)
    print("Hourly verified", json.dumps(hourly, ensure_ascii=False), flush=True)
    fast = validate_fast(gain, shift)
    print("Fast verified", json.dumps(fast, ensure_ascii=False), flush=True)
    hashes = {
        "Revision_006": verify_checksums(DEST_H),
        "Fast_Variation_Revision_002": verify_checksums(DEST_F),
    }
    report = {
        "status": "VERIFIED",
        "scope": "retrospective equipment gain; not clear-sky attenuation",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_bjt": BOUNDARY.isoformat(),
        "calibration_clear_hours_before": before_n,
        "calibration_clear_hours_after": after_n,
        "a1_w": a1,
        "a2_w": a2,
        "linear_gain": gain,
        "db_shift": shift,
        "hourly": hourly,
        "fast": fast,
        "checksums_verified": hashes,
    }
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    for root in (DEST_H, DEST_F):
        (root / "validation_report.json").write_text(content, encoding="utf-8")
        (root / "VALIDATION.md").write_text(
            "# 独立验证\n\n状态：VERIFIED。" 
            "全量逐行验证报告见 `validation_report.json`。"
            "该状态仅证明设备增益校准与源数据关系正确，"
            "不表示已生成相对晴空衰减标签或完成模型实验。\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
