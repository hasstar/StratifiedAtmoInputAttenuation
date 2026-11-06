"""Stage source-specific hourly signals without inventing clear-sky references.

The local and OpenSat outputs contain observed power, not attenuation. CEDA GBS
already supplies attenuation, which is averaged via linear transmission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import netcdf_file


PROJECT = Path(__file__).resolve().parents[3]
LOCAL = PROJECT / "Outputs/Datasets/Revision_005"
LOCAL_QUALITY = PROJECT / "Outputs/Datasets/Revision_002/measurement_hourly.csv"
OPENSAT = PROJECT / "Outputs/Datasets/External_OpenSat4Weather_001"
GBS = PROJECT / "外部样例数据CEDA/GBS_20p7GHz"
OUTPUT = PROJECT / "Outputs/Datasets/ClearSky_Alignment_Revision_002"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_utc_hour(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    source = frame["hour_utc"] if "hour_utc" in frame else frame["hour_bjt"]
    frame["hour_utc"] = pd.to_datetime(source, utc=True)
    require(frame["hour_utc"].notna().all(), f"{name}: invalid time")
    require(frame["hour_utc"].is_unique, f"{name}: duplicate time")
    return frame


def local_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    report = json.loads((LOCAL / "validation_report.json").read_text(encoding="utf-8"))
    require(report.get("status") == "VERIFIED", "Revision_005 is not verified")
    model = add_utc_hour(
        pd.read_parquet(
            LOCAL / "model_table_primary_p10_all_measurements.parquet",
            columns=["hour_bjt", "hour_utc", "split_event_chrono_24h", "received_power_dbm"],
        ), "local model",
    )
    quality = add_utc_hour(
        pd.read_csv(
            LOCAL_QUALITY,
            usecols=[
                "hour_utc", "measurement_coverage_fraction", "measurement_high_coverage_95",
                "quality_zero_fraction", "quality_bit16_fraction", "quality_bit32_fraction",
                "source_k11_fraction", "source_k101_fraction",
            ],
        ), "local quality",
    )
    events = add_utc_hour(
        pd.read_csv(
            LOCAL / "hourly_event_labels.csv",
            usecols=["hour_bjt", "clear_dry", "rain", "heavy_rain", "dry_overcast", "snow"],
        ), "local events",
    ).drop(columns="hour_bjt")
    result = model.merge(quality, on="hour_utc", how="left", validate="one_to_one")
    result = result.merge(events, on="hour_utc", how="left", validate="one_to_one")
    require(len(result) == 3869, "local row count changed")
    require(result["quality_zero_fraction"].notna().all(), "local quality mismatch")
    require(result["clear_dry"].notna().all(), "local event mismatch")
    result = result.rename(columns={"received_power_dbm": "observed_power_dbm"})
    result.insert(0, "dataset", "local_Run011_Revision005")
    result.insert(1, "link_id", "local_single_link")
    result["clear_sky_attenuation_db"] = np.nan
    result["target_status"] = "baseline_not_established"
    result = result.sort_values("hour_utc").reset_index(drop=True)

    result["month_bjt"] = pd.to_datetime(result["hour_bjt"]).dt.strftime("%Y-%m")
    monthly = []
    for month, group in result.groupby("month_bjt", sort=True):
        clear = group.loc[group["clear_dry"], "observed_power_dbm"]
        screened = group.loc[
            group["clear_dry"]
            & group["measurement_high_coverage_95"]
            & (group["quality_zero_fraction"] >= 0.95),
            "observed_power_dbm",
        ]
        monthly.append({
            "month_bjt": month,
            "measured_hours": len(group),
            "clear_dry_hours": len(clear),
            "screened_clear_dry_hours": len(screened),
            "clear_power_p10_dbm": clear.quantile(0.1) if len(clear) else np.nan,
            "clear_power_p50_dbm": clear.median() if len(clear) else np.nan,
            "clear_power_p90_dbm": clear.quantile(0.9) if len(clear) else np.nan,
            "screened_clear_power_p50_dbm": screened.median() if len(screened) else np.nan,
        })
    return result, pd.DataFrame(monthly)


def opensat_hourly() -> pd.DataFrame:
    frames = []
    native_count = 0
    with h5py.File(OPENSAT / "extracted/sml_data_2022.nc", "r") as source:
        rsl = source["rsl"]
        n_links, n_minutes = rsl.shape
        require((n_links, n_minutes) == (215, 220320), "OpenSat shape changed")
        require(np.array_equal(source["time"][:], np.arange(n_minutes)), "OpenSat time changed")
        fill = int(np.asarray(rsl.attrs["_FillValue"]).ravel()[0])
        scale = float(np.asarray(rsl.attrs["scale_factor"]).ravel()[0])
        require(fill == -9999 and np.isclose(scale, 0.001), "OpenSat encoding changed")
        hours = pd.date_range("2022-08-01", periods=n_minutes // 60, freq="h", tz="UTC")
        ids = source["sml_id"][:]
        require(len(set(ids.tolist())) == 215, "OpenSat link IDs are not unique")
        for i in range(n_links):
            raw = rsl[i, :]
            valid = raw != fill
            counts = valid.reshape(-1, 60).sum(axis=1).astype(np.int32)
            dbm = raw.astype(np.float64) * scale
            milliwatts = np.zeros(n_minutes, dtype=np.float64)
            milliwatts[valid] = np.power(10.0, dbm[valid] / 10.0)
            hourly_mw = np.divide(
                milliwatts.reshape(-1, 60).sum(axis=1), counts,
                out=np.full(len(counts), np.nan), where=counts > 0,
            )
            hourly_dbm = np.full(len(counts), np.nan)
            np.log10(hourly_mw, out=hourly_dbm, where=counts > 0)
            hourly_dbm *= 10.0
            native_count += int(counts.sum())
            frames.append(pd.DataFrame({
                "dataset": "OpenSat4Weather",
                "link_id": ids[i].decode("utf-8"),
                "hour_utc": hours,
                "observed_power_dbm": hourly_dbm,
                "valid_minutes": counts,
                "coverage_fraction": counts / 60.0,
                "site_lat_deg": float(source["site_0_lat"][i]),
                "site_lon_deg": float(source["site_0_lon"][i]),
                "site_alt_m": int(source["site_0_alt"][i]),
                "satellite": source["satellite"][i].decode("utf-8"),
                "hardware": source["hardware"][i].decode("utf-8"),
                "clear_sky_attenuation_db": np.nan,
                "target_status": "baseline_not_established",
            }))
    require(native_count == 46944206, "OpenSat nonmissing count changed")
    result = pd.concat(frames, ignore_index=True)
    require(len(result) == 215 * 3672, "OpenSat hourly row count changed")
    return result


def gbs_hourly() -> tuple[pd.DataFrame, dict[str, int]]:
    with (GBS / "source_manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    require(len(entries) == 1982, "GBS file count changed")
    qa = json.loads((GBS / "QA_Revision_002/qa_summary.json").read_text(encoding="utf-8"))
    require(qa.get("status_counts") == {"readable_verified": 1982}, "GBS QA changed")
    require(len({(e["site"], e["date_utc"]) for e in entries}) == 1982,
            "GBS has duplicate site/day")
    rows = []
    valid_total = fill_total = out_of_range_total = bad_time_with_signal = 0
    boundary_zero_total = 0
    for entry in entries:
        path = GBS / entry["relative_path"]
        require(path.is_file() and path.stat().st_size == int(entry["bytes"]),
                f"GBS file missing or size changed: {path}")
        with netcdf_file(path, "r", mmap=False) as source:
            signal = source.variables["clearsky_attenuation"]
            attenuation = np.asarray(signal.data, dtype=np.float64).ravel()
            time_s = np.asarray(source.variables["time"].data, dtype=np.float64).ravel()
            require(len(attenuation) == len(time_s), f"GBS pair mismatch: {path}")
            attrs = signal._attributes
            require(attrs["units"] == b"dB", f"GBS unit changed: {path}")
            fill = float(attrs["_FillValue"])
            minimum = float(attrs["valid_min"])
            maximum = float(attrs["valid_max"])
            finite = np.isfinite(attenuation)
            missing = ~finite | (attenuation == fill)
            nonfill = ~missing
            out_of_range = nonfill & ((attenuation < minimum) | (attenuation > maximum))
            valid_time = np.isfinite(time_s) & (time_s >= 0) & (time_s <= 86400)
            valid = nonfill & ~out_of_range & valid_time
            bad_time_with_signal += int(np.count_nonzero(nonfill & ~valid_time))
            fill_total += int(np.count_nonzero(missing))
            out_of_range_total += int(np.count_nonzero(out_of_range))
            valid_total += int(np.count_nonzero(valid))

            # The native time marks the END of each one-second integration.
            # Exactly 01:00:00 therefore belongs to the preceding 00:00 hour.
            # t=0 is a boundary observation ending in the previous day; do
            # not put it into this day's first hour as a 3,601st second.
            boundary_zero_total += int(np.count_nonzero(valid & (time_s == 0)))
            hourly_valid = valid & (time_s > 0)
            hour_index = np.ceil(time_s[hourly_valid] / 3600).astype(np.int64) - 1
            require(np.all(hour_index < 24), f"GBS hour index outside day: {path}")
            values = attenuation[hourly_valid]
            count = np.bincount(hour_index, minlength=24)
            require(np.all(count <= 3600), f"GBS overfull hour: {path}")
            sum_transmission = np.bincount(
                hour_index, weights=np.power(10.0, -values / 10.0), minlength=24
            )
            sum_db = np.bincount(hour_index, weights=values, minlength=24)
            ge3 = np.bincount(hour_index, weights=values >= 3.0, minlength=24)
            ge10 = np.bincount(hour_index, weights=values >= 10.0, minlength=24)
            hourly_attenuation = np.full(24, np.nan)
            hourly_arithmetic = np.full(24, np.nan)
            has_data = count > 0
            hourly_attenuation[has_data] = -10.0 * np.log10(
                sum_transmission[has_data] / count[has_data]
            )
            hourly_arithmetic[has_data] = sum_db[has_data] / count[has_data]
            day = pd.Timestamp(entry["date_utc"], tz="UTC")
            for hour in range(24):
                rows.append({
                    "dataset": "CEDA_GBS",
                    "site": entry["site"],
                    "hour_utc": day + pd.Timedelta(hours=hour),
                    "clear_sky_attenuation_db": hourly_attenuation[hour],
                    "native_arithmetic_mean_db_diagnostic": hourly_arithmetic[hour],
                    "valid_seconds": int(count[hour]),
                    "coverage_fraction": float(count[hour] / 3600.0),
                    "seconds_ge_3db": int(ge3[hour]),
                    "seconds_ge_10db": int(ge10[hour]),
                    "source_file": entry["relative_path"],
                })
    require(bad_time_with_signal == 0, "GBS has nonfill measurements with invalid time")
    require(fill_total == 2632499, "GBS fill count changed")
    require(out_of_range_total == 20886, "GBS out-of-range count changed")
    require(valid_total == 168379274, "GBS valid count changed")
    result = pd.DataFrame(rows).sort_values(["site", "hour_utc"]).reset_index(drop=True)
    require(len(result) == 1982 * 24, "GBS hourly count changed")
    require(not result.duplicated(["site", "hour_utc"]).any(), "GBS duplicate hour")
    require(int(result["valid_seconds"].sum()) == valid_total - boundary_zero_total,
            "GBS hourly count conservation failed")
    stats = {
        "source_files": len(entries),
        "valid_native_seconds": valid_total,
        "excluded_midnight_boundary_seconds": boundary_zero_total,
        "aggregated_valid_seconds": int(result["valid_seconds"].sum()),
        "fill_seconds": fill_total,
        "out_of_range_seconds": out_of_range_total,
        "hour_rows": len(result),
        "hours_with_valid_attenuation": int(result["clear_sky_attenuation_db"].notna().sum()),
        "high_coverage_95_hours": int((result["coverage_fraction"] >= 0.95).sum()),
    }
    return result, stats


def main() -> None:
    require(not OUTPUT.exists(), f"Output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    local, monthly = local_hourly()
    opensat = opensat_hourly()
    gbs, gbs_stats = gbs_hourly()

    local.to_parquet(OUTPUT / "local_hourly_power_audit.parquet", index=False)
    monthly.to_csv(OUTPUT / "local_clear_candidate_monthly.csv", index=False)
    opensat.to_parquet(OUTPUT / "opensat_hourly_power.parquet", index=False)
    gbs.to_parquet(OUTPUT / "ceda_gbs_hourly_attenuation.parquet", index=False)
    pd.DataFrame([
        {
            "dataset": "local_Run011_Revision005", "native_field": "peak_power_dbm",
            "native_unit": "dBm", "signal_band": "Ka beacon",
            "native_resolution": "100 ms; existing model table 1 h",
            "hour_rows": len(local), "ready_attenuation_hours": 0,
            "reference_status": "not established; local weather and receiver-state audit needed",
        },
        {
            "dataset": "OpenSat4Weather", "native_field": "rsl",
            "native_unit": "dBm", "signal_band": "Ku wideband TV",
            "native_resolution": "1 min; prepared 1 h",
            "hour_rows": len(opensat), "ready_attenuation_hours": 0,
            "reference_status": "not established; site-specific clear-sky and hardware audit plus cloud data needed",
        },
        {
            "dataset": "CEDA_GBS", "native_field": "clearsky_attenuation",
            "native_unit": "dB", "signal_band": "20.7 GHz beacon",
            "native_resolution": "1 s; prepared 1 h",
            "hour_rows": len(gbs),
            "ready_attenuation_hours": gbs_stats["hours_with_valid_attenuation"],
            "reference_status": "provided by source archive; no re-baselining",
        },
    ]).to_csv(OUTPUT / "source_registry.csv", index=False)

    readme = f"""# 三数据源相对晴空衰减指标整理（Revision 002）

状态：**仅 CEDA GBS 已有可用的相对晴空衰减标签；不是可直接合并训练的三源衰减数据集。**
本目录为新建的独立整理包，不修改任何 H5、ERA5、Run_011 或既有 Revision 数据集。

| 来源 | 原生观测 | 整理后的小时表 | 可用衰减标签 |
|---|---|---|---|
| 本地 Ka 信标 | `peak_power_dbm`，dBm | `local_hourly_power_audit.parquet`，{len(local):,} 小时 | 0；晴空基线未建立 |
| OpenSat4Weather，法国 Ku 电视链路 | `rsl`，dBm | `opensat_hourly_power.parquet`，{len(opensat):,} 链路-小时 | 0；逐链路晴空基线未建立 |
| CEDA GBS，英国 20.7 GHz 信标 | `clearsky_attenuation`，dB | `ceda_gbs_hourly_attenuation.parquet`，{len(gbs):,} 站点-小时；其中 {gbs_stats['hours_with_valid_attenuation']:,} 小时有值 | 原生标签，可用 |

## 共同目标定义

若功率以 dBm 表示，目标为 `A_cs(t) = P_clear_reference(t) - P_observed(t)`，单位 dB；
它表示相对于该链路**同一设备状态、同时段附近的无雨无云参考功率**的衰减，不是自由空间/真空总路径损耗。
CEDA 数据源已经提供这种定义的标签，原生注释明确写明晴空为“无雨、无云”。
本地和 OpenSat 目前只有观测功率，不能在未确定参考功率时把 dBm 改名为 dB，也不能从三站/215 条链路共用一个基准值。
有效负值保留；不进行零截断。三套数据频段、接收机和观测目标不同，共同单位不等于可以无条件合并。

## 本轮处理和文件

- `source_registry.csv`：三源原生字段、单位、频段、小时数量和标签可用状态。
- `local_hourly_power_audit.parquet`：本地原有小时功率、质量比例、既有划分与天气候选标志；`clear_sky_attenuation_db` 全为空。
- `local_clear_candidate_monthly.csv`：本地 ERA5“少雨少雪少云”候选时段的功率月度分布，仅用于检查基线漂移，**不是**正式晴空基线。
- `opensat_hourly_power.parquet`：逐链路分钟 `rsl` 转线性毫瓦均值再转 dBm，保存原生有效分钟数、覆盖率、站点坐标和卫星名；`clear_sky_attenuation_db` 全为空。
- `ceda_gbs_hourly_attenuation.parquet`：原生 1 秒 dB 衰减经有效性过滤后，逐小时在相对线性传输率域汇总：
  `A_hour = -10 log10(mean(10^(-A_second/10)))`。同时保留算术 dB 均值供诊断，不作为目标。
  NetCDF 时间标记为积分**结束时刻**，整点记录归入结束前的那个小时；本日 00:00 边界记录归属上日，
  在本日小时表中排除（本版共 {gbs_stats['excluded_midnight_boundary_seconds']:,} 秒），避免首小时出现 3,601 秒。
  无有效秒的小时标签为空。
- `manifest.json`：输入文件哈希、输出文件哈希、环境和关键计数。

CEDA 的填充值 -100 dB 与超出原生有效范围 [-5, 100] dB 的记录均剔除，
但 [-5, 0) dB 的有效负值保留；不把这些质量过滤等同于天气事件筛选。
本整理未下载或匹配额外 ERA5，也未重新定义既有训练/验证/测试划分。

## 生成正式三源标签之前

1. 本地：依据无雨无云证据和接收系统状态，建立分段/随时间变化且**仅从训练期拟合**的晴空参考；先检查采集配置、质量标志与功率漂移，避免把仪器变化算作大气衰减。
2. OpenSat：为每条链路匹配同期云、降水等天气证据，审计设备状态/功率漂移，再为每条链路独立拟合训练期晴空参考，并在留出时间和站点验证。
3. CEDA：保留原生标签，不根据本地方法重算；后续另行匹配英国三站 2003–2006 年的 ERA5，并统一小时有效覆盖门槛。
4. 跨数据源比较应分开报告、按各自站点/链路划分训练与测试，不能将 Ku 与 Ka/20.7 GHz 误作同频同设备直接混训。

生成脚本：`Code/Dataset_Build/ClearSky_Alignment_Revision_002/build_alignment.py`。
Revision 001 为 QA 审计发现首小时边界重复计数的保留版本，不用于下游分析。
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    sources = {
        "local_validation_report": LOCAL / "validation_report.json",
        "local_event_labels": LOCAL / "hourly_event_labels.csv",
        "local_quality_table": LOCAL_QUALITY,
        "opensat_archive": OPENSAT / "OpenSat4Weather_data.zip",
        "gbs_source_manifest": GBS / "source_manifest.csv",
        "gbs_qa_report": GBS / "QA_Revision_002/qa_summary.json",
        "builder": Path(__file__),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGING_ONLY_NOT_A_THREE_DATASET_ATTENUATION_TARGET",
        "scope": "hourly signal alignment and existing CEDA attenuation; no new clear-sky reference fitted",
        "formula": "A_h=-10*log10(mean(10^(-A_native_db/10))) for valid CEDA samples",
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()
        },
        "counts": {
            "local_hours": len(local),
            "local_high_coverage_hours": int(local["measurement_high_coverage_95"].sum()),
            "opensat_links": int(opensat["link_id"].nunique()),
            "opensat_hour_rows": len(opensat),
            "opensat_valid_native_minutes": int(opensat["valid_minutes"].sum()),
            "gbs": gbs_stats,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
