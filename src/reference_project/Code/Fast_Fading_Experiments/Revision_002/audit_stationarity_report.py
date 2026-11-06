"""Independent read-only upstream audit and documentation of the frozen C3 report.

Does not fit a generator, choose new thresholds, or read final holdout samples.
Only adds its own audit and descriptive Markdown inside Fast_Fading_Revision_002.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_002"
DEST = OUT / "stationarity"
ENDPOINTS = ("half_power_drift_db", "half_sd_difference_db", "half_p05_difference_db",
             "half_p50_difference_db", "half_p95_difference_db", "half_lag1_difference")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, (20261007,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**63)


def bootstrap(values: pd.Series, length: int, tag: str) -> dict:
    # Independently vectorized, retaining the frozen random stream and date gaps.
    times = pd.to_datetime(values.index).to_numpy(dtype="datetime64[D]")
    starts = [i for i in range(len(values) - length + 1)
              if int((times[i + length - 1] - times[i]) / np.timedelta64(1, "D")) == length - 1]
    if not starts:
        return {"status": "no_contiguous_date_blocks", "date_union": 0}
    blocks = np.array([np.arange(i, i + length) for i in starts])
    rng = np.random.default_rng(stable_seed("C3-bootstrap", tag, length))
    chosen = rng.integers(len(blocks), size=(2000, math.ceil(len(values) / length)))
    indices = blocks[chosen].reshape(2000, -1)[:, :len(values)]
    draws = values.to_numpy(dtype=float)[indices].mean(axis=1)
    return {"status": "complete", "ci025": float(np.quantile(draws, .025)),
            "ci975": float(np.quantile(draws, .975)), "eligible_blocks": len(blocks),
            "date_union": len(np.unique(blocks))}


def literal_probability(observed: np.ndarray, reference: np.ndarray, half: bool) -> float:
    n = 32 if half else 64
    fit, calibration = reference[:n], reference[64:64 + n]
    if not np.isfinite(observed).all() or not np.isfinite(reference).all():
        return float("nan")
    obs_terms, cal_terms = [], []
    for j in range(6):
        mean = sum(float(v) for v in fit[:, j]) / n
        scale = math.sqrt(sum((float(v) - mean)**2 for v in fit[:, j]) / (n - 1))
        if scale > 0:
            obs_terms.append((float(observed[j]) - mean) / scale)
            cal_terms.append([(float(v) - mean) / scale for v in calibration[:, j]])
        else:
            obs_terms.append(float("inf") if observed[j] > mean else 0.)
            cal_terms.append([float("inf") if v > mean else 0. for v in calibration[:, j]])
    test = max(obs_terms)
    extremes = [max(cal_terms[j][i] for j in range(6)) for i in range(n)]
    return (1 + sum(v >= test for v in extremes)) / (n + 1)


def run() -> dict:
    qa, selection = read_json(DEST / "qa.json"), read_json(DEST / "selection.json")
    if qa["status"] != "PASS" or selection["qa_sha256"] != digest(DEST / "qa.json"):
        raise RuntimeError("Unverified C3 upstream")
    if qa["manifest_sha256"] != digest(DEST / "manifest.json"):
        raise RuntimeError("Manifest changed")
    checks = 0
    for name, expected in selection["output_sha256"].items():
        if digest(DEST / name) != expected:
            raise RuntimeError(f"Report output changed: {name}")
        checks += 1
    assignments = pd.read_csv(OUT / "marginal_fold_hours.csv").set_index("hour_bjt")
    context = {i: read_json(OUT / f"dynamics/folds/fold_{i}/context.json") for i in range(5)}
    # Cache schema is checked explicitly; no held-out observations enter the fit pool.
    for value in context.values():
        if set(value["fit_days"]) & set(value["held_days"]):
            raise RuntimeError("Date leakage in fold context")
    pieces, hours = [], []
    max_source_error = 0.
    for marker in sorted((DEST / "chunks").glob("fold_*.json")):
        record = read_json(marker)
        if digest(marker) != qa["checkpoint_hashes"][marker.name]:
            raise RuntimeError("Checkpoint changed after complete QA")
        checks += 1
        for name, expected in record["output_sha256"].items():
            if digest(DEST / "chunks" / name) != expected:
                raise RuntimeError("Completed chunk changed")
            checks += 1
        for meta in record["metadata"]:
            actual = assignments.loc[meta["hour"]]
            if int(actual["fold"]) != int(meta["fold"]):
                raise RuntimeError("Hour fold mismatch")
            if meta["day"] not in context[int(meta["fold"])]["held_days"]:
                raise RuntimeError("Reference uses wrong fold")
            max_source_error = max(max_source_error, float(meta["source_endpoint_max_error"]))
        piece = pd.read_parquet(DEST / "chunks" / (marker.stem + ".parquet"))
        piece["_checkpoint"] = marker.stem
        pieces.append(piece)
        hours.extend(record["hours"])
    windows = pd.concat(pieces, ignore_index=True)
    windows["month"] = windows.date_bjt.astype(str).str[:7]
    key = ["quality_gate", "hour_key", "window_minutes", "slot"]
    if len(hours) != 2404 or len(set(hours)) != 2404 or set(hours) != set(assignments.index):
        raise RuntimeError("Official training hour scope mismatch")
    if len(windows) != 653430 or windows.duplicated(key).any():
        raise RuntimeError("Window completeness mismatch")
    # Boundary-preserving probability spot-checks supplement the complete 302793-window QA.
    witnesses = windows.groupby(["month", "quality_gate", "window_minutes"], sort=True).head(1)
    probability_errors = []
    cache = {}
    for _, row in witnesses.iterrows():
        stem = row["_checkpoint"]
        if stem not in cache:
            checkpoint = read_json(DEST / "chunks" / (stem + ".json"))
            hour_indices = {h: i for i, h in enumerate(checkpoint["hours"])}
            with np.load(DEST / "chunks" / (stem + "_reference.npz")) as bank:
                cache[stem] = (hour_indices, {name: bank[name] for name in bank.files})
        hour_indices, banks = cache[stem]
        reference = banks[f"hour_{hour_indices[row.hour_key]}"][int(row.reference_index_within_hour)]
        observed = row[list(ENDPOINTS)].to_numpy(dtype=float)
        for half, column in ((False, "reference_tail_probability"), (True, "half_reference_tail_probability")):
            expected = literal_probability(observed, reference, half)
            actual = float(row[column])
            if not (np.isnan(expected) and np.isnan(actual)):
                probability_errors.append(abs(expected - actual))
    del cache
    keys = ["quality_gate", "window_minutes", "event", "date_bjt", "hour_key"]
    complete = windows.loc[windows.compatibility_status.eq("complete")].copy()
    complete["classification_changed"] = complete.compatible != complete.half_compatible
    quantities = ["compatible", "half_compatible", "classification_changed"]
    hourly = complete.groupby(keys)[quantities].mean().reset_index()
    all_hours = hourly.assign(event="all_train")
    hourly = pd.concat([hourly, all_hours], ignore_index=True)
    daily = hourly.groupby(keys[:-1])[quantities].mean().reset_index()
    saved_hourly, saved_daily = pd.read_parquet(DEST / "hour_summary.parquet"), pd.read_parquet(DEST / "date_summary.parquet")
    for actual, expected, indices in ((saved_hourly, hourly, keys), (saved_daily, daily, keys[:-1])):
        pd.testing.assert_frame_equal(actual.sort_values(indices).reset_index(drop=True),
                                      expected.sort_values(indices).reset_index(drop=True), check_dtype=False)
    summary = pd.read_csv(DEST / "window_compatibility.csv")
    expected_rows = daily.groupby(keys[:3]).ngroups * 9
    if len(summary) != expected_rows:
        raise RuntimeError("Incomplete report states")
    summary_errors = []
    for row in summary.itertuples(index=False):
        part = daily.loc[daily.quality_gate.eq(row.quality_gate) & daily.window_minutes.eq(row.window_minutes)
                         & daily.event.eq(row.event)]
        values = part.set_index("date_bjt")[row.quantity].sort_index()
        result = bootstrap(values, int(row.block_days), f"{(row.quality_gate, row.window_minutes, row.event)}|{row.quantity}")
        if result["status"] != row.status or result["date_union"] != row.date_union:
            raise RuntimeError("Date block definition mismatch")
        summary_errors.append(abs(values.mean() - row.mean_date_proportion))
        if result["status"] == "complete":
            if int(result["eligible_blocks"]) != int(row.eligible_blocks):
                raise RuntimeError("Date block count mismatch")
            summary_errors.extend(abs(result[name] - getattr(row, name)) for name in ("ci025", "ci975"))
    frozen_suggestions = pd.read_csv(DEST / "engineering_window_suggestions.csv")
    for row in frozen_suggestions.itertuples(index=False):
        part = summary.loc[summary.quality_gate.eq("primary90_gap5") & summary.event.eq(row.event)
                           & summary.block_days.eq(1) & summary.quantity.isin(("compatible", "half_compatible"))]
        supported = []
        for minutes in (1, 2, 5, 10, 15, 30, 60):
            prefix = part.loc[part.window_minutes.le(minutes)]
            if (len(prefix) == 2 * sum(t <= minutes for t in (1, 2, 5, 10, 15, 30, 60))
                    and prefix.ci025.notna().all() and prefix.ci025.ge(row.engineering_coverage).all()):
                supported.append(minutes)
        actual = None if pd.isna(row.supported_fixed_window_minutes) else int(row.supported_fixed_window_minutes)
        if actual != (max(supported) if supported else None):
            raise RuntimeError("Frozen engineering suggestion mismatch")
    if max(probability_errors, default=0.) > 1e-12 or max(summary_errors, default=0.) > 1e-12:
        raise RuntimeError("Independent numerical audit failed")
    audit = {"status": "PASS", "upstream_fingerprint_checks": checks,
             "training_hours": len(hours), "window_states": len(windows), "folds": 5,
             "date_leakage": 0, "max_recorded_source_endpoint_error": max_source_error,
             "independent_probability_witness_states": len(witnesses),
             "max_independent_probability_error": max(probability_errors, default=0.),
             "hour_summary_rows": len(hourly), "date_summary_rows": len(daily),
             "summary_rows_recomputed": len(summary), "bootstrap_replicates": 2000,
             "max_independent_summary_or_interval_error": max(summary_errors, default=0.),
             "engineering_suggestions_recomputed": len(frozen_suggestions),
             "manifest_sha256": digest(DEST / "manifest.json"), "qa_sha256": digest(DEST / "qa.json"),
             "selection_sha256": digest(DEST / "selection.json"), "script_sha256": digest(Path(__file__)),
             "final_validation_test_reads": 0, "whole_fast_fading_stage_complete": False}
    (DEST / "independent_report_qa.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# C3 模型相容窗口诊断结果", "", "只使用2,404训练小时。C3运行、完整端点/参考概率QA与本独立汇总审计均PASS。",
             "六端点衡量窗口两半的位置、尺度、分位数和lag-1依赖变化；与冻结模拟模型相容不等于证明传播过程严格平稳。",
             "256秒是C2选出的生成块长，不是本节的诊断窗口，也不自动成为平稳区间。", "",
             "## 主质量门槛：日期均衡窗口相容比例", "",
             "表中为64+64主参考银行；括号为2,000次独立日期重采样的95%区间。32+32敏感性、全部四质量门槛及1/3/7日块结果保存在CSV。",
             "", "| 天气 | 窗长min | 比例 | 95%日期区间 | 小时/日期 | 主半预算分类变化比例 |", "|---|---:|---:|---|---|---:|"]
    primary = summary.loc[summary.quality_gate.eq("primary90_gap5") & summary.block_days.eq(1)]
    for event in ("all_train", "clear_dry", "rain", "dry_overcast", "other"):
        for row in primary.loc[primary.event.eq(event) & primary.quantity.eq("compatible")].sort_values("window_minutes").itertuples(index=False):
            change = primary.loc[primary.event.eq(event) & primary.window_minutes.eq(row.window_minutes)
                                 & primary.quantity.eq("classification_changed"), "mean_date_proportion"].iloc[0]
            lines.append(f"| {event} | {row.window_minutes} | {row.mean_date_proportion:.4f} | [{row.ci025:.4f}, {row.ci975:.4f}] | {row.n_hours}/{row.n_days} | {change:.4f} |")
    lines += ["", "## 冻结工程窗口建议", "", "预先登记的0.90覆盖要求：所有更短候选的独立日期95%区间下界在主/半参考银行中均须达到0.90。没有满足候选时保留无统一固定窗口支持；不修改阈值追求15分钟。", ""]
    for row in frozen_suggestions.loc[np.isclose(frozen_suggestions.engineering_coverage, .90)].itertuples(index=False):
        window = "无统一固定窗支持" if pd.isna(row.supported_fixed_window_minutes) else f"{int(row.supported_fixed_window_minutes)}分钟"
        lines.append(f"- {row.event}：{window}。")
    lines += ["", "## 统计口径与复盘", "", "先对每小时有效窗比例取均值，再对同日小时取均值，最终日期等权；窗口和128条模拟实现不是独立实测样本。",
              "连续3/7日块不跨缺日期，CSV的date_union/eligible_blocks展示相应覆盖；未覆盖的零散日期不能被当作连续日。",
              "各长度纳入小时数不同；长度间变化不能自动解释成同一小时配对效应。严格10/10秒门槛无完整小时，空状态不解释为无漂移。",
              "0.05参考尾概率是估计生成模型条件下的诊断设置，不是未知传播过程的精确平稳检验显著性。",
              "本阶段未读取343验证/687测试快变残差。C1/C2/C3共享开发池；最终评价、100ms敏感性、设备/谱配置、慢变合成仍需独立计划、实现与QA。",
              "", "## 文件", "", "- manifest.json：C3源码/计划/源/折内缓存冻结指纹。", "- chunks：653,430状态和302,793唯一窗的128×6参考端点，完整种子及相位接缝记录。",
              "- qa.json：全窗独立参考概率复算。", "- independent_report_qa.json：独立日期汇总/全部区间/工程建议复算。",
              "- window_compatibility.csv：全部天气、质量、长度、主半预算、日期块区间。", "- engineering_window_suggestions.csv：0.80/0.90/0.95全部建议。", ""]
    (DEST / "STATIONARITY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return audit


if __name__ == "__main__":
    from window_diagnostics import worker_lock
    with worker_lock("audit_stationarity_report.py"):
        run()
