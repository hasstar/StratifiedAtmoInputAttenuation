"""Train-only refits of every layer mask; score validation and test only after lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

import common as c
import run_layers as layers
import support


VERSION = "heldout-all-masks-v1-20260930"
ROOT = c.OUT / "layers" / "heldout_all_masks"
PLAN = ROOT / "training_plan.json"
MANIFEST = ROOT / "train_manifest.json"
LOCK = ROOT / "TRAIN_LOCK.json"
BASE_MODEL = ROOT / "models" / "base188.ubj"
BASE_META = ROOT / "models" / "base188.json"
RESIDUAL_MODELS = ROOT / "models" / "residual"
OLD_MASKS = c.OUT / "layers" / "subsets"
EVENTS = ("clear_dry", "rain", "heavy_rain", "dry_overcast", "snow")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def feature_names() -> list[str]:
    names = pd.read_csv(c.DATA / "feature_columns.csv")["column"].tolist()
    ensure(len(names) == len(set(names)) == 348, "348-column feature list changed")
    ensure(tuple(int(names[188 + i * 16].split("__")[0][1:]) for i in range(10))
           == c.LEVELS, "Pressure-layer order changed")
    return names


def train_table(names: list[str]) -> pd.DataFrame:
    # This function reads train.parquet only; no validation/test/status file.
    frame = pd.read_parquet(c.DATA / "train.parquet")
    ensure(len(frame) == 2404 and list(frame.columns) ==
           ["hour_bjt", "hour_utc", *names, c.TARGET], "Training schema changed")
    ensure(frame.hour_bjt.is_unique and np.isfinite(frame[c.TARGET]).all(),
           "Invalid training hours/targets")
    frame = frame.sort_values("hour_bjt").reset_index(drop=True)
    frame["split"] = "train"
    return frame


def frozen_winner() -> dict:
    selected = read(c.OUT / "model_selection.json")
    qa = read(c.OUT / "model_selection_qa.json")
    ensure(selected["primary_winner"] == "xgboost" and qa["status"] == "PASS"
           and qa["fingerprint"] == selected["fingerprint"],
           "Frozen ten-model selection did not pass")
    row = next(row for row in selected["model_rows"] if row["name"] == "xgboost")
    ensure(row["best_config"] == {"n_estimators": 500, "max_depth": 5,
                                  "learning_rate": 0.05}
           and row["formal_rounds"] == 499, "Frozen XGBoost configuration changed")
    return {"name": "xgboost", "best_config": row["best_config"],
            "formal_rounds": 499, "selection_fingerprint": selected["fingerprint"]}


def build_plan() -> dict:
    names = feature_names()
    train_table(names)
    winner = frozen_winner()
    old_manifest = read(c.OUT / "layers" / "run_manifest.json")
    old_qa = read(c.OUT / "layers" / "validation_qa.json")
    ensure(old_qa["status"] == "PASS" and old_qa["fingerprint"] ==
           old_manifest["fingerprint"], "Source mask training QA failed")
    masks = []
    for mask in range(1024):
        source = OLD_MASKS / f"mask_{mask:04d}.json"
        item = read(source)
        ensure(item["fingerprint"] == old_manifest["fingerprint"] and
               item["mask"] == mask, f"Source mask {mask} mismatch")
        active, expected = layers.columns_for_mask(names, mask)
        ensure(item["features"] == expected and
               item["levels_hpa"] == [c.LEVELS[i] for i in active],
               f"Mask {mask} feature/level mismatch")
        if mask:
            scores = item["alpha_scores"]
            alpha = min(c.ALPHAS, key=lambda a: (scores[str(a)], a))
            rounds = int(np.median([f["selected_round"] for f in item["folds"]]))
            ensure(len(item["folds"]) == 3 and item["alpha"] == alpha and
                   item["formal_rounds"] == rounds and 1 <= rounds <= 120 and
                   math.isclose(item["cv_pooled_rmse_db"], scores[str(alpha)],
                                rel_tol=0, abs_tol=1e-10),
                   f"Mask {mask} training-only alpha/round rules changed")
        else:
            alpha, rounds = 0.0, 0
            ensure(item["alpha"] == 0.0 and item["formal_rounds"] == 0,
                   "Empty-mask settings changed")
        # No validation prediction/metric field is copied into the training plan.
        masks.append({"mask": mask, "levels_hpa": item["levels_hpa"],
                      "features": expected, "alpha": float(alpha),
                      "formal_rounds": rounds,
                      "training_internal_cv_rmse_db": item["cv_pooled_rmse_db"],
                      "source_mask_sha256": support.sha256(source)})
    proposed = {"version": VERSION, "winner": winner, "n_train": 2404,
                "n_schemes": 1024, "n_residual_models": 1023,
                "train_parquet_sha256": support.sha256(c.DATA / "train.parquet"),
                "feature_columns_sha256": support.sha256(c.DATA / "feature_columns.csv"),
                "source_layer_fingerprint": old_manifest["fingerprint"],
                "mask_plans": masks,
                "forbidden_training_sources": ["validation.parquet", "test.parquet",
                                               "hourly_sample_status.parquet"]}
    if PLAN.exists():
        ensure(read(PLAN) == proposed, "Frozen training plan changed")
    else:
        support.atomic_json(PLAN, proposed)
    return proposed


def train_manifest() -> dict:
    paths = [c.DATA / "train.parquet", c.DATA / "feature_columns.csv", PLAN,
             c.OUT / "model_selection.json", c.OUT / "model_selection_qa.json",
             c.HERE / "models.py", c.HERE / "support.py", c.HERE / "common.py",
             c.HERE / "run_layers.py", c.HERE / "run_test_by_mask.py",
             c.HERE / "PER_LAYER_TEST_PROTOCOL.md"]
    candidate = c.hash_sources(paths, VERSION)
    if MANIFEST.exists():
        existing = read(MANIFEST)
        ensure(existing["fingerprint"] == candidate["fingerprint"],
               "Training source fingerprint changed")
        return existing
    support.atomic_json(MANIFEST, candidate)
    return candidate


def save_model(model: XGBRegressor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.ubj")
    model.save_model(str(temporary))
    os.replace(temporary, path)
    return support.sha256(path)


def impute(x: np.ndarray, medians: list[float]) -> np.ndarray:
    stats = np.asarray(medians, dtype=float)
    ensure(x.ndim == 2 and x.shape[1] == len(stats) and np.isfinite(stats).all(),
           "Imputer statistics/feature shape mismatch")
    return np.where(np.isfinite(x), x, stats[None, :])


def checked_model(model_path: Path, meta_path: Path, fingerprint: str) -> tuple[dict, XGBRegressor]:
    meta = read(meta_path)
    ensure(meta["fingerprint"] == fingerprint and model_path.exists() and
           meta["model_sha256"] == support.sha256(model_path),
           f"Saved model mismatch: {model_path}")
    model = XGBRegressor()
    model.load_model(str(model_path))
    return meta, model


def residual_paths(mask: int) -> tuple[Path, Path]:
    stem = f"mask_{mask:04d}"
    return RESIDUAL_MODELS / f"{stem}.ubj", RESIDUAL_MODELS / f"{stem}.json"


def train_all() -> None:
    ensure(not (ROOT / "test_qa.json").exists() and not (ROOT / "test").exists(),
           "Test already opened; training is permanently locked")
    plan = build_plan()
    manifest = train_manifest()
    fingerprint = manifest["fingerprint"]
    frame = train_table(feature_names())
    names = feature_names()
    chosen = plan["winner"]
    positions = np.arange(len(frame), dtype=int)
    # Training-only 5-fold OOF basis is intentionally rebuilt, not inherited.
    oof, inner_audit = layers.crossfit_base(frame, positions, names[:188], chosen)
    truth = frame[c.TARGET].to_numpy(dtype=float)
    residual_target = truth - oof
    ensure(np.isfinite(residual_target).all(), "Invalid training OOF residuals")
    x_base_raw = c.raw(frame, positions, names[:188])
    if BASE_META.exists():
        base_meta, base_model = checked_model(BASE_MODEL, BASE_META, fingerprint)
        ensure(base_meta["features"] == names[:188] and
               base_meta["rounds"] == chosen["formal_rounds"],
               "Saved baseline configuration changed")
    else:
        base_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_base = base_imputer.fit_transform(x_base_raw)
        base_model, history, _, fit_meta = c.fit_logged(
            "xgboost", chosen["best_config"], 17, (), x_base, truth,
            fixed_rounds=chosen["formal_rounds"])
        base_sha = save_model(base_model, BASE_MODEL)
        base_meta = {"fingerprint": fingerprint, "features": names[:188],
                     "rounds": chosen["formal_rounds"], "seed": 17,
                     "medians": base_imputer.statistics_.astype(float).tolist(),
                     "model_sha256": base_sha, "train_fit_rmse_db": c.rmse(truth, base_model.predict(x_base)),
                     "train_round_history": history, "fit_metadata": fit_meta,
                     "completed_utc": support.utc_now()}
        support.atomic_json(BASE_META, base_meta)
    ensure(len(inner_audit) == 5 and len(oof) == 2404, "Training OOF incomplete")
    completed = 0
    for item in plan["mask_plans"][1:]:
        mask = item["mask"]
        model_path, meta_path = residual_paths(mask)
        if meta_path.exists():
            meta, _ = checked_model(model_path, meta_path, fingerprint)
            ensure(meta["mask"] == mask and meta["features"] == item["features"] and
                   meta["alpha"] == item["alpha"] and
                   meta["formal_rounds"] == item["formal_rounds"],
                   f"Saved residual mask {mask} configuration changed")
            continue
        raw = c.raw(frame, positions, item["features"])
        model, fitted_imputer, history, _, _, seconds = layers.fit_g(
            raw, residual_target, None, None, rounds=item["formal_rounds"])
        model_sha = save_model(model, model_path)
        meta = {"fingerprint": fingerprint, "mask": mask,
                "levels_hpa": item["levels_hpa"], "features": item["features"],
                "alpha": item["alpha"], "formal_rounds": item["formal_rounds"],
                "seed": 17, "config": c.RESIDUAL_CONFIG,
                "medians": fitted_imputer.statistics_.astype(float).tolist(),
                "model_sha256": model_sha, "train_round_history": history,
                "train_residual_rmse_db": c.rmse(
                    residual_target, model.predict(fitted_imputer.transform(raw))),
                "fit_seconds": seconds, "completed_utc": support.utc_now()}
        support.atomic_json(meta_path, meta)
        completed += 1
        if mask % 25 == 0 or mask == 1023:
            print(f"TRAINED mask={mask}/1023 new={completed}", flush=True)
    verify_models(plan, fingerprint)
    lock = {"status": "TRAIN_LOCKED", "fingerprint": fingerprint,
            "n_training_hours": 2404, "n_schemes": 1024,
            "n_residual_models": 1023, "base_model_sha256": support.sha256(BASE_MODEL),
            "oof_residual_sha256": hashlib.sha256(residual_target.tobytes()).hexdigest(),
            "inner_group_folds": [{"fold": x["inner_fold"], "n_fit": x["n_fit"],
                                    "n_holdout": x["n_holdout"]} for x in inner_audit],
            "validation_test_not_read_by_training": True,
            "completed_utc": support.utc_now()}
    support.atomic_json(LOCK, lock)
    print(json.dumps({"status": lock["status"], "n_schemes": 1024,
                      "new_residual_models": completed}), flush=True)


def verify_models(plan: dict, fingerprint: str) -> None:
    checked_model(BASE_MODEL, BASE_META, fingerprint)
    for item in plan["mask_plans"][1:]:
        model_path, meta_path = residual_paths(item["mask"])
        meta, _ = checked_model(model_path, meta_path, fingerprint)
        ensure(meta["mask"] == item["mask"] and meta["features"] == item["features"] and
               meta["alpha"] == item["alpha"] and
               meta["formal_rounds"] == item["formal_rounds"],
               f"Training plan/model mismatch: {item['mask']}")
    ensure(len(list(RESIDUAL_MODELS.glob("mask_*.json"))) == 1023 and
           len(list(RESIDUAL_MODELS.glob("mask_*.ubj"))) == 1023,
           "Residual model file count mismatch")


def load_evaluation(split: str, names: list[str]) -> pd.DataFrame:
    ensure(split in ("validation", "test"), "Unknown evaluation split")
    table = pd.read_parquet(c.DATA / f"{split}.parquet")
    expected = 343 if split == "validation" else 687
    ensure(len(table) == expected and list(table.columns) ==
           ["hour_bjt", "hour_utc", *names, c.TARGET], "Evaluation schema changed")
    status = pd.read_parquet(c.DATA / "hourly_sample_status.parquet", columns=[
        "hour_bjt", "split", "use_for_regression", *EVENTS])
    ensure(status.hour_bjt.is_unique, "Status hour duplicated")
    table = table.merge(status, on="hour_bjt", how="left", validate="one_to_one")
    ensure(table["split"].eq(split).all() and table["use_for_regression"].eq(True).all()
           and table[list(EVENTS)].notna().all().all(), "Split/status mismatch")
    table = table.sort_values("hour_bjt").reset_index(drop=True)
    ensure(table.hour_bjt.is_unique and np.isfinite(table[c.TARGET]).all(),
           "Invalid evaluation hours/targets")
    return table


def predict_all(split: str) -> None:
    ensure(LOCK.exists() and read(LOCK)["status"] == "TRAIN_LOCKED",
           "All training models must be locked before evaluation")
    plan = read(PLAN)
    manifest = train_manifest()
    fingerprint = manifest["fingerprint"]
    ensure(read(LOCK)["fingerprint"] == fingerprint, "Training lock fingerprint changed")
    if split == "test":
        ensure((ROOT / "validation_qa.json").exists() and
               read(ROOT / "validation_qa.json")["status"] == "PASS" and
               read(ROOT / "validation_qa.json")["fingerprint"] == fingerprint,
               "Validation diagnostic QA must pass before test")
    names = feature_names()
    frame = load_evaluation(split, names)
    train_hours = pd.read_parquet(c.DATA / "train.parquet", columns=["hour_bjt"])
    ensure(set(frame.hour_bjt).isdisjoint(set(train_hours.hour_bjt)),
           "Evaluation/training hours overlap")
    if split == "test":
        validation_hours = pd.read_parquet(c.DATA / "validation.parquet", columns=["hour_bjt"])
        ensure(set(frame.hour_bjt).isdisjoint(set(validation_hours.hour_bjt)),
               "Test/validation hours overlap")
    positions = np.arange(len(frame), dtype=int)
    base_meta, base_model = checked_model(BASE_MODEL, BASE_META, fingerprint)
    base = np.asarray(base_model.predict(impute(c.raw(frame, positions, names[:188]),
                                                base_meta["medians"])), dtype=float)
    ensure(np.isfinite(base).all(), "Invalid baseline predictions")
    output_dir = ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)
    data_hash = support.sha256(c.DATA / f"{split}.parquet")
    new = 0
    for item in plan["mask_plans"]:
        mask = item["mask"]
        path = output_dir / f"mask_{mask:04d}.json"
        if path.exists():
            record = read(path)
            ensure(record["fingerprint"] == fingerprint and
                   record["data_sha256"] == data_hash and record["mask"] == mask,
                   f"Existing {split} result mismatch: mask {mask}")
            continue
        prediction = base.copy()
        if mask:
            model_path, meta_path = residual_paths(mask)
            meta, model = checked_model(model_path, meta_path, fingerprint)
            raw = c.raw(frame, positions, item["features"])
            prediction += item["alpha"] * np.asarray(model.predict(
                impute(raw, meta["medians"])), dtype=float)
        ensure(np.isfinite(prediction).all(), f"Invalid {split} prediction: {mask}")
        record = {"fingerprint": fingerprint, "split": split,
                  "data_sha256": data_hash, "mask": mask,
                  "levels_hpa": item["levels_hpa"], "features": item["features"],
                  "alpha": item["alpha"], "formal_rounds": item["formal_rounds"],
                  "n": len(frame), "prediction_db": prediction.tolist(),
                  "metrics": c.metrics(frame, positions, prediction),
                  "completed_utc": support.utc_now()}
        support.atomic_json(path, record)
        new += 1
        if mask % 100 == 0 or mask == 1023:
            print(f"{split.upper()} mask={mask}/1023 new={new}", flush=True)
    print(json.dumps({"split": split, "new_records": new,
                      "total_records": len(list(output_dir.glob("mask_*.json")))}), flush=True)


def audit_split(split: str) -> dict:
    plan = read(PLAN)
    manifest = train_manifest()
    fingerprint = manifest["fingerprint"]
    ensure(LOCK.exists() and read(LOCK)["fingerprint"] == fingerprint,
           "Training lock missing/mismatched")
    frame = load_evaluation(split, feature_names())
    truth = frame[c.TARGET].to_numpy(dtype=float)
    output_dir = ROOT / split
    ensure(len(list(output_dir.glob("mask_*.json"))) == 1024,
           f"{split}: expected 1024 records")
    data_hash = support.sha256(c.DATA / f"{split}.parquet")
    max_metric_diff = 0.0
    max_replay_diff = 0.0
    for item in plan["mask_plans"]:
        mask = item["mask"]
        record = read(output_dir / f"mask_{mask:04d}.json")
        ensure(record["fingerprint"] == fingerprint and
               record["data_sha256"] == data_hash and record["mask"] == mask and
               record["features"] == item["features"] and
               record["alpha"] == item["alpha"] and
               record["formal_rounds"] == item["formal_rounds"] and
               record["n"] == len(frame), f"{split}: mask metadata mismatch {mask}")
        pred = np.asarray(record["prediction_db"], dtype=float)
        ensure(len(pred) == len(truth) and np.isfinite(pred).all(),
               f"{split}: invalid predictions {mask}")
        recalculated = c.metrics(frame, np.arange(len(frame)), pred)
        for group in c.GROUPS:
            for metric in ("rmse", "mae", "bias"):
                old, new = record["metrics"][group][metric], recalculated[group][metric]
                if old is None or new is None:
                    ensure(old is None and new is None, f"{split}: null metric mismatch")
                else:
                    max_metric_diff = max(max_metric_diff, abs(old - new))
            ensure(record["metrics"][group]["n"] == recalculated[group]["n"],
                   f"{split}: group count mismatch {mask}")
        if split == "validation":
            previous = read(OLD_MASKS / f"mask_{mask:04d}.json")
            old_pred = np.asarray(previous["validation_prediction_db"], dtype=float)
            max_replay_diff = max(max_replay_diff, float(np.max(np.abs(pred - old_pred))))
    ensure(max_metric_diff <= 1e-12, f"{split}: metric recomputation mismatch")
    if split == "validation":
        ensure(max_replay_diff <= 1e-6, "Refitted validation predictions differ from locked source")
    qa = {"status": "PASS", "split": split, "fingerprint": fingerprint,
          "n_schemes": 1024, "n_hours": len(frame),
          "all_metrics_recomputed": True, "max_metric_difference_db": max_metric_diff,
          "max_locked_validation_replay_difference_db":
              max_replay_diff if split == "validation" else None,
          "completed_utc": support.utc_now()}
    support.atomic_json(ROOT / f"{split}_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False), flush=True)
    return qa


def report() -> None:
    qa = read(ROOT / "test_qa.json")
    ensure(qa["status"] == "PASS" and qa["n_schemes"] == 1024,
           "Test QA must pass before report")
    records = {mask: read(ROOT / "test" / f"mask_{mask:04d}.json")
               for mask in range(1024)}
    rows = []
    for mask, item in records.items():
        row = {"mask": mask, "levels_hpa": ";".join(map(str, item["levels_hpa"])),
               "n_levels": len(item["levels_hpa"]), "alpha": item["alpha"],
               "formal_rounds": item["formal_rounds"]}
        for group in c.GROUPS:
            for key in ("n", "days", "rmse", "mae", "bias"):
                row[f"test_{group}_{key}"] = item["metrics"][group][key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "test_subset_summary.csv", index=False)
    pairs = []
    for level_index, level in enumerate(c.LEVELS):
        bit = 1 << level_index
        for without in range(1024):
            if without & bit:
                continue
            with_layer = without | bit
            row = {"level_hpa": level, "without_mask": without,
                   "with_mask": with_layer, "other_level_count": without.bit_count()}
            for group in c.GROUPS:
                a = records[without]["metrics"][group]["rmse"]
                b = records[with_layer]["metrics"][group]["rmse"]
                row[f"test_{group}_rmse_improvement_db"] = None if a is None or b is None else a - b
            pairs.append(row)
    ensure(len(pairs) == 5120, "Expected 5120 layer pairs")
    paired = pd.DataFrame(pairs)
    paired.to_csv(ROOT / "test_layer_pairwise.csv", index=False)
    summary = []
    for level in c.LEVELS:
        part = paired.loc[paired.level_hpa.eq(level)]
        row = {"level_hpa": level, "paired_contexts": len(part),
               "single_layer_test_overall_rmse_db":
                   records[1 << c.LEVELS.index(level)]["metrics"]["overall"]["rmse"],
               "base188_test_overall_rmse_db": records[0]["metrics"]["overall"]["rmse"]}
        row["single_layer_improvement_db"] = (row["base188_test_overall_rmse_db"] -
                                                row["single_layer_test_overall_rmse_db"])
        for group in ("overall", "rain", "heavy_rain", "dry_overcast"):
            values = part[f"test_{group}_rmse_improvement_db"].dropna()
            row[f"{group}_mean_improvement_db"] = float(values.mean())
            row[f"{group}_median_improvement_db"] = float(values.median())
            row[f"{group}_fraction_improved"] = float((values > 0).mean())
        summary.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(ROOT / "test_layer_summary.csv", index=False)
    lines = ["# 1024 组逐层重训：测试集结果", "",
             "本轮 1024 种方案在读取验证/测试数据前已全部以 2404 个训练小时拟合并锁定。",
             "验证仅作诊断；测试集 687 小时只用于报告，不据此重新选择模型、层、轮数或 α。",
             "历史上该本地测试集曾被查看；本结果证明本轮代码没有数据泄漏，不称全新独立盲测。", "",
             "正贡献 = 不含该层 RMSE − 含该层 RMSE（dB）；正数表示该搭配下加层更准。", "",
             "| hPa | 单独加入该层的测试 RMSE | 相对 Base188 改善 | 512 搭配平均改善 | 改善搭配比例 | 雨平均改善 | 强雨平均改善 | 干燥阴天平均改善 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| {level_hpa} | {single_layer_test_overall_rmse_db:.4f} | {single_layer_improvement_db:+.4f} | {overall_mean_improvement_db:+.4f} | {overall_fraction_improved:.1%} | {rain_mean_improvement_db:+.4f} | {heavy_rain_mean_improvement_db:+.4f} | {dry_overcast_mean_improvement_db:+.4f} |".format(**row))
    lines += ["", "完整 1024 组总体与全部天气 RMSE/MAE/bias：`test_subset_summary.csv`。",
              "5120 个成对差值：`test_layer_pairwise.csv`。逐小时预测：`test/mask_XXXX.json`。",
              "训练配置、逐轮曲线和填补参数：`models/residual/mask_XXXX.json`；训练锁：`TRAIN_LOCK.json`。",
              "测试集雨、强雨、阴天分别只有 51、11、158 小时；雪只有 2 小时，仅作个案。",
              ""]
    (ROOT / "TEST_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "REPORT_WRITTEN", "schemes": 1024,
                      "pairs": 5120, "base_test_rmse": records[0]["metrics"]["overall"]["rmse"]}), flush=True)


def status() -> None:
    print(json.dumps({"plan": PLAN.exists(), "train_lock": LOCK.exists(),
                      "residual_models": len(list(RESIDUAL_MODELS.glob("mask_*.ubj"))),
                      "validation_records": len(list((ROOT / "validation").glob("mask_*.json"))),
                      "validation_qa": (ROOT / "validation_qa.json").exists(),
                      "test_records": len(list((ROOT / "test").glob("mask_*.json"))),
                      "test_qa": (ROOT / "test_qa.json").exists(),
                      "report": (ROOT / "TEST_RESULTS.md").exists()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "validation", "validation-qa",
                                            "test", "test-qa", "report", "status"))
    command = parser.parse_args().command
    if command == "status":
        status()
    else:
        with c.writer_lock():
            if command == "plan":
                plan = build_plan()
                print(json.dumps({"status": "PLAN_FROZEN", "schemes": len(plan["mask_plans"]),
                                  "base_config": plan["winner"]["best_config"]}))
            elif command == "train":
                train_all()
            elif command in ("validation", "test"):
                predict_all(command)
            elif command in ("validation-qa", "test-qa"):
                audit_split(command.removesuffix("-qa"))
            else:
                report()


if __name__ == "__main__":
    main()
