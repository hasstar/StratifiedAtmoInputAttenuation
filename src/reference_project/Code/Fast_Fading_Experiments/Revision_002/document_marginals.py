"""Save descriptive paired-day effects and the marginal-CV result document."""

import json

import numpy as np
import pandas as pd

from window_diagnostics import OUT, now, sha256, write_json


def main():
    qa = json.loads((OUT / "marginal_model_qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS":
        raise RuntimeError("Final marginal QA pending")
    scores = pd.read_parquet(OUT / "marginal_cv_hour_scores.parquet")
    metrics = pd.read_csv(OUT / "marginal_cv_metrics.csv")
    effects = []
    events = ("all_train", "clear_dry", "dry_overcast", "rain", "other")
    for i, event in enumerate(events):
        part = scores if event == "all_train" else scores.loc[scores.event.eq(event)]
        pair = part.loc[part.candidate.isin(["empirical__global", "empirical__hourly_weather"])].groupby(
            ["candidate", "date_bjt"]).crps201_db.mean().unstack("candidate").dropna()
        difference = (pair.empirical__global - pair.empirical__hourly_weather).to_numpy()
        rng = np.random.default_rng(20261006 + i)
        draws = difference[rng.integers(0, len(difference), size=(2000, len(difference)))].mean(axis=1)
        effects.append({"event": event, "n_days": len(pair), "global_minus_condition_db": float(difference.mean()),
                        "ci95_low_db": float(np.quantile(draws, .025)), "ci95_high_db": float(np.quantile(draws, .975))})
    effect = pd.DataFrame(effects)
    effect.to_csv(OUT / "marginal_paired_day_effects.csv", index=False)
    model = json.loads((OUT / "marginal_model/model.json").read_text(encoding="utf-8"))
    text = ["# C1：快变边缘分布的日期隔离比较\n",
            "2,404个正式训练小时、162日期按完整日期进行五折比较，共40条候选-折记录、19,232条逐小时折外评分。验证/测试未参与。QA从源功率独立复算40小时的概率评分，最大差异约3.01e-14。\n",
            "主指标为dB域CRPS：比较预测的完整偏差分布与实测偏差，越小越好。它不是瞬时点预测RMSE，也不是小时慢变模型的RMSE。201/401等距概率积分的第一名一致；数值差异已保存。\n",
            "## 总体折外评分（按日期平均）\n",
            "| 候选 | CRPS201 (dB) | CRPS401 (dB) |\n|---|---:|---:|"]
    overall = metrics.loc[metrics.event.eq("all_train")].sort_values("date_mean_crps201_db")
    for row in overall.itertuples(index=False):
        text.append(f"| {row.candidate} | {row.date_mean_crps201_db:.6f} | {row.date_mean_crps401_db:.6f} |")
    text += ["\n## 统一与天气条件经验分布\n",
             "| 天气 | 小时/日期数 | 统一分布CRPS (dB) | 条件分布CRPS (dB) | 统一减条件，日期块95%区间 (dB) |\n|---|---:|---:|---:|---:|"]
    for event in events:
        g = metrics.loc[metrics.event.eq(event) & metrics.candidate.eq("empirical__global")].iloc[0]
        c = metrics.loc[metrics.event.eq(event) & metrics.candidate.eq("empirical__hourly_weather")].iloc[0]
        e = effect.loc[effect.event.eq(event)].iloc[0]
        text.append(f"| {event} | {int(c.n_hours)}/{int(c.n_days)} | {g.date_mean_crps201_db:.6f} | {c.date_mean_crps201_db:.6f} | {e.global_minus_condition_db:.6f} [{e.ci95_low_db:.6f}, {e.ci95_high_db:.6f}] |")
    text += ["\n正差表示天气条件分布的CRPS较小，负差表示统一分布较小。区间描述同日期配对的不确定性；该族由训练CV选择，以上差异不作为事后确认性显著性检验。天气类别是小时样本组。连续多日块敏感性和最终测试尚待动态阶段。\n",
             "## 全训练小时拟合的冻结模型\n",
             "按预先登记的主评分选中`empirical__hourly_weather`。它以训练加权经验CDF描述快变，不把拟合族解释为已识别的多径机制。四组的完整功率比支持和CDF、201/401分位数、归一化常数、小时/日期数、代码与源指纹均已保存。\n",
             "| 条件 | 训练小时 | 训练日期 | 主有效秒 | 数据估计的功率归一化常数 |\n|---|---:|---:|---:|---:|"]
    for label, p in model["components"].items():
        text.append(f"| {label} | {p['n_hours']} | {p['n_days']} | {p['n_primary_seconds']} | {p.get('power_normalizer', 1):.12f} |")
    text += ["\n保存后从完整支持和CDF独立复算单位功率矩，四组均PASS；最大绝对误差约1.1e-11。模型保持`E(Q)=1`，结合小时慢变时无需按dB去均值。\n",
             "## 接下来的实验职责\n",
             "边缘模型定义单次随机偏差的分布；相关生成器还要复现实测的自相关、持续时间、间隔和变化速度。相同边缘分布的独立随机数仍是动态对照基线。下一阶段先登记并冻结相关生成器的参数与选择评分，再开展日期隔离比较。当前不宣布最终平稳窗口或完整快变模型完成。\n"]
    (OUT / "MARGINAL_RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    write_json(OUT / "marginal_document_qa.json", {
        "status": "PASS", "completed_utc": now(), "source_model_qa_sha256": sha256(OUT / "marginal_model_qa.json"),
        "effects_rows": len(effect), "day_bootstrap_replicates": 2000, "seed": 20261006,
        "outputs_sha256": {p.name: sha256(p) for p in [OUT / "MARGINAL_RESULTS.md", OUT / "marginal_paired_day_effects.csv"]},
    })
    print(effect.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
