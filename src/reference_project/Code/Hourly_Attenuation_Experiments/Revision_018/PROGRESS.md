# Revision 018 进度

- 2026-10-01：我将训练期成对绝对 Spearman 相关冗余阈值从 0.95 调整为 0.90；其他数据、十模型参数预算及分层训练规则保持不变。只读预检查预计保留 200/348 列，基础 88 列，十气压面均非空；正式数量以特征筛选文件及独立 QA 为准。
- 已创建本聊天每 15 分钟续跑任务 `tvt-0-90`。在完整结果及 QA 出来前，不得称为新实验完成。
- 正式训练集筛选已完成且独立 QA 为 PASS：保留 200/348 列，其中基础 88、十气压面 112；原始 `|rho|≥0.90` 的相关对 542 对，筛后最大 `|rho|=0.8999409187`；特征指纹 `dad9a4fc6615094470d7e1c9a92968bd21b6ce88b163e1982f50cb099c7e8d14`。
- 十模型短轮次 smoke 全部通过。正式 worker 于 2026-10-01 13:39 北京时间启动，初始 PID 26224，模型指纹 `0f88ecf788110794f63c4d25dc4474da00f658468629c68f3c2ec4c0168cfc8c`。当前只可只读监测；进程运行中绝不重复启动。
## 2026-10-01 14:05 (Asia/Shanghai) — model screening QA and layer analysis

- Ten-model stage finished: 40 candidate trials and 30 three-seed validation fits. Independent `verify-models` returned PASS with model fingerprint `0f88ecf788110794f63c4d25dc4474da00f658468629c68f3c2ec4c0168cfc8c`; the 50/50 validation RMSE and single-row latency rule selected XGBoost. Test data were not read in this stage.
- Layer preflight returned `PREFLIGHT_OK`, layer fingerprint `9b39b0820659235d2aee7171c5bfa8d31e100e198c346480496f83733deb2e10`; 200 retained features comprise 88 base and 112 across all ten pressure levels. Layer smoke returned `SMOKE_OK`.
- Started the hidden `layer_pipeline.py subsets` worker at 14:05:38, PID 14576. Command line and baseline cross-validation output were checked. The 1024-mask analysis is still running and must not be started again while this worker is alive.

## 2026-10-01 14:22 (Asia/Shanghai) — completed and independently checked

- The layer worker exited normally after 1024/1024 masks; stdout ended with `NEW_SUBSETS 1024`, stderr was empty. `report` selected mask 218 using training-only criteria: 850, 650, 500, 200, and 30 hPa.
- `verify` returned PASS for 1024 masks, 5120 paired comparisons, complete per-mask validation predictions, and layer fingerprint `9b39b0820659235d2aee7171c5bfa8d31e100e198c346480496f83733deb2e10`.
- The one planned 687-hour test evaluation was then run. `verify-all` returned PASS, including independently recomputed test metrics. `document_results.py` wrote `Outputs/Experiments/Hourly_Attenuation_Revision_018_Correlation_090/RESULTS.md`.
- Final test overall RMSE (dB): screened 88-base `Base188` 0.233816; screened 200-feature direct `Direct348` 0.217233; training-selected 5-layer residual `SelectedResidual` 0.240202; full ten-layer residual `Full10Residual` 0.234454. The direct model improved over screened base, but the residual-selected mask did not generalize better on this previously viewed test set. This is not a fresh blind test.
- All requested Revision_018 stages are complete. The recurring `tvt-0-90` monitor can be stopped.

