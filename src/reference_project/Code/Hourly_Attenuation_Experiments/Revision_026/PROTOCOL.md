# Revision 026 — frozen ten-model test benchmark

User-authorized scope (2026-10-11): evaluate all ten hourly models on the same existing 687-hour test partition, report overall/clear/rain/dry-overcast/other metrics, and rank them on test performance. Do not retune any model after test scoring. Preserve Revision 023–025, the slow/fast frozen packages, original H5/ERA5/Run_011, and the dataset unchanged. All new code/output belongs to Revision 026.

## Frozen inputs and models

- Dataset: Signal_Detectability_Revision_003; 2404 train, 343 validation, 687 test hours; target `clear_sky_attenuation_db`.
- Reuse exactly the 156 training-selected features in Revision 023: 50 base/column variables and 106 pressure-level variables. No new feature selection. Event labels are used only for reporting.
- Reuse each family's Revision 023 validation-selected configuration and stopping round, with seeds 17/29/43. The original ten-model worker discarded estimator weights after saving predictions. Therefore reconstruct all 30 estimators under these already-frozen settings, using training samples only. There is no new configuration search, stopping choice, or weight update using validation/test.
- Use the identical frozen Revision 014 model implementations and Revision 017/023 preprocessing. Median imputation, scaling, and neural target mean/scale are fitted on training only. Structured models retain neutral excluded slots in their original vertical layout; flat models receive only the 156 retained columns.
- Before test loading, require all 30 saved reconstructed models to reproduce their corresponding old validation predictions within 1e-6 dB maximum absolute difference, and saved/reloaded predictions within 1e-10 dB. The former tolerance is much smaller than the displayed 1e-4 dB precision and permits insignificant numerical roundoff, not model changes. Save per-round history, effective parameters, hashes, training/validation predictions, and all failures.

## Evaluation and ranking (fixed before new test scores)

After reconstruction QA PASS, load saved weights only. Evaluate each model/seed once on all 687 test hours, with no exclusions beyond the already-cleaned dataset. There is no fitting during evaluation.

For each seed and weather group compute hour-equal RMSE, MAE, and Bias = prediction minus observation. Rain merges all rain including strong rain. Clear, rain, dry-overcast must be disjoint; other is their complement. Expected test counts: overall 687, clear 322, rain 51, dry-overcast 158, other 156. Save all per-hour predictions and individual-seed results.

Family metrics are arithmetic means of the three seed-specific metrics, not metrics of averaged predictions or an ensemble. Also save seed standard deviations and seed-17 scores for comparison with previous single-seed reports. The three seeds do not constitute independent field observations.

1. Main requested performance ranking: ascending mean overall test RMSE; tie by mean overall MAE then stable model name.
2. Supplementary prior 50/50 rule: 0.5 × min-max normalized mean test RMSE + 0.5 × min-max normalized log(single-row latency). Latency is measured after training, including preprocessing and prediction, not training time. Use 20 warm-up calls and 300 timed calls (100 fixed evenly spaced test rows, three passes); family latency is the median of three seed medians. Tie by overall RMSE, MAE, name. Save all timing samples. Do not use new rankings to alter existing layer/RFE/fast packages.

The old two-model table used Revision 023's separately tuned layer SVR (C=10), whereas the frozen ten-model comparison selected SVR C=1. Preserve the old C=10 result as a labeled supplementary reference, not an eleventh member or substituted score. Confirm BiLSTM seed-17 test predictions reproduce Revision 025 full-input results.

## Gates and evidence boundary

prepare → refit → verify-refit PASS → evaluate → independent verify PASS → report/workbook. Require SHA-256 source preservation, 30 reconstructed weights, 30 training records, 30 test records, 20,610 test predictions, 150 model/seed/weather rows, and complete 156-feature names/configuration/curves. There is one operating-system byte lock for writers; never start duplicate workers. Failed stages resume under the same frozen manifest and preserve failure/log records.

This is explicitly a user-requested ranking on a historically accessed local test set. It is not a fresh blind test; choosing a family using these test scores is test-based model selection, even though no test sample enters parameter fitting. All 156 test calendar dates overlap training dates in this random-hour split. No unseen-day or causal conclusion is claimed, and no old test-access audit is deleted.
