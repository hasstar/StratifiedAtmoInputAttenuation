# Revision 019: test-ranked ten models and fixed-weight zero-mask analysis

This revision implements the experiment I revised. It does not train a residual model or enumerate 1024 pressure-layer subsets. The original H5, ERA5, Run_011, and dataset revisions are read-only. New outputs go only to `Outputs/Experiments/Hourly_Attenuation_Revision_019_Test_Ranked_Zero_Mask`.

## Data and frozen inputs

- Target: same-hour `clear_sky_attenuation_db`. The source dataset `Signal_Detectability_Revision_003` has 2404 training, 343 validation, and 687 test hours. Time, target, quality flags, and weather-event labels are never model inputs.
- Reproduce the 0.90 absolute Spearman train-only pruning from 348 meteorological columns; the expected retained list has 200 columns (88 base, 112 from ten pressure levels). Recalculate train-column reference medians and freeze the ordered list, source hashes, and references before fitting.
- Each retained raw meteorological column is shifted by its training median. For unscaled estimators, input zero represents the training median; for estimators with training-fitted standardization, zero at the *final model input* represents the fitted training center. Never interpret a mask as zero kelvin, zero pressure, or physically absent atmosphere. Pruned columns remain unavailable, not silently reintroduced.
- Imputation and any standardization are fitted only on the training split. The same training-fitted transforms are used for validation and test. The original 348-slot vertical geometry is retained for Transformer, GCN, and BiLSTM, with pruned slots filled by training-only imputation.

## Ten-model comparison

- Candidate families and four candidate configurations each are exactly those in Revision_014: Elastic Net, RBF-SVR, Random Forest, XGBoost, LightGBM, CatBoost, MLP, vertical Transformer, vertical GCN, and vertical BiLSTM. Seeds are 17, 29, and 43.
- Fit all 40 candidate configurations on the 2404 training hours. Use the 343 validation hours only to select each family's hyperparameters and stopping round, and to detect overfitting. Save per-round curves and validation predictions. Validation does **not** select the winning family.
- Freeze each family's configuration and round before reading `test.parquet`. Independently refit all ten frozen families on the training set at three seeds, then evaluate all ten on the same 687 test hours once per seed. Save predictions, overall and weather-event errors, and single-row inference latency (including preprocessing). The previously established ranking rule is 50% min-max-normalized test overall RMSE plus 50% min-max-normalized log latency; also report pure-RMSE order and the 25/75 and 75/25 weight sensitivities. No tuning, rerun, or feature choice may use a test score.
- The test set has been seen in historical experiments. Its role here is an explicit **benchmark comparison and model selection**, not a newly blind final generalization estimate. Removing historical files cannot change this fact.

## Frozen-model pressure-level sensitivity

- Fix the test-ranked winning family, its validation-chosen configuration and stopping round. Refit the same three seeds on training hours only. Verify each unmasked prediction matches the already saved test comparison prediction for the same seed.
- Do **not** retrain for any mask. At the final numerical model input, set the selected pressure-layer coordinates to exactly zero. Calculate 22 configurations: unmasked full input; all ten levels masked (base-only); each of ten levels masked individually from full input; and each of ten levels kept individually with the other nine masked. Retain the 88 base columns in every configuration. This is a local zero-reference perturbation, not a causal estimate of the physical effect of removing an atmospheric layer.
- Save every test-hour prediction and overall/rain-including-heavy-rain/dry-overcast metrics; report prediction shifts as well as error shifts. No weather event is a model input. Do not use the masked results to reselect the winning model or to describe the original test as independent blind confirmation.

## Gates and cleanup

1. Reproduce and independently verify the feature plan and train-only reference values.
2. Finish 40/40 candidate trials and 30/30 three-seed validation refits; verify their fingerprints, curves, and training/validation boundary. Freeze ten configurations.
3. Only then score 30/30 frozen model-seed pairs on the test set and generate the ten-model test ranking. Verify predictions and metric recomputation.
4. Only then perform and verify 22 zero-mask configurations at each of three seeds, with no retraining between masks. Generate a full result report.
5. Previous experiment directories are not deleted until I specify the exact target directories and the new run has passed QA. Preserve a minimal test-access audit record and any shared runtime dependencies required to reproduce this revision.
