# Revision 021: actual retrained recursive feature elimination (RFE)

This is a separate experiment from the fixed-model zero masks in Revision 020. It uses the same 200 retained meteorological features and XGBoost family selected in Revision 019, but at **every** feature count k=200,...,1 trains new models on the remaining k features. No eliminated input is replaced by zero. The 148 columns removed by the train-only correlation screen before Revision 019 are outside this RFE search.

## Data, target, and separation

- Target: same-hour `clear_sky_attenuation_db`. Use 2404 training, 343 validation, and 687 test hours from the same frozen `Signal_Detectability_Revision_003` split. Time, event flags, quality flags, measured history, and target-derived columns are never inputs.
- The ordered starting list is the 200 meteorological features in Revision 019 `feature_plan.json` (88 base and 112 from ten pressure levels). The original 348-to-200 correlation screen and its training-only reference medians are frozen and not rerun differently here.
- Each subset has its own median imputer fitted only on training rows. XGBoost does not use a scaler. The same training-fitted imputer transforms validation and, after the RFE path is frozen, test rows.
- The model configuration from Revision 019 is frozen: 300 maximum trees, depth 5, learning rate 0.03, squared-error objective, histogram growth, subsample 0.8, colsample_bytree 0.8, L2 regularization 1, two CPU threads. Use seeds 17, 29, and 43. The only changed input across k is the selected feature subset. Do not tune hyperparameters or select a subset by test score.
- At each k, fit three fresh XGBoost models on the same training hours. Monitor training and validation RMSE for every boosting round, with predeclared patience 30 and cap 300. The 343 validation hours determine the effective best round/early stop only; they do not directly rank features or choose the final k. Record the full curves, trained rounds, effective best round, stopping reason, training/validation predictions and metrics, all hyperparameters, seed, feature names, and fitted model/imputer digests.

## Recursive deletion rule

For each trained model at k>1, compute tree `total_gain` importance from only the trees through the validation-chosen effective best round. Normalize each model's nonnegative importance vector to sum to one, then average across the three seeds. Remove exactly the currently retained feature with the smallest mean normalized total gain. Exact ties follow the original 200-column order. Train from scratch at k-1 and recompute all importances. Thus RFE makes 199 deletion decisions and 600 three-seed fits overall, not 20,099 candidate subset refits. Save each k's full retained list, three-seed importances, mean importance, removed feature, and model references. This is model-dependent RFE; gain importance should not be read as a causal physical contribution.

## Gates and outputs

1. Freeze and independently verify source hashes, 200-column order, the Revision 019 winner/configuration, and input boundaries.
2. Run checkpointed RFE from k=200 through k=1. The `k_*.json` state is written only after all three model artifacts for that k are complete and verified. `verify-rfe` checks 200 states, 600 training records/models, 199 exact one-column deletions, every score/importance vector, best-round selection, full feature names, and immutable fingerprint. The test split is not read during RFE.
3. Only after RFE QA PASS, load the 600 saved fitted models and their train-only imputers, score each on the same 687 test hours, and save 600 per-seed test prediction records with overall/rain-including-heavy-rain/dry-overcast metrics. Do not use these scores to revise deletion order, choose k, or retune. `verify-test` recomputes metrics and checks model digests and all feature sets. Produce a 200-row table, per-step feature CSV, and detailed result report.
4. The local test set was previously accessed and used to select XGBoost in Revision 019. This run is an exploratory model-path comparison, not a new blind estimate. Choosing a final sparse k from its test curve would create further test-selection bias; a deployable sparse model requires a separate selection criterion or new independent data.

All new files stay in Revision 021 code/output directories. Original H5, ERA5, Run_011, dataset revisions, and all earlier experiments remain read-only. Never launch a duplicate worker while one is alive; resume a failed stage only after confirming process exit and matching checkpoints/fingerprint.
