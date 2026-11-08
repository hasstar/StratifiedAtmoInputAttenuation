# Revision 027 — paired 50/156-input benchmark

Authorized 2026-10-11: continue toward a model whose full-156-input test overall RMSE ranks first among ten families and whose full-input overall/rain/dry-overcast RMSE is below its own base-50-input RMSE. This is an empirical acceptance criterion, not permission to manufacture a positive result. This first stage corrects the old single-seed vs three-seed aggregation mismatch before trying any new tuning strategy.

## Reuse and scope

ADAPT Revision 026's QA-passed saved estimators, selected configurations and rounds, data loading, train-fitted preprocessing, and Revision 014 implementations. The registry and local Revision 023–026 code were inspected on 2026-10-11; no new package or external service is needed. Official references: scikit-learn 1.7 common pitfalls (https://scikit-learn.org/1.7/common_pitfalls.html), SciPy paired-bootstrap documentation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Only the paired record/verification/report adapter is new.

Raw H5, ERA5, Run_011, the clean dataset, all old revisions, and slow/fast frozen packages are READ ONLY. New files only in Revision 027 and its new output directory. No old output is overwritten. Source digests, process command lines, OS writer lock, complete records and failure audit are required. Scoped process/capacity checks found no hourly/fast worker and about 119 GB free on E before creation; this is not a permission to terminate unrelated processes. No new automation is implied by this protocol.

## Frozen controlled comparison

- Dataset Signal_Detectability_Revision_003: train 2404, validation 343, test 687 hours; target clear_sky_attenuation_db. No repartition, row removal or feature re-selection.
- Full list: the same 156 retained meteorological variables, 50 base/single-level/column variables plus 106 pressure-level variables. Base list: exactly the 50 surface__ entries in that same list. Full is not a new feature screen.
- Ten families and seeds 17/29/43 exactly as Revision 026. For each family, the base-input model receives the SAME configuration and fixed stopping round as its full-input model. These settings were selected on full-input validation in Revision 023; this is a fixed-setting ablation, not proof that each input pool received its own optimal search.
- Fit all 30 base estimators from scratch on train only. Median imputation, feature scaling, and neural target normalization fit train only. Flat models receive the 50 columns. Vertical models retain the 188-slot base layout with nonretained slots neutral after train preprocessing, and NO pressure tokens; their existing unused vertical modules remain instantiated to preserve the legacy initialization. This is new training, not zero-masking a trained full model.
- Validation predictions are recorded after fitting, not used to update weights or choose new stopping rounds. All existing full weights and their test records are reused READ ONLY. Test targets are never read during base fitting. Test weather/event labels are reporting masks, not inputs.
- Complete 30 training records and weights and PASS fit QA before test scoring. Saved/reloaded base validation predictions agree to 1e-10. Base BiLSTM seed17 must reproduce Revision 025 mask0000 validation predictions to 1e-6 and, after test opening, test predictions to 1e-6. Full predictions must reproduce Revision 026 for every family/seed.

## Acceptance, metrics and uncertainty

At fixed rules, rank the ten FULL-input families by arithmetic mean of the three seed-specific overall test RMSEs (ties MAE/name). Then check the FIRST family, without filtering the ranking for favorable layer effects. Acceptance is the conjunction full_mean_RMSE < base_mean_RMSE for overall, rain and dry-overcast. A failure of any one endpoint is NOT MET. Do not change seed, model, row list or metric to rescue a failure.

Each seed's RMSE/MAE/Bias is hour-equal; family values are means of seed-specific metrics, NOT metrics of averaged predictions. Report every family and all weather groups, plus each seed's paired difference and standard deviation. Keep seed17's negative result even if the mean passes. Rain merges all rain; expected test groups are 322 clear, 51 rain, 158 dry-overcast, 156 other, disjoint and exhaustive.

Exploratory paired uncertainty: keep all three fitted seed instances fixed and resample whole observed calendar dates with replacement, using the SAME hour multiplicities for base and full. For each replicate recompute each seed's hour-equal RMSE then average the three base-minus-full RMSE differences. 4096 replicates and RNG20261011 are fixed computational choices. Save ordinary percentile95% intervals, and for the winner's three requested endpoints Bonferroni percentile bounds using alpha/(2*3). These conditional resampling intervals do not correct historical/test-based model selection, temporal dependence, or same-day train/test overlap; no confirmatory significance claim or new blind-test claim. Seeds are algorithm repetitions, not independent weather observations.

This existing test has historical access and all its dates overlap training. Any later new strategy must be specified separately, selected using train/validation only, retain all failures, and preserve this diagnostic as exploratory. Never iterate on test scores until an expected conclusion appears.

## Gates

prepare → fit-base → verify-fit PASS → evaluate → independent verify PASS → report. Expected 30 new weights, 30 base training records, 30 paired evaluation records referencing 30 full weights, 41,220 paired test predictions, 300 per-seed/input/weather rows, ten full-family ranked rows. Each state retains input names, effective config, rounds, train curves and train/validation/test hourly predictions. All source digests must remain unchanged. Report QA/criterion separately; a complete diagnostic with NOT MET is not fulfillment of my objective.
