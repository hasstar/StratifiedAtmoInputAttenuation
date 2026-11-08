# Revision 028 — symmetric weather-aware direct refits

Authorized 2026-10-11: continue experiments, and restore 15-minute monitoring. Revision 027 paired seed audit was complete/PASS but NOT_MET: all three full BiLSTM rain scores worsened. That failed hypothesis stays read-only. This is one bounded, explicitly exploratory response, not an unlimited search on test scores for an expected conclusion.

## Reuse and scope

ADAPT Revision 014's ten estimator definitions, Revision 026's train-only preprocessing/loaders, and Revision 027's paired/date reporting. All source digests are checked. No new dependency, data resplit, target change, pressure quota, feature-count search, residual head, or inference masking experiment. Original H5/ERA5/Run_011, dataset, previous revisions and slow/fast frozen packages remain read-only. Write only Revision_028/new output. Dataset train2404/validation343/test687 hours; target clear_sky_attenuation_db. Full156 is exactly base50 + pressure106, fixed from physical screening. Weather masks never enter X.

## Hypotheses and bounded choices

1. Fixed full-derived settings may not be optimal for base inputs. Give BOTH inputs independent, identical eight-candidate searches.
2. Global MSE can trade off small rain/overcast groups. Compare unweighted fits against weather-aware weights; weights depend only on training membership/counts, not target magnitude or test outcomes.
3. More pressure inputs can overfit. Compare stronger regularization, and for structured networks a learned base-context pressure fusion gate. Gate is not an imposed pressure contribution or a residual learner. No guaranteed gain.

Four original candidate configurations are the uniform-weight control. Four extended configurations pair the same grid with balanced training weights and stronger regularization; vertical neural variants also use the gate. All candidates are selected with the SAME predeclared validation criterion:

S = sqrt((MSE_overall + MSE_rain + MSE_dry_overcast)/3).

I give the three selected endpoints equal objective weight in this score. It is not the test ranking rule, not a criterion that forces a positive layer effect, and not evidence that these are optimal physical weights. For n fitting hours, normalized balanced sample weights are w_i=(1+n*I_rain/n_rain+n*I_overcast/n_overcast)/3, mean1. Thus mean(w*error^2) is exactly the three-MSE objective. Uniform control w_i=1. No validation/test row is used to compute training weights, median, scale, target normalization or gradients.

Grid constants and reasons are in PARAMETERS.md. Structured full models receive original348-slot layout, nonretained entries neutral after training imputation/scaling, all ten pressure tokens. Base structured models receive188 slots and no pressure tokens. Flat models receive exactly50/156 columns. The gate sigmoid(W*base_embedding+b) multiplies the learned pressure representation before direct concatenated prediction. It may shrink/retain information naturally, does not force rain improvement, and remains an architecture variant of the named family, not a replication of TFT.

## Training and isolation

Search: 2 pools x10 families x8 configs =160 fresh fits, seed17. Save full config, feature names, selected-round predictions, metrics, all per-round train/validation curves, actual completed/selected round, stopping reason, warning audit and elapsed time. Validation selects each pool/family's configuration and round independently. Neural search caps160 epochs, patience20 unsuccessful epochs, tolerance1e-12 on validation S, restore the earliest best checkpoint. Nonneural trees fit the preset cap, retrospectively select the earliest minimum-S prefix; do NOT call that stopping the fit at the selected tree. Solvers record actual iterations and convergence/warnings, not invented epochs.

Then freeze all20 plans and SHA256. Refit 20 plans x3 seeds17/29/43 =60 fresh models on2404 train hours only, at fixed selected rounds. Do not fit on train+validation, do not use validation/test for refit stopping. Validation diagnostics after refit cannot revise the plan. Save every weight, complete feature list, training curves and hourly predictions. Expected total220 fits. QA PASS before test. Resume only missing checkpoints, same fingerprint, distinct recovery logs; live matching worker only permits read-only health checks. OS lock protects stages. Never change frozen code/plan to rescue an unfavorable result.

## Final test, ranking and acceptance

Only after train QA PASS load60 frozen models once for the same687 test hours; zero fitting. Save all predictions and RMSE/MAE/Bias for overall/clear/rain/dry-overcast/other. Rain includes all rain. Report mean of three seed-specific metrics, not metrics of a prediction ensemble. First rank the ten FULL families by pure overall mean RMSE, then check that first family's full RMSE is lower than its own independently tuned base RMSE for all three requested endpoints. Never first filter favorable families. Also require the new full winner not be worse than the old full benchmark mean0.28567316472555604 dB, so satisfying the comparison by degrading the competition is not success. Record MET/NOT_MET separately from execution COMPLETE.

Conditional paired uncertainty resamples whole observed dates with matched base/full hour multiplicities, fixed three seeds,4096 replicates,RNG20261011. Ordinary95% and three-endpoint Bonferroni percentile bounds; no claim these correct test-based ranking, repeated historical test access, temporal dependence or same-day train/test overlap. Exact hours are disjoint, but all test dates overlap training. This is exploratory, not a new blind or prospective temporal test. Test must not select a seed, a configuration, a layer, an epoch or a later retry. If NOT_MET, preserve all failures, finish the bounded report and ask for the next scientifically distinct study; do not automatically continue test-chasing.

## Gates

Pure-synthetic self-test and all-ten weighted API preflight PASS -> prepare -> search -> freeze (160 records) -> refit (60 records/weights) -> verify-train PASS -> test (60 records) -> independent verify PASS -> report -> independent delivery audit. Save paired metrics, full ranking, parameters, round histories, all weather counts, bootstrap intervals and source/output digests. Tests and acceptance are distinct. At end pause monitoring after all deliverables complete, explaining whether the objective was met.

## References and boundaries

Weighted fitting API: https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVR.html and https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.ElasticNet.html. Train-fitted transformations/test isolation: https://scikit-learn.org/1.7/common_pitfalls.html. Gating motivation: Lim et al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting, arXiv1912.09363, https://arxiv.org/abs/1912.09363. The local gate is a modest adaptation of gating as an unnecessary-component suppression mechanism; no claim to implement that paper's full time-series method. Objective coefficients and search multipliers are predeclared engineering choices with explicit provenance, not literature-derived optimality guarantees.
