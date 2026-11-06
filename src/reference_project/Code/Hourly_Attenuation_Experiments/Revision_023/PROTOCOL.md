# Revision 023 — physical-domain screen and train-only redundancy control

## Aim and evidentiary boundary

Test whether removing terrestrial/ancillary proxies and highly redundant ERA5 columns improves same-hour clear-sky-relative attenuation estimation and clarifies the predictive contribution of pressure-level atmosphere. The result is not required to favor pressure levels. Do not force any layer, column, feature count, model family, or performance sign. Never replace an unfavorable result with a test-driven filter adjustment.

Use the existing Signal_Detectability_Revision_003 split without alteration: 2404 train, 343 validation, 687 test hours; target `clear_sky_attenuation_db`. The random-hour test set was accessed in prior revisions and shares calendar days with training, so it is not a fresh blind or unseen-day generalization test. Original H5, ERA5, Run_011, datasets, and older experiment revisions are read-only. All new files belong to Revision_023.

## Feature eligibility, fixed before new model fitting

Begin with the original 348 same-hour ERA5 columns. No event flag, target, time, measurement, quality or history column is a predictor.

1. Apply a **name/physics-based exclusion** without consulting any target or split statistics. Exclude land-cover, vegetation, soil and soil-water, land albedo, skin/snowpack state, runoff, surface roughness/stress/exchange flux, and top/surface radiative-energy proxy columns. Keep precipitation and snowfall rates/amounts, atmospheric temperature/humidity, wind, pressure, clouds, atmospheric column contents/transport, boundary-layer and refractivity descriptors, and pressure-level atmospheric fields. Save the exact original column, exclusion class and rule for every decision. This filter restricts inputs to atmospheric propagation/weather state; it does not declare every retained variable causally important.
2. On **training weather columns only**, remove >50% missing or nonconstant failures. On the remaining columns calculate pairwise Pearson and Spearman correlations with at least 200 common finite training hours. Mark an edge when **either** absolute coefficient is at least 0.90. In a deterministic greedy independent-set pass, prefer fewer training missing values, then more distinct observed values, then a stable name hash. A rejected column records the selected representative, both correlations and common sample count. Retained pairs must have neither coefficient at or above 0.90 where estimable. No pressure level is protected from removal; no result-dependent exception is permitted. This removes strong linear/monotone pairwise redundancy but does not prove statistical independence or remove all nonlinear dependence.
3. Save original, physically eligible, quality eligible and final retained lists; every pairwise high-correlation edge; per-level counts; all source SHA-256 hashes; and an independent screen QA. No label, event, validation or test statistic enters this screen.

## Model and layer evaluation

After screen QA PASS, rerun the frozen ten candidate families and four configurations per family from Revision_014, using the same 2404 train and 343 validation hours. Flat learners see retained columns only. Structured vertical learners retain the fixed 188+10×16 input layout, but excluded slots are missing and become neutral values from training-only imputation/standardization. Save 40 configuration trials, 30 three-seed validation fits, complete parameters/round curves, per-hour predictions, overall/rain/dry-overcast metrics and single-row inference latency. The pre-existing validation ranking rule is 50% normalized overall RMSE plus 50% normalized log inference latency; also show pure RMSE ranking and all weather metrics. Test does not choose model family, configuration, or training round.

For the selected family, compare retained base-only inputs and retained base-plus-pressure inputs with comparable search budgets and training/validation protocol. Do **not** use residual heads or zero-masked inference as proof of layer value. Then undertake direct-refit layer-subset analysis from all naturally surviving pressure groups, without requiring any group to survive. Freeze the exact subset-selection method and final comparison before opening test. Candidate layer subsets are selected using training-only, day-disjoint folds; validation is used for final stopping/diagnosis, test for final metrics only. Save every subset, full feature names, training history and event-stratified metrics. If no pressure group remains after objective screening, report that outcome rather than changing the rules.

All rain metrics include strong-rain hours. Emphasize overall, rain and dry-overcast; do not optimize on test or choose the headline model by whichever test event happens to look strongest. Report paired day-block uncertainty for frozen test comparisons and explicitly state the same-day overlap limitation.

## Gates

`prepare` → `verify-features` PASS → ten-model worker → `verify-models` PASS → direct layer-refit worker → layer QA PASS → freeze final assessment → one test scoring stage → final QA and report. Running workers are inspected read-only and never duplicated; failed workers resume only with the same command and fingerprint. The heartbeat stops when all gates and documentation are complete.
