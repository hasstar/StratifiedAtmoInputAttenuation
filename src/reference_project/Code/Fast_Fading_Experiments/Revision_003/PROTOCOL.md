# Revision 003 — frozen fast generator + current TabM k10

Frozen before new evaluation, 2026-10-24. I selected TabM only and LOCAL validation/test plus both UK stations. The current submission's k10 state is used; no UK/test-based state or seed selection. Historical validation/test/UK data have been inspected previously: this is a descriptive supplementary evaluation, not a new blind test.

## Inputs and immutable lineage

- RFE: Hourly034 Final Delivery/delivery_002 paths and retained-feature dictionary, independently PASS QA. Export all 1560 states and 1550 deletions, baseline round 0; distinguish the feature removed to ENTER a state from removed_next which LEAVES that state. Preserve native ordered feature names and authoritative field metadata.
- Slow: saved Hourly032 tabm/k_0010/evaluation.json (343 validation,687 test) and Hourly037 all_hourly_predictions.parquet filtered tabm,k10 (5664 candidate,5593 eligible). Verify local state SHA and model-file SHA; no model calls or fits.
- Fast: Fast002/final_generator/model.json and its pinned weather empirical CDFs and training-only libraries. Reuse load_package/block_library/generate with fold=-1, block_0256, 32 repetitions; seed 20261007 with SHA-derived identifiers. LOCAL key is the existing D-final-hour key; UK key T016-UK-station-UTC-hour. No distribution re-estimation or UK templates. The existing empirical CDF is hour-balanced, with valid-record weighting within training hours; it is not a newly pooled all-seconds CDF.
- LOCAL: Fast001/clean_1s. A valid second has >=9 quality-zero 100ms records, finite positive power. Preserve all3600 positions; unobserved positions are NaN. F=-10log10(P_second/P_hour); slow truth must match the frozen clear reference (-76.19362933508154dBm) minus hour dBm.
- UK: native clearsky_attenuation 1s integration endpoints (t0,t1], valid time0<t<=86400, native fill/range checks. Existing primary_score_eligible mask unchanged; observed fast F=A_native-A_hour. Recompute A_hour=-10log10(mean(10^(-A_native/10))) and match saved truth. Existing236 daily source hashes must match.
- UK weather: apply the already-defined four-way rule to Combined008 saved p/s/TCC: rain p>=0.1mm,s<0.1; dry_overcast p<0.1,s<0.1,TCC>=0.8; clear_dry p<0.1,s<0.1,TCC<=0.2; remaining known values other. Keep original 3-way event alongside new generator_event; rain/overcast must match. No attenuation-derived weather, no extra download.

## Estimands and computation

Generate32 independent realizations, each3600Q samples with rank ordering from15 training blocks (14×256 +16 seconds), random tie breaking and circular phase. All synthetic seams remain in evaluation. Population E(Q)=1; do not force a finite realization's average Q=1 or meanF=0. Save all32 float64 curves, seeds and phase/provenance fingerprints for reproduction.

Per valid second, empirical-ensemble CRPS = mean_m |x_m-y| - sum_(m,n)|x_m-x_n|/(2M²), M=32, units dB. Use this for fast-only F and combined Ahat_TabM+F. Slow-only is a degenerate predictive distribution with CRPS=absolute error at the SAME native-second positions. Fast zero baseline similarly uses abs(F). Also save fair CRPS with M(M-1) in the second term as Monte Carlo diagnostic, explicitly not the primary reported score; no tuning.

Primary hour score is the arithmetic mean over valid seconds (exactly3600 only for complete hours). Missing seconds are not imputed, dropped across time, or assigned zero loss. Report n_seconds and exclusions. This differs from older valid-record-weighted201/401-quantile integrated CRPS; do not label them identical.

For each observed hour and each of32 generated curves, calculate SD(ddof0), ACF at1,2,5,10,30,60,120,300,600,900,1800s, full-interval valid pair masks, and fade episodes above frozen weather F95/F99 thresholds. Episodes touching gaps/hour boundaries are censored, not exact durations. Store complete episode lengths and censor counts; summarize per-hour mean complete duration with its nonempty support count, not arbitrary zero replacement. Time-order diagnostics are distinct from marginal CRPS. Slow additive constants cancel from SD/ACF/relative-fast fade durations; these are not evidence of slow-model accuracy.

Summaries: LOCAL validation/test separately; UK pooled and each station, overall and four weather strata. Primary descriptive mean gives each scored hour equal weight; date-equal sensitivity and paired date-cluster bootstrap95% intervals for combined-minus-slow CRPS (2000 replicates, fixed seed20261024, joint UK stations on sameUTCdate) are reported. CIs do not establish independence of adjacent dates. No inferential claims from treating seconds/32synthetic curves as independent observed replicates.

## Operational/QA gates

Single worker, 2-thread numeric limits, OS byte lock, no network/installs. Record all source/code/manifest hashes before execution; each completed hour has a SHA receipt. Preserve failures; do not rerun completed historical experiments or silently replace failed output. Resume only this version with all hashes matching, released lock, no new stop request, checked output receipts. Run IDs/timestamps append; status may update only within new output.

Unit controls: brute-force CRPS versus sorted formula, degenerate=MAE, translation/permutation invariance, missing-interval mask, constant shift cancellation. Independent audit: all stored per-second CRPS via a separate CDF-interval formula; all hour/aggregate counts and saved source matches; sampled native reconstruction, generator replay and independent dynamics. Numerical failure blocks delivery. Render and inspect each final plot and workbook sheet. Original submission4files remain byte-identical.

Reference checked2026-10-24: scoringrules official finite-ensemble CRPS documentation, https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html . No external library install or private data transmission.
