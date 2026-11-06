# Expanded UK comparison — Revision 003

Frozen before expanded-cohort inference, 2026-10-16. Latest human instruction:
“请做完所有的工作”, following the request for LOCAL-test and expanded-UK comparisons.

## Conflict check and execution lock

Holder /root, current original thread. New code and outputs in Revision_003 only.
Elevated CIM at 15:54:42 +08:00 found zero Python processes in two samples.
No download, training, or old evaluation entrypoint will be started. STOP_REQUESTED
is protected; the existing tvt automation remains paused. New work has an exclusive
OS byte lock. Existing experiment outputs, source payloads, labels, checkpoints,
normalizers and reference definitions are read-only. Fixed T012/PROGRESS indexes
may receive an appended/prepended handoff only after byte-for-byte before snapshots.
C1/C2: new version paths and source SHA guard; C3: no frozen code changes;
C4/C5: current human authorization, bounded scope below; C6: two CPU threads,
serial inference; C7: independent raw-field/native-label gates and common keys;
C8: reuse LOCAL and old39 predictions; no training, parameter or seed selection.

## Cohort fixed without looking at expanded outcomes

Use the intersection of completed pressure/surface/rain acquisition-unit hours,
separately for each original station, requiring complete 24-hour source dates.
Expected counts from the user-stop receipt: Chilbolton 143 days/3432 hours,
Sparsholt 93 days/2232 hours, total 236 station-days/5664 station-hours.
This comprises old39 days per station plus 104/54 newly acquired 2003 dates through
November30. Preserve original native calendar gaps. December incomplete pressure
is excluded by acquisition completeness, not by targets or errors. No new GET.

Primary scoring requires a finite source-native hourly label and >=95% native
second coverage (>=3420/3600). Retain missing/lower-coverage rows, count reasons,
and report all-finite sensitivity separately. Native integration-end seconds are
assigned to (h,h+1h]; hourly attenuation is -10log10(mean linear transmittance).
No refitting/re-estimating clear-sky reference, imputation, scaling, bias or CDF.

## Field and geometry gate

Independently decode actual ASCII/NetCDF bytes, check SHA and available response
sizes, parameter/table/short name/units, dimensions/maps, UTC, pressure level,
quarter-degree grid and finite values. Compare decoded values with canonical
collection records; collection records alone do not establish field QA.
Check every station-hour at station-nearest surface and WGS84 forward LOS pressure
points; 1hPa uses its own hourly z, four other levels retain their declared fixed
heights. No spatial/temporal interpolation or missing-value replacement.
Top10 order and definitions are exactly the frozen k_0010 checkpoint. Raw MTPR
(235055) and MSR (235031) are diagnostic only, kg m-2 s-1, valid=t+1h for beacon
[t,t+1h). 3600*rate is water-equivalent hourly mm; liquid estimate is 3600*(MTPR-MSR),
negative differences retained and flagged. These are reanalysis, not rain-gauge data.

## Frozen inference and comparisons

Use exactly the existing 20 common-input run IDs from Revision002: proposed
natural-Top10 BiLSTM, train mean, Ridge, ElasticNet, SVR, and three fixed seeds each
for RF, XGBoost, Bai2022 MLP adaptation, Bai2022 zero-history LSTM adaptation and
official-source TabM2025 fixed-budget adaptation. Check checkpoint/preprocessing
hashes before/after. Only transform/predict are permitted. No fit/partial_fit/
fit_transform/backward/optimizer step, no external-label adaptation. Import only
needed model definitions/pure inference functions, never execute old main/run.
Reuse old39 prediction values verbatim after overlap checks. LOCAL687 saved
predictions are reused, not rerun. Expanded new-hour predictions are new inference.

Report LOCAL separately from UK. UK panels: all acquired dates, each station,
original39, new extension only, each month, and predeclared reanalysis liquid-rate
bands (<0, [0,.1), [.1,1), >=1 mm/h). Bands are diagnostic and not an observed
weather classification. Common scoring mask across all20 runs. RMSE/MAE/bias/R2
and target distribution are reported; no cross-dataset pooled metric.
Family mean/SD across fixed seeds is descriptive, not a confidence interval.
Paired UTC-date cluster bootstrap (2000 draws, seed20261016) compares each run
against proposed and train mean; both nearby stations on a date remain together.
These exploratory intervals are unadjusted for many comparisons, assume dates
exchangeable, and do not establish independent-site generalization or superiority.
No model, date, seed or hyperparameter changes based on these results.

## Physical and literature comparators

Reuse the two existing LOCAL circular-polarization P.838+geometric-path and
P.618-hourly-proxy results, separately from common-Top10 methods. UK has neither
audited zero-degree/rain-height inputs nor confirmed matching polarization;
do not replace them with LOCAL medians or arbitrary heights. Mark UK physical
attenuation evaluation unavailable for missing inputs. P.618 annual statistics,
P.1853/Maseng–Bakken/SST stochastic synthesis are different output tasks; do not
score them as contemporaneous hourly predictions. Models using past channel
measurements to forecast future channels are excluded by direct user instruction.
Document remaining candidates and exact input/reproduction boundaries, not fake
paper replications. This closes the current feasible hourly comparison package,
not every historical T012 goal or new fast/annual experiment.

## Verification and deliverables

Raw-field QA, native-label QA, per-value provenance, date coverage, frozen-source
manifest, per-hour predictions, all-run/group metrics, seed summaries, exploratory
date-cluster intervals, LOCAL/UK comparison figures, independent verification,
input/feasibility matrix, Chinese report and handoff. Independent verification
recomputes metrics and label masks, checks saved-source SHA, all model IDs,
old39/LOCAL equality, and frozen prediction replay on fixed non-outcome-selected
rows. Parser negative controls test identity/UTC/shape/finite/grid/SHA failures.
Failures are retained; no silent repairs or claims of completion before PASS.
