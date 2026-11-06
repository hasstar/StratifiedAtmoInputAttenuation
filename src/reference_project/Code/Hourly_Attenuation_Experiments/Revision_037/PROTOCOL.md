# T015 — UK frozen LOCAL RFE transfer, k=10..1

2026-10-20, same user thread. User explicitly requested frozen inference for ten models on the existing UK dataset and then explicitly approved targeted acquisition of the missing atmospheric inputs and necessary supporting fields. This does not resume any old download queue or authorize UK training.

## Fixed contract

- Freeze the delivered T014 model lineage (all 6240 roles); use only the 100 full-training roles (fold=0) at k=10..1 for inference. Keep each model's own ordered RFE path. Never retrain, refit imputers/scalers, change weights, select a UK-optimal k, or tune to UK labels.
- Input cohort remains Combined003/010: 5664 station-hours, 236 station-days, Chilbolton 143 days and Sparsholt 93 days. Primary scores use the existing 5593 eligible native attenuation labels. Do not use the ITU 5538-hour mask for these ten ML models. Keep the 71 quality exclusions and actual UTC dates.
- Weather classification is the frozen Combined008 proxy: rain total-water>=0.1 mm and snow-water<0.1 mm; dry_overcast total-water<0.1, snow-water<0.1, TCC>=0.8. Overall includes all weather. Rain rates have valid UTC=t+1h; instantaneous analysis at t. Existing labels/columns/masks are read-only.
- Six requested metrics: overall/rain/dry_overcast coefficient of determination R²=1-SSE/SST and RMSE dB. Negative R² retained; insufficient/constant-target groups null with reason. Predictions and n/date coverage retained. Test-derived comparisons are descriptive; this UK cohort was already inspected in earlier studies.
- Model input union is determined from the ten frozen k10 states, not a shared Top10. Existing UK data supply 11 of the 38 required fields; 27 require whole-cohort supplementation. Do not replace entirely unacquired fields with medians or zero. Missing metadata/geometry/coverage stops affected computation.

## Targeted acquisition

Reuse verified UK station geometry, nearest 0.25° publication grid, existing source receipts and pure parsing functions. Existing pressure input convention is forward LOS sampling: 1 hPa uses its already verified hourly geopotential-based point, other levels use the established standard-atmosphere fixed-height approximation. Extend the same analytic standard-height definition to newly required pressure levels, explicitly documenting that these are not observed hourly heights. Do not use a station-vertical column or interpolation as a silent substitute.

Only public NCAR GDEX/ECMWF metadata and bounded exact OPeNDAP subsets for the existing cohort and required variables are authorized. No credentials, uploads, paid services, new environment, whole global grids, or old queue resumption. Archive source URL, raw response, metadata identity, UTC/level/grid, SHA and actual byte count. Anonymous GET, no credentials, no redirects, bounded timeout/retry. Existing old STOP_REQUESTED files remain unchanged. This new output may have its own stop marker.

## Conflict / reuse / lock

Root E:/0_Workflow_System/README.md was fully read this turn; SHA 39741aa8c69136661b4b02948b29714a3654f3f989ab7b488a54f0c02f93f7c7. Read T014, WF01, final034 handoff/QA, 032/033 protocols, Combined003/008/010 and existing UK input audit025. Control-center registry/blockers reviewed; they do not establish resource availability. Host-level read-only CIM on 2026-10-20 ~10:08 +08 returned no Python processes. Unknown unregistered sessions remain UNKNOWN.

ADAPT/COMBINE: reuse immutable T014 bundle loader and complete model SHA table, UK native labels/weather and field-provenance tables, existing NCAR exact-subset transport/parser design, existing six-figure styling. No new model library or external research is needed. Official archive metadata will be checked only for missing field acquisition.

C1/C2/C3 LOW: new Code/Hourly_Attenuation_Experiments/Revision_037 and Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE, both checked unoccupied. All historical code/data/model files READ ONLY with SHA guards. C4 MEDIUM: parameter identity, time semantics and LOS alignment must pass before prediction. C5 latest user authorization allows only frozen UK inference and bounded input completion. C6 single low-thread local process, single anonymous request stream; no GPU. C7 missing input gate unresolved until acquisition and independent QA. C8 reuse the accepted data and weights, never rerun RFE. Decision PARALLEL_ALLOWED only within this isolated write domain; no second worker.

Execution lock is an OS byte lock in the new output. Freeze has no network or model execution. Subsequent stages must preserve failed attempts and source snapshots; no frozen032/033 edits. Completion requires 100 valid states, independent numerical/input/model QA, detailed result tables, predictions, six figures (ten curves each, TabM thick blue/cyan glow), Chinese report, and post-run SHA verification. Partial acquisition or partial inference is not completion.
