# Approximate ITU rain baselines — Revision 004

Frozen before physical calculation on 2026-10-16. Human authorization:
“雨层高度可以近似”; “近似 ITU 基线、没完成的请补算”.

## Conflict check / execution lock

Holder /root, original thread 01a0bda1-92d1-7832-9508-60f4e0ec1634.
WRITE only new Code/Combined_Channel_Experiments/Revision_004 and
Outputs/Experiments/Combined_Channel_Revision_004; fixed T012 and Revision001
PROGRESS may receive a new handoff after complete before snapshots. All historical
files, labels, predictions, model weights, normalization, clear-sky references,
failed attempts and weather acquisition state remain read-only.
Elevated CIM at 16:40:33.0440503 and 16:40:36.1628495 +08:00 found zero Python
processes, zero missing command lines. This is a recorded tool observation, not a
claim about unobserved external writers. A new exclusive OS byte lock covers work.
C1/C2: new paths, source hashes, byte-preserving fixed-index snapshots;
C3: immutable executed code, version any repair; C4/C5: latest human scope above;
C6: serial arithmetic, at most two numerical threads; C7: standard formulas,
source-height datum, native headers, common sample mask and independent QA;
C8: reuse all ML and LOCAL physical predictions, no training or inference rerun.

STOP_REQUESTED stays intact and tvt remains PAUSED. No ERA5/NCAR/CEDA retrieval or
download entrypoint. Four bounded anonymous GETs of official ITU method references
(P.838-3, P.839-4, P.618-14 PDFs and P.839-4 static climate-map ZIP) are allowed;
these are reference acquisition, not a restart of stopped weather acquisition.
No credentials, proxy changes, global weather files or background download jobs.

## Inputs and preregistered approximations

Use all 5664 already audited expanded UK station-hours (Chilbolton143 days,
Sparsholt93 days). Reuse Revision003 labels, eligibility and all20 saved ML runs.
Mean liquid water-equivalent rate R=3600*(MTPR-MSR), numerically mm/h for these
one-hour means. MTPR/MSR valid end is t+1h for label [t,t+1h). This is reanalysis,
not gauge rain or one-minute R_0.01. Preserve both raw rates and negative
differences. Negative/nonfinite R is invalid for ITU power laws, never clipped to
zero or imputed. Positive finite R and exact zero are valid. Labels are not clipped.
Primary common comparison is prior eligibility AND physical-valid rate. Expected
5593-55=5538 scored hours; report exclusions and per-station/date counts. Keep the
original5593 metrics unchanged, and rescore every saved ML run on the same5538.

Use P.839-4 official h0 grid at each station, bilinearly interpolated as stipulated
by the standard (NOT spatial interpolation of weather data). h0 is annual mean
0-degree height above mean sea level; hR=h0+0.36 km. Main hR plus predeclared
-0.5/+0.5 km sensitivity; this bracket is an assumption range, not uncertainty CI.
Subtract native receiver AMSL height84/118m to obtain vertical rain depth. Do not
transfer LOCAL medians. Frequency20.7GHz; frozen station/satellite nominal WGS84
geometry from the audited plan, elevation about28degrees. Confirm native metadata.
UK polarization is not verified: retain horizontal, vertical and circular at every
height. Circular/main-height is a fixed reference scenario for continuity with
the earlier LOCAL baseline, NOT a claim of true UK polarization or a selected best.
No tuning of height, polarization, rain thresholds, dates or seeds using outcomes.

## Two explicitly approximate deterministic comparators

1. P.838-3 coefficients: gamma_R=k*R^alpha (dB/km), with frequency/elevation/
polarization mixing per equations4/5. Geometric rain-only baseline
A_geom=gamma_R*(hR-hs)/sin(theta), for theta>=5degrees and hR>hs; zero rain gives
zero attenuation, rain below receiver gives zero path. No gas/cloud/scintillation
component is silently added. Uniform hourly surface rate along the slant path is
an approximation, not the complete P.618 annual method.
2. P.618-14 steps6–9 with the same hourly gamma_R, giving reduced/effective path
attenuation. It is labelled P618_hourly_proxy: replacing stipulated one-minute
R_0.01 with an hourly mean does NOT produce standard annual A_0.01 or exceedance
probabilities. Report separately as exploratory, no annual performance claim.
All2*3*3=18 UK physical prediction columns retained, not just the lowest error.

LOCAL687 rows reuse the existing circular P838 geometric and P618 hourly proxy
predictions exactly. The LOCAL baseline used its previously documented hourly
zero-degree height, not the newly approximated UK climate height. No fit, predict,
model import, optimizer, reference/CDF/scaler/imputer re-estimation anywhere here.

## Scoring / verification

RMSE, MAE, signed bias and R2 on same finite hourly targets. UK all, each station,
original39, extension2003, station-extension and fixed rain-rate bands. Physical
comparators use additional rain inputs; label that fact rather than presenting
them as common-Top10 learners. LOCAL and UK not pooled. Family mean/SD across
the three already frozen seeds are descriptive; do not choose a seed by score.
Paired UTC-date bootstrap2000 draws, seed20261016, for main circular physical
scenarios versus proposed and train-mean controls, both stations kept together.
Exploratory unadjusted intervals; dates may be temporally dependent, two nearby
sites do not establish independent-site generalization.

Independent checks: official map CRC/shape/grid/corners/weights and interpolation;
ASL/AGL/km/m distinctions; P838 table controls and legacy pure-function parity;
separately coded scalar P838/P618 equations for every UK value/scenario;
native frequency/coordinates/altitude and WGS84 pointing; UTC intervals, rain
conversion, masks, unchanged ML values, per-group metric recomputation and source
SHA before/after. Keep failed stages and repair separately. Status stays pending
until verification passes. Save per-hour values, all scenarios, metrics, formulas,
source provenance, independent QA, Chinese report and handoff.

Annual P618, P1853/SST/Maseng–Bakken stochastic dynamics, missing complete physical
fields, and models using historical channel measurements remain task-incompatible
or input-unavailable, not fabricated into a contemporaneous hourly comparison.
