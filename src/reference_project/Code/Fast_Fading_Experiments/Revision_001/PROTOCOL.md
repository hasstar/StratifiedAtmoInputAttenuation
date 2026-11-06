# Local rapid-variation experiment — Revision 001

## Scientific question and separation of scales

The hourly model estimates the W-domain mean received power expressed as relative clear-sky attenuation. The rapid-variation experiment asks what **distribution and temporal dynamics** remain around that hourly mean at the measured site. It does not claim to predict a unique 100 ms realization from hourly ERA5, and it does not assume that every weather class shares one fluctuation law.

Let `P_t` be a quality-accepted calibrated 100 ms received power in watts and `Pbar_h` the W-domain mean of quality-accepted 100 ms powers in hour `h`. With the frozen training-only clear-sky reference `B` in dBm:

```
A_t = B - 10 log10(1000 P_t)                 [dB]
Abar_h = B - 10 log10(1000 Pbar_h)           [dB]
F_t = A_t - Abar_h = 10 log10(Pbar_h / P_t)  [dB]
Q_t = P_t / Pbar_h = 10^(-F_t/10)            [unitless]
E_h[Q_t] = 1                                [over the accepted records of hour h]
```

Therefore `F_t` need **not** have arithmetic mean zero in dB. A synthesized conditional fast process must be normalized in the **power-ratio** domain so that its expected `Q_t` is one before being combined with a predicted slow attenuation. Do not add a dB-centering correction and call it W-domain mean preservation.

## Immutable inputs and stage gates

- Read-only 100 ms input: `Outputs/Datasets/Fast_Variation_Revision_002/fast_100ms/*.parquet`. Its existing `fast_residual_db`, `normalized_power_ratio`, `normalized_envelope`, `slow_hourly_power_*`, and 1 s aggregates were built before the current detectability rule. **Do not use those fields as the formal residual or formal 1 s series.**
- Read-only official hourly target/split: `Outputs/Datasets/Signal_Detectability_Revision_003`, with 3,434 detected hours and 435 censored hours. The training-derived clear-sky reference is frozen in its `manifest.json`.
- Primary accepted 100 ms observations require `quality_code == 0`, physical validity, finite positive `peak_power_w`, and membership in a detected hour. A detected hour has at least 34,200 quality-zero records of the nominal 36,000 slots. Invalid/non-detected periods remain censored; never assign their receiver-background peaks exact attenuation labels.
- Stage 0 scans all monthly 100 ms partitions and independently recomputes per-hour quality-zero count and W mean. The computed count and attenuation must match the official hourly table within the predeclared tolerance. QA must pass before residual modeling.
- New files are confined to `Code/Fast_Fading_Experiments/Revision_001` and `Outputs/Experiments/Fast_Fading_Revision_001`; H5, ERA5, Run_011, existing slow experiments and existing fast dataset revisions remain unchanged.

## Analysis sequence after Stage 0 QA

1. Reaggregate accepted 100 ms observations to 1 s in W, keeping within-second count, coverage and censored gaps. Use 1 s for the primary time-series analysis and 100 ms for resolution sensitivity; never treat 100 ms points as independent weather observations.
2. Join hourly `clear_dry`, `dry_overcast`, and `rain` labels by `hour_bjt` only for conditioning and scoring; rain includes strong rain. Snow has too few independent hours for a standalone distribution. The event label is not an instantaneous 12:38 weather measurement.
3. Check the rapid-residual time scale with autocorrelation, low-frequency drift, and within-hour variation. Compare candidate non-overlapping windows of 1, 2, 5, 10, 15, and 30 min (60 min as the hourly reference). Do not assume 15 min is stationary. A window must not cross an hour, large data gap, equipment epoch, or held-out split boundary. Report eligible-window counts by weather and acquisition configuration.
4. Define an **operational local-stationarity window** from train-day evidence: adjacent-half changes in power-domain mean, scale, quantiles and autocorrelation versus their day-block bootstrap uncertainty, plus held-out train-day predictive scores. Freeze the rule/window before final held-out-day scoring. A distribution-only significance test on millions of correlated samples is insufficient.
5. Fit predeclared descriptive families to positive normalized envelope/power ratio and a heavy-tailed dB-residual alternative, with an empirical distribution baseline. Compare weather-conditional and weather-agnostic laws on unseen days using likelihood/CRPS or quantile calibration. Do not interpret a fitted Rician/Nakagami parameter as a physically identified multipath component without complex I/Q data.
6. Test dynamics as well as the marginal CDF: 1 s and 100 ms autocorrelation, threshold exceedance frequency, fade duration/interfade duration, and fade slope. If independent draws fail temporal metrics, fit or resample a temporally correlated process; a good histogram alone is not a valid 12:38 time-series generator.
7. Demonstrate synthesis as `A_generated(t) = A_slow_predicted(h) + F_generated(t | hour class, context)` with `E[10^(-F/10)] = 1`. Evaluate the slow model and the rapid process separately, then together. A generated sample is a stochastic scenario, not an exact instantaneous point forecast.

The temporal-dynamics endpoints align with [ITU-R P.1623](https://www.itu.int/rec/R-REC-P.1623-1-200503-I/en) (fade duration/slope). The time-series-synthesis framing aligns with [ITU-R P.1853-2](https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en). Neither recommendation is evidence that this site's 15-minute process is stationary; that must be tested here.

## Stage 0 output and acceptance

Run `python audit_alignment.py`. It writes `alignment_by_hour.csv` and `alignment_qa.json` in the new output directory. QA passes only if all 3,869 official measurement hours have matching quality-zero record counts, all 3,434 detected hours have a finite recomputed attenuation, the maximum target difference is at most 0.001 dB, and all 435 censored hours remain excluded from numerical fast-residual fitting. Any failure halts modeling and requires source diagnosis rather than a relaxed tolerance chosen from downstream model performance.
