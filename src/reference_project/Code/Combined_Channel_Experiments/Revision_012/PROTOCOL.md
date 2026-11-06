# T012 七类名单核验与冻结结果重评分 — Revision 012

2026-10-17；当前人工决定和 tvt 七类比较提示优先。此版本只读复用，不训练、不重新推理、不下载天气，不改变任何历史评价。

## 执行与隔离

Execution Lock: LOCKED at preparation; holder /root, thread 01a0bda1-92d1-7832-9508-60f4e0ec1634. Release is recorded in the new HANDOFF. 2026-10-17 12:17:29 / 12:17:32 +08 elevated CIM succeeded; both Python process lists were empty. Earlier sandbox CIM failure is not absence evidence. Other unregistered sessions are not asserted conflict-free.

READ: frozen Revision011 predictions/QA, their protected sources, Revision008 weather classification, Revision010 full-cohort inputs, Revision002 frozen model register/checkpoints, LOCAL original test table and raw ERA5 campaign manifest/one representative header, supplied RMOP PDF. WRITE ONLY: this new Revision012 code directory and Outputs/Experiments/Combined_Channel_Revision_012. Both absent at preflight. New arithmetic uses its own exclusive OS lock and exclusive output creation. No old entrypoint is invoked. All previous failures/results remain untouched.

C1/C2: separate new paths and input/output SHA. C3: no frozen edits. C4/C5: finite one-shot offline audit and scoring, then independent QA. C6: CPU arithmetic only, two BLAS threads, no GPU/models loaded. C7: exact station-hour/UTC/target/weather/mask alignment. C8: reuse complete predictions. The one expected historical guard change is tvt configuration, documented in Outputs/Automation_Changes/TVT_Comparison_20261017_001; STOP must remain unchanged.

## Roster and estimand

Exactly seven families: proposed frozen Top10, Bai LSTM, TabM, combined ITU approximation, E-SST, RMOP, NWP–propagation chain. Only the first four have verified full-cohort predictions at preparation. The last three remain explicit blocked rows with null metrics, not zeros or ranks. Bai MLP, generic screening models, TabPFN and TabICL are absent from new score outputs. P.618 hourly sensitivity is not an eighth family.

The first three use ten contemporaneous weather features, without measured channel history or labeled training examples at inference. Bai LSTM and TabM use existing LOCAL-adapted frozen runs, not original author pretrained task weights. Existing seeds 17/29/43 are all retained; family metrics are means of run-level metrics, not a newly formed prediction ensemble or a selected seed. SD is sample SD across runs, not a confidence interval or independent climate replication.

Target: contemporaneous hourly relative-clear-sky attenuation in dB. LOCAL original test has 687 hours, UK full completed source dates have 5664 station-hours (Chilbolton 143 days, Sparsholt 93 days). Original label-valid mask and physical-input mask remain separate; common mask is expected 5538. All 5664 UK records receive a ledger entry, including invalid/excluded records. No silent reduction to the 39-day historical subset. Calendar gaps are retained. N counts station-hours, not independent samples; reported ranks are descriptive, with no new significance test.

Primary events are rain and dry_overcast, inherited from the frozen classification and independently checked from 3600*MTPR, 3600*MSR and TCC for UK: rain iff total >=0.1 mm and snow <0.1 mm; dry_overcast iff total <0.1, snow <0.1 and TCC >=0.8. These are hourly weather classes, not labels assigned to entire calendar days. LOCAL uses the pre-existing flags, checked against its earlier frozen prediction table. All full-cohort dates remain in coverage, even if they have zero eligible hours for an event.

RMSE=sqrt(mean((prediction-truth)^2)), MAE=mean(abs(prediction-truth)), Bias=mean(prediction-truth). Four scored families use exactly the same mask per cohort/event. Rank by unrounded mean RMSE; ties receive competition ranks. LOCAL, pooled UK, and each UK station are separate cohorts. A missing prediction on an eligible row fails rather than shrinking the mask.

## Physics and further-method gates

Combined ITU is reused byte-for-byte: relative gas (absolute gas minus approved fixed physical reference) + single cloud-liquid component + geometric-path rain. It is approximate, not full ITU probabilistic synthesis; scintillation absent, cloud/fog liquid water not counted twice, hourly rain not R0.01. Signed gas changes and existing invalid rain flags remain. LOCAL legacy rain/height preprocessing and geometry approximations are disclosed, not refitted.

E-SST hourly adaptation requires verified equations, actual 3600 s step, liquid rain units, wind translation, h0/receiver height, rain type, geometry and boundary handling. Sparse random LOCAL test hours cannot be concatenated into a storm trajectory; complete meteorological context (without channel labels) must be used if implemented. No minute observations fabricated. No default wind or guessed rain-type threshold.

RMOP requires the atmospheric attenuation branch of WRF/GSDSU, not SNR or symbol-rate optimization. Native output is hourly; 5-minute values are interpolation. Existing ERA5 headers/point features are not the original WRF/GSDSU chain. NWP2024 PRM integrates gas/cloud/rain extinction vertically then uses cosecant elevation; its paper-specific scattering coefficients require verification. Full MTS needs the original event library and is a stochastic, non-time-aligned statistical comparator. No adding project fast fading to full MTS; any common-fast alternative must be a separately named project hybrid with unchanged existing CDF.

Missing weather/author coefficients/event library are explicit blockers. Public source verification may continue; author contact or new weather acquisition needs new authority. The new input audit is bounded to named local assets, not proof that an asset cannot exist elsewhere. No complete seven-family claim or automatic task completion.

## Deliverables and validation

Exact selected frozen predictions, all-hour exclusion ledger, per-date coverage, run metrics, seven-family rank table with blocked entries, input availability audit, provenance matrix, dynamic-comparison gate, Chinese report and handoff. Independent verification uses scalar math.fsum (not production scoring), separately constructs cohorts/masks and weather labels, checks prediction identity, date counts, source SHAs, and malformed-input negative tests. Arithmetic QA is an independent implementation by the same agent, not an external scientific review. New outputs are published only after QA PASS.
