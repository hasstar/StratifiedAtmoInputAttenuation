# T012 comparison extension — Revision 002

Frozen before new fits, 2026-10-16, same user thread. This is an experiment
implementation, not a weather-acquisition restart.

## Authorization and isolation

Latest direct human instruction: “你直接进行对比实验……有开源代码的……找找开源代码……没有开源的……大致的复现……输入不一样……整理好……统一对比。开始吧”. This authorizes training NEW comparison models and their train-only preprocessing. It does not authorize changing/retraining the proposed frozen model, changing historical results, resuming stopped weather downloads, or fitting/calibrating on external evaluation labels. Earlier download-only/zero-fit instructions remain applicable to frozen Revision 001 artifacts, not to these newly requested comparison fits.

Execution Lock: LOCKED; holder /root; thread 01a0bda1-92d1-7832-9508-60f4e0ec1634.
READ: existing LOCAL Signal_Detectability_Revision_003, Rev029 k_0010 state and saved predictions, Rev016 saved physical predictions, independently audited GBS39 acquisition/labels/predictions, relevant prior code/protocols. WRITE ONLY: this new code directory and Outputs/Experiments/Combined_Channel_Revision_002. No old run/prepare/initialize entrypoint. New scripts use exclusive creation and an OS byte lock. C1/C2: new paths, SHA protection; C3: no old code changes; C4/C5: latest explicit experiment authorization, first finite batch specified below; C6: CPU two threads, no GPU, serial fits, bounded epochs; C7: common identities/targets and external field QA mandatory; C8: reuse old predictions rather than rerunning old evaluations. T006/T010 and download locks were released; elevated CIM showed no Python worker in preflight. Unregistered external sessions are not asserted conflict-free.

STOP_REQUESTED remains present and tvt remains paused. Public scholarly/code GETs are distinct from stopped meteorological acquisition. No credentials, authenticated session extraction, or data uploads.

## Scientific task and cohorts

Task A is contemporaneous hourly relative-clear-sky attenuation estimation in dB, NOT future forecasting. LOCAL labels are negative 10 log10 of mean linear transmission; original reference and random-hour split remain unchanged (train 2404, validation 343, test 687). Historical test outcomes have already been seen; this is not a new blind or time-separated test. All new selection uses validation only. Same-hour overlap across dates/splits is disclosed.

Common-input main comparison: precisely the ten names in frozen Rev029 rfe/states/k_0010.json; no rain columns added, no temporal signal history, no new feature selection, no external normalization. Proposed model is its frozen natural-RFE ten-feature checkpoint, NOT the full156 model padded with invented data. Its structural BiLSTM axis represents pressure levels, not elapsed time.

External cohort: the previously independently audited two-station39-day/1872-hour cohort, using exactly the existing primary eligible mask (1847 expected; finite native target and >=95% coverage). Both new models and frozen proposed predictions use identical station-hour keys. External labels never enter fit, early stopping, hyperparameter or seed selection. Report stations separately. GBS20.7GHz vs LOCAL19.13GHz, native vs LOCAL clear reference, and fixed-height nearest-LOS vs LOCAL dynamic-level bilinear meteorology are domain shifts, not physically calibrated equality.

The stopped download also contains 143 Chilbolton / 93 Sparsholt fully acquired days by collection-record intersection. Newly expanded days have NOT passed the same independent full-field evaluation gate and are NOT silently included in this first model-comparison batch. Their evaluation is a separate data-QA milestone. December partial-pressure days are excluded, not filled.

## Finite first batch fixed before fitting

1. Frozen proposed Top10 BiLSTM: reuse validation/test and GBS saved predictions verbatim, SHA-checked. No retraining or new inference.
2. Train-mean constant, Ridge, ElasticNet, RBF-SVR, RandomForest, XGBoost: common ten inputs. Three fixed configurations each for nonconstant controls, selected by validation RMSE; tie by candidate index. Seed17 for selection. For stochastic selected models, also seeds29/43 with no seed selection; report all runs, mean and SD. Classical deterministic methods run once.
3. Bai et al., IEEE TAP2022, DOI10.1109/TAP.2021.3137285: independently implement published MLP widths16/32/64/32/16 and LSTM32 + Dense16/32/64, scalar output. Use their zero-history variant, not a made-up five-minute sequence. Label BOTH as paper-architecture, Top10/hourly adaptations, not author implementation or full reproduction. Original seven ground-sensor inputs, LASSO selection, one-minute data and Q-band task are not available identically here. Max1000epochs, patience5, lr1e-4, Xavier input/dense matrices, MSE, RMSProp. Unspecified dense activations fixed ReLU, batch128, standard LSTM tanh/sigmoid, RMSProp alpha0.9/eps1e-6/momentum0 (paper's 'momentum0.9' terminology is ambiguous). Our train-only standardization, target scaling, PyTorch implementation and restore-best policy are explicit adaptations. No performance-driven architecture revision. Seeds17/29/43, report all.
4. TabM, ICLR2025, DOI/arXiv10.48550/arXiv.2410.24210: official yandex-research/tabm source, pinned commit and license retained. Common ten-input tabular-regression control, not satellite-specific prior art. Basic TabM without numerical embeddings, k32, n_blocks3, d_block256, dropout0.1; AdamW lr0.002 wd0.0003, batch256, MSE of EACH ensemble member (not only ensemble-mean MSE), ensemble mean for validation/inference. Max200epochs, patience20, seeds17/29/43, no hyperparameter search. Bounded-budget official-architecture transfer, not full reproduction of the paper's50–100trial optimization.
5. Standards/physics: reuse existing Rev016 circular P.838+geometric path and P.618-hourly-proxy predictions on exactly the same687 test keys. Additional rain/height/geometry inputs and rain-only-vs-total excess attenuation target mismatch are explicitly separate from common-input ranking. The hourly proxy is NOT standard annual P.618/R0.01 validation. Do not refit the historical clear reference/height imputer.

No pooling LOCAL and GBS errors. Do not rank annual CCDF models or random synthetic paths by instantaneous point RMSE. P.1853 / Maseng–Bakken / SST / recent meteorology-driven stochastic synthesizers require a separate fast/dynamic experiment; record feasibility and missing inputs rather than invent a success. Input/history eligibility matrix saved with citations and code-search evidence. Original research results are not treated as this experiment's results.

## Fitting, QA and reporting

One new imputer/scaler is fitted on LOCAL train only for the new family, never substituted into frozen models. Require at least one finite training value per selected column. Nonfinite features become missing, targets are never imputed. Float64 target statistics, float32 neural inputs; preserve signed predictions without clipping/bias correction. Uniform training weights. No checkpoint trained with test/GBS labels.

Save source SHA, versions, all candidate validation scores, all epoch curves, chosen epochs, elapsed CPU wall time, checkpoints, per-hour outputs and failure tracebacks. Test/external outputs are emitted only after all fitting and selection are frozen. Reruns do not overwrite existing completed artifacts. Independent script checks source hashes, keys/masks, finite values, old-prediction equality, train-only preprocessing, recorded epoch selection, and recomputes RMSE/MAE/bias without using training metrics code. Test day-cluster bootstrap (2000 draws, fixed seed20261016) compares each model with frozen proposed predictions; it does not change model selection. No significance or superiority claim unsupported by interval/data.

Preflight includes synthetic shape, parameter-count, determinism, train-only scaling, ensemble-loss and key-alignment tests. Outcomes, including weaker baselines and unresolved paper inputs, are retained. New comparison completion is not automatically completion of all T012 scientific work.
