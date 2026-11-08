# T014 / Revision032 — frozen before new fits

## Scope and interpretation

Ten learning families: tabm, bai_lstm, bilstm, transformer, mlp, catboost, gcn, random_forest, svr_rbf, elastic_net. Each starts from the SAME ordered 156 columns and independently removes exactly one feature to 1. I treat TabM superiority as a hypothesis, not a required ranking. No test-driven parameter search, deletion, seed selection or suppression of poor states.

LOCAL Signal_Detectability_Revision_003: 2404 training / 343 validation / 687 test hours. Rev023 retained list: 50 surface/base + 106 pressure variables; pressure levels 925/850/700/650/500/300/200/30/5/1 hPa. Target clear_sky_attenuation_db is unchanged. No historical measured channel as predictor. No new weather, UK fitting, 1024-layer training, fast-CDF fitting, or excluded-model work.

## Configurations

Reuse Rev028 validation-frozen FULL configurations for its eight current families, including weather weight policies and gated GCN. This is a comparison of existing fixed LOCAL adaptations, not equal-budget fresh hyperparameter optimization or author pretrained networks. Fixed rounds: BiLSTM53, Transformer41, MLP48, GCN56, CatBoost500, Random Forest7; SVR/ElasticNet use frozen solver settings. The RF 7-tree prefix was selected in the earlier validation study, not in this experiment or on the test set.

TabM uses Rev031 official_base width256 / blocks3 / k32 / dropout0.1 / AdamW lr0.002 weight_decay0.01 / 54 epochs. Original official architecture, LOCAL trained, not author pretrained weights. Do not continue the four unsuccessful architecture proposals.

Bai LSTM is the existing one-step concurrent-weather adaptation: LSTM(input=k, hidden32), dense16→32→64→1, ReLU; Xavier initialization and forget input bias1; RMSprop lr1e-4, alpha0.9, eps1e-6; batch128. First select its full156 epoch using the fixed architecture, original validation overall RMSE, earliest strict minimum, patience5, cap1000. Freeze that epoch for all subsequent folds/subsets. No architecture or seed search. This is formal full156 training, NOT a timing pilot. Test stays closed for this choice.

All seeds17, CPU two threads, one worker. No multi-seed claim. Preprocessing fits only the respective fitting rows: median imputation; standard scaling for linear/kernel/neural families, not tree families; target scaling for neural models. TabM/Bai use the Rev031 numeric-median implementation; structured networks use legacy atmospheric slots with discarded inputs masked and entire absent levels removed. Actual selected input measurements decrease each step; zero masked slots are architectural placeholders, NOT replacement for subset refitting. Negative permutation importance retained.

## RFE / reporting

Three GroupKFold folds by Beijing calendar date WITHIN original training. Every state has three fold fits and one original-training fit: 6240 role slots, less exact-match reused fits (explicit ledger). For k>1, permute each retained column once per fold, measure increase in fold-held-out overall RMSE, average three deltas, remove minimum with lexical name tie break. A permutation acts on transformed held-out columns; identical imputed constants stay constant. Common deterministic permutation namespace is Rev029 fingerprint cdda7a1eaa7e14477a8edef39702d5b5e34e03f2a29a337d01b91c88ba57b4cd; SHA256(namespace|k|fold|feature), first8 bytes big-endian. This reuses the same random mechanism, NOT another model's ranking. 367350 permutation evaluations over ten paths. k=1 no further deletion.

Freeze and commit the training-only state and next deletion BEFORE evaluating validation/test. Test results at each k are descriptive requested outputs; selecting the best k from those curves is exploratory, not unbiased generalization. No internal-test metric determines future fits. Prior full156 configurations had used official validation; internal CV is used for RFE, not claimed nested validation of all hyperparameters. Historical test has been examined before.

Each state saves model/fit-role lineage, training and fold predictions, per-epoch histories where applicable, retained names/level metadata, deleted next feature, permutation predictions, validation/test predictions, and all five groups: overall, clear_dry, rain, dry_overcast, other. R²=1−sum((prediction−truth)^2)/sum((truth−mean(truth))^2), NOT Pearson r². Also save r and r². R² may be negative; zero target variance or n<2 yields null with reason, never forced zero. Pearson is null for constant vectors. Dates are actual hours' Beijing dates, not all-day weather labels. Scores are dB except dimensionless R²/r, hours count n.

## ITU source contract

My manuscript Materials/Original_Submission/English_2026_TVT.pdf page4 section II.D, Eq(4): gas+cloud+rain. Prose says four mechanisms although the equation has three; no extra scintillation is invented. Reuse approved Combined011 corrected hourly geometric approximation plus Combined027 liquid-input legality audit. The printed vapour-density Eq(5) has an extra RH factor when vapour pressure is already from dew point; corrected implementation does not multiply humidity twice. Relative-clear target conversion uses fixed physical gas reference, NOT the manuscript's fitted 0.5dB empirical offset. Cloud liquid counted once, hourly liquid rain proxy not R0.01. This is explicitly a corrected hourly combined ITU approximation, not exact reproduction of every manuscript sentence or a full ITU probability model.

ITU has no learned RFE path. Keep all687 ML hours, separately compare all models on the684 physically legal hours. Exclude only the three pre-identified negative tp−snow liquid differences (027 ledger); no clipping/missing fill. Retain full-hour missing ledger. No change of reference based on labels.

## Reuse, QA, execution

Reuse existing identical BiLSTM029 fits/path; verify sources, key/feature/config agreement, replay weights, recompute metrics and all permutation deletion calculations with new independent QA. Reuse028 full156 fit only where configuration/preprocessing/keys/seed match; reuse031 full156 formal and three folds. Do not count reused roles as newly trained. Model SHA and upstream receipts pinned before work.

Synthetic correctness checks only; no runtime pilot. New files before freezing are preparation. After manifest freeze, implementation defects require a new unoccupied repair revision; failures and old outputs preserved. Committed states/results never overwritten. Mutable execution progress/recovery files are explicitly distinguished. Recover only after real PID+creation+command and OS lock released, manifest/input SHA checks pass, and interruption is classified. Unknown failures require inspection, not blind retry. New worker checks a task-local STOP_REQUESTED between operations and epochs, no old stop modification.

Independent QA uses scalar math.fsum formulas, verifies key/weather/date/target alignment, full preprocessing fitted-row provenance and model replay, checks fold/date partition, permutation matrices and deletion. PASS is not equivalent to completion; all1560 states, fixed ITU, summaries, detailed retained-feature inventory, negative/failure ledger and Chinese final report must be delivered. Same tvt every15min monitors real process and progress until full acceptance; no duplicate automation. Original two-day target is at risk with this expanded work, not a promised deadline.
