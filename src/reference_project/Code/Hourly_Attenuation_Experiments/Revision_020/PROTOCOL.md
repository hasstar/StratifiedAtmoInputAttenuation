# Revision 020: sequential single-feature masking of the frozen XGBoost

This is the next, explicitly exploratory step after Revision 019. It follows the existing winning XGBoost model without refitting. The model has 200 actual numerical meteorological inputs after training-only correlation screening of the original 348; therefore the path is **200 retained inputs down to 1**, not 348 down to 1. The 148 pre-screened columns cannot be revived without a different experiment.

## Inputs and separation

- Use the three stored XGBoost models (seeds 17, 29, 43), training-fitted imputers, and fixed 200-column ordered feature plan from Revision 019. Verify model SHA-256 hashes, source fingerprints, and the Revision 019 all-stage QA before doing any work. No model fitting or hyperparameter changes occur in this revision.
- At the final model input, set every eliminated coordinate to numeric zero. In this pipeline, zero denotes the training-reference value (raw input was shifted by its training median). It does **not** mean that a meteorological variable or atmospheric layer is physically absent.
- Select the removal path using only the 2404 original training hours. Do not use validation, test, event labels, or weather-specific test errors to choose a feature or stop the path. The existing validation-selected model settings remain frozen.
- After the complete 200-to-1 path is frozen and independently verified, evaluate each state once on the same 687 test hours. These test results only describe the predefined path; they must not be used to revise the path, select a best feature count, retune the model, or claim fresh blind generalization. This test set was previously accessed and was used to select XGBoost in Revision 019.

## Exact greedy rule

Start with all 200 retained inputs. At each current size k from 200 to 2, try masking each of the k currently retained features in turn, while all previously eliminated inputs remain zero. For each candidate, predict every training hour with each of the three frozen models and compute the mean of their three training RMSE values. Permanently mask the candidate with the smallest mean training RMSE. Break an exact tie by the original feature-column order. Never unmask a previously eliminated feature. The selection objective is overall training RMSE only; rain and dry-overcast labels are not used to select.

This is greedy **fixed-model input masking**, not classical recursive feature elimination with model refitting. Because the fitted trees were learned with all 200 inputs, a k-feature masked state is not a newly trained k-feature model. Correlated features and masking-induced input combinations can make the errors non-monotone. The path is a sensitivity/exploration result, not a causal or deployment-ready reduced feature set.

## Output and QA

- Save an immutable manifest with source/model hashes and the exact rule; a resume must match it.
- Save 200 state files for k=200,...,1. Each state contains the full retained and eliminated feature-name lists, feature removed at that transition, original input indices, all three training scores, mean training score, and fingerprint.
- Save 199 candidate files, one for each transition, listing all candidates and their three training RMSE values. There should be 200+199+...+2 = 20099 candidate assessments, each scored using three stored models.
- After a selection QA PASS, save all 200 x 3 = 600 per-seed test records with hourly predictions and overall/rain-including-heavy-rain/dry-overcast metrics. Full-input test predictions must equal the Revision 019 saved predictions for each seed. Save a 200-row summary and feature-path CSV, a human-readable result document, and final QA.
- Source H5, ERA5, Run_011, datasets, Revision 019, and earlier experiments are read-only. All new files stay in the Revision 020 code/output folders. Never run two workers on the same revision concurrently; an active worker is monitored read-only and a failed worker resumes only after confirmed exit.
