# Revision 020 continuation runbook

Workspace: `E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`.
Python: `C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`.
Code: `Code\Hourly_Attenuation_Experiments\Revision_020\run_elimination.py`.
Output: `Outputs\Experiments\Hourly_Attenuation_Revision_020_Sequential_Zero_Mask`.

At each wakeup, read `PROTOCOL.md`, `PROGRESS.md`, this runbook, and the output manifest. Check the exact Python process command line and CPU, stdout/stderr tail, status, and checkpoint counts. If the worker is alive, perform read-only checks only; never start a duplicate. Keep all prior revisions and source data read-only.

Expected gates: `prepare` -> `verify-plan` -> `select` (200 state files, 199 candidate files, 20099 candidate assessments) -> `verify-selection` PASS -> `evaluate` (600 per-seed test records) -> `verify-evaluation` PASS -> `report`. Execute each next stage only after the preceding QA PASS. Resume `select` or `evaluate` with the same command only after confirming that the previous process exited and diagnosing any failure. The script refuses manifest fingerprint changes.

The selected path must use training hours only. The test set is historically accessed and XGBoost was test-ranked in Revision 019; do not call these scores blind, do not select a feature count from them, and do not conflate zero-masking with retraining. Once final QA and report are complete, update `PROGRESS.md` and delete this thread's heartbeat automation. Notify the user only on a meaningful stage completion, failure, needed decision, or final completion.
