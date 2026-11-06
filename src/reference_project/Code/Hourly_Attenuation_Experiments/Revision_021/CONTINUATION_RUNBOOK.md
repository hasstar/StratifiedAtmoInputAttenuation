# Revision 021 continuation runbook

Workspace: `E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`. Python: `C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`. Code: `Code\Hourly_Attenuation_Experiments\Revision_021\run_rfe.py`. Output: `Outputs\Experiments\Hourly_Attenuation_Revision_021_RFE_Refit`.

At every wakeup read `PROTOCOL.md`, `PROGRESS.md`, this runbook, and `run_manifest.json`. Check `run_rfe.py status`, exact Python command lines/CPU, rfe.stdout.log/rfe.stderr.log, `states/k_*.json`, `training/k_*_seed*.json`, `models/k_*_seed*.pkl`, `test/k_*_seed*.json`, QA gates, and source fingerprint. If a matching worker is alive, only perform read-only health checks and never duplicate it. If absent, first distinguish normal completion from failure; only confirmed failure may use the same stage command to resume verified checkpoints.

Stages: `prepare` -> `verify-plan` PASS -> `rfe` (200 states, 600 training records/models, 199 deletions) -> `verify-rfe` PASS -> `test` (600 saved test predictions) -> `verify-test` PASS -> `report`. Test is not read before RFE QA. No zero-mask comparison, no 1024 subsets, and no residual model are part of this run. Weather labels evaluate only, including all heavy rain within rain. The local test set is historically accessed and the family was test-ranked earlier; do not call it newly blind or choose final k from its curve. All earlier versions and source data are read-only.

When all QA gates pass, update `PROGRESS.md` and delete the 15-minute heartbeat. Notify only on a meaningful stage completion, failure, need for user decision, or final completion. Stay quiet when a healthy worker continues unchanged.
