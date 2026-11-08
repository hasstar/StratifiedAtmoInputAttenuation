# Revision 019 continuation runbook

Workspace: `E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction`. Python: `C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`. Code: `Code\Hourly_Attenuation_Experiments\Revision_019`. Output: `Outputs\Experiments\Hourly_Attenuation_Revision_019_Test_Ranked_Zero_Mask`.

Read `PROTOCOL.md`, `PROGRESS.md`, `run_manifest.json`, and the latest QA records first. Check exact process command lines, CPU, stdout/stderr, record counts, and fingerprints before doing anything. If a worker is alive, perform read-only health checks only; do not launch a duplicate. If it has failed, diagnose and resume the same stage from its verified checkpoint. All sources and previous revisions are read-only.

Expected stages: `prepare` and `verify-plan`; `train` (40 trials, 30 validation refits) and `verify-train`; `test-models` (30 test refits) and `verify-test`; `zero-masks` (22 mask configurations times three seeds, no per-mask refits) and `verify-all`; `report`. Never score test before train QA PASS; never zero-mask before ten-model test-ranking QA PASS. Report meaningful stage changes, failure, or completion only. The test is historically accessed and is being used for the explicit ten-model benchmark ranking, so do not call it fresh blind.

When all final QA is PASS, update `PROGRESS.md` and delete this thread heartbeat. Old experiments may be removed only after resolving the exact deletion targets I specify and protecting the audit trail and dependencies.
