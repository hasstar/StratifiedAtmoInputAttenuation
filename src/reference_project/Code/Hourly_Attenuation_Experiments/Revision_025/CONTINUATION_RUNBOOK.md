# Continuation runbook

Use `C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe` in this directory. Never launch a second writer while an identical worker is alive. Check `run_experiment.py status`, exact process command line/PID/CPU, stdout/stderr, inventory counts and manifest fingerprint first. If the worker is alive, only read health. If it ended, identify successful completion versus failure before resuming the same stage under the same fingerprint.

1. `run_experiment.py prepare` creates the frozen manifest after checking parent QA and source schema. `run_experiment.py smoke` tests two tiny training-only forward/backward examples without changing frozen results.
2. `run_experiment.py layer-run` writes 1024 states and models, 4096 total fits. Resume with the same command if confirmed failed and no worker is alive. `layer-verify-run` must PASS and freeze the train-fold mask.
3. `layer-evaluate` scores saved models on validation/test without fitting. `layer-verify-evaluation` must PASS; then `layer-report` writes full tables and documentation.
4. `rfe-run` writes 156 states and models, 624 total fits and 155 deletions. Resume only after confirmed failure and no live writer. `rfe-verify-run` must PASS and freeze train-fold k.
5. `rfe-evaluate` only loads models and scores validation/test. `rfe-verify-evaluation` must PASS; then `rfe-report` writes all-k tables and documentation. Final audit checks both reports/QA and updates PROGRESS. Stop this heartbeat only after all gates PASS.

The source H5, original ERA5, Run_011, dataset and all older revisions are read-only. Test hours were viewed in historical work; do not call them fresh blind. Rain includes strong rain. The ten-model primary selection remains the earlier validation-selected SVR; BiLSTM here is a controlled comparator.
