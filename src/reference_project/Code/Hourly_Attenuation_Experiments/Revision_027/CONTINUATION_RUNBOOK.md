# Revision 027 continuation

Python C:/Users/SDR_HH/.conda/envs/pytorchmy/python.exe; use -B.
Runner Code/Hourly_Attenuation_Experiments/Revision_027/run_pair.py.
Output Outputs/Experiments/Hourly_Attenuation_Revision_027_Paired_Base_Full.

Read protocol, progress, manifest/QA first. Inspect exact process command/PID/CPU, stdout/stderr, writer lock and record counts. A live matching worker permits READ-ONLY checks only; never duplicate it. Confirm failure and no same-command worker before restoring checkpoints with the same source fingerprint and command, distinct recovery logs and preserved failures.

Commands: self-test, prepare, fit-base, verify-fit, evaluate, report, status. Run self-test before prepare. Fit stage only reads train/validation; no new hyperparameter search. After 30 records/weights exist, verify-fit PASS before evaluate. Evaluate loads models only and reuses old full records. Run separate verify_pair.py; independent QA PASS before report. Preserve old Revision 025 seed17 base and Revision 026 all full predictions by reproduction checks. If complete, run verification only, never refit/rescore.

Report ten FULL-input ranks before checking the first family's three improvements; don't rank only favorable families. All rain samples combined. Complete counts: 30 weights, 30 training records, 30 paired evaluations, 41,220 paired hourly values, 300 seed/input/weather summaries. Test views/history/daily overlap stay disclosed. MET is only the predeclared mean-metric condition; robustness by seed/date is reported separately. If NOT MET, this diagnostic is done but my objective is not achieved; design any further strategy in a separate frozen stage using train/validation, not an open-ended test chase.
