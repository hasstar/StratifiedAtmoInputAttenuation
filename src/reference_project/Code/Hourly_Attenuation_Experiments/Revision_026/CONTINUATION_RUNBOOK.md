# Revision 026 continuation

Python: `C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe`.
Runner: `Code\Hourly_Attenuation_Experiments\Revision_026\run_test_benchmark.py`.
Output: `Outputs\Experiments\Hourly_Attenuation_Revision_026_Ten_Model_Test`.

Read PROTOCOL.md, PROGRESS.md, manifest and QA first. Inspect exact process command/PID/CPU, byte lock, logs and counts before resuming. Running matching workers receive read-only checks only. Do not recreate any previous layer, RFE, or fast experiment.

Commands: `prepare`, `refit`, `verify-refit`, `evaluate`, `report`, `status`.
After refit completes: require `verify-refit` PASS before `evaluate`. The evaluator loads the 30 saved weights and never fits. Then run the separate `verify_independent.py`; require independent test QA PASS before `report`. Re-run verification, not scoring, to check completed records. Create the reader workbook from the saved final report with the documented bundled spreadsheet runtime. Preserve source hashes and all failures. No test-driven retuning, retraining, feature changes, or seed cherry-picking.

Expected counts: 30 training JSONs and weights, 30 test JSONs, 20,610 per-hour test predictions, 150 per-seed weather metric rows, ten family rows. Report arithmetic mean seed metrics, not ensemble-prediction metrics. Old BiLSTM seed-17 and separate SVR C=10 references must be clearly distinguished from three-seed ranked models.

Completed 2026-10-11: reconstruction, independent metric QA, full saved-weight replay and delivery QA all PASS; workbook formula/value/visual QA PASS. No worker remains. Do not refit or rescore this completed benchmark on an old heartbeat. To check integrity, use the independent verification scripts only. Report the existing test ranking without modifying any previous experiment or calling it a new blind test.
