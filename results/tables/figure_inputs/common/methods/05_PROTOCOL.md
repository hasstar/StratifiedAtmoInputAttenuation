# Revision 041 / T017 — TabM surface-only independent RFE

Frozen before the first scientific fit, 2026-10-26. This protocol adapts the completed T014 experiment without changing its files.

## Scientific contract

1. The question is the predictive contribution of the pressure-level candidate pool under the same TabM training/evaluation procedure. Candidate pool A is the original ordered 156 names from Rev023, whose entire completed RFE path is read-only. Pool B is exactly its ordered 50 `surface__` names, eliminated independently from 50 to 1. These single-level/base variables include column integrals and diagnostics; “surface-only” is shorthand, not a strict surface-station-only claim. No new physical decorrelation selection is fitted.
2. Use only Signal_Detectability_Revision_003: train 2404, validation 343, test 687 hours. Keep target clear_sky_attenuation_db, timestamps and weather flags. Test counts overall/clear_dry/rain/dry_overcast/other are 687/322/51/158/156. Three excluded ITU hours are NOT excluded from the primary ML comparison. No new dates/sites, external data or weather download; no UK training or refitting.
3. TabM uses the existing official_base architecture (width256, 3blocks, 32internal members, dropout0.1, no embedding/gate), AdamW lr0.002/weight_decay0.01, batch256, uniform sample weight, 54 fixed epochs, seed17, CPU2threads. The 32 members are averaged for point predictions, not declared a calibrated predictive distribution. No new hyperparameter search, epoch/seed search or timing pilot.
4. For every state, 3 GroupKFold folds by Beijing date within training plus one full-training fit. Imputation, feature standardization and target scaling use each role's fitting hours only. Recomputing fitting-only preprocessing for QA is an audit, not an additional model experiment.
5. For k>1, independently permute each retained feature once on each training fold's holdout. The mean over 3 folds of permuted RMSE minus unpermuted RMSE defines importance. Remove its minimum, lexical feature-name order for exact ties. Preserve namespace cdda7a1eaa7e14477a8edef39702d5b5e34e03f2a29a337d01b91c88ba57b4cd and SHA256(namespace|k|fold|feature), first8bytes big-endian. Do not reuse the 156-pool deletion sequence.
6. Commit the training-only deletion decision before validation/test inference. All steps are reported, including negative R² and poor states. A descriptive representative uses minimum training-CV mean RMSE, smaller k for exact ties, never minimum test RMSE. Both these data and the full-pool reference have been examined historically; no new untouched-test claim, confidence/significance claim or universal superiority conclusion.

## Reuse and implementation

Reuse unchanged Rev032 runtime.py, run.py functions and verify.py through an in-memory output-path adapter. Never run its ten-model main, its Bai epoch search or any old CLI. Before import, check old code digests against its original manifest. New files and data dependency hashes are then frozen in the new manifest.

The existing Rev031 layer/states/0000.json provides only the k50 starting weights. Reuse each of its 4 roles only after QA-file digests, ordered features, exact fitting/holdout keys, config, seed17, fixed epochs54, fitting-only preprocessing and saved prediction replay match. Record these as reused, not newly fitted. k49→1 are new fits (196 roles). No old mask/RFE queue is resumed.

All outputs go to Outputs/Experiments/Hourly_Attenuation_Revision_041_TabM_Surface_RFE. Pin reference state/evaluation/QA hashes to the completed Rev034 delivery table; independently recompute metrics from predictions. Reuse shared numerical routines for reproducibility; the scalar audit uses math.fsum independently of NumPy production metrics. This is independently implemented computational QA, not external peer review.

## Operational contract

Single local worker, OS byte lock writer.lock, PID/create_time/command in active_worker.json, timestamped immutable runs and failures. Startup refuses another project Python worker or occupied old writer locks. Use the existing pytorchmy environment and at least 3GiB free disk. Preserve old STOP_REQUESTED files. The new STOP_REQUESTED stops only this new experiment. No automatic failed-run retry or new service/automation.

First use: prepare_surface.py; then run_surface.py run. New source must not be edited after execution/freezing. An interrupted run needs inspected, explicit resume authorization referencing a newly saved diagnosis; the worker rechecks source/input hashes and same checkpoint identity. A PASS state is reused only when its state/evaluation hashes and feature path match. Unknown failures remain visible. Store checkpoints every5epochs and at epoch54 as in Rev032.

## Delivery and acceptance

50/50 states and 200/200 roles, with 4 true reused roles and 196 new fits if start-state reuse passes. 3822 training-fold feature permutations (3 × sum(2..50)). Report per-state kept/removed names, per-feature importance, CV and all 5 weather groups, validation/test hourly predictions, metrics RMSE/R²/MAE/bias/Pearson r and prediction variability, model/lineage SHA. The old all-pool 156 states are kept intact.

Compare pool A and B at the same k1–50; show k1–10 zoom and full-range context. Export PDF+PNG, independent CSV-readback QA and Chinese interpretation. Plots retain real negative values and undefined values as unavailable, not zero. Final visual review is separate from numerical PASS. The worker ends at NUMERICS_COMPLETE_VISUAL_REVIEW_PENDING; only a real image inspection and final acceptance can mark overall delivery complete or copy the accepted package into submission materials.
