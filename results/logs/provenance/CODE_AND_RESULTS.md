# Code, data and result catalogue

All paths below are relative to the **SupplementaryMaterial** root, so the package can be moved.
The public scripts are the supported entry points. Historical scientific source is preserved
under `src/reference_project/Code/`; its absolute paths are rebound in memory by `package_support.py`.
Never start an old acquisition queue or a historical training runner directly.

## Section II: data and physical definitions

| Function | Code | Inputs and outputs |
|---|---|---|
| Validate fixed data, splits, 156/50 feature sets and external cohort | `src/data_preparation/prepare.py` | Reads archived data manifests, train/validation/test Parquet and UK input table; writes `input_qa.json` in a new run |
| Local ERA5 surface interpolation and pressure-surface ray intersections | `src/data_preparation/rebuild_era5.py` and `src/data_preparation/reference_era5/prepare_era5_los_csv.py` | Reads `data/raw/era5/local/raw/` and exact campaign geometry; writes 12 hourly CSVs and extraction QA |
| Exact official data requests | `src/data_preparation/download_data.py` | Reads `src/data_preparation/requests/`; dry run by default, downloads only when explicitly requested into a new directory |
| Historical measurement processing and fixed random-hour split | `src/reference_project/Code/Dataset_Build/` | Includes clear-sky alignment, equipment calibration, detectability and `Signal_Detectability_Revision_003/build_random_hour_split.py`; original receiver recordings before processing are not bundled |
| One-second linear-power construction | `src/reference_project/Code/Fast_Fading_Experiments/Revision_001/build_clean_1s.py` | Archived local 100-ms measurements to `Outputs/Experiments/Fast_Fading_Revision_001/clean_1s/` |
| UK hourly linear-power mean and within-hour component | `src/fast/evaluate.py`, original `Fast_Fading_Experiments/Revision_003/evaluate.py` | Reads 236 native CEDA day files; retains native clear-sky reference; computes hourly and within-hour attenuation |
| ITU physical baseline | `src/reference_project/Code/Hourly_Attenuation_Experiments/Revision_016/run_itu_baseline.py` | Archived ERA5 physical inputs, P.676/P.840/P.838/P.839 implementation and fixed gaseous reference; 684 common test hours |

`Outputs/...` in this catalogue means `results/tables/reference_project/Outputs/...` inside this package.
Local model-ready splits are `Outputs/Datasets/Signal_Detectability_Revision_003/`.
UK acquisition plans, payload receipts, complete inputs and QA are under
`Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE/`.
The exact payload/request manifests and original source URLs are indexed in the data README.
Input QA is not a claim that a new remote download was performed.

## Section III: hourly models, training and elimination

**Public runner:** `src/hourly/train_rfe.py`. It trains each model independently and saves a
full-training fit plus three date-group fold fits per state. Each fold imputes/scales only its
fitting rows. A feature's deletion score is its held-out RMSE increase under permutation,
averaged over three folds. The deletion is recorded before validation/test evaluation.

| Model | Main scientific implementation |
|---|---|
| TabM | `Hourly_Attenuation_Experiments/Revision_031/core.py`, official `tabm` implementation in `Combined_Channel_Experiments/Revision_002/vendor/`, orchestrated by Revision_032/runtime.py |
| Bai LSTM | `Hourly_Attenuation_Experiments/Revision_032/runtime.py`; one concurrent atmospheric input step, LSTM 32 and dense 16–32–64–1 |
| MLP, BiLSTM, Transformer | `Hourly_Attenuation_Experiments/Revision_014/models.py`, fitted by `Revision_028/engine.py` through Revision_032/runtime.py |
| GCN | `Revision_014/models.py` and `Revision_028/engine.py` (`GatedProfileNet`); graph structure/fusion retained from the selected implementation |
| SVR-RBF, CatBoost, Random Forest, Elastic Net | `Revision_014/models.py` estimators and `Revision_028/engine.py` fixed configuration/weight handling |

These paths are below `src/reference_project/Code/`. Hyperparameters, selected epochs/tree
prefixes and seed 17 are centralized in `config.yaml`. The original configuration search is
retained, but the public RFE runner uses the selected configurations; it does not silently
retune at each feature count. TabM's 50-feature path is independent of its 156-feature path.

**Frozen results:**
- Full 156-to-1 paths: `Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE/models/<model>/k_<0001..0156>/`.
- Base 50-to-1 TabM path: `Outputs/Experiments/Hourly_Attenuation_Revision_041_TabM_Surface_RFE/models/tabm/k_<0001..0050>/`.
- Every state's `state.json`: ordered retained names, next deletion, fold importances and model references.
- `evaluation.json`: validation/test hour identities, truths, predictions and weather-specific metrics.
- `qa.json`: state/evaluation integrity checks. Model references locate actual weights and preprocessing.
- `results/logs/release_verification_001/all_states_and_models.csv`: searchable index of all 1,610 states and 6,440 fitting roles.
- `all_retained_and_removed_features.csv`: every retained set and deletion, including the final one-feature state.
- `all_local_metrics.csv`: all 24,150 state/split/weather records independently recomputed from saved predictions.

**New runs:** `results/tables/reproductions/<run>/hourly/<full156|base50>/<model>/k_XXXX/`.
`fold1/`, `fold2/`, `fold3/`, and `full/` contain fitted objects, preprocessing, identities and
training progress. `permutation_fold*.npz` stores every permuted held-out prediction.
`state.json` and `evaluation.json` retain the complete elimination and scoring trail.
Use `--resume` only for that same new run after its previous process has exited; identities
and hashes must match. Do not copy a reference checkpoint over a failed new state.

## Section III-C: frozen UK inference and k=8 SHAP

`src/evaluation/verify_reference.py` audits all local states and replays 100 frozen estimators.
The external state is the locally fitted k=10...1 model, with identical features, order,
scaling and fitted weights. UK rows never fit a model or scaler. Replayed predictions and
station/weather metrics are in `results/logs/release_verification_003/`.
All 5,593 eligible station-hours are scored; the archived input contains 5,664 rows before
the fixed eligibility mask. A newly trained path that requires an absent UK feature stops
with an explicit error rather than filling a never-acquired field.

`src/evaluation/shap_k08.py` recomputes both frozen TabM k=8 explanations. Its unchanged
scientific source is `figsrc/upstream/TabM_K08_SHAP_20261104_001/code/calculate_shap_v001.py`.
The analysis uses 64 training background rows, two independent permutation replicates and
paired forward/reverse paths. Outputs include per-hour SHAP arrays/CSV, feature summaries,
background/test inputs, additivity checks and Monte Carlo agreement checks.
The original full and base results are in that package's `results/` and `qa/` folders.
`figsrc/plot_shap_k08.py` redraws these stored attributions with “set” labels and the colour
bar at the far right. Plotting alone does not recompute explanations. These are marginal
model attributions, not causal or path-integrated physical attenuation contributions.

## Section IV: second-scale generation and combination

| Public script | Scientific source | Saved output |
|---|---|---|
| `src/fast/generator.py` | `Fast_Fading_Experiments/Revision_002/run_dynamics.py` and `Revision_003/freeze_generator.py` | Five date-fold candidate evaluations; selected length; weather CDF supports and contiguous training segment libraries |
| `src/fast/evaluate.py` | `Fast_Fading_Experiments/Revision_003/evaluate.py` | 32 realizations per hour, observations/masks, CRPS, SD, ACF, complete/censored fade events, VS and coverage diagnostics |
| `src/evaluation/summarize.py` | `Revision_003/summarize_and_plot.py` aggregation; saved dense-ACF calculation | All weather/cohort summaries, paired date-bootstrap intervals, full 1–1800 s ACF, fade-duration distributions, LaTeX tables and figure inputs |

Frozen generation results: `Outputs/Experiments/Fast_Fading_Revision_003/`.
The per-hour records are also exported in `results/tables/figure_inputs/common/frozen_exports/fast_hourly.csv`.
The published slow component is the frozen full-set TabM **k=10**, not the local compact k=8
SHAP state. The same 32 generated fast curves are added to that hourly prediction.
Fast-only CRPS is compared with zero within-hour fluctuation; combined CRPS is compared with
the hourly-only point predictor on the same total second-scale observations.

The new verified generation is in `results/tables/reproductions/release_check_002/fast/`:
`hours/<identity>.npz` contains observed/generated arrays, masks, metadata and diagnostics;
the adjacent JSON records its SHA and scalar scores. `hourly_results.parquet` and CSV
contain all 6,623 hours (343 validation + 687 test + 5,593 UK).
New training CDFs/libraries are in the sibling `generator/` folder. This release check
refitted those distributions and regenerated the full evaluation, but did not rerun the
five-fold generator search. The published complete search results are supplied separately.

`results/tables/reproductions/release_check_004/summary/` holds regenerated summaries and
readable score/duration/combination tables. Full CSVs retain all diagnostics, including
sample counts and metrics that do not fit in a compact manuscript table.

## Figures: exact current manuscript and additional support

The table below is generated from active `\\includegraphics` commands after removing
LaTeX comments. Filenames are stable asset IDs; the printed figure number follows the
current manuscript order. Numerical entries output both PDF and PNG. Photography and
editable schematics reuse verified PDF exports unless PowerPoint export is requested.

| Current order | PDF asset | Public entry point | Inputs |
|---|---|---|---|
| 1 | `高轨卫星实测图.pdf` | `figsrc/plot_fig01_measurement.py` | Supplied measurement photograph; PDF copied exactly. |
| 2 | `Fig04_All_Hourly_Observations_And_ITU_Test.pdf` | `figsrc/plot_fig02_itu.py` | ITU小时观测图_Revision_003: all 3434 hourly observations; ITU comparison on common 684 test hours. |
| 3 | `Fig02_Stratified_Atmosphere_Editable.pdf` | `figsrc/plot_fig02_atmosphere.py` | figsrc/editable/Fig02_Editable_Revision_002; editable PPTX and verified PDF. |
| 4 | `Fig05_TabM_Architecture_And_Protocol.pdf` | `figsrc/plot_fig05_tabm.py` | figsrc/editable/Fig05_TabM_Revision_003/editable; editable PPTX and verified PDF. |
| 5 | `Fig06_Ten_Model_RFE.pdf` | `figsrc/plot_fig06_rfe.py` | common/frozen_exports/local_metrics.csv; all 156 states per model. |
| 6 | `Fig07_TabM_Independent_RFE_Paths.pdf` | `figsrc/plot_fig07_independent_rfe.py` | common/frozen_exports/tabm_same_k.csv; two independently selected paths. |
| 7 | `Fig_SHAP_TabM_K08.pdf` | `figsrc/plot_shap_k08.py` | TabM_K08_SHAP_20261104_001/results/{full,base}_shap.npz and test input CSVs. |
| 8 | `Fig08_UK_Frozen_Model_Comparison.pdf` | `figsrc/plot_fig08_uk.py` | common/frozen_exports/uk_metrics.csv and all_hourly_predictions.parquet; 100 frozen states. |
| 9 | `Fig10_Fast_Temporal_Statistics.pdf` | `figsrc/plot_fig10_fast_temporal.py` | common/Fig10_dense_acf_1_to_1800.csv and Fig10_duration_distributions.csv. |

An additional supporting entry is `plot_s01_combined_examples.py` (preselected earliest
rainy examples). It is not an active manuscript figure in this snapshot.
Every generated set has `render_manifest.json`; all copied originals have source SHA records.

## Tables and textual evidence

The following are exact extracted LaTeX environments from the manuscript snapshot; they
remain author-formatted publication references. New numerical tables from `summarize.py`
are separately named and never overwrite these source tables.

| Manuscript table | Label | Source file |
|---|---|---|
| 1 | `tab:Table01_Measurement_Configuration` | `results/tables/manuscript/table_01_tab_Table01_Measurement_Configuration.tex` |
| 2 | `tab:Table02_Dataset_Overview` | `results/tables/manuscript/table_02_tab_Table02_Dataset_Overview.tex` |
| 3 | `tab:Table03_Training_And_RFE_Protocol` | `results/tables/manuscript/table_03_tab_Table03_Training_And_RFE_Protocol.tex` |
| 4 | `tab:Table04_Local_Ten_Model_Comparison` | `results/tables/manuscript/table_04_tab_Table04_Local_Ten_Model_Comparison.tex` |
| 5 | `tab:Table05_TabM_Independent_Pool_Comparison` | `results/tables/manuscript/table_05_tab_Table05_TabM_Independent_Pool_Comparison.tex` |
| 6 | `tab:fast_block_selection` | `results/tables/manuscript/table_06_tab_fast_block_selection.tex` |
| 7 | `tab:fast_amplitude_temporal` | `results/tables/manuscript/table_07_tab_fast_amplitude_temporal.tex` |
| 8 | `tab:slow_fast_crps` | `results/tables/manuscript/table_08_tab_slow_fast_crps.tex` |

The training table comes from the frozen configuration; data counts come from fixed split
and cohort manifests. RFE tables come from per-state evaluation records. Generator selection
comes from the training-only candidate/fold results. Fast/combined tables come from 32-curve
per-hour diagnostics and paired aggregation. Table 04 reports relative RMSE reduction and
absolute R² difference, not a percentage change in R².

## Provenance, logs and verified limits

- `results/logs/RELEASE_VALIDATION.json`: acceptance checks and limits, including unexecuted full retraining.
- `results/logs/reference_copy_plan.jsonl` and copy receipts: original location, package location and SHA for the archived experiment.
- `results/logs/additional_assets_manifest.json`, `added_dependency_manifest_001.json`, `added_data_dependencies.json`: current manuscript assets and dependency closure.
- `results/logs/bounded_training_and_era5_qa.json`: exact TabM full-fit replay, two complete Elastic Net RFE states and decoded ERA5-table agreement.
- `results/logs/full_fast_comparison.json`: all 6,623 newly generated hourly metric records versus archived latest evaluation.
- `results/logs/VISUAL_QA.json`: actual inspection of all ten delivered figure entries.
- `results/logs/RELEASE_FILES.jsonl`: timestamped final package file-size/SHA inventory; the inventory itself is excluded.
- `results/logs/provenance/REPRODUCTION.md`: execution boundaries and restart details.

Read-only means that supplied data/reference copies are not targets of the public workflows.
No old source, model, result or acquisition queue was altered during packaging.
Versioned failure logs remain available; successful repairs do not erase earlier failures.
The exact 1,610-state full retraining pipeline and external downloads have not been executed
end-to-end for this release. Tested scopes are listed explicitly rather than inferred from
the presence of an entry-point script.
