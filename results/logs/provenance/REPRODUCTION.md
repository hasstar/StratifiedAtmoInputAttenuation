# Execution, restart and evidence boundaries

## Three different operations

1. `bash scripts/run_all.sh` creates new fitted weights, independently computed
   RFE deletions, generator cross-validation, a fitted training CDF/library,
   local and external predictions, 32-member second-scale results, tables and
   numerical plots. It does not replace the manuscript's stored results.
2. `bash scripts/verify_frozen.sh` checks all 1,610 archived states and feature
   paths, independently recalculates local metrics, replays all 100 external
   models, and regenerates all 6,623 hours from the frozen generator.
3. `bash scripts/make_figures.sh` only redraws publication figures from saved
   numerical source data and reuses verified vector artwork. This is the quick
   route for caption/plot-layout changes. It does not train, select, or infer.

The full numerical workflow is provided and its components have been exercised;
the complete 6,440-role refit plus generator selection has not been claimed as
executed in the packaging session. `results/logs/RELEASE_VALIDATION.json` is the
authority for the checks actually run. Historical complete results and QA are
provided separately from the new release-check results.

## Output and failure records

Each launch creates a new `results/logs/run_<UTC>/` and numerical run under
`results/tables/reproductions/<UTC>/`. Logs record command, environment,
start time, progress, completion, or `FAILED_REQUIRES_INSPECTION`.
The OS lock prevents two public pipelines from running concurrently. Hourly
training also holds a separate lock. Never remove a lock or use a second
worker as a way of bypassing a healthy first worker.

Failures are preserved. A failed source/hash check is not permission to skip
data, silently change an RFE deletion, or accept a different statistical target.
Download failures stop without automatic retries. No historical STOP marker,
acquisition queue, frozen manifest or result is rewritten.

For a stopped new hourly run, first confirm that the actual worker process has
exited. Then use the original options with the same `--output` and add
`--resume` to `src/hourly/train_rfe.py`. Resume verifies the saved protocol,
feature list, fitting identity, model SHA and preprocessing statistics. Neural
fits retain the original optimizer/RNG recovery checkpoints. A failed
incomplete classical fit may restart that fit; completed fits/states are
validated and reused. Resume is not a way to apply changed hyperparameters to
an old checkpoint. The top-level launcher intentionally starts a fresh run.

## Configuration

`config.yaml` centralizes package-relative paths, selected model configurations,
epochs/rounds, the fixed seed 17, CV folds, candidate lengths, lags, and historical
analysis random namespaces. This is a reproduction configuration. Changing a
protocol requires a new experiment and a new output directory. The public
hourly adapter enforces the published seed and CPU-thread count.

TabM has three width-256 blocks and 32 members. It fits 54 epochs using the
published validation-selected configuration. Classical tree/linear/kernel
settings and the other neural epochs are recorded individually. The 50-feature
path has its own elimination trajectory. It is not obtained by simply removing
pressure-level inputs from an already selected full-set checkpoint.

Raw inputs, reference weights and historical source bytes are immutable.
`package_support.py` rebinds historical path constants in memory to the delivered
directory; it does not patch old scientific source files. Vendored Python sources
are retained for provenance, while compiled libraries come from the pinned
installed packages. Historical absolute paths in manifests identify the source
machine and are not runtime locations required by the public adapter.

## Data construction versus new model fitting

The release begins from quality-screened local measurement tables. Raw ERA5
interpolation is executable and independently checked, but the historical
measurement calibration, signal-detectability decisions, random hour split,
156/50 feature eligibility and three/five-fold date memberships stay fixed.
Their code and intermediate outputs are included as provenance. This boundary
lets a reader reproduce the reported model experiment without quietly changing
the underlying measured target.

`src/data_preparation/prepare.py` checks identities, row counts, unique and
disjoint hour keys, finite targets, feature counts and all 236 CEDA day hashes.
It is not labeled a new independent QA of every downloaded ERA5 field. The
original acquisition, field-completeness, geometric and label QA remain in the
reference result directories.

## Figures following a new run

The summary stage creates new plot input CSVs. Numerical RFE, UK and temporal
figures read them through `TVT_FIGURE_INPUTS`; UK audits use the corresponding
new prediction file. The render manifest names the input directory. The
measurement photograph, atmospheric diagram, architecture, ITU benchmark and
published SHAP figure have their own fixed provenance and are identified as
reference assets. Retaining them does not imply a new SHAP computation or a new
physical experiment. Every numerical table and figure can be traced back to
the saved per-hour or per-state data.

## Release portability

The scientific Python entry points were checked on Windows with Python 3.12.
Git Bash is the recommended shell for the supplied bash launchers. The bundled
MSYS shell could not start under this host’s OS permissions, so shell execution
itself is not recorded as validated; the equivalent Python entry points were run.
All public paths are resolved from the package root. UTF-8 paths, long archive
names and non-ASCII diagram/source folders are intentional. Do not flatten or
rename an individual reference directory without also rebuilding its manifest.
Keep the source directory structure when moving the entire package. Exact
floating-point reproducibility across different library builds, BLAS libraries
or operating systems has not been asserted.
