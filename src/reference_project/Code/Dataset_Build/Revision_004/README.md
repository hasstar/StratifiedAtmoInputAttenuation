# Revision_004 build and validation

Run from the paper workspace:

```powershell
D:\Anaconda202410\python.exe Code\Dataset_Build\Revision_004\build_event_chronological_dataset.py
D:\Anaconda202410\python.exe Code\Dataset_Build\Revision_004\build_event_chronological_dataset.py --write
D:\Anaconda202410\python.exe Code\Dataset_Build\Revision_004\validate_event_chronological_dataset.py
```

The first command is read-only and prints the proposed split. The `--write`
command creates `Outputs/Datasets/Revision_004` and refuses to overwrite an
existing directory. Validation writes only its report and summary inside that
new output directory. The builder reads verified `Revision_003` and the
existing fast-window statistics, but never alters them or any ERA5, H5, or
Run_011 source.

Only the chronological split changes. Use `split_event_chrono_24h` for new
experiments and keep `snow_case` and `embargo` out of formal fitting and
scoring. Weather event labels are for stratification, and the low-power flag
is outcome-derived audit metadata; neither is a model predictor. The full
11-pressure-level feature set, original five-layer control, 0-pressure
control, and strict local control are all materialized for both coverage
scopes. The 2,048 subset catalog is copied unchanged, not trained.

To compare with the old `split_strict_24h` protocol, retrain and select from
scratch under that old protocol. A model fit on the new training interval has
already seen some hours that belong to the old validation/test intervals, so
its score there is not an independent old-protocol test.
