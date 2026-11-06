# Fast Variation Dataset Revision_001

This code builds the fast-varying branch paired with `Outputs/Datasets/Revision_002`.

## Run

```powershell
python build_fast_dataset.py
python validate_fast_dataset.py
```

The builder never modifies Run_011, the raw H5 archive, ERA5, or the verified hourly dataset. It creates a new output directory atomically. A failed build is retained in a timestamped `Fast_Variation_Revision_001_Failed_*` directory for diagnosis.

`smoke_test.py` constructs a tiny synthetic aligned Run_011-like input and checks deduplication, W-domain aggregation, segmentation, one-second output, and window generation before the full run.

The independent validator is intentionally separate from the builder and rechecks source hashes, schemas, time keys, hourly power consistency, derived variables, one-second aggregation, windows, and split inheritance.
