# Distributed data

I have included 236 CEDA station-day NetCDF files (Chilbolton and Sparsholt)
and the targeted UK ERA5 payloads used for the frozen evaluation. These are
reference copies and must be treated as read-only. I have not uploaded my local
measurements or local ERA5 datasets.

Official dataset records:
- Chilbolton Global Broadcast Service propagation observations:
  https://doi.org/10.5285/639a3714-bc74-46a6-9026-64931f355e07
- Sparsholt Global Broadcast Service propagation observations:
  https://doi.org/10.5285/E8F43A51-0198-4323-A926-FE69225D57DD
- ERA5: https://cds.climate.copernicus.eu/ (single-level and pressure-level products).

The exact UK requests are under `src/data_preparation/requests/`.
The downloader defaults to a dry run:

```bash
python src/data_preparation/download_data.py --source ceda_days
python src/data_preparation/download_data.py --source cds_uk
python src/data_preparation/download_data.py --source ncar_exact
```

To explicitly retrieve a source, add `--download --destination data/incoming_uk`;
each invocation requires a new destination directory. Obtain any necessary
provider registration/terms acceptance through the official service. CDS uses
the operator's normal CDS API configuration; CEDA may use the operator-provided
`CEDA_ACCESS_TOKEN` environment variable. Do not put credentials into this
repository. The script does not recover old remote requests or overwrite the
delivered raw files. Local request lists are omitted with the local dataset.

Provider terms and original attribution apply. No replacement data licence is
created by this private research repository.
