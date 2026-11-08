# Bounded CDS authorization — 2026-10-20

I explicitly answered “允许这3项使用官方CDS接口” to the scoped question about K index, Total totals index and precipitation type, after authorizing completion of missing inputs. This addendum supersedes only the original protocol's no-credentials/no-CDS restriction for these three fields.

Use the existing local credential only to authenticate to https://cds.climate.copernicus.eu/api. Never display or persist credentials or signed asset URLs. Submit new requests only for the frozen two-station, 236-station-day cohort, at their previously verified nearest surface grid points. Request ERA5 hourly single-level values at the existing UTC hours, without deriving these products from substitutes. Preserve response metadata, units, grid, time, raw bytes and hashes. No paid service, installation, old request listing, old request IDs, old queue entrypoint, or old stop-marker change is authorized.

At most one new CDS request may be unresolved at a time. Save its local intent before POST and its new remote ID immediately after acknowledgement; do not retry an ambiguous POST. Only these newly registered IDs may be polled or downloaded. A pending remote CDS job may process while the single local worker reads the bounded anonymous NCAR subsets. This does not add a second local worker. All stages share the new execution lock. Errors are sanitized before recording; HTTP credential material and signed URLs must not enter failures or console output.

The frozen model, date, weather, label and preprocessing contracts remain unchanged. A source's successful request is not input QA or completed model evaluation.
