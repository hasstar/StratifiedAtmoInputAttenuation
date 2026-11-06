# Clear-sky attenuation indicator preparation

Run `python build_alignment.py` from any working directory. It reads the already
verified local, OpenSat4Weather and CEDA GBS sources and creates
`Outputs/Datasets/ClearSky_Alignment_Revision_002` in the paper workspace.
It refuses to overwrite an existing output directory.

Only CEDA has a native relative-to-clear-sky attenuation label. The local and
OpenSat tables retain observed received power in dBm and set the intended dB
target to null until a defensible, source-specific clear-sky reference is
established. This prevents a unit rename from being mistaken for a target
conversion. Nothing is written to the source directories.

The generated output README details the target definition, aggregation and
remaining baseline/ERA5 work. The manifest records hashes and QA counts.
Run `python validate_alignment.py` after building for independent sample
recomputation and source-hour conservation checks. This revision excludes
valid CEDA timestamps exactly at 00:00, which are end-of-integration boundary
observations of the previous day; Revision_001 is retained as a failed-QA
audit trail and must not be used downstream.
