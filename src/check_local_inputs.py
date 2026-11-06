"""Fail clearly when intentionally unpublished local data have not been restored."""
from pathlib import Path
import sys
r=Path(__file__).resolve().parents[1]
required=['data/raw/era5/local','results/tables/reference_project/Outputs/Datasets/Signal_Detectability_Revision_003','results/tables/reference_project/Outputs/Datasets/Fast_Variation_Revision_002']
missing=[x for x in required if not (r/x).exists()]
if missing:
    print('The local dataset is intentionally not distributed. Full reproduction requires separately restoring the original local inputs:')
    for x in missing:print('  '+x)
    print('For the distributed aggregate figures, run: bash scripts/make_figures.sh')
    sys.exit(2)
