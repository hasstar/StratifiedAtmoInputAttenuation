#!/usr/bin/env bash
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python}" -X utf8 -B src/check_local_inputs.py
"${PYTHON:-python}" -X utf8 -B src/pipeline.py --mode full "$@"
