"""Revision 018 entry point: train-only |Spearman rho| >= 0.90 pruning.

All estimator implementations and record formats are reused from frozen prior
versions, but their output roots, source hashes and screening threshold are
redirected to this independent experiment revision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "Revision_017" / "run_experiment.py"
_spec = importlib.util.spec_from_file_location("revision017_core_for_revision018", PREVIOUS)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Revision 017 core could not be loaded")
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)

OUT = ROOT / "Outputs/Experiments/Hourly_Attenuation_Revision_018_Correlation_090"
DATA = _core.DATA
PRIOR = _core.PRIOR
SCREEN = OUT / "feature_selection.json"
MANIFEST = OUT / "run_manifest.json"
MODEL_QA = OUT / "model_selection_qa.json"
THRESHOLD = 0.90
VERSION = "correlation-pruned-ten-models-v18-20261001"

_core.HERE = HERE
_core.OUT = OUT
_core.SCREEN = SCREEN
_core.MANIFEST = MANIFEST
_core.MODEL_QA = MODEL_QA
_core.THRESHOLD = THRESHOLD
_core.VERSION = VERSION
_core.c.OUT = OUT
_core.prior_selection.OUT = OUT

_original_screen = _core.train_only_screen


def train_only_screen():
    header, audit, pairs = _original_screen()
    header["source_hashes"][str(PREVIOUS.relative_to(ROOT))] = _core.source.sha256(PREVIOUS)
    header["method"] = (
        "nonconstant; <=50% missing; train-only Spearman absolute rho <0.90 "
        "greedy independent set, >=200 pairwise observations"
    )
    header["fingerprint"] = _core.fingerprint(
        {key: value for key, value in header.items() if key not in ("prepared_utc", "fingerprint")}
    )
    return header, audit, pairs


_core.train_only_screen = train_only_screen

_original_model_manifest = _core.model_manifest


def model_manifest(saved):
    payload = _original_model_manifest(saved)
    payload["source_hashes"][str(PREVIOUS.relative_to(ROOT))] = _core.source.sha256(PREVIOUS)
    payload["fingerprint"] = _core.fingerprint(
        {key: value for key, value in payload.items()
         if key not in ("fingerprint", "created_utc")}
    )
    return payload


_core.model_manifest = model_manifest

# Export the frozen helpers to the Revision 018 layer adapter.
c = _core.c
source = _core.source
prior_selection = _core.prior_selection
hashes = _core.hashes
fingerprint = _core.fingerprint
require = _core.require
SubsetImputer = _core.SubsetImputer
StandardScaler = _core.StandardScaler
STRUCTURED = _core.STRUCTURED
ORIGINAL_PREPROCESS = _core.ORIGINAL_PREPROCESS
prepare = _core.prepare
load_screen = _core.load_screen
masked_training = _core.masked_training
configure_pruned_preprocess = _core.configure_pruned_preprocess
run_models = _core.run_models
verify_models = _core.verify_models
status = _core.status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "models", "verify-models", "status"))
    parser.add_argument("--models", nargs="+", choices=c.MODEL_NAMES)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif args.command == "models":
        run_models(None if args.models is None else set(args.models))
    elif args.command == "verify-models":
        print(json.dumps(verify_models(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
