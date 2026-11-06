"""Run the frozen 1024-mask residual analysis using Revision 018 columns."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import run_experiment as stage

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "Revision_017" / "layer_pipeline.py"
_spec = importlib.util.spec_from_file_location("revision017_layer_adapter_for_revision018", PREVIOUS)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Revision 017 layer adapter could not be loaded")
_adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _adapter
_spec.loader.exec_module(_adapter)

_adapter.LAYER_VERSION = "correlation-pruned-1024-residual-v18-20261001"
_original_manifest = _adapter.layer_manifest


def layer_manifest(frame, names, folds, chosen):
    payload = _original_manifest(frame, names, folds, chosen)
    key = str(Path(__file__).resolve().relative_to(stage.ROOT))
    payload["source_hashes"][key] = stage.source.sha256(Path(__file__).resolve())
    payload["fingerprint"] = stage.fingerprint(
        {name: value for name, value in payload.items()
         if name not in ("fingerprint", "created_utc")}
    )
    return payload


_adapter.layer_manifest = layer_manifest


if __name__ == "__main__":
    _adapter.main()
