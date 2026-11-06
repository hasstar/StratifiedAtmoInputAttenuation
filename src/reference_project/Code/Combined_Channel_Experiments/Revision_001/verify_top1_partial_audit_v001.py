"""Offline delivery checks for partial field QA and the recorded input gap."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top1_v003"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    output = DEST / "field_audit_001/delivery_qa.json"
    require(not output.exists(), "Immutable delivery audit exists; use a new version")
    test_log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("test_audit_top1_available_fields_v001")
    result = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == 12, "Field positive/negative tests failed")
    field_path = DEST / "field_audit_001/qa.json"
    compatibility_path = DEST / "input_compatibility_001.json"
    fields, compatibility = load(field_path), load(compatibility_path)
    require(fields["status"] == "PASS_REUSED_TOP1_FIELDS_ONLY" and
            not fields["full_top1_field_qa_pass"], "Partial scope mislabeled")
    require(compatibility["status"] == "BLOCKED_SPATIAL_INPUT_DEFINITION" and
            compatibility["new_fit_calls"] == compatibility["model_predictions"] == 0,
            "Incompatible input produced fits/predictions")
    sources = {**fields["source_sha256"], **compatibility["source_sha256"]}
    for path, digest in sources.items():
        require(sha(path) == digest, "Recorded source changed: " + path)
    total_hours, total_values = 0, 0
    for row in fields["checks"]:
        path = ROOT / row["payload"]
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(row["member"])
        else:
            raw = path.read_bytes()
        with netCDF4.Dataset("independent_top1.nc", memory=raw) as ds:
            idx = np.flatnonzero(np.asarray(ds["pressure_level"][:]) == 1).item()
            values = np.asarray(ds["z"][:, idx, :, :], dtype="<f8")
            require(values.size == row["grid_values"] and np.isfinite(values).all(), "Field count differs")
            require(hashlib.sha256(values.tobytes()).hexdigest() == row["float64_c_order_sha256"],
                    "Independently extracted field values differ")
            total_values += values.size
            total_hours += len(ds["valid_time"])
    require((total_hours, total_values) == (864, 1728), "Independent coverage differs")

    # Execute only the original pure interpolation function, not its module/main.
    # The original source file and old campaign outputs remain read-only.
    los_path = Path(r"E:\0_Workflow_System\01_PhD\Papers\3_TAP_被拒\scripts\prepare_era5_los_csv.py")
    tree = ast.parse(los_path.read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_bilinear_time_series")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(los_path), "exec"), namespace)
    interpolation = namespace["_bilinear_time_series"]
    supported = interpolation(np.array([[[0., 10.], [20., 30.]]]), np.array([51., 51.25]),
                              np.array([-1.5, -1.25]), np.array([51.125]), np.array([-1.375]))
    require(np.array_equal(supported, [15.]), "Original interpolation control failed")
    with np.errstate(divide="ignore", invalid="ignore"):
        degenerate = interpolation(np.ones((1, 2, 1)), np.array([51., 51.25]), np.array([-1.5]),
                                   np.array([51.125]), np.array([-1.5]))
    require(np.isnan(degenerate).all(), "Reassess original function's one-longitude behavior")
    for path, digest in sources.items():
        require(sha(path) == digest, "Source changed during verification: " + path)
    artifacts = [field_path, compatibility_path, Path(__file__),
                 Path(__file__).with_name("test_audit_top1_available_fields_v001.py")]
    report = {
        "status": "PASS_PARTIAL_AUDIT_AND_GAP_PROVENANCE_ONLY",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "positive_negative_field_tests": result.testsRun, "test_log": test_log.getvalue(),
        "independent_extracted_request_hours": total_hours, "independent_extracted_grid_values": total_values,
        "original_interpolation_checks": {"two_by_two_control": "PASS", "one_longitude_returns_nan": "PASS"},
        "checked_source_files": len(sources), "source_files_unchanged": True,
        "external_model_compatibility_pass": False, "external_predictions": 0, "new_fit_calls": 0,
        "full_top1_field_qa_pass": False, "overall_completion": False,
        "artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in artifacts},
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "positive_negative_field_tests",
                                            "checked_source_files", "original_interpolation_checks")}, indent=2))


if __name__ == "__main__":
    main()
