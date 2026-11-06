"""Independent power-domain identity check for the derived clean 1 s series."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Outputs/Experiments/Fast_Fading_Revision_001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = OUT / "clean_1s_qa.json"
    qa = json.loads(source.read_text(encoding="utf-8"))
    if qa["status"] != "PASS":
        raise RuntimeError("Clean 1 s QA is not PASS")
    rows = 0
    max_ratio_error = 0.0
    max_f_error_db = 0.0
    max_envelope_error = 0.0
    hourly_parts = []
    for name, expected_sha in qa["monthly_parquet_sha256"].items():
        path = OUT / "clean_1s" / name
        if sha256(path) != expected_sha:
            raise RuntimeError(f"Partition hash differs: {name}")
        frame = pd.read_parquet(
            path,
            columns=["hour_bjt", "quality_zero_records", "received_power_w",
                     "hour_mean_power_w", "normalized_power_ratio",
                     "normalized_envelope", "fast_residual_db"],
        )
        rows += len(frame)
        p = frame.received_power_w.to_numpy(dtype=float)
        base = frame.hour_mean_power_w.to_numpy(dtype=float)
        q = frame.normalized_power_ratio.to_numpy(dtype=float)
        env = frame.normalized_envelope.to_numpy(dtype=float)
        f = frame.fast_residual_db.to_numpy(dtype=float)
        if not (np.isfinite(p).all() and np.isfinite(base).all()
                and np.isfinite(q).all() and np.isfinite(env).all()
                and np.isfinite(f).all() and (p > 0).all() and (base > 0).all()):
            raise RuntimeError("Nonfinite or nonpositive derived observation")
        max_ratio_error = max(max_ratio_error, float(np.max(np.abs(q - p / base))))
        max_f_error_db = max(max_f_error_db, float(np.max(np.abs(f + 10 * np.log10(q)))))
        max_envelope_error = max(max_envelope_error, float(np.max(np.abs(env ** 2 - q))))
        weighted = frame[["hour_bjt", "quality_zero_records"]].copy()
        weighted["weighted_ratio"] = weighted.quality_zero_records * q
        hourly_parts.append(weighted.groupby("hour_bjt").agg(
            n=("quality_zero_records", "sum"), weighted_q=("weighted_ratio", "sum")
        ))
    hourly = pd.concat(hourly_parts).groupby(level=0).sum()
    max_hour_normalization_error = float(np.max(np.abs(hourly.weighted_q / hourly.n - 1)))
    passed = (rows == qa["clean_1s_observed_seconds"] and len(hourly) == 3434
              and max_ratio_error < 1e-10 and max_f_error_db < 1e-10
              and max_envelope_error < 1e-10 and max_hour_normalization_error < 1e-10)
    report = {
        "status": "PASS" if passed else "FAIL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verified_seconds": rows,
        "verified_hours": len(hourly),
        "max_abs_ratio_identity_error": max_ratio_error,
        "max_abs_fast_residual_identity_error_db": max_f_error_db,
        "max_abs_envelope_square_identity_error": max_envelope_error,
        "max_abs_hour_weighted_power_ratio_minus_one": max_hour_normalization_error,
        "clean_1s_qa_sha256": sha256(source),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "power_normalization_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit("Power normalization QA failed")


if __name__ == "__main__":
    main()
