from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import build_fast_dataset as build


def write_metric(path: Path, times: np.ndarray, values: np.ndarray, value_type: pa.DataType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [pa.array(times.astype("datetime64[ms]"), type=pa.timestamp("ms")), pa.array(values, type=value_type)],
        names=["time", "value"],
    )
    pq.write_table(table, path, row_group_size=7)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fast_dataset_smoke_") as folder:
        root = Path(folder)
        run = root / "Run_011"
        base = np.datetime64("2025-09-30T14:00:00.100", "ms").astype(np.int64)
        times = base + np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 900, 1_000, 1_100, 7_000, 7_100], dtype=np.int64)
        peak = np.array([-77.0, -76.9, -77.1, -77.2, -77.0, -76.8, -77.2, -77.1, -76.9, -77.0, -77.0, -77.1, -77.2, -78.0, -78.1], dtype=np.float32)
        quality = np.zeros(len(times), dtype=np.uint16)
        quality[-2:] = 32
        source_k = np.full(len(times), 101, dtype=np.int32)
        for name, values, dtype in (
            ("peak_power_dbm", peak, pa.float32()),
            ("quality_code", quality, pa.uint16()),
            ("source_bin_count", source_k, pa.int32()),
        ):
            write_metric(run / "metrics" / name / f"{name}.parquet", times, values, dtype)
        mean_w = np.mean(np.power(10.0, (peak.astype(np.float64) - 30.0) / 10.0))
        hour = pd.Timestamp("2025-09-30 14:00:00", tz=build.TZ_BJT)
        hourly = pd.DataFrame(
            {
                "hour_bjt": [hour],
                "received_power_w": [mean_w],
                "received_power_dbm": [10.0 * np.log10(mean_w * 1000.0)],
                "split_blocked": ["train"],
                "split_strict_24h": ["train"],
            }
        )
        hourly_path = root / "measurement_hourly.parquet"
        hourly.to_parquet(hourly_path, index=False)
        output = root / "output"
        old_argv = __import__("sys").argv
        __import__("sys").argv = ["build_fast_dataset.py", "--run011", str(run), "--hourly", str(hourly_path), "--output", str(output), "--batch-size", "5"]
        try:
            result = build.main()
        finally:
            __import__("sys").argv = old_argv
        assert result == 0
        fast100 = pd.read_parquet(next((output / "fast_100ms").glob("*.parquet")))
        fast1 = pd.read_parquet(next((output / "fast_1s").glob("*.parquet")))
        assert len(fast100) == len(np.unique(times))
        assert fast100["time_bjt"].is_monotonic_increasing
        assert int(fast100["source_record_multiplicity"].max()) == 2
        assert fast100["continuous_segment_id"].nunique() == 2
        assert len(fast1) == 3
        assert (output / "fast_window_statistics.parquet").is_file()
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
