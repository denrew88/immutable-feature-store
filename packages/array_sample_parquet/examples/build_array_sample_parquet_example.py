from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SRC = REPO_ROOT / "packages" / "array_sample_parquet" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetRawDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
    open_array_sample_parquet,
)


def _columns(offset: float, length: int, label: str) -> dict[str, object]:
    return {
        "time": np.arange(length, dtype=np.float64),
        "value": offset + np.arange(length, dtype=np.float64) * 0.1,
        "ch_step": [label] * length,
    }


def main() -> None:
    root = REPO_ROOT / "data" / "tmp_array_sample_parquet_python_example"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    pl.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "sample_key": ["sample_000000", "sample_000001", "sample_000002", "sample_000003"],
            "split": ["train", "train", "valid", "valid"],
        }
    ).write_parquet(sample_meta_path)
    pl.DataFrame(
        {
            "feature_id": [0, 1],
            "feature_key": ["feature_a", "feature_b"],
            "group": ["alpha", "beta"],
        }
    ).write_parquet(feature_meta_path)

    point_schema = [
        PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL),
    ]
    options = ArraySampleParquetBuildOptions(
        target_part_bytes=16 * 1024 * 1024,
        max_part_rows=1_000_000,
        max_part_samples=0,
        compression="zstd",
    )
    dataset_dir = root / "array_sample_parquet"

    # 첫 실행에서는 sample 2개만 완료하고 종료된 상황을 가정합니다.
    with ArraySampleParquetRawDatasetBuilder.open_session(
        dataset_dir,
        sample_meta_path,
        point_schema,
        feature_meta_path=feature_meta_path,
        options=options,
    ) as builder:
        with builder.sample(sample_id=2) as sample:
            sample.add_trace(feature_key="feature_b", columns=_columns(20.0, 2, "B"))
        with builder.sample(sample_id=0) as sample:
            sample.add_trace(feature_key="feature_a", columns=_columns(10.0, 3, "A"))
            sample.add_trace(feature_key="feature_b", columns=_columns(30.0, 0, "empty"))
        print("first_pending=", builder.pending_sample_ids())

    # 같은 session을 다시 열고 pending sample만 채운 뒤 compact합니다.
    with ArraySampleParquetRawDatasetBuilder.open_session(
        dataset_dir,
        sample_meta_path,
        point_schema,
        feature_meta_path=feature_meta_path,
        options=options,
    ) as builder:
        with builder.sample(sample_id=0, skip_if_completed=True) as skipped:
            print("sample_0_skipped=", skipped.skipped)
        for sample_id in builder.pending_sample_ids():
            with builder.sample(sample_id=sample_id, skip_if_completed=True) as sample:
                if sample.skipped:
                    continue
                sample.add_trace(feature_key="feature_a", columns=_columns(100.0 + sample_id, 2, "A"))
        manifest_path = builder.compact(require_all=True)

    reader = open_array_sample_parquet(manifest_path)
    payload = reader.get_traces_json(
        sample_keys=["sample_000000", "sample_000002"],
        feature_keys=["feature_a", "feature_b"],
        include_missing=True,
        layout="nested",
    )
    print("trace_count=", payload["trace_count"])
    print("sample_meta=", sample_meta_path)
    print("feature_meta=", feature_meta_path)
    print("manifest=", manifest_path)


if __name__ == "__main__":
    main()
