from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from fs.array.metadata import write_feature_meta, write_sample_meta
from fs.array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    open_array_sample_parquet,
)
from fs.types import LogicalType, PointColumnSpec, StorageType
from scripts.serve_array_api import (
    ArraySampleParquetSchemaRequest,
    ArraySampleParquetTraceRequest,
    array_sample_parquet_schema,
    array_sample_parquet_traces,
)


def _row(**kwargs):
    return dict(kwargs)


def _columns(ts, value, phase, ch_step):
    return {
        "ts": np.asarray(ts, dtype="datetime64[ns]"),
        "value": np.asarray(value, dtype=np.float64),
        "phase": np.asarray(phase, dtype=np.int32),
        "ch_step": list(ch_step),
    }


def main():
    root = Path("data/tmp_array_sample_parquet_test").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    sample_meta_path = write_sample_meta(
        [
            _row(sample_key="sample_000000", split="train"),
            _row(sample_key="sample_000001", split="train"),
            _row(sample_key="sample_000002", split="valid"),
            _row(sample_key="sample_000003", split="test"),
        ],
        root / "sample_meta.parquet",
    )
    feature_meta_path = write_feature_meta(
        [
            _row(feature_key="feature_a", group="alpha"),
            _row(feature_key="feature_b", group="beta"),
        ],
        root / "feature_meta.parquet",
    )
    schema = [
        PointColumnSpec("ts", StorageType.INT64, LogicalType.TIMESTAMP_NS),
        PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER),
        PointColumnSpec("ch_step", StorageType.UINT32, LogicalType.CATEGORICAL),
    ]
    options = ArraySampleParquetBuildOptions(
        target_part_bytes=1024 * 1024,
        max_part_samples=2,
        compression="none",
    )
    out_dir = root / "dataset"

    with ArraySampleParquetDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        schema,
        feature_meta_path=feature_meta_path,
        options=options,
    ) as builder:
        with builder.sample(sample_id=0) as sample:
            sample.add_trace(
                feature_key="feature_a",
                columns=_columns(
                    ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                    [1.0, np.nan],
                    [10, 11],
                    ["A", "B"],
                ),
            )
        with builder.sample(sample_id=1):
            pass
        status = builder.status()
        assert status.next_expected_sample_id == 2, status
        assert status.committed_part_count == 1, status

    with ArraySampleParquetDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        schema,
        feature_meta_path=feature_meta_path,
        options=options,
    ) as builder:
        status = builder.status()
        assert status.next_expected_sample_id == 2, status
        with builder.sample(sample_key="sample_000002") as sample:
            sample.add_trace(
                feature_key="feature_b",
                columns=_columns(
                    ["2024-01-01T00:00:02"],
                    [3.0],
                    [12],
                    ["B"],
                ),
            )
        with builder.sample(sample_id=3) as sample:
            sample.add_trace(
                feature_key="feature_a",
                columns=_columns([], [], [], []),
            )
        manifest_path = builder.finish()

    assert os.path.exists(manifest_path), manifest_path
    reader = open_array_sample_parquet(manifest_path)
    traces = reader.get_traces(
        sample_keys=["sample_000000", "sample_000001", "sample_000002", "sample_000003"],
        feature_keys=["feature_a", "feature_b"],
        include_missing=True,
        decode_categorical=True,
    )
    by_pair = {(trace.sample_id, trace.feature_id): trace for trace in traces}
    assert by_pair[(0, 0)].present
    assert by_pair[(0, 0)].columns["ch_step"] == ["A", "B"]
    assert by_pair[(0, 0)].trace_len == 2
    assert not by_pair[(1, 0)].present
    assert by_pair[(2, 1)].columns["ch_step"] == ["B"]
    assert by_pair[(3, 0)].present
    assert by_pair[(3, 0)].trace_len == 0

    nested = reader.get_traces_json(
        sample_ids=[0, 2],
        feature_ids=[0, 1],
        include_missing=True,
        decode_categorical=True,
        layout="nested",
    )
    assert nested["layout"] == "nested"
    assert nested["sample_count"] == 2
    assert nested["trace_count"] == 4

    flat = reader.get_traces_json(
        sample_ids=[0],
        feature_ids=[0],
        decode_categorical=True,
        layout="flat",
    )
    assert flat["layout"] == "flat"
    assert flat["traces"][0]["sample_key"] == "sample_000000"
    assert flat["traces"][0]["feature_key"] == "feature_a"

    api_schema = array_sample_parquet_schema(
        ArraySampleParquetSchemaRequest(manifest_path=manifest_path, include_dictionaries=True)
    )
    assert api_schema["format"] == "array-sample-parquet"
    assert "ch_step" in api_schema["categorical_dictionaries"]

    api_traces = array_sample_parquet_traces(
        ArraySampleParquetTraceRequest(
            manifest_path=manifest_path,
            sample_keys=["sample_000000", "sample_000002"],
            feature_keys=["feature_a", "feature_b"],
            include_missing=True,
            decode_categorical=True,
            layout="nested",
        )
    )
    assert api_traces["trace_count"] == 4
    assert api_traces["samples"][0]["sample_key"] == "sample_000000"
    print("array_sample_parquet tests passed")


if __name__ == "__main__":
    main()
