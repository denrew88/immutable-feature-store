import shutil
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SRC = REPO_ROOT / "packages" / "array_sample_parquet" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
    open_array_sample_parquet,
)


def main():
    root = Path("data/tmp_array_sample_parquet_pkg_test").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    pl.DataFrame({"sample_id": [0], "sample_key": ["s0"]}).write_parquet(root / "sample_meta.parquet")
    pl.DataFrame({"feature_id": [0, 1], "feature_key": ["f0", "f1"]}).write_parquet(root / "feature_meta.parquet")
    schema = [PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS)]
    builder = ArraySampleParquetDatasetBuilder.open_session(
        root / "dataset",
        root / "sample_meta.parquet",
        schema,
        feature_meta_path=root / "feature_meta.parquet",
        options=ArraySampleParquetBuildOptions(compression="none"),
    )
    with builder.sample(sample_id=0) as sample:
        sample.add_trace(feature_key="f1", columns={"value": [3.0]})
        sample.add_trace(feature_key="f0", columns={"value": [1.0, 2.0]})
    manifest_path = builder.finish()
    reader = open_array_sample_parquet(manifest_path)
    part = reader.manifest.parts[0]
    point_df = pl.read_parquet(part.path)
    assert point_df.select(["sample_id", "feature_id", "point_idx"]).equals(
        point_df.select(["sample_id", "feature_id", "point_idx"]).sort(["sample_id", "feature_id", "point_idx"])
    )
    traces = reader.get_traces(sample_keys=["s0"], feature_keys=["f0"])
    assert traces[0].columns["value"] == [1.0, 2.0]

    raw_schema = [
        PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("step", StorageType.STRING, LogicalType.CATEGORICAL),
    ]
    raw_builder = ArraySampleParquetDatasetBuilder.open_session(
        root / "raw_dataset",
        root / "sample_meta.parquet",
        raw_schema,
        feature_meta_path=root / "feature_meta.parquet",
        options=ArraySampleParquetBuildOptions(compression="none"),
    )
    assert raw_builder.pending_sample_ids() == [0]
    with raw_builder.sample(sample_id=0) as sample:
        sample.add_trace(feature_key="f1", columns={"value": [3.0], "step": ["A"]})
        sample.add_trace(feature_key="f0", columns={"value": [1.0, 2.0], "step": ["B", "A"]})
    raw_manifest = raw_builder.compact()
    raw_sample_df = pl.read_parquet(root / "raw_dataset" / "raw_samples" / "sample_000000000000.parquet")
    raw_trace_index_df = pl.read_parquet(root / "raw_dataset" / "raw_trace_index" / "sample_000000000000.parquet")
    assert "point_idx" in raw_sample_df.columns
    assert "trace_len" not in raw_sample_df.columns
    assert raw_trace_index_df.select(pl.col("trace_len").sum()).item() == raw_sample_df.height
    raw_reader = open_array_sample_parquet(raw_manifest)
    raw_part = raw_reader.manifest.parts[0]
    raw_point_df = pl.read_parquet(raw_part.path)
    raw_trace_index_df = pl.read_parquet(raw_part.trace_index_path)
    assert "point_idx" in raw_point_df.columns
    assert "trace_len" not in raw_point_df.columns
    assert raw_trace_index_df.select(pl.col("trace_len").sum()).item() == raw_point_df.height
    assert raw_point_df.select(["sample_id", "feature_id", "point_idx"]).equals(
        raw_point_df.select(["sample_id", "feature_id", "point_idx"]).sort(["sample_id", "feature_id", "point_idx"])
    )
    raw_traces = raw_reader.get_traces(sample_keys=["s0"], feature_keys=["f0"], decode_categorical=True)
    assert raw_traces[0].columns["step"] == ["B", "A"]
    assert raw_reader.point_schema()[1].storage_type == StorageType.STRING
    print("array_sample_parquet package smoke passed")


if __name__ == "__main__":
    main()
