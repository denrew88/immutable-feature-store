import shutil
from pathlib import Path

import polars as pl

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
    pl.DataFrame({"feature_id": [0], "feature_key": ["f0"]}).write_parquet(root / "feature_meta.parquet")
    schema = [PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS)]
    builder = ArraySampleParquetDatasetBuilder.open_session(
        root / "dataset",
        root / "sample_meta.parquet",
        schema,
        feature_meta_path=root / "feature_meta.parquet",
        options=ArraySampleParquetBuildOptions(compression="none"),
    )
    with builder.sample(sample_id=0) as sample:
        sample.add_trace(feature_key="f0", columns={"value": [1.0, 2.0]})
    manifest_path = builder.finish()
    reader = open_array_sample_parquet(manifest_path)
    traces = reader.get_traces(sample_keys=["s0"], feature_keys=["f0"])
    assert traces[0].columns["value"] == [1.0, 2.0]
    print("array_sample_parquet package smoke passed")


if __name__ == "__main__":
    main()
