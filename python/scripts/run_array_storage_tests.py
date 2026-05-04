import shutil
from pathlib import Path

import numpy as np
import polars as pl

from fs.array import open_shard
from fs.array.binary_storage import build_array_binary_shards_from_bundles, load_array_binary_shard_manifest
from fs.array.storage import ArraySampleBundleWriter, FLAG_EMPTY, FLAG_PRESENT
from fs.config import ArrayBundleConfig, ArrayShardConfig
from fs.types import LogicalType, PointColumnSpec, StorageType


def assert_trace(trace, expected_flags, expected_time, expected_value):
    assert trace is not None
    assert int(trace.flags) == int(expected_flags)
    np.testing.assert_equal(trace.columns["time"], np.asarray(expected_time, dtype=np.float64))
    np.testing.assert_equal(trace.columns["value"], np.asarray(expected_value, dtype=np.float64))


def main():
    """Run bundle-stage and binary array shard smoke tests."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_array_storage_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    pl.DataFrame(
        {
            "sample_id": pl.Series("sample_id", np.arange(10, dtype=np.int64), dtype=pl.Int64),
            "sample_key": pl.Series("sample_key", [f"sample_{i:06d}" for i in range(10)], dtype=pl.String),
        }
    ).write_parquet(sample_meta_path)
    pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", np.arange(2, dtype=np.int32), dtype=pl.Int32),
            "feature_key": pl.Series("feature_key", [f"feature_{i:06d}" for i in range(2)], dtype=pl.String),
        }
    ).write_parquet(feature_meta_path)

    bundle_cfg = ArrayBundleConfig(max_bundle_rows=3, max_bundle_bytes=1 << 20)
    point_schema = [
        PointColumnSpec(name="time", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
        PointColumnSpec(name="value", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
    ]
    with ArraySampleBundleWriter(
        str(root),
        str(sample_meta_path),
        n_samples=10,
        feature_meta_path=str(feature_meta_path),
        config=bundle_cfg,
        point_schema=point_schema,
    ) as writer:
        writer.append_trace(0, 0, columns={"time": [0.0, 1.0, 2.0], "value": [10.0, 11.0, 12.0]})
        writer.append_trace(2, 0, columns={"time": [0.0, 1.0], "value": [20.0, 21.0]})
        writer.append_trace(5, 0, columns={"time": [], "value": []})
        writer.append_trace(8, 0, columns={"time": [0.0, np.nan], "value": [80.0, np.inf]})
        writer.append_trace(1, 1, columns={"time": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]})
        writer.append_trace(7, 1, columns={"time": [5.0], "value": [7.0]})
        bundle_manifest_path = writer.finish()

    shard_manifest_path = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "shards"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=300),
    )
    spill_manifest_path = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "shards_spill"),
        config=ArrayShardConfig(
            samples_per_block=4,
            target_shard_bytes=300,
            use_tmp_spill=True,
            spill_bucket_target_bytes=150,
        ),
    )

    manifest = load_array_binary_shard_manifest(shard_manifest_path)
    spill_manifest = load_array_binary_shard_manifest(spill_manifest_path)
    assert manifest.samples_per_block == 4
    assert manifest.n_features == 2
    assert manifest.blocks_per_feature == 3
    assert len(manifest.shards) >= 1
    assert not (root / "shards_spill" / "_tmp").exists()

    with open_shard(shard_manifest_path) as ds, open_shard(spill_manifest_path) as spill_ds:
        traces10 = ds.get_traces(feature_id=0, sample_ids=[0, 1, 2, 5, 8, 9])
        spill_traces10 = spill_ds.get_traces(feature_id=0, sample_ids=[0, 1, 2, 5, 8, 9])
        assert_trace(traces10.traces[0], FLAG_PRESENT, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
        assert_trace(traces10.traces[1], 0, [], [])
        assert_trace(traces10.traces[2], FLAG_PRESENT, [0.0, 1.0], [20.0, 21.0])
        assert_trace(traces10.traces[3], FLAG_PRESENT | FLAG_EMPTY, [], [])
        assert_trace(traces10.traces[4], FLAG_PRESENT, [0.0, np.nan], [80.0, np.inf])
        assert_trace(traces10.traces[5], 0, [], [])
        for left, right in zip(traces10.traces, spill_traces10.traces):
            assert int(left.flags) == int(right.flags)
            np.testing.assert_equal(left.columns["time"], right.columns["time"])
            np.testing.assert_equal(left.columns["value"], right.columns["value"])

        traces20 = ds.get_traces(feature_id=1, sample_ids=[1, 7])
        spill_traces20 = spill_ds.get_traces(feature_id=1, sample_ids=[1, 7])
        assert_trace(traces20.traces[0], FLAG_PRESENT, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert_trace(traces20.traces[1], FLAG_PRESENT, [5.0], [7.0])
        for left, right in zip(traces20.traces, spill_traces20.traces):
            assert int(left.flags) == int(right.flags)
            np.testing.assert_equal(left.columns["time"], right.columns["time"])
            np.testing.assert_equal(left.columns["value"], right.columns["value"])

    print("python array storage tests passed")


if __name__ == "__main__":
    main()
