import shutil
from pathlib import Path

import numpy as np
import polars as pl

from fs.array.storage import (
    ArraySampleBundleWriter,
    ArrayShardReader,
    FLAG_EMPTY,
    FLAG_HAS_NONFINITE_TIME,
    FLAG_HAS_NONFINITE_VALUE,
    FLAG_PRESENT,
    build_array_feature_locator_index,
    build_array_shards_from_bundles,
    load_array_shard_manifest,
)
from fs.config import ArrayBundleConfig, ArrayShardConfig


def assert_trace(trace, expected_flags, expected_time, expected_value):
    """Assert that one trace matches the expected flags and payloads."""
    assert trace is not None
    assert int(trace.flags) == int(expected_flags)
    np.testing.assert_equal(trace.time, np.asarray(expected_time, dtype=np.float64))
    np.testing.assert_equal(trace.value, np.asarray(expected_value, dtype=np.float64))


def main():
    """Run correctness tests for parquet array shard build and lookup paths."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_array_storage_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    bundle_cfg = ArrayBundleConfig(max_bundle_rows=3, max_bundle_bytes=1 << 20)
    with ArraySampleBundleWriter(str(root), "", n_samples=10, config=bundle_cfg) as writer:
        writer.append_trace(0, 1000, 10, columns={"time": [0.0, 1.0, 2.0], "value": [10.0, 11.0, 12.0]})
        writer.append_trace(2, 1002, 10, columns={"time": [0.0, 1.0], "value": [20.0, 21.0]})
        writer.append_trace(5, 1005, 10, columns={"time": [], "value": []})
        writer.append_trace(8, 1008, 10, columns={"time": [0.0, np.nan], "value": [80.0, np.inf]})
        writer.append_trace(1, 1001, 20, columns={"time": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]})
        writer.append_trace(7, 1007, 20, columns={"time": [5.0], "value": [7.0]})
        bundle_manifest_path = writer.finish()

    shard_cfg = ArrayShardConfig(samples_per_block=4, target_shard_bytes=300)
    shard_manifest_path = build_array_shards_from_bundles(
        bundle_manifest_path,
        str(root / "shards"),
        config=shard_cfg,
    )
    spill_manifest_path = build_array_shards_from_bundles(
        bundle_manifest_path,
        str(root / "shards_spill"),
        config=ArrayShardConfig(
            samples_per_block=4,
            target_shard_bytes=300,
            use_tmp_spill=True,
            spill_bucket_target_bytes=150,
        ),
    )

    manifest = load_array_shard_manifest(shard_manifest_path)
    spill_manifest = load_array_shard_manifest(spill_manifest_path)
    locator_index = build_array_feature_locator_index(manifest.locator_path)
    spill_locator_index = build_array_feature_locator_index(spill_manifest.locator_path)
    assert len(locator_index[10]) == 3
    assert len(locator_index[20]) == 2
    assert {loc.shard_id for loc in locator_index[10]} == {0}
    assert {loc.shard_id for loc in locator_index[20]} == {1}
    assert pl.read_parquet(manifest.locator_path).sort(["feature_id", "block_id"]).equals(
        pl.read_parquet(spill_manifest.locator_path).sort(["feature_id", "block_id"])
    )
    assert not (root / "shards_spill" / "_tmp").exists()

    reader = ArrayShardReader(manifest)
    spill_reader = ArrayShardReader(spill_manifest)
    traces10 = reader.load_feature_samples(10, [0, 1, 2, 5, 8, 9], locator_index=locator_index)
    spill_traces10 = spill_reader.load_feature_samples(10, [0, 1, 2, 5, 8, 9], locator_index=spill_locator_index)
    assert_trace(traces10[0], FLAG_PRESENT, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
    assert_trace(traces10[1], 0, [], [])
    assert_trace(traces10[2], FLAG_PRESENT, [0.0, 1.0], [20.0, 21.0])
    assert_trace(traces10[5], FLAG_PRESENT | FLAG_EMPTY, [], [])
    assert_trace(
        traces10[8],
        FLAG_PRESENT | FLAG_HAS_NONFINITE_TIME | FLAG_HAS_NONFINITE_VALUE,
        [0.0, np.nan],
        [80.0, np.inf],
    )
    assert_trace(traces10[9], 0, [], [])
    for sample_row, trace in traces10.items():
        spill_trace = spill_traces10[sample_row]
        assert int(trace.flags) == int(spill_trace.flags)
        np.testing.assert_equal(trace.time, spill_trace.time)
        np.testing.assert_equal(trace.value, spill_trace.value)

    traces20 = reader.load_feature_samples(20, [1, 7], locator_index=locator_index)
    spill_traces20 = spill_reader.load_feature_samples(20, [1, 7], locator_index=spill_locator_index)
    assert_trace(traces20[1], FLAG_PRESENT, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert_trace(traces20[7], FLAG_PRESENT, [5.0], [7.0])
    for sample_row, trace in traces20.items():
        spill_trace = spill_traces20[sample_row]
        assert int(trace.flags) == int(spill_trace.flags)
        np.testing.assert_equal(trace.time, spill_trace.time)
        np.testing.assert_equal(trace.value, spill_trace.value)

    print("python array storage tests passed")


if __name__ == "__main__":
    main()
