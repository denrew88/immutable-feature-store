import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from fs.array import open_shard
from fs.array.binary_storage import (
    build_array_binary_shards_from_bundles,
    load_array_binary_categorical_dictionaries,
    load_array_binary_shard_manifest,
)
from fs.array.builder import ArrayDatasetBuilder
from fs.array.metadata import write_feature_meta, write_sample_meta
from fs.array.storage import ArraySampleBundleWriter
from fs.config import ArrayBinaryBuildOptions, ArrayBundleConfig, ArrayShardConfig
from fs.types import LogicalType, PointColumnSpec, StorageType


def assert_trace(trace, expected_flags, expected_time, expected_value):
    assert trace is not None
    assert int(trace.flags) == int(expected_flags)
    np.testing.assert_equal(trace.columns["time"], np.asarray(expected_time, dtype=np.float64))
    np.testing.assert_equal(trace.columns["value"], np.asarray(expected_value, dtype=np.float64))


def main():
    """Run correctness tests for binary array shard build and lookup paths."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_array_binary_storage_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"

    pl.DataFrame(
        {
            "sample_id": pl.Series("sample_id", np.arange(10, dtype=np.int64), dtype=pl.Int64),
            "sample_key": pl.Series("sample_key", [f"sample_{i:03d}" for i in range(10)], dtype=pl.String),
        }
    ).write_parquet(sample_meta_path)
    pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", np.arange(3, dtype=np.int32), dtype=pl.Int32),
            "feature_key": pl.Series("feature_key", [f"feature_{i:03d}" for i in range(3)], dtype=pl.String),
        }
    ).write_parquet(feature_meta_path)

    generated_sample_meta_path = Path(
        write_sample_meta(
            [
                {"sample_key": "sample_gen_000", "split": "train"},
                {"sample_key": "sample_gen_001", "split": "train"},
                {"sample_key": "sample_gen_002", "split": "test"},
            ],
            root / "generated_sample_meta.parquet",
        )
    )
    generated_feature_meta_path = Path(
        write_feature_meta(
            [
                {"feature_key": "feature_gen_000", "family": "a"},
                {"feature_key": "feature_gen_001", "family": "b"},
            ],
            root / "generated_feature_meta.parquet",
        )
    )
    generated_sample_meta = pl.read_parquet(generated_sample_meta_path)
    generated_feature_meta = pl.read_parquet(generated_feature_meta_path)
    assert tuple(generated_sample_meta.columns) == ("sample_id", "sample_key", "split")
    assert tuple(generated_sample_meta["sample_id"].to_list()) == (0, 1, 2)
    assert tuple(generated_feature_meta.columns) == ("feature_id", "feature_key", "family")
    assert tuple(generated_feature_meta["feature_id"].to_list()) == (0, 1)
    assert PointColumnSpec(
        name="elapsed",
        storage_type=StorageType.INT64,
        logical_type=LogicalType.TIMEDELTA_NS,
    ).to_json() == {
        "name": "elapsed",
        "storage_type": "int64",
        "logical_type": "timedelta_ns",
    }
    try:
        PointColumnSpec(name="bad_time", storage_type=StorageType.UINT32, logical_type=LogicalType.TIMESTAMP_NS)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid timestamp storage type to raise ValueError")

    bundle_cfg = ArrayBundleConfig(max_bundle_rows=3, max_bundle_bytes=1 << 20)
    time_value_schema = [
        PointColumnSpec(name="time", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
        PointColumnSpec(name="value", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
    ]
    with ArraySampleBundleWriter(
        str(root),
        str(sample_meta_path),
        n_samples=10,
        feature_meta_path=str(feature_meta_path),
        config=bundle_cfg,
        point_schema=time_value_schema,
    ) as writer:
        writer.append_trace(0, 0, columns={"time": [0.0, 1.0, 2.0], "value": [10.0, 11.0, 12.0]})
        writer.append_trace(2, 0, columns={"time": [0.0, 1.0], "value": [20.0, 21.0]})
        writer.append_trace(5, 0, columns={"time": [], "value": []})
        writer.append_trace(8, 0, columns={"time": [0.0, np.nan], "value": [80.0, np.inf]})
        writer.append_trace(1, 1, columns={"time": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]})
        writer.append_trace(7, 1, columns={"time": [5.0], "value": [7.0]})
        bundle_manifest_path = writer.finish()

    binary_manifest_path, binary_stats = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "binary_shards"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=300),
        return_stats=True,
    )
    binary_default_manifest_path, binary_default_stats = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "binary_default_shards"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=300),
        return_stats=True,
    )
    binary_zstd_manifest_path, binary_zstd_stats = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "binary_zstd_shards"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=300),
        codec="zstd",
        zstd_level=3,
        return_stats=True,
    )
    binary_spill_manifest_path, binary_spill_stats = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "binary_spill_shards"),
        config=ArrayShardConfig(
            samples_per_block=4,
            target_shard_bytes=300,
            use_tmp_spill=True,
            spill_bucket_target_bytes=150,
        ),
        return_stats=True,
    )

    binary_manifest = load_array_binary_shard_manifest(binary_manifest_path)
    binary_default_manifest = load_array_binary_shard_manifest(binary_default_manifest_path)
    binary_zstd_manifest = load_array_binary_shard_manifest(binary_zstd_manifest_path)
    binary_spill_manifest = load_array_binary_shard_manifest(binary_spill_manifest_path)

    with open(binary_manifest_path, "r", encoding="utf-8") as f:
        binary_manifest_json = json.load(f)
    assert binary_manifest_json["sample_meta_path"] == "sample_meta.parquet"
    assert binary_manifest_json["feature_meta_path"] == "feature_meta.parquet"
    assert binary_manifest_json["shard_path"] == "array_binary_feature_shards"
    assert (root / "binary_shards" / "sample_meta.parquet").exists()
    assert (root / "binary_shards" / "feature_meta.parquet").exists()
    assert (root / "binary_shards" / "array_binary_feature_shards").is_dir()

    assert binary_manifest.n_shards >= 1
    assert binary_manifest.samples_per_block == 4
    assert binary_manifest.n_features == 3
    assert binary_manifest.blocks_per_feature == 3
    assert binary_manifest.id_scheme == "dense_row_ids"
    assert binary_manifest.sample_key_col == "sample_key"
    assert binary_manifest.feature_key_col == "feature_key"
    assert int(binary_stats["block_count"]) == 9
    assert int(binary_stats["feature_count"]) == 3
    assert int(binary_default_stats["block_count"]) == 9
    assert int(binary_zstd_stats["block_count"]) == 9
    assert int(binary_spill_stats["block_count"]) == 9
    assert binary_zstd_manifest.default_codec == "zstd"
    assert not (root / "binary_spill_shards" / "_tmp").exists()

    with open_shard(binary_manifest_path) as ds, open_shard(binary_default_manifest_path) as default_ds, open_shard(
        binary_zstd_manifest_path
    ) as zstd_ds, open_shard(binary_spill_manifest_path) as spill_ds:
        sample_rows = [0, 1, 2, 5, 8, 9]
        traces10 = ds.get_traces(feature_id=0, sample_ids=sample_rows)
        default_traces10 = default_ds.get_traces(feature_id=0, sample_ids=sample_rows)
        zstd_traces10 = zstd_ds.get_traces(feature_id=0, sample_ids=sample_rows)
        spill_traces10 = spill_ds.get_traces(feature_id=0, sample_ids=sample_rows)
        assert_trace(traces10.traces[0], 1, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
        assert_trace(traces10.traces[1], 0, [], [])
        assert_trace(traces10.traces[2], 1, [0.0, 1.0], [20.0, 21.0])
        assert_trace(traces10.traces[3], 3, [], [])
        assert_trace(traces10.traces[4], 1, [0.0, np.nan], [80.0, np.inf])
        assert_trace(traces10.traces[5], 0, [], [])
        for left_batch, right_batch in [
            (traces10, default_traces10),
            (traces10, zstd_traces10),
            (traces10, spill_traces10),
        ]:
            for left, right in zip(left_batch.traces, right_batch.traces):
                assert int(left.flags) == int(right.flags)
                np.testing.assert_equal(left.columns["time"], right.columns["time"])
                np.testing.assert_equal(left.columns["value"], right.columns["value"])

        traces20 = ds.get_traces(feature_id=1, sample_ids=[1, 7])
        default_traces20 = default_ds.get_traces(feature_id=1, sample_ids=[1, 7])
        zstd_traces20 = zstd_ds.get_traces(feature_id=1, sample_ids=[1, 7])
        spill_traces20 = spill_ds.get_traces(feature_id=1, sample_ids=[1, 7])
        assert_trace(traces20.traces[0], 1, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert_trace(traces20.traces[1], 1, [5.0], [7.0])
        for left_batch, right_batch in [
            (traces20, default_traces20),
            (traces20, zstd_traces20),
            (traces20, spill_traces20),
        ]:
            for left, right in zip(left_batch.traces, right_batch.traces):
                assert int(left.flags) == int(right.flags)
                np.testing.assert_equal(left.columns["time"], right.columns["time"])
                np.testing.assert_equal(left.columns["value"], right.columns["value"])

        by_ids = ds.get_traces(feature_id=0, sample_ids=[0, 2, 5, 8, 9999])
        assert_trace(by_ids.traces[0], 1, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
        assert_trace(by_ids.traces[1], 1, [0.0, 1.0], [20.0, 21.0])
        assert_trace(by_ids.traces[2], 3, [], [])
        assert_trace(by_ids.traces[3], 1, [0.0, np.nan], [80.0, np.inf])
        assert int(by_ids.traces[4].flags) == 0
        assert by_ids.traces[4].columns["time"].size == 0
        assert by_ids.traces[4].columns["value"].size == 0

        empty_feature = ds.get_traces(feature_id=2, sample_ids=[0, 4, 9])
        for trace in empty_feature.traces:
            assert int(trace.flags) == 0
            assert trace.columns["time"].size == 0
            assert trace.columns["value"].size == 0

    dict_dir = root / "categorical_dictionaries_source"
    dict_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = dict_dir / "state_code.json"
    event_dict_path = dict_dir / "event_type.json"
    state_dict_path.write_text(
        json.dumps(
            {
                "column": "state_code",
                "items": [
                    {"code": 1, "label": "OK"},
                    {"code": 2, "label": "WARN"},
                    {"code": 3, "label": "FAIL"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    event_dict_path.write_text(
        json.dumps(
            {
                "column": "event_type",
                "items": [
                    {"code": 1, "label": "START"},
                    {"code": 2, "label": "STOP"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    custom_schema = [
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(
            name="state_code",
            storage_type=StorageType.UINT32,
            logical_type=LogicalType.CATEGORICAL,
            dictionary_path=str(state_dict_path),
        ),
        PointColumnSpec(
            name="event_type",
            storage_type=StorageType.UINT32,
            logical_type=LogicalType.CATEGORICAL,
            dictionary_path=str(event_dict_path),
        ),
    ]
    with ArraySampleBundleWriter(
        str(root / "v3_bundles"),
        str(sample_meta_path),
        n_samples=10,
        feature_meta_path=str(feature_meta_path),
        config=bundle_cfg,
        point_schema=custom_schema,
    ) as writer:
        writer.append_trace(
            0,
            0,
            columns={
                "phase": np.asarray([10, 11, 12], dtype=np.int32),
                "state_code": np.asarray([1, 1, 2], dtype=np.uint32),
                "event_type": np.asarray([1, 1, 2], dtype=np.uint32),
            },
        )
        writer.append_trace(
            2,
            0,
            columns={
                "phase": np.asarray([20], dtype=np.int32),
                "state_code": np.asarray([3], dtype=np.uint32),
                "event_type": np.asarray([2], dtype=np.uint32),
            },
        )
        custom_bundle_manifest_path = writer.finish()

    binary_v3_manifest_path = build_array_binary_shards_from_bundles(
        custom_bundle_manifest_path,
        str(root / "binary_v3_shards"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=300),
    )
    binary_v3_manifest = load_array_binary_shard_manifest(binary_v3_manifest_path)
    assert int(binary_v3_manifest.version) == 3
    assert [spec.name for spec in binary_v3_manifest.point_schema] == ["phase", "state_code", "event_type"]
    assert (root / "binary_v3_shards" / "categorical_dictionaries" / "state_code.json").exists()
    assert (root / "binary_v3_shards" / "categorical_dictionaries" / "event_type.json").exists()
    dictionaries = load_array_binary_categorical_dictionaries(binary_v3_manifest)
    assert dictionaries["state_code"][1] == "OK"
    assert dictionaries["event_type"][2] == "STOP"

    with open_shard(binary_v3_manifest_path) as ds:
        v3_traces = ds.get_traces(feature_id=0, sample_ids=[0, 1, 2, 9])
        np.testing.assert_equal(v3_traces.traces[0].columns["phase"], np.asarray([10, 11, 12], dtype=np.int32))
        np.testing.assert_equal(v3_traces.traces[0].columns["state_code"], np.asarray([1, 1, 2], dtype=np.uint32))
        np.testing.assert_equal(v3_traces.traces[0].columns["event_type"], np.asarray([1, 1, 2], dtype=np.uint32))
        assert "time" not in v3_traces.traces[0].columns
        assert "value" not in v3_traces.traces[0].columns
        np.testing.assert_equal(v3_traces.traces[2].columns["phase"], np.asarray([20], dtype=np.int32))
        assert int(v3_traces.traces[1].flags) == 0
        assert set(v3_traces.traces[1].columns.keys()) == {"phase", "state_code", "event_type"}
        for values in v3_traces.traces[1].columns.values():
            assert values.size == 0

        decoded_trace = ds.get_trace(feature_id=0, sample_id=0, decode_categorical=True)
        assert tuple(decoded_trace.columns["state_code"]) == ("OK", "OK", "WARN")
        assert tuple(decoded_trace.columns["event_type"]) == ("START", "START", "STOP")

    builder_schema = [
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ]
    known_builder = ArrayDatasetBuilder(
        out_dir=str(root / "builder_known"),
        sample_meta_path=str(sample_meta_path),
        point_schema=builder_schema,
        feature_keys=["feature_alpha", "feature_beta"],
        build_options=ArrayBinaryBuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    with known_builder.sample(sample_key="sample_000") as sample:
        sample.add_trace(
            feature_key="feature_alpha",
            columns={"phase": [1, 2], "state_code": ["OK", "WARN"]},
        )
    with known_builder.sample(sample_key="sample_001"):
        pass
    with known_builder.sample(sample_key="sample_002"):
        pass
    with known_builder.sample(sample_key="sample_003") as sample:
        sample.add_trace(
            feature_id=1,
            columns={"phase": [9], "state_code": ["FAIL"]},
        )
    known_status = known_builder.status()
    assert known_status.last_committed_sample_id is None
    assert known_status.next_expected_sample_id == 0
    assert known_status.buffered_through_sample_id == 3
    assert known_status.in_progress_sample_id is None
    known_builder_manifest_path = known_builder.finish()
    with open_shard(known_builder_manifest_path) as ds:
        known_trace = ds.get_trace_by_key(
            feature_key="feature_alpha",
            sample_key="sample_000",
            decode_categorical=True,
        )
        np.testing.assert_equal(known_trace.columns["phase"], np.asarray([1, 2], dtype=np.int32))
        assert tuple(known_trace.columns["state_code"]) == ("OK", "WARN")
        assert ds.categorical_dictionaries()["state_code"][3] == "FAIL"

    discovered_builder = ArrayDatasetBuilder(
        out_dir=str(root / "builder_discovered"),
        sample_meta_path=str(sample_meta_path),
        point_schema=builder_schema,
        build_options=ArrayBinaryBuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    try:
        with discovered_builder.sample(sample_id=0) as sample:
            sample.add_trace(
                feature_id=0,
                columns={"phase": [1], "state_code": ["OK"]},
            )
    except ValueError:
        pass
    else:
        raise AssertionError("expected discovered-feature mode to require feature_key")
    with discovered_builder.sample(sample_id=0) as sample:
        sample.add_trace(
            feature_key="feature_zeta",
            columns={"phase": [4, 5], "state_code": ["WARN", "FAIL"]},
        )
    with discovered_builder.sample(sample_id=1) as sample:
        sample.add_trace(
            feature_key="feature_alpha",
            columns={"phase": [3], "state_code": ["OK"]},
        )
    updated_feature_meta_path = Path(
        discovered_builder.update_feature_meta(
            [
                {"feature_key": "feature_zeta", "group": "latent_a", "rank_hint": 10},
                {"feature_key": "feature_alpha", "group": "latent_b", "rank_hint": 20},
            ],
            require_all=True,
        )
    )
    updated_feature_meta = pl.read_parquet(updated_feature_meta_path)
    assert tuple(updated_feature_meta.columns) == ("feature_id", "feature_key", "group", "rank_hint")
    discovered_builder_manifest_path = discovered_builder.build_shards()
    with open_shard(discovered_builder_manifest_path) as ds:
        discovered_feature_meta = pl.read_parquet(Path(discovered_builder_manifest_path).resolve().parent / "feature_meta.parquet")
        assert tuple(discovered_feature_meta.columns) == ("feature_id", "feature_key", "group", "rank_hint")
        assert tuple(discovered_feature_meta["feature_key"].to_list()) == ("feature_zeta", "feature_alpha")
        discovered_trace = ds.get_trace_by_key(
            feature_key="feature_zeta",
            sample_key="sample_000",
            decode_categorical=True,
        )
        np.testing.assert_equal(discovered_trace.columns["phase"], np.asarray([4, 5], dtype=np.int32))
        assert tuple(discovered_trace.columns["state_code"]) == ("WARN", "FAIL")

    strict_builder = ArrayDatasetBuilder(
        out_dir=str(root / "builder_strict"),
        sample_meta_path=str(sample_meta_path),
        point_schema=builder_schema,
        feature_keys=["feature_only"],
        build_options=ArrayBinaryBuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    try:
        with strict_builder.sample(sample_id=0) as sample:
            sample.add_trace(
                feature_key="feature_only",
                columns={"phase": [1], "state_code": ["OK"], "extra_column": [99]},
            )
    except ValueError:
        pass
    else:
        raise AssertionError("expected extra point column to raise ValueError")

    print("python array binary storage tests passed")


if __name__ == "__main__":
    main()
