import shutil
import json
from pathlib import Path

import numpy as np
import polars as pl

from fs.array.binary_storage import (
    ArrayBinaryShardReader,
    build_array_binary_shards_from_bundles,
    build_array_binary_shards_from_array_manifest,
    load_array_binary_categorical_dictionaries,
    load_array_binary_shard_manifest,
)
from fs.array.builder import ArrayDatasetBuilder
from fs.array.metadata import write_feature_meta, write_sample_meta
from fs.array.storage import (
    ArraySampleBundleWriter,
    ArrayShardReader,
    build_array_shards_from_bundles,
    load_array_shard_manifest,
)
from fs.config import ArrayBinaryBuildOptions, ArrayBundleConfig, ArrayShardConfig
from fs.types import LogicalType, PointColumnSpec, StorageType


def assert_trace(trace, expected_flags, expected_time, expected_value):
    """Assert that one trace matches the expected flags and payloads."""
    assert trace is not None
    assert int(trace.flags) == int(expected_flags)
    np.testing.assert_equal(trace.time, np.asarray(expected_time, dtype=np.float64))
    np.testing.assert_equal(trace.value, np.asarray(expected_value, dtype=np.float64))


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
            "sample_row": pl.Series("sample_row", np.arange(10, dtype=np.int64), dtype=pl.Int64),
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
    assert tuple(generated_sample_meta["sample_key"].to_list()) == (
        "sample_gen_000",
        "sample_gen_001",
        "sample_gen_002",
    )
    assert tuple(generated_feature_meta.columns) == ("feature_id", "feature_key", "family")
    assert tuple(generated_feature_meta["feature_id"].to_list()) == (0, 1)
    assert tuple(generated_feature_meta["feature_key"].to_list()) == (
        "feature_gen_000",
        "feature_gen_001",
    )
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
        PointColumnSpec(
            name="bad_time",
            storage_type=StorageType.UINT32,
            logical_type=LogicalType.TIMESTAMP_NS,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected invalid timestamp storage type to raise ValueError")

    bundle_cfg = ArrayBundleConfig(max_bundle_rows=3, max_bundle_bytes=1 << 20)
    with ArraySampleBundleWriter(
        str(root),
        str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        n_samples=10,
        config=bundle_cfg,
    ) as writer:
        writer.append_trace(0, 0, 0, columns={"time": [0.0, 1.0, 2.0], "value": [10.0, 11.0, 12.0]})
        writer.append_trace(2, 2, 0, columns={"time": [0.0, 1.0], "value": [20.0, 21.0]})
        writer.append_trace(5, 5, 0, columns={"time": [], "value": []})
        writer.append_trace(8, 8, 0, columns={"time": [0.0, np.nan], "value": [80.0, np.inf]})
        writer.append_trace(1, 1, 1, columns={"time": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]})
        writer.append_trace(7, 7, 1, columns={"time": [5.0], "value": [7.0]})
        bundle_manifest_path = writer.finish()

    parquet_manifest_path = build_array_shards_from_bundles(
        bundle_manifest_path,
        str(root / "parquet_shards"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=300, row_group_size=2),
    )
    binary_manifest_path, binary_stats = build_array_binary_shards_from_array_manifest(
        parquet_manifest_path,
        str(root / "binary_shards"),
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

    parquet_manifest = load_array_shard_manifest(parquet_manifest_path)
    binary_manifest = load_array_binary_shard_manifest(binary_manifest_path)
    binary_default_manifest = load_array_binary_shard_manifest(binary_default_manifest_path)
    binary_zstd_manifest = load_array_binary_shard_manifest(binary_zstd_manifest_path)
    binary_spill_manifest = load_array_binary_shard_manifest(binary_spill_manifest_path)
    parquet_reader = ArrayShardReader(parquet_manifest)
    binary_reader = ArrayBinaryShardReader(binary_manifest)
    binary_default_reader = ArrayBinaryShardReader(binary_default_manifest)
    binary_zstd_reader = ArrayBinaryShardReader(binary_zstd_manifest)
    binary_spill_reader = ArrayBinaryShardReader(binary_spill_manifest)

    with open(binary_manifest_path, "r", encoding="utf-8") as f:
        binary_manifest_json = json.load(f)
    assert binary_manifest_json["sample_meta_path"] == "sample_meta.parquet"
    assert binary_manifest_json["feature_meta_path"] == "feature_meta.parquet"
    assert binary_manifest_json["shard_path"] == "array_binary_feature_shards"
    assert (root / "binary_shards" / "sample_meta.parquet").exists()
    assert (root / "binary_shards" / "feature_meta.parquet").exists()
    assert (root / "binary_shards" / "array_binary_feature_shards").is_dir()

    assert binary_manifest.n_shards == parquet_manifest.n_shards
    assert binary_manifest.samples_per_block == parquet_manifest.samples_per_block
    assert binary_manifest.n_features == 3
    assert binary_manifest.blocks_per_feature == 3
    assert binary_manifest.id_scheme == "dense_row_ids"
    assert binary_manifest.sample_key_col == "sample_key"
    assert binary_manifest.feature_key_col == "feature_key"
    locator_rows = pl.read_parquet(parquet_manifest.locator_path).height
    assert int(binary_stats["block_count"]) == 9
    assert int(binary_stats["feature_count"]) == 3
    assert int(binary_default_stats["block_count"]) == 9
    assert int(binary_default_stats["feature_count"]) == 3
    assert int(binary_zstd_stats["block_count"]) == 9
    assert int(binary_zstd_stats["feature_count"]) == 3
    assert int(binary_spill_stats["block_count"]) == 9
    assert int(binary_spill_stats["feature_count"]) == 3
    assert int(locator_rows) == 5
    assert binary_zstd_manifest.default_codec == "zstd"
    assert not (root / "binary_spill_shards" / "_tmp").exists()

    sample_rows = [0, 1, 2, 5, 8, 9]
    parquet_traces10 = parquet_reader.load_feature_samples(0, sample_rows)
    binary_traces10 = binary_reader.load_feature_samples(0, sample_rows)
    binary_default_traces10 = binary_default_reader.load_feature_samples(0, sample_rows)
    binary_zstd_traces10 = binary_zstd_reader.load_feature_samples(0, sample_rows)
    binary_spill_traces10 = binary_spill_reader.load_feature_samples(0, sample_rows)
    assert_trace(binary_traces10[0], parquet_traces10[0].flags, parquet_traces10[0].time, parquet_traces10[0].value)
    assert_trace(binary_traces10[1], parquet_traces10[1].flags, parquet_traces10[1].time, parquet_traces10[1].value)
    assert_trace(binary_traces10[2], parquet_traces10[2].flags, parquet_traces10[2].time, parquet_traces10[2].value)
    assert_trace(binary_traces10[5], parquet_traces10[5].flags, parquet_traces10[5].time, parquet_traces10[5].value)
    assert_trace(binary_traces10[8], parquet_traces10[8].flags, parquet_traces10[8].time, parquet_traces10[8].value)
    assert_trace(binary_traces10[9], parquet_traces10[9].flags, parquet_traces10[9].time, parquet_traces10[9].value)
    for sample_row in sample_rows:
        assert_trace(
            binary_default_traces10[sample_row],
            parquet_traces10[sample_row].flags,
            parquet_traces10[sample_row].time,
            parquet_traces10[sample_row].value,
        )
        assert_trace(
            binary_zstd_traces10[sample_row],
            parquet_traces10[sample_row].flags,
            parquet_traces10[sample_row].time,
            parquet_traces10[sample_row].value,
        )
        assert_trace(
            binary_spill_traces10[sample_row],
            parquet_traces10[sample_row].flags,
            parquet_traces10[sample_row].time,
            parquet_traces10[sample_row].value,
        )

    parquet_traces20 = parquet_reader.load_feature_samples(1, [1, 7])
    binary_traces20 = binary_reader.load_feature_samples(1, [1, 7])
    binary_default_traces20 = binary_default_reader.load_feature_samples(1, [1, 7])
    binary_zstd_traces20 = binary_zstd_reader.load_feature_samples(1, [1, 7])
    binary_spill_traces20 = binary_spill_reader.load_feature_samples(1, [1, 7])
    assert_trace(binary_traces20[1], parquet_traces20[1].flags, parquet_traces20[1].time, parquet_traces20[1].value)
    assert_trace(binary_traces20[7], parquet_traces20[7].flags, parquet_traces20[7].time, parquet_traces20[7].value)
    for sample_row in [1, 7]:
        assert_trace(
            binary_default_traces20[sample_row],
            parquet_traces20[sample_row].flags,
            parquet_traces20[sample_row].time,
            parquet_traces20[sample_row].value,
        )
        assert_trace(
            binary_zstd_traces20[sample_row],
            parquet_traces20[sample_row].flags,
            parquet_traces20[sample_row].time,
            parquet_traces20[sample_row].value,
        )
        assert_trace(
            binary_spill_traces20[sample_row],
            parquet_traces20[sample_row].flags,
            parquet_traces20[sample_row].time,
            parquet_traces20[sample_row].value,
        )

    binary_by_ids = binary_reader.load_feature_samples_by_sample_ids(
        0,
        [0, 2, 5, 8, 9999],
    )
    assert_trace(binary_by_ids[0], parquet_traces10[0].flags, parquet_traces10[0].time, parquet_traces10[0].value)
    assert_trace(binary_by_ids[2], parquet_traces10[2].flags, parquet_traces10[2].time, parquet_traces10[2].value)
    assert_trace(binary_by_ids[5], parquet_traces10[5].flags, parquet_traces10[5].time, parquet_traces10[5].value)
    assert_trace(binary_by_ids[8], parquet_traces10[8].flags, parquet_traces10[8].time, parquet_traces10[8].value)
    assert int(binary_by_ids[9999].sample_row) == -1
    assert int(binary_by_ids[9999].flags) == 0
    assert binary_by_ids[9999].time.size == 0
    assert binary_by_ids[9999].value.size == 0

    empty_feature = binary_reader.load_feature_samples(2, [0, 4, 9])
    for sample_row in [0, 4, 9]:
        assert int(empty_feature[sample_row].flags) == 0
        assert empty_feature[sample_row].time.size == 0
        assert empty_feature[sample_row].value.size == 0

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
        feature_meta_path=str(feature_meta_path),
        n_samples=10,
        config=bundle_cfg,
        point_schema=custom_schema,
    ) as writer:
        writer.append_trace(
            0,
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
    binary_v3_reader = ArrayBinaryShardReader(binary_v3_manifest)
    assert int(binary_v3_manifest.version) == 3
    assert binary_v3_manifest.time_dtype == ""
    assert binary_v3_manifest.value_dtype == ""
    assert [spec.name for spec in binary_v3_manifest.point_schema] == ["phase", "state_code", "event_type"]
    assert (root / "binary_v3_shards" / "categorical_dictionaries" / "state_code.json").exists()
    assert (root / "binary_v3_shards" / "categorical_dictionaries" / "event_type.json").exists()
    dictionaries = load_array_binary_categorical_dictionaries(binary_v3_manifest)
    assert dictionaries["state_code"][1] == "OK"
    assert dictionaries["event_type"][2] == "STOP"

    v3_traces = binary_v3_reader.load_feature_samples(0, [0, 1, 2, 9])
    np.testing.assert_equal(v3_traces[0].columns["phase"], np.asarray([10, 11, 12], dtype=np.int32))
    np.testing.assert_equal(v3_traces[0].columns["state_code"], np.asarray([1, 1, 2], dtype=np.uint32))
    np.testing.assert_equal(v3_traces[0].columns["event_type"], np.asarray([1, 1, 2], dtype=np.uint32))
    assert v3_traces[0].time.size == 0
    assert v3_traces[0].value.size == 0
    np.testing.assert_equal(v3_traces[2].columns["phase"], np.asarray([20], dtype=np.int32))
    assert int(v3_traces[1].flags) == 0
    assert set(v3_traces[1].columns.keys()) == {"phase", "state_code", "event_type"}
    for values in v3_traces[1].columns.values():
        assert values.size == 0

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
            columns={
                "phase": [1, 2],
                "state_code": ["OK", "WARN"],
            },
        )
    known_builder.add_trace(
        sample_key="sample_003",
        feature_id=1,
        columns={
            "phase": [9],
            "state_code": ["FAIL"],
        },
    )
    known_builder_manifest_path = known_builder.finish()
    known_builder_manifest = load_array_binary_shard_manifest(known_builder_manifest_path)
    known_builder_reader = ArrayBinaryShardReader(known_builder_manifest)
    known_builder_dicts = load_array_binary_categorical_dictionaries(known_builder_manifest)
    assert known_builder_dicts["state_code"][1] == "OK"
    assert known_builder_dicts["state_code"][3] == "FAIL"
    known_traces = known_builder_reader.load_feature_samples(0, [0, 1])
    np.testing.assert_equal(known_traces[0].columns["phase"], np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_equal(known_traces[0].columns["state_code"], np.asarray([1, 2], dtype=np.uint32))
    assert int(known_traces[1].flags) == 0

    discovered_builder = ArrayDatasetBuilder(
        out_dir=str(root / "builder_discovered"),
        sample_meta_path=str(sample_meta_path),
        point_schema=builder_schema,
        build_options=ArrayBinaryBuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    try:
        discovered_builder.add_trace(
            sample_id=0,
            feature_id=0,
            columns={
                "phase": [1],
                "state_code": ["OK"],
            },
        )
    except ValueError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected discovered-feature mode to require feature_key")
    discovered_builder.add_trace(
        sample_id=0,
        feature_key="feature_zeta",
        columns={
            "phase": [4, 5],
            "state_code": ["WARN", "FAIL"],
        },
    )
    discovered_builder.add_trace(
        sample_id=1,
        feature_key="feature_alpha",
        columns={
            "phase": [3],
            "state_code": ["OK"],
        },
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
    discovered_builder_manifest = load_array_binary_shard_manifest(discovered_builder_manifest_path)
    discovered_builder_reader = ArrayBinaryShardReader(discovered_builder_manifest)
    discovered_builder_dicts = load_array_binary_categorical_dictionaries(discovered_builder_manifest)
    assert discovered_builder_dicts["state_code"][1] == "WARN"
    discovered_feature_meta = pl.read_parquet(Path(discovered_builder_manifest.feature_meta_path))
    assert tuple(discovered_feature_meta.columns) == ("feature_id", "feature_key", "group", "rank_hint")
    assert tuple(discovered_feature_meta["feature_key"].to_list()) == (
        "feature_zeta",
        "feature_alpha",
    )
    assert tuple(discovered_feature_meta["group"].to_list()) == ("latent_a", "latent_b")
    discovered_traces = discovered_builder_reader.load_feature_samples(0, [0])
    np.testing.assert_equal(discovered_traces[0].columns["phase"], np.asarray([4, 5], dtype=np.int32))
    np.testing.assert_equal(discovered_traces[0].columns["state_code"], np.asarray([1, 2], dtype=np.uint32))

    strict_schema = [
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ]
    strict_builder = ArrayDatasetBuilder(
        out_dir=str(root / "builder_strict"),
        sample_meta_path=str(sample_meta_path),
        point_schema=strict_schema,
        feature_keys=["feature_only"],
        build_options=ArrayBinaryBuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    try:
        strict_builder.add_trace(
            sample_id=0,
            feature_key="feature_only",
            columns={
                "phase": [1],
                "state_code": ["OK"],
                "extra_column": [99],
            },
        )
    except ValueError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected extra point column to raise ValueError")

    print("python array binary storage tests passed")


if __name__ == "__main__":
    main()
