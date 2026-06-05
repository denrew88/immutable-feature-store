import shutil
import sys
import json
from pathlib import Path

import numpy as np
import polars as pl

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "packages" / "array_binary_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))
PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from array_binary_shard import (
    ArrayDatasetBuilder,
    BuildOptions,
    FeatureNotFoundError,
    LogicalType,
    PointColumnSpec,
    SampleNotFoundError,
    StorageType,
    build_shard,
    open_shard,
    write_feature_meta,
    write_sample_meta,
)
from fs.array.storage import ArraySampleBundleWriter
from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArraySyntheticConfig


def main():
    """Run smoke tests for the public `array_binary_shard` package facade."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_array_binary_package_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    cfg = ArraySyntheticConfig(
        n_samples=24,
        n_features=12,
        n_latent_groups=4,
        min_trace_len=8,
        max_trace_len=16,
        seed=11,
    )
    synth = generate_array_synthetic(
        bundle_out_dir=str(root / "bundles"),
        sample_meta_path=str(root / "sample_meta.parquet"),
        config=cfg,
    )
    bundle_manifest_path = synth["bundle_manifest_path"]

    generated_sample_meta_path = Path(
        write_sample_meta(
            [
                {"sample_key": "sample_pkg_000", "split": "train"},
                {"sample_key": "sample_pkg_001", "split": "test"},
            ],
            root / "generated_sample_meta.parquet",
        )
    )
    generated_feature_meta_path = Path(
        write_feature_meta(
            [
                {"feature_key": "feature_pkg_000", "group": "g0"},
                {"feature_key": "feature_pkg_001", "group": "g1"},
            ],
            root / "generated_feature_meta.parquet",
        )
    )
    generated_sample_meta = pl.read_parquet(generated_sample_meta_path)
    generated_feature_meta = pl.read_parquet(generated_feature_meta_path)
    assert tuple(generated_sample_meta.columns) == ("sample_id", "sample_key", "split")
    assert tuple(generated_sample_meta["sample_id"].to_list()) == (0, 1)
    assert tuple(generated_feature_meta.columns) == ("feature_id", "feature_key", "group")
    assert tuple(generated_feature_meta["feature_id"].to_list()) == (0, 1)
    assert PointColumnSpec(
        name="ts",
        storage_type=StorageType.INT64,
        logical_type=LogicalType.TIMESTAMP_NS,
    ).to_json() == {
        "name": "ts",
        "storage_type": "int64",
        "logical_type": "timestamp_ns",
    }
    try:
        PointColumnSpec(
            name="bad_state",
            storage_type=StorageType.FLOAT64,
            logical_type=LogicalType.CATEGORICAL,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected invalid logical/storage combination to raise ValueError")

    manifest_path = build_shard(
        bundle_manifest_path,
        str(root / "binary_shards"),
        options=BuildOptions(samples_per_block=16, target_shard_mb=8, codec="none"),
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_json = json.load(f)
    assert manifest_json["sample_meta_path"] == "sample_meta.parquet"
    assert manifest_json["feature_meta_path"] == "feature_meta.parquet"
    assert manifest_json["shard_path"] == "array_binary_feature_shards"
    assert (root / "binary_shards" / "sample_meta.parquet").exists()
    assert (root / "binary_shards" / "feature_meta.parquet").exists()
    assert (root / "binary_shards" / "array_binary_feature_shards").is_dir()

    with open_shard(manifest_path) as ds:
        assert ds.n_samples == cfg.n_samples
        assert ds.samples_per_block == 16
        assert ds.feature_count == cfg.n_features
        assert len(ds.feature_ids()) == cfg.n_features
        assert len(ds.sample_ids()) == cfg.n_samples

        feature_id = int(ds.feature_ids()[0])
        sample_id = int(ds.sample_ids()[0])
        trace = ds.get_trace(feature_id=feature_id, sample_id=sample_id)
        assert trace.feature_id == feature_id
        assert trace.sample_id == sample_id
        assert isinstance(trace.present, bool)
        np.testing.assert_equal(trace.columns["time"].shape, trace.columns["value"].shape)

        feature_key = str(ds.feature_keys()[0])
        sample_key = str(ds.sample_keys()[0])
        keyed_trace = ds.get_trace_by_key(feature_key=feature_key, sample_key=sample_key)
        assert keyed_trace.feature_id == feature_id
        assert keyed_trace.sample_id == sample_id
        assert keyed_trace.feature_key == feature_key
        assert keyed_trace.sample_key == sample_key

        batch = ds.get_traces(feature_id=feature_id, sample_ids=ds.sample_ids()[:4])
        assert batch.feature_id == feature_id
        assert len(batch.sample_ids) == 4
        assert len(batch.traces) == 4

        keyed_batch = ds.get_traces_by_key(feature_key=feature_key, sample_keys=ds.sample_keys()[:4])
        assert keyed_batch.feature_id == feature_id
        assert keyed_batch.feature_key == feature_key
        assert tuple(keyed_batch.sample_ids) == tuple(ds.sample_ids()[:4])
        assert tuple(keyed_batch.sample_keys) == tuple(ds.sample_keys()[:4])

        feature_ids = ds.feature_ids()[:3]
        sample_ids = ds.sample_ids()[:5]
        result = ds.get_many(feature_ids=feature_ids, sample_ids=sample_ids)
        assert tuple(result.feature_ids) == tuple(feature_ids)
        assert tuple(result.sample_ids) == tuple(sample_ids)
        assert len(result.features) == len(feature_ids)

        keyed_result = ds.get_many_by_key(feature_keys=ds.feature_keys()[:3], sample_keys=ds.sample_keys()[:5])
        assert tuple(keyed_result.feature_ids) == tuple(feature_ids)
        assert tuple(keyed_result.sample_ids) == tuple(sample_ids)
        assert tuple(keyed_result.feature_keys) == tuple(ds.feature_keys()[:3])
        assert tuple(keyed_result.sample_keys) == tuple(ds.sample_keys()[:5])

        try:
            ds.get_trace(feature_id=999999, sample_id=sample_id, strict=True)
        except FeatureNotFoundError:
            pass
        else:  # pragma: no cover - sanity guard
            raise AssertionError("expected FeatureNotFoundError")

        try:
            ds.get_trace(feature_id=feature_id, sample_id=999999, strict=True)
        except SampleNotFoundError:
            pass
        else:  # pragma: no cover - sanity guard
            raise AssertionError("expected SampleNotFoundError")

        try:
            ds.resolve_feature_key("missing_feature_key")
        except FeatureNotFoundError:
            pass
        else:  # pragma: no cover - sanity guard
            raise AssertionError("expected FeatureNotFoundError for missing feature key")

        try:
            ds.resolve_sample_key("missing_sample_key")
        except SampleNotFoundError:
            pass
        else:  # pragma: no cover - sanity guard
            raise AssertionError("expected SampleNotFoundError for missing sample key")

    try:
        ds.get_trace(feature_id=feature_id, sample_id=sample_id)
    except RuntimeError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected closed-dataset RuntimeError")

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
        str(root / "sample_meta.parquet"),
        feature_meta_path=str(synth["feature_meta_path"]),
        n_samples=cfg.n_samples,
        config=ArrayBundleConfig(max_bundle_rows=64, max_bundle_bytes=1 << 20),
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
            3,
            0,
            columns={
                "phase": np.asarray([20], dtype=np.int32),
                "state_code": np.asarray([3], dtype=np.uint32),
                "event_type": np.asarray([2], dtype=np.uint32),
            },
        )
        custom_bundle_manifest_path = writer.finish()

    custom_manifest_path = build_shard(
        custom_bundle_manifest_path,
        str(root / "binary_v3_shards"),
        options=BuildOptions(samples_per_block=16, target_shard_mb=8, codec="none"),
    )
    with open_shard(custom_manifest_path) as ds:
        assert [spec.name for spec in ds.point_schema] == ["phase", "state_code", "event_type"]
        assert ds.categorical_dictionaries()["state_code"][1] == "OK"
        raw_trace = ds.get_trace(feature_id=0, sample_id=0)
        np.testing.assert_equal(raw_trace.columns["phase"], np.asarray([10, 11, 12], dtype=np.int32))
        np.testing.assert_equal(raw_trace.columns["state_code"], np.asarray([1, 1, 2], dtype=np.uint32))
        assert "time" not in raw_trace.columns
        decoded_trace = ds.get_trace(feature_id=0, sample_id=0, decode_categorical=True)
        assert tuple(decoded_trace.columns["state_code"]) == ("OK", "OK", "WARN")
        assert tuple(decoded_trace.columns["event_type"]) == ("START", "START", "STOP")

    ts_builder = ArrayDatasetBuilder(
        out_dir=str(root / "builder_temporal"),
        sample_meta_path=str(root / "sample_meta.parquet"),
        point_schema=[
            {"name": "ts", "storage_type": StorageType.INT64, "logical_type": LogicalType.TIMESTAMP_NS},
            {"name": "dt", "storage_type": StorageType.INT64, "logical_type": LogicalType.TIMEDELTA_NS},
        ],
        feature_keys=["feature_temporal"],
        build_options=BuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    with ts_builder.sample(sample_id=0) as sample:
        sample.add_trace(
            feature_key="feature_temporal",
            columns={
                "ts": np.asarray(["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"),
                "dt": np.asarray([0, 1_000_000_000], dtype="timedelta64[ns]"),
            },
        )
    try:
        with ts_builder.sample(sample_id=1) as sample:
            sample.add_trace(
                feature_key="feature_temporal",
                columns={
                    "ts": np.asarray(["2024-01-01T00:00:02"], dtype="datetime64[ns]"),
                    "dt": np.asarray([2_000_000_000], dtype="timedelta64[ns]"),
                    "extra_column": [123],
                },
            )
    except ValueError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected extra point column to raise ValueError")
    ts_manifest_path = ts_builder.build_shards()
    with open_shard(ts_manifest_path) as ds:
        temporal_trace = ds.get_trace(feature_id=0, sample_id=0)
        assert str(temporal_trace.columns["ts"].dtype) == "datetime64[ns]"
        assert str(temporal_trace.columns["dt"].dtype) == "timedelta64[ns]"
        np.testing.assert_equal(
            temporal_trace.columns["ts"],
            np.asarray(["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"),
        )
        np.testing.assert_equal(
            temporal_trace.columns["dt"],
            np.asarray([0, 1_000_000_000], dtype="timedelta64[ns]"),
        )

    builder_schema = [
        {"name": "phase", "storage_type": StorageType.INT32, "logical_type": LogicalType.INTEGER},
        {"name": "state_code", "storage_type": StorageType.UINT32, "logical_type": LogicalType.CATEGORICAL},
    ]
    known_builder_out = root / "builder_known"
    known_builder = ArrayDatasetBuilder(
        out_dir=str(known_builder_out),
        sample_meta_path=str(root / "sample_meta.parquet"),
        point_schema=builder_schema,
        feature_keys=["feature_alpha", "feature_beta"],
        build_options=BuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    with known_builder.sample(sample_key="sample_000000") as sample:
        sample.add_trace(
            feature_key="feature_alpha",
            columns={
                "phase": [1, 2],
                "state_code": ["OK", "WARN"],
            },
        )
        sample.add_trace(
            feature_key="feature_beta",
            columns={
                "phase": [9],
                "state_code": ["FAIL"],
            },
        )
    with known_builder.sample(sample_key="sample_000001"):
        pass
    with known_builder.sample(sample_key="sample_000002"):
        pass
    with known_builder.sample(sample_key="sample_000003") as sample:
        sample.add_trace(
            feature_id=0,
            columns={
                "phase": [7, 8, 9],
                "state_code": ["WARN", "WARN", "OK"],
            },
        )
    known_status = known_builder.status()
    assert known_status.last_committed_sample_id is None
    assert known_status.next_expected_sample_id == 0
    assert known_status.buffered_through_sample_id == 3
    assert known_status.in_progress_sample_id is None
    known_manifest_path = known_builder.finish()
    with open_shard(known_manifest_path) as ds:
        assert tuple(ds.feature_keys()) == ("feature_alpha", "feature_beta")
        known_trace = ds.get_trace_by_key(
            feature_key="feature_alpha",
            sample_key=str(ds.sample_keys()[0]),
            decode_categorical=True,
        )
        np.testing.assert_equal(known_trace.columns["phase"], np.asarray([1, 2], dtype=np.int32))
        assert tuple(known_trace.columns["state_code"]) == ("OK", "WARN")
        assert ds.categorical_dictionaries()["state_code"][3] == "FAIL"

    discovered_builder_out = root / "builder_discovered"
    discovered_builder = ArrayDatasetBuilder(
        out_dir=str(discovered_builder_out),
        sample_meta_path=str(root / "sample_meta.parquet"),
        point_schema=builder_schema,
        build_options=BuildOptions(samples_per_block=4, target_shard_mb=8, codec="none"),
    )
    try:
        with discovered_builder.sample(sample_id=0) as sample:
            sample.add_trace(
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
    with discovered_builder.sample(sample_id=0) as sample:
        sample.add_trace(
            feature_key="feature_zeta",
            columns={
                "phase": [4, 5],
                "state_code": ["WARN", "FAIL"],
            },
        )
    with discovered_builder.sample(sample_id=1) as sample:
        sample.add_trace(
            feature_key="feature_alpha",
            columns={
                "phase": [3],
                "state_code": ["OK"],
            },
        )
    with discovered_builder.sample(sample_id=2) as sample:
        sample.add_trace(
            feature_key="feature_zeta",
            columns={
                "phase": [6],
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
    discovered_manifest_path = discovered_builder.build_shards()
    with open_shard(discovered_manifest_path) as ds:
        assert tuple(ds.feature_keys()) == ("feature_zeta", "feature_alpha")
        assert ds.resolve_feature_key("feature_zeta") == 0
        assert ds.resolve_feature_key("feature_alpha") == 1
        discovered_feature_meta = pl.read_parquet(Path(discovered_manifest_path).resolve().parent / "feature_meta.parquet")
        assert tuple(discovered_feature_meta.columns) == ("feature_id", "feature_key", "group", "rank_hint")
        discovered_trace = ds.get_trace_by_key(
            feature_key="feature_zeta",
            sample_key=str(ds.sample_keys()[0]),
            decode_categorical=True,
        )
        np.testing.assert_equal(discovered_trace.columns["phase"], np.asarray([4, 5], dtype=np.int32))
        assert tuple(discovered_trace.columns["state_code"]) == ("WARN", "FAIL")

    print("python array binary package tests passed")


if __name__ == "__main__":
    main()
