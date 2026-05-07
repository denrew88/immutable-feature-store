import shutil
import sys
import json
from pathlib import Path

import numpy as np
import polars as pl

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "packages" / "scalar_feature_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scalar_feature_shard import (
    BuildOptions,
    FeatureNotFoundError,
    SampleNotFoundError,
    ScalarDatasetBuilder,
    SelectionOptions,
    build_shard,
    open_shard,
    select_features,
    write_feature_meta,
    write_sample_meta,
)


def main():
    """Run smoke tests for the public `scalar_feature_shard` package facade."""

    root = Path(__file__).resolve().parents[2] / "data" / "tmp_scalar_package_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_meta_path = write_sample_meta(
        [
            {"sample_key": "sample_000000", "y": 1.0, "y_alt": 1.5},
            {"sample_key": "sample_000001", "y": 2.0, "y_alt": 2.5},
            {"sample_key": "sample_000002", "y": 3.0, "y_alt": 3.5},
            {"sample_key": "sample_000003", "y": 4.0, "y_alt": 4.5},
        ],
        root / "sample_meta.parquet",
    )
    feature_meta_path = write_feature_meta(
        [
            {"feature_key": "feature_a", "group": "alpha"},
            {"feature_key": "feature_b", "group": "beta"},
            {"feature_key": "feature_c", "group": "gamma"},
        ],
        root / "feature_meta.parquet",
    )

    generated_sample_meta = pl.read_parquet(sample_meta_path)
    generated_feature_meta = pl.read_parquet(feature_meta_path)
    assert tuple(generated_sample_meta.columns) == ("sample_id", "sample_key", "y", "y_alt")
    assert tuple(generated_feature_meta.columns) == ("feature_id", "feature_key", "group")

    known_out = root / "known_shards"
    known_builder = ScalarDatasetBuilder(
        out_dir=str(known_out),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=BuildOptions(
            target_shard_mb=1,
            stats_y_cols=("y", "y_alt"),
        ),
    )
    known_builder.write_sample(0, {"feature_a": 10.0, "feature_c": None})
    known_builder.write_sample(1, {"feature_a": 11.0, "feature_b": 21.0})
    known_builder.write_sample(2, {"feature_b": 20.0})
    try:
        known_builder.write_sample(1, {"feature_c": 31.0})
    except ValueError as exc:
        assert "expects sample_id" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected duplicate sample write to fail")

    known_stage_manifest = Path(known_builder.finish_sample_major())
    assert known_stage_manifest.exists()
    known_stage_payload = json.loads(known_stage_manifest.read_text(encoding="utf-8"))
    assert known_stage_payload["format"] == "scalar-sample-bundles"
    assert len(known_stage_payload["bundle_paths"]) >= 1
    assert (known_out / "sample_major_stage" / "sample_bundles").exists()
    known_status = known_builder.status()
    assert known_status.last_committed_sample_id == 2
    assert known_status.next_expected_sample_id == 3

    wrapper_manifest_path = build_shard(
        str(known_stage_manifest),
        str(root / "wrapper_shards"),
        feature_meta_path=str(known_builder.sample_major_feature_meta_path),
        options=BuildOptions(target_shard_mb=1, stats_y_cols=("y", "y_alt")),
    )
    assert Path(wrapper_manifest_path).exists()

    manifest_path = known_builder.build_shards(keep_sample_major=True)
    with open_shard(manifest_path) as ds:
        assert ds.n_samples == 4
        assert ds.feature_count == 3
        assert ds.n_shards >= 1
        assert tuple(ds.feature_ids()) == (0, 1, 2)
        assert tuple(ds.sample_ids()) == (0, 1, 2, 3)
        assert tuple(ds.feature_keys()) == ("feature_a", "feature_b", "feature_c")
        assert tuple(ds.sample_keys()) == ("sample_000000", "sample_000001", "sample_000002", "sample_000003")

        value_a0 = ds.get_value(feature_id=0, sample_id=0)
        assert value_a0.present is True
        assert np.isclose(value_a0.value, 10.0)
        assert value_a0.feature_key == "feature_a"
        assert value_a0.sample_key == "sample_000000"

        keyed = ds.get_value_by_key(feature_key="feature_b", sample_key="sample_000002")
        assert keyed.present is True
        assert np.isclose(keyed.value, 20.0)

        batch = ds.get_values(feature_id=0, sample_ids=[0, 1, 2, 3])
        assert [item.present for item in batch.values] == [True, True, False, False]
        assert batch.feature_key == "feature_a"
        assert tuple(batch.sample_keys) == ("sample_000000", "sample_000001", "sample_000002", "sample_000003")

        keyed_batch = ds.get_values_by_key("feature_a", ["sample_000000", "sample_000001"])
        assert tuple(item.sample_key for item in keyed_batch.values) == ("sample_000000", "sample_000001")

        result = ds.get_many(feature_ids=[0, 1], sample_ids=[0, 2, 3])
        assert tuple(result.feature_ids) == (0, 1)
        assert tuple(result.sample_ids) == (0, 2, 3)
        assert tuple(result.feature_keys) == ("feature_a", "feature_b")
        streamed = ds.get_many(feature_ids=[0, 1], sample_ids=[0, 2, 3], stream=True, batch_size=1)
        streamed_features = tuple(streamed.features)
        assert len(streamed_features) == 2
        assert streamed_features[0].feature_id == 0
        assert streamed_features[1].feature_id == 1
        iter_features = tuple(ds.iter_many(feature_ids=[0, 1], sample_ids=[0, 2, 3], batch_size=1))
        assert [item.feature_id for item in iter_features] == [0, 1]

        keyed_result = ds.get_many_by_key(
            feature_keys=["feature_a", "feature_b"],
            sample_keys=["sample_000000", "sample_000002"],
        )
        assert tuple(keyed_result.feature_ids) == (0, 1)
        assert tuple(keyed_result.sample_ids) == (0, 2)
        assert tuple(keyed_result.sample_keys) == ("sample_000000", "sample_000002")
        keyed_streamed = ds.get_many_by_key(
            feature_keys=["feature_a", "feature_b"],
            sample_keys=["sample_000000", "sample_000002"],
            stream=True,
            batch_size=1,
        )
        keyed_streamed_features = tuple(keyed_streamed.features)
        assert [item.feature_key for item in keyed_streamed_features] == ["feature_a", "feature_b"]
        iter_keyed = tuple(
            ds.iter_many_by_key(
                feature_keys=["feature_a", "feature_b"],
                sample_keys=["sample_000000", "sample_000002"],
                batch_size=1,
            )
        )
        assert [item.feature_key for item in iter_keyed] == ["feature_a", "feature_b"]

        try:
            ds.get_value(feature_id=999999, sample_id=0, strict=True)
        except FeatureNotFoundError:
            pass
        else:  # pragma: no cover - sanity guard
            raise AssertionError("expected FeatureNotFoundError")

        try:
            ds.get_value(feature_id=0, sample_id=999999, strict=True)
        except SampleNotFoundError:
            pass
        else:  # pragma: no cover - sanity guard
            raise AssertionError("expected SampleNotFoundError")

    selection = select_features(
        manifest_path,
        options=SelectionOptions(
            y_col="y_alt",
            top_m=2,
            min_non_null_y=1,
            min_non_null_pair=1,
            initial_cap=4,
            max_step=4,
            batch_size=4,
            max_gap=2,
        ),
        include_candidates=True,
    )
    assert selection.y_col == "y_alt"
    assert selection.used_precomputed_stats is True
    assert selection.selected_count <= 2
    assert selection.candidate_count >= selection.selected_count
    assert len(selection.selected_feature_ids) == selection.selected_count
    assert len(selection.selected_feature_keys) == selection.selected_count
    assert len(selection.candidates) == selection.candidate_count

    discovered_out = root / "discovered_shards"
    with ScalarDatasetBuilder(
        out_dir=str(discovered_out),
        sample_meta_path=str(sample_meta_path),
        build_options=BuildOptions(target_shard_mb=1, stats_y_cols=("y",)),
    ) as discovered_builder:
        discovered_builder.write_sample(0, {"feature_x": 102.0, "feature_y": 202.0})
        discovered_builder.write_sample(1, {"feature_x": 101.0, "feature_y": None})

        discovered_builder.finish_sample_major()
        discovered_builder.update_feature_meta(
            [
                {"feature_key": "feature_x", "group": "left"},
                {"feature_key": "feature_y", "group": "right"},
            ],
            require_all=True,
        )
        discovered_manifest_path = discovered_builder.build_shards(keep_sample_major=True)

    with open_shard(discovered_manifest_path) as ds:
        assert tuple(ds.feature_keys()) == ("feature_x", "feature_y")
        discovered_value = ds.get_value_by_key("feature_x", "sample_000001")
        assert discovered_value.present is True
        assert np.isclose(discovered_value.value, 101.0)

    ordered_feature_meta_path = write_feature_meta(
        [{"feature_key": f"feature_{idx:02d}"} for idx in range(6)],
        root / "ordered_feature_meta.parquet",
    )
    ordered_out = root / "ordered_shards"
    ordered_builder = ScalarDatasetBuilder(
        out_dir=str(ordered_out),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(ordered_feature_meta_path),
        build_options=BuildOptions(
            target_shard_mb=1,
            n_shards=2,
            stats_y_cols=("y",),
        ),
    )
    ordered_builder.write_sample(
        0,
        {f"feature_{idx:02d}": float(idx) for idx in range(6)},
    )
    ordered_manifest_path = ordered_builder.build_shards()

    with open_shard(ordered_manifest_path) as ds:
        shuffled_feature_ids = [4, 1, 5, 0]
        kept_order = tuple(
            item.feature_id
            for item in ds.iter_many(
                feature_ids=shuffled_feature_ids,
                sample_ids=[0],
                batch_size=4,
                maintain_order=True,
            )
        )
        shard_order = tuple(
            item.feature_id
            for item in ds.iter_many(
                feature_ids=shuffled_feature_ids,
                sample_ids=[0],
                batch_size=4,
                maintain_order=False,
            )
        )
        assert kept_order == (4, 1, 5, 0)
        assert shard_order == (0, 1, 4, 5)

        grouped_many = ds.get_many(
            feature_ids=shuffled_feature_ids,
            sample_ids=[0],
            batch_size=4,
            maintain_order=False,
        )
        assert tuple(grouped_many.feature_ids) == shard_order
        assert [item.feature_id for item in grouped_many.features] == list(shard_order)

        shuffled_feature_keys = ["feature_04", "feature_01", "feature_05", "feature_00"]
        keyed_shard_order = tuple(
            item.feature_key
            for item in ds.iter_many_by_key(
                feature_keys=shuffled_feature_keys,
                sample_keys=["sample_000000"],
                batch_size=4,
                maintain_order=False,
            )
        )
        assert keyed_shard_order == ("feature_00", "feature_01", "feature_04", "feature_05")

        grouped_many_by_key = ds.get_many_by_key(
            feature_keys=shuffled_feature_keys,
            sample_keys=["sample_000000"],
            batch_size=4,
            maintain_order=False,
        )
        assert tuple(grouped_many_by_key.feature_keys) == keyed_shard_order
        assert [item.feature_key for item in grouped_many_by_key.features] == list(keyed_shard_order)

    print("python scalar package tests passed")


if __name__ == "__main__":
    main()
