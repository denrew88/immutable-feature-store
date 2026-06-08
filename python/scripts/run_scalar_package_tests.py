import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "packages" / "scalar_feature_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scalar_feature_shard import (  # noqa: E402
    BuildOptions,
    ScalarDatasetBuilder,
    SelectionOptions,
    build_shard,
    open_dense_long_shard,
    select_features,
    write_feature_meta,
    write_sample_meta,
)
from validate_scalar_dense_long_exhaustive import validate_manifest


def _assert_stats_overlap_matches_y_valid(manifest_path: str, y_col: str) -> None:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_rel = manifest["selection_stats"][y_col]
    stats_path = manifest_path.parent / stats_rel
    sample_meta_path = manifest_path.parent / manifest["sample_meta_path"]
    sample_meta = pl.read_parquet(sample_meta_path)
    y_values = sample_meta[y_col].to_numpy().astype(np.float64, copy=False)
    y_valid = ~np.isnan(y_values)
    y_valid_count = int(y_valid.sum())
    saw_feature_limited_count = False
    stats = pl.read_parquet(stats_path).sort("feature_id")

    with open_dense_long_shard(str(manifest_path)) as ds:
        for row in stats.iter_rows(named=True):
            feature_id = int(row["feature_id"])
            _, valid = ds.load_feature_by_id(feature_id)
            expected = int((valid.astype(bool, copy=False) & y_valid).sum())
            assert int(row["n_y_overlap"]) == expected, (feature_id, row["n_y_overlap"], expected)
            if expected < y_valid_count:
                saw_feature_limited_count = True
    assert saw_feature_limited_count, f"{y_col} fixture did not prove n_y_overlap is feature-limited"


def _assert_stats_r2_all_zero(manifest_path: str, y_col: str) -> None:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_path = manifest_path.parent / manifest["selection_stats"][y_col]
    stats = pl.read_parquet(stats_path).sort("feature_id")
    r2 = stats["r2y"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
    assert np.array_equal(r2, np.zeros_like(r2)), (y_col, r2)


def main():
    """Run smoke tests for the dense-long scalar package facade."""

    root = Path(__file__).resolve().parents[2] / "data" / "tmp_scalar_package_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_meta_path = write_sample_meta(
        [
            {"sample_key": "sample_000000", "y": 1.0, "y_alt": 1.5, "y_const": 7.0},
            {"sample_key": "sample_000001", "y": 2.0, "y_alt": 2.5, "y_const": 7.0},
            {"sample_key": "sample_000002", "y": 3.0, "y_alt": None, "y_const": 7.0},
            {"sample_key": "sample_000003", "y": 4.0, "y_alt": 4.5, "y_const": 7.0},
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

    assert tuple(pl.read_parquet(sample_meta_path).columns) == ("sample_id", "sample_key", "y", "y_alt", "y_const")
    assert tuple(pl.read_parquet(feature_meta_path).columns) == ("feature_id", "feature_key", "group")

    known_out = root / "known_scalar_shard"
    builder = ScalarDatasetBuilder(
        out_dir=str(known_out),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=BuildOptions(target_shard_mb=1, stats_y_cols=("y", "y_alt", "y_const")),
    )
    stale_tmp = known_out / "raw_samples" / "sample_000000000002.parquet.tmp"
    stale_tmp.parent.mkdir(parents=True, exist_ok=True)
    with stale_tmp.open("wb") as stale_handle:
        stale_handle.write(b"locked stale tmp")
        stale_handle.flush()
        builder.write_sample(2, {"feature_b": 20.0})
    try:
        stale_tmp.unlink()
    except FileNotFoundError:
        pass
    builder.write_sample(0, {"feature_a": 10.0, "feature_c": None})
    builder.write_sample(1, {"feature_a": 11.0, "feature_b": 21.0})
    builder.write_sample(3, {})
    assert builder.pending_sample_ids() == []
    stage_manifest = Path(builder.finish_stage())
    assert stage_manifest.exists()
    assert json.loads(stage_manifest.read_text(encoding="utf-8"))["format"] == "scalar-sample-major-v1"

    wrapper_manifest_path = build_shard(
        str(stage_manifest),
        str(root / "wrapper_scalar_shard"),
        feature_meta_path=str(builder.feature_meta_path),
        options=BuildOptions(target_shard_mb=1, stats_y_cols=("y", "y_alt", "y_const")),
    )
    assert Path(wrapper_manifest_path).name == "scalar_shard_manifest.json"

    manifest_path = builder.build_shards(keep_raw=True)
    assert Path(manifest_path).name == "scalar_shard_manifest.json"
    with open_dense_long_shard(manifest_path) as ds:
        values_a, valid_a = ds.load_feature_by_id(0)
        assert bool(valid_a[0]) and np.isclose(values_a[0], 10.0)
        assert bool(valid_a[1]) and np.isclose(values_a[1], 11.0)
        assert not bool(valid_a[2])
        assert np.isnan(values_a[2])
        values_b, valid_b = ds.load_feature_by_id(1)
        assert bool(valid_b[1]) and np.isclose(values_b[1], 21.0)
        assert bool(valid_b[2]) and np.isclose(values_b[2], 20.0)
        sample2_values, sample2_valid = ds.load_sample_by_id(2)
        assert bool(sample2_valid[1]) and np.isclose(sample2_values[1], 20.0)
        assert not bool(sample2_valid[0])
        assert np.isnan(sample2_values[0])
        assert ds.top_features_from_stats("y_alt", top_k=2).height == 2
    _assert_stats_overlap_matches_y_valid(manifest_path, "y")
    _assert_stats_overlap_matches_y_valid(manifest_path, "y_alt")
    _assert_stats_overlap_matches_y_valid(manifest_path, "y_const")
    _assert_stats_r2_all_zero(manifest_path, "y_const")

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
        ),
        include_candidates=True,
    )
    assert selection.y_col == "y_alt"
    assert selection.used_precomputed_stats is True
    assert selection.selected_count <= 2
    assert selection.candidate_count >= selection.selected_count

    raw_out = root / "raw_dense_stage"
    raw_builder = ScalarDatasetBuilder(
        out_dir=str(raw_out),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=BuildOptions(target_shard_mb=1, stats_y_cols=("y", "y_alt", "y_const")),
    )
    assert raw_builder.pending_sample_ids() == [0, 1, 2, 3]
    raw_builder.write_sample(2, {"feature_b": 20.0})
    raw_builder.close()

    resumed_builder = ScalarDatasetBuilder(
        out_dir=str(raw_out),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=BuildOptions(target_shard_mb=1, stats_y_cols=("y", "y_alt", "y_const")),
    )
    assert resumed_builder.completed_sample_ids() == [2]
    assert resumed_builder.pending_sample_ids() == [0, 1, 3]
    resumed_builder.write_sample(0, {"feature_a": 10.0, "feature_c": None})
    resumed_builder.write_sample(1, {"feature_a": 11.0, "feature_b": 21.0})
    resumed_builder.write_sample(3, {})
    assert resumed_builder.completed_sample_ids() == [0, 1, 2, 3]
    assert resumed_builder.pending_sample_ids() == []
    assert resumed_builder.write_sample(1, {"feature_c": 31.0}, skip_if_completed=True) is False
    dense_manifest_path = resumed_builder.build_dense_long_shards(out_dir=str(root / "raw_scalar_shard"))
    validate_manifest(
        Path(dense_manifest_path),
        sample_major_manifest_path=Path(raw_out) / "sample_major_manifest.json",
        progress_every=0,
    )
    with open_dense_long_shard(dense_manifest_path) as ds:
        dense_values_b, dense_valid_b = ds.load_feature_by_id(1)
        assert bool(dense_valid_b[2]) and np.isclose(dense_values_b[2], 20.0)
        dense_values_a, dense_valid_a = ds.load_feature_by_id(0)
        assert not bool(dense_valid_a[2])
        assert np.isnan(dense_values_a[2])
        dense_values_c, dense_valid_c = ds.load_feature_by_id(2)
        assert not bool(dense_valid_c[1])
        assert np.isnan(dense_values_c[1])

    print("python scalar package tests passed")


if __name__ == "__main__":
    main()
