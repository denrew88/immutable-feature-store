import shutil
import json
from pathlib import Path

import numpy as np
import polars as pl

from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder, write_feature_meta, write_sample_meta
from fs.scalar.parquet_storage import (
    ParquetShardReader,
    build_feature_locator_index,
    load_manifest,
    resolve_selection_stats_path,
)


def main():
    """Run direct-ingestion scalar builder correctness checks."""

    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_scalar_builder_test"
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

    known_out = root / "known_shards"
    known_builder = ScalarDatasetBuilder(
        out_dir=known_out,
        sample_meta_path=sample_meta_path,
        feature_meta_path=feature_meta_path,
        build_options=ScalarShardBuildOptions(
            target_shard_mb=1,
            stats_y_cols=("y", "y_alt"),
        ),
    )
    known_builder.write_sample(0, {"feature_a": 10.0, "feature_c": float("nan")})
    known_builder.write_sample(1, {"feature_a": 11.0, "feature_b": 21.0})
    known_builder.write_sample(2, {"feature_b": 20.0})
    try:
        known_builder.write_sample(1, {"feature_c": 31.0})
        raise AssertionError("expected duplicate sample write to fail")
    except ValueError as exc:
        assert "expects sample_id" in str(exc)

    known_stage_manifest = Path(known_builder.finish_sample_major())
    assert known_stage_manifest.exists()
    known_stage_payload = json.loads(known_stage_manifest.read_text(encoding="utf-8"))
    assert known_stage_payload["format"] == "scalar-sample-bundles"
    assert len(known_stage_payload["bundle_paths"]) >= 1
    assert (known_out / "sample_major_stage" / "sample_bundles").exists()
    known_status = known_builder.status()
    assert known_status.last_committed_sample_id == 2
    assert known_status.next_expected_sample_id == 3

    known_manifest_path = known_builder.build_shards()
    known_manifest = load_manifest(known_manifest_path)
    assert Path(resolve_selection_stats_path(known_manifest, "y")).exists()
    assert Path(resolve_selection_stats_path(known_manifest, "y_alt")).exists()
    assert not (known_out / "sample_major_stage").exists()

    known_reader = ParquetShardReader(known_manifest)
    known_locator = build_feature_locator_index(known_manifest.feature_locator_path)
    values_a, valid_a = known_reader.load_feature_by_id(0, locator_index=known_locator)
    values_b, valid_b = known_reader.load_feature_by_id(1, locator_index=known_locator)
    values_c, valid_c = known_reader.load_feature_by_id(2, locator_index=known_locator)

    assert bool(valid_a[0]) and np.isclose(values_a[0], 10.0)
    assert bool(valid_a[1]) and np.isclose(values_a[1], 11.0)
    assert not bool(valid_a[2]) and np.isclose(values_a[2], 0.0)
    assert not bool(valid_a[3]) and np.isclose(values_a[3], 0.0)

    assert not bool(valid_b[0]) and np.isclose(values_b[0], 0.0)
    assert bool(valid_b[1]) and np.isclose(values_b[1], 21.0)
    assert bool(valid_b[2]) and np.isclose(values_b[2], 20.0)
    assert not bool(valid_b[3]) and np.isclose(values_b[3], 0.0)

    assert not bool(valid_c[0]) and np.isclose(values_c[0], 0.0)
    assert not bool(valid_c[1]) and np.isclose(values_c[1], 0.0)
    assert not bool(valid_c[2]) and np.isclose(values_c[2], 0.0)
    assert not bool(valid_c[3]) and np.isclose(values_c[3], 0.0)

    discovered_out = root / "discovered_shards"
    with ScalarDatasetBuilder(
        out_dir=discovered_out,
        sample_meta_path=sample_meta_path,
        build_options=ScalarShardBuildOptions(
            target_shard_mb=1,
            stats_y_cols=("y",),
        ),
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

    discovered_feature_meta_df = pl.read_parquet(discovered_out / "sample_major_stage" / "feature_meta.parquet")
    assert "group" in discovered_feature_meta_df.columns
    assert discovered_feature_meta_df["group"].to_list() == ["left", "right"]
    discovered_manifest = load_manifest(discovered_manifest_path)
    discovered_stage_manifest = discovered_out / "sample_major_stage" / "sample_major_manifest.json"
    assert discovered_stage_manifest.exists()
    discovered_stage_payload = json.loads(discovered_stage_manifest.read_text(encoding="utf-8"))
    assert discovered_stage_payload["format"] == "scalar-sample-bundles"
    assert len(discovered_stage_payload["bundle_paths"]) >= 1
    discovered_reader = ParquetShardReader(discovered_manifest)
    discovered_locator = build_feature_locator_index(discovered_manifest.feature_locator_path)
    values_x, valid_x = discovered_reader.load_feature_by_id(0, locator_index=discovered_locator)
    values_y, valid_y = discovered_reader.load_feature_by_id(1, locator_index=discovered_locator)

    assert bool(valid_x[0]) and np.isclose(values_x[0], 102.0)
    assert bool(valid_x[1]) and np.isclose(values_x[1], 101.0)
    assert bool(valid_y[0]) and np.isclose(values_y[0], 202.0)
    assert not bool(valid_y[1]) and np.isclose(values_y[1], 0.0)

    print("python scalar builder tests passed")


if __name__ == "__main__":
    main()
