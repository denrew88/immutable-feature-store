import shutil
from pathlib import Path

import numpy as np

from fs.array.storage import (
    ArrayShardReader,
    build_array_feature_locator_index,
    load_array_shard_manifest,
)
from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig
from fs.scalar.parquet_storage import build_sample_id_index


def main():
    """Run smoke tests for synthetic array data generation and shard building."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_array_synth_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    result = generate_array_synthetic(
        bundle_out_dir=str(root / "bundles"),
        sample_meta_path=str(root / "sample_meta.parquet"),
        config=ArraySyntheticConfig(
            n_samples=24,
            n_features=12,
            min_trace_len=24,
            max_trace_len=64,
            seed=7,
        ),
        bundle_config=ArrayBundleConfig(max_bundle_rows=64, max_bundle_bytes=1 << 20),
        shard_out_dir=str(root / "shards"),
        shard_config=ArrayShardConfig(n_shards=4, samples_per_block=6),
    )

    manifest = load_array_shard_manifest(result["shard_manifest_path"])
    locator_index = build_array_feature_locator_index(manifest.locator_path)
    sample_id_index = build_sample_id_index(result["sample_meta_path"])
    for feature_locs in locator_index.values():
        assert len({loc.shard_id for loc in feature_locs}) == 1
    assert 0 in sample_id_index
    assert sample_id_index[5] == 5

    feature_id = next(iter(locator_index))
    sample_ids = [0, 3, 7, 11]
    reader = ArrayShardReader(manifest)
    traces = reader.load_feature_samples_by_sample_ids(
        feature_id=feature_id,
        sample_ids=sample_ids,
        locator_index=locator_index,
        sample_id_index=sample_id_index,
    )
    assert set(traces.keys()) == set(sample_ids)
    for sample_id in sample_ids:
        trace = traces[sample_id]
        assert trace is not None
        assert trace.sample_row in (-1, sample_id_index[sample_id])
        assert trace.time.shape == trace.value.shape
        if trace.sample_row >= 0 and trace.flags != 0:
            assert trace.time.ndim == 1
            assert trace.value.ndim == 1
            assert trace.time.size >= 0

    print("python array synthetic tests passed")


if __name__ == "__main__":
    main()
