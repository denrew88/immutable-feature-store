import shutil
from pathlib import Path

from fs.array import open_shard
from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig


def main():
    """Run smoke tests for synthetic array data generation and binary shard build."""
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

    with open_shard(result["shard_manifest_path"]) as ds:
        assert ds.n_samples == 24
        assert ds.feature_count == 12
        assert [spec.name for spec in ds.point_schema] == ["time", "value"]

        feature_id = int(ds.feature_ids()[0])
        sample_ids = [0, 3, 7, 11]
        traces = ds.get_traces(feature_id=feature_id, sample_ids=sample_ids)
        assert tuple(traces.sample_ids) == tuple(sample_ids)
        assert len(traces.traces) == len(sample_ids)
        for trace in traces.traces:
            assert trace is not None
            assert trace.columns["time"].shape == trace.columns["value"].shape
            if trace.flags != 0:
                assert trace.columns["time"].ndim == 1
                assert trace.columns["value"].ndim == 1
                assert trace.columns["time"].size >= 0

    print("python array synthetic tests passed")


if __name__ == "__main__":
    main()
