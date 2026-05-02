import argparse
from pathlib import Path

from line_profiler import LineProfiler

from fs.array.storage import ArrayShardReader, _find_locator_block, load_array_feature_block
from scripts import serve_array_api as api


def main():
    """CLI entry point for line-profiling array feature lookup code paths."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "array_synth_py"
            / "shards"
            / "array_shard_manifest.json"
        ),
    )
    ap.add_argument("--n-features", type=int, default=20)
    ap.add_argument("--sample-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    req = api.ArrayFeatureRequest(
        manifest_path=args.manifest_path,
        feature_ids=list(range(args.n_features)),
        sample_ids=[args.sample_id],
        sanitize_nonfinite=True,
    )

    # Warm cache and filesystem page cache once before profiling.
    api.array_feature(req)

    profiler = LineProfiler()
    profiler.add_function(api.array_feature)
    profiler.add_function(api._resolve_array_feature_ids)
    profiler.add_function(api._json_safe_array)
    profiler.add_function(ArrayShardReader.load_feature_samples_by_sample_ids)
    profiler.add_function(ArrayShardReader.load_feature_samples)
    profiler.add_function(load_array_feature_block)
    profiler.add_function(_find_locator_block)

    for _ in range(args.repeats):
        profiler.runcall(api.array_feature, req)

    profiler.print_stats()


if __name__ == "__main__":
    main()
