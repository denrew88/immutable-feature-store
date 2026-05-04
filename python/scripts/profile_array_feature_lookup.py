import argparse
from pathlib import Path

from line_profiler import LineProfiler

from fs.array.binary_storage import ArrayBinaryShardReader, load_array_binary_shard_manifest
from scripts import serve_array_api as api


def main():
    """line-profiler entry point for binary array feature lookup."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "tmp_py_array_api_test"
            / "binary_v3_shards"
            / "array_binary_shard_manifest.json"
        ),
    )
    ap.add_argument("--n-features", type=int, default=20)
    ap.add_argument("--sample-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    manifest = load_array_binary_shard_manifest(args.manifest_path)
    reader = ArrayBinaryShardReader(manifest)
    req = api.ArrayFeatureRequest(
        manifest_path=args.manifest_path,
        feature_ids=list(range(args.n_features)),
        sample_ids=[args.sample_id],
        sanitize_nonfinite=True,
    )

    api.array_feature(req)

    profiler = LineProfiler()
    profiler.add_function(api.array_feature)
    profiler.add_function(api._resolve_array_feature_ids)
    profiler.add_function(api._json_safe_array)
    profiler.add_function(ArrayBinaryShardReader.load_feature_samples_by_sample_ids)
    profiler.add_function(ArrayBinaryShardReader.load_feature_samples)

    for _ in range(args.repeats):
        profiler.runcall(api.array_feature, req)

    profiler.print_stats()
    reader.close()


if __name__ == "__main__":
    main()
