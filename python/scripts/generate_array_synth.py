import argparse

from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig


def main():
    """CLI entry point for generating synthetic array data and optional shards."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-out-dir", required=True)
    ap.add_argument("--sample-meta", required=True)
    ap.add_argument("--shard-out-dir")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--min-trace-len", type=int, default=96)
    ap.add_argument("--max-trace-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bundle-rows", type=int, default=10000)
    ap.add_argument("--max-bundle-bytes", type=int, default=128 * 1024 * 1024)
    ap.add_argument("--target-shard-mb", type=int, default=256)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--samples-per-block", type=int, default=8)
    ap.add_argument("--row-group-size", type=int, default=64)
    ap.add_argument("--use-tmp-spill", action="store_true")
    ap.add_argument("--spill-bucket-mb", type=int, default=8)
    args = ap.parse_args()

    synth_cfg = ArraySyntheticConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        min_trace_len=args.min_trace_len,
        max_trace_len=args.max_trace_len,
        seed=args.seed,
    )
    bundle_cfg = ArrayBundleConfig(
        max_bundle_rows=args.max_bundle_rows,
        max_bundle_bytes=args.max_bundle_bytes,
    )
    shard_cfg = ArrayShardConfig(
        samples_per_block=args.samples_per_block,
        target_shard_bytes=args.target_shard_mb * 1024 * 1024,
        n_shards=(args.n_shards or 0),
        row_group_size=max(0, int(args.row_group_size)),
        use_tmp_spill=bool(args.use_tmp_spill),
        spill_bucket_target_bytes=args.spill_bucket_mb * 1024 * 1024,
    )

    result = generate_array_synthetic(
        bundle_out_dir=args.bundle_out_dir,
        sample_meta_path=args.sample_meta,
        config=synth_cfg,
        bundle_config=bundle_cfg,
        shard_out_dir=args.shard_out_dir,
        shard_config=shard_cfg,
    )
    print(result.get("shard_manifest_path", result["bundle_manifest_path"]))


if __name__ == "__main__":
    main()
