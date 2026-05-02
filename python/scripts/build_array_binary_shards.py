import argparse

from fs.array.binary_storage import build_array_binary_shards_from_bundles
from fs.config import ArrayShardConfig


def main():
    """CLI entry point for building binary array shards from bundle manifests."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-shard-mb", type=int, default=32)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--samples-per-block", type=int, default=16)
    ap.add_argument("--codec", choices=["none", "zstd"], default="none")
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--sample-key-col", default="sample_key")
    ap.add_argument("--feature-key-col", default="feature_key")
    args = ap.parse_args()

    cfg = ArrayShardConfig(
        samples_per_block=args.samples_per_block,
        target_shard_bytes=args.target_shard_mb * 1024 * 1024,
        n_shards=(args.n_shards or 0),
        row_group_size=0,
        use_tmp_spill=False,
        spill_bucket_target_bytes=8 * 1024 * 1024,
    )
    manifest_path = build_array_binary_shards_from_bundles(
        args.bundle_manifest,
        args.out_dir,
        config=cfg,
        codec=args.codec,
        zstd_level=args.zstd_level,
        sample_key_col=args.sample_key_col,
        feature_key_col=args.feature_key_col,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
