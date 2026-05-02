import argparse

from fs.array.storage import build_array_shards_from_bundles
from fs.config import ArrayShardConfig


def main():
    """CLI entry point for building parquet array shards from bundle manifests."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-shard-mb", type=int, default=256)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--samples-per-block", type=int, default=8)
    ap.add_argument("--row-group-size", type=int, default=64)
    ap.add_argument("--use-tmp-spill", action="store_true")
    ap.add_argument("--spill-bucket-mb", type=int, default=8)
    args = ap.parse_args()

    cfg = ArrayShardConfig(
        samples_per_block=args.samples_per_block,
        target_shard_bytes=args.target_shard_mb * 1024 * 1024,
        n_shards=(args.n_shards or 0),
        row_group_size=max(0, int(args.row_group_size)),
        use_tmp_spill=bool(args.use_tmp_spill),
        spill_bucket_target_bytes=args.spill_bucket_mb * 1024 * 1024,
    )
    manifest_path = build_array_shards_from_bundles(
        args.bundle_manifest,
        args.out_dir,
        config=cfg,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
