import argparse

from fs._package_sources import ensure_package_source

ensure_package_source("scalar_feature_shard")

from scalar_feature_shard import build_shard


def main():
    """CLI entry point for building dense-long scalar shards from sample-major data."""
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-meta")
    source.add_argument("--sample-major-manifest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feature-meta")
    ap.add_argument("--target-shard-mb", type=int, default=256)
    ap.add_argument("--feature-id-col", default="feature_id")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--sample-id-col", default="sample_id")
    ap.add_argument("--sample-key-col", default="sample_key")
    ap.add_argument("--feature-key-col", default="feature_key")
    ap.add_argument("--path-col", default="sample_path")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--stats-y-col", action="append", dest="stats_y_cols")
    args = ap.parse_args()

    if args.stats_y_cols:
        stats_y_cols = list(dict.fromkeys(str(value) for value in args.stats_y_cols))
    else:
        stats_y_cols = [str(args.y_col)]

    manifest_path = build_shard(
        args.sample_major_manifest or args.sample_meta,
        args.out_dir,
        feature_meta_path=args.feature_meta,
        target_shard_mb=int(args.target_shard_mb),
        feature_id_col=args.feature_id_col,
        value_col=args.value_col,
        sample_id_col=args.sample_id_col,
        sample_key_col=args.sample_key_col,
        feature_key_col=args.feature_key_col,
        path_col=args.path_col,
        y_col=args.y_col,
        stats_y_cols=stats_y_cols,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
