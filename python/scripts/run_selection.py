import argparse

from fs.config import SelectionConfig
from fs.scalar.parquet_storage import (
    ParquetShardReader,
    list_shard_paths,
    load_manifest,
    load_sample_meta,
    locator_has_candidate_stats,
    resolve_selection_stats_path,
)
from fs.feature_selection.candidates import build_candidates_from_shards, build_candidates_from_stats
from fs.feature_selection.incremental import select_features_incremental


def main():
    """CLI entry point for running scalar feature selection from a manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--y-r2", type=float, default=0.01)
    ap.add_argument("--min-non-null-y", type=int, default=200)
    ap.add_argument("--ff-r2", type=float, default=0.9)
    ap.add_argument("--min-non-null-pair", type=int, default=200)
    ap.add_argument("--top-m", type=int, default=100)
    ap.add_argument("--initial-cap", type=int, default=2048)
    ap.add_argument("--max-step", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-gap", type=int, default=64)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--mask-fastpath-min-group", type=int, default=64)
    ap.add_argument("--mask-fastpath-min-pairs", type=int, default=8192)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    reader = ParquetShardReader(manifest, max_gap=args.max_gap)
    stats_path = resolve_selection_stats_path(manifest, args.y_col)
    if stats_path:
        candidates = build_candidates_from_stats(
            stats_path,
            min_non_null_y=args.min_non_null_y,
            y_r2_threshold=args.y_r2,
            max_candidates=args.max_candidates,
        )
    elif manifest.stats_y_col == args.y_col and locator_has_candidate_stats(manifest.feature_locator_path):
        candidates = build_candidates_from_stats(
            manifest.feature_locator_path,
            min_non_null_y=args.min_non_null_y,
            y_r2_threshold=args.y_r2,
            max_candidates=args.max_candidates,
        )
    else:
        _, y, y_mask, _ = load_sample_meta(manifest.sample_meta_path, y_col=args.y_col)
        candidates = build_candidates_from_shards(
            list_shard_paths(manifest),
            y,
            y_mask,
            min_non_null_y=args.min_non_null_y,
            y_r2_threshold=args.y_r2,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
        )

    config = SelectionConfig(
        y_r2_threshold=args.y_r2,
        min_non_null_y=args.min_non_null_y,
        ff_r2_threshold=args.ff_r2,
        min_non_null_pair=args.min_non_null_pair,
        top_m=args.top_m,
        initial_cap=args.initial_cap,
        max_step=args.max_step,
        batch_size=args.batch_size,
        max_gap=args.max_gap,
        max_candidates=args.max_candidates,
        mask_fastpath_min_group=args.mask_fastpath_min_group,
        mask_fastpath_min_pairs=args.mask_fastpath_min_pairs,
    )

    selected = select_features_incremental(candidates, reader, config)
    for c in selected:
        print(c.feature_id)


if __name__ == "__main__":
    main()
