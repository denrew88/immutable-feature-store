import argparse
import io

from line_profiler import LineProfiler

from fs.feature_selection import incremental as selection
from fs.feature_selection import pearson
from fs.feature_selection.candidates import build_candidates_from_shards, build_candidates_from_stats
from fs.config import SelectionConfig
from fs.scalar.parquet_storage import (
    ParquetShardReader,
    list_shard_paths,
    load_manifest,
    load_sample_targets,
    locator_has_candidate_stats,
    resolve_selection_stats_path,
)


def _build_candidates(manifest, args):
    """Build feature candidates for line-profiling the selection pipeline."""
    stats_path = resolve_selection_stats_path(manifest, args.y_col)
    if stats_path:
        return build_candidates_from_stats(
            stats_path,
            min_non_null_y=args.min_non_null_y,
            y_r2_threshold=args.y_r2,
            max_candidates=args.max_candidates,
        )
    if manifest.stats_y_col == args.y_col and locator_has_candidate_stats(manifest.feature_locator_path):
        return build_candidates_from_stats(
            manifest.feature_locator_path,
            min_non_null_y=args.min_non_null_y,
            y_r2_threshold=args.y_r2,
            max_candidates=args.max_candidates,
        )

    _, y, y_mask = load_sample_targets(manifest.sample_meta_path, y_col=args.y_col)
    return build_candidates_from_shards(
        list_shard_paths(manifest),
        y,
        y_mask,
        min_non_null_y=args.min_non_null_y,
        y_r2_threshold=args.y_r2,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
    )


def _run_selection(args):
    """Run one full selection pass for profiling."""
    manifest = load_manifest(args.manifest)
    candidates = _build_candidates(manifest, args)
    reader = ParquetShardReader(manifest, max_gap=args.max_gap)
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
    return selection.select_features_incremental(candidates, reader, config)


def main():
    """Profile the scalar feature-selection pipeline with line_profiler."""
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
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    profiler = LineProfiler()
    for fn in [
        _run_selection,
        selection.select_features_incremental,
        selection._build_stage_cache,
        selection._build_stage_adjacency,
        selection._build_mask_groups,
        selection._process_mask_group_pair,
        selection._get_group_values,
        selection._stack_local_values,
        selection._same_mask_tile_stats,
        selection._or_edge_rows,
        selection._overlap_counts_many_to_many,
        selection._stack_feature_tile,
        selection._pack_valid_rows,
        selection._pack_bool_rows,
        pearson.batch_r2_many_vs_many,
        pearson.batch_r2_one_vs_many,
        ParquetShardReader._group_offsets,
        ParquetShardReader.load_rows,
    ]:
        profiler.add_function(fn)

    selected = profiler(_run_selection)(args)
    print(f"selected_count={len(selected)}")
    print("selected_feature_ids=" + ",".join(str(c.feature_id) for c in selected[:10]))

    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            profiler.print_stats(stream=f)
    else:
        stream = io.StringIO()
        profiler.print_stats(stream=stream)
        print(stream.getvalue())


if __name__ == "__main__":
    main()
