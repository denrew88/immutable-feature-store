import argparse
import io

from line_profiler import LineProfiler

from scalar_feature_shard import SelectionOptions, select_features
from scalar_feature_shard._impl import incremental as selection_incremental
from scalar_feature_shard._impl import pearson
from scalar_feature_shard._impl.candidates import build_candidates_from_stats


def _run_selection(args):
    """Run one dense-long selection pass for line profiling."""

    return select_features(
        args.manifest,
        y_col=args.y_col,
        options=SelectionOptions(
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
        ),
    )


def main():
    """Profile the dense-long scalar feature-selection pipeline."""

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
        select_features,
        build_candidates_from_stats,
        selection_incremental.select_features_incremental,
        selection_incremental._build_stage_cache,
        selection_incremental._build_stage_adjacency,
        selection_incremental._build_mask_groups,
        selection_incremental._process_mask_group_pair,
        selection_incremental._get_group_values,
        selection_incremental._stack_local_values,
        selection_incremental._same_mask_tile_stats,
        selection_incremental._or_edge_rows,
        selection_incremental._overlap_counts_many_to_many,
        selection_incremental._stack_feature_tile,
        selection_incremental._pack_valid_rows,
        selection_incremental._pack_bool_rows,
        pearson.batch_r2_many_vs_many,
        pearson.batch_r2_one_vs_many,
    ]:
        profiler.add_function(fn)

    result = profiler(_run_selection)(args)
    print(f"selected_count={result.selected_count}")
    print("selected_feature_ids=" + ",".join(str(feature_id) for feature_id in result.selected_feature_ids[:10]))

    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            profiler.print_stats(stream=f)
    else:
        stream = io.StringIO()
        profiler.print_stats(stream=stream)
        print(stream.getvalue())


if __name__ == "__main__":
    main()
