import argparse

from scalar_feature_shard import SelectionOptions, select_features


def main():
    """CLI entry point for running dense-long scalar feature selection."""

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

    result = select_features(
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
    for feature_id in result.selected_feature_ids:
        print(feature_id)


if __name__ == "__main__":
    main()
