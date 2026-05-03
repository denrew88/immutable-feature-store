import argparse
import os

from fs.scalar.synthetic import SyntheticConfig, generate_synthetic, write_sample_major


def main():
    """CLI entry point for generating scalar synthetic data and optional shards."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-meta", required=True)
    ap.add_argument("--feature-meta")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-features", type=int, default=2000)
    ap.add_argument("--n-sample-cohorts", type=int, default=16)
    ap.add_argument("--n-missing-patterns", type=int, default=32)
    ap.add_argument("--shared-missing-feature-ratio", type=float, default=0.8)
    ap.add_argument("--residual-missing-rate", type=float, default=0.01)
    ap.add_argument("--y-col", action="append", dest="y_cols")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SyntheticConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_sample_cohorts=args.n_sample_cohorts,
        n_missing_patterns=args.n_missing_patterns,
        shared_missing_feature_ratio=args.shared_missing_feature_ratio,
        residual_missing_rate=args.residual_missing_rate,
        y_cols=None if not args.y_cols else tuple(str(value) for value in args.y_cols),
        seed=args.seed,
    )
    data = generate_synthetic(cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    write_sample_major(data, args.out_dir, args.sample_meta, feature_meta_path=args.feature_meta)


if __name__ == "__main__":
    main()
