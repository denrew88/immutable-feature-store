import argparse

import numpy as np

from fs.config import SelectionConfig
from fs.scalar.synthetic import SyntheticConfig, generate_synthetic
from fs.feature_selection.validate import validate_batch_kernel, validate_incremental_vs_naive


def main():
    """Run the core scalar feature-selection correctness test suite."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    synth_cfg = SyntheticConfig(
        n_samples=200,
        n_features=300,
        n_sample_cohorts=8,
        n_missing_patterns=12,
        shared_missing_feature_ratio=0.9,
        residual_missing_rate=0.01,
        seed=args.seed,
    )
    data = generate_synthetic(synth_cfg)

    X = data["X"]
    M = data["M"]
    y = data["y"]
    y_mask = ~np.isnan(y)

    validate_batch_kernel(X, M, y, y_mask, min_non_null=20)

    config = SelectionConfig(
        y_r2_threshold=0.01,
        min_non_null_y=20,
        ff_r2_threshold=0.9,
        min_non_null_pair=20,
        top_m=30,
        initial_cap=50,
        max_step=100,
        batch_size=64,
        mask_fastpath_min_group=8,
        mask_fastpath_min_pairs=64,
    )
    validate_incremental_vs_naive(X, M, y, y_mask, config)
    print("tests passed")


if __name__ == "__main__":
    main()
