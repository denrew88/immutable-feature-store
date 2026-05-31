import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import polars as pl


@dataclass
class SyntheticConfig:
    """Configuration for scalar synthetic dataset generation."""

    n_samples: int = 1000
    n_features: int = 2000
    n_latent_groups: int = 20
    group_size_mean: float = 50.0
    informative_group_ratio: float = 0.3
    n_latent_for_y: int = 5
    noise_scale: float = 1.0
    redundant_strength: float = 1.5
    noise_feature_ratio: float = 0.2
    neg_corr_ratio: float = 0.2
    missing_rate: float = 0.1
    missing_rate_variability: float = 0.05
    sparse_feature_ratio: float = 0.05
    sparse_target_valid: int = None
    group_missing_share_prob: float = 0.2
    n_sample_cohorts: int = 16
    n_missing_patterns: int = 32
    shared_missing_feature_ratio: float = 0.8
    residual_missing_rate: float = 0.01
    min_present_cohorts: int = 2
    y_missing_rate: float = 0.0
    y_cols: tuple[str, ...] | None = None
    seed: int = 0


def _resolve_y_cols(config: SyntheticConfig) -> list[str]:
    """Resolve the ordered list of synthetic target column names."""
    if not config.y_cols:
        return ["y"]
    ordered: list[str] = []
    for value in config.y_cols:
        name = str(value)
        if not name:
            raise ValueError("y_cols must not contain empty names")
        if name not in ordered:
            ordered.append(name)
    if not ordered:
        raise ValueError("at least one y column is required")
    return ordered


def _generate_target(
    rng,
    latent: np.ndarray,
    informative_groups: np.ndarray,
    config: SyntheticConfig,
):
    """Generate one synthetic target from the latent factors."""
    n_latent_for_y = min(config.n_latent_for_y, len(informative_groups))
    y_groups = rng.choice(informative_groups, size=n_latent_for_y, replace=False)
    weights = rng.normal(1.0, 0.2, size=n_latent_for_y)

    y = np.zeros(config.n_samples, dtype=np.float64)
    for g, w in zip(y_groups, weights):
        y += w * latent[g]
    y += rng.normal(0.0, config.noise_scale, size=config.n_samples)

    if config.y_missing_rate > 0.0:
        y_mask = rng.random(config.n_samples) > config.y_missing_rate
        y = y.copy()
        y[~y_mask] = np.nan

    return y, y_groups


def _sample_group_sizes(rng, n_groups: int, total: int, mean: float) -> List[int]:
    """Sample positive latent-group sizes that sum to the requested total."""
    if total <= 0:
        return [0 for _ in range(n_groups)]
    if total < n_groups:
        sizes = np.zeros(n_groups, dtype=np.int32)
        chosen = rng.choice(n_groups, size=total, replace=False)
        sizes[chosen] = 1
        return sizes.tolist()
    sizes = rng.poisson(lam=mean, size=n_groups)
    sizes = np.maximum(sizes, 1)
    diff = int(total - sizes.sum())
    while diff != 0:
        if diff > 0:
            idx = rng.integers(0, n_groups)
            sizes[idx] += 1
            diff -= 1
        else:
            idx = rng.integers(0, n_groups)
            if sizes[idx] > 1:
                sizes[idx] -= 1
                diff += 1
    return sizes.tolist()


def _build_sample_cohorts(rng, n_samples: int, n_cohorts: int) -> np.ndarray:
    """Assign each sample to a shuffled cohort id."""
    n_cohorts = max(1, min(n_cohorts, n_samples))
    cohort_ids = np.arange(n_samples, dtype=np.int32) % n_cohorts
    rng.shuffle(cohort_ids)
    return cohort_ids


def _build_missing_pattern_pool(rng, cohort_ids: np.ndarray, config: SyntheticConfig) -> List[np.ndarray]:
    """Build reusable cohort-level missingness masks."""
    n_samples = cohort_ids.shape[0]
    n_cohorts = int(cohort_ids.max()) + 1
    cohort_present_prob = float(np.clip(1.0 - config.missing_rate, 0.05, 0.98))
    min_present = max(1, min(config.min_present_cohorts, n_cohorts))
    patterns = [np.ones(n_samples, dtype=bool)]
    for _ in range(max(0, config.n_missing_patterns - 1)):
        present_cohorts = rng.random(n_cohorts) < cohort_present_prob
        if int(present_cohorts.sum()) < min_present:
            chosen = rng.choice(n_cohorts, size=min_present, replace=False)
            present_cohorts[chosen] = True
        patterns.append(present_cohorts[cohort_ids])
    return patterns


def generate_synthetic(config: SyntheticConfig):
    """Generate a scalar synthetic dataset with correlated groups and missingness."""
    rng = np.random.default_rng(config.seed)
    y_cols = _resolve_y_cols(config)

    n_noise = int(config.n_features * config.noise_feature_ratio)
    n_group_features = config.n_features - n_noise

    group_sizes = _sample_group_sizes(rng, config.n_latent_groups, n_group_features, config.group_size_mean)

    latent = rng.normal(0.0, 1.0, size=(config.n_latent_groups, config.n_samples))

    # informative groups for Y-like targets
    n_informative_groups = max(1, int(config.n_latent_groups * config.informative_group_ratio))
    informative_groups = rng.choice(config.n_latent_groups, size=n_informative_groups, replace=False)
    targets = {}
    target_groups = {}
    for y_col in y_cols:
        target, groups = _generate_target(rng, latent, informative_groups, config)
        targets[str(y_col)] = target
        target_groups[str(y_col)] = groups

    cohort_ids = _build_sample_cohorts(rng, config.n_samples, config.n_sample_cohorts)
    missing_patterns = _build_missing_pattern_pool(rng, cohort_ids, config)
    residual_missing_rate = float(np.clip(config.residual_missing_rate, 0.0, 0.95))

    X = np.zeros((config.n_features, config.n_samples), dtype=np.float64)
    M = np.ones_like(X, dtype=np.uint8)

    feature_meta = []
    feat_idx = 0

    sparse_target_valid = config.sparse_target_valid
    if sparse_target_valid is None:
        sparse_target_valid = max(2, int(0.05 * config.n_samples))

    sparse_count = int(config.n_features * config.sparse_feature_ratio)
    sparse_features = set(rng.choice(config.n_features, size=sparse_count, replace=False).tolist())

    for g, size in enumerate(group_sizes):
        group_mask = None
        group_pattern_idx = None
        if missing_patterns and rng.random() < config.group_missing_share_prob:
            group_pattern_idx = int(rng.integers(0, len(missing_patterns)))
            group_mask = missing_patterns[group_pattern_idx]

        for _ in range(size):
            sign = -1.0 if rng.random() < config.neg_corr_ratio else 1.0
            strength = config.redundant_strength * rng.uniform(0.8, 1.2)
            x = sign * strength * latent[g] + rng.normal(0.0, config.noise_scale, size=config.n_samples)

            pattern_idx = group_pattern_idx
            base_mask = group_mask
            if base_mask is None and missing_patterns and rng.random() < config.shared_missing_feature_ratio:
                pattern_idx = int(rng.integers(0, len(missing_patterns)))
                base_mask = missing_patterns[pattern_idx]

            mr = np.clip(residual_missing_rate + rng.normal(0.0, config.missing_rate_variability * 0.25), 0.0, 0.5)
            if feat_idx in sparse_features:
                mr = 1.0 - (sparse_target_valid / config.n_samples)
                mr = np.clip(mr, 0.0, 0.98)

            if feat_idx in sparse_features:
                if base_mask is None:
                    available = np.arange(config.n_samples, dtype=np.int32)
                else:
                    available = np.flatnonzero(base_mask)
                if available.size > sparse_target_valid:
                    keep = rng.choice(available, size=sparse_target_valid, replace=False)
                else:
                    keep = available
                mask = np.zeros(config.n_samples, dtype=bool)
                mask[keep] = True
            else:
                mask = np.ones(config.n_samples, dtype=bool) if base_mask is None else base_mask.copy()
                if mr > 0.0:
                    mask &= rng.random(config.n_samples) > mr

            x = x.astype(np.float64)
            x[~mask] = np.nan

            X[feat_idx] = x
            M[feat_idx] = mask.astype(np.uint8)
            feature_meta.append({"group": g, "type": "group", "pattern_idx": pattern_idx})
            feat_idx += 1

    for _ in range(n_noise):
        x = rng.normal(0.0, config.noise_scale, size=config.n_samples)
        pattern_idx = None
        base_mask = None
        if missing_patterns and rng.random() < config.shared_missing_feature_ratio:
            pattern_idx = int(rng.integers(0, len(missing_patterns)))
            base_mask = missing_patterns[pattern_idx]
        mr = np.clip(residual_missing_rate + rng.normal(0.0, config.missing_rate_variability * 0.25), 0.0, 0.5)
        if feat_idx in sparse_features:
            mr = 1.0 - (sparse_target_valid / config.n_samples)
            mr = np.clip(mr, 0.0, 0.98)
        if feat_idx in sparse_features:
            if base_mask is None:
                available = np.arange(config.n_samples, dtype=np.int32)
            else:
                available = np.flatnonzero(base_mask)
            if available.size > sparse_target_valid:
                keep = rng.choice(available, size=sparse_target_valid, replace=False)
            else:
                keep = available
            mask = np.zeros(config.n_samples, dtype=bool)
            mask[keep] = True
        else:
            mask = np.ones(config.n_samples, dtype=bool) if base_mask is None else base_mask.copy()
            if mr > 0.0:
                mask &= rng.random(config.n_samples) > mr
        x = x.astype(np.float64)
        x[~mask] = np.nan
        X[feat_idx] = x
        M[feat_idx] = mask.astype(np.uint8)
        feature_meta.append({"group": None, "type": "noise", "pattern_idx": pattern_idx})
        feat_idx += 1

    feature_ids = np.arange(config.n_features, dtype=np.int32)
    primary_y_col = "y" if "y" in targets else y_cols[0]
    Y = np.column_stack([np.asarray(targets[y_col], dtype=np.float64) for y_col in y_cols])

    return {
        "X": X.T.copy(),
        "M": M.T.copy(),
        "Y": Y,
        "y_cols": tuple(str(y_col) for y_col in y_cols),
        "y": targets[primary_y_col],
        "targets": targets,
        "primary_y_col": primary_y_col,
        "feature_ids": feature_ids,
        "feature_meta": feature_meta,
        "informative_groups": informative_groups,
        "y_groups": target_groups[primary_y_col],
        "target_groups": target_groups,
        "sample_cohort_ids": cohort_ids,
    }


def write_sample_major(dataset: Dict, out_dir: str, sample_meta_path: str, feature_meta_path: str = None):
    """Write a scalar synthetic dataset into sample-major parquet files.

    The scalar v2 convention treats both sample ids and feature ids as dense
    zero-based row indices defined by metadata row order. This writer therefore
    emits:

    - `sample_meta.parquet` with dense `sample_id` and external `sample_key`
    - `feature_meta.parquet` with dense `feature_id` and external `feature_key`
    - per-sample long-format parquet files containing dense `feature_id`

    Args:
        dataset: Synthetic dataset dictionary produced by `generate_synthetic`.
        out_dir: Directory where per-sample parquet files are written.
        sample_meta_path: Output path for `sample_meta.parquet`.
        feature_meta_path: Optional output path for `feature_meta.parquet`. When
            omitted, the file is written next to `sample_meta_path`.
    """
    out_dir_abs = os.path.abspath(out_dir)
    sample_meta_path_abs = os.path.abspath(sample_meta_path)
    sample_meta_dir = os.path.dirname(sample_meta_path_abs)
    os.makedirs(out_dir_abs, exist_ok=True)
    X = dataset["X"]
    M = dataset["M"]
    y_cols = tuple(str(value) for value in dataset.get("y_cols", ()))
    Y = dataset.get("Y")
    if Y is not None and y_cols:
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError("dataset['Y'] must be a 2D array of shape (n_samples, n_y_cols)")
        if Y.shape[1] != len(y_cols):
            raise ValueError(
                f"dataset['Y'] column count must match dataset['y_cols']: {Y.shape[1]} != {len(y_cols)}"
            )
        targets = {y_col: Y[:, idx] for idx, y_col in enumerate(y_cols)}
    else:
        targets = dict(dataset.get("targets") or {"y": dataset["y"]})
    feature_meta = dataset["feature_meta"]

    n_samples, n_features = X.shape
    sample_paths = []
    if feature_meta_path is None:
        feature_meta_path = os.path.join(sample_meta_dir, "feature_meta.parquet")
    feature_meta_path_abs = os.path.abspath(feature_meta_path)

    for s_idx in range(n_samples):
        values = X[s_idx]
        valid = M[s_idx].astype(bool)
        fids = np.arange(n_features, dtype=np.int32)
        df = pl.DataFrame({
            "sample_id": pl.Series("sample_id", np.full(int(valid.sum()), s_idx, dtype=np.int64), dtype=pl.Int64),
            "feature_id": pl.Series("feature_id", fids[valid], dtype=pl.Int32),
            "value": values[valid],
        })
        sample_path = os.path.join(out_dir_abs, f"sample_{s_idx:06d}.parquet")
        df.write_parquet(sample_path)
        sample_paths.append(os.path.relpath(sample_path, sample_meta_dir).replace("\\", "/"))

    meta_data = {
        "sample_id": np.arange(n_samples, dtype=np.int64),
        "sample_key": [f"sample_{sample_id:06d}" for sample_id in range(n_samples)],
    }
    for y_col, values in targets.items():
        meta_data[str(y_col)] = values
    meta_data["sample_path"] = sample_paths
    meta = pl.DataFrame(meta_data)
    meta.write_parquet(sample_meta_path_abs)

    feature_meta_df = pl.DataFrame(
        {
            "feature_id": np.arange(n_features, dtype=np.int32),
            "feature_key": [f"feature_{feature_id:06d}" for feature_id in range(n_features)],
            "group": [row["group"] for row in feature_meta],
            "type": [row["type"] for row in feature_meta],
            "pattern_idx": [row["pattern_idx"] for row in feature_meta],
        }
    )
    feature_meta_df.write_parquet(feature_meta_path_abs)
