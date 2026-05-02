import os

import numpy as np
import polars as pl

from .storage import ArraySampleBundleWriter, build_array_shards_from_bundles
from ..config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig


def _sample_group_sizes(rng, n_groups: int, total: int, mean: float):
    """Sample positive group sizes whose sum exactly matches the requested total."""
    sizes = rng.poisson(lam=max(mean, 1.0), size=n_groups)
    sizes = np.maximum(sizes, 1)
    diff = int(total - sizes.sum())
    while diff != 0:
        idx = int(rng.integers(0, n_groups))
        if diff > 0:
            sizes[idx] += 1
            diff -= 1
        else:
            if sizes[idx] > 1:
                sizes[idx] -= 1
                diff += 1
    return sizes.astype(np.int32, copy=False)


def generate_array_synthetic(
    bundle_out_dir: str,
    sample_meta_path: str,
    config: ArraySyntheticConfig = None,
    bundle_config: ArrayBundleConfig = None,
    shard_out_dir: str = None,
    shard_config: ArrayShardConfig = None,
):
    """Generate synthetic array traces and optionally build array shards."""
    config = config or ArraySyntheticConfig()
    bundle_config = bundle_config or ArrayBundleConfig()
    rng = np.random.default_rng(config.seed)

    n_noise = int(round(config.n_features * config.noise_feature_ratio))
    n_group_features = max(0, config.n_features - n_noise)
    n_groups = max(1, min(config.n_latent_groups, max(config.n_features, 1)))
    n_group_feature_groups = min(n_groups, n_group_features) if n_group_features > 0 else 0
    if n_group_feature_groups > 0:
        group_sizes = _sample_group_sizes(rng, n_group_feature_groups, n_group_features, config.group_size_mean)
    else:
        group_sizes = np.zeros(0, dtype=np.int32)

    group_freqs = rng.uniform(0.6, 2.4, size=n_groups)
    group_phase_shift = rng.uniform(-np.pi, np.pi, size=n_groups)
    group_time_scale = rng.uniform(0.85, 1.2, size=n_groups)
    sample_amp = rng.normal(1.0, 0.35, size=(n_groups, config.n_samples))
    sample_phase = rng.normal(0.0, 0.45, size=(n_groups, config.n_samples))
    sample_baseline = rng.normal(0.0, 0.2, size=(n_groups, config.n_samples))

    n_informative_groups = max(1, int(round(n_groups * config.informative_group_ratio)))
    informative_groups = rng.choice(n_groups, size=n_informative_groups, replace=False)
    n_latent_for_y = min(config.n_latent_for_y, informative_groups.shape[0])
    y_groups = rng.choice(informative_groups, size=n_latent_for_y, replace=False)
    y_weights = rng.normal(1.0, 0.25, size=n_latent_for_y)
    y = np.zeros(config.n_samples, dtype=np.float64)
    for group_id, weight in zip(y_groups, y_weights):
        y += weight * sample_amp[group_id]
        y += 0.15 * weight * sample_baseline[group_id]
    y += rng.normal(0.0, 0.3, size=config.n_samples)

    feature_group = np.full(config.n_features, -1, dtype=np.int32)
    feature_kind = np.empty(config.n_features, dtype=object)
    feature_sign = rng.choice(np.array([-1.0, 1.0]), size=config.n_features)
    feature_scale = rng.uniform(0.9, config.redundant_strength, size=config.n_features)
    feature_local_freq = rng.uniform(0.75, 1.35, size=config.n_features)
    feature_phase_offset = rng.uniform(-0.8, 0.8, size=config.n_features)

    feat_idx = 0
    for group_id, group_size in enumerate(group_sizes):
        for _ in range(int(group_size)):
            if feat_idx >= config.n_features:
                break
            feature_group[feat_idx] = group_id
            feature_kind[feat_idx] = "group"
            feat_idx += 1
    while feat_idx < config.n_features:
        feature_kind[feat_idx] = "noise"
        feat_idx += 1

    sample_rows = np.arange(config.n_samples, dtype=np.int64)
    sample_ids = sample_rows.copy()
    external_sample_ids = sample_rows + np.int64(config.sample_id_offset)
    sample_keys = np.asarray([f"sample_{sample_id:06d}" for sample_id in external_sample_ids], dtype=object)
    os.makedirs(bundle_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(sample_meta_path) or ".", exist_ok=True)
    feature_meta_path = os.path.join(bundle_out_dir, "array_feature_meta.parquet")

    with ArraySampleBundleWriter(
        bundle_out_dir,
        sample_meta_path,
        feature_meta_path=feature_meta_path,
        n_samples=config.n_samples,
        config=bundle_config,
    ) as writer:
        for sample_row in range(config.n_samples):
            sample_id = int(sample_ids[sample_row])
            for feature_id in range(config.n_features):
                if rng.random() < config.missing_feature_rate:
                    continue
                if rng.random() < config.empty_trace_rate:
                    writer.append_trace(
                        sample_row,
                        sample_id,
                        feature_id,
                        columns={
                            "time": [],
                            "value": [],
                        },
                    )
                    continue

                trace_len = int(rng.integers(config.min_trace_len, config.max_trace_len + 1))
                duration = config.time_duration * rng.uniform(0.8, 1.2)
                dt = rng.lognormal(mean=0.0, sigma=max(config.time_jitter, 1e-6), size=trace_len)
                time = np.cumsum(dt)
                time = time / time[-1] * duration

                group_id = int(feature_group[feature_id])
                if group_id >= 0:
                    freq = group_freqs[group_id] * feature_local_freq[feature_id]
                    phase = group_phase_shift[group_id] + sample_phase[group_id, sample_row] + feature_phase_offset[feature_id]
                    base = np.sin(freq * time + phase)
                    harmonic = 0.35 * np.cos(0.5 * freq * time - phase)
                    envelope = 1.0 + 0.2 * np.sin(time / max(duration, 1e-6) * np.pi)
                    value = (
                        feature_sign[feature_id]
                        * feature_scale[feature_id]
                        * sample_amp[group_id, sample_row]
                        * (base + harmonic)
                        * envelope
                    )
                    value += sample_baseline[group_id, sample_row]
                    value += rng.normal(0.0, config.noise_scale, size=trace_len)
                    time *= group_time_scale[group_id]
                else:
                    drift = rng.normal(0.0, 0.15) * time
                    value = rng.normal(0.0, 0.8, size=trace_len) + drift
                    value += rng.normal(0.0, config.noise_scale, size=trace_len)

                if rng.random() < config.nonfinite_trace_rate and trace_len > 0:
                    n_bad = max(1, int(round(trace_len * config.nonfinite_point_rate)))
                    bad_idx = rng.choice(trace_len, size=min(trace_len, n_bad), replace=False)
                    split = len(bad_idx) // 2
                    if split > 0:
                        time[bad_idx[:split]] = np.nan
                    for j, idx in enumerate(bad_idx[split:]):
                        value[idx] = np.inf if (j % 2 == 0) else -np.inf

                writer.append_trace(
                    sample_row,
                    sample_id,
                    feature_id,
                    columns={
                        "time": time,
                        "value": value,
                    },
                )
        bundle_manifest_path = writer.finish()

    meta = pl.DataFrame(
        {
            "sample_row": pl.Series("sample_row", sample_rows, dtype=pl.Int64),
            "sample_id": pl.Series("sample_id", sample_ids, dtype=pl.Int64),
            "sample_key": pl.Series("sample_key", sample_keys.tolist(), dtype=pl.String),
            "y": pl.Series("y", y, dtype=pl.Float64),
        }
    )
    meta.write_parquet(sample_meta_path)

    feature_meta = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", np.arange(config.n_features, dtype=np.int32), dtype=pl.Int32),
            "feature_key": pl.Series(
                "feature_key",
                [f"feature_{feature_id:06d}" for feature_id in range(config.n_features)],
                dtype=pl.String,
            ),
            "feature_kind": pl.Series("feature_kind", feature_kind.tolist(), dtype=pl.String),
            "group_id": pl.Series("group_id", feature_group, dtype=pl.Int32),
            "scale": pl.Series("scale", feature_scale, dtype=pl.Float64),
        }
    )
    feature_meta.write_parquet(feature_meta_path)

    result = {
        "bundle_manifest_path": bundle_manifest_path,
        "sample_meta_path": sample_meta_path,
        "feature_meta_path": feature_meta_path,
    }
    if shard_out_dir is not None:
        shard_manifest_path = build_array_shards_from_bundles(
            bundle_manifest_path=bundle_manifest_path,
            out_dir=shard_out_dir,
            config=shard_config or ArrayShardConfig(),
        )
        result["shard_manifest_path"] = shard_manifest_path
    return result
