from dataclasses import dataclass


@dataclass
class SelectionConfig:
    y_r2_threshold: float = 0.01
    min_non_null_y: int = 200
    ff_r2_threshold: float = 0.9
    min_non_null_pair: int = 200
    top_m: int = 100
    initial_cap: int = 2048
    max_step: int = 4096
    batch_size: int = 512
    max_gap: int = 64
    max_candidates: int = 0
    mask_fastpath_min_group: int = 64
    mask_fastpath_min_pairs: int = 8192


@dataclass
class BuildShardConfig:
    n_shards: int = 16
    feature_id_col: str = "feature_id"
    value_col: str = "value"
    sample_id_col: str = "sample_id"
    path_col: str = "sample_path"
    y_col: str = "y"
    values_dtype: str = "float64"
    valid_dtype: str = "uint8"


@dataclass
class ArrayBundleConfig:
    max_bundle_rows: int = 10000
    max_bundle_bytes: int = 128 * 1024 * 1024


@dataclass
class ArrayShardConfig:
    samples_per_block: int = 8
    target_shard_bytes: int = 256 * 1024 * 1024
    n_shards: int = 0
    row_group_size: int = 64
    use_tmp_spill: bool = False
    spill_bucket_target_bytes: int = 8 * 1024 * 1024


@dataclass
class ArrayBinaryBuildOptions:
    target_shard_mb: int = 32
    samples_per_block: int = 16
    n_shards: int | None = None
    codec: str = "none"
    zstd_level: int = 3
    sample_key_col: str = "sample_key"
    feature_key_col: str = "feature_key"


@dataclass
class ArraySyntheticConfig:
    n_samples: int = 1000
    n_features: int = 256
    n_latent_groups: int = 12
    informative_group_ratio: float = 0.3
    n_latent_for_y: int = 4
    group_size_mean: float = 24.0
    noise_feature_ratio: float = 0.2
    redundant_strength: float = 1.25
    noise_scale: float = 0.15
    min_trace_len: int = 96
    max_trace_len: int = 384
    empty_trace_rate: float = 0.02
    missing_feature_rate: float = 0.1
    nonfinite_trace_rate: float = 0.03
    nonfinite_point_rate: float = 0.02
    time_duration: float = 10.0
    time_jitter: float = 0.25
    sample_id_offset: int = 0
    seed: int = 0
