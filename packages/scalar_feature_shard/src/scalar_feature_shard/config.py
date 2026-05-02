"""Internal configuration dataclasses for the scalar feature shard package."""

from dataclasses import dataclass


@dataclass
class ScalarShardBuildOptions:
    """Low-level build options used by the scalar shard builder implementation."""

    target_shard_mb: int = 32
    n_shards: int | None = None
    feature_id_col: str = "feature_id"
    value_col: str = "value"
    sample_id_col: str = "sample_id"
    sample_key_col: str = "sample_key"
    feature_key_col: str = "feature_key"
    path_col: str = "sample_path"
    y_col: str = "y"
    stats_y_cols: tuple[str, ...] | None = None
    values_dtype: str = "float64"
    valid_dtype: str = "uint8"


@dataclass
class SelectionConfig:
    """Low-level selection configuration used by the incremental selector."""

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
