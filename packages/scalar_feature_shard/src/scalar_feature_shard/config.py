"""Internal configuration dataclasses for the scalar feature shard package."""

from dataclasses import dataclass


@dataclass
class ScalarShardBuildOptions:
    """Scalar dense-long builder의 low-level 설정.

    public API에서는 보통 `scalar_feature_shard.BuildOptions`를 사용합니다. 이
    클래스는 내부 구현과 legacy wrapper가 같은 설정을 공유하기 위한 형태입니다.
    필드 의미는 `BuildOptions`와 같습니다.
    """

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
    """Incremental selector의 low-level 설정.

    public API의 `SelectionOptions`와 같은 의미입니다. 내부 selector는 이 값을
    사용해 y-candidate 필터링, feature-feature redundancy 필터링, batch 크기를
    결정합니다.
    """

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
