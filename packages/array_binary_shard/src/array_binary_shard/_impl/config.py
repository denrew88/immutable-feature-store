from dataclasses import dataclass


@dataclass
class SelectionConfig:
    """Scalar selection과 같은 incremental selector 기준값.

    array-binary package 내부 테스트/호환 경로에서 사용합니다. 일반 array custom
    binary build에는 직접 필요하지 않습니다.
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


@dataclass
class BuildShardConfig:
    """Legacy scalar shard build 설정.

    array-binary package 내부 호환 경로에 남아 있는 설정입니다. 새 array build에는
    `ArrayShardConfig` 또는 public `BuildOptions`를 사용하십시오.
    """

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
    """Array custom binary 중간 bundle flush 설정.

    bundle 파일이 너무 많이 생기면 값을 키우고, bundle 생성 중 메모리 피크가
    크면 값을 줄입니다.
    """

    max_bundle_rows: int = 10000
    max_bundle_bytes: int = 128 * 1024 * 1024


@dataclass
class ArrayShardConfig:
    """Array custom binary shard materialize 설정.

    Attributes:
        samples_per_block: feature 하나를 sample 축으로 자르는 block 크기.
        target_shard_bytes: shard 하나의 목표 크기(byte).
        n_shards: 명시 shard 개수. 0이면 목표 크기 기준 자동 분할.
        row_group_size: estimate/metadata parquet 작성 시 사용하는 row group 기준.
        use_tmp_spill: 큰 입력을 정렬할 때 임시 spill 파일을 사용할지 여부.
        spill_bucket_target_bytes: spill bucket 하나의 목표 크기(byte).
    """

    samples_per_block: int = 8
    target_shard_bytes: int = 256 * 1024 * 1024
    n_shards: int = 0
    row_group_size: int = 64
    use_tmp_spill: bool = False
    spill_bucket_target_bytes: int = 8 * 1024 * 1024


@dataclass
class ArraySyntheticConfig:
    """Array synthetic data 생성 설정.

    저장 포맷이 아니라 테스트 데이터의 sample/feature 수, trace 길이, missing 비율,
    nonfinite 비율, seed를 제어합니다.
    """

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
