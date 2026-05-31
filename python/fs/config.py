from dataclasses import dataclass


@dataclass
class SelectionConfig:
    """Scalar feature selection 설정.

    y와의 1차 상관 후보를 만들고, 이미 선택된 feature와 너무 비슷한 후보를
    걸러내는 incremental selection의 기준값입니다. 처음에는 `top_m`만 원하는
    개수로 바꾸고 나머지는 기본값을 쓰는 것을 권장합니다.

    Attributes:
        y_r2_threshold: y와의 최소 R^2. 낮을수록 후보가 많이 남습니다.
        min_non_null_y: feature와 y가 동시에 present인 sample 최소 개수.
        ff_r2_threshold: 후보 feature와 이미 선택된 feature 사이의 최대 허용 R^2.
        min_non_null_pair: feature-feature R^2를 계산할 때 필요한 공통 present sample 수.
        top_m: 최종 선택할 feature 수.
        initial_cap: selection 시작 시 가져올 후보 수.
        max_step: 후보가 부족할 때 한 번에 늘리는 최대 후보 수.
        batch_size: 후보를 reader에서 읽어오는 batch 크기.
        max_gap: 추가 후보 탐색 중 개선 없이 허용할 gap. 0이면 제한을 두지 않습니다.
        max_candidates: 후보 수 상한. 0이면 제한을 두지 않습니다.
        mask_fastpath_min_group: mask fast-path를 적용할 최소 그룹 크기.
        mask_fastpath_min_pairs: mask fast-path를 적용할 최소 pair 수.
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

    현재 표준 scalar 포맷은 `ScalarShardBuildOptions`/`BuildOptions`의 dense-long
    설정을 사용합니다. 이 클래스는 오래된 script와 내부 호환 경로에서 사용되며,
    새 코드에서는 가능하면 `ScalarShardBuildOptions`를 사용하십시오.
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
    """Array custom binary의 중간 bundle flush 설정.

    bundle은 custom binary shard를 만들기 전의 중간 parquet 묶음입니다. 사용자가
    직접 tuning할 일은 많지 않습니다. bundle 파일이 너무 많이 생기면 값을 키우고,
    bundle 생성 중 메모리 피크가 크면 값을 줄입니다.

    Attributes:
        max_bundle_rows: bundle 하나에 넣을 최대 trace row 수.
        max_bundle_bytes: bundle 하나의 목표 최대 byte 수.
    """

    max_bundle_rows: int = 10000
    max_bundle_bytes: int = 128 * 1024 * 1024


@dataclass
class ArrayShardConfig:
    """Array custom binary shard materialize 설정.

    bundle에서 feature-major custom binary shard를 만들 때 쓰는 low-level 설정입니다.
    유지보수성보다 조회 속도가 중요한 특수 경로에서만 직접 조정하는 것을 권장합니다.

    Attributes:
        samples_per_block: feature 하나를 sample 축으로 자르는 block 크기.
            작을수록 부분 sample 조회 낭비가 줄지만 index와 header overhead가 커집니다.
        target_shard_bytes: feature partitioning 시 shard 하나의 목표 크기.
        n_shards: 명시적인 shard 개수. 0이면 `target_shard_bytes` 기준으로 자동 분할합니다.
        row_group_size: feature estimate를 만들 때 쓰는 parquet row group 기준값.
        use_tmp_spill: 큰 bundle을 정렬할 때 임시 spill 파일을 사용할지 여부.
        spill_bucket_target_bytes: spill bucket 하나의 목표 크기.
    """

    samples_per_block: int = 8
    target_shard_bytes: int = 256 * 1024 * 1024
    n_shards: int = 0
    row_group_size: int = 64
    use_tmp_spill: bool = False
    spill_bucket_target_bytes: int = 8 * 1024 * 1024


@dataclass
class ArrayBinaryBuildOptions:
    """Array custom binary direct builder의 public 설정.

    sample context로 trace를 직접 넣고 마지막에 custom binary shard를 만들 때 쓰는
    설정입니다. 처음에는 `samples_per_block`, `target_shard_mb`, `codec` 정도만
    지정하면 됩니다.

    Attributes:
        target_shard_mb: shard 하나의 목표 크기(MB).
        samples_per_block: feature 하나를 sample 축으로 자르는 block 크기.
        n_shards: 명시적인 shard 개수. None이면 목표 크기 기준으로 자동 분할합니다.
        codec: payload codec. 현재 기본/권장은 `"none"`입니다.
        zstd_level: `codec="zstd"`일 때만 쓰는 압축 level.
        sample_key_col: sample metadata의 external key column.
        feature_key_col: feature metadata의 external key column.
    """

    target_shard_mb: int = 32
    samples_per_block: int = 16
    n_shards: int | None = None
    codec: str = "none"
    zstd_level: int = 3
    sample_key_col: str = "sample_key"
    feature_key_col: str = "feature_key"


@dataclass
class ScalarShardBuildOptions:
    """Scalar dense-long shard build 설정.

    scalar 표준 builder가 sample별 raw parquet stage를 만들고, 마지막에 dense-long
    shard로 materialize할 때 사용합니다. 처음에는 `target_shard_mb`와
    `stats_y_cols`만 지정하면 충분합니다.

    Attributes:
        target_shard_mb: dense-long part 하나의 목표 크기(MB).
        n_shards: legacy shard 개수 override. dense-long 표준 경로에서는 보통 쓰지 않습니다.
        feature_id_col: raw/sample-major 입력에서 feature id column 이름.
        value_col: raw/sample-major 입력에서 scalar value column 이름.
        sample_id_col: sample id column 이름.
        sample_key_col: sample metadata의 external key column.
        feature_key_col: feature metadata의 external key column.
        path_col: sample-major manifest에서 raw sample path column 이름.
        y_col: selection stats target column. `stats_y_cols`가 없을 때 사용합니다.
        stats_y_cols: build 중 selection stats를 만들 y column 목록.
        values_dtype: value 저장 dtype. 현재 표준은 `float64`입니다.
        valid_dtype: mask 저장 dtype. 현재 표준은 `uint8`입니다.
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
class ArraySyntheticConfig:
    """Array synthetic data 생성 설정.

    벤치마크와 예제 데이터 생성을 위한 설정입니다. 실제 저장 포맷 설정이 아니라,
    synthetic trace의 크기, missing/nonfinite 비율, random seed를 제어합니다.
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
