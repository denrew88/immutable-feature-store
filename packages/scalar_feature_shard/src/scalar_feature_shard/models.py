"""Public dataclasses for the scalar feature shard package."""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class BuildOptions:
    """Scalar dense-long shard build 설정.

    sample별 raw parquet stage를 만든 뒤 dense-long shard로 materialize할 때
    사용합니다. 일반 사용자는 `target_shard_mb`와 `stats_y_cols`만 지정하면
    충분합니다.

    Attributes:
        target_shard_mb: dense-long part 하나의 목표 크기(MB). 너무 작으면 part가
            많아지고, 너무 크면 build/query 시 한 파일의 작업량이 커집니다.
        n_shards: legacy shard 개수 override. dense-long 표준 경로에서는 보통 쓰지 않습니다.
        feature_id_col: sample-major/raw 입력의 feature id column 이름.
        value_col: scalar value column 이름.
        sample_id_col: sample id column 이름.
        sample_key_col: sample metadata의 external key column 이름.
        feature_key_col: feature metadata의 external key column 이름.
        path_col: sample-major manifest에서 raw sample path를 담는 column 이름.
        y_col: `stats_y_cols`를 생략했을 때 selection stats를 만들 target column.
        stats_y_cols: build 중 `selection_stats/<y>.parquet`를 만들 target column 목록.
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


@dataclass(frozen=True)
class SelectionOptions:
    """Scalar feature selection 설정.

    y와의 상관 후보를 만들고, 이미 선택된 feature와 너무 비슷한 후보를
    제거하는 incremental selection의 기준값입니다. 처음에는 `y_col`과
    `top_m`만 바꾸고 나머지는 기본값을 쓰는 것을 권장합니다.

    Attributes:
        y_col: 사용할 precomputed selection stats의 target column.
        y_r2_threshold: y와의 최소 R^2. 낮을수록 후보가 많이 남습니다.
        min_non_null_y: feature와 y가 동시에 present인 sample 최소 개수.
        ff_r2_threshold: 후보 feature와 이미 선택된 feature 사이의 최대 허용 R^2.
        min_non_null_pair: feature-feature R^2 계산에 필요한 공통 present sample 수.
        top_m: 최종 선택할 feature 수.
        initial_cap: selection 시작 시 가져올 후보 수.
        max_step: 후보가 부족할 때 한 번에 늘리는 최대 후보 수.
        batch_size: 후보 feature를 reader에서 읽어오는 batch 크기.
        max_gap: 추가 후보 탐색 중 개선 없이 허용할 gap. 0이면 제한 없음입니다.
        max_candidates: 후보 수 상한. 0이면 제한 없음입니다.
        mask_fastpath_min_group: mask fast-path를 적용할 최소 그룹 크기.
        mask_fastpath_min_pairs: mask fast-path를 적용할 최소 pair 수.
    """

    y_col: str = "y"
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


@dataclass(frozen=True)
class ScalarValue:
    """One scalar feature value aligned to one sample id."""

    feature_id: int
    sample_id: int
    present: bool
    value: Optional[float]
    feature_key: Optional[str] = None
    sample_key: Optional[str] = None


@dataclass(frozen=True)
class FeatureValues:
    """Batch of scalar values for one feature across multiple samples."""

    feature_id: int
    sample_ids: Sequence[int]
    values: Sequence[ScalarValue]
    feature_key: Optional[str] = None
    sample_keys: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class QueryResult:
    """Batch result covering multiple features and multiple samples."""

    feature_ids: Sequence[int]
    sample_ids: Sequence[int]
    features: Sequence[FeatureValues]
    feature_keys: Optional[Sequence[str]] = None
    sample_keys: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class SelectionCandidate:
    """One ranked feature-vs-y candidate."""

    feature_id: int
    feature_key: Optional[str]
    r2_y: float
    n_valid_y: int


@dataclass(frozen=True)
class SelectionResult:
    """Public result returned by the scalar selection facade."""

    y_col: str
    selected_feature_ids: Sequence[int]
    selected_feature_keys: Sequence[Optional[str]]
    candidates: Sequence[SelectionCandidate]
    candidate_count: int
    selected_count: int
    used_precomputed_stats: bool
