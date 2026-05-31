"""Public dataclasses used by the array binary shard package."""

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class BuildOptions:
    """Array custom binary build 설정.

    bundle 또는 direct builder stage에서 feature-major custom binary shard를 만들 때
    사용합니다. 처음에는 `samples_per_block`, `target_shard_mb`, `codec`만
    지정하면 됩니다.

    Attributes:
        target_shard_mb: shard 하나의 목표 크기(MB).
        samples_per_block: feature 하나를 sample 축으로 자르는 block 크기.
        n_shards: 명시적인 shard 개수. `None`이면 목표 크기 기준 자동 분할입니다.
        codec: payload codec. 현재 권장 기본값은 `none`입니다.
        zstd_level: `codec='zstd'`일 때만 쓰는 압축 level.
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


@dataclass(frozen=True)
class Trace:
    """One decoded trace returned by the public reader API.

    Attributes:
        feature_id: Logical feature identifier that was requested.
        sample_id: External sample identifier that was requested.
        present: Whether the feature has a stored trace for the requested sample.
        flags: Stored per-sample trace flags.
        feature_key: Optional external feature key when the request path used keys.
        sample_key: Optional external sample key when the request path used keys.
        columns: Mapping from manifest point-column name to the decoded values for
            this trace. Numeric columns are returned as NumPy arrays. When
            `decode_categorical=True` is used, categorical columns are returned as
            tuples of original labels or `None`.
    """

    feature_id: int
    sample_id: int
    present: bool
    flags: int
    feature_key: Optional[str]
    sample_key: Optional[str]
    columns: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureTraces:
    """Batch of traces for one feature across multiple sample ids.

    Attributes:
        feature_id: Logical feature identifier associated with the batch.
        sample_ids: Requested sample ids in response order.
        traces: Per-sample trace results aligned with `sample_ids`.
        feature_key: Optional external feature key when the batch was requested by key.
        sample_keys: Optional external sample keys aligned with `sample_ids`.
    """

    feature_id: int
    sample_ids: Sequence[int]
    traces: Sequence[Trace]
    feature_key: Optional[str] = None
    sample_keys: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class QueryResult:
    """Batch result covering multiple features and sample ids.

    Attributes:
        feature_ids: Requested feature ids in response order.
        sample_ids: Requested sample ids shared by all feature batches.
        features: Per-feature trace batches aligned with `feature_ids`.
        feature_keys: Optional external feature keys aligned with `feature_ids`.
        sample_keys: Optional external sample keys aligned with `sample_ids`.
    """

    feature_ids: Sequence[int]
    sample_ids: Sequence[int]
    features: Sequence[FeatureTraces]
    feature_keys: Optional[Sequence[str]] = None
    sample_keys: Optional[Sequence[str]] = None
