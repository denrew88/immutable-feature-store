"""Public dataclasses used by the array binary shard package."""

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class BuildOptions:
    """Options controlling bundle-to-binary-shard conversion.

    Attributes:
        target_shard_mb: Preferred maximum shard size used for feature partitioning.
        samples_per_block: Number of samples packed into one logical block.
        n_shards: Optional explicit shard count override. `None` means target-size partitioning.
        codec: Payload codec name. The current recommended default is `none`.
        zstd_level: Compression level used only when `codec='zstd'`.
        sample_key_col: Sample metadata column containing external keys.
        feature_key_col: Feature metadata column containing external keys.
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
