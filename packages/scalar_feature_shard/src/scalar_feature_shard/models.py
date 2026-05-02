"""Public dataclasses for the scalar feature shard package."""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class BuildOptions:
    """Options controlling sample-major to scalar-shard conversion."""

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
    """Options controlling incremental scalar feature selection."""

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
