"""Public reader/writer facade for standalone scalar feature shards."""

from .builder import ScalarBuildSessionStatus, ScalarDatasetBuilder
from .dense_long import (
    ScalarDenseLongDataset,
    build_dense_long_shards_from_sample_bundles,
    open_dense_long_shard,
)
from .exceptions import (
    FeatureNotFoundError,
    ManifestFormatError,
    SampleNotFoundError,
    ScalarFeatureShardError,
)
from .metadata import write_feature_meta, write_sample_meta
from .models import (
    BuildOptions,
    FeatureValues,
    QueryResult,
    ScalarValue,
    SelectionCandidate,
    SelectionOptions,
    SelectionResult,
)
from .reader import ScalarShardDataset, open_shard
from .raw_builder import ScalarRawBuildStatus, ScalarRawDatasetBuilder
from .selection import run_selection, select_features
from .writer import build_shard

__all__ = [
    "BuildOptions",
    "FeatureNotFoundError",
    "FeatureValues",
    "ManifestFormatError",
    "QueryResult",
    "SampleNotFoundError",
    "ScalarDatasetBuilder",
    "ScalarDenseLongDataset",
    "ScalarFeatureShardError",
    "ScalarBuildSessionStatus",
    "ScalarRawBuildStatus",
    "ScalarRawDatasetBuilder",
    "ScalarShardDataset",
    "ScalarValue",
    "SelectionCandidate",
    "SelectionOptions",
    "SelectionResult",
    "build_dense_long_shards_from_sample_bundles",
    "build_shard",
    "open_dense_long_shard",
    "open_shard",
    "run_selection",
    "select_features",
    "write_feature_meta",
    "write_sample_meta",
]
