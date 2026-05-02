"""Public reader/writer facade for standalone scalar feature shards."""

from .builder import ScalarDatasetBuilder, ScalarSampleContext
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
    "ScalarFeatureShardError",
    "ScalarSampleContext",
    "ScalarShardDataset",
    "ScalarValue",
    "SelectionCandidate",
    "SelectionOptions",
    "SelectionResult",
    "build_shard",
    "open_shard",
    "run_selection",
    "select_features",
    "write_feature_meta",
    "write_sample_meta",
]
