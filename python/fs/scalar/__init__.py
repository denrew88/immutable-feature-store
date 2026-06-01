"""Dense-long scalar feature compatibility package."""

from .._package_sources import ensure_package_source

ensure_package_source("scalar_feature_shard")

from .builder import ScalarBuildSessionStatus, ScalarDatasetBuilder
from .metadata import write_feature_meta, write_sample_meta
from scalar_feature_shard import build_shard, select_features
from .reader import (
    FeatureValues,
    QueryResult,
    ScalarDenseLongDataset,
    ScalarValue,
    open_dense_long_shard,
)

__all__ = [
    "FeatureValues",
    "QueryResult",
    "ScalarBuildSessionStatus",
    "ScalarDatasetBuilder",
    "ScalarDenseLongDataset",
    "ScalarValue",
    "build_shard",
    "open_dense_long_shard",
    "select_features",
    "write_feature_meta",
    "write_sample_meta",
]
