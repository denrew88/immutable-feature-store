"""Compatibility wrapper for the package scalar dense-long reader."""

from .._package_sources import ensure_package_source

ensure_package_source("scalar_feature_shard")

from scalar_feature_shard.models import FeatureValues, QueryResult, ScalarValue
from scalar_feature_shard.reader import ScalarDenseLongDataset, open_dense_long_shard

__all__ = [
    "FeatureValues",
    "QueryResult",
    "ScalarDenseLongDataset",
    "ScalarValue",
    "open_dense_long_shard",
]
