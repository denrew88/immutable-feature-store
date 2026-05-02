"""Public reader/writer facade for custom array binary shards."""

from .builder import ArrayDatasetBuilder, SampleContext
from ._impl.types import LogicalType, PointColumnSpec, StorageType
from .exceptions import (
    ArrayBinaryShardError,
    FeatureNotFoundError,
    ManifestFormatError,
    SampleNotFoundError,
)
from .metadata import write_feature_meta, write_sample_meta
from .models import BuildOptions, FeatureTraces, QueryResult, Trace
from .reader import BinaryShardDataset, open_shard
from .writer import build_shard, convert_parquet_shard

__all__ = [
    "ArrayBinaryShardError",
    "ArrayDatasetBuilder",
    "BinaryShardDataset",
    "BuildOptions",
    "FeatureNotFoundError",
    "FeatureTraces",
    "LogicalType",
    "ManifestFormatError",
    "PointColumnSpec",
    "QueryResult",
    "SampleNotFoundError",
    "SampleContext",
    "StorageType",
    "Trace",
    "build_shard",
    "convert_parquet_shard",
    "open_shard",
    "write_feature_meta",
    "write_sample_meta",
]
