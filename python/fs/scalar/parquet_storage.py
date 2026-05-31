"""Dense-long scalar storage helpers used by legacy script entrypoints."""

from __future__ import annotations

import numpy as np
import polars as pl

from scalar_feature_shard._impl.storage_common import load_feature_meta, load_sample_targets
from scalar_feature_shard.dense_long import (
    ScalarDenseLongDataset,
    build_dense_long_shards_from_sample_major_manifest,
    load_dense_long_manifest,
    open_dense_long_shard,
)
from scalar_feature_shard.writer import build_shard


def load_manifest(manifest_path: str):
    """Load a dense-long scalar manifest."""

    return load_dense_long_manifest(manifest_path)


def build_feature_locator_index(locator_path: str):
    """Build feature -> `(part_id, offset_in_part)` lookup for dense-long shards."""

    df = pl.read_parquet(locator_path)
    fids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    part_ids = df["part_id"].to_numpy().astype(np.int32, copy=False)
    offsets = df["offset_in_part"].to_numpy().astype(np.int32, copy=False)
    return {int(fid): (int(part_id), int(offset)) for fid, part_id, offset in zip(fids, part_ids, offsets)}


def resolve_selection_stats_path(manifest, y_col: str):
    """Resolve the precomputed selection-stats parquet for one y column."""

    return (getattr(manifest, "selection_stats", None) or {}).get(str(y_col))


def validate_dense_sample_ids(sample_meta_path: str, sample_id_col: str = "sample_id", sample_row_col: str = "sample_row"):
    """Validate dense sample ids and return the sample count."""

    del sample_row_col
    df = pl.read_parquet(sample_meta_path)
    expected = np.arange(df.height, dtype=np.int64)
    if sample_id_col in df.columns:
        stored = df[sample_id_col].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(stored, expected):
            raise ValueError(f"sample_meta {sample_id_col} must equal dense row order 0..n-1")
    return int(df.height)


__all__ = [
    "ScalarDenseLongDataset",
    "build_dense_long_shards_from_sample_major_manifest",
    "build_feature_locator_index",
    "build_shard",
    "load_dense_long_manifest",
    "load_feature_meta",
    "load_manifest",
    "load_sample_targets",
    "open_dense_long_shard",
    "resolve_selection_stats_path",
    "validate_dense_sample_ids",
]
