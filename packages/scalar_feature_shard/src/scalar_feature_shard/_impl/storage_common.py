"""Shared helpers for dense-long scalar storage."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import polars as pl


def load_sample_targets(
    sample_meta_path: str,
    y_col: str = "y",
    sample_id_col: str = "sample_id",
):
    """Load dense sample ids, target values, and a non-null target mask."""

    df = pl.read_parquet(sample_meta_path)
    if y_col not in df.columns:
        raise ValueError(f"sample_meta parquet must have target column: {y_col}")
    sample_ids = np.arange(df.height, dtype=np.int64)
    if sample_id_col in df.columns:
        stored_ids = df[sample_id_col].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(stored_ids, sample_ids):
            raise ValueError(f"sample_meta {sample_id_col} must equal dense row order 0..n-1")
    y = df[y_col].to_numpy().astype(np.float64, copy=False)
    y_mask = ~np.isnan(y)
    return sample_ids.tolist(), y, y_mask


def load_feature_meta(
    feature_meta_path: str,
    feature_id_col: str = "feature_id",
):
    """Load feature metadata and validate dense row-order feature ids."""

    df = pl.read_parquet(feature_meta_path)
    feature_ids = np.arange(df.height, dtype=np.int32)
    if feature_id_col in df.columns:
        stored_ids = df[feature_id_col].to_numpy().astype(np.int32, copy=False)
        if not np.array_equal(stored_ids, feature_ids):
            raise ValueError(f"feature_meta {feature_id_col} must equal dense row order 0..n-1")
    return feature_ids, df


def close_memmap(mm):
    """Flush and close a NumPy memmap backing handle if it is still open."""

    if mm is None:
        return
    try:
        mm.flush()
    except Exception:
        pass
    backing = getattr(mm, "_mmap", None)
    if backing is not None:
        try:
            backing.close()
        except Exception:
            pass


def cleanup_backing_file(path: Optional[str]):
    """Delete one temporary memmap backing file if it exists."""

    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def cleanup_empty_dir(path: Optional[str]):
    """Remove one temporary directory when it exists and is empty."""

    if not path:
        return
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass


def load_sample_bundle_manifest(manifest_path: str):
    """Load a scalar sample-bundle/raw-sample manifest and resolve paths."""

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("format", "")) != "scalar-sample-bundles":
        raise ValueError(f"unsupported sample-major manifest format: {data.get('format')}")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    def resolve(value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(manifest_dir, value))

    return {
        "sample_meta_path": resolve(str(data["sample_meta_path"])),
        "feature_meta_path": resolve(str(data["feature_meta_path"])),
        "bundle_paths": [resolve(str(value)) for value in list(data.get("bundle_paths", []))],
        "bundle_sample_ids": data.get("bundle_sample_ids"),
        "sample_id_col": str(data.get("sample_id_col", "sample_id")),
        "feature_id_col": str(data.get("feature_id_col", "feature_id")),
        "value_col": str(data.get("value_col", "value")),
    }
