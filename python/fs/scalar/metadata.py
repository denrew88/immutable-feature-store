"""Helpers for writing dense scalar sample/feature metadata parquet files."""

from __future__ import annotations

from ..array.metadata import write_feature_meta, write_sample_meta

__all__ = ["write_sample_meta", "write_feature_meta"]
