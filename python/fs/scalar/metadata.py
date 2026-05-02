"""dense scalar sample/feature metadata parquet를 쓰는 helper."""

from __future__ import annotations

from ..array.metadata import write_feature_meta, write_sample_meta

__all__ = ["write_sample_meta", "write_feature_meta"]
