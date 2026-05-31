"""Dense-long compatible feature-selection candidate helpers."""

from scalar_feature_shard._impl.candidates import (
    build_candidates_from_inmemory,
    build_candidates_from_stats,
)
from fs.types import Candidate
from .pearson import pairwise_r2

import numpy as np
import polars as pl


def build_candidates_from_shards(
    shard_paths,
    y,
    y_mask,
    min_non_null_y,
    y_r2_threshold,
    max_candidates=0,
    batch_size=512,
    feature_id_col="feature_id",
):
    """Scan dense-long part files and rank feature candidates by r^2."""

    del batch_size
    candidates = []
    for part_id, path in enumerate(shard_paths):
        df = (
            pl.scan_parquet(path)
            .select([feature_id_col, "sample_id", "mask", "value"])
            .sort([feature_id_col, "sample_id"])
            .collect()
        )
        offset = 0
        for feature_key, group in df.group_by(feature_id_col, maintain_order=True):
            feature_id = int(feature_key[0] if isinstance(feature_key, tuple) else feature_key)
            values = group["value"].to_numpy().astype(np.float64, copy=False)
            valid = group["mask"].to_numpy().astype(np.uint8, copy=False)
            r2, n = pairwise_r2(values, valid, y, y_mask, min_non_null=min_non_null_y)
            if n >= min_non_null_y and r2 >= y_r2_threshold:
                candidates.append(Candidate(feature_id, int(part_id), int(offset), float(r2), int(n)))
            offset += 1
    candidates.sort(key=lambda c: c.r2_y, reverse=True)
    if max_candidates and len(candidates) > max_candidates:
        return candidates[:max_candidates]
    return candidates


def build_candidates_from_locator(locator_path, min_non_null_y, y_r2_threshold, max_candidates=0):
    """Build candidates directly from a selection-stats/locator parquet file."""

    return build_candidates_from_stats(locator_path, min_non_null_y, y_r2_threshold, max_candidates=max_candidates)


__all__ = [
    "build_candidates_from_inmemory",
    "build_candidates_from_locator",
    "build_candidates_from_shards",
    "build_candidates_from_stats",
]
