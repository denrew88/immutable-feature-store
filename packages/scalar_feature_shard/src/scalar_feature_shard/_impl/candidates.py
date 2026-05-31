"""Candidate helpers for dense-long scalar feature selection."""

import numpy as np
import polars as pl

from ..types import Candidate
from .pearson import batch_r2_one_vs_many


def build_candidates_from_stats(stats_path, min_non_null_y, y_r2_threshold, max_candidates=0):
    """Build ranked candidates from a dense-long selection-stats sidecar."""

    df = pl.read_parquet(stats_path)
    if {"part_id", "offset_in_part"}.issubset(df.columns):
        shard_col = "part_id"
        offset_col = "offset_in_part"
    elif {"shard_id", "offset_in_shard"}.issubset(df.columns):
        shard_col = "shard_id"
        offset_col = "offset_in_shard"
    else:
        raise ValueError("selection stats must include part_id/offset_in_part")

    required = {"feature_id", shard_col, offset_col, "r2y", "n_y_overlap"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"selection stats missing required columns: {sorted(missing)}")
    df = df.filter((pl.col("n_y_overlap") >= min_non_null_y) & (pl.col("r2y") >= y_r2_threshold)).sort(
        by=["r2y", "feature_id"], descending=[True, False]
    )
    if max_candidates and df.height > max_candidates:
        df = df.head(max_candidates)
    if df.height == 0:
        return []
    fids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    shard_ids = df[shard_col].to_numpy().astype(np.int32, copy=False)
    offsets = df[offset_col].to_numpy().astype(np.int32, copy=False)
    r2y = df["r2y"].to_numpy().astype(np.float64, copy=False)
    n_valid = df["n_y_overlap"].to_numpy().astype(np.int32, copy=False)
    return [
        Candidate(int(fids[i]), int(shard_ids[i]), int(offsets[i]), float(r2y[i]), int(n_valid[i]))
        for i in range(df.height)
    ]


def build_candidates_from_inmemory(shards, y, y_mask, min_non_null_y, y_r2_threshold, max_candidates=0):
    """Build candidates from in-memory feature rows for tests and validation."""

    candidates = []
    for shard_id, shard in enumerate(shards):
        z = shard["values"]
        mz = shard["valid"]
        r2, n = batch_r2_one_vs_many(y, y_mask, z, mz, min_non_null=min_non_null_y, sanitize=True)
        fids = shard["feature_id"]
        for i in range(z.shape[0]):
            if n[i] >= min_non_null_y and r2[i] >= y_r2_threshold:
                candidates.append(Candidate(int(fids[i]), shard_id, i, float(r2[i]), int(n[i])))
    candidates.sort(key=lambda c: c.r2_y, reverse=True)
    if max_candidates and len(candidates) > max_candidates:
        return candidates[:max_candidates]
    return candidates
