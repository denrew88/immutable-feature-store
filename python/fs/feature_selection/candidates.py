import numpy as np
import polars as pl

from .pearson import batch_r2_one_vs_many
from ..types import Candidate
from ..scalar.parquet_storage import _decode_values_blob, _decode_valid_blob


def build_candidates_from_shards(
    shard_paths,
    y,
    y_mask,
    min_non_null_y,
    y_r2_threshold,
    max_candidates=0,
    batch_size=512,
    value_len_col="value_len",
    values_blob_col="values_blob",
    valid_blob_col="valid_blob",
    feature_id_col="feature_id",
):
    """Scan scalar shards and rank feature candidates by r^2 against y."""
    candidates = []

    for shard_id, path in enumerate(shard_paths):
        scan = pl.scan_parquet(path)
        n_rows = int(scan.select(pl.len()).collect().item())
        for start in range(0, n_rows, batch_size):
            df = scan.slice(start, batch_size).collect()
            rows = df.height
            if rows == 0:
                continue
            value_lens = df[value_len_col].to_numpy()
            values_blob = df[values_blob_col].to_list()
            valid_blob = df[valid_blob_col].to_list()
            Z = np.empty((rows, y.shape[0]), dtype=np.float64)
            Mz = np.empty((rows, y.shape[0]), dtype=np.uint8)
            for i in range(rows):
                value_len = int(value_lens[i])
                if value_len != y.shape[0]:
                    raise ValueError(f"value_len mismatch: {value_len} != {y.shape[0]}")
                Z[i] = _decode_values_blob(values_blob[i], value_len)
                Mz[i] = _decode_valid_blob(valid_blob[i], value_len)
            r2, n = batch_r2_one_vs_many(y, y_mask, Z, Mz, min_non_null=min_non_null_y, sanitize=True)

            fids = df[feature_id_col].to_numpy()
            for i in range(Z.shape[0]):
                if n[i] >= min_non_null_y and r2[i] >= y_r2_threshold:
                    candidates.append(Candidate(int(fids[i]), shard_id, start + i, float(r2[i]), int(n[i])))

    candidates.sort(key=lambda c: c.r2_y, reverse=True)
    if max_candidates and len(candidates) > max_candidates:
        return candidates[:max_candidates]
    return candidates


def build_candidates_from_locator(locator_path, min_non_null_y, y_r2_threshold, max_candidates=0):
    """Build candidates directly from precomputed locator statistics."""
    return build_candidates_from_stats(locator_path, min_non_null_y, y_r2_threshold, max_candidates=max_candidates)


def build_candidates_from_stats(stats_path, min_non_null_y, y_r2_threshold, max_candidates=0):
    """Build candidates directly from one precomputed selection-stats sidecar.

    Args:
        stats_path: Path to a parquet file containing `feature_id`, `shard_id`,
            `offset_in_shard`, `r2y`, and `n_y_overlap`.
        min_non_null_y: Minimum overlap with y required for candidacy.
        y_r2_threshold: Minimum feature-vs-y r^2 required for candidacy.
        max_candidates: Optional hard cap on the number of candidates returned.

    Returns:
        A list of ranked `Candidate` objects sorted by descending `r2y`.
    """
    df = pl.read_parquet(stats_path)
    required = {"feature_id", "shard_id", "offset_in_shard", "r2y", "n_y_overlap"}
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
    shard_ids = df["shard_id"].to_numpy().astype(np.int32, copy=False)
    offsets = df["offset_in_shard"].to_numpy().astype(np.int32, copy=False)
    r2y = df["r2y"].to_numpy().astype(np.float64, copy=False)
    n_valid = df["n_y_overlap"].to_numpy().astype(np.int32, copy=False)
    return [
        Candidate(int(fids[i]), int(shard_ids[i]), int(offsets[i]), float(r2y[i]), int(n_valid[i]))
        for i in range(df.height)
    ]


def build_candidates_from_inmemory(shards, y, y_mask, min_non_null_y, y_r2_threshold, max_candidates=0):
    """Build candidates from in-memory shards for tests and benchmarks."""
    candidates = []
    for shard_id, shard in enumerate(shards):
        Z = shard["values"]
        Mz = shard["valid"]
        r2, n = batch_r2_one_vs_many(y, y_mask, Z, Mz, min_non_null=min_non_null_y, sanitize=True)
        fids = shard["feature_id"]
        for i in range(Z.shape[0]):
            if n[i] >= min_non_null_y and r2[i] >= y_r2_threshold:
                candidates.append(Candidate(int(fids[i]), shard_id, i, float(r2[i]), int(n[i])))
    candidates.sort(key=lambda c: c.r2_y, reverse=True)
    if max_candidates and len(candidates) > max_candidates:
        return candidates[:max_candidates]
    return candidates
