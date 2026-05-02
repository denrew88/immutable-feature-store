import numpy as np

from .pearson import pairwise_r2, batch_r2_one_vs_many
from .incremental import select_features_incremental
from .candidates import build_candidates_from_inmemory
from ..scalar.parquet_storage import InMemoryShardReader


def build_inmemory_shards(X, M, shard_size=128):
    """Split dense arrays into in-memory shard dictionaries for tests."""
    n_features = X.shape[0]
    shards = []
    for start in range(0, n_features, shard_size):
        end = min(start + shard_size, n_features)
        shards.append({
            "feature_id": np.arange(start, end, dtype=np.int32),
            "values": X[start:end].astype(np.float64),
            "valid": M[start:end].astype(np.uint8),
        })
    return shards


def naive_greedy_full(X, M, y, y_mask, min_non_null_y, y_r2_threshold, min_non_null_pair, ff_r2_threshold, top_m):
    """Reference greedy selector that evaluates all pairwise redundancies directly."""
    n_features = X.shape[0]
    candidates = []
    for i in range(n_features):
        r2, n = pairwise_r2(X[i], M[i], y, y_mask, min_non_null=min_non_null_y)
        if n >= min_non_null_y and r2 >= y_r2_threshold:
            candidates.append((i, r2))
    candidates.sort(key=lambda t: t[1], reverse=True)

    alive = {i for i, _ in candidates}
    selected = []
    for i, _ in candidates:
        if i not in alive:
            continue
        selected.append(i)
        if len(selected) >= top_m:
            break
        for j, _ in candidates:
            if j <= i or j not in alive:
                continue
            r2, n = pairwise_r2(X[i], M[i], X[j], M[j], min_non_null=min_non_null_pair)
            if n >= min_non_null_pair and r2 >= ff_r2_threshold:
                alive.remove(j)
    return selected


def validate_batch_kernel(X, M, y, y_mask, min_non_null=2, tol=1e-6):
    """Check the batched one-vs-many kernel against the scalar reference."""
    r2_batch, n_batch = batch_r2_one_vs_many(y, y_mask, X, M, min_non_null=min_non_null, sanitize=True)
    for i in range(X.shape[0]):
        r2, n = pairwise_r2(y, y_mask, X[i], M[i], min_non_null=min_non_null)
        if n != n_batch[i]:
            raise AssertionError("n_valid mismatch")
        if abs(r2 - r2_batch[i]) > tol:
            raise AssertionError("r2 mismatch")
    return True


def validate_incremental_vs_naive(X, M, y, y_mask, config, shard_size=128):
    """Verify incremental selection matches the naive reference implementation."""
    shards = build_inmemory_shards(X, M, shard_size=shard_size)
    reader = InMemoryShardReader(shards)
    candidates = build_candidates_from_inmemory(shards, y, y_mask, config.min_non_null_y, config.y_r2_threshold)
    selected_opt = select_features_incremental(candidates, reader, config)
    selected_opt_ids = [c.feature_id for c in selected_opt]

    selected_naive = naive_greedy_full(
        X, M, y, y_mask,
        config.min_non_null_y, config.y_r2_threshold,
        config.min_non_null_pair, config.ff_r2_threshold,
        config.top_m,
    )
    if selected_naive != selected_opt_ids:
        raise AssertionError("optimized selection does not match naive")
    return True
