from typing import Dict, List

import numpy as np

from .pearson import batch_r2_many_vs_many, batch_r2_one_vs_many


def grow_cap(current_cap: int, max_step: int) -> int:
    """Grow the active candidate cap with a bounded step size."""
    return current_cap + min(current_cap, max_step)


def _rebucket_candidates(candidates, start: int, end: int) -> Dict[int, List[int]]:
    """Group candidate indices by shard for efficient staged reads."""
    buckets: Dict[int, List[int]] = {}
    for idx in range(start, end):
        c = candidates[idx]
        buckets.setdefault(c.shard_id, []).append(idx)
    for shard_id in buckets:
        buckets[shard_id].sort(key=lambda i: candidates[i].offset_in_shard)
    return buckets


def _iter_batches(candidates, idxs: List[int], batch_size: int):
    """Yield shard-local candidate batches and their row offsets."""
    for i in range(0, len(idxs), batch_size):
        batch = idxs[i : i + batch_size]
        offsets = [candidates[j].offset_in_shard for j in batch]
        yield batch, offsets


def _pack_valid_rows(valid):
    """Pack one or more validity rows into uint64 bit masks."""
    valid = np.asarray(valid, dtype=np.uint8)
    squeeze = False
    if valid.ndim == 1:
        valid = valid.reshape(1, -1)
        squeeze = True
    packed = np.packbits(valid, axis=1, bitorder="little")
    pad = (-packed.shape[1]) % 8
    if pad:
        packed = np.pad(packed, ((0, 0), (0, pad)), mode="constant")
    packed = np.ascontiguousarray(packed)
    words = packed.view(np.uint64)
    return words[0] if squeeze else words


def _pack_bool_rows(mask):
    """Pack a 2D boolean mask into uint64 words per row."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    packed = np.packbits(mask, axis=1, bitorder="little")
    pad = (-packed.shape[1]) % 8
    if pad:
        packed = np.pad(packed, ((0, 0), (0, pad)), mode="constant")
    packed = np.ascontiguousarray(packed)
    return packed.view(np.uint64)


def _overlap_counts_one_to_many(valid_words, batch_valid_words):
    """Count valid overlaps between one feature and a feature batch."""
    return np.bitwise_count(np.bitwise_and(batch_valid_words, valid_words[None, :])).sum(axis=1, dtype=np.int32)


def _overlap_counts_many_to_many(valid_words_left, valid_words_right):
    """Count valid overlaps for every pair across two feature tiles."""
    overlap_words = np.bitwise_and(valid_words_left[:, None, :], valid_words_right[None, :, :])
    return np.bitwise_count(overlap_words).sum(axis=2, dtype=np.int32)


def _build_stage_cache(candidates, reader, start: int, end: int):
    """Load values and packed masks for one active candidate stage."""
    buckets = _rebucket_candidates(candidates, start, end)
    stage = {}
    feature_cache = {}
    for shard_id, idxs in buckets.items():
        offsets = [candidates[j].offset_in_shard for j in idxs]
        values, valid = reader.load_rows(shard_id, offsets)
        valid_words = _pack_valid_rows(valid)
        idxs_arr = np.asarray(idxs, dtype=np.int32)
        stage[shard_id] = {
            "idxs": idxs_arr,
            "values": values,
            "valid": valid,
            "valid_words": valid_words,
        }
        for pos, cand_idx in enumerate(idxs):
            feature_cache[cand_idx] = (values[pos], valid[pos], valid_words[pos])
    return stage, feature_cache


def _stack_feature_tile(feature_cache, global_idxs):
    """Stack cached feature rows into dense tile arrays."""
    x_rows = []
    mx_rows = []
    word_rows = []
    for global_idx in global_idxs:
        x, mx, words = feature_cache[int(global_idx)]
        x_rows.append(x)
        mx_rows.append(mx)
        word_rows.append(words)
    return (
        np.stack(x_rows, axis=0),
        np.stack(mx_rows, axis=0),
        np.stack(word_rows, axis=0),
    )


def _build_mask_groups(active_global_idxs, feature_cache):
    """Group active features by identical validity-mask bit patterns."""
    grouped = {}
    for local_idx, global_idx in enumerate(active_global_idxs):
        _, valid, words = feature_cache[int(global_idx)]
        key = np.ascontiguousarray(words).tobytes()
        entry = grouped.get(key)
        if entry is None:
            entry = {
                "local_idxs": [],
                "valid": valid.astype(bool, copy=False),
                "words": np.asarray(words, dtype=np.uint64),
            }
            grouped[key] = entry
        entry["local_idxs"].append(local_idx)

    groups = []
    group_ids = np.empty(active_global_idxs.shape[0], dtype=np.int32)
    for gid, entry in enumerate(grouped.values()):
        local_idxs = np.asarray(entry["local_idxs"], dtype=np.int32)
        group_ids[local_idxs] = gid
        groups.append({
            "gid": gid,
            "local_idxs": local_idxs,
            "valid": entry["valid"],
            "words": entry["words"],
        })
    return groups, group_ids


def _stack_local_values(feature_cache, active_global_idxs, local_idxs):
    """Stack feature values for local adjacency indices."""
    rows = []
    for local_idx in local_idxs:
        x, _, _ = feature_cache[int(active_global_idxs[int(local_idx)])]
        rows.append(x)
    return np.stack(rows, axis=0)


def _same_mask_tile_stats(values, n_valid):
    """Compute row sums and centered variances for a same-mask tile."""
    sums = values.sum(axis=1)
    sums2 = (values * values).sum(axis=1)
    variances = sums2 - (sums * sums) / n_valid
    return sums, variances


def _get_group_values(group_values_cache, feature_cache, active_global_idxs, group):
    """Memoize stacked values for a mask group."""
    gid = int(group["gid"])
    values = group_values_cache.get(gid)
    if values is None:
        values = _stack_local_values(feature_cache, active_global_idxs, group["local_idxs"])
        group_values_cache[gid] = values
    return values


def _or_edge_rows(adjacency, active_n: int, row_local, col_local, edge_mask):
    """OR a boolean edge mask into both directions of the adjacency matrix."""
    if not np.any(edge_mask):
        return
    edge_rows = np.zeros((row_local.shape[0], active_n), dtype=bool)
    edge_rows[:, col_local] = edge_mask
    adjacency[row_local, :] = adjacency[row_local, :] | _pack_bool_rows(edge_rows)


def _process_mask_group_pair(
    adjacency,
    active_global_idxs,
    feature_cache,
    group_values_cache,
    left_group,
    right_group,
    config,
    tile_size,
):
    """Populate adjacency edges for one pair of validity-mask groups."""
    overlap_mask = left_group["valid"] & right_group["valid"]
    n_valid = int(overlap_mask.sum())
    if n_valid < config.min_non_null_pair:
        return False

    active_n = int(active_global_idxs.shape[0])
    threshold = float(config.ff_r2_threshold)
    left_local_all = left_group["local_idxs"]
    right_local_all = right_group["local_idxs"]
    left_values_all = _get_group_values(group_values_cache, feature_cache, active_global_idxs, left_group)
    if left_group["gid"] == right_group["gid"]:
        right_values_all = left_values_all
    else:
        right_values_all = _get_group_values(group_values_cache, feature_cache, active_global_idxs, right_group)

    use_overlap_slice = n_valid != overlap_mask.shape[0]
    left_range = range(0, left_local_all.shape[0], tile_size)
    for left_start in left_range:
        left_end = min(left_start + tile_size, left_local_all.shape[0])
        left_local = left_local_all[left_start:left_end]
        X = left_values_all[left_start:left_end]
        if use_overlap_slice:
            X = X[:, overlap_mask]
        sum_x, var_x = _same_mask_tile_stats(X, n_valid)

        if left_group["gid"] == right_group["gid"]:
            right_range = range(left_start, right_local_all.shape[0], tile_size)
        else:
            right_range = range(0, right_local_all.shape[0], tile_size)

        for right_start in right_range:
            right_end = min(right_start + tile_size, right_local_all.shape[0])
            right_local = right_local_all[right_start:right_end]
            Z = right_values_all[right_start:right_end]
            if use_overlap_slice:
                Z = Z[:, overlap_mask]
            sum_z, var_z = _same_mask_tile_stats(Z, n_valid)

            cov = X @ Z.T
            cov -= (sum_x[:, None] * sum_z[None, :]) / n_valid
            denom = var_x[:, None] * var_z[None, :]
            edge_mask = (denom > 0.0) & ((cov * cov) >= (threshold * denom))
            if not np.any(edge_mask):
                continue

            if left_group["gid"] == right_group["gid"]:
                order_mask = left_local[:, None] < right_local[None, :]
                _or_edge_rows(adjacency, active_n, left_local, right_local, edge_mask & order_mask)
                continue

            order_ab = left_local[:, None] < right_local[None, :]
            _or_edge_rows(adjacency, active_n, left_local, right_local, edge_mask & order_ab)
            order_ba = right_local[:, None] < left_local[None, :]
            _or_edge_rows(adjacency, active_n, right_local, left_local, edge_mask.T & order_ba)
    return True


def _build_stage_adjacency(active_global_idxs, feature_cache, config):
    """Build the feature-feature adjacency matrix for one active stage."""
    active_n = int(active_global_idxs.shape[0])
    word_count = (active_n + 63) // 64
    adjacency = np.zeros((active_n, word_count), dtype=np.uint64)
    if active_n <= 1:
        return adjacency

    tile_size = min(max(64, ((config.batch_size + 63) // 64) * 64), 256)
    mask_groups, group_ids = _build_mask_groups(active_global_idxs, feature_cache)
    group_values_cache = {}
    group_sizes = np.asarray([group["local_idxs"].shape[0] for group in mask_groups], dtype=np.int32)
    processed_pairs = np.zeros((len(mask_groups), len(mask_groups)), dtype=bool)

    same_group_ids = np.flatnonzero(group_sizes >= config.mask_fastpath_min_group)
    for gid in same_group_ids:
        if _process_mask_group_pair(
            adjacency,
            active_global_idxs,
            feature_cache,
            group_values_cache,
            mask_groups[int(gid)],
            mask_groups[int(gid)],
            config,
            tile_size,
        ):
            processed_pairs[int(gid), int(gid)] = True

    if group_sizes.size:
        max_group_size = int(group_sizes.max())
        cross_group_ids = np.flatnonzero(group_sizes * max_group_size >= config.mask_fastpath_min_pairs)
        for pos, left_gid in enumerate(cross_group_ids):
            left_gid = int(left_gid)
            for right_gid in cross_group_ids[pos + 1 :]:
                right_gid = int(right_gid)
                if group_sizes[left_gid] * group_sizes[right_gid] < config.mask_fastpath_min_pairs:
                    continue
                if _process_mask_group_pair(
                    adjacency,
                    active_global_idxs,
                    feature_cache,
                    group_values_cache,
                    mask_groups[left_gid],
                    mask_groups[right_gid],
                    config,
                    tile_size,
                ):
                    processed_pairs[left_gid, right_gid] = True
                    processed_pairs[right_gid, left_gid] = True

    for left_start in range(0, active_n, tile_size):
        left_end = min(left_start + tile_size, active_n)
        left_global = active_global_idxs[left_start:left_end]
        X, MX, X_words = _stack_feature_tile(feature_cache, left_global)
        left_group_slice = group_ids[left_start:left_end]

        for right_start in range(left_start, active_n, tile_size):
            right_end = min(right_start + tile_size, active_n)
            right_group_slice = group_ids[right_start:right_end]
            right_global = active_global_idxs[right_start:right_end]
            Z, MZ, Z_words = _stack_feature_tile(feature_cache, right_global)

            overlap = _overlap_counts_many_to_many(X_words, Z_words)
            keep_mask = overlap >= config.min_non_null_pair
            keep_mask &= ~processed_pairs[left_group_slice[:, None], right_group_slice[None, :]]
            if right_start == left_start:
                tri_mask = np.triu(np.ones_like(keep_mask, dtype=bool), k=1)
                keep_mask &= tri_mask
            if not np.any(keep_mask):
                continue

            r2, n_valid = batch_r2_many_vs_many(
                X,
                MX,
                Z,
                MZ,
                min_non_null=config.min_non_null_pair,
                sanitize=True,
                precomputed_n=overlap,
            )
            edge_mask = keep_mask & (n_valid >= config.min_non_null_pair) & (r2 >= config.ff_r2_threshold)
            if not np.any(edge_mask):
                continue

            packed_edges = _pack_bool_rows(edge_mask)
            word_start = right_start // 64
            word_span = packed_edges.shape[1]
            adjacency[left_start:left_end, word_start : word_start + word_span] |= packed_edges

    return adjacency


def _bit_is_set(words, bit_index: int) -> bool:
    """Return whether a packed-bit vector contains a set bit."""
    return bool((words[bit_index >> 6] >> (bit_index & 63)) & np.uint64(1))


def _clear_bit(words, bit_index: int):
    """Clear one bit in a packed-bit vector in place."""
    words[bit_index >> 6] &= ~np.uint64(1 << (bit_index & 63))


def select_features_incremental(candidates, reader, config):
    """Run staged greedy feature selection with batched redundancy pruning."""
    if config.max_candidates and len(candidates) > config.max_candidates:
        candidates = candidates[: config.max_candidates]
    n = len(candidates)
    alive = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    selected_idx: List[int] = []
    selected_cache = {}
    selected_tile_size = 16

    cap_end = min(config.initial_cap, n)
    processed = 0

    while len(selected_idx) < config.top_m and processed < n:
        stage, stage_feature_cache = _build_stage_cache(candidates, reader, processed, cap_end)

        # prune new range by already selected
        for tile_start in range(0, len(selected_idx), selected_tile_size):
            tile = selected_idx[tile_start : tile_start + selected_tile_size]
            if not tile:
                continue
            X_rows = []
            MX_rows = []
            X_word_rows = []
            for s_idx in tile:
                if s_idx in selected_cache:
                    x, mx, x_words = selected_cache[s_idx]
                else:
                    x, mx = reader.load_feature_by_offset(
                        candidates[s_idx].shard_id,
                        candidates[s_idx].offset_in_shard,
                    )
                    x_words = _pack_valid_rows(mx)
                    selected_cache[s_idx] = (x, mx, x_words)
                X_rows.append(x)
                MX_rows.append(mx)
                X_word_rows.append(x_words)
            X = np.stack(X_rows, axis=0)
            MX = np.stack(MX_rows, axis=0)
            X_words = np.stack(X_word_rows, axis=0)
            for bucket in stage.values():
                active_pos = np.flatnonzero(alive[bucket["idxs"]])
                if active_pos.size == 0:
                    continue
                for pos_start in range(0, active_pos.size, config.batch_size):
                    pos_batch = active_pos[pos_start : pos_start + config.batch_size]
                    overlap = _overlap_counts_many_to_many(X_words, bucket["valid_words"][pos_batch])
                    keep = np.any(overlap >= config.min_non_null_pair, axis=0)
                    if not np.any(keep):
                        continue
                    pos_keep = pos_batch[keep]
                    Z = bucket["values"][pos_keep]
                    Mz = bucket["valid"][pos_keep]
                    r2, n_valid = batch_r2_many_vs_many(
                        X,
                        MX,
                        Z,
                        Mz,
                        min_non_null=config.min_non_null_pair,
                        sanitize=True,
                        precomputed_n=overlap[:, keep],
                    )
                    kill = np.any(
                        (n_valid >= config.min_non_null_pair) & (r2 >= config.ff_r2_threshold),
                        axis=0,
                    )
                    if np.any(kill):
                        alive[bucket["idxs"][pos_keep[kill]]] = False

        stage_global_idxs = np.flatnonzero(alive[processed:cap_end]) + processed
        if stage_global_idxs.size > 0:
            adjacency = _build_stage_adjacency(stage_global_idxs, stage_feature_cache, config)
            alive_words = np.zeros(((stage_global_idxs.size + 63) // 64,), dtype=np.uint64)
            for local_idx in range(stage_global_idxs.size):
                alive_words[local_idx >> 6] |= np.uint64(1 << (local_idx & 63))

            for local_idx, global_idx in enumerate(stage_global_idxs):
                if not _bit_is_set(alive_words, local_idx):
                    continue
                global_idx = int(global_idx)
                selected[global_idx] = True
                selected_idx.append(global_idx)
                if global_idx not in selected_cache:
                    x, mx, x_words = stage_feature_cache.get(global_idx, (None, None, None))
                    if x is None:
                        x, mx = reader.load_feature_by_offset(
                            candidates[global_idx].shard_id,
                            candidates[global_idx].offset_in_shard,
                        )
                        x_words = _pack_valid_rows(mx)
                    selected_cache[global_idx] = (x, mx, x_words)
                _clear_bit(alive_words, local_idx)
                alive_words &= ~adjacency[local_idx]
                if len(selected_idx) >= config.top_m:
                    break

            dead_local = []
            for local_idx in range(stage_global_idxs.size):
                if not _bit_is_set(alive_words, local_idx):
                    dead_local.append(local_idx)
            if dead_local:
                alive[stage_global_idxs[np.asarray(dead_local, dtype=np.int32)]] = False
            if len(selected_idx) >= config.top_m:
                break

        if cap_end == n:
            break
        processed = cap_end
        cap_end = min(grow_cap(cap_end, config.max_step), n)

    return [candidates[i] for i in selected_idx]
