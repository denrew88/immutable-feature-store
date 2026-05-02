import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import polars as pl

from ..feature_selection.pearson import batch_r2_one_vs_many


def load_sample_meta(
    sample_meta_path: str,
    y_col: str = "y",
    sample_id_col: str = "sample_id",
    path_col: str = "sample_path",
):
    """Load dense sample ids, target values, validity mask, and sample paths.

    The scalar v2 convention defines `sample_id` as the row index of
    `sample_meta.parquet`. When a `sample_id` column is present it must match
    that dense zero-based ordering.

    Args:
        sample_meta_path: Path to `sample_meta.parquet`.
        y_col: Target-value column name.
        sample_id_col: Optional dense id column to validate.
        path_col: Column containing per-sample parquet paths.

    Returns:
        A tuple `(sample_ids, y, y_mask, sample_paths)` where `sample_ids`
        always equals `range(n_samples)`.
    """
    sample_ids, y, y_mask = load_sample_targets(
        sample_meta_path,
        y_col=y_col,
        sample_id_col=sample_id_col,
    )
    df = pl.read_parquet(sample_meta_path)
    if path_col not in df.columns:
        raise ValueError("sample_meta parquet must have y and sample_path columns")
    sample_paths = []
    meta_dir = os.path.dirname(os.path.abspath(sample_meta_path))
    for raw_path in df[path_col].to_list():
        path_value = str(raw_path)
        if os.path.isabs(path_value):
            sample_paths.append(path_value)
        else:
            sample_paths.append(os.path.normpath(os.path.join(meta_dir, path_value)))
    return sample_ids, y, y_mask, sample_paths


def load_sample_targets(
    sample_meta_path: str,
    y_col: str = "y",
    sample_id_col: str = "sample_id",
):
    """Load dense sample ids, target values, and validity mask without paths.

    Args:
        sample_meta_path: Path to `sample_meta.parquet`.
        y_col: Target-value column name.
        sample_id_col: Optional dense id column to validate.

    Returns:
        A tuple `(sample_ids, y, y_mask)` where `sample_ids` always equals
        `range(n_samples)`.
    """
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
    """Load dense feature ids from `feature_meta.parquet`.

    The scalar v2 convention defines `feature_id` as the row index of
    `feature_meta.parquet`. When a `feature_id` column is present it must match
    that dense zero-based ordering.

    Args:
        feature_meta_path: Path to `feature_meta.parquet`.
        feature_id_col: Optional dense id column to validate.

    Returns:
        A tuple `(feature_ids, feature_meta_df)` where `feature_ids` is a dense
        `int32` array equal to `0..n_features-1`.
    """
    df = pl.read_parquet(feature_meta_path)
    feature_ids = np.arange(df.height, dtype=np.int32)
    if feature_id_col in df.columns:
        stored_ids = df[feature_id_col].to_numpy().astype(np.int32, copy=False)
        if not np.array_equal(stored_ids, feature_ids):
            raise ValueError(f"feature_meta {feature_id_col} must equal dense row order 0..n-1")
    return feature_ids, df


def _collect_feature_ids(sample_paths: List[str], feature_id_col: str) -> np.ndarray:
    """Collect the sorted set of feature ids from sample-major parquet files."""
    feature_ids = set()
    for path in sample_paths:
        df = pl.read_parquet(path, columns=[feature_id_col])
        feature_ids.update(int(v) for v in df[feature_id_col].to_list())
    if not feature_ids:
        return np.array([], dtype=np.int32)
    lo = min(feature_ids)
    hi = max(feature_ids)
    if lo < np.iinfo(np.int32).min or hi > np.iinfo(np.int32).max:
        raise ValueError(f"feature_id out of int32 range: min={lo}, max={hi}")
    return np.array(sorted(feature_ids), dtype=np.int32)


def _assign_shards(n_features: int, n_shards: int):
    """Compute equal-count shard boundaries for scalar features."""
    shard_size = (n_features + n_shards - 1) // n_shards
    shard_starts = [i * shard_size for i in range(n_shards)]
    shard_ends = [min((i + 1) * shard_size, n_features) for i in range(n_shards)]
    return shard_size, shard_starts, shard_ends


def _estimate_scalar_feature_bytes(n_samples: int) -> int:
    """Estimate bytes required to store one scalar feature row.

    Args:
        n_samples: Number of dense samples stored per feature.

    Returns:
        Approximate bytes for one feature row, including blob payloads and a
        small fixed overhead for parquet row metadata.
    """
    return int(n_samples) * (8 + 1) + 64


def _assign_shards_by_target_bytes(
    n_features: int,
    n_samples: int,
    target_shard_bytes: int,
    n_shards_override: Optional[int] = None,
):
    """Compute scalar shard boundaries from a target shard size.

    Args:
        n_features: Total number of dense features.
        n_samples: Total number of dense samples.
        target_shard_bytes: Desired shard size target in bytes.
        n_shards_override: Optional explicit shard-count override.

    Returns:
        A tuple `(n_shards, shard_size, shard_starts, shard_ends)`.
    """
    if n_features <= 0:
        return 1, 1, [0], [0]
    if n_shards_override is not None:
        n_shards = int(n_shards_override)
        shard_size, shard_starts, shard_ends = _assign_shards(n_features, n_shards)
        return n_shards, shard_size, shard_starts, shard_ends
    if target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be > 0")
    feature_bytes = max(1, _estimate_scalar_feature_bytes(n_samples))
    shard_size = max(1, int(target_shard_bytes) // feature_bytes)
    n_shards = max(1, (n_features + shard_size - 1) // shard_size)
    shard_starts = [shard_id * shard_size for shard_id in range(n_shards)]
    shard_ends = [min((shard_id + 1) * shard_size, n_features) for shard_id in range(n_shards)]
    return n_shards, shard_size, shard_starts, shard_ends


def _close_memmap(mm):
    """Close one NumPy memmap backing handle if it is still open.

    Args:
        mm: Memmap instance or `None`.
    """
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


def _cleanup_backing_file(path: Optional[str]):
    """Delete one temporary backing file if it exists.

    Args:
        path: Path to a temporary memmap backing file.
    """
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _cleanup_empty_dir(path: Optional[str]):
    """Remove one directory when it exists and is empty.

    Args:
        path: Directory path to remove.
    """
    if not path:
        return
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass


def _resolve_manifest_relative(manifest_path: str, value: str) -> str:
    """Resolve one manifest path relative to the manifest location.

    Args:
        manifest_path: Path to `shard_manifest.json`.
        value: Stored manifest path, absolute or relative.

    Returns:
        Absolute path resolved against the manifest directory when needed.
    """
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(manifest_path), value))


def _decode_values_blob(values_blob: bytes, value_len: int) -> np.ndarray:
    """Decode a float64 values blob with strict length validation."""
    if values_blob is None:
        raise ValueError("values_blob is None")
    expected = value_len * 8
    if len(values_blob) != expected:
        raise ValueError(f"invalid values_blob length: got={len(values_blob)} expected={expected}")
    return np.frombuffer(values_blob, dtype="<f8", count=value_len).astype(np.float64, copy=True)


def _decode_valid_blob(valid_blob: bytes, value_len: int) -> np.ndarray:
    """Decode a uint8 validity blob with strict length validation."""
    if valid_blob is None:
        raise ValueError("valid_blob is None")
    if len(valid_blob) != value_len:
        raise ValueError(f"invalid valid_blob length: got={len(valid_blob)} expected={value_len}")
    return np.frombuffer(valid_blob, dtype=np.uint8, count=value_len).astype(np.uint8, copy=True)


def validate_dense_sample_ids(
    sample_meta_path: str,
    sample_id_col: str = "sample_id",
    sample_row_col: str = "sample_row",
):
    """Validate that sample ids already equal dense row-order ids.

    Args:
        sample_meta_path: Path to the sample metadata parquet file.
        sample_id_col: Column containing dense sample ids when present.
        sample_row_col: Optional legacy column containing dense row indices.

    Returns:
        The number of dense sample ids validated in the metadata file.
    """
    df = pl.read_parquet(sample_meta_path)
    sample_rows = np.arange(df.height, dtype=np.int64)
    if sample_id_col in df.columns:
        sample_ids = df[sample_id_col].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(sample_ids, sample_rows):
            raise ValueError(f"sample_meta {sample_id_col} must equal dense row order 0..n-1")
    elif sample_row_col in df.columns:
        sample_ids = df[sample_row_col].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(sample_ids, sample_rows):
            raise ValueError(f"sample_meta {sample_row_col} must equal dense row order 0..n-1")
    return int(df.height)


def build_sample_id_index(sample_meta_path: str, sample_id_col: str = "sample_id", sample_row_col: str = "sample_row"):
    """Build a legacy identity mapping for dense sample ids.

    Args:
        sample_meta_path: Path to the sample metadata parquet file.
        sample_id_col: Column containing dense sample ids when present.
        sample_row_col: Optional legacy column containing dense row indices.

    Returns:
        A dictionary whose keys and values are identical dense sample ids.

    Notes:
        New dense-id code paths should call `validate_dense_sample_ids(...)`
        and use `sample_id` directly. This wrapper remains only for callers
        that still expect a mapping object.
    """
    n_samples = validate_dense_sample_ids(
        sample_meta_path,
        sample_id_col=sample_id_col,
        sample_row_col=sample_row_col,
    )
    return {sample_id: sample_id for sample_id in range(n_samples)}


def build_feature_locator_index(locator_path: str):
    """Build a feature locator dictionary from locator parquet rows.

    Args:
        locator_path: Path to the feature locator parquet file.

    Returns:
        A dictionary mapping feature id to `(shard_id, offset_in_shard)`.
    """
    df = pl.read_parquet(locator_path)
    fids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    shard_ids = df["shard_id"].to_numpy().astype(np.int32, copy=False)
    offsets = df["offset_in_shard"].to_numpy().astype(np.int32, copy=False)
    return {int(fid): (int(sid), int(off)) for fid, sid, off in zip(fids, shard_ids, offsets)}


def shard_file_path(shard_path: str, shard_id: int) -> str:
    """Return the parquet path for a scalar shard.

    Args:
        shard_path: Directory containing shard parquet files.
        shard_id: Zero-based shard identifier.

    Returns:
        Absolute or relative path to the shard parquet file.
    """
    return os.path.join(shard_path, f"shard_{shard_id:04d}.parquet")


def list_shard_paths(manifest) -> List[str]:
    """List all scalar shard parquet files described by a manifest.

    Args:
        manifest: Loaded scalar shard manifest.

    Returns:
        A list of shard parquet paths ordered by shard id.
    """
    return [shard_file_path(manifest.shard_path, shard_id) for shard_id in range(manifest.n_shards)]


@dataclass
class ShardManifest:
    """Metadata describing a scalar shard set and its companion locator."""

    sample_meta_path: str
    feature_meta_path: str
    n_samples: int
    n_features: int
    shard_path: str
    n_shards: int
    feature_locator_path: str
    feature_locator_format: str
    feature_id_dtype: str
    values_dtype: str
    valid_dtype: str
    id_scheme: str = "dense_row_ids"
    sample_key_col: str = "sample_key"
    feature_key_col: str = "feature_key"
    target_shard_bytes: Optional[int] = None
    selection_stats: Optional[Dict[str, str]] = None
    stats_y_col: Optional[str] = None

    def to_json(self):
        """Convert the manifest into a JSON-serializable dictionary.

        Returns:
            A dictionary ready to be written as `shard_manifest.json`.
        """
        data = {
            "sample_meta_path": self.sample_meta_path,
            "feature_meta_path": self.feature_meta_path,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "shard_path": self.shard_path,
            "n_shards": self.n_shards,
            "feature_locator_path": self.feature_locator_path,
            "feature_locator_format": self.feature_locator_format,
            "feature_id_dtype": self.feature_id_dtype,
            "values_dtype": self.values_dtype,
            "valid_dtype": self.valid_dtype,
            "id_scheme": self.id_scheme,
            "sample_key_col": self.sample_key_col,
            "feature_key_col": self.feature_key_col,
        }
        if self.target_shard_bytes is not None:
            data["target_shard_bytes"] = int(self.target_shard_bytes)
        if self.selection_stats:
            data["selection_stats"] = {str(key): str(value) for key, value in self.selection_stats.items()}
        return data


def locator_has_candidate_stats(locator_path: str) -> bool:
    """Check whether a locator already carries feature-vs-y candidate stats.

    Args:
        locator_path: Path to the feature locator parquet file.

    Returns:
        `True` when `r2y` and `n_y_overlap` columns are present, otherwise `False`.
    """
    df = pl.read_parquet(locator_path)
    return "r2y" in df.columns and "n_y_overlap" in df.columns


def _selection_stats_filename(y_col: str) -> str:
    """Encode one y-column name into a safe selection-stats filename."""
    return f"{quote(str(y_col), safe='')}.parquet"


def resolve_selection_stats_path(manifest: ShardManifest, y_col: str) -> Optional[str]:
    """Resolve one selection-stats sidecar path for a requested target column.

    Args:
        manifest: Loaded scalar shard manifest.
        y_col: Requested target column name.

    Returns:
        Absolute path to the matching selection-stats sidecar when present,
        otherwise `None`.
    """
    mapping = getattr(manifest, "selection_stats", None) or {}
    value = mapping.get(str(y_col))
    if not value:
        return None
    return str(value)


def _load_sample_bundle_manifest(manifest_path: str):
    """Load one sample-bundle stage manifest and resolve relative paths.

    Args:
        manifest_path: Path to `sample_major_manifest.json`.

    Returns:
        A dictionary with resolved absolute paths.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("format", "")) != "scalar-sample-bundles":
        raise ValueError(f"unsupported sample-major manifest format: {data.get('format')}")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    def _resolve(value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(manifest_dir, value))

    return {
        "sample_meta_path": _resolve(str(data["sample_meta_path"])),
        "feature_meta_path": _resolve(str(data["feature_meta_path"])),
        "bundle_paths": [_resolve(str(value)) for value in list(data.get("bundle_paths", []))],
        "sample_id_col": str(data.get("sample_id_col", "sample_id")),
        "feature_id_col": str(data.get("feature_id_col", "feature_id")),
        "value_col": str(data.get("value_col", "value")),
    }


def build_shards_from_sample_major(
    sample_meta_path: str,
    out_dir: str,
    feature_meta_path: str = None,
    n_shards: Optional[int] = None,
    target_shard_bytes: int = 256 * 1024 * 1024,
    feature_id_col: str = "feature_id",
    value_col: str = "value",
    sample_id_col: str = "sample_id",
    sample_key_col: str = "sample_key",
    feature_key_col: str = "feature_key",
    path_col: str = "sample_path",
    y_col: str = "y",
    stats_y_cols: Optional[List[str]] = None,
    values_dtype: str = "float64",
    valid_dtype: str = "uint8",
    tmp_dir: str = None,
    return_stats: bool = False,
):
    """Build feature-major scalar shards from sample-major parquet inputs.

    Args:
        sample_meta_path: Path to the sample metadata parquet file.
        out_dir: Output directory where shards, locator, and manifest are written.
        feature_meta_path: Path to the dense feature metadata parquet file. When
            omitted, defaults to `feature_meta.parquet` next to sample metadata.
        n_shards: Optional explicit shard-count override kept for legacy use.
        target_shard_bytes: Target parquet shard size in bytes. Used when
            `n_shards` is not provided.
        feature_id_col: Feature id column name in sample-major files.
        value_col: Scalar value column name in sample-major files.
        sample_id_col: Sample id column name in sample metadata.
        sample_key_col: External sample-key column name stored in sample metadata.
        feature_key_col: External feature-key column name stored in feature metadata.
        path_col: Sample parquet path column name in sample metadata.
        y_col: Legacy default target column name used when `stats_y_cols` is
            not provided.
        stats_y_cols: Target columns for precomputing feature-vs-y candidate
            stats. Each column is written as its own sidecar parquet file under
            `selection_stats/`.
        values_dtype: Expected encoded values dtype. Only float64 is supported.
        valid_dtype: Expected encoded validity dtype. Only uint8 is supported.
        tmp_dir: Optional directory for temporary memmap backing files.

    Returns:
        Path to the generated scalar shard manifest JSON file, or
        `(manifest_path, stats)` when `return_stats=True`.
    """
    total_t0 = time.perf_counter()
    stats = {
        "load_metadata_s": 0.0,
        "allocate_memmaps_s": 0.0,
        "fill_memmaps_s": 0.0,
        "compute_selection_stats_s": 0.0,
        "write_shards_s": 0.0,
        "cleanup_backing_files_s": 0.0,
        "copy_metadata_s": 0.0,
        "write_locator_s": 0.0,
        "write_selection_stats_s": 0.0,
        "write_manifest_s": 0.0,
        "total_s": 0.0,
    }
    if values_dtype not in ("float64", "double"):
        raise ValueError("values_dtype must be float64/double for BLOB shard format")
    if valid_dtype != "uint8":
        raise ValueError("valid_dtype must be uint8 for BLOB shard format")
    if n_shards is not None and int(n_shards) <= 0:
        raise ValueError("n_shards must be > 0")
    if int(target_shard_bytes) <= 0:
        raise ValueError("target_shard_bytes must be > 0")

    resolved_stats_y_cols: List[str] = []
    for value in (stats_y_cols or [y_col]):
        name = str(value)
        if name and name not in resolved_stats_y_cols:
            resolved_stats_y_cols.append(name)
    if not resolved_stats_y_cols:
        raise ValueError("at least one stats y column is required")

    os.makedirs(out_dir, exist_ok=True)
    shard_path = os.path.join(out_dir, "feature_shards")
    os.makedirs(shard_path, exist_ok=True)
    if tmp_dir is None:
        tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    if feature_meta_path is None:
        feature_meta_path = os.path.join(os.path.dirname(sample_meta_path), "feature_meta.parquet")

    phase_t0 = time.perf_counter()
    sample_ids, _, _, sample_paths = load_sample_meta(
        sample_meta_path,
        y_col=resolved_stats_y_cols[0],
        sample_id_col=sample_id_col,
        path_col=path_col,
    )
    n_samples = len(sample_ids)
    sample_meta_df = pl.read_parquet(sample_meta_path, columns=[sample_key_col] if sample_key_col else None)
    if sample_key_col not in sample_meta_df.columns:
        raise ValueError(f"sample_meta parquet must have key column: {sample_key_col}")
    sample_keys = sample_meta_df[sample_key_col]
    if sample_keys.null_count() != 0:
        raise ValueError(f"sample_meta {sample_key_col} must not contain nulls")
    if int(sample_keys.n_unique()) != int(sample_meta_df.height):
        raise ValueError(f"sample_meta {sample_key_col} must be unique")

    feature_ids, feature_meta_df = load_feature_meta(feature_meta_path, feature_id_col=feature_id_col)
    if feature_key_col not in feature_meta_df.columns:
        raise ValueError(f"feature_meta parquet must have key column: {feature_key_col}")
    feature_keys = feature_meta_df[feature_key_col]
    if feature_keys.null_count() != 0:
        raise ValueError(f"feature_meta {feature_key_col} must not contain nulls")
    if int(feature_keys.n_unique()) != int(feature_meta_df.height):
        raise ValueError(f"feature_meta {feature_key_col} must be unique")
    n_features = int(feature_ids.shape[0])
    n_shards, shard_size, shard_starts, shard_ends = _assign_shards_by_target_bytes(
        n_features,
        n_samples,
        target_shard_bytes=int(target_shard_bytes),
        n_shards_override=None if n_shards is None else int(n_shards),
    )
    stats["load_metadata_s"] = time.perf_counter() - phase_t0

    # allocate memmaps (values are always float64, valid mask is uint8)
    values_maps = []
    valid_maps = []
    values_paths = []
    valid_paths = []
    try:
        phase_t0 = time.perf_counter()
        for shard_id in range(n_shards):
            start = shard_starts[shard_id]
            end = shard_ends[shard_id]
            n_rows = max(0, end - start)
            values_path = os.path.join(tmp_dir, f"shard_{shard_id:04d}_values.dat")
            valid_path = os.path.join(tmp_dir, f"shard_{shard_id:04d}_valid.dat")
            values_mm = np.memmap(values_path, dtype=np.float64, mode="w+", shape=(n_rows, n_samples))
            valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="w+", shape=(n_rows, n_samples))
            values_mm[:] = 0.0
            valid_mm[:] = 0
            values_maps.append(values_mm)
            valid_maps.append(valid_mm)
            values_paths.append(values_path)
            valid_paths.append(valid_path)
        stats["allocate_memmaps_s"] = time.perf_counter() - phase_t0

        # fill memmaps
        phase_t0 = time.perf_counter()
        for s_idx, path in enumerate(sample_paths):
            df = pl.read_parquet(path, columns=[feature_id_col, value_col])
            fids = df[feature_id_col].to_numpy().astype(np.int32, copy=False)
            vals = df[value_col].to_numpy().astype(np.float64, copy=False)

            valid_val = ~np.isnan(vals)
            if not np.all(valid_val):
                fids = fids[valid_val]
                vals = vals[valid_val]
            if fids.size == 0:
                continue
            if int(np.min(fids)) < 0 or int(np.max(fids)) >= n_features:
                raise ValueError(
                    f"sample-major feature ids must be dense 0..{n_features - 1}; "
                    f"found range [{int(fids.min())}, {int(fids.max())}]"
                )
            idx = fids.astype(np.int64, copy=False)
            if shard_size <= 0:
                continue

            shard_ids = idx // shard_size
            offsets = idx - shard_ids * shard_size
            for shard_id in np.unique(shard_ids):
                mask = shard_ids == shard_id
                off = offsets[mask]
                val = vals[mask]
                values_maps[int(shard_id)][off, s_idx] = val
                valid_maps[int(shard_id)][off, s_idx] = 1

        for mm in values_maps:
            mm.flush()
        for mm in valid_maps:
            mm.flush()
        stats["fill_memmaps_s"] = time.perf_counter() - phase_t0

        selection_stats_arrays = {
            stats_col: {
                "r2y": np.full(n_features, np.nan, dtype=np.float64),
                "n_y_overlap": np.full(n_features, -1, dtype=np.int32),
            }
            for stats_col in resolved_stats_y_cols
        }
        stats_targets = {}
        for stats_col in resolved_stats_y_cols:
            _, stats_y, stats_y_mask, _ = load_sample_meta(
                sample_meta_path,
                y_col=stats_col,
                sample_id_col=sample_id_col,
                path_col=path_col,
            )
            stats_targets[stats_col] = (
                np.asarray(stats_y, dtype=np.float64),
                np.asarray(stats_y_mask, dtype=np.uint8),
            )

        # compute stats, materialize parquet, then delete each shard backing file eagerly
        compute_stats_s = 0.0
        write_shards_s = 0.0
        cleanup_backing_files_s = 0.0
        for shard_id in range(n_shards):
            start = shard_starts[shard_id]
            end = shard_ends[shard_id]
            n_rows = max(0, end - start)
            shard_file = shard_file_path(shard_path, shard_id)
            values_mm = values_maps[shard_id]
            valid_mm = valid_maps[shard_id]

            if n_rows == 0:
                write_t0 = time.perf_counter()
                df = pl.DataFrame(
                    {
                        "feature_id": pl.Series("feature_id", [], dtype=pl.Int32),
                        "value_len": pl.Series("value_len", [], dtype=pl.Int32),
                        "values_blob": pl.Series("values_blob", [], dtype=pl.Binary),
                        "valid_blob": pl.Series("valid_blob", [], dtype=pl.Binary),
                    }
                )
                df.write_parquet(shard_file)
                write_shards_s += time.perf_counter() - write_t0
                cleanup_t0 = time.perf_counter()
                _close_memmap(values_mm)
                _close_memmap(valid_mm)
                values_maps[shard_id] = None
                valid_maps[shard_id] = None
                _cleanup_backing_file(values_paths[shard_id])
                _cleanup_backing_file(valid_paths[shard_id])
                cleanup_backing_files_s += time.perf_counter() - cleanup_t0
                continue

            compute_t0 = time.perf_counter()
            for stats_col, (stats_y, stats_y_mask) in stats_targets.items():
                shard_r2, shard_n = batch_r2_one_vs_many(
                    stats_y,
                    stats_y_mask.astype(np.uint8, copy=False),
                    values_mm,
                    valid_mm,
                    min_non_null=0,
                    sanitize=True,
                )
                selection_stats_arrays[stats_col]["r2y"][start:end] = shard_r2.astype(np.float64, copy=False)
                selection_stats_arrays[stats_col]["n_y_overlap"][start:end] = shard_n.astype(np.int32, copy=False)
            compute_stats_s += time.perf_counter() - compute_t0

            shard_feature_ids = feature_ids[start:end]
            value_len = np.full(n_rows, n_samples, dtype=np.int32)
            values_blob = [np.asarray(values_mm[row], dtype="<f8").tobytes() for row in range(n_rows)]
            valid_blob = [np.asarray(valid_mm[row], dtype=np.uint8).tobytes() for row in range(n_rows)]
            write_t0 = time.perf_counter()
            df = pl.DataFrame(
                {
                    "feature_id": pl.Series("feature_id", shard_feature_ids, dtype=pl.Int32),
                    "value_len": pl.Series("value_len", value_len, dtype=pl.Int32),
                    "values_blob": pl.Series("values_blob", values_blob, dtype=pl.Binary),
                    "valid_blob": pl.Series("valid_blob", valid_blob, dtype=pl.Binary),
                }
            )
            df.write_parquet(shard_file)
            write_shards_s += time.perf_counter() - write_t0

            cleanup_t0 = time.perf_counter()
            _close_memmap(values_mm)
            _close_memmap(valid_mm)
            values_maps[shard_id] = None
            valid_maps[shard_id] = None
            _cleanup_backing_file(values_paths[shard_id])
            _cleanup_backing_file(valid_paths[shard_id])
            cleanup_backing_files_s += time.perf_counter() - cleanup_t0
        stats["compute_selection_stats_s"] = compute_stats_s
        stats["write_shards_s"] = write_shards_s
        stats["cleanup_backing_files_s"] = cleanup_backing_files_s
    finally:
        for mm in values_maps:
            _close_memmap(mm)
        for mm in valid_maps:
            _close_memmap(mm)
        for path in values_paths:
            _cleanup_backing_file(path)
        for path in valid_paths:
            _cleanup_backing_file(path)
        _cleanup_empty_dir(tmp_dir)

    # copy metadata into the output artifact so the shard set is self-contained
    phase_t0 = time.perf_counter()
    sample_meta_out = os.path.join(out_dir, "sample_meta.parquet")
    feature_meta_out = os.path.join(out_dir, "feature_meta.parquet")
    if os.path.normcase(os.path.abspath(sample_meta_out)) != os.path.normcase(os.path.abspath(sample_meta_path)):
        shutil.copy2(sample_meta_path, sample_meta_out)
    if os.path.normcase(os.path.abspath(feature_meta_out)) != os.path.normcase(os.path.abspath(feature_meta_path)):
        shutil.copy2(feature_meta_path, feature_meta_out)
    stats["copy_metadata_s"] = time.perf_counter() - phase_t0

    # write feature locator
    phase_t0 = time.perf_counter()
    locator_path = os.path.join(out_dir, "feature_locator.parquet")
    global_rank = np.arange(n_features, dtype=np.int32)
    if shard_size > 0:
        shard_id = (global_rank // shard_size).astype(np.int32, copy=False)
        offset_in_shard = (global_rank - shard_id * shard_size).astype(np.int32, copy=False)
    else:
        shard_id = np.zeros(n_features, dtype=np.int32)
        offset_in_shard = np.zeros(n_features, dtype=np.int32)
    locator_df = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
            "global_rank": pl.Series("global_rank", global_rank, dtype=pl.Int32),
            "shard_id": pl.Series("shard_id", shard_id, dtype=pl.Int32),
            "offset_in_shard": pl.Series("offset_in_shard", offset_in_shard, dtype=pl.Int32),
        }
    )
    locator_df.write_parquet(locator_path)
    stats["write_locator_s"] = time.perf_counter() - phase_t0

    phase_t0 = time.perf_counter()
    selection_stats_dir = os.path.join(out_dir, "selection_stats")
    os.makedirs(selection_stats_dir, exist_ok=True)
    selection_stats = {}
    for stats_col in resolved_stats_y_cols:
        filename = _selection_stats_filename(stats_col)
        relative_path = os.path.join("selection_stats", filename)
        absolute_path = os.path.join(out_dir, relative_path)
        stats_df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
                "shard_id": pl.Series("shard_id", shard_id, dtype=pl.Int32),
                "offset_in_shard": pl.Series("offset_in_shard", offset_in_shard, dtype=pl.Int32),
                "r2y": pl.Series("r2y", selection_stats_arrays[stats_col]["r2y"], dtype=pl.Float64),
                "n_y_overlap": pl.Series(
                    "n_y_overlap",
                    selection_stats_arrays[stats_col]["n_y_overlap"],
                    dtype=pl.Int32,
                ),
            }
        )
        stats_df.write_parquet(absolute_path)
        selection_stats[str(stats_col)] = relative_path
    stats["write_selection_stats_s"] = time.perf_counter() - phase_t0

    manifest = ShardManifest(
        sample_meta_path="sample_meta.parquet",
        feature_meta_path="feature_meta.parquet",
        n_samples=n_samples,
        n_features=n_features,
        shard_path="feature_shards",
        n_shards=n_shards,
        feature_locator_path="feature_locator.parquet",
        feature_locator_format="parquet_v1",
        feature_id_dtype="INT32",
        values_dtype="blob_float64_le_len",
        valid_dtype="blob_uint8_len",
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
        target_shard_bytes=None if n_shards is not None else int(target_shard_bytes),
        selection_stats=selection_stats,
    )

    phase_t0 = time.perf_counter()
    manifest_path = os.path.join(out_dir, "shard_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2)
    stats["write_manifest_s"] = time.perf_counter() - phase_t0
    stats["total_s"] = time.perf_counter() - total_t0

    if return_stats:
        return manifest_path, stats
    return manifest_path


def build_shards_from_sample_bundles(
    sample_major_manifest_path: str,
    out_dir: str,
    feature_meta_path: str = None,
    n_shards: Optional[int] = None,
    target_shard_bytes: int = 256 * 1024 * 1024,
    feature_id_col: str = "feature_id",
    value_col: str = "value",
    sample_id_col: str = "sample_id",
    sample_key_col: str = "sample_key",
    feature_key_col: str = "feature_key",
    path_col: str = "sample_path",
    y_col: str = "y",
    stats_y_cols: Optional[List[str]] = None,
    values_dtype: str = "float64",
    valid_dtype: str = "uint8",
    tmp_dir: str = None,
    return_stats: bool = False,
):
    """Build feature-major scalar shards from sample-bundle parquet inputs.

    Args:
        sample_major_manifest_path: Path to the visible sample-bundle stage
            manifest written by `ScalarDatasetBuilder.finish_sample_major()`.
        out_dir: Output directory where shards, locator, and manifest are written.
        feature_meta_path: Optional override for dense feature metadata.
        n_shards: Optional explicit shard-count override kept for legacy use.
        target_shard_bytes: Target parquet shard size in bytes. Used when
            `n_shards` is not provided.
        feature_id_col: Feature id column name in bundle parquet files.
        value_col: Scalar value column name in bundle parquet files.
        sample_id_col: Sample id column name in bundle parquet files and metadata.
        sample_key_col: External sample-key column name stored in sample metadata.
        feature_key_col: External feature-key column name stored in feature metadata.
        path_col: Legacy unused parameter kept for API symmetry.
        y_col: Legacy default target column name used when `stats_y_cols` is
            not provided.
        stats_y_cols: Target columns for precomputing feature-vs-y candidate
            stats. Each column is written as its own sidecar parquet file under
            `selection_stats/`.
        values_dtype: Expected encoded values dtype. Only float64 is supported.
        valid_dtype: Expected encoded validity dtype. Only uint8 is supported.
        tmp_dir: Optional directory for temporary memmap backing files.

    Returns:
        Path to the generated scalar shard manifest JSON file, or
        `(manifest_path, stats)` when `return_stats=True`.
    """
    del path_col
    total_t0 = time.perf_counter()
    stats = {
        "load_metadata_s": 0.0,
        "allocate_memmaps_s": 0.0,
        "fill_memmaps_s": 0.0,
        "compute_selection_stats_s": 0.0,
        "write_shards_s": 0.0,
        "cleanup_backing_files_s": 0.0,
        "copy_metadata_s": 0.0,
        "write_locator_s": 0.0,
        "write_selection_stats_s": 0.0,
        "write_manifest_s": 0.0,
        "total_s": 0.0,
    }
    if values_dtype not in ("float64", "double"):
        raise ValueError("values_dtype must be float64/double for BLOB shard format")
    if valid_dtype != "uint8":
        raise ValueError("valid_dtype must be uint8 for BLOB shard format")
    if n_shards is not None and int(n_shards) <= 0:
        raise ValueError("n_shards must be > 0")
    if int(target_shard_bytes) <= 0:
        raise ValueError("target_shard_bytes must be > 0")

    resolved_stats_y_cols: List[str] = []
    for value in (stats_y_cols or [y_col]):
        name = str(value)
        if name and name not in resolved_stats_y_cols:
            resolved_stats_y_cols.append(name)
    if not resolved_stats_y_cols:
        raise ValueError("at least one stats y column is required")

    os.makedirs(out_dir, exist_ok=True)
    shard_path = os.path.join(out_dir, "feature_shards")
    os.makedirs(shard_path, exist_ok=True)
    if tmp_dir is None:
        tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    phase_t0 = time.perf_counter()
    stage_manifest = _load_sample_bundle_manifest(sample_major_manifest_path)
    sample_meta_path = str(stage_manifest["sample_meta_path"])
    if feature_meta_path is None:
        feature_meta_path = str(stage_manifest["feature_meta_path"])
    bundle_paths = list(stage_manifest["bundle_paths"])

    sample_ids, _, _ = load_sample_targets(
        sample_meta_path,
        y_col=resolved_stats_y_cols[0],
        sample_id_col=sample_id_col,
    )
    n_samples = len(sample_ids)
    sample_meta_df = pl.read_parquet(sample_meta_path, columns=[sample_key_col] if sample_key_col else None)
    if sample_key_col not in sample_meta_df.columns:
        raise ValueError(f"sample_meta parquet must have key column: {sample_key_col}")
    sample_keys = sample_meta_df[sample_key_col]
    if sample_keys.null_count() != 0:
        raise ValueError(f"sample_meta {sample_key_col} must not contain nulls")
    if int(sample_keys.n_unique()) != int(sample_meta_df.height):
        raise ValueError(f"sample_meta {sample_key_col} must be unique")

    feature_ids, feature_meta_df = load_feature_meta(feature_meta_path, feature_id_col=feature_id_col)
    if feature_key_col not in feature_meta_df.columns:
        raise ValueError(f"feature_meta parquet must have key column: {feature_key_col}")
    feature_keys = feature_meta_df[feature_key_col]
    if feature_keys.null_count() != 0:
        raise ValueError(f"feature_meta {feature_key_col} must not contain nulls")
    if int(feature_keys.n_unique()) != int(feature_meta_df.height):
        raise ValueError(f"feature_meta {feature_key_col} must be unique")
    n_features = int(feature_ids.shape[0])
    n_shards, shard_size, shard_starts, shard_ends = _assign_shards_by_target_bytes(
        n_features,
        n_samples,
        target_shard_bytes=int(target_shard_bytes),
        n_shards_override=None if n_shards is None else int(n_shards),
    )
    stats["load_metadata_s"] = time.perf_counter() - phase_t0

    values_maps = []
    valid_maps = []
    values_paths = []
    valid_paths = []
    try:
        phase_t0 = time.perf_counter()
        for shard_id in range(n_shards):
            start = shard_starts[shard_id]
            end = shard_ends[shard_id]
            n_rows = max(0, end - start)
            values_path = os.path.join(tmp_dir, f"shard_{shard_id:04d}_values.dat")
            valid_path = os.path.join(tmp_dir, f"shard_{shard_id:04d}_valid.dat")
            values_mm = np.memmap(values_path, dtype=np.float64, mode="w+", shape=(n_rows, n_samples))
            valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="w+", shape=(n_rows, n_samples))
            values_mm[:] = 0.0
            valid_mm[:] = 0
            values_maps.append(values_mm)
            valid_maps.append(valid_mm)
            values_paths.append(values_path)
            valid_paths.append(valid_path)
        stats["allocate_memmaps_s"] = time.perf_counter() - phase_t0

        phase_t0 = time.perf_counter()
        for path in bundle_paths:
            df = pl.read_parquet(path, columns=[sample_id_col, feature_id_col, value_col])
            if df.height <= 0:
                continue
            sids = df[sample_id_col].to_numpy().astype(np.int64, copy=False)
            fids = df[feature_id_col].to_numpy().astype(np.int32, copy=False)
            vals = df[value_col].to_numpy().astype(np.float64, copy=False)

            valid_val = ~np.isnan(vals)
            if not np.all(valid_val):
                sids = sids[valid_val]
                fids = fids[valid_val]
                vals = vals[valid_val]
            if fids.size == 0:
                continue
            if int(np.min(sids)) < 0 or int(np.max(sids)) >= n_samples:
                raise ValueError(
                    f"bundle sample ids must be dense 0..{n_samples - 1}; "
                    f"found range [{int(sids.min())}, {int(sids.max())}]"
                )
            if int(np.min(fids)) < 0 or int(np.max(fids)) >= n_features:
                raise ValueError(
                    f"bundle feature ids must be dense 0..{n_features - 1}; "
                    f"found range [{int(fids.min())}, {int(fids.max())}]"
                )
            idx = fids.astype(np.int64, copy=False)
            if shard_size <= 0:
                continue
            shard_ids = idx // shard_size
            offsets = idx - shard_ids * shard_size
            for shard_id in np.unique(shard_ids):
                mask = shard_ids == shard_id
                off = offsets[mask]
                sid = sids[mask]
                val = vals[mask]
                values_maps[int(shard_id)][off, sid] = val
                valid_maps[int(shard_id)][off, sid] = 1

        for mm in values_maps:
            mm.flush()
        for mm in valid_maps:
            mm.flush()
        stats["fill_memmaps_s"] = time.perf_counter() - phase_t0

        selection_stats_arrays = {
            stats_col: {
                "r2y": np.full(n_features, np.nan, dtype=np.float64),
                "n_y_overlap": np.full(n_features, -1, dtype=np.int32),
            }
            for stats_col in resolved_stats_y_cols
        }
        stats_targets = {}
        for stats_col in resolved_stats_y_cols:
            _, stats_y, stats_y_mask = load_sample_targets(
                sample_meta_path,
                y_col=stats_col,
                sample_id_col=sample_id_col,
            )
            stats_targets[stats_col] = (
                np.asarray(stats_y, dtype=np.float64),
                np.asarray(stats_y_mask, dtype=np.uint8),
            )

        compute_stats_s = 0.0
        write_shards_s = 0.0
        cleanup_backing_files_s = 0.0
        for shard_id in range(n_shards):
            start = shard_starts[shard_id]
            end = shard_ends[shard_id]
            n_rows = max(0, end - start)
            shard_file = shard_file_path(shard_path, shard_id)
            values_mm = values_maps[shard_id]
            valid_mm = valid_maps[shard_id]

            if n_rows == 0:
                write_t0 = time.perf_counter()
                df = pl.DataFrame(
                    {
                        "feature_id": pl.Series("feature_id", [], dtype=pl.Int32),
                        "value_len": pl.Series("value_len", [], dtype=pl.Int32),
                        "values_blob": pl.Series("values_blob", [], dtype=pl.Binary),
                        "valid_blob": pl.Series("valid_blob", [], dtype=pl.Binary),
                    }
                )
                df.write_parquet(shard_file)
                write_shards_s += time.perf_counter() - write_t0
                cleanup_t0 = time.perf_counter()
                _close_memmap(values_mm)
                _close_memmap(valid_mm)
                values_maps[shard_id] = None
                valid_maps[shard_id] = None
                _cleanup_backing_file(values_paths[shard_id])
                _cleanup_backing_file(valid_paths[shard_id])
                cleanup_backing_files_s += time.perf_counter() - cleanup_t0
                continue

            compute_t0 = time.perf_counter()
            for stats_col, (stats_y, stats_y_mask) in stats_targets.items():
                shard_r2, shard_n = batch_r2_one_vs_many(
                    stats_y,
                    stats_y_mask.astype(np.uint8, copy=False),
                    values_mm,
                    valid_mm,
                    min_non_null=0,
                    sanitize=True,
                )
                selection_stats_arrays[stats_col]["r2y"][start:end] = shard_r2.astype(np.float64, copy=False)
                selection_stats_arrays[stats_col]["n_y_overlap"][start:end] = shard_n.astype(np.int32, copy=False)
            compute_stats_s += time.perf_counter() - compute_t0

            shard_feature_ids = feature_ids[start:end]
            value_len = np.full(n_rows, n_samples, dtype=np.int32)
            values_blob = [np.asarray(values_mm[row], dtype="<f8").tobytes() for row in range(n_rows)]
            valid_blob = [np.asarray(valid_mm[row], dtype=np.uint8).tobytes() for row in range(n_rows)]
            write_t0 = time.perf_counter()
            df = pl.DataFrame(
                {
                    "feature_id": pl.Series("feature_id", shard_feature_ids, dtype=pl.Int32),
                    "value_len": pl.Series("value_len", value_len, dtype=pl.Int32),
                    "values_blob": pl.Series("values_blob", values_blob, dtype=pl.Binary),
                    "valid_blob": pl.Series("valid_blob", valid_blob, dtype=pl.Binary),
                }
            )
            df.write_parquet(shard_file)
            write_shards_s += time.perf_counter() - write_t0

            cleanup_t0 = time.perf_counter()
            _close_memmap(values_mm)
            _close_memmap(valid_mm)
            values_maps[shard_id] = None
            valid_maps[shard_id] = None
            _cleanup_backing_file(values_paths[shard_id])
            _cleanup_backing_file(valid_paths[shard_id])
            cleanup_backing_files_s += time.perf_counter() - cleanup_t0
        stats["compute_selection_stats_s"] = compute_stats_s
        stats["write_shards_s"] = write_shards_s
        stats["cleanup_backing_files_s"] = cleanup_backing_files_s
    finally:
        for mm in values_maps:
            _close_memmap(mm)
        for mm in valid_maps:
            _close_memmap(mm)
        for path in values_paths:
            _cleanup_backing_file(path)
        for path in valid_paths:
            _cleanup_backing_file(path)
        _cleanup_empty_dir(tmp_dir)

    phase_t0 = time.perf_counter()
    sample_meta_out = os.path.join(out_dir, "sample_meta.parquet")
    feature_meta_out = os.path.join(out_dir, "feature_meta.parquet")
    if os.path.normcase(os.path.abspath(sample_meta_out)) != os.path.normcase(os.path.abspath(sample_meta_path)):
        shutil.copy2(sample_meta_path, sample_meta_out)
    if os.path.normcase(os.path.abspath(feature_meta_out)) != os.path.normcase(os.path.abspath(feature_meta_path)):
        shutil.copy2(feature_meta_path, feature_meta_out)
    stats["copy_metadata_s"] = time.perf_counter() - phase_t0

    phase_t0 = time.perf_counter()
    locator_path = os.path.join(out_dir, "feature_locator.parquet")
    global_rank = np.arange(n_features, dtype=np.int32)
    if shard_size > 0:
        shard_id = (global_rank // shard_size).astype(np.int32, copy=False)
        offset_in_shard = (global_rank - shard_id * shard_size).astype(np.int32, copy=False)
    else:
        shard_id = np.zeros(n_features, dtype=np.int32)
        offset_in_shard = np.zeros(n_features, dtype=np.int32)
    locator_df = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
            "global_rank": pl.Series("global_rank", global_rank, dtype=pl.Int32),
            "shard_id": pl.Series("shard_id", shard_id, dtype=pl.Int32),
            "offset_in_shard": pl.Series("offset_in_shard", offset_in_shard, dtype=pl.Int32),
        }
    )
    locator_df.write_parquet(locator_path)
    stats["write_locator_s"] = time.perf_counter() - phase_t0

    phase_t0 = time.perf_counter()
    selection_stats_dir = os.path.join(out_dir, "selection_stats")
    os.makedirs(selection_stats_dir, exist_ok=True)
    selection_stats = {}
    for stats_col in resolved_stats_y_cols:
        filename = _selection_stats_filename(stats_col)
        relative_path = os.path.join("selection_stats", filename)
        absolute_path = os.path.join(out_dir, relative_path)
        stats_df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
                "shard_id": pl.Series("shard_id", shard_id, dtype=pl.Int32),
                "offset_in_shard": pl.Series("offset_in_shard", offset_in_shard, dtype=pl.Int32),
                "r2y": pl.Series("r2y", selection_stats_arrays[stats_col]["r2y"], dtype=pl.Float64),
                "n_y_overlap": pl.Series(
                    "n_y_overlap",
                    selection_stats_arrays[stats_col]["n_y_overlap"],
                    dtype=pl.Int32,
                ),
            }
        )
        stats_df.write_parquet(absolute_path)
        selection_stats[str(stats_col)] = relative_path
    stats["write_selection_stats_s"] = time.perf_counter() - phase_t0

    manifest = ShardManifest(
        sample_meta_path="sample_meta.parquet",
        feature_meta_path="feature_meta.parquet",
        n_samples=n_samples,
        n_features=n_features,
        shard_path="feature_shards",
        n_shards=n_shards,
        feature_locator_path="feature_locator.parquet",
        feature_locator_format="parquet_v1",
        feature_id_dtype="INT32",
        values_dtype="blob_float64_le_len",
        valid_dtype="blob_uint8_len",
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
        target_shard_bytes=None if n_shards is not None else int(target_shard_bytes),
        selection_stats=selection_stats,
    )

    phase_t0 = time.perf_counter()
    manifest_path = os.path.join(out_dir, "shard_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2)
    stats["write_manifest_s"] = time.perf_counter() - phase_t0
    stats["total_s"] = time.perf_counter() - total_t0

    if return_stats:
        return manifest_path, stats
    return manifest_path


def load_manifest(manifest_path: str) -> ShardManifest:
    """Load a scalar shard manifest from JSON.

    Args:
        manifest_path: Path to `shard_manifest.json`.

    Returns:
        A populated `ShardManifest` instance.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest_path = os.path.abspath(manifest_path)
    return ShardManifest(
        sample_meta_path=_resolve_manifest_relative(manifest_path, data["sample_meta_path"]),
        feature_meta_path=_resolve_manifest_relative(manifest_path, data.get("feature_meta_path", "")),
        n_samples=data["n_samples"],
        n_features=data.get("n_features", 0),
        shard_path=_resolve_manifest_relative(manifest_path, data["shard_path"]),
        n_shards=data["n_shards"],
        feature_locator_path=_resolve_manifest_relative(manifest_path, data["feature_locator_path"]),
        feature_locator_format=data["feature_locator_format"],
        feature_id_dtype=data["feature_id_dtype"],
        values_dtype=data["values_dtype"],
        valid_dtype=data["valid_dtype"],
        id_scheme=str(data.get("id_scheme", "legacy")),
        sample_key_col=str(data.get("sample_key_col", "sample_key")),
        feature_key_col=str(data.get("feature_key_col", "feature_key")),
        target_shard_bytes=data.get("target_shard_bytes"),
        selection_stats={
            str(key): _resolve_manifest_relative(manifest_path, str(value))
            for key, value in (data.get("selection_stats") or {}).items()
        }
        or None,
        stats_y_col=data.get("stats_y_col"),
    )


class ParquetShardReader:
    def __init__(
        self,
        manifest: ShardManifest,
        value_len_col: str = "value_len",
        values_blob_col: str = "values_blob",
        valid_blob_col: str = "valid_blob",
        feature_id_col: str = "feature_id",
        max_gap: int = 0,
    ):
        """Create a reader for scalar parquet shards.

        Args:
            manifest: Loaded scalar shard manifest.
            value_len_col: Column storing the dense vector length for each row.
            values_blob_col: Column storing float64-encoded value blobs.
            valid_blob_col: Column storing uint8 validity blobs.
            feature_id_col: Feature id column name in the shard parquet file.
            max_gap: Allowed gap when merging sparse row offsets into larger scan
                slices.
        """
        self.manifest = manifest
        self.value_len_col = value_len_col
        self.values_blob_col = values_blob_col
        self.valid_blob_col = valid_blob_col
        self.feature_id_col = feature_id_col
        self.max_gap = max_gap
        self._scans: Dict[int, pl.LazyFrame] = {}

    def _scan(self, shard_id: int) -> pl.LazyFrame:
        """Return and cache a lazy parquet scan for one shard.

        Args:
            shard_id: Zero-based shard identifier.

        Returns:
            A cached Polars lazy frame over the shard parquet file.
        """
        if shard_id not in self._scans:
            path = shard_file_path(self.manifest.shard_path, shard_id)
            self._scans[shard_id] = pl.scan_parquet(path)
        return self._scans[shard_id]

    def shard_row_count(self, shard_id: int) -> int:
        """Return the number of feature rows stored in a shard.

        Args:
            shard_id: Zero-based shard identifier.

        Returns:
            Number of rows present in the shard parquet file.
        """
        scan = self._scan(shard_id)
        return int(scan.select(pl.len()).collect().item())

    def _group_offsets(self, offsets: List[int]) -> List[Tuple[int, int]]:
        """Merge sorted row offsets into contiguous scan slices.

        Args:
            offsets: Shard-local row offsets to load.

        Returns:
            A list of inclusive `(start, end)` ranges used for parquet slicing.
        """
        if not offsets:
            return []
        offsets_sorted = sorted(offsets)
        ranges = []
        start = offsets_sorted[0]
        prev = start
        for off in offsets_sorted[1:]:
            if off <= prev + 1 + self.max_gap:
                prev = off
                continue
            ranges.append((start, prev))
            start = off
            prev = off
        ranges.append((start, prev))
        return ranges

    def load_rows(self, shard_id: int, offsets: List[int]):
        """Load and decode multiple scalar feature rows from one shard.

        Args:
            shard_id: Zero-based shard identifier.
            offsets: Shard-local row offsets to read.

        Returns:
            A pair `(values, valid)` where both arrays have shape
            `(len(offsets), manifest.n_samples)`.
        """
        if not offsets:
            return (
                np.empty((0, self.manifest.n_samples), dtype=np.float64),
                np.empty((0, self.manifest.n_samples), dtype=np.uint8),
            )

        offsets = list(offsets)
        order = np.argsort(offsets)
        offsets_sorted = [offsets[i] for i in order]
        ranges = self._group_offsets(offsets_sorted)

        values_list = []
        valid_list = []
        scan = self._scan(shard_id)
        pos = 0

        for start, end in ranges:
            length = end - start + 1
            df = scan.slice(start, length).collect()
            value_lens = df[self.value_len_col].to_numpy()
            values_blob = df[self.values_blob_col].to_list()
            valid_blob = df[self.valid_blob_col].to_list()

            while pos < len(offsets_sorted) and offsets_sorted[pos] <= end:
                off = offsets_sorted[pos]
                row = off - start
                value_len = int(value_lens[row])
                if value_len != self.manifest.n_samples:
                    raise ValueError(
                        f"value_len mismatch at shard={shard_id}, offset={off}: "
                        f"{value_len} != {self.manifest.n_samples}"
                    )
                v = _decode_values_blob(values_blob[row], value_len)
                m = _decode_valid_blob(valid_blob[row], value_len)
                values_list.append(v)
                valid_list.append(m)
                pos += 1

        values = np.vstack(values_list)
        valid = np.vstack(valid_list)

        inv_order = np.argsort(order)
        values = values[inv_order]
        valid = valid[inv_order]
        return values, valid

    def load_feature_by_offset(self, shard_id: int, offset: int):
        """Load one scalar feature row by shard-local offset.

        Args:
            shard_id: Zero-based shard identifier.
            offset: Zero-based row offset inside the shard.

        Returns:
            A pair `(values, valid)` for the requested feature row.
        """
        values, valid = self.load_rows(shard_id, [offset])
        return values[0], valid[0]

    def load_feature_by_id(self, feature_id: int, locator_index=None):
        """Load one scalar feature row by logical feature id.

        Args:
            feature_id: Logical feature identifier.
            locator_index: Optional prebuilt feature locator dictionary.

        Returns:
            A pair `(values, valid)` for the requested feature. Missing features
            return zero-filled arrays.
        """
        locator_index = locator_index or build_feature_locator_index(self.manifest.feature_locator_path)
        loc = locator_index.get(int(feature_id))
        if loc is None:
            return (
                np.zeros(self.manifest.n_samples, dtype=np.float64),
                np.zeros(self.manifest.n_samples, dtype=np.uint8),
            )
        shard_id, offset = loc
        return self.load_feature_by_offset(int(shard_id), int(offset))


class InMemoryShardReader:
    def __init__(self, shards):
        """Create an in-memory shard reader for tests and benchmarks.

        Args:
            shards: List of shard dictionaries containing `values` and `valid`.
        """
        self.shards = shards
        self.n_samples = shards[0]["values"].shape[1] if shards else 0

    def load_rows(self, shard_id: int, offsets: List[int]):
        """Load multiple rows from an in-memory shard.

        Args:
            shard_id: Zero-based shard identifier.
            offsets: Row offsets to load.

        Returns:
            A pair `(values, valid)` sliced directly from the shard arrays.
        """
        if not offsets:
            return (
                np.empty((0, self.n_samples), dtype=np.float64),
                np.empty((0, self.n_samples), dtype=np.uint8),
            )
        values = self.shards[shard_id]["values"][offsets]
        valid = self.shards[shard_id]["valid"][offsets]
        return values, valid

    def load_feature_by_offset(self, shard_id: int, offset: int):
        """Load one feature row from an in-memory shard.

        Args:
            shard_id: Zero-based shard identifier.
            offset: Zero-based row offset inside the shard.

        Returns:
            A pair `(values, valid)` for the requested feature row.
        """
        return self.shards[shard_id]["values"][offset], self.shards[shard_id]["valid"][offset]


def load_scalar_feature_by_sample_ids(
    manifest,
    feature_id: int,
    sample_ids,
    locator_index=None,
    reader: ParquetShardReader = None,
):
    """Load scalar values for one feature aligned to requested sample ids.

    Args:
        manifest: Loaded manifest or path to `shard_manifest.json`.
        feature_id: Logical feature identifier.
        sample_ids: External sample ids requested by the caller.
        locator_index: Optional prebuilt feature locator dictionary.
        reader: Optional shard reader instance to reuse.

    Returns:
        A dictionary keyed by dense sample id. Each value contains
        `sample_id`, `present`, and `value`.
    """
    if isinstance(manifest, str):
        manifest = load_manifest(manifest)
    locator_index = locator_index or build_feature_locator_index(manifest.feature_locator_path)
    validate_dense_sample_ids(manifest.sample_meta_path)
    reader = reader or ParquetShardReader(manifest)

    values, valid = reader.load_feature_by_id(feature_id, locator_index=locator_index)
    out = {}
    for sample_id in sample_ids:
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= int(manifest.n_samples):
            out[sample_id] = {
                "sample_id": sample_id,
                "present": False,
                "value": None,
            }
            continue
        present = bool(valid[sample_id])
        out[sample_id] = {
            "sample_id": sample_id,
            "present": present,
            "value": float(values[sample_id]) if present else None,
        }
    return out
