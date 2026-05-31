"""Dense-long scalar shard builder and reader."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..models import FeatureValues, QueryResult, ScalarValue
from .pearson import batch_r2_one_vs_many
from .storage_common import (
    cleanup_backing_file,
    cleanup_empty_dir,
    close_memmap,
    load_feature_meta,
    load_sample_major_manifest,
    load_sample_targets,
)


DENSE_LONG_FORMAT_NAME = "scalar-dense-long-shard-v1"
DENSE_LONG_ROW_VALUE_BYTES = 8
DENSE_LONG_ROW_MASK_BYTES = 1
DENSE_LONG_ROW_ID_BYTES = 12
DENSE_LONG_ROW_GROUP_FEATURES = 128


@dataclass(frozen=True)
class ScalarDenseLongPart:
    """One feature-range parquet part in a dense-long scalar shard."""

    part_id: int
    path: str
    first_feature_id: int
    last_feature_id: int
    feature_count: int
    row_count: int
    byte_size: int


@dataclass(frozen=True)
class ScalarDenseLongManifest:
    """Manifest for dense-long scalar shards."""

    manifest_path: str
    sample_meta_path: str
    feature_meta_path: str
    n_samples: int
    n_features: int
    parts_path: str
    parts: list[ScalarDenseLongPart]
    feature_locator_path: str
    sample_key_col: str
    feature_key_col: str
    sample_id_col: str
    feature_id_col: str
    value_col: str
    mask_col: str
    compression: str
    target_part_bytes: Optional[int]
    selection_stats: Optional[dict[str, str]]


def _write_json_atomic(path: str, payload: dict):
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _resolve_manifest_relative(manifest_path: str, value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(manifest_path)), value))


def _selection_stats_filename(y_col: str) -> str:
    return f"{quote(str(y_col), safe='')}.parquet"


def _pyarrow_compression(compression: str) -> Optional[str]:
    value = str(compression or "").strip().lower()
    if value in {"", "none", "uncompressed"}:
        return None
    return value


def _estimate_dense_long_feature_bytes(n_samples: int) -> int:
    return int(n_samples) * (DENSE_LONG_ROW_VALUE_BYTES + DENSE_LONG_ROW_MASK_BYTES + DENSE_LONG_ROW_ID_BYTES)


def _assign_parts_by_target_bytes(
    n_features: int,
    n_samples: int,
    target_part_bytes: int,
    *,
    min_features_per_part: int = 1,
):
    if n_features <= 0:
        return [0], [0]
    if target_part_bytes <= 0:
        raise ValueError("target_part_bytes must be > 0")
    feature_bytes = max(1, _estimate_dense_long_feature_bytes(n_samples))
    features_per_part = max(1, int(min_features_per_part), int(target_part_bytes) // feature_bytes)
    starts = list(range(0, n_features, features_per_part))
    ends = [min(start + features_per_part, n_features) for start in starts]
    return starts, ends


def _read_sample_path_batch(paths: list[str], columns: list[str], sample_ids: list[int] | None = None) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(schema={columns[0]: pl.Int64, columns[1]: pl.Int32, columns[2]: pl.Float64})
    try:
        return pl.scan_parquet(paths, glob=False).select(columns).collect()
    except Exception as exc:
        if sample_ids is None or "unable to find column" not in str(exc):
            raise
        sample_id_col, feature_id_col, value_col = columns
        frames = []
        for path, sample_id in zip(paths, sample_ids):
            frame = pl.read_parquet(path, columns=[feature_id_col, value_col])
            frames.append(frame.with_columns(pl.lit(int(sample_id), dtype=pl.Int64).alias(sample_id_col)).select(columns))
        if not frames:
            return pl.DataFrame(schema={sample_id_col: pl.Int64, feature_id_col: pl.Int32, value_col: pl.Float64})
        return pl.concat(frames, how="vertical")


def _write_dense_long_part(
    path: str,
    feature_ids: np.ndarray,
    sample_ids: np.ndarray,
    values,
    valid,
    *,
    compression: str,
    row_group_features: int,
) -> int:
    if feature_ids.size <= 0:
        table = pa.table(
            {
                "feature_id": pa.array([], type=pa.int32()),
                "sample_id": pa.array([], type=pa.int64()),
                "mask": pa.array([], type=pa.uint8()),
                "value": pa.array([], type=pa.float64()),
            }
        )
        pq.write_table(table, path, compression=_pyarrow_compression(compression))
        return 0

    n_samples = int(sample_ids.shape[0])
    flat_mask = np.asarray(valid, dtype=np.uint8).reshape(-1)
    flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
    flat_values = np.where(flat_mask.astype(bool, copy=False), flat_values, 0.0).astype(np.float64, copy=False)
    table = pa.table(
        {
            "feature_id": pa.array(np.repeat(feature_ids, n_samples), type=pa.int32()),
            "sample_id": pa.array(np.tile(sample_ids, int(feature_ids.shape[0])), type=pa.int64()),
            "mask": pa.array(flat_mask, type=pa.uint8()),
            "value": pa.array(flat_values, type=pa.float64()),
        }
    )
    # A row group covers a small feature block. This keeps feature_id pruning
    # useful without paying the heavy metadata and dictionary overhead of one
    # row group per feature.
    pq.write_table(
        table,
        path,
        compression=_pyarrow_compression(compression),
        use_dictionary=True,
        row_group_size=n_samples * max(1, int(row_group_features)),
    )
    return int(table.num_rows)


def _write_empty_dense_long_part(path: str, *, compression: str, row_group_features: int):
    _write_dense_long_part(
        path,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.float64),
        np.empty((0, 0), dtype=np.uint8),
        compression=compression,
        row_group_features=row_group_features,
    )


def build_dense_long_shards_from_sample_major_manifest(
    sample_major_manifest_path: str,
    out_dir: str,
    *,
    feature_meta_path: str = None,
    target_part_bytes: int = 256 * 1024 * 1024,
    feature_id_col: str = "feature_id",
    value_col: str = "value",
    sample_id_col: str = "sample_id",
    sample_key_col: str = "sample_key",
    feature_key_col: str = "feature_key",
    y_col: str = "y",
    stats_y_cols: Optional[list[str]] = None,
    tmp_dir: str = None,
    compression: str = "zstd",
    input_batch_files: int = 256,
    row_group_features: int = DENSE_LONG_ROW_GROUP_FEATURES,
    return_stats: bool = False,
):
    """Build dense-long scalar shards from sample-major raw rows.

    Input parquet files are expected to contain `(sample_id, feature_id, value)`
    rows. The builder reads those files in batches, fills feature-major memmaps,
    computes selection stats while the dense arrays are hot, then writes final
    Parquet parts sorted by `(feature_id, sample_id)`.
    """

    total_t0 = time.perf_counter()
    stats = {
        "load_metadata_s": 0.0,
        "allocate_memmaps_s": 0.0,
        "fill_memmaps_s": 0.0,
        "compute_selection_stats_s": 0.0,
        "write_parts_s": 0.0,
        "cleanup_backing_files_s": 0.0,
        "copy_metadata_s": 0.0,
        "write_locator_s": 0.0,
        "write_selection_stats_s": 0.0,
        "write_manifest_s": 0.0,
        "total_s": 0.0,
    }

    os.makedirs(out_dir, exist_ok=True)
    parts_path = os.path.join(out_dir, "dense_long_parts")
    os.makedirs(parts_path, exist_ok=True)
    if tmp_dir is None:
        tmp_dir = os.path.join(out_dir, "_tmp_dense_long")
    os.makedirs(tmp_dir, exist_ok=True)

    phase_t0 = time.perf_counter()
    stage_manifest = load_sample_major_manifest(sample_major_manifest_path)
    sample_meta_path = str(stage_manifest["sample_meta_path"])
    if feature_meta_path is None:
        feature_meta_path = str(stage_manifest["feature_meta_path"])
    sample_paths = list(stage_manifest["sample_paths"])
    sample_ids_raw = stage_manifest.get("sample_ids")
    sample_file_ids = None if sample_ids_raw is None else [int(value) for value in sample_ids_raw]
    if sample_file_ids is not None and len(sample_file_ids) != len(sample_paths):
        raise ValueError("sample_ids length must match sample_paths length")
    sample_id_col = str(stage_manifest.get("sample_id_col", sample_id_col))
    feature_id_col = str(stage_manifest.get("feature_id_col", feature_id_col))
    value_col = str(stage_manifest.get("value_col", value_col))

    resolved_stats_y_cols: list[str] = []
    for value in (stats_y_cols or [y_col]):
        name = str(value)
        if name and name not in resolved_stats_y_cols:
            resolved_stats_y_cols.append(name)
    if not resolved_stats_y_cols:
        raise ValueError("at least one stats y column is required")

    sample_ids, _, _ = load_sample_targets(
        sample_meta_path,
        y_col=resolved_stats_y_cols[0],
        sample_id_col=sample_id_col,
    )
    n_samples = len(sample_ids)
    feature_ids, feature_meta_df = load_feature_meta(feature_meta_path, feature_id_col=feature_id_col)
    n_features = int(feature_ids.shape[0])

    if sample_key_col:
        sample_meta_df = pl.read_parquet(sample_meta_path, columns=[sample_key_col])
        if sample_key_col not in sample_meta_df.columns:
            raise ValueError(f"sample_meta parquet must have key column: {sample_key_col}")
        if sample_meta_df[sample_key_col].null_count() != 0:
            raise ValueError(f"sample_meta {sample_key_col} must not contain nulls")
        if int(sample_meta_df[sample_key_col].n_unique()) != int(sample_meta_df.height):
            raise ValueError(f"sample_meta {sample_key_col} must be unique")
    if feature_key_col:
        if feature_key_col not in feature_meta_df.columns:
            raise ValueError(f"feature_meta parquet must have key column: {feature_key_col}")
        if feature_meta_df[feature_key_col].null_count() != 0:
            raise ValueError(f"feature_meta {feature_key_col} must not contain nulls")
        if int(feature_meta_df[feature_key_col].n_unique()) != int(feature_meta_df.height):
            raise ValueError(f"feature_meta {feature_key_col} must be unique")

    part_starts, part_ends = _assign_parts_by_target_bytes(
        n_features,
        n_samples,
        int(target_part_bytes),
        min_features_per_part=int(row_group_features),
    )
    part_count = len(part_starts)
    stats["load_metadata_s"] = time.perf_counter() - phase_t0

    values_maps = []
    valid_maps = []
    values_paths = []
    valid_paths = []
    try:
        phase_t0 = time.perf_counter()
        for part_id, (start, end) in enumerate(zip(part_starts, part_ends)):
            row_count = max(0, int(end) - int(start))
            values_path = os.path.join(tmp_dir, f"part_{part_id:04d}_values.dat")
            valid_path = os.path.join(tmp_dir, f"part_{part_id:04d}_valid.dat")
            values_mm = np.memmap(values_path, dtype=np.float64, mode="w+", shape=(row_count, n_samples))
            valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="w+", shape=(row_count, n_samples))
            values_mm[:] = 0.0
            valid_mm[:] = 0
            values_maps.append(values_mm)
            valid_maps.append(valid_mm)
            values_paths.append(values_path)
            valid_paths.append(valid_path)
        stats["allocate_memmaps_s"] = time.perf_counter() - phase_t0

        features_per_part = max(1, int(part_ends[0] - part_starts[0])) if part_count else 1
        phase_t0 = time.perf_counter()
        batch_size = max(1, int(input_batch_files))
        for batch_start in range(0, len(sample_paths), batch_size):
            paths = sample_paths[batch_start : batch_start + batch_size]
            path_sample_ids = None if sample_file_ids is None else sample_file_ids[batch_start : batch_start + batch_size]
            df = _read_sample_path_batch(paths, [sample_id_col, feature_id_col, value_col], path_sample_ids)
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
                    f"sample ids must be dense 0..{n_samples - 1}; "
                    f"found range [{int(sids.min())}, {int(sids.max())}]"
                )
            if int(np.min(fids)) < 0 or int(np.max(fids)) >= n_features:
                raise ValueError(
                    f"feature ids must be dense 0..{n_features - 1}; "
                    f"found range [{int(fids.min())}, {int(fids.max())}]"
                )
            part_ids = fids.astype(np.int64, copy=False) // features_per_part
            part_ids = np.minimum(part_ids, part_count - 1)
            for part_id in np.unique(part_ids):
                mask = part_ids == part_id
                start = int(part_starts[int(part_id)])
                offsets = fids[mask].astype(np.int64, copy=False) - start
                values_maps[int(part_id)][offsets, sids[mask]] = vals[mask]
                valid_maps[int(part_id)][offsets, sids[mask]] = 1

        for mm in values_maps:
            mm.flush()
        for mm in valid_maps:
            mm.flush()
        stats["fill_memmaps_s"] = time.perf_counter() - phase_t0

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
        selection_stats_arrays = {
            stats_col: {
                "r2y": np.full(n_features, np.nan, dtype=np.float64),
                "n_y_overlap": np.full(n_features, -1, dtype=np.int32),
            }
            for stats_col in resolved_stats_y_cols
        }

        sample_id_array = np.arange(n_samples, dtype=np.int64)
        parts: list[ScalarDenseLongPart] = []
        compute_stats_s = 0.0
        write_parts_s = 0.0
        cleanup_s = 0.0
        for part_id, (start, end) in enumerate(zip(part_starts, part_ends)):
            start = int(start)
            end = int(end)
            row_count = max(0, end - start)
            values_mm = values_maps[part_id]
            valid_mm = valid_maps[part_id]
            part_file = os.path.join(parts_path, f"part_{part_id:04d}.parquet")

            if row_count <= 0:
                write_t0 = time.perf_counter()
                _write_empty_dense_long_part(
                    part_file,
                    compression=compression,
                    row_group_features=row_group_features,
                )
                write_parts_s += time.perf_counter() - write_t0
            else:
                compute_t0 = time.perf_counter()
                for stats_col, (stats_y, stats_y_mask) in stats_targets.items():
                    part_r2, part_n = batch_r2_one_vs_many(
                        stats_y,
                        stats_y_mask.astype(np.uint8, copy=False),
                        values_mm,
                        valid_mm,
                        min_non_null=0,
                        sanitize=True,
                    )
                    selection_stats_arrays[stats_col]["r2y"][start:end] = part_r2.astype(np.float64, copy=False)
                    selection_stats_arrays[stats_col]["n_y_overlap"][start:end] = part_n.astype(np.int32, copy=False)
                compute_stats_s += time.perf_counter() - compute_t0

                write_t0 = time.perf_counter()
                written_rows = _write_dense_long_part(
                    part_file,
                    feature_ids[start:end],
                    sample_id_array,
                    values_mm,
                    valid_mm,
                    compression=compression,
                    row_group_features=row_group_features,
                )
                write_parts_s += time.perf_counter() - write_t0

            byte_size = int(os.path.getsize(part_file))
            parts.append(
                ScalarDenseLongPart(
                    part_id=int(part_id),
                    path=part_file,
                    first_feature_id=start,
                    last_feature_id=max(start, end - 1),
                    feature_count=row_count,
                    row_count=row_count * n_samples,
                    byte_size=byte_size,
                )
            )
            cleanup_t0 = time.perf_counter()
            close_memmap(values_mm)
            close_memmap(valid_mm)
            values_maps[part_id] = None
            valid_maps[part_id] = None
            cleanup_backing_file(values_paths[part_id])
            cleanup_backing_file(valid_paths[part_id])
            cleanup_s += time.perf_counter() - cleanup_t0
        stats["compute_selection_stats_s"] = compute_stats_s
        stats["write_parts_s"] = write_parts_s
        stats["cleanup_backing_files_s"] = cleanup_s
    finally:
        for mm in values_maps:
            close_memmap(mm)
        for mm in valid_maps:
            close_memmap(mm)
        for path in values_paths:
            cleanup_backing_file(path)
        for path in valid_paths:
            cleanup_backing_file(path)
        cleanup_empty_dir(tmp_dir)

    phase_t0 = time.perf_counter()
    sample_meta_out = os.path.join(out_dir, "sample_meta.parquet")
    feature_meta_out = os.path.join(out_dir, "feature_meta.parquet")
    if os.path.normcase(os.path.abspath(sample_meta_out)) != os.path.normcase(os.path.abspath(sample_meta_path)):
        shutil.copy2(sample_meta_path, sample_meta_out)
    if os.path.normcase(os.path.abspath(feature_meta_out)) != os.path.normcase(os.path.abspath(feature_meta_path)):
        shutil.copy2(feature_meta_path, feature_meta_out)
    stats["copy_metadata_s"] = time.perf_counter() - phase_t0

    phase_t0 = time.perf_counter()
    global_rank = np.arange(n_features, dtype=np.int32)
    features_per_part = max(1, int(part_ends[0] - part_starts[0])) if part_count else 1
    part_ids = np.minimum(global_rank // features_per_part, max(0, part_count - 1)).astype(np.int32, copy=False)
    offset_in_part = (global_rank - part_ids * features_per_part).astype(np.int32, copy=False)
    first_row_in_part = (offset_in_part.astype(np.int64) * int(n_samples)).astype(np.int64, copy=False)
    locator_path = os.path.join(out_dir, "feature_locator.parquet")
    pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
            "global_rank": pl.Series("global_rank", global_rank, dtype=pl.Int32),
            "part_id": pl.Series("part_id", part_ids, dtype=pl.Int32),
            "offset_in_part": pl.Series("offset_in_part", offset_in_part, dtype=pl.Int32),
            "first_row_in_part": pl.Series("first_row_in_part", first_row_in_part, dtype=pl.Int64),
        }
    ).write_parquet(locator_path, compression=compression)
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
                "part_id": pl.Series("part_id", part_ids, dtype=pl.Int32),
                "offset_in_part": pl.Series("offset_in_part", offset_in_part, dtype=pl.Int32),
                "r2y": pl.Series("r2y", selection_stats_arrays[stats_col]["r2y"], dtype=pl.Float64),
                "n_y_overlap": pl.Series(
                    "n_y_overlap",
                    selection_stats_arrays[stats_col]["n_y_overlap"],
                    dtype=pl.Int32,
                ),
            }
        )
        stats_df.write_parquet(absolute_path, compression=compression)
        selection_stats[str(stats_col)] = relative_path
    stats["write_selection_stats_s"] = time.perf_counter() - phase_t0

    phase_t0 = time.perf_counter()
    manifest_path = os.path.join(out_dir, "dense_long_shard_manifest.json")
    manifest_payload = {
        "format": DENSE_LONG_FORMAT_NAME,
        "sample_meta_path": "sample_meta.parquet",
        "feature_meta_path": "feature_meta.parquet",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "parts_path": "dense_long_parts",
        "parts": [
            {
                "part_id": part.part_id,
                "path": os.path.relpath(part.path, out_dir).replace("\\", "/"),
                "first_feature_id": part.first_feature_id,
                "last_feature_id": part.last_feature_id,
                "feature_count": part.feature_count,
                "row_count": part.row_count,
                "byte_size": part.byte_size,
            }
            for part in parts
        ],
        "feature_locator_path": "feature_locator.parquet",
        "id_scheme": "dense_row_ids",
        "sample_key_col": str(sample_key_col),
        "feature_key_col": str(feature_key_col),
        "sample_id_col": str(sample_id_col),
        "feature_id_col": str(feature_id_col),
        "value_col": "value",
        "mask_col": "mask",
        "compression": str(compression),
        "row_group_features": int(row_group_features),
        "target_part_bytes": int(target_part_bytes),
        "selection_stats": selection_stats,
    }
    _write_json_atomic(manifest_path, manifest_payload)
    stats["write_manifest_s"] = time.perf_counter() - phase_t0
    stats["total_s"] = time.perf_counter() - total_t0

    if return_stats:
        return manifest_path, stats
    return manifest_path


def load_dense_long_manifest(manifest_path: str) -> ScalarDenseLongManifest:
    manifest_path = str(Path(manifest_path).expanduser().resolve())
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("format")) != DENSE_LONG_FORMAT_NAME:
        raise ValueError(f"unsupported dense-long manifest format: {data.get('format')}")
    out_dir = os.path.dirname(manifest_path)
    parts = [
        ScalarDenseLongPart(
            part_id=int(item["part_id"]),
            path=_resolve_manifest_relative(manifest_path, str(item["path"])),
            first_feature_id=int(item["first_feature_id"]),
            last_feature_id=int(item["last_feature_id"]),
            feature_count=int(item["feature_count"]),
            row_count=int(item["row_count"]),
            byte_size=int(item["byte_size"]),
        )
        for item in data.get("parts", [])
    ]
    selection_stats = {
        str(key): _resolve_manifest_relative(manifest_path, str(value))
        for key, value in (data.get("selection_stats") or {}).items()
    }
    return ScalarDenseLongManifest(
        manifest_path=manifest_path,
        sample_meta_path=_resolve_manifest_relative(manifest_path, str(data["sample_meta_path"])),
        feature_meta_path=_resolve_manifest_relative(manifest_path, str(data["feature_meta_path"])),
        n_samples=int(data["n_samples"]),
        n_features=int(data["n_features"]),
        parts_path=_resolve_manifest_relative(manifest_path, str(data["parts_path"])),
        parts=parts,
        feature_locator_path=_resolve_manifest_relative(manifest_path, str(data["feature_locator_path"])),
        sample_key_col=str(data.get("sample_key_col", "sample_key")),
        feature_key_col=str(data.get("feature_key_col", "feature_key")),
        sample_id_col=str(data.get("sample_id_col", "sample_id")),
        feature_id_col=str(data.get("feature_id_col", "feature_id")),
        value_col=str(data.get("value_col", "value")),
        mask_col=str(data.get("mask_col", "mask")),
        compression=str(data.get("compression", "zstd")),
        target_part_bytes=data.get("target_part_bytes"),
        selection_stats=selection_stats or None,
    )


class ScalarDenseLongDataset:
    """Reader for dense-long scalar shard artifacts."""

    def __init__(self, manifest_path: str):
        self.manifest = load_dense_long_manifest(manifest_path)
        self._locator = None
        self._sample_keys = None
        self._sample_key_to_id = None
        self._feature_keys = None
        self._feature_key_to_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _locator_df(self) -> pl.DataFrame:
        if self._locator is None:
            self._locator = pl.read_parquet(self.manifest.feature_locator_path)
        return self._locator

    @property
    def n_samples(self) -> int:
        return int(self.manifest.n_samples)

    @property
    def feature_count(self) -> int:
        return int(self.manifest.n_features)

    @property
    def n_shards(self) -> int:
        return int(len(self.manifest.parts))

    def feature_ids(self):
        return tuple(range(int(self.manifest.n_features)))

    def sample_ids(self):
        return tuple(range(int(self.manifest.n_samples)))

    def _load_sample_keys(self):
        if self._sample_keys is not None:
            return
        key_col = str(self.manifest.sample_key_col)
        df = pl.read_parquet(self.manifest.sample_meta_path, columns=[key_col])
        keys = tuple(None if value is None else str(value) for value in df[key_col].to_list())
        self._sample_keys = keys
        self._sample_key_to_id = {str(key): idx for idx, key in enumerate(keys) if key is not None}

    def _load_feature_keys(self):
        if self._feature_keys is not None:
            return
        key_col = str(self.manifest.feature_key_col)
        df = pl.read_parquet(self.manifest.feature_meta_path, columns=[key_col])
        keys = tuple(None if value is None else str(value) for value in df[key_col].to_list())
        self._feature_keys = keys
        self._feature_key_to_id = {str(key): idx for idx, key in enumerate(keys) if key is not None}

    def sample_keys(self):
        self._load_sample_keys()
        return self._sample_keys

    def feature_keys(self):
        self._load_feature_keys()
        return self._feature_keys

    def resolve_sample_key(self, sample_key: str) -> int:
        self._load_sample_keys()
        sample_id = self._sample_key_to_id.get(str(sample_key))
        if sample_id is None:
            raise KeyError(f"sample key not found: {sample_key}")
        return int(sample_id)

    def resolve_feature_key(self, feature_key: str) -> int:
        self._load_feature_keys()
        feature_id = self._feature_key_to_id.get(str(feature_key))
        if feature_id is None:
            raise KeyError(f"feature key not found: {feature_key}")
        return int(feature_id)

    def _part_path_for_feature(self, feature_id: int) -> str:
        feature_id = int(feature_id)
        if feature_id < 0 or feature_id >= int(self.manifest.n_features):
            raise KeyError(f"feature_id out of range: {feature_id}")
        loc = self._locator_df().filter(pl.col("feature_id") == feature_id)
        if loc.height != 1:
            raise KeyError(f"feature_id not found: {feature_id}")
        part_id = int(loc["part_id"][0])
        return self.manifest.parts[part_id].path

    def load_feature_by_id(self, feature_id: int):
        part_path = self._part_path_for_feature(int(feature_id))
        df = (
            pl.scan_parquet(part_path)
            .filter(pl.col("feature_id") == int(feature_id))
            .select(["sample_id", "mask", "value"])
            .sort("sample_id")
            .collect()
        )
        values = np.zeros(int(self.manifest.n_samples), dtype=np.float64)
        valid = np.zeros(int(self.manifest.n_samples), dtype=np.uint8)
        if df.height > 0:
            sample_ids = df["sample_id"].to_numpy().astype(np.int64, copy=False)
            values[sample_ids] = df["value"].to_numpy().astype(np.float64, copy=False)
            valid[sample_ids] = df["mask"].to_numpy().astype(np.uint8, copy=False)
        return values, valid

    def load_feature_by_key(self, feature_key: str):
        return self.load_feature_by_id(self.resolve_feature_key(feature_key))

    def load_features_by_ids(self, feature_ids):
        feature_ids = [int(value) for value in feature_ids]
        values = np.zeros((len(feature_ids), int(self.manifest.n_samples)), dtype=np.float64)
        valid = np.zeros((len(feature_ids), int(self.manifest.n_samples)), dtype=np.uint8)
        if not feature_ids:
            return values, valid

        order_by_feature = {feature_id: idx for idx, feature_id in enumerate(feature_ids)}
        locator = self._locator_df().filter(pl.col("feature_id").is_in(feature_ids))
        by_part: dict[int, list[int]] = {}
        for row in locator.iter_rows(named=True):
            by_part.setdefault(int(row["part_id"]), []).append(int(row["feature_id"]))

        for part_id, ids in by_part.items():
            df = (
                pl.scan_parquet(self.manifest.parts[int(part_id)].path)
                .filter(pl.col("feature_id").is_in(ids))
                .select(["feature_id", "sample_id", "mask", "value"])
                .collect()
            )
            for feature_id, group in df.group_by("feature_id", maintain_order=False):
                fid = int(feature_id[0] if isinstance(feature_id, tuple) else feature_id)
                out_idx = order_by_feature[fid]
                sample_ids = group["sample_id"].to_numpy().astype(np.int64, copy=False)
                values[out_idx, sample_ids] = group["value"].to_numpy().astype(np.float64, copy=False)
                valid[out_idx, sample_ids] = group["mask"].to_numpy().astype(np.uint8, copy=False)
        return values, valid

    def load_rows(self, part_id: int, offsets):
        """Load feature rows by dense-long part id and feature offset.

        The selection pipeline uses a generic `(shard_id, offset)` reader
        interface. For dense-long shards, `shard_id` means `part_id` and
        `offset` means the feature offset inside that part.
        """

        part = self.manifest.parts[int(part_id)]
        feature_ids = [int(part.first_feature_id) + int(offset) for offset in offsets]
        return self.load_features_by_ids(feature_ids)

    def load_feature_by_offset(self, part_id: int, offset: int):
        """Load one feature row by dense-long part id and offset."""

        values, valid = self.load_rows(int(part_id), [int(offset)])
        return values[0], valid[0]

    def load_sample_by_id(self, sample_id: int):
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= int(self.manifest.n_samples):
            raise KeyError(f"sample_id out of range: {sample_id}")
        df = (
            pl.scan_parquet([part.path for part in self.manifest.parts], glob=False)
            .filter(pl.col("sample_id") == sample_id)
            .select(["feature_id", "mask", "value"])
            .sort("feature_id")
            .collect()
        )
        values = np.zeros(int(self.manifest.n_features), dtype=np.float64)
        valid = np.zeros(int(self.manifest.n_features), dtype=np.uint8)
        if df.height > 0:
            feature_ids = df["feature_id"].to_numpy().astype(np.int64, copy=False)
            values[feature_ids] = df["value"].to_numpy().astype(np.float64, copy=False)
            valid[feature_ids] = df["mask"].to_numpy().astype(np.uint8, copy=False)
        return values, valid

    def load_sample_by_key(self, sample_key: str):
        return self.load_sample_by_id(self.resolve_sample_key(sample_key))

    def _sample_key_for_id(self, sample_id: int):
        self._load_sample_keys()
        if sample_id < 0 or sample_id >= len(self._sample_keys):
            return None
        return self._sample_keys[int(sample_id)]

    def _feature_key_for_id(self, feature_id: int):
        self._load_feature_keys()
        if feature_id < 0 or feature_id >= len(self._feature_keys):
            return None
        return self._feature_keys[int(feature_id)]

    def _value_model(self, feature_id: int, sample_id: int, values, valid, *, feature_key=None, sample_key=None):
        present = 0 <= sample_id < int(valid.shape[0]) and bool(valid[int(sample_id)])
        return ScalarValue(
            feature_id=int(feature_id),
            sample_id=int(sample_id),
            present=present,
            value=float(values[int(sample_id)]) if present else None,
            feature_key=self._feature_key_for_id(feature_id) if feature_key is None else feature_key,
            sample_key=self._sample_key_for_id(sample_id) if sample_key is None else sample_key,
        )

    def get_value(self, feature_id: int, sample_id: int, strict: bool = False) -> ScalarValue:
        if strict and (int(feature_id) < 0 or int(feature_id) >= int(self.manifest.n_features)):
            raise KeyError(f"feature id not found: {feature_id}")
        if strict and (int(sample_id) < 0 or int(sample_id) >= int(self.manifest.n_samples)):
            raise KeyError(f"sample id not found: {sample_id}")
        batch = self.get_values(feature_id=int(feature_id), sample_ids=[int(sample_id)], strict=False)
        return batch.values[0]

    def get_value_by_key(self, feature_key: str, sample_key: str, strict: bool = True) -> ScalarValue:
        del strict
        return self.get_value(self.resolve_feature_key(feature_key), self.resolve_sample_key(sample_key), strict=True)

    def get_values(self, feature_id: int, sample_ids, strict: bool = False) -> FeatureValues:
        if strict and (int(feature_id) < 0 or int(feature_id) >= int(self.manifest.n_features)):
            raise KeyError(f"feature id not found: {feature_id}")
        values, valid = self.load_features_by_ids([int(feature_id)])
        sample_ids = [int(value) for value in sample_ids]
        feature_key = self._feature_key_for_id(int(feature_id))
        items = [
            self._value_model(int(feature_id), sample_id, values[0], valid[0], feature_key=feature_key)
            for sample_id in sample_ids
        ]
        return FeatureValues(
            feature_id=int(feature_id),
            sample_ids=tuple(sample_ids),
            values=tuple(items),
            feature_key=feature_key,
            sample_keys=tuple(item.sample_key for item in items),
        )

    def get_values_by_key(self, feature_key: str, sample_keys, strict: bool = True) -> FeatureValues:
        del strict
        feature_id = self.resolve_feature_key(feature_key)
        sample_ids = [self.resolve_sample_key(value) for value in sample_keys]
        return self.get_values(feature_id, sample_ids, strict=True)

    def _ordered_feature_ids(self, feature_ids, maintain_order: bool):
        feature_ids = [int(value) for value in feature_ids]
        if maintain_order:
            return feature_ids
        locator = self._locator_df().filter(pl.col("feature_id").is_in(feature_ids))
        rank = {
            int(row["feature_id"]): (int(row["part_id"]), int(row["offset_in_part"]))
            for row in locator.iter_rows(named=True)
        }
        return sorted(feature_ids, key=lambda fid: rank.get(int(fid), (10**9, int(fid))))

    def iter_many(self, feature_ids, sample_ids, strict: bool = False, batch_size: int = 128, maintain_order: bool = True):
        del batch_size
        ordered = self._ordered_feature_ids(feature_ids, bool(maintain_order))
        for feature_id in ordered:
            yield self.get_values(feature_id, sample_ids, strict=bool(strict))

    def get_many(
        self,
        feature_ids,
        sample_ids,
        strict: bool = False,
        batch_size: int = 128,
        stream: bool = False,
        maintain_order: bool = True,
    ) -> QueryResult:
        ordered = self._ordered_feature_ids(feature_ids, bool(maintain_order))
        features_iter = self.iter_many(ordered, sample_ids, strict=bool(strict), batch_size=int(batch_size), maintain_order=True)
        self._load_feature_keys()
        self._load_sample_keys()
        return QueryResult(
            feature_ids=tuple(ordered),
            sample_ids=tuple(int(value) for value in sample_ids),
            features=features_iter if bool(stream) else tuple(features_iter),
            feature_keys=tuple(self._feature_key_for_id(feature_id) for feature_id in ordered),
            sample_keys=tuple(self._sample_key_for_id(int(sample_id)) for sample_id in sample_ids),
        )

    def iter_many_by_key(
        self,
        feature_keys,
        sample_keys,
        strict: bool = True,
        batch_size: int = 128,
        maintain_order: bool = True,
    ):
        del strict
        feature_ids = [self.resolve_feature_key(value) for value in feature_keys]
        sample_ids = [self.resolve_sample_key(value) for value in sample_keys]
        return self.iter_many(feature_ids, sample_ids, strict=True, batch_size=batch_size, maintain_order=maintain_order)

    def get_many_by_key(
        self,
        feature_keys,
        sample_keys,
        strict: bool = True,
        batch_size: int = 128,
        stream: bool = False,
        maintain_order: bool = True,
    ) -> QueryResult:
        del strict
        feature_ids = [self.resolve_feature_key(value) for value in feature_keys]
        sample_ids = [self.resolve_sample_key(value) for value in sample_keys]
        return self.get_many(
            feature_ids,
            sample_ids,
            strict=True,
            batch_size=batch_size,
            stream=stream,
            maintain_order=maintain_order,
        )

    def top_features_from_stats(self, y_col: str = "y", top_k: int = 256) -> pl.DataFrame:
        mapping = self.manifest.selection_stats or {}
        path = mapping.get(str(y_col))
        if not path:
            raise KeyError(f"selection stats not found for y column: {y_col}")
        return (
            pl.scan_parquet(path)
            .sort(["r2y", "feature_id"], descending=[True, False], nulls_last=True)
            .head(int(top_k))
            .collect()
        )


def open_dense_long_shard(manifest_path: str) -> ScalarDenseLongDataset:
    return ScalarDenseLongDataset(manifest_path)
