"""High-level builder facade for creating scalar shards from per-sample values."""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ..config import ScalarShardBuildOptions
from .parquet_storage import build_shards_from_sample_bundles


def _load_dense_metadata(
    meta_path: str,
    *,
    id_col: str,
    entity_name: str,
    key_col: str,
) -> pl.DataFrame:
    """Load dense metadata and validate row-order ids and optional keys."""

    df = pl.read_parquet(meta_path)
    dense_ids = np.arange(df.height, dtype=np.int64 if id_col == "sample_id" else np.int32)
    if id_col in df.columns:
        stored = df[id_col].to_numpy().astype(dense_ids.dtype, copy=False)
        if not np.array_equal(stored, dense_ids):
            raise ValueError(f"{entity_name} metadata {id_col} must equal dense row order 0..n-1")

    if key_col and key_col in df.columns:
        values = df[key_col].to_list()
        seen = set()
        for row_idx, value in enumerate(values):
            if value is None:
                raise ValueError(f"{entity_name} metadata {key_col} cannot be null at row {row_idx}")
            key = str(value)
            if key in seen:
                raise ValueError(f"duplicate {entity_name} {key_col}: {key}")
            seen.add(key)

    return df


def _prepare_empty_dir(path: str):
    """Create an empty directory or fail when it already contains files."""

    if os.path.exists(path):
        if os.path.isdir(path) and not os.listdir(path):
            return
        raise ValueError(f"sample_major_out_dir already exists and is not empty: {path}")
    os.makedirs(path, exist_ok=True)


@dataclass
class ScalarSampleContext:
    """Optional helper for sample-scoped scalar ingestion."""

    builder: "ScalarDatasetBuilder"
    sample_id: int

    def __enter__(self):
        """Open the sample-scoped context."""
        self.builder._begin_sample(self.sample_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        """Flush the sample when the context exits without swallowing errors."""
        self.builder._end_sample(abort=exc_type is not None)
        return False

    def write_value(self, feature, value):
        """Add one scalar feature value to the open sample.

        Args:
            feature: Dense `feature_id` or external `feature_key`.
            value: Scalar value to store for the feature. `None` or `NaN`
                becomes missing.
        """

        self.builder._append_open_sample_value(feature, value)

    def write_values(self, values: Mapping):
        """Add many scalar feature values to the open sample."""

        for feature_ref, value in dict(values).items():
            self.write_value(feature_ref, value)


class ScalarDatasetBuilder:
    """High-level builder that writes sample-bundle inputs and final scalar shards.

    The primary public API is sample-scoped on purpose:

    - `write_sample(sample_id, values=...)`
    - `with builder.open_sample(sample_id) as sample: ...`

    This keeps memory bounded and makes sample revisits impossible. Internally,
    completed samples are appended to bundle parquet files instead of creating
    one parquet file per sample.
    """

    _DEFAULT_BUNDLE_FLUSH_ROWS = 1_000_000

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        *,
        feature_meta_path: Optional[str] = None,
        feature_keys: Optional[Sequence[str]] = None,
        build_options: ScalarShardBuildOptions | None = None,
        sample_major_out_dir: Optional[str] = None,
    ):
        """Create a new scalar dataset builder."""

        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.source_sample_meta_path = str(Path(sample_meta_path).expanduser().resolve())
        self.build_options = build_options or ScalarShardBuildOptions()
        self._closed = False
        self._sample_major_finalized = False
        self._shards_built = False
        self._manifest_path = None

        sample_meta_df = _load_dense_metadata(
            self.source_sample_meta_path,
            id_col=str(self.build_options.sample_id_col),
            entity_name="sample",
            key_col=str(self.build_options.sample_key_col),
        )
        for y_col in self._stats_y_cols():
            if y_col not in sample_meta_df.columns:
                raise ValueError(f"sample metadata is missing required target column: {y_col}")
        self._source_sample_meta_df = sample_meta_df
        self.n_samples = int(sample_meta_df.height)

        default_sample_major_root = os.path.join(self.out_dir, "sample_major_stage")
        self.sample_major_out_dir = str(Path(sample_major_out_dir or default_sample_major_root).expanduser().resolve())
        _prepare_empty_dir(self.sample_major_out_dir)
        self._bundle_files_dir = os.path.join(self.sample_major_out_dir, "sample_bundles")
        os.makedirs(self._bundle_files_dir, exist_ok=True)

        self.sample_major_manifest_path = os.path.join(self.sample_major_out_dir, "sample_major_manifest.json")
        self.sample_major_sample_meta_path = os.path.join(self.sample_major_out_dir, "sample_meta.parquet")
        self.sample_major_feature_meta_path = os.path.join(self.sample_major_out_dir, "feature_meta.parquet")

        self._feature_meta_source_path = None
        self._feature_key_to_id: OrderedDict[str, int] = OrderedDict()
        self._feature_keys_in_order: list[str] = []
        self._known_feature_mode = False
        self._known_feature_count: Optional[int] = None
        self._writes_feature_meta = False

        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")

        if feature_meta_path:
            self._known_feature_mode = True
            self._feature_meta_source_path = str(Path(feature_meta_path).expanduser().resolve())
            feature_meta_df = _load_dense_metadata(
                self._feature_meta_source_path,
                id_col=str(self.build_options.feature_id_col),
                entity_name="feature",
                key_col=str(self.build_options.feature_key_col),
            )
            self._known_feature_count = int(feature_meta_df.height)
            key_col = str(self.build_options.feature_key_col)
            if key_col and key_col in feature_meta_df.columns:
                for idx, value in enumerate(feature_meta_df[key_col].to_list()):
                    key = str(value)
                    self._feature_key_to_id[key] = int(idx)
                    self._feature_keys_in_order.append(key)
        elif feature_keys is not None:
            self._known_feature_mode = True
            self._writes_feature_meta = True
            for idx, feature_key in enumerate(feature_keys):
                key = str(feature_key)
                if key in self._feature_key_to_id:
                    raise ValueError(f"duplicate feature key: {key}")
                self._feature_key_to_id[key] = int(idx)
                self._feature_keys_in_order.append(key)
            self._known_feature_count = int(len(self._feature_keys_in_order))
        else:
            self._writes_feature_meta = True

        self._sample_written = np.zeros(self.n_samples, dtype=bool)
        self._open_sample_id: Optional[int] = None
        self._open_sample_values: Optional[dict[int, float]] = None

        self._bundle_paths: list[str] = []
        self._bundle_index = 0
        self._bundle_sample_id_chunks: list[np.ndarray] = []
        self._bundle_feature_id_chunks: list[np.ndarray] = []
        self._bundle_value_chunks: list[np.ndarray] = []
        self._bundle_row_count = 0
        self._bundle_flush_rows_target = int(self._DEFAULT_BUNDLE_FLUSH_ROWS)

    def _stats_y_cols(self) -> list[str]:
        """Return the unique ordered list of target columns to precompute."""

        values = self.build_options.stats_y_cols
        if not values:
            return [str(self.build_options.y_col)]
        ordered: list[str] = []
        for value in values:
            name = str(value)
            if name and name not in ordered:
                ordered.append(name)
        if not ordered:
            raise ValueError("at least one stats y column is required")
        return ordered

    def _ensure_open(self):
        """Raise if the builder has already been closed or finished."""

        if self._closed:
            raise RuntimeError("scalar dataset builder is closed")
        if self._shards_built:
            raise RuntimeError("scalar dataset builder has already built shards")

    def _ensure_sample_major_open(self):
        """Raise if sample-major ingestion is no longer allowed."""

        self._ensure_open()
        if self._sample_major_finalized:
            raise RuntimeError("sample-major stage has already been finalized")

    def _normalize_scalar_value(self, value) -> Optional[float]:
        """Normalize one scalar value or treat `None`/`NaN` as missing."""

        if value is None:
            return None
        scalar = float(value)
        if np.isnan(scalar):
            return None
        return scalar

    def _resolve_feature_id(self, feature_ref) -> int:
        """Resolve one feature reference into a dense feature id."""

        if isinstance(feature_ref, (int, np.integer)):
            resolved = int(feature_ref)
            if resolved < 0:
                raise ValueError(f"feature_id out of range: {resolved}")
            if not self._known_feature_mode:
                raise ValueError("feature_id cannot be used in discovered-feature mode; use feature_key instead")
            if self._known_feature_count is not None and resolved >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {resolved}")
            return resolved

        key = str(feature_ref)
        resolved = self._feature_key_to_id.get(key)
        if resolved is not None:
            return int(resolved)
        if self._known_feature_mode:
            raise ValueError(f"unknown feature key: {key}")
        resolved = len(self._feature_keys_in_order)
        self._feature_key_to_id[key] = int(resolved)
        self._feature_keys_in_order.append(key)
        return int(resolved)

    def _flush_bundle(self):
        """Write the currently buffered scalar rows into one bundle parquet file."""

        if self._bundle_row_count <= 0:
            return
        bundle_path = os.path.join(self._bundle_files_dir, f"bundle_{self._bundle_index:06d}.parquet")
        sample_ids = np.concatenate(self._bundle_sample_id_chunks).astype(np.int64, copy=False)
        feature_ids = np.concatenate(self._bundle_feature_id_chunks).astype(np.int32, copy=False)
        values = np.concatenate(self._bundle_value_chunks).astype(np.float64, copy=False)
        df = pl.DataFrame(
            {
                str(self.build_options.sample_id_col): pl.Series(
                    str(self.build_options.sample_id_col), sample_ids, dtype=pl.Int64
                ),
                str(self.build_options.feature_id_col): pl.Series(
                    str(self.build_options.feature_id_col), feature_ids, dtype=pl.Int32
                ),
                str(self.build_options.value_col): pl.Series(
                    str(self.build_options.value_col), values, dtype=pl.Float64
                ),
            }
        )
        df.write_parquet(bundle_path)
        self._bundle_paths.append(bundle_path)
        self._bundle_index += 1
        self._bundle_sample_id_chunks.clear()
        self._bundle_feature_id_chunks.clear()
        self._bundle_value_chunks.clear()
        self._bundle_row_count = 0

    def _append_sample_rows(self, sample_id: int, feature_values: Mapping[int, float]):
        """Append one finished sample into the current bundle buffer."""

        if feature_values:
            feature_ids = np.asarray(sorted(feature_values.keys()), dtype=np.int32)
            values = np.asarray([float(feature_values[int(fid)]) for fid in feature_ids], dtype=np.float64)
            sample_ids = np.full(feature_ids.shape[0], int(sample_id), dtype=np.int64)
            self._bundle_sample_id_chunks.append(sample_ids)
            self._bundle_feature_id_chunks.append(feature_ids)
            self._bundle_value_chunks.append(values)
            self._bundle_row_count += int(feature_ids.shape[0])
            if self._bundle_row_count >= int(self._bundle_flush_rows_target):
                self._flush_bundle()
        self._sample_written[int(sample_id)] = True

    def _copy_sample_meta(self):
        """Copy the source sample metadata into the visible stage unchanged."""

        if os.path.normcase(os.path.abspath(self.sample_major_sample_meta_path)) != os.path.normcase(
            os.path.abspath(self.source_sample_meta_path)
        ):
            shutil.copy2(self.source_sample_meta_path, self.sample_major_sample_meta_path)

    def _write_feature_meta(self):
        """Write or copy feature metadata for the sample-major stage."""

        if self._feature_meta_source_path:
            shutil.copy2(self._feature_meta_source_path, self.sample_major_feature_meta_path)
            return
        if not self._writes_feature_meta:
            return
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {
            str(self.build_options.feature_id_col): pl.Series(
                str(self.build_options.feature_id_col),
                feature_ids,
                dtype=pl.Int32,
            )
        }
        key_col = str(self.build_options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self.sample_major_feature_meta_path)

    def _write_sample_major_manifest(self):
        """Write the visible sample-bundle manifest for the intermediate stage."""

        data = {
            "format": "scalar-sample-bundles",
            "version": 1,
            "sample_meta_path": "sample_meta.parquet",
            "feature_meta_path": "feature_meta.parquet",
            "bundle_paths": [
                os.path.join("sample_bundles", os.path.basename(path))
                for path in self._bundle_paths
            ],
            "sample_id_col": str(self.build_options.sample_id_col),
            "feature_id_col": str(self.build_options.feature_id_col),
            "value_col": str(self.build_options.value_col),
        }
        with open(self.sample_major_manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _begin_sample(self, sample_id: int):
        """Open one sample-scoped buffer."""

        self._ensure_sample_major_open()
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= self.n_samples:
            raise ValueError(f"sample_id out of range: {sample_id}")
        if self._sample_written[sample_id]:
            raise ValueError(f"sample_id has already been written and cannot be revisited: {sample_id}")
        if self._open_sample_id is not None:
            raise RuntimeError("another sample context is already open")
        self._open_sample_id = sample_id
        self._open_sample_values = {}

    def _append_open_sample_value(self, feature_ref, value):
        """Append one feature value to the currently open sample."""

        if self._open_sample_id is None or self._open_sample_values is None:
            raise RuntimeError("no sample context is currently open")
        normalized = self._normalize_scalar_value(value)
        resolved_feature_id = self._resolve_feature_id(feature_ref)
        if resolved_feature_id in self._open_sample_values:
            raise ValueError(
                f"duplicate feature assignment within sample {self._open_sample_id}: feature_id={resolved_feature_id}"
            )
        if normalized is not None:
            self._open_sample_values[int(resolved_feature_id)] = float(normalized)

    def _end_sample(self, *, abort: bool):
        """Flush or discard the currently open sample buffer."""

        if self._open_sample_id is None:
            return
        sample_id = int(self._open_sample_id)
        values = dict(self._open_sample_values or {})
        self._open_sample_id = None
        self._open_sample_values = None
        if abort:
            return
        self._append_sample_rows(sample_id, values)

    def write_sample(self, sample_id: int, values: Mapping):
        """Write one complete sample and flush it immediately.

        This is the recommended public API because it keeps memory bounded and
        makes sample revisits impossible: each sample can be written exactly once.
        """

        self._begin_sample(int(sample_id))
        try:
            for feature_ref, value in dict(values).items():
                self._append_open_sample_value(feature_ref, value)
        except Exception:
            self._end_sample(abort=True)
            raise
        self._end_sample(abort=False)

    def open_sample(self, sample_id: int) -> ScalarSampleContext:
        """Return a sample-scoped context manager for incremental value addition."""

        return ScalarSampleContext(self, int(sample_id))

    def update_feature_meta(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        on: Optional[str] = None,
        require_all: bool = False,
    ) -> str:
        """Merge extra feature metadata columns into the stage feature metadata."""

        self.finish_sample_major()
        base = pl.read_parquet(self.sample_major_feature_meta_path)
        join_col = str(on or "")
        if not join_col:
            key_col = str(self.build_options.feature_key_col)
            join_col = key_col if key_col and key_col in base.columns else str(self.build_options.feature_id_col)
        if join_col not in base.columns:
            raise ValueError(f"feature metadata join column not found: {join_col}")

        rows = [dict(record) for record in records]
        if not rows:
            return self.sample_major_feature_meta_path
        updates = pl.from_dicts(rows, infer_schema_length=None)
        if join_col not in updates.columns:
            raise ValueError(f"update records must include join column: {join_col}")

        seen = set()
        for row_idx, value in enumerate(updates[join_col].to_list()):
            if value is None:
                raise ValueError(f"feature metadata join column {join_col} cannot be null at row {row_idx}")
            key = int(value) if join_col == str(self.build_options.feature_id_col) else str(value)
            if key in seen:
                raise ValueError(f"duplicate feature metadata join key: {value}")
            seen.add(key)

        overlapping = [name for name in updates.columns if name != join_col and name in base.columns]
        if overlapping:
            raise ValueError(
                f"feature metadata updates must add new columns only; overlapping columns: {', '.join(overlapping)}"
            )

        merged = base.join(updates, on=join_col, how="left")
        if require_all:
            new_columns = [name for name in updates.columns if name != join_col]
            for column_name in new_columns:
                if merged[column_name].null_count() > 0:
                    raise ValueError(f"missing values remain in required feature metadata column: {column_name}")

        merged.write_parquet(self.sample_major_feature_meta_path)
        return self.sample_major_feature_meta_path

    def finish_sample_major(self):
        """Finalize the visible sample-major stage."""

        if self._sample_major_finalized:
            return self.sample_major_manifest_path
        self._ensure_open()
        if self._open_sample_id is not None:
            self._end_sample(abort=False)

        self._flush_bundle()
        self._write_feature_meta()
        self._copy_sample_meta()
        self._write_sample_major_manifest()
        self._sample_major_finalized = True
        return self.sample_major_manifest_path

    def build_shards(self, *, keep_sample_major: bool = False, return_stats: bool = False):
        """Build the final scalar shard artifact from the sample-major stage.

        Notes:
            When `keep_sample_major=False`, the visible stage directory is
            deleted after shard construction.
        """

        if self._shards_built:
            return (self._manifest_path, None) if return_stats else self._manifest_path
        self._ensure_open()
        sample_major_manifest_path = self.finish_sample_major()
        build_result = build_shards_from_sample_bundles(
            sample_major_manifest_path,
            self.out_dir,
            feature_meta_path=self.sample_major_feature_meta_path,
            n_shards=None if self.build_options.n_shards is None else int(self.build_options.n_shards),
            target_shard_bytes=int(self.build_options.target_shard_mb) * 1024 * 1024,
            feature_id_col=str(self.build_options.feature_id_col),
            value_col=str(self.build_options.value_col),
            sample_id_col=str(self.build_options.sample_id_col),
            sample_key_col=str(self.build_options.sample_key_col),
            feature_key_col=str(self.build_options.feature_key_col),
            path_col=str(self.build_options.path_col),
            y_col=str(self.build_options.y_col),
            stats_y_cols=self._stats_y_cols(),
            values_dtype=str(self.build_options.values_dtype),
            valid_dtype=str(self.build_options.valid_dtype),
            return_stats=bool(return_stats),
        )
        if return_stats:
            manifest_path, build_stats = build_result
        else:
            manifest_path = build_result
            build_stats = None
        if (not keep_sample_major) and os.path.exists(self.sample_major_out_dir):
            shutil.rmtree(self.sample_major_out_dir)
        self._manifest_path = str(manifest_path)
        self._shards_built = True
        self._closed = True
        if return_stats:
            return self._manifest_path, build_stats
        return self._manifest_path

    def close(self):
        """Close the builder and drop partial sample-major output when unfinished."""

        if self._closed:
            return
        self._end_sample(abort=True)
        if not self._sample_major_finalized and os.path.exists(self.sample_major_out_dir):
            shutil.rmtree(self.sample_major_out_dir)
        self._closed = True

    def __enter__(self):
        """Return the builder itself for `with` usage."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the builder when leaving the context."""

        self.close()
        return False
