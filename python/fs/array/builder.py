"""High-level builder facade for creating array binary shards from traces."""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ..config import ArrayBinaryBuildOptions, ArrayBundleConfig, ArrayShardConfig
from ..types import LogicalType, PointColumnSpec, point_storage_dtype
from .binary_storage import _load_dense_meta, build_array_binary_shards_from_bundles
from .storage import ArraySampleBundleWriter, _normalize_point_schema


@dataclass
class _CategoricalRegistry:
    """Mutable string-to-code mapping for one categorical point column."""

    label_to_code: dict
    code_to_label: list

    @classmethod
    def create(cls):
        """Create an empty categorical registry."""
        return cls(label_to_code={}, code_to_label=[])

    def encode(self, values) -> np.ndarray:
        """Encode one categorical trace column into uint32 codes.

        Args:
            values: Sequence of labels for one point column.

        Returns:
            A `uint32` array where `0` is reserved for null values and all other
            labels are assigned first-seen integer codes.
        """
        arr = np.asarray(values, dtype=object)
        out = np.zeros(int(arr.size), dtype=np.uint32)
        for idx, value in enumerate(arr.tolist()):
            if value is None:
                out[idx] = np.uint32(0)
                continue
            label = str(value)
            code = self.label_to_code.get(label)
            if code is None:
                code = len(self.code_to_label) + 1
                self.label_to_code[label] = int(code)
                self.code_to_label.append(label)
            out[idx] = np.uint32(code)
        return out

    def to_frame(self) -> pl.DataFrame:
        """Materialize the registry as a parquet-ready dictionary table.

        Returns:
            A Polars dataframe with `code` and `label` columns suitable for
            writing as a categorical dictionary sidecar.
        """
        codes = np.arange(1, len(self.code_to_label) + 1, dtype=np.uint32)
        return pl.DataFrame(
            {
                "code": pl.Series("code", codes, dtype=pl.UInt32),
                "label": pl.Series("label", list(self.code_to_label), dtype=pl.String),
            }
        )


class ArraySampleContext:
    """Optional helper for sample-scoped trace ingestion."""

    def __init__(self, builder: "ArrayDatasetBuilder", sample_id: int):
        """Bind one builder and one dense sample id."""
        self._builder = builder
        self._sample_id = int(sample_id)

    def __enter__(self):
        """Return the sample-scoped helper for `with` usage."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Do not swallow exceptions raised inside the sample context."""
        return False

    def add_trace(self, feature_id: Optional[int] = None, feature_key: Optional[str] = None, *, columns):
        """Append one trace for the bound sample id.

        Args:
            feature_id: Optional dense feature id.
            feature_key: Optional external feature key.
            columns: Explicit point-column mapping.
        """
        self._builder.add_trace(
            sample_id=self._sample_id,
            feature_id=feature_id,
            feature_key=feature_key,
            columns=columns,
        )


class ArrayDatasetBuilder:
    """High-level builder that turns traces directly into binary array shards.

    The builder accepts one trace at a time, buffers them into sample-major
    bundle parquet files, and later turns those bundles into the final binary
    shard artifact.

    Feature ids can be supplied in two modes:

    - known-feature mode:
      pass `feature_meta_path` or `feature_keys` up front
    - discovered-feature mode:
      omit both and always add traces by `feature_key`; dense feature ids are
      assigned in first-seen order
    """

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        point_schema,
        *,
        feature_meta_path: Optional[str] = None,
        feature_keys: Optional[Sequence[str]] = None,
        build_options: ArrayBinaryBuildOptions | None = None,
        bundle_config: ArrayBundleConfig | None = None,
        bundle_out_dir: Optional[str] = None,
    ):
        """Create a new direct-ingestion array dataset builder.

        Args:
            out_dir: Final output directory for the standalone binary artifact.
            sample_meta_path: Path to dense sample metadata parquet.
            point_schema: Manifest-wide fixed point-column schema.
            feature_meta_path: Optional dense feature metadata parquet for
                known-feature mode.
            feature_keys: Optional external feature keys for known-feature mode
                when a feature metadata file does not already exist.
            build_options: High-level binary shard build options.
            bundle_config: Optional internal bundle flush thresholds.
            bundle_out_dir: Optional directory where the intermediate bundle
                artifact should be materialized. When omitted, a visible sibling
                directory under `out_dir` is used.
        """
        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.sample_meta_path = str(Path(sample_meta_path).expanduser().resolve())
        self.build_options = build_options or ArrayBinaryBuildOptions()
        self.bundle_config = bundle_config or ArrayBundleConfig()
        self.point_schema = _normalize_point_schema(point_schema)
        self._closed = False
        self._finished = False
        self._bundles_finalized = False
        self._manifest_path = None

        sample_meta = _load_dense_meta(
            self.sample_meta_path,
            "sample_id",
            "sample",
            str(self.build_options.sample_key_col),
        )
        self.n_samples = int(sample_meta.height)

        default_bundle_root = os.path.join(self.out_dir, "bundle_stage")
        self.bundle_out_dir = str(Path(bundle_out_dir or default_bundle_root).expanduser().resolve())
        self._prepare_empty_dir(self.bundle_out_dir)

        self._bundle_sample_meta_path = os.path.join(self.bundle_out_dir, "sample_meta.parquet")
        shutil.copy2(self.sample_meta_path, self._bundle_sample_meta_path)

        self._feature_meta_path = os.path.join(self.bundle_out_dir, "feature_meta.parquet")
        self._feature_key_to_id = OrderedDict()
        self._feature_keys_in_order = []
        self._known_feature_mode = False
        self._known_feature_count = None
        self._writes_feature_meta = False

        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")

        if feature_meta_path:
            self._known_feature_mode = True
            feature_meta_source_path = str(Path(feature_meta_path).expanduser().resolve())
            feature_meta = _load_dense_meta(
                feature_meta_source_path,
                "feature_id",
                "feature",
                str(self.build_options.feature_key_col),
            )
            shutil.copy2(feature_meta_source_path, self._feature_meta_path)
            self._known_feature_count = int(feature_meta.height)
            key_col = str(self.build_options.feature_key_col)
            if key_col and key_col in feature_meta.columns:
                for idx, value in enumerate(feature_meta[key_col].to_list()):
                    self._feature_key_to_id[str(value)] = int(idx)
                    self._feature_keys_in_order.append(str(value))
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

        self._categorical_registries = {
            spec.name: _CategoricalRegistry.create()
            for spec in self.point_schema
            if spec.logical_type == LogicalType.CATEGORICAL
        }

        self._bundle_writer = ArraySampleBundleWriter(
            self.bundle_out_dir,
            self._bundle_sample_meta_path,
            feature_meta_path=self._feature_meta_path,
            n_samples=self.n_samples,
            config=self.bundle_config,
            point_schema=self.point_schema,
        )
        self.bundle_manifest_path = self._bundle_writer.manifest_path
        self.bundle_path = self._bundle_writer.bundle_path

    @staticmethod
    def _prepare_empty_dir(path: str):
        """Create an empty output directory or fail if it already has contents."""
        if os.path.exists(path):
            if os.path.isdir(path) and not os.listdir(path):
                return
            raise ValueError(f"bundle_out_dir already exists and is not empty: {path}")
        os.makedirs(path, exist_ok=True)

    def _ensure_open(self):
        """Raise when the builder has already been closed or finished."""
        if self._closed:
            raise RuntimeError("array dataset builder is closed")
        if self._finished:
            raise RuntimeError("array dataset builder is already finished")

    def _ensure_trace_stage_open(self):
        """Raise when trace ingestion is no longer allowed."""
        self._ensure_open()
        if self._bundles_finalized:
            raise RuntimeError("bundle stage has already been finalized")

    def close(self):
        """Close the builder.

        When the bundle stage has not yet been finalized, this also removes the
        partially written bundle artifact directory.
        """
        if self._closed:
            return
        try:
            if not self._bundles_finalized and os.path.exists(self.bundle_out_dir):
                shutil.rmtree(self.bundle_out_dir)
        finally:
            self._closed = True

    def __enter__(self):
        """Return the builder for `with` usage."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Finalize shards on clean exit and clean up partial bundle state on failure."""
        if exc_type is not None:
            self.close()
            return False
        if not self._finished:
            self.build_shards()
        return False

    def sample(self, sample_id: int) -> ArraySampleContext:
        """Return a sample-scoped helper object.

        Args:
            sample_id: Dense sample id for the nested context.

        Returns:
            A helper that automatically fills `sample_id` on `add_trace(...)`.
        """
        self._ensure_trace_stage_open()
        return ArraySampleContext(self, sample_id)

    def _resolve_feature_id(self, feature_id: Optional[int], feature_key: Optional[str]) -> int:
        """Resolve one user-supplied feature reference into a dense feature id."""
        if feature_id is None and feature_key is None:
            raise ValueError("provide either feature_id or feature_key")
        if feature_id is not None and feature_key is not None:
            resolved = self._resolve_feature_id(None, feature_key)
            if int(feature_id) != int(resolved):
                raise ValueError(f"feature_id/feature_key mismatch: {feature_id} != {feature_key}")
            return int(feature_id)
        if feature_id is not None:
            feature_id = int(feature_id)
            if feature_id < 0:
                raise ValueError("feature_id must be >= 0")
            if self._known_feature_mode and self._known_feature_count is not None and feature_id >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {feature_id}")
            if not self._known_feature_mode:
                raise ValueError("discovered-feature mode requires feature_key inputs")
            return feature_id

        key = str(feature_key)
        resolved = self._feature_key_to_id.get(key)
        if resolved is not None:
            return int(resolved)
        if self._known_feature_mode:
            raise ValueError(f"unknown feature key: {key}")
        resolved = len(self._feature_keys_in_order)
        self._feature_key_to_id[key] = int(resolved)
        self._feature_keys_in_order.append(key)
        return int(resolved)

    def _normalize_columns(self, *, columns):
        """Normalize one trace into typed 1D NumPy arrays for every point column."""
        if columns is None:
            raise ValueError("columns is required")
        expected_names = {spec.name for spec in self.point_schema}
        actual_names = {str(name) for name in dict(columns).keys()}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"point columns must exactly match point_schema ({', '.join(details)})")
        out = {}
        expected_length = None
        for spec in self.point_schema:
            if spec.name not in columns:
                raise ValueError(f"missing point column: {spec.name}")
            values = columns[spec.name]
            logical_type = spec.logical_type
            if logical_type == LogicalType.CATEGORICAL:
                registry = self._categorical_registries[spec.name]
                arr = registry.encode(values)
            elif logical_type == LogicalType.TIMESTAMP_NS:
                arr = np.asarray(values, dtype="datetime64[ns]").reshape(-1).astype(point_storage_dtype(spec.storage_type), copy=False)
            elif logical_type == LogicalType.TIMEDELTA_NS:
                arr = np.asarray(values, dtype="timedelta64[ns]").reshape(-1).astype(point_storage_dtype(spec.storage_type), copy=False)
            else:
                arr = np.asarray(values, dtype=point_storage_dtype(spec.storage_type))
            if arr.ndim != 1:
                raise ValueError(f"point column must be 1D: {spec.name}")
            if expected_length is None:
                expected_length = int(arr.shape[0])
            elif int(arr.shape[0]) != int(expected_length):
                raise ValueError(
                    f"point column length mismatch for {spec.name}: expected={expected_length} got={arr.shape[0]}"
                )
            out[spec.name] = arr
        return out

    def add_trace(
        self,
        *,
        sample_id: int,
        feature_id: Optional[int] = None,
        feature_key: Optional[str] = None,
        columns,
    ):
        """Append one trace into the bundle stage.

        Args:
            sample_id: Dense sample id.
            feature_id: Optional dense feature id.
            feature_key: Optional external feature key.
            columns: Full point-column mapping.
        """
        self._ensure_trace_stage_open()
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= int(self.n_samples):
            raise ValueError(f"sample_id out of range: {sample_id}")
        resolved_feature_id = self._resolve_feature_id(feature_id, feature_key)
        normalized_columns = self._normalize_columns(columns=columns)
        self._bundle_writer.append_trace(
            sample_row=sample_id,
            sample_id=sample_id,
            feature_id=resolved_feature_id,
            columns=normalized_columns,
        )

    def _write_feature_meta(self):
        """Write generated dense feature metadata when the builder owns it."""
        if not self._writes_feature_meta:
            return
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {
            "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
        }
        key_col = str(self.build_options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self._feature_meta_path)

    def _write_categorical_dictionaries(self):
        """Write generated categorical dictionary parquet files into the bundle artifact."""
        dict_root = os.path.join(self.bundle_out_dir, "categorical_dictionaries")
        os.makedirs(dict_root, exist_ok=True)
        for idx, spec in enumerate(self.point_schema):
            if spec.logical_type != LogicalType.CATEGORICAL:
                continue
            registry = self._categorical_registries[spec.name]
            dict_path = os.path.join(dict_root, f"{spec.name}.parquet")
            registry.to_frame().write_parquet(dict_path)
            spec.dictionary_path = dict_path
            self._bundle_writer.point_schema[idx].dictionary_path = dict_path

    def update_feature_meta(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        on: Optional[str] = None,
        require_all: bool = False,
    ) -> str:
        """Merge extra feature metadata columns into the bundle-stage feature metadata.

        This method finalizes the bundle stage first when needed, so discovered
        feature ids are frozen before metadata enrichment happens.

        Args:
            records: Extra feature metadata rows. Each row must contain the join
                column specified by `on`.
            on: Join column name. Defaults to the configured feature key column
                when present, otherwise `feature_id`.
            require_all: When true, require every feature row in the current
                metadata table to receive a non-null value for each new column.

        Returns:
            Absolute path to the updated `feature_meta.parquet`.
        """
        self.finish_bundles()
        base = pl.read_parquet(self._feature_meta_path)
        join_col = str(on or "")
        if not join_col:
            key_col = str(self.build_options.feature_key_col)
            join_col = key_col if key_col and key_col in base.columns else "feature_id"
        if join_col not in base.columns:
            raise ValueError(f"feature metadata join column not found: {join_col}")

        rows = [dict(record) for record in records]
        if not rows:
            return self._feature_meta_path
        updates = pl.from_dicts(rows, infer_schema_length=None)
        if join_col not in updates.columns:
            raise ValueError(f"update records must include join column: {join_col}")

        update_keys = updates[join_col].to_list()
        seen_keys = set()
        for row_idx, value in enumerate(update_keys):
            if value is None:
                raise ValueError(f"feature metadata join column {join_col} cannot be null at row {row_idx}")
            key = str(value) if join_col != "feature_id" else int(value)
            if key in seen_keys:
                raise ValueError(f"duplicate feature metadata join key: {value}")
            seen_keys.add(key)

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

        merged.write_parquet(self._feature_meta_path)
        return self._feature_meta_path

    def finish_bundles(self):
        """Finalize the intermediate sample-major bundle artifact.

        Returns:
            Path to the generated `array_bundle_manifest.json`.
        """
        if self._bundles_finalized:
            return self.bundle_manifest_path
        self._ensure_open()
        self._write_feature_meta()
        self._write_categorical_dictionaries()
        self.bundle_manifest_path = self._bundle_writer.finish()
        self.bundle_path = self._bundle_writer.bundle_path
        self._bundles_finalized = True
        return self.bundle_manifest_path

    def build_shards(self, *, cleanup_bundles: bool = False, return_stats: bool = False):
        """Convert the finalized bundle stage into the final binary shard artifact.

        Args:
            cleanup_bundles: Whether to delete the intermediate bundle artifact
                directory after shard construction succeeds.
            return_stats: Whether to return build statistics with the manifest path.

        Returns:
            Manifest path, or `(manifest_path, stats)` when `return_stats=True`.
        """
        if self._finished:
            if return_stats:
                raise RuntimeError("array dataset builder has already built shards; build stats are no longer available")
            return self._manifest_path
        self._ensure_open()
        bundle_manifest_path = self.finish_bundles()
        config = ArrayShardConfig(
            samples_per_block=int(self.build_options.samples_per_block),
            target_shard_bytes=int(self.build_options.target_shard_mb) * 1024 * 1024,
            n_shards=0 if self.build_options.n_shards is None else int(self.build_options.n_shards),
            row_group_size=0,
            use_tmp_spill=False,
            spill_bucket_target_bytes=8 * 1024 * 1024,
        )
        result = build_array_binary_shards_from_bundles(
            bundle_manifest_path,
            self.out_dir,
            config=config,
            codec=str(self.build_options.codec),
            zstd_level=int(self.build_options.zstd_level),
            sample_key_col=str(self.build_options.sample_key_col),
            feature_key_col=str(self.build_options.feature_key_col),
            return_stats=bool(return_stats),
        )
        if cleanup_bundles and os.path.exists(self.bundle_out_dir):
            shutil.rmtree(self.bundle_out_dir)
        self._finished = True
        self._closed = True
        self._manifest_path = result[0] if isinstance(result, tuple) else result
        return result

    def finish(self, *, cleanup_bundles: bool = False, return_stats: bool = False):
        """Compatibility alias for `build_shards(...)`."""
        return self.build_shards(cleanup_bundles=cleanup_bundles, return_stats=return_stats)
