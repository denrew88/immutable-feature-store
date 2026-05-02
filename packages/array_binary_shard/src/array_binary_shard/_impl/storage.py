import json
import os
import shutil
import struct
from functools import lru_cache

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .config import ArrayBundleConfig, ArrayShardConfig
from .sample_index import build_sample_id_index
from .types import (
    ArrayBlockLocation,
    ArrayBundleManifest,
    ArrayFeatureBlock,
    ArrayShardManifest,
    ArrayTrace,
    LogicalType,
    PointColumnSpec,
    StorageType,
    normalize_logical_type,
    point_storage_dtype,
)

FLAG_PRESENT = 0x01
FLAG_EMPTY = 0x02
FLAG_HAS_NONFINITE_TIME = 0x04
FLAG_HAS_NONFINITE_VALUE = 0x08
_SPILL_RECORD_HEADER = struct.Struct("<iqBi")

def _default_point_schema():
    return [
        PointColumnSpec(name="time", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
        PointColumnSpec(name="value", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
    ]


def _normalize_point_schema(point_schema):
    specs = []
    for spec in (point_schema or _default_point_schema()):
        if hasattr(spec, "name") and hasattr(spec, "storage_type") and hasattr(spec, "logical_type"):
            item = PointColumnSpec(
                name=spec.name,
                storage_type=spec.storage_type,
                logical_type=spec.logical_type,
                dictionary_path=getattr(spec, "dictionary_path", ""),
            )
        else:
            item = PointColumnSpec(
                name=str(spec["name"]),
                storage_type=spec["storage_type"],
                logical_type=spec.get("logical_type", LogicalType.CONTINUOUS),
                dictionary_path=str(spec.get("dictionary_path", "")),
            )
        specs.append(item)
    if not specs:
        raise ValueError("point_schema must not be empty")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("point_schema column names must be unique")
    return specs


def _point_blob_column_name(column_name: str) -> str:
    return f"{str(column_name)}_blob"


def _encode_point_values(values, spec: PointColumnSpec) -> np.ndarray:
    """Encode one point-column payload into the canonical storage dtype."""
    logical_type = normalize_logical_type(spec.logical_type)
    storage_dtype = point_storage_dtype(spec.storage_type)
    if logical_type == LogicalType.TIMESTAMP_NS:
        return np.asarray(values, dtype="datetime64[ns]").reshape(-1).astype(storage_dtype, copy=False)
    if logical_type == LogicalType.TIMEDELTA_NS:
        return np.asarray(values, dtype="timedelta64[ns]").reshape(-1).astype(storage_dtype, copy=False)
    return np.asarray(values, dtype=storage_dtype).reshape(-1)


def compute_array_trace_flags(time, value) -> int:
    """Compute per-trace flag bits from time and value arrays.

    Args:
        time: 1D array-like of time coordinates.
        value: 1D array-like of values aligned to `time`.

    Returns:
        Bitwise flag integer describing presence, emptiness, and nonfinite data.

    Raises:
        ValueError: If `time` and `value` do not have the same shape.
    """
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if time.shape != value.shape:
        raise ValueError(f"time/value length mismatch: {time.shape} != {value.shape}")
    flags = FLAG_PRESENT
    if time.size == 0:
        return flags | FLAG_EMPTY
    if np.any(~np.isfinite(time)):
        flags |= FLAG_HAS_NONFINITE_TIME
    if np.any(~np.isfinite(value)):
        flags |= FLAG_HAS_NONFINITE_VALUE
    return flags


def bundle_file_path(bundle_path: str, bundle_id: int) -> str:
    """Return the parquet path for one sample-major bundle file.

    Args:
        bundle_path: Directory containing bundle parquet files.
        bundle_id: Zero-based bundle identifier.

    Returns:
        Path to `bundle_{bundle_id:06d}.parquet`.
    """
    return os.path.join(bundle_path, f"bundle_{bundle_id:06d}.parquet")


def shard_file_path(shard_path: str, shard_id: int) -> str:
    """Return the parquet path for one array shard file.

    Args:
        shard_path: Directory containing shard parquet files.
        shard_id: Zero-based shard identifier.

    Returns:
        Path to `shard_{shard_id:04d}.parquet`.
    """
    return os.path.join(shard_path, f"shard_{shard_id:04d}.parquet")


def list_bundle_paths(manifest: ArrayBundleManifest):
    """List all bundle parquet files described by a bundle manifest.

    Args:
        manifest: Loaded bundle manifest.

    Returns:
        Bundle parquet paths ordered by bundle id.
    """
    return [bundle_file_path(manifest.bundle_path, bundle_id) for bundle_id in range(manifest.n_bundles)]


def list_array_shard_paths(manifest: ArrayShardManifest):
    """List all array shard parquet files described by a shard manifest.

    Args:
        manifest: Loaded shard manifest.

    Returns:
        Shard parquet paths ordered by shard id.
    """
    return [shard_file_path(manifest.shard_path, shard_id) for shard_id in range(manifest.n_shards)]


def _empty_array_trace_df():
    """Create an empty dataframe with the canonical array trace schema.

    Returns:
        An empty Polars dataframe with the columns used by bundle and shard code.
    """
    return pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", [], dtype=pl.Int32),
            "sample_row": pl.Series("sample_row", [], dtype=pl.Int64),
            "flags": pl.Series("flags", [], dtype=pl.UInt8),
            "trace_len": pl.Series("trace_len", [], dtype=pl.Int32),
            "time_blob": pl.Series("time_blob", [], dtype=pl.Binary),
            "value_blob": pl.Series("value_blob", [], dtype=pl.Binary),
        }
    )


class ArraySampleBundleWriter:
    def __init__(
        self,
        out_dir: str,
        sample_meta_path: str,
        n_samples: int,
        feature_meta_path: str = "",
        config: ArrayBundleConfig = None,
        point_schema=None,
    ):
        """Create a writer that groups sample-major traces into parquet bundles.

        Args:
            out_dir: Output directory that will contain bundle parquet files.
            sample_meta_path: Path to the sample metadata parquet file.
            feature_meta_path: Optional path to the feature metadata parquet file.
            n_samples: Total number of samples expected in the dataset.
            config: Optional bundle writer configuration.
        """
        self.config = config or ArrayBundleConfig()
        self.sample_meta_path = sample_meta_path or ""
        self.feature_meta_path = feature_meta_path or ""
        self.n_samples = int(n_samples)
        self.point_schema = _normalize_point_schema(point_schema)
        self.out_dir = out_dir
        self.bundle_path = os.path.join(out_dir, "array_sample_bundles")
        self.manifest_path = os.path.join(out_dir, "array_bundle_manifest.json")
        os.makedirs(self.bundle_path, exist_ok=True)
        self._reset_buffer()
        self.n_bundles = 0
        self._finished = False

    def __enter__(self):
        """Return the writer for context-manager usage."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Flush pending bundles when leaving a context manager."""
        self.finish()
        return False

    def append_trace(self, sample_row: int, sample_id: int, feature_id: int, *, columns):
        """Append one `(sample, feature)` trace into the current bundle buffer.

        Args:
            sample_row: Dense zero-based sample row index.
            sample_id: External sample identifier stored alongside the row.
            feature_id: Logical feature identifier.
            columns: Mapping from schema column name to 1D arrays.

        Raises:
            RuntimeError: If the writer has already been finished.
            ValueError: If indices are invalid or trace shapes do not match.
        """
        if self._finished:
            raise RuntimeError("writer already finished")
        sample_row = int(sample_row)
        if sample_row < 0 or sample_row >= self.n_samples:
            raise ValueError(f"sample_row out of range: {sample_row}")
        column_map = {str(name): value for name, value in dict(columns).items()}
        expected_names = {spec.name for spec in self.point_schema}
        actual_names = set(column_map.keys())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"point columns must exactly match point_schema ({', '.join(details)})")
        encoded_columns = {}
        trace_len = None
        for spec in self.point_schema:
            arr = _encode_point_values(column_map[spec.name], spec)
            if trace_len is None:
                trace_len = int(arr.size)
            elif int(arr.size) != int(trace_len):
                raise ValueError(f"point column length mismatch for {spec.name}: expected={trace_len} got={arr.size}")
            encoded_columns[spec.name] = np.asarray(arr, dtype=point_storage_dtype(spec.storage_type)).tobytes()
        trace_len = int(trace_len or 0)
        if "time" in column_map and "value" in column_map:
            time_arr = np.asarray(column_map["time"], dtype=np.float64).reshape(-1)
            value_arr = np.asarray(column_map["value"], dtype=np.float64).reshape(-1)
            flags = compute_array_trace_flags(time_arr, value_arr)
        else:
            flags = FLAG_PRESENT | (FLAG_EMPTY if trace_len == 0 else 0)

        self._sample_rows.append(sample_row)
        self._sample_ids.append(int(sample_id))
        self._feature_ids.append(int(feature_id))
        self._flags.append(flags)
        self._trace_lens.append(trace_len)
        for spec in self.point_schema:
            self._column_blobs[spec.name].append(encoded_columns[spec.name])
        self._current_rows += 1
        self._current_bytes += 8 + 8 + 4 + 1 + 4 + sum(len(encoded_columns[spec.name]) for spec in self.point_schema)

        if self._current_rows >= self.config.max_bundle_rows or self._current_bytes >= self.config.max_bundle_bytes:
            self.flush_bundle()

    def flush_bundle(self):
        """Write the current in-memory bundle buffer to a parquet file.

        Returns:
            None. When the buffer is empty this function does nothing.
        """
        if self._current_rows == 0:
            return
        order = np.lexsort((np.asarray(self._feature_ids), np.asarray(self._sample_rows)))
        df = pl.DataFrame(
            {
                "sample_row": pl.Series(
                    "sample_row",
                    [self._sample_rows[i] for i in order],
                    dtype=pl.Int64,
                ),
                "sample_id": pl.Series(
                    "sample_id",
                    [self._sample_ids[i] for i in order],
                    dtype=pl.Int64,
                ),
                "feature_id": pl.Series(
                    "feature_id",
                    [self._feature_ids[i] for i in order],
                    dtype=pl.Int32,
                ),
                "flags": pl.Series(
                    "flags",
                    [self._flags[i] for i in order],
                    dtype=pl.UInt8,
                ),
                "trace_len": pl.Series(
                    "trace_len",
                    [self._trace_lens[i] for i in order],
                    dtype=pl.Int32,
                ),
                **{
                    _point_blob_column_name(spec.name): pl.Series(
                        _point_blob_column_name(spec.name),
                        [self._column_blobs[spec.name][i] for i in order],
                        dtype=pl.Binary,
                    )
                    for spec in self.point_schema
                },
            }
        )
        df.write_parquet(bundle_file_path(self.bundle_path, self.n_bundles))
        self.n_bundles += 1
        self._reset_buffer()

    def finish(self):
        """Flush remaining data and write the bundle manifest.

        Returns:
            Path to the written bundle manifest JSON file.
        """
        if self._finished:
            return self.manifest_path
        self.flush_bundle()
        manifest = ArrayBundleManifest(
            sample_meta_path=self.sample_meta_path,
            feature_meta_path=self.feature_meta_path,
            n_samples=self.n_samples,
            bundle_path=self.bundle_path,
            n_bundles=self.n_bundles,
            feature_id_dtype="INT32",
            flags_dtype="UINT8",
            time_dtype="FLOAT64_LE_BLOB" if any(spec.name == "time" for spec in self.point_schema) else "",
            value_dtype="FLOAT64_LE_BLOB" if any(spec.name == "value" for spec in self.point_schema) else "",
            point_schema=self.point_schema,
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_json(), f, indent=2)
        self._finished = True
        return self.manifest_path

    def _reset_buffer(self):
        """Reset the in-memory append buffers for the next bundle."""
        self._sample_rows = []
        self._sample_ids = []
        self._feature_ids = []
        self._flags = []
        self._trace_lens = []
        self._column_blobs = {spec.name: [] for spec in getattr(self, "point_schema", _default_point_schema())}
        self._current_rows = 0
        self._current_bytes = 0


class _BlockAccumulator:
    def __init__(self, feature_id: int, block_id: int, sample_row: int, samples_per_block: int, n_samples: int):
        """Create an accumulator for one feature block inside a shard.

        Args:
            feature_id: Logical feature identifier.
            block_id: Zero-based block ordinal for the feature.
            sample_row: First sample row seen for this block.
            samples_per_block: Maximum number of samples stored per block.
            n_samples: Total number of samples in the dataset.
        """
        self.feature_id = int(feature_id)
        self.block_id = int(block_id)
        self.sample_row_start = self.block_id * int(samples_per_block)
        self.sample_count = int(min(samples_per_block, n_samples - self.sample_row_start))
        if self.sample_count <= 0:
            raise ValueError(f"invalid block sample_count for sample_row={sample_row}")
        self.sample_flags = np.zeros(self.sample_count, dtype=np.uint8)
        self.sample_offsets = np.zeros(self.sample_count + 1, dtype=np.int64)
        self.time_chunks = []
        self.value_chunks = []
        self.point_count = 0
        self.next_relative_sample = 0
        self.present_count = 0

    def append(self, sample_row: int, flags: int, trace_len: int, time_blob: bytes, value_blob: bytes):
        """Append one sample trace into the current block.

        Args:
            sample_row: Global sample row index.
            flags: Trace state flags.
            trace_len: Number of points in the trace.
            time_blob: Little-endian float64 time payload.
            value_blob: Little-endian float64 value payload.

        Raises:
            ValueError: If sample rows are outside the block range or not sorted.
        """
        relative_sample = int(sample_row - self.sample_row_start)
        if relative_sample < 0 or relative_sample >= self.sample_count:
            raise ValueError(f"sample_row out of block range: {sample_row}")
        if relative_sample < self.next_relative_sample:
            raise ValueError(
                f"duplicate or unsorted (feature_id={self.feature_id}, sample_row={sample_row})"
            )

        while self.next_relative_sample < relative_sample:
            self.sample_offsets[self.next_relative_sample + 1] = self.point_count
            self.next_relative_sample += 1

        out_flags = int(flags)
        if (out_flags & FLAG_PRESENT) == 0:
            out_flags |= FLAG_PRESENT
        if trace_len == 0 and (out_flags & FLAG_EMPTY) == 0:
            out_flags |= FLAG_EMPTY
        self.sample_flags[relative_sample] = out_flags

        if trace_len > 0:
            self.time_chunks.append(time_blob)
            self.value_chunks.append(value_blob)
            self.point_count += int(trace_len)

        self.sample_offsets[relative_sample + 1] = self.point_count
        self.next_relative_sample = relative_sample + 1
        self.present_count += 1

    def finish(self):
        """Seal trailing offsets for samples that were not explicitly appended."""
        while self.next_relative_sample < self.sample_count:
            self.sample_offsets[self.next_relative_sample + 1] = self.point_count
            self.next_relative_sample += 1

    def has_present_rows(self) -> bool:
        """Return whether the block contains at least one present sample."""
        return self.present_count > 0


def build_array_shards_from_bundles(
    bundle_manifest_path: str,
    out_dir: str,
    config: ArrayShardConfig = None,
    return_stats: bool = False,
):
    """Build feature-major array parquet shards from sample-major bundles.

    Args:
        bundle_manifest_path: Path to `array_bundle_manifest.json`.
        out_dir: Output directory for shard parquet files, locator, and manifest.
        config: Optional shard build configuration.
        return_stats: Whether to return build statistics alongside the manifest path.

    Returns:
        Either the shard manifest path, or `(manifest_path, stats)` when
        `return_stats=True`.
    """
    config = config or ArrayShardConfig()
    if config.samples_per_block <= 0:
        raise ValueError("samples_per_block must be > 0")
    if config.target_shard_bytes <= 0 and config.n_shards <= 0:
        raise ValueError("either target_shard_bytes or n_shards must be > 0")
    if config.use_tmp_spill and config.spill_bucket_target_bytes <= 0:
        raise ValueError("spill_bucket_target_bytes must be > 0 when use_tmp_spill=True")

    bundle_manifest = load_array_bundle_manifest(bundle_manifest_path)
    bundle_paths = list_bundle_paths(bundle_manifest)
    feature_ids, estimated_feature_bytes = _collect_array_feature_stats(bundle_paths, config.samples_per_block)
    shard_partitions = _build_array_shard_partitions(feature_ids, estimated_feature_bytes, config)
    if not shard_partitions:
        shard_partitions = [np.empty(0, dtype=np.int32)]

    if config.use_tmp_spill:
        manifest_path, stats = _build_array_shards_with_tmp_spill(
            bundle_manifest=bundle_manifest,
            bundle_paths=bundle_paths,
            feature_ids=feature_ids,
            estimated_feature_bytes=estimated_feature_bytes,
            shard_partitions=shard_partitions,
            out_dir=out_dir,
            config=config,
        )
    else:
        manifest_path, stats = _build_array_shards_direct(
            bundle_manifest=bundle_manifest,
            bundle_paths=bundle_paths,
            shard_partitions=shard_partitions,
            out_dir=out_dir,
            config=config,
        )
    if return_stats:
        return manifest_path, stats
    return manifest_path


def _collect_feature_ids(bundle_paths):
    """Collect the sorted feature id domain across all bundle files.

    Args:
        bundle_paths: Bundle parquet paths to scan.

    Returns:
        Sorted `int32` numpy array of unique feature ids.
    """
    if not bundle_paths:
        return np.array([], dtype=np.int32)
    df = pl.scan_parquet(bundle_paths).select(pl.col("feature_id").unique().sort()).collect()
    return df["feature_id"].to_numpy().astype(np.int32, copy=False)


def _estimate_block_overhead_bytes(samples_per_block: int) -> int:
    """Estimate fixed bytes added per serialized block.

    Args:
        samples_per_block: Maximum sample count per block.

    Returns:
        Approximate fixed byte overhead per block used for partition planning.
    """
    return int(64 + 9 * samples_per_block)


def _build_array_shard_partitions(feature_ids, estimated_feature_bytes, config: ArrayShardConfig):
    """Choose shard partitions using fixed count or target-byte partitioning.

    Args:
        feature_ids: Sorted feature ids.
        estimated_feature_bytes: Estimated bytes per feature.
        config: Array shard build configuration.

    Returns:
        A list of numpy arrays, one per shard partition.
    """
    if config.n_shards > 0:
        return _partition_feature_ids_by_count(feature_ids, config.n_shards)
    return _partition_feature_ids_by_target_bytes(
        feature_ids,
        estimated_feature_bytes,
        config.target_shard_bytes,
    )


def _prepare_array_shard_output_paths(out_dir: str):
    """Create output directories and derive standard parquet shard paths.

    Args:
        out_dir: Root output directory for the shard build.

    Returns:
        `(shard_path, locator_path, manifest_path)`.
    """
    os.makedirs(out_dir, exist_ok=True)
    shard_path = os.path.join(out_dir, "array_feature_shards")
    os.makedirs(shard_path, exist_ok=True)
    locator_path = os.path.join(out_dir, "array_feature_locator.parquet")
    manifest_path = os.path.join(out_dir, "array_shard_manifest.json")
    return shard_path, locator_path, manifest_path


def _write_array_shard_manifest(
    bundle_manifest: ArrayBundleManifest,
    shard_path: str,
    locator_path: str,
    manifest_path: str,
    n_shards: int,
    samples_per_block: int,
    row_group_size: int,
):
    """Write the JSON manifest for a parquet array shard set.

    Args:
        bundle_manifest: Source bundle manifest.
        shard_path: Directory containing shard parquet files.
        locator_path: Path to the block locator parquet file.
        manifest_path: Output manifest JSON path.
        n_shards: Number of generated shards.
        samples_per_block: Logical block size used during build.
        row_group_size: Physical parquet row group size used for shard writing.

    Returns:
        Path to the written shard manifest JSON file.
    """
    manifest = ArrayShardManifest(
        sample_meta_path=bundle_manifest.sample_meta_path,
        feature_meta_path=bundle_manifest.feature_meta_path,
        n_samples=bundle_manifest.n_samples,
        shard_path=shard_path,
        n_shards=int(n_shards),
        locator_path=locator_path,
        samples_per_block=int(samples_per_block),
        feature_id_dtype="INT32",
        flags_dtype="UINT8",
        offset_dtype="INT64",
        time_dtype="FLOAT64_LE_BLOB",
        value_dtype="FLOAT64_LE_BLOB",
        row_group_size=int(row_group_size),
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2)
    return manifest_path


def _collect_array_feature_stats(bundle_paths, samples_per_block: int):
    """Estimate per-feature payload bytes and block counts from bundle metadata.

    Args:
        bundle_paths: Bundle parquet paths to scan.
        samples_per_block: Logical block size used to derive block ids.

    Returns:
        A pair `(feature_ids, estimated_feature_bytes)` as numpy arrays.
    """
    if not bundle_paths:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int64)
    block_overhead = _estimate_block_overhead_bytes(samples_per_block)
    df = (
        pl.scan_parquet(bundle_paths)
        .select(
            [
                pl.col("feature_id").cast(pl.Int32),
                pl.col("trace_len").cast(pl.Int64),
                (pl.col("sample_row") // samples_per_block).cast(pl.Int64).alias("block_id"),
            ]
        )
        .group_by("feature_id")
        .agg(
            [
                pl.col("trace_len").sum().alias("total_trace_len"),
                pl.col("block_id").n_unique().alias("block_count"),
            ]
        )
        .sort("feature_id")
        .collect()
    )
    feature_ids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    total_trace_len = df["total_trace_len"].to_numpy().astype(np.int64, copy=False)
    block_count = df["block_count"].to_numpy().astype(np.int64, copy=False)
    estimated_bytes = total_trace_len * 16 + block_count * np.int64(block_overhead)
    return feature_ids, estimated_bytes


def _partition_feature_ids_by_count(feature_ids, n_shards: int):
    """Split sorted feature ids into roughly even contiguous partitions.

    Args:
        feature_ids: Sorted feature ids.
        n_shards: Number of desired partitions.

    Returns:
        A list of numpy arrays, one per shard partition.
    """
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    if len(feature_ids) == 0:
        return [np.empty(0, dtype=np.int32) for _ in range(n_shards)]
    shard_size = max(1, (len(feature_ids) + n_shards - 1) // n_shards)
    partitions = []
    for shard_id in range(n_shards):
        start = shard_id * shard_size
        end = min((shard_id + 1) * shard_size, len(feature_ids))
        partitions.append(np.asarray(feature_ids[start:end], dtype=np.int32))
    return partitions


def _partition_feature_ids_by_target_bytes(feature_ids, estimated_feature_bytes, target_shard_bytes: int):
    """Split feature ids so each partition stays near a target byte budget.

    Args:
        feature_ids: Sorted feature ids.
        estimated_feature_bytes: Estimated bytes per feature.
        target_shard_bytes: Desired upper target for one shard partition.

    Returns:
        A list of contiguous feature-id partitions.
    """
    if target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be > 0")
    if len(feature_ids) == 0:
        return [np.empty(0, dtype=np.int32)]
    partitions = []
    current_feature_ids = []
    current_bytes = 0
    for feature_id, est_bytes in zip(feature_ids, estimated_feature_bytes):
        est_bytes = max(int(est_bytes), 1)
        if current_feature_ids and current_bytes + est_bytes > target_shard_bytes:
            partitions.append(np.asarray(current_feature_ids, dtype=np.int32))
            current_feature_ids = []
            current_bytes = 0
        current_feature_ids.append(int(feature_id))
        current_bytes += est_bytes
    if current_feature_ids:
        partitions.append(np.asarray(current_feature_ids, dtype=np.int32))
    return partitions


def _build_array_shards_direct(
    bundle_manifest: ArrayBundleManifest,
    bundle_paths,
    shard_partitions,
    out_dir: str,
    config: ArrayShardConfig,
):
    """Build parquet shards directly from bundle scans without spill files.

    Args:
        bundle_manifest: Source bundle manifest.
        bundle_paths: Bundle parquet paths to read.
        shard_partitions: Feature-id partitions, one per output shard.
        out_dir: Output directory for shards.
        config: Shard build configuration.

    Returns:
        `(manifest_path, stats)` where `stats` contains temp-file metrics.
    """
    shard_path, locator_path, manifest_path = _prepare_array_shard_output_paths(out_dir)
    locator_rows = []
    for shard_id, shard_feature_ids in enumerate(shard_partitions):
        df = _collect_sorted_rows_for_shard(bundle_paths, shard_feature_ids)
        shard_rows, shard_locator_rows = _process_sorted_rows(
            df,
            bundle_manifest.n_samples,
            config.samples_per_block,
            shard_id,
        )
        _write_array_shard_file(
            shard_rows,
            shard_file_path(shard_path, shard_id),
            row_group_size=config.row_group_size,
        )
        locator_rows.extend(shard_locator_rows)

    _write_array_locator(locator_rows, locator_path)
    manifest_path = _write_array_shard_manifest(
        bundle_manifest=bundle_manifest,
        shard_path=shard_path,
        locator_path=locator_path,
        manifest_path=manifest_path,
        n_shards=len(shard_partitions),
        samples_per_block=config.samples_per_block,
        row_group_size=config.row_group_size,
    )
    stats = {
        "n_shards": int(len(shard_partitions)),
        "n_buckets": 0,
        "temp_files_created": 0,
        "peak_live_temp_files": 0,
        "peak_live_temp_bytes": 0,
    }
    return manifest_path, stats


def _build_bucket_partitions(shard_partitions, feature_ids, estimated_feature_bytes, spill_bucket_target_bytes: int):
    """Subdivide shard partitions into bounded spill buckets.

    Args:
        shard_partitions: Feature-id partitions, one per shard.
        feature_ids: Sorted feature ids.
        estimated_feature_bytes: Estimated bytes per feature.
        spill_bucket_target_bytes: Target bucket size during spill builds.

    Returns:
        `(feature_map_df, shard_bucket_partitions)` used by spill mode.
    """
    est_lookup = {
        int(feature_id): int(est_bytes)
        for feature_id, est_bytes in zip(feature_ids.tolist(), estimated_feature_bytes.tolist())
    }
    feature_map_rows = []
    shard_bucket_partitions = []
    for shard_id, shard_feature_ids in enumerate(shard_partitions):
        shard_estimates = np.asarray(
            [est_lookup[int(feature_id)] for feature_id in shard_feature_ids.tolist()],
            dtype=np.int64,
        )
        bucket_partitions = _partition_feature_ids_by_target_bytes(
            shard_feature_ids,
            shard_estimates,
            spill_bucket_target_bytes,
        )
        shard_bucket_partitions.append(bucket_partitions)
        for bucket_id, bucket_feature_ids in enumerate(bucket_partitions):
            for feature_id in bucket_feature_ids.tolist():
                feature_map_rows.append(
                    {
                        "feature_id": int(feature_id),
                        "shard_id": int(shard_id),
                        "bucket_id": int(bucket_id),
                    }
                )
    feature_map_df = pl.DataFrame(
        {
            "feature_id": pl.Series(
                "feature_id",
                [row["feature_id"] for row in feature_map_rows],
                dtype=pl.Int32,
            ),
            "shard_id": pl.Series(
                "shard_id",
                [row["shard_id"] for row in feature_map_rows],
                dtype=pl.Int32,
            ),
            "bucket_id": pl.Series(
                "bucket_id",
                [row["bucket_id"] for row in feature_map_rows],
                dtype=pl.Int32,
            ),
        }
    )
    return feature_map_df, shard_bucket_partitions


def _bucket_spill_path(tmp_root: str, shard_id: int, bucket_id: int):
    """Return the temporary spill file path for one shard bucket.

    Args:
        tmp_root: Root temporary spill directory.
        shard_id: Zero-based shard identifier.
        bucket_id: Zero-based bucket identifier within the shard.

    Returns:
        Path to the binary spill file.
    """
    shard_dir = os.path.join(tmp_root, f"shard_{shard_id:04d}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, f"bucket_{bucket_id:04d}.spill")


def _append_frame_to_spill_file(df: pl.DataFrame, spill_path: str) -> int:
    """Append one dataframe chunk to a binary spill file.

    Args:
        df: Trace rows to spill.
        spill_path: Destination spill file path.

    Returns:
        Number of bytes appended to the spill file.
    """
    if df.height == 0:
        return 0
    feature_ids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    sample_rows = df["sample_row"].to_numpy().astype(np.int64, copy=False)
    flags = df["flags"].to_numpy().astype(np.uint8, copy=False)
    trace_lens = df["trace_len"].to_numpy().astype(np.int32, copy=False)
    time_blobs = df["time_blob"].to_list()
    value_blobs = df["value_blob"].to_list()

    written_bytes = 0
    with open(spill_path, "ab") as f:
        for idx in range(df.height):
            trace_len = int(trace_lens[idx])
            time_blob = time_blobs[idx] or b""
            value_blob = value_blobs[idx] or b""
            header = _SPILL_RECORD_HEADER.pack(
                int(feature_ids[idx]),
                int(sample_rows[idx]),
                int(flags[idx]),
                trace_len,
            )
            f.write(header)
            written_bytes += _SPILL_RECORD_HEADER.size
            if time_blob:
                f.write(time_blob)
                written_bytes += len(time_blob)
            if value_blob:
                f.write(value_blob)
                written_bytes += len(value_blob)
    return written_bytes


def _load_spill_file_to_sorted_df(spill_path: str):
    """Load a spill file back into a sorted dataframe of trace rows.

    Args:
        spill_path: Path to a spill file produced by `_append_frame_to_spill_file`.

    Returns:
        A Polars dataframe sorted by `(feature_id, sample_row)`.
    """
    feature_ids = []
    sample_rows = []
    flags = []
    trace_lens = []
    time_blobs = []
    value_blobs = []
    with open(spill_path, "rb") as f:
        while True:
            header = f.read(_SPILL_RECORD_HEADER.size)
            if not header:
                break
            if len(header) != _SPILL_RECORD_HEADER.size:
                raise ValueError(f"corrupt spill header: {spill_path}")
            feature_id, sample_row, flag, trace_len = _SPILL_RECORD_HEADER.unpack(header)
            expected = int(trace_len) * 8
            time_blob = f.read(expected)
            value_blob = f.read(expected)
            if len(time_blob) != expected:
                raise ValueError(f"corrupt spill time_blob: {spill_path}")
            if len(value_blob) != expected:
                raise ValueError(f"corrupt spill value_blob: {spill_path}")
            feature_ids.append(int(feature_id))
            sample_rows.append(int(sample_row))
            flags.append(int(flag))
            trace_lens.append(int(trace_len))
            time_blobs.append(time_blob)
            value_blobs.append(value_blob)

    if not feature_ids:
        return _empty_array_trace_df()

    order = np.lexsort(
        (
            np.asarray(sample_rows, dtype=np.int64),
            np.asarray(feature_ids, dtype=np.int32),
        )
    )
    return pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", [feature_ids[i] for i in order], dtype=pl.Int32),
            "sample_row": pl.Series("sample_row", [sample_rows[i] for i in order], dtype=pl.Int64),
            "flags": pl.Series("flags", [flags[i] for i in order], dtype=pl.UInt8),
            "trace_len": pl.Series("trace_len", [trace_lens[i] for i in order], dtype=pl.Int32),
            "time_blob": pl.Series("time_blob", [time_blobs[i] for i in order], dtype=pl.Binary),
            "value_blob": pl.Series("value_blob", [value_blobs[i] for i in order], dtype=pl.Binary),
        }
    )


def _build_array_shards_with_tmp_spill(
    bundle_manifest: ArrayBundleManifest,
    bundle_paths,
    feature_ids,
    estimated_feature_bytes,
    shard_partitions,
    out_dir: str,
    config: ArrayShardConfig,
):
    """Build parquet shards through temporary spill buckets.

    Args:
        bundle_manifest: Source bundle manifest.
        bundle_paths: Bundle parquet paths to read.
        feature_ids: Sorted feature ids present in the input.
        estimated_feature_bytes: Estimated bytes per feature.
        shard_partitions: Feature-id partitions, one per shard.
        out_dir: Output directory for shards.
        config: Shard build configuration.

    Returns:
        `(manifest_path, stats)` where `stats` contains spill-related metrics.
    """
    shard_path, locator_path, manifest_path = _prepare_array_shard_output_paths(out_dir)
    tmp_root = os.path.join(out_dir, "_tmp")
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    feature_map_df, shard_bucket_partitions = _build_bucket_partitions(
        shard_partitions,
        feature_ids,
        estimated_feature_bytes,
        config.spill_bucket_target_bytes,
    )
    spill_paths = {}
    spill_sizes = {}
    temp_files_created = 0
    live_temp_files = 0
    peak_live_temp_files = 0
    live_temp_bytes = 0
    peak_live_temp_bytes = 0

    try:
        for bundle_path in bundle_paths:
            bundle_df = pl.read_parquet(
                bundle_path,
                columns=[
                    "feature_id",
                    "sample_row",
                    "flags",
                    "trace_len",
                    "time_blob",
                    "value_blob",
                ],
            )
            if bundle_df.height == 0:
                continue
            joined = bundle_df.join(feature_map_df, on="feature_id", how="inner")
            if joined.height == 0:
                continue
            parts = joined.partition_by(["shard_id", "bucket_id"], as_dict=True)
            for key, part_df in parts.items():
                shard_id = int(key[0])
                bucket_id = int(key[1])
                spill_key = (shard_id, bucket_id)
                spill_path = spill_paths.get(spill_key)
                if spill_path is None:
                    spill_path = _bucket_spill_path(tmp_root, shard_id, bucket_id)
                    spill_paths[spill_key] = spill_path
                    temp_files_created += 1
                    live_temp_files += 1
                delta = _append_frame_to_spill_file(
                    part_df.select(
                        [
                            "feature_id",
                            "sample_row",
                            "flags",
                            "trace_len",
                            "time_blob",
                            "value_blob",
                        ]
                    ),
                    spill_path,
                )
                spill_sizes[spill_key] = spill_sizes.get(spill_key, 0) + int(delta)
                live_temp_bytes += int(delta)
                peak_live_temp_files = max(peak_live_temp_files, live_temp_files)
                peak_live_temp_bytes = max(peak_live_temp_bytes, live_temp_bytes)

        locator_rows = []
        for shard_id, bucket_partitions in enumerate(shard_bucket_partitions):
            shard_rows = []
            row_in_shard_offset = 0
            for bucket_id, _bucket_feature_ids in enumerate(bucket_partitions):
                spill_key = (shard_id, bucket_id)
                spill_path = spill_paths.get(spill_key)
                if spill_path is None or not os.path.exists(spill_path):
                    continue
                df = _load_spill_file_to_sorted_df(spill_path)
                bucket_rows, bucket_locator_rows = _process_sorted_rows(
                    df,
                    bundle_manifest.n_samples,
                    config.samples_per_block,
                    shard_id,
                )
                for row in bucket_locator_rows:
                    row["row_in_shard"] += row_in_shard_offset
                row_in_shard_offset += len(bucket_rows)
                shard_rows.extend(bucket_rows)
                locator_rows.extend(bucket_locator_rows)

                spill_size = spill_sizes.pop(spill_key, 0)
                os.remove(spill_path)
                live_temp_files -= 1
                live_temp_bytes -= int(spill_size)

            _write_array_shard_file(
                shard_rows,
                shard_file_path(shard_path, shard_id),
                row_group_size=config.row_group_size,
            )

        _write_array_locator(locator_rows, locator_path)
        manifest_path = _write_array_shard_manifest(
            bundle_manifest=bundle_manifest,
            shard_path=shard_path,
            locator_path=locator_path,
            manifest_path=manifest_path,
            n_shards=len(shard_partitions),
            samples_per_block=config.samples_per_block,
            row_group_size=config.row_group_size,
        )
        stats = {
            "n_shards": int(len(shard_partitions)),
            "n_buckets": int(sum(len(bucket_partitions) for bucket_partitions in shard_bucket_partitions)),
            "temp_files_created": int(temp_files_created),
            "peak_live_temp_files": int(peak_live_temp_files),
            "peak_live_temp_bytes": int(peak_live_temp_bytes),
        }
        return manifest_path, stats
    finally:
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)


def _collect_sorted_rows_for_shard(bundle_paths, shard_feature_ids):
    """Collect and sort all bundle rows belonging to one shard partition.

    Args:
        bundle_paths: Bundle parquet paths to scan.
        shard_feature_ids: Feature ids assigned to one shard.

    Returns:
        A Polars dataframe sorted by `(feature_id, sample_row)`.
    """
    if not bundle_paths or len(shard_feature_ids) == 0:
        return _empty_array_trace_df()
    feature_id_list = [int(feature_id) for feature_id in shard_feature_ids.tolist()]
    return (
        pl.scan_parquet(bundle_paths)
        .filter(pl.col("feature_id").is_in(feature_id_list))
        .select(
            [
                "feature_id",
                "sample_row",
                "flags",
                "trace_len",
                "time_blob",
                "value_blob",
            ]
        )
        .sort(["feature_id", "sample_row"])
        .collect()
    )


def _process_sorted_rows(
    df: pl.DataFrame,
    n_samples: int,
    samples_per_block: int,
    shard_id: int,
):
    """Convert sorted trace rows into shard rows and locator entries.

    Args:
        df: Trace rows sorted by `(feature_id, sample_row)`.
        n_samples: Total number of samples in the dataset.
        samples_per_block: Logical block size used for grouping samples.
        shard_id: Zero-based shard identifier being produced.

    Returns:
        `(shard_rows, locator_rows)` ready for shard and locator serialization.
    """
    shard_rows = []
    locator_rows = []
    feature_ids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    sample_rows = df["sample_row"].to_numpy().astype(np.int64, copy=False)
    flags = df["flags"].to_numpy().astype(np.uint8, copy=False)
    trace_lens = df["trace_len"].to_numpy().astype(np.int32, copy=False)
    time_blobs = df["time_blob"].to_list()
    value_blobs = df["value_blob"].to_list()

    current_feature_id = None
    block = None
    row_in_shard = 0

    for idx in range(df.height):
        feature_id = int(feature_ids[idx])
        sample_row = int(sample_rows[idx])
        trace_len = int(trace_lens[idx])
        time_blob = time_blobs[idx]
        value_blob = value_blobs[idx]
        _validate_trace_row(sample_row, n_samples, trace_len, time_blob, value_blob)

        if feature_id != current_feature_id:
            row_in_shard = _finalize_array_block(
                block,
                current_feature_id,
                shard_id,
                row_in_shard,
                shard_rows,
                locator_rows,
            )
            block = None
            current_feature_id = feature_id

        block_id = sample_row // samples_per_block
        if block is None or block.block_id != block_id:
            row_in_shard = _finalize_array_block(
                block,
                current_feature_id,
                shard_id,
                row_in_shard,
                shard_rows,
                locator_rows,
            )
            block = _BlockAccumulator(
                feature_id=feature_id,
                block_id=block_id,
                sample_row=sample_row,
                samples_per_block=samples_per_block,
                n_samples=n_samples,
            )

        block.append(
            sample_row=sample_row,
            flags=int(flags[idx]),
            trace_len=trace_len,
            time_blob=time_blob,
            value_blob=value_blob,
        )

    _finalize_array_block(
        block,
        current_feature_id,
        shard_id,
        row_in_shard,
        shard_rows,
        locator_rows,
    )
    return shard_rows, locator_rows


def _validate_trace_row(sample_row: int, n_samples: int, trace_len: int, time_blob: bytes, value_blob: bytes):
    """Validate one raw trace row before block packing.

    Args:
        sample_row: Global sample row index.
        n_samples: Total number of samples in the dataset.
        trace_len: Number of points in the trace.
        time_blob: Serialized time payload.
        value_blob: Serialized value payload.

    Raises:
        ValueError: If row indices or blob lengths are inconsistent.
    """
    if sample_row < 0 or sample_row >= n_samples:
        raise ValueError(f"sample_row out of range: {sample_row}")
    if trace_len < 0:
        raise ValueError("trace_len must be >= 0")
    expected = trace_len * 8
    actual_time = len(time_blob or b"")
    actual_value = len(value_blob or b"")
    if actual_time != expected:
        raise ValueError(f"time_blob length mismatch: expected={expected} got={actual_time}")
    if actual_value != expected:
        raise ValueError(f"value_blob length mismatch: expected={expected} got={actual_value}")


def _finalize_array_block(block, current_feature_id, shard_id: int, row_in_shard: int, shard_rows, locator_rows):
    """Flush a completed block into shard rows and locator rows.

    Args:
        block: Current `_BlockAccumulator` or `None`.
        current_feature_id: Feature id associated with the block.
        shard_id: Zero-based shard identifier.
        row_in_shard: Next physical row offset in the shard.
        shard_rows: Output list receiving serialized shard rows.
        locator_rows: Output list receiving locator entries.

    Returns:
        Updated `row_in_shard` after the block is emitted.
    """
    if block is None or current_feature_id is None:
        return row_in_shard
    block.finish()
    if not block.has_present_rows():
        return row_in_shard
    sample_row_end = block.sample_row_start + block.sample_count - 1
    shard_rows.append(
        {
            "feature_id": int(current_feature_id),
            "block_id": int(block.block_id),
            "sample_row_start": int(block.sample_row_start),
            "sample_count": int(block.sample_count),
            "point_count": int(block.point_count),
            "sample_flags_blob": block.sample_flags.tobytes(),
            "sample_offsets_blob": np.asarray(block.sample_offsets, dtype="<i8").tobytes(),
            "time_blob": b"".join(block.time_chunks),
            "value_blob": b"".join(block.value_chunks),
        }
    )
    locator_rows.append(
        {
            "feature_id": int(current_feature_id),
            "block_id": int(block.block_id),
            "shard_id": shard_id,
            "row_in_shard": row_in_shard,
            "sample_row_start": int(block.sample_row_start),
            "sample_row_end": int(sample_row_end),
        }
    )
    return row_in_shard + 1


def _write_array_shard_file(rows, path: str, row_group_size: int = 0):
    """Write one parquet shard file from serialized block rows.

    Args:
        rows: Serialized block rows for one shard.
        path: Output parquet path.
        row_group_size: Optional physical parquet row-group size.
    """
    if not rows:
        df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", [], dtype=pl.Int32),
                "block_id": pl.Series("block_id", [], dtype=pl.Int32),
                "sample_row_start": pl.Series("sample_row_start", [], dtype=pl.Int64),
                "sample_count": pl.Series("sample_count", [], dtype=pl.Int32),
                "point_count": pl.Series("point_count", [], dtype=pl.Int64),
                "sample_flags_blob": pl.Series("sample_flags_blob", [], dtype=pl.Binary),
                "sample_offsets_blob": pl.Series("sample_offsets_blob", [], dtype=pl.Binary),
                "time_blob": pl.Series("time_blob", [], dtype=pl.Binary),
                "value_blob": pl.Series("value_blob", [], dtype=pl.Binary),
            }
        )
    else:
        df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", [row["feature_id"] for row in rows], dtype=pl.Int32),
                "block_id": pl.Series("block_id", [row["block_id"] for row in rows], dtype=pl.Int32),
                "sample_row_start": pl.Series(
                    "sample_row_start",
                    [row["sample_row_start"] for row in rows],
                    dtype=pl.Int64,
                ),
                "sample_count": pl.Series(
                    "sample_count",
                    [row["sample_count"] for row in rows],
                    dtype=pl.Int32,
                ),
                "point_count": pl.Series(
                    "point_count",
                    [row["point_count"] for row in rows],
                    dtype=pl.Int64,
                ),
                "sample_flags_blob": pl.Series(
                    "sample_flags_blob",
                    [row["sample_flags_blob"] for row in rows],
                    dtype=pl.Binary,
                ),
                "sample_offsets_blob": pl.Series(
                    "sample_offsets_blob",
                    [row["sample_offsets_blob"] for row in rows],
                    dtype=pl.Binary,
                ),
                "time_blob": pl.Series(
                    "time_blob",
                    [row["time_blob"] for row in rows],
                    dtype=pl.Binary,
                ),
                "value_blob": pl.Series(
                    "value_blob",
                    [row["value_blob"] for row in rows],
                    dtype=pl.Binary,
                ),
            }
        )
    write_kwargs = {}
    if int(row_group_size) > 0:
        write_kwargs["row_group_size"] = int(row_group_size)
    df.write_parquet(path, **write_kwargs)


def _write_array_locator(locator_rows, locator_path: str):
    """Write the array feature locator parquet file.

    Args:
        locator_rows: Locator records produced during shard building.
        locator_path: Output locator parquet path.
    """
    if not locator_rows:
        df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", [], dtype=pl.Int32),
                "block_id": pl.Series("block_id", [], dtype=pl.Int32),
                "shard_id": pl.Series("shard_id", [], dtype=pl.Int32),
                "row_in_shard": pl.Series("row_in_shard", [], dtype=pl.Int32),
                "sample_row_start": pl.Series("sample_row_start", [], dtype=pl.Int64),
                "sample_row_end": pl.Series("sample_row_end", [], dtype=pl.Int64),
            }
        )
    else:
        df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", [row["feature_id"] for row in locator_rows], dtype=pl.Int32),
                "block_id": pl.Series("block_id", [row["block_id"] for row in locator_rows], dtype=pl.Int32),
                "shard_id": pl.Series("shard_id", [row["shard_id"] for row in locator_rows], dtype=pl.Int32),
                "row_in_shard": pl.Series(
                    "row_in_shard",
                    [row["row_in_shard"] for row in locator_rows],
                    dtype=pl.Int32,
                ),
                "sample_row_start": pl.Series(
                    "sample_row_start",
                    [row["sample_row_start"] for row in locator_rows],
                    dtype=pl.Int64,
                ),
                "sample_row_end": pl.Series(
                    "sample_row_end",
                    [row["sample_row_end"] for row in locator_rows],
                    dtype=pl.Int64,
                ),
            }
        )
    df.write_parquet(locator_path)


def load_array_bundle_manifest(manifest_path: str) -> ArrayBundleManifest:
    """Load an array bundle manifest from JSON.

    Args:
        manifest_path: Path to `array_bundle_manifest.json`.

    Returns:
        A populated `ArrayBundleManifest` instance.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ArrayBundleManifest(
        sample_meta_path=data["sample_meta_path"],
        feature_meta_path=data.get("feature_meta_path", ""),
        n_samples=int(data["n_samples"]),
        bundle_path=data["bundle_path"],
        n_bundles=int(data["n_bundles"]),
        feature_id_dtype=data["feature_id_dtype"],
        flags_dtype=data["flags_dtype"],
        time_dtype=data.get("time_dtype", "FLOAT64_LE_BLOB"),
        value_dtype=data.get("value_dtype", "FLOAT64_LE_BLOB"),
        point_schema=_normalize_point_schema(data.get("point_schema")),
    )


def load_array_shard_manifest(manifest_path: str) -> ArrayShardManifest:
    """Load an array parquet shard manifest from JSON.

    Args:
        manifest_path: Path to `array_shard_manifest.json`.

    Returns:
        A populated `ArrayShardManifest` instance.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ArrayShardManifest(
        sample_meta_path=data["sample_meta_path"],
        feature_meta_path=data.get("feature_meta_path", ""),
        n_samples=int(data["n_samples"]),
        shard_path=data["shard_path"],
        n_shards=int(data["n_shards"]),
        locator_path=data["locator_path"],
        samples_per_block=int(data["samples_per_block"]),
        feature_id_dtype=data["feature_id_dtype"],
        flags_dtype=data["flags_dtype"],
        offset_dtype=data["offset_dtype"],
        time_dtype=data["time_dtype"],
        value_dtype=data["value_dtype"],
        row_group_size=int(data.get("row_group_size", 0)),
    )


def build_array_feature_locator_index(locator_path: str):
    """Build an in-memory locator index keyed by feature id.

    Args:
        locator_path: Path to the array feature locator parquet file.

    Returns:
        A dictionary mapping feature id to sorted `ArrayBlockLocation` lists.
    """
    df = pl.read_parquet(locator_path)
    out = {}
    for row in df.iter_rows(named=True):
        loc = ArrayBlockLocation(
            feature_id=int(row["feature_id"]),
            block_id=int(row["block_id"]),
            shard_id=int(row["shard_id"]),
            row_in_shard=int(row["row_in_shard"]),
            sample_row_start=int(row["sample_row_start"]),
            sample_row_end=int(row["sample_row_end"]),
        )
        out.setdefault(loc.feature_id, []).append(loc)
    for feature_id in out:
        out[feature_id].sort(key=lambda loc: (loc.sample_row_start, loc.block_id))
    return out


@lru_cache(maxsize=512)
def _cached_array_parquet_file(path: str):
    """Cache a pyarrow `ParquetFile` for repeated shard access.

    Args:
        path: Shard parquet path.

    Returns:
        Cached `pyarrow.parquet.ParquetFile` instance.
    """
    return pq.ParquetFile(path, memory_map=True)


@lru_cache(maxsize=512)
def _cached_array_row_group_starts(path: str):
    """Cache cumulative row-group starts for one shard parquet file.

    Args:
        path: Shard parquet path.

    Returns:
        A numpy array where each element is the starting row offset of a row group.
    """
    parquet_file = _cached_array_parquet_file(path)
    starts = np.empty(parquet_file.num_row_groups, dtype=np.int64)
    start = 0
    for idx in range(parquet_file.num_row_groups):
        starts[idx] = start
        start += parquet_file.metadata.row_group(idx).num_rows
    return starts


def _load_array_parquet_row(path: str, row_in_shard: int):
    """Load one physical row from a shard using row-group metadata.

    Args:
        path: Shard parquet path.
        row_in_shard: Zero-based row offset inside the shard.

    Returns:
        A Python dictionary for the requested row.
    """
    row_group_starts = _cached_array_row_group_starts(path)
    if row_group_starts.size == 0:
        raise ValueError(f"empty shard file: {path}")
    row_group_id = int(np.searchsorted(row_group_starts, int(row_in_shard), side="right") - 1)
    if row_group_id < 0:
        raise ValueError(f"invalid row offset: {row_in_shard}")
    row_offset = int(row_in_shard - int(row_group_starts[row_group_id]))
    table = _cached_array_parquet_file(path).read_row_group(row_group_id)
    if row_offset < 0 or row_offset >= table.num_rows:
        raise ValueError(f"block not found: path={path} row_in_shard={row_in_shard}")
    return table.slice(row_offset, 1).to_pylist()[0]


def load_array_feature_block(manifest: ArrayShardManifest, shard_id: int, row_in_shard: int):
    """Decode one feature block from a parquet shard row.

    Args:
        manifest: Loaded array shard manifest.
        shard_id: Zero-based shard identifier.
        row_in_shard: Zero-based row offset inside the shard.

    Returns:
        A decoded `ArrayFeatureBlock`.
    """
    row = _load_array_parquet_row(shard_file_path(manifest.shard_path, shard_id), row_in_shard)
    sample_count = int(row["sample_count"])
    point_count = int(row["point_count"])
    sample_flags = np.frombuffer(row["sample_flags_blob"], dtype=np.uint8, count=sample_count).copy()
    sample_offsets = np.frombuffer(row["sample_offsets_blob"], dtype="<i8", count=sample_count + 1).copy()
    time = np.frombuffer(row["time_blob"], dtype="<f8", count=point_count).astype(np.float64, copy=True)
    value = np.frombuffer(row["value_blob"], dtype="<f8", count=point_count).astype(np.float64, copy=True)
    return ArrayFeatureBlock(
        feature_id=int(row["feature_id"]),
        block_id=int(row["block_id"]),
        sample_row_start=int(row["sample_row_start"]),
        sample_count=sample_count,
        point_count=point_count,
        sample_flags=sample_flags,
        sample_offsets=sample_offsets,
        columns={
            "time": time,
            "value": value,
        },
    )


class ArrayShardReader:
    def __init__(self, manifest: ArrayShardManifest):
        """Create a reader for parquet array shards.

        Args:
            manifest: Loaded array shard manifest.
        """
        self.manifest = manifest

    def load_block(self, shard_id: int, row_in_shard: int):
        """Load one decoded block by physical shard row location.

        Args:
            shard_id: Zero-based shard identifier.
            row_in_shard: Zero-based row offset inside the shard.

        Returns:
            A decoded `ArrayFeatureBlock`.
        """
        return load_array_feature_block(self.manifest, shard_id, row_in_shard)

    def load_feature_samples(self, feature_id: int, sample_rows, locator_index=None):
        """Load traces for one feature at requested sample rows.

        Args:
            feature_id: Logical feature identifier.
            sample_rows: Iterable of global sample row indices.
            locator_index: Optional prebuilt locator index.

        Returns:
            A dictionary keyed by sample row with `ArrayTrace` values. Missing rows
            are represented by empty traces with `flags=0`.
        """
        locator_index = locator_index or build_array_feature_locator_index(self.manifest.locator_path)
        requested = [int(sample_row) for sample_row in sample_rows]
        out = {
            sample_row: ArrayTrace(
                sample_row=sample_row,
                flags=0,
                columns={
                    "time": np.empty(0, dtype=np.float64),
                    "value": np.empty(0, dtype=np.float64),
                },
            )
            for sample_row in requested
        }
        blocks = locator_index.get(int(feature_id), [])
        if not blocks:
            return out

        rows_by_block = {}
        block_by_key = {}
        for sample_row in requested:
            loc = _find_locator_block(blocks, sample_row)
            if loc is None:
                continue
            key = (loc.shard_id, loc.row_in_shard)
            rows_by_block.setdefault(key, []).append(sample_row)
            block_by_key[key] = loc

        for key, rows in rows_by_block.items():
            loc = block_by_key[key]
            block = self.load_block(loc.shard_id, loc.row_in_shard)
            for sample_row in rows:
                trace = block.trace_for_sample_row(sample_row)
                if trace is not None:
                    out[sample_row] = trace
        return out

    def load_feature_samples_by_sample_ids(
        self,
        feature_id: int,
        sample_ids,
        locator_index=None,
        sample_id_index=None,
        sample_meta_path: str = None,
    ):
        """Resolve sample ids to rows and load traces for one feature.

        Args:
            feature_id: Logical feature identifier.
            sample_ids: External sample ids requested by the caller.
            locator_index: Optional prebuilt locator index.
            sample_id_index: Optional sample id to row lookup.
            sample_meta_path: Optional override path for sample metadata.

        Returns:
            A dictionary keyed by sample id with `ArrayTrace` values.
        """
        locator_index = locator_index or build_array_feature_locator_index(self.manifest.locator_path)
        if sample_id_index is None:
            meta_path = sample_meta_path or self.manifest.sample_meta_path
            if not meta_path:
                raise ValueError("sample_meta_path is required to resolve sample ids")
            sample_id_index = build_sample_id_index(meta_path)
        sample_id_list = [int(sample_id) for sample_id in sample_ids]
        sample_rows = [sample_id_index[sample_id] for sample_id in sample_id_list if sample_id in sample_id_index]
        traces_by_row = self.load_feature_samples(feature_id, sample_rows, locator_index=locator_index)
        out = {}
        for sample_id in sample_id_list:
            sample_row = sample_id_index.get(sample_id)
            if sample_row is None:
                out[sample_id] = ArrayTrace(
                    sample_row=-1,
                    flags=0,
                    columns={
                        "time": np.empty(0, dtype=np.float64),
                        "value": np.empty(0, dtype=np.float64),
                    },
                )
            else:
                out[sample_id] = traces_by_row.get(
                    sample_row,
                    ArrayTrace(
                        sample_row=sample_row,
                        flags=0,
                        columns={
                            "time": np.empty(0, dtype=np.float64),
                            "value": np.empty(0, dtype=np.float64),
                        },
                    ),
                )
        return out


def load_array_feature_samples_by_sample_ids(
    manifest,
    feature_id: int,
    sample_ids,
    locator_index=None,
    sample_id_index=None,
    sample_meta_path: str = None,
):
    """Convenience wrapper to load array traces from a manifest path or object.

    Args:
        manifest: Loaded manifest or path to `array_shard_manifest.json`.
        feature_id: Logical feature identifier.
        sample_ids: External sample ids requested by the caller.
        locator_index: Optional prebuilt locator index.
        sample_id_index: Optional sample id to row lookup.
        sample_meta_path: Optional override path for sample metadata.

    Returns:
        A dictionary keyed by sample id with `ArrayTrace` values.
    """
    if isinstance(manifest, str):
        manifest = load_array_shard_manifest(manifest)
    reader = ArrayShardReader(manifest)
    return reader.load_feature_samples_by_sample_ids(
        feature_id=feature_id,
        sample_ids=sample_ids,
        locator_index=locator_index,
        sample_id_index=sample_id_index,
        sample_meta_path=sample_meta_path,
    )


def _find_locator_block(blocks, sample_row: int):
    """Find the locator block that contains a sample row.

    Args:
        blocks: Sorted `ArrayBlockLocation` list for one feature.
        sample_row: Global sample row index to resolve.

    Returns:
        The matching `ArrayBlockLocation`, or `None` if no block covers the row.
    """
    for loc in blocks:
        if loc.contains_sample_row(sample_row):
            return loc
    return None

