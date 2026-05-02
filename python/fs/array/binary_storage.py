import json
import mmap
import os
import shutil
import struct
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np
import polars as pl

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - exercised only when the optional dependency is absent.
    zstd = None

from .storage import (
    _build_array_shard_partitions,
    _build_bucket_partitions,
    _bucket_spill_path,
    _estimate_block_overhead_bytes,
    _point_blob_column_name,
    _default_point_schema,
    _normalize_point_schema,
    list_array_shard_paths,
    load_array_bundle_manifest,
    load_array_shard_manifest,
)
from ..config import ArrayShardConfig
from ..types import (
    ArrayBinaryShardInfo,
    ArrayBinaryShardManifest,
    ArrayFeatureBlock,
    ArrayTrace,
    LogicalType,
    PointColumnSpec,
    point_storage_dtype,
)

CODEC_NONE = 0
CODEC_ZSTD = 1
FILE_VERSION = 3
LEGACY_FILE_VERSION = 2
FILE_ENDIANNESS = "little"
DEFAULT_CODEC_NAME = "none"
DEFAULT_SAMPLE_KEY_COL = "sample_key"
DEFAULT_FEATURE_KEY_COL = "feature_key"

_BINARY_BLOCK_RECORDS_CACHE = OrderedDict()
_BINARY_DATA_MMAP_CACHE = OrderedDict()
_BINARY_CACHE_LOCK = threading.Lock()
_BINARY_BLOCK_RECORDS_CACHE_BYTES = 0
_BINARY_BLOCK_RECORDS_CACHE_MAX_BYTES = 512 * 1024 * 1024
_BINARY_BLOCK_RECORDS_CACHE_MAX_ENTRIES = 256
_BINARY_DATA_MMAP_CACHE_MAX_OPEN = 64

_BLOCKS_INDEX_MAGIC = b"ABLOCKIX"
_BLOCKS_DATA_MAGIC = b"ABLOCKSB"

_FILE_HEADER_STRUCT = struct.Struct("<8sHHHHQQI28x")
_BLOCK_PAYLOAD_HEADER_STRUCT = struct.Struct("<iiqIBBHQIIII")

_BLOCK_RECORD_DTYPE = np.dtype(
    [
        ("data_offset", "<u8"),
        ("data_length", "<u8"),
        ("point_count", "<u8"),
        ("codec", "u1"),
        ("block_flags", "u1"),
        ("reserved0", "<u2"),
        ("crc32_optional", "<u4"),
    ]
)

if _BLOCK_RECORD_DTYPE.itemsize != 32:
    raise AssertionError("unexpected block record size")
if _BLOCK_PAYLOAD_HEADER_STRUCT.size != 48:
    raise AssertionError("unexpected block payload header size")

def _legacy_point_schema():
    return _normalize_point_schema(_default_point_schema())


def _normalize_binary_point_schema(point_schema):
    return _normalize_point_schema(point_schema or _legacy_point_schema())


def _point_dtype(spec) -> np.dtype:
    storage_type = spec.storage_type if hasattr(spec, "storage_type") else spec["storage_type"]
    return point_storage_dtype(storage_type)


def _point_total_bytes(point_schema, point_count: int) -> int:
    return int(sum(int(point_count) * int(_point_dtype(spec).itemsize) for spec in point_schema))


def _build_point_schema_from_manifest_json(data: dict):
    point_schema = data.get("point_schema")
    if point_schema:
        return _normalize_binary_point_schema(point_schema)
    return _legacy_point_schema()


def _copy_categorical_dictionaries(manifest_path: str, point_schema, out_dir: str):
    """Copy categorical dictionary files into the output artifact and rewrite paths."""
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    out_specs = []
    for spec in point_schema:
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
        dictionary_path = str(item.dictionary_path or "")
        if dictionary_path:
            source_path = dictionary_path if os.path.isabs(dictionary_path) else os.path.join(manifest_dir, dictionary_path)
            dict_dir = os.path.join(out_dir, "categorical_dictionaries")
            os.makedirs(dict_dir, exist_ok=True)
            destination = os.path.join(dict_dir, os.path.basename(source_path))
            if os.path.normcase(os.path.abspath(source_path)) != os.path.normcase(os.path.abspath(destination)):
                shutil.copy2(source_path, destination)
            item = PointColumnSpec(
                name=item.name,
                storage_type=item.storage_type,
                logical_type=item.logical_type,
                dictionary_path=os.path.join("categorical_dictionaries", os.path.basename(destination)),
            )
        out_specs.append(item)
    return out_specs


def _binary_shard_base(shard_path: str, shard_id: int) -> str:
    """Return the common basename used by one binary shard's files.

    Args:
        shard_path: Directory containing binary shard files.
        shard_id: Zero-based shard identifier.

    Returns:
        Base path without the `.blocks.idx` or `.blocks.bin` suffix.
    """
    return os.path.join(shard_path, f"shard_{int(shard_id):04d}")


def binary_blocks_index_path(shard_path: str, shard_id: int) -> str:
    """Return the `blocks.idx` path for one binary shard."""
    return _binary_shard_base(shard_path, shard_id) + ".blocks.idx"


def binary_blocks_data_path(shard_path: str, shard_id: int) -> str:
    """Return the `blocks.bin` path for one binary shard."""
    return _binary_shard_base(shard_path, shard_id) + ".blocks.bin"


def _artifact_relative_path(manifest_path: str, target_path: str) -> str:
    """Return a manifest-relative path for one artifact file or directory.

    Args:
        manifest_path: Path to the manifest that will reference the artifact.
        target_path: Absolute or relative filesystem path to store in the manifest.

    Returns:
        A normalized relative path from the manifest directory to `target_path`.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    return os.path.relpath(os.path.abspath(target_path), manifest_dir)


def _resolve_manifest_path(manifest_path: str, stored_path: str) -> str:
    """Resolve one manifest path field against the manifest location.

    Args:
        manifest_path: Path to the manifest file on disk.
        stored_path: Path string stored inside the manifest JSON.

    Returns:
        An absolute path suitable for runtime file access.
    """
    stored = str(stored_path)
    if os.path.isabs(stored):
        return stored
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    return os.path.abspath(os.path.join(manifest_dir, stored))


def _materialize_metadata_file(source_path: str, out_dir: str, target_name: str) -> str:
    """Copy one metadata file into the binary artifact root.

    Args:
        source_path: Original metadata file path.
        out_dir: Binary artifact root directory.
        target_name: Standardized filename inside the artifact root.

    Returns:
        Absolute path to the copied metadata file inside `out_dir`.
    """
    destination = os.path.abspath(os.path.join(out_dir, target_name))
    source_abs = os.path.abspath(source_path)
    if source_abs != destination:
        shutil.copy2(source_abs, destination)
    return destination


def _materialize_binary_metadata(out_dir: str, sample_meta_path: str, feature_meta_path: str):
    """Copy dataset metadata into one self-contained binary artifact directory.

    Args:
        out_dir: Binary artifact root directory.
        sample_meta_path: Source sample metadata parquet path.
        feature_meta_path: Source feature metadata parquet path.

    Returns:
        Tuple of absolute destination paths:
        `(artifact_sample_meta_path, artifact_feature_meta_path)`.
    """
    os.makedirs(out_dir, exist_ok=True)
    artifact_sample_meta_path = _materialize_metadata_file(sample_meta_path, out_dir, "sample_meta.parquet")
    artifact_feature_meta_path = _materialize_metadata_file(feature_meta_path, out_dir, "feature_meta.parquet")
    return artifact_sample_meta_path, artifact_feature_meta_path


def _write_file_header(path: str, magic: bytes, record_bytes: int, entry_count: int, aux_count: int, shard_id: int):
    """Write a fixed-size binary file header in-place.

    Args:
        path: Destination binary file path.
        magic: File magic identifying the file kind.
        record_bytes: Record size stored in the header.
        entry_count: Number of logical records in the file.
        aux_count: Auxiliary count field whose meaning depends on the file type.
        shard_id: Owning shard identifier.
    """
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(
            _FILE_HEADER_STRUCT.pack(
                magic,
                FILE_VERSION,
                _FILE_HEADER_STRUCT.size,
                int(record_bytes),
                0,
                int(entry_count),
                int(aux_count),
                int(shard_id),
            )
        )


def _read_file_header(path: str, expected_magic: bytes):
    """Read and validate a binary file header.

    Args:
        path: Binary file path.
        expected_magic: Expected magic bytes for the file type.

    Returns:
        A dictionary with parsed header fields.

    Raises:
        ValueError: If the magic or version does not match expectations.
    """
    with open(path, "rb") as f:
        data = f.read(_FILE_HEADER_STRUCT.size)
    if len(data) != _FILE_HEADER_STRUCT.size:
        raise ValueError(f"corrupt binary shard header: {path}")
    magic, version, header_bytes, record_bytes, flags, entry_count, aux_count, shard_id = _FILE_HEADER_STRUCT.unpack(
        data
    )
    if magic != expected_magic:
        raise ValueError(f"unexpected magic for {path}: {magic!r}")
    if int(version) not in {int(FILE_VERSION), int(LEGACY_FILE_VERSION)}:
        raise ValueError(f"unsupported version for {path}: {version}")
    if header_bytes != _FILE_HEADER_STRUCT.size:
        raise ValueError(f"unexpected header size for {path}: {header_bytes}")
    return {
        "record_bytes": int(record_bytes),
        "version": int(version),
        "flags": int(flags),
        "entry_count": int(entry_count),
        "aux_count": int(aux_count),
        "shard_id": int(shard_id),
    }


def _write_empty_binary_files(shard_path: str, shard_id: int):
    """Create empty binary shard files with valid headers."""
    for path, magic, record_bytes in [
        (binary_blocks_index_path(shard_path, shard_id), _BLOCKS_INDEX_MAGIC, _BLOCK_RECORD_DTYPE.itemsize),
        (binary_blocks_data_path(shard_path, shard_id), _BLOCKS_DATA_MAGIC, 0),
    ]:
        with open(path, "wb") as f:
            f.write(b"\x00" * _FILE_HEADER_STRUCT.size)
        _write_file_header(path, magic, record_bytes, 0, 0, shard_id)


def _normalize_codec(codec: str) -> int:
    """Convert a codec name into the binary codec enum."""
    name = str(codec).strip().lower()
    if name in {"", "none", "raw"}:
        return CODEC_NONE
    if name == "zstd":
        return CODEC_ZSTD
    raise ValueError(f"unsupported binary shard codec: {codec}")


def _codec_name(codec_id: int) -> str:
    """Convert a binary codec enum back into its canonical name."""
    if int(codec_id) == CODEC_NONE:
        return "none"
    if int(codec_id) == CODEC_ZSTD:
        return "zstd"
    raise ValueError(f"unsupported codec id: {codec_id}")


def _require_zstd():
    """Return the imported `zstandard` module or raise a helpful error."""
    if zstd is None:
        raise RuntimeError("zstandard is required for codec='zstd'. Install the 'zstandard' package.")
    return zstd


def _maybe_compress_payload(codec_id: int, payload: bytes, compressor):
    """Optionally compress a payload according to the requested codec."""
    if int(codec_id) == CODEC_NONE or not payload:
        return payload
    if int(codec_id) == CODEC_ZSTD:
        if compressor is None:
            compressor = _require_zstd().ZstdCompressor(level=3)
        return compressor.compress(payload)
    raise ValueError(f"unsupported codec id: {codec_id}")


def _maybe_decompress_payload(codec_id: int, payload: bytes, decompressor, expected_bytes: int):
    """Optionally decompress a payload and validate its decoded size."""
    if not payload:
        if int(expected_bytes) != 0:
            raise ValueError(f"empty payload for expected decoded length {expected_bytes}")
        return b""
    if int(codec_id) == CODEC_NONE:
        return payload
    if int(codec_id) == CODEC_ZSTD:
        if decompressor is None:
            decompressor = _require_zstd().ZstdDecompressor()
        out = decompressor.decompress(payload, max_output_size=int(expected_bytes))
        if len(out) != int(expected_bytes):
            raise ValueError(f"decoded payload length mismatch: expected={expected_bytes} got={len(out)}")
        return out
    raise ValueError(f"unsupported codec id: {codec_id}")


def _validate_dense_id_column(df: pl.DataFrame, id_col: str, entity_name: str):
    """Validate that an optional id column matches dense row-order ids."""
    if id_col not in df.columns:
        return
    ids = df[id_col].to_numpy().astype(np.int64, copy=False)
    expected = np.arange(df.height, dtype=np.int64)
    if ids.shape[0] != expected.shape[0] or not np.array_equal(ids, expected):
        raise ValueError(
            f"{entity_name} metadata column '{id_col}' must equal dense row ids 0..N-1 in row order"
        )


def _validate_unique_key_column(df: pl.DataFrame, key_col: str, entity_name: str):
    """Validate that an optional external key column is unique and non-null."""
    if not key_col:
        return
    if key_col not in df.columns:
        raise ValueError(f"{entity_name} metadata must have key column: {key_col}")
    series = df[key_col]
    if series.null_count() != 0:
        raise ValueError(f"{entity_name} metadata key column '{key_col}' must not contain nulls")
    if int(series.n_unique()) != int(df.height):
        raise ValueError(f"{entity_name} metadata key column '{key_col}' must be unique")


def _load_dense_meta(meta_path: str, id_col: str, entity_name: str, key_col: str = "") -> pl.DataFrame:
    """Load metadata whose row order defines dense internal ids."""
    if not meta_path:
        raise ValueError(f"{entity_name} metadata path is required")
    df = pl.read_parquet(meta_path)
    _validate_dense_id_column(df, id_col, entity_name)
    _validate_unique_key_column(df, key_col, entity_name)
    return df


def _blocks_per_feature(n_samples: int, samples_per_block: int) -> int:
    """Return the number of logical blocks assigned to one feature."""
    if samples_per_block <= 0:
        raise ValueError("samples_per_block must be > 0")
    return int((int(n_samples) + int(samples_per_block) - 1) // int(samples_per_block))


def _sample_count_for_block(n_samples: int, samples_per_block: int, block_id: int) -> int:
    """Return the number of samples covered by one block id."""
    start = int(block_id) * int(samples_per_block)
    if start >= int(n_samples):
        return 0
    return int(min(int(samples_per_block), int(n_samples) - start))


def _point_column_blob_names(point_schema):
    """Return ordered bundle/blob column names for one point schema."""
    return [_point_blob_column_name(spec.name) for spec in point_schema]


def _empty_spill_frame(point_schema):
    """Create an empty spill dataframe matching the v3 bundle row layout."""
    columns = {
        "feature_id": pl.Series("feature_id", [], dtype=pl.Int32),
        "sample_row": pl.Series("sample_row", [], dtype=pl.Int64),
        "flags": pl.Series("flags", [], dtype=pl.UInt8),
        "trace_len": pl.Series("trace_len", [], dtype=pl.Int32),
    }
    for spec in point_schema:
        columns[_point_blob_column_name(spec.name)] = pl.Series(_point_blob_column_name(spec.name), [], dtype=pl.Binary)
    return pl.DataFrame(columns)


def _validate_point_trace_row(sample_row: int, n_samples: int, trace_len: int, row: dict, point_schema):
    """Validate one generic point-trace row before block packing or spill serialization."""
    if int(sample_row) < 0 or int(sample_row) >= int(n_samples):
        raise ValueError(f"sample_row out of range: {sample_row}")
    if int(trace_len) < 0:
        raise ValueError("trace_len must be >= 0")
    for spec in point_schema:
        blob_name = _point_blob_column_name(spec.name)
        blob = row.get(blob_name) or b""
        expected = int(trace_len) * int(_point_dtype(spec).itemsize)
        actual = len(blob)
        if actual != expected:
            raise ValueError(
                f"{blob_name} length mismatch: expected={expected} got={actual}"
            )


class _DensePointBlockAccumulator:
    """Accumulate one feature block with arbitrary fixed-width point columns."""

    def __init__(self, feature_id: int, block_id: int, sample_row: int, samples_per_block: int, n_samples: int, point_schema):
        self.feature_id = int(feature_id)
        self.block_id = int(block_id)
        self.sample_row_start = self.block_id * int(samples_per_block)
        self.sample_count = int(min(samples_per_block, n_samples - self.sample_row_start))
        if self.sample_count <= 0:
            raise ValueError(f"invalid block sample_count for sample_row={sample_row}")
        self.point_schema = list(point_schema)
        self.sample_flags = np.zeros(self.sample_count, dtype=np.uint8)
        self.sample_offsets = np.zeros(self.sample_count + 1, dtype=np.int64)
        self.column_chunks = {spec.name: [] for spec in self.point_schema}
        self.point_count = 0
        self.next_relative_sample = 0
        self.present_count = 0

    def append(self, sample_row: int, flags: int, trace_len: int, row: dict):
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
        if (out_flags & 0x01) == 0:
            out_flags |= 0x01
        if int(trace_len) == 0 and (out_flags & 0x02) == 0:
            out_flags |= 0x02
        self.sample_flags[relative_sample] = out_flags

        if int(trace_len) > 0:
            for spec in self.point_schema:
                blob_name = _point_blob_column_name(spec.name)
                self.column_chunks[spec.name].append(row[blob_name] or b"")
            self.point_count += int(trace_len)

        self.sample_offsets[relative_sample + 1] = self.point_count
        self.next_relative_sample = relative_sample + 1
        self.present_count += 1

    def finish(self):
        while self.next_relative_sample < self.sample_count:
            self.sample_offsets[self.next_relative_sample + 1] = self.point_count
            self.next_relative_sample += 1

    def has_present_rows(self) -> bool:
        return self.present_count > 0


def _finalize_dense_point_block(block, current_feature_id, shard_rows):
    """Flush one completed generic point block into serialized shard rows."""
    if block is None or current_feature_id is None:
        return
    block.finish()
    if not block.has_present_rows():
        return
    shard_rows.append(
        {
            "feature_id": int(current_feature_id),
            "block_id": int(block.block_id),
            "sample_row_start": int(block.sample_row_start),
            "sample_count": int(block.sample_count),
            "point_count": int(block.point_count),
            "sample_flags_blob": block.sample_flags.tobytes(),
            "sample_offsets_blob": np.asarray(block.sample_offsets, dtype="<i8").tobytes(),
            "column_blobs": {
                spec.name: b"".join(block.column_chunks[spec.name])
                for spec in block.point_schema
            },
        }
    )


def _process_sorted_rows_v3(df: pl.DataFrame, n_samples: int, samples_per_block: int, point_schema):
    """Compact sorted spill rows into dense block rows for one shard."""
    if df.height == 0:
        return []

    feature_ids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    sample_rows = df["sample_row"].to_numpy().astype(np.int64, copy=False)
    flags = df["flags"].to_numpy().astype(np.uint8, copy=False)
    trace_lens = df["trace_len"].to_numpy().astype(np.int32, copy=False)
    blob_columns = {
        _point_blob_column_name(spec.name): df[_point_blob_column_name(spec.name)].to_list()
        for spec in point_schema
    }

    shard_rows = []
    current_feature_id = None
    current_block_id = None
    block = None

    for idx in range(df.height):
        feature_id = int(feature_ids[idx])
        sample_row = int(sample_rows[idx])
        trace_len = int(trace_lens[idx])
        row = {name: values[idx] for name, values in blob_columns.items()}
        _validate_point_trace_row(sample_row, n_samples, trace_len, row, point_schema)
        block_id = sample_row // int(samples_per_block)

        if current_feature_id is None or feature_id != current_feature_id or block_id != current_block_id:
            _finalize_dense_point_block(block, current_feature_id, shard_rows)
            current_feature_id = feature_id
            current_block_id = block_id
            block = _DensePointBlockAccumulator(
                feature_id=feature_id,
                block_id=block_id,
                sample_row=sample_row,
                samples_per_block=samples_per_block,
                n_samples=n_samples,
                point_schema=point_schema,
            )

        block.append(
            sample_row=sample_row,
            flags=int(flags[idx]),
            trace_len=trace_len,
            row=row,
        )

    _finalize_dense_point_block(block, current_feature_id, shard_rows)
    return shard_rows


def _append_frame_to_spill_file_v3(df: pl.DataFrame, spill_path: str, point_schema):
    """Append generic point-trace rows to one spill file."""
    os.makedirs(os.path.dirname(spill_path), exist_ok=True)
    blob_names = _point_column_blob_names(point_schema)
    delta = 0
    with open(spill_path, "ab") as f:
        for row in df.iter_rows(named=True):
            trace_len = int(row["trace_len"])
            _validate_point_trace_row(int(row["sample_row"]), 1 << 60, trace_len, row, point_schema)
            header = struct.pack("<iqBi", int(row["feature_id"]), int(row["sample_row"]), int(row["flags"]), trace_len)
            f.write(header)
            delta += len(header)
            for blob_name in blob_names:
                blob = row[blob_name] or b""
                f.write(blob)
                delta += len(blob)
    return int(delta)


def _load_spill_file_to_sorted_df_v3(spill_path: str, point_schema):
    """Load one generic point spill file into a sorted dataframe."""
    if not os.path.exists(spill_path) or os.path.getsize(spill_path) == 0:
        return _empty_spill_frame(point_schema)

    rows = {
        "feature_id": [],
        "sample_row": [],
        "flags": [],
        "trace_len": [],
    }
    for spec in point_schema:
        rows[_point_blob_column_name(spec.name)] = []

    header_struct = struct.Struct("<iqBi")
    blob_names = _point_column_blob_names(point_schema)
    dtypes = [_point_dtype(spec) for spec in point_schema]

    with open(spill_path, "rb") as f:
        while True:
            header = f.read(header_struct.size)
            if not header:
                break
            if len(header) != header_struct.size:
                raise ValueError(f"corrupt spill header: {spill_path}")
            feature_id, sample_row, flags, trace_len = header_struct.unpack(header)
            trace_len = int(trace_len)
            rows["feature_id"].append(int(feature_id))
            rows["sample_row"].append(int(sample_row))
            rows["flags"].append(int(flags))
            rows["trace_len"].append(trace_len)
            for blob_name, dtype in zip(blob_names, dtypes):
                byte_count = int(trace_len) * int(dtype.itemsize)
                blob = f.read(byte_count)
                if len(blob) != byte_count:
                    raise ValueError(f"corrupt spill payload: {spill_path}")
                rows[blob_name].append(blob)

    frame = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", rows["feature_id"], dtype=pl.Int32),
            "sample_row": pl.Series("sample_row", rows["sample_row"], dtype=pl.Int64),
            "flags": pl.Series("flags", rows["flags"], dtype=pl.UInt8),
            "trace_len": pl.Series("trace_len", rows["trace_len"], dtype=pl.Int32),
            **{
                blob_name: pl.Series(blob_name, rows[blob_name], dtype=pl.Binary)
                for blob_name in blob_names
            },
        }
    )
    if frame.height == 0:
        return frame
    return frame.sort(["feature_id", "sample_row"])


def _dense_feature_estimates_from_bundles(bundle_paths, n_features: int, n_samples: int, samples_per_block: int, point_schema):
    """Estimate per-feature bytes for dense-id shard partition planning."""
    blocks_per_feature = _blocks_per_feature(n_samples, samples_per_block)
    estimates = np.full(
        int(n_features),
        int(blocks_per_feature) * int(_BLOCK_RECORD_DTYPE.itemsize),
        dtype=np.int64,
    )
    if not bundle_paths:
        return estimates
    bytes_per_point = int(sum(int(_point_dtype(spec).itemsize) for spec in point_schema))
    block_overhead = int(_estimate_block_overhead_bytes(samples_per_block))
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
    observed = total_trace_len * np.int64(bytes_per_point) + block_count * np.int64(block_overhead)
    for feature_id, est in zip(feature_ids.tolist(), observed.tolist()):
        feature_id = int(feature_id)
        if feature_id < 0 or feature_id >= int(n_features):
            raise ValueError(f"bundle feature_id out of dense metadata range: {feature_id}")
        estimates[feature_id] += int(est)
    return estimates


def _dense_feature_estimates_from_parquet_shards(shard_paths, n_features: int, n_samples: int, samples_per_block: int):
    """Estimate per-feature bytes from existing parquet shard rows."""
    blocks_per_feature = _blocks_per_feature(n_samples, samples_per_block)
    estimates = np.full(
        int(n_features),
        int(blocks_per_feature) * int(_BLOCK_RECORD_DTYPE.itemsize),
        dtype=np.int64,
    )
    if not shard_paths:
        return estimates
    df = (
        pl.scan_parquet(shard_paths)
        .select(
            [
                pl.col("feature_id").cast(pl.Int32),
                pl.col("point_count").cast(pl.Int64),
            ]
        )
        .group_by("feature_id")
        .agg(pl.col("point_count").sum().alias("point_count"))
        .sort("feature_id")
        .collect()
    )
    feature_ids = df["feature_id"].to_numpy().astype(np.int32, copy=False)
    point_counts = df["point_count"].to_numpy().astype(np.int64, copy=False)
    for feature_id, point_count in zip(feature_ids.tolist(), point_counts.tolist()):
        feature_id = int(feature_id)
        if feature_id < 0 or feature_id >= int(n_features):
            raise ValueError(f"parquet shard feature_id out of dense metadata range: {feature_id}")
        estimates[feature_id] += int(point_count) * 16
    return estimates


def _collect_sorted_rows_from_parquet_shards(shard_paths, shard_feature_ids):
    """Collect and sort parquet array-shard rows belonging to one dense shard partition."""
    if not shard_paths or len(shard_feature_ids) == 0:
        return pl.DataFrame(
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
    feature_id_list = [int(feature_id) for feature_id in shard_feature_ids.tolist()]
    return (
        pl.scan_parquet(shard_paths)
        .filter(pl.col("feature_id").is_in(feature_id_list))
        .select(
            [
                "feature_id",
                "block_id",
                "sample_row_start",
                "sample_count",
                "point_count",
                "sample_flags_blob",
                "sample_offsets_blob",
                "time_blob",
                "value_blob",
            ]
        )
        .sort(["feature_id", "block_id"])
        .collect()
    )


def _write_dense_binary_shard(
    shard_path: str,
    shard_id: int,
    shard_feature_ids,
    rows,
    *,
    n_samples: int,
    samples_per_block: int,
    point_schema,
    codec: str = "none",
    zstd_level: int = 3,
):
    """Write one dense-id binary shard from serialized block rows.

    Args:
        shard_path: Directory containing binary shard files.
        shard_id: Zero-based shard identifier.
        shard_feature_ids: Dense feature ids assigned to the shard.
        rows: Serialized block rows for the shard. Missing blocks may be omitted.
        n_samples: Total number of dense sample ids.
        samples_per_block: Logical block size.
        codec: Payload codec name.
        zstd_level: Compression level when `codec='zstd'`.

    Returns:
        A statistics dictionary including the emitted `ArrayBinaryShardInfo`.
    """
    blocks_index_path = binary_blocks_index_path(shard_path, shard_id)
    blocks_data_path = binary_blocks_data_path(shard_path, shard_id)
    codec_id = _normalize_codec(codec)
    compressor = _require_zstd().ZstdCompressor(level=int(zstd_level)) if codec_id == CODEC_ZSTD else None
    point_schema = _normalize_binary_point_schema(point_schema)

    feature_count = int(len(shard_feature_ids))
    blocks_per_feature = _blocks_per_feature(n_samples, samples_per_block)
    if feature_count == 0:
        _write_empty_binary_files(shard_path, shard_id)
        return {
            "shard_info": ArrayBinaryShardInfo(
                shard_id=shard_id,
                feature_id_start=-1,
                feature_id_end=-1,
                feature_count=0,
                block_count=0,
                blocks_index_name=os.path.basename(blocks_index_path),
                blocks_data_name=os.path.basename(blocks_data_path),
            ),
            "data_bytes": os.path.getsize(blocks_data_path),
            "index_bytes": os.path.getsize(blocks_index_path),
        }

    feature_id_start = int(shard_feature_ids[0])
    feature_id_end = int(shard_feature_ids[-1])
    block_count = int(feature_count) * int(blocks_per_feature)
    block_records = np.zeros(block_count, dtype=_BLOCK_RECORD_DTYPE)

    with open(blocks_data_path, "wb") as data_f:
        data_f.write(b"\x00" * _FILE_HEADER_STRUCT.size)

        for row in rows:
            feature_id = int(row["feature_id"])
            block_id = int(row["block_id"])
            local_feature = feature_id - feature_id_start
            if local_feature < 0 or local_feature >= feature_count:
                raise ValueError(f"feature_id out of shard range: {feature_id}")
            if block_id < 0 or block_id >= blocks_per_feature:
                raise ValueError(f"block_id out of range: {block_id}")

            sample_row_start = int(row["sample_row_start"])
            sample_count = int(row["sample_count"])
            point_count = int(row["point_count"])
            sample_flags_blob = row["sample_flags_blob"] or b""
            sample_offsets_blob = row["sample_offsets_blob"] or b""
            column_blobs = row.get("column_blobs") or {}
            raw_columns_payload = b"".join((column_blobs.get(spec.name) or b"") for spec in point_schema)
            expected_raw_bytes = _point_total_bytes(point_schema, point_count)
            if len(raw_columns_payload) != int(expected_raw_bytes):
                raise ValueError(
                    f"column payload length mismatch for feature_id={feature_id} block_id={block_id}: "
                    f"expected={expected_raw_bytes} got={len(raw_columns_payload)}"
                )
            encoded_columns_payload = _maybe_compress_payload(codec_id, raw_columns_payload, compressor)

            record_index = int(local_feature) * int(blocks_per_feature) + int(block_id)
            data_offset = data_f.tell()
            payload_header = _BLOCK_PAYLOAD_HEADER_STRUCT.pack(
                feature_id,
                block_id,
                sample_row_start,
                sample_count,
                codec_id,
                0,
                len(point_schema),
                point_count,
                len(sample_flags_blob),
                len(sample_offsets_blob),
                len(encoded_columns_payload),
                0,
            )
            data_f.write(payload_header)
            data_f.write(sample_flags_blob)
            data_f.write(sample_offsets_blob)
            data_f.write(encoded_columns_payload)
            data_length = data_f.tell() - data_offset

            block_records[record_index]["data_offset"] = np.uint64(data_offset)
            block_records[record_index]["data_length"] = np.uint64(data_length)
            block_records[record_index]["point_count"] = np.uint64(point_count)
            block_records[record_index]["codec"] = np.uint8(codec_id)
            block_records[record_index]["block_flags"] = np.uint8(0)
            block_records[record_index]["crc32_optional"] = np.uint32(0)

    with open(blocks_index_path, "wb") as f:
        f.write(b"\x00" * _FILE_HEADER_STRUCT.size)
        if block_records.size > 0:
            block_records.tofile(f)

    _write_file_header(
        blocks_index_path,
        _BLOCKS_INDEX_MAGIC,
        _BLOCK_RECORD_DTYPE.itemsize,
        block_records.size,
        feature_count,
        shard_id,
    )
    data_file_size = os.path.getsize(blocks_data_path)
    _write_file_header(
        blocks_data_path,
        _BLOCKS_DATA_MAGIC,
        0,
        block_records.size,
        max(data_file_size - _FILE_HEADER_STRUCT.size, 0),
        shard_id,
    )
    return {
        "shard_info": ArrayBinaryShardInfo(
            shard_id=shard_id,
            feature_id_start=feature_id_start,
            feature_id_end=feature_id_end,
            feature_count=feature_count,
            block_count=block_count,
            blocks_index_name=os.path.basename(blocks_index_path),
            blocks_data_name=os.path.basename(blocks_data_path),
        ),
        "data_bytes": os.path.getsize(blocks_data_path),
        "index_bytes": os.path.getsize(blocks_index_path),
    }


def _write_binary_manifest(
    *,
    sample_meta_path: str,
    feature_meta_path: str,
    n_samples: int,
    n_features: int,
    shard_path: str,
    manifest_path: str,
    samples_per_block: int,
    blocks_per_feature: int,
    shard_infos,
    default_codec: str,
    point_schema,
    sample_key_col: str = DEFAULT_SAMPLE_KEY_COL,
    feature_key_col: str = DEFAULT_FEATURE_KEY_COL,
):
    """Write the top-level binary shard manifest JSON file."""
    binary_manifest = ArrayBinaryShardManifest(
        sample_meta_path=_artifact_relative_path(manifest_path, sample_meta_path),
        feature_meta_path=_artifact_relative_path(manifest_path, feature_meta_path),
        n_samples=int(n_samples),
        n_features=int(n_features),
        shard_path=_artifact_relative_path(manifest_path, shard_path),
        n_shards=len(shard_infos),
        samples_per_block=int(samples_per_block),
        blocks_per_feature=int(blocks_per_feature),
        feature_id_dtype="INT32",
        flags_dtype="UINT8",
        offset_dtype="INT64",
        time_dtype="FLOAT64_LE" if any(spec.name == "time" for spec in point_schema) else "",
        value_dtype="FLOAT64_LE" if any(spec.name == "value" for spec in point_schema) else "",
        default_codec=str(default_codec),
        endianness=FILE_ENDIANNESS,
        id_scheme="dense_row_ids",
        sample_key_col=str(sample_key_col),
        feature_key_col=str(feature_key_col),
        shards=shard_infos,
        point_schema=point_schema,
        version=3,
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(binary_manifest.to_json(), f, indent=2)
    return manifest_path


def _build_array_binary_shards_with_tmp_spill(
    *,
    bundle_manifest_path: str,
    bundle_manifest,
    bundle_paths,
    feature_ids,
    estimated_feature_bytes,
    shard_partitions,
    out_dir: str,
    config: ArrayShardConfig,
    codec: str,
    zstd_level: int,
    n_features: int,
    sample_key_col: str,
    feature_key_col: str,
    point_schema,
):
    """Build binary shards through append-only temporary spill buckets.

    Each `(shard_id, bucket_id)` pair owns one spill file under `_tmp/`. As bundle
    files are scanned, rows are partitioned by `(shard_id, bucket_id)` and appended
    to that fixed spill file. Later, each spill file is loaded once, sorted, compacted
    into block rows, and then deleted.
    """
    artifact_sample_meta_path, artifact_feature_meta_path = _materialize_binary_metadata(
        out_dir,
        bundle_manifest.sample_meta_path,
        bundle_manifest.feature_meta_path,
    )
    artifact_point_schema = _copy_categorical_dictionaries(bundle_manifest_path, point_schema, out_dir)
    shard_path = os.path.join(out_dir, "array_binary_feature_shards")
    os.makedirs(shard_path, exist_ok=True)
    manifest_path = os.path.join(out_dir, "array_binary_shard_manifest.json")
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
    blocks_per_feature = _blocks_per_feature(bundle_manifest.n_samples, config.samples_per_block)
    bundle_blob_columns = _point_column_blob_names(point_schema)

    try:
        for bundle_path in bundle_paths:
            bundle_df = pl.read_parquet(
                bundle_path,
                columns=[
                    "feature_id",
                    "sample_row",
                    "flags",
                    "trace_len",
                    *bundle_blob_columns,
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
                delta = _append_frame_to_spill_file_v3(
                    part_df.select(
                        [
                            "feature_id",
                            "sample_row",
                            "flags",
                            "trace_len",
                            *bundle_blob_columns,
                        ]
                    ),
                    spill_path,
                    point_schema,
                )
                spill_sizes[spill_key] = spill_sizes.get(spill_key, 0) + int(delta)
                live_temp_bytes += int(delta)
                peak_live_temp_files = max(peak_live_temp_files, live_temp_files)
                peak_live_temp_bytes = max(peak_live_temp_bytes, live_temp_bytes)

        shard_infos = []
        total_data_bytes = 0
        total_index_bytes = 0
        total_block_count = 0
        total_feature_count = 0

        for shard_id, bucket_partitions in enumerate(shard_bucket_partitions):
            shard_rows = []
            for bucket_id, _bucket_feature_ids in enumerate(bucket_partitions):
                spill_key = (shard_id, bucket_id)
                spill_path = spill_paths.get(spill_key)
                if spill_path is None or not os.path.exists(spill_path):
                    continue
                df = _load_spill_file_to_sorted_df_v3(spill_path, point_schema)
                bucket_rows = _process_sorted_rows_v3(
                    df,
                    bundle_manifest.n_samples,
                    config.samples_per_block,
                    point_schema,
                )
                shard_rows.extend(bucket_rows)

                spill_size = spill_sizes.pop(spill_key, 0)
                os.remove(spill_path)
                live_temp_files -= 1
                live_temp_bytes -= int(spill_size)

            shard_write = _write_dense_binary_shard(
                shard_path,
                shard_id,
                shard_partitions[shard_id],
                shard_rows,
                n_samples=bundle_manifest.n_samples,
                samples_per_block=config.samples_per_block,
                point_schema=point_schema,
                codec=codec,
                zstd_level=zstd_level,
            )
            shard_infos.append(shard_write["shard_info"])
            total_data_bytes += int(shard_write["data_bytes"])
            total_index_bytes += int(shard_write["index_bytes"])
            total_block_count += int(shard_write["shard_info"].block_count)
            total_feature_count += int(shard_write["shard_info"].feature_count)

        _write_binary_manifest(
            sample_meta_path=artifact_sample_meta_path,
            feature_meta_path=artifact_feature_meta_path,
            n_samples=bundle_manifest.n_samples,
            n_features=n_features,
            shard_path=shard_path,
            manifest_path=manifest_path,
            samples_per_block=config.samples_per_block,
            blocks_per_feature=blocks_per_feature,
            shard_infos=shard_infos,
            default_codec=_codec_name(_normalize_codec(codec)),
            point_schema=artifact_point_schema,
            sample_key_col=sample_key_col,
            feature_key_col=feature_key_col,
        )
        stats = {
            "n_shards": int(len(shard_infos)),
            "n_buckets": int(sum(len(parts) for parts in shard_bucket_partitions)),
            "temp_files_created": int(temp_files_created),
            "peak_live_temp_files": int(peak_live_temp_files),
            "peak_live_temp_bytes": int(peak_live_temp_bytes),
            "feature_count": int(total_feature_count),
            "block_count": int(total_block_count),
            "data_bytes": int(total_data_bytes),
            "index_bytes": int(total_index_bytes),
            "total_bytes": int(total_data_bytes + total_index_bytes + os.path.getsize(manifest_path)),
        }
        return manifest_path, stats
    finally:
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)


def build_array_binary_shards_from_bundles(
    bundle_manifest_path: str,
    out_dir: str,
    *,
    config: ArrayShardConfig = None,
    codec: str = DEFAULT_CODEC_NAME,
    zstd_level: int = 3,
    return_stats: bool = False,
    sample_key_col: str = DEFAULT_SAMPLE_KEY_COL,
    feature_key_col: str = DEFAULT_FEATURE_KEY_COL,
):
    """Build dense-id binary shards from a sample-major bundle manifest.

    Args:
        bundle_manifest_path: Path to `array_bundle_manifest.json`.
        out_dir: Output directory for the binary shard set.
        config: Optional shard build configuration. Binary builds always use the
            append-only temporary spill path even if `config.use_tmp_spill` is `False`.
        codec: Payload codec name.
        zstd_level: Compression level when `codec='zstd'`.
        return_stats: Whether to return `(manifest_path, stats)` instead of only the manifest path.
        sample_key_col: Metadata column containing external sample keys.
        feature_key_col: Metadata column containing external feature keys.

    Returns:
        Manifest path, or `(manifest_path, stats)` when `return_stats=True`.
    """
    config = config or ArrayShardConfig()
    bundle_manifest = load_array_bundle_manifest(bundle_manifest_path)
    bundle_paths = [os.path.join(bundle_manifest.bundle_path, f"bundle_{bundle_id:06d}.parquet") for bundle_id in range(bundle_manifest.n_bundles)]
    point_schema = _normalize_binary_point_schema(getattr(bundle_manifest, "point_schema", None))
    sample_meta_df = _load_dense_meta(bundle_manifest.sample_meta_path, "sample_id", "sample", sample_key_col)
    feature_meta_df = _load_dense_meta(bundle_manifest.feature_meta_path, "feature_id", "feature", feature_key_col)
    if int(sample_meta_df.height) != int(bundle_manifest.n_samples):
        raise ValueError("bundle manifest n_samples does not match sample metadata row count")
    n_features = int(feature_meta_df.height)
    feature_ids = np.arange(n_features, dtype=np.int32)
    estimated_feature_bytes = _dense_feature_estimates_from_bundles(
        bundle_paths,
        n_features=n_features,
        n_samples=bundle_manifest.n_samples,
        samples_per_block=config.samples_per_block,
        point_schema=point_schema,
    )
    shard_partitions = _build_array_shard_partitions(feature_ids, estimated_feature_bytes, config)
    manifest_path, stats = _build_array_binary_shards_with_tmp_spill(
        bundle_manifest_path=bundle_manifest_path,
        bundle_manifest=bundle_manifest,
        bundle_paths=bundle_paths,
        feature_ids=feature_ids,
        estimated_feature_bytes=estimated_feature_bytes,
        shard_partitions=shard_partitions,
        out_dir=out_dir,
        config=config,
        codec=codec,
        zstd_level=zstd_level,
        n_features=n_features,
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
        point_schema=point_schema,
    )
    if return_stats:
        return manifest_path, stats
    return manifest_path


def build_array_binary_shards_from_array_manifest(
    array_manifest_path: str,
    out_dir: str,
    *,
    codec: str = DEFAULT_CODEC_NAME,
    zstd_level: int = 3,
    return_stats: bool = False,
    sample_key_col: str = DEFAULT_SAMPLE_KEY_COL,
    feature_key_col: str = DEFAULT_FEATURE_KEY_COL,
):
    """Convert parquet array shards into dense-id binary shards.

    The conversion may repartition features across shards. It preserves the logical
    dataset contents but does not guarantee shard-for-shard identity with the
    source parquet layout.
    """
    parquet_manifest = load_array_shard_manifest(array_manifest_path)
    sample_meta_df = _load_dense_meta(parquet_manifest.sample_meta_path, "sample_id", "sample", sample_key_col)
    feature_meta_df = _load_dense_meta(parquet_manifest.feature_meta_path, "feature_id", "feature", feature_key_col)
    n_features = int(feature_meta_df.height)
    if int(sample_meta_df.height) != int(parquet_manifest.n_samples):
        raise ValueError("parquet manifest n_samples does not match sample metadata row count")
    shard_paths = list_array_shard_paths(parquet_manifest)
    point_schema = _legacy_point_schema()
    feature_ids = np.arange(n_features, dtype=np.int32)
    blocks_per_feature = _blocks_per_feature(parquet_manifest.n_samples, parquet_manifest.samples_per_block)
    estimated_feature_bytes = _dense_feature_estimates_from_parquet_shards(
        shard_paths,
        n_features=n_features,
        n_samples=parquet_manifest.n_samples,
        samples_per_block=parquet_manifest.samples_per_block,
    )
    target_shard_bytes = int(max(np.sum(estimated_feature_bytes), 1) // max(int(parquet_manifest.n_shards), 1))
    config = ArrayShardConfig(
        samples_per_block=int(parquet_manifest.samples_per_block),
        target_shard_bytes=max(int(target_shard_bytes), 1),
        n_shards=int(parquet_manifest.n_shards),
        row_group_size=0,
        use_tmp_spill=False,
        spill_bucket_target_bytes=8 * 1024 * 1024,
    )
    shard_partitions = _build_array_shard_partitions(feature_ids, estimated_feature_bytes, config)
    artifact_sample_meta_path, artifact_feature_meta_path = _materialize_binary_metadata(
        out_dir,
        parquet_manifest.sample_meta_path,
        parquet_manifest.feature_meta_path,
    )

    shard_path = os.path.join(out_dir, "array_binary_feature_shards")
    os.makedirs(shard_path, exist_ok=True)
    manifest_path = os.path.join(out_dir, "array_binary_shard_manifest.json")
    shard_infos = []
    total_data_bytes = 0
    total_index_bytes = 0
    total_block_count = 0
    total_feature_count = 0

    for shard_id, shard_feature_ids in enumerate(shard_partitions):
        df = _collect_sorted_rows_from_parquet_shards(shard_paths, shard_feature_ids)
        rows = []
        if df.height > 0:
            rows = [
                {
                    "feature_id": int(row["feature_id"]),
                    "block_id": int(row["block_id"]),
                    "sample_row_start": int(row["sample_row_start"]),
                    "sample_count": int(row["sample_count"]),
                    "point_count": int(row["point_count"]),
                    "sample_flags_blob": row["sample_flags_blob"],
                    "sample_offsets_blob": row["sample_offsets_blob"],
                    "column_blobs": {
                        "time": row["time_blob"],
                        "value": row["value_blob"],
                    },
                }
                for row in df.iter_rows(named=True)
            ]
        shard_write = _write_dense_binary_shard(
            shard_path,
            shard_id,
            shard_feature_ids,
            rows,
            n_samples=parquet_manifest.n_samples,
            samples_per_block=parquet_manifest.samples_per_block,
            point_schema=point_schema,
            codec=codec,
            zstd_level=zstd_level,
        )
        shard_infos.append(shard_write["shard_info"])
        total_data_bytes += int(shard_write["data_bytes"])
        total_index_bytes += int(shard_write["index_bytes"])
        total_block_count += int(shard_write["shard_info"].block_count)
        total_feature_count += int(shard_write["shard_info"].feature_count)

    _write_binary_manifest(
        sample_meta_path=artifact_sample_meta_path,
        feature_meta_path=artifact_feature_meta_path,
        n_samples=parquet_manifest.n_samples,
        n_features=n_features,
        shard_path=shard_path,
        manifest_path=manifest_path,
        samples_per_block=parquet_manifest.samples_per_block,
        blocks_per_feature=blocks_per_feature,
        shard_infos=shard_infos,
        default_codec=_codec_name(_normalize_codec(codec)),
        point_schema=point_schema,
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
    )
    stats = {
        "n_shards": int(len(shard_infos)),
        "n_buckets": 0,
        "temp_files_created": 0,
        "peak_live_temp_files": 0,
        "peak_live_temp_bytes": 0,
        "feature_count": int(total_feature_count),
        "block_count": int(total_block_count),
        "data_bytes": int(total_data_bytes),
        "index_bytes": int(total_index_bytes),
        "total_bytes": int(total_data_bytes + total_index_bytes + os.path.getsize(manifest_path)),
    }
    if return_stats:
        return manifest_path, stats
    return manifest_path


def _resolve_point_schema_paths(manifest_path: str, point_schema):
    """Resolve dictionary paths inside one point schema against the manifest location."""
    out = []
    for spec in _normalize_binary_point_schema(point_schema):
        dictionary_path = str(spec.dictionary_path or "")
        if dictionary_path:
            dictionary_path = _resolve_manifest_path(manifest_path, dictionary_path)
        out.append(
            PointColumnSpec(
                name=spec.name,
                storage_type=spec.storage_type,
                logical_type=spec.logical_type,
                dictionary_path=dictionary_path,
            )
        )
    return out


def load_array_binary_shard_manifest(manifest_path: str) -> ArrayBinaryShardManifest:
    """Load a dense-id binary shard manifest JSON file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "array-binary-shard":
        raise ValueError(f"unsupported binary shard manifest format: {manifest_path}")
    version = int(data.get("version", 2))
    if version not in {2, 3}:
        raise ValueError(f"unsupported binary shard manifest version: {data.get('version')}")
    shards = [
        ArrayBinaryShardInfo(
            shard_id=int(shard["shard_id"]),
            feature_id_start=int(shard["feature_id_start"]),
            feature_id_end=int(shard["feature_id_end"]),
            feature_count=int(shard["feature_count"]),
            block_count=int(shard["block_count"]),
            blocks_index_name=str(shard["blocks_index_name"]),
            blocks_data_name=str(shard["blocks_data_name"]),
        )
        for shard in data.get("shards", [])
    ]
    point_schema = _resolve_point_schema_paths(manifest_path, _build_point_schema_from_manifest_json(data))
    return ArrayBinaryShardManifest(
        sample_meta_path=_resolve_manifest_path(manifest_path, str(data["sample_meta_path"])),
        feature_meta_path=_resolve_manifest_path(manifest_path, str(data["feature_meta_path"])),
        n_samples=int(data["n_samples"]),
        n_features=int(data["n_features"]),
        shard_path=_resolve_manifest_path(manifest_path, str(data["shard_path"])),
        n_shards=int(data["n_shards"]),
        samples_per_block=int(data["samples_per_block"]),
        blocks_per_feature=int(data["blocks_per_feature"]),
        feature_id_dtype=str(data["feature_id_dtype"]),
        flags_dtype=str(data["flags_dtype"]),
        offset_dtype=str(data["offset_dtype"]),
        time_dtype=str(data.get("time_dtype", "")),
        value_dtype=str(data.get("value_dtype", "")),
        default_codec=str(data.get("default_codec", DEFAULT_CODEC_NAME)),
        endianness=str(data.get("endianness", FILE_ENDIANNESS)),
        id_scheme=str(data.get("id_scheme", "dense_row_ids")),
        sample_key_col=str(data.get("sample_key_col", DEFAULT_SAMPLE_KEY_COL)),
        feature_key_col=str(data.get("feature_key_col", DEFAULT_FEATURE_KEY_COL)),
        shards=shards,
        point_schema=point_schema,
        version=version,
    )


def get_array_binary_point_schema(manifest):
    """Return the normalized point schema for a binary shard manifest."""
    manifest_obj = load_array_binary_shard_manifest(manifest) if isinstance(manifest, str) else manifest
    return list(_normalize_binary_point_schema(getattr(manifest_obj, "point_schema", None)))


def load_array_binary_categorical_dictionaries(manifest):
    """Load categorical dictionaries declared by a binary shard manifest."""
    manifest_obj = load_array_binary_shard_manifest(manifest) if isinstance(manifest, str) else manifest
    out = {}
    for spec in get_array_binary_point_schema(manifest_obj):
        if spec.logical_type != LogicalType.CATEGORICAL:
            continue
        dictionary_path = str(spec.dictionary_path or "")
        if not dictionary_path:
            continue
        df = pl.read_parquet(dictionary_path)
        if "code" not in df.columns or "label" not in df.columns:
            raise ValueError(f"categorical dictionary must contain code/label columns: {dictionary_path}")
        out[spec.name] = {
            int(code): None if label is None else str(label)
            for code, label in zip(df["code"].to_list(), df["label"].to_list())
        }
    return out


def _binary_shard_info(manifest: ArrayBinaryShardManifest, shard_id: int) -> ArrayBinaryShardInfo:
    """Return the manifest shard entry for one shard id."""
    return manifest.shards[int(shard_id)]


def _binary_blocks_index_file(manifest: ArrayBinaryShardManifest, shard_id: int) -> str:
    """Return the `blocks.idx` path for one binary shard manifest entry."""
    shard = _binary_shard_info(manifest, shard_id)
    return os.path.join(manifest.shard_path, shard.blocks_index_name)


def _binary_blocks_data_file(manifest: ArrayBinaryShardManifest, shard_id: int) -> str:
    """Return the `blocks.bin` path for one binary shard manifest entry."""
    shard = _binary_shard_info(manifest, shard_id)
    return os.path.join(manifest.shard_path, shard.blocks_data_name)


def _cached_binary_block_records(path: str):
    """Load and cache the fixed-size block record array for one shard."""
    path = os.path.abspath(path)
    global _BINARY_BLOCK_RECORDS_CACHE_BYTES
    with _BINARY_CACHE_LOCK:
        cached = _BINARY_BLOCK_RECORDS_CACHE.get(path)
        if cached is not None:
            _BINARY_BLOCK_RECORDS_CACHE.move_to_end(path)
            return cached
        header = _read_file_header(path, _BLOCKS_INDEX_MAGIC)
        if header["record_bytes"] != _BLOCK_RECORD_DTYPE.itemsize:
            raise ValueError(f"unexpected block record size for {path}: {header['record_bytes']}")
        count = int(header["entry_count"])
        with open(path, "rb") as f:
            f.seek(_FILE_HEADER_STRUCT.size)
            raw = f.read(count * _BLOCK_RECORD_DTYPE.itemsize)
        records = np.frombuffer(raw, dtype=_BLOCK_RECORD_DTYPE, count=count).copy()
        _BINARY_BLOCK_RECORDS_CACHE[path] = records
        _BINARY_BLOCK_RECORDS_CACHE.move_to_end(path)
        _BINARY_BLOCK_RECORDS_CACHE_BYTES += int(records.nbytes)
        while (
            len(_BINARY_BLOCK_RECORDS_CACHE) > int(_BINARY_BLOCK_RECORDS_CACHE_MAX_ENTRIES)
            or _BINARY_BLOCK_RECORDS_CACHE_BYTES > int(_BINARY_BLOCK_RECORDS_CACHE_MAX_BYTES)
        ):
            old_path, old_records = _BINARY_BLOCK_RECORDS_CACHE.popitem(last=False)
            _BINARY_BLOCK_RECORDS_CACHE_BYTES -= int(old_records.nbytes)
        return records


def _cached_binary_data_mmap(path: str):
    """Open and cache the memory-mapped data file for one shard."""
    path = os.path.abspath(path)
    with _BINARY_CACHE_LOCK:
        cached = _BINARY_DATA_MMAP_CACHE.get(path)
        if cached is not None:
            _BINARY_DATA_MMAP_CACHE.move_to_end(path)
            return cached
        file_obj = open(path, "rb")
        data = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        _BINARY_DATA_MMAP_CACHE[path] = (file_obj, data)
        _BINARY_DATA_MMAP_CACHE.move_to_end(path)
        while len(_BINARY_DATA_MMAP_CACHE) > int(_BINARY_DATA_MMAP_CACHE_MAX_OPEN):
            old_path, (old_file_obj, old_data) = _BINARY_DATA_MMAP_CACHE.popitem(last=False)
            try:
                old_data.close()
            finally:
                old_file_obj.close()
        return _BINARY_DATA_MMAP_CACHE[path]


def _close_memmap(cache, path: str):
    """Close one cached mmap entry if present."""
    path = os.path.abspath(path)
    with _BINARY_CACHE_LOCK:
        cached = cache.pop(path, None)
        if cached is None:
            return
        file_obj, data = cached
        try:
            data.close()
        finally:
            file_obj.close()


def _binary_manifest_paths(manifest: ArrayBinaryShardManifest):
    """Return all binary shard file paths referenced by a manifest."""
    paths = []
    for shard in manifest.shards:
        paths.append(os.path.join(manifest.shard_path, shard.blocks_index_name))
        paths.append(os.path.join(manifest.shard_path, shard.blocks_data_name))
    return paths


def close_array_binary_resources(manifest: ArrayBinaryShardManifest = None):
    """Close cached mmaps for one manifest or for all cached binary shards."""
    global _BINARY_BLOCK_RECORDS_CACHE_BYTES
    if manifest is None:
        with _BINARY_CACHE_LOCK:
            for _path, (file_obj, data) in list(_BINARY_DATA_MMAP_CACHE.items()):
                try:
                    data.close()
                finally:
                    file_obj.close()
            _BINARY_DATA_MMAP_CACHE.clear()
            _BINARY_BLOCK_RECORDS_CACHE.clear()
            _BINARY_BLOCK_RECORDS_CACHE_BYTES = 0
        return
    manifest_obj = load_array_binary_shard_manifest(manifest) if isinstance(manifest, str) else manifest
    with _BINARY_CACHE_LOCK:
        for path in _binary_manifest_paths(manifest_obj):
            cached = _BINARY_DATA_MMAP_CACHE.pop(os.path.abspath(path), None)
            if cached is not None:
                file_obj, data = cached
                try:
                    data.close()
                finally:
                    file_obj.close()
            old_records = _BINARY_BLOCK_RECORDS_CACHE.pop(os.path.abspath(path), None)
            if old_records is not None:
                _BINARY_BLOCK_RECORDS_CACHE_BYTES -= int(old_records.nbytes)


def get_array_binary_cache_stats():
    """Return current in-process cache statistics for binary array reads.

    Returns:
        A JSON-serializable dictionary describing cache occupancy and limits for
        binary block-record arrays and open mmaps.
    """
    with _BINARY_CACHE_LOCK:
        return {
            "block_records_entries": int(len(_BINARY_BLOCK_RECORDS_CACHE)),
            "block_records_bytes": int(_BINARY_BLOCK_RECORDS_CACHE_BYTES),
            "block_records_max_entries": int(_BINARY_BLOCK_RECORDS_CACHE_MAX_ENTRIES),
            "block_records_max_bytes": int(_BINARY_BLOCK_RECORDS_CACHE_MAX_BYTES),
            "open_mmaps": int(len(_BINARY_DATA_MMAP_CACHE)),
            "mmap_max_open": int(_BINARY_DATA_MMAP_CACHE_MAX_OPEN),
        }


def list_array_binary_feature_ids(manifest: ArrayBinaryShardManifest):
    """Return all dense feature ids present in the binary manifest."""
    manifest_obj = load_array_binary_shard_manifest(manifest) if isinstance(manifest, str) else manifest
    return np.arange(int(manifest_obj.n_features), dtype=np.int32)


def _empty_columns(point_schema):
    """Create empty NumPy arrays for every column declared by a point schema."""
    return {
        spec.name: np.empty(0, dtype=_point_dtype(spec))
        for spec in _normalize_binary_point_schema(point_schema)
    }


def _empty_trace(sample_row: int, point_schema=None):
    """Create an empty trace placeholder for a missing sample/feature pair."""
    return ArrayTrace(
        sample_row=int(sample_row),
        flags=0,
        columns=_empty_columns(point_schema or _legacy_point_schema()),
    )


def _empty_block(feature_id: int, block_id: int, sample_row_start: int, sample_count: int, point_schema):
    """Create an empty decoded block for an unmaterialized dense block slot."""
    return ArrayFeatureBlock(
        feature_id=int(feature_id),
        block_id=int(block_id),
        sample_row_start=int(sample_row_start),
        sample_count=int(sample_count),
        point_count=0,
        sample_flags=np.zeros(int(sample_count), dtype=np.uint8),
        sample_offsets=np.zeros(int(sample_count) + 1, dtype=np.int64),
        columns=_empty_columns(point_schema),
    )


def _decode_block_record(
    manifest: ArrayBinaryShardManifest,
    shard: ArrayBinaryShardInfo,
    feature_id: int,
    block_id: int,
    record,
):
    """Decode one block payload from a dense block record."""
    sample_row_start = int(block_id) * int(manifest.samples_per_block)
    sample_count = _sample_count_for_block(manifest.n_samples, manifest.samples_per_block, block_id)
    point_schema = get_array_binary_point_schema(manifest)
    data_length = int(record["data_length"])
    if data_length == 0:
        return _empty_block(feature_id, block_id, sample_row_start, sample_count, point_schema)

    path = _binary_blocks_data_file(manifest, shard.shard_id)
    _file_obj, mm = _cached_binary_data_mmap(path)
    data_offset = int(record["data_offset"])
    payload = mm[data_offset : data_offset + data_length]
    if len(payload) != data_length:
        raise ValueError(f"binary payload truncated: shard={shard.shard_id} offset={data_offset}")

    manifest_version = int(getattr(manifest, "version", LEGACY_FILE_VERSION))
    (
        header_feature_id,
        header_block_id,
        header_sample_row_start,
        header_sample_count,
        header_codec,
        _header_flags,
        header_schema_column_count,
        point_count,
        flags_bytes,
        offsets_bytes,
        encoded_columns_or_time_bytes,
        value_bytes_or_reserved,
    ) = _BLOCK_PAYLOAD_HEADER_STRUCT.unpack_from(payload, 0)

    if int(header_feature_id) != int(feature_id):
        raise ValueError(f"binary payload feature mismatch: expected={feature_id} got={header_feature_id}")
    if int(header_block_id) != int(block_id):
        raise ValueError(f"binary payload block mismatch: expected={block_id} got={header_block_id}")
    if int(header_sample_row_start) != int(sample_row_start):
        raise ValueError(
            f"binary payload sample_row_start mismatch: expected={sample_row_start} got={header_sample_row_start}"
        )
    if int(header_sample_count) != int(sample_count):
        raise ValueError(f"binary payload sample_count mismatch: expected={sample_count} got={header_sample_count}")

    cursor = _BLOCK_PAYLOAD_HEADER_STRUCT.size
    sample_flags_blob = payload[cursor : cursor + int(flags_bytes)]
    cursor += int(flags_bytes)
    sample_offsets_blob = payload[cursor : cursor + int(offsets_bytes)]
    cursor += int(offsets_bytes)

    sample_flags = np.frombuffer(sample_flags_blob, dtype=np.uint8, count=sample_count).copy()
    sample_offsets = np.frombuffer(sample_offsets_blob, dtype="<i8", count=sample_count + 1).copy()
    columns = {}
    if manifest_version == LEGACY_FILE_VERSION:
        encoded_time_blob = payload[cursor : cursor + int(encoded_columns_or_time_bytes)]
        cursor += int(encoded_columns_or_time_bytes)
        encoded_value_blob = payload[cursor : cursor + int(value_bytes_or_reserved)]
        cursor += int(value_bytes_or_reserved)
        if cursor != len(payload):
            raise ValueError(
                f"binary payload length mismatch for shard={shard.shard_id} feature={feature_id} block={block_id}"
            )
        expected_value_bytes = int(point_count) * 8
        time_blob = _maybe_decompress_payload(int(header_codec), encoded_time_blob, None, expected_value_bytes)
        decompressor = _require_zstd().ZstdDecompressor() if int(header_codec) == CODEC_ZSTD else None
        value_blob = _maybe_decompress_payload(int(header_codec), encoded_value_blob, decompressor, expected_value_bytes)
        columns = {
            "time": np.frombuffer(time_blob, dtype="<f8", count=int(point_count)).astype(np.float64, copy=True),
            "value": np.frombuffer(value_blob, dtype="<f8", count=int(point_count)).astype(np.float64, copy=True),
        }
    else:
        if int(header_schema_column_count) != len(point_schema):
            raise ValueError(
                f"binary payload schema column count mismatch: expected={len(point_schema)} got={header_schema_column_count}"
            )
        encoded_columns_payload = payload[cursor : cursor + int(encoded_columns_or_time_bytes)]
        cursor += int(encoded_columns_or_time_bytes)
        if cursor != len(payload):
            raise ValueError(
                f"binary payload length mismatch for shard={shard.shard_id} feature={feature_id} block={block_id}"
            )
        expected_column_bytes = _point_total_bytes(point_schema, point_count)
        decoded_columns_payload = _maybe_decompress_payload(
            int(header_codec),
            encoded_columns_payload,
            _require_zstd().ZstdDecompressor() if int(header_codec) == CODEC_ZSTD else None,
            expected_column_bytes,
        )
        column_cursor = 0
        for spec in point_schema:
            dtype = _point_dtype(spec)
            byte_count = int(point_count) * int(dtype.itemsize)
            blob = decoded_columns_payload[column_cursor : column_cursor + byte_count]
            if len(blob) != byte_count:
                raise ValueError(
                    f"decoded payload length mismatch for column={spec.name}: expected={byte_count} got={len(blob)}"
                )
            column_cursor += byte_count
            columns[spec.name] = np.frombuffer(blob, dtype=dtype, count=int(point_count)).copy()
        if column_cursor != len(decoded_columns_payload):
            raise ValueError(
                f"decoded payload trailing bytes mismatch: expected={column_cursor} got={len(decoded_columns_payload)}"
            )
    return ArrayFeatureBlock(
        feature_id=int(feature_id),
        block_id=int(block_id),
        sample_row_start=int(sample_row_start),
        sample_count=int(sample_count),
        point_count=int(point_count),
        sample_flags=sample_flags,
        sample_offsets=sample_offsets,
        columns=columns,
    )


def load_array_binary_feature_block(manifest: ArrayBinaryShardManifest, shard_id: int, row_in_shard: int):
    """Decode one block given a shard-local dense record index.

    Args:
        manifest: Loaded binary shard manifest.
        shard_id: Zero-based shard identifier.
        row_in_shard: Zero-based dense block record index inside the shard.

    Returns:
        A decoded `ArrayFeatureBlock`.
    """
    manifest_obj = load_array_binary_shard_manifest(manifest) if isinstance(manifest, str) else manifest
    shard = _binary_shard_info(manifest_obj, shard_id)
    row_in_shard = int(row_in_shard)
    if row_in_shard < 0 or row_in_shard >= int(shard.block_count):
        raise ValueError(f"row_in_shard out of range: shard={shard_id} row={row_in_shard}")
    local_feature = row_in_shard // int(manifest_obj.blocks_per_feature)
    block_id = row_in_shard % int(manifest_obj.blocks_per_feature)
    if local_feature >= int(shard.feature_count):
        raise ValueError(f"invalid block record index for shard {shard_id}: {row_in_shard}")
    feature_id = int(shard.feature_id_start) + int(local_feature)
    records = _cached_binary_block_records(_binary_blocks_index_file(manifest_obj, shard_id))
    return _decode_block_record(manifest_obj, shard, feature_id, block_id, records[row_in_shard])


class ArrayBinaryShardReader:
    """Reader for dense-id binary array shards."""

    def __init__(self, manifest: ArrayBinaryShardManifest):
        """Create a reader for binary array shards.

        Args:
            manifest: Loaded binary shard manifest.
        """
        self.manifest = manifest
        self._feature_to_shard = np.full(int(self.manifest.n_features), -1, dtype=np.int32)
        for shard in self.manifest.shards:
            if int(shard.feature_count) <= 0:
                continue
            self._feature_to_shard[int(shard.feature_id_start) : int(shard.feature_id_end) + 1] = int(shard.shard_id)
        self._point_schema = get_array_binary_point_schema(self.manifest)

    def close(self):
        """Release cached mmap-backed resources for this manifest."""
        close_array_binary_resources(self.manifest)

    def has_feature(self, feature_id: int) -> bool:
        """Return whether a dense feature id exists in this dataset."""
        feature_id = int(feature_id)
        return 0 <= feature_id < int(self.manifest.n_features)

    def feature_ids(self):
        """Return all dense feature ids present in the dataset."""
        return list_array_binary_feature_ids(self.manifest)

    def point_schema(self):
        """Return the normalized point schema for this dataset."""
        return list(self._point_schema)

    def load_block(self, shard_id: int, row_in_shard: int):
        """Load one decoded block by dense shard record index."""
        return load_array_binary_feature_block(self.manifest, shard_id, row_in_shard)

    def _shard_for_feature(self, feature_id: int) -> Optional[int]:
        """Return the owning shard id for a feature, or `None` if out of range."""
        feature_id = int(feature_id)
        if feature_id < 0 or feature_id >= int(self.manifest.n_features):
            return None
        shard_id = int(self._feature_to_shard[feature_id])
        if shard_id < 0:
            return None
        return shard_id

    def _record_index(self, shard: ArrayBinaryShardInfo, feature_id: int, block_id: int) -> int:
        """Compute the dense block-record index for one `(feature_id, block_id)` pair."""
        local_feature = int(feature_id) - int(shard.feature_id_start)
        return int(local_feature) * int(self.manifest.blocks_per_feature) + int(block_id)

    def load_feature_samples(self, feature_id: int, sample_ids):
        """Load traces for one dense feature id at requested dense sample ids.

        Args:
            feature_id: Dense feature identifier.
            sample_ids: Iterable of dense sample ids. The binary format uses the same
                dense ids internally and externally.

        Returns:
            A dictionary keyed by requested sample id with `ArrayTrace` values. Missing
            or out-of-range sample ids map to empty traces with `flags=0`.
        """
        requested_ids = [int(sample_id) for sample_id in sample_ids]
        out = {sample_id: _empty_trace(sample_id, self._point_schema) for sample_id in requested_ids}
        shard_id = self._shard_for_feature(feature_id)
        if shard_id is None:
            return out

        valid_sample_ids = [
            sample_id for sample_id in requested_ids if 0 <= int(sample_id) < int(self.manifest.n_samples)
        ]
        if not valid_sample_ids:
            return out

        shard = _binary_shard_info(self.manifest, shard_id)
        rows_by_block = {}
        for sample_id in valid_sample_ids:
            block_id = int(sample_id) // int(self.manifest.samples_per_block)
            rows_by_block.setdefault(block_id, []).append(int(sample_id))

        block_records = _cached_binary_block_records(_binary_blocks_index_file(self.manifest, shard_id))
        for block_id, block_sample_ids in rows_by_block.items():
            record_index = self._record_index(shard, feature_id, block_id)
            block = _decode_block_record(self.manifest, shard, int(feature_id), int(block_id), block_records[record_index])
            for sample_id in block_sample_ids:
                trace = block.trace_for_sample_row(sample_id)
                if trace is not None:
                    out[int(sample_id)] = trace
        return out

    def load_feature_samples_by_sample_ids(
        self,
        feature_id: int,
        sample_ids,
        locator_index=None,
        sample_id_index=None,
        sample_meta_path: str = None,
    ):
        """Load traces for one feature using dense sample ids.

        The dense-id v2 format defines `sample_id == sample_row`. The legacy
        `sample_id_index` and `sample_meta_path` arguments are accepted only for
        API compatibility and are ignored.
        """
        _ = locator_index
        _ = sample_id_index
        _ = sample_meta_path
        sample_id_list = [int(sample_id) for sample_id in sample_ids]
        traces_by_row = self.load_feature_samples(feature_id, sample_id_list)
        out = {}
        for sample_id in sample_id_list:
            if 0 <= int(sample_id) < int(self.manifest.n_samples):
                out[sample_id] = traces_by_row[int(sample_id)]
            else:
                out[sample_id] = _empty_trace(-1, self._point_schema)
        return out


def load_array_binary_feature_samples_by_sample_ids(
    manifest,
    feature_id: int,
    sample_ids,
    locator_index=None,
    sample_id_index=None,
    sample_meta_path: str = None,
):
    """Convenience wrapper to load binary traces from a manifest path or object."""
    if isinstance(manifest, str):
        manifest = load_array_binary_shard_manifest(manifest)
    reader = ArrayBinaryShardReader(manifest)
    return reader.load_feature_samples_by_sample_ids(
        feature_id=feature_id,
        sample_ids=sample_ids,
        locator_index=locator_index,
        sample_id_index=sample_id_index,
        sample_meta_path=sample_meta_path,
    )
