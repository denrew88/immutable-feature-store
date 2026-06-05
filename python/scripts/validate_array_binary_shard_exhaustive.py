from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from fs.array.binary_storage import (  # noqa: E402
    ArrayBinaryShardReader,
    _BLOCKS_DATA_MAGIC,
    _BLOCKS_INDEX_MAGIC,
    _BLOCK_PAYLOAD_HEADER_STRUCT,
    _BLOCK_RECORD_DTYPE,
    _FILE_HEADER_STRUCT,
    _binary_blocks_data_file,
    _binary_blocks_index_file,
    _read_file_header,
    load_array_binary_shard_manifest,
)
from fs.array.storage import _point_blob_column_name, list_bundle_paths, load_array_bundle_manifest  # noqa: E402
from fs.types import point_storage_dtype  # noqa: E402


def _fail(label: str, message: str):
    raise AssertionError(f"{label}: {message}")


def _assert_array_equal(label: str, actual: np.ndarray, expected: np.ndarray):
    if actual.dtype.kind == "f" or expected.dtype.kind == "f":
        actual_f = actual.astype(np.float64, copy=False)
        expected_f = expected.astype(np.float64, copy=False)
        equal = (actual_f == expected_f) | (np.isnan(actual_f) & np.isnan(expected_f))
        if bool(np.all(equal)):
            return
        idx = int(np.flatnonzero(~equal)[0])
        _fail(label, f"float mismatch at index={idx}: actual={actual_f[idx]!r} expected={expected_f[idx]!r}")
    if not np.array_equal(actual, expected):
        actual_flat = actual.reshape(-1)
        expected_flat = expected.reshape(-1)
        idx = int(np.flatnonzero(actual_flat != expected_flat)[0])
        _fail(label, f"mismatch at index={idx}: actual={actual_flat[idx]!r} expected={expected_flat[idx]!r}")


def _read_bundle_reference(bundle_manifest_path: str | Path) -> tuple[dict[tuple[int, int], dict[str, Any]], Any]:
    bundle_manifest = load_array_bundle_manifest(str(bundle_manifest_path))
    point_schema = list(bundle_manifest.point_schema)
    reference: dict[tuple[int, int], dict[str, Any]] = {}
    bundle_paths = [path for path in list_bundle_paths(bundle_manifest) if Path(path).exists()]
    if not bundle_paths:
        return reference, bundle_manifest
    frames = [pl.read_parquet(path) for path in bundle_paths]
    bundle_df = pl.concat(frames, how="vertical").sort(["feature_id", "sample_id"])
    for row_idx, row in enumerate(bundle_df.iter_rows(named=True)):
        sample_id = int(row["sample_id"])
        feature_id = int(row["feature_id"])
        key = (feature_id, sample_id)
        if key in reference:
            _fail("bundle-reference", f"duplicate feature/sample row at bundle row {row_idx}: {key}")
        trace_len = int(row["trace_len"])
        columns = {}
        for spec in point_schema:
            blob = row[_point_blob_column_name(spec.name)] or b""
            dtype = point_storage_dtype(spec.storage_type)
            expected_bytes = int(trace_len) * int(dtype.itemsize)
            if len(blob) != expected_bytes:
                _fail(
                    "bundle-reference",
                    f"blob length mismatch row={row_idx} column={spec.name}: got={len(blob)} expected={expected_bytes}",
                )
            columns[spec.name] = np.frombuffer(blob, dtype=dtype, count=trace_len).copy()
        reference[key] = {
            "flags": int(row["flags"]),
            "trace_len": trace_len,
            "columns": columns,
        }
    return reference, bundle_manifest


def _validate_shard_coverage(manifest):
    covered: list[int] = []
    for shard in manifest.shards:
        feature_count = int(shard.feature_count)
        if feature_count == 0:
            if int(shard.feature_id_start) != -1 or int(shard.feature_id_end) != -1:
                _fail("manifest", f"empty shard has non-empty range: shard_id={shard.shard_id}")
            if int(shard.block_count) != 0:
                _fail("manifest", f"empty shard has block_count={shard.block_count}: shard_id={shard.shard_id}")
            continue
        first = int(shard.feature_id_start)
        last = int(shard.feature_id_end)
        if last - first + 1 != feature_count:
            _fail("manifest", f"feature_count/range mismatch shard_id={shard.shard_id}")
        if int(shard.block_count) != feature_count * int(manifest.blocks_per_feature):
            _fail("manifest", f"block_count mismatch shard_id={shard.shard_id}")
        covered.extend(range(first, last + 1))
    if sorted(covered) != list(range(int(manifest.n_features))):
        _fail("manifest", "shard feature coverage has gap or overlap")


def _validate_binary_files(manifest):
    for shard in manifest.shards:
        shard_id = int(shard.shard_id)
        idx_path = _binary_blocks_index_file(manifest, shard_id)
        bin_path = _binary_blocks_data_file(manifest, shard_id)
        idx_header = _read_file_header(idx_path, _BLOCKS_INDEX_MAGIC)
        bin_header = _read_file_header(bin_path, _BLOCKS_DATA_MAGIC)
        if idx_header["record_bytes"] != _BLOCK_RECORD_DTYPE.itemsize:
            _fail("blocks.idx", f"record size mismatch shard_id={shard_id}")
        if idx_header["entry_count"] != int(shard.block_count):
            _fail("blocks.idx", f"entry count mismatch shard_id={shard_id}")
        if idx_header["aux_count"] != int(shard.feature_count):
            _fail("blocks.idx", f"feature aux count mismatch shard_id={shard_id}")
        if idx_header["shard_id"] != shard_id or bin_header["shard_id"] != shard_id:
            _fail("binary-header", f"shard id header mismatch shard_id={shard_id}")
        if bin_header["entry_count"] != int(shard.block_count):
            _fail("blocks.bin", f"entry count mismatch shard_id={shard_id}")
        data_size = Path(bin_path).stat().st_size
        if bin_header["aux_count"] != max(0, data_size - _FILE_HEADER_STRUCT.size):
            _fail("blocks.bin", f"payload byte aux count mismatch shard_id={shard_id}")

        with open(idx_path, "rb") as f:
            f.seek(_FILE_HEADER_STRUCT.size)
            raw = f.read(int(shard.block_count) * _BLOCK_RECORD_DTYPE.itemsize)
        records = np.frombuffer(raw, dtype=_BLOCK_RECORD_DTYPE, count=int(shard.block_count))
        with open(bin_path, "rb") as f:
            for record_idx, record in enumerate(records):
                data_offset = int(record["data_offset"])
                data_length = int(record["data_length"])
                if int(record["reserved0"]) != 0 or int(record["crc32_optional"]) != 0:
                    _fail("blocks.idx", f"reserved/checksum fields must be zero shard_id={shard_id} record={record_idx}")
                if data_length == 0:
                    if data_offset != 0 or int(record["point_count"]) != 0:
                        _fail("blocks.idx", f"empty record metadata mismatch shard_id={shard_id} record={record_idx}")
                    continue
                if data_offset < _FILE_HEADER_STRUCT.size:
                    _fail("blocks.idx", f"payload offset before file header shard_id={shard_id} record={record_idx}")
                if data_offset + data_length > data_size:
                    _fail("blocks.idx", f"payload range outside file shard_id={shard_id} record={record_idx}")
                f.seek(data_offset)
                payload = f.read(data_length)
                if len(payload) != data_length:
                    _fail("blocks.bin", f"payload truncated shard_id={shard_id} record={record_idx}")
                header = _BLOCK_PAYLOAD_HEADER_STRUCT.unpack_from(payload, 0)
                (
                    feature_id,
                    block_id,
                    sample_id_start,
                    sample_count,
                    codec,
                    header_flags,
                    schema_count,
                    point_count,
                    flags_bytes,
                    offsets_bytes,
                    encoded_bytes,
                    reserved,
                ) = header
                expected_block_id = int(record_idx) % int(manifest.blocks_per_feature)
                expected_feature_id = int(shard.feature_id_start) + int(record_idx) // int(manifest.blocks_per_feature)
                expected_sample_start = expected_block_id * int(manifest.samples_per_block)
                expected_sample_count = min(
                    int(manifest.samples_per_block),
                    max(0, int(manifest.n_samples) - expected_sample_start),
                )
                if int(feature_id) != expected_feature_id or int(block_id) != expected_block_id:
                    _fail("blocks.bin", f"payload feature/block mismatch shard_id={shard_id} record={record_idx}")
                if int(sample_id_start) != expected_sample_start or int(sample_count) != expected_sample_count:
                    _fail("blocks.bin", f"payload sample range mismatch shard_id={shard_id} record={record_idx}")
                if int(header_flags) != 0 or int(reserved) != 0:
                    _fail("blocks.bin", f"payload reserved fields must be zero shard_id={shard_id} record={record_idx}")
                if int(schema_count) != len(manifest.point_schema):
                    _fail("blocks.bin", f"schema count mismatch shard_id={shard_id} record={record_idx}")
                if int(point_count) != int(record["point_count"]):
                    _fail("blocks.bin", f"point_count mismatch shard_id={shard_id} record={record_idx}")
                expected_len = _BLOCK_PAYLOAD_HEADER_STRUCT.size + int(flags_bytes) + int(offsets_bytes) + int(encoded_bytes)
                if expected_len != int(data_length):
                    _fail("blocks.bin", f"payload length header mismatch shard_id={shard_id} record={record_idx}")
                if int(flags_bytes) != expected_sample_count:
                    _fail("blocks.bin", f"sample flags byte count mismatch shard_id={shard_id} record={record_idx}")
                if int(offsets_bytes) != (expected_sample_count + 1) * 8:
                    _fail("blocks.bin", f"sample offsets byte count mismatch shard_id={shard_id} record={record_idx}")
                if int(codec) not in {0, 1}:
                    _fail("blocks.bin", f"unknown codec id shard_id={shard_id} record={record_idx}: {codec}")


def _validate_reader_against_bundle(manifest, reference: dict[tuple[int, int], dict[str, Any]]):
    reader = ArrayBinaryShardReader(manifest)
    samples = list(range(int(manifest.n_samples)))
    for feature_id in range(int(manifest.n_features)):
        traces = reader.load_feature_samples_by_sample_ids(feature_id, samples)
        for sample_id in samples:
            trace = traces[int(sample_id)]
            expected = reference.get((int(feature_id), int(sample_id)))
            if expected is None:
                if int(trace.flags) != 0:
                    _fail("reader", f"missing trace flags mismatch feature={feature_id} sample={sample_id}: {trace.flags}")
                for spec in manifest.point_schema:
                    if int(trace.columns[spec.name].size) != 0:
                        _fail("reader", f"missing trace column not empty feature={feature_id} sample={sample_id} column={spec.name}")
                continue
            if int(trace.flags) != int(expected["flags"]):
                _fail(
                    "reader",
                    f"flags mismatch feature={feature_id} sample={sample_id}: actual={trace.flags} expected={expected['flags']}",
                )
            for spec in manifest.point_schema:
                _assert_array_equal(
                    f"reader feature={feature_id} sample={sample_id} column={spec.name}",
                    trace.columns[spec.name],
                    expected["columns"][spec.name],
                )


def validate_manifest(binary_manifest_path: str | Path, bundle_manifest_path: str | Path):
    binary_manifest = load_array_binary_shard_manifest(str(binary_manifest_path))
    reference, bundle_manifest = _read_bundle_reference(bundle_manifest_path)
    if int(binary_manifest.n_samples) != int(bundle_manifest.n_samples):
        _fail("manifest", "n_samples mismatch between binary and bundle manifests")
    binary_schema = [(spec.name, str(spec.storage_type), str(spec.logical_type)) for spec in binary_manifest.point_schema]
    bundle_schema = [(spec.name, str(spec.storage_type), str(spec.logical_type)) for spec in bundle_manifest.point_schema]
    if binary_schema != bundle_schema:
        _fail("manifest", "point_schema mismatch between binary and bundle manifests")
    _validate_shard_coverage(binary_manifest)
    _validate_binary_files(binary_manifest)
    _validate_reader_against_bundle(binary_manifest, reference)
    print(
        "array binary validation passed "
        f"samples={binary_manifest.n_samples} features={binary_manifest.n_features} "
        f"bundle_traces={len(reference)} shards={binary_manifest.n_shards}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("binary_manifest_path")
    parser.add_argument("bundle_manifest_path")
    args = parser.parse_args(argv)
    validate_manifest(args.binary_manifest_path, args.bundle_manifest_path)


if __name__ == "__main__":
    main(sys.argv[1:])
