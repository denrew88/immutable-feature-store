import json
import os

import numpy as np
import polars as pl

from ..config import ArrayBundleConfig, ArrayShardConfig
from ..types import LogicalType, PointColumnSpec, normalize_logical_type, point_storage_dtype

FLAG_PRESENT = 0x01
FLAG_EMPTY = 0x02


def _normalize_point_schema(point_schema):
    if point_schema is None:
        raise ValueError("point_schema must be provided explicitly")
    specs = []
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
    logical_type = normalize_logical_type(spec.logical_type)
    storage_dtype = point_storage_dtype(spec.storage_type)
    if logical_type == LogicalType.TIMESTAMP_NS:
        return np.asarray(values, dtype="datetime64[ns]").reshape(-1).astype(storage_dtype, copy=False)
    if logical_type == LogicalType.TIMEDELTA_NS:
        return np.asarray(values, dtype="timedelta64[ns]").reshape(-1).astype(storage_dtype, copy=False)
    return np.asarray(values, dtype=storage_dtype).reshape(-1)


def bundle_file_path(bundle_path: str, bundle_id: int) -> str:
    return os.path.join(bundle_path, f"bundle_{bundle_id:06d}.parquet")


def list_bundle_paths(manifest) -> list[str]:
    return [bundle_file_path(manifest.bundle_path, bundle_id) for bundle_id in range(int(manifest.n_bundles))]


class ArraySampleBundleWriter:
    def __init__(
        self,
        out_dir: str,
        sample_meta_path: str,
        n_samples: int,
        *,
        feature_meta_path: str = "",
        config: ArrayBundleConfig = None,
        point_schema,
    ):
        self.config = config or ArrayBundleConfig()
        self.sample_meta_path = str(sample_meta_path or "")
        self.feature_meta_path = str(feature_meta_path or "")
        self.n_samples = int(n_samples)
        self.point_schema = _normalize_point_schema(point_schema)
        self.out_dir = out_dir
        self.bundle_path = os.path.join(out_dir, "array_sample_bundles")
        self.manifest_path = os.path.join(out_dir, "array_bundle_manifest.json")
        os.makedirs(self.bundle_path, exist_ok=True)
        self.n_bundles = 0
        self._finished = False
        self._reset_buffer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()
        return False

    def append_trace(self, sample_id: int, feature_id: int, *, columns):
        if self._finished:
            raise RuntimeError("writer already finished")
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= self.n_samples:
            raise ValueError(f"sample_id out of range: {sample_id}")

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
        flags = FLAG_PRESENT | (FLAG_EMPTY if trace_len == 0 else 0)

        self._sample_ids.append(sample_id)
        self._feature_ids.append(int(feature_id))
        self._flags.append(int(flags))
        self._trace_lens.append(trace_len)
        for spec in self.point_schema:
            self._column_blobs[spec.name].append(encoded_columns[spec.name])
        self._current_rows += 1
        self._current_bytes += 8 + 4 + 1 + 4 + sum(len(encoded_columns[spec.name]) for spec in self.point_schema)

        if self._current_rows >= self.config.max_bundle_rows or self._current_bytes >= self.config.max_bundle_bytes:
            self.flush_bundle()

    def flush_bundle(self):
        if self._current_rows == 0:
            return
        order = np.lexsort((np.asarray(self._feature_ids), np.asarray(self._sample_ids)))
        df = pl.DataFrame(
            {
                "sample_id": pl.Series("sample_id", [self._sample_ids[i] for i in order], dtype=pl.Int64),
                "feature_id": pl.Series("feature_id", [self._feature_ids[i] for i in order], dtype=pl.Int32),
                "flags": pl.Series("flags", [self._flags[i] for i in order], dtype=pl.UInt8),
                "trace_len": pl.Series("trace_len", [self._trace_lens[i] for i in order], dtype=pl.Int32),
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
        if self._finished:
            return self.manifest_path
        self.flush_bundle()
        manifest = {
            "sample_meta_path": self.sample_meta_path,
            "feature_meta_path": self.feature_meta_path,
            "n_samples": self.n_samples,
            "bundle_path": self.bundle_path,
            "n_bundles": self.n_bundles,
            "feature_id_dtype": "INT32",
            "flags_dtype": "UINT8",
            "point_schema": [spec.to_json() for spec in self.point_schema],
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self._finished = True
        return self.manifest_path

    def _reset_buffer(self):
        self._sample_ids = []
        self._feature_ids = []
        self._flags = []
        self._trace_lens = []
        self._column_blobs = {spec.name: [] for spec in self.point_schema}
        self._current_rows = 0
        self._current_bytes = 0


def load_array_bundle_manifest(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from ..types import ArrayBundleManifest

    return ArrayBundleManifest(
        sample_meta_path=data["sample_meta_path"],
        feature_meta_path=data.get("feature_meta_path", ""),
        n_samples=int(data["n_samples"]),
        bundle_path=data["bundle_path"],
        n_bundles=int(data["n_bundles"]),
        feature_id_dtype=data["feature_id_dtype"],
        flags_dtype=data["flags_dtype"],
        point_schema=_normalize_point_schema(data.get("point_schema")),
    )


def _estimate_block_overhead_bytes(samples_per_block: int) -> int:
    return int(64 + 9 * int(samples_per_block))


def _partition_feature_ids_by_count(feature_ids, n_shards: int):
    if int(n_shards) <= 0:
        raise ValueError("n_shards must be > 0")
    if len(feature_ids) == 0:
        return [np.empty(0, dtype=np.int32) for _ in range(int(n_shards))]
    shard_size = max(1, (len(feature_ids) + int(n_shards) - 1) // int(n_shards))
    partitions = []
    for shard_id in range(int(n_shards)):
        start = shard_id * shard_size
        end = min((shard_id + 1) * shard_size, len(feature_ids))
        partitions.append(np.asarray(feature_ids[start:end], dtype=np.int32))
    return partitions


def _partition_feature_ids_by_target_bytes(feature_ids, estimated_feature_bytes, target_shard_bytes: int):
    if int(target_shard_bytes) <= 0:
        raise ValueError("target_shard_bytes must be > 0")
    if len(feature_ids) == 0:
        return [np.empty(0, dtype=np.int32)]
    partitions = []
    current_feature_ids = []
    current_bytes = 0
    for feature_id, est_bytes in zip(feature_ids, estimated_feature_bytes):
        est_bytes = max(int(est_bytes), 1)
        if current_feature_ids and current_bytes + est_bytes > int(target_shard_bytes):
            partitions.append(np.asarray(current_feature_ids, dtype=np.int32))
            current_feature_ids = []
            current_bytes = 0
        current_feature_ids.append(int(feature_id))
        current_bytes += est_bytes
    if current_feature_ids:
        partitions.append(np.asarray(current_feature_ids, dtype=np.int32))
    return partitions


def _build_array_shard_partitions(feature_ids, estimated_feature_bytes, config: ArrayShardConfig):
    if int(config.n_shards) > 0:
        return _partition_feature_ids_by_count(feature_ids, int(config.n_shards))
    return _partition_feature_ids_by_target_bytes(feature_ids, estimated_feature_bytes, int(config.target_shard_bytes))


def _build_bucket_partitions(shard_partitions, feature_ids, estimated_feature_bytes, spill_bucket_target_bytes: int):
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
            int(spill_bucket_target_bytes),
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
            "feature_id": pl.Series("feature_id", [row["feature_id"] for row in feature_map_rows], dtype=pl.Int32),
            "shard_id": pl.Series("shard_id", [row["shard_id"] for row in feature_map_rows], dtype=pl.Int32),
            "bucket_id": pl.Series("bucket_id", [row["bucket_id"] for row in feature_map_rows], dtype=pl.Int32),
        }
    )
    return feature_map_df, shard_bucket_partitions


def _bucket_spill_path(tmp_root: str, shard_id: int, bucket_id: int):
    shard_dir = os.path.join(tmp_root, f"shard_{int(shard_id):04d}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, f"bucket_{int(bucket_id):04d}.spill")
