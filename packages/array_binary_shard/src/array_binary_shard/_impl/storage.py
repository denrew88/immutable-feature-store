import json
import os
import time
import uuid

import numpy as np
import polars as pl

from .config import ArrayBundleConfig, ArrayShardConfig
from .types import LogicalType, PointColumnSpec, normalize_logical_type, point_storage_dtype

FLAG_PRESENT = 0x01
FLAG_EMPTY = 0x02

SAMPLE_ID_BYTES = 8
FEATURE_ID_BYTES = 4
FLAGS_BYTES = 1
TRACE_LEN_BYTES = 4
BUNDLE_TRACE_ROW_FIXED_BYTES = SAMPLE_ID_BYTES + FEATURE_ID_BYTES + FLAGS_BYTES + TRACE_LEN_BYTES
BLOCK_HEADER_ESTIMATE_BYTES = 64
BLOCK_PER_SAMPLE_CONTROL_BYTES = 9
FILE_REPLACE_RETRY_COUNT = 10
FILE_REPLACE_RETRY_BASE_SECONDS = 0.025
JSON_REPLACE_RETRY_COUNT = 8


def _replace_file_with_retry(tmp_path: str, final_path: str):
    last_error = None
    for attempt in range(FILE_REPLACE_RETRY_COUNT):
        try:
            # bundle parquet를 tmp에 완전히 쓴 뒤 final 이름으로 바꿉니다.
            # 실패하면 final bundle이 없으므로 manifest에는 아직 포함되지 않습니다.
            os.replace(tmp_path, final_path)
            return
        except OSError as exc:
            last_error = exc
            if attempt == FILE_REPLACE_RETRY_COUNT - 1:
                break
            time.sleep(FILE_REPLACE_RETRY_BASE_SECONDS * float(attempt + 1))
    raise last_error or OSError(f"failed to replace {final_path!r} with {tmp_path!r}")


def _write_json_atomic(path: str, payload: dict):
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        # manifest JSON은 reader가 바로 여는 파일이므로 partial write를 허용하지 않습니다.
        # UUID tmp에 먼저 쓰고 replace해야 reader가 항상 완성된 JSON만 봅니다.
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        last_error = None
        for attempt in range(JSON_REPLACE_RETRY_COUNT):
            try:
                os.replace(tmp_path, path)
                return
            except OSError as exc:
                last_error = exc
                if attempt == JSON_REPLACE_RETRY_COUNT - 1:
                    break
                time.sleep(0.025 * float(attempt + 1))
        raise last_error or OSError(f"failed to replace {path!r} with {tmp_path!r}")
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


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


def _estimate_bundle_trace_row_bytes(encoded_columns: dict[str, bytes], point_schema: list[PointColumnSpec]) -> int:
    # Bundle flush thresholds work on the serialized row footprint:
    # sample_id + feature_id + flags + trace_len fixed fields,
    # plus the already-encoded point-column blobs stored in parquet.
    return int(BUNDLE_TRACE_ROW_FIXED_BYTES + sum(len(encoded_columns[spec.name]) for spec in point_schema))


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
        start_bundle_id: int = 0,
        auto_flush: bool = True,
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
        self.n_bundles = int(start_bundle_id)
        self.auto_flush = bool(auto_flush)
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
        self._current_bytes += _estimate_bundle_trace_row_bytes(encoded_columns, self.point_schema)

        if self.auto_flush and self.should_flush_bundle():
            self.flush_bundle()

    def should_flush_bundle(self) -> bool:
        return bool(
            self._current_rows >= int(self.config.max_bundle_rows)
            or self._current_bytes >= int(self.config.max_bundle_bytes)
        )

    def flush_bundle(self):
        if self._current_rows == 0:
            return None
        bundle_id = int(self.n_bundles)
        row_count = int(self._current_rows)
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
        final_path = bundle_file_path(self.bundle_path, bundle_id)
        tmp_path = final_path + ".tmp"
        try:
            # parquet write 자체는 tmp 파일에 수행합니다. final path로 이동하기 전까지는
            # manifest의 n_bundles가 증가하지 않으므로, 중간 장애 시 reader가 보지 않습니다.
            df.write_parquet(tmp_path)
            _replace_file_with_retry(tmp_path, final_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
        byte_size = os.path.getsize(final_path)
        self.n_bundles += 1
        self._reset_buffer()
        return {
            "bundle_id": bundle_id,
            "path": final_path,
            "row_count": row_count,
            "byte_size": int(byte_size),
        }

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
        # 모든 pending bundle이 flush된 뒤 최종 manifest를 atomic하게 씁니다.
        # manifest가 보인다는 것은 해당 bundle parquet들이 final 위치에 있다는 의미입니다.
        _write_json_atomic(self.manifest_path, manifest)
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
    from .types import ArrayBundleManifest

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
    # block 하나에는 고정 binary payload header가 있고, sample마다 flags/offset을 담는
    # control section이 붙습니다. 이 추정치는 shard partitioning용이라 실제 직렬화와
    # byte 단위로 완전히 같을 필요는 없지만, 같은 구성요소를 기준으로 잡아야 합니다.
    return int(BLOCK_HEADER_ESTIMATE_BYTES + BLOCK_PER_SAMPLE_CONTROL_BYTES * int(samples_per_block))


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
    """Feature 목록을 shard 단위 partition으로 나눈다.

    이 함수의 역할은 "전체 feature를 몇 개의 shard로 어떤 기준으로 나눌지"
    최종 결정하는 것이다. 실제 분할 로직 자체는 하위 helper에 있고,
    여기서는 `ArrayShardConfig`를 보고 어떤 분할 방식을 사용할지만 고른다.

    분기 기준은 두 가지다.

    - `config.n_shards > 0`
      - shard 개수를 사용자가 직접 지정한 경우다.
      - feature 수를 기준으로 거의 균등하게 잘라서 shard를 만든다.
    - 그 외
      - `config.target_shard_bytes`를 기준으로 shard를 만든다.
      - `estimated_feature_bytes`를 앞에서부터 누적하면서,
        shard 하나의 예상 총량이 목표 바이트를 넘기기 직전에서 끊는다.

    반환값은 shard별 feature id 배열 목록이다.
    이후 단계에서는 이 결과를 바탕으로
    - feature -> shard 매핑을 만들고
    - 각 shard 내부에서 다시 spill bucket partition을 계산한다.
    """
    if int(config.n_shards) > 0:
        return _partition_feature_ids_by_count(feature_ids, int(config.n_shards))
    return _partition_feature_ids_by_target_bytes(feature_ids, estimated_feature_bytes, int(config.target_shard_bytes))


def _build_bucket_partitions(shard_partitions, feature_ids, estimated_feature_bytes, spill_bucket_target_bytes: int):
    """Shard 내부 feature를 spill bucket 단위로 다시 나눈다.

    이 함수의 목적은 한 shard 안에서도 임시 spill 파일이 너무 커지지 않게,
    feature들을 여러 bucket으로 잘라 두는 것이다.

    입력으로는 이미 shard 단위로 배정된 `shard_partitions`와
    feature별 예상 바이트 수 `estimated_feature_bytes`를 받는다.
    각 shard 안에서는 예상 바이트 합이 `spill_bucket_target_bytes`를 넘지 않도록
    feature들을 순서대로 잘라서 bucket partition을 만든다.

    반환값은 두 가지다.

    - `feature_map_df`
      - 각 `feature_id`가 어느 `(shard_id, bucket_id)`에 속하는지 나타내는 매핑 테이블
      - 이후 bundle row를 읽을 때 이 테이블과 join해서 어느 spill 파일로 보낼지 결정한다.
    - `shard_bucket_partitions`
      - shard별 bucket 목록
      - 이후 spill 파일을 다시 읽어 shard를 복원할 때 bucket 순회를 위해 사용한다.
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
