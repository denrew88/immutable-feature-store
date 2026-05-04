from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np


class StorageType(str, Enum):
    """point column 하나의 물리 저장 dtype."""

    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    UINT32 = "uint32"
    UINT64 = "uint64"


class LogicalType(str, Enum):
    """point column 하나의 논리적 의미."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    TIMESTAMP_NS = "timestamp_ns"
    TIMEDELTA_NS = "timedelta_ns"


POINT_STORAGE_DTYPES = {
    StorageType.FLOAT64: np.dtype("<f8"),
    StorageType.INT32: np.dtype("<i4"),
    StorageType.INT64: np.dtype("<i8"),
    StorageType.UINT32: np.dtype("<u4"),
    StorageType.UINT64: np.dtype("<u8"),
}

ALLOWED_STORAGE_TYPES_BY_LOGICAL = {
    LogicalType.CONTINUOUS: {StorageType.FLOAT64},
    LogicalType.INTEGER: {
        StorageType.INT32,
        StorageType.INT64,
        StorageType.UINT32,
        StorageType.UINT64,
    },
    LogicalType.CATEGORICAL: {StorageType.UINT32},
    LogicalType.TIMESTAMP_NS: {StorageType.INT64},
    LogicalType.TIMEDELTA_NS: {StorageType.INT64},
}


def normalize_storage_type(value: Union[StorageType, str]) -> StorageType:
    """사용자가 넘긴 storage type을 표준 enum 값으로 정규화한다."""
    if isinstance(value, StorageType):
        return value
    if isinstance(value, Enum):
        value = value.value
    try:
        return StorageType(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in StorageType)
        raise ValueError(f"unsupported point storage type: {value!r}; allowed: {allowed}") from exc


def normalize_logical_type(value: Union[LogicalType, str]) -> LogicalType:
    """사용자가 넘긴 logical type을 표준 enum 값으로 정규화한다."""
    if isinstance(value, LogicalType):
        return value
    if isinstance(value, Enum):
        value = value.value
    try:
        return LogicalType(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in LogicalType)
        raise ValueError(f"unsupported point logical type: {value!r}; allowed: {allowed}") from exc


def point_storage_dtype(value: Union[StorageType, str]) -> np.dtype:
    """물리 point storage type 하나에 대응하는 표준 NumPy dtype을 반환한다."""
    return POINT_STORAGE_DTYPES[normalize_storage_type(value)]


def validate_point_type_pair(
    storage_type: Union[StorageType, str],
    logical_type: Union[LogicalType, str],
):
    """`(storage_type, logical_type)` 조합 하나를 검증한다.

    Args:
        storage_type: 물리 dtype 선언.
        logical_type: 논리 타입 선언.

    Returns:
        표준 enum 형태의 `(storage_enum, logical_enum)` 튜플.

    Raises:
        ValueError: 포맷 스펙이 허용하지 않는 조합일 때 발생한다.
    """
    storage_enum = normalize_storage_type(storage_type)
    logical_enum = normalize_logical_type(logical_type)
    allowed = ALLOWED_STORAGE_TYPES_BY_LOGICAL[logical_enum]
    if storage_enum not in allowed:
        allowed_values = ", ".join(item.value for item in sorted(allowed, key=lambda item: item.value))
        raise ValueError(
            f"logical_type={logical_enum.value!r} requires storage_type in {{{allowed_values}}}; "
            f"got {storage_enum.value!r}"
        )
    return storage_enum, logical_enum


@dataclass(frozen=True)
class Candidate:
    feature_id: int
    shard_id: int
    offset_in_shard: int
    r2_y: float
    n_valid_y: int


@dataclass
class PointColumnSpec:
    name: str
    storage_type: Union[StorageType, str]
    logical_type: Union[LogicalType, str]
    dictionary_path: str = ""

    def __post_init__(self):
        """point-column specification 하나를 정규화하고 검증한다."""
        self.name = str(self.name)
        storage_enum, logical_enum = validate_point_type_pair(self.storage_type, self.logical_type)
        self.storage_type = storage_enum
        self.logical_type = logical_enum
        self.dictionary_path = str(self.dictionary_path or "")

    def to_json(self):
        data = {
            "name": self.name,
            "storage_type": self.storage_type.value,
            "logical_type": self.logical_type.value,
        }
        if self.dictionary_path:
            data["dictionary_path"] = self.dictionary_path
        return data


@dataclass
class ArrayBundleManifest:
    sample_meta_path: str
    feature_meta_path: str
    n_samples: int
    bundle_path: str
    n_bundles: int
    feature_id_dtype: str
    flags_dtype: str
    point_schema: list = field(default_factory=list)

    def to_json(self):
        """bundle manifest를 JSON-compatible dictionary로 직렬화한다.

        Returns:
            `array_bundle_manifest.json`으로 바로 쓸 수 있는 dictionary.
        """
        return {
            "sample_meta_path": self.sample_meta_path,
            "feature_meta_path": self.feature_meta_path,
            "n_samples": self.n_samples,
            "bundle_path": self.bundle_path,
            "n_bundles": self.n_bundles,
            "feature_id_dtype": self.feature_id_dtype,
            "flags_dtype": self.flags_dtype,
            "point_schema": [
                spec.to_json() if hasattr(spec, "to_json") else spec
                for spec in self.point_schema
            ],
        }


@dataclass
class ArrayBinaryShardInfo:
    shard_id: int
    feature_id_start: int
    feature_id_end: int
    feature_count: int
    block_count: int
    blocks_index_name: str
    blocks_data_name: str

    def to_json(self):
        """top-level manifest의 binary shard 엔트리 하나를 직렬화한다.

        Returns:
            binary shard 하나를 설명하는 JSON-compatible dictionary.
        """
        return {
            "shard_id": self.shard_id,
            "feature_id_start": self.feature_id_start,
            "feature_id_end": self.feature_id_end,
            "feature_count": self.feature_count,
            "block_count": self.block_count,
            "blocks_index_name": self.blocks_index_name,
            "blocks_data_name": self.blocks_data_name,
        }


@dataclass
class ArrayBinaryShardManifest:
    sample_meta_path: str
    feature_meta_path: str
    n_samples: int
    n_features: int
    shard_path: str
    n_shards: int
    samples_per_block: int
    blocks_per_feature: int
    feature_id_dtype: str
    flags_dtype: str
    offset_dtype: str
    default_codec: str
    endianness: str
    id_scheme: str
    sample_key_col: str
    feature_key_col: str
    shards: list
    point_schema: list = field(default_factory=list)
    version: int = 3

    def to_json(self):
        """binary array shard manifest를 JSON 형태로 직렬화한다.

        Returns:
            `array_binary_shard_manifest.json`으로 바로 쓸 수 있는 dictionary.
        """
        return {
            "format": "array-binary-shard",
            "version": int(self.version),
            "endianness": self.endianness,
            "sample_meta_path": self.sample_meta_path,
            "feature_meta_path": self.feature_meta_path,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "shard_path": self.shard_path,
            "n_shards": self.n_shards,
            "samples_per_block": self.samples_per_block,
            "blocks_per_feature": self.blocks_per_feature,
            "feature_id_dtype": self.feature_id_dtype,
            "flags_dtype": self.flags_dtype,
            "offset_dtype": self.offset_dtype,
            "default_codec": self.default_codec,
            "id_scheme": self.id_scheme,
            "sample_key_col": self.sample_key_col,
            "feature_key_col": self.feature_key_col,
            "point_schema": [
                spec.to_json() if hasattr(spec, "to_json") else spec
                for spec in self.point_schema
            ],
            "shards": [
                shard.to_json() if hasattr(shard, "to_json") else shard
                for shard in self.shards
            ],
        }


@dataclass
class ArrayTrace:
    sample_id: int
    flags: int
    columns: dict = field(default_factory=dict)


@dataclass
class ArrayFeatureBlock:
    feature_id: int
    block_id: int
    sample_id_start: int
    sample_count: int
    point_count: int
    sample_flags: np.ndarray
    sample_offsets: np.ndarray
    columns: dict = field(default_factory=dict)

    def trace_for_sample_id(self, sample_id: int):
        """디코딩된 block에서 sample trace 하나를 추출한다.

        Args:
            sample_id: 추출할 전역 dense sample id.

        Returns:
            sample id가 이 block에 속하면 `ArrayTrace`, 아니면 `None`.

        Raises:
            ValueError: 저장된 sample offset이 payload 배열과 일치하지 않을 때 발생한다.
        """
        idx = int(sample_id - self.sample_id_start)
        if idx < 0 or idx >= self.sample_count:
            return None
        start = int(self.sample_offsets[idx])
        end = int(self.sample_offsets[idx + 1])
        if end < start:
            raise ValueError(f"invalid offsets for sample_id={sample_id}")
        for name, values in self.columns.items():
            if end > int(values.shape[0]):
                raise ValueError(f"invalid offsets for sample_id={sample_id} column={name}")
        return ArrayTrace(
            sample_id=int(sample_id),
            flags=int(self.sample_flags[idx]),
            columns={
                name: values[start:end].copy()
                for name, values in self.columns.items()
            },
        )
