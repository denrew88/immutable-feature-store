from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np


class StorageType(str, Enum):
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    STRING = "string"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"


class LogicalType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    TIMESTAMP_NS = "timestamp_ns"
    TIMEDELTA_NS = "timedelta_ns"


POINT_STORAGE_DTYPES = {
    StorageType.FLOAT64: np.dtype("<f8"),
    StorageType.INT32: np.dtype("<i4"),
    StorageType.INT64: np.dtype("<i8"),
    StorageType.UINT8: np.dtype("u1"),
    StorageType.UINT16: np.dtype("<u2"),
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
    LogicalType.CATEGORICAL: {StorageType.STRING, StorageType.UINT8, StorageType.UINT16, StorageType.UINT32},
    LogicalType.TIMESTAMP_NS: {StorageType.INT64},
    LogicalType.TIMEDELTA_NS: {StorageType.INT64},
}


def normalize_storage_type(value: Union[StorageType, str]) -> StorageType:
    if isinstance(value, StorageType):
        return value
    if isinstance(value, Enum):
        value = value.value
    return StorageType(str(value).strip().lower())


def normalize_logical_type(value: Union[LogicalType, str]) -> LogicalType:
    if isinstance(value, LogicalType):
        return value
    if isinstance(value, Enum):
        value = value.value
    return LogicalType(str(value).strip().lower())


def point_storage_dtype(value: Union[StorageType, str]) -> np.dtype:
    return POINT_STORAGE_DTYPES[normalize_storage_type(value)]


def validate_point_type_pair(storage_type: Union[StorageType, str], logical_type: Union[LogicalType, str]):
    storage_enum = normalize_storage_type(storage_type)
    logical_enum = normalize_logical_type(logical_type)
    if storage_enum not in ALLOWED_STORAGE_TYPES_BY_LOGICAL[logical_enum]:
        raise ValueError(f"logical_type={logical_enum.value!r} does not allow storage_type={storage_enum.value!r}")
    return storage_enum, logical_enum


@dataclass
class PointColumnSpec:
    name: str
    storage_type: Union[StorageType, str]
    logical_type: Union[LogicalType, str]

    def __post_init__(self):
        self.name = str(self.name)
        self.storage_type, self.logical_type = validate_point_type_pair(self.storage_type, self.logical_type)

    def to_json(self):
        return {
            "name": self.name,
            "storage_type": self.storage_type.value,
            "logical_type": self.logical_type.value,
        }
