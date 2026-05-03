"""array shard를 여는 core reader facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import polars as pl

from ..types import LogicalType
from .binary_storage import (
    DEFAULT_FEATURE_KEY_COL,
    DEFAULT_SAMPLE_KEY_COL,
    ArrayBinaryShardReader,
    _legacy_point_schema,
    get_array_binary_point_schema,
    load_array_binary_categorical_dictionaries,
    load_array_binary_shard_manifest,
)
from .storage import ArrayShardReader as ArrayParquetShardReader
from .storage import build_array_feature_locator_index, load_array_shard_manifest


@dataclass(frozen=True)
class Trace:
    """public array trace 결과."""

    feature_id: int
    sample_id: int
    sample_row: int
    present: bool
    flags: int
    feature_key: Optional[str]
    sample_key: Optional[str]
    columns: dict = field(default_factory=dict)

    @property
    def time(self):
        """`time` 컬럼이 있으면 반환하고, 없으면 빈 배열을 반환한다."""
        return self.columns.get("time", np.empty(0, dtype=np.float64))

    @property
    def value(self):
        """`value` 컬럼이 있으면 반환하고, 없으면 빈 배열을 반환한다."""
        return self.columns.get("value", np.empty(0, dtype=np.float64))


@dataclass(frozen=True)
class FeatureTraces:
    """feature 하나에 대한 trace 배치 결과."""

    feature_id: int
    sample_ids: Sequence[int]
    traces: Sequence[Trace]
    feature_key: Optional[str] = None
    sample_keys: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class QueryResult:
    """여러 feature와 sample을 함께 조회한 결과."""

    feature_ids: Sequence[int]
    sample_ids: Sequence[int]
    features: Sequence[FeatureTraces]
    feature_keys: Optional[Sequence[str]] = None
    sample_keys: Optional[Sequence[str]] = None


class ArrayShardDataset:
    """array shard를 읽는 core dataset facade.

    binary v2/v3 manifest와 기존 parquet array manifest를 모두 연다.
    """

    def __init__(self, manifest_path):
        self._manifest_path = str(Path(manifest_path).expanduser().resolve())
        self._closed = False
        self._sample_key_to_id = None
        self._sample_keys = None
        self._feature_key_to_id = None
        self._feature_keys = None
        self._feature_ids = None

        binary_error = None
        try:
            self._manifest = load_array_binary_shard_manifest(self._manifest_path)
            self._kind = "binary"
            self._reader = ArrayBinaryShardReader(self._manifest)
            self._locator_index = None
            self._point_schema = tuple(get_array_binary_point_schema(self._manifest))
            self._categorical_dictionaries = None
            self._sample_key_col = str(self._manifest.sample_key_col)
            self._feature_key_col = str(self._manifest.feature_key_col)
            self._feature_ids = tuple(range(int(self._manifest.n_features)))
        except Exception as exc:
            binary_error = exc
            try:
                self._manifest = load_array_shard_manifest(self._manifest_path)
                self._kind = "parquet"
                self._reader = ArrayParquetShardReader(self._manifest)
                self._locator_index = build_array_feature_locator_index(self._manifest.locator_path)
                self._point_schema = tuple(_legacy_point_schema())
                self._categorical_dictionaries = {}
                self._sample_key_col = DEFAULT_SAMPLE_KEY_COL
                self._feature_key_col = DEFAULT_FEATURE_KEY_COL
            except Exception as parquet_exc:
                raise ValueError(
                    f"array shard manifest를 읽지 못했다: {self._manifest_path}\n"
                    f"binary parser: {binary_error}\n"
                    f"parquet parser: {parquet_exc}"
                ) from parquet_exc

        self._sample_ids = tuple(range(int(self._manifest.n_samples)))

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def manifest_path(self) -> str:
        """열린 manifest 경로를 반환한다."""
        return self._manifest_path

    @property
    def n_samples(self) -> int:
        """dense sample 개수를 반환한다."""
        return int(self._manifest.n_samples)

    @property
    def n_shards(self) -> int:
        """shard 개수를 반환한다."""
        return int(self._manifest.n_shards)

    @property
    def feature_count(self) -> int:
        """dense feature 개수를 반환한다."""
        if self._feature_ids is None:
            self._load_feature_ids()
        return int(len(self._feature_ids))

    @property
    def point_schema(self):
        """point schema를 반환한다."""
        return self._point_schema

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("array shard dataset is closed")

    def close(self):
        """dataset이 잡고 있는 reader 상태를 정리한다."""
        if self._closed:
            return
        close_fn = getattr(self._reader, "close", None)
        if callable(close_fn):
            close_fn()
        self._closed = True

    def schema(self):
        """point schema를 tuple로 반환한다."""
        self._ensure_open()
        return self._point_schema

    def categorical_dictionaries(self):
        """categorical dictionary를 반환한다."""
        self._ensure_open()
        if self._kind != "binary":
            return {}
        if self._categorical_dictionaries is None:
            self._categorical_dictionaries = load_array_binary_categorical_dictionaries(self._manifest)
        return self._categorical_dictionaries

    def _load_feature_ids(self):
        if self._feature_ids is not None:
            return
        df = pl.read_parquet(self._manifest.feature_meta_path, columns=["feature_id"])
        self._feature_ids = tuple(int(value) for value in df["feature_id"].to_list())

    def _load_sample_key_index(self):
        if self._sample_key_to_id is not None:
            return
        key_col = str(self._sample_key_col)
        df = pl.read_parquet(self._manifest.sample_meta_path, columns=[key_col])
        if key_col not in df.columns:
            raise LookupError(f"sample metadata에 key 컬럼이 없다: {key_col}")
        keys = df[key_col].to_list()
        self._sample_keys = tuple(None if key is None else str(key) for key in keys)
        self._sample_key_to_id = {str(key): idx for idx, key in enumerate(keys) if key is not None}

    def _load_feature_key_index(self):
        if self._feature_key_to_id is not None:
            return
        key_col = str(self._feature_key_col)
        df = pl.read_parquet(self._manifest.feature_meta_path, columns=[key_col])
        if key_col not in df.columns:
            raise LookupError(f"feature metadata에 key 컬럼이 없다: {key_col}")
        keys = df[key_col].to_list()
        self._feature_keys = tuple(None if key is None else str(key) for key in keys)
        self._feature_key_to_id = {str(key): idx for idx, key in enumerate(keys) if key is not None}
        if self._feature_ids is None:
            self._feature_ids = tuple(range(len(keys)))

    def has_feature(self, feature_id: int) -> bool:
        """해당 dense feature id가 존재하는지 반환한다."""
        self._ensure_open()
        feature_id = int(feature_id)
        if self._kind == "binary":
            return bool(self._reader.has_feature(feature_id))
        return bool(self._locator_index.get(feature_id))

    def has_sample(self, sample_id: int) -> bool:
        """해당 dense sample id가 존재하는지 반환한다."""
        self._ensure_open()
        return 0 <= int(sample_id) < int(self._manifest.n_samples)

    def feature_ids(self):
        """모든 dense feature id를 반환한다."""
        self._ensure_open()
        if self._feature_ids is None:
            self._load_feature_ids()
        return self._feature_ids

    def sample_ids(self):
        """모든 dense sample id를 반환한다."""
        self._ensure_open()
        return self._sample_ids

    def feature_keys(self):
        """모든 feature key를 dense id 순서대로 반환한다."""
        self._ensure_open()
        self._load_feature_key_index()
        return self._feature_keys

    def sample_keys(self):
        """모든 sample key를 dense id 순서대로 반환한다."""
        self._ensure_open()
        self._load_sample_key_index()
        return self._sample_keys

    def resolve_feature_key(self, feature_key: str) -> int:
        """feature key를 dense feature id로 변환한다."""
        self._ensure_open()
        self._load_feature_key_index()
        feature_id = self._feature_key_to_id.get(str(feature_key))
        if feature_id is None:
            raise LookupError(f"feature key not found: {feature_key}")
        return int(feature_id)

    def resolve_sample_key(self, sample_key: str) -> int:
        """sample key를 dense sample id로 변환한다."""
        self._ensure_open()
        self._load_sample_key_index()
        sample_id = self._sample_key_to_id.get(str(sample_key))
        if sample_id is None:
            raise LookupError(f"sample key not found: {sample_key}")
        return int(sample_id)

    def _decode_trace_columns(self, trace, decode_categorical: bool):
        dictionaries = self.categorical_dictionaries()
        schema_by_name = {spec.name: spec for spec in self._point_schema}
        out = {}
        for name, values in trace.columns.items():
            spec = schema_by_name.get(name)
            logical_type = None if spec is None else spec.logical_type
            if logical_type == LogicalType.CATEGORICAL:
                if not decode_categorical:
                    out[name] = values.copy()
                    continue
                mapping = dictionaries.get(name, {})
                out[name] = tuple(None if int(value) == 0 else mapping.get(int(value)) for value in values.tolist())
                continue
            if logical_type == LogicalType.TIMESTAMP_NS:
                out[name] = values.astype("datetime64[ns]", copy=True)
                continue
            if logical_type == LogicalType.TIMEDELTA_NS:
                out[name] = values.astype("timedelta64[ns]", copy=True)
                continue
            out[name] = values.copy()
        return out

    def _to_public_trace(
        self,
        feature_id: int,
        sample_id: int,
        trace,
        *,
        feature_key=None,
        sample_key=None,
        decode_categorical: bool = False,
    ):
        return Trace(
            feature_id=int(feature_id),
            sample_id=int(sample_id),
            sample_row=int(trace.sample_row),
            present=bool(int(trace.flags) & 0x01),
            flags=int(trace.flags),
            feature_key=None if feature_key is None else str(feature_key),
            sample_key=None if sample_key is None else str(sample_key),
            columns=self._decode_trace_columns(trace, bool(decode_categorical)),
        )

    def _validate_requests(self, feature_id: int, sample_ids, strict: bool):
        if strict and not self.has_feature(feature_id):
            raise LookupError(f"feature id not found: {feature_id}")
        if strict:
            missing = [int(sample_id) for sample_id in sample_ids if not self.has_sample(int(sample_id))]
            if missing:
                raise LookupError(f"sample ids not found: {missing}")

    def get_trace(self, feature_id: int, sample_id: int, strict: bool = False, decode_categorical: bool = False) -> Trace:
        """feature 하나와 sample 하나의 trace를 읽는다."""
        batch = self.get_traces(
            feature_id=feature_id,
            sample_ids=[sample_id],
            strict=strict,
            decode_categorical=decode_categorical,
        )
        return batch.traces[0]

    def get_trace_by_key(self, feature_key: str, sample_key: str, strict: bool = True, decode_categorical: bool = False) -> Trace:
        """feature key와 sample key로 trace 하나를 읽는다."""
        batch = self.get_traces_by_key(
            feature_key=feature_key,
            sample_keys=[sample_key],
            strict=strict,
            decode_categorical=decode_categorical,
        )
        return batch.traces[0]

    def get_traces(self, feature_id: int, sample_ids, strict: bool = False, decode_categorical: bool = False) -> FeatureTraces:
        """feature 하나에 대해 여러 sample의 trace를 읽는다."""
        self._ensure_open()
        feature_id = int(feature_id)
        sample_id_list = [int(sample_id) for sample_id in sample_ids]
        self._validate_requests(feature_id, sample_id_list, bool(strict))
        if self._kind == "binary":
            traces = self._reader.load_feature_samples_by_sample_ids(feature_id=feature_id, sample_ids=sample_id_list)
        else:
            traces = self._reader.load_feature_samples_by_sample_ids(
                feature_id=feature_id,
                sample_ids=sample_id_list,
                locator_index=self._locator_index,
            )
        public_traces = [
            self._to_public_trace(feature_id, sample_id, traces[int(sample_id)], decode_categorical=decode_categorical)
            for sample_id in sample_id_list
        ]
        return FeatureTraces(
            feature_id=feature_id,
            sample_ids=tuple(sample_id_list),
            traces=tuple(public_traces),
        )

    def get_traces_by_key(self, feature_key: str, sample_keys, strict: bool = True, decode_categorical: bool = False) -> FeatureTraces:
        """feature key와 여러 sample key로 trace를 읽는다."""
        self._ensure_open()
        feature_id = self.resolve_feature_key(feature_key)
        sample_key_list = [str(sample_key) for sample_key in sample_keys]
        sample_id_list = [self.resolve_sample_key(sample_key) for sample_key in sample_key_list]
        self._validate_requests(feature_id, sample_id_list, bool(strict))
        if self._kind == "binary":
            traces = self._reader.load_feature_samples_by_sample_ids(feature_id=feature_id, sample_ids=sample_id_list)
        else:
            traces = self._reader.load_feature_samples_by_sample_ids(
                feature_id=feature_id,
                sample_ids=sample_id_list,
                locator_index=self._locator_index,
            )
        public_traces = [
            self._to_public_trace(
                feature_id,
                sample_id,
                traces[int(sample_id)],
                feature_key=feature_key,
                sample_key=sample_key,
                decode_categorical=decode_categorical,
            )
            for sample_id, sample_key in zip(sample_id_list, sample_key_list)
        ]
        return FeatureTraces(
            feature_id=feature_id,
            sample_ids=tuple(sample_id_list),
            traces=tuple(public_traces),
            feature_key=str(feature_key),
            sample_keys=tuple(sample_key_list),
        )

    def get_many(self, feature_ids, sample_ids, strict: bool = False, decode_categorical: bool = False) -> QueryResult:
        """여러 feature를 공통 sample 집합에 맞춰 읽는다."""
        self._ensure_open()
        feature_id_list = [int(feature_id) for feature_id in feature_ids]
        sample_id_list = [int(sample_id) for sample_id in sample_ids]
        features = [
            self.get_traces(
                feature_id=feature_id,
                sample_ids=sample_id_list,
                strict=strict,
                decode_categorical=decode_categorical,
            )
            for feature_id in feature_id_list
        ]
        return QueryResult(
            feature_ids=tuple(feature_id_list),
            sample_ids=tuple(sample_id_list),
            features=tuple(features),
        )

    def get_many_by_key(self, feature_keys, sample_keys, strict: bool = True, decode_categorical: bool = False) -> QueryResult:
        """여러 feature key와 sample key를 사용해 trace를 읽는다."""
        self._ensure_open()
        feature_key_list = [str(feature_key) for feature_key in feature_keys]
        sample_key_list = [str(sample_key) for sample_key in sample_keys]
        feature_id_list = [self.resolve_feature_key(feature_key) for feature_key in feature_key_list]
        sample_id_list = [self.resolve_sample_key(sample_key) for sample_key in sample_key_list]
        features = [
            self.get_traces_by_key(
                feature_key=feature_key,
                sample_keys=sample_key_list,
                strict=strict,
                decode_categorical=decode_categorical,
            )
            for feature_key in feature_key_list
        ]
        return QueryResult(
            feature_ids=tuple(feature_id_list),
            sample_ids=tuple(sample_id_list),
            features=tuple(features),
            feature_keys=tuple(feature_key_list),
            sample_keys=tuple(sample_key_list),
        )


def open_shard(manifest_path) -> ArrayShardDataset:
    """array shard manifest를 열어 dataset facade를 반환한다."""
    return ArrayShardDataset(manifest_path)
