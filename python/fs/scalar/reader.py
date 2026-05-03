"""scalar shard를 여는 core reader facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from .parquet_storage import ParquetShardReader, build_feature_locator_index, load_manifest


@dataclass(frozen=True)
class ScalarValue:
    """sample 하나에 정렬된 scalar 값 하나."""

    feature_id: int
    sample_id: int
    present: bool
    value: Optional[float]
    feature_key: Optional[str] = None
    sample_key: Optional[str] = None


@dataclass(frozen=True)
class FeatureValues:
    """feature 하나에 대한 scalar 값 배치 결과."""

    feature_id: int
    sample_ids: Sequence[int]
    values: Sequence[ScalarValue]
    feature_key: Optional[str] = None
    sample_keys: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class QueryResult:
    """여러 feature와 여러 sample을 함께 조회한 결과."""

    feature_ids: Sequence[int]
    sample_ids: Sequence[int]
    features: Sequence[FeatureValues]
    feature_keys: Optional[Sequence[str]] = None
    sample_keys: Optional[Sequence[str]] = None


class ScalarShardDataset:
    """scalar parquet shard를 읽는 core dataset facade."""

    def __init__(self, manifest_path):
        self._manifest_path = str(Path(manifest_path).expanduser().resolve())
        try:
            self._manifest = load_manifest(self._manifest_path)
        except Exception as exc:
            raise ValueError(f"scalar shard manifest를 읽지 못했다: {self._manifest_path}") from exc
        self._reader = ParquetShardReader(self._manifest)
        self._locator_index = build_feature_locator_index(self._manifest.feature_locator_path)
        self._feature_ids = tuple(range(int(self._manifest.n_features)))
        self._sample_ids = tuple(range(int(self._manifest.n_samples)))
        self._sample_key_to_id = None
        self._sample_keys = None
        self._feature_key_to_id = None
        self._feature_keys = None
        self._closed = False

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
        return int(self._manifest.n_features)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("scalar shard dataset is closed")

    def close(self):
        """lazy parquet scan cache를 비우고 dataset을 닫는다."""
        if self._closed:
            return
        self._reader._scans.clear()
        self._closed = True

    def feature_ids(self):
        """모든 dense feature id를 반환한다."""
        self._ensure_open()
        return self._feature_ids

    def sample_ids(self):
        """모든 dense sample id를 반환한다."""
        self._ensure_open()
        return self._sample_ids

    def has_feature(self, feature_id: int) -> bool:
        """해당 dense feature id가 존재하는지 반환한다."""
        self._ensure_open()
        return 0 <= int(feature_id) < int(self._manifest.n_features)

    def has_sample(self, sample_id: int) -> bool:
        """해당 dense sample id가 존재하는지 반환한다."""
        self._ensure_open()
        return 0 <= int(sample_id) < int(self._manifest.n_samples)

    def _load_sample_key_index(self):
        if self._sample_key_to_id is not None:
            return
        key_col = str(self._manifest.sample_key_col)
        df = pl.read_parquet(self._manifest.sample_meta_path, columns=[key_col])
        if key_col not in df.columns:
            raise LookupError(f"sample metadata에 key 컬럼이 없다: {key_col}")
        keys = df[key_col].to_list()
        self._sample_keys = tuple(None if key is None else str(key) for key in keys)
        self._sample_key_to_id = {str(key): idx for idx, key in enumerate(keys) if key is not None}

    def _load_feature_key_index(self):
        if self._feature_key_to_id is not None:
            return
        key_col = str(self._manifest.feature_key_col)
        df = pl.read_parquet(self._manifest.feature_meta_path, columns=[key_col])
        if key_col not in df.columns:
            raise LookupError(f"feature metadata에 key 컬럼이 없다: {key_col}")
        keys = df[key_col].to_list()
        self._feature_keys = tuple(None if key is None else str(key) for key in keys)
        self._feature_key_to_id = {str(key): idx for idx, key in enumerate(keys) if key is not None}

    def _maybe_load_sample_keys(self):
        if self._sample_keys is not None:
            return
        try:
            self._load_sample_key_index()
        except LookupError:
            pass

    def _maybe_load_feature_keys(self):
        if self._feature_keys is not None:
            return
        try:
            self._load_feature_key_index()
        except LookupError:
            pass

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

    def _validate_requests(self, feature_id: int, sample_ids, strict: bool):
        if strict and not self.has_feature(feature_id):
            raise LookupError(f"feature id not found: {feature_id}")
        if strict:
            missing = [int(sample_id) for sample_id in sample_ids if not self.has_sample(int(sample_id))]
            if missing:
                raise LookupError(f"sample ids not found: {missing}")

    def _to_public_value(self, feature_id: int, sample_id: int, values, valid, *, feature_key=None, sample_key=None):
        present = False
        value = None
        sample_id = int(sample_id)
        if 0 <= sample_id < int(self._manifest.n_samples):
            present = bool(valid[sample_id])
            value = float(values[sample_id]) if present else None
        return ScalarValue(
            feature_id=int(feature_id),
            sample_id=sample_id,
            present=present,
            value=value,
            feature_key=None if feature_key is None else str(feature_key),
            sample_key=None if sample_key is None else str(sample_key),
        )

    def get_value(self, feature_id: int, sample_id: int, strict: bool = False) -> ScalarValue:
        """feature 하나와 sample 하나의 scalar 값을 읽는다."""
        batch = self.get_values(feature_id=feature_id, sample_ids=[sample_id], strict=strict)
        return batch.values[0]

    def get_value_by_key(self, feature_key: str, sample_key: str, strict: bool = True) -> ScalarValue:
        """feature key와 sample key로 scalar 값 하나를 읽는다."""
        batch = self.get_values_by_key(feature_key=feature_key, sample_keys=[sample_key], strict=strict)
        return batch.values[0]

    def get_values(self, feature_id: int, sample_ids, strict: bool = False) -> FeatureValues:
        """feature 하나에 대해 여러 sample의 scalar 값을 읽는다."""
        self._ensure_open()
        sample_ids = [int(sample_id) for sample_id in sample_ids]
        self._validate_requests(int(feature_id), sample_ids, strict=bool(strict))
        self._maybe_load_sample_keys()
        self._maybe_load_feature_keys()
        values, valid = self._reader.load_feature_by_id(int(feature_id), locator_index=self._locator_index)
        sample_key_lookup = self._sample_keys if self._sample_keys is not None else None
        feature_key = None if self._feature_keys is None or not self.has_feature(int(feature_id)) else self._feature_keys[int(feature_id)]
        out = []
        for sample_id in sample_ids:
            sample_key = None
            if sample_key_lookup is not None and 0 <= sample_id < len(sample_key_lookup):
                sample_key = sample_key_lookup[sample_id]
            out.append(
                self._to_public_value(
                    feature_id=int(feature_id),
                    sample_id=sample_id,
                    values=values,
                    valid=valid,
                    feature_key=feature_key,
                    sample_key=sample_key,
                )
            )
        sample_keys = None if sample_key_lookup is None else tuple(value.sample_key for value in out)
        return FeatureValues(
            feature_id=int(feature_id),
            sample_ids=tuple(sample_ids),
            values=tuple(out),
            feature_key=feature_key,
            sample_keys=sample_keys,
        )

    def get_values_by_key(self, feature_key: str, sample_keys, strict: bool = True) -> FeatureValues:
        """feature key와 여러 sample key로 scalar 값을 읽는다."""
        self._ensure_open()
        feature_id = self.resolve_feature_key(feature_key)
        self._load_sample_key_index()
        sample_ids = [self.resolve_sample_key(sample_key) for sample_key in sample_keys]
        batch = self.get_values(feature_id=feature_id, sample_ids=sample_ids, strict=bool(strict))
        return FeatureValues(
            feature_id=batch.feature_id,
            sample_ids=batch.sample_ids,
            values=tuple(
                ScalarValue(
                    feature_id=value.feature_id,
                    sample_id=value.sample_id,
                    present=value.present,
                    value=value.value,
                    feature_key=str(feature_key),
                    sample_key=str(sample_key),
                )
                for value, sample_key in zip(batch.values, sample_keys)
            ),
            feature_key=str(feature_key),
            sample_keys=tuple(str(sample_key) for sample_key in sample_keys),
        )

    def get_many(self, feature_ids, sample_ids, strict: bool = False) -> QueryResult:
        """여러 feature를 공통 sample 집합에 맞춰 읽는다."""
        self._ensure_open()
        feature_ids = [int(feature_id) for feature_id in feature_ids]
        sample_ids = [int(sample_id) for sample_id in sample_ids]
        self._maybe_load_feature_keys()
        self._maybe_load_sample_keys()
        features = [self.get_values(feature_id=feature_id, sample_ids=sample_ids, strict=bool(strict)) for feature_id in feature_ids]
        feature_keys = None
        if self._feature_keys is not None:
            feature_keys = tuple(None if not self.has_feature(feature_id) else self._feature_keys[feature_id] for feature_id in feature_ids)
        sample_keys = None
        if self._sample_keys is not None:
            sample_keys = tuple(None if not self.has_sample(sample_id) else self._sample_keys[sample_id] for sample_id in sample_ids)
        return QueryResult(
            feature_ids=tuple(feature_ids),
            sample_ids=tuple(sample_ids),
            features=tuple(features),
            feature_keys=feature_keys,
            sample_keys=sample_keys,
        )

    def get_many_by_key(self, feature_keys, sample_keys, strict: bool = True) -> QueryResult:
        """여러 feature key와 sample key를 사용해 scalar 값을 읽는다."""
        self._ensure_open()
        self._load_feature_key_index()
        self._load_sample_key_index()
        feature_ids = [self.resolve_feature_key(feature_key) for feature_key in feature_keys]
        sample_ids = [self.resolve_sample_key(sample_key) for sample_key in sample_keys]
        result = self.get_many(feature_ids=feature_ids, sample_ids=sample_ids, strict=bool(strict))
        return QueryResult(
            feature_ids=result.feature_ids,
            sample_ids=result.sample_ids,
            features=tuple(
                FeatureValues(
                    feature_id=feature_id,
                    sample_ids=result.sample_ids,
                    values=feature_batch.values,
                    feature_key=str(feature_key),
                    sample_keys=tuple(str(sample_key) for sample_key in sample_keys),
                )
                for feature_id, feature_key, feature_batch in zip(result.feature_ids, feature_keys, result.features)
            ),
            feature_keys=tuple(str(feature_key) for feature_key in feature_keys),
            sample_keys=tuple(str(sample_key) for sample_key in sample_keys),
        )


def open_shard(manifest_path) -> ScalarShardDataset:
    """scalar shard manifest를 열어 dataset facade를 반환한다."""
    return ScalarShardDataset(manifest_path)
