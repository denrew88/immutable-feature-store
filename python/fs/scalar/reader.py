"""scalar shard용 core reader facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
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
    """여러 feature와 여러 sample에 대한 조회 결과."""

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
        """현재 manifest 경로를 반환한다."""
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

    def _feature_key_for_id(self, feature_id: int):
        if self._feature_keys is None or not self.has_feature(int(feature_id)):
            return None
        return self._feature_keys[int(feature_id)]

    def _normalize_feature_iteration(self, feature_ids, feature_keys_override=None, maintain_order: bool = True):
        pairs = [
            (
                int(feature_id),
                None if feature_keys_override is None else feature_keys_override[idx],
            )
            for idx, feature_id in enumerate(feature_ids)
        ]
        if bool(maintain_order):
            return pairs

        def sort_key(item):
            feature_id, _feature_key = item
            loc = self._locator_index.get(int(feature_id))
            if loc is None:
                return (1, int(feature_id), 0)
            shard_id, offset = loc
            return (0, int(shard_id), int(offset))

        return sorted(pairs, key=sort_key)

    def _load_feature_arrays_batch(self, feature_ids):
        feature_ids = [int(feature_id) for feature_id in feature_ids]
        cached = {}
        shard_groups = {}
        for feature_id in feature_ids:
            if feature_id in cached:
                continue
            loc = self._locator_index.get(int(feature_id))
            if loc is None:
                cached[feature_id] = (
                    np.zeros(self._manifest.n_samples, dtype=np.float64),
                    np.zeros(self._manifest.n_samples, dtype=np.uint8),
                )
                continue
            shard_id, offset = loc
            shard_groups.setdefault(int(shard_id), []).append((feature_id, int(offset)))

        for shard_id, items in shard_groups.items():
            offsets = [offset for _, offset in items]
            values_batch, valid_batch = self._reader.load_rows(int(shard_id), offsets)
            for row_idx, (feature_id, _) in enumerate(items):
                cached[int(feature_id)] = (values_batch[row_idx], valid_batch[row_idx])
        return cached

    def _build_feature_values(
        self,
        feature_id: int,
        sample_ids,
        values,
        valid,
        *,
        feature_key=None,
        sample_key_lookup=None,
        sample_keys_override=None,
    ):
        out = []
        for idx, sample_id in enumerate(sample_ids):
            sample_key = None
            if sample_keys_override is not None:
                sample_key = sample_keys_override[idx]
            elif sample_key_lookup is not None and 0 <= sample_id < len(sample_key_lookup):
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
        sample_keys = None if all(value.sample_key is None for value in out) else tuple(value.sample_key for value in out)
        return FeatureValues(
            feature_id=int(feature_id),
            sample_ids=tuple(sample_ids),
            values=tuple(out),
            feature_key=feature_key,
            sample_keys=sample_keys,
        )

    def _iter_feature_values(
        self,
        feature_ids,
        sample_ids,
        strict: bool,
        batch_size: int,
        *,
        feature_keys_override=None,
        sample_keys_override=None,
    ):
        feature_ids = [int(feature_id) for feature_id in feature_ids]
        sample_ids = [int(sample_id) for sample_id in sample_ids]
        if bool(strict):
            for feature_id in feature_ids:
                self._validate_requests(feature_id, sample_ids, strict=True)
        self._maybe_load_feature_keys()
        self._maybe_load_sample_keys()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        sample_key_lookup = None if sample_keys_override is not None else self._sample_keys
        for start in range(0, len(feature_ids), batch_size):
            stop = start + batch_size
            chunk_feature_ids = feature_ids[start:stop]
            feature_arrays = self._load_feature_arrays_batch(chunk_feature_ids)
            chunk_feature_keys = None if feature_keys_override is None else feature_keys_override[start:stop]
            for idx, feature_id in enumerate(chunk_feature_ids):
                feature_key = self._feature_key_for_id(feature_id)
                if chunk_feature_keys is not None and chunk_feature_keys[idx] is not None:
                    feature_key = str(chunk_feature_keys[idx])
                yield self._build_feature_values(
                    feature_id=feature_id,
                    sample_ids=sample_ids,
                    values=feature_arrays[feature_id][0],
                    valid=feature_arrays[feature_id][1],
                    feature_key=feature_key,
                    sample_key_lookup=sample_key_lookup,
                    sample_keys_override=sample_keys_override,
                )

    def get_value(self, feature_id: int, sample_id: int, strict: bool = False) -> ScalarValue:
        """feature 하나와 sample 하나에 대한 scalar 값을 읽는다."""
        batch = self.get_values(feature_id=feature_id, sample_ids=[sample_id], strict=strict)
        return batch.values[0]

    def get_value_by_key(self, feature_key: str, sample_key: str, strict: bool = True) -> ScalarValue:
        """feature key와 sample key로 scalar 값 하나를 읽는다."""
        batch = self.get_values_by_key(feature_key=feature_key, sample_keys=[sample_key], strict=strict)
        return batch.values[0]

    def get_values(self, feature_id: int, sample_ids, strict: bool = False) -> FeatureValues:
        """feature 하나에 대한 여러 sample 값을 읽는다."""
        self._ensure_open()
        return next(self._iter_feature_values([feature_id], sample_ids, bool(strict), 1))

    def get_values_by_key(self, feature_key: str, sample_keys, strict: bool = True) -> FeatureValues:
        """feature key와 sample key 목록으로 값을 읽는다."""
        self._ensure_open()
        feature_id = self.resolve_feature_key(feature_key)
        self._load_sample_key_index()
        sample_ids = [self.resolve_sample_key(sample_key) for sample_key in sample_keys]
        sample_key_list = tuple(str(sample_key) for sample_key in sample_keys)
        return next(
            self._iter_feature_values(
                [feature_id],
                sample_ids,
                bool(strict),
                1,
                feature_keys_override=[feature_key],
                sample_keys_override=sample_key_list,
            )
        )

    def iter_many(
        self,
        feature_ids,
        sample_ids,
        strict: bool = False,
        batch_size: int = 128,
        maintain_order: bool = True,
    ):
        """여러 feature를 generator로 반환한다. maintain_order=False면 shard 순서로 재정렬한다."""
        self._ensure_open()
        feature_pairs = self._normalize_feature_iteration(feature_ids, maintain_order=bool(maintain_order))
        ordered_feature_ids = [feature_id for feature_id, _feature_key in feature_pairs]
        return self._iter_feature_values(
            ordered_feature_ids,
            sample_ids,
            bool(strict),
            int(batch_size),
        )

    def get_many(
        self,
        feature_ids,
        sample_ids,
        strict: bool = False,
        batch_size: int = 128,
        stream: bool = False,
        maintain_order: bool = True,
    ) -> QueryResult:
        """여러 feature를 공통 sample 집합에 맞춰 읽는다."""
        self._ensure_open()
        feature_pairs = self._normalize_feature_iteration(feature_ids, maintain_order=bool(maintain_order))
        ordered_feature_ids = [feature_id for feature_id, _feature_key in feature_pairs]
        sample_ids = [int(sample_id) for sample_id in sample_ids]
        self._maybe_load_feature_keys()
        self._maybe_load_sample_keys()
        features_iter = self._iter_feature_values(
            ordered_feature_ids,
            sample_ids,
            bool(strict),
            int(batch_size),
        )
        feature_keys = None
        if self._feature_keys is not None:
            feature_keys = tuple(
                None if not self.has_feature(feature_id) else self._feature_keys[feature_id]
                for feature_id in ordered_feature_ids
            )
        sample_keys = None
        if self._sample_keys is not None:
            sample_keys = tuple(
                None if not self.has_sample(sample_id) else self._sample_keys[sample_id]
                for sample_id in sample_ids
            )
        return QueryResult(
            feature_ids=tuple(ordered_feature_ids),
            sample_ids=tuple(sample_ids),
            features=features_iter if bool(stream) else tuple(features_iter),
            feature_keys=feature_keys,
            sample_keys=sample_keys,
        )

    def iter_many_by_key(
        self,
        feature_keys,
        sample_keys,
        strict: bool = True,
        batch_size: int = 128,
        maintain_order: bool = True,
    ):
        """여러 feature key를 generator로 반환한다. maintain_order=False면 shard 순서로 재정렬한다."""
        self._ensure_open()
        self._load_feature_key_index()
        self._load_sample_key_index()
        feature_key_list = [str(feature_key) for feature_key in feature_keys]
        sample_key_list = tuple(str(sample_key) for sample_key in sample_keys)
        feature_pairs = self._normalize_feature_iteration(
            [self.resolve_feature_key(feature_key) for feature_key in feature_key_list],
            feature_keys_override=feature_key_list,
            maintain_order=bool(maintain_order),
        )
        ordered_feature_ids = [feature_id for feature_id, _feature_key in feature_pairs]
        ordered_feature_keys = [str(feature_key) if feature_key is not None else "" for _feature_id, feature_key in feature_pairs]
        sample_ids = [self.resolve_sample_key(sample_key) for sample_key in sample_key_list]
        return self._iter_feature_values(
            ordered_feature_ids,
            sample_ids,
            bool(strict),
            int(batch_size),
            feature_keys_override=ordered_feature_keys,
            sample_keys_override=sample_key_list,
        )

    def get_many_by_key(
        self,
        feature_keys,
        sample_keys,
        strict: bool = True,
        batch_size: int = 128,
        stream: bool = False,
        maintain_order: bool = True,
    ) -> QueryResult:
        """여러 feature key와 sample key를 사용해 scalar 값을 읽는다."""
        self._ensure_open()
        self._load_feature_key_index()
        self._load_sample_key_index()
        feature_key_list = [str(feature_key) for feature_key in feature_keys]
        sample_key_list = tuple(str(sample_key) for sample_key in sample_keys)
        feature_pairs = self._normalize_feature_iteration(
            [self.resolve_feature_key(feature_key) for feature_key in feature_key_list],
            feature_keys_override=feature_key_list,
            maintain_order=bool(maintain_order),
        )
        ordered_feature_ids = [feature_id for feature_id, _feature_key in feature_pairs]
        ordered_feature_keys = [str(feature_key) if feature_key is not None else "" for _feature_id, feature_key in feature_pairs]
        sample_ids = [self.resolve_sample_key(sample_key) for sample_key in sample_key_list]
        features_iter = self._iter_feature_values(
            ordered_feature_ids,
            sample_ids,
            bool(strict),
            int(batch_size),
            feature_keys_override=ordered_feature_keys,
            sample_keys_override=sample_key_list,
        )
        return QueryResult(
            feature_ids=tuple(ordered_feature_ids),
            sample_ids=tuple(sample_ids),
            features=features_iter if bool(stream) else tuple(features_iter),
            feature_keys=tuple(ordered_feature_keys),
            sample_keys=tuple(sample_key_list),
        )


def open_shard(manifest_path) -> ScalarShardDataset:
    """scalar shard manifest를 열고 dataset facade를 반환한다."""
    return ScalarShardDataset(manifest_path)
