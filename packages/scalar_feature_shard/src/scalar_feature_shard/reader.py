"""High-level reader facade for scalar parquet shards."""

from pathlib import Path

import polars as pl

from ._impl.parquet_storage import (
    ParquetShardReader,
    build_feature_locator_index,
    load_manifest,
)
from .exceptions import FeatureNotFoundError, ManifestFormatError, SampleNotFoundError
from .models import FeatureValues, QueryResult, ScalarValue


class ScalarShardDataset:
    """User-facing dataset object for querying scalar feature shards."""

    def __init__(self, manifest_path):
        """Open a scalar shard dataset from its manifest path."""

        self._manifest_path = str(Path(manifest_path).expanduser().resolve())
        try:
            self._manifest = load_manifest(self._manifest_path)
        except Exception as exc:  # pragma: no cover - parser internals are implementation-defined.
            raise ManifestFormatError(f"failed to load scalar shard manifest: {self._manifest_path}") from exc
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
        """Return the dataset for context-manager usage."""

        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Drop cached scan state when leaving a context manager."""

        self.close()
        return False

    @property
    def manifest_path(self) -> str:
        """Return the absolute manifest path used to open the dataset."""

        return self._manifest_path

    @property
    def n_samples(self) -> int:
        """Return the total number of dense samples in the dataset."""

        return int(self._manifest.n_samples)

    @property
    def n_shards(self) -> int:
        """Return the number of parquet shard files in the dataset."""

        return int(self._manifest.n_shards)

    @property
    def feature_count(self) -> int:
        """Return the number of dense feature ids in the dataset."""

        return int(self._manifest.n_features)

    def _ensure_open(self):
        """Raise if the dataset has already been closed."""

        if self._closed:
            raise RuntimeError("scalar shard dataset is closed")

    def close(self):
        """Drop cached lazy parquet scans and mark the dataset closed."""

        if self._closed:
            return
        self._reader._scans.clear()
        self._closed = True

    def feature_ids(self):
        """Return all dense feature ids in ascending order."""

        self._ensure_open()
        return self._feature_ids

    def sample_ids(self):
        """Return all dense sample ids in ascending order."""

        self._ensure_open()
        return self._sample_ids

    def has_feature(self, feature_id: int) -> bool:
        """Return whether the dataset contains the requested dense feature id."""

        self._ensure_open()
        return 0 <= int(feature_id) < int(self._manifest.n_features)

    def has_sample(self, sample_id: int) -> bool:
        """Return whether the dataset contains the requested dense sample id."""

        self._ensure_open()
        return 0 <= int(sample_id) < int(self._manifest.n_samples)

    def _load_sample_key_index(self):
        """Lazily load sample-key lookup structures from sample metadata."""

        if self._sample_key_to_id is not None:
            return
        key_col = str(self._manifest.sample_key_col)
        if not key_col:
            raise SampleNotFoundError("sample key column is not configured in the manifest")
        df = pl.read_parquet(self._manifest.sample_meta_path, columns=[key_col])
        if key_col not in df.columns:
            raise SampleNotFoundError(f"sample metadata does not contain key column: {key_col}")
        keys = df[key_col].to_list()
        self._sample_keys = tuple(None if key is None else str(key) for key in keys)
        self._sample_key_to_id = {str(key): idx for idx, key in enumerate(keys)}

    def _load_feature_key_index(self):
        """Lazily load feature-key lookup structures from feature metadata."""

        if self._feature_key_to_id is not None:
            return
        key_col = str(self._manifest.feature_key_col)
        if not key_col:
            raise FeatureNotFoundError("feature key column is not configured in the manifest")
        df = pl.read_parquet(self._manifest.feature_meta_path, columns=[key_col])
        if key_col not in df.columns:
            raise FeatureNotFoundError(f"feature metadata does not contain key column: {key_col}")
        keys = df[key_col].to_list()
        self._feature_keys = tuple(None if key is None else str(key) for key in keys)
        self._feature_key_to_id = {str(key): idx for idx, key in enumerate(keys)}

    def _maybe_load_sample_keys(self):
        """Best-effort sample-key load used by id-based query paths."""

        if self._sample_keys is not None:
            return
        try:
            self._load_sample_key_index()
        except SampleNotFoundError:
            pass

    def _maybe_load_feature_keys(self):
        """Best-effort feature-key load used by id-based query paths."""

        if self._feature_keys is not None:
            return
        try:
            self._load_feature_key_index()
        except FeatureNotFoundError:
            pass

    def feature_keys(self):
        """Return all external feature keys in dense id order."""

        self._ensure_open()
        self._load_feature_key_index()
        return self._feature_keys

    def sample_keys(self):
        """Return all external sample keys in dense id order."""

        self._ensure_open()
        self._load_sample_key_index()
        return self._sample_keys

    def resolve_feature_key(self, feature_key: str) -> int:
        """Resolve one external feature key into its dense internal id."""

        self._ensure_open()
        self._load_feature_key_index()
        feature_id = self._feature_key_to_id.get(str(feature_key))
        if feature_id is None:
            raise FeatureNotFoundError(f"feature key not found: {feature_key}")
        return int(feature_id)

    def resolve_sample_key(self, sample_key: str) -> int:
        """Resolve one external sample key into its dense internal id."""

        self._ensure_open()
        self._load_sample_key_index()
        sample_id = self._sample_key_to_id.get(str(sample_key))
        if sample_id is None:
            raise SampleNotFoundError(f"sample key not found: {sample_key}")
        return int(sample_id)

    def _validate_requests(self, feature_id: int, sample_ids, strict: bool):
        """Optionally raise public exceptions for missing feature/sample ids."""

        if strict and not self.has_feature(feature_id):
            raise FeatureNotFoundError(f"feature id not found: {feature_id}")
        if strict:
            missing = [int(sample_id) for sample_id in sample_ids if not self.has_sample(int(sample_id))]
            if missing:
                raise SampleNotFoundError(f"sample ids not found: {missing}")

    def _to_public_value(self, feature_id: int, sample_id: int, values, valid, feature_key=None, sample_key=None):
        """Convert one decoded scalar row position into the public value model."""

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
        """Load one scalar value for one feature and one sample id."""

        batch = self.get_values(feature_id=feature_id, sample_ids=[sample_id], strict=strict)
        return batch.values[0]

    def get_value_by_key(self, feature_key: str, sample_key: str, strict: bool = True) -> ScalarValue:
        """Load one scalar value addressed by external feature/sample keys."""

        batch = self.get_values_by_key(feature_key=feature_key, sample_keys=[sample_key], strict=strict)
        return batch.values[0]

    def get_values(self, feature_id: int, sample_ids, strict: bool = False) -> FeatureValues:
        """Load one feature row aligned to multiple dense sample ids."""

        self._ensure_open()
        sample_ids = [int(sample_id) for sample_id in sample_ids]
        self._validate_requests(int(feature_id), sample_ids, strict=bool(strict))
        self._maybe_load_sample_keys()
        self._maybe_load_feature_keys()
        values, valid = self._reader.load_feature_by_id(int(feature_id), locator_index=self._locator_index)
        sample_key_lookup = None
        if self._sample_keys is not None:
            sample_key_lookup = self._sample_keys
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
        """Load one feature row addressed by external keys."""

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
        """Load multiple feature rows aligned to the same sample ids."""

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
        """Load multiple features addressed by external keys."""

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
    """Open a scalar shard dataset from `shard_manifest.json`."""

    return ScalarShardDataset(manifest_path)
