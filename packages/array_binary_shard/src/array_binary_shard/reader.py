"""High-level reader facade for custom array binary shards."""

from pathlib import Path

import polars as pl

from ._impl.binary_storage import (
    ArrayBinaryShardReader,
    get_array_binary_point_schema,
    load_array_binary_categorical_dictionaries,
    load_array_binary_shard_manifest,
)
from ._impl.types import LogicalType
from .exceptions import FeatureNotFoundError, ManifestFormatError, SampleNotFoundError
from .models import FeatureTraces, QueryResult, Trace


class BinaryShardDataset:
    """User-facing dataset object for querying array binary shards.

    The public API exposes dense internal ids directly:

    - `sample_id == sample_meta` row index
    - `feature_id == feature_meta` row index

    Optional external keys are loaded lazily from metadata only when key-based
    methods are used.
    """

    def __init__(self, manifest_path):
        """Open a binary shard dataset from its manifest path.

        Args:
            manifest_path: Path to `array_binary_shard_manifest.json`.

        Raises:
            ManifestFormatError: If the manifest cannot be parsed as a binary shard manifest.
        """
        self._manifest_path = str(Path(manifest_path).expanduser().resolve())
        try:
            self._manifest = load_array_binary_shard_manifest(self._manifest_path)
        except Exception as exc:  # pragma: no cover - exact parser exception type is implementation-defined.
            raise ManifestFormatError(f"failed to load binary shard manifest: {self._manifest_path}") from exc
        self._reader = ArrayBinaryShardReader(self._manifest)
        self._feature_ids = tuple(range(int(self._manifest.n_features)))
        self._sample_ids = tuple(range(int(self._manifest.n_samples)))
        self._sample_key_to_id = None
        self._sample_keys = None
        self._feature_key_to_id = None
        self._feature_keys = None
        self._point_schema = tuple(get_array_binary_point_schema(self._manifest))
        self._categorical_dictionaries = None
        self._closed = False

    def __enter__(self):
        """Return the dataset for context-manager usage."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close internal resources when leaving a context manager."""
        self.close()
        return False

    @property
    def manifest_path(self) -> str:
        """Return the absolute manifest path used to open the dataset."""
        return self._manifest_path

    @property
    def n_samples(self) -> int:
        """Return the total number of dense sample ids described by the manifest."""
        return int(self._manifest.n_samples)

    @property
    def n_shards(self) -> int:
        """Return the number of binary shards in the dataset."""
        return int(self._manifest.n_shards)

    @property
    def samples_per_block(self) -> int:
        """Return the logical block size used by the dataset."""
        return int(self._manifest.samples_per_block)

    @property
    def default_codec(self) -> str:
        """Return the manifest-level default payload codec name."""
        return str(self._manifest.default_codec)

    @property
    def feature_count(self) -> int:
        """Return the number of logical dense feature ids in the dataset."""
        return int(self._manifest.n_features)

    @property
    def point_schema(self):
        """Return the manifest point-column schema in stored order."""
        return self._point_schema

    def _ensure_open(self):
        """Raise if the dataset has already been closed."""
        if self._closed:
            raise RuntimeError("binary shard dataset is closed")

    def close(self):
        """Close mmap-backed resources owned by the dataset."""
        if self._closed:
            return
        self._reader.close()
        self._closed = True

    def schema(self):
        """Return the manifest point-column schema as a tuple."""
        self._ensure_open()
        return self._point_schema

    def categorical_dictionaries(self):
        """Return categorical dictionary mappings declared by the manifest."""
        self._ensure_open()
        if self._categorical_dictionaries is None:
            self._categorical_dictionaries = load_array_binary_categorical_dictionaries(self._manifest)
        return self._categorical_dictionaries

    def _load_sample_key_index(self):
        """Lazily load the sample-key lookup structures from sample metadata."""
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
        """Lazily load the feature-key lookup structures from feature metadata."""
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

    def has_feature(self, feature_id: int) -> bool:
        """Return whether the dataset contains a dense feature id."""
        self._ensure_open()
        return bool(self._reader.has_feature(int(feature_id)))

    def has_sample(self, sample_id: int) -> bool:
        """Return whether the dataset contains a dense sample id."""
        self._ensure_open()
        return 0 <= int(sample_id) < int(self._manifest.n_samples)

    def feature_ids(self):
        """Return all dense feature ids in ascending order."""
        self._ensure_open()
        return self._feature_ids

    def sample_ids(self):
        """Return all dense sample ids in ascending order."""
        self._ensure_open()
        return self._sample_ids

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
        """Resolve one external feature key into its dense internal feature id."""
        self._ensure_open()
        self._load_feature_key_index()
        feature_id = self._feature_key_to_id.get(str(feature_key))
        if feature_id is None:
            raise FeatureNotFoundError(f"feature key not found: {feature_key}")
        return int(feature_id)

    def resolve_sample_key(self, sample_key: str) -> int:
        """Resolve one external sample key into its dense internal sample id."""
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

    def _decode_trace_columns(self, trace, decode_categorical: bool):
        """Convert internal trace columns into the public representation."""
        dictionaries = self.categorical_dictionaries()
        point_schema_by_name = {spec.name: spec for spec in self._point_schema}
        out = {}
        for name, values in trace.columns.items():
            spec = point_schema_by_name.get(name)
            logical_type = None if spec is None else spec.logical_type
            if logical_type == LogicalType.CATEGORICAL:
                if not decode_categorical:
                    out[name] = values.copy()
                    continue
                mapping = dictionaries.get(name, {})
                decoded = []
                for value in values.tolist():
                    code = int(value)
                    if code == 0:
                        decoded.append(None)
                    else:
                        decoded.append(mapping.get(code))
                out[name] = tuple(decoded)
                continue
            if logical_type == LogicalType.TIMESTAMP_NS:
                out[name] = values.astype("datetime64[ns]", copy=True)
                continue
            if logical_type == LogicalType.TIMEDELTA_NS:
                out[name] = values.astype("timedelta64[ns]", copy=True)
                continue
            out[name] = values.copy()
        return out

    def _to_public_trace(self, feature_id: int, sample_id: int, trace, feature_key=None, sample_key=None, decode_categorical: bool = False):
        """Convert an internal `ArrayTrace` into the public `Trace` model."""
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

    def get_trace(self, feature_id: int, sample_id: int, strict: bool = False, decode_categorical: bool = False) -> Trace:
        """Load one trace for one feature and one dense sample id."""
        batch = self.get_traces(feature_id=feature_id, sample_ids=[sample_id], strict=strict, decode_categorical=decode_categorical)
        return batch.traces[0]

    def get_trace_by_key(self, feature_key: str, sample_key: str, strict: bool = True, decode_categorical: bool = False) -> Trace:
        """Load one trace using external feature and sample keys."""
        result = self.get_traces_by_key(feature_key=feature_key, sample_keys=[sample_key], strict=strict, decode_categorical=decode_categorical)
        return result.traces[0]

    def get_traces(self, feature_id: int, sample_ids, strict: bool = False, decode_categorical: bool = False) -> FeatureTraces:
        """Load traces for one feature across multiple dense sample ids."""
        self._ensure_open()
        feature_id = int(feature_id)
        sample_id_list = [int(sample_id) for sample_id in sample_ids]
        self._validate_requests(feature_id, sample_id_list, bool(strict))
        traces = self._reader.load_feature_samples_by_sample_ids(
            feature_id=feature_id,
            sample_ids=sample_id_list,
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
        """Load traces for one feature across multiple external sample keys."""
        self._ensure_open()
        feature_id = self.resolve_feature_key(feature_key)
        sample_key_list = [str(sample_key) for sample_key in sample_keys]
        sample_id_list = [self.resolve_sample_key(sample_key) for sample_key in sample_key_list]
        self._validate_requests(feature_id, sample_id_list, bool(strict))
        traces = self._reader.load_feature_samples_by_sample_ids(
            feature_id=feature_id,
            sample_ids=sample_id_list,
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
        """Load traces for multiple features over a shared dense sample-id set."""
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
        """Load traces for multiple external feature keys and sample keys."""
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


def open_shard(manifest_path) -> BinaryShardDataset:
    """Open a binary shard dataset from its manifest path."""
    return BinaryShardDataset(manifest_path)
