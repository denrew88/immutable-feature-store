"""커스텀 array binary shard를 읽기 위한 고수준 facade."""

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
    """array binary shard를 조회하는 사용자용 dataset 객체.

    public API는 dense 내부 id를 그대로 노출한다.

    - `sample_id == sample_meta`의 row index
    - `feature_id == feature_meta`의 row index

    외부 key는 key 기반 메서드를 실제로 호출할 때만 metadata에서 지연 로드한다.
    """

    def __init__(self, manifest_path):
        """manifest 경로로 binary shard dataset을 연다.

        Args:
            manifest_path: `array_binary_shard_manifest.json` 경로.

        Raises:
            ManifestFormatError: manifest를 binary shard manifest로 해석할 수 없을 때 발생한다.
        """
        self._manifest_path = str(Path(manifest_path).expanduser().resolve())
        try:
            self._manifest = load_array_binary_shard_manifest(self._manifest_path)
        except Exception as exc:  # pragma: no cover - 내부 parser 예외 타입은 구현 세부사항이다.
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
        """`with` 문에서 사용할 수 있도록 자기 자신을 반환한다."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """context manager를 빠져나갈 때 내부 자원을 닫는다."""
        self.close()
        return False

    @property
    def manifest_path(self) -> str:
        """dataset을 열 때 사용한 절대 manifest 경로를 반환한다."""
        return self._manifest_path

    @property
    def n_samples(self) -> int:
        """manifest가 설명하는 dense sample id의 총 개수를 반환한다."""
        return int(self._manifest.n_samples)

    @property
    def n_shards(self) -> int:
        """dataset에 포함된 binary shard 개수를 반환한다."""
        return int(self._manifest.n_shards)

    @property
    def samples_per_block(self) -> int:
        """dataset이 사용하는 논리 block 크기를 반환한다."""
        return int(self._manifest.samples_per_block)

    @property
    def default_codec(self) -> str:
        """manifest 수준의 기본 payload codec 이름을 반환한다."""
        return str(self._manifest.default_codec)

    @property
    def feature_count(self) -> int:
        """논리 dense feature id의 개수를 반환한다."""
        return int(self._manifest.n_features)

    @property
    def point_schema(self):
        """manifest의 point-column schema를 저장된 순서 그대로 반환한다."""
        return self._point_schema

    def _ensure_open(self):
        """dataset이 이미 닫혔으면 예외를 발생시킨다."""
        if self._closed:
            raise RuntimeError("binary shard dataset is closed")

    def close(self):
        """dataset이 보유한 mmap 기반 자원을 닫는다."""
        if self._closed:
            return
        self._reader.close()
        self._closed = True

    def schema(self):
        """manifest의 point-column schema를 tuple로 반환한다."""
        self._ensure_open()
        return self._point_schema

    def categorical_dictionaries(self):
        """manifest가 선언한 categorical dictionary 매핑을 반환한다."""
        self._ensure_open()
        if self._categorical_dictionaries is None:
            self._categorical_dictionaries = load_array_binary_categorical_dictionaries(self._manifest)
        return self._categorical_dictionaries

    def _load_sample_key_index(self):
        """sample metadata에서 sample key 조회 구조를 지연 로드한다."""
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
        """feature metadata에서 feature key 조회 구조를 지연 로드한다."""
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
        """해당 dense feature id가 dataset에 존재하는지 반환한다."""
        self._ensure_open()
        return bool(self._reader.has_feature(int(feature_id)))

    def has_sample(self, sample_id: int) -> bool:
        """해당 dense sample id가 dataset에 존재하는지 반환한다."""
        self._ensure_open()
        return 0 <= int(sample_id) < int(self._manifest.n_samples)

    def feature_ids(self):
        """모든 dense feature id를 오름차순으로 반환한다."""
        self._ensure_open()
        return self._feature_ids

    def sample_ids(self):
        """모든 dense sample id를 오름차순으로 반환한다."""
        self._ensure_open()
        return self._sample_ids

    def feature_keys(self):
        """모든 외부 feature key를 dense id 순서로 반환한다."""
        self._ensure_open()
        self._load_feature_key_index()
        return self._feature_keys

    def sample_keys(self):
        """모든 외부 sample key를 dense id 순서로 반환한다."""
        self._ensure_open()
        self._load_sample_key_index()
        return self._sample_keys

    def resolve_feature_key(self, feature_key: str) -> int:
        """외부 feature key 하나를 dense 내부 feature id로 변환한다."""
        self._ensure_open()
        self._load_feature_key_index()
        feature_id = self._feature_key_to_id.get(str(feature_key))
        if feature_id is None:
            raise FeatureNotFoundError(f"feature key not found: {feature_key}")
        return int(feature_id)

    def resolve_sample_key(self, sample_key: str) -> int:
        """외부 sample key 하나를 dense 내부 sample id로 변환한다."""
        self._ensure_open()
        self._load_sample_key_index()
        sample_id = self._sample_key_to_id.get(str(sample_key))
        if sample_id is None:
            raise SampleNotFoundError(f"sample key not found: {sample_key}")
        return int(sample_id)

    def _validate_requests(self, feature_id: int, sample_ids, strict: bool):
        """strict 모드일 때 누락된 feature/sample id에 대한 public 예외를 발생시킨다."""
        if strict and not self.has_feature(feature_id):
            raise FeatureNotFoundError(f"feature id not found: {feature_id}")
        if strict:
            missing = [int(sample_id) for sample_id in sample_ids if not self.has_sample(int(sample_id))]
            if missing:
                raise SampleNotFoundError(f"sample ids not found: {missing}")

    def _decode_trace_columns(self, trace, decode_categorical: bool):
        """내부 trace column 표현을 public 표현으로 변환한다."""
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

    def _to_public_trace(
        self,
        feature_id: int,
        sample_id: int,
        trace,
        feature_key=None,
        sample_key=None,
        decode_categorical: bool = False,
    ):
        """내부 `ArrayTrace`를 public `Trace` 모델로 변환한다."""
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
        """feature 하나와 dense sample id 하나에 대한 trace를 읽는다."""
        batch = self.get_traces(
            feature_id=feature_id,
            sample_ids=[sample_id],
            strict=strict,
            decode_categorical=decode_categorical,
        )
        return batch.traces[0]

    def get_trace_by_key(self, feature_key: str, sample_key: str, strict: bool = True, decode_categorical: bool = False) -> Trace:
        """외부 feature/sample key를 사용해 trace 하나를 읽는다."""
        result = self.get_traces_by_key(
            feature_key=feature_key,
            sample_keys=[sample_key],
            strict=strict,
            decode_categorical=decode_categorical,
        )
        return result.traces[0]

    def get_traces(self, feature_id: int, sample_ids, strict: bool = False, decode_categorical: bool = False) -> FeatureTraces:
        """feature 하나에 대해 여러 dense sample id의 trace를 읽는다."""
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
        """feature 하나에 대해 여러 외부 sample key의 trace를 읽는다."""
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
        """여러 feature를 공통 dense sample id 집합에 맞춰 읽는다."""
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
        """여러 외부 feature key와 sample key를 사용해 trace를 읽는다."""
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
    """manifest 경로로 binary shard dataset을 연다."""
    return BinaryShardDataset(manifest_path)
