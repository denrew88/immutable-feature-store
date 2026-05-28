import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fs.feature_selection.candidates import build_candidates_from_shards, build_candidates_from_stats
from fs.config import SelectionConfig
from fs.array.binary_storage import (
    ArrayBinaryShardReader,
    get_array_binary_point_schema,
    get_array_binary_cache_stats,
    load_array_binary_categorical_dictionaries,
    load_array_binary_shard_manifest,
)
from fs.array_sample_parquet import (
    ArraySampleParquetReader,
    load_array_sample_parquet_manifest,
)
from fs.scalar.parquet_storage import (
    ParquetShardReader,
    build_feature_locator_index,
    list_shard_paths,
    load_sample_targets,
    load_manifest,
    locator_has_candidate_stats,
    resolve_selection_stats_path,
    validate_dense_sample_ids,
)
from fs.feature_selection.incremental import select_features_incremental
from fs.types import LogicalType, StorageType, normalize_logical_type, normalize_storage_type


class ArrayFeatureRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to array_binary_shard_manifest.json")
    feature_id: Optional[int] = None
    feature_ids: Optional[List[int]] = None
    feature_key: Optional[str] = None
    feature_keys: Optional[List[str]] = None
    sample_ids: Optional[List[int]] = None
    sample_keys: Optional[List[str]] = None
    sanitize_nonfinite: bool = True
    decode_categorical: bool = False
    temporal_format: str = "iso"


class ArrayTraceResponse(BaseModel):
    sample_id: int
    sample_key: Optional[str] = None
    flags: int
    columns: dict


class ArrayFeatureItemResponse(BaseModel):
    feature_id: int
    feature_key: Optional[str] = None
    traces: List[ArrayTraceResponse]


class ArrayFeatureResponse(BaseModel):
    manifest_path: str
    feature_id: Optional[int] = None
    feature_key: Optional[str] = None
    feature_ids: Optional[List[int]] = None
    feature_keys: Optional[List[Optional[str]]] = None
    sample_count: int
    traces: Optional[List[ArrayTraceResponse]] = None
    features: Optional[List[ArrayFeatureItemResponse]] = None


class ArraySchemaRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to array_binary_shard_manifest.json")
    include_dictionaries: bool = False


class ArraySchemaResponse(BaseModel):
    manifest_path: str
    version: int
    n_samples: int
    n_features: Optional[int] = None
    samples_per_block: int
    default_codec: Optional[str] = None
    point_schema: List[dict]
    categorical_dictionaries: Optional[dict] = None


class ArraySampleParquetSchemaRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to array_sample_parquet_manifest.json")
    include_dictionaries: bool = False


class ArraySampleParquetTraceRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to array_sample_parquet_manifest.json")
    sample_ids: Optional[List[int]] = None
    sample_keys: Optional[List[str]] = None
    feature_ids: Optional[List[int]] = None
    feature_keys: Optional[List[str]] = None
    layout: str = "nested"
    decode_categorical: bool = False
    include_missing: bool = False
    max_traces: int = 10000


class ScalarFeatureRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to scalar shard_manifest.json")
    feature_id: Optional[int] = None
    feature_key: Optional[str] = None
    sample_ids: Optional[List[int]] = None
    sample_keys: Optional[List[str]] = None
    sanitize_nonfinite: bool = True


class ScalarValueResponse(BaseModel):
    sample_id: int
    sample_key: Optional[str] = None
    present: bool
    value: Optional[float]


class ScalarFeatureResponse(BaseModel):
    manifest_path: str
    feature_id: int
    feature_key: Optional[str] = None
    sample_count: int
    values: List[ScalarValueResponse]


class SelectionRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to scalar shard_manifest.json")
    y_col: str = "y"
    y_r2: float = 0.01
    min_non_null_y: int = 20
    ff_r2: float = 0.8
    min_non_null_pair: int = 20
    top_m: int = 100
    initial_cap: int = 2048
    max_step: int = 4096
    batch_size: int = 512
    max_gap: int = 64
    max_candidates: int = 0
    mask_fastpath_min_group: int = 64
    mask_fastpath_min_pairs: int = 8192


class SelectionResponse(BaseModel):
    manifest_path: str
    y_col: str
    top_m: int
    candidate_count: int
    selected_count: int
    selected_feature_ids: List[int]
    selected_feature_keys: Optional[List[Optional[str]]] = None
    used_locator_stats: bool
    candidate_build_ms: int
    selection_ms: int
    elapsed_ms: int


class CacheStatsResponse(BaseModel):
    manifest_cache: dict
    array_binary_cache: dict
    array_parquet_cache: dict
    scalar_parquet_cache: dict


@dataclass
class _ManifestCacheEntry:
    manifest_path: str
    manifest: object
    locator_index: dict
    reader: object
    last_access_ts: float = 0.0
    sample_key_col: str = "sample_key"
    feature_key_col: str = "feature_key"
    sample_key_index: Optional[dict] = None
    feature_key_index: Optional[dict] = None
    sample_keys_by_id: Optional[dict] = None
    feature_keys_by_id: Optional[dict] = None
    point_schema: Optional[list] = None
    categorical_dictionaries: Optional[dict] = None


app = FastAPI(title="Feature API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_CACHE_LOCK = threading.Lock()
_ARRAY_CACHE = OrderedDict()
_ARRAY_SAMPLE_PARQUET_CACHE = OrderedDict()
_SCALAR_CACHE = OrderedDict()
_MANIFEST_CACHE_MAX_ENTRIES = 16
_MANIFEST_CACHE_TTL_SECONDS = 30 * 60


def _normalize_manifest_path(manifest_path: str) -> str:
    """캐시 키로 쓰기 좋게 manifest 경로를 정규화한다.

    Args:
        manifest_path: 사용자가 넘긴 manifest 경로.

    Returns:
        정규화된 절대 manifest 경로.
    """
    return os.path.abspath(os.path.expanduser(manifest_path))


def _load_manifest_json(manifest_path: str):
    """manifest JSON 파일을 읽어 dictionary로 반환한다.

    Args:
        manifest_path: manifest JSON 파일 경로.

    Returns:
        파싱된 JSON dictionary.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _close_manifest_entry(entry: _ManifestCacheEntry):
    """캐시된 manifest 엔트리 하나가 잡고 있는 reader 자원을 해제한다."""
    close_fn = getattr(entry.reader, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def _sweep_manifest_cache(cache: OrderedDict, now: float):
    """만료된 manifest 캐시 엔트리를 비우고 최대 개수 제한을 맞춥니다."""
    expired_keys = [
        key
        for key, entry in cache.items()
        if (now - float(entry.last_access_ts)) > float(_MANIFEST_CACHE_TTL_SECONDS)
    ]
    for key in expired_keys:
        entry = cache.pop(key, None)
        if entry is not None:
            _close_manifest_entry(entry)
    while len(cache) > int(_MANIFEST_CACHE_MAX_ENTRIES):
        _key, entry = cache.popitem(last=False)
        _close_manifest_entry(entry)


def _touch_manifest_entry(cache: OrderedDict, key: str, entry: _ManifestCacheEntry, now: float):
    """manifest 캐시 엔트리 하나를 최근 사용 상태로 갱신한다."""
    entry.last_access_ts = float(now)
    cache[key] = entry
    cache.move_to_end(key)


def _manifest_cache_snapshot(cache: OrderedDict, now: float):
    """manifest 캐시 하나에 대한 JSON-safe 요약 정보를 반환한다."""
    entries = []
    for key, entry in cache.items():
        entries.append(
            {
                "manifest_path": key,
                "idle_seconds": max(0.0, float(now - float(entry.last_access_ts))),
                "has_sample_key_index": entry.sample_key_index is not None,
                "has_feature_key_index": entry.feature_key_index is not None,
                "has_sample_keys_by_id": entry.sample_keys_by_id is not None,
                "has_feature_keys_by_id": entry.feature_keys_by_id is not None,
            }
        )
    return entries


def _build_key_to_id_index(meta_path: str, key_col: str, id_col: str):
    """metadata parquet 파일 하나에서 외부 key -> id 인덱스를 생성한다.

    Args:
        meta_path: `sample_meta.parquet` 또는 `feature_meta.parquet` 경로.
        key_col: 외부 key가 들어 있는 컬럼 이름.
        id_col: 우선 사용할 정수 id 컬럼 이름. 없으면 row 순서를 id로 사용한다.

    Returns:
        문자열 key를 정수 id로 매핑하는 dictionary.

    Raises:
        ValueError: key 컬럼이 없거나, null이 있거나, unique하지 않을 때 발생한다.
    """
    df = pl.read_parquet(meta_path)
    if key_col not in df.columns:
        raise ValueError(f"metadata key column not found: {key_col}")
    key_series = df[key_col]
    if key_series.null_count() != 0:
        raise ValueError(f"metadata key column contains nulls: {key_col}")
    if int(key_series.n_unique()) != int(df.height):
        raise ValueError(f"metadata key column must be unique: {key_col}")
    if id_col and id_col in df.columns:
        ids = [int(value) for value in df[id_col].to_list()]
    else:
        ids = list(range(df.height))
    return {str(key): int(value) for key, value in zip(key_series.to_list(), ids)}


def _build_id_to_key_index(meta_path: str, key_col: str, id_col: str):
    """metadata parquet 파일 하나에서 id -> 외부 key 인덱스를 생성한다.

    Args:
        meta_path: `sample_meta.parquet` 또는 `feature_meta.parquet` 경로.
        key_col: 외부 key가 들어 있는 컬럼 이름.
        id_col: 우선 사용할 정수 id 컬럼 이름. 없으면 row 순서를 id로 사용한다.

    Returns:
        정수 id를 문자열 key로 매핑하는 dictionary. key 컬럼이 없으면 `None`.

    Raises:
        ValueError: key 컬럼에 null 또는 중복 값이 있을 때 발생한다.
    """
    df = pl.read_parquet(meta_path)
    if key_col not in df.columns:
        return None
    key_series = df[key_col]
    if key_series.null_count() != 0:
        raise ValueError(f"metadata key column contains nulls: {key_col}")
    if int(key_series.n_unique()) != int(df.height):
        raise ValueError(f"metadata key column must be unique: {key_col}")
    if id_col and id_col in df.columns:
        ids = [int(value) for value in df[id_col].to_list()]
    else:
        ids = list(range(df.height))
    return {int(value): str(key) for key, value in zip(key_series.to_list(), ids)}


def _get_array_cache_entry(manifest_path: str) -> _ManifestCacheEntry:
    """array manifest용 reader 상태를 로드하고 캐시에 저장한다.

    Args:
        manifest_path: binary array manifest 경로.

    Returns:
        array 조회에 필요한 manifest metadata와 reader 객체를 담은 캐시 엔트리.
    """
    normalized = _normalize_manifest_path(manifest_path)
    with _CACHE_LOCK:
        now = time.monotonic()
        _sweep_manifest_cache(_ARRAY_CACHE, now)
        entry = _ARRAY_CACHE.get(normalized)
        if entry is not None:
            _touch_manifest_entry(_ARRAY_CACHE, normalized, entry, now)
            return entry
        manifest_json = _load_manifest_json(normalized)
        if manifest_json.get("format") != "array-binary-shard":
            raise ValueError(f"unsupported array manifest format: {manifest_json.get('format')!r}")
        manifest = load_array_binary_shard_manifest(normalized)
        reader = ArrayBinaryShardReader(manifest)
        locator_index = None
        sample_key_col = str(getattr(manifest, "sample_key_col", "sample_key"))
        feature_key_col = str(getattr(manifest, "feature_key_col", "feature_key"))
        entry = _ManifestCacheEntry(
            manifest_path=normalized,
            manifest=manifest,
            locator_index=locator_index,
            reader=reader,
            last_access_ts=now,
            sample_key_col=sample_key_col,
            feature_key_col=feature_key_col,
        )
        _touch_manifest_entry(_ARRAY_CACHE, normalized, entry, now)
        _sweep_manifest_cache(_ARRAY_CACHE, now)
        return entry


def _get_array_sample_parquet_cache_entry(manifest_path: str) -> _ManifestCacheEntry:
    """sample-major Parquet array manifest와 reader를 로드하고 캐시한다."""
    normalized = _normalize_manifest_path(manifest_path)
    with _CACHE_LOCK:
        now = time.monotonic()
        _sweep_manifest_cache(_ARRAY_SAMPLE_PARQUET_CACHE, now)
        entry = _ARRAY_SAMPLE_PARQUET_CACHE.get(normalized)
        if entry is not None:
            _touch_manifest_entry(_ARRAY_SAMPLE_PARQUET_CACHE, normalized, entry, now)
            return entry
        manifest_json = _load_manifest_json(normalized)
        if manifest_json.get("format") != "array-sample-parquet":
            raise ValueError(f"unsupported array sample parquet manifest format: {manifest_json.get('format')!r}")
        manifest = load_array_sample_parquet_manifest(normalized)
        reader = ArraySampleParquetReader(manifest)
        entry = _ManifestCacheEntry(
            manifest_path=normalized,
            manifest=manifest,
            locator_index=None,
            reader=reader,
            last_access_ts=now,
            sample_key_col=str(getattr(manifest, "sample_key_col", "sample_key")),
            feature_key_col=str(getattr(manifest, "feature_key_col", "feature_key")),
            point_schema=list(manifest.point_schema),
        )
        _touch_manifest_entry(_ARRAY_SAMPLE_PARQUET_CACHE, normalized, entry, now)
        _sweep_manifest_cache(_ARRAY_SAMPLE_PARQUET_CACHE, now)
        return entry


def _get_scalar_cache_entry(manifest_path: str) -> _ManifestCacheEntry:
    """scalar manifest용 reader 상태를 로드하고 캐시에 저장한다.

    Args:
        manifest_path: scalar shard manifest 경로.

    Returns:
        scalar 조회에 필요한 manifest metadata, 인덱스, reader 객체를 담은 캐시 엔트리.
    """
    normalized = _normalize_manifest_path(manifest_path)
    with _CACHE_LOCK:
        now = time.monotonic()
        _sweep_manifest_cache(_SCALAR_CACHE, now)
        entry = _SCALAR_CACHE.get(normalized)
        if entry is not None:
            _touch_manifest_entry(_SCALAR_CACHE, normalized, entry, now)
            return entry
        manifest = load_manifest(normalized)
        validate_dense_sample_ids(manifest.sample_meta_path)
        reader = ParquetShardReader(manifest)
        locator_index = build_feature_locator_index(manifest.feature_locator_path)
        entry = _ManifestCacheEntry(
            manifest_path=normalized,
            manifest=manifest,
            locator_index=locator_index,
            reader=reader,
            last_access_ts=now,
            sample_key_col=str(getattr(manifest, "sample_key_col", "sample_key")),
            feature_key_col=str(getattr(manifest, "feature_key_col", "feature_key")),
        )
        _touch_manifest_entry(_SCALAR_CACHE, normalized, entry, now)
        _sweep_manifest_cache(_SCALAR_CACHE, now)
        return entry


def _json_safe_array(values, sanitize_nonfinite: bool):
    """숫자 배열을 JSON-safe한 Python 값으로 변환한다.

    Args:
        values: 숫자 값 iterable.
        sanitize_nonfinite: NaN과 infinity를 `None`으로 치환할지 여부.

    Returns:
        JSON 응답에 바로 넣을 수 있는 float 또는 `None` 목록.
    """
    out = []
    for value in values:
        scalar = float(value)
        if sanitize_nonfinite and not (scalar == scalar and abs(scalar) != float("inf")):
            out.append(None)
        else:
            out.append(scalar)
    return out


def _timedelta_ns_to_iso(value_ns: int) -> str:
    """부호가 있는 nanosecond delta 하나를 ISO 8601 duration 문자열로 변환한다."""
    total_ns = int(value_ns)
    sign = "-" if total_ns < 0 else ""
    total_ns = abs(total_ns)
    ns_per_day = 24 * 60 * 60 * 1_000_000_000
    ns_per_hour = 60 * 60 * 1_000_000_000
    ns_per_minute = 60 * 1_000_000_000
    ns_per_second = 1_000_000_000
    days, rem = divmod(total_ns, ns_per_day)
    hours, rem = divmod(rem, ns_per_hour)
    minutes, rem = divmod(rem, ns_per_minute)
    seconds, ns = divmod(rem, ns_per_second)
    second_text = f"{seconds}"
    if ns:
        second_text = f"{seconds}.{ns:09d}".rstrip("0")
    time_parts = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if seconds or ns or not (days or hours or minutes):
        time_parts.append(f"{second_text}S")
    date_part = f"{days}D" if days else ""
    time_part = f"T{''.join(time_parts)}" if time_parts else ""
    return f"{sign}P{date_part}{time_part or 'T0S'}"


def _normalize_temporal_format(temporal_format: str) -> str:
    """요청된 temporal JSON 렌더링 모드를 검증한다."""
    normalized = str(temporal_format or "iso").strip().lower()
    if normalized not in {"iso", "raw_ns"}:
        raise HTTPException(status_code=400, detail="temporal_format must be 'iso' or 'raw_ns'")
    return normalized


def _array_point_schema(entry: _ManifestCacheEntry):
    """캐시된 array manifest 하나에 대한 정규화된 point schema를 반환한다."""
    if entry.point_schema is not None:
        return entry.point_schema
    entry.point_schema = get_array_binary_point_schema(entry.manifest)
    return entry.point_schema


def _get_array_categorical_dictionaries(entry: _ManifestCacheEntry):
    """array dataset 하나에 대한 categorical dictionary 매핑을 로드하고 캐시한다."""
    if entry.categorical_dictionaries is not None:
        return entry.categorical_dictionaries
    entry.categorical_dictionaries = load_array_binary_categorical_dictionaries(entry.manifest)
    return entry.categorical_dictionaries


def _json_safe_column(
    values,
    spec,
    sanitize_nonfinite: bool,
    decode_categorical: bool,
    categorical_dictionary,
    temporal_format: str,
):
    """point-column 배열 하나를 JSON-safe scalar 목록으로 변환한다."""
    logical_type = normalize_logical_type(
        spec.logical_type if hasattr(spec, "logical_type") else spec.get("logical_type", LogicalType.CONTINUOUS)
    )
    storage_type = normalize_storage_type(
        spec.storage_type if hasattr(spec, "storage_type") else spec.get("storage_type", StorageType.FLOAT64)
    )
    if logical_type == LogicalType.CATEGORICAL:
        if decode_categorical:
            out = []
            for value in values.tolist():
                code = int(value)
                if code == 0:
                    out.append(None)
                else:
                    label = None if categorical_dictionary is None else categorical_dictionary.get(code)
                    out.append(None if label is None else str(label))
            return out
        return [int(value) for value in values.tolist()]
    if logical_type == LogicalType.TIMESTAMP_NS:
        if temporal_format == "raw_ns":
            return [int(value) for value in values.tolist()]
        arr = values.astype("datetime64[ns]", copy=False)
        return [np.datetime_as_string(value, unit="ns", timezone="naive") for value in arr]
    if logical_type == LogicalType.TIMEDELTA_NS:
        if temporal_format == "raw_ns":
            return [int(value) for value in values.tolist()]
        return [_timedelta_ns_to_iso(int(value)) for value in values.tolist()]
    if storage_type == StorageType.FLOAT64:
        return _json_safe_array(values, sanitize_nonfinite)
    return [int(value) for value in values.tolist()]


def _schema_json(specs):
    """point-schema 객체를 JSON-safe dictionary로 변환한다."""
    out = []
    for spec in specs:
        if hasattr(spec, "to_json"):
            out.append(spec.to_json())
        else:
            out.append(
                {
                    "name": str(spec["name"]),
                    "storage_type": str(spec["storage_type"]),
                    "logical_type": str(spec.get("logical_type", "continuous")),
                    **({"dictionary_path": str(spec["dictionary_path"])} if spec.get("dictionary_path") else {}),
                }
            )
    return out


def _get_array_sample_key_index(entry: _ManifestCacheEntry):
    """array dataset 하나에 대한 sample-key 매핑을 로드하고 캐시한다."""
    if entry.sample_key_index is None:
        entry.sample_key_index = _build_key_to_id_index(
            entry.manifest.sample_meta_path,
            entry.sample_key_col,
            "sample_id",
        )
    return entry.sample_key_index


def _get_array_feature_key_index(entry: _ManifestCacheEntry):
    """array dataset 하나에 대한 feature-key 매핑을 로드하고 캐시한다."""
    feature_meta_path = str(getattr(entry.manifest, "feature_meta_path", "") or "")
    if not feature_meta_path:
        raise ValueError("feature metadata path is required for feature_key requests")
    if entry.feature_key_index is None:
        entry.feature_key_index = _build_key_to_id_index(
            feature_meta_path,
            entry.feature_key_col,
            "feature_id",
        )
    return entry.feature_key_index


def _get_array_sample_keys_by_id(entry: _ManifestCacheEntry):
    """array dataset 하나에 대한 sample-id -> sample-key 매핑을 로드하고 캐시한다."""
    if entry.sample_keys_by_id is None:
        entry.sample_keys_by_id = _build_id_to_key_index(
            entry.manifest.sample_meta_path,
            entry.sample_key_col,
            "sample_id",
        )
    return entry.sample_keys_by_id


def _get_array_feature_keys_by_id(entry: _ManifestCacheEntry):
    """array dataset 하나에 대한 feature-id -> feature-key 매핑을 로드하고 캐시한다."""
    feature_meta_path = str(getattr(entry.manifest, "feature_meta_path", "") or "")
    if not feature_meta_path:
        return None
    if entry.feature_keys_by_id is None:
        entry.feature_keys_by_id = _build_id_to_key_index(
            feature_meta_path,
            entry.feature_key_col,
            "feature_id",
        )
    return entry.feature_keys_by_id


def _get_scalar_sample_key_index(entry: _ManifestCacheEntry):
    """scalar dataset 하나에 대한 sample-key 매핑을 로드하고 캐시한다."""
    if entry.sample_key_index is None:
        entry.sample_key_index = _build_key_to_id_index(
            entry.manifest.sample_meta_path,
            entry.sample_key_col,
            "sample_id",
        )
    return entry.sample_key_index


def _get_scalar_feature_key_index(entry: _ManifestCacheEntry):
    """scalar dataset 하나에 대한 feature-key 매핑을 로드하고 캐시한다."""
    feature_meta_path = str(getattr(entry.manifest, "feature_meta_path", "") or "")
    if not feature_meta_path:
        raise ValueError("feature metadata path is required for feature_key requests")
    if entry.feature_key_index is None:
        entry.feature_key_index = _build_key_to_id_index(
            feature_meta_path,
            entry.feature_key_col,
            "feature_id",
        )
    return entry.feature_key_index


def _get_scalar_sample_keys_by_id(entry: _ManifestCacheEntry):
    """scalar dataset 하나에 대한 sample-id -> sample-key 매핑을 로드하고 캐시한다."""
    if entry.sample_keys_by_id is None:
        entry.sample_keys_by_id = _build_id_to_key_index(
            entry.manifest.sample_meta_path,
            entry.sample_key_col,
            "sample_id",
        )
    return entry.sample_keys_by_id


def _get_scalar_feature_keys_by_id(entry: _ManifestCacheEntry):
    """scalar dataset 하나에 대한 feature-id -> feature-key 매핑을 로드하고 캐시한다."""
    feature_meta_path = str(getattr(entry.manifest, "feature_meta_path", "") or "")
    if not feature_meta_path:
        return None
    if entry.feature_keys_by_id is None:
        entry.feature_keys_by_id = _build_id_to_key_index(
            feature_meta_path,
            entry.feature_key_col,
            "feature_id",
        )
    return entry.feature_keys_by_id


def _resolve_array_request_feature_ids(req: ArrayFeatureRequest, entry: _ManifestCacheEntry):
    """array feature 요청 필드를 검증하고 정규화한다.

    Args:
        req: 파싱된 array feature 요청 모델.
        entry: 캐시된 manifest metadata 및 reader 상태.

    Returns:
        비어 있지 않은 feature id 목록.

    Raises:
        HTTPException: 요청이 feature를 모호하게 지정했을 때 발생한다.
    """
    provided = sum(
        int(value is not None)
        for value in [req.feature_id, req.feature_ids, req.feature_key, req.feature_keys]
    )
    if provided != 1:
        raise HTTPException(
            status_code=400,
            detail="provide exactly one of feature_id, feature_ids, feature_key, or feature_keys",
        )
    if req.feature_id is not None:
        return [int(req.feature_id)]
    if req.feature_ids is not None:
        feature_ids = [int(feature_id) for feature_id in req.feature_ids]
        if not feature_ids:
            raise HTTPException(status_code=400, detail="feature_ids must not be empty")
        return feature_ids
    try:
        feature_key_index = _get_array_feature_key_index(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if req.feature_key is not None:
        feature_id = feature_key_index.get(str(req.feature_key))
        if feature_id is None:
            raise HTTPException(status_code=404, detail=f"feature key not found: {req.feature_key}")
        return [int(feature_id)]
    feature_keys = [str(feature_key) for feature_key in req.feature_keys]
    if not feature_keys:
        raise HTTPException(status_code=400, detail="feature_keys must not be empty")
    feature_ids = []
    for feature_key in feature_keys:
        feature_id = feature_key_index.get(feature_key)
        if feature_id is None:
            raise HTTPException(status_code=404, detail=f"feature key not found: {feature_key}")
        feature_ids.append(int(feature_id))
    return feature_ids


def _resolve_array_request_sample_ids(req: ArrayFeatureRequest, entry: _ManifestCacheEntry):
    """array sample 요청 필드를 검증하고 정규화한다."""
    has_ids = req.sample_ids is not None
    has_keys = req.sample_keys is not None
    if has_ids == has_keys:
        raise HTTPException(status_code=400, detail="provide exactly one of sample_ids or sample_keys")
    if req.sample_ids is not None:
        sample_ids = [int(sample_id) for sample_id in req.sample_ids]
        if not sample_ids:
            raise HTTPException(status_code=400, detail="sample_ids must not be empty")
        return sample_ids
    try:
        sample_key_index = _get_array_sample_key_index(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sample_keys = [str(sample_key) for sample_key in req.sample_keys]
    if not sample_keys:
        raise HTTPException(status_code=400, detail="sample_keys must not be empty")
    sample_ids = []
    for sample_key in sample_keys:
        sample_id = sample_key_index.get(sample_key)
        if sample_id is None:
            raise HTTPException(status_code=404, detail=f"sample key not found: {sample_key}")
        sample_ids.append(int(sample_id))
    return sample_ids


def _resolve_scalar_request_feature_id(req: ScalarFeatureRequest, entry: _ManifestCacheEntry) -> int:
    """scalar feature 요청 필드를 검증하고 정규화한다."""
    provided = int(req.feature_id is not None) + int(req.feature_key is not None)
    if provided != 1:
        raise HTTPException(status_code=400, detail="provide exactly one of feature_id or feature_key")
    if req.feature_id is not None:
        return int(req.feature_id)
    try:
        feature_key_index = _get_scalar_feature_key_index(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    feature_id = feature_key_index.get(str(req.feature_key))
    if feature_id is None:
        raise HTTPException(status_code=404, detail=f"feature key not found: {req.feature_key}")
    return int(feature_id)


def _resolve_scalar_request_sample_ids(req: ScalarFeatureRequest, entry: _ManifestCacheEntry) -> List[int]:
    """scalar sample 요청 필드를 검증하고 정규화한다."""
    has_ids = req.sample_ids is not None
    has_keys = req.sample_keys is not None
    if has_ids == has_keys:
        raise HTTPException(status_code=400, detail="provide exactly one of sample_ids or sample_keys")
    if req.sample_ids is not None:
        sample_ids = [int(sample_id) for sample_id in req.sample_ids]
        if not sample_ids:
            raise HTTPException(status_code=400, detail="sample_ids must not be empty")
        return sample_ids
    try:
        sample_key_index = _get_scalar_sample_key_index(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sample_keys = [str(sample_key) for sample_key in req.sample_keys]
    if not sample_keys:
        raise HTTPException(status_code=400, detail="sample_keys must not be empty")
    sample_ids = []
    for sample_key in sample_keys:
        sample_id = sample_key_index.get(sample_key)
        if sample_id is None:
            raise HTTPException(status_code=404, detail=f"sample key not found: {sample_key}")
        sample_ids.append(int(sample_id))
    return sample_ids


@app.get("/healthz")
def healthz():
    """liveness check용 가벼운 health 응답을 반환한다."""
    return {"ok": True}


@app.get("/cache-stats", response_model=CacheStatsResponse)
def cache_stats():
    """서빙 서버의 in-process cache 사용량과 제한을 반환한다."""
    now = time.monotonic()
    with _CACHE_LOCK:
        _sweep_manifest_cache(_ARRAY_CACHE, now)
        _sweep_manifest_cache(_ARRAY_SAMPLE_PARQUET_CACHE, now)
        _sweep_manifest_cache(_SCALAR_CACHE, now)
        array_entries = _manifest_cache_snapshot(_ARRAY_CACHE, now)
        array_sample_parquet_entries = _manifest_cache_snapshot(_ARRAY_SAMPLE_PARQUET_CACHE, now)
        scalar_entries = _manifest_cache_snapshot(_SCALAR_CACHE, now)

        scalar_open_scans = 0
        scalar_open_scan_manifests = 0
        for entry in _SCALAR_CACHE.values():
            scans = getattr(entry.reader, "_scans", {})
            scan_count = int(len(scans))
            scalar_open_scans += scan_count
            if scan_count > 0:
                scalar_open_scan_manifests += 1

    return CacheStatsResponse(
        manifest_cache={
            "array_entries": int(len(array_entries)),
            "scalar_entries": int(len(scalar_entries)),
            "max_entries": int(_MANIFEST_CACHE_MAX_ENTRIES),
            "ttl_seconds": int(_MANIFEST_CACHE_TTL_SECONDS),
            "array_manifests": array_entries,
            "scalar_manifests": scalar_entries,
        },
        array_binary_cache=get_array_binary_cache_stats(),
        array_parquet_cache={
            "entries": array_sample_parquet_entries,
            "max_entries": int(_MANIFEST_CACHE_MAX_ENTRIES),
            "ttl_seconds": int(_MANIFEST_CACHE_TTL_SECONDS),
        },
        scalar_parquet_cache={
            "open_scan_manifests": int(scalar_open_scan_manifests),
            "open_scans": int(scalar_open_scans),
        },
    )


@app.post("/array-schema", response_model=ArraySchemaResponse)
def array_schema(req: ArraySchemaRequest):
    """array manifest 하나에 대한 point-column schema를 반환한다."""
    entry = _get_array_cache_entry(req.manifest_path)
    point_schema = _array_point_schema(entry)
    categorical_dictionaries = None
    if req.include_dictionaries:
        categorical_dictionaries = {
            column_name: {
                str(code): label
                for code, label in mapping.items()
            }
            for column_name, mapping in _get_array_categorical_dictionaries(entry).items()
        }
    return ArraySchemaResponse(
        manifest_path=entry.manifest_path,
        version=int(entry.manifest.version),
        n_samples=int(entry.manifest.n_samples),
        n_features=int(entry.manifest.n_features),
        samples_per_block=int(entry.manifest.samples_per_block),
        default_codec=str(entry.manifest.default_codec),
        point_schema=_schema_json(point_schema),
        categorical_dictionaries=categorical_dictionaries,
    )


@app.post("/array-sample-parquet/schema")
def array_sample_parquet_schema(req: ArraySampleParquetSchemaRequest):
    """sample-major Parquet array dataset의 schema와 dictionary 정보를 반환한다."""
    try:
        entry = _get_array_sample_parquet_cache_entry(req.manifest_path)
        dictionaries = None
        if req.include_dictionaries:
            dictionaries = entry.reader.categorical_dictionaries()
        return {
            "manifest_path": entry.manifest_path,
            "format": "array-sample-parquet",
            "version": int(entry.manifest.version),
            "n_samples": int(entry.manifest.n_samples),
            "n_features": int(entry.manifest.n_features),
            "sample_key_col": str(entry.manifest.sample_key_col),
            "feature_key_col": str(entry.manifest.feature_key_col),
            "part_count": int(len(entry.manifest.parts)),
            "point_schema": _schema_json(entry.manifest.point_schema),
            "categorical_dictionaries": dictionaries,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/array-sample-parquet/traces")
def array_sample_parquet_traces(req: ArraySampleParquetTraceRequest):
    """sample-major Parquet array trace를 sample/key 또는 feature/key 기준으로 조회한다.

    sample은 `sample_ids` 또는 `sample_keys` 중 정확히 하나가 필요하다.
    feature는 `feature_ids` 또는 `feature_keys` 중 최대 하나만 허용하며,
    둘 다 생략하면 요청 sample에 존재하는 모든 trace를 반환한다.
    """
    try:
        entry = _get_array_sample_parquet_cache_entry(req.manifest_path)
        result = entry.reader.get_traces_json(
            sample_ids=req.sample_ids,
            sample_keys=req.sample_keys,
            feature_ids=req.feature_ids,
            feature_keys=req.feature_keys,
            decode_categorical=bool(req.decode_categorical),
            include_missing=bool(req.include_missing),
            layout=str(req.layout),
        )
        trace_count = int(result.get("trace_count", 0))
        if trace_count > int(req.max_traces):
            raise HTTPException(
                status_code=413,
                detail=f"trace_count exceeds max_traces: {trace_count} > {int(req.max_traces)}",
            )
        return {
            "manifest_path": entry.manifest_path,
            **result,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/array-feature", response_model=ArrayFeatureResponse)
def array_feature(req: ArrayFeatureRequest):
    """하나 이상의 feature와 요청한 sample id에 대한 array trace를 반환한다.

    Args:
        req: manifest, feature id, sample id를 담은 요청 객체.

    Returns:
        feature 하나의 trace 목록 또는 feature별 결과 목록을 담은 응답.
    """
    entry = _get_array_cache_entry(req.manifest_path)
    feature_ids = _resolve_array_request_feature_ids(req, entry)
    sample_ids = _resolve_array_request_sample_ids(req, entry)
    sample_keys_by_id = _get_array_sample_keys_by_id(entry)
    feature_keys_by_id = _get_array_feature_keys_by_id(entry)
    point_schema = _array_point_schema(entry)
    point_schema_by_name = {
        (spec.name if hasattr(spec, "name") else str(spec["name"])): spec
        for spec in point_schema
    }
    categorical_dictionaries = _get_array_categorical_dictionaries(entry) if req.decode_categorical else {}
    temporal_format = _normalize_temporal_format(req.temporal_format)
    response_items = []
    for feature_id in feature_ids:
        traces = entry.reader.load_feature_samples_by_sample_ids(
            feature_id=feature_id,
            sample_ids=sample_ids,
        )
        response_traces = []
        for sample_id in sample_ids:
            trace = traces[int(sample_id)]
            response_columns = {}
            for column_name, values in trace.columns.items():
                spec = point_schema_by_name.get(column_name)
                if spec is None:
                    continue
                response_columns[column_name] = _json_safe_column(
                    values,
                    spec,
                    req.sanitize_nonfinite,
                    bool(req.decode_categorical),
                    categorical_dictionaries.get(column_name),
                    temporal_format,
                )
            response_traces.append(
                ArrayTraceResponse(
                    sample_id=int(sample_id),
                    sample_key=None if sample_keys_by_id is None else sample_keys_by_id.get(int(sample_id)),
                    flags=int(trace.flags),
                    columns=response_columns,
                )
            )
        response_items.append(
            ArrayFeatureItemResponse(
                feature_id=int(feature_id),
                feature_key=None if feature_keys_by_id is None else feature_keys_by_id.get(int(feature_id)),
                traces=response_traces,
            )
        )
    if len(response_items) == 1:
        item = response_items[0]
        return ArrayFeatureResponse(
            manifest_path=entry.manifest_path,
            feature_id=item.feature_id,
            feature_key=item.feature_key,
            sample_count=len(sample_ids),
            traces=item.traces,
        )
    return ArrayFeatureResponse(
        manifest_path=entry.manifest_path,
        feature_ids=[item.feature_id for item in response_items],
        feature_keys=[item.feature_key for item in response_items],
        sample_count=len(sample_ids),
        features=response_items,
    )


@app.post("/scalar-feature", response_model=ScalarFeatureResponse)
def scalar_feature(req: ScalarFeatureRequest):
    """요청한 sample id에 대한 scalar feature 값을 반환한다.

    Args:
        req: manifest, feature id, sample id를 담은 요청 객체.

    Returns:
        요청한 sample id마다 scalar value 레코드 하나를 담은 응답.
    """
    entry = _get_scalar_cache_entry(req.manifest_path)
    feature_id = _resolve_scalar_request_feature_id(req, entry)
    sample_ids = _resolve_scalar_request_sample_ids(req, entry)
    values, valid = entry.reader.load_feature_by_id(feature_id, locator_index=entry.locator_index)
    sample_keys_by_id = _get_scalar_sample_keys_by_id(entry)
    feature_keys_by_id = _get_scalar_feature_keys_by_id(entry)
    response_values = []
    for sample_id in sample_ids:
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= int(entry.manifest.n_samples):
            response_values.append(
                ScalarValueResponse(
                    sample_id=sample_id,
                    sample_key=None if sample_keys_by_id is None else sample_keys_by_id.get(sample_id),
                    present=False,
                    value=None,
                )
            )
            continue
        present = bool(valid[sample_id])
        value = float(values[sample_id]) if present else None
        if req.sanitize_nonfinite and value is not None and not (value == value and abs(value) != float("inf")):
            value = None
        response_values.append(
            ScalarValueResponse(
                sample_id=sample_id,
                sample_key=None if sample_keys_by_id is None else sample_keys_by_id.get(sample_id),
                present=present,
                value=value,
            )
        )
    return ScalarFeatureResponse(
        manifest_path=entry.manifest_path,
        feature_id=feature_id,
        feature_key=None if feature_keys_by_id is None else feature_keys_by_id.get(feature_id),
        sample_count=len(sample_ids),
        values=response_values,
    )


@app.post("/run-selection", response_model=SelectionResponse)
def run_selection(req: SelectionRequest):
    """scalar feature selection을 실행하고 선택된 feature id를 반환한다.

    Args:
        req: selection threshold와 batching 파라미터를 담은 요청 객체.

    Returns:
        selection 결과 metadata, timing, 선택된 feature id를 담은 응답.
    """
    entry = _get_scalar_cache_entry(req.manifest_path)
    total_started = time.perf_counter()

    candidate_started = time.perf_counter()
    stats_path = resolve_selection_stats_path(entry.manifest, req.y_col)
    used_locator_stats = bool(stats_path)
    if used_locator_stats:
        candidates = build_candidates_from_stats(
            stats_path,
            min_non_null_y=req.min_non_null_y,
            y_r2_threshold=req.y_r2,
            max_candidates=req.max_candidates,
        )
    elif entry.manifest.stats_y_col == req.y_col and locator_has_candidate_stats(entry.manifest.feature_locator_path):
        used_locator_stats = True
        candidates = build_candidates_from_stats(
            entry.manifest.feature_locator_path,
            min_non_null_y=req.min_non_null_y,
            y_r2_threshold=req.y_r2,
            max_candidates=req.max_candidates,
        )
    else:
        _, y, y_mask = load_sample_targets(entry.manifest.sample_meta_path, y_col=req.y_col)
        candidates = build_candidates_from_shards(
            list_shard_paths(entry.manifest),
            y,
            y_mask,
            min_non_null_y=req.min_non_null_y,
            y_r2_threshold=req.y_r2,
            max_candidates=req.max_candidates,
            batch_size=req.batch_size,
        )
    candidate_build_ms = int(round((time.perf_counter() - candidate_started) * 1000.0))

    reader = ParquetShardReader(entry.manifest, max_gap=req.max_gap)
    config = SelectionConfig(
        y_r2_threshold=req.y_r2,
        min_non_null_y=req.min_non_null_y,
        ff_r2_threshold=req.ff_r2,
        min_non_null_pair=req.min_non_null_pair,
        top_m=req.top_m,
        initial_cap=req.initial_cap,
        max_step=req.max_step,
        batch_size=req.batch_size,
        max_gap=req.max_gap,
        max_candidates=req.max_candidates,
        mask_fastpath_min_group=req.mask_fastpath_min_group,
        mask_fastpath_min_pairs=req.mask_fastpath_min_pairs,
    )

    selection_started = time.perf_counter()
    selected = select_features_incremental(candidates, reader, config)
    selection_ms = int(round((time.perf_counter() - selection_started) * 1000.0))
    total_elapsed_ms = int(round((time.perf_counter() - total_started) * 1000.0))

    feature_keys_by_id = _get_scalar_feature_keys_by_id(entry)
    return SelectionResponse(
        manifest_path=entry.manifest_path,
        y_col=req.y_col,
        top_m=req.top_m,
        candidate_count=len(candidates),
        selected_count=len(selected),
        selected_feature_ids=[int(c.feature_id) for c in selected],
        selected_feature_keys=None
        if feature_keys_by_id is None
        else [feature_keys_by_id.get(int(c.feature_id)) for c in selected],
        used_locator_stats=used_locator_stats,
        candidate_build_ms=candidate_build_ms,
        selection_ms=selection_ms,
        elapsed_ms=total_elapsed_ms,
    )


def main():
    """uvicorn으로 로컬 FastAPI 서버를 실행한다."""
    uvicorn.run(
        "scripts.serve_array_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
