from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
for package_src in [
    REPO_ROOT / "packages" / "scalar_feature_shard" / "src",
    REPO_ROOT / "packages" / "array_sample_parquet" / "src",
]:
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))

from array_sample_parquet import open_array_sample_parquet
from scalar_feature_shard import open_dense_long_shard


SCALAR_DENSE_LONG_FORMAT = "scalar-dense-long-shard-v1"
ARRAY_SAMPLE_PARQUET_FORMAT = "array-sample-parquet"


class ManifestRequest(BaseModel):
    manifest_path: str = Field(..., description="조회할 manifest JSON 경로입니다.")


class ArraySampleParquetSchemaRequest(ManifestRequest):
    pass


class ArraySampleParquetTraceRequest(ManifestRequest):
    sample_ids: Optional[list[int]] = None
    sample_keys: Optional[list[str]] = None
    feature_ids: Optional[list[int]] = None
    feature_keys: Optional[list[str]] = None
    include_missing: bool = False
    decode_categorical: bool = True
    layout: str = "nested"
    max_traces: int = 10000


class ScalarSchemaRequest(ManifestRequest):
    scalar_format: str = "auto"


class ScalarFeatureQueryRequest(ManifestRequest):
    scalar_format: str = "auto"
    feature_ids: Optional[list[int]] = None
    feature_keys: Optional[list[str]] = None
    sample_ids: Optional[list[int]] = None
    sample_keys: Optional[list[str]] = None
    max_cells: int = 100000
    strict: bool = True


class ScalarSampleQueryRequest(ManifestRequest):
    scalar_format: str = "auto"
    sample_id: Optional[int] = None
    sample_key: Optional[str] = None
    feature_ids: Optional[list[int]] = None
    feature_keys: Optional[list[str]] = None
    max_features: int = 10000
    strict: bool = True


class ScalarTopFeaturesRequest(ManifestRequest):
    scalar_format: str = "auto"
    y_col: str = "y"
    top_k: int = 256


@dataclass
class _MetadataIndex:
    sample_meta_path: str
    feature_meta_path: str
    n_samples: int
    n_features: int
    sample_key_col: str
    feature_key_col: str
    _sample_key_to_id: Optional[dict[str, int]] = None
    _feature_key_to_id: Optional[dict[str, int]] = None
    _sample_keys_by_id: Optional[dict[int, str]] = None
    _feature_keys_by_id: Optional[dict[int, str]] = None

    def sample_key_to_id(self) -> dict[str, int]:
        if self._sample_key_to_id is None:
            self._sample_key_to_id = _build_key_to_id(self.sample_meta_path, self.sample_key_col, "sample_id")
        return self._sample_key_to_id

    def feature_key_to_id(self) -> dict[str, int]:
        if self._feature_key_to_id is None:
            self._feature_key_to_id = _build_key_to_id(self.feature_meta_path, self.feature_key_col, "feature_id")
        return self._feature_key_to_id

    def sample_keys_by_id(self) -> dict[int, str]:
        if self._sample_keys_by_id is None:
            self._sample_keys_by_id = _build_id_to_key(self.sample_meta_path, self.sample_key_col, "sample_id")
        return self._sample_keys_by_id

    def feature_keys_by_id(self) -> dict[int, str]:
        if self._feature_keys_by_id is None:
            self._feature_keys_by_id = _build_id_to_key(self.feature_meta_path, self.feature_key_col, "feature_id")
        return self._feature_keys_by_id


@dataclass
class _CachedDataset:
    manifest_path: str
    kind: str
    manifest_json: dict[str, Any]
    reader: Any
    metadata: Optional[_MetadataIndex]
    last_access: float


app = FastAPI(title="Feature Query API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: OrderedDict[str, _CachedDataset] = OrderedDict()
_MAX_CACHE_ENTRIES = 16
_CACHE_TTL_SECONDS = 30 * 60


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest_path(manifest_path: str, value: str) -> str:
    if not value:
        return ""
    raw = Path(value)
    if raw.is_absolute():
        return str(raw.resolve())
    return str((Path(manifest_path).resolve().parent / raw).resolve())


def _detect_kind(manifest_path: str, scalar_format: str = "auto") -> str:
    payload = _read_json(manifest_path)
    manifest_format = str(payload.get("format") or "")
    requested = str(scalar_format or "auto").strip().lower().replace("_", "-")
    if manifest_format == ARRAY_SAMPLE_PARQUET_FORMAT:
        return "array-sample-parquet"
    if requested in {"dense-long", "scalar-dense-long"}:
        return "scalar-dense-long"
    if manifest_format == SCALAR_DENSE_LONG_FORMAT:
        return "scalar-dense-long"
    raise ValueError(f"unsupported manifest format: {manifest_format or '<missing>'}")


def _build_metadata_index(manifest_path: str, manifest_json: dict[str, Any]) -> _MetadataIndex:
    return _MetadataIndex(
        sample_meta_path=_resolve_manifest_path(manifest_path, str(manifest_json["sample_meta_path"])),
        feature_meta_path=_resolve_manifest_path(manifest_path, str(manifest_json["feature_meta_path"])),
        n_samples=int(manifest_json["n_samples"]),
        n_features=int(manifest_json["n_features"]),
        sample_key_col=str(manifest_json.get("sample_key_col", "sample_key")),
        feature_key_col=str(manifest_json.get("feature_key_col", "feature_key")),
    )


def _build_key_to_id(meta_path: str, key_col: str, id_col: str) -> dict[str, int]:
    if not key_col:
        raise ValueError("metadata key column is not configured")
    df = pl.read_parquet(meta_path)
    if key_col not in df.columns:
        raise ValueError(f"metadata key column not found: {key_col}")
    if int(df[key_col].null_count()) != 0:
        raise ValueError(f"metadata key column contains nulls: {key_col}")
    if int(df[key_col].n_unique()) != int(df.height):
        raise ValueError(f"metadata key column must be unique: {key_col}")
    ids = df[id_col].to_list() if id_col in df.columns else list(range(df.height))
    return {str(key): int(value) for key, value in zip(df[key_col].to_list(), ids)}


def _build_id_to_key(meta_path: str, key_col: str, id_col: str) -> dict[int, str]:
    if not key_col:
        return {}
    df = pl.read_parquet(meta_path)
    if key_col not in df.columns:
        return {}
    ids = df[id_col].to_list() if id_col in df.columns else list(range(df.height))
    return {int(value): str(key) for key, value in zip(df[key_col].to_list(), ids)}


def _close_reader(reader: Any):
    close_fn = getattr(reader, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def _sweep_cache(now: float):
    expired = [
        key
        for key, entry in _DATASET_CACHE.items()
        if now - float(entry.last_access) > _CACHE_TTL_SECONDS
    ]
    for key in expired:
        entry = _DATASET_CACHE.pop(key, None)
        if entry is not None:
            _close_reader(entry.reader)
    while len(_DATASET_CACHE) > _MAX_CACHE_ENTRIES:
        _key, entry = _DATASET_CACHE.popitem(last=False)
        _close_reader(entry.reader)


def _get_dataset(manifest_path: str, scalar_format: str = "auto") -> _CachedDataset:
    normalized = _normalize_path(manifest_path)
    kind = _detect_kind(normalized, scalar_format=scalar_format)
    cache_key = f"{kind}:{normalized}"
    with _CACHE_LOCK:
        now = time.monotonic()
        _sweep_cache(now)
        cached = _DATASET_CACHE.get(cache_key)
        if cached is not None:
            cached.last_access = now
            _DATASET_CACHE.move_to_end(cache_key)
            return cached

        manifest_json = _read_json(normalized)
        if kind == "array-sample-parquet":
            reader = open_array_sample_parquet(normalized)
            metadata = None
        elif kind == "scalar-dense-long":
            reader = open_dense_long_shard(normalized)
            metadata = _build_metadata_index(normalized, manifest_json)
        else:
            raise ValueError(f"unsupported dataset kind: {kind}")

        entry = _CachedDataset(
            manifest_path=normalized,
            kind=kind,
            manifest_json=manifest_json,
            reader=reader,
            metadata=metadata,
            last_access=now,
        )
        _DATASET_CACHE[cache_key] = entry
        _sweep_cache(now)
        return entry


def _schema_json(specs) -> list[dict[str, str]]:
    out = []
    for spec in specs:
        out.append(
            {
                "name": str(spec.name),
                "storage_type": str(spec.storage_type.value if hasattr(spec.storage_type, "value") else spec.storage_type),
                "logical_type": str(spec.logical_type.value if hasattr(spec.logical_type, "value") else spec.logical_type),
            }
        )
    return out


def _exactly_one(left_name: str, left_value, right_name: str, right_value):
    if (left_value is None) == (right_value is None):
        raise HTTPException(status_code=400, detail=f"provide exactly one of {left_name} or {right_name}")


def _resolve_ids(
    *,
    ids: Optional[list[int]],
    keys: Optional[list[str]],
    key_to_id: dict[str, int],
    id_name: str,
    key_name: str,
) -> list[int]:
    _exactly_one(id_name, ids, key_name, keys)
    if ids is not None:
        out = [int(value) for value in ids]
    else:
        out = []
        for key in keys or []:
            value = key_to_id.get(str(key))
            if value is None:
                raise HTTPException(status_code=404, detail=f"{key_name[:-1]} not found: {key}")
            out.append(int(value))
    if not out:
        raise HTTPException(status_code=400, detail=f"{id_name} must not be empty")
    return out


def _resolve_optional_feature_ids(metadata: _MetadataIndex, feature_ids, feature_keys, max_features: int) -> list[int]:
    if feature_ids is not None and feature_keys is not None:
        raise HTTPException(status_code=400, detail="provide at most one of feature_ids or feature_keys")
    if feature_ids is not None:
        out = [int(value) for value in feature_ids]
    elif feature_keys is not None:
        index = metadata.feature_key_to_id()
        out = []
        for key in feature_keys:
            value = index.get(str(key))
            if value is None:
                raise HTTPException(status_code=404, detail=f"feature_key not found: {key}")
            out.append(int(value))
    else:
        out = list(range(int(metadata.n_features)))
    if len(out) > int(max_features):
        raise HTTPException(status_code=413, detail=f"feature count exceeds max_features: {len(out)} > {max_features}")
    return out


def _resolve_sample_id(metadata: _MetadataIndex, sample_id: Optional[int], sample_key: Optional[str]) -> int:
    _exactly_one("sample_id", sample_id, "sample_key", sample_key)
    if sample_id is not None:
        return int(sample_id)
    value = metadata.sample_key_to_id().get(str(sample_key))
    if value is None:
        raise HTTPException(status_code=404, detail=f"sample_key not found: {sample_key}")
    return int(value)


def _require_scalar(entry: _CachedDataset) -> _MetadataIndex:
    if not entry.kind.startswith("scalar-"):
        raise HTTPException(status_code=400, detail="manifest is not a scalar dataset")
    if entry.metadata is None:
        raise HTTPException(status_code=500, detail="scalar metadata index is missing")
    return entry.metadata


def _json_float(value: float | None) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _json_safe_nested(value):
    """JSON 응답에 직접 넣을 수 없는 NaN/Inf float를 null로 바꾼다."""

    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {key: _json_safe_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_nested(item) for item in value]
    return value


def _validate_cells(feature_count: int, sample_count: int, max_cells: int):
    cells = int(feature_count) * int(sample_count)
    if cells > int(max_cells):
        raise HTTPException(status_code=413, detail=f"requested cells exceed max_cells: {cells} > {int(max_cells)}")


def _scalar_feature_rows(entry: _CachedDataset, feature_ids: list[int], sample_ids: list[int]) -> list[dict[str, Any]]:
    metadata = _require_scalar(entry)
    feature_keys = metadata.feature_keys_by_id()
    sample_keys = metadata.sample_keys_by_id()
    rows = []
    if entry.kind == "scalar-dense-long":
        values, valid = entry.reader.load_features_by_ids(feature_ids)
        for feature_idx, feature_id in enumerate(feature_ids):
            rows.append(
                {
                    "feature_id": int(feature_id),
                    "feature_key": feature_keys.get(int(feature_id)),
                    "values": [
                        {
                            "sample_id": int(sample_id),
                            "sample_key": sample_keys.get(int(sample_id)),
                            "present": bool(valid[feature_idx, int(sample_id)]),
                            "value": _json_float(values[feature_idx, int(sample_id)])
                            if bool(valid[feature_idx, int(sample_id)])
                            else None,
                        }
                        for sample_id in sample_ids
                    ],
                }
            )
        return rows
    raise HTTPException(status_code=400, detail=f"unsupported scalar dataset kind: {entry.kind}")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/cache-stats")
def cache_stats():
    now = time.monotonic()
    with _CACHE_LOCK:
        _sweep_cache(now)
        return {
            "entries": [
                {
                    "manifest_path": entry.manifest_path,
                    "kind": entry.kind,
                    "idle_seconds": round(now - entry.last_access, 3),
                }
                for entry in _DATASET_CACHE.values()
            ],
            "max_entries": _MAX_CACHE_ENTRIES,
            "ttl_seconds": _CACHE_TTL_SECONDS,
        }


@app.post("/array-sample-parquet/schema")
def array_sample_parquet_schema(req: ArraySampleParquetSchemaRequest):
    try:
        entry = _get_dataset(req.manifest_path)
        if entry.kind != "array-sample-parquet":
            raise HTTPException(status_code=400, detail="manifest is not array_sample_parquet")
        manifest = entry.reader.manifest
        return {
            "manifest_path": entry.manifest_path,
            "format": ARRAY_SAMPLE_PARQUET_FORMAT,
            "version": int(manifest.version),
            "n_samples": int(manifest.n_samples),
            "n_features": int(manifest.n_features),
            "sample_key_col": str(manifest.sample_key_col),
            "feature_key_col": str(manifest.feature_key_col),
            "part_count": int(len(manifest.parts)),
            "point_schema": _schema_json(manifest.point_schema),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/array-sample-parquet/traces")
def array_sample_parquet_traces(req: ArraySampleParquetTraceRequest):
    try:
        entry = _get_dataset(req.manifest_path)
        if entry.kind != "array-sample-parquet":
            raise HTTPException(status_code=400, detail="manifest is not array_sample_parquet")
        result = entry.reader.get_traces_json(
            sample_ids=req.sample_ids,
            sample_keys=req.sample_keys,
            feature_ids=req.feature_ids,
            feature_keys=req.feature_keys,
            include_missing=bool(req.include_missing),
            decode_categorical=bool(req.decode_categorical),
            layout=str(req.layout),
        )
        trace_count = int(result.get("trace_count", 0))
        if trace_count > int(req.max_traces):
            raise HTTPException(
                status_code=413,
                detail=f"trace_count exceeds max_traces: {trace_count} > {int(req.max_traces)}",
            )
        return {"manifest_path": entry.manifest_path, **_json_safe_nested(result)}
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/scalar/schema")
def scalar_schema(req: ScalarSchemaRequest):
    try:
        entry = _get_dataset(req.manifest_path, scalar_format=req.scalar_format)
        metadata = _require_scalar(entry)
        return {
            "manifest_path": entry.manifest_path,
            "format": entry.kind,
            "n_samples": int(metadata.n_samples),
            "n_features": int(metadata.n_features),
            "sample_key_col": metadata.sample_key_col,
            "feature_key_col": metadata.feature_key_col,
            "selection_stats": sorted((entry.manifest_json.get("selection_stats") or {}).keys()),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/scalar/features")
def scalar_features(req: ScalarFeatureQueryRequest):
    try:
        entry = _get_dataset(req.manifest_path, scalar_format=req.scalar_format)
        metadata = _require_scalar(entry)
        feature_ids = _resolve_ids(
            ids=req.feature_ids,
            keys=req.feature_keys,
            key_to_id=metadata.feature_key_to_id(),
            id_name="feature_ids",
            key_name="feature_keys",
        )
        sample_ids = _resolve_ids(
            ids=req.sample_ids,
            keys=req.sample_keys,
            key_to_id=metadata.sample_key_to_id(),
            id_name="sample_ids",
            key_name="sample_keys",
        )
        _validate_cells(len(feature_ids), len(sample_ids), req.max_cells)
        if req.strict:
            _validate_scalar_id_ranges(metadata, feature_ids, sample_ids)
        return {
            "manifest_path": entry.manifest_path,
            "format": entry.kind,
            "layout": "features",
            "feature_count": len(feature_ids),
            "sample_count": len(sample_ids),
            "features": _scalar_feature_rows(entry, feature_ids, sample_ids),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/scalar/sample")
def scalar_sample(req: ScalarSampleQueryRequest):
    try:
        entry = _get_dataset(req.manifest_path, scalar_format=req.scalar_format)
        metadata = _require_scalar(entry)
        sample_id = _resolve_sample_id(metadata, req.sample_id, req.sample_key)
        feature_ids = _resolve_optional_feature_ids(metadata, req.feature_ids, req.feature_keys, req.max_features)
        if req.strict:
            _validate_scalar_id_ranges(metadata, feature_ids, [sample_id])
        feature_keys = metadata.feature_keys_by_id()
        sample_keys = metadata.sample_keys_by_id()

        values_by_feature = []
        if entry.kind == "scalar-dense-long" and req.feature_ids is None and req.feature_keys is None:
            values, valid = entry.reader.load_sample_by_id(sample_id)
            for feature_id in feature_ids:
                present = bool(valid[int(feature_id)])
                values_by_feature.append(
                    {
                        "feature_id": int(feature_id),
                        "feature_key": feature_keys.get(int(feature_id)),
                        "present": present,
                        "value": _json_float(values[int(feature_id)]) if present else None,
                    }
                )
        else:
            rows = _scalar_feature_rows(entry, feature_ids, [sample_id])
            for row in rows:
                value = row["values"][0]
                values_by_feature.append(
                    {
                        "feature_id": int(row["feature_id"]),
                        "feature_key": row.get("feature_key"),
                        "present": bool(value["present"]),
                        "value": value["value"],
                    }
                )

        return {
            "manifest_path": entry.manifest_path,
            "format": entry.kind,
            "layout": "sample",
            "sample_id": int(sample_id),
            "sample_key": sample_keys.get(int(sample_id)),
            "feature_count": len(values_by_feature),
            "values": values_by_feature,
        }
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/scalar/top-features")
def scalar_top_features(req: ScalarTopFeaturesRequest):
    try:
        entry = _get_dataset(req.manifest_path, scalar_format=req.scalar_format)
        metadata = _require_scalar(entry)
        stats_mapping = entry.manifest_json.get("selection_stats") or {}
        raw_path = stats_mapping.get(str(req.y_col))
        if not raw_path:
            raise HTTPException(status_code=404, detail=f"selection stats not found for y_col: {req.y_col}")
        stats_path = _resolve_manifest_path(entry.manifest_path, str(raw_path))
        df = (
            pl.scan_parquet(stats_path)
            .sort(["r2y", "feature_id"], descending=[True, False], nulls_last=True)
            .head(int(req.top_k))
            .collect()
        )
        feature_keys = metadata.feature_keys_by_id()
        rows = []
        for row in df.iter_rows(named=True):
            feature_id = int(row["feature_id"])
            rows.append(
                {
                    "feature_id": feature_id,
                    "feature_key": feature_keys.get(feature_id),
                    "r2y": _json_float(row.get("r2y")),
                    "n_y_overlap": None if row.get("n_y_overlap") is None else int(row.get("n_y_overlap")),
                    "part_id": None if row.get("part_id") is None else int(row.get("part_id")),
                    "offset_in_part": None if row.get("offset_in_part") is None else int(row.get("offset_in_part")),
                    "shard_id": None if row.get("shard_id") is None else int(row.get("shard_id")),
                    "offset_in_shard": None if row.get("offset_in_shard") is None else int(row.get("offset_in_shard")),
                }
            )
        return {
            "manifest_path": entry.manifest_path,
            "format": entry.kind,
            "y_col": str(req.y_col),
            "top_k": int(req.top_k),
            "features": rows,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _validate_scalar_id_ranges(metadata: _MetadataIndex, feature_ids: list[int], sample_ids: list[int]):
    bad_features = [value for value in feature_ids if value < 0 or value >= int(metadata.n_features)]
    bad_samples = [value for value in sample_ids if value < 0 or value >= int(metadata.n_samples)]
    if bad_features:
        raise HTTPException(status_code=404, detail=f"feature ids out of range: {bad_features}")
    if bad_samples:
        raise HTTPException(status_code=404, detail=f"sample ids out of range: {bad_samples}")


def main():
    parser = argparse.ArgumentParser(description="Feature query FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    target = "scripts.serve_feature_query_api:app" if bool(args.reload) else app
    uvicorn.run(target, host=args.host, port=int(args.port), reload=bool(args.reload))


if __name__ == "__main__":
    main()
