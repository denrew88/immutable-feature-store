from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


class SyntheticValueRequest(BaseModel):
    sample_meta_path: str
    feature_meta_path: str
    sample_id: Optional[int] = None
    sample_key: Optional[str] = None
    feature_ids: Optional[list[int]] = None
    feature_keys: Optional[list[str]] = None
    sample_key_col: str = "sample_key"
    feature_key_col: str = "feature_key"
    seed: int = 0


class ScalarValueRequest(SyntheticValueRequest):
    missing_rate: float = Field(0.1, ge=0.0, le=1.0)
    n_latent_groups: int = Field(16, ge=1)
    noise_scale: float = Field(0.25, ge=0.0)


class ArrayTraceRequest(SyntheticValueRequest):
    missing_feature_rate: float = Field(0.05, ge=0.0, le=1.0)
    empty_trace_rate: float = Field(0.02, ge=0.0, le=1.0)
    min_trace_len: int = Field(24, ge=0)
    max_trace_len: int = Field(48, ge=0)
    n_latent_groups: int = Field(16, ge=1)
    noise_scale: float = Field(0.1, ge=0.0)
    include_missing: bool = False


@dataclass(frozen=True)
class _MetaIndex:
    row_count: int
    key_to_id: dict[str, int]
    id_to_key: dict[int, str]


app = FastAPI(
    title="Synthetic Feature Value API",
    version="1.0.0",
    description=(
        "Synthetic value source used by Java builders. "
        "sample_meta/feature_meta define dense ids and optional keys; this service only returns values."
    ),
)


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/scalar/values")
def scalar_values(req: ScalarValueRequest) -> dict[str, Any]:
    try:
        sample_meta = _load_meta(req.sample_meta_path, "sample_id", req.sample_key_col)
        feature_meta = _load_meta(req.feature_meta_path, "feature_id", req.feature_key_col)
        sample_id = _resolve_sample_id(req, sample_meta)
        feature_ids = _resolve_feature_ids(req, feature_meta)
        rows = []
        for feature_id in feature_ids:
            present = _scalar_present(req.seed, sample_id, feature_id, req.missing_rate)
            rows.append(
                {
                    "feature_id": int(feature_id),
                    "feature_key": feature_meta.id_to_key.get(int(feature_id)),
                    "present": bool(present),
                    "value": _scalar_value(req, sample_id, feature_id) if present else None,
                }
            )
        return {
            "sample_id": int(sample_id),
            "sample_key": sample_meta.id_to_key.get(int(sample_id)),
            "feature_count": len(rows),
            "values": rows,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/array/traces")
def array_traces(req: ArrayTraceRequest) -> dict[str, Any]:
    try:
        if int(req.max_trace_len) < int(req.min_trace_len):
            raise HTTPException(status_code=400, detail="max_trace_len must be >= min_trace_len")
        sample_meta = _load_meta(req.sample_meta_path, "sample_id", req.sample_key_col)
        feature_meta = _load_meta(req.feature_meta_path, "feature_id", req.feature_key_col)
        sample_id = _resolve_sample_id(req, sample_meta)
        feature_ids = _resolve_feature_ids(req, feature_meta)
        traces = []
        for feature_id in feature_ids:
            present = _array_present(req.seed, sample_id, feature_id, req.missing_feature_rate)
            if not present:
                if req.include_missing:
                    traces.append(
                        {
                            "feature_id": int(feature_id),
                            "feature_key": feature_meta.id_to_key.get(int(feature_id)),
                            "present": False,
                            "trace_len": 0,
                            "columns": {"time": [], "value": [], "ch_step": []},
                        }
                    )
                continue
            columns = _array_columns(req, sample_id, feature_id)
            traces.append(
                {
                    "feature_id": int(feature_id),
                    "feature_key": feature_meta.id_to_key.get(int(feature_id)),
                    "present": True,
                    "trace_len": len(columns["time"]),
                    "columns": columns,
                }
            )
        return {
            "sample_id": int(sample_id),
            "sample_key": sample_meta.id_to_key.get(int(sample_id)),
            "feature_count": len(feature_ids),
            "trace_count": len(traces),
            "point_schema": [
                {"name": "time", "storage_type": "float64", "logical_type": "continuous"},
                {"name": "value", "storage_type": "float64", "logical_type": "continuous"},
                {"name": "ch_step", "storage_type": "string", "logical_type": "categorical"},
            ],
            "traces": traces,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@lru_cache(maxsize=32)
def _load_meta(path: str, id_col: str, key_col: str) -> _MetaIndex:
    resolved = str(Path(path).expanduser().resolve())
    df = pl.read_parquet(resolved)
    row_count = int(df.height)
    ids = df[id_col].to_list() if id_col in df.columns else list(range(row_count))
    if [int(value) for value in ids] != list(range(row_count)):
        raise ValueError(f"{id_col} must be dense ids 0..N-1 in row order: {resolved}")

    key_to_id: dict[str, int] = {}
    id_to_key: dict[int, str] = {}
    if key_col and key_col in df.columns:
        keys = df[key_col].to_list()
        if any(value is None for value in keys):
            raise ValueError(f"{key_col} must not contain nulls: {resolved}")
        if len(set(str(value) for value in keys)) != row_count:
            raise ValueError(f"{key_col} must be unique: {resolved}")
        for idx, key in enumerate(keys):
            text = str(key)
            key_to_id[text] = int(idx)
            id_to_key[int(idx)] = text
    return _MetaIndex(row_count=row_count, key_to_id=key_to_id, id_to_key=id_to_key)


def _resolve_sample_id(req: SyntheticValueRequest, meta: _MetaIndex) -> int:
    if (req.sample_id is None) == (req.sample_key is None):
        raise HTTPException(status_code=400, detail="provide exactly one of sample_id or sample_key")
    if req.sample_id is not None:
        sample_id = int(req.sample_id)
    else:
        sample_id = meta.key_to_id.get(str(req.sample_key), -1)
    if sample_id < 0 or sample_id >= meta.row_count:
        raw_ref = req.sample_id if req.sample_id is not None else req.sample_key
        raise HTTPException(status_code=404, detail=f"sample not found: {raw_ref}")
    return int(sample_id)


def _resolve_feature_ids(req: SyntheticValueRequest, meta: _MetaIndex) -> list[int]:
    if (req.feature_ids is None) == (req.feature_keys is None):
        raise HTTPException(status_code=400, detail="provide exactly one of feature_ids or feature_keys")
    if req.feature_ids is not None:
        out = [int(value) for value in req.feature_ids]
    else:
        out = []
        for key in req.feature_keys or []:
            value = meta.key_to_id.get(str(key))
            if value is None:
                raise HTTPException(status_code=404, detail=f"feature_key not found: {key}")
            out.append(int(value))
    if not out:
        raise HTTPException(status_code=400, detail="feature list must not be empty")
    bad = [value for value in out if value < 0 or value >= meta.row_count]
    if bad:
        raise HTTPException(status_code=404, detail=f"feature ids out of range: {bad}")
    return out


def _scalar_present(seed: int, sample_id: int, feature_id: int, missing_rate: float) -> bool:
    return _uniform(seed, "scalar-present", sample_id, feature_id) >= float(missing_rate)


def _scalar_value(req: ScalarValueRequest, sample_id: int, feature_id: int) -> float:
    group_id = int(feature_id) % int(req.n_latent_groups)
    latent = _normal(req.seed, "scalar-latent", group_id, sample_id)
    sign = -1.0 if _uniform(req.seed, "scalar-sign", feature_id, 0) < 0.25 else 1.0
    strength = 0.8 + 0.6 * _uniform(req.seed, "scalar-strength", feature_id, 0)
    noise = float(req.noise_scale) * _normal(req.seed, "scalar-noise", feature_id, sample_id)
    return float(sign * strength * latent + noise)


def _array_present(seed: int, sample_id: int, feature_id: int, missing_rate: float) -> bool:
    return _uniform(seed, "array-present", sample_id, feature_id) >= float(missing_rate)


def _array_columns(req: ArrayTraceRequest, sample_id: int, feature_id: int) -> dict[str, list[Any]]:
    if _uniform(req.seed, "array-empty", sample_id, feature_id) < float(req.empty_trace_rate):
        return {"time": [], "value": [], "ch_step": []}

    trace_span = int(req.max_trace_len) - int(req.min_trace_len) + 1
    trace_len = int(req.min_trace_len) + int(_uniform(req.seed, "array-len", sample_id, feature_id) * trace_span)
    trace_len = max(int(req.min_trace_len), min(int(req.max_trace_len), trace_len))
    if trace_len <= 0:
        return {"time": [], "value": [], "ch_step": []}

    group_id = int(feature_id) % int(req.n_latent_groups)
    freq = 0.8 + 2.2 * _uniform(req.seed, "array-freq", group_id, 0)
    phase = 2.0 * math.pi * _uniform(req.seed, "array-phase", group_id, sample_id)
    amp = 0.6 + 1.4 * _uniform(req.seed, "array-amp", feature_id, 0)
    sample_shift = 0.2 * _normal(req.seed, "array-shift", group_id, sample_id)
    labels = ("pre", "stim", "post")

    time_values: list[float] = []
    value_values: list[float] = []
    ch_steps: list[str] = []
    denom = max(trace_len - 1, 1)
    for point_idx in range(trace_len):
        t = point_idx / denom
        time_value = 10.0 * t
        signal = amp * math.sin(freq * time_value + phase)
        harmonic = 0.3 * math.cos(0.5 * freq * time_value - phase)
        noise = float(req.noise_scale) * _normal(req.seed, "array-noise", feature_id, sample_id * 100000 + point_idx)
        time_values.append(float(time_value))
        value_values.append(float(signal + harmonic + sample_shift + noise))
        ch_steps.append(labels[min(len(labels) - 1, int(point_idx * len(labels) / trace_len))])
    return {"time": time_values, "value": value_values, "ch_step": ch_steps}


def _uniform(seed: int, label: str, a: int, b: int) -> float:
    raw = _hash_u64(seed, label, a, b)
    return ((raw >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def _normal(seed: int, label: str, a: int, b: int) -> float:
    u1 = max(_uniform(seed, label + "-u1", a, b), 1e-12)
    u2 = _uniform(seed, label + "-u2", a, b)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _hash_u64(seed: int, label: str, a: int, b: int) -> int:
    payload = f"{int(seed)}|{label}|{int(a)}|{int(b)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic scalar/array value FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    target = "scripts.serve_synthetic_value_api:app" if bool(args.reload) else app
    uvicorn.run(target, host=args.host, port=int(args.port), reload=bool(args.reload))


if __name__ == "__main__":
    main()
