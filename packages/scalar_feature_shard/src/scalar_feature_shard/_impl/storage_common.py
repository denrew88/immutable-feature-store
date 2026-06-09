"""Shared helpers for dense-long scalar storage."""

from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np
import polars as pl

from .file_lock import FilePathLock


_JSON_REPLACE_RETRY_COUNT = 8
_JSON_LOCK_TIMEOUT_SECONDS = 30.0


def write_json_atomic(path: str, payload: dict):
    """Write JSON via a unique temporary file and atomic replace.

    Windows can briefly reject replace when an IDE, antivirus, or another reader
    has the target JSON open. A unique tmp path avoids writer-side tmp-name
    collisions, and the short retry handles those transient locks.
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    lock = FilePathLock(path + ".lock", timeout_seconds=_JSON_LOCK_TIMEOUT_SECONDS)
    lock.acquire()
    try:
        # final JSON을 직접 덮어쓰지 않고 UUID tmp에 먼저 씁니다.
        # 이렇게 해야 reader가 깨진 중간 JSON을 보는 일을 막을 수 있습니다.
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        last_error = None
        for attempt in range(_JSON_REPLACE_RETRY_COUNT):
            try:
                # Windows에서는 백신/IDE가 final JSON을 잠깐 열어 replace가 실패할 수
                # 있으므로 짧은 backoff로 재시도합니다.
                os.replace(tmp_path, path)
                return
            except OSError as exc:
                last_error = exc
                if attempt == _JSON_REPLACE_RETRY_COUNT - 1:
                    break
                time.sleep(0.025 * (attempt + 1))
        raise last_error
    finally:
        lock.release()
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def load_dense_metadata(
    meta_path: str,
    *,
    id_col: str,
    entity_name: str,
    key_col: str,
) -> pl.DataFrame:
    """Load dense metadata and validate row-order ids and optional keys."""

    df = pl.read_parquet(meta_path)
    dense_ids = np.arange(df.height, dtype=np.int64 if id_col == "sample_id" else np.int32)
    if id_col in df.columns:
        stored = df[id_col].to_numpy().astype(dense_ids.dtype, copy=False)
        if not np.array_equal(stored, dense_ids):
            raise ValueError(f"{entity_name} metadata {id_col} must equal dense row order 0..n-1")

    if key_col and key_col in df.columns:
        values = df[key_col].to_list()
        seen = set()
        for row_idx, value in enumerate(values):
            if value is None:
                raise ValueError(f"{entity_name} metadata {key_col} cannot be null at row {row_idx}")
            key = str(value)
            if key in seen:
                raise ValueError(f"duplicate {entity_name} {key_col}: {key}")
            seen.add(key)

    return df


def load_sample_targets(
    sample_meta_path: str,
    y_col: str = "y",
    sample_id_col: str = "sample_id",
):
    """Load dense sample ids, target values, and a non-null target mask."""

    df = pl.read_parquet(sample_meta_path)
    if y_col not in df.columns:
        raise ValueError(f"sample_meta parquet must have target column: {y_col}")
    sample_ids = np.arange(df.height, dtype=np.int64)
    if sample_id_col in df.columns:
        stored_ids = df[sample_id_col].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(stored_ids, sample_ids):
            raise ValueError(f"sample_meta {sample_id_col} must equal dense row order 0..n-1")
    y = df[y_col].to_numpy().astype(np.float64, copy=False)
    y_mask = ~np.isnan(y)
    return sample_ids.tolist(), y, y_mask


def load_feature_meta(
    feature_meta_path: str,
    feature_id_col: str = "feature_id",
):
    """Load feature metadata and validate dense row-order feature ids."""

    df = pl.read_parquet(feature_meta_path)
    feature_ids = np.arange(df.height, dtype=np.int32)
    if feature_id_col in df.columns:
        stored_ids = df[feature_id_col].to_numpy().astype(np.int32, copy=False)
        if not np.array_equal(stored_ids, feature_ids):
            raise ValueError(f"feature_meta {feature_id_col} must equal dense row order 0..n-1")
    return feature_ids, df


def close_memmap(mm):
    """Flush and close a NumPy memmap backing handle if it is still open."""

    if mm is None:
        return
    try:
        mm.flush()
    except Exception:
        pass
    backing = getattr(mm, "_mmap", None)
    if backing is not None:
        try:
            backing.close()
        except Exception:
            pass


def cleanup_backing_file(path: Optional[str]):
    """Delete one temporary memmap backing file if it exists."""

    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def cleanup_empty_dir(path: Optional[str]):
    """Remove one temporary directory when it exists and is empty."""

    if not path:
        return
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass


SAMPLE_MAJOR_MANIFEST_FORMAT = "scalar-sample-major-v1"


def load_sample_major_manifest(manifest_path: str):
    """Load a scalar sample-major manifest and resolve paths."""

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if str(data.get("format", "")) != SAMPLE_MAJOR_MANIFEST_FORMAT:
        raise ValueError(f"unsupported sample-major manifest format: {data.get('format')}")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    def resolve(value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(manifest_dir, value))

    return {
        "sample_meta_path": resolve(str(data["sample_meta_path"])),
        "feature_meta_path": resolve(str(data["feature_meta_path"])),
        "sample_paths": [resolve(str(value)) for value in list(data.get("sample_paths", []))],
        "sample_ids": data.get("sample_ids"),
        "sample_id_col": str(data.get("sample_id_col", "sample_id")),
        "feature_id_col": str(data.get("feature_id_col", "feature_id")),
        "value_col": str(data.get("value_col", "value")),
    }
