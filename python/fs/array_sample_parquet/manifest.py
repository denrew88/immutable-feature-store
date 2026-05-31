"""Manifest models for array_sample_parquet v1."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..types import PointColumnSpec
from ..array.storage import _normalize_point_schema


FORMAT_NAME = "array-sample-parquet"
FORMAT_VERSION = 1
DEFAULT_SAMPLE_KEY_COL = "sample_key"
DEFAULT_FEATURE_KEY_COL = "feature_key"


@dataclass
class ArraySampleParquetBuildOptions:
    """sample-major Parquet dataset 생성 옵션.

    part 크기는 sample 개수가 아니라 추정 payload byte 기준으로 결정한다.
    다만 commit은 sample 경계에서만 수행하므로, 재시작 시 사용자는
    `status().next_expected_sample_id`부터 다시 데이터를 수집해서 넣으면 된다.
    """

    target_part_bytes: int = 128 * 1024 * 1024
    max_part_rows: int = 10_000_000
    max_part_samples: int = 0
    compression: str = "zstd"
    sample_key_col: str = DEFAULT_SAMPLE_KEY_COL
    feature_key_col: str = DEFAULT_FEATURE_KEY_COL


@dataclass(frozen=True)
class ArraySampleParquetBuildSessionStatus:
    """resume 가능한 build session의 현재 checkpoint 상태."""

    last_committed_sample_id: Optional[int]
    last_committed_sample_key: Optional[str]
    next_expected_sample_id: int
    next_expected_sample_key: Optional[str]
    committed_part_count: int
    finished: bool
    manifest_path: Optional[str]
    buffered_through_sample_id: Optional[int]
    buffered_through_sample_key: Optional[str]
    in_progress_sample_id: Optional[int]
    in_progress_sample_key: Optional[str]


@dataclass
class ArraySampleParquetPart:
    part_id: int
    path: str
    trace_index_path: str
    first_sample_id: int
    last_sample_id: int
    sample_count: int
    trace_count: int
    row_count: int
    byte_size: int
    trace_index_byte_size: int = 0

    def to_json(self) -> dict:
        return {
            "part_id": int(self.part_id),
            "path": str(self.path),
            "trace_index_path": str(self.trace_index_path),
            "first_sample_id": int(self.first_sample_id),
            "last_sample_id": int(self.last_sample_id),
            "sample_count": int(self.sample_count),
            "trace_count": int(self.trace_count),
            "row_count": int(self.row_count),
            "byte_size": int(self.byte_size),
            "trace_index_byte_size": int(self.trace_index_byte_size),
        }


@dataclass
class ArraySampleParquetManifest:
    sample_meta_path: str
    feature_meta_path: str
    n_samples: int
    n_features: int
    sample_parts_path: str
    trace_index_parts_path: str
    parts: list[ArraySampleParquetPart]
    point_schema: list[PointColumnSpec]
    sample_key_col: str = DEFAULT_SAMPLE_KEY_COL
    feature_key_col: str = DEFAULT_FEATURE_KEY_COL
    id_scheme: str = "dense_row_ids"
    version: int = FORMAT_VERSION

    def to_json(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": int(self.version),
            "id_scheme": self.id_scheme,
            "sample_meta_path": str(self.sample_meta_path),
            "feature_meta_path": str(self.feature_meta_path),
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "sample_parts_path": str(self.sample_parts_path),
            "trace_index_parts_path": str(self.trace_index_parts_path),
            "sample_key_col": str(self.sample_key_col),
            "feature_key_col": str(self.feature_key_col),
            "point_schema": [spec.to_json() for spec in self.point_schema],
            "parts": [part.to_json() for part in self.parts],
        }


def _relative_to(manifest_path: str, target_path: str) -> str:
    if not target_path:
        return ""
    target = Path(target_path)
    if not target.is_absolute():
        return str(target).replace("\\", "/")
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    return os.path.relpath(str(target.resolve()), str(manifest_dir)).replace("\\", "/")


def _resolve_against(manifest_path: str, stored_path: str) -> str:
    if not stored_path:
        return ""
    stored = Path(stored_path)
    if stored.is_absolute():
        return str(stored.resolve())
    return str((Path(manifest_path).expanduser().resolve().parent / stored).resolve())


def write_array_sample_parquet_manifest(path: str, manifest: ArraySampleParquetManifest):
    """manifest를 artifact-relative path로 정리해서 저장한다."""

    payload = manifest.to_json()
    payload["sample_meta_path"] = _relative_to(path, manifest.sample_meta_path)
    payload["feature_meta_path"] = _relative_to(path, manifest.feature_meta_path)
    payload["sample_parts_path"] = _relative_to(path, manifest.sample_parts_path)
    payload["trace_index_parts_path"] = _relative_to(path, manifest.trace_index_parts_path)
    for part in payload["parts"]:
        part["path"] = _relative_to(path, part["path"])
        part["trace_index_path"] = _relative_to(path, part["trace_index_path"])
    point_schema = []
    for spec in manifest.point_schema:
        point_schema.append(spec.to_json())
    payload["point_schema"] = point_schema
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_array_sample_parquet_manifest(path: str) -> ArraySampleParquetManifest:
    """array_sample_parquet manifest JSON을 읽고 모든 path를 절대경로로 해석한다."""

    manifest_path = str(Path(path).expanduser().resolve())
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported array sample parquet format: {data.get('format')!r}")
    version = int(data.get("version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported array sample parquet version: {version}")
    point_schema_payload = []
    for item in data.get("point_schema") or []:
        point_schema_payload.append(dict(item))
    return ArraySampleParquetManifest(
        sample_meta_path=_resolve_against(manifest_path, data["sample_meta_path"]),
        feature_meta_path=_resolve_against(manifest_path, data["feature_meta_path"]),
        n_samples=int(data["n_samples"]),
        n_features=int(data["n_features"]),
        sample_parts_path=_resolve_against(manifest_path, data["sample_parts_path"]),
        trace_index_parts_path=_resolve_against(manifest_path, data["trace_index_parts_path"]),
        parts=[
            ArraySampleParquetPart(
                part_id=int(item["part_id"]),
                path=_resolve_against(manifest_path, item["path"]),
                trace_index_path=_resolve_against(manifest_path, item["trace_index_path"]),
                first_sample_id=int(item["first_sample_id"]),
                last_sample_id=int(item["last_sample_id"]),
                sample_count=int(item["sample_count"]),
                trace_count=int(item["trace_count"]),
                row_count=int(item["row_count"]),
                byte_size=int(item["byte_size"]),
                trace_index_byte_size=int(item.get("trace_index_byte_size", 0)),
            )
            for item in data.get("parts") or []
        ],
        point_schema=_normalize_point_schema(point_schema_payload),
        sample_key_col=str(data.get("sample_key_col", DEFAULT_SAMPLE_KEY_COL)),
        feature_key_col=str(data.get("feature_key_col", DEFAULT_FEATURE_KEY_COL)),
        id_scheme=str(data.get("id_scheme", "dense_row_ids")),
        version=version,
    )
