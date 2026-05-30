"""Resumable builder for array_sample_parquet v1."""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ..array.binary_storage import _load_dense_meta
from ..array.storage import _normalize_point_schema
from ..types import LogicalType, StorageType, point_storage_dtype
from .dictionaries import append_jsonl, load_jsonl, write_json_atomic
from .manifest import (
    FORMAT_NAME,
    ArraySampleParquetBuildOptions,
    ArraySampleParquetBuildSessionStatus,
    ArraySampleParquetManifest,
    ArraySampleParquetPart,
    write_array_sample_parquet_manifest,
)
from .parquet_io import StreamingTracePartWriter, normalize_array_sample_point_schema


class ArraySampleParquetSampleContext:
    """sample 하나의 trace들을 명시적으로 묶는 context.

    이 포맷의 checkpoint 단위는 part parquet이고, part는 항상 sample 경계에서만
    commit된다. 따라서 trace가 하나도 없는 sample도 `with builder.sample(...):`
    블록을 통과하면 완료된 sample로 기록된다.
    """

    def __init__(self, builder: "ArraySampleParquetDatasetBuilder", sample_id: int):
        self._builder = builder
        self._sample_id = int(sample_id)

    def __enter__(self):
        self._builder._begin_sample(self._sample_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._builder._end_sample(abort=exc_type is not None)
        return False

    def add_trace(self, feature_id: Optional[int] = None, feature_key: Optional[str] = None, *, columns):
        """현재 sample에 feature trace 하나를 추가한다."""

        self._builder.add_trace(
            sample_id=self._sample_id,
            feature_id=feature_id,
            feature_key=feature_key,
            columns=columns,
        )


class ArraySampleParquetDatasetBuilder:
    """sample-major Parquet array dataset을 resume 가능하게 생성한다.

    사용자는 sample 순서대로 데이터를 넣는다. builder는 sample 경계에서만 part를
    flush하므로, 중간 실패 후에는 `status().next_expected_sample_id`부터 다시
    외부 데이터를 수집해서 이어 넣으면 된다.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        point_schema,
        *,
        feature_meta_path: Optional[str] = None,
        feature_keys: Optional[Sequence[str]] = None,
        options: ArraySampleParquetBuildOptions | None = None,
    ):
        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.sample_meta_source_path = str(Path(sample_meta_path).expanduser().resolve())
        self.options = options or ArraySampleParquetBuildOptions()
        self.point_schema = normalize_array_sample_point_schema(_normalize_point_schema(point_schema))

        self.sample_parts_path = os.path.join(self.out_dir, "sample_parts")
        self.trace_index_parts_path = os.path.join(self.out_dir, "trace_index_parts")
        self.manifest_path = os.path.join(self.out_dir, "array_sample_parquet_manifest.json")
        self.state_path = os.path.join(self.out_dir, "state.json")
        self.parts_log_path = os.path.join(self.out_dir, "parts.jsonl")
        self.sample_meta_path = os.path.join(self.out_dir, "sample_meta.parquet")
        self.feature_meta_path = os.path.join(self.out_dir, "feature_meta.parquet")

        sample_meta = _load_dense_meta(
            self.sample_meta_source_path,
            "sample_id",
            "sample",
            str(self.options.sample_key_col),
        )
        self.n_samples = int(sample_meta.height)
        self._sample_keys = self._load_keys(sample_meta, str(self.options.sample_key_col), self.n_samples)
        self._sample_key_to_id = {
            key: int(idx) for idx, key in enumerate(self._sample_keys) if key is not None
        }

        self._feature_meta_source_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        self._known_feature_mode = bool(feature_meta_path or feature_keys is not None)
        self._writes_feature_meta = not bool(feature_meta_path)
        self._feature_key_to_id: OrderedDict[str, int] = OrderedDict()
        self._feature_keys_in_order: list[str] = []
        self._known_feature_count: Optional[int] = None

        self._part_writer: Optional[StreamingTracePartWriter] = None
        self._part_rows = 0
        self._part_bytes = 0
        self._pending_first_sample_id: Optional[int] = None
        self._pending_last_sample_id: Optional[int] = None
        self._pending_sample_count = 0
        self._pending_trace_count = 0
        self._committed_part_count = 0
        self._last_committed_sample_id: Optional[int] = None
        self._cursor_sample_id = 0
        self._open_sample_id: Optional[int] = None
        self._current_sample_trace_count = 0
        self._current_sample_traces: list[tuple[int, int, dict[str, np.ndarray]]] = []
        self._closed = False
        self._finished = False

        if os.path.exists(self.state_path):
            self._resume(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        else:
            self._initialize(feature_meta_path=feature_meta_path, feature_keys=feature_keys)

    @classmethod
    def open_session(cls, *args, **kwargs) -> "ArraySampleParquetDatasetBuilder":
        return cls(*args, **kwargs)

    @staticmethod
    def _load_keys(df: pl.DataFrame, key_col: str, count: int) -> list[Optional[str]]:
        if not key_col or key_col not in df.columns:
            return [None] * int(count)
        return [None if value is None else str(value) for value in df[key_col].to_list()]

    def _sample_key_for_id(self, sample_id: Optional[int]) -> Optional[str]:
        if sample_id is None or int(sample_id) < 0 or int(sample_id) >= int(self.n_samples):
            return None
        return self._sample_keys[int(sample_id)]

    def _part_path(self, part_id: int) -> str:
        return os.path.join(self.sample_parts_path, f"part_{int(part_id):06d}.parquet")

    def _trace_index_part_path(self, part_id: int) -> str:
        return os.path.join(self.trace_index_parts_path, f"part_{int(part_id):06d}.parquet")

    def _ensure_part_writer(self) -> StreamingTracePartWriter:
        if self._part_writer is None:
            part_id = int(self._committed_part_count)
            self._part_writer = StreamingTracePartWriter(
                self._part_path(part_id),
                trace_index_path=self._trace_index_part_path(part_id),
                point_schema=self.point_schema,
                compression=str(self.options.compression),
            )
        return self._part_writer

    def _options_payload(self) -> dict:
        return asdict(self.options)

    def _point_schema_payload(self) -> list[dict]:
        return [spec.to_json() for spec in self.point_schema]

    def _state_payload(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "state_version": self._STATE_VERSION,
            "sample_meta_source_path": self.sample_meta_source_path,
            "feature_meta_source_path": self._feature_meta_source_path,
            "options": self._options_payload(),
            "point_schema": self._point_schema_payload(),
            "known_feature_mode": bool(self._known_feature_mode),
            "writes_feature_meta": bool(self._writes_feature_meta),
            "feature_keys_in_order": list(self._feature_keys_in_order),
            "known_feature_count": self._known_feature_count,
            "committed_part_count": int(self._committed_part_count),
            "last_committed_sample_id": self._last_committed_sample_id,
            "last_committed_sample_key": self._sample_key_for_id(self._last_committed_sample_id),
            "next_expected_sample_id": self._resume_next_sample_id(),
            "next_expected_sample_key": self._sample_key_for_id(self._resume_next_sample_id()),
            "finished": bool(self._finished),
            "manifest_path": self.manifest_path if self._finished else None,
        }

    def _save_state(self):
        write_json_atomic(self.state_path, self._state_payload())

    def _cleanup_tmp_files(self):
        for root in [self.out_dir, self.sample_parts_path, self.trace_index_parts_path]:
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                if name.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(root, name))
                    except FileNotFoundError:
                        pass

    def _initialize(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if os.path.exists(self.out_dir) and os.listdir(self.out_dir):
            raise ValueError(f"out_dir already exists and is not empty: {self.out_dir}")
        os.makedirs(self.sample_parts_path, exist_ok=True)
        os.makedirs(self.trace_index_parts_path, exist_ok=True)
        shutil.copy2(self.sample_meta_source_path, self.sample_meta_path)
        self._initialize_feature_meta(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        self._save_state()

    def _initialize_feature_meta(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")
        if feature_meta_path:
            feature_meta = _load_dense_meta(
                self._feature_meta_source_path,
                "feature_id",
                "feature",
                str(self.options.feature_key_col),
            )
            shutil.copy2(self._feature_meta_source_path, self.feature_meta_path)
            self._known_feature_count = int(feature_meta.height)
            self._load_feature_keys_from_meta(feature_meta)
            return
        if feature_keys is not None:
            for idx, feature_key in enumerate(feature_keys):
                key = str(feature_key)
                if key in self._feature_key_to_id:
                    raise ValueError(f"duplicate feature key: {key}")
                self._feature_key_to_id[key] = int(idx)
                self._feature_keys_in_order.append(key)
            self._known_feature_count = int(len(self._feature_keys_in_order))

    def _load_feature_keys_from_meta(self, feature_meta: pl.DataFrame):
        key_col = str(self.options.feature_key_col)
        if key_col and key_col in feature_meta.columns:
            for idx, value in enumerate(feature_meta[key_col].to_list()):
                key = str(value)
                self._feature_key_to_id[key] = int(idx)
                self._feature_keys_in_order.append(key)

    def _resume(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        self._cleanup_tmp_files()
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._validate_resume_state(state, feature_meta_path=feature_meta_path, feature_keys=feature_keys)

        records = load_jsonl(self.parts_log_path)
        self._committed_part_count = len(records)
        self._last_committed_sample_id = None if not records else int(records[-1]["last_sample_id"])
        self._cursor_sample_id = self._resume_next_sample_id()
        self._finished = bool(state.get("finished"))

        self._known_feature_mode = bool(state.get("known_feature_mode"))
        self._writes_feature_meta = bool(state.get("writes_feature_meta"))
        self._known_feature_count = state.get("known_feature_count")
        self._feature_keys_in_order = [str(value) for value in state.get("feature_keys_in_order") or []]
        self._feature_key_to_id = OrderedDict(
            (feature_key, int(idx)) for idx, feature_key in enumerate(self._feature_keys_in_order)
        )

    def _validate_resume_state(self, state: dict, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if state.get("format") != FORMAT_NAME:
            raise ValueError(f"unsupported build session format: {state.get('format')!r}")
        if str(Path(state["sample_meta_source_path"]).expanduser().resolve()) != self.sample_meta_source_path:
            raise ValueError("sample_meta_path does not match existing build session")
        if state.get("options") != self._options_payload():
            raise ValueError("options do not match existing build session")
        if state.get("point_schema") != self._point_schema_payload():
            raise ValueError("point_schema does not match existing build session")
        normalized_feature_meta_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        stored_feature_meta_path = str(state.get("feature_meta_source_path") or "")
        if normalized_feature_meta_path and stored_feature_meta_path and normalized_feature_meta_path != stored_feature_meta_path:
            raise ValueError("feature_meta_path does not match existing build session")
        if feature_keys is not None and [str(value) for value in feature_keys] != list(state.get("feature_keys_in_order") or []):
            raise ValueError("feature_keys do not match existing build session")

    def _resume_next_sample_id(self) -> int:
        return 0 if self._last_committed_sample_id is None else int(self._last_committed_sample_id) + 1

    def _ensure_open_for_writes(self):
        if self._closed:
            raise RuntimeError("array sample parquet builder is closed")
        if self._finished:
            raise RuntimeError("array sample parquet builder is already finished")

    def _resolve_sample_id(self, sample_id: Optional[int], sample_key: Optional[str]) -> int:
        if sample_id is None and sample_key is None:
            raise ValueError("provide either sample_id or sample_key")
        if sample_id is not None and sample_key is not None:
            resolved = self._resolve_sample_id(None, sample_key)
            if int(sample_id) != int(resolved):
                raise ValueError(f"sample_id/sample_key mismatch: {sample_id} != {sample_key}")
            return int(sample_id)
        if sample_id is not None:
            sample_id = int(sample_id)
            if sample_id < 0 or sample_id >= int(self.n_samples):
                raise ValueError(f"sample_id out of range: {sample_id}")
            return sample_id
        key = str(sample_key)
        resolved = self._sample_key_to_id.get(key)
        if resolved is None:
            raise ValueError(f"unknown sample key: {key}")
        return int(resolved)

    def _resolve_feature_id(self, feature_id: Optional[int], feature_key: Optional[str]) -> int:
        if feature_id is None and feature_key is None:
            raise ValueError("provide either feature_id or feature_key")
        if feature_id is not None and feature_key is not None:
            resolved = self._resolve_feature_id(None, feature_key)
            if int(feature_id) != int(resolved):
                raise ValueError(f"feature_id/feature_key mismatch: {feature_id} != {feature_key}")
            return int(feature_id)
        if feature_id is not None:
            feature_id = int(feature_id)
            if feature_id < 0:
                raise ValueError("feature_id must be >= 0")
            if not self._known_feature_mode:
                raise ValueError("feature_id inputs require feature_meta_path or feature_keys")
            if self._known_feature_count is not None and feature_id >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {feature_id}")
            return feature_id

        key = str(feature_key)
        resolved = self._feature_key_to_id.get(key)
        if resolved is not None:
            return int(resolved)
        if self._known_feature_mode:
            raise ValueError(f"unknown feature key: {key}")
        resolved = len(self._feature_keys_in_order)
        self._feature_key_to_id[key] = int(resolved)
        self._feature_keys_in_order.append(key)
        return int(resolved)

    def _normalize_columns(self, columns) -> dict[str, np.ndarray]:
        if columns is None:
            raise ValueError("columns is required")
        column_map = {str(name): value for name, value in dict(columns).items()}
        expected_names = {spec.name for spec in self.point_schema}
        actual_names = set(column_map.keys())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"point columns must exactly match point_schema ({', '.join(details)})")

        out: dict[str, np.ndarray] = {}
        trace_len = None
        for spec in self.point_schema:
            values = column_map[spec.name]
            if spec.logical_type == LogicalType.CATEGORICAL:
                arr = _normalize_categorical_values(values, spec.name)
            elif spec.logical_type == LogicalType.TIMESTAMP_NS:
                arr = np.asarray(values, dtype="datetime64[ns]").reshape(-1).astype(point_storage_dtype(spec.storage_type), copy=False)
            elif spec.logical_type == LogicalType.TIMEDELTA_NS:
                arr = np.asarray(values, dtype="timedelta64[ns]").reshape(-1).astype(point_storage_dtype(spec.storage_type), copy=False)
            else:
                arr = np.asarray(values, dtype=point_storage_dtype(spec.storage_type)).reshape(-1)
            if trace_len is None:
                trace_len = int(arr.size)
            elif int(arr.size) != int(trace_len):
                raise ValueError(f"point column length mismatch for {spec.name}: expected={trace_len} got={arr.size}")
            out[spec.name] = arr
        return out

    def _begin_sample(self, sample_id: int):
        self._ensure_open_for_writes()
        sample_id = int(sample_id)
        if self._open_sample_id is not None:
            if int(self._open_sample_id) == sample_id:
                return
            raise RuntimeError("another sample is already open")
        if sample_id != int(self._cursor_sample_id):
            raise ValueError(
                f"array_sample_parquet session expects sample_id {self._cursor_sample_id}; got {sample_id}. "
                "Resume from status().next_expected_sample_id and process samples sequentially."
            )
        self._open_sample_id = sample_id
        self._current_sample_trace_count = 0
        self._current_sample_traces = []

    def _end_sample(self, *, abort: bool):
        if self._open_sample_id is None:
            return
        sample_id = int(self._open_sample_id)
        traces = list(self._current_sample_traces)
        self._open_sample_id = None
        self._current_sample_trace_count = 0
        self._current_sample_traces = []
        if abort:
            # current sample은 아직 part writer에 쓰기 전이므로 버퍼만 버리면 된다.
            return
        try:
            trace_count = self._write_sample_traces_sorted(sample_id, traces)
        except Exception:
            # sample 기록 도중 실패하면 일부 row만 part writer에 들어갔을 수 있다.
            # parquet row 단위 rollback은 불가능하므로 현재 미commit part를 버린다.
            self._discard_part_buffer()
            raise
        if self._pending_first_sample_id is None:
            self._pending_first_sample_id = sample_id
        self._pending_last_sample_id = sample_id
        self._pending_sample_count += 1
        self._pending_trace_count += trace_count
        self._cursor_sample_id = sample_id + 1
        if self._should_flush_part():
            self.flush_part()

    def _write_sample_traces_sorted(self, sample_id: int, traces: list[tuple[int, int, dict[str, np.ndarray]]]) -> int:
        """sample 내부 trace를 feature_id 순서로 정렬해 part에 기록한다.

        builder는 sample_id를 순차 처리하도록 강제한다. 따라서 sample 하나 안에서
        feature_id만 정렬하면 최종 point parquet의 물리 순서는
        `(sample_id, feature_id, point_idx)`가 된다.
        """

        if not traces:
            return 0
        ordered = traces
        if any(int(ordered[idx - 1][0]) > int(ordered[idx][0]) for idx in range(1, len(ordered))):
            ordered = sorted(traces, key=lambda item: int(item[0]))
        for feature_id, trace_len, normalized in ordered:
            self._ensure_part_writer().write_row(
                sample_id=int(sample_id),
                feature_id=int(feature_id),
                trace_len=int(trace_len),
                columns=normalized,
            )
            self._part_rows += int(trace_len)
            self._part_bytes += self._estimate_row_bytes(int(trace_len))
        return len(ordered)

    def sample(self, sample_id: Optional[int] = None, sample_key: Optional[str] = None) -> ArraySampleParquetSampleContext:
        return ArraySampleParquetSampleContext(self, self._resolve_sample_id(sample_id, sample_key))

    def add_trace(
        self,
        *,
        sample_id: Optional[int] = None,
        sample_key: Optional[str] = None,
        feature_id: Optional[int] = None,
        feature_key: Optional[str] = None,
        columns,
    ):
        """현재 sample에 feature trace 하나를 추가한다."""

        self._ensure_open_for_writes()
        resolved_sample_id = self._resolve_sample_id(sample_id, sample_key)
        if self._open_sample_id is None:
            self._begin_sample(resolved_sample_id)
        elif int(self._open_sample_id) != int(resolved_sample_id):
            raise RuntimeError("sample boundary crossed without closing previous sample")
        resolved_feature_id = self._resolve_feature_id(feature_id, feature_key)
        normalized = self._normalize_columns(columns)
        trace_len = int(next(iter(normalized.values())).shape[0]) if normalized else 0

        self._current_sample_traces.append((int(resolved_feature_id), trace_len, normalized))
        self._current_sample_trace_count += 1

    def _estimate_row_bytes(self, trace_len: int) -> int:
        # Parquet 내부 인코딩까지 정확히 맞추려는 값이 아니라, 자동 flush가 너무
        # 늦어지지 않도록 trace payload 크기를 근사하는 제어용 추정치다.
        fixed = 8 + 4 + 4
        return int(fixed + sum(int(trace_len) * _estimated_value_bytes(spec) for spec in self.point_schema))

    def _should_flush_part(self) -> bool:
        if self._pending_sample_count <= 0:
            return False
        if int(self.options.max_part_samples) > 0 and self._pending_sample_count >= int(self.options.max_part_samples):
            return True
        return bool(
            self._part_rows >= int(self.options.max_part_rows)
            or self._part_bytes >= int(self.options.target_part_bytes)
        )

    def flush_part(self):
        """현재까지 완료된 sample들을 part parquet 하나로 commit한다."""

        if self._pending_sample_count <= 0:
            return None
        part_id = int(self._committed_part_count)
        part_path = self._part_path(part_id)
        trace_index_path = self._trace_index_part_path(part_id)
        self._ensure_part_writer().commit()
        self._part_writer = None
        byte_size = os.path.getsize(part_path)
        trace_index_byte_size = os.path.getsize(trace_index_path)
        record = {
            "part_id": part_id,
            "path": os.path.relpath(part_path, self.out_dir).replace("\\", "/"),
            "trace_index_path": os.path.relpath(trace_index_path, self.out_dir).replace("\\", "/"),
            "first_sample_id": int(self._pending_first_sample_id),
            "last_sample_id": int(self._pending_last_sample_id),
            "first_sample_key": self._sample_key_for_id(self._pending_first_sample_id),
            "last_sample_key": self._sample_key_for_id(self._pending_last_sample_id),
            "sample_count": int(self._pending_sample_count),
            "trace_count": int(self._pending_trace_count),
            "row_count": int(self._part_rows),
            "byte_size": int(byte_size),
            "trace_index_byte_size": int(trace_index_byte_size),
        }
        append_jsonl(self.parts_log_path, record)
        self._committed_part_count += 1
        self._last_committed_sample_id = self._pending_last_sample_id
        self._reset_part_buffer()
        self._save_state()
        return record

    def _reset_part_buffer(self):
        self._part_writer = None
        self._part_rows = 0
        self._part_bytes = 0
        self._pending_first_sample_id = None
        self._pending_last_sample_id = None
        self._pending_sample_count = 0
        self._pending_trace_count = 0

    def _discard_part_buffer(self):
        if self._part_writer is not None:
            self._part_writer.abort()
        self._reset_part_buffer()
        self._cursor_sample_id = self._resume_next_sample_id()

    def _write_feature_meta(self):
        if not self._writes_feature_meta:
            return
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {"feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32)}
        key_col = str(self.options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self.feature_meta_path)
        self._known_feature_count = int(len(self._feature_keys_in_order))

    def finish(self) -> str:
        """남은 part를 commit하고 최종 manifest를 쓴다."""

        if self._finished:
            return self.manifest_path
        self._ensure_open_for_writes()
        self._end_sample(abort=False)
        self.flush_part()
        self._write_feature_meta()
        parts = [
            ArraySampleParquetPart(
                part_id=int(item["part_id"]),
                path=os.path.join(self.out_dir, item["path"]),
                trace_index_path=os.path.join(self.out_dir, item["trace_index_path"]),
                first_sample_id=int(item["first_sample_id"]),
                last_sample_id=int(item["last_sample_id"]),
                sample_count=int(item["sample_count"]),
                trace_count=int(item["trace_count"]),
                row_count=int(item["row_count"]),
                byte_size=int(item["byte_size"]),
                trace_index_byte_size=int(item.get("trace_index_byte_size", 0)),
            )
            for item in load_jsonl(self.parts_log_path)
        ]
        manifest = ArraySampleParquetManifest(
            sample_meta_path=self.sample_meta_path,
            feature_meta_path=self.feature_meta_path,
            n_samples=int(self.n_samples),
            n_features=int(self._known_feature_count or len(self._feature_keys_in_order)),
            sample_parts_path=self.sample_parts_path,
            trace_index_parts_path=self.trace_index_parts_path,
            parts=parts,
            point_schema=self.point_schema,
            sample_key_col=str(self.options.sample_key_col),
            feature_key_col=str(self.options.feature_key_col),
        )
        write_array_sample_parquet_manifest(self.manifest_path, manifest)
        self._finished = True
        self._save_state()
        return self.manifest_path

    def status(self) -> ArraySampleParquetBuildSessionStatus:
        next_expected = self._resume_next_sample_id()
        return ArraySampleParquetBuildSessionStatus(
            last_committed_sample_id=self._last_committed_sample_id,
            last_committed_sample_key=self._sample_key_for_id(self._last_committed_sample_id),
            next_expected_sample_id=next_expected,
            next_expected_sample_key=self._sample_key_for_id(next_expected),
            committed_part_count=int(self._committed_part_count),
            finished=bool(self._finished),
            manifest_path=self.manifest_path if self._finished else None,
            buffered_through_sample_id=self._pending_last_sample_id,
            buffered_through_sample_key=self._sample_key_for_id(self._pending_last_sample_id),
            in_progress_sample_id=self._open_sample_id,
            in_progress_sample_key=self._sample_key_for_id(self._open_sample_id),
        )

    def close(self):
        """현재 sample/part 경계까지 checkpoint를 남기고 닫는다."""

        if self._closed:
            return
        if not self._finished:
            self._end_sample(abort=False)
            self.flush_part()
            self._save_state()
        self._closed = True

    def __enter__(self):
        self._ensure_open_for_writes()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def build_array_sample_parquet_dataset(*args, **kwargs) -> str:
    """편의 함수: session을 열고 caller가 직접 채운 뒤 finish해야 하는 경우보다 단순한 wrapper."""

    with ArraySampleParquetDatasetBuilder(*args, **kwargs) as builder:
        return builder.finish()


def _normalize_categorical_values(values, column_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if bool(np.any(arr == None)):
        raise ValueError(f"categorical point column {column_name} does not support null values")
    return arr


def _estimated_value_bytes(spec) -> int:
    if spec.storage_type == StorageType.STRING:
        return 16
    return int(point_storage_dtype(spec.storage_type).itemsize)
