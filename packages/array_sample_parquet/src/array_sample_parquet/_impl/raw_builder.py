"""Resume-safe builder for array_sample_parquet v1.

표준 builder는 sample별 raw parquet를 먼저 commit하고, 마지막 compact 단계에서
최종 `sample_parts/`와 `trace_index_parts/`를 만듭니다.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..types import LogicalType, PointColumnSpec, StorageType, point_storage_dtype
from .dictionaries import append_jsonl, load_jsonl, write_json_atomic
from .file_lock import FilePathLock as _FileLock
from .manifest import (
    FORMAT_NAME,
    ArraySampleParquetBuildOptions,
    ArraySampleParquetManifest,
    ArraySampleParquetPart,
    write_array_sample_parquet_manifest,
)
from .parquet_io import arrow_value_type, empty_point_columns, normalize_array_sample_point_schema, replace_file_with_retry
from .support import _load_dense_meta, _normalize_point_schema


RAW_STATE_VERSION = 1
RAW_SAMPLE_PADDING = 12
RAW_WRITER_BATCH_TRACES = 1024
RAW_WRITER_BATCH_POINTS = 262_144


@dataclass(frozen=True)
class _CompactPartPlan:
    raw_paths: list[str]
    raw_trace_index_paths: list[str]
    row_count: int
    trace_count: int
    byte_size: int
    sample_count: int
    first_sample_id: int
    last_sample_id: int


@dataclass(frozen=True)
class ArraySampleParquetBuildSessionStatus:
    """raw-sample build session의 완료/미완료 sample 현황."""

    n_samples: int
    completed_sample_count: int
    pending_sample_count: int
    completed_sample_ids: list[int]
    pending_sample_ids: list[int]
    finished: bool
    manifest_path: Optional[str]


class ArraySampleParquetSampleContext:
    """sample 하나를 raw parquet 파일 하나로 쓰는 context."""

    def __init__(self, builder: "ArraySampleParquetDatasetBuilder", sample_id: int, *, skip_if_completed: bool):
        self._builder = builder
        self._sample_id = int(sample_id)
        self._skip_if_completed = bool(skip_if_completed)
        self.skipped = False

    def __enter__(self):
        self.skipped = self._builder._begin_sample(
            self._sample_id,
            skip_if_completed=self._skip_if_completed,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self._builder._end_sample(abort=exc_type is not None)
        return False

    def add_trace(self, feature_id: Optional[int] = None, feature_key: Optional[str] = None, *, columns):
        if self.skipped:
            return
        self._builder.add_trace(
            sample_id=self._sample_id,
            feature_id=feature_id,
            feature_key=feature_key,
            columns=columns,
        )


class ArraySampleParquetDatasetBuilder:
    """sample별 raw parquet을 먼저 만들고, 나중에 최종 part로 compact하는 builder.

    이 builder는 기존 순차 builder와 다르게 sample_id 순서를 강제하지 않는다. worker는
    각자 맡은 sample 파일만 `raw_samples/sample_*.parquet.tmp`에 쓰고, 정상 종료 시
    `.parquet`으로 rename한 뒤 `raw_samples.jsonl`에 commit record를 남긴다.

    categorical column은 raw와 final 모두 문자열로 저장한다. Parquet 자체 dictionary
    encoding에 맡기므로 별도 sidecar dictionary나 code mapping을 만들지 않는다.
    """

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
        if not feature_meta_path and feature_keys is None:
            raise ValueError("raw sample builder requires feature_meta_path or feature_keys")

        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.sample_meta_source_path = str(Path(sample_meta_path).expanduser().resolve())
        self.options = options or ArraySampleParquetBuildOptions()
        self.point_schema = normalize_array_sample_point_schema(_normalize_point_schema(point_schema))

        self.raw_samples_path = os.path.join(self.out_dir, "raw_samples")
        self.raw_trace_index_path = os.path.join(self.out_dir, "raw_trace_index")
        self.sample_parts_path = os.path.join(self.out_dir, "sample_parts")
        self.trace_index_parts_path = os.path.join(self.out_dir, "trace_index_parts")
        self.manifest_path = os.path.join(self.out_dir, "array_sample_parquet_manifest.json")
        self.state_path = os.path.join(self.out_dir, "raw_state.json")
        self.raw_log_path = os.path.join(self.out_dir, "raw_samples.jsonl")
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
        self._feature_key_to_id: OrderedDict[str, int] = OrderedDict()
        self._feature_keys_in_order: list[str] = []
        self._known_feature_count: Optional[int] = None
        self._finished = False

        self._open_sample_id: Optional[int] = None
        self._open_writer: Optional[_RawSampleFileWriter] = None
        self._open_lock: Optional[_FileLock] = None
        self._open_trace_count = 0
        self._open_point_count = 0

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

    def _state_payload(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "raw_state_version": RAW_STATE_VERSION,
            "sample_meta_source_path": self.sample_meta_source_path,
            "feature_meta_source_path": self._feature_meta_source_path,
            "options": asdict(self.options),
            "point_schema": [spec.to_json() for spec in self.point_schema],
            "feature_keys_in_order": list(self._feature_keys_in_order),
            "known_feature_count": self._known_feature_count,
            "finished": bool(self._finished),
            "manifest_path": self.manifest_path if self._finished else None,
        }

    def _save_state(self):
        write_json_atomic(self.state_path, self._state_payload())

    def _initialize(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if os.path.exists(self.out_dir) and os.listdir(self.out_dir):
            raise ValueError(f"out_dir already exists and is not empty: {self.out_dir}")
        os.makedirs(self.raw_samples_path, exist_ok=True)
        os.makedirs(self.raw_trace_index_path, exist_ok=True)
        os.makedirs(self.sample_parts_path, exist_ok=True)
        os.makedirs(self.trace_index_parts_path, exist_ok=True)
        shutil.copy2(self.sample_meta_source_path, self.sample_meta_path)
        self._initialize_feature_meta(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        self._cleanup_tmp_files()
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

        for idx, feature_key in enumerate(feature_keys or []):
            key = str(feature_key)
            if key in self._feature_key_to_id:
                raise ValueError(f"duplicate feature key: {key}")
            self._feature_key_to_id[key] = int(idx)
            self._feature_keys_in_order.append(key)
        self._known_feature_count = int(len(self._feature_keys_in_order))
        _write_feature_meta(
            [{"feature_key": key} for key in self._feature_keys_in_order],
            self.feature_meta_path,
            feature_key_col=str(self.options.feature_key_col),
        )

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
        self._finished = bool(state.get("finished"))
        self._known_feature_count = state.get("known_feature_count")
        self._feature_keys_in_order = [str(value) for value in state.get("feature_keys_in_order") or []]
        self._feature_key_to_id = OrderedDict(
            (feature_key, int(idx)) for idx, feature_key in enumerate(self._feature_keys_in_order)
        )

    def _validate_resume_state(self, state: dict, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if state.get("format") != FORMAT_NAME:
            raise ValueError(f"unsupported raw build session format: {state.get('format')!r}")
        if int(state.get("raw_state_version", 0)) != RAW_STATE_VERSION:
            raise ValueError(f"unsupported raw build session version: {state.get('raw_state_version')}")
        if str(Path(state["sample_meta_source_path"]).expanduser().resolve()) != self.sample_meta_source_path:
            raise ValueError("sample_meta_path does not match existing raw build session")
        if state.get("options") != asdict(self.options):
            raise ValueError("options do not match existing raw build session")
        if state.get("point_schema") != [spec.to_json() for spec in self.point_schema]:
            raise ValueError("point_schema does not match existing raw build session")
        normalized_feature_meta_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        stored_feature_meta_path = str(state.get("feature_meta_source_path") or "")
        if normalized_feature_meta_path and stored_feature_meta_path and normalized_feature_meta_path != stored_feature_meta_path:
            raise ValueError("feature_meta_path does not match existing raw build session")
        if feature_keys is not None and [str(value) for value in feature_keys] != list(state.get("feature_keys_in_order") or []):
            raise ValueError("feature_keys do not match existing raw build session")

    def _cleanup_tmp_files(self):
        for root in [self.raw_samples_path, self.raw_trace_index_path, self.sample_parts_path, self.trace_index_parts_path]:
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                if name.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(root, name))
                    except FileNotFoundError:
                        pass

    def _raw_sample_path(self, sample_id: int) -> str:
        return os.path.join(self.raw_samples_path, f"sample_{int(sample_id):0{RAW_SAMPLE_PADDING}d}.parquet")

    def _raw_sample_rel_path(self, sample_id: int) -> str:
        return os.path.relpath(self._raw_sample_path(sample_id), self.out_dir).replace("\\", "/")

    def _raw_trace_index_path(self, sample_id: int) -> str:
        return os.path.join(self.raw_trace_index_path, f"sample_{int(sample_id):0{RAW_SAMPLE_PADDING}d}.parquet")

    def _raw_trace_index_rel_path(self, sample_id: int) -> str:
        return os.path.relpath(self._raw_trace_index_path(sample_id), self.out_dir).replace("\\", "/")

    def _part_path(self, part_id: int) -> str:
        return os.path.join(self.sample_parts_path, f"part_{int(part_id):06d}.parquet")

    def _trace_index_part_path(self, part_id: int) -> str:
        return os.path.join(self.trace_index_parts_path, f"part_{int(part_id):06d}.parquet")

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
            if self._known_feature_count is not None and feature_id >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {feature_id}")
            return feature_id
        key = str(feature_key)
        resolved = self._feature_key_to_id.get(key)
        if resolved is None:
            raise ValueError(f"unknown feature key: {key}")
        return int(resolved)

    def _normalize_columns_for_raw(self, columns) -> dict[str, np.ndarray]:
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

    def sample(
        self,
        sample_id: Optional[int] = None,
        sample_key: Optional[str] = None,
        *,
        skip_if_completed: bool = False,
    ) -> ArraySampleParquetSampleContext:
        return ArraySampleParquetSampleContext(
            self,
            self._resolve_sample_id(sample_id, sample_key),
            skip_if_completed=skip_if_completed,
        )

    def _begin_sample(self, sample_id: int, *, skip_if_completed: bool) -> bool:
        if self._finished:
            raise RuntimeError("raw sample builder is already compacted")
        if self._open_sample_id is not None:
            raise RuntimeError("another raw sample is already open in this builder")
        if self.is_sample_completed(sample_id):
            if skip_if_completed:
                return True
            raise ValueError(f"sample already completed: {sample_id}")

        # sample 하나는 point parquet와 trace-index parquet 두 파일로 commit됩니다.
        # sample별 lock을 잡아야 두 worker가 같은 sample_id의 파일 쌍을 동시에 만들지 않습니다.
        lock = _FileLock(self._raw_sample_path(sample_id) + ".lock")
        lock.acquire()
        try:
            if self.is_sample_completed(sample_id):
                if skip_if_completed:
                    lock.release()
                    return True
                raise ValueError(f"sample already completed: {sample_id}")
            tmp_path = self._raw_sample_path(sample_id) + ".tmp"
            trace_index_tmp_path = self._raw_trace_index_path(sample_id) + ".tmp"
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            try:
                os.remove(trace_index_tmp_path)
            except FileNotFoundError:
                pass
            self._open_writer = _RawSampleFileWriter(
                tmp_path,
                trace_index_tmp_path,
                point_schema=self.point_schema,
                compression=str(self.options.compression),
            )
            self._open_sample_id = int(sample_id)
            self._open_trace_count = 0
            self._open_point_count = 0
            self._open_lock = lock
            return False
        except Exception:
            lock.release()
            raise

    def _end_sample(self, *, abort: bool):
        if self._open_sample_id is None:
            return
        sample_id = int(self._open_sample_id)
        writer = self._open_writer
        lock = self._open_lock
        # builder 내부 open 상태를 먼저 비워 재진입을 막습니다. 이후 close/rename/log append
        # 중 예외가 나도 finally에서 sample lock은 반드시 release됩니다.
        self._open_sample_id = None
        self._open_writer = None
        self._open_lock = None
        trace_count = int(self._open_trace_count)
        point_count = int(self._open_point_count)
        self._open_trace_count = 0
        self._open_point_count = 0

        try:
            if abort:
                if writer is not None:
                    writer.abort()
                return
            if writer is None:
                return
            writer.close()
            final_path = self._raw_sample_path(sample_id)
            final_trace_index_path = self._raw_trace_index_path(sample_id)
            # point parquet와 trace-index parquet가 모두 final 이름으로 이동한 뒤에만
            # sample commit log를 append합니다. 둘 중 하나라도 실패하면 log를 쓰지
            # 않으므로 resume/compact가 불완전한 sample을 완료로 보지 않습니다.
            replace_file_with_retry(writer.path, final_path)
            replace_file_with_retry(writer.trace_index_path, final_trace_index_path)
            record = {
                "sample_id": int(sample_id),
                "sample_key": self._sample_key_for_id(sample_id),
                "path": self._raw_sample_rel_path(sample_id),
                "trace_index_path": self._raw_trace_index_rel_path(sample_id),
                "trace_count": int(trace_count),
                "row_count": int(point_count),
                "byte_size": int(os.path.getsize(final_path)),
                "trace_index_byte_size": int(os.path.getsize(final_trace_index_path)),
            }
            self._append_raw_commit(record)
        finally:
            if lock is not None:
                lock.release()

    def add_trace(
        self,
        *,
        sample_id: Optional[int] = None,
        sample_key: Optional[str] = None,
        feature_id: Optional[int] = None,
        feature_key: Optional[str] = None,
        columns,
    ):
        resolved_sample_id = self._resolve_sample_id(sample_id, sample_key)
        if self._open_sample_id is None:
            self._begin_sample(resolved_sample_id, skip_if_completed=False)
        elif int(self._open_sample_id) != int(resolved_sample_id):
            raise RuntimeError("sample boundary crossed without closing previous raw sample")
        if self._open_writer is None:
            raise RuntimeError("raw sample writer is not open")
        resolved_feature_id = self._resolve_feature_id(feature_id, feature_key)
        normalized = self._normalize_columns_for_raw(columns)
        trace_len = int(next(iter(normalized.values())).shape[0]) if normalized else 0
        self._open_writer.write_row(
            sample_id=int(resolved_sample_id),
            feature_id=int(resolved_feature_id),
            trace_len=trace_len,
            columns=normalized,
        )
        self._open_trace_count += 1
        self._open_point_count += int(trace_len)

    def _append_raw_commit(self, record: dict):
        # raw_samples.jsonl은 모든 worker가 공유하는 append-only commit log입니다.
        # sample 파일 쌍이 final 위치에 있다는 사실을 한 줄 JSON으로 남기므로, append는
        # log 전용 lock 아래에서 직렬화합니다.
        lock = _FileLock(self.raw_log_path + ".lock")
        lock.acquire()
        try:
            append_jsonl(self.raw_log_path, record)
        finally:
            lock.release()

    def _raw_commit_records(self) -> dict[int, dict]:
        records: dict[int, dict] = {}
        for item in load_jsonl(self.raw_log_path):
            try:
                sample_id = int(item["sample_id"])
            except (KeyError, TypeError, ValueError):
                continue
            path = os.path.join(self.out_dir, str(item.get("path") or ""))
            trace_index_path = os.path.join(self.out_dir, str(item.get("trace_index_path") or ""))
            if sample_id < 0 or sample_id >= int(self.n_samples):
                continue
            if not os.path.exists(path):
                continue
            if not trace_index_path or not os.path.exists(trace_index_path):
                continue
            records[sample_id] = dict(item)
        return records

    def completed_sample_ids(self) -> list[int]:
        return sorted(self._raw_commit_records().keys())

    def pending_sample_ids(self) -> list[int]:
        completed = set(self.completed_sample_ids())
        return [idx for idx in range(int(self.n_samples)) if idx not in completed]

    def is_sample_completed(self, sample_id: int) -> bool:
        return int(sample_id) in self._raw_commit_records()

    def recover_raw_samples(self) -> int:
        """log에는 없지만 정상 parquet 파일로 남은 raw sample을 commit log에 채택한다."""

        known = set(self._raw_commit_records())
        recovered = 0
        if not os.path.isdir(self.raw_samples_path):
            return 0
        for name in sorted(os.listdir(self.raw_samples_path)):
            if not (name.startswith("sample_") and name.endswith(".parquet")):
                continue
            try:
                sample_id = int(name[len("sample_") : -len(".parquet")])
            except ValueError:
                continue
            if sample_id in known or sample_id < 0 or sample_id >= int(self.n_samples):
                continue
            path = os.path.join(self.raw_samples_path, name)
            trace_index_path = self._raw_trace_index_path(sample_id)
            if not os.path.exists(trace_index_path):
                continue
            try:
                df = pl.read_parquet(path, columns=["sample_id"])
                trace_index_df = pl.read_parquet(trace_index_path, columns=["sample_id", "trace_len"])
            except Exception:
                continue
            if df.height and set(int(value) for value in df["sample_id"].to_list()) != {int(sample_id)}:
                continue
            if trace_index_df.height and set(int(value) for value in trace_index_df["sample_id"].to_list()) != {int(sample_id)}:
                continue
            record = {
                "sample_id": int(sample_id),
                "sample_key": self._sample_key_for_id(sample_id),
                "path": os.path.relpath(path, self.out_dir).replace("\\", "/"),
                "trace_index_path": self._raw_trace_index_rel_path(sample_id),
                "trace_count": int(trace_index_df.height),
                "row_count": int(df.height),
                "byte_size": int(os.path.getsize(path)),
                "trace_index_byte_size": int(os.path.getsize(trace_index_path)),
                "recovered": True,
            }
            self._append_raw_commit(record)
            known.add(sample_id)
            recovered += 1
        return recovered

    def status(self) -> ArraySampleParquetBuildSessionStatus:
        completed = self.completed_sample_ids()
        pending = [idx for idx in range(int(self.n_samples)) if idx not in set(completed)]
        return ArraySampleParquetBuildSessionStatus(
            n_samples=int(self.n_samples),
            completed_sample_count=len(completed),
            pending_sample_count=len(pending),
            completed_sample_ids=completed,
            pending_sample_ids=pending,
            finished=bool(self._finished),
            manifest_path=self.manifest_path if self._finished else None,
        )

    def compact(self, *, require_all: bool = True, cleanup_raw: bool = False, overwrite: bool = False) -> str:
        """raw sample 파일들을 size-based 최종 part parquet으로 묶고 manifest를 쓴다."""

        if self._finished and os.path.exists(self.manifest_path) and not overwrite:
            return self.manifest_path
        records_by_sample = self._raw_commit_records()
        pending = [idx for idx in range(int(self.n_samples)) if idx not in records_by_sample]
        if require_all and pending:
            raise ValueError(f"cannot compact: {len(pending)} samples are still pending")
        if os.path.exists(self.manifest_path) and not overwrite:
            raise ValueError(f"manifest already exists: {self.manifest_path}")
        if overwrite:
            shutil.rmtree(self.sample_parts_path, ignore_errors=True)
            shutil.rmtree(self.trace_index_parts_path, ignore_errors=True)
        elif os.path.isdir(self.sample_parts_path) and os.listdir(self.sample_parts_path):
            raise ValueError(f"sample_parts directory is not empty: {self.sample_parts_path}")
        os.makedirs(self.sample_parts_path, exist_ok=True)
        os.makedirs(self.trace_index_parts_path, exist_ok=True)

        records = [records_by_sample[sample_id] for sample_id in sorted(records_by_sample)]
        final_schema = list(self.point_schema)
        parts = self._write_compact_parts(records, final_schema)
        manifest = ArraySampleParquetManifest(
            sample_meta_path=self.sample_meta_path,
            feature_meta_path=self.feature_meta_path,
            n_samples=int(self.n_samples),
            n_features=int(self._known_feature_count or len(self._feature_keys_in_order)),
            sample_parts_path=self.sample_parts_path,
            trace_index_parts_path=self.trace_index_parts_path,
            parts=parts,
            point_schema=final_schema,
            sample_key_col=str(self.options.sample_key_col),
            feature_key_col=str(self.options.feature_key_col),
        )
        write_array_sample_parquet_manifest(self.manifest_path, manifest)
        self._finished = True
        self._save_state()
        if cleanup_raw:
            shutil.rmtree(self.raw_samples_path, ignore_errors=True)
            shutil.rmtree(self.raw_trace_index_path, ignore_errors=True)
        return self.manifest_path

    def finish(self) -> str:
        """Finalize all completed raw samples into public dataset parts."""

        return self.compact(require_all=True, cleanup_raw=False, overwrite=False)

    def _write_compact_parts(
        self,
        records: list[dict],
        final_schema: list[PointColumnSpec],
    ) -> list[ArraySampleParquetPart]:
        parts: list[ArraySampleParquetPart] = []
        plans = _plan_compact_parts(records, out_dir=self.out_dir, options=self.options)
        for part_id, plan in enumerate(plans):
            part_path = self._part_path(part_id)
            trace_index_path = self._trace_index_part_path(part_id)
            part_df = _read_compact_part_frame(plan.raw_paths)
            trace_index_df = _read_compact_trace_index_frame(plan.raw_trace_index_paths)
            row_count = _write_long_compact_part(
                part_df,
                trace_index_df,
                part_path,
                trace_index_path,
                final_schema=final_schema,
                compression=str(self.options.compression),
            )
            parts.append(
                ArraySampleParquetPart(
                    part_id=int(part_id),
                    path=part_path,
                    trace_index_path=trace_index_path,
                    first_sample_id=int(plan.first_sample_id),
                    last_sample_id=int(plan.last_sample_id),
                    sample_count=int(plan.sample_count),
                    trace_count=int(plan.trace_count),
                    row_count=row_count,
                    byte_size=int(os.path.getsize(part_path)),
                    trace_index_byte_size=int(os.path.getsize(trace_index_path)),
                )
            )
        return parts

    def close(self):
        self._end_sample(abort=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _RawSampleFileWriter:
    def __init__(self, path: str, trace_index_path: str, *, point_schema: list[PointColumnSpec], compression: str):
        self.path = os.path.abspath(path)
        self.trace_index_path = os.path.abspath(trace_index_path)
        self.point_schema = list(point_schema)
        self.compression = None if str(compression).lower() in {"", "none"} else str(compression)
        self.schema = _raw_point_arrow_schema(self.point_schema)
        self.trace_index_schema = _raw_trace_index_arrow_schema()
        self._stream_path = self.path + ".stream"
        self._stream_trace_index_path = self.trace_index_path + ".stream"
        self._trace_sample_ids: list[int] = []
        self._trace_feature_ids: list[int] = []
        self._trace_lens: list[int] = []
        self._point_columns = empty_point_columns(self.point_schema)
        self._batch_point_count = 0
        self._seen_trace_keys: set[tuple[int, int]] = set()
        self._last_trace_key: Optional[tuple[int, int]] = None
        self._is_sorted = True
        self._closed = False
        self._writers_closed = False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trace_index_path), exist_ok=True)
        for stale_path in [self._stream_path, self._stream_trace_index_path]:
            try:
                os.remove(stale_path)
            except FileNotFoundError:
                pass
        self._writer = pq.ParquetWriter(self._stream_path, self.schema, compression=self.compression)
        self._trace_index_writer = pq.ParquetWriter(
            self._stream_trace_index_path,
            self.trace_index_schema,
            compression=self.compression,
        )

    def write_row(self, *, sample_id: int, feature_id: int, trace_len: int, columns: dict[str, np.ndarray]):
        if self._closed:
            raise RuntimeError("raw sample writer is already closed")
        trace_key = (int(sample_id), int(feature_id))
        if trace_key in self._seen_trace_keys:
            raise ValueError(f"duplicate trace for sample_id={sample_id}, feature_id={feature_id}")
        if self._last_trace_key is not None and self._last_trace_key > trace_key:
            self._is_sorted = False
        self._seen_trace_keys.add(trace_key)
        self._last_trace_key = trace_key

        copied = {spec.name: np.asarray(columns[spec.name]).reshape(-1).copy() for spec in self.point_schema}
        self._trace_sample_ids.append(int(sample_id))
        self._trace_feature_ids.append(int(feature_id))
        self._trace_lens.append(int(trace_len))
        for spec in self.point_schema:
            self._point_columns[spec.name].append(copied[spec.name])
        self._batch_point_count += int(trace_len)
        if (
            len(self._trace_sample_ids) >= RAW_WRITER_BATCH_TRACES
            or int(self._batch_point_count) >= RAW_WRITER_BATCH_POINTS
        ):
            self.flush()

    def flush(self):
        if not self._trace_sample_ids:
            return

        trace_index_arrays = [
            pa.array(self._trace_sample_ids, type=pa.int64()),
            pa.array(self._trace_feature_ids, type=pa.int32()),
            pa.array(self._trace_lens, type=pa.int32()),
        ]
        self._trace_index_writer.write_table(pa.Table.from_arrays(trace_index_arrays, schema=self.trace_index_schema))

        point_count = int(sum(self._trace_lens))
        if point_count > 0:
            point_arrays = [
                pa.array(np.repeat(np.asarray(self._trace_sample_ids, dtype=np.int64), self._trace_lens), type=pa.int64()),
                pa.array(np.repeat(np.asarray(self._trace_feature_ids, dtype=np.int32), self._trace_lens), type=pa.int32()),
                pa.array(_point_indices_from_lengths(self._trace_lens), type=pa.int32()),
            ]
            for spec in self.point_schema:
                point_arrays.append(_raw_value_array_from_rows(self._point_columns[spec.name], spec, point_count))
            self._writer.write_table(pa.Table.from_arrays(point_arrays, schema=self.schema))

        self._trace_sample_ids = []
        self._trace_feature_ids = []
        self._trace_lens = []
        self._point_columns = empty_point_columns(self.point_schema)
        self._batch_point_count = 0

    def _close_writers(self):
        if self._writers_closed:
            return
        self.flush()
        self._writer.close()
        self._trace_index_writer.close()
        self._writers_closed = True

    def close(self):
        if self._closed:
            return
        try:
            self._close_writers()
            if self._is_sorted:
                replace_file_with_retry(self._stream_path, self.path)
                replace_file_with_retry(self._stream_trace_index_path, self.trace_index_path)
            else:
                _sort_raw_point_file(
                    self._stream_path,
                    self.path,
                    point_schema=self.point_schema,
                    compression=self.compression,
                )
                _sort_raw_trace_index_file(
                    self._stream_trace_index_path,
                    self.trace_index_path,
                    compression=self.compression,
                )
                for stale_path in [self._stream_path, self._stream_trace_index_path]:
                    try:
                        os.remove(stale_path)
                    except FileNotFoundError:
                        pass
        finally:
            self._closed = True

    def abort(self):
        if not self._writers_closed:
            for writer in [getattr(self, "_writer", None), getattr(self, "_trace_index_writer", None)]:
                try:
                    if writer is not None:
                        writer.close()
                except Exception:
                    pass
            self._writers_closed = True
        self._closed = True
        for path in [self.path, self.trace_index_path, self._stream_path, self._stream_trace_index_path]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _raw_point_arrow_schema(point_schema: list[PointColumnSpec]) -> pa.Schema:
    fields = [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("feature_id", pa.int32(), nullable=False),
        pa.field("point_idx", pa.int32(), nullable=False),
    ]
    for spec in point_schema:
        value_type = pa.string() if spec.logical_type == LogicalType.CATEGORICAL else arrow_value_type(spec)
        fields.append(pa.field(spec.name, value_type, nullable=False))
    return pa.schema(fields, metadata={b"format": b"array-sample-parquet-raw-points", b"version": b"1"})


def _raw_trace_index_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.int64(), nullable=False),
            pa.field("feature_id", pa.int32(), nullable=False),
            pa.field("trace_len", pa.int32(), nullable=False),
        ],
        metadata={b"format": b"array-sample-parquet-raw-trace-index", b"version": b"1"},
    )


def _point_indices_from_lengths(lengths: list[int]) -> np.ndarray:
    non_empty = [np.arange(int(length), dtype=np.int32) for length in lengths if int(length) > 0]
    return np.concatenate(non_empty) if non_empty else np.empty(0, dtype=np.int32)


def _raw_value_array_from_rows(rows: list[np.ndarray], spec: PointColumnSpec, total_len: int) -> pa.Array:
    if spec.storage_type == StorageType.STRING:
        if total_len <= 0:
            values = np.empty(0, dtype=object)
        else:
            values = np.concatenate([row for row in rows if int(row.size) > 0]).astype(object, copy=False)
        return pa.array(values, type=pa.string())
    if total_len <= 0:
        values = np.empty(0, dtype=point_storage_dtype(spec.storage_type))
    else:
        values = np.concatenate([row for row in rows if int(row.size) > 0]).astype(point_storage_dtype(spec.storage_type), copy=False)
    return pa.array(values, type=arrow_value_type(spec))


def _sort_raw_point_file(
    src_path: str,
    out_path: str,
    *,
    point_schema: list[PointColumnSpec],
    compression: Optional[str],
):
    columns = ["sample_id", "feature_id", "point_idx", *[spec.name for spec in point_schema]]
    df = pl.scan_parquet(src_path).select(columns).sort(["sample_id", "feature_id", "point_idx"]).collect()
    _write_part_frame(df, out_path, compression=compression or "none")


def _sort_raw_trace_index_file(src_path: str, out_path: str, *, compression: Optional[str]):
    df = pl.scan_parquet(src_path).select(["sample_id", "feature_id", "trace_len"]).sort(["sample_id", "feature_id"]).collect()
    _write_part_frame(df, out_path, compression=compression or "none")

def _would_exceed_part(
    *,
    batch_rows: int,
    batch_bytes: int,
    batch_sample_count: int,
    add_rows: int,
    add_bytes: int,
    add_sample_count: int,
    options: ArraySampleParquetBuildOptions,
) -> bool:
    if int(options.max_part_samples) > 0 and batch_sample_count + add_sample_count > int(options.max_part_samples):
        return True
    if batch_rows + add_rows > int(options.max_part_rows):
        return True
    return bool(batch_bytes + add_bytes > int(options.target_part_bytes))


def _plan_compact_parts(
    records: list[dict],
    *,
    out_dir: str,
    options: ArraySampleParquetBuildOptions,
) -> list[_CompactPartPlan]:
    plans: list[_CompactPartPlan] = []
    batch_records: list[dict] = []
    batch_paths: list[str] = []
    batch_trace_index_paths: list[str] = []
    batch_rows = 0
    batch_trace_count = 0
    batch_bytes = 0
    batch_sample_count = 0
    batch_first_sample_id: Optional[int] = None
    batch_last_sample_id: Optional[int] = None

    def commit_plan():
        nonlocal batch_records, batch_paths, batch_trace_index_paths, batch_rows, batch_trace_count, batch_bytes, batch_sample_count, batch_first_sample_id, batch_last_sample_id
        if not batch_records:
            return
        plans.append(
            _CompactPartPlan(
                raw_paths=list(batch_paths),
                raw_trace_index_paths=list(batch_trace_index_paths),
                row_count=int(batch_rows),
                trace_count=int(batch_trace_count),
                byte_size=int(batch_bytes),
                sample_count=int(batch_sample_count),
                first_sample_id=int(batch_first_sample_id),
                last_sample_id=int(batch_last_sample_id),
            )
        )
        batch_records = []
        batch_paths = []
        batch_trace_index_paths = []
        batch_rows = 0
        batch_trace_count = 0
        batch_bytes = 0
        batch_sample_count = 0
        batch_first_sample_id = None
        batch_last_sample_id = None

    for record in records:
        sample_rows = _raw_record_row_count(record)
        sample_trace_count = _raw_record_trace_count(record)
        if sample_trace_count <= 0:
            continue
        sample_id = int(record["sample_id"])
        sample_bytes = _raw_record_byte_size(record, out_dir)
        if batch_records and _would_exceed_part(
            batch_rows=batch_rows,
            batch_bytes=batch_bytes,
            batch_sample_count=batch_sample_count,
            add_rows=sample_rows,
            add_bytes=sample_bytes,
            add_sample_count=1,
            options=options,
        ):
            commit_plan()
        if batch_first_sample_id is None:
            batch_first_sample_id = sample_id
        batch_last_sample_id = sample_id
        batch_records.append(record)
        batch_paths.append(_raw_record_path(record, out_dir))
        batch_trace_index_paths.append(_raw_record_trace_index_path(record, out_dir))
        batch_rows += sample_rows
        batch_trace_count += sample_trace_count
        batch_bytes += sample_bytes
        batch_sample_count += 1
    commit_plan()
    return plans


def _read_compact_part_frame(raw_paths: list[str]) -> pl.DataFrame:
    if not raw_paths:
        return pl.DataFrame()
    return pl.scan_parquet(raw_paths, glob=False).collect()


def _read_compact_trace_index_frame(raw_paths: list[str]) -> pl.DataFrame:
    if not raw_paths:
        return pl.DataFrame()
    return pl.scan_parquet(raw_paths, glob=False).collect()


def _raw_record_row_count(record: dict) -> int:
    try:
        return int(record.get("row_count", record.get("trace_count", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _raw_record_trace_count(record: dict) -> int:
    try:
        return int(record.get("trace_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _raw_record_byte_size(record: dict, out_dir: str) -> int:
    try:
        value = int(record.get("byte_size", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value + int(record.get("trace_index_byte_size", 0) or 0)
    try:
        return int(os.path.getsize(_raw_record_path(record, out_dir))) + int(os.path.getsize(_raw_record_trace_index_path(record, out_dir)))
    except OSError:
        return 0


def _raw_record_path(record: dict, out_dir: str) -> str:
    return os.path.join(out_dir, str(record.get("path") or ""))


def _raw_record_trace_index_path(record: dict, out_dir: str) -> str:
    return os.path.join(out_dir, str(record.get("trace_index_path") or ""))


def _write_long_compact_part(
    df: pl.DataFrame,
    trace_index_df: pl.DataFrame,
    point_path: str,
    trace_index_path: str,
    *,
    final_schema: list[PointColumnSpec],
    compression: str,
) -> int:
    trace_index_df = trace_index_df.sort(["sample_id", "feature_id"]) if trace_index_df.height > 0 else trace_index_df
    _write_part_frame(trace_index_df, trace_index_path, compression=compression)

    if df.height == 0:
        point_df = _empty_long_point_frame(final_schema)
    else:
        point_df = (
            df
            .sort(["sample_id", "feature_id", "point_idx"])
            .select(["sample_id", "feature_id", "point_idx", *[spec.name for spec in final_schema]])
        )
    _write_part_frame(point_df, point_path, compression=compression)
    return int(point_df.height)


def _empty_long_point_frame(final_schema: list[PointColumnSpec]) -> pl.DataFrame:
    schema = {
        "sample_id": pl.Int64,
        "feature_id": pl.Int32,
        "point_idx": pl.Int32,
    }
    for spec in final_schema:
        schema[spec.name] = _polars_storage_dtype(spec.storage_type)
    return pl.DataFrame(schema=schema)


def _polars_storage_dtype(storage_type: StorageType) -> pl.DataType:
    if storage_type == StorageType.FLOAT64:
        return pl.Float64
    if storage_type == StorageType.STRING:
        return pl.String
    if storage_type == StorageType.INT32:
        return pl.Int32
    if storage_type == StorageType.INT64:
        return pl.Int64
    if storage_type in {StorageType.UINT8, StorageType.UINT16, StorageType.UINT32, StorageType.UINT64}:
        return _polars_unsigned_dtype(storage_type)
    raise ValueError(f"unsupported storage_type: {storage_type}")


def _polars_unsigned_dtype(storage_type: StorageType) -> pl.DataType:
    if storage_type == StorageType.UINT8:
        return pl.UInt8
    if storage_type == StorageType.UINT16:
        return pl.UInt16
    if storage_type == StorageType.UINT32:
        return pl.UInt32
    if storage_type == StorageType.UINT64:
        return pl.UInt64
    raise ValueError(f"unsupported unsigned storage_type: {storage_type}")


def _normalize_categorical_values(values, column_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if bool(np.any(arr == None)):
        raise ValueError(f"categorical point column {column_name} does not support null values")
    return arr


def _write_part_frame(df: pl.DataFrame, path: str, *, compression: str):
    pq.write_table(df.to_arrow(), path, compression=_pyarrow_compression(compression))


def _pyarrow_compression(compression: str) -> Optional[str]:
    value = str(compression or "").strip().lower()
    if value in {"", "none", "uncompressed"}:
        return None
    return value


def _should_flush_part(row_count: int, byte_count: int, sample_count: int, options: ArraySampleParquetBuildOptions) -> bool:
    if sample_count <= 0:
        return False
    if int(options.max_part_samples) > 0 and sample_count >= int(options.max_part_samples):
        return True
    return bool(row_count >= int(options.max_part_rows) or byte_count >= int(options.target_part_bytes))


def _list_or_empty(value) -> list:
    if value is None:
        return []
    return list(value)


def _write_feature_meta(records: list[dict], out_path: str, *, feature_key_col: str) -> str:
    rows = [dict(record) for record in records]
    for idx, row in enumerate(rows):
        row["feature_id"] = int(idx)
    columns = ["feature_id"] + [name for name in rows[0].keys() if name != "feature_id"] if rows else ["feature_id"]
    if rows:
        df = pl.from_dicts(rows, infer_schema_length=None).with_columns(pl.col("feature_id").cast(pl.Int32)).select(columns)
        if feature_key_col and feature_key_col in df.columns:
            if df[feature_key_col].null_count() != 0:
                raise ValueError(f"feature {feature_key_col} cannot contain nulls")
            if int(df[feature_key_col].n_unique()) != int(df.height):
                raise ValueError(f"feature {feature_key_col} must be unique")
    else:
        df = pl.DataFrame({"feature_id": pl.Series("feature_id", [], dtype=pl.Int32)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return str(Path(out_path).resolve())
