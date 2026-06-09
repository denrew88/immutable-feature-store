"""Resume-safe scalar builder backed by one raw parquet file per sample."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ScalarShardBuildOptions
from .dense_long import build_dense_long_shards_from_sample_major_manifest
from .file_lock import FilePathLock as _FileLock
from .storage_common import SAMPLE_MAJOR_MANIFEST_FORMAT, load_dense_metadata, write_json_atomic


RAW_STATE_VERSION = 1
RAW_SAMPLE_PADDING = 12
RAW_FORMAT_NAME = "scalar-raw-samples"
FILE_REPLACE_RETRY_COUNT = 10
FILE_REPLACE_RETRY_BASE_SECONDS = 0.025
DEFAULT_STAGE_LOCK_TIMEOUT_SECONDS = 30.0
STAGE_LOCK_TIMEOUT_ENV = "SCALAR_RAW_STAGE_LOCK_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class ScalarBuildSessionStatus:
    """표준 scalar build session의 진행 상태.

    현재 builder는 sample 하나를 raw parquet 파일 하나로 commit합니다. 따라서 재개
    기준은 순차 watermark 하나가 아니라 완료된 sample id 목록과 아직 남은 sample id
    목록입니다. 순차 실행을 원하면 `pending_sample_ids`를 앞에서부터 처리하면 됩니다.
    """

    n_samples: int
    completed_sample_count: int
    pending_sample_count: int
    completed_sample_ids: list[int]
    pending_sample_ids: list[int]
    finished_stage: bool
    sample_major_manifest_path: Optional[str]

    @property
    def next_pending_sample_id(self) -> Optional[int]:
        return self.pending_sample_ids[0] if self.pending_sample_ids else None


class ScalarDatasetBuilder:
    """sample별 raw parquet 파일에서 scalar dense-long shard를 만드는 builder.

    완료된 sample은 `raw_samples/sample_*.parquet` 파일 하나와 `raw_samples.jsonl`
    commit log 한 줄로 기록됩니다. sample id 순서를 강제하지 않으므로 supervisor가
    `pending_sample_ids()`를 worker에게 나눠주고, 마지막에 한 프로세스가
    `build_dense_long_shards()`를 호출해 최종 dense-long shard를 만들 수 있습니다.
    """

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        *,
        feature_meta_path: Optional[str] = None,
        feature_keys: Optional[Sequence[str]] = None,
        build_options: ScalarShardBuildOptions | None = None,
    ):
        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")
        if not feature_meta_path and feature_keys is None:
            raise ValueError("scalar builder requires feature_meta_path or feature_keys")

        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.source_sample_meta_path = str(Path(sample_meta_path).expanduser().resolve())
        self.build_options = build_options or ScalarShardBuildOptions()

        self.raw_samples_path = os.path.join(self.out_dir, "raw_samples")
        self.state_path = os.path.join(self.out_dir, "raw_state.json")
        self.raw_stage_lock_path = os.path.join(self.out_dir, "raw_stage.lock")
        self.raw_log_path = os.path.join(self.out_dir, "raw_samples.jsonl")
        self.raw_log_lock_path = os.path.join(self.out_dir, "raw_samples.jsonl.lock")
        self.sample_major_manifest_path = os.path.join(self.out_dir, "sample_major_manifest.json")
        self.sample_meta_path = os.path.join(self.out_dir, "sample_meta.parquet")
        self.feature_meta_path = os.path.join(self.out_dir, "feature_meta.parquet")

        sample_meta_df = load_dense_metadata(
            self.source_sample_meta_path,
            id_col=str(self.build_options.sample_id_col),
            entity_name="sample",
            key_col=str(self.build_options.sample_key_col),
        )
        for y_col in self._stats_y_cols():
            if y_col not in sample_meta_df.columns:
                raise ValueError(f"sample metadata is missing required target column: {y_col}")
        self.n_samples = int(sample_meta_df.height)
        self._sample_keys = (
            [None] * self.n_samples
            if str(self.build_options.sample_key_col) not in sample_meta_df.columns
            else [
                None if value is None else str(value)
                for value in sample_meta_df[str(self.build_options.sample_key_col)].to_list()
            ]
        )
        self._sample_key_to_id = {
            key: int(idx) for idx, key in enumerate(self._sample_keys) if key is not None
        }

        self._feature_meta_source_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        self._feature_key_to_id: OrderedDict[str, int] = OrderedDict()
        self._feature_keys_in_order: list[str] = []
        self._known_feature_count: Optional[int] = None
        self._finished_stage = False

        if os.path.exists(self.state_path):
            self._resume(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        else:
            self._initialize(feature_meta_path=feature_meta_path, feature_keys=feature_keys)

    @classmethod
    def open_session(cls, *args, **kwargs) -> "ScalarDatasetBuilder":
        return cls(*args, **kwargs)

    def _stats_y_cols(self) -> tuple[str, ...]:
        values = self.build_options.stats_y_cols
        if values is None:
            return (str(self.build_options.y_col),)
        out = []
        for value in values:
            name = str(value)
            if name and name not in out:
                out.append(name)
        return tuple(out)

    def _options_payload(self) -> dict:
        payload = asdict(self.build_options)
        if payload.get("stats_y_cols") is not None:
            payload["stats_y_cols"] = [str(value) for value in payload["stats_y_cols"]]
        return payload

    def _state_payload(self) -> dict:
        finalized = bool(self._finished_stage or os.path.exists(self.sample_major_manifest_path))
        return {
            "format": RAW_FORMAT_NAME,
            "raw_state_version": RAW_STATE_VERSION,
            "sample_meta_source_path": self.source_sample_meta_path,
            "feature_meta_source_path": self._feature_meta_source_path,
            "options": self._options_payload(),
            "feature_keys_in_order": list(self._feature_keys_in_order),
            "known_feature_count": self._known_feature_count,
            "finished_stage": finalized,
            "sample_major_manifest_path": self.sample_major_manifest_path if finalized else None,
        }

    def _save_state(self):
        write_json_atomic(self.state_path, self._state_payload())

    def _initialize(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if os.path.exists(self.out_dir) and os.listdir(self.out_dir):
            raise ValueError(f"out_dir already exists and is not empty: {self.out_dir}")
        os.makedirs(self.raw_samples_path, exist_ok=True)
        shutil.copy2(self.source_sample_meta_path, self.sample_meta_path)
        self._initialize_feature_meta(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        self._cleanup_tmp_files()
        self._save_state()

    def _initialize_feature_meta(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if feature_meta_path:
            feature_meta = load_dense_metadata(
                self._feature_meta_source_path,
                id_col=str(self.build_options.feature_id_col),
                entity_name="feature",
                key_col=str(self.build_options.feature_key_col),
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
        self._write_feature_meta_from_keys()

    def _load_feature_keys_from_meta(self, feature_meta: pl.DataFrame):
        key_col = str(self.build_options.feature_key_col)
        if key_col and key_col in feature_meta.columns:
            for idx, value in enumerate(feature_meta[key_col].to_list()):
                key = str(value)
                self._feature_key_to_id[key] = int(idx)
                self._feature_keys_in_order.append(key)

    def _write_feature_meta_from_keys(self):
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {
            str(self.build_options.feature_id_col): pl.Series(
                str(self.build_options.feature_id_col), feature_ids, dtype=pl.Int32
            )
        }
        key_col = str(self.build_options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self.feature_meta_path)

    def _resume(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        self._cleanup_tmp_files()
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._validate_resume_state(state, feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        self._finished_stage = bool(state.get("finished_stage"))
        self._known_feature_count = state.get("known_feature_count")
        self._feature_keys_in_order = [str(value) for value in state.get("feature_keys_in_order") or []]
        self._feature_key_to_id = OrderedDict(
            (feature_key, int(idx)) for idx, feature_key in enumerate(self._feature_keys_in_order)
        )

    def _validate_resume_state(self, state: dict, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if state.get("format") != RAW_FORMAT_NAME:
            raise ValueError(f"unsupported scalar build session format: {state.get('format')!r}")
        if int(state.get("raw_state_version", 0)) != RAW_STATE_VERSION:
            raise ValueError(f"unsupported raw scalar build session version: {state.get('raw_state_version')}")
        if str(Path(state["sample_meta_source_path"]).expanduser().resolve()) != self.source_sample_meta_path:
            raise ValueError("sample_meta_path does not match existing scalar build session")
        if state.get("options") != self._options_payload():
            raise ValueError("options do not match existing raw scalar build session")
        normalized_feature_meta_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        stored_feature_meta_path = str(state.get("feature_meta_source_path") or "")
        if normalized_feature_meta_path and stored_feature_meta_path and normalized_feature_meta_path != stored_feature_meta_path:
            raise ValueError("feature_meta_path does not match existing scalar build session")
        if feature_keys is not None and [str(value) for value in feature_keys] != list(state.get("feature_keys_in_order") or []):
            raise ValueError("feature_keys do not match existing scalar build session")

    def _cleanup_tmp_files(self):
        if not os.path.isdir(self.raw_samples_path):
            return
        for name in os.listdir(self.raw_samples_path):
            if name.endswith(".tmp"):
                _remove_file_best_effort(os.path.join(self.raw_samples_path, name))

    def _cleanup_sample_tmp_files(self, sample_id: int):
        if not os.path.isdir(self.raw_samples_path):
            return
        final_name = os.path.basename(self._raw_sample_path(sample_id))
        legacy_name = final_name + ".tmp"
        unique_prefix = final_name + "."
        for name in os.listdir(self.raw_samples_path):
            if name.endswith(".tmp") and (name == legacy_name or name.startswith(unique_prefix)):
                _remove_file_best_effort(os.path.join(self.raw_samples_path, name))

    def _active_sample_lock_paths(self) -> list[str]:
        if not os.path.isdir(self.raw_samples_path):
            return []
        return sorted(
            os.path.join(self.raw_samples_path, name)
            for name in os.listdir(self.raw_samples_path)
            if name.startswith("sample_") and name.endswith(".parquet.lock")
        )

    def _wait_for_no_active_sample_locks(self):
        deadline = time.monotonic() + _stage_lock_timeout_seconds()
        while True:
            locks = self._active_sample_lock_paths()
            if not locks:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"active scalar sample locks remain: {locks[:5]}")
            time.sleep(0.05)

    def _sample_key_for_id(self, sample_id: Optional[int]) -> Optional[str]:
        if sample_id is None or int(sample_id) < 0 or int(sample_id) >= int(self.n_samples):
            return None
        return self._sample_keys[int(sample_id)]

    def _resolve_sample_id(self, sample_id: Optional[int] = None, sample_key: Optional[str] = None) -> int:
        if sample_id is None and sample_key is None:
            raise ValueError("provide either sample_id or sample_key")
        if sample_id is not None and sample_key is not None:
            resolved = self._resolve_sample_id(sample_key=sample_key)
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

    def _resolve_feature_id(self, feature_ref) -> int:
        if isinstance(feature_ref, (int, np.integer)):
            feature_id = int(feature_ref)
            if feature_id < 0:
                raise ValueError("feature_id must be >= 0")
            if self._known_feature_count is not None and feature_id >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {feature_id}")
            return feature_id
        key = str(feature_ref)
        resolved = self._feature_key_to_id.get(key)
        if resolved is None:
            raise ValueError(f"unknown feature key: {key}")
        return int(resolved)

    def _normalize_scalar_value(self, value):
        if value is None:
            return None
        out = float(value)
        if np.isnan(out):
            return None
        return out

    def _raw_sample_path(self, sample_id: int) -> str:
        return os.path.join(self.raw_samples_path, f"sample_{int(sample_id):0{RAW_SAMPLE_PADDING}d}.parquet")

    def _raw_sample_rel_path(self, sample_id: int) -> str:
        return os.path.relpath(self._raw_sample_path(sample_id), self.out_dir).replace("\\", "/")

    def _raw_sample_id_from_filename(self, name: str) -> Optional[int]:
        prefix = "sample_"
        suffix = ".parquet"
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        raw_id = name[len(prefix) : -len(suffix)]
        if not raw_id.isdigit():
            return None
        sample_id = int(raw_id)
        if sample_id < 0 or sample_id >= int(self.n_samples):
            return None
        return sample_id

    def _read_raw_commit_records_unlocked(self) -> dict[int, dict]:
        records: dict[int, dict] = {}
        if not os.path.exists(self.raw_log_path):
            return records
        with open(self.raw_log_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                sample_id = int(row["sample_id"])
                path = os.path.join(self.out_dir, str(row.get("path", "")))
                if 0 <= sample_id < int(self.n_samples) and os.path.exists(path):
                    records[sample_id] = row
        return records

    def _reconcile_raw_commit_records(self):
        # final parquet 이동에는 성공했지만 raw_samples.jsonl append 전에 프로세스가 죽으면
        # raw 파일만 남습니다. log lock을 잡은 상태에서 디렉터리를 스캔해 이런 sample을
        # commit record로 복구해야 여러 builder가 같은 record를 중복 append하지 않습니다.
        lock = _FileLock(self.raw_log_lock_path, timeout_seconds=_stage_lock_timeout_seconds())
        lock.acquire()
        try:
            records = self._read_raw_commit_records_unlocked()
            additions: list[dict] = []
            if os.path.isdir(self.raw_samples_path):
                for name in sorted(os.listdir(self.raw_samples_path)):
                    sample_id = self._raw_sample_id_from_filename(name)
                    if sample_id is None or sample_id in records:
                        continue
                    path = self._raw_sample_path(sample_id)
                    if not os.path.isfile(path):
                        continue
                    parquet_file = pq.ParquetFile(path)
                    record = {
                        "sample_id": int(sample_id),
                        "sample_key": self._sample_key_for_id(sample_id),
                        "path": self._raw_sample_rel_path(sample_id),
                        "row_count": int(parquet_file.metadata.num_rows),
                        "byte_size": int(os.path.getsize(path)),
                    }
                    additions.append(record)
                    records[sample_id] = record
            if additions:
                with open(self.raw_log_path, "a", encoding="utf-8") as f:
                    for record in sorted(additions, key=lambda item: int(item["sample_id"])):
                        f.write(json.dumps(record, ensure_ascii=False))
                        f.write("\n")
        finally:
            lock.release()

    def _raw_commit_records(self) -> dict[int, dict]:
        self._reconcile_raw_commit_records()
        return self._read_raw_commit_records_unlocked()

    def _append_raw_commit(self, record: dict):
        # JSONL append는 모든 worker가 공유하는 단일 파일 write입니다. 한 줄 단위 JSON이
        # 서로 섞이지 않도록 log lock으로 직렬화합니다.
        lock = _FileLock(self.raw_log_lock_path)
        lock.acquire()
        try:
            records = self._read_raw_commit_records_unlocked()
            sample_id = int(record["sample_id"])
            if sample_id in records:
                return
            with open(self.raw_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
        finally:
            lock.release()

    def is_sample_completed(self, sample_id: int) -> bool:
        return int(sample_id) in self._raw_commit_records()

    def _is_raw_sample_file_completed(self, sample_id: int) -> bool:
        return os.path.exists(self._raw_sample_path(sample_id))

    def completed_sample_ids(self) -> list[int]:
        return sorted(self._raw_commit_records().keys())

    def pending_sample_ids(self) -> list[int]:
        completed = set(self.completed_sample_ids())
        return [sample_id for sample_id in range(int(self.n_samples)) if sample_id not in completed]

    def status(self) -> ScalarBuildSessionStatus:
        completed = self.completed_sample_ids()
        pending = [idx for idx in range(int(self.n_samples)) if idx not in set(completed)]
        return ScalarBuildSessionStatus(
            n_samples=int(self.n_samples),
            completed_sample_count=len(completed),
            pending_sample_count=len(pending),
            completed_sample_ids=completed,
            pending_sample_ids=pending,
            finished_stage=bool(self._finished_stage),
            sample_major_manifest_path=self.sample_major_manifest_path if self._finished_stage else None,
        )

    def write_sample(
        self,
        sample_id: Optional[int] = None,
        values: Mapping = None,
        *,
        sample_key: Optional[str] = None,
        skip_if_completed: bool = False,
    ) -> bool:
        """sample 하나를 raw parquet 파일로 commit합니다.

        `values`는 feature id 또는 feature key를 key로 갖는 mapping입니다. 값이
        `None`이거나 `NaN`이면 missing으로 취급해서 raw row를 쓰지 않습니다.
        새 sample 파일을 commit하면 `True`를 반환하고, `skip_if_completed=True`로
        이미 완료된 sample을 건너뛰면 `False`를 반환합니다.
        """

        if self._finished_stage or os.path.exists(self.sample_major_manifest_path):
            raise RuntimeError("raw scalar stage is already finalized")
        sample_id = self._resolve_sample_id(sample_id=sample_id, sample_key=sample_key)
        final_path = self._raw_sample_path(sample_id)
        stage_lock = _FileLock(self.raw_stage_lock_path, timeout_seconds=_stage_lock_timeout_seconds())
        sample_lock: Optional[_FileLock] = None
        stage_lock.acquire()
        try:
            if self._finished_stage or os.path.exists(self.sample_major_manifest_path):
                raise RuntimeError("raw scalar stage is already finalized")
            if self._is_raw_sample_file_completed(sample_id):
                if skip_if_completed:
                    return False
                raise ValueError(f"sample already completed: {sample_id}")
            sample_lock = _FileLock(final_path + ".lock", timeout_seconds=_stage_lock_timeout_seconds())
            sample_lock.acquire()
        finally:
            stage_lock.release()
        tmp_path = f"{final_path}.{uuid.uuid4().hex}.tmp"
        try:
            if self._is_raw_sample_file_completed(sample_id):
                if skip_if_completed:
                    return False
                raise ValueError(f"sample already completed: {sample_id}")
            self._cleanup_sample_tmp_files(sample_id)
            feature_values: dict[int, float] = {}
            for feature_ref, value in dict(values or {}).items():
                normalized = self._normalize_scalar_value(value)
                feature_id = self._resolve_feature_id(feature_ref)
                if feature_id in feature_values:
                    raise ValueError(f"duplicate feature assignment within sample {sample_id}: feature_id={feature_id}")
                if normalized is not None:
                    feature_values[int(feature_id)] = float(normalized)

            feature_ids = np.asarray(sorted(feature_values.keys()), dtype=np.int32)
            vals = np.asarray([feature_values[int(fid)] for fid in feature_ids], dtype=np.float64)
            sample_ids = np.full(feature_ids.shape[0], int(sample_id), dtype=np.int64)
            table = pa.table(
                {
                    str(self.build_options.sample_id_col): pa.array(sample_ids, type=pa.int64()),
                    str(self.build_options.feature_id_col): pa.array(feature_ids, type=pa.int32()),
                    str(self.build_options.value_col): pa.array(vals, type=pa.float64()),
                }
            )
            pq.write_table(table, tmp_path, compression=_pyarrow_compression("zstd"), use_dictionary=True)
            # tmp parquet를 final path로 옮긴 뒤에만 commit log를 append합니다.
            # 장애가 이 줄 이전에 나면 final parquet가 없거나 tmp만 남으므로 resume에서
            # 완료 sample로 보지 않습니다.
            _replace_file_with_retry(tmp_path, final_path)
            record = {
                "sample_id": int(sample_id),
                "sample_key": self._sample_key_for_id(sample_id),
                "path": self._raw_sample_rel_path(sample_id),
                "row_count": int(feature_ids.shape[0]),
                "byte_size": int(os.path.getsize(final_path)),
            }
            # raw_samples.jsonl은 "sample 파일이 최종 위치에 있다"는 commit 기록입니다.
            # 파일 이동과 log append 사이에서 죽어도 reconcile 단계가 실제 parquet를
            # 스캔해 빠진 commit record를 복구합니다.
            self._append_raw_commit(record)
            return True
        except Exception:
            _remove_file_best_effort(tmp_path)
            raise
        finally:
            if sample_lock is not None:
                sample_lock.release()

    def finish_stage(self, *, require_all: bool = False):
        """raw sample 파일 목록을 sample-major manifest로 확정합니다.

        이 단계는 raw parquet를 다시 쓰지 않습니다. commit log에 있는 sample 파일
        경로를 `sample_major_manifest.json`에 연결할 뿐입니다. raw 파일 schema가
        `(sample_id, feature_id, value)` long schema이므로 dense-long build가 이
        manifest를 그대로 입력으로 사용할 수 있습니다.
        """

        if self._finished_stage or os.path.exists(self.sample_major_manifest_path):
            self._finished_stage = True
            return self.sample_major_manifest_path
        stage_lock = _FileLock(self.raw_stage_lock_path, timeout_seconds=_stage_lock_timeout_seconds())
        stage_lock.acquire()
        try:
            if self._finished_stage or os.path.exists(self.sample_major_manifest_path):
                self._finished_stage = True
                self._save_state()
                return self.sample_major_manifest_path
            self._wait_for_no_active_sample_locks()
            records = self._raw_commit_records()
            if require_all:
                pending = [sample_id for sample_id in range(int(self.n_samples)) if sample_id not in records]
                if pending:
                    raise ValueError(f"cannot finish scalar stage: {len(pending)} samples are still pending")
            sample_paths = [
                os.path.relpath(os.path.join(self.out_dir, records[sample_id]["path"]), self.out_dir).replace("\\", "/")
                for sample_id in sorted(records)
            ]
            payload = {
                "format": SAMPLE_MAJOR_MANIFEST_FORMAT,
                "sample_meta_path": os.path.relpath(self.sample_meta_path, self.out_dir).replace("\\", "/"),
                "feature_meta_path": os.path.relpath(self.feature_meta_path, self.out_dir).replace("\\", "/"),
                "sample_paths": sample_paths,
                "sample_ids": [int(sample_id) for sample_id in sorted(records)],
                "sample_id_col": str(self.build_options.sample_id_col),
                "feature_id_col": str(self.build_options.feature_id_col),
                "value_col": str(self.build_options.value_col),
                "raw_sample_stage": True,
                "completed_sample_count": int(len(records)),
            }
            write_json_atomic(self.sample_major_manifest_path, payload)
            self._finished_stage = True
            self._save_state()
            return self.sample_major_manifest_path
        finally:
            stage_lock.release()

    def build_dense_long_shards(
        self,
        *,
        require_all: bool = True,
        out_dir: Optional[str] = None,
        target_part_mb: Optional[int] = None,
        row_group_features: int = 128,
        keep_raw: bool = True,
        return_stats: bool = False,
    ):
        manifest_path = self.finish_stage(require_all=bool(require_all))
        dense_out_dir = str(Path(out_dir or os.path.join(self.out_dir, "scalar_shard")).expanduser().resolve())
        result = build_dense_long_shards_from_sample_major_manifest(
            manifest_path,
            dense_out_dir,
            feature_meta_path=self.feature_meta_path,
            target_part_bytes=int(target_part_mb or self.build_options.target_shard_mb) * 1024 * 1024,
            feature_id_col=str(self.build_options.feature_id_col),
            value_col=str(self.build_options.value_col),
            sample_id_col=str(self.build_options.sample_id_col),
            sample_key_col=str(self.build_options.sample_key_col),
            feature_key_col=str(self.build_options.feature_key_col),
            y_col=str(self.build_options.y_col),
            stats_y_cols=list(self._stats_y_cols()),
            compression="zstd",
            row_group_features=int(row_group_features),
            return_stats=bool(return_stats),
        )
        if not keep_raw:
            shutil.rmtree(self.raw_samples_path, ignore_errors=True)
        return result

    def build_shards(
        self,
        *,
        require_all: bool = True,
        out_dir: Optional[str] = None,
        target_part_mb: Optional[int] = None,
        row_group_features: int = 128,
        keep_raw: bool = True,
        return_stats: bool = False,
    ):
        """Build the final dense-long shard from committed sample files."""

        return self.build_dense_long_shards(
            require_all=require_all,
            out_dir=out_dir,
            target_part_mb=target_part_mb,
            row_group_features=row_group_features,
            keep_raw=keep_raw,
            return_stats=return_stats,
        )

    def close(self):
        self._save_state()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _replace_file_with_retry(tmp_path: str, final_path: str):
    last_error: Optional[OSError] = None
    for attempt in range(FILE_REPLACE_RETRY_COUNT):
        try:
            os.replace(tmp_path, final_path)
            return
        except OSError as exc:
            last_error = exc
            if attempt == FILE_REPLACE_RETRY_COUNT - 1:
                break
            time.sleep(FILE_REPLACE_RETRY_BASE_SECONDS * float(attempt + 1))
    raise last_error or OSError(f"failed to replace {final_path!r} with {tmp_path!r}")


def _stage_lock_timeout_seconds() -> float:
    raw = os.environ.get(STAGE_LOCK_TIMEOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_STAGE_LOCK_TIMEOUT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_STAGE_LOCK_TIMEOUT_SECONDS


def _remove_file_best_effort(path: str):
    last_error: Optional[OSError] = None
    for attempt in range(FILE_REPLACE_RETRY_COUNT):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt == FILE_REPLACE_RETRY_COUNT - 1:
                break
            time.sleep(FILE_REPLACE_RETRY_BASE_SECONDS * float(attempt + 1))
    if last_error is not None:
        # Windows에서 백신/IDE/이전 디버그 프로세스가 stale tmp를 잠깐 잡고 있으면
        # cleanup 실패가 새 sample commit을 막으면 안 된다. unique tmp를 쓰므로
        # 잠긴 stale tmp는 다음 resume cleanup이나 OS 정리 대상으로 남겨 둔다.
        return


def _pyarrow_compression(compression: str) -> Optional[str]:
    value = str(compression or "").strip().lower()
    if value in {"", "none", "uncompressed"}:
        return None
    return value
