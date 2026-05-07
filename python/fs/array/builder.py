"""Resumable direct-ingestion builder for array binary shards."""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ..config import ArrayBinaryBuildOptions, ArrayBundleConfig, ArrayShardConfig
from ..types import LogicalType, PointColumnSpec, point_storage_dtype
from .binary_storage import _load_dense_meta, build_array_binary_shards_from_bundles
from .storage import ArraySampleBundleWriter, _normalize_point_schema


def _write_json_atomic(path: str, payload: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _append_jsonl(path: str, payload: dict):
    line = json.dumps(payload, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _cleanup_tmp_files(path: str):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name.endswith(".tmp"):
            try:
                os.remove(os.path.join(path, name))
            except FileNotFoundError:
                pass


@dataclass
class _CategoricalRegistry:
    """Mutable string-to-code registry for one categorical point column."""

    label_to_code: dict
    code_to_label: list

    @classmethod
    def create(cls):
        return cls(label_to_code={}, code_to_label=[])

    def encode(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=object)
        out = np.zeros(int(arr.size), dtype=np.uint32)
        for idx, value in enumerate(arr.tolist()):
            if value is None:
                out[idx] = np.uint32(0)
                continue
            label = str(value)
            code = self.label_to_code.get(label)
            if code is None:
                code = len(self.code_to_label) + 1
                self.label_to_code[label] = int(code)
                self.code_to_label.append(label)
            out[idx] = np.uint32(code)
        return out

    def to_json_dict(self, column_name: str) -> dict:
        return {
            "column": str(column_name),
            "items": [
                {"code": int(idx + 1), "label": str(label)}
                for idx, label in enumerate(self.code_to_label)
            ],
        }


@dataclass(frozen=True)
class ArrayBuildSessionStatus:
    """Resume-safe array build session snapshot."""

    last_committed_sample_id: Optional[int]
    last_committed_sample_key: Optional[str]
    next_expected_sample_id: int
    next_expected_sample_key: Optional[str]
    committed_bundle_count: int
    finished_stage: bool
    bundle_manifest_path: Optional[str]
    buffered_through_sample_id: Optional[int]
    buffered_through_sample_key: Optional[str]
    in_progress_sample_id: Optional[int]
    in_progress_sample_key: Optional[str]


class ArraySampleContext:
    """Sample-scoped helper that groups traces into one resumable sample unit."""

    def __init__(self, builder: "ArrayDatasetBuilder", sample_id: int):
        self._builder = builder
        self._sample_id = int(sample_id)

    def __enter__(self):
        self._builder._begin_sample(self._sample_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._builder._end_sample(abort=exc_type is not None)
        return False

    def add_trace(self, feature_id: Optional[int] = None, feature_key: Optional[str] = None, *, columns):
        self._builder.add_trace(
            sample_id=self._sample_id,
            feature_id=feature_id,
            feature_key=feature_key,
            columns=columns,
        )


class ArrayDatasetBuilder:
    """Resumable direct-ingestion builder for array binary shards.

    Automatic chunking remains internal. One committed bundle parquet file is
    one durable checkpoint, and bundle commits only happen after a sample
    boundary so resume positions stay easy to reason about.
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
        build_options: ArrayBinaryBuildOptions | None = None,
        bundle_config: ArrayBundleConfig | None = None,
        bundle_out_dir: Optional[str] = None,
    ):
        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.sample_meta_path = str(Path(sample_meta_path).expanduser().resolve())
        self.build_options = build_options or ArrayBinaryBuildOptions()
        self.bundle_config = bundle_config or ArrayBundleConfig()
        self.point_schema = _normalize_point_schema(point_schema)
        self._closed = False
        self._finished = False
        self._bundles_finalized = False
        self._manifest_path = None

        sample_meta = _load_dense_meta(
            self.sample_meta_path,
            "sample_id",
            "sample",
            str(self.build_options.sample_key_col),
        )
        self.n_samples = int(sample_meta.height)
        self._sample_key_col = str(self.build_options.sample_key_col)
        self._sample_keys = (
            [None] * self.n_samples
            if self._sample_key_col not in sample_meta.columns
            else [
                None if sample_key is None else str(sample_key)
                for sample_key in sample_meta[self._sample_key_col].to_list()
            ]
        )
        self._sample_key_to_id = None
        if self._sample_key_col and self._sample_key_col in sample_meta.columns:
            self._sample_key_to_id = {
                str(sample_key): int(idx)
                for idx, sample_key in enumerate(sample_meta[self._sample_key_col].to_list())
                if sample_key is not None
            }

        default_bundle_root = os.path.join(self.out_dir, "bundle_stage")
        self.bundle_out_dir = str(Path(bundle_out_dir or default_bundle_root).expanduser().resolve())
        self._state_path = os.path.join(self.bundle_out_dir, "state.json")
        self._bundle_log_path = os.path.join(self.bundle_out_dir, "bundles.jsonl")
        self._tmp_dir = os.path.join(self.bundle_out_dir, "tmp")
        self._feature_meta_path = os.path.join(self.bundle_out_dir, "feature_meta.parquet")
        self._bundle_sample_meta_path = os.path.join(self.bundle_out_dir, "sample_meta.parquet")

        self._feature_meta_source_path = ""
        self._feature_key_to_id: OrderedDict[str, int] = OrderedDict()
        self._feature_keys_in_order: list[str] = []
        self._known_feature_mode = False
        self._known_feature_count: Optional[int] = None
        self._writes_feature_meta = False

        self._categorical_registries = {
            spec.name: _CategoricalRegistry.create()
            for spec in self.point_schema
            if spec.logical_type == LogicalType.CATEGORICAL
        }

        self._bundle_writer: Optional[ArraySampleBundleWriter] = None
        self.bundle_manifest_path = os.path.join(self.bundle_out_dir, "array_bundle_manifest.json")
        self.bundle_path = os.path.join(self.bundle_out_dir, "array_sample_bundles")

        self._last_committed_sample_id: Optional[int] = None
        self._cursor_sample_id = 0
        self._committed_bundle_count = 0
        self._pending_bundle_first_sample_id: Optional[int] = None
        self._pending_bundle_last_sample_id: Optional[int] = None
        self._pending_bundle_sample_count = 0
        self._pending_bundle_trace_count = 0
        self._open_sample_id: Optional[int] = None
        self._current_sample_trace_count = 0

        if os.path.exists(self._state_path):
            self._resume_stage(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        else:
            self._initialize_new_stage(feature_meta_path=feature_meta_path, feature_keys=feature_keys)

    @classmethod
    def open_session(cls, *args, **kwargs) -> "ArrayDatasetBuilder":
        """Open a new array build session or resume an existing one."""

        return cls(*args, **kwargs)

    @staticmethod
    def _prepare_empty_dir(path: str):
        if os.path.exists(path):
            if os.path.isdir(path) and not os.listdir(path):
                return
            raise ValueError(f"bundle_out_dir already exists and is not empty: {path}")
        os.makedirs(path, exist_ok=True)

    def _build_options_payload(self) -> dict:
        return asdict(self.build_options)

    def _point_schema_payload(self) -> list[dict]:
        return [spec.to_json() for spec in self.point_schema]

    def _sample_key_for_id(self, sample_id: Optional[int]) -> Optional[str]:
        if sample_id is None:
            return None
        if sample_id < 0 or sample_id >= self.n_samples:
            return None
        return self._sample_keys[int(sample_id)]

    def _resume_next_sample_id(self) -> int:
        if self._last_committed_sample_id is None:
            return 0
        return int(self._last_committed_sample_id) + 1

    def _bundle_relative_path(self, bundle_path: str) -> str:
        return os.path.relpath(bundle_path, self.bundle_out_dir)

    def _stage_state_payload(self) -> dict:
        return {
            "format_version": self._STATE_VERSION,
            "stage_type": "array_bundle_stage_v1",
            "sample_meta_path": self.sample_meta_path,
            "feature_meta_source_path": self._feature_meta_source_path,
            "build_options": self._build_options_payload(),
            "point_schema": self._point_schema_payload(),
            "known_feature_mode": bool(self._known_feature_mode),
            "writes_feature_meta": bool(self._writes_feature_meta),
            "known_feature_count": self._known_feature_count,
            "feature_keys_in_order": list(self._feature_keys_in_order),
            "next_bundle_id": int(self._bundle_writer.n_bundles if self._bundle_writer is not None else self._committed_bundle_count),
            "last_committed_sample_id": self._last_committed_sample_id,
            "last_committed_sample_key": self._sample_key_for_id(self._last_committed_sample_id),
            "next_expected_sample_id": self._resume_next_sample_id(),
            "next_expected_sample_key": self._sample_key_for_id(self._resume_next_sample_id()),
            "committed_bundle_count": int(self._committed_bundle_count),
            "finished_stage": bool(self._bundles_finalized),
            "bundle_manifest_path": self.bundle_manifest_path if self._bundles_finalized else None,
        }

    def _save_state(self):
        os.makedirs(self.bundle_out_dir, exist_ok=True)
        _write_json_atomic(self._state_path, self._stage_state_payload())

    def _cleanup_stage_tmp(self):
        _cleanup_tmp_files(self.bundle_out_dir)
        _cleanup_tmp_files(self._tmp_dir)
        _cleanup_tmp_files(self.bundle_path)

    def _validate_resume_state(self, state: dict, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if str(state.get("stage_type")) != "array_bundle_stage_v1":
            raise ValueError(f"unsupported array build session type: {state.get('stage_type')}")
        if str(Path(state["sample_meta_path"]).expanduser().resolve()) != self.sample_meta_path:
            raise ValueError("sample_meta_path does not match existing array build session")
        if state.get("build_options") != self._build_options_payload():
            raise ValueError("build_options do not match existing array build session")
        if state.get("point_schema") != self._point_schema_payload():
            raise ValueError("point_schema does not match existing array build session")

        stored_source = str(state.get("feature_meta_source_path") or "")
        normalized_feature_meta_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        if normalized_feature_meta_path and stored_source and normalized_feature_meta_path != stored_source:
            raise ValueError("feature_meta_path does not match existing array build session")
        if feature_keys is not None and list(map(str, feature_keys)) != list(state.get("feature_keys_in_order") or []):
            raise ValueError("feature_keys do not match existing array build session")

    def _initialize_new_stage(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        self._prepare_empty_dir(self.bundle_out_dir)
        os.makedirs(self._tmp_dir, exist_ok=True)
        shutil.copy2(self.sample_meta_path, self._bundle_sample_meta_path)

        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")

        if feature_meta_path:
            self._known_feature_mode = True
            self._feature_meta_source_path = str(Path(feature_meta_path).expanduser().resolve())
            feature_meta = _load_dense_meta(
                self._feature_meta_source_path,
                "feature_id",
                "feature",
                str(self.build_options.feature_key_col),
            )
            shutil.copy2(self._feature_meta_source_path, self._feature_meta_path)
            self._known_feature_count = int(feature_meta.height)
            key_col = str(self.build_options.feature_key_col)
            if key_col and key_col in feature_meta.columns:
                for idx, value in enumerate(feature_meta[key_col].to_list()):
                    key = str(value)
                    self._feature_key_to_id[key] = int(idx)
                    self._feature_keys_in_order.append(key)
        elif feature_keys is not None:
            self._known_feature_mode = True
            self._writes_feature_meta = True
            for idx, feature_key in enumerate(feature_keys):
                key = str(feature_key)
                if key in self._feature_key_to_id:
                    raise ValueError(f"duplicate feature key: {key}")
                self._feature_key_to_id[key] = int(idx)
                self._feature_keys_in_order.append(key)
            self._known_feature_count = int(len(self._feature_keys_in_order))
        else:
            self._writes_feature_meta = True

        self._bundle_writer = ArraySampleBundleWriter(
            self.bundle_out_dir,
            self._bundle_sample_meta_path,
            n_samples=self.n_samples,
            feature_meta_path=self._feature_meta_path,
            config=self.bundle_config,
            point_schema=self.point_schema,
            start_bundle_id=0,
            auto_flush=False,
        )
        self.bundle_manifest_path = self._bundle_writer.manifest_path
        self.bundle_path = self._bundle_writer.bundle_path
        self._save_state()

    def _resume_stage(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        with open(self._state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._validate_resume_state(state, feature_meta_path, feature_keys)
        self._cleanup_stage_tmp()

        self._known_feature_mode = bool(state.get("known_feature_mode"))
        self._writes_feature_meta = bool(state.get("writes_feature_meta"))
        self._known_feature_count = state.get("known_feature_count")
        self._feature_meta_source_path = str(state.get("feature_meta_source_path") or "")
        self._feature_keys_in_order = [str(value) for value in state.get("feature_keys_in_order") or []]
        self._feature_key_to_id = OrderedDict(
            (feature_key, int(idx)) for idx, feature_key in enumerate(self._feature_keys_in_order)
        )

        bundle_records = _load_jsonl(self._bundle_log_path)
        self._committed_bundle_count = len(bundle_records)
        self._last_committed_sample_id = None if not bundle_records else int(bundle_records[-1]["last_sample_id"])
        self._cursor_sample_id = int(self._resume_next_sample_id())
        self._bundles_finalized = bool(state.get("finished_stage"))

        next_bundle_id = int(state.get("next_bundle_id", self._committed_bundle_count))
        self._bundle_writer = ArraySampleBundleWriter(
            self.bundle_out_dir,
            self._bundle_sample_meta_path,
            n_samples=self.n_samples,
            feature_meta_path=self._feature_meta_path,
            config=self.bundle_config,
            point_schema=self.point_schema,
            start_bundle_id=next_bundle_id,
            auto_flush=False,
        )
        self.bundle_manifest_path = self._bundle_writer.manifest_path
        self.bundle_path = self._bundle_writer.bundle_path
        if self._bundles_finalized:
            self._manifest_path = self.bundle_manifest_path

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("array dataset builder is closed")
        if self._finished:
            raise RuntimeError("array dataset builder is already finished")

    def _ensure_trace_stage_open(self):
        self._ensure_open()
        if self._bundles_finalized:
            raise RuntimeError("bundle stage has already been finalized")

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

        if self._sample_key_to_id is None:
            raise ValueError(f"sample metadata does not expose sample keys: {self._sample_key_col}")
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
            if self._known_feature_mode and self._known_feature_count is not None and feature_id >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {feature_id}")
            if not self._known_feature_mode:
                raise ValueError("discovered-feature mode requires feature_key inputs")
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

    def _normalize_columns(self, *, columns):
        if columns is None:
            raise ValueError("columns is required")
        expected_names = {spec.name for spec in self.point_schema}
        actual_names = {str(name) for name in dict(columns).keys()}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"point columns must exactly match point_schema ({', '.join(details)})")
        out = {}
        expected_length = None
        for spec in self.point_schema:
            values = columns[spec.name]
            if spec.logical_type == LogicalType.CATEGORICAL:
                arr = self._categorical_registries[spec.name].encode(values)
            elif spec.logical_type == LogicalType.TIMESTAMP_NS:
                arr = np.asarray(values, dtype="datetime64[ns]").reshape(-1).astype(
                    point_storage_dtype(spec.storage_type), copy=False
                )
            elif spec.logical_type == LogicalType.TIMEDELTA_NS:
                arr = np.asarray(values, dtype="timedelta64[ns]").reshape(-1).astype(
                    point_storage_dtype(spec.storage_type), copy=False
                )
            else:
                arr = np.asarray(values, dtype=point_storage_dtype(spec.storage_type))
            if arr.ndim != 1:
                raise ValueError(f"point column must be 1D: {spec.name}")
            if expected_length is None:
                expected_length = int(arr.shape[0])
            elif int(arr.shape[0]) != int(expected_length):
                raise ValueError(
                    f"point column length mismatch for {spec.name}: expected={expected_length} got={arr.shape[0]}"
                )
            out[spec.name] = arr
        return out

    def _begin_sample(self, sample_id: int):
        self._ensure_trace_stage_open()
        sample_id = int(sample_id)
        if self._open_sample_id is not None:
            if int(self._open_sample_id) == sample_id:
                return
            raise RuntimeError("another sample is already open")
        if sample_id != int(self._cursor_sample_id):
            raise ValueError(
                f"array session expects sample_id {self._cursor_sample_id}; got {sample_id}. "
                "Resume from status().next_expected_sample_id and process samples sequentially."
            )
        self._open_sample_id = sample_id
        self._current_sample_trace_count = 0

    def _record_processed_sample(self, sample_id: int, *, trace_count: int):
        if self._pending_bundle_first_sample_id is None:
            self._pending_bundle_first_sample_id = int(sample_id)
        self._pending_bundle_last_sample_id = int(sample_id)
        self._pending_bundle_sample_count += 1
        self._pending_bundle_trace_count += int(trace_count)
        self._cursor_sample_id = int(sample_id) + 1

    def _commit_pending_bundle(self, *, force: bool):
        if self._bundle_writer is None:
            return None
        if not force and not self._bundle_writer.should_flush_bundle():
            return None
        commit = self._bundle_writer.flush_bundle()
        if commit is None:
            return None
        record = {
            "bundle_id": int(commit["bundle_id"]),
            "path": self._bundle_relative_path(str(commit["path"])),
            "first_sample_id": self._pending_bundle_first_sample_id,
            "last_sample_id": self._pending_bundle_last_sample_id,
            "first_sample_key": self._sample_key_for_id(self._pending_bundle_first_sample_id),
            "last_sample_key": self._sample_key_for_id(self._pending_bundle_last_sample_id),
            "sample_count": int(self._pending_bundle_sample_count),
            "trace_count": int(self._pending_bundle_trace_count),
            "row_count": int(commit["row_count"]),
            "byte_size": int(commit["byte_size"]),
        }
        _append_jsonl(self._bundle_log_path, record)
        self._committed_bundle_count += 1
        self._last_committed_sample_id = self._pending_bundle_last_sample_id
        self._pending_bundle_first_sample_id = None
        self._pending_bundle_last_sample_id = None
        self._pending_bundle_sample_count = 0
        self._pending_bundle_trace_count = 0
        self._save_state()
        return record

    def _end_sample(self, *, abort: bool):
        if self._open_sample_id is None:
            return
        sample_id = int(self._open_sample_id)
        trace_count = int(self._current_sample_trace_count)
        self._open_sample_id = None
        self._current_sample_trace_count = 0
        if abort:
            return
        self._record_processed_sample(sample_id, trace_count=trace_count)
        self._commit_pending_bundle(force=False)

    def sample(self, sample_id: Optional[int] = None, sample_key: Optional[str] = None) -> ArraySampleContext:
        self._ensure_trace_stage_open()
        return ArraySampleContext(self, self._resolve_sample_id(sample_id, sample_key))

    def add_trace(
        self,
        *,
        sample_id: Optional[int] = None,
        sample_key: Optional[str] = None,
        feature_id: Optional[int] = None,
        feature_key: Optional[str] = None,
        columns,
    ):
        self._ensure_trace_stage_open()
        resolved_sample_id = self._resolve_sample_id(sample_id, sample_key)
        if self._open_sample_id is None:
            self._begin_sample(resolved_sample_id)
        elif int(self._open_sample_id) != int(resolved_sample_id):
            raise RuntimeError(
                "add_trace(...) crossed a sample boundary without closing the previous sample. "
                "Use builder.sample(sample_id=...) contexts or process traces for each sample together."
            )

        resolved_feature_id = self._resolve_feature_id(feature_id, feature_key)
        normalized_columns = self._normalize_columns(columns=columns)
        if self._bundle_writer is None:
            raise RuntimeError("array bundle writer is not initialized")
        self._bundle_writer.append_trace(
            sample_id=resolved_sample_id,
            feature_id=resolved_feature_id,
            columns=normalized_columns,
        )
        self._current_sample_trace_count += 1

    def _write_feature_meta(self):
        if not self._writes_feature_meta:
            return
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {"feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32)}
        key_col = str(self.build_options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self._feature_meta_path)

    def _write_categorical_dictionaries(self):
        dict_root = os.path.join(self.bundle_out_dir, "categorical_dictionaries")
        os.makedirs(dict_root, exist_ok=True)
        for idx, spec in enumerate(self.point_schema):
            if spec.logical_type != LogicalType.CATEGORICAL:
                continue
            registry = self._categorical_registries[spec.name]
            dict_path = os.path.join(dict_root, f"{spec.name}.json")
            _write_json_atomic(dict_path, registry.to_json_dict(spec.name))
            spec.dictionary_path = dict_path
            if self._bundle_writer is not None:
                self._bundle_writer.point_schema[idx].dictionary_path = dict_path

    def update_feature_meta(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        on: Optional[str] = None,
        require_all: bool = False,
    ) -> str:
        self.finish_stage()
        base = pl.read_parquet(self._feature_meta_path)
        join_col = str(on or "")
        if not join_col:
            key_col = str(self.build_options.feature_key_col)
            join_col = key_col if key_col and key_col in base.columns else "feature_id"
        if join_col not in base.columns:
            raise ValueError(f"feature metadata join column not found: {join_col}")

        rows = [dict(record) for record in records]
        if not rows:
            return self._feature_meta_path
        updates = pl.from_dicts(rows, infer_schema_length=None)
        if join_col not in updates.columns:
            raise ValueError(f"update records must include join column: {join_col}")

        update_keys = updates[join_col].to_list()
        seen_keys = set()
        for row_idx, value in enumerate(update_keys):
            if value is None:
                raise ValueError(f"feature metadata join column {join_col} cannot be null at row {row_idx}")
            key = str(value) if join_col != "feature_id" else int(value)
            if key in seen_keys:
                raise ValueError(f"duplicate feature metadata join key: {value}")
            seen_keys.add(key)

        overlapping = [name for name in updates.columns if name != join_col and name in base.columns]
        if overlapping:
            raise ValueError(
                f"feature metadata updates must add new columns only; overlapping columns: {', '.join(overlapping)}"
            )

        merged = base.join(updates, on=join_col, how="left")
        if require_all:
            new_columns = [name for name in updates.columns if name != join_col]
            for column_name in new_columns:
                if merged[column_name].null_count() > 0:
                    raise ValueError(f"missing values remain in required feature metadata column: {column_name}")

        merged.write_parquet(self._feature_meta_path)
        return self._feature_meta_path

    def status(self) -> ArrayBuildSessionStatus:
        next_expected = self._resume_next_sample_id()
        return ArrayBuildSessionStatus(
            last_committed_sample_id=self._last_committed_sample_id,
            last_committed_sample_key=self._sample_key_for_id(self._last_committed_sample_id),
            next_expected_sample_id=next_expected,
            next_expected_sample_key=self._sample_key_for_id(next_expected),
            committed_bundle_count=int(self._committed_bundle_count),
            finished_stage=bool(self._bundles_finalized),
            bundle_manifest_path=self.bundle_manifest_path if self._bundles_finalized else None,
            buffered_through_sample_id=self._pending_bundle_last_sample_id,
            buffered_through_sample_key=self._sample_key_for_id(self._pending_bundle_last_sample_id),
            in_progress_sample_id=self._open_sample_id,
            in_progress_sample_key=self._sample_key_for_id(self._open_sample_id),
        )

    def finish_stage(self):
        if self._bundles_finalized:
            return self.bundle_manifest_path
        self._ensure_open()
        self._end_sample(abort=False)
        self._commit_pending_bundle(force=True)
        self._write_feature_meta()
        self._write_categorical_dictionaries()
        if self._bundle_writer is None:
            raise RuntimeError("array bundle writer is not initialized")
        self.bundle_manifest_path = self._bundle_writer.finish()
        self.bundle_path = self._bundle_writer.bundle_path
        self._bundles_finalized = True
        self._save_state()
        return self.bundle_manifest_path

    def finish_bundles(self):
        """Legacy alias for `finish_stage()`."""

        return self.finish_stage()

    def build_shards(self, *, cleanup_bundles: bool = False, return_stats: bool = False):
        if self._finished:
            if return_stats:
                raise RuntimeError("array dataset builder has already built shards; build stats are no longer available")
            return self._manifest_path
        self._ensure_open()
        bundle_manifest_path = self.finish_stage()
        config = ArrayShardConfig(
            samples_per_block=int(self.build_options.samples_per_block),
            target_shard_bytes=int(self.build_options.target_shard_mb) * 1024 * 1024,
            n_shards=0 if self.build_options.n_shards is None else int(self.build_options.n_shards),
            row_group_size=0,
            use_tmp_spill=False,
            spill_bucket_target_bytes=8 * 1024 * 1024,
        )
        result = build_array_binary_shards_from_bundles(
            bundle_manifest_path,
            self.out_dir,
            config=config,
            codec=str(self.build_options.codec),
            zstd_level=int(self.build_options.zstd_level),
            sample_key_col=str(self.build_options.sample_key_col),
            feature_key_col=str(self.build_options.feature_key_col),
            return_stats=bool(return_stats),
        )
        if cleanup_bundles and os.path.exists(self.bundle_out_dir):
            shutil.rmtree(self.bundle_out_dir)
        self._finished = True
        self._closed = True
        self._manifest_path = result[0] if isinstance(result, tuple) else result
        return result

    def finish(self, *, cleanup_bundles: bool = False, return_stats: bool = False):
        return self.build_shards(cleanup_bundles=cleanup_bundles, return_stats=return_stats)

    def close(self):
        """Close the builder without discarding committed checkpoints."""

        if self._closed:
            return
        self._end_sample(abort=False)
        self._commit_pending_bundle(force=True)
        self._save_state()
        self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
