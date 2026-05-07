"""Resumable sample-scoped builder for scalar feature shards."""

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

from ..config import ScalarShardBuildOptions
from .parquet_storage import build_shards_from_sample_bundles


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


def _load_dense_metadata(
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


def _prepare_empty_dir(path: str):
    """Create an empty directory or fail when it already contains files."""

    if os.path.exists(path):
        if os.path.isdir(path) and not os.listdir(path):
            return
        raise ValueError(f"sample_major_out_dir already exists and is not empty: {path}")
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class ScalarBuildSessionStatus:
    """Resume-safe scalar build session snapshot.

    `last_committed_sample_id` and `next_expected_sample_id` are the important
    fields for restart logic. Anything after `buffered_through_sample_id` may
    still be sitting in memory and can be lost on a crash.
    """

    last_committed_sample_id: Optional[int]
    last_committed_sample_key: Optional[str]
    next_expected_sample_id: int
    next_expected_sample_key: Optional[str]
    committed_bundle_count: int
    finished_stage: bool
    bundle_manifest_path: Optional[str]
    buffered_through_sample_id: Optional[int]
    buffered_through_sample_key: Optional[str]


class ScalarDatasetBuilder:
    """Sample-scoped scalar builder with automatic bundle checkpoints.

    The builder still chunks automatically, but every committed checkpoint is
    one sealed bundle parquet file. Users resume from `status()` and continue
    with `write_sample(...)` starting at `next_expected_sample_id`.
    """

    _DEFAULT_BUNDLE_FLUSH_ROWS = 1_000_000
    _STATE_VERSION = 1

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        *,
        feature_meta_path: Optional[str] = None,
        feature_keys: Optional[Sequence[str]] = None,
        build_options: ScalarShardBuildOptions | None = None,
        sample_major_out_dir: Optional[str] = None,
    ):
        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.source_sample_meta_path = str(Path(sample_meta_path).expanduser().resolve())
        self.build_options = build_options or ScalarShardBuildOptions()
        self._closed = False
        self._sample_major_finalized = False
        self._shards_built = False
        self._manifest_path = None

        sample_meta_df = _load_dense_metadata(
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

        default_sample_major_root = os.path.join(self.out_dir, "sample_major_stage")
        self.sample_major_out_dir = str(Path(sample_major_out_dir or default_sample_major_root).expanduser().resolve())
        self._bundle_files_dir = os.path.join(self.sample_major_out_dir, "sample_bundles")
        self._tmp_dir = os.path.join(self.sample_major_out_dir, "tmp")
        self._state_path = os.path.join(self.sample_major_out_dir, "state.json")
        self._bundle_log_path = os.path.join(self.sample_major_out_dir, "bundles.jsonl")

        self.sample_major_manifest_path = os.path.join(self.sample_major_out_dir, "sample_major_manifest.json")
        self.sample_major_sample_meta_path = os.path.join(self.sample_major_out_dir, "sample_meta.parquet")
        self.sample_major_feature_meta_path = os.path.join(self.sample_major_out_dir, "feature_meta.parquet")

        self._feature_meta_source_path = ""
        self._feature_key_to_id: OrderedDict[str, int] = OrderedDict()
        self._feature_keys_in_order: list[str] = []
        self._known_feature_mode = False
        self._known_feature_count: Optional[int] = None
        self._writes_feature_meta = False

        self._bundle_paths: list[str] = []
        self._bundle_index = 0
        self._bundle_sample_id_chunks: list[np.ndarray] = []
        self._bundle_feature_id_chunks: list[np.ndarray] = []
        self._bundle_value_chunks: list[np.ndarray] = []
        self._bundle_row_count = 0
        self._bundle_flush_rows_target = int(self._DEFAULT_BUNDLE_FLUSH_ROWS)

        self._last_committed_sample_id: Optional[int] = None
        self._cursor_sample_id = 0
        self._committed_bundle_count = 0
        self._pending_bundle_first_sample_id: Optional[int] = None
        self._pending_bundle_last_sample_id: Optional[int] = None
        self._pending_bundle_sample_count = 0
        self._pending_bundle_trace_count = 0

        if os.path.exists(self._state_path):
            self._resume_stage(feature_meta_path=feature_meta_path, feature_keys=feature_keys)
        else:
            self._initialize_new_stage(feature_meta_path=feature_meta_path, feature_keys=feature_keys)

    @classmethod
    def open_session(cls, *args, **kwargs) -> "ScalarDatasetBuilder":
        """Open a new scalar build session or resume an existing one."""

        return cls(*args, **kwargs)

    def _stats_y_cols(self) -> list[str]:
        values = self.build_options.stats_y_cols
        if not values:
            return [str(self.build_options.y_col)]
        ordered: list[str] = []
        for value in values:
            name = str(value)
            if name and name not in ordered:
                ordered.append(name)
        if not ordered:
            raise ValueError("at least one stats y column is required")
        return ordered

    def _build_options_payload(self) -> dict:
        payload = asdict(self.build_options)
        if payload.get("stats_y_cols") is not None:
            payload["stats_y_cols"] = list(payload["stats_y_cols"])
        return payload

    def _stage_state_payload(self) -> dict:
        return {
            "format_version": self._STATE_VERSION,
            "stage_type": "scalar_bundle_stage_v1",
            "sample_meta_path": self.source_sample_meta_path,
            "feature_meta_source_path": self._feature_meta_source_path,
            "build_options": self._build_options_payload(),
            "known_feature_mode": bool(self._known_feature_mode),
            "writes_feature_meta": bool(self._writes_feature_meta),
            "known_feature_count": self._known_feature_count,
            "feature_keys_in_order": list(self._feature_keys_in_order),
            "next_bundle_id": int(self._bundle_index),
            "last_committed_sample_id": self._last_committed_sample_id,
            "last_committed_sample_key": self._sample_key_for_id(self._last_committed_sample_id),
            "next_expected_sample_id": self._resume_next_sample_id(),
            "next_expected_sample_key": self._sample_key_for_id(self._resume_next_sample_id()),
            "committed_bundle_count": int(self._committed_bundle_count),
            "finished_stage": bool(self._sample_major_finalized),
            "bundle_manifest_path": self.sample_major_manifest_path if self._sample_major_finalized else None,
        }

    def _save_state(self):
        os.makedirs(self.sample_major_out_dir, exist_ok=True)
        _write_json_atomic(self._state_path, self._stage_state_payload())

    def _resume_next_sample_id(self) -> int:
        if self._last_committed_sample_id is None:
            return 0
        return int(self._last_committed_sample_id) + 1

    def _sample_key_for_id(self, sample_id: Optional[int]) -> Optional[str]:
        if sample_id is None:
            return None
        if sample_id < 0 or sample_id >= self.n_samples:
            return None
        return self._sample_keys[int(sample_id)]

    def _bundle_relative_path(self, bundle_path: str) -> str:
        return os.path.relpath(bundle_path, self.sample_major_out_dir)

    def _cleanup_stage_tmp(self):
        _cleanup_tmp_files(self.sample_major_out_dir)
        _cleanup_tmp_files(self._tmp_dir)
        _cleanup_tmp_files(self._bundle_files_dir)

    def _validate_resume_state(self, state: dict, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        if str(state.get("stage_type")) != "scalar_bundle_stage_v1":
            raise ValueError(f"unsupported scalar build session type: {state.get('stage_type')}")
        if str(Path(state["sample_meta_path"]).expanduser().resolve()) != self.source_sample_meta_path:
            raise ValueError("sample_meta_path does not match existing scalar build session")
        if state.get("build_options") != self._build_options_payload():
            raise ValueError("build_options do not match existing scalar build session")

        stored_source = str(state.get("feature_meta_source_path") or "")
        normalized_feature_meta_path = "" if not feature_meta_path else str(Path(feature_meta_path).expanduser().resolve())
        if normalized_feature_meta_path and stored_source and normalized_feature_meta_path != stored_source:
            raise ValueError("feature_meta_path does not match existing scalar build session")
        if feature_keys is not None and list(map(str, feature_keys)) != list(state.get("feature_keys_in_order") or []):
            raise ValueError("feature_keys do not match existing scalar build session")

    def _initialize_new_stage(self, *, feature_meta_path: Optional[str], feature_keys: Optional[Sequence[str]]):
        _prepare_empty_dir(self.sample_major_out_dir)
        os.makedirs(self._bundle_files_dir, exist_ok=True)
        os.makedirs(self._tmp_dir, exist_ok=True)
        shutil.copy2(self.source_sample_meta_path, self.sample_major_sample_meta_path)

        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")

        if feature_meta_path:
            self._known_feature_mode = True
            self._feature_meta_source_path = str(Path(feature_meta_path).expanduser().resolve())
            feature_meta_df = _load_dense_metadata(
                self._feature_meta_source_path,
                id_col=str(self.build_options.feature_id_col),
                entity_name="feature",
                key_col=str(self.build_options.feature_key_col),
            )
            self._known_feature_count = int(feature_meta_df.height)
            shutil.copy2(self._feature_meta_source_path, self.sample_major_feature_meta_path)
            key_col = str(self.build_options.feature_key_col)
            if key_col and key_col in feature_meta_df.columns:
                for idx, value in enumerate(feature_meta_df[key_col].to_list()):
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
        self._bundle_paths = [
            str(Path(self.sample_major_out_dir, str(record["path"])).resolve())
            for record in bundle_records
        ]
        self._bundle_index = int(state.get("next_bundle_id", len(bundle_records)))
        self._committed_bundle_count = len(bundle_records)
        self._last_committed_sample_id = None if not bundle_records else int(bundle_records[-1]["last_sample_id"])
        self._cursor_sample_id = int(self._resume_next_sample_id())
        self._sample_major_finalized = bool(state.get("finished_stage"))
        if self._sample_major_finalized:
            self._manifest_path = self.sample_major_manifest_path

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("scalar dataset builder is closed")
        if self._shards_built:
            raise RuntimeError("scalar dataset builder has already built shards")

    def _ensure_sample_major_open(self):
        self._ensure_open()
        if self._sample_major_finalized:
            raise RuntimeError("sample-major stage has already been finalized")

    def _normalize_scalar_value(self, value) -> Optional[float]:
        if value is None:
            return None
        scalar = float(value)
        if np.isnan(scalar):
            return None
        return scalar

    def _resolve_feature_id(self, feature_ref) -> int:
        if isinstance(feature_ref, (int, np.integer)):
            resolved = int(feature_ref)
            if resolved < 0:
                raise ValueError(f"feature_id out of range: {resolved}")
            if not self._known_feature_mode:
                raise ValueError("feature_id cannot be used in discovered-feature mode; use feature_key instead")
            if self._known_feature_count is not None and resolved >= int(self._known_feature_count):
                raise ValueError(f"feature_id out of range: {resolved}")
            return resolved

        key = str(feature_ref)
        resolved = self._feature_key_to_id.get(key)
        if resolved is not None:
            return int(resolved)
        if self._known_feature_mode:
            raise ValueError(f"unknown feature key: {key}")
        resolved = len(self._feature_keys_in_order)
        self._feature_key_to_id[key] = int(resolved)
        self._feature_keys_in_order.append(key)
        return int(resolved)

    def _record_processed_sample(self, sample_id: int, *, row_count: int):
        if self._pending_bundle_first_sample_id is None:
            self._pending_bundle_first_sample_id = int(sample_id)
        self._pending_bundle_last_sample_id = int(sample_id)
        self._pending_bundle_sample_count += 1
        self._pending_bundle_trace_count += int(row_count)
        self._cursor_sample_id = int(sample_id) + 1

    def _flush_bundle(self):
        if self._bundle_row_count <= 0:
            return None
        bundle_id = int(self._bundle_index)
        final_path = os.path.join(self._bundle_files_dir, f"bundle_{bundle_id:06d}.parquet")
        tmp_path = final_path + ".tmp"
        sample_ids = np.concatenate(self._bundle_sample_id_chunks).astype(np.int64, copy=False)
        feature_ids = np.concatenate(self._bundle_feature_id_chunks).astype(np.int32, copy=False)
        values = np.concatenate(self._bundle_value_chunks).astype(np.float64, copy=False)
        pl.DataFrame(
            {
                str(self.build_options.sample_id_col): pl.Series(
                    str(self.build_options.sample_id_col), sample_ids, dtype=pl.Int64
                ),
                str(self.build_options.feature_id_col): pl.Series(
                    str(self.build_options.feature_id_col), feature_ids, dtype=pl.Int32
                ),
                str(self.build_options.value_col): pl.Series(
                    str(self.build_options.value_col), values, dtype=pl.Float64
                ),
            }
        ).write_parquet(tmp_path)
        os.replace(tmp_path, final_path)
        row_count = int(self._bundle_row_count)
        byte_size = int(os.path.getsize(final_path))
        self._bundle_paths.append(final_path)
        self._bundle_index += 1
        self._bundle_sample_id_chunks.clear()
        self._bundle_feature_id_chunks.clear()
        self._bundle_value_chunks.clear()
        self._bundle_row_count = 0
        return {
            "bundle_id": bundle_id,
            "path": final_path,
            "row_count": row_count,
            "byte_size": byte_size,
        }

    def _commit_pending_bundle(self, *, force: bool):
        if not force and self._bundle_row_count < int(self._bundle_flush_rows_target):
            return None
        commit = self._flush_bundle()
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

    def _append_sample_rows(self, sample_id: int, feature_values: Mapping[int, float]):
        row_count = 0
        if feature_values:
            feature_ids = np.asarray(sorted(feature_values.keys()), dtype=np.int32)
            values = np.asarray([float(feature_values[int(fid)]) for fid in feature_ids], dtype=np.float64)
            sample_ids = np.full(feature_ids.shape[0], int(sample_id), dtype=np.int64)
            self._bundle_sample_id_chunks.append(sample_ids)
            self._bundle_feature_id_chunks.append(feature_ids)
            self._bundle_value_chunks.append(values)
            row_count = int(feature_ids.shape[0])
            self._bundle_row_count += row_count
        self._record_processed_sample(int(sample_id), row_count=row_count)
        self._commit_pending_bundle(force=False)

    def _copy_sample_meta(self):
        if os.path.normcase(os.path.abspath(self.sample_major_sample_meta_path)) != os.path.normcase(
            os.path.abspath(self.source_sample_meta_path)
        ):
            shutil.copy2(self.source_sample_meta_path, self.sample_major_sample_meta_path)

    def _write_feature_meta(self):
        if self._feature_meta_source_path:
            shutil.copy2(self._feature_meta_source_path, self.sample_major_feature_meta_path)
            return
        if not self._writes_feature_meta:
            return
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {
            str(self.build_options.feature_id_col): pl.Series(
                str(self.build_options.feature_id_col),
                feature_ids,
                dtype=pl.Int32,
            )
        }
        key_col = str(self.build_options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self.sample_major_feature_meta_path)

    def _write_sample_major_manifest(self):
        payload = {
            "format": "scalar-sample-bundles",
            "version": 1,
            "sample_meta_path": "sample_meta.parquet",
            "feature_meta_path": "feature_meta.parquet",
            "bundle_paths": [self._bundle_relative_path(path) for path in self._bundle_paths],
            "sample_id_col": str(self.build_options.sample_id_col),
            "feature_id_col": str(self.build_options.feature_id_col),
            "value_col": str(self.build_options.value_col),
        }
        _write_json_atomic(self.sample_major_manifest_path, payload)

    def status(self) -> ScalarBuildSessionStatus:
        """Return the current resume-safe scalar build status."""

        next_expected = self._resume_next_sample_id()
        return ScalarBuildSessionStatus(
            last_committed_sample_id=self._last_committed_sample_id,
            last_committed_sample_key=self._sample_key_for_id(self._last_committed_sample_id),
            next_expected_sample_id=next_expected,
            next_expected_sample_key=self._sample_key_for_id(next_expected),
            committed_bundle_count=int(self._committed_bundle_count),
            finished_stage=bool(self._sample_major_finalized),
            bundle_manifest_path=self.sample_major_manifest_path if self._sample_major_finalized else None,
            buffered_through_sample_id=self._pending_bundle_last_sample_id,
            buffered_through_sample_key=self._sample_key_for_id(self._pending_bundle_last_sample_id),
        )

    def write_sample(self, sample_id: int, values: Mapping):
        """Write one complete sample.

        Scalar session ingestion is intentionally sample-scoped. This keeps the
        API aligned with resume semantics and avoids the very slow per-value
        public write path.
        """

        self._ensure_sample_major_open()
        sample_id = int(sample_id)
        if sample_id != int(self._cursor_sample_id):
            raise ValueError(
                f"scalar session expects sample_id {self._cursor_sample_id}; got {sample_id}. "
                "Resume from status().next_expected_sample_id and write samples sequentially."
            )

        feature_values: dict[int, float] = {}
        for feature_ref, value in dict(values).items():
            normalized = self._normalize_scalar_value(value)
            resolved_feature_id = self._resolve_feature_id(feature_ref)
            if resolved_feature_id in feature_values:
                raise ValueError(
                    f"duplicate feature assignment within sample {sample_id}: feature_id={resolved_feature_id}"
                )
            if normalized is not None:
                feature_values[int(resolved_feature_id)] = float(normalized)
        self._append_sample_rows(sample_id, feature_values)

    def open_sample(self, sample_id: int):
        raise RuntimeError("scalar sample contexts are disabled; use write_sample(sample_id=..., values=...)")

    def update_feature_meta(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        on: Optional[str] = None,
        require_all: bool = False,
    ) -> str:
        self.finish_stage()
        base = pl.read_parquet(self.sample_major_feature_meta_path)
        join_col = str(on or "")
        if not join_col:
            key_col = str(self.build_options.feature_key_col)
            join_col = key_col if key_col and key_col in base.columns else str(self.build_options.feature_id_col)
        if join_col not in base.columns:
            raise ValueError(f"feature metadata join column not found: {join_col}")

        rows = [dict(record) for record in records]
        if not rows:
            return self.sample_major_feature_meta_path
        updates = pl.from_dicts(rows, infer_schema_length=None)
        if join_col not in updates.columns:
            raise ValueError(f"update records must include join column: {join_col}")

        seen = set()
        for row_idx, value in enumerate(updates[join_col].to_list()):
            if value is None:
                raise ValueError(f"feature metadata join column {join_col} cannot be null at row {row_idx}")
            key = int(value) if join_col == str(self.build_options.feature_id_col) else str(value)
            if key in seen:
                raise ValueError(f"duplicate feature metadata join key: {value}")
            seen.add(key)

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

        merged.write_parquet(self.sample_major_feature_meta_path)
        return self.sample_major_feature_meta_path

    def finish_stage(self):
        """Finalize the resumable scalar stage and materialize its manifest."""

        if self._sample_major_finalized:
            return self.sample_major_manifest_path
        self._ensure_open()
        self._commit_pending_bundle(force=True)
        self._write_feature_meta()
        self._copy_sample_meta()
        self._write_sample_major_manifest()
        self._sample_major_finalized = True
        self._save_state()
        return self.sample_major_manifest_path

    def finish_sample_major(self):
        """Legacy alias for `finish_stage()`."""

        return self.finish_stage()

    def build_shards(self, *, keep_sample_major: bool = False, return_stats: bool = False):
        if self._shards_built:
            return (self._manifest_path, None) if return_stats else self._manifest_path
        self._ensure_open()
        sample_major_manifest_path = self.finish_stage()
        build_result = build_shards_from_sample_bundles(
            sample_major_manifest_path,
            self.out_dir,
            feature_meta_path=self.sample_major_feature_meta_path,
            n_shards=None if self.build_options.n_shards is None else int(self.build_options.n_shards),
            target_shard_bytes=int(self.build_options.target_shard_mb) * 1024 * 1024,
            feature_id_col=str(self.build_options.feature_id_col),
            value_col=str(self.build_options.value_col),
            sample_id_col=str(self.build_options.sample_id_col),
            sample_key_col=str(self.build_options.sample_key_col),
            feature_key_col=str(self.build_options.feature_key_col),
            path_col=str(self.build_options.path_col),
            y_col=str(self.build_options.y_col),
            stats_y_cols=self._stats_y_cols(),
            values_dtype=str(self.build_options.values_dtype),
            valid_dtype=str(self.build_options.valid_dtype),
            return_stats=bool(return_stats),
        )
        if return_stats:
            manifest_path, build_stats = build_result
        else:
            manifest_path = build_result
            build_stats = None
        if (not keep_sample_major) and os.path.exists(self.sample_major_out_dir):
            shutil.rmtree(self.sample_major_out_dir)
        self._manifest_path = str(manifest_path)
        self._shards_built = True
        self._closed = True
        if return_stats:
            return self._manifest_path, build_stats
        return self._manifest_path

    def close(self):
        """Close the builder while preserving committed checkpoints.

        `close()` no longer deletes the stage directory. If there are buffered
        bundle rows, they are flushed first so the next process can resume from
        the latest sealed bundle checkpoint.
        """

        if self._closed:
            return
        if not self._sample_major_finalized:
            self._commit_pending_bundle(force=True)
            self._save_state()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
