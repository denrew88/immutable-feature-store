"""Reader for array_sample_parquet v1."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from .manifest import ArraySampleParquetManifest, load_array_sample_parquet_manifest


@dataclass
class ArraySampleParquetTrace:
    sample_id: int
    feature_id: int
    trace_len: int
    columns: dict[str, list] = field(default_factory=dict)
    sample_key: Optional[str] = None
    feature_key: Optional[str] = None
    present: bool = True

    def to_json(self) -> dict:
        return {
            "sample_id": int(self.sample_id),
            "sample_key": self.sample_key,
            "feature_id": int(self.feature_id),
            "feature_key": self.feature_key,
            "present": bool(self.present),
            "trace_len": int(self.trace_len),
            "columns": self.columns,
        }


def _build_key_to_id_index(meta_path: str, key_col: str, id_col: str) -> Optional[dict[str, int]]:
    df = pl.read_parquet(meta_path)
    if not key_col or key_col not in df.columns:
        return None
    ids = df[id_col].to_list() if id_col in df.columns else list(range(df.height))
    return {str(key): int(value) for key, value in zip(df[key_col].to_list(), ids)}


def _build_id_to_key_index(meta_path: str, key_col: str, id_col: str) -> Optional[dict[int, str]]:
    df = pl.read_parquet(meta_path)
    if not key_col or key_col not in df.columns:
        return None
    ids = df[id_col].to_list() if id_col in df.columns else list(range(df.height))
    return {int(value): str(key) for key, value in zip(df[key_col].to_list(), ids)}


class ArraySampleParquetReader:
    """sample-major Parquet array dataset reader.

    여러 sample/feature 요청은 관련 part만 골라 읽는다. 이 포맷은 viewer/debugging용이라
    feature-major binary shard처럼 O(1) block seek를 목표로 하지 않는다.
    """

    def __init__(self, manifest: ArraySampleParquetManifest):
        self.manifest = manifest
        self._sample_key_index = None
        self._feature_key_index = None
        self._sample_keys_by_id = None
        self._feature_keys_by_id = None

    def close(self):
        """현재 reader는 mmap 같은 장기 리소스를 잡지 않으므로 no-op이다."""

    def point_schema(self):
        return list(self.manifest.point_schema)

    def _sample_key_to_id(self) -> Optional[dict[str, int]]:
        if self._sample_key_index is None:
            self._sample_key_index = _build_key_to_id_index(
                self.manifest.sample_meta_path,
                self.manifest.sample_key_col,
                "sample_id",
            )
        return self._sample_key_index

    def _feature_key_to_id(self) -> Optional[dict[str, int]]:
        if self._feature_key_index is None:
            self._feature_key_index = _build_key_to_id_index(
                self.manifest.feature_meta_path,
                self.manifest.feature_key_col,
                "feature_id",
            )
        return self._feature_key_index

    def sample_keys_by_id(self) -> Optional[dict[int, str]]:
        if self._sample_keys_by_id is None:
            self._sample_keys_by_id = _build_id_to_key_index(
                self.manifest.sample_meta_path,
                self.manifest.sample_key_col,
                "sample_id",
            )
        return self._sample_keys_by_id

    def feature_keys_by_id(self) -> Optional[dict[int, str]]:
        if self._feature_keys_by_id is None:
            self._feature_keys_by_id = _build_id_to_key_index(
                self.manifest.feature_meta_path,
                self.manifest.feature_key_col,
                "feature_id",
            )
        return self._feature_keys_by_id

    def _resolve_sample_ids(self, sample_ids=None, sample_keys=None) -> list[int]:
        has_ids = sample_ids is not None
        has_keys = sample_keys is not None
        if has_ids == has_keys:
            raise ValueError("provide exactly one of sample_ids or sample_keys")
        if sample_ids is not None:
            out = [int(value) for value in sample_ids]
        else:
            index = self._sample_key_to_id()
            if index is None:
                raise ValueError(f"sample metadata does not expose key column: {self.manifest.sample_key_col}")
            out = []
            for key in sample_keys:
                value = index.get(str(key))
                if value is None:
                    raise KeyError(f"sample key not found: {key}")
                out.append(int(value))
        if not out:
            raise ValueError("sample request must not be empty")
        return out

    def _resolve_feature_ids(self, feature_ids=None, feature_keys=None) -> Optional[list[int]]:
        has_ids = feature_ids is not None
        has_keys = feature_keys is not None
        if has_ids and has_keys:
            raise ValueError("provide at most one of feature_ids or feature_keys")
        if feature_ids is not None:
            out = [int(value) for value in feature_ids]
        elif feature_keys is not None:
            index = self._feature_key_to_id()
            if index is None:
                raise ValueError(f"feature metadata does not expose key column: {self.manifest.feature_key_col}")
            out = []
            for key in feature_keys:
                value = index.get(str(key))
                if value is None:
                    raise KeyError(f"feature key not found: {key}")
                out.append(int(value))
        else:
            return None
        if not out:
            raise ValueError("feature request must not be empty")
        return out

    def _candidate_part_paths(self, sample_ids: list[int]) -> list[str]:
        requested_min = min(sample_ids)
        requested_max = max(sample_ids)
        paths = []
        for part in self.manifest.parts:
            if int(part.last_sample_id) < requested_min or int(part.first_sample_id) > requested_max:
                continue
            if os.path.exists(part.path):
                paths.append(part.path)
        return paths

    def _candidate_trace_index_paths(self, sample_ids: list[int]) -> list[str]:
        requested_min = min(sample_ids)
        requested_max = max(sample_ids)
        paths = []
        for part in self.manifest.parts:
            if int(part.last_sample_id) < requested_min or int(part.first_sample_id) > requested_max:
                continue
            if os.path.exists(part.trace_index_path):
                paths.append(part.trace_index_path)
        return paths

    def get_traces(
        self,
        *,
        sample_ids=None,
        sample_keys=None,
        feature_ids=None,
        feature_keys=None,
        decode_categorical: bool = False,
        include_missing: bool = False,
    ) -> list[ArraySampleParquetTrace]:
        """여러 sample/feature trace를 flat list로 조회한다."""

        resolved_sample_ids = self._resolve_sample_ids(sample_ids=sample_ids, sample_keys=sample_keys)
        resolved_feature_ids = self._resolve_feature_ids(feature_ids=feature_ids, feature_keys=feature_keys)
        paths = self._candidate_part_paths(resolved_sample_ids)
        trace_index_paths = self._candidate_trace_index_paths(resolved_sample_ids)
        sample_keys_by_id = self.sample_keys_by_id() or {}
        feature_keys_by_id = self.feature_keys_by_id() or {}
        specs = self.manifest.point_schema
        traces: list[ArraySampleParquetTrace] = []

        trace_df = pl.DataFrame({"sample_id": [], "feature_id": [], "trace_len": []})
        if trace_index_paths:
            trace_lazy = pl.scan_parquet(trace_index_paths, glob=False).filter(pl.col("sample_id").is_in(resolved_sample_ids))
            if resolved_feature_ids is not None:
                trace_lazy = trace_lazy.filter(pl.col("feature_id").is_in(resolved_feature_ids))
            trace_df = trace_lazy.sort(["sample_id", "feature_id"]).collect()

        point_rows_by_trace: dict[tuple[int, int], dict] = {}
        if paths and trace_df.height:
            point_lazy = pl.scan_parquet(paths, glob=False).filter(pl.col("sample_id").is_in(resolved_sample_ids))
            if resolved_feature_ids is not None:
                point_lazy = point_lazy.filter(pl.col("feature_id").is_in(resolved_feature_ids))
            point_df = (
                point_lazy
                .sort(["sample_id", "feature_id", "point_idx"])
                .group_by(["sample_id", "feature_id"], maintain_order=True)
                .agg([pl.col(spec.name).alias(spec.name) for spec in specs])
                .collect()
            )
            point_rows_by_trace = {
                (int(row["sample_id"]), int(row["feature_id"])): row
                for row in point_df.iter_rows(named=True)
            }

        for row in trace_df.iter_rows(named=True):
            sample_id = int(row["sample_id"])
            feature_id = int(row["feature_id"])
            trace_len = int(row["trace_len"])
            point_row = point_rows_by_trace.get((sample_id, feature_id), {})
            columns = {}
            for spec in specs:
                values = list(point_row.get(spec.name) or [])
                if trace_len != len(values):
                    if trace_len == 0 and not values:
                        values = []
                    else:
                        raise ValueError(
                            f"trace_len/point row mismatch for sample_id={sample_id}, feature_id={feature_id}, column={spec.name}"
                        )
                columns[spec.name] = values
            traces.append(
                ArraySampleParquetTrace(
                    sample_id=sample_id,
                    sample_key=sample_keys_by_id.get(sample_id),
                    feature_id=feature_id,
                    feature_key=feature_keys_by_id.get(feature_id),
                    trace_len=trace_len,
                    columns=columns,
                    present=True,
                )
            )

        if include_missing and resolved_feature_ids is not None:
            seen = {(trace.sample_id, trace.feature_id) for trace in traces}
            empty_columns = {spec.name: [] for spec in specs}
            for sample_id in resolved_sample_ids:
                for feature_id in resolved_feature_ids:
                    key = (int(sample_id), int(feature_id))
                    if key in seen:
                        continue
                    traces.append(
                        ArraySampleParquetTrace(
                            sample_id=int(sample_id),
                            sample_key=sample_keys_by_id.get(int(sample_id)),
                            feature_id=int(feature_id),
                            feature_key=feature_keys_by_id.get(int(feature_id)),
                            trace_len=0,
                            columns={name: list(values) for name, values in empty_columns.items()},
                            present=False,
                        )
                    )
        traces.sort(key=lambda item: (int(item.sample_id), int(item.feature_id)))
        return traces

    def get_traces_json(self, *, layout: str = "nested", **kwargs) -> dict:
        """API 서버가 그대로 사용할 수 있는 JSON-compatible layout으로 조회한다."""

        traces = [trace.to_json() for trace in self.get_traces(**kwargs)]
        layout = str(layout or "nested").lower()
        if layout == "flat":
            return {"layout": "flat", "trace_count": len(traces), "traces": traces}
        if layout != "nested":
            raise ValueError("layout must be 'nested' or 'flat'")
        samples = []
        current = None
        for trace in traces:
            if current is None or current["sample_id"] != trace["sample_id"]:
                current = {
                    "sample_id": trace["sample_id"],
                    "sample_key": trace["sample_key"],
                    "traces": [],
                }
                samples.append(current)
            current["traces"].append(
                {
                    "feature_id": trace["feature_id"],
                    "feature_key": trace["feature_key"],
                    "present": trace["present"],
                    "trace_len": trace["trace_len"],
                    "columns": trace["columns"],
                }
            )
        return {"layout": "nested", "sample_count": len(samples), "trace_count": len(traces), "samples": samples}


def open_array_sample_parquet(manifest_path: str) -> ArraySampleParquetReader:
    return ArraySampleParquetReader(load_array_sample_parquet_manifest(manifest_path))
