"""trace 입력으로 array binary shard를 만드는 고수준 builder facade."""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ._impl.binary_storage import _load_dense_meta, build_array_binary_shards_from_bundles
from ._impl.config import ArrayBundleConfig, ArrayShardConfig
from ._impl.storage import ArraySampleBundleWriter, _normalize_point_schema
from ._impl.types import LogicalType, point_storage_dtype
from .models import BuildOptions


@dataclass
class _CategoricalRegistry:
    """categorical point column 하나에 대한 가변 string-to-code 매핑."""

    label_to_code: dict
    code_to_label: list

    @classmethod
    def create(cls):
        """비어 있는 categorical registry를 생성한다."""
        return cls(label_to_code={}, code_to_label=[])

    def encode(self, values) -> np.ndarray:
        """categorical trace column 하나를 uint32 code로 인코딩한다.

        Args:
            values: point column 하나에 대한 label 시퀀스.

        Returns:
            `0`은 null 값에 예약하고, 나머지 label에는 처음 등장한 순서대로
            정수 code를 부여한 `uint32` 배열을 반환한다.
        """
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
        """registry를 JSON dictionary sidecar로 기록할 수 있는 구조로 만든다."""
        return {
            "column": str(column_name),
            "items": [
                {"code": int(idx + 1), "label": str(label)}
                for idx, label in enumerate(self.code_to_label)
            ],
        }


class SampleContext:
    """sample 범위로 trace를 넣을 때 사용하는 선택적 helper."""

    def __init__(self, builder: "ArrayDatasetBuilder", sample_id: int):
        """builder 하나와 dense sample id 하나를 묶습니다."""
        self._builder = builder
        self._sample_id = int(sample_id)

    def __enter__(self):
        """`with` 문에서 사용할 sample-scoped helper를 반환한다."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """sample context 내부에서 발생한 예외를 삼키지 않는다."""
        return False

    def add_trace(self, feature_id: Optional[int] = None, feature_key: Optional[str] = None, *, columns):
        """묶여 있는 sample id에 trace 하나를 추가한다.

        Args:
            feature_id: 선택적 dense feature id.
            feature_key: 선택적 외부 feature key.
            columns: point-column 매핑.
        """
        self._builder.add_trace(
            sample_id=self._sample_id,
            feature_id=feature_id,
            feature_key=feature_key,
            columns=columns,
        )


class ArrayDatasetBuilder:
    """trace를 바로 받아 최종 binary array shard로 바꾸는 고수준 builder.

    이 builder는 trace를 하나씩 받아 sample-major bundle parquet에 버퍼링한 뒤,
    나중에 그 bundle을 최종 binary shard artifact로 변환한다.

    feature id는 두 가지 방식으로 입력할 수 있다.

    - known-feature mode:
      시작할 때 `feature_meta_path` 또는 `feature_keys`를 넘깁니다.
    - discovered-feature mode:
      둘 다 넘기지 않고 항상 `feature_key`로 trace를 추가한다.
      dense feature id는 처음 등장한 순서대로 부여된다.
    """

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        point_schema,
        *,
        feature_meta_path: Optional[str] = None,
        feature_keys: Optional[Sequence[str]] = None,
        build_options: BuildOptions | None = None,
        bundle_config: ArrayBundleConfig | None = None,
        bundle_out_dir: Optional[str] = None,
    ):
        """새 direct-ingestion array dataset builder를 생성한다.

        Args:
            out_dir: 최종 standalone binary artifact를 쓸 출력 디렉터리.
            sample_meta_path: dense sample metadata parquet 경로.
            point_schema: manifest 전체에 고정되는 point-column schema.
            feature_meta_path: known-feature mode에서 사용할 선택적 dense feature
                metadata parquet 경로.
            feature_keys: feature metadata 파일이 없을 때 known-feature mode에서
                사용할 선택적 외부 feature key 목록.
            build_options: 고수준 binary shard build 옵션.
            bundle_config: 내부 bundle flush 기준.
            bundle_out_dir: 중간 bundle artifact를 기록할 선택적 디렉터리.
                생략하면 `out_dir` 아래에 보이는 하위 디렉터리를 사용한다.
        """
        self.out_dir = str(Path(out_dir).expanduser().resolve())
        self.sample_meta_path = str(Path(sample_meta_path).expanduser().resolve())
        self.build_options = build_options or BuildOptions()
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
        self._sample_key_to_id = None
        if self._sample_key_col and self._sample_key_col in sample_meta.columns:
            sample_keys = sample_meta[self._sample_key_col].to_list()
            self._sample_key_to_id = {
                str(sample_key): int(idx)
                for idx, sample_key in enumerate(sample_keys)
                if sample_key is not None
            }

        default_bundle_root = os.path.join(self.out_dir, "bundle_stage")
        self.bundle_out_dir = str(Path(bundle_out_dir or default_bundle_root).expanduser().resolve())
        self._prepare_empty_dir(self.bundle_out_dir)

        self._bundle_sample_meta_path = os.path.join(self.bundle_out_dir, "sample_meta.parquet")
        shutil.copy2(self.sample_meta_path, self._bundle_sample_meta_path)

        self._feature_meta_path = os.path.join(self.bundle_out_dir, "feature_meta.parquet")
        self._feature_key_to_id = OrderedDict()
        self._feature_keys_in_order = []
        self._known_feature_mode = False
        self._known_feature_count = None
        self._writes_feature_meta = False

        if feature_meta_path and feature_keys is not None:
            raise ValueError("provide at most one of feature_meta_path or feature_keys")

        if feature_meta_path:
            self._known_feature_mode = True
            feature_meta_source_path = str(Path(feature_meta_path).expanduser().resolve())
            feature_meta = _load_dense_meta(
                feature_meta_source_path,
                "feature_id",
                "feature",
                str(self.build_options.feature_key_col),
            )
            shutil.copy2(feature_meta_source_path, self._feature_meta_path)
            self._known_feature_count = int(feature_meta.height)
            key_col = str(self.build_options.feature_key_col)
            if key_col and key_col in feature_meta.columns:
                for idx, value in enumerate(feature_meta[key_col].to_list()):
                    self._feature_key_to_id[str(value)] = int(idx)
                    self._feature_keys_in_order.append(str(value))
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

        self._categorical_registries = {
            spec.name: _CategoricalRegistry.create()
            for spec in self.point_schema
            if spec.logical_type == LogicalType.CATEGORICAL
        }

        self._bundle_writer = ArraySampleBundleWriter(
            self.bundle_out_dir,
            self._bundle_sample_meta_path,
            feature_meta_path=self._feature_meta_path,
            n_samples=self.n_samples,
            config=self.bundle_config,
            point_schema=self.point_schema,
        )
        self.bundle_manifest_path = self._bundle_writer.manifest_path
        self.bundle_path = self._bundle_writer.bundle_path

    @staticmethod
    def _prepare_empty_dir(path: str):
        """비어 있는 출력 디렉터리를 만들고, 이미 내용이 있으면 실패한다."""
        if os.path.exists(path):
            if os.path.isdir(path) and not os.listdir(path):
                return
            raise ValueError(f"bundle_out_dir already exists and is not empty: {path}")
        os.makedirs(path, exist_ok=True)

    def _ensure_open(self):
        """builder가 이미 닫혔거나 종료됐으면 예외를 발생시킨다."""
        if self._closed:
            raise RuntimeError("array dataset builder is closed")
        if self._finished:
            raise RuntimeError("array dataset builder is already finished")

    def _ensure_trace_stage_open(self):
        """더 이상 trace 입력을 받을 수 없으면 예외를 발생시킨다."""
        self._ensure_open()
        if self._bundles_finalized:
            raise RuntimeError("bundle stage has already been finalized")

    def close(self):
        """builder를 닫는다.

        bundle 단계가 아직 finalize되지 않았다면, 부분적으로 기록된 bundle
        artifact 디렉터리도 함께 삭제한다.
        """
        if self._closed:
            return
        try:
            if not self._bundles_finalized and os.path.exists(self.bundle_out_dir):
                shutil.rmtree(self.bundle_out_dir)
        finally:
            self._closed = True

    def __enter__(self):
        """`with` 문에서 사용할 builder 자신을 반환한다."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """정상 종료 시 shard를 마무리하고, 실패 시 부분 bundle 상태를 정리한다."""
        if exc_type is not None:
            self.close()
            return False
        if not self._finished:
            self.build_shards()
        return False

    def sample(self, sample_id: Optional[int] = None, sample_key: Optional[str] = None) -> SampleContext:
        """sample 범위 helper 객체를 반환한다.

        Args:
            sample_id: 선택적 dense sample id.
            sample_key: 선택적 외부 sample key.

        Returns:
            `add_trace(...)` 호출 시 `sample_id`를 자동으로 채워주는 helper.
        """
        self._ensure_trace_stage_open()
        return SampleContext(self, self._resolve_sample_id(sample_id, sample_key))

    def _resolve_sample_id(self, sample_id: Optional[int], sample_key: Optional[str]) -> int:
        """사용자가 넘긴 sample 참조를 dense sample id 하나로 정규화한다."""
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
        """사용자가 넘긴 feature 참조를 dense feature id 하나로 정규화한다."""
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
        """trace 하나를 point column별 타입이 맞는 1D NumPy 배열로 정규화한다."""
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
            if spec.name not in columns:
                raise ValueError(f"missing point column: {spec.name}")
            values = columns[spec.name]
            logical_type = spec.logical_type
            if logical_type == LogicalType.CATEGORICAL:
                registry = self._categorical_registries[spec.name]
                arr = registry.encode(values)
            elif logical_type == LogicalType.TIMESTAMP_NS:
                arr = np.asarray(values, dtype="datetime64[ns]").reshape(-1).astype(point_storage_dtype(spec.storage_type), copy=False)
            elif logical_type == LogicalType.TIMEDELTA_NS:
                arr = np.asarray(values, dtype="timedelta64[ns]").reshape(-1).astype(point_storage_dtype(spec.storage_type), copy=False)
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

    def add_trace(
        self,
        *,
        sample_id: Optional[int] = None,
        sample_key: Optional[str] = None,
        feature_id: Optional[int] = None,
        feature_key: Optional[str] = None,
        columns,
    ):
        """bundle 단계에 trace 하나를 추가한다.

        Args:
            sample_id: 선택적 dense sample id.
            sample_key: 선택적 외부 sample key.
            feature_id: 선택적 dense feature id.
            feature_key: 선택적 외부 feature key.
            columns: 전체 point-column 매핑.
        """
        self._ensure_trace_stage_open()
        sample_id = self._resolve_sample_id(sample_id, sample_key)
        resolved_feature_id = self._resolve_feature_id(feature_id, feature_key)
        normalized_columns = self._normalize_columns(columns=columns)
        self._bundle_writer.append_trace(
            sample_row=sample_id,
            sample_id=sample_id,
            feature_id=resolved_feature_id,
            columns=normalized_columns,
        )

    def _write_feature_meta(self):
        """builder가 소유한 경우 생성된 dense feature metadata를 기록한다."""
        if not self._writes_feature_meta:
            return
        feature_ids = np.arange(len(self._feature_keys_in_order), dtype=np.int32)
        data = {
            "feature_id": pl.Series("feature_id", feature_ids, dtype=pl.Int32),
        }
        key_col = str(self.build_options.feature_key_col)
        if key_col:
            data[key_col] = pl.Series(key_col, list(self._feature_keys_in_order), dtype=pl.String)
        pl.DataFrame(data).write_parquet(self._feature_meta_path)

    def _write_categorical_dictionaries(self):
        """생성된 categorical dictionary JSON 파일을 bundle artifact에 기록한다."""
        dict_root = os.path.join(self.bundle_out_dir, "categorical_dictionaries")
        os.makedirs(dict_root, exist_ok=True)
        for idx, spec in enumerate(self.point_schema):
            if spec.logical_type != LogicalType.CATEGORICAL:
                continue
            registry = self._categorical_registries[spec.name]
            dict_path = os.path.join(dict_root, f"{spec.name}.json")
            with open(dict_path, "w", encoding="utf-8") as f:
                json.dump(registry.to_json_dict(spec.name), f, ensure_ascii=False, indent=2)
            spec.dictionary_path = dict_path
            self._bundle_writer.point_schema[idx].dictionary_path = dict_path

    def update_feature_meta(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        on: Optional[str] = None,
        require_all: bool = False,
    ) -> str:
        """bundle 단계의 feature metadata에 추가 컬럼을 병합한다.

        필요하면 먼저 bundle 단계를 finalize하므로, metadata를 보강하기 전에
        discovered feature id가 고정된다.

        Args:
            records: 추가할 feature metadata 행. 각 행은 `on`으로 지정한 join
                컬럼을 반드시 포함해야 한다.
            on: join 컬럼 이름. 생략하면 설정된 feature key 컬럼이 있으면 그것을,
                없으면 `feature_id`를 사용한다.
            require_all: `True`이면 현재 metadata 테이블의 모든 feature row가
                새 컬럼에 대해 null이 아닌 값을 가져야 한다.

        Returns:
            갱신된 `feature_meta.parquet`의 절대 경로.
        """
        self.finish_bundles()
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

    def finish_bundles(self):
        """중간 sample-major bundle artifact를 finalize한다.

        Returns:
            생성된 `array_bundle_manifest.json` 경로.
        """
        if self._bundles_finalized:
            return self.bundle_manifest_path
        self._ensure_open()
        self._write_feature_meta()
        self._write_categorical_dictionaries()
        self.bundle_manifest_path = self._bundle_writer.finish()
        self.bundle_path = self._bundle_writer.bundle_path
        self._bundles_finalized = True
        return self.bundle_manifest_path

    def build_shards(self, *, cleanup_bundles: bool = False, return_stats: bool = False):
        """finalize된 bundle 단계를 최종 binary shard artifact로 변환한다.

        Args:
            cleanup_bundles: shard 생성이 성공한 뒤 중간 bundle artifact
                디렉터리를 삭제할지 여부.
            return_stats: manifest 경로와 함께 build 통계를 반환할지 여부.

        Returns:
            manifest 경로 또는 `return_stats=True`일 때 `(manifest_path, stats)`.
        """
        if self._finished:
            if return_stats:
                raise RuntimeError("array dataset builder has already built shards; build stats are no longer available")
            return self._manifest_path
        self._ensure_open()
        bundle_manifest_path = self.finish_bundles()
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
        """`build_shards(...)`와 동일한 호환용 alias입니다."""
        return self.build_shards(cleanup_bundles=cleanup_bundles, return_stats=return_stats)
