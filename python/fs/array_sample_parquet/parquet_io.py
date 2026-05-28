"""Low-level Parquet IO for sample-major trace parts."""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

from ..types import LogicalType, PointColumnSpec, StorageType


def arrow_value_type(spec: PointColumnSpec) -> pa.DataType:
    """point schema storage_type을 Parquet list element type으로 변환한다."""

    if spec.storage_type == StorageType.FLOAT64:
        return pa.float64()
    if spec.storage_type == StorageType.INT32:
        return pa.int32()
    if spec.storage_type == StorageType.INT64:
        return pa.int64()
    if spec.storage_type == StorageType.UINT32:
        return pa.uint32()
    if spec.storage_type == StorageType.UINT64:
        return pa.uint64()
    raise ValueError(f"unsupported storage_type: {spec.storage_type}")


def arrow_schema(point_schema: list[PointColumnSpec]) -> pa.Schema:
    """part parquet의 물리 schema를 만든다.

    행 하나는 `(sample_id, feature_id)` trace 하나이고, point column은 모두
    nullable이 아닌 list column이다. point-level null은 포맷에서 지원하지 않는다.
    """

    fields = [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("feature_id", pa.int32(), nullable=False),
        pa.field("trace_len", pa.int32(), nullable=False),
    ]
    for spec in point_schema:
        metadata = {
            b"logical_type": spec.logical_type.value.encode("utf-8"),
            b"storage_type": spec.storage_type.value.encode("utf-8"),
        }
        if spec.logical_type == LogicalType.CATEGORICAL:
            metadata[b"categorical"] = b"true"
        fields.append(pa.field(spec.name, pa.list_(arrow_value_type(spec)), nullable=False, metadata=metadata))
    return pa.schema(fields, metadata={b"format": b"array-sample-parquet", b"version": b"1"})


class StreamingTracePartWriter:
    """trace row를 part parquet `.tmp` 파일에 작은 batch 단위로 바로 쓴다.

    Parquet footer는 writer close 시점에 완성된다. 따라서 build session은 항상
    `.parquet.tmp`에 먼저 쓰고, sample 경계에서 part를 commit할 때만 close 후
    최종 `.parquet` 경로로 rename한다.
    """

    def __init__(self, path: str, *, point_schema: list[PointColumnSpec], compression: str, batch_rows: int = 256):
        self.path = os.path.abspath(path)
        self.tmp_path = self.path + ".tmp"
        self.point_schema = list(point_schema)
        self.compression = None if str(compression).lower() in {"", "none"} else str(compression)
        self.batch_rows = max(1, int(batch_rows))
        self.schema = arrow_schema(self.point_schema)
        self._sample_ids: list[int] = []
        self._feature_ids: list[int] = []
        self._trace_lens: list[int] = []
        self._point_columns = empty_point_columns(self.point_schema)
        self._closed = False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._writer = pq.ParquetWriter(self.tmp_path, self.schema, compression=self.compression)

    def write_row(self, *, sample_id: int, feature_id: int, trace_len: int, columns: dict):
        if self._closed:
            raise RuntimeError("streaming parquet part writer is already closed")
        self._sample_ids.append(int(sample_id))
        self._feature_ids.append(int(feature_id))
        self._trace_lens.append(int(trace_len))
        for spec in self.point_schema:
            self._point_columns[spec.name].append(columns[spec.name].tolist())
        if len(self._sample_ids) >= self.batch_rows:
            self.flush()

    def flush(self):
        if not self._sample_ids:
            return
        arrays = [
            pa.array(self._sample_ids, type=pa.int64()),
            pa.array(self._feature_ids, type=pa.int32()),
            pa.array(self._trace_lens, type=pa.int32()),
        ]
        for spec in self.point_schema:
            arrays.append(pa.array(self._point_columns[spec.name], type=pa.list_(arrow_value_type(spec))))
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))
        self._sample_ids = []
        self._feature_ids = []
        self._trace_lens = []
        self._point_columns = empty_point_columns(self.point_schema)

    def commit(self):
        if self._closed:
            raise RuntimeError("streaming parquet part writer is already closed")
        self.flush()
        self._writer.close()
        self._closed = True
        os.replace(self.tmp_path, self.path)

    def abort(self):
        if not self._closed:
            try:
                self._writer.close()
            except Exception:
                pass
            self._closed = True
        try:
            os.remove(self.tmp_path)
        except FileNotFoundError:
            pass


def empty_point_columns(point_schema: list[PointColumnSpec]) -> dict[str, list[list]]:
    return {spec.name: [] for spec in point_schema}
