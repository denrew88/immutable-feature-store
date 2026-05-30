"""Low-level Parquet IO for long-format array_sample_parquet parts."""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..types import LogicalType, PointColumnSpec, StorageType, point_storage_dtype


def normalize_array_sample_point_schema(point_schema: list[PointColumnSpec]) -> list[PointColumnSpec]:
    """array_sample_parquet에서는 categorical column을 문자열로 저장하도록 표준화한다."""

    out: list[PointColumnSpec] = []
    for spec in point_schema:
        if spec.logical_type == LogicalType.CATEGORICAL:
            out.append(PointColumnSpec(spec.name, StorageType.STRING, spec.logical_type))
        else:
            out.append(PointColumnSpec(spec.name, spec.storage_type, spec.logical_type))
    return out


def arrow_value_type(spec: PointColumnSpec) -> pa.DataType:
    """point schema storage_type을 Parquet primitive type으로 변환한다."""

    if spec.storage_type == StorageType.FLOAT64:
        return pa.float64()
    if spec.storage_type == StorageType.STRING:
        return pa.string()
    if spec.storage_type == StorageType.INT32:
        return pa.int32()
    if spec.storage_type == StorageType.INT64:
        return pa.int64()
    if spec.storage_type == StorageType.UINT8:
        return pa.uint8()
    if spec.storage_type == StorageType.UINT16:
        return pa.uint16()
    if spec.storage_type == StorageType.UINT32:
        return pa.uint32()
    if spec.storage_type == StorageType.UINT64:
        return pa.uint64()
    raise ValueError(f"unsupported storage_type: {spec.storage_type}")


def arrow_schema(point_schema: list[PointColumnSpec]) -> pa.Schema:
    """최종 point part schema를 만든다.

    `sample_parts/*.parquet`은 long format이다. row 하나는 trace 하나가 아니라
    point 하나이며 `(sample_id, feature_id, point_idx)`와 primitive point column
    값들을 가진다. empty trace는 point row가 없으므로 별도 trace index parquet에
    기록한다.
    """

    fields = [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("feature_id", pa.int32(), nullable=False),
        pa.field("point_idx", pa.int32(), nullable=False),
    ]
    for spec in point_schema:
        metadata = {
            b"logical_type": spec.logical_type.value.encode("utf-8"),
            b"storage_type": spec.storage_type.value.encode("utf-8"),
        }
        if spec.logical_type == LogicalType.CATEGORICAL:
            metadata[b"categorical"] = b"true"
        fields.append(pa.field(spec.name, arrow_value_type(spec), nullable=False, metadata=metadata))
    return pa.schema(fields, metadata={b"format": b"array-sample-parquet", b"version": b"1"})


def trace_index_schema() -> pa.Schema:
    """present trace를 보존하기 위한 trace index schema를 만든다."""

    return pa.schema(
        [
            pa.field("sample_id", pa.int64(), nullable=False),
            pa.field("feature_id", pa.int32(), nullable=False),
            pa.field("trace_len", pa.int32(), nullable=False),
        ],
        metadata={b"format": b"array-sample-parquet-trace-index", b"version": b"1"},
    )


class StreamingTracePartWriter:
    """trace를 long-format point parquet와 trace index parquet에 기록한다.

    point parquet에는 point가 하나도 없는 empty trace를 표현할 row가 없다. 따라서
    모든 present trace는 trace index에 `(sample_id, feature_id, trace_len)`으로
    기록하고, `trace_len > 0`인 trace만 point parquet에 `point_idx`별 row로 펼쳐 쓴다.
    """

    def __init__(
        self,
        path: str,
        *,
        trace_index_path: str,
        point_schema: list[PointColumnSpec],
        compression: str,
        batch_rows: int = 64,
    ):
        self.path = os.path.abspath(path)
        self.tmp_path = self.path + ".tmp"
        self.trace_index_path = os.path.abspath(trace_index_path)
        self.trace_index_tmp_path = self.trace_index_path + ".tmp"
        self.point_schema = list(point_schema)
        self.compression = None if str(compression).lower() in {"", "none"} else str(compression)
        self.batch_rows = max(1, int(batch_rows))
        self.schema = arrow_schema(self.point_schema)
        self.trace_index_schema = trace_index_schema()
        self._trace_sample_ids: list[int] = []
        self._trace_feature_ids: list[int] = []
        self._trace_lens: list[int] = []
        self._point_columns = empty_point_columns(self.point_schema)
        self._closed = False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trace_index_path), exist_ok=True)
        self._writer = pq.ParquetWriter(self.tmp_path, self.schema, compression=self.compression)
        self._trace_index_writer = pq.ParquetWriter(
            self.trace_index_tmp_path,
            self.trace_index_schema,
            compression=self.compression,
        )

    def write_row(self, *, sample_id: int, feature_id: int, trace_len: int, columns: dict):
        if self._closed:
            raise RuntimeError("streaming parquet part writer is already closed")
        self._trace_sample_ids.append(int(sample_id))
        self._trace_feature_ids.append(int(feature_id))
        self._trace_lens.append(int(trace_len))
        for spec in self.point_schema:
            self._point_columns[spec.name].append(np.asarray(columns[spec.name]).reshape(-1).copy())
        if len(self._trace_sample_ids) >= self.batch_rows:
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
            arrays = [
                pa.array(np.repeat(np.asarray(self._trace_sample_ids, dtype=np.int64), self._trace_lens), type=pa.int64()),
                pa.array(np.repeat(np.asarray(self._trace_feature_ids, dtype=np.int32), self._trace_lens), type=pa.int32()),
                pa.array(point_indices_from_lengths(self._trace_lens), type=pa.int32()),
            ]
            for spec in self.point_schema:
                arrays.append(value_array_from_numpy_rows(self._point_columns[spec.name], spec))
            self._writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))

        self._trace_sample_ids = []
        self._trace_feature_ids = []
        self._trace_lens = []
        self._point_columns = empty_point_columns(self.point_schema)

    def commit(self):
        if self._closed:
            raise RuntimeError("streaming parquet part writer is already closed")
        self.flush()
        self._writer.close()
        self._trace_index_writer.close()
        self._closed = True
        os.replace(self.tmp_path, self.path)
        os.replace(self.trace_index_tmp_path, self.trace_index_path)

    def abort(self):
        if not self._closed:
            for writer in [self._writer, self._trace_index_writer]:
                try:
                    writer.close()
                except Exception:
                    pass
            self._closed = True
        for path in [self.tmp_path, self.trace_index_tmp_path]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def empty_point_columns(point_schema: list[PointColumnSpec]) -> dict[str, list[np.ndarray]]:
    return {spec.name: [] for spec in point_schema}


def point_indices_from_lengths(lengths: list[int]) -> np.ndarray:
    if not lengths:
        return np.empty(0, dtype=np.int32)
    non_empty = [np.arange(int(length), dtype=np.int32) for length in lengths if int(length) > 0]
    return np.concatenate(non_empty) if non_empty else np.empty(0, dtype=np.int32)


def value_array_from_numpy_rows(rows: list[np.ndarray], spec: PointColumnSpec) -> pa.Array:
    """trace별 NumPy row buffer를 long-format primitive Arrow array로 붙인다."""

    total_len = int(sum(int(row.size) for row in rows))
    if spec.storage_type == StorageType.STRING:
        if total_len == 0:
            values = np.empty(0, dtype=object)
        elif len(rows) == 1:
            values = np.asarray(rows[0], dtype=object).reshape(-1)
        else:
            non_empty = [row for row in rows if int(row.size) > 0]
            values = (
                np.concatenate(non_empty).astype(object, copy=False)
                if non_empty
                else np.empty(0, dtype=object)
            )
        return pa.array(values, type=pa.string())
    if total_len == 0:
        values = np.empty(0, dtype=point_storage_dtype(spec.storage_type))
    elif len(rows) == 1:
        values = np.asarray(rows[0], dtype=point_storage_dtype(spec.storage_type)).reshape(-1)
    else:
        non_empty = [row for row in rows if int(row.size) > 0]
        values = (
            np.concatenate(non_empty).astype(point_storage_dtype(spec.storage_type), copy=False)
            if non_empty
            else np.empty(0, dtype=point_storage_dtype(spec.storage_type))
        )
    return pa.array(values, type=arrow_value_type(spec))
