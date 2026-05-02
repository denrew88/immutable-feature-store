"""Helpers for writing dense scalar sample/feature metadata parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import polars as pl


def _build_dense_meta_frame(
    records: Sequence[Mapping[str, object]],
    *,
    id_col: str,
    id_dtype: pl.DataType,
    entity_name: str,
    key_col: str,
) -> pl.DataFrame:
    """Convert list-of-dict metadata records into one dense metadata dataframe."""

    rows = [dict(record) for record in records]
    column_order = []
    seen_columns = set()
    for row in rows:
        for column_name in row:
            if column_name not in seen_columns:
                seen_columns.add(column_name)
                column_order.append(column_name)

    if rows:
        for row_idx, row in enumerate(rows):
            row_value = row.get(id_col, row_idx)
            try:
                dense_id = int(row_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{entity_name} {id_col} must be an integer at row {row_idx}") from exc
            if dense_id != row_idx:
                raise ValueError(
                    f"{entity_name} metadata must use dense {id_col}; row {row_idx} has {dense_id}"
                )
            row[id_col] = dense_id

    ordered_columns = [id_col] + [name for name in column_order if name != id_col]
    if rows:
        df = pl.from_dicts(rows, infer_schema_length=None)
        df = df.with_columns(pl.col(id_col).cast(id_dtype, strict=True)).select(ordered_columns)
    else:
        df = pl.DataFrame({id_col: pl.Series(id_col, [], dtype=id_dtype)})

    if key_col and key_col in df.columns:
        key_values = df[key_col].to_list()
        seen_keys = set()
        for row_idx, value in enumerate(key_values):
            if value is None:
                raise ValueError(f"{entity_name} {key_col} cannot be null at row {row_idx}")
            key = str(value)
            if key in seen_keys:
                raise ValueError(f"duplicate {entity_name} {key_col}: {key}")
            seen_keys.add(key)

    return df


def write_sample_meta(
    records: Sequence[Mapping[str, object]],
    out_path,
    *,
    sample_id_col: str = "sample_id",
    sample_key_col: str = "sample_key",
) -> str:
    """Write dense sample metadata parquet from list-of-dict records."""

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_dense_meta_frame(
        records,
        id_col=str(sample_id_col),
        id_dtype=pl.Int64,
        entity_name="sample",
        key_col=str(sample_key_col),
    )
    df.write_parquet(out_path)
    return str(out_path)


def write_feature_meta(
    records: Sequence[Mapping[str, object]],
    out_path,
    *,
    feature_id_col: str = "feature_id",
    feature_key_col: str = "feature_key",
) -> str:
    """Write dense feature metadata parquet from list-of-dict records."""

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_dense_meta_frame(
        records,
        id_col=str(feature_id_col),
        id_dtype=pl.Int32,
        entity_name="feature",
        key_col=str(feature_key_col),
    )
    df.write_parquet(out_path)
    return str(out_path)
