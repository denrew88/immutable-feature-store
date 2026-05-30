from __future__ import annotations

import polars as pl

from ..types import LogicalType, PointColumnSpec


def _normalize_point_schema(point_schema):
    if point_schema is None:
        raise ValueError("point_schema must be provided explicitly")
    specs = []
    for spec in point_schema:
        if hasattr(spec, "name") and hasattr(spec, "storage_type") and hasattr(spec, "logical_type"):
            item = PointColumnSpec(
                name=spec.name,
                storage_type=spec.storage_type,
                logical_type=spec.logical_type,
            )
        else:
            item = PointColumnSpec(
                name=str(spec["name"]),
                storage_type=spec["storage_type"],
                logical_type=spec.get("logical_type", LogicalType.CONTINUOUS),
            )
        specs.append(item)
    if not specs:
        raise ValueError("point_schema must not be empty")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("point_schema column names must be unique")
    return specs


def _load_dense_meta(meta_path: str, id_col: str, entity_name: str, key_col: str = "") -> pl.DataFrame:
    df = pl.read_parquet(meta_path)
    if id_col in df.columns:
        ids = [int(value) for value in df[id_col].to_list()]
        expected = list(range(df.height))
        if ids != expected:
            raise ValueError(f"{entity_name} metadata column '{id_col}' must equal dense row ids 0..N-1")
    if key_col:
        if key_col not in df.columns:
            raise ValueError(f"{entity_name} metadata must have key column: {key_col}")
        series = df[key_col]
        if series.null_count() != 0:
            raise ValueError(f"{entity_name} metadata key column '{key_col}' must not contain nulls")
        if int(series.n_unique()) != int(df.height):
            raise ValueError(f"{entity_name} metadata key column '{key_col}' must be unique")
    return df
