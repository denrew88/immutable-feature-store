"""Public dense-long scalar writer facade."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from ._impl.dense_long import build_dense_long_shards_from_sample_major_manifest
from ._impl.storage_common import SAMPLE_MAJOR_MANIFEST_FORMAT
from .config import ScalarShardBuildOptions
from .models import BuildOptions


def _resolve_options(
    options: BuildOptions | ScalarShardBuildOptions | None,
    *,
    target_shard_mb,
    feature_id_col,
    value_col,
    sample_id_col,
    sample_key_col,
    feature_key_col,
    path_col,
    y_col,
    stats_y_cols,
):
    """Merge object-style build options with explicit keyword overrides."""

    base = options or BuildOptions()
    if isinstance(base, ScalarShardBuildOptions):
        base = BuildOptions(
            target_shard_mb=base.target_shard_mb,
            n_shards=base.n_shards,
            feature_id_col=base.feature_id_col,
            value_col=base.value_col,
            sample_id_col=base.sample_id_col,
            sample_key_col=base.sample_key_col,
            feature_key_col=base.feature_key_col,
            path_col=base.path_col,
            y_col=base.y_col,
            stats_y_cols=base.stats_y_cols,
        )
    return BuildOptions(
        target_shard_mb=int(base.target_shard_mb if target_shard_mb is None else target_shard_mb),
        n_shards=None,
        feature_id_col=str(base.feature_id_col if feature_id_col is None else feature_id_col),
        value_col=str(base.value_col if value_col is None else value_col),
        sample_id_col=str(base.sample_id_col if sample_id_col is None else sample_id_col),
        sample_key_col=str(base.sample_key_col if sample_key_col is None else sample_key_col),
        feature_key_col=str(base.feature_key_col if feature_key_col is None else feature_key_col),
        path_col=str(base.path_col if path_col is None else path_col),
        y_col=str(base.y_col if y_col is None else y_col),
        stats_y_cols=base.stats_y_cols if stats_y_cols is None else tuple(str(value) for value in stats_y_cols),
    )


def _resolve_sample_paths_from_df(df: pl.DataFrame, sample_meta_path: str, path_col: str) -> list[str]:
    if path_col not in df.columns:
        raise ValueError(f"sample_meta parquet must contain sample path column: {path_col}")
    base_dir = os.path.dirname(os.path.abspath(sample_meta_path))
    out: list[str] = []
    for raw_path in df[path_col].to_list():
        value = str(raw_path)
        out.append(value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value)))
    return out


def _stage_manifest_from_sample_meta(
    source_path: str,
    out_dir: str,
    *,
    feature_meta_path: str | None,
    options: BuildOptions,
) -> str:
    sample_meta_path = str(Path(source_path).expanduser().resolve())
    sample_meta_df = pl.read_parquet(sample_meta_path)
    feature_meta = (
        str(Path(feature_meta_path).expanduser().resolve())
        if feature_meta_path is not None
        else os.path.join(os.path.dirname(sample_meta_path), "feature_meta.parquet")
    )
    if not os.path.exists(feature_meta):
        raise ValueError(f"feature_meta_path does not exist: {feature_meta}")

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "sample_major_manifest.json")
    payload = {
        "format": SAMPLE_MAJOR_MANIFEST_FORMAT,
        "version": 1,
        "sample_meta_path": sample_meta_path,
        "feature_meta_path": feature_meta,
        "sample_paths": _resolve_sample_paths_from_df(sample_meta_df, sample_meta_path, str(options.path_col)),
        "sample_ids": (
            [int(value) for value in sample_meta_df[str(options.sample_id_col)].to_list()]
            if str(options.sample_id_col) in sample_meta_df.columns
            else list(range(sample_meta_df.height))
        ),
        "sample_id_col": str(options.sample_id_col),
        "feature_id_col": str(options.feature_id_col),
        "value_col": str(options.value_col),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return manifest_path


def build_shard(
    source,
    out_dir,
    *,
    feature_meta_path=None,
    options: BuildOptions | ScalarShardBuildOptions | None = None,
    target_shard_mb: int | None = None,
    n_shards: int | None = None,
    feature_id_col: str | None = None,
    value_col: str | None = None,
    sample_id_col: str | None = None,
    sample_key_col: str | None = None,
    feature_key_col: str | None = None,
    path_col: str | None = None,
    y_col: str | None = None,
    stats_y_cols=None,
    values_dtype: str | None = None,
    valid_dtype: str | None = None,
):
    """Build a dense-long scalar shard from sample-major rows.

    `source` may be a `scalar-sample-major-v1` manifest or a sample metadata
    parquet containing `path_col`. The generated artifact is always
    `dense_long_shard_manifest.json` plus `dense_long_parts/*.parquet`.
    Deprecated fixed-shard options (`n_shards`, `values_dtype`, `valid_dtype`)
    are accepted for call-site convenience but ignored.
    """

    if n_shards is not None:
        # dense-long partitions by target bytes, not fixed shard count.
        n_shards = None
    if values_dtype is not None and str(values_dtype).lower() not in {"float64", "double"}:
        raise ValueError("dense-long scalar shards store value as float64")
    if valid_dtype is not None and str(valid_dtype).lower() != "uint8":
        raise ValueError("dense-long scalar shards store mask as uint8")

    resolved = _resolve_options(
        options,
        target_shard_mb=target_shard_mb,
        feature_id_col=feature_id_col,
        value_col=value_col,
        sample_id_col=sample_id_col,
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
        path_col=path_col,
        y_col=y_col,
        stats_y_cols=stats_y_cols,
    )
    source_path = str(Path(source).expanduser().resolve())
    if Path(source_path).suffix.lower() == ".json":
        with open(source_path, "r", encoding="utf-8") as f:
            source_json = json.load(f)
        if str(source_json.get("format", "")) != SAMPLE_MAJOR_MANIFEST_FORMAT:
            raise ValueError(f"unsupported scalar build manifest format: {source_json.get('format')}")
        stage_manifest = source_path
    else:
        stage_manifest = _stage_manifest_from_sample_meta(
            source_path,
            str(Path(out_dir).expanduser().resolve()),
            feature_meta_path=None if feature_meta_path is None else str(feature_meta_path),
            options=resolved,
        )

    return build_dense_long_shards_from_sample_major_manifest(
        stage_manifest,
        str(out_dir),
        feature_meta_path=None if feature_meta_path is None else str(feature_meta_path),
        target_part_bytes=int(resolved.target_shard_mb) * 1024 * 1024,
        feature_id_col=resolved.feature_id_col,
        value_col=resolved.value_col,
        sample_id_col=resolved.sample_id_col,
        sample_key_col=resolved.sample_key_col,
        feature_key_col=resolved.feature_key_col,
        y_col=resolved.y_col,
        stats_y_cols=None if resolved.stats_y_cols is None else list(resolved.stats_y_cols),
        compression="zstd",
    )
