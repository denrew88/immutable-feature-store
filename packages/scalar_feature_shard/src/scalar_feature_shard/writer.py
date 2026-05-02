"""Thin public writer facade for scalar feature shards."""

from ._impl.parquet_storage import build_shards_from_sample_major
from .config import ScalarShardBuildOptions
from .models import BuildOptions


def _resolve_options(
    options: BuildOptions | ScalarShardBuildOptions | None,
    *,
    target_shard_mb,
    n_shards,
    feature_id_col,
    value_col,
    sample_id_col,
    sample_key_col,
    feature_key_col,
    path_col,
    y_col,
    stats_y_cols,
    values_dtype,
    valid_dtype,
):
    """Merge explicit keyword arguments over optional build options."""

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
            values_dtype=base.values_dtype,
            valid_dtype=base.valid_dtype,
        )
    return BuildOptions(
        target_shard_mb=int(base.target_shard_mb if target_shard_mb is None else target_shard_mb),
        n_shards=base.n_shards if n_shards is None else int(n_shards),
        feature_id_col=str(base.feature_id_col if feature_id_col is None else feature_id_col),
        value_col=str(base.value_col if value_col is None else value_col),
        sample_id_col=str(base.sample_id_col if sample_id_col is None else sample_id_col),
        sample_key_col=str(base.sample_key_col if sample_key_col is None else sample_key_col),
        feature_key_col=str(base.feature_key_col if feature_key_col is None else feature_key_col),
        path_col=str(base.path_col if path_col is None else path_col),
        y_col=str(base.y_col if y_col is None else y_col),
        stats_y_cols=base.stats_y_cols if stats_y_cols is None else tuple(str(value) for value in stats_y_cols),
        values_dtype=str(base.values_dtype if values_dtype is None else values_dtype),
        valid_dtype=str(base.valid_dtype if valid_dtype is None else valid_dtype),
    )


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
    """Build scalar shards from sample-major metadata and per-sample parquet files.

    Args:
        source: Path to sample-major `sample_meta.parquet`.
        out_dir: Output directory for the standalone shard artifact.
        feature_meta_path: Optional path to dense feature metadata.
        options: Optional `BuildOptions` bundle.
        target_shard_mb: Preferred maximum shard size in megabytes.
        n_shards: Optional explicit shard-count override.
        feature_id_col: Feature id column name in sample files.
        value_col: Scalar value column name in sample files.
        sample_id_col: Dense sample id column name in sample metadata.
        sample_key_col: External sample-key column name in sample metadata.
        feature_key_col: External feature-key column name in feature metadata.
        path_col: Sample file path column name in sample metadata.
        y_col: Default target column name used when `stats_y_cols` is omitted.
        stats_y_cols: Target columns to precompute into `selection_stats/`.
        values_dtype: Encoded values dtype.
        valid_dtype: Encoded validity dtype.

    Returns:
        Path to the generated `shard_manifest.json`.
    """

    resolved = _resolve_options(
        options,
        target_shard_mb=target_shard_mb,
        n_shards=n_shards,
        feature_id_col=feature_id_col,
        value_col=value_col,
        sample_id_col=sample_id_col,
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
        path_col=path_col,
        y_col=y_col,
        stats_y_cols=stats_y_cols,
        values_dtype=values_dtype,
        valid_dtype=valid_dtype,
    )
    return build_shards_from_sample_major(
        str(source),
        str(out_dir),
        feature_meta_path=None if feature_meta_path is None else str(feature_meta_path),
        n_shards=None if resolved.n_shards is None else int(resolved.n_shards),
        target_shard_bytes=int(resolved.target_shard_mb) * 1024 * 1024,
        feature_id_col=resolved.feature_id_col,
        value_col=resolved.value_col,
        sample_id_col=resolved.sample_id_col,
        sample_key_col=resolved.sample_key_col,
        feature_key_col=resolved.feature_key_col,
        path_col=resolved.path_col,
        y_col=resolved.y_col,
        stats_y_cols=None if resolved.stats_y_cols is None else list(resolved.stats_y_cols),
        values_dtype=resolved.values_dtype,
        valid_dtype=resolved.valid_dtype,
    )
