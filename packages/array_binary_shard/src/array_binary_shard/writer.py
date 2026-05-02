"""High-level writer facade for custom array binary shards."""

from ._impl.binary_storage import (
    build_array_binary_shards_from_array_manifest,
    build_array_binary_shards_from_bundles,
)
from ._impl.config import ArrayShardConfig

from .models import BuildOptions


def _resolve_options(
    options: BuildOptions | None,
    *,
    target_shard_mb,
    samples_per_block,
    n_shards,
    codec,
    zstd_level,
    sample_key_col,
    feature_key_col,
):
    """Merge explicit keyword arguments over an optional `BuildOptions` object."""
    base = options or BuildOptions()
    return BuildOptions(
        target_shard_mb=int(base.target_shard_mb if target_shard_mb is None else target_shard_mb),
        samples_per_block=int(base.samples_per_block if samples_per_block is None else samples_per_block),
        n_shards=base.n_shards if n_shards is None else int(n_shards),
        codec=str(base.codec if codec is None else codec),
        zstd_level=int(base.zstd_level if zstd_level is None else zstd_level),
        sample_key_col=str(base.sample_key_col if sample_key_col is None else sample_key_col),
        feature_key_col=str(base.feature_key_col if feature_key_col is None else feature_key_col),
    )


def build_shard(
    source,
    out_dir,
    *,
    options: BuildOptions | None = None,
    target_shard_mb: int | None = None,
    samples_per_block: int | None = None,
    n_shards: int | None = None,
    codec: str | None = None,
    zstd_level: int | None = None,
    sample_key_col: str | None = None,
    feature_key_col: str | None = None,
    return_stats: bool = False,
):
    """Build binary array shards from a sample-major bundle manifest.

    Args:
        source: Path to `array_bundle_manifest.json`.
        out_dir: Output directory for the binary shard set.
        options: Optional `BuildOptions` bundle.
        target_shard_mb: Preferred maximum shard size in megabytes.
        samples_per_block: Number of samples packed into one logical block.
        n_shards: Optional explicit shard-count override.
        codec: Payload codec name.
        zstd_level: Compression level used only when `codec='zstd'`.
        sample_key_col: Sample metadata column containing external keys.
        feature_key_col: Feature metadata column containing external keys.
        return_stats: Whether to return build statistics.

    Returns:
        Either the binary shard manifest path, or `(manifest_path, stats)` when
        `return_stats=True`.
    """

    resolved = _resolve_options(
        options,
        target_shard_mb=target_shard_mb,
        samples_per_block=samples_per_block,
        n_shards=n_shards,
        codec=codec,
        zstd_level=zstd_level,
        sample_key_col=sample_key_col,
        feature_key_col=feature_key_col,
    )
    config = ArrayShardConfig(
        samples_per_block=int(resolved.samples_per_block),
        target_shard_bytes=int(resolved.target_shard_mb) * 1024 * 1024,
        n_shards=0 if resolved.n_shards is None else int(resolved.n_shards),
        row_group_size=0,
        use_tmp_spill=False,
        spill_bucket_target_bytes=8 * 1024 * 1024,
    )
    return build_array_binary_shards_from_bundles(
        str(source),
        str(out_dir),
        config=config,
        codec=resolved.codec,
        zstd_level=int(resolved.zstd_level),
        sample_key_col=resolved.sample_key_col,
        feature_key_col=resolved.feature_key_col,
        return_stats=bool(return_stats),
    )


def convert_parquet_shard(
    source,
    out_dir,
    *,
    codec: str = "none",
    zstd_level: int = 3,
    sample_key_col: str = "sample_key",
    feature_key_col: str = "feature_key",
    return_stats: bool = False,
):
    """Convert an existing parquet array shard set into binary shards.

    Args:
        source: Path to `array_shard_manifest.json`.
        out_dir: Output directory for the converted binary shard set.
        codec: Payload codec name.
        zstd_level: Compression level used only when `codec='zstd'`.
        sample_key_col: Sample metadata column containing external keys.
        feature_key_col: Feature metadata column containing external keys.
        return_stats: Whether to return conversion statistics.

    Returns:
        Either the converted binary shard manifest path, or `(manifest_path, stats)`
        when `return_stats=True`.
    """

    return build_array_binary_shards_from_array_manifest(
        str(source),
        str(out_dir),
        codec=str(codec),
        zstd_level=int(zstd_level),
        sample_key_col=str(sample_key_col),
        feature_key_col=str(feature_key_col),
        return_stats=bool(return_stats),
    )
