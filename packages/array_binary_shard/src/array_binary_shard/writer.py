"""커스텀 array binary shard를 만드는 고수준 writer facade."""

from ._impl.binary_storage import build_array_binary_shards_from_bundles
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
    """선택적 `BuildOptions` 객체 위에 명시적 keyword 인자를 덮어써서 병합한다."""
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
    """sample-major bundle manifest로부터 binary array shard를 생성한다."""

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
