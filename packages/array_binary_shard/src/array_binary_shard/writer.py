"""커스텀 array binary shard를 만드는 고수준 writer facade."""

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
    """sample-major bundle manifest로부터 binary array shard를 생성한다.

    Args:
        source: `array_bundle_manifest.json` 경로.
        out_dir: binary shard 세트를 쓸 출력 디렉터리.
        options: 선택적 `BuildOptions` 묶음.
        target_shard_mb: 선호하는 최대 shard 크기(MB).
        samples_per_block: 논리 block 하나에 담을 sample 수.
        n_shards: 선택적 명시 shard 개수 override.
        codec: payload codec 이름.
        zstd_level: `codec='zstd'`일 때만 사용하는 압축 레벨.
        sample_key_col: 외부 key가 들어 있는 sample metadata 컬럼 이름.
        feature_key_col: 외부 key가 들어 있는 feature metadata 컬럼 이름.
        return_stats: build 통계를 함께 반환할지 여부.

    Returns:
        binary shard manifest 경로 또는 `return_stats=True`일 때
        `(manifest_path, stats)`.
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
    """기존 parquet array shard 세트를 binary shard로 변환한다.

    Args:
        source: `array_shard_manifest.json` 경로.
        out_dir: 변환된 binary shard 세트를 쓸 출력 디렉터리.
        codec: payload codec 이름.
        zstd_level: `codec='zstd'`일 때만 사용하는 압축 레벨.
        sample_key_col: 외부 key가 들어 있는 sample metadata 컬럼 이름.
        feature_key_col: 외부 key가 들어 있는 feature metadata 컬럼 이름.
        return_stats: 변환 통계를 함께 반환할지 여부.

    Returns:
        변환된 binary shard manifest 경로 또는 `return_stats=True`일 때
        `(manifest_path, stats)`.
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
