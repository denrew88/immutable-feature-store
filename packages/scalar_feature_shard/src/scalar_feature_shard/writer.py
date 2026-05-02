"""scalar feature shard를 위한 얇은 public writer facade."""

import json
from pathlib import Path

from ._impl.parquet_storage import build_shards_from_sample_bundles, build_shards_from_sample_major
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
    """선택적 build 옵션 위에 명시적 keyword 인자를 덮어써서 병합한다."""

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
    """sample-major metadata와 sample별 parquet 파일에서 scalar shard를 생성한다.

    Args:
        source: sample-major `sample_meta.parquet` 경로.
        out_dir: standalone shard artifact를 쓸 출력 디렉터리.
        feature_meta_path: 선택적 dense feature metadata 경로.
        options: 선택적 `BuildOptions` 묶음.
        target_shard_mb: 선호하는 최대 shard 크기(MB).
        n_shards: 선택적 명시 shard 개수 override.
        feature_id_col: sample 파일의 feature id 컬럼 이름.
        value_col: sample 파일의 scalar value 컬럼 이름.
        sample_id_col: sample metadata의 dense sample id 컬럼 이름.
        sample_key_col: sample metadata의 외부 sample-key 컬럼 이름.
        feature_key_col: feature metadata의 외부 feature-key 컬럼 이름.
        path_col: sample metadata의 sample 파일 경로 컬럼 이름.
        y_col: `stats_y_cols`를 생략했을 때 사용할 기본 target 컬럼 이름.
        stats_y_cols: `selection_stats/`에 미리 계산해 둘 target 컬럼 목록.
        values_dtype: 인코딩된 values dtype.
        valid_dtype: 인코딩된 validity dtype.

    Returns:
        생성된 `shard_manifest.json` 경로.
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
    source_path = str(source)
    build_fn = build_shards_from_sample_major
    if Path(source_path).suffix.lower() == ".json":
        with open(source_path, "r", encoding="utf-8") as f:
            source_json = json.load(f)
        if str(source_json.get("format", "")) == "scalar-sample-bundles":
            build_fn = build_shards_from_sample_bundles

    return build_fn(
        source_path,
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
