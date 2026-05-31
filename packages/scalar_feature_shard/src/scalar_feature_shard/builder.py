"""Public scalar builder facade."""

from ._impl.raw_builder import ScalarBuildSessionStatus
from ._impl.raw_builder import ScalarDatasetBuilder as _ImplScalarDatasetBuilder
from .config import ScalarShardBuildOptions
from .models import BuildOptions


def _resolve_build_options(build_options: BuildOptions | ScalarShardBuildOptions | None):
    if build_options is None:
        return ScalarShardBuildOptions()
    if isinstance(build_options, ScalarShardBuildOptions):
        return build_options
    return ScalarShardBuildOptions(
        target_shard_mb=int(build_options.target_shard_mb),
        n_shards=None if build_options.n_shards is None else int(build_options.n_shards),
        feature_id_col=str(build_options.feature_id_col),
        value_col=str(build_options.value_col),
        sample_id_col=str(build_options.sample_id_col),
        sample_key_col=str(build_options.sample_key_col),
        feature_key_col=str(build_options.feature_key_col),
        path_col=str(build_options.path_col),
        y_col=str(build_options.y_col),
        stats_y_cols=None if build_options.stats_y_cols is None else tuple(str(value) for value in build_options.stats_y_cols),
        values_dtype=str(build_options.values_dtype),
        valid_dtype=str(build_options.valid_dtype),
    )


class ScalarDatasetBuilder(_ImplScalarDatasetBuilder):
    """Public scalar dataset builder.

    Samples are committed as independent raw parquet files, so callers may
    write sample ids in any order. A sequential loop over `pending_sample_ids()`
    is just a special case of the same API.
    """

    def __init__(
        self,
        out_dir,
        sample_meta_path,
        *,
        feature_meta_path=None,
        feature_keys=None,
        build_options: BuildOptions | ScalarShardBuildOptions | None = None,
    ):
        super().__init__(
            out_dir=out_dir,
            sample_meta_path=sample_meta_path,
            feature_meta_path=feature_meta_path,
            feature_keys=feature_keys,
            build_options=_resolve_build_options(build_options),
        )


__all__ = [
    "ScalarBuildSessionStatus",
    "ScalarDatasetBuilder",
]
