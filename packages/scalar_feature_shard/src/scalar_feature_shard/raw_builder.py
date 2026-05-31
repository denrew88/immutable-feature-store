"""Public facade for out-of-order scalar raw-sample builds."""

from ._impl.raw_builder import ScalarRawBuildStatus
from ._impl.raw_builder import ScalarRawDatasetBuilder as _ImplScalarRawDatasetBuilder
from .builder import _resolve_build_options
from .config import ScalarShardBuildOptions
from .models import BuildOptions


class ScalarRawDatasetBuilder(_ImplScalarRawDatasetBuilder):
    """Raw scalar builder that accepts samples in any order."""

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
    "ScalarRawBuildStatus",
    "ScalarRawDatasetBuilder",
]
