"""Compatibility wrapper for the package scalar dense-long builder."""

from .._package_sources import ensure_package_source

ensure_package_source("scalar_feature_shard")

from scalar_feature_shard.builder import ScalarBuildSessionStatus, ScalarDatasetBuilder

__all__ = ["ScalarBuildSessionStatus", "ScalarDatasetBuilder"]
