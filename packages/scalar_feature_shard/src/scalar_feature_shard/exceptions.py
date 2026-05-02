"""Public exception types for the scalar feature shard package."""


class ScalarFeatureShardError(Exception):
    """Base class for package-level scalar feature shard errors."""


class ManifestFormatError(ScalarFeatureShardError):
    """Raised when a manifest cannot be parsed as a supported scalar shard manifest."""


class FeatureNotFoundError(ScalarFeatureShardError):
    """Raised when a requested feature id or feature key does not exist."""


class SampleNotFoundError(ScalarFeatureShardError):
    """Raised when a requested sample id or sample key does not exist."""
