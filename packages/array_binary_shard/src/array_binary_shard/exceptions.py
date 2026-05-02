"""Public exception types for the array binary shard package."""


class ArrayBinaryShardError(Exception):
    """Base class for package-level array binary shard errors."""


class ManifestFormatError(ArrayBinaryShardError):
    """Raised when a manifest cannot be parsed as a supported binary shard manifest."""


class FeatureNotFoundError(ArrayBinaryShardError):
    """Raised when a requested feature id does not exist in the shard set."""


class SampleNotFoundError(ArrayBinaryShardError):
    """Raised when a requested sample id does not exist in sample metadata."""

