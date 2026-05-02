"""Internal lightweight types shared by the scalar feature shard package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One feature-vs-y candidate used by the incremental selector."""

    feature_id: int
    shard_id: int
    offset_in_shard: int
    r2_y: float
    n_valid_y: int
