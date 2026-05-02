"""Private helpers for resolving sample ids."""

import numpy as np
import polars as pl


def build_sample_id_index(sample_meta_path: str, sample_id_col: str = "sample_id", sample_row_col: str = "sample_row"):
    """Build a sample id to sample row lookup table.

    Args:
        sample_meta_path: Path to the sample metadata parquet file.
        sample_id_col: Column containing external sample ids.
        sample_row_col: Optional column containing dense sample row indices.

    Returns:
        A dictionary mapping each sample id to its sample row index.
    """
    df = pl.read_parquet(sample_meta_path)
    if sample_id_col not in df.columns:
        raise ValueError(f"sample_meta parquet must have column: {sample_id_col}")
    sample_ids = df[sample_id_col].to_numpy()
    if sample_row_col in df.columns:
        sample_rows = df[sample_row_col].to_numpy()
    else:
        sample_rows = np.arange(df.height, dtype=np.int64)
    return {int(sample_id): int(sample_row) for sample_id, sample_row in zip(sample_ids, sample_rows)}
