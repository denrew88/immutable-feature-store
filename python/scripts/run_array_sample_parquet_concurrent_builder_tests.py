from __future__ import annotations

import shutil
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for path in (
    PYTHON_ROOT,
    REPO_ROOT / "packages" / "array_sample_parquet" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fs.array.metadata import write_feature_meta, write_sample_meta
from fs.types import LogicalType, PointColumnSpec, StorageType
from array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    open_array_sample_parquet,
)
from validate_array_sample_parquet_exhaustive import validate_manifest


def _schema():
    return [
        PointColumnSpec("ts", StorageType.INT64, LogicalType.TIMESTAMP_NS),
        PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER),
        PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL),
    ]


def _options():
    return ArraySampleParquetBuildOptions(
        target_part_bytes=1024 * 1024,
        max_part_samples=4,
        compression="none",
    )


def _columns_for(rng: np.random.Generator, sample_id: int, feature_id: int, trace_len: int):
    trace_len = int(trace_len)
    base = np.datetime64("2026-06-05T00:00:00", "ns") + np.timedelta64(
        int(sample_id) * 1000 + int(feature_id),
        "ns",
    )
    ts = base + np.arange(trace_len, dtype=np.int64).astype("timedelta64[ns]")
    value = rng.normal(loc=float(sample_id) * 0.1, scale=3.0, size=trace_len).astype(np.float64)
    if trace_len > 0 and rng.random() < 0.25:
        value[int(rng.integers(0, trace_len))] = np.nan
    return {
        "ts": ts,
        "value": value,
        "phase": rng.integers(-3, 4, size=trace_len, dtype=np.int32),
        "ch_step": rng.choice(np.asarray(["worker_a", "worker_b", "worker_c", "idle"], dtype=object), size=trace_len).tolist(),
    }


def _write_samples(args) -> list[int]:
    out_dir, sample_meta_path, feature_meta_path, sample_ids, n_features = args
    with ArraySampleParquetDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        _schema(),
        feature_meta_path=feature_meta_path,
        options=_options(),
    ) as builder:
        for sample_id in sample_ids:
            rng = np.random.default_rng(0xA11A_0000 + int(sample_id))
            with builder.sample(sample_id=int(sample_id), skip_if_completed=True) as sample:
                if sample.skipped:
                    continue
                for feature_id in rng.permutation(int(n_features)):
                    if rng.random() < 0.30:
                        continue
                    trace_len = int(rng.integers(0, 8))
                    sample.add_trace(
                        feature_id=int(feature_id),
                        columns=_columns_for(rng, int(sample_id), int(feature_id), trace_len),
                    )
    return [int(sample_id) for sample_id in sample_ids]


def main(argv=None):
    ap = ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--n-features", type=int, default=12)
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args(argv)

    root = Path(args.out_dir).resolve() if args.out_dir else REPO_ROOT / "data" / "tmp_array_sample_parquet_concurrent_builder_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    n_samples = int(args.n_samples)
    n_features = int(args.n_features)
    n_workers = int(args.n_workers)
    started = time.perf_counter()

    sample_meta_path = write_sample_meta(
        [{"sample_key": f"sample_{sample_id:06d}", "split": "test"} for sample_id in range(n_samples)],
        root / "sample_meta.parquet",
    )
    feature_meta_path = write_feature_meta(
        [{"feature_key": f"feature_{feature_id:06d}", "group": f"g{feature_id % 3}"} for feature_id in range(n_features)],
        root / "feature_meta.parquet",
    )
    out_dir = str(root / "dataset")

    init_started = time.perf_counter()
    with ArraySampleParquetDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        _schema(),
        feature_meta_path=feature_meta_path,
        options=_options(),
    ) as builder:
        assert builder.pending_sample_ids() == list(range(n_samples))
    init_sec = time.perf_counter() - init_started

    assignments = [[] for _ in range(n_workers)]
    for sample_id in range(n_samples):
        assignments[sample_id % n_workers].append(sample_id)

    write_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(
            pool.map(
                _write_samples,
                [(out_dir, sample_meta_path, feature_meta_path, ids, n_features) for ids in assignments],
            )
        )
    write_sec = time.perf_counter() - write_started
    committed = sorted(sample_id for group in results for sample_id in group)
    assert committed == list(range(n_samples)), committed

    finish_started = time.perf_counter()
    with ArraySampleParquetDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        _schema(),
        feature_meta_path=feature_meta_path,
        options=_options(),
    ) as builder:
        assert builder.completed_sample_ids() == list(range(n_samples))
        assert builder.pending_sample_ids() == []
        manifest_path = builder.finish()
    finish_sec = time.perf_counter() - finish_started

    validate_manifest(manifest_path, stage_dir=out_dir, label="array-sample-parquet-concurrent")
    reader = open_array_sample_parquet(manifest_path)
    traces = reader.get_traces(
        sample_ids=[0, n_samples - 1],
        feature_ids=list(range(min(3, n_features))),
        include_missing=True,
        decode_categorical=True,
    )
    assert len(traces) == 2 * min(3, n_features)

    total_sec = time.perf_counter() - started
    print(
        "array sample parquet concurrent builder tests passed "
        f"n_samples={n_samples} n_features={n_features} n_workers={n_workers} "
        f"init_sec={init_sec:.3f} write_sec={write_sec:.3f} "
        f"finish_sec={finish_sec:.3f} total_sec={total_sec:.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
