from __future__ import annotations

import shutil
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "packages" / "scalar_feature_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder, write_feature_meta, write_sample_meta
from validate_scalar_dense_long_exhaustive import validate_manifest


def _values_for(sample_id: int, n_features: int) -> dict[int, float]:
    rng = np.random.default_rng(0x5EED_0000 + int(sample_id))
    present = rng.random(int(n_features)) >= 0.23
    values = rng.normal(loc=0.0, scale=10.0, size=int(n_features)) + rng.uniform(-3.0, 3.0, size=int(n_features))
    return {int(feature_id): float(values[feature_id]) for feature_id in np.flatnonzero(present)}


def _write_samples(args) -> list[int]:
    out_dir, sample_meta_path, feature_meta_path, sample_ids, n_features = args
    with ScalarDatasetBuilder(
        out_dir=out_dir,
        sample_meta_path=sample_meta_path,
        feature_meta_path=feature_meta_path,
        build_options=BuildOptions(target_shard_mb=1),
    ) as builder:
        for sample_id in sample_ids:
            builder.write_sample(sample_id, _values_for(sample_id, n_features), skip_if_completed=True)
    return [int(sample_id) for sample_id in sample_ids]


def main():
    ap = ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--n-features", type=int, default=128)
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_dir).resolve() if args.out_dir else Path(__file__).resolve().parents[2] / "data" / "tmp_scalar_concurrent_builder_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n_samples = int(args.n_samples)
    n_features = int(args.n_features)
    n_workers = int(args.n_workers)

    sample_meta_path = write_sample_meta(
        [{"sample_key": f"sample_{sample_id:06d}", "y": float(sample_id % 3)} for sample_id in range(n_samples)],
        root / "sample_meta.parquet",
    )
    feature_meta_path = write_feature_meta(
        [{"feature_key": f"feature_{feature_id:06d}"} for feature_id in range(n_features)],
        root / "feature_meta.parquet",
    )
    out_dir = str(root / "stage")

    # 병렬 worker가 동시에 처음 초기화하지 않도록 supervisor가 먼저 stage를 만든다.
    with ScalarDatasetBuilder(
        out_dir=out_dir,
        sample_meta_path=sample_meta_path,
        feature_meta_path=feature_meta_path,
        build_options=BuildOptions(target_shard_mb=1),
    ) as builder:
        assert builder.pending_sample_ids() == list(range(n_samples))
    init_sec = time.perf_counter() - started

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
    with ScalarDatasetBuilder(
        out_dir=out_dir,
        sample_meta_path=sample_meta_path,
        feature_meta_path=feature_meta_path,
        build_options=BuildOptions(target_shard_mb=1),
    ) as builder:
        assert builder.completed_sample_ids() == list(range(n_samples))
        assert builder.pending_sample_ids() == []
        if args.skip_build:
            manifest_path = builder.finish_stage()
        else:
            manifest_path = builder.build_dense_long_shards(out_dir=str(root / "scalar_shard"), keep_raw=True)
    finish_sec = time.perf_counter() - finish_started

    if not args.skip_build:
        validate_manifest(
            Path(manifest_path),
            sample_major_manifest_path=Path(out_dir) / "sample_major_manifest.json",
            progress_every=0,
        )

    total_sec = time.perf_counter() - started
    print(
        "python scalar concurrent builder tests passed "
        f"n_samples={n_samples} n_features={n_features} n_workers={n_workers} "
        f"skip_build={bool(args.skip_build)} init_sec={init_sec:.3f} "
        f"write_sec={write_sec:.3f} finish_sec={finish_sec:.3f} total_sec={total_sec:.3f}"
    )


if __name__ == "__main__":
    main()
