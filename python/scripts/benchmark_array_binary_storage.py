import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fs.array.binary_storage import (
    ArrayBinaryShardReader,
    build_array_binary_shards_from_array_manifest,
    load_array_binary_shard_manifest,
)
from fs.array.storage import ArrayShardReader, build_array_feature_locator_index, load_array_shard_manifest


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _default_parquet_manifest() -> str:
    """Return the default parquet manifest used by the benchmark script."""
    return str(
        _repo_root()
        / "data"
        / "tmp_array_shard_bench_2000x1024"
        / "build_spill_b8_rg64"
        / "array_shard_manifest.json"
    )


def _size_of_path(path: str) -> int:
    """Return the total size of a file or directory tree in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def _artifact_bytes_for_parquet(manifest) -> int:
    """Return the total on-disk bytes used by a parquet shard artifact."""
    return _size_of_path(os.path.dirname(manifest.locator_path))


def _artifact_bytes_for_binary(manifest_path: str, manifest) -> int:
    """Return the total on-disk bytes used by a binary shard artifact."""
    return _size_of_path(manifest.shard_path) + _size_of_path(manifest_path)


def _median_ms(fn, repeats: int, warmup: int) -> float:
    """Measure the median execution time of a callable in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return round(float(np.median(np.asarray(samples, dtype=np.float64))), 1)


def main():
    """Benchmark parquet versus binary array shard lookup speed and artifact size."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-manifest", default=_default_parquet_manifest())
    ap.add_argument("--binary-out-dir", default="")
    ap.add_argument("--rebuild-binary", action="store_true")
    ap.add_argument("--codec", choices=["none", "zstd"], default="none")
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--n-features", type=int, default=20)
    ap.add_argument("--feature-id", type=int, default=0)
    ap.add_argument("--contiguous-count", type=int, default=256)
    ap.add_argument("--spread-count", type=int, default=256)
    args = ap.parse_args()

    parquet_manifest = load_array_shard_manifest(args.parquet_manifest)
    parquet_reader = ArrayShardReader(parquet_manifest)
    parquet_locator_index = build_array_feature_locator_index(parquet_manifest.locator_path)

    binary_out_dir = args.binary_out_dir or str(Path(args.parquet_manifest).resolve().parent.parent / "build_binary_ref")
    binary_manifest_path = os.path.join(binary_out_dir, "array_binary_shard_manifest.json")
    binary_build_s = None
    if args.rebuild_binary or not os.path.exists(binary_manifest_path):
        if os.path.exists(binary_out_dir):
            shutil.rmtree(binary_out_dir)
        started = time.perf_counter()
        binary_manifest_path = build_array_binary_shards_from_array_manifest(
            args.parquet_manifest,
            binary_out_dir,
            codec=args.codec,
            zstd_level=args.zstd_level,
        )
        binary_build_s = round(time.perf_counter() - started, 3)

    binary_manifest = load_array_binary_shard_manifest(binary_manifest_path)
    binary_reader = ArrayBinaryShardReader(binary_manifest)

    contiguous_rows = list(range(min(args.contiguous_count, parquet_manifest.n_samples)))
    spread_rows = np.linspace(
        0,
        max(parquet_manifest.n_samples - 1, 0),
        num=min(args.spread_count, parquet_manifest.n_samples),
        dtype=np.int64,
    ).tolist()
    many_feature_ids = list(range(args.n_features))

    def parquet_many_features_one_sample():
        """Benchmark many-feature single-sample lookup on parquet shards."""
        for feature_id in many_feature_ids:
            parquet_reader.load_feature_samples(feature_id, [0], locator_index=parquet_locator_index)

    def binary_many_features_one_sample():
        """Benchmark many-feature single-sample lookup on binary shards."""
        for feature_id in many_feature_ids:
            binary_reader.load_feature_samples(feature_id, [0])

    def parquet_single_feature_contiguous():
        """Benchmark one-feature contiguous-sample lookup on parquet shards."""
        parquet_reader.load_feature_samples(args.feature_id, contiguous_rows, locator_index=parquet_locator_index)

    def binary_single_feature_contiguous():
        """Benchmark one-feature contiguous-sample lookup on binary shards."""
        binary_reader.load_feature_samples(args.feature_id, contiguous_rows)

    def parquet_single_feature_spread():
        """Benchmark one-feature spread-sample lookup on parquet shards."""
        parquet_reader.load_feature_samples(args.feature_id, spread_rows, locator_index=parquet_locator_index)

    def binary_single_feature_spread():
        """Benchmark one-feature spread-sample lookup on binary shards."""
        binary_reader.load_feature_samples(args.feature_id, spread_rows)

    parquet_bytes = _artifact_bytes_for_parquet(parquet_manifest)
    binary_bytes = _artifact_bytes_for_binary(binary_manifest_path, binary_manifest)

    result = {
        "parquet_manifest": args.parquet_manifest,
        "binary_manifest": binary_manifest_path,
        "build_binary_s": binary_build_s,
        "n_samples": parquet_manifest.n_samples,
        "samples_per_block": parquet_manifest.samples_per_block,
        "binary_codec": binary_manifest.default_codec,
        "disk": {
            "parquet_bytes": int(parquet_bytes),
            "binary_bytes": int(binary_bytes),
            "binary_vs_parquet_ratio": round(float(binary_bytes) / float(parquet_bytes), 3) if parquet_bytes else None,
        },
        "lookup_ms": {
            "many_features_one_sample": {
                "parquet": _median_ms(parquet_many_features_one_sample, args.repeats, args.warmup),
                "binary": _median_ms(binary_many_features_one_sample, args.repeats, args.warmup),
            },
            "single_feature_contiguous": {
                "parquet": _median_ms(parquet_single_feature_contiguous, args.repeats, args.warmup),
                "binary": _median_ms(binary_single_feature_contiguous, args.repeats, args.warmup),
            },
            "single_feature_spread": {
                "parquet": _median_ms(parquet_single_feature_spread, args.repeats, args.warmup),
                "binary": _median_ms(binary_single_feature_spread, args.repeats, args.warmup),
            },
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
