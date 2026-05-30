"""Build/query benchmark for array binary shard vs array_sample_parquet.

The default size matches the recent large comparison:

    1024 samples x 768 features x trace_len 512

The script intentionally generates trace rows sample-by-sample and does not
pre-build a full dataset in memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from fs.array import (
    ArrayDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
    write_feature_meta,
    write_sample_meta,
)
from fs.array.binary_storage import ArrayBinaryShardReader, load_array_binary_shard_manifest
from fs.array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    open_array_sample_parquet,
)
from fs.config import ArrayBinaryBuildOptions, ArrayBundleConfig


FORMAT_BINARY = "binary"
FORMAT_SAMPLE_PARQUET = "sample-parquet"


@dataclass
class BuildResult:
    manifest_path: str
    seconds: float | None
    artifact_mb: float


class SyntheticTraceFactory:
    def __init__(
        self,
        *,
        n_samples: int,
        n_features: int,
        trace_len: int,
        include_categorical: bool,
        seed: int,
        max_precompute_mb: int,
    ):
        self.n_samples = int(n_samples)
        self.n_features = int(n_features)
        self.trace_len = int(trace_len)
        self.include_categorical = bool(include_categorical)
        self.rng = np.random.default_rng(int(seed))
        self.time = np.linspace(0.0, 1.0, self.trace_len, dtype=np.float64)
        self.sample_offsets = ((np.arange(self.n_samples, dtype=np.float64) % 97.0) - 48.0) * 0.001
        self.feature_bias = ((np.arange(self.n_features, dtype=np.float64) % 131.0) - 65.0) * 0.0005
        self.feature_freq = 1.0 + (np.arange(self.n_features, dtype=np.float64) % 37.0) * 0.031
        self.feature_phase = (np.arange(self.n_features, dtype=np.float64) % 19.0) * 0.17
        self.feature_waves = None
        estimated_wave_mb = self.n_features * self.trace_len * np.dtype(np.float64).itemsize / (1024 * 1024)
        if estimated_wave_mb <= float(max_precompute_mb):
            self.feature_waves = np.sin(
                self.feature_freq[:, None] * self.time[None, :] + self.feature_phase[:, None]
            ).astype(np.float64, copy=False)
        self.ch_step = None
        if self.include_categorical:
            labels = np.array(["idle", "ramp", "hold", "cool"], dtype=object)
            self.ch_step = labels[np.arange(self.trace_len) % labels.size]

    def point_schema(self) -> list[PointColumnSpec]:
        schema = [
            PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
            PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        ]
        if self.include_categorical:
            schema.append(PointColumnSpec("ch_step", StorageType.UINT32, LogicalType.CATEGORICAL))
        return schema

    def columns(self, sample_id: int, feature_id: int) -> dict[str, np.ndarray]:
        feature_id = int(feature_id)
        sample_id = int(sample_id)
        if self.feature_waves is None:
            value = np.sin(self.feature_freq[feature_id] * self.time + self.feature_phase[feature_id])
        else:
            value = self.feature_waves[feature_id]
        value = (value + self.sample_offsets[sample_id] + self.feature_bias[feature_id]).astype(np.float64, copy=False)
        columns = {
            "time": self.time,
            "value": value,
        }
        if self.include_categorical:
            columns["ch_step"] = self.ch_step
        return columns


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_out_dir(n_samples: int, n_features: int, trace_len: int) -> Path:
    return repo_root() / "data" / f"tmp_array_query_bench_{n_samples}x{n_features}x{trace_len}"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 2)


def median_ms(fn, *, repeats: int, warmup: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples = []
    for _ in range(int(repeats)):
        gc.collect()
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return round(float(np.median(np.asarray(samples, dtype=np.float64))), 1)


def sample_rows(n_samples: int) -> list[dict[str, object]]:
    return [
        {
            "sample_key": f"sample_{idx:06d}",
            "split": "train" if idx % 5 else "valid",
            "lot": f"lot_{idx // 25:04d}",
        }
        for idx in range(int(n_samples))
    ]


def feature_rows(n_features: int) -> list[dict[str, object]]:
    return [
        {
            "feature_key": f"feature_{idx:06d}",
            "group": f"group_{idx % 32:02d}",
        }
        for idx in range(int(n_features))
    ]


def ensure_metadata(root: Path, *, n_samples: int, n_features: int) -> tuple[str, str]:
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    if not sample_meta_path.exists():
        write_sample_meta(sample_rows(n_samples), sample_meta_path)
    if not feature_meta_path.exists():
        write_feature_meta(feature_rows(n_features), feature_meta_path)
    return str(sample_meta_path), str(feature_meta_path)


def progress(prefix: str, sample_id: int, total: int, started: float, every: int):
    if int(every) <= 0:
        return
    if sample_id == 0 or sample_id + 1 == total or (sample_id + 1) % int(every) == 0:
        elapsed = time.perf_counter() - started
        rate = (sample_id + 1) / elapsed if elapsed > 0 else 0.0
        print(f"{prefix}: sample {sample_id + 1}/{total} ({rate:.2f} samples/sec)", flush=True)


def write_all_traces(builder, factory: SyntheticTraceFactory, *, progress_prefix: str, progress_every: int):
    started = time.perf_counter()
    status = builder.status()
    start_sample_id = int(status.next_expected_sample_id)
    for sample_id in range(start_sample_id, factory.n_samples):
        with builder.sample(sample_id=sample_id) as sample:
            for feature_id in range(factory.n_features):
                sample.add_trace(feature_id=feature_id, columns=factory.columns(sample_id, feature_id))
        progress(progress_prefix, sample_id, factory.n_samples, started, progress_every)


def build_binary(
    root: Path,
    *,
    sample_meta_path: str,
    feature_meta_path: str,
    factory: SyntheticTraceFactory,
    args,
) -> BuildResult:
    out_dir = root / "binary_shards"
    manifest_path = out_dir / "array_binary_shard_manifest.json"
    if args.reuse_existing and manifest_path.exists():
        return BuildResult(str(manifest_path), None, mb(dir_size_bytes(out_dir)))

    build_options = ArrayBinaryBuildOptions(
        target_shard_mb=int(args.binary_target_shard_mb),
        samples_per_block=int(args.samples_per_block),
        n_shards=None if int(args.binary_n_shards) <= 0 else int(args.binary_n_shards),
        codec=str(args.binary_codec),
    )
    bundle_config = ArrayBundleConfig(
        max_bundle_rows=int(args.bundle_max_rows),
        max_bundle_bytes=int(args.bundle_max_mb) * 1024 * 1024,
    )
    started = time.perf_counter()
    with ArrayDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        factory.point_schema(),
        feature_meta_path=feature_meta_path,
        build_options=build_options,
        bundle_config=bundle_config,
    ) as builder:
        write_all_traces(builder, factory, progress_prefix="binary build", progress_every=int(args.progress_every))
        manifest = builder.build_shards(cleanup_bundles=bool(args.cleanup_bundles))
    seconds = round(time.perf_counter() - started, 3)
    return BuildResult(str(manifest), seconds, mb(dir_size_bytes(out_dir)))


def build_sample_parquet(
    root: Path,
    *,
    sample_meta_path: str,
    feature_meta_path: str,
    factory: SyntheticTraceFactory,
    args,
) -> BuildResult:
    out_dir = root / "sample_parquet"
    manifest_path = out_dir / "array_sample_parquet_manifest.json"
    if args.reuse_existing and manifest_path.exists():
        return BuildResult(str(manifest_path), None, mb(dir_size_bytes(out_dir)))

    options = ArraySampleParquetBuildOptions(
        target_part_bytes=int(args.target_part_mb) * 1024 * 1024,
        max_part_rows=int(args.sample_parquet_max_part_rows),
        max_part_samples=int(args.sample_parquet_max_part_samples),
        compression=str(args.sample_parquet_compression),
    )
    started = time.perf_counter()
    with ArraySampleParquetDatasetBuilder.open_session(
        out_dir,
        sample_meta_path,
        factory.point_schema(),
        feature_meta_path=feature_meta_path,
        options=options,
    ) as builder:
        write_all_traces(builder, factory, progress_prefix="sample parquet build", progress_every=int(args.progress_every))
        manifest = builder.finish()
    seconds = round(time.perf_counter() - started, 3)
    return BuildResult(str(manifest), seconds, mb(dir_size_bytes(out_dir)))


def contiguous_ids(count: int, limit: int) -> list[int]:
    return list(range(min(int(count), int(limit))))


def binary_query_bench(manifest_path: str, args) -> dict[str, float]:
    manifest = load_array_binary_shard_manifest(manifest_path)
    reader = ArrayBinaryShardReader(manifest)
    all_samples = list(range(int(manifest.n_samples)))
    all_features = list(range(int(manifest.n_features)))
    sample_id = min(max(int(args.query_sample_id), 0), max(int(manifest.n_samples) - 1, 0))
    feature_id = min(max(int(args.query_feature_id), 0), max(int(manifest.n_features) - 1, 0))
    selected_samples = contiguous_ids(int(args.query_sample_count), int(manifest.n_samples))
    selected_features = contiguous_ids(int(args.query_feature_count), int(manifest.n_features))

    def one_feature_all_samples():
        reader.load_feature_samples(feature_id, all_samples)

    def one_sample_all_features():
        for fid in all_features:
            reader.load_feature_samples(fid, [sample_id])

    def multi_sample_multi_feature():
        for fid in selected_features:
            reader.load_feature_samples(fid, selected_samples)

    try:
        return {
            "one_feature_all_samples": median_ms(one_feature_all_samples, repeats=args.query_repeats, warmup=args.query_warmup),
            "one_sample_all_features": median_ms(one_sample_all_features, repeats=args.query_repeats, warmup=args.query_warmup),
            f"{len(selected_samples)}_samples_x_{len(selected_features)}_features": median_ms(
                multi_sample_multi_feature,
                repeats=args.query_repeats,
                warmup=args.query_warmup,
            ),
        }
    finally:
        reader.close()


def sample_parquet_query_bench(manifest_path: str, args) -> dict[str, float]:
    reader = open_array_sample_parquet(manifest_path)
    n_samples = int(reader.manifest.n_samples)
    n_features = int(reader.manifest.n_features)
    all_samples = list(range(n_samples))
    all_features = list(range(n_features))
    sample_id = min(max(int(args.query_sample_id), 0), max(n_samples - 1, 0))
    feature_id = min(max(int(args.query_feature_id), 0), max(n_features - 1, 0))
    selected_samples = contiguous_ids(int(args.query_sample_count), n_samples)
    selected_features = contiguous_ids(int(args.query_feature_count), n_features)

    def one_feature_all_samples():
        reader.get_traces(sample_ids=all_samples, feature_ids=[feature_id])

    def one_sample_all_features():
        reader.get_traces(sample_ids=[sample_id], feature_ids=all_features)

    def multi_sample_multi_feature():
        reader.get_traces(sample_ids=selected_samples, feature_ids=selected_features)

    return {
        "one_feature_all_samples": median_ms(one_feature_all_samples, repeats=args.query_repeats, warmup=args.query_warmup),
        "one_sample_all_features": median_ms(one_sample_all_features, repeats=args.query_repeats, warmup=args.query_warmup),
        f"{len(selected_samples)}_samples_x_{len(selected_features)}_features": median_ms(
            multi_sample_multi_feature,
            repeats=args.query_repeats,
            warmup=args.query_warmup,
        ),
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--n-features", type=int, default=768)
    ap.add_argument("--trace-len", type=int, default=512)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--formats", nargs="+", choices=[FORMAT_BINARY, FORMAT_SAMPLE_PARQUET], default=[FORMAT_BINARY, FORMAT_SAMPLE_PARQUET])
    ap.add_argument("--clean", action="store_true", help="delete out-dir before building")
    ap.add_argument("--reuse-existing", action="store_true", help="reuse existing manifests when present")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-query", action="store_true")
    ap.add_argument("--cleanup-bundles", action="store_true")
    ap.add_argument("--include-categorical", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-precompute-mb", type=int, default=512)
    ap.add_argument("--progress-every", type=int, default=16)
    ap.add_argument("--binary-target-shard-mb", type=int, default=32)
    ap.add_argument("--binary-n-shards", type=int, default=0)
    ap.add_argument("--binary-codec", default="none")
    ap.add_argument("--samples-per-block", type=int, default=16)
    ap.add_argument("--bundle-max-rows", type=int, default=10000)
    ap.add_argument("--bundle-max-mb", type=int, default=256)
    ap.add_argument("--target-part-mb", type=int, default=128)
    ap.add_argument("--sample-parquet-max-part-rows", type=int, default=10_000_000)
    ap.add_argument("--sample-parquet-max-part-samples", type=int, default=0)
    ap.add_argument("--sample-parquet-compression", default="zstd")
    ap.add_argument("--query-warmup", type=int, default=1)
    ap.add_argument("--query-repeats", type=int, default=5)
    ap.add_argument("--query-sample-id", type=int, default=0)
    ap.add_argument("--query-feature-id", type=int, default=0)
    ap.add_argument("--query-sample-count", type=int, default=16)
    ap.add_argument("--query-feature-count", type=int, default=64)
    ap.add_argument("--result-json", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir(args.n_samples, args.n_features, args.trace_len)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_build and any(root.iterdir()) and not args.reuse_existing:
        expected = {"sample_meta.parquet", "feature_meta.parquet"}
        present = {child.name for child in root.iterdir()}
        if not present.issubset(expected):
            raise ValueError(f"out-dir is not empty; pass --clean or --reuse-existing: {root}")

    sample_meta_path, feature_meta_path = ensure_metadata(root, n_samples=args.n_samples, n_features=args.n_features)
    factory = SyntheticTraceFactory(
        n_samples=args.n_samples,
        n_features=args.n_features,
        trace_len=args.trace_len,
        include_categorical=args.include_categorical,
        seed=args.seed,
        max_precompute_mb=args.max_precompute_mb,
    )

    result = {
        "out_dir": str(root),
        "config": {
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "trace_len": int(args.trace_len),
            "include_categorical": bool(args.include_categorical),
            "formats": list(args.formats),
        },
        "build": {},
        "query_ms": {},
    }

    binary_manifest = str(root / "binary_shards" / "array_binary_shard_manifest.json")
    sample_parquet_manifest = str(root / "sample_parquet" / "array_sample_parquet_manifest.json")

    if not args.skip_build:
        if FORMAT_BINARY in args.formats:
            binary = build_binary(
                root,
                sample_meta_path=sample_meta_path,
                feature_meta_path=feature_meta_path,
                factory=factory,
                args=args,
            )
            binary_manifest = binary.manifest_path
            result["build"][FORMAT_BINARY] = {
                "manifest_path": binary.manifest_path,
                "seconds": binary.seconds,
                "artifact_mb": binary.artifact_mb,
            }
        if FORMAT_SAMPLE_PARQUET in args.formats:
            sample_parquet = build_sample_parquet(
                root,
                sample_meta_path=sample_meta_path,
                feature_meta_path=feature_meta_path,
                factory=factory,
                args=args,
            )
            sample_parquet_manifest = sample_parquet.manifest_path
            result["build"][FORMAT_SAMPLE_PARQUET] = {
                "manifest_path": sample_parquet.manifest_path,
                "seconds": sample_parquet.seconds,
                "artifact_mb": sample_parquet.artifact_mb,
            }

    if not args.skip_query:
        if FORMAT_BINARY in args.formats:
            result["query_ms"][FORMAT_BINARY] = binary_query_bench(binary_manifest, args)
        if FORMAT_SAMPLE_PARQUET in args.formats:
            result["query_ms"][FORMAT_SAMPLE_PARQUET] = sample_parquet_query_bench(sample_parquet_manifest, args)

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    result_path = Path(args.result_json).expanduser().resolve() if args.result_json else root / "benchmark_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(output + "\n", encoding="utf-8")
    print(f"wrote result: {result_path}")


if __name__ == "__main__":
    main()
