from __future__ import annotations

import shutil
import sys
import time
from argparse import ArgumentParser
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

from fs.array import open_shard  # noqa: E402
from fs.array.metadata import write_feature_meta, write_sample_meta  # noqa: E402
from fs.array.storage import ArraySampleBundleWriter  # noqa: E402
from fs.array.binary_storage import build_array_binary_shards_from_bundles  # noqa: E402
from fs.config import ArrayBundleConfig, ArrayShardConfig  # noqa: E402
from fs.types import LogicalType, PointColumnSpec, StorageType  # noqa: E402
from array_sample_parquet import (  # noqa: E402
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    open_array_sample_parquet,
)
from validate_array_binary_shard_exhaustive import validate_manifest as validate_binary_manifest  # noqa: E402
from validate_array_sample_parquet_exhaustive import validate_manifest as validate_parquet_manifest  # noqa: E402


def _schema():
    return [
        PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
        PointColumnSpec("ch_step", StorageType.UINT32, LogicalType.INTEGER),
    ]


def _columns(rng: np.random.Generator, sample_id: int, feature_id: int, trace_len: int):
    trace_len = int(trace_len)
    time_values = np.arange(trace_len, dtype=np.float64) + sample_id * 0.01 + feature_id * 0.001
    values = rng.normal(loc=float(feature_id), scale=2.0, size=trace_len).astype(np.float64)
    if trace_len > 0 and rng.random() < 0.25:
        values[int(rng.integers(0, trace_len))] = np.nan
    return {
        "time": time_values,
        "value": values,
        "ch_step": rng.integers(1, 5, size=trace_len, dtype=np.uint32),
    }


def _assert_array_equal(label: str, actual, expected):
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    if actual_arr.dtype.kind == "f" or expected_arr.dtype.kind == "f":
        actual_f = actual_arr.astype(np.float64, copy=False)
        expected_f = expected_arr.astype(np.float64, copy=False)
        equal = (actual_f == expected_f) | (np.isnan(actual_f) & np.isnan(expected_f))
        if bool(np.all(equal)):
            return
        idx = int(np.flatnonzero(~equal)[0])
        raise AssertionError(f"{label} mismatch at index={idx}: actual={actual_f[idx]!r} expected={expected_f[idx]!r}")
    if not np.array_equal(actual_arr, expected_arr):
        actual_flat = actual_arr.reshape(-1)
        expected_flat = expected_arr.reshape(-1)
        idx = int(np.flatnonzero(actual_flat != expected_flat)[0])
        raise AssertionError(f"{label} mismatch at index={idx}: actual={actual_flat[idx]!r} expected={expected_flat[idx]!r}")


def _compare_parquet_and_binary(parquet_manifest_path: str, binary_manifest_path: str, *, n_samples: int, n_features: int):
    parquet_reader = open_array_sample_parquet(parquet_manifest_path)
    binary_reader = open_shard(binary_manifest_path)
    parquet_traces = parquet_reader.get_traces(
        sample_ids=list(range(n_samples)),
        feature_ids=list(range(n_features)),
        include_missing=True,
        decode_categorical=False,
    )
    parquet_by_pair = {(int(trace.sample_id), int(trace.feature_id)): trace for trace in parquet_traces}
    for feature_id in range(n_features):
        binary_batch = binary_reader.get_traces(feature_id=feature_id, sample_ids=list(range(n_samples)))
        for sample_id, binary_trace in zip(range(n_samples), binary_batch.traces):
            parquet_trace = parquet_by_pair[(sample_id, feature_id)]
            binary_present = bool(int(binary_trace.flags) & 0x01)
            binary_empty = bool(int(binary_trace.flags) & 0x02)
            if bool(parquet_trace.present) != binary_present:
                raise AssertionError(f"present mismatch sample={sample_id} feature={feature_id}")
            if not parquet_trace.present:
                if int(binary_trace.flags) != 0:
                    raise AssertionError(f"missing flags mismatch sample={sample_id} feature={feature_id}")
                continue
            if int(parquet_trace.trace_len) != int(binary_trace.columns["time"].shape[0]):
                raise AssertionError(f"trace_len mismatch sample={sample_id} feature={feature_id}")
            if int(parquet_trace.trace_len) == 0 and not binary_empty:
                raise AssertionError(f"empty trace flag mismatch sample={sample_id} feature={feature_id}")
            for name in ["time", "value", "ch_step"]:
                _assert_array_equal(
                    f"sample={sample_id} feature={feature_id} column={name}",
                    binary_trace.columns[name],
                    parquet_trace.columns[name],
                )
    binary_reader.close()


def main(argv=None):
    ap = ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--n-features", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args(argv)

    root = Path(args.out_dir).resolve() if args.out_dir else REPO_ROOT / "data" / "tmp_array_parquet_binary_parity_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    n_samples = int(args.n_samples)
    n_features = int(args.n_features)
    started = time.perf_counter()

    sample_meta_path = write_sample_meta(
        [{"sample_key": f"sample_{sample_id:06d}"} for sample_id in range(n_samples)],
        root / "sample_meta.parquet",
    )
    feature_meta_path = write_feature_meta(
        [{"feature_key": f"feature_{feature_id:06d}"} for feature_id in range(n_features)],
        root / "feature_meta.parquet",
    )

    schema = _schema()
    parquet_out = root / "array_sample_parquet"
    bundle_out = root / "bundles"
    rng = np.random.default_rng(20260605)
    with ArraySampleParquetDatasetBuilder.open_session(
        parquet_out,
        sample_meta_path,
        schema,
        feature_meta_path=feature_meta_path,
        options=ArraySampleParquetBuildOptions(target_part_bytes=1024 * 1024, max_part_samples=4, compression="none"),
    ) as parquet_builder, ArraySampleBundleWriter(
        str(bundle_out),
        str(sample_meta_path),
        n_samples=n_samples,
        feature_meta_path=str(feature_meta_path),
        config=ArrayBundleConfig(max_bundle_rows=16, max_bundle_bytes=1 << 20),
        point_schema=schema,
    ) as bundle_writer:
        for sample_id in range(n_samples):
            with parquet_builder.sample(sample_id=sample_id) as sample:
                for feature_id in range(n_features):
                    if rng.random() < 0.25:
                        continue
                    trace_len = int(rng.integers(0, 9))
                    cols = _columns(rng, sample_id, feature_id, trace_len)
                    sample.add_trace(feature_id=feature_id, columns=cols)
                    bundle_writer.append_trace(sample_id, feature_id, columns=cols)
        parquet_manifest_path = parquet_builder.finish()
        bundle_manifest_path = bundle_writer.finish()

    binary_manifest_path = build_array_binary_shards_from_bundles(
        bundle_manifest_path,
        str(root / "array_binary"),
        config=ArrayShardConfig(samples_per_block=4, target_shard_bytes=4096),
    )

    validate_parquet_manifest(parquet_manifest_path, stage_dir=parquet_out, label="array-parquet-binary-parity-parquet")
    validate_binary_manifest(binary_manifest_path, bundle_manifest_path)
    _compare_parquet_and_binary(
        parquet_manifest_path,
        binary_manifest_path,
        n_samples=n_samples,
        n_features=n_features,
    )

    total_sec = time.perf_counter() - started
    print(
        "array sample parquet/custom binary parity tests passed "
        f"n_samples={n_samples} n_features={n_features} total_sec={total_sec:.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
