import json
import shutil
import time
from pathlib import Path

import numpy as np

from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder, write_feature_meta, write_sample_meta


def _format_seconds(value: float) -> str:
    """Format one elapsed time with three fractional digits."""

    return f"{value:.3f}s"


def main():
    """Benchmark scalar direct-ingestion builder on a large dense workload."""

    n_samples = 5000
    n_features = 20000
    target_shard_mb = 32

    root = Path(__file__).resolve().parents[2] / "data" / "tmp_scalar_builder_bench_5000x20000"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    shard_out_dir = root / "scalar_shards"

    timings = {}

    t0 = time.perf_counter()
    sample_records = [
        {"sample_key": f"sample_{sample_id:06d}", "y": float(sample_id)}
        for sample_id in range(n_samples)
    ]
    timings["build_sample_meta_records_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_sample_meta(sample_records, sample_meta_path)
    timings["write_sample_meta_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feature_records = [
        {"feature_key": f"feature_{feature_id:06d}"}
        for feature_id in range(n_features)
    ]
    timings["build_feature_meta_records_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_feature_meta(feature_records, feature_meta_path)
    timings["write_feature_meta_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder = ScalarDatasetBuilder(
        out_dir=str(shard_out_dir),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=ScalarShardBuildOptions(
            target_shard_mb=target_shard_mb,
            stats_y_cols=("y",),
        ),
    )
    timings["builder_init_s"] = time.perf_counter() - t0

    feature_ids = list(range(n_features))
    feature_values_base = np.arange(n_features, dtype=np.float64)

    mapping_build_s = 0.0
    writer_write_sample_s = 0.0
    ingest_total_t0 = time.perf_counter()
    for sample_id in range(n_samples):
        value_t0 = time.perf_counter()
        values = dict(zip(feature_ids, feature_values_base + float(sample_id)))
        mapping_build_s += time.perf_counter() - value_t0

        write_t0 = time.perf_counter()
        builder.write_sample(sample_id, values)
        writer_write_sample_s += time.perf_counter() - write_t0

        if (sample_id + 1) % 500 == 0:
            print(f"[progress] wrote {sample_id + 1}/{n_samples} samples")

    timings["build_sample_value_mappings_s"] = mapping_build_s
    timings["writer_write_sample_calls_s"] = writer_write_sample_s
    timings["writer_ingest_total_s"] = time.perf_counter() - ingest_total_t0

    t0 = time.perf_counter()
    builder.finish_sample_major()
    timings["finish_sample_major_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    manifest_path, build_stats = builder.build_shards(return_stats=True)
    timings["build_shards_wall_s"] = time.perf_counter() - t0

    manifest_path = str(manifest_path)
    shard_bytes = sum(path.stat().st_size for path in shard_out_dir.rglob("*") if path.is_file())

    report = {
        "shape": {
            "n_samples": n_samples,
            "n_features": n_features,
            "target_shard_mb": target_shard_mb,
        },
        "artifact": {
            "root": str(root),
            "manifest_path": manifest_path,
            "total_bytes": int(shard_bytes),
        },
        "timings": timings,
        "build_stats": build_stats,
        "throughput": {
            "samples_per_second": (
                float(n_samples) / timings["writer_ingest_total_s"] if timings["writer_ingest_total_s"] > 0 else None
            ),
            "feature_values_per_second": (
                float(n_samples * n_features) / timings["writer_ingest_total_s"]
                if timings["writer_ingest_total_s"] > 0
                else None
            ),
        },
    }

    print()
    print("Scalar builder benchmark")
    print(f"- shape: {n_samples} samples x {n_features} features")
    print(f"- target_shard_mb: {target_shard_mb}")
    print(f"- manifest: {manifest_path}")
    print(f"- artifact bytes: {shard_bytes}")
    print()
    print("Top-level timings")
    for key in (
        "build_sample_meta_records_s",
        "write_sample_meta_s",
        "build_feature_meta_records_s",
        "write_feature_meta_s",
        "builder_init_s",
        "build_sample_value_mappings_s",
        "writer_write_sample_calls_s",
        "writer_ingest_total_s",
        "finish_sample_major_s",
        "build_shards_wall_s",
    ):
        print(f"- {key}: {_format_seconds(float(timings[key]))}")

    print()
    print("build_shards internal stats")
    for key, value in build_stats.items():
        print(f"- {key}: {_format_seconds(float(value))}")

    print()
    print("Throughput")
    print(f"- samples_per_second: {report['throughput']['samples_per_second']:.3f}")
    print(f"- feature_values_per_second: {report['throughput']['feature_values_per_second']:.3f}")

    report_path = root / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
