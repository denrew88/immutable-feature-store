import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fs.array.storage import (
    _collect_array_feature_stats,
    _partition_feature_ids_by_count,
    _partition_feature_ids_by_target_bytes,
    _process_sorted_rows,
    _write_array_locator,
    _write_array_shard_file,
    build_array_shards_from_bundles,
    list_bundle_paths,
    load_array_bundle_manifest,
    shard_file_path,
)
from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig
from fs.types import ArrayShardManifest


def _mb(value: int) -> float:
    """Convert bytes to rounded mebibytes."""
    return round(float(value) / (1024 * 1024), 2)


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _data_root(name: str) -> Path:
    """Return a data subdirectory path under the repository root."""
    return _repo_root() / "data" / name


def _clean_dir(path: Path):
    """Remove and recreate a directory used by a benchmark run."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _measure_process_stats_start():
    """Capture initial process timing and working-set information."""
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "started_at": time.perf_counter(),
        "working_set_start_mb": _mb(getattr(mem, "wset", mem.rss)),
    }


def _measure_process_stats_finish(start_stats: dict):
    """Capture final process timing and working-set information."""
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "elapsed_s": round(time.perf_counter() - start_stats["started_at"], 3),
        "working_set_start_mb": start_stats["working_set_start_mb"],
        "working_set_end_mb": _mb(getattr(mem, "wset", mem.rss)),
        "peak_working_set_mb": _mb(getattr(mem, "peak_wset", getattr(mem, "wset", mem.rss))),
    }


def _sum_file_sizes(paths) -> int:
    """Return the total size of all existing files in an iterable."""
    total = 0
    for path in paths:
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total


def _build_bucket_partitions(shard_partitions, feature_ids, estimated_feature_bytes, spill_bucket_target_bytes: int):
    """Build shard/bucket partitions for the spill prototype benchmark."""
    est_lookup = {
        int(feature_id): int(est_bytes)
        for feature_id, est_bytes in zip(feature_ids.tolist(), estimated_feature_bytes.tolist())
    }
    feature_map_rows = []
    shard_bucket_partitions = []
    for shard_id, shard_feature_ids in enumerate(shard_partitions):
        shard_estimates = np.asarray(
            [est_lookup[int(feature_id)] for feature_id in shard_feature_ids.tolist()],
            dtype=np.int64,
        )
        bucket_partitions = _partition_feature_ids_by_target_bytes(
            shard_feature_ids,
            shard_estimates,
            spill_bucket_target_bytes,
        )
        shard_bucket_partitions.append(bucket_partitions)
        for bucket_id, bucket_feature_ids in enumerate(bucket_partitions):
            for feature_id in bucket_feature_ids.tolist():
                feature_map_rows.append(
                    {
                        "feature_id": int(feature_id),
                        "shard_id": int(shard_id),
                        "bucket_id": int(bucket_id),
                    }
                )
    feature_map_df = pl.DataFrame(
        {
            "feature_id": pl.Series(
                "feature_id",
                [row["feature_id"] for row in feature_map_rows],
                dtype=pl.Int32,
            ),
            "shard_id": pl.Series(
                "shard_id",
                [row["shard_id"] for row in feature_map_rows],
                dtype=pl.Int32,
            ),
            "bucket_id": pl.Series(
                "bucket_id",
                [row["bucket_id"] for row in feature_map_rows],
                dtype=pl.Int32,
            ),
        }
    )
    return feature_map_df, shard_bucket_partitions


def build_array_shards_from_bundles_spill_prototype(
    bundle_manifest_path: str,
    out_dir: str,
    config: ArrayShardConfig = None,
    spill_bucket_target_bytes: int = 8 * 1024 * 1024,
):
    """Build array shards with the older part-file spill prototype.

    This benchmark helper exists only to compare memory and temp-file behavior
    against the current production spill implementation.
    """
    config = config or ArrayShardConfig()
    if config.samples_per_block <= 0:
        raise ValueError("samples_per_block must be > 0")
    if config.target_shard_bytes <= 0 and config.n_shards <= 0:
        raise ValueError("either target_shard_bytes or n_shards must be > 0")
    if spill_bucket_target_bytes <= 0:
        raise ValueError("spill_bucket_target_bytes must be > 0")

    bundle_manifest = load_array_bundle_manifest(bundle_manifest_path)
    bundle_paths = list_bundle_paths(bundle_manifest)
    feature_ids, estimated_feature_bytes = _collect_array_feature_stats(bundle_paths, config.samples_per_block)
    if config.n_shards > 0:
        shard_partitions = _partition_feature_ids_by_count(feature_ids, config.n_shards)
    else:
        shard_partitions = _partition_feature_ids_by_target_bytes(
            feature_ids,
            estimated_feature_bytes,
            config.target_shard_bytes,
        )
    if not shard_partitions:
        shard_partitions = [np.empty(0, dtype=np.int32)]

    os.makedirs(out_dir, exist_ok=True)
    shard_path = os.path.join(out_dir, "array_feature_shards")
    os.makedirs(shard_path, exist_ok=True)
    locator_path = os.path.join(out_dir, "array_feature_locator.parquet")
    manifest_path = os.path.join(out_dir, "array_shard_manifest.json")
    tmp_root = os.path.join(out_dir, "_tmp")
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    feature_map_df, shard_bucket_partitions = _build_bucket_partitions(
        shard_partitions,
        feature_ids,
        estimated_feature_bytes,
        spill_bucket_target_bytes,
    )

    bucket_part_paths = {}
    temp_files_created = 0
    live_temp_files = 0
    peak_live_temp_files = 0
    live_temp_bytes = 0
    peak_live_temp_bytes = 0

    for bundle_id, bundle_path in enumerate(bundle_paths):
        bundle_df = pl.read_parquet(
            bundle_path,
            columns=[
                "feature_id",
                "sample_row",
                "flags",
                "trace_len",
                "time_blob",
                "value_blob",
            ],
        )
        if bundle_df.height == 0:
            continue
        joined = bundle_df.join(feature_map_df, on="feature_id", how="inner")
        if joined.height == 0:
            continue
        parts = joined.partition_by(["shard_id", "bucket_id"], as_dict=True)
        for key, part_df in parts.items():
            shard_id = int(key[0])
            bucket_id = int(key[1])
            bucket_dir = os.path.join(tmp_root, f"shard_{shard_id:04d}", f"bucket_{bucket_id:04d}")
            os.makedirs(bucket_dir, exist_ok=True)
            part_path = os.path.join(bucket_dir, f"part_{bundle_id:06d}.parquet")
            (
                part_df.select(
                    [
                        "feature_id",
                        "sample_row",
                        "flags",
                        "trace_len",
                        "time_blob",
                        "value_blob",
                    ]
                )
                .write_parquet(part_path)
            )
            part_size = os.path.getsize(part_path)
            bucket_part_paths.setdefault((shard_id, bucket_id), []).append((part_path, part_size))
            temp_files_created += 1
            live_temp_files += 1
            live_temp_bytes += part_size
            peak_live_temp_files = max(peak_live_temp_files, live_temp_files)
            peak_live_temp_bytes = max(peak_live_temp_bytes, live_temp_bytes)

    locator_rows = []
    for shard_id, bucket_partitions in enumerate(shard_bucket_partitions):
        shard_rows = []
        row_in_shard_offset = 0
        for bucket_id, _bucket_feature_ids in enumerate(bucket_partitions):
            part_entries = bucket_part_paths.get((shard_id, bucket_id), [])
            if not part_entries:
                continue
            part_paths = [part_path for part_path, _ in part_entries]
            df = (
                pl.scan_parquet(part_paths)
                .select(
                    [
                        "feature_id",
                        "sample_row",
                        "flags",
                        "trace_len",
                        "time_blob",
                        "value_blob",
                    ]
                )
                .sort(["feature_id", "sample_row"])
                .collect()
            )
            bucket_rows, bucket_locator_rows = _process_sorted_rows(
                df,
                bundle_manifest.n_samples,
                config.samples_per_block,
                shard_id,
            )
            for row in bucket_locator_rows:
                row["row_in_shard"] += row_in_shard_offset
            row_in_shard_offset += len(bucket_rows)
            shard_rows.extend(bucket_rows)
            locator_rows.extend(bucket_locator_rows)

            for part_path, part_size in part_entries:
                if os.path.exists(part_path):
                    os.remove(part_path)
                live_temp_files -= 1
                live_temp_bytes -= part_size
            bucket_dir = os.path.dirname(part_entries[0][0])
            if os.path.isdir(bucket_dir):
                os.rmdir(bucket_dir)

        _write_array_shard_file(shard_rows, shard_file_path(shard_path, shard_id))
        shard_tmp_dir = os.path.join(tmp_root, f"shard_{shard_id:04d}")
        if os.path.isdir(shard_tmp_dir):
            os.rmdir(shard_tmp_dir)

    _write_array_locator(locator_rows, locator_path)
    manifest = ArrayShardManifest(
        sample_meta_path=bundle_manifest.sample_meta_path,
        n_samples=bundle_manifest.n_samples,
        shard_path=shard_path,
        n_shards=len(shard_partitions),
        locator_path=locator_path,
        samples_per_block=config.samples_per_block,
        feature_id_dtype="INT32",
        flags_dtype="UINT8",
        offset_dtype="INT64",
        time_dtype="FLOAT64_LE_BLOB",
        value_dtype="FLOAT64_LE_BLOB",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2)

    if os.path.isdir(tmp_root):
        shutil.rmtree(tmp_root)

    return {
        "manifest_path": manifest_path,
        "n_shards": len(shard_partitions),
        "n_buckets": int(sum(len(bucket_partitions) for bucket_partitions in shard_bucket_partitions)),
        "spill_bucket_target_mb": round(spill_bucket_target_bytes / (1024 * 1024), 2),
        "temp_files_created": int(temp_files_created),
        "peak_live_temp_files": int(peak_live_temp_files),
        "peak_live_temp_bytes_mb": _mb(peak_live_temp_bytes),
        "final_live_temp_files": int(live_temp_files),
        "final_live_temp_bytes_mb": _mb(live_temp_bytes),
    }


def _bundle_bytes(bundle_manifest_path: str) -> float:
    """Return total bundle artifact size in mebibytes."""
    manifest = load_array_bundle_manifest(bundle_manifest_path)
    return _mb(_sum_file_sizes(list_bundle_paths(manifest)))


def _run_baseline_child(bundle_manifest_path: str, out_dir: str, target_shard_mb: int, samples_per_block: int):
    """Run the direct parquet shard build in a child process and print JSON stats."""
    _clean_dir(Path(out_dir))
    start = _measure_process_stats_start()
    manifest_path, build_stats = build_array_shards_from_bundles(
        bundle_manifest_path,
        out_dir,
        config=ArrayShardConfig(
            samples_per_block=samples_per_block,
            target_shard_bytes=target_shard_mb * 1024 * 1024,
        ),
        return_stats=True,
    )
    stats = _measure_process_stats_finish(start)
    stats.update(
        {
            "mode": "baseline",
            "target_shard_mb": target_shard_mb,
            "manifest_path": manifest_path,
            **build_stats,
        }
    )
    if "peak_live_temp_bytes" in stats:
        stats["peak_live_temp_bytes_mb"] = _mb(stats.pop("peak_live_temp_bytes"))
    stats["shard_bytes_mb"] = _mb(
        _sum_file_sizes(
            Path(manifest_path).with_name("array_feature_shards").glob("*.parquet")
        )
    )
    print(json.dumps(stats))


def _run_spill_child(
    bundle_manifest_path: str,
    out_dir: str,
    target_shard_mb: int,
    samples_per_block: int,
    spill_bucket_mb: int,
):
    """Run the spill parquet shard build in a child process and print JSON stats."""
    _clean_dir(Path(out_dir))
    start = _measure_process_stats_start()
    manifest_path, build_stats = build_array_shards_from_bundles(
        bundle_manifest_path,
        out_dir,
        config=ArrayShardConfig(
            samples_per_block=samples_per_block,
            target_shard_bytes=target_shard_mb * 1024 * 1024,
            use_tmp_spill=True,
            spill_bucket_target_bytes=spill_bucket_mb * 1024 * 1024,
        ),
        return_stats=True,
    )
    stats = _measure_process_stats_finish(start)
    stats.update(
        {
            "mode": "spill",
            "target_shard_mb": target_shard_mb,
            "manifest_path": manifest_path,
            "spill_bucket_target_mb": float(spill_bucket_mb),
            **build_stats,
        }
    )
    if "peak_live_temp_bytes" in stats:
        stats["peak_live_temp_bytes_mb"] = _mb(stats.pop("peak_live_temp_bytes"))
    stats["shard_bytes_mb"] = _mb(
        _sum_file_sizes(
            Path(manifest_path).with_name("array_feature_shards").glob("*.parquet")
        )
    )
    print(json.dumps(stats))


def _run_child_command(args):
    """Execute a child benchmark command and parse its final JSON line."""
    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"child produced no output: {completed.stderr}")
    return json.loads(lines[-1])


def _generate_dataset(root: Path, n_samples: int, n_features: int, seed: int):
    """Generate the synthetic array dataset used by the benchmark harness."""
    bundle_out_dir = root / "bundles"
    sample_meta_path = root / "sample_meta.parquet"
    if bundle_out_dir.exists():
        shutil.rmtree(bundle_out_dir)
    if sample_meta_path.exists():
        sample_meta_path.unlink()
    result = generate_array_synthetic(
        bundle_out_dir=str(bundle_out_dir),
        sample_meta_path=str(sample_meta_path),
        config=ArraySyntheticConfig(
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
        ),
        bundle_config=ArrayBundleConfig(),
        shard_out_dir=None,
    )
    return result["bundle_manifest_path"]


def run_bench(n_samples: int, n_features: int, target_shard_mb: int, samples_per_block: int, spill_bucket_mb: int, seed: int):
    """Run the end-to-end baseline versus spill memory benchmark."""
    root = _data_root(f"tmp_array_shard_bench_{n_samples}x{n_features}")
    root.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path = _generate_dataset(root, n_samples, n_features, seed)

    baseline_out_dir = root / "build_baseline"
    spill_out_dir = root / "build_spill"
    baseline_stats = _run_child_command(
        [
            sys.executable,
            __file__,
            "--mode",
            "build-baseline",
            "--bundle-manifest",
            str(bundle_manifest_path),
            "--out-dir",
            str(baseline_out_dir),
            "--target-shard-mb",
            str(target_shard_mb),
            "--samples-per-block",
            str(samples_per_block),
        ]
    )
    spill_stats = _run_child_command(
        [
            sys.executable,
            __file__,
            "--mode",
            "build-spill",
            "--bundle-manifest",
            str(bundle_manifest_path),
            "--out-dir",
            str(spill_out_dir),
            "--target-shard-mb",
            str(target_shard_mb),
            "--samples-per-block",
            str(samples_per_block),
            "--spill-bucket-mb",
            str(spill_bucket_mb),
        ]
    )

    report = {
        "dataset": {
            "root": str(root),
            "bundle_manifest_path": str(bundle_manifest_path),
            "bundle_bytes_mb": _bundle_bytes(bundle_manifest_path),
            "n_samples": n_samples,
            "n_features": n_features,
            "target_shard_mb": target_shard_mb,
            "samples_per_block": samples_per_block,
            "spill_bucket_mb": spill_bucket_mb,
            "seed": seed,
        },
        "baseline": baseline_stats,
        "spill": spill_stats,
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    """CLI entry point for array shard build memory benchmarks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "build-baseline", "build-spill"], default="bench")
    ap.add_argument("--bundle-manifest")
    ap.add_argument("--out-dir")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--n-features", type=int, default=1024)
    ap.add_argument("--target-shard-mb", type=int, default=32)
    ap.add_argument("--samples-per-block", type=int, default=8)
    ap.add_argument("--spill-bucket-mb", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.mode == "bench":
        run_bench(
            n_samples=args.n_samples,
            n_features=args.n_features,
            target_shard_mb=args.target_shard_mb,
            samples_per_block=args.samples_per_block,
            spill_bucket_mb=args.spill_bucket_mb,
            seed=args.seed,
        )
        return

    if not args.bundle_manifest or not args.out_dir:
        raise ValueError("bundle-manifest and out-dir are required for build modes")

    if args.mode == "build-baseline":
        _run_baseline_child(
            args.bundle_manifest,
            args.out_dir,
            args.target_shard_mb,
            args.samples_per_block,
        )
        return

    _run_spill_child(
        args.bundle_manifest,
        args.out_dir,
        args.target_shard_mb,
        args.samples_per_block,
        args.spill_bucket_mb,
    )


if __name__ == "__main__":
    main()
