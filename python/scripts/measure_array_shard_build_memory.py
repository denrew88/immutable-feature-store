import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fs.array.binary_storage import build_array_binary_shards_from_bundles
from fs.array.storage import list_bundle_paths, load_array_bundle_manifest
from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig


def _mb(value: int) -> float:
    return round(float(value) / (1024 * 1024), 2)


def _measure_process_stats_start():
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "started_at": time.perf_counter(),
        "working_set_start_mb": _mb(getattr(mem, "wset", mem.rss)),
    }


def _measure_process_stats_finish(start_stats: dict):
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "elapsed_s": round(time.perf_counter() - start_stats["started_at"], 3),
        "working_set_start_mb": start_stats["working_set_start_mb"],
        "working_set_end_mb": _mb(getattr(mem, "wset", mem.rss)),
        "peak_working_set_mb": _mb(getattr(mem, "peak_wset", getattr(mem, "wset", mem.rss))),
    }


def _clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sum_existing_file_sizes(paths) -> int:
    total = 0
    for path in paths:
        path_obj = Path(path)
        if path_obj.exists():
            total += path_obj.stat().st_size
    return total


def _run_build_child(bundle_manifest_path: str, out_dir: str, target_shard_mb: int, samples_per_block: int, use_tmp_spill: bool):
    _clean_dir(Path(out_dir))
    start = _measure_process_stats_start()
    manifest_path, build_stats = build_array_binary_shards_from_bundles(
        bundle_manifest_path=bundle_manifest_path,
        out_dir=out_dir,
        config=ArrayShardConfig(
            samples_per_block=samples_per_block,
            target_shard_bytes=target_shard_mb * 1024 * 1024,
            use_tmp_spill=bool(use_tmp_spill),
            spill_bucket_target_bytes=max(1, target_shard_mb // 2) * 1024 * 1024,
        ),
        return_stats=True,
    )
    stats = _measure_process_stats_finish(start)
    stats.update(
        {
            "mode": "spill" if use_tmp_spill else "direct",
            "target_shard_mb": int(target_shard_mb),
            "manifest_path": manifest_path,
            **build_stats,
        }
    )
    print(json.dumps(stats))


def _run_child_command(args):
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"child produced no output: {completed.stderr}")
    return json.loads(lines[-1])


def _generate_dataset(root: Path, n_samples: int, n_features: int, seed: int):
    bundle_out_dir = root / "bundles"
    sample_meta_path = root / "sample_meta.parquet"
    result = generate_array_synthetic(
        bundle_out_dir=str(bundle_out_dir),
        sample_meta_path=str(sample_meta_path),
        config=ArraySyntheticConfig(
            n_samples=int(n_samples),
            n_features=int(n_features),
            seed=int(seed),
        ),
        bundle_config=ArrayBundleConfig(max_bundle_rows=4096, max_bundle_bytes=256 * 1024 * 1024),
    )
    bundle_manifest = load_array_bundle_manifest(result["bundle_manifest_path"])
    return {
        "bundle_manifest_path": result["bundle_manifest_path"],
        "bundle_bytes_mb": _mb(_sum_existing_file_sizes(list_bundle_paths(bundle_manifest))),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure current array binary shard build memory usage.")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--target-shard-mb", type=int, default=32)
    parser.add_argument("--samples-per-block", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "tmp_array_build_mem"))
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--bundle-manifest-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--spill", action="store_true")
    args = parser.parse_args()

    if args.child:
        _run_build_child(
            bundle_manifest_path=args.bundle_manifest_path,
            out_dir=args.out_dir,
            target_shard_mb=args.target_shard_mb,
            samples_per_block=args.samples_per_block,
            use_tmp_spill=bool(args.spill),
        )
        return

    root = Path(args.root).resolve()
    _clean_dir(root)
    dataset = _generate_dataset(root, args.n_samples, args.n_features, args.seed)

    direct = _run_child_command(
        [
            sys.executable,
            __file__,
            "--child",
            "--bundle-manifest-path",
            dataset["bundle_manifest_path"],
            "--out-dir",
            str(root / "direct"),
            "--target-shard-mb",
            str(args.target_shard_mb),
            "--samples-per-block",
            str(args.samples_per_block),
        ]
    )
    spill = _run_child_command(
        [
            sys.executable,
            __file__,
            "--child",
            "--bundle-manifest-path",
            dataset["bundle_manifest_path"],
            "--out-dir",
            str(root / "spill"),
            "--target-shard-mb",
            str(args.target_shard_mb),
            "--samples-per-block",
            str(args.samples_per_block),
            "--spill",
        ]
    )

    report = {
        "dataset": {
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "seed": int(args.seed),
            "bundle_bytes_mb": dataset["bundle_bytes_mb"],
            "bundle_manifest_path": dataset["bundle_manifest_path"],
        },
        "direct": direct,
        "spill": spill,
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
