import argparse
import shutil
import time
from pathlib import Path

from line_profiler import LineProfiler

from fs.array import binary_storage as binary_storage_mod
from fs.array.synthetic import generate_array_synthetic
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig


def main():
    """Profile the append-only tmp-spill binary build path on a moderate dataset."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-shard-mb", type=int, default=32)
    ap.add_argument("--samples-per-block", type=int, default=16)
    ap.add_argument("--bundle-max-rows", type=int, default=10000)
    ap.add_argument(
        "--out-root",
        default=str(
            Path(__file__).resolve().parents[2] / "data" / "tmp_profile_array_binary_spill_build"
        ),
    )
    args = ap.parse_args()

    root = Path(args.out_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    bundle_out_dir = root / "bundles"
    sample_meta_path = root / "sample_meta.parquet"
    binary_out_dir = root / "binary_spill"

    synth_cfg = ArraySyntheticConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
    )
    bundle_cfg = ArrayBundleConfig(max_bundle_rows=args.bundle_max_rows)
    shard_cfg = ArrayShardConfig(
        samples_per_block=args.samples_per_block,
        target_shard_bytes=int(args.target_shard_mb) * 1024 * 1024,
        use_tmp_spill=True,
    )

    synth = generate_array_synthetic(
        bundle_out_dir=str(bundle_out_dir),
        sample_meta_path=str(sample_meta_path),
        config=synth_cfg,
        bundle_config=bundle_cfg,
    )
    bundle_manifest_path = synth["bundle_manifest_path"]

    profiler = LineProfiler(
        binary_storage_mod.build_array_binary_shards_from_bundles,
        binary_storage_mod._build_array_binary_shards_with_tmp_spill,
        binary_storage_mod._dense_feature_estimates_from_bundles,
        binary_storage_mod._write_dense_binary_shard,
    )
    wrapped_build = profiler(binary_storage_mod.build_array_binary_shards_from_bundles)

    started = time.perf_counter()
    manifest_path, stats = wrapped_build(
        bundle_manifest_path,
        str(binary_out_dir),
        config=shard_cfg,
        return_stats=True,
    )
    elapsed = time.perf_counter() - started

    print("profile target: append-only tmp-spill binary build from bundles")
    print(
        {
            "bundle_manifest_path": bundle_manifest_path,
            "manifest_path": manifest_path,
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "target_shard_mb": args.target_shard_mb,
            "samples_per_block": args.samples_per_block,
            "bundle_max_rows": args.bundle_max_rows,
            "elapsed_s": round(elapsed, 3),
            "stats": stats,
        }
    )
    profiler.print_stats()


if __name__ == "__main__":
    main()
