import argparse
import json
import time
from pathlib import Path

import numpy as np

from fs.array.binary_storage import ArrayBinaryShardReader, load_array_binary_shard_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_manifest() -> str:
    return str(
        _repo_root()
        / "data"
        / "tmp_py_array_api_test"
        / "binary_v3_shards"
        / "array_binary_shard_manifest.json"
    )


def _median_ms(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return round(float(np.median(np.asarray(samples, dtype=np.float64))), 1)


def main():
    """binary array shard lookup benchmark."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=_default_manifest())
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--n-features", type=int, default=20)
    ap.add_argument("--feature-id", type=int, default=0)
    ap.add_argument("--contiguous-count", type=int, default=256)
    ap.add_argument("--spread-count", type=int, default=256)
    args = ap.parse_args()

    manifest = load_array_binary_shard_manifest(args.manifest)
    reader = ArrayBinaryShardReader(manifest)

    contiguous_sample_ids = list(range(min(args.contiguous_count, manifest.n_samples)))
    spread_sample_ids = np.linspace(
        0,
        max(manifest.n_samples - 1, 0),
        num=min(args.spread_count, manifest.n_samples),
        dtype=np.int64,
    ).tolist()
    many_feature_ids = list(range(min(args.n_features, manifest.n_features)))

    def many_features_one_sample():
        for feature_id in many_feature_ids:
            reader.load_feature_samples(feature_id, [0])

    def single_feature_contiguous():
        reader.load_feature_samples(args.feature_id, contiguous_sample_ids)

    def single_feature_spread():
        reader.load_feature_samples(args.feature_id, spread_sample_ids)

    result = {
        "manifest": args.manifest,
        "n_samples": manifest.n_samples,
        "n_features": manifest.n_features,
        "samples_per_block": manifest.samples_per_block,
        "codec": manifest.default_codec,
        "lookup_ms": {
            "many_features_one_sample": _median_ms(many_features_one_sample, args.repeats, args.warmup),
            "single_feature_contiguous": _median_ms(single_feature_contiguous, args.repeats, args.warmup),
            "single_feature_spread": _median_ms(single_feature_spread, args.repeats, args.warmup),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
