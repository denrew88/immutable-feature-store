from __future__ import annotations

import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SRC = REPO_ROOT / "packages" / "scalar_feature_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scalar_feature_shard import (  # noqa: E402
    BuildOptions,
    ScalarDatasetBuilder,
    open_dense_long_shard,
    write_feature_meta,
    write_sample_meta,
)


def main() -> None:
    root = REPO_ROOT / "data" / "tmp_scalar_feature_shard_python_example"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    sample_meta_path = write_sample_meta(
        [
            {"sample_key": "sample_000000", "split": "train", "y": 1.0},
            {"sample_key": "sample_000001", "split": "train", "y": 2.0},
            {"sample_key": "sample_000002", "split": "valid", "y": 3.0},
            {"sample_key": "sample_000003", "split": "valid", "y": 4.0},
        ],
        root / "sample_meta.parquet",
    )
    feature_meta_path = write_feature_meta(
        [
            {"feature_key": "feature_a", "group": "alpha"},
            {"feature_key": "feature_b", "group": "beta"},
            {"feature_key": "feature_c", "group": "gamma"},
        ],
        root / "feature_meta.parquet",
    )

    build_options = BuildOptions(target_shard_mb=16, stats_y_cols=("y",))
    stage_dir = root / "scalar_stage"
    dense_dir = root / "scalar_shard"

    # 첫 실행에서는 일부 sample만 완료하고 종료한 상황을 가정합니다.
    with ScalarDatasetBuilder.open_session(
        stage_dir,
        sample_meta_path,
        feature_meta_path=feature_meta_path,
        build_options=build_options,
    ) as builder:
        builder.write_sample(2, {"feature_b": 20.0}, skip_if_completed=True)
        builder.write_sample(0, {"feature_a": 10.0, "feature_c": None}, skip_if_completed=True)
        print("first_pending=", builder.pending_sample_ids())

    # 같은 out_dir로 다시 열면 raw_state.json과 raw_samples.jsonl을 읽고 이어서 진행합니다.
    with ScalarDatasetBuilder.open_session(
        stage_dir,
        sample_meta_path,
        feature_meta_path=feature_meta_path,
        build_options=build_options,
    ) as builder:
        for sample_id in builder.pending_sample_ids():
            values = {"feature_a": 11.0, "feature_b": 21.0} if sample_id == 1 else {"feature_c": 40.0}
            builder.write_sample(sample_id, values, skip_if_completed=True)

        dense_manifest_path = builder.build_dense_long_shards(
            out_dir=dense_dir,
            require_all=True,
            row_group_features=128,
        )

    with open_dense_long_shard(dense_manifest_path) as dataset:
        values, valid = dataset.load_feature_by_id(0)
        top = dataset.top_features_from_stats("y", top_k=2)
        print("feature_a_values=", [None if not valid[idx] else float(values[idx]) for idx in range(len(values))])
        print("top_features=", top.select(["feature_id", "r2y", "n_y_overlap"]).to_dicts())

    print("sample_meta=", sample_meta_path)
    print("feature_meta=", feature_meta_path)
    print("dense_manifest=", dense_manifest_path)


if __name__ == "__main__":
    main()
