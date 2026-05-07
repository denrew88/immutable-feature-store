import json
import shutil
from pathlib import Path

from fs.array import ArrayDatasetBuilder, LogicalType, PointColumnSpec, StorageType
from fs.array import write_feature_meta as write_array_feature_meta
from fs.array import write_sample_meta as write_array_sample_meta
from fs.config import ArrayBinaryBuildOptions, ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder
from fs.scalar import write_feature_meta as write_scalar_feature_meta
from fs.scalar import write_sample_meta as write_scalar_sample_meta


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _run_scalar_session_test(root: Path):
    sample_meta_path = Path(
        write_scalar_sample_meta(
            [
                {"sample_key": "sample_000000", "y": 0.0},
                {"sample_key": "sample_000001", "y": 1.0},
                {"sample_key": "sample_000002", "y": 0.0},
            ],
            root / "sample_meta.parquet",
        )
    )
    feature_meta_path = Path(
        write_scalar_feature_meta(
            [
                {"feature_key": "feature_a"},
                {"feature_key": "feature_b"},
            ],
            root / "feature_meta.parquet",
        )
    )
    out_dir = root / "scalar_shards"
    stage_dir = out_dir / "sample_major_stage"
    state_path = stage_dir / "state.json"
    log_path = stage_dir / "bundles.jsonl"

    with ScalarDatasetBuilder.open_session(
        out_dir=str(out_dir),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=ScalarShardBuildOptions(target_shard_mb=1),
    ) as builder:
        status0 = builder.status()
        assert status0.last_committed_sample_id is None
        assert status0.next_expected_sample_id == 0
        builder.write_sample(0, {"feature_a": 1.0})
        builder.write_sample(1, {"feature_b": 2.0})
        status1 = builder.status()
        assert status1.last_committed_sample_id is None
        assert status1.buffered_through_sample_id == 1
        assert status1.next_expected_sample_id == 0

    assert state_path.exists()
    assert log_path.exists()
    state1 = _load_json(state_path)
    log1 = _load_jsonl(log_path)
    assert state1["last_committed_sample_id"] == 1
    assert state1["next_expected_sample_id"] == 2
    assert len(log1) == 1
    assert log1[0]["first_sample_id"] == 0
    assert log1[0]["last_sample_id"] == 1

    with ScalarDatasetBuilder.open_session(
        out_dir=str(out_dir),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=ScalarShardBuildOptions(target_shard_mb=1),
    ) as builder:
        resumed = builder.status()
        assert resumed.last_committed_sample_id == 1
        assert resumed.next_expected_sample_id == 2
        builder.write_sample(2, {"feature_a": 3.0, "feature_b": 4.0})
        stage_manifest_path = Path(builder.finish_stage())
        assert stage_manifest_path.exists()
        finished = builder.status()
        assert finished.finished_stage
        assert finished.bundle_manifest_path == str(stage_manifest_path)
        manifest_path = Path(builder.build_shards(keep_sample_major=True))
        assert manifest_path.exists()


def _run_array_session_test(root: Path):
    sample_meta_path = Path(
        write_array_sample_meta(
            [
                {"sample_key": "sample_000000"},
                {"sample_key": "sample_000001"},
                {"sample_key": "sample_000002"},
            ],
            root / "sample_meta.parquet",
        )
    )
    feature_meta_path = Path(
        write_array_feature_meta(
            [
                {"feature_key": "feature_a"},
                {"feature_key": "feature_b"},
            ],
            root / "feature_meta.parquet",
        )
    )
    out_dir = root / "array_shards"
    stage_dir = out_dir / "bundle_stage"
    state_path = stage_dir / "state.json"
    log_path = stage_dir / "bundles.jsonl"
    point_schema = [
        PointColumnSpec(name="time", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
        PointColumnSpec(name="value", storage_type=StorageType.FLOAT64, logical_type=LogicalType.CONTINUOUS),
    ]
    build_options = ArrayBinaryBuildOptions(samples_per_block=2, target_shard_mb=1, codec="none")

    with ArrayDatasetBuilder.open_session(
        out_dir=str(out_dir),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        point_schema=point_schema,
        build_options=build_options,
    ) as builder:
        status0 = builder.status()
        assert status0.last_committed_sample_id is None
        assert status0.next_expected_sample_id == 0
        with builder.sample(sample_id=0) as sample:
            sample.add_trace(feature_key="feature_a", columns={"time": [0.0, 1.0], "value": [10.0, 11.0]})
        with builder.sample(sample_id=1) as sample:
            sample.add_trace(feature_key="feature_b", columns={"time": [0.5], "value": [20.0]})
        status1 = builder.status()
        assert status1.last_committed_sample_id is None
        assert status1.buffered_through_sample_id == 1
        assert status1.next_expected_sample_id == 0

    assert state_path.exists()
    assert log_path.exists()
    state1 = _load_json(state_path)
    log1 = _load_jsonl(log_path)
    assert state1["last_committed_sample_id"] == 1
    assert state1["next_expected_sample_id"] == 2
    assert len(log1) == 1
    assert log1[0]["first_sample_id"] == 0
    assert log1[0]["last_sample_id"] == 1

    with ArrayDatasetBuilder.open_session(
        out_dir=str(out_dir),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        point_schema=point_schema,
        build_options=build_options,
    ) as builder:
        resumed = builder.status()
        assert resumed.last_committed_sample_id == 1
        assert resumed.next_expected_sample_id == 2
        with builder.sample(sample_id=2) as sample:
            sample.add_trace(feature_key="feature_a", columns={"time": [2.0], "value": [30.0]})
        stage_manifest_path = Path(builder.finish_stage())
        assert stage_manifest_path.exists()
        finished = builder.status()
        assert finished.finished_stage
        assert finished.bundle_manifest_path == str(stage_manifest_path)
        manifest_path = Path(builder.build_shards())
        assert manifest_path.exists()


def main():
    """Run dedicated resumable session tests for array and scalar builders."""

    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_builder_session_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    _run_scalar_session_test(root / "scalar")
    _run_array_session_test(root / "array")
    print("python builder session tests passed")


if __name__ == "__main__":
    main()
