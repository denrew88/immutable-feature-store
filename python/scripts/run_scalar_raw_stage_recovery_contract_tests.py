from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "packages" / "scalar_feature_shard" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder, open_dense_long_shard, write_feature_meta, write_sample_meta
from validate_scalar_dense_long_exhaustive import validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_SAMPLE_PADDING = 12


def _sample_file(stage_dir: Path, sample_id: int) -> Path:
    return stage_dir / "raw_samples" / f"sample_{sample_id:0{RAW_SAMPLE_PADDING}d}.parquet"


def _sample_lock(stage_dir: Path, sample_id: int) -> Path:
    return Path(str(_sample_file(stage_dir, sample_id)) + ".lock")


def _raw_stage_lock(stage_dir: Path) -> Path:
    return stage_dir / "raw_stage.lock"


def _write_fresh_lock(path: Path, *, kind: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "pid": os.getpid(),
                "host": os.environ.get("COMPUTERNAME", "local"),
                "created_at_ms": int(time.time() * 1000),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _make_metadata(root: Path) -> tuple[Path, Path]:
    sample_meta_path = Path(
        write_sample_meta(
            [
                {"sample_key": "sample_000000", "y": 1.0},
                {"sample_key": "sample_000001", "y": 2.0},
                {"sample_key": "sample_000002", "y": 3.0},
                {"sample_key": "sample_000003", "y": 4.0},
            ],
            root / "sample_meta.parquet",
        )
    )
    feature_meta_path = Path(
        write_feature_meta(
            [
                {"feature_key": "feature_a"},
                {"feature_key": "feature_b"},
                {"feature_key": "feature_c"},
            ],
            root / "feature_meta.parquet",
        )
    )
    return sample_meta_path, feature_meta_path


def _open_builder(stage_dir: Path, sample_meta_path: Path, feature_meta_path: Path) -> ScalarDatasetBuilder:
    return ScalarDatasetBuilder(
        out_dir=str(stage_dir),
        sample_meta_path=str(sample_meta_path),
        feature_meta_path=str(feature_meta_path),
        build_options=BuildOptions(target_shard_mb=1, stats_y_cols=("y",)),
    )


def _drop_commit_records(raw_log_path: Path, sample_ids: set[int]):
    rows = []
    for line in raw_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["sample_id"]) not in sample_ids:
            rows.append(record)
    raw_log_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rows),
        encoding="utf-8",
    )


def _read_commit_ids(raw_log_path: Path) -> list[int]:
    if not raw_log_path.exists():
        return []
    out = []
    for line in raw_log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(int(json.loads(line)["sample_id"]))
    return sorted(out)


def _assert_direct_values(manifest_path: Path):
    with open_dense_long_shard(str(manifest_path)) as ds:
        values_a, valid_a = ds.load_feature_by_id(0)
        assert bool(valid_a[0]) and np.isclose(values_a[0], 10.0)
        assert not bool(valid_a[1]) and np.isnan(values_a[1])
        assert not bool(valid_a[2]) and np.isnan(values_a[2])
        values_b, valid_b = ds.load_feature_by_id(1)
        assert not bool(valid_b[0]) and np.isnan(values_b[0])
        assert bool(valid_b[1]) and np.isclose(values_b[1], 21.0)
        assert not bool(valid_b[2]) and np.isnan(values_b[2])
        values_c, valid_c = ds.load_feature_by_id(2)
        assert bool(valid_c[3]) and np.isclose(values_c[3], 33.0)


def test_reconciles_final_parquets_missing_from_commit_log(root: Path):
    """final parquet는 있는데 commit log append만 실패한 stage를 복구해야 합니다."""

    sample_meta_path, feature_meta_path = _make_metadata(root)
    stage_dir = root / "stage_reconcile"
    builder = _open_builder(stage_dir, sample_meta_path, feature_meta_path)
    builder.write_sample(0, {"feature_a": 10.0})
    builder.write_sample(1, {"feature_b": 21.0})
    builder.write_sample(2, {})  # empty sample도 파일명만으로 완료 상태를 복구해야 한다.
    builder.write_sample(3, {"feature_c": 33.0})
    builder.close()

    _drop_commit_records(stage_dir / "raw_samples.jsonl", {1, 2})
    assert _sample_file(stage_dir, 1).exists()
    assert _sample_file(stage_dir, 2).exists()
    assert _read_commit_ids(stage_dir / "raw_samples.jsonl") == [0, 3]

    recovered = _open_builder(stage_dir, sample_meta_path, feature_meta_path)
    assert recovered.completed_sample_ids() == [0, 1, 2, 3]
    assert recovered.pending_sample_ids() == []
    assert _read_commit_ids(stage_dir / "raw_samples.jsonl") == [0, 1, 2, 3]

    stage_manifest_path = Path(recovered.finish_stage())
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    assert stage_manifest["sample_ids"] == [0, 1, 2, 3]
    assert stage_manifest["completed_sample_count"] == 4

    final_manifest_path = Path(
        recovered.build_dense_long_shards(
            require_all=True,
            out_dir=str(root / "reconciled_scalar_shard"),
        )
    )
    validate_manifest(final_manifest_path, sample_major_manifest_path=stage_manifest_path, progress_every=0)
    _assert_direct_values(final_manifest_path)


def test_finish_stage_refuses_active_sample_lock(root: Path):
    """sample write가 진행 중이면 partial manifest를 확정하면 안 됩니다."""

    sample_meta_path, feature_meta_path = _make_metadata(root)
    stage_dir = root / "stage_active_sample_lock"
    builder = _open_builder(stage_dir, sample_meta_path, feature_meta_path)
    builder.write_sample(0, {"feature_a": 10.0})
    _write_fresh_lock(_sample_lock(stage_dir, 1), kind="sample")

    try:
        builder.finish_stage()
    except Exception:
        pass
    else:
        raise AssertionError("finish_stage must not finalize while an active sample lock exists")

    assert not (stage_dir / "sample_major_manifest.json").exists()


def test_stage_lock_blocks_new_write(root: Path):
    """finish/build가 stage lock을 잡고 있으면 새 write_sample 진입이 막혀야 합니다."""

    sample_meta_path, feature_meta_path = _make_metadata(root)
    stage_dir = root / "stage_finalize_lock"
    builder = _open_builder(stage_dir, sample_meta_path, feature_meta_path)
    _write_fresh_lock(_raw_stage_lock(stage_dir), kind="stage")

    try:
        builder.write_sample(1, {"feature_b": 21.0})
    except Exception:
        pass
    else:
        raise AssertionError("write_sample must not commit while the raw stage finalize lock is active")

    assert not _sample_file(stage_dir, 1).exists()
    assert _read_commit_ids(stage_dir / "raw_samples.jsonl") == []


def main():
    # 실제 구현에서는 테스트 시간을 줄이기 위해 이 env를 lock timeout으로 존중해야 합니다.
    os.environ.setdefault("SCALAR_RAW_STAGE_LOCK_TIMEOUT_SECONDS", "0.2")
    root = REPO_ROOT / "data" / "tmp_scalar_raw_stage_recovery_contract_tests"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    tests = (
        test_reconciles_final_parquets_missing_from_commit_log,
        test_finish_stage_refuses_active_sample_lock,
        test_stage_lock_blocks_new_write,
    )
    failures: list[str] = []
    for test in tests:
        case_root = root / test.__name__
        case_root.mkdir(parents=True, exist_ok=True)
        try:
            test(case_root)
        except Exception:
            failures.append(f"{test.__name__}\n{traceback.format_exc()}")
    if failures:
        raise AssertionError(
            "scalar raw stage recovery contract tests failed:\n\n" + "\n\n".join(failures)
        )
    print("scalar raw stage recovery contract tests passed")


if __name__ == "__main__":
    main()
