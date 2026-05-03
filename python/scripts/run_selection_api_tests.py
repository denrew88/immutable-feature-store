import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from fs.feature_selection.candidates import build_candidates_from_stats
from fs.config import SelectionConfig
from fs.scalar.parquet_storage import (
    ParquetShardReader,
    build_shards_from_sample_major,
    load_manifest,
    resolve_selection_stats_path,
)
from fs.feature_selection.incremental import select_features_incremental
from fs.scalar.synthetic import SyntheticConfig, generate_synthetic, write_sample_major
from scripts.serve_array_api import app


def main():
    """Run integration tests for the selection API endpoint."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_selection_api_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_dir = root / "samples"
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    shard_dir = root / "shards"

    data = generate_synthetic(
        SyntheticConfig(
            n_samples=64,
            n_features=128,
            y_cols=("y", "y_alt"),
            seed=13,
        )
    )
    assert data["Y"].shape == (64, 2)
    assert tuple(data["y_cols"]) == ("y", "y_alt")
    write_sample_major(data, str(sample_dir), str(sample_meta_path), str(feature_meta_path))
    manifest_path = build_shards_from_sample_major(
        str(sample_meta_path),
        str(shard_dir),
        feature_meta_path=str(feature_meta_path),
        n_shards=4,
        stats_y_cols=["y", "y_alt"],
    )

    client = TestClient(app)
    health = client.get("/healthz")
    assert health.status_code == 200

    request = {
        "manifest_path": manifest_path,
        "y_col": "y_alt",
        "min_non_null_y": 20,
        "min_non_null_pair": 20,
        "top_m": 12,
        "initial_cap": 32,
        "max_step": 64,
        "batch_size": 32,
        "max_gap": 16,
        "mask_fastpath_min_group": 8,
        "mask_fastpath_min_pairs": 64,
    }
    resp = client.post("/run-selection", json=request)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    manifest = load_manifest(manifest_path)
    stats_path = resolve_selection_stats_path(manifest, "y_alt")
    assert stats_path is not None
    reader = ParquetShardReader(manifest, max_gap=request["max_gap"])
    candidates = build_candidates_from_stats(
        stats_path,
        min_non_null_y=request["min_non_null_y"],
        y_r2_threshold=0.01,
        max_candidates=0,
    )
    selected = select_features_incremental(
        candidates,
        reader,
        SelectionConfig(
            y_r2_threshold=0.01,
            min_non_null_y=request["min_non_null_y"],
            ff_r2_threshold=0.9,
            min_non_null_pair=request["min_non_null_pair"],
            top_m=request["top_m"],
            initial_cap=request["initial_cap"],
            max_step=request["max_step"],
            batch_size=request["batch_size"],
            max_gap=request["max_gap"],
            max_candidates=0,
            mask_fastpath_min_group=request["mask_fastpath_min_group"],
            mask_fastpath_min_pairs=request["mask_fastpath_min_pairs"],
        ),
    )
    expected_ids = [int(c.feature_id) for c in selected]
    expected_keys = [f"feature_{feature_id:06d}" for feature_id in expected_ids]

    assert payload["top_m"] == 12
    assert payload["candidate_count"] == len(candidates)
    assert payload["selected_count"] == len(expected_ids)
    assert payload["selected_feature_ids"] == expected_ids
    assert payload["selected_feature_keys"] == expected_keys
    assert payload["used_locator_stats"] is True
    assert payload["candidate_build_ms"] >= 0
    assert payload["selection_ms"] >= 0
    assert payload["elapsed_ms"] >= payload["selection_ms"]

    print("python selection api tests passed")


if __name__ == "__main__":
    main()
