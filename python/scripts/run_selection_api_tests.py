import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
PACKAGE_SRC = REPO_ROOT / "packages" / "scalar_feature_shard" / "src"
for path in (PYTHON_ROOT, PACKAGE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fs.scalar.synthetic import SyntheticConfig, generate_synthetic, write_sample_major
from scalar_feature_shard import build_shard, open_dense_long_shard
from scripts.serve_feature_query_api import app


def main():
    """Run integration tests for dense-long scalar top-feature API endpoint."""

    root = REPO_ROOT / "data" / "tmp_py_selection_api_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_dir = root / "samples"
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    shard_dir = root / "dense_long"

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
    manifest_path = build_shard(
        str(sample_meta_path),
        str(shard_dir),
        feature_meta_path=str(feature_meta_path),
        target_shard_mb=1,
        stats_y_cols=("y", "y_alt"),
    )

    client = TestClient(app)
    assert client.get("/healthz").status_code == 200

    request = {
        "manifest_path": manifest_path,
        "y_col": "y_alt",
        "top_k": 12,
    }
    resp = client.post("/scalar/top-features", json=request)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    with open_dense_long_shard(manifest_path) as ds:
        expected = ds.top_features_from_stats("y_alt", top_k=12)
    expected_ids = [int(value) for value in expected["feature_id"].to_list()]
    expected_keys = [f"feature_{feature_id:06d}" for feature_id in expected_ids]

    assert payload["format"] == "scalar-dense-long"
    assert payload["y_col"] == "y_alt"
    assert payload["top_k"] == 12
    assert [row["feature_id"] for row in payload["features"]] == expected_ids
    assert [row["feature_key"] for row in payload["features"]] == expected_keys
    assert all(row["part_id"] is not None for row in payload["features"])
    assert all(row["offset_in_part"] is not None for row in payload["features"])

    print("python selection api tests passed")


if __name__ == "__main__":
    main()
