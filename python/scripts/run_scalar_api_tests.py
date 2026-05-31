import json
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
from scalar_feature_shard import build_shard
from scripts.serve_feature_query_api import app


def main():
    """Run integration tests for dense-long scalar API endpoints."""

    root = REPO_ROOT / "data" / "tmp_py_scalar_api_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_dir = root / "samples"
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    shard_dir = root / "dense_long"

    data = generate_synthetic(SyntheticConfig(n_samples=16, n_features=20, seed=13))
    write_sample_major(data, str(sample_dir), str(sample_meta_path), str(feature_meta_path))
    manifest_path = build_shard(
        str(sample_meta_path),
        str(shard_dir),
        feature_meta_path=str(feature_meta_path),
        target_shard_mb=1,
        stats_y_cols=("y",),
    )
    manifest_json = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest_json["format"] == "scalar-dense-long-shard-v1"
    assert manifest_json["parts_path"] == "dense_long_parts"
    assert manifest_json["feature_locator_path"] == "feature_locator.parquet"
    assert (shard_dir / "sample_meta.parquet").exists()
    assert (shard_dir / "feature_meta.parquet").exists()
    assert (shard_dir / "feature_locator.parquet").exists()

    client = TestClient(app)
    assert client.get("/healthz").status_code == 200

    missing_resp = client.post(
        "/scalar/features",
        json={
            "manifest_path": manifest_path,
            "feature_keys": ["feature_000000"],
            "sample_keys": ["sample_000000", "missing_sample_key"],
        },
    )
    assert missing_resp.status_code == 404, missing_resp.text
    assert "sample_key not found" in missing_resp.text

    resp = client.post(
        "/scalar/features",
        json={
            "manifest_path": manifest_path,
            "feature_keys": ["feature_000000"],
            "sample_keys": ["sample_000000", "sample_000003", "sample_000007"],
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["format"] == "scalar-dense-long"
    assert payload["layout"] == "features"
    assert payload["feature_count"] == 1
    assert payload["sample_count"] == 3
    feature = payload["features"][0]
    assert feature["feature_id"] == 0
    assert feature["feature_key"] == "feature_000000"
    assert [item["sample_id"] for item in feature["values"]] == [0, 3, 7]
    assert [item["sample_key"] for item in feature["values"]] == [
        "sample_000000",
        "sample_000003",
        "sample_000007",
    ]

    sample_resp = client.post(
        "/scalar/sample",
        json={
            "manifest_path": manifest_path,
            "sample_key": "sample_000000",
            "feature_keys": ["feature_000000", "feature_000001"],
        },
    )
    assert sample_resp.status_code == 200, sample_resp.text
    sample_payload = sample_resp.json()
    assert sample_payload["format"] == "scalar-dense-long"
    assert sample_payload["layout"] == "sample"
    assert sample_payload["sample_id"] == 0
    assert sample_payload["sample_key"] == "sample_000000"
    assert sample_payload["feature_count"] == 2

    print("python scalar api tests passed")


if __name__ == "__main__":
    main()
