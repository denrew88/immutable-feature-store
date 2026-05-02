import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from fs.scalar.parquet_storage import build_shards_from_sample_major
from fs.scalar.synthetic import SyntheticConfig, generate_synthetic, write_sample_major
from scripts.serve_array_api import app


def main():
    """Run integration tests for the scalar-serving API endpoint."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_scalar_api_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_dir = root / "samples"
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    shard_dir = root / "shards"

    data = generate_synthetic(
        SyntheticConfig(
            n_samples=16,
            n_features=20,
            seed=13,
        )
    )
    write_sample_major(data, str(sample_dir), str(sample_meta_path), str(feature_meta_path))
    manifest_path = build_shards_from_sample_major(
        str(sample_meta_path),
        str(shard_dir),
        feature_meta_path=str(feature_meta_path),
        n_shards=4,
    )
    manifest_json = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest_json["sample_meta_path"] == "sample_meta.parquet"
    assert manifest_json["feature_meta_path"] == "feature_meta.parquet"
    assert manifest_json["shard_path"] == "feature_shards"
    assert manifest_json["feature_locator_path"] == "feature_locator.parquet"
    assert (shard_dir / "sample_meta.parquet").exists()
    assert (shard_dir / "feature_meta.parquet").exists()
    assert (shard_dir / "feature_locator.parquet").exists()

    client = TestClient(app)
    health = client.get("/healthz")
    assert health.status_code == 200

    resp = client.post(
        "/scalar-feature",
        json={
            "manifest_path": manifest_path,
            "feature_key": "feature_000000",
            "sample_keys": ["sample_000000", "sample_000003", "sample_000007", "missing_sample_key"],
            "sanitize_nonfinite": True,
        },
    )
    assert resp.status_code == 404, resp.text
    assert "sample key not found" in resp.text

    resp = client.post(
        "/scalar-feature",
        json={
            "manifest_path": manifest_path,
            "feature_key": "feature_000000",
            "sample_keys": ["sample_000000", "sample_000003", "sample_000007"],
            "sanitize_nonfinite": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["feature_id"] == 0
    assert data["feature_key"] == "feature_000000"
    assert data["sample_count"] == 3
    assert [item["sample_id"] for item in data["values"]] == [0, 3, 7]
    assert [item["sample_key"] for item in data["values"]] == [
        "sample_000000",
        "sample_000003",
        "sample_000007",
    ]
    for item in data["values"]:
        assert "sample_row" not in item
        assert "present" in item
        if item["present"]:
            assert isinstance(item["value"], float) or item["value"] is None
        else:
            assert item["value"] is None

    mixed_resp = client.post(
        "/scalar-feature",
        json={
            "manifest_path": manifest_path,
            "feature_id": 0,
            "sample_keys": ["sample_000000", "sample_000003", "sample_000007"],
            "sanitize_nonfinite": True,
        },
    )
    assert mixed_resp.status_code == 200, mixed_resp.text
    mixed_data = mixed_resp.json()
    assert mixed_data["feature_key"] == "feature_000000"
    assert [item["sample_key"] for item in mixed_data["values"]] == [
        "sample_000000",
        "sample_000003",
        "sample_000007",
    ]

    print("python scalar api tests passed")


if __name__ == "__main__":
    main()
