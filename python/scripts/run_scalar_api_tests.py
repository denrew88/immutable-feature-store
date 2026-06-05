import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
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


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _load_final_rows(manifest_path: str) -> pl.DataFrame:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    frames = [
        pl.read_parquet(_resolve(manifest_file, str(part["path"])), columns=["feature_id", "sample_id", "mask", "value"])
        for part in manifest["parts"]
    ]
    return pl.concat(frames, how="vertical").sort(["feature_id", "sample_id"])


def _assert_float_same_or_nan(actual: float, expected: float):
    if math.isnan(float(expected)):
        assert math.isnan(float(actual)), (actual, expected)
    else:
        assert float(actual) == float(expected), (actual, expected)


def _expected_cell(final_rows: pl.DataFrame, feature_id: int, sample_id: int) -> dict:
    row = final_rows.filter((pl.col("feature_id") == int(feature_id)) & (pl.col("sample_id") == int(sample_id)))
    assert row.height == 1
    return row.row(0, named=True)


def _assert_api_value_matches_cell(value_payload: dict, cell: dict):
    assert bool(value_payload["present"]) == bool(int(cell["mask"]))
    if bool(value_payload["present"]):
        assert value_payload["value"] is not None
        assert float(value_payload["value"]) == float(cell["value"])
    else:
        assert value_payload["value"] is None
        assert math.isnan(float(cell["value"]))


def main():
    """Run integration tests for dense-long scalar API endpoints."""

    root = REPO_ROOT / "data" / "tmp_py_scalar_api_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    sample_dir = root / "samples"
    sample_meta_path = root / "sample_meta.parquet"
    feature_meta_path = root / "feature_meta.parquet"
    shard_dir = root / "scalar_shard"

    data = generate_synthetic(SyntheticConfig(n_samples=16, n_features=20, y_missing_rate=0.2, seed=13))
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
    assert manifest_json["parts_path"] == "parts"
    assert manifest_json["feature_locator_path"] == "feature_locator.parquet"
    assert (shard_dir / "sample_meta.parquet").exists()
    assert (shard_dir / "feature_meta.parquet").exists()
    assert (shard_dir / "feature_locator.parquet").exists()
    final_rows = _load_final_rows(manifest_path)
    missing_cell = final_rows.filter(pl.col("mask") == 0).row(0, named=True)
    present_cell = final_rows.filter(pl.col("mask") == 1).row(0, named=True)
    assert math.isnan(float(missing_cell["value"]))

    with open_dense_long_shard(manifest_path) as ds:
        feature_values, feature_valid = ds.load_feature_by_id(int(present_cell["feature_id"]))
        feature_key_values, feature_key_valid = ds.load_feature_by_key(f"feature_{int(present_cell['feature_id']):06d}")
        assert np.array_equal(feature_valid, feature_key_valid)
        assert np.allclose(feature_values, feature_key_values, equal_nan=True)
        for sample_id in range(int(manifest_json["n_samples"])):
            cell = _expected_cell(final_rows, int(present_cell["feature_id"]), sample_id)
            assert int(feature_valid[sample_id]) == int(cell["mask"])
            _assert_float_same_or_nan(float(feature_values[sample_id]), float(cell["value"]))

        sample_values, sample_valid = ds.load_sample_by_id(int(present_cell["sample_id"]))
        sample_key_values, sample_key_valid = ds.load_sample_by_key(f"sample_{int(present_cell['sample_id']):06d}")
        assert np.array_equal(sample_valid, sample_key_valid)
        assert np.allclose(sample_values, sample_key_values, equal_nan=True)
        for feature_id in range(int(manifest_json["n_features"])):
            cell = _expected_cell(final_rows, feature_id, int(present_cell["sample_id"]))
            assert int(sample_valid[feature_id]) == int(cell["mask"])
            _assert_float_same_or_nan(float(sample_values[feature_id]), float(cell["value"]))

    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.post("/scalar/schema", json={"manifest_path": manifest_path}).status_code == 200

    both_feature_axes = client.post(
        "/scalar/features",
        json={
            "manifest_path": manifest_path,
            "feature_ids": [0],
            "feature_keys": ["feature_000000"],
            "sample_ids": [0],
        },
    )
    assert both_feature_axes.status_code == 400, both_feature_axes.text

    both_sample_axes = client.post(
        "/scalar/features",
        json={
            "manifest_path": manifest_path,
            "feature_ids": [0],
            "sample_ids": [0],
            "sample_keys": ["sample_000000"],
        },
    )
    assert both_sample_axes.status_code == 400, both_sample_axes.text

    max_cells_resp = client.post(
        "/scalar/features",
        json={"manifest_path": manifest_path, "feature_ids": [0, 1], "sample_ids": [0, 1], "max_cells": 3},
    )
    assert max_cells_resp.status_code == 413, max_cells_resp.text

    out_of_range_resp = client.post(
        "/scalar/features",
        json={"manifest_path": manifest_path, "feature_ids": [9999], "sample_ids": [0]},
    )
    assert out_of_range_resp.status_code == 404, out_of_range_resp.text

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
    for value_payload in feature["values"]:
        cell = _expected_cell(final_rows, feature["feature_id"], int(value_payload["sample_id"]))
        _assert_api_value_matches_cell(value_payload, cell)

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
    for value_payload in sample_payload["values"]:
        cell = _expected_cell(final_rows, int(value_payload["feature_id"]), sample_payload["sample_id"])
        _assert_api_value_matches_cell(value_payload, cell)

    sample_by_id_resp = client.post(
        "/scalar/sample",
        json={"manifest_path": manifest_path, "sample_id": 0, "feature_ids": [0, 1, 2]},
    )
    assert sample_by_id_resp.status_code == 200, sample_by_id_resp.text
    assert sample_by_id_resp.json()["sample_key"] == "sample_000000"

    both_sample_id_key = client.post(
        "/scalar/sample",
        json={"manifest_path": manifest_path, "sample_id": 0, "sample_key": "sample_000000", "feature_ids": [0]},
    )
    assert both_sample_id_key.status_code == 400, both_sample_id_key.text

    max_features_resp = client.post(
        "/scalar/sample",
        json={"manifest_path": manifest_path, "sample_id": 0, "max_features": 1},
    )
    assert max_features_resp.status_code == 413, max_features_resp.text

    missing_feature_resp = client.post(
        "/scalar/sample",
        json={"manifest_path": manifest_path, "sample_id": 0, "feature_keys": ["missing_feature_key"]},
    )
    assert missing_feature_resp.status_code == 404, missing_feature_resp.text

    missing_json_resp = client.post(
        "/scalar/features",
        json={
            "manifest_path": manifest_path,
            "feature_ids": [int(missing_cell["feature_id"])],
            "sample_ids": [int(missing_cell["sample_id"])],
        },
    )
    assert missing_json_resp.status_code == 200, missing_json_resp.text
    missing_value = missing_json_resp.json()["features"][0]["values"][0]
    assert missing_value["present"] is False
    assert missing_value["value"] is None

    print("python scalar api tests passed")


if __name__ == "__main__":
    main()
