import json
import shutil
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from fs.array.binary_storage import build_array_binary_shards_from_bundles
from fs.array.synthetic import generate_array_synthetic
from fs.array.storage import ArraySampleBundleWriter
from fs.config import ArrayBundleConfig, ArrayShardConfig, ArraySyntheticConfig
from fs.types import PointColumnSpec
from scripts.serve_array_api import app


def main():
    """Run integration tests for the array-serving API endpoints."""
    root = Path(__file__).resolve().parents[2] / "data" / "tmp_py_array_api_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    result = generate_array_synthetic(
        bundle_out_dir=str(root / "bundles"),
        sample_meta_path=str(root / "sample_meta.parquet"),
        config=ArraySyntheticConfig(
            n_samples=16,
            n_features=10,
            min_trace_len=12,
            max_trace_len=24,
            seed=11,
        ),
        bundle_config=ArrayBundleConfig(max_bundle_rows=64, max_bundle_bytes=1 << 20),
        shard_out_dir=str(root / "shards"),
        shard_config=ArrayShardConfig(n_shards=2, samples_per_block=4),
    )
    binary_manifest_path = build_array_binary_shards_from_bundles(
        result["bundle_manifest_path"],
        str(root / "binary_shards"),
        config=ArrayShardConfig(n_shards=2, samples_per_block=4),
    )

    dict_dir = root / "categorical_dictionaries_source"
    dict_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = dict_dir / "state_code.json"
    event_dict_path = dict_dir / "event_type.json"
    state_dict_path.write_text(
        json.dumps(
            {
                "column": "state_code",
                "items": [
                    {"code": 1, "label": "OK"},
                    {"code": 2, "label": "WARN"},
                    {"code": 3, "label": "FAIL"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    event_dict_path.write_text(
        json.dumps(
            {
                "column": "event_type",
                "items": [
                    {"code": 1, "label": "START"},
                    {"code": 2, "label": "STOP"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    custom_schema = [
        PointColumnSpec(name="phase", storage_type="int32", logical_type="integer"),
        PointColumnSpec(
            name="state_code",
            storage_type="uint32",
            logical_type="categorical",
            dictionary_path=str(state_dict_path),
        ),
        PointColumnSpec(
            name="event_type",
            storage_type="uint32",
            logical_type="categorical",
            dictionary_path=str(event_dict_path),
        ),
    ]
    with ArraySampleBundleWriter(
        str(root / "v3_bundles"),
        str(root / "sample_meta.parquet"),
        feature_meta_path=str(result["feature_meta_path"]),
        n_samples=16,
        config=ArrayBundleConfig(max_bundle_rows=64, max_bundle_bytes=1 << 20),
        point_schema=custom_schema,
    ) as writer:
        writer.append_trace(
            0,
            0,
            columns={
                "phase": np.asarray([10, 11, 12], dtype=np.int32),
                "state_code": np.asarray([1, 1, 2], dtype=np.uint32),
                "event_type": np.asarray([1, 1, 2], dtype=np.uint32),
            },
        )
        writer.append_trace(
            3,
            0,
            columns={
                "phase": np.asarray([20], dtype=np.int32),
                "state_code": np.asarray([3], dtype=np.uint32),
                "event_type": np.asarray([2], dtype=np.uint32),
            },
        )
        custom_bundle_manifest_path = writer.finish()
    binary_v3_manifest_path = build_array_binary_shards_from_bundles(
        custom_bundle_manifest_path,
        str(root / "binary_v3_shards"),
        config=ArrayShardConfig(n_shards=1, samples_per_block=4),
    )
    temporal_schema = [
        PointColumnSpec(name="ts", storage_type="int64", logical_type="timestamp_ns"),
        PointColumnSpec(name="dt", storage_type="int64", logical_type="timedelta_ns"),
    ]
    with ArraySampleBundleWriter(
        str(root / "temporal_bundles"),
        str(root / "sample_meta.parquet"),
        feature_meta_path=str(result["feature_meta_path"]),
        n_samples=16,
        config=ArrayBundleConfig(max_bundle_rows=64, max_bundle_bytes=1 << 20),
        point_schema=temporal_schema,
    ) as writer:
        writer.append_trace(
            0,
            0,
            columns={
                "ts": np.asarray(["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"),
                "dt": np.asarray([0, 1_000_000_000], dtype="timedelta64[ns]"),
            },
        )
        writer.append_trace(
            3,
            0,
            columns={
                "ts": np.asarray(["2024-01-01T00:00:05"], dtype="datetime64[ns]"),
                "dt": np.asarray([5_000_000_000], dtype="timedelta64[ns]"),
            },
        )
        temporal_bundle_manifest_path = writer.finish()
    binary_temporal_manifest_path = build_array_binary_shards_from_bundles(
        temporal_bundle_manifest_path,
        str(root / "binary_temporal_shards"),
        config=ArrayShardConfig(n_shards=1, samples_per_block=4),
    )

    client = TestClient(app)
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    resp = client.post(
        "/array-feature",
        json={
            "manifest_path": result["shard_manifest_path"],
            "feature_id": 0,
            "sample_ids": [0, 3, 7],
            "sanitize_nonfinite": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["feature_id"] == 0
    assert data["feature_key"] == "feature_000000"
    assert data["sample_count"] == 3
    assert [trace["sample_id"] for trace in data["traces"]] == [0, 3, 7]
    assert [trace["sample_key"] for trace in data["traces"]] == [
        "sample_000000",
        "sample_000003",
        "sample_000007",
    ]
    for trace in data["traces"]:
        assert "sample_row" not in trace
        assert "sample_key" in trace
        assert set(trace["columns"].keys()) == {"time", "value"}
        assert len(trace["columns"]["time"]) == len(trace["columns"]["value"])

    multi = client.post(
        "/array-feature",
        json={
            "manifest_path": result["shard_manifest_path"],
            "feature_ids": [0, 1],
            "sample_ids": [0, 3, 7],
            "sanitize_nonfinite": True,
        },
    )
    assert multi.status_code == 200, multi.text
    multi_data = multi.json()
    assert multi_data["feature_ids"] == [0, 1]
    assert multi_data["feature_keys"] == ["feature_000000", "feature_000001"]
    assert multi_data["sample_count"] == 3
    assert len(multi_data["features"]) == 2
    assert [item["feature_id"] for item in multi_data["features"]] == [0, 1]
    assert [item["feature_key"] for item in multi_data["features"]] == [
        "feature_000000",
        "feature_000001",
    ]
    for item in multi_data["features"]:
        assert [trace["sample_id"] for trace in item["traces"]] == [0, 3, 7]
        assert [trace["sample_key"] for trace in item["traces"]] == [
            "sample_000000",
            "sample_000003",
            "sample_000007",
        ]
        for trace in item["traces"]:
            assert "sample_row" not in trace
            assert "sample_key" in trace
            assert set(trace["columns"].keys()) == {"time", "value"}
            assert len(trace["columns"]["time"]) == len(trace["columns"]["value"])

    binary = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_manifest_path,
            "feature_ids": [0, 1],
            "sample_ids": [0, 3, 7],
            "sanitize_nonfinite": True,
        },
    )
    assert binary.status_code == 200, binary.text
    binary_data = binary.json()
    assert binary_data["feature_ids"] == [0, 1]
    assert binary_data["sample_count"] == 3
    assert len(binary_data["features"]) == 2
    assert binary_data["features"] == multi_data["features"]

    keyed = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_manifest_path,
            "feature_keys": ["feature_000000", "feature_000001"],
            "sample_keys": ["sample_000000", "sample_000003", "sample_000007"],
            "sanitize_nonfinite": True,
        },
    )
    assert keyed.status_code == 200, keyed.text
    keyed_data = keyed.json()
    assert keyed_data["feature_ids"] == [0, 1]
    assert keyed_data["sample_count"] == 3
    assert keyed_data["features"] == binary_data["features"]

    keyed_single = client.post(
        "/array-feature",
        json={
            "manifest_path": result["shard_manifest_path"],
            "feature_key": "feature_000000",
            "sample_keys": ["sample_000000", "sample_000003", "sample_000007"],
            "sanitize_nonfinite": True,
        },
    )
    assert keyed_single.status_code == 200, keyed_single.text
    keyed_single_data = keyed_single.json()
    assert keyed_single_data["feature_id"] == 0
    assert keyed_single_data["feature_key"] == "feature_000000"
    assert keyed_single_data["sample_count"] == 3
    assert keyed_single_data["traces"] == data["traces"]

    schema_resp = client.post(
        "/array-schema",
        json={
            "manifest_path": binary_v3_manifest_path,
            "include_dictionaries": True,
        },
    )
    assert schema_resp.status_code == 200, schema_resp.text
    schema_data = schema_resp.json()
    assert schema_data["version"] == 3
    assert [spec["name"] for spec in schema_data["point_schema"]] == ["phase", "state_code", "event_type"]
    assert schema_data["categorical_dictionaries"]["state_code"]["1"] == "OK"
    assert schema_data["categorical_dictionaries"]["event_type"]["2"] == "STOP"

    v3_raw = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_v3_manifest_path,
            "feature_id": 0,
            "sample_keys": ["sample_000000", "sample_000003"],
            "sanitize_nonfinite": True,
            "decode_categorical": False,
        },
    )
    assert v3_raw.status_code == 200, v3_raw.text
    v3_raw_data = v3_raw.json()
    assert v3_raw_data["feature_id"] == 0
    assert set(v3_raw_data["traces"][0]["columns"].keys()) == {"phase", "state_code", "event_type"}
    assert v3_raw_data["traces"][0]["columns"]["phase"] == [10, 11, 12]
    assert v3_raw_data["traces"][0]["columns"]["state_code"] == [1, 1, 2]
    assert v3_raw_data["traces"][1]["columns"]["event_type"] == [2]

    v3_decoded = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_v3_manifest_path,
            "feature_key": "feature_000000",
            "sample_ids": [0, 3],
            "sanitize_nonfinite": True,
            "decode_categorical": True,
        },
    )
    assert v3_decoded.status_code == 200, v3_decoded.text
    v3_decoded_data = v3_decoded.json()
    assert v3_decoded_data["feature_key"] == "feature_000000"
    assert v3_decoded_data["traces"][0]["columns"]["state_code"] == ["OK", "OK", "WARN"]
    assert v3_decoded_data["traces"][0]["columns"]["event_type"] == ["START", "START", "STOP"]
    assert v3_decoded_data["traces"][1]["columns"]["state_code"] == ["FAIL"]

    temporal_iso = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_temporal_manifest_path,
            "feature_id": 0,
            "sample_ids": [0, 3],
            "sanitize_nonfinite": True,
        },
    )
    assert temporal_iso.status_code == 200, temporal_iso.text
    temporal_iso_data = temporal_iso.json()
    assert temporal_iso_data["traces"][0]["columns"]["ts"] == [
        "2024-01-01T00:00:00.000000000",
        "2024-01-01T00:00:01.000000000",
    ]
    assert temporal_iso_data["traces"][0]["columns"]["dt"] == ["PT0S", "PT1S"]
    assert temporal_iso_data["traces"][1]["columns"]["ts"] == ["2024-01-01T00:00:05.000000000"]
    assert temporal_iso_data["traces"][1]["columns"]["dt"] == ["PT5S"]

    temporal_raw = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_temporal_manifest_path,
            "feature_id": 0,
            "sample_ids": [0, 3],
            "sanitize_nonfinite": True,
            "temporal_format": "raw_ns",
        },
    )
    assert temporal_raw.status_code == 200, temporal_raw.text
    temporal_raw_data = temporal_raw.json()
    assert temporal_raw_data["traces"][0]["columns"]["dt"] == [0, 1_000_000_000]
    assert temporal_raw_data["traces"][1]["columns"]["dt"] == [5_000_000_000]

    temporal_bad = client.post(
        "/array-feature",
        json={
            "manifest_path": binary_temporal_manifest_path,
            "feature_id": 0,
            "sample_ids": [0],
            "temporal_format": "bad_format",
        },
    )
    assert temporal_bad.status_code == 400, temporal_bad.text

    cache_stats = client.get("/cache-stats")
    assert cache_stats.status_code == 200, cache_stats.text
    stats = cache_stats.json()
    assert "manifest_cache" in stats
    assert "array_binary_cache" in stats
    assert "array_parquet_cache" in stats
    assert "scalar_parquet_cache" in stats
    assert stats["manifest_cache"]["array_entries"] >= 1
    assert stats["manifest_cache"]["max_entries"] >= stats["manifest_cache"]["array_entries"]
    assert isinstance(stats["manifest_cache"]["array_manifests"], list)
    assert isinstance(stats["manifest_cache"]["scalar_manifests"], list)
    assert "block_records_entries" in stats["array_binary_cache"]
    assert "open_mmaps" in stats["array_binary_cache"]
    assert "entries" in stats["array_parquet_cache"]
    assert "max_entries" in stats["array_parquet_cache"]
    assert "ttl_seconds" in stats["array_parquet_cache"]
    assert "open_scans" in stats["scalar_parquet_cache"]
    assert "open_scan_manifests" in stats["scalar_parquet_cache"]

    bad = client.post(
        "/array-feature",
        json={
            "manifest_path": result["shard_manifest_path"],
            "sample_ids": [0, 3, 7],
        },
    )
    assert bad.status_code == 400, bad.text

    print("python array api tests passed")


if __name__ == "__main__":
    main()
