Feature selection pipeline (Parquet + Polars + NumPy only)

This repo contains a practical implementation skeleton for large-scale feature selection with:
- Pairwise-valid Pearson r^2
- Greedy redundancy pruning
- Incremental cap expansion
- Feature-major Parquet shards

Key entry points:
- scripts/build_shards.py: sample-major -> feature-major shard conversion
- scripts/run_selection.py: candidate generation + incremental greedy selection
- scripts/generate_synth.py: synthetic data generation (sample-major)
- scripts/locate_feature.py: feature_id -> shard_id/offset lookup (via locator)
- scripts/run_tests.py: small correctness checks
- scripts/build_array_shards.py: array bundle -> array feature micro-block shard conversion
- scripts/generate_array_synth.py: synthetic array bundle/sample_meta generation (optional shard build)
- scripts/locate_array_feature.py: array feature -> block lookup (via locator)
- scripts/serve_array_api.py: FastAPI POST endpoints for array/scalar lookup and scalar feature selection
- scripts/run_scalar_api_tests.py: FastAPI scalar endpoint checks
- scripts/run_selection_api_tests.py: FastAPI selection endpoint checks
- scripts/run_array_storage_tests.py: array storage correctness checks
- scripts/run_array_synthetic_tests.py: synthetic array generation + sample_id retrieval checks
- scripts/run_array_api_tests.py: FastAPI array endpoint checks
- scripts/run_array_binary_package_tests.py: public binary-shard package facade smoke tests

Expected input:
- scalar sample metadata must contain columns: `y`, `sample_path`, and typically
  dense `sample_id` plus external `sample_key`
- scalar feature metadata should live in `feature_meta.parquet` with dense
  `feature_id` and external `feature_key`
- scalar v2 treats both `sample_id` and `feature_id` as dense zero-based row
  indices defined by metadata row order
- sample_path points to per-sample Parquet files in long format with columns:
  `feature_id`, `value`

Shard output schema:
- shard parquet: `feature_id(Int32 signed), value_len(Int32), values_blob(Binary), valid_blob(Binary)`
- `values_blob`: little-endian float64 buffer (`value_len * 8` bytes)
- `valid_blob`: uint8 mask buffer (`value_len` bytes)
- feature locator parquet: `feature_id(Int32), global_rank(Int32), shard_id(Int32), offset_in_shard(Int32)`
- selection stats sidecars: `selection_stats/<y_col>.parquet` with columns `feature_id(Int32), shard_id(Int32), offset_in_shard(Int32), r2y(Float64), n_y_overlap(Int32)`
- shard files are stored in `<out-dir>/feature_shards/`
- manifest includes `shard_path` (the subfolder path) and `n_shards` (instead of per-file `shard_paths`)
- scalar shard build prefers `--target-shard-mb`; `--n-shards` remains as a legacy override
- scalar shard output is self-contained: `shard_manifest.json`, copied `sample_meta.parquet`, copied `feature_meta.parquet`, `feature_locator.parquet`, `selection_stats/`, and `feature_shards/` live under one output folder
- manifest stores a `selection_stats` mapping from each precomputed `y_col` to its sidecar parquet path; `scripts.run_selection` reuses that sidecar when available and falls back to shard rescans otherwise
- manifest `feature_id_dtype` is unified to `INT32`
- redundant fields (`n_features`, `shard_starts`, `shard_ends`) are removed
- Python selection defaults are tuned for stage-graph execution: `initial_cap=2048`, `max_step=4096`, `batch_size=512`, `max_gap=64`.
- Mask-group fast path is available for structured missingness. Defaults are conservative: `mask_fastpath_min_group=64`, `mask_fastpath_min_pairs=8192`.

Run from the `python/` directory so package imports resolve:
```
cd python
```

Scalar pipeline examples:
```
python -m scripts.generate_synth --out-dir ..\data\synth_py\samples --sample-meta ..\data\synth_py\sample_meta.parquet --feature-meta ..\data\synth_py\feature_meta.parquet
python -m scripts.build_shards --sample-meta ..\data\synth_py\sample_meta.parquet --feature-meta ..\data\synth_py\feature_meta.parquet --out-dir ..\data\synth_py\shards --target-shard-mb 32 --stats-y-col y --stats-y-col y_alt
python -m scripts.run_selection --manifest ..\data\synth_py\shards\shard_manifest.json --top-m 30
python -m scripts.locate_feature --manifest ..\data\synth_py\shards\shard_manifest.json --feature-id 12
python -m scripts.run_tests
python -m scripts.run_scalar_api_tests
python -m scripts.run_selection_api_tests
```

Direct scalar ingestion example:
```python
from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder, write_feature_meta, write_sample_meta

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "y": 1.0, "y_alt": 1.5},
        {"sample_key": "sample_000001", "y": 2.0, "y_alt": 2.5},
    ],
    "..\\data\\scalar_sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    "..\\data\\scalar_feature_meta.parquet",
)

builder = ScalarDatasetBuilder(
    out_dir="..\\data\\scalar_shards",
    sample_meta_path=sample_meta_path,
    feature_meta_path=feature_meta_path,
    build_options=ScalarShardBuildOptions(
        target_shard_mb=32,
        stats_y_cols=("y", "y_alt"),
    ),
)

builder.write_sample(0, {"feature_a": 1.23, "feature_b": 4.56})
builder.write_sample(1, {"feature_a": 7.89})
manifest_path = builder.build_shards()
```

Discovered-feature mode is also supported:
```python
from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder

with ScalarDatasetBuilder(
    out_dir="..\\data\\scalar_shards",
    sample_meta_path="..\\data\\scalar_sample_meta.parquet",
    build_options=ScalarShardBuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
) as builder:
    with builder.open_sample(0) as sample:
        sample.write_value("feature_x", 1.23)
        sample.write_value("feature_y", 4.56)

    with builder.open_sample(1) as sample:
        sample.write_values({"feature_y": 7.89, "feature_z": 0.12})

    builder.finish_sample_major()
    builder.update_feature_meta(
        [
            {"feature_key": "feature_x", "group": "alpha"},
            {"feature_key": "feature_y", "group": "beta"},
            {"feature_key": "feature_z", "group": "gamma"},
        ],
        require_all=True,
    )
    manifest_path = builder.build_shards(keep_sample_major=True)
```

Scalar builder notes:
- The primary safe API is sample-scoped. Each sample is flushed immediately, so memory use is bounded by one sample's feature map.
- Samples may be written in any order, but the same `sample_id` cannot be written twice.
- `finish_sample_major()` makes the visible intermediate sample-major stage explicit.
- `keep_sample_major=True` keeps the visible sample-major stage after shard build. The default is to remove it.

Structured-missing synthetic example:
```
python -m scripts.generate_synth --out-dir ..\data\synth_py\samples --sample-meta ..\data\synth_py\sample_meta.parquet --n-sample-cohorts 16 --n-missing-patterns 32 --shared-missing-feature-ratio 0.9 --residual-missing-rate 0.01
python -m scripts.run_selection --manifest ..\data\synth_py\shards\shard_manifest.json --top-m 30 --mask-fastpath-min-group 64 --mask-fastpath-min-pairs 8192
```

Array bundle / shard pipeline:
- Raw sample bundle row schema:
  `sample_row(Int64), sample_id(Int64), feature_id(Int32), flags(UInt8), trace_len(Int32), time_blob(Binary), value_blob(Binary)`
- Array shard row schema:
  `feature_id(Int32), block_id(Int32), sample_row_start(Int64), sample_count(Int32), point_count(Int64), sample_flags_blob(Binary), sample_offsets_blob(Binary), time_blob(Binary), value_blob(Binary)`
- Array locator schema:
  `feature_id(Int32), block_id(Int32), shard_id(Int32), row_in_shard(Int32), sample_row_start(Int64), sample_row_end(Int64)`
- Array blocks are feature-major sample micro-blocks; defaults are `samples_per_block=8`, `row_group_size=64`.
- Array shard build is one-shard-at-a-time. A feature never spans multiple shard files.
- Preferred array shard sizing input is `--target-shard-mb`; `--n-shards` remains as a legacy override.

Array pipeline examples:
```
python -m scripts.generate_array_synth --bundle-out-dir ..\data\array_bundles --sample-meta ..\data\array_sample_meta.parquet --shard-out-dir ..\data\array_shards --n-samples 256 --n-features 64 --seed 7 --target-shard-mb 256 --samples-per-block 8 --row-group-size 64
python -m scripts.build_array_shards --bundle-manifest ..\data\array_bundle_manifest.json --out-dir ..\data\array_shards --target-shard-mb 256 --samples-per-block 8 --row-group-size 64
python -m scripts.locate_array_feature --manifest ..\data\array_shards\array_shard_manifest.json --feature-id 12345
python -m scripts.serve_array_api
python -m scripts.run_array_storage_tests
python -m scripts.run_array_synthetic_tests
python -m scripts.run_array_api_tests
```

Binary array shard library examples:
```
python -m scripts.build_array_binary_shards --bundle-manifest ..\data\array_bundle_manifest.json --out-dir ..\data\array_binary_shards --target-shard-mb 32 --samples-per-block 16 --codec none
```

Direct array trace ingestion example:
```python
from fs.array import (
    ArrayDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
    write_feature_meta,
    write_sample_meta,
)
from fs.config import ArrayBinaryBuildOptions

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "split": "train"},
        {"sample_key": "sample_000001", "split": "test"},
    ],
    "..\\data\\sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    "..\\data\\feature_meta.parquet",
)

with ArrayDatasetBuilder(
    out_dir="..\\data\\array_binary_shards",
    sample_meta_path=sample_meta_path,
    point_schema=[
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ],
    feature_meta_path=feature_meta_path,
    build_options=ArrayBinaryBuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
) as builder:
    builder.add_trace(
        sample_id=0,
        feature_key="feature_a",
        columns={
            "phase": [10, 11, 12],
            "state_code": ["OK", "OK", "WARN"],
        },
    )
    manifest_path = builder.build_shards()
```

The core builder also supports discovered-feature mode: omit `feature_keys` and
`feature_meta_path`, then always ingest by `feature_key`. Dense feature ids are
assigned in first-seen order, and categorical string labels are encoded into
dictionary-backed integer codes automatically. In that discovered mode, the
generated `feature_meta.parquet` currently contains only `feature_id` and
`feature_key`. If richer feature metadata columns are needed, create them up
front with `write_feature_meta(...)` and pass `feature_meta_path=...`, or
finalize the bundle stage and enrich discovered features before shard build:

```python
builder.finish_bundles()
builder.update_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha", "rank_hint": 10},
        {"feature_key": "feature_b", "group": "beta", "rank_hint": 20},
    ],
    require_all=True,
)
manifest_path = builder.build_shards()
```

If temporal point columns are needed, use `storage_type="int64"` with
`logical_type="timestamp_ns"` or `logical_type="timedelta_ns"`. The shard stores
raw nanosecond counts; the public wheel reader converts them back to NumPy
`datetime64[ns]` / `timedelta64[ns]`.

`storage_type` and `logical_type` may still be passed as strings, but the
recommended API is to use the exported enums so typos fail earlier:

```python
from fs.array import LogicalType, PointColumnSpec, StorageType

PointColumnSpec(
    name="ts",
    storage_type=StorageType.INT64,
    logical_type=LogicalType.TIMESTAMP_NS,
)
```

The bundle stage is explicit. If the intermediate sample-major bundle artifact
should remain visible, use:

```python
builder = ArrayDatasetBuilder(
    out_dir="..\\data\\array_binary_shards",
    bundle_out_dir="..\\data\\array_bundle_stage",
    sample_meta_path="..\\data\\sample_meta.parquet",
    point_schema=[
        {"name": "phase", "storage_type": "int32", "logical_type": "integer"},
    ],
    feature_keys=["feature_a"],
)

builder.add_trace(
    sample_id=0,
    feature_key="feature_a",
    columns={"phase": [10, 11, 12]},
)

bundle_manifest_path = builder.finish_bundles()
manifest_path = builder.build_shards(cleanup_bundles=False)
```

Use `cleanup_bundles=True` to delete the visible bundle stage after the final
binary shard artifact has been built.

Standalone package location:
```
..\packages\array_binary_shard
```

Public package API:
```python
from array_binary_shard import BuildOptions, build_shard, open_shard

manifest_path = build_shard(
    source="..\\data\\array_bundle_manifest.json",
    out_dir="..\\data\\array_binary_shards",
    options=BuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
)

with open_shard(manifest_path) as ds:
    trace = ds.get_trace(feature_id=123, sample_id=1001)
    batch = ds.get_traces(feature_id=123, sample_ids=[1001, 1007, 1015])
    result = ds.get_many(feature_ids=[123, 456], sample_ids=[1001, 1007])
```

Package build:
```
cd ..\packages\array_binary_shard
python -m pip install build
python -m build
```

Notes:
- The wheel-friendly public API lives under `packages/array_binary_shard/src/array_binary_shard/`.
- The standalone package carries its own private implementation under `array_binary_shard._impl`.
- The current recommended binary defaults are `samples_per_block=16` and `codec=none`.

Python sample_id retrieval API:
- `fs.scalar.parquet_storage.build_sample_id_index(sample_meta_path)`
- `fs.array.storage.ArrayShardReader.load_feature_samples_by_sample_ids(feature_id, sample_ids, ...)`
- Synthetic `sample_meta.parquet` includes `sample_row, sample_id, y`.

HTTP POST example:
```json
POST /array-feature
{
  "manifest_path": "C:\\data\\array_shards\\array_shard_manifest.json",
  "feature_id": 123,
  "sample_ids": [101, 205, 999],
  "sanitize_nonfinite": true
}
```

Multiple features example:
```json
POST /array-feature
{
  "manifest_path": "C:\\data\\array_shards\\array_shard_manifest.json",
  "feature_ids": [123, 456],
  "sample_ids": [101, 205, 999],
  "sanitize_nonfinite": true
}
```

Response shape:
- `traces[i].sample_id`: requested sample id
- `traces[i].sample_row`: resolved sample row, `-1` if sample id not found
- `traces[i].flags`: stored trace flags
- `traces[i].time`, `traces[i].value`: JSON-safe arrays; when `sanitize_nonfinite=true`, `NaN/Inf/-Inf` become `null`
- `temporal_format` may be set to `iso` or `raw_ns` for `timestamp_ns` / `timedelta_ns` point columns
- Use exactly one of `feature_id` or `feature_ids`.
- When `feature_ids` is used, the response contains `features[j].feature_id` and `features[j].traces`.

Scalar HTTP POST example:
```json
POST /scalar-feature
{
  "manifest_path": "C:\\data\\shards\\shard_manifest.json",
  "feature_key": "feature_000123",
  "sample_keys": ["sample_000101", "sample_000205", "sample_000999"],
  "sanitize_nonfinite": true
}
```

Scalar response shape:
- `feature_id`: dense internal feature id
- `feature_key`: external feature key
- `values[i].sample_id`: dense internal sample id
- `values[i].sample_key`: external sample key
- `values[i].present`: whether the sample has a valid scalar value for that feature
- `values[i].value`: scalar feature value, or `null` if missing / sanitized nonfinite
- Use exactly one of `feature_id` or `feature_key`.
- Use exactly one of `sample_ids` or `sample_keys`.

Selection HTTP POST example:
```json
POST /run-selection
{
  "manifest_path": "C:\\data\\shards\\shard_manifest.json",
  "y_col": "y",
  "top_m": 128,
  "initial_cap": 2048,
  "max_step": 4096,
  "batch_size": 512,
  "max_gap": 64,
  "mask_fastpath_min_group": 64,
  "mask_fastpath_min_pairs": 8192
}
```

Selection response shape:
- `selected_feature_ids`: selected feature ids in greedy order
- `selected_feature_keys`: selected feature keys aligned with `selected_feature_ids`
- `candidate_count`: candidate count before redundancy pruning
- `used_locator_stats`: whether precomputed selection stats were used instead of rescanning shard payloads
- `candidate_build_ms`, `selection_ms`, `elapsed_ms`: API timing breakdown in milliseconds
