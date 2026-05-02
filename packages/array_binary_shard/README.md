# array-binary-shard

Standalone Python package for the dense-id array binary shard format.

## Design summary

This package implements the v3 binary format described in
[`docs/array_binary_shard_format_v3.md`](../../docs/array_binary_shard_format_v3.md).

The core rule is simple:

- `sample_id` is the dense row index of `sample_meta.parquet`
- `feature_id` is the dense row index of `feature_meta.parquet`

Optional external identifiers live in metadata columns:

- `sample_key`
- `feature_key`

The package therefore supports both:

- fast dense-id lookup
- convenient external-key lookup

Within one manifest, the point-column schema is fixed. The schema may differ
between manifests. `time` and `value` are no longer required columns in v3.

Categorical point columns are stored as integer codes inside shard payloads and
optionally mapped back to their original string labels through sidecar parquet
dictionary files.

Binary shard output is a self-contained dataset directory. The manifest stores
relative paths to:

- `sample_meta.parquet`
- `feature_meta.parquet`
- `array_binary_feature_shards/`

This means the whole output directory can be copied or moved as one artifact.

Bundle-to-binary builds always use append-only temporary spill files internally.
That keeps build cost proportional to the input size and avoids the pathological
per-shard full rescans that a direct no-spill implementation would trigger.

## Public API

- reader
  - `open_shard(...)`
  - `BinaryShardDataset`
- writer
  - `build_shard(...)`
  - `convert_parquet_shard(...)`
  - `ArrayDatasetBuilder`
- models / exceptions
- schema enums
  - `StorageType`
  - `LogicalType`
  - `PointColumnSpec`

## Layout

```text
packages/array_binary_shard/
  pyproject.toml
  README.md
  src/
    array_binary_shard/
      __init__.py
      reader.py
      writer.py
      models.py
      exceptions.py
      _impl/
        binary_storage.py
        storage.py
        config.py
        types.py
        sample_index.py
```

## Build

```bash
cd packages/array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

## Build from sample-major bundles

```python
from array_binary_shard import BuildOptions, build_shard

manifest_path = build_shard(
    source=".../array_bundle_manifest.json",
    out_dir=".../array_binary_shards",
    options=BuildOptions(
        samples_per_block=16,
        target_shard_mb=32,
        codec="none",
        sample_key_col="sample_key",
        feature_key_col="feature_key",
    ),
)
```

The builder always uses the append-only spill path internally. Spill controls are
not exposed through the public package API.

## Build directly from traces

For user-facing ingestion, prefer `ArrayDatasetBuilder` instead of manually
assembling sample-major parquet bundles.

Known-feature mode:

```python
from array_binary_shard import (
    ArrayDatasetBuilder,
    BuildOptions,
    LogicalType,
    PointColumnSpec,
    StorageType,
    write_feature_meta,
    write_sample_meta,
)

sample_meta_path = write_sample_meta(
    [
        {"sample_key": "sample_000000", "split": "train"},
        {"sample_key": "sample_000001", "split": "test"},
    ],
    ".../sample_meta.parquet",
)
feature_meta_path = write_feature_meta(
    [
        {"feature_key": "feature_a", "group": "alpha"},
        {"feature_key": "feature_b", "group": "beta"},
    ],
    ".../feature_meta.parquet",
)

builder = ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    sample_meta_path=sample_meta_path,
    point_schema=[
        PointColumnSpec(name="phase", storage_type=StorageType.INT32, logical_type=LogicalType.INTEGER),
        PointColumnSpec(name="state_code", storage_type=StorageType.UINT32, logical_type=LogicalType.CATEGORICAL),
    ],
    feature_meta_path=feature_meta_path,
    build_options=BuildOptions(samples_per_block=16, target_shard_mb=32, codec="none"),
)

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

Discovered-feature mode:

```python
from array_binary_shard import ArrayDatasetBuilder, LogicalType, StorageType

with ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    sample_meta_path=".../sample_meta.parquet",
    point_schema=[
        {"name": "phase", "storage_type": StorageType.INT32, "logical_type": LogicalType.INTEGER},
        {"name": "state_code", "storage_type": StorageType.UINT32, "logical_type": LogicalType.CATEGORICAL},
    ],
) as builder:
    with builder.sample(sample_id=0) as sample:
        sample.add_trace(
            feature_key="feature_a",
            columns={
                "phase": [10, 11, 12],
                "state_code": ["OK", "OK", "WARN"],
            },
        )
        sample.add_trace(
            feature_key="feature_b",
            columns={
                "phase": [3],
                "state_code": ["FAIL"],
            },
        )
```

Categorical point columns accept original string labels during ingestion. The
builder assigns integer codes automatically and writes the corresponding
dictionary parquet files into the bundle stage and final artifact.

Temporal point columns are also supported when stored as `int64`:
- `logical_type="timestamp_ns"` for `datetime64[ns]`
- `logical_type="timedelta_ns"` for `timedelta64[ns]`

The package stores both as raw nanosecond `int64` values and converts them back
to NumPy `datetime64[ns]` / `timedelta64[ns]` arrays in the public reader API.

`storage_type` and `logical_type` may still be passed as strings, but the
recommended public API is to use the exported enums so typos fail earlier:

```python
from array_binary_shard import LogicalType, PointColumnSpec, StorageType

PointColumnSpec(
    name="ts",
    storage_type=StorageType.INT64,
    logical_type=LogicalType.TIMESTAMP_NS,
)
```

The builder now exposes the intermediate bundle stage explicitly:

```python
from array_binary_shard import ArrayDatasetBuilder

builder = ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    bundle_out_dir=".../array_bundle_stage",
    sample_meta_path=".../sample_meta.parquet",
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

Use:
- `finish_bundles()` when the sample-major bundle artifact itself is useful
- `build_shards()` for the final binary shard artifact
- `cleanup_bundles=True` if the bundle stage should be deleted after shard build

In discovered-feature mode, the generated `feature_meta.parquet` currently
contains only `feature_id` and `feature_key`. If richer feature metadata
columns are required, prebuild `feature_meta.parquet` with
`write_feature_meta(...)` and use known-feature mode, or enrich discovered
features after the bundle stage has been frozen:

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

The resulting directory layout is:

```text
array_binary_shards/
  array_binary_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  array_binary_feature_shards/
    shard_0000.blocks.idx
    shard_0000.blocks.bin
    ...
```

## Read by dense ids

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    schema = ds.point_schema
    trace = ds.get_trace(feature_id=123, sample_id=1001)
    batch = ds.get_traces(feature_id=123, sample_ids=[1001, 1007, 1015])
    result = ds.get_many(feature_ids=[123, 456], sample_ids=[1001, 1007])
```

## Read by external keys

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    trace = ds.get_trace_by_key(
        feature_key="feature_000123",
        sample_key="sample_001001",
    )
    batch = ds.get_traces_by_key(
        feature_key="feature_000123",
        sample_keys=["sample_001001", "sample_001007"],
    )
    result = ds.get_many_by_key(
        feature_keys=["feature_000123", "feature_000456"],
        sample_keys=["sample_001001", "sample_001007"],
    )
```

## Decode categorical columns

```python
from array_binary_shard import open_shard

with open_shard(".../array_binary_shard_manifest.json") as ds:
    dictionaries = ds.categorical_dictionaries()
    trace = ds.get_trace(
        feature_id=123,
        sample_id=1001,
        decode_categorical=True,
    )
    print(trace.columns["state_code"])
```
```

## Notes

- Dense ids are the fastest path.
- Key-based lookups lazily load metadata-to-id dictionaries and then use the same
  dense-id fast path internally.
- The recommended default codec is `none`.
- `trace.columns` is the canonical response shape. `trace.time` and
  `trace.value` remain as convenience accessors and may be empty when those
  columns are not present in the manifest schema.

## Optional compression support

If you need `codec="zstd"`, install the optional extra:

```bash
python -m pip install "array-binary-shard[zstd]"
```
